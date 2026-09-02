"""Render policy rollouts to a self-contained HTML page."""

from __future__ import annotations

import jax
import wandb
from brax.io import html
from flax import linen as nn

from srl.envs import make_env


def render_rollouts(cfg, agent, training_state, sys, save_path) -> None:
    """Roll the deterministic policy out in an unwrapped env and write `vis.html`.

    Rendering is best-effort: a failure here must not lose a finished training run.
    """
    try:
        rollout_states = []
        for i in range(cfg.train.num_render):
            raw_env, _, _, _ = make_env(cfg.env.env_id)
            raw_reset = jax.jit(raw_env.reset)
            raw_step = jax.jit(raw_env.step)

            @jax.jit
            def policy_step(es, actor_params):
                mean, _ = agent.actor.apply(actor_params, es.obs)
                a = nn.tanh(mean)
                ns = raw_step(es, a)
                return ns, es

            es = raw_reset(jax.random.PRNGKey(i + 1))
            for _ in range(cfg.train.vis_length):
                es, cur = policy_step(es, training_state.actor_state.params)
                rollout_states.append(cur.pipeline_state)

        html_string = html.render(sys, rollout_states)
        with open(str(save_path / "vis.html"), "w") as f:
            f.write(html_string)
        if cfg.train.track:
            wandb.log({"vis": wandb.Html(html_string)})
    except Exception as e:
        print(f"Render failed: {e}", flush=True)
