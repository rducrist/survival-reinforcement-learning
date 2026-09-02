"""Training entry point for Survival Reinforcement Learning (SRL).

    uv run -m srl.main agent.m=1 env.env_id=ant_u4_maze seed=0

Each epoch evaluates the current policy, then runs `train.num_training_steps_per_epoch`
rounds of: collect a rollout, relabel it into survival labels, and take SGD steps on the
hazard critic (and, every `train.actor_update_every` steps, on the actor and alpha).
"""

from __future__ import annotations

import pickle
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from brax import envs
from omegaconf import DictConfig, OmegaConf

from srl.agent import SRLAgent
from srl.envs import make_env
from srl.trainer import make_trainer
from srl.utils.buffer import make_replay_buffer
from srl.utils.common import (
    calculate_steps,
    get_run_name,
    init_wandb,
    make_save_dir,
    print_config,
    set_global_seeds,
)
from srl.utils.evaluation import make_evaluator
from srl.utils.flax_utils import Transition, checkpoint_params, save_params
from srl.utils.relabeling import make_relabel_fn
from srl.utils.visualization import render_rollouts

# Used by config/hydra/launcher/cleps.yaml for conditional SLURM node exclusion.
OmegaConf.register_new_resolver("eval", lambda s: eval(s, globals()), replace=True)


def make_envs(cfg: DictConfig):
    """Build the train and eval environments, and record the goal layout in the config."""
    env, state_dim, goal_start_idx, goal_end_idx = make_env(cfg.env.env_id)
    cfg.env.state_dim = state_dim
    cfg.env.goal_start_idx = goal_start_idx
    cfg.env.goal_end_idx = goal_end_idx

    obs_size = env.observation_size
    action_dim = env.action_size
    env = envs.training.wrap(env, episode_length=cfg.env.episode_length)

    eval_env, _, _, _ = make_env(cfg.env.env_id)
    eval_env = envs.training.wrap(eval_env, episode_length=cfg.env.episode_length)
    eval_env.step = jax.jit(eval_env.step)

    return env, eval_env, obs_size, action_dim


@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(cfg: DictConfig):
    cfg = calculate_steps(cfg)
    print_config(cfg)

    if cfg.agent.agent_name != "srl":
        raise ValueError(f"Unknown agent: {cfg.agent.agent_name}. Only 'srl' exists.")

    if cfg.seed is None:
        cfg.seed = np.random.randint(2**15)
    set_global_seeds(cfg.seed)

    key = jax.random.PRNGKey(cfg.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, critic_key = jax.random.split(
        key, 6
    )

    run_name = get_run_name(cfg)
    trigger_sync = init_wandb(cfg, run_name)
    save_path = make_save_dir(cfg, run_name)

    # ------------------------------------------------------------------ setup ---
    env, eval_env, obs_size, action_dim = make_envs(cfg)
    env_state = jax.jit(env.reset)(jax.random.split(env_key, cfg.env.num_envs))
    env.step = jax.jit(env.step)

    state_dim = cfg.env.state_dim
    goal_dim = cfg.env.goal_end_idx - cfg.env.goal_start_idx

    agent, training_state = SRLAgent.create(
        cfg,
        obs_size=obs_size,
        action_dim=action_dim,
        state_dim=state_dim,
        goal_dim=goal_dim,
        actor_key=actor_key,
        critic_key=critic_key,
    )

    dummy_transition = Transition(
        observation=jnp.zeros((obs_size,), dtype=jnp.float32),
        action=jnp.zeros((action_dim,), dtype=jnp.float32),
        reward=0.0,
        discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}},
    )
    replay_buffer, buffer_state = make_replay_buffer(cfg, dummy_transition, buffer_key)

    relabel_fn, relabel_cfg = make_relabel_fn(
        cfg, state_dim, cfg.env.goal_start_idx, cfg.env.goal_end_idx
    )
    trainer = make_trainer(cfg, agent, env, replay_buffer, relabel_fn, relabel_cfg)

    key, prefill_key = jax.random.split(key)
    training_state, env_state, buffer_state, key = trainer.prefill(
        training_state, env_state, buffer_state, prefill_key
    )

    evaluator, key = make_evaluator(cfg, agent, eval_env, eval_env_key, key)

    # -------------------------------------------------------------- training ---
    print("Start training...", flush=True)
    training_walltime = 0.0
    start_time = time.time()
    log = {}

    for ne in range(cfg.train.num_epochs):
        t0 = time.time()
        key, epoch_key = jax.random.split(key)
        log = evaluator.run_evaluation(training_state, log)
        training_state, env_state, buffer_state, metrics, key = trainer.epoch(
            training_state, env_state, buffer_state, epoch_key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_time = time.time() - t0
        training_walltime += epoch_time
        sps = (
            cfg.train.env_steps_per_actor_step * cfg.train.num_training_steps_per_epoch
        ) / max(epoch_time, 1e-6)

        log.update(
            {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": int(training_state.env_steps),
                "training/gradsteps": int(training_state.gradient_steps),
                **{
                    f"training/{k}": float(v) if np.ndim(v) == 0 else v
                    for k, v in metrics.items()
                },
            }
        )
        print(
            f"epoch {ne}/{cfg.train.num_epochs} - "
            f"envsteps={log['training/envsteps']} - sps={log['training/sps']:.1f}",
            flush=True,
        )

        if cfg.train.checkpoint and save_path is not None:
            if ne < 5 or ne >= cfg.train.num_epochs - 5 or ne % 10 == 0:
                save_params(
                    str(save_path / f"step_{int(training_state.env_steps)}.pkl"),
                    checkpoint_params(training_state),
                )

        if cfg.train.track:
            wandb.log(log, step=ne)
            if trigger_sync is not None:
                trigger_sync()
        print(f"walltime: {(time.time()-start_time)/3600:.3f} hours", flush=True)

    log = evaluator.run_evaluation(training_state, log)
    if cfg.train.track:
        wandb.log(log, step=ne)
        if trigger_sync is not None:
            trigger_sync()

    # --------------------------------------------------------------- outputs ---
    if cfg.train.checkpoint and save_path is not None:
        save_params(str(save_path / "final.pkl"), checkpoint_params(training_state))
        with open(str(save_path / "args.pkl"), "wb") as f:
            pickle.dump(cfg, f)

    if cfg.train.capture_vis and save_path is not None:
        render_rollouts(cfg, agent, training_state, env.sys, save_path)


if __name__ == "__main__":
    main()
