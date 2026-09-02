"""Evaluation rollouts and the metrics they produce."""

from __future__ import annotations

import time

import jax
import numpy as np
from brax import envs

# Episode metrics reported when the environment provides them.
EPISODE_METRICS = (
    "reward",
    "success",
    "success_easy",
    "success_hard",
    "dist",
    "distance_from_origin",
)


def generate_unroll(
    actor_step, training_state, env, env_state, unroll_length, extra_fields=()
):
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state = carry
        nstate, transition = actor_step(
            training_state, env, state, extra_fields=extra_fields
        )
        return nstate, transition

    final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
    return final_state, data


class Evaluator:
    """Runs one full-length episode per eval env and aggregates episode metrics."""

    def __init__(self, actor_step, eval_env, num_eval_envs, episode_length, key):
        self._key = key
        self._eval_walltime = 0.0

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                actor_step,
                training_state,
                eval_env,
                eval_first_state,
                unroll_length=episode_length,
            )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, training_state, training_metrics, aggregate_episodes=True):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t

        episode_metrics = eval_metrics.episode_metrics
        metrics = {
            f"eval/episode_{name}": (
                np.mean(episode_metrics[name])
                if aggregate_episodes
                else episode_metrics[name]
            )
            for name in EPISODE_METRICS
            if name in episode_metrics
        }

        # In how many envs was there at least one successful step?
        if "success" in episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(
                episode_metrics["success"] > 0.0
            )

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time

        return {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}


def make_evaluator(cfg, agent, eval_env, eval_env_key, key):
    """Build the evaluator for the configured eval policy.

    Returns `(evaluator, key)`; `key` is only consumed when `train.eval_actor` asks for
    a stochastic evaluation policy.
    """
    if cfg.train.eval_actor:
        key, eval_actor_key = jax.random.split(key)

        def actor_step(training_state, env, env_state, extra_fields):
            return agent.actor_step(
                training_state, env, env_state, eval_actor_key, extra_fields
            )
    else:
        actor_step = agent.deterministic_actor_step

    evaluator = Evaluator(
        actor_step,
        eval_env,
        num_eval_envs=cfg.env.num_eval_envs,
        episode_length=cfg.env.episode_length,
        key=eval_env_key,
    )
    return evaluator, key
