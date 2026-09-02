"""The online training loop: collect -> relabel -> SGD.

`make_trainer` closes over everything that is static for a run (config, agent,
environment, buffer, relabeling function) and returns the two entry points the
training script drives: a buffer prefill and a jitted training epoch.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Trainer(NamedTuple):
    """Entry points into the online loop."""

    prefill: Callable
    """(training_state, env_state, buffer_state, key) -> same, after the prefill."""

    epoch: Callable
    """(training_state, env_state, buffer_state, key) -> (..., metrics, key). Jitted."""


def make_trainer(cfg, agent, env, replay_buffer, relabel_fn, relabel_cfg) -> Trainer:
    env_steps_per_actor_step = cfg.train.env_steps_per_actor_step
    batch_size = cfg.train.batch_size

    # ------------------------------------------------------------ collection ---
    def get_experience(training_state, env_state, buffer_state, key):
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if cfg.train.expl_actor:
                env_state, transition = agent.actor_step(
                    training_state,
                    env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "seed"),
                )
            else:
                env_state, transition = agent.deterministic_actor_step(
                    training_state,
                    env,
                    env_state,
                    extra_fields=("truncation", "seed"),
                )
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(
            f, (env_state, key), (), length=cfg.train.unroll_length
        )
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        def f(carry, _):
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state, env_state, buffer_state, key
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )
            return (training_state, env_state, buffer_state, new_key), ()

        (training_state, env_state, buffer_state, key), _ = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=cfg.train.num_prefill_actor_steps,
        )
        return training_state, env_state, buffer_state, key

    # ------------------------------------------------------- batch assembly ---
    def sample_and_relabel(buffer_state, sampling_key, perm_key, sgd_batches_key):
        """Sample trajectories, relabel them, and cut them into SGD batches."""
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        for _ in range(1, cfg.train.num_episodes_per_env):
            buffer_state, extra = replay_buffer.sample(buffer_state)
            transitions = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x, y], axis=0), transitions, extra
            )

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(relabel_fn, in_axes=(None, 0, 0))(
            relabel_cfg, transitions, batch_keys
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
        )
        perm = jax.random.permutation(perm_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[perm], transitions)
        n_full = len(transitions.observation) // batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[: n_full * batch_size], transitions
        )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]), transitions
        )

        if not cfg.train.use_all_batches:
            nb = transitions.observation.shape[0]
            sel = jax.random.permutation(sgd_batches_key, nb)[
                : cfg.train.num_sgd_batches_per_training_step
            ]
            transitions = jax.tree_util.tree_map(lambda x: x[sel], transitions)

        return buffer_state, transitions

    # --------------------------------------------------------- training loop ---
    def training_step(training_state, env_state, buffer_state, key):
        (
            experience_key1,
            experience_key2,
            sampling_key,
            training_key,
            sgd_batches_key,
        ) = jax.random.split(key, 5)

        env_state, buffer_state = get_experience(
            training_state, env_state, buffer_state, experience_key1
        )
        training_state = training_state.replace(
            env_steps=training_state.env_steps + env_steps_per_actor_step
        )

        buffer_state, transitions = sample_and_relabel(
            buffer_state, sampling_key, experience_key2, sgd_batches_key
        )

        # The critic updates every step; the actor only every `actor_update_every`.
        actor_update_every = jnp.array(
            max(1, int(cfg.train.actor_update_every)), jnp.int32
        )
        do_actor = (training_state.gradient_steps % actor_update_every) == 0

        def run_critic_only(_):
            (ts, _), metrics = jax.lax.scan(
                agent.sgd_step_critic_only, (training_state, training_key), transitions
            )
            return ts, metrics

        def run_full(_):
            (ts, _), metrics = jax.lax.scan(
                agent.sgd_step_full, (training_state, training_key), transitions
            )
            return ts, metrics

        training_state, metrics = jax.lax.cond(
            ~do_actor, run_critic_only, run_full, operand=None
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        def f(carry, _):
            ts, es, bs, k = carry
            k, nk = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, nk), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            jnp.arange(
                cfg.train.num_training_steps_per_epoch
                * cfg.train.training_steps_multiplier
            ),
        )
        return training_state, env_state, buffer_state, metrics, key

    return Trainer(prefill=prefill_replay_buffer, epoch=training_epoch)
