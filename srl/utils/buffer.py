"""Replay buffer: a FIFO queue of per-env trajectories, sampled as contiguous windows."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp
from brax.training.types import PRNGKey
from jax import flatten_util


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jnp.ndarray
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    insert_count: jnp.ndarray
    key: PRNGKey


class TrajectoryUniformSamplingQueue:
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
        num_envs: int,
        episode_length: int,
    ):
        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))

        self._data_shape = (max_replay_size, num_envs, len(dummy_flatten))
        self._data_dtype = dummy_flatten.dtype
        self.num_envs = num_envs
        self.max_replay_size = max_replay_size
        self.episode_length = episode_length

    def init(self, key):
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            insert_count=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state, samples):
        return self.insert_internal(buffer_state, samples)

    def insert_internal(self, buffer_state, samples):
        update = self._flatten_fn(samples)
        num_new = update.shape[0]
        data = buffer_state.data
        position = buffer_state.insert_position

        roll = jnp.minimum(0, self.max_replay_size - position - num_new)
        data = jax.lax.cond(
            roll < 0, lambda: jnp.roll(data, roll, axis=0), lambda: data
        )
        position = position + roll
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)

        new_position = position + num_new
        new_count = jnp.minimum(
            buffer_state.insert_count + num_new, self.max_replay_size
        )

        return buffer_state.replace(
            data=data, insert_position=new_position, insert_count=new_count
        )

    def sample_internal(self, buffer_state):
        key, key_envs, key_start = jax.random.split(buffer_state.key, 3)

        envs_idxs = jax.random.choice(
            key_envs, jnp.arange(self.num_envs), shape=(self.num_envs,), replace=False
        )

        max_start = jnp.maximum(0, buffer_state.insert_position - self.episode_length)

        start_indices = jax.random.randint(
            key_start,
            shape=(self.num_envs,),
            minval=0,
            maxval=max_start + 1,
        )

        matrix = start_indices[:, jnp.newaxis] + jnp.arange(self.episode_length)

        env_data = buffer_state.data[:, envs_idxs, :]

        def gather_env(data_slice, indices):
            return data_slice[indices]

        batch = jax.vmap(gather_env, in_axes=(1, 0))(env_data, matrix)
        transition_batch = self._unflatten_fn(batch)

        return buffer_state.replace(key=key), transition_batch

    def sample(self, buffer_state):
        return self.sample_internal(buffer_state)

    def size(self, buffer_state):
        return buffer_state.insert_count


def make_replay_buffer(cfg, dummy_transition, key):
    """Build the replay buffer (with its hot paths jitted) and its initial state."""
    buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=cfg.train.max_replay_size,
        dummy_data_sample=dummy_transition,
        num_envs=cfg.env.num_envs,
        episode_length=cfg.env.episode_length,
    )
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer.sample_internal = jax.jit(buffer.sample_internal)
    return buffer, jax.jit(buffer.init)(key)
