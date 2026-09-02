"""Flax/JAX plumbing: initializers, the training-state containers and checkpoint I/O."""

from __future__ import annotations

import pickle
from typing import Any, NamedTuple

import flax
import jax.numpy as jnp
from etils import epath
from flax.training.train_state import TrainState
from jax.nn.initializers import variance_scaling, zeros


def default_init():
    """Lecun-uniform kernel initializer."""
    return variance_scaling(1 / 3, "fan_in", "uniform")


def bias_init():
    """Zero bias initializer."""
    return zeros


class Transition(NamedTuple):
    """A single environment transition (batched over envs and/or time)."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


@flax.struct.dataclass
class TrainingState:
    """Everything the learner carries across gradient steps."""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


def checkpoint_params(training_state: TrainingState):
    """Params tuple written to disk, in the order expected when reloading."""
    return (
        training_state.alpha_state.params,
        training_state.actor_state.params,
        training_state.critic_state.params,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        return pickle.loads(fin.read())


def save_params(path: str, params: Any) -> None:
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))
