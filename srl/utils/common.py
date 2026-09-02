"""Run-level helpers: config derivation, seeding, naming, logging and W&B setup."""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
import wandb_osh
from flax import linen as nn
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.rule import Rule
from rich.syntax import Syntax
from wandb_osh.hooks import TriggerWandbSyncHook

ACTIVATIONS = {
    "swish": nn.swish,
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "elu": nn.elu,
}


def get_activation(name: str):
    """Return an activation function by name."""
    if name.lower() not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}."
        )
    return ACTIVATIONS[name.lower()]


def calculate_steps(cfg: DictConfig) -> DictConfig:
    """Fill in the `train.*` fields that are derived from the step budget."""
    train = cfg.train
    env_cfg = cfg.env
    cfg.train.env_steps_per_actor_step = train.unroll_length * env_cfg.num_envs
    cfg.train.num_prefill_env_steps = train.min_replay_size * env_cfg.num_envs
    cfg.train.num_prefill_actor_steps = train.min_replay_size // train.unroll_length
    numerator = train.total_env_steps - train.num_prefill_env_steps
    denominator = train.num_epochs * train.env_steps_per_actor_step
    cfg.train.num_training_steps_per_epoch = numerator // denominator
    return cfg


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_run_name(cfg: DictConfig) -> str:
    return (
        f"srl_{cfg.env.env_id}_seed:{cfg.seed}_B:{cfg.train.batch_size}_nenvs:{cfg.env.num_envs}"
        f"_H:{cfg.agent.surv_horizon}_K:{cfg.agent.num_bins}_m:{cfg.agent.m}_ms:{cfg.agent.m_stride}"
        f"_criticwidth:{cfg.architecture.critic_network_width}"
        f"_actorwidth:{cfg.architecture.actor_network_width}"
        f"_criticdepth:{cfg.architecture.critic_network_depth}"
        f"_actordepth:{cfg.architecture.actor_network_depth}"
        f"_nepoch:{cfg.train.num_epochs}"
    )


def init_wandb(cfg: DictConfig, run_name: str):
    """Start the W&B run. Returns the offline sync hook, or None."""
    if not cfg.train.track:
        return None

    wandb.init(
        project=cfg.wandb.wandb_project_name,
        entity=cfg.wandb.wandb_entity if cfg.wandb.wandb_entity else None,
        mode=cfg.wandb.wandb_mode,
        group=cfg.wandb.wandb_group,
        dir=cfg.wandb.wandb_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        name=run_name,
        save_code=True,
    )
    if cfg.wandb.wandb_mode == "offline":
        wandb_osh.set_log_level("ERROR")
        return TriggerWandbSyncHook()
    return None


def make_save_dir(cfg: DictConfig, run_name: str):
    """Directory for checkpoints and rollout videos, or None if neither is enabled."""
    if not (cfg.train.checkpoint or cfg.train.capture_vis):
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = Path(cfg.wandb.wandb_dir) / "runs" / f"{run_name}_{stamp}"
    os.makedirs(save_path, exist_ok=True)
    return save_path


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """Pretty-print the Hydra configuration using Rich."""
    console = Console()
    syntax = Syntax(
        OmegaConf.to_yaml(cfg, resolve=resolve),
        "yaml",
        theme="monokai",
        line_numbers=False,
    )
    console.print()
    console.print(Rule("[bold blue] CONFIGURATION "))
    console.print(syntax)
    console.print(Rule())
    console.print()
