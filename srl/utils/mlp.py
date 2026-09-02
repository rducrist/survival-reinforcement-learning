"""Feed-forward building blocks shared by the actor and the critic."""

from __future__ import annotations

import warnings
from typing import Any

from flax import linen as nn

from srl.utils.flax_utils import bias_init, default_init


def dense(features: int, **kwargs) -> nn.Dense:
    """nn.Dense with default initializers."""
    return nn.Dense(
        features, kernel_init=default_init(), bias_init=bias_init(), **kwargs
    )


class _MLPBase(nn.Module):
    """Internal base that holds common hyper-parameters for MLP variants."""

    network_width: int = 256
    network_depth: int = 4
    activation: Any = nn.swish


class MLP(_MLPBase):
    """Simple feedforward MLP without residual connections."""

    @nn.compact
    def __call__(self, x):
        for _ in range(self.network_depth):
            x = dense(self.network_width)(x)
            x = nn.LayerNorm()(x)
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """A single residual block with its own scoped parameters."""

    width: int
    depth: int
    activation: Any

    @nn.compact
    def __call__(self, x):
        identity = x
        if x.shape[-1] != self.width:
            identity = dense(self.width, name="proj")(identity)

        for i in range(self.depth):
            x = dense(self.width, name=f"dense_{i}")(x)
            x = nn.LayerNorm(name=f"norm_{i}")(x)
            x = self.activation(x)
        return x + identity


class ResidualMLP(_MLPBase):
    """MLP with residual blocks (4 layers per block)."""

    @nn.compact
    def __call__(self, x):
        x = dense(self.network_width)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)

        for i in range(self.network_depth // 4):
            x = ResidualBlock(
                width=self.network_width,
                depth=4,
                activation=self.activation,
                name=f"block_{i}",
            )(x)
        return x


def build_mlp(width: int, depth: int, act, use_residual: bool, name: str = None):
    """Factory: returns a ResidualMLP or MLP depending on `use_residual`."""
    kwargs = dict(network_width=width, network_depth=depth, activation=act)
    if name:
        kwargs["name"] = name

    if use_residual and depth < 4:
        warnings.warn(
            f"use_residual=True requires depth >= 4; got depth={depth}. "
            "Falling back to plain MLP.",
            stacklevel=2,
        )

    cls = ResidualMLP if (use_residual and depth >= 4) else MLP
    return cls(**kwargs)
