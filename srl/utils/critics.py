"""The SRL hazard critic: (state, action, goal) -> discrete-time hazard logits."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn

from srl.utils.mlp import ResidualBlock, build_mlp, dense


class _EncoderBase(nn.Module):
    """Shared hyper-parameters for all encoders."""

    width: int = 512
    depth: int = 8
    embed_dim: int = 128
    act: Any = nn.swish
    use_residual: bool = True

    def _project(self, h):
        """Project backbone output to `embed_dim` and apply LayerNorm."""
        z = dense(self.embed_dim)(h)
        return nn.LayerNorm()(z)


class EncSA(_EncoderBase):
    """State-action encoder: (s, a) -> z."""

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([s, a], axis=-1)
        h = build_mlp(
            self.width, self.depth, self.act, self.use_residual, name="backbone"
        )(x)
        return self._project(h)


class EncG(_EncoderBase):
    """Goal encoder: g -> z. For m > 1, `g` is the concatenated anchor sequence."""

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        h = build_mlp(
            self.width, self.depth, self.act, self.use_residual, name="backbone"
        )(g)
        return self._project(h)


class HazardHead(nn.Module):
    """Hazard head: (z_sa, z_g) -> (logit0, logits over bins).

    1. FiLM modulation of z_sa by z_g
    2. richer fusion: [z_sa_film, z_g, diff, prod]
    3. low-rank temporal basis for the logits over bins
    """

    num_bins: int
    embed_dim: int = 128
    trunk_width: int = 512
    trunk_depth: int = 8
    rank: int = 64
    act: Any = nn.swish

    @nn.compact
    def __call__(self, z_sa, z_g):
        film = dense(2 * self.embed_dim, name="film")(z_g)
        gamma, beta = film[..., : self.embed_dim], film[..., self.embed_dim :]

        z_sa_film = (1.0 + 0.1 * gamma) * z_sa + 0.1 * beta
        diff = z_sa_film - z_g
        prod = z_sa_film * z_g
        x = jnp.concatenate([z_sa_film, z_g, diff, prod], axis=-1)

        h = dense(self.trunk_width, name="input_proj")(x)
        h = nn.LayerNorm(name="input_ln")(h)
        h = self.act(h)

        for i in range(self.trunk_depth // 4):
            h = ResidualBlock(
                width=self.trunk_width,
                depth=4,
                activation=self.act,
                name=f"trunk_block_{i}",
            )(h)

        logit0 = dense(1, name="logits_0")(h).squeeze(-1)

        u = dense(self.rank, name="time_coeff")(h)

        B = self.param(
            "time_basis",
            nn.initializers.normal(0.02),
            (self.num_bins, self.rank),
        )
        b = self.param(
            "time_bias",
            nn.initializers.zeros,
            (self.num_bins,),
        )

        logits = u @ B.T + b  # (batch, num_bins)

        return logit0, logits


class HazardCritic(nn.Module):
    """Full hazard critic: (state, action, goal) -> (logit0, bin logits)."""

    num_bins: int
    embed_dim: int = 128
    enc_width: int = 512
    enc_depth: int = 8
    trunk_width: int = 512
    trunk_depth: int = 16
    act: Any = nn.swish
    use_residuals: bool = True

    @nn.compact
    def __call__(self, state, action, goal):
        enc_kwargs = dict(
            width=self.enc_width,
            depth=self.enc_depth,
            embed_dim=self.embed_dim,
            act=self.act,
            use_residual=self.use_residuals,
        )
        z_sa = EncSA(**enc_kwargs, name="enc_sa")(state, action)
        z_g = EncG(**enc_kwargs, name="enc_g")(goal)

        return HazardHead(
            num_bins=self.num_bins,
            embed_dim=self.embed_dim,
            trunk_width=self.trunk_width,
            trunk_depth=self.trunk_depth,
            act=self.act,
            name="head",
        )(z_sa, z_g)
