"""The SRL policy: a tanh-squashed diagonal-Gaussian actor over [state, goal]."""

from __future__ import annotations

from typing import Any

from flax import linen as nn

from srl.utils.mlp import build_mlp, dense


class Actor(nn.Module):
    """Diagonal-Gaussian actor with tanh-squashed log-std."""

    action_dim: int
    network_width: int = 512
    network_depth: int = 4
    use_residuals: bool = True
    activation: Any = nn.swish

    # Kept as plain class-level constants — not per-instance config.
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, x):
        x = build_mlp(
            self.network_width, self.network_depth, self.activation, self.use_residuals
        )(x)

        mean = dense(self.action_dim)(x)
        log_std = dense(self.action_dim)(x)

        # Clamp log_std to [LOG_STD_MIN, LOG_STD_MAX] via tanh (SpinUp / Yarats)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )

        return mean, log_std
