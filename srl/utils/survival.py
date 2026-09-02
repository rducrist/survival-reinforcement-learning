"""Discrete-time survival analysis: bin layout, the PCS value estimator and the NLL.

The critic emits `logit0` (the instant-hit probability p0 = P(T = 0)) plus one hazard
logit per time bin. This module turns those logits into

  * a goal-conditioned value, via the piecewise-constant-survival (PCS) estimator, and
  * a grouped-time negative log-likelihood over `(is_event, tau, censor)` labels.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def create_log_bins(horizon: int, num_bins: int) -> Tuple[jnp.ndarray, int]:
    """
    Create bin boundaries b_0..b_K with b_0=0, b_K=horizon.

    Returns:
      bins: int32 array of shape (K+1,)
      K: number of bins
    """
    if num_bins <= 0 or num_bins >= horizon:
        bins = jnp.arange(horizon + 1, dtype=jnp.int32)
        return bins, horizon

    # Geometric spacing then de-dup.
    ratio = jnp.power(horizon / 1.0, 1.0 / num_bins)
    bins = jnp.round(jnp.power(ratio, jnp.arange(num_bins + 1))).astype(jnp.int32)
    bins = jnp.clip(bins, 0, horizon)

    bins = jnp.unique(bins)
    bins = jnp.unique(jnp.concatenate([jnp.array([0, horizon], dtype=jnp.int32), bins]))
    K = int(bins.shape[0] - 1)
    return bins, K


def get_bin_discount_factors(bins: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Computes D_k = sum_{t=b_k}^{b_{k+1}-1} gamma^t
    """
    bins = jnp.asarray(bins, dtype=jnp.int32)
    starts = bins[:-1].astype(jnp.float32)
    lengths = (bins[1:] - bins[:-1]).astype(jnp.float32)

    if gamma == 1.0:
        return lengths

    gamma_start = jnp.power(gamma, starts)
    gamma_len = jnp.power(gamma, lengths)
    return gamma_start * (1.0 - gamma_len) / (1.0 - gamma)


def create_perbin_value_fn(
    bins: jnp.ndarray,
    gamma: float,
    add_tail: bool = False,
):
    """
    Per-bin hazard model value under approximation:
      S(t) is constant within bin k and equals S(b_k).
    """
    bins = jnp.asarray(bins, dtype=jnp.int32)
    K = int(bins.shape[0] - 1)
    H = int(bins[-1])

    bin_discounts = get_bin_discount_factors(bins, gamma)  # (K,)

    @jax.jit
    def value_from_logits(logit0: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Inputs:
          logit0: (B,)  logit for p0 = P(T=0)
          logits: (B,K) logits for per-bin hazard h_k = sigmoid(logits[:,k])

        Output:
          v: (B,) approximate discounted negative survival sum
        """
        B, K_logits = logits.shape
        assert K_logits == K

        # log(1-p0)
        log1mp0 = jax.nn.log_sigmoid(-logit0)  # (B,)

        # log(1-h_k)
        log1mh = jax.nn.log_sigmoid(-logits)  # (B,K)

        # S at bin starts:
        # log S(b_0) = log(1-p0)
        # log S(b_k) = log(1-p0) + sum_{j<k} log(1-h_j)
        prefix = jnp.cumsum(log1mh, axis=-1)  # (B,K) prefix up to k
        prefix_before = jnp.concatenate(
            [jnp.zeros((B, 1), dtype=logits.dtype), prefix[:, :-1]],
            axis=1,
        )  # (B,K) sum_{j<k} log(1-h_j)

        logS_start = log1mp0[:, None] + prefix_before  # (B,K)
        S_start = jnp.exp(logS_start)  # (B,K)

        # V ≈ - sum_k D_k * S(b_k)
        v = -jnp.sum(S_start * bin_discounts[None, :], axis=-1)

        if add_tail and gamma < 1.0:
            # S(b_K) = (1-p0)*prod_{k=0..K-1}(1-h_k)
            logS_end = log1mp0 + prefix[:, -1]  # (B,)
            S_end = jnp.exp(logS_end)
            v = v - (jnp.power(gamma, H) * S_end) / (1.0 - gamma)

        return v

    return value_from_logits


def create_perbin_nll_fn(bins: jnp.ndarray):
    """
    Per-bin hazard NLL:
      - Each bin k contributes a single Bernoulli hazard h_k
      - Survival updates only across bins:
          S(b_{k+1}) = S(b_k) (1-h_k)


    Convention for labels:
      - tau in [0..H]
        tau=0 means event at time 0 via p0
        tau>0: event belongs to the bin containing tau (side='right')
      - censor in [0..H]
        censor=c means right-censored after surviving up to time c
        In this implementation we only credit survival up to the boundary BEFORE c:
          censor_edge = searchsorted(bins, c, side='left')
          -> survival through bins [0..censor_edge-1] only
    """
    bins = jnp.asarray(bins, dtype=jnp.int32)
    K = int(bins.shape[0] - 1)

    @jax.jit
    def nll_from_logits(
        logit0: jnp.ndarray,  # (B,)
        logits: jnp.ndarray,  # (B,K)
        is_event: jnp.ndarray,  # (B,) float in {0,1}
        tau: jnp.ndarray,  # (B,) int in [0..H]
        censor: jnp.ndarray,  # (B,) int in [0..H]
    ) -> jnp.ndarray:
        B, K_logits = logits.shape
        assert K_logits == K
        bidx = jnp.arange(B)

        # log p0 and log(1-p0)
        logp0 = jax.nn.log_sigmoid(logit0)  # (B,)
        log1mp0 = jax.nn.log_sigmoid(-logit0)  # (B,)

        # per-bin hazard logs
        logh = jax.nn.log_sigmoid(logits)  # log(h_k)
        log1mh = jax.nn.log_sigmoid(-logits)  # log(1-h_k)

        # prefix_before[k] = sum_{j<k} log(1-h_j)
        prefix = jnp.cumsum(log1mh, axis=-1)  # (B,K) sum_{j<=k}
        prefix_before = jnp.concatenate(
            [jnp.zeros((B, 1), dtype=logits.dtype), prefix[:, :-1]],
            axis=1,
        )  # (B,K) sum_{j<k}

        tau = jnp.asarray(tau, dtype=jnp.int32)
        censor = jnp.asarray(censor, dtype=jnp.int32)

        # ---------------- event log-likelihood ----------------
        # tau == 0 uses p0
        ll_tau0 = logp0

        # tau>0: event in bin containing tau
        # bin index: k such that tau in [b_k, b_{k+1}) => searchsorted(..., side="right")-1
        tau_bin = jnp.searchsorted(bins, tau, side="right") - 1
        tau_bin = jnp.clip(tau_bin, 0, K - 1).astype(jnp.int32)

        # log P(event in bin k) = log(1-p0) + sum_{j<k} log(1-h_j) + log(h_k)
        ll_event_pos = log1mp0 + prefix_before[bidx, tau_bin] + logh[bidx, tau_bin]
        ll_event = jnp.where(tau == 0, ll_tau0, ll_event_pos)

        # ---------------- censor log-likelihood ----------------
        # censor at c: only know survived up to boundary before c
        # edge = number of boundaries <= c (side="left" gives first boundary >= c)
        censor_edge = jnp.searchsorted(bins, censor, side="left")  # in [0..K]
        censor_edge = jnp.clip(censor_edge, 0, K).astype(jnp.int32)

        # survival through bins [0..edge-1]:
        # if edge==0 => no bin survival terms, only (1-p0)
        # if edge>0 => add sum_{j<edge} log(1-h_j) = prefix[:, edge-1]
        surv_bins = jnp.where(
            censor_edge == 0,
            jnp.zeros((B,), dtype=logits.dtype),
            prefix[bidx, censor_edge - 1],
        )
        ll_censor = log1mp0 + surv_bins

        ll = jnp.where(is_event > 0.5, ll_event, ll_censor)
        return -ll

    return nll_from_logits
