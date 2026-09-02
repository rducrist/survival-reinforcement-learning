"""Hindsight relabeling: turn sampled trajectories into survival-labelled transitions.

A relabeled transition carries `(surv_is_event, surv_time, surv_censor_time)`: a
goal-reach is an *event* at time tau, and a trajectory that never reaches its
hindsight goal is *right-censored* at the horizon H.

There are two implementations, selected by `make_relabel_fn`:

* `flatten_srl_single_fn` (m == 1) — the critic is conditioned on a single goal.
* `flatten_srl_multi_fn`  (m  > 1) — the critic is conditioned on an m-anchor goal
  sequence and an event requires the whole sequence to match.

The two are bitwise-identical at m == 1; they are kept separate so the m == 1 path
stays exactly the code that produced the published results.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_srl_single_fn(buffer_config, transition, sample_key):
    """SRL relabeling for m = 1: the critic is conditioned on a single goal.

    buffer_config: (gamma, obs_dim, g_start, g_end, H, eps, p_same, p_cur, p_random)
    """
    (
        gamma,
        obs_dim,
        g_start,
        g_end,
        H,
        eps,
        p_same,
        p_cur,
        p_random,
    ) = buffer_config

    state = transition.observation[:, :obs_dim]  # (T, obs_dim)
    T = state.shape[0]

    seeds = jnp.squeeze(transition.extras["state_extras"]["seed"]).astype(jnp.int32)

    key_strat, key_future, key_rand = jax.random.split(sample_key, 3)

    # indices
    t_idx = jnp.arange(T)[:, None]          # (T, 1)
    j_idx = jnp.arange(T)[None, :]            # (1, T)
    dt = j_idx - t_idx                        # (T, T)

    same_ep_mat = (seeds[:T, None] == seeds[None, :])  # (T, T)

    # -------------------------
    # Strategy B: current goal
    # -------------------------
    goal_cur = state[:T, g_start:g_end]  # (T, goal_dim)

    # -------------------------
    # Strategy A: future goal (FIXED)
    # -------------------------
    future_mask = same_ep_mat & (dt >= 1)  # strictly future, same episode
    valid_future = jnp.any(future_mask, axis=1)  # (T,)

    # Only compute gamma**dt where dt is valid; otherwise 0.0
    dt_f = dt.astype(jnp.float32)
    w = jnp.where(future_mask, jnp.power(gamma, dt_f), 0.0)  # (T, T)
    future_logits = jnp.log(w + 1e-10)

    future_idx = jax.random.categorical(key_future, future_logits, axis=1)  # (T,)
    goal_future = state[future_idx, g_start:g_end]

    # Edge-case: if no valid future exists, fall back to current
    goal_future = jnp.where(valid_future[:, None], goal_future, goal_cur)

    # -------------------------
    # Strategy C: random goal
    # -------------------------
    rand_idx = jax.random.randint(key_rand, (T,), 0, T)
    goal_rand = state[rand_idx, g_start:g_end]

    # -------------------------
    # Strategy selection
    # -------------------------
    probs = jnp.array([p_same, p_cur, p_random], dtype=jnp.float32)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    selection = jax.random.choice(key_strat, 3, shape=(T,), p=probs)

    goal = jnp.where(
        selection[:, None] == 0,
        goal_future,
        jnp.where(selection[:, None] == 1, goal_cur, goal_rand),
    )

    # -------------------------
    # Distance / survival labeling
    # -------------------------
    s_match = state[:, g_start:g_end]  # achieved goal projection (T, goal_dim)

    dmat = jnp.linalg.norm(goal[:, None, :] - s_match[None, :, :], axis=-1)  # (T, T)
    d0 = jnp.linalg.norm(s_match[:T] - goal, axis=-1)  # (Tm1,)

    hit_future = same_ep_mat & (dt >= 1) & (dmat <= eps)

    big = jnp.array(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    tau_hat = jnp.min(jnp.where(hit_future, dt.astype(jnp.int32), big), axis=1)

    is_event = (d0 <= eps) | (tau_hat < big)
    tau = jnp.where(d0 <= eps, 0, tau_hat)

    sliced_extras = jax.tree_util.tree_map(lambda x: x[:T], transition.extras)
    new_extras = {
        **sliced_extras,
        "surv_is_event": is_event.astype(jnp.float32),
        "surv_time": jnp.clip(tau, 0, int(H)).astype(jnp.int32),
        "surv_censor_time": (jnp.ones(T)*int(H)).astype(jnp.int32),
        "surv_valid": jnp.ones(T).astype(jnp.float32),
        "goal_strategy": selection,
        "valid_future": valid_future.astype(jnp.float32),
    }

    return transition._replace(
        observation=jnp.concatenate([state[:T], goal], axis=-1),
        action=transition.action[:T],
        reward=transition.reward[:T],
        discount=transition.discount[:T],
        extras=new_extras,
    )


@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_srl_multi_fn(buffer_config, transition, sample_key):
    """SRL relabeling for m > 1: the critic is conditioned on an m-anchor goal sequence.

    The actor observation stays [state, first_goal_anchor]; the critic goal becomes
    [g_u, g_{u+stride}, ..., g_{u+(m-1)stride}].

    Survival labels:
        event = first future index j such that the entire m-anchor goal sequence
                at j matches the selected hindsight goal sequence within eps.

    Because each start index needs a full m-anchor suffix, this drops the last
    (m - 1) * stride timesteps of every sampled trajectory.

    buffer_config: (gamma, obs_dim, g_start, g_end, H, eps,
                    p_same, p_cur, p_random, m, m_stride)

    At m = 1 this is bitwise-identical to `flatten_srl_single_fn`; the two are kept
    separate so the m = 1 path stays exactly the code that produced the SRL results.
    """
    (
        gamma,
        obs_dim,
        g_start,
        g_end,
        H,
        eps,
        p_same,
        p_cur,
        p_random,
        m,
        m_stride,
    ) = buffer_config

    state_all = transition.observation[:, :obs_dim]
    full_T = state_all.shape[0]
    goal_dim = g_end - g_start
    max_offset = (m - 1) * m_stride

    # Keep only starts that can support a full k-anchor suffix.
    T = full_T - max_offset
    state = state_all[:T]

    seeds = jnp.squeeze(transition.extras["state_extras"]["seed"]).astype(jnp.int32)

    key_strat, key_future, key_rand = jax.random.split(sample_key, 3)

    def build_suffix_valid(seeds_vec):
        valid = jnp.arange(full_T) < (full_T - max_offset)
        for i in range(1, m):
            off = i * m_stride
            cur_valid = jnp.concatenate(
                [
                    seeds_vec[:-off] == seeds_vec[off:],
                    jnp.zeros((off,), dtype=bool),
                ],
                axis=0,
            )
            valid = valid & cur_valid
        return valid

    suffix_valid = build_suffix_valid(seeds)

    t_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(full_T)[None, :]
    dt = j_idx - t_idx

    same_ep_tj = seeds[:T, None] == seeds[None, :]
    future_mask = same_ep_tj & (dt >= 1) & suffix_valid[None, :]
    valid_future = jnp.any(future_mask, axis=1)

    dt_f = dt.astype(jnp.float32)
    w = jnp.where(future_mask, jnp.power(gamma, dt_f), 0.0)
    future_logits = jnp.log(w + 1e-10)

    future_idx = jax.random.categorical(key_future, future_logits, axis=1)
    future_idx = jnp.where(valid_future, future_idx, jnp.arange(T))

    goal_all = state_all[:, g_start:g_end]

    def gather_mode_from_start(start_idx):
        anchors = []
        for i in range(m):
            idx = start_idx + i * m_stride
            anchors.append(goal_all[idx])
        return jnp.concatenate(anchors, axis=-1)

    future_mode_goal = jax.vmap(gather_mode_from_start)(future_idx)
    cur_mode_goal = jax.vmap(gather_mode_from_start)(jnp.arange(T))

    rand_idx = jax.random.randint(key_rand, (T,), 0, T)
    goal_rand = state[rand_idx, g_start:g_end]
    rand_mode_goal = jnp.tile(goal_rand, (1, m))

    probs = jnp.array([p_same, p_cur, p_random], dtype=jnp.float32)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    selection = jax.random.choice(key_strat, 3, shape=(T,), p=probs)

    mode_goal = jnp.where(
        selection[:, None] == 0,
        future_mode_goal,
        jnp.where(selection[:, None] == 1, cur_mode_goal, rand_mode_goal),
    )

    point_goal = mode_goal[:, :goal_dim]

    # Build all candidate mode sequences starting at j = 0..T-1
    candidate_mode_goal = jax.vmap(gather_mode_from_start)(jnp.arange(T))  # (T, k*gd)

    mode_goal_reshaped = mode_goal.reshape(T, m, goal_dim)
    candidate_reshaped = candidate_mode_goal.reshape(T, m, goal_dim)

    # Compare every selected mode (row t) with every candidate start j
    diffs = mode_goal_reshaped[:, None, :, :] - candidate_reshaped[None, :, :, :]
    dseq = jnp.linalg.norm(diffs, axis=-1)  # (T, T, k)
    match_seq = jnp.all(dseq <= eps, axis=-1)  # (T, T)

    same_ep_tt = seeds[:T, None] == seeds[:T][None, :]
    dt_tt = jnp.arange(T)[None, :] - jnp.arange(T)[:, None]

    future_match = match_seq & same_ep_tt & (dt_tt >= 1)

    big = jnp.array(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    tau_hat = jnp.min(jnp.where(future_match, dt_tt.astype(jnp.int32), big), axis=1)

    d0_match = jnp.diag(match_seq)
    is_event = d0_match | (tau_hat < big)
    tau = jnp.where(d0_match, 0, tau_hat)

    sliced_extras = jax.tree_util.tree_map(lambda x: x[:T], transition.extras)
    new_extras = {
        **sliced_extras,
        "surv_is_event": is_event.astype(jnp.float32),
        "surv_time": jnp.clip(tau, 0, int(H)).astype(jnp.int32),
        "surv_censor_time": (jnp.ones(T) * int(H)).astype(jnp.int32),
        "surv_valid": jnp.ones(T, dtype=jnp.float32),
        "goal_strategy": selection,
        "valid_future": valid_future.astype(jnp.float32),
        "mode_goal": mode_goal.astype(jnp.float32),
        "mode_valid": jnp.ones(T, dtype=jnp.float32),
        "point_goal": point_goal.astype(jnp.float32),
    }

    return transition._replace(
        observation=jnp.concatenate([state, point_goal], axis=-1),
        action=transition.action[:T],
        reward=transition.reward[:T],
        discount=transition.discount[:T],
        extras=new_extras,
    )


def make_relabel_fn(cfg, state_dim: int, goal_start_idx: int, goal_end_idx: int):
    """Pick the relabeling implementation for `agent.m` and freeze its static config.

    Returns `(relabel_fn, relabel_cfg)`; `relabel_cfg` is a hashable tuple passed as a
    static argument, so the choice is made once, at trace time.
    """
    m = int(cfg.agent.m)
    m_stride = int(cfg.agent.m_stride)

    relabel_cfg = (
        float(cfg.agent.discount),
        int(state_dim),
        int(goal_start_idx),
        int(goal_end_idx),
        int(cfg.agent.surv_horizon),
        float(cfg.agent.surv_eps),
        float(cfg.agent.p_same),
        float(cfg.agent.p_cur),
        float(cfg.agent.p_random),
    )
    if m > 1:
        return flatten_srl_multi_fn, relabel_cfg + (m, m_stride)
    return flatten_srl_single_fn, relabel_cfg
