"""The SRL agent: hazard critic + entropy-regularised actor.

The critic is trained by maximum likelihood on survival labels — no Bellman backup and
no target network. The actor maximises the value recovered in closed form from the
hazard (`srl.utils.survival.create_perbin_value_fn`).

`SRLAgent` is a frozen container of *static* objects (the network definitions, the
survival functions, and the scalars read off the config). All mutable learner state
lives in `TrainingState`, which every update takes in and returns, so the agent can be
closed over by jitted code.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from srl.utils.actors import Actor
from srl.utils.common import get_activation
from srl.utils.critics import HazardCritic
from srl.utils.flax_utils import TrainingState, Transition
from srl.utils.survival import (
    create_log_bins,
    create_perbin_nll_fn,
    create_perbin_value_fn,
)


@dataclasses.dataclass(frozen=True)
class SRLAgent:
    """Networks, survival math and hyper-parameters for one SRL run."""

    actor: Actor
    critic: HazardCritic
    value_from_logits: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    nll_from_logits: Callable[..., jnp.ndarray]

    state_dim: int
    m: int
    surv_horizon: int
    batch_size: int
    target_entropy: float
    disable_entropy: bool

    # ------------------------------------------------------------------ setup ---
    @classmethod
    def create(
        cls,
        cfg,
        obs_size: int,
        action_dim: int,
        state_dim: int,
        goal_dim: int,
        actor_key,
        critic_key,
    ) -> tuple["SRLAgent", TrainingState]:
        """Build the networks and the initial `TrainingState`."""
        m = int(cfg.agent.m)
        if m < 1:
            raise ValueError(f"agent.m must be >= 1, got {m}")

        activation = get_activation(cfg.architecture.activation)

        actor = Actor(
            action_dim=action_dim,
            activation=activation,
            network_width=cfg.architecture.actor_network_width,
            network_depth=cfg.architecture.actor_network_depth,
            use_residuals=bool(cfg.architecture.use_residuals),
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, jnp.ones((1, obs_size))),
            tx=optax.adam(learning_rate=cfg.optimizer.actor_lr),
        )

        print(
            f"Training the Survival Reinforcement Learning (SRL) agent with m={m}",
            flush=True,
        )

        # Bin edges are geometric over [0, H] and de-duplicated, so the actual bin
        # count can be smaller than the requested agent.num_bins.
        bins, real_num_bins = create_log_bins(cfg.agent.surv_horizon, cfg.agent.num_bins)
        print(f"real_num_bins: {real_num_bins}", flush=True)

        critic = HazardCritic(
            num_bins=real_num_bins,
            embed_dim=cfg.agent.embed_dim,
            enc_depth=cfg.architecture.critic_network_depth // 2,
            enc_width=cfg.architecture.critic_network_width,
            trunk_width=cfg.architecture.critic_network_width,
            trunk_depth=cfg.architecture.critic_network_depth,
            use_residuals=bool(cfg.architecture.use_residuals),
            act=activation,
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(
                critic_key,
                jnp.ones((1, state_dim)),
                jnp.ones((1, action_dim)),
                jnp.ones((1, goal_dim * m)),
            ),
            tx=optax.adam(learning_rate=cfg.optimizer.critic_lr),
        )

        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.array(0.0, dtype=jnp.float32)},
            tx=optax.adam(learning_rate=cfg.optimizer.alpha_lr),
        )

        agent = cls(
            actor=actor,
            critic=critic,
            value_from_logits=create_perbin_value_fn(bins, float(cfg.agent.discount)),
            nll_from_logits=create_perbin_nll_fn(bins),
            state_dim=int(state_dim),
            m=m,
            surv_horizon=int(cfg.agent.surv_horizon),
            batch_size=int(cfg.train.batch_size),
            target_entropy=-cfg.agent.entropy_param * action_dim,
            disable_entropy=bool(cfg.agent.disable_entropy),
        )
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )
        return agent, training_state

    # ------------------------------------------------------------- acting ------
    def sample_action(self, actor_params, obs, key):
        """Tanh-squashed Gaussian sample plus its log-probability."""
        mean, log_std = self.actor.apply(actor_params, obs)
        std = jnp.exp(log_std)
        x = mean + std * jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        a = nn.tanh(x)
        logp = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=std)
        logp = logp - jnp.log(1.0 - a**2 + 1e-6)
        logp = jnp.sum(logp, axis=-1)
        return a, logp

    def actor_step(self, training_state, env, env_state, key, extra_fields):
        """One stochastic environment step, returned as a `Transition`."""
        a, _ = self.sample_action(training_state.actor_state.params, env_state.obs, key)
        return self._step(env, env_state, a, extra_fields)

    def deterministic_actor_step(self, training_state, env, env_state, extra_fields):
        """One environment step at the distribution mode."""
        mean, _ = self.actor.apply(training_state.actor_state.params, env_state.obs)
        return self._step(env, env_state, nn.tanh(mean), extra_fields)

    @staticmethod
    def _step(env, env_state, action, extra_fields):
        nstate = env.step(env_state, action)
        state_extras = {x: env_state.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=action,
            reward=nstate.reward,
            discount=1.0 - nstate.done,
            extras={"state_extras": state_extras},
        )

    # ------------------------------------------------------------- learning ----
    def update_critic(self, transitions, training_state, key):
        """Maximum-likelihood step on the grouped-time survival NLL."""
        obs = transitions.observation
        state = obs[:, : self.state_dim]
        action = transitions.action
        # For m > 1 the critic is supervised on the relabeled m-anchor sequence; for
        # m == 1 on the single goal relabeling already concatenated into the observation.
        goal = (
            transitions.extras["mode_goal"] if self.m > 1 else obs[:, self.state_dim :]
        )

        is_event = transitions.extras["surv_is_event"]
        tau = jnp.clip(transitions.extras["surv_time"], 0, self.surv_horizon).astype(
            jnp.int32
        )
        censor = jnp.clip(
            transitions.extras["surv_censor_time"], 0, self.surv_horizon
        ).astype(jnp.int32)
        valid = (transitions.extras["surv_valid"] > 0).astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(valid), 1.0)

        def loss_fn(params):
            logits = self.critic.apply(params, state, action, goal)
            nll = self.nll_from_logits(logits[0], logits[1], is_event, tau, censor)
            loss = jnp.sum(valid * nll) / denom
            return loss, {
                "critic_loss": loss,
                "event_frac": jnp.mean(is_event),
                "mean_tau": jnp.mean(tau.astype(jnp.float32)),
            }

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            training_state.critic_state.params
        )
        new_critic_state = training_state.critic_state.apply_gradients(grads=grads)
        return training_state.replace(critic_state=new_critic_state), info

    def update_actor_and_alpha(self, transitions, training_state, key):
        """Maximise the survival value, and adapt the entropy temperature."""
        transitions = jax.tree_util.tree_map(
            lambda x: x[: self.batch_size], transitions
        )
        obs = transitions.observation

        def actor_loss_fn(actor_params, critic_params, log_alpha):
            a, logp = self.sample_action(actor_params, obs, key)
            state = obs[:, : self.state_dim]
            # At act time the policy only sees a point goal, so for m > 1 we query the
            # critic with the constant sequence [g, ..., g]. At m == 1 the tile is a
            # no-op (verified bitwise-identical to indexing the goal directly).
            goal = jnp.tile(obs[:, self.state_dim :], (1, self.m))
            logits = self.critic.apply(critic_params, state, a, goal)
            q = self.value_from_logits(logits[0], logits[1])
            if self.disable_entropy:
                loss = -jnp.mean(q)
            else:
                loss = jnp.mean(jnp.exp(log_alpha) * logp - q)
            info = {
                "actor_loss": loss,
                "q_mean": jnp.mean(q),
                "q_min": jnp.min(q),
                "q_max": jnp.max(q),
                "sample_entropy": jnp.mean(-logp),
            }
            return loss, (logp, info)

        (loss, (logp, info)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            training_state.actor_state.params,
            training_state.critic_state.params,
            training_state.alpha_state.params["log_alpha"],
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=grads)

        def alpha_loss_fn(alpha_params):
            return jnp.mean(
                jnp.exp(alpha_params["log_alpha"])
                * jax.lax.stop_gradient(-logp - self.target_entropy)
            )

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
            training_state.alpha_state.params
        )
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grads)
        return training_state.replace(
            actor_state=new_actor_state, alpha_state=new_alpha_state
        ), {
            **info,
            "alpha_loss": alpha_loss,
            "log_alpha": new_alpha_state.params["log_alpha"],
        }

    # ------------------------------------------------------------ sgd steps ----
    @staticmethod
    def zero_metrics() -> dict[str, Any]:
        """Metric skeleton, so both SGD variants report the same keys."""
        z = jnp.array(0.0, dtype=jnp.float32)
        return {
            "actor/actor_loss": z,
            "actor/alpha_loss": z,
            "actor/sample_entropy": z,
            "actor/log_alpha": z,
            "actor/q_max": z,
            "actor/q_mean": z,
            "actor/q_min": z,
            "actor/skipped": z,
            "critic/critic_loss": z,
            "critic/event_frac": z,
            "critic/mean_tau": z,
        }

    def sgd_step_critic_only(self, carry, batch):
        """`lax.scan` body for the steps where the actor update is skipped."""
        training_state, key = carry
        key, critic_key = jax.random.split(key)
        training_state, cinfo = self.update_critic(batch, training_state, critic_key)
        training_state = training_state.replace(
            gradient_steps=training_state.gradient_steps + 1
        )
        metrics = self.zero_metrics()
        metrics.update({f"critic/{k}": v for k, v in cinfo.items()})
        metrics["actor/skipped"] = jnp.array(1.0, dtype=jnp.float32)
        return (training_state, key), metrics

    def sgd_step_full(self, carry, batch):
        """`lax.scan` body for a critic + actor + alpha step."""
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)
        training_state, cinfo = self.update_critic(batch, training_state, critic_key)
        training_state, ainfo = self.update_actor_and_alpha(
            batch, training_state, actor_key
        )
        training_state = training_state.replace(
            gradient_steps=training_state.gradient_steps + 1
        )
        metrics = self.zero_metrics()
        metrics.update({f"critic/{k}": v for k, v in cinfo.items()})
        metrics.update({f"actor/{k}": v for k, v in ainfo.items()})
        metrics["actor/skipped"] = jnp.array(0.0, dtype=jnp.float32)
        return (training_state, key), metrics
