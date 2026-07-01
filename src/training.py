from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from src.games import NormalFormGame

PyTree = Any


class PlayerState(NamedTuple):
    """All parameter sets and optimizer state for one player.

    Only `params` is updated by gradients. `ema_params` is an exponential
    moving average of `params`. `periodic_params` is a snapshot copied from
    `params` every `periodic_interval` steps.
    """
    params: PyTree
    ema_params: PyTree
    periodic_params: PyTree
    opt_state: optax.OptState


def init_player_state(
    network: nn.Module,
    state_dim: int,
    rng: jax.Array,
    optimizer: optax.GradientTransformation,
) -> PlayerState:
    """Initialise a PlayerState with all three param sets set to the same initial params."""
    params = network.init(rng, jnp.zeros(state_dim))
    opt_state = optimizer.init(params)
    return PlayerState(
        params=params,
        ema_params=params,
        periodic_params=params,
        opt_state=opt_state,
    )


def collect_episodes(
    network: nn.Module,
    params_p1: PyTree,
    params_p2: PyTree,
    game: NormalFormGame,
    rng: jax.Array,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample episodes on-policy using current params of both players.

    Logits are computed once (deterministic given params and game state) and
    only action sampling varies across episodes.

    Returns:
        actions_batch:    Integer action indices, shape (batch_size, 2).
        utilities_batch:  Per-player utilities, shape (batch_size, 2), where
                          [:, 0] is player 1's utility and [:, 1] player 2's.
    """
    logits_p1 = network.apply(params_p1, game.state_representation())
    logits_p2 = network.apply(params_p2, game.state_representation())

    def single_episode(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        rng1, rng2 = jax.random.split(rng)
        a1 = jax.random.categorical(rng1, logits_p1)
        a2 = jax.random.categorical(rng2, logits_p2)
        u = game.matrix[a1, a2]
        return jnp.stack([a1, a2]), jnp.stack([u, -u])

    rngs = jax.random.split(rng, batch_size)
    return jax.vmap(single_episode)(rngs)


def _magnet_kl(
    log_probs: jax.Array,
    magnet_logits: jax.Array,
    direction: str,
) -> jax.Array:
    """KL divergence between the current policy and the magnet (EMA) policy.

    ``"reverse"`` computes KL(policy || magnet) = sum_a policy_a * (log policy_a
    - log magnet_a); this is mode-seeking and is the original behaviour.
    ``"forward"`` computes KL(magnet || policy) = sum_a magnet_a * (log magnet_a
    - log policy_a); this is mass-covering.
    """
    probs = jnp.exp(log_probs)
    magnet_log_probs = jax.nn.log_softmax(magnet_logits)
    magnet_probs = jnp.exp(magnet_log_probs)
    if direction == "reverse":
        return jnp.sum(probs * (log_probs - magnet_log_probs))
    elif direction == "forward":
        return jnp.sum(magnet_probs * (magnet_log_probs - log_probs))
    else:
        raise ValueError(f"Unknown kl_direction: {direction!r}")


def _reinforce_loss(
    params: PyTree,
    network: nn.Module,
    game_state: jax.Array,
    player_actions: jax.Array,
    player_utilities: jax.Array,
    magnet_logits: jax.Array,
    entropy_coef: float,
    kl_coef: float,
    kl_direction: str,
) -> jax.Array:
    """REINFORCE loss with entropy bonus and KL magnet regularisation."""
    logits = network.apply(params, game_state)
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)

    pg_loss = -jnp.mean(log_probs[player_actions] * player_utilities)

    entropy = -jnp.sum(probs * log_probs)
    entropy_loss = -entropy_coef * entropy

    kl_loss = kl_coef * _magnet_kl(log_probs, magnet_logits, kl_direction)

    return pg_loss + entropy_loss + kl_loss


def _ppo_loss(
    params: PyTree,
    network: nn.Module,
    game_state: jax.Array,
    player_actions: jax.Array,
    player_utilities: jax.Array,
    old_log_probs: jax.Array,
    magnet_logits: jax.Array,
    clip_eps: float,
    entropy_coef: float,
    kl_coef: float,
    kl_direction: str,
) -> jax.Array:
    """PPO clipped surrogate loss with entropy bonus and KL magnet regularisation.

    Uses per-episode advantages (utility minus mean utility as baseline) and
    clips the probability ratio to [1-clip_eps, 1+clip_eps].
    """
    logits = network.apply(params, game_state)
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)

    new_log_probs = log_probs[player_actions]
    ratio = jnp.exp(new_log_probs - old_log_probs)

    advantages = player_utilities - jnp.mean(player_utilities)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    ppo_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

    entropy = -jnp.sum(probs * log_probs)
    entropy_loss = -entropy_coef * entropy

    kl_loss = kl_coef * _magnet_kl(log_probs, magnet_logits, kl_direction)

    return ppo_loss + entropy_loss + kl_loss


@functools.partial(
    jax.jit,
    static_argnames=('network', 'optimizer', 'batch_size', 'periodic_interval', 'algorithm', 'ppo_epochs', 'kl_direction'),
)
def train_step(
    player1_state: PlayerState,
    player2_state: PlayerState,
    network: nn.Module,
    optimizer: optax.GradientTransformation,
    game: NormalFormGame,
    rng: jax.Array,
    step: jax.Array,
    batch_size: int = 256,
    target_update_rate: float = 0.005,
    periodic_interval: int = 100,
    entropy_coef: float = 0.01,
    kl_coef: float = 0.01,
    kl_direction: str = 'reverse',
    algorithm: str = 'reinforce',
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
) -> tuple[PlayerState, PlayerState, jax.Array, jax.Array]:
    """Single JIT-compiled training step for both players.

    Supports REINFORCE (one gradient step per collected batch) and PPO
    (multiple clipped-surrogate epochs over the same batch).

    `network`, `optimizer`, `batch_size`, `periodic_interval`, `algorithm`,
    and `ppo_epochs` are static. `step` must be a traced jnp integer array.
    """
    game_state = game.state_representation()

    rng, ep_rng = jax.random.split(rng)
    actions_batch, utilities_batch = collect_episodes(
        network, player1_state.params, player2_state.params,
        game, ep_rng, batch_size,
    )

    magnet_logits_p1 = network.apply(player1_state.ema_params, game_state)
    magnet_logits_p2 = network.apply(player2_state.ema_params, game_state)

    def apply_updates(state: PlayerState, grads: PyTree) -> tuple[PyTree, optax.OptState]:
        updates, new_opt_state = optimizer.update(grads, state.opt_state, params=state.params)
        return optax.apply_updates(state.params, updates), new_opt_state

    if algorithm == 'reinforce':
        loss_p1, grads_p1 = jax.value_and_grad(_reinforce_loss)(
            player1_state.params, network, game_state,
            actions_batch[:, 0], utilities_batch[:, 0],
            magnet_logits_p1, entropy_coef, kl_coef, kl_direction,
        )
        loss_p2, grads_p2 = jax.value_and_grad(_reinforce_loss)(
            player2_state.params, network, game_state,
            actions_batch[:, 1], utilities_batch[:, 1],
            magnet_logits_p2, entropy_coef, kl_coef, kl_direction,
        )
        new_params_p1, new_opt_p1 = apply_updates(player1_state, grads_p1)
        new_params_p2, new_opt_p2 = apply_updates(player2_state, grads_p2)
        player1_state = PlayerState(new_params_p1, player1_state.ema_params, player1_state.periodic_params, new_opt_p1)
        player2_state = PlayerState(new_params_p2, player2_state.ema_params, player2_state.periodic_params, new_opt_p2)

    else:  # ppo
        old_log_probs_p1 = jax.nn.log_softmax(
            network.apply(player1_state.params, game_state)
        )[actions_batch[:, 0]]
        old_log_probs_p2 = jax.nn.log_softmax(
            network.apply(player2_state.params, game_state)
        )[actions_batch[:, 1]]

        def ppo_epoch(carry: tuple, _: None) -> tuple:
            p1_s, p2_s = carry
            loss_p1, grads_p1 = jax.value_and_grad(_ppo_loss)(
                p1_s.params, network, game_state,
                actions_batch[:, 0], utilities_batch[:, 0],
                old_log_probs_p1, magnet_logits_p1, clip_eps, entropy_coef, kl_coef, kl_direction,
            )
            loss_p2, grads_p2 = jax.value_and_grad(_ppo_loss)(
                p2_s.params, network, game_state,
                actions_batch[:, 1], utilities_batch[:, 1],
                old_log_probs_p2, magnet_logits_p2, clip_eps, entropy_coef, kl_coef, kl_direction,
            )
            new_params_p1, new_opt_p1 = apply_updates(p1_s, grads_p1)
            new_params_p2, new_opt_p2 = apply_updates(p2_s, grads_p2)
            new_p1 = PlayerState(new_params_p1, p1_s.ema_params, p1_s.periodic_params, new_opt_p1)
            new_p2 = PlayerState(new_params_p2, p2_s.ema_params, p2_s.periodic_params, new_opt_p2)
            return (new_p1, new_p2), (loss_p1, loss_p2)

        (player1_state, player2_state), (losses_p1, losses_p2) = jax.lax.scan(
            ppo_epoch, (player1_state, player2_state), None, length=ppo_epochs,
        )
        loss_p1, loss_p2 = losses_p1[-1], losses_p2[-1]

    # EMA and periodic updates applied once after all gradient steps.
    new_ema_p1 = optax.incremental_update(player1_state.params, player1_state.ema_params, step_size=target_update_rate)
    new_ema_p2 = optax.incremental_update(player2_state.params, player2_state.ema_params, step_size=target_update_rate)

    new_periodic_p1 = optax.periodic_update(player1_state.params, player1_state.periodic_params, steps=step, update_period=periodic_interval)
    new_periodic_p2 = optax.periodic_update(player2_state.params, player2_state.periodic_params, steps=step, update_period=periodic_interval)

    new_p1 = PlayerState(player1_state.params, new_ema_p1, new_periodic_p1, player1_state.opt_state)
    new_p2 = PlayerState(player2_state.params, new_ema_p2, new_periodic_p2, player2_state.opt_state)

    return new_p1, new_p2, loss_p1, loss_p2
