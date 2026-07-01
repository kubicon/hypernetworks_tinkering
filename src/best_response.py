"""Best-response networks over policy-network embeddings.

Given the embedding of an *opponent* policy network (produced by the
hypernetwork auto-encoder in src/hypernet.py), predict the best-response
strategy against that opponent in the normal-form game.

We train one network per player:
  * ``br_p1``: input = embedding of a player-2 network, output = player 1's
               best-response distribution over its actions.
  * ``br_p2``: input = embedding of a player-1 network, output = player 2's
               best-response distribution.

In a 2-player zero-sum game with player-1 payoff matrix ``A`` the exact best
responses are trivial to compute, so they serve as supervised targets:

    against P2 mix q:  P1 wants to maximise (A q)      -> argmax_i (A q)_i
    against P1 mix p:  P2 wants to minimise (Aᵀ p)     -> argmin_j (Aᵀ p)_j

The point of the proof of concept is that the BR network never sees the
opponent policy directly -- only its embedding -- so it must recover the
relevant behaviour from the latent code.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class BRNet(nn.Module):
    """Small MLP mapping an embedding to best-response action logits."""
    n_actions: int
    hidden_dims: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = z
        for h in self.hidden_dims:
            x = nn.gelu(nn.Dense(h)(x))
        return nn.Dense(self.n_actions)(x)


def best_response_targets(
    matrix: np.ndarray,
    policies: np.ndarray,
    players: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split samples by player and compute exact best-response targets.

    Args:
        matrix:   (a1, a2) player-1 payoff matrix.
        policies: (N, .) each network's own action distribution.
        players:  (N,) 0 = player-1 network (policy p), 1 = player-2 network
                  (policy q).

    Returns:
        inputs:  {"p1": mask, "p2": mask} -- boolean masks selecting the
                 *opponent* networks each BR net consumes.
        targets: {"p1": br_actions, "p2": br_actions} -- integer best-response
                 action for each selected sample.
    """
    A = np.asarray(matrix)
    is_p1_net = players == 0          # policy p (player-1 networks)
    is_p2_net = players == 1          # policy q (player-2 networks)

    q = policies[is_p2_net]
    p = policies[is_p1_net]

    # P1 best-responds to P2 networks: maximise (A q).
    br_p1 = np.argmax(q @ A.T, axis=1)
    # P2 best-responds to P1 networks: minimise (Aᵀ p).
    br_p2 = np.argmin(p @ A, axis=1)

    inputs = {"p1": is_p2_net, "p2": is_p1_net}
    targets = {"p1": br_p1.astype(np.int32), "p2": br_p2.astype(np.int32)}
    return inputs, targets


def best_response_value(
    matrix: np.ndarray,
    strategy: np.ndarray,
    opponent: np.ndarray,
    player: str,
) -> np.ndarray:
    """Expected payoff (to the responding player) of a strategy vs an opponent.

    ``strategy`` and ``opponent`` are (N, a) distributions. For ``player='p1'``
    returns sᵀ A q; for ``player='p2'`` returns the player-2 payoff -sᵀ ... i.e.
    -(pᵀ A o) so that larger is better for player 2.
    """
    A = np.asarray(matrix)
    if player == "p1":
        return np.einsum("ni,ij,nj->n", strategy, A, opponent)
    else:  # p2 responds; opponent is the P1 network (policy p)
        return -np.einsum("ni,ij,nj->n", opponent, A, strategy)
