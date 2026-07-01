import jax
import jax.numpy as jnp
from flax import linen as nn

from src.games import NormalFormGame


def compute_exploitability(
    network: nn.Module,
    params_p1: dict,
    params_p2: dict,
    game: NormalFormGame,
) -> float:
    """Nash exploitability of a strategy profile (p, q).

    Equals max_{a1} (A q)_{a1} - min_{a2} (A^T p)_{a2}, which is 0 at Nash
    equilibrium and positive otherwise.
    """
    state = game.state_representation()
    p = jax.nn.softmax(network.apply(params_p1, state))
    q = jax.nn.softmax(network.apply(params_p2, state))
    br1 = jnp.max(game.matrix @ q)
    br2 = jnp.min(game.matrix.T @ p)
    return float(br1 - br2)
