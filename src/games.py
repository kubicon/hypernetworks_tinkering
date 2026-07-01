import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class NormalFormGame:
    """Two-player zero-sum normal form game.

    Args:
        matrix: Payoff matrix for player 1, shape (n_actions_p1, n_actions_p2).
                Player 2's payoff is the negation of player 1's (zero-sum).
    """
    matrix: jnp.ndarray

    def step(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Return player 1's utility for pure-strategy action profile.

        Args:
            actions: Integer action indices of shape (2,), where actions[0]
                     is player 1's action and actions[1] is player 2's action.

        Returns:
            Scalar utility for player 1 (player 2's utility is its negation).
        """
        return self.matrix[actions[0], actions[1]]

    def state_representation(self) -> jnp.ndarray:
        """Return the payoff matrix as a flat vector."""
        return self.matrix.flatten()


# ---------------------------------------------------------------------------
# Concrete games
# ---------------------------------------------------------------------------

class MatchingPennies(NormalFormGame):
    """Standard Matching Pennies. Actions: 0=Heads, 1=Tails.

    Player 1 wins when both players match; player 2 wins when they differ.
    """
    def __init__(self) -> None:
        super().__init__(matrix=jnp.array([
            [ 1., -1.],   # Heads
            [-1.,  1.],   # Tails
        ]))


class BiasedMatchingPennies(NormalFormGame):
    """Matching Pennies with a bias added to the Tails-Tails payoff.

    Breaks the symmetry between Heads and Tails so the Nash equilibrium
    deviates from the uniform 50/50 mix.
    """
    def __init__(self, bias: float) -> None:
        super().__init__(matrix=jnp.array([
            [ 1.,        -1.],   # Heads
            [-1.,  1. + bias],   # Tails — bias on Tails-Tails
        ]))


class RockPaperScissors(NormalFormGame):
    """Standard Rock-Paper-Scissors. Actions: 0=Rock, 1=Paper, 2=Scissors."""
    def __init__(self) -> None:
        super().__init__(matrix=jnp.array([
            [ 0., -1.,  1.],   # Rock
            [ 1.,  0., -1.],   # Paper
            [-1.,  1.,  0.],   # Scissors
        ]))


class BiasedRockPaperScissors(NormalFormGame):
    """Rock-Paper-Scissors with a bias on the Rock-Scissors payoff.

    Bias is added only when player 1 plays Rock (0) and player 2 plays
    Scissors (2), making Rock a more attractive deviation and breaking the
    symmetric Nash equilibrium.
    """
    def __init__(self, bias: float) -> None:
        super().__init__(matrix=jnp.array([
            [ 0., -1.,  1. + bias],   # Rock — bias on Rock vs Scissors
            [ 1.,  0., -1.        ],   # Paper
            [-1.,  1.,  0.        ],   # Scissors
        ]))


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------
# All game classes are registered so they can be passed to jit-compiled
# functions. Unflatten always produces a NormalFormGame since all subclasses
# carry only a matrix and have the same interface.

def _game_flatten(game):
    return (game.matrix,), None

def _game_unflatten(_, children):
    return NormalFormGame(matrix=children[0])

for _cls in [NormalFormGame, MatchingPennies, BiasedMatchingPennies,
             RockPaperScissors, BiasedRockPaperScissors]:
    jax.tree_util.register_pytree_node(_cls, _game_flatten, _game_unflatten)


# ---------------------------------------------------------------------------
# Short CLI names
# ---------------------------------------------------------------------------
# Short codes used for the --game CLI argument. Data on disk keeps using the
# long names (see GAME_LONG_NAME), so existing data/<long_name>/ directories
# stay valid.
GAME_CHOICES = ["mp", "bmp", "rps", "brps"]

GAME_LONG_NAME = {
    "mp": "matching_pennies",
    "bmp": "biased_matching_pennies",
    "rps": "rock_paper_scissors",
    "brps": "biased_rock_paper_scissors",
}

BIASED_GAME_CHOICES = {"bmp", "brps"}
