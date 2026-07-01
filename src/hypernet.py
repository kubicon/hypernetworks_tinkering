"""Hypernetwork auto-encoder over trained policy-network weights.

This is a proof-of-concept that learns a low-dimensional *embedding* of an
entire trained network.  The pipeline is an auto-encoder whose encoder reads
weight space but whose decoder outputs behaviour, not weights:

    encoder:  theta (flat weight vector)  ->  z (latent embedding)
    decoder:  z                           ->  pi_hat (reconstructed strategy)

The decoder never reconstructs any of the 1346 parameters of the GeneralMLP
policy; it maps the latent code straight to action-distribution logits, so it
is trained solely against a behavioural signal:
  * behavioural KL -- match the policy's action distribution on the (fixed)
                      game input.

See ``train_hypernet.py`` for the training loop and the notes printed there
about weight-space symmetries, which are the main subtlety with this approach.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from src.games import (
    BiasedMatchingPennies,
    BiasedRockPaperScissors,
    MatchingPennies,
    NormalFormGame,
    RockPaperScissors,
)
from src.networks import GeneralMLP

# Default architecture/game for biased_matching_pennies (back-compat). For other
# games/sizes, infer the architecture from a checkpoint and build the matching
# network with the helpers below.
POLICY_NET = GeneralMLP(
    hidden_dims=(32, 32), output_dim=2,
    normalization="rms_norm", activation="gelu",
)
GAME = BiasedMatchingPennies(1.0)


# ---------------------------------------------------------------------------
# Architecture inference and config-driven construction
# ---------------------------------------------------------------------------

def load_sample_params(data_dir: str | Path):
    """Return the params pytree of the first checkpoint found under data_dir."""
    path = sorted(Path(data_dir).glob("*/*/p1_state.pkl"))[0]
    with open(path, "rb") as f:
        return pickle.load(f).params


def infer_arch_config(params) -> dict:
    """Infer the GeneralMLP architecture from a params pytree.

    Returns input_dim, output_dim, hidden_dims and the normalization key
    ('rms_norm', 'layer_norm', or 'none'). The activation cannot be recovered
    from weights and must be supplied separately.
    """
    p = params.get("params", params)
    dense = sorted((k for k in p if k.startswith("Dense_")),
                   key=lambda s: int(s.split("_")[1]))
    in_dim = int(p[dense[0]]["kernel"].shape[0])
    out_dim = int(p[dense[-1]]["kernel"].shape[1])
    hidden = tuple(int(p[d]["bias"].shape[0]) for d in dense[:-1])

    norm = "none"
    norm_keys = [k for k in p if k.startswith("Normalization_")]
    if norm_keys:
        sub = next(iter(p[sorted(norm_keys)[0]]))
        norm = "rms_norm" if "RMSNorm" in sub else "layer_norm"
    return dict(input_dim=in_dim, output_dim=out_dim,
                hidden_dims=hidden, normalization=norm)


def build_policy_net(cfg: dict, activation: str = "gelu") -> GeneralMLP:
    """Build the GeneralMLP matching an inferred config."""
    norm = None if cfg["normalization"] == "none" else cfg["normalization"]
    return GeneralMLP(
        hidden_dims=cfg["hidden_dims"], output_dim=cfg["output_dim"],
        normalization=norm, activation=activation,
    )


def make_game(name: str, bias: float = 1.0) -> NormalFormGame:
    """Construct a game by short code (mirrors main.py)."""
    match name:
        case "mp":
            return MatchingPennies()
        case "bmp":
            return BiasedMatchingPennies(bias)
        case "rps":
            return RockPaperScissors()
        case "brps":
            return BiasedRockPaperScissors(bias)
        case _:
            raise ValueError(f"Unknown game: {name!r}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class WeightDataset:
    """Flattened weight vectors plus per-sample metadata.

    Attributes:
        weights:   (N, D) raw flattened policy params.
        seeds:     (N,) training seed each sample came from.
        steps:     (N,) training step (checkpoint) each sample came from.
        players:   (N,) 0 for player 1, 1 for player 2.
        policies:  (N, 2) softmax action distribution of each network on the
                   fixed game input -- the "behaviour" of the network.
        unravel:   callable turning a flat (D,) vector back into a params pytree.
        mean/std:  (D,) per-dimension standardization stats over the dataset.
    """
    weights: np.ndarray
    seeds: np.ndarray
    steps: np.ndarray
    players: np.ndarray
    policies: np.ndarray
    unravel: callable
    mean: np.ndarray
    std: np.ndarray

    @property
    def dim(self) -> int:
        return self.weights.shape[1]

    def standardized(self) -> np.ndarray:
        return (self.weights - self.mean) / self.std

    def destandardize(self, z: jnp.ndarray) -> jnp.ndarray:
        return z * self.std + self.mean


def load_weight_dataset(
    data_dir: str | Path,
    network: nn.Module = POLICY_NET,
    game: NormalFormGame = GAME,
    steps: str = "all",
) -> WeightDataset:
    """Load every checkpoint into a flat weight dataset.

    Args:
        data_dir: e.g. ``data/biased_matching_pennies``.
        network:  the policy network matching the checkpoints (build via
                  build_policy_net on an inferred config).
        game:     the game the networks were trained on (for the fixed input).
        steps:    "all" to use every saved checkpoint, or "final" to use only
                  the last checkpoint of each seed.
    """
    data_dir = Path(data_dir)
    seed_dirs = sorted(
        (d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    weights, seeds, step_ids, players = [], [], [], []
    unravel = None
    state_in = game.state_representation()

    for sd in seed_dirs:
        ckpt_dirs = sorted(
            (c for c in sd.iterdir() if c.is_dir() and c.name.isdigit()),
            key=lambda c: int(c.name),
        )
        if steps == "final":
            ckpt_dirs = ckpt_dirs[-1:]
        for cd in ckpt_dirs:
            for pidx, fname in enumerate(("p1_state.pkl", "p2_state.pkl")):
                with open(cd / fname, "rb") as f:
                    state = pickle.load(f)
                flat, unravel = ravel_pytree(state.params)
                weights.append(np.asarray(flat))
                seeds.append(int(sd.name))
                step_ids.append(int(cd.name))
                players.append(pidx)

    weights = np.stack(weights).astype(np.float32)

    # Behavioural policy of every network on the fixed game input.
    @jax.jit
    def policy_of(flat):
        logits = network.apply(unravel(flat), state_in)
        return jax.nn.softmax(logits)

    policies = np.asarray(jax.vmap(policy_of)(jnp.asarray(weights)))

    mean = weights.mean(0)
    std = weights.std(0) + 1e-8
    return WeightDataset(
        weights=weights,
        seeds=np.asarray(seeds),
        steps=np.asarray(step_ids),
        players=np.asarray(players),
        policies=policies,
        unravel=unravel,
        mean=mean,
        std=std,
    )


# ---------------------------------------------------------------------------
# Hypernetwork auto-encoder
# ---------------------------------------------------------------------------

class HyperAE(nn.Module):
    """Weight-space encoder / strategy-space decoder.

    The encoder compresses a flat weight vector to ``latent_dim`` numbers; the
    decoder does not regenerate any weights -- it maps the latent straight to
    the policy's action-distribution logits, so it only ever has to learn to
    reconstruct the *strategy*, never the underlying parameters.
    """
    weight_dim: int
    output_dim: int = 2
    latent_dim: int = 8
    enc_dims: tuple[int, ...] = (256, 128)
    dec_dims: tuple[int, ...] = (128, 256)

    def setup(self) -> None:
        self.enc_layers = [nn.Dense(h) for h in self.enc_dims]
        self.to_latent = nn.Dense(self.latent_dim)
        self.dec_layers = [nn.Dense(h) for h in self.dec_dims]
        self.to_policy = nn.Dense(self.output_dim)

    def __call__(self, theta: jnp.ndarray):
        z = self.encode(theta)
        policy_logits = self.decode(z)
        return policy_logits, z

    def encode(self, theta: jnp.ndarray) -> jnp.ndarray:
        x = theta
        for layer in self.enc_layers:
            x = nn.gelu(layer(x))
        return self.to_latent(x)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Latent -> policy logits (softmax gives the action distribution)."""
        x = z
        for layer in self.dec_layers:
            x = nn.gelu(layer(x))
        return self.to_policy(x)
