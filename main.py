#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from src.games import (
    BIASED_GAME_CHOICES,
    GAME_CHOICES,
    GAME_LONG_NAME,
    BiasedMatchingPennies,
    BiasedRockPaperScissors,
    MatchingPennies,
    NormalFormGame,
    RockPaperScissors,
)
from src.metrics import compute_exploitability
from src.networks import GeneralMLP
from src.training import PlayerState, init_player_state, train_step


def make_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    match args.optimizer:
        case "adam":
            return optax.adam(args.lr)
        case "adamw":
            return optax.adamw(args.lr)
        case "sgd":
            return optax.sgd(args.lr)
        case "rmsprop":
            return optax.rmsprop(args.lr)
        case "muon":
            return optax.contrib.muon(args.lr)
        case "optimistic_adam":
            return optax.optimistic_adam(args.lr)
        case "optimistic_gd":
            return optax.optimistic_gradient_descent(args.lr)
        case _:
            raise ValueError(f"Unknown optimizer: {args.optimizer!r}")


def make_game(args: argparse.Namespace) -> NormalFormGame:
    match args.game:
        case "mp":
            return MatchingPennies()
        case "bmp":
            return BiasedMatchingPennies(args.bias)
        case "rps":
            return RockPaperScissors()
        case "brps":
            return BiasedRockPaperScissors(args.bias)
        case _:
            raise ValueError(f"Unknown game: {args.game!r}")



def save_checkpoint(p1_state: PlayerState, p2_state: PlayerState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "p1_state.pkl", "wb") as f:
        pickle.dump(p1_state, f)
    with open(path / "p2_state.pkl", "wb") as f:
        pickle.dump(p2_state, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Nash equilibrium strategies via policy gradient."
    )

    # Game
    p.add_argument(
        "--game", default="bmp",
        choices=GAME_CHOICES,
        help="mp=matching_pennies, bmp=biased_matching_pennies, "
             "rps=rock_paper_scissors, brps=biased_rock_paper_scissors.",
    )
    p.add_argument("--bias", type=float, default=1.0,
                   help="Reward bias for biased game variants.")

    # Training
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Episodes sampled per training step.")
    p.add_argument("--optimizer", type=str, default="muon",
                   choices=["adam", "adamw", "sgd", "rmsprop", "muon",
                            "optimistic_adam", "optimistic_gd"],
                   help="Gradient optimizer.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate.")

    # Network
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 32])
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--normalization", type=str, default="rms_norm")

    # Algorithm
    p.add_argument("--algorithm", type=str, default="ppo",
                   choices=["reinforce", "ppo"],
                   help="Policy gradient algorithm.")
    p.add_argument("--ppo_epochs", type=int, default=1,
                   help="Number of PPO update epochs per collected batch.")
    p.add_argument("--clip_eps", type=float, default=0.1,
                   help="PPO clipping epsilon.")

    # Regularisation
    p.add_argument("--entropy_coef", type=float, default=0.02)
    p.add_argument("--kl_coef", type=float, default=0.8)
    p.add_argument("--kl_direction", type=str, default="reverse",
                   choices=["reverse", "forward"],
                   help="Magnet KL direction: 'reverse' = KL(policy || magnet) "
                        "(mode-seeking), 'forward' = KL(magnet || policy) "
                        "(mass-covering).")

    # EMA / periodic snapshot
    p.add_argument("--target_update_rate", type=float, default=0.005,
                   help="Step size for EMA update (smaller = slower tracking).")
    p.add_argument("--periodic_interval", type=int, default=500,
                   help="Copy current params to periodic snapshot every N steps.")

    # Logging / checkpointing
    p.add_argument("--save_every", type=int, default=200,
                   help="Save weights every N steps.")
    p.add_argument("--eval_every", type=int, default=200,
                   help="Print exploitability every M steps.")
    p.add_argument("--out_dir", type=str, default="data",
                   help="Root directory to save checkpoints under.")

    return p.parse_args()


def build_env(args: argparse.Namespace) -> tuple[NormalFormGame, GeneralMLP, optax.GradientTransformation]:
    """Build the seed-independent training environment (game, network, optimizer).

    Sharing these across seeds lets JAX reuse a single compiled `train_step`
    instead of retracing it per seed.
    """
    game = make_game(args)
    n_actions = game.matrix.shape[0]   # both players have same count for our games

    network = GeneralMLP(
        hidden_dims=tuple(args.hidden_dims),
        output_dim=n_actions,
        activation=args.activation,
        normalization=args.normalization,
    )
    optimizer = make_optimizer(args)
    return game, network, optimizer


def run_training(
    args: argparse.Namespace,
    seed: int,
    game: NormalFormGame,
    network: GeneralMLP,
    optimizer: optax.GradientTransformation,
) -> None:
    state_dim = int(game.state_representation().shape[0])

    rng = jax.random.PRNGKey(seed)
    rng, rng_p1, rng_p2 = jax.random.split(rng, 3)

    p1_state = init_player_state(network, state_dim, rng_p1, optimizer)
    p2_state = init_player_state(network, state_dim, rng_p2, optimizer)

    save_root = Path(args.out_dir) / GAME_LONG_NAME[args.game] / str(seed)
    print(f"Game:  {args.game}" + (f"  bias={args.bias}" if args.game in BIASED_GAME_CHOICES else ""))
    print(f"Saves: {save_root}/<step>/")
    print()

    for step in range(args.n_steps + 1):
        if step % args.save_every == 0:
            save_checkpoint(p1_state, p2_state, save_root / str(step))

        if step % args.eval_every == 0:
            expl_cur = compute_exploitability(network, p1_state.params, p2_state.params, game)
            expl_ema = compute_exploitability(network, p1_state.ema_params, p2_state.ema_params, game)
            print(
                f"step {step:6d} | "
                f"exploit_current {expl_cur:.4f} | "
                f"exploit_ema {expl_ema:.4f}"
            )

        if step == args.n_steps:
            break

        rng, step_rng = jax.random.split(rng)
        p1_state, p2_state, _, _ = train_step(
            p1_state, p2_state, network, optimizer,
            game, step_rng, jnp.int32(step),
            batch_size=args.batch_size,
            target_update_rate=args.target_update_rate,
            periodic_interval=args.periodic_interval,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            kl_direction=args.kl_direction,
            algorithm=args.algorithm,
            ppo_epochs=args.ppo_epochs,
            clip_eps=args.clip_eps,
        )

    print(f"\nDone. Final weights saved to {save_root}/{args.n_steps}/")


def main() -> None:
    args = parse_args()
    game, network, optimizer = build_env(args)
    run_training(args, args.seed, game, network, optimizer)


if __name__ == "__main__":
    main()
