#!/usr/bin/env python3
"""Build a weight-dataset by training with the same hyperparameters across
multiple seeds.

Runs all seeds in a single process (rather than one subprocess per seed) so
JAX's jit cache for `train_step` is populated once and reused across seeds,
instead of being recompiled from scratch for every seed. Checkpoints are
saved directly under ``out_dir/<seed>/<step>`` (no game-name subfolder).
"""
import argparse
import sys
import traceback
from pathlib import Path

from main import build_env, run_training
from src.games import BIASED_GAME_CHOICES, GAME_CHOICES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run main.py with the same settings across multiple seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1100, 2000)),
                   help="Seeds to run.")

    # Game
    p.add_argument("--game", default="brps", choices=GAME_CHOICES,
                   help="mp=matching_pennies, bmp=biased_matching_pennies, "
                        "rps=rock_paper_scissors, brps=biased_rock_paper_scissors.")
    p.add_argument("--bias", type=float, default=1.0)

    # Training
    p.add_argument("--n_steps", type=int, default=5_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--optimizer", type=str, default="muon",
                   choices=["adam", "adamw", "sgd", "rmsprop", "muon",
                            "optimistic_adam", "optimistic_gd"])
    p.add_argument("--lr", type=float, default=1e-3)

    # Network
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[16, 16])
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--normalization", type=str, default="none")

    # Algorithm
    p.add_argument("--algorithm", type=str, default="ppo",
                   choices=["reinforce", "ppo"])
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--clip_eps", type=float, default=0.1)

    # Regularisation
    p.add_argument("--entropy_coef", type=float, default=0.02)
    p.add_argument("--kl_coef", type=float, default=0.5)
    p.add_argument("--kl_direction", type=str, default="reverse",
                   choices=["reverse", "forward"])

    # EMA / periodic snapshot
    p.add_argument("--target_update_rate", type=float, default=0.005)
    p.add_argument("--periodic_interval", type=int, default=200)

    # Logging / checkpointing
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="data/biased_rps_16",
                   help="Root directory to save checkpoints under.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Running {len(args.seeds)} seeds: {args.seeds}")
    print(f"Game: {args.game}" + (f"  bias={args.bias}" if args.game in BIASED_GAME_CHOICES else ""))
    print(f"Optimizer: {args.optimizer}  lr={args.lr}  n_steps={args.n_steps}")
    print()

    game, network, optimizer = build_env(args)

    results: dict[int, bool] = {}
    for seed in args.seeds:
        print(f"{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        try:
            run_training(args, seed, game, network, optimizer, game_subdir=False)
            results[seed] = True
        except Exception:
            traceback.print_exc()
            results[seed] = False
        print()

    print(f"{'='*60}")
    print("Summary:")
    for seed, ok in results.items():
        status = "OK" if ok else "FAILED"
        save_path = Path(args.out_dir) / str(seed) / str(args.n_steps)
        print(f"  seed {seed:4d}: {status}  -> {save_path}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed seeds: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
