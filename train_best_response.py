#!/usr/bin/env python3
"""Train per-player best-response networks over policy-network embeddings.

Consumes the embeddings produced by train_hypernet.py and trains two small
MLPs (one per player) that map an opponent network's embedding to a
best-response strategy.  Seeds are split into train/test so we measure
generalisation to embeddings of *unseen* trained networks.

Run:  uv run python train_best_response.py
(after train_hypernet.py has written hypernet_out/embeddings.npz)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.best_response import (
    BRNet,
    best_response_targets,
    best_response_value,
)
from src.games import BiasedMatchingPennies
from src.trainlog import TrainMonitor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb", default="hypernet_out/embeddings.npz")
    p.add_argument("--out_dir", default="hypernet_out")
    p.add_argument("--n_test_seeds", type=int, default=10,
                   help="Seeds held out for the test split.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=50,
                   help="Print epoch stats every N epochs.")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="Checkpoint every N epochs (0 = only at the end). Each "
                        "save is a distinct <tag>_step<N>.pkl.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_one(
    name: str,
    Z: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    args: argparse.Namespace,
):
    """Train a single BRNet; return params and a (logits_fn)."""
    model = BRNet(n_actions=int(y.max()) + 1)
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.asarray(Z[:1]))
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    Ztr, ytr = jnp.asarray(Z[train_mask]), jnp.asarray(y[train_mask])

    def loss_fn(params, z, t):
        logits = model.apply(params, z)
        return optax.softmax_cross_entropy_with_integer_labels(logits, t).mean()

    @jax.jit
    def update(params, opt_state, z, t):
        loss, grads = jax.value_and_grad(loss_fn)(params, z, t)
        updates, opt_state = opt.update(grads, opt_state)
        correct = jnp.sum(jnp.argmax(model.apply(params, z), -1) == t)
        return optax.apply_updates(params, updates), opt_state, loss, correct

    monitor = TrainMonitor(args.out_dir, f"br_{name}", log_every=args.log_every,
                           ckpt_every=args.ckpt_every)
    start_epoch = 0
    if args.resume:
        ckpt = monitor.resume()
        if ckpt is not None:
            params, opt_state = ckpt["params"], ckpt["opt_state"]
            start_epoch = ckpt["step"] + 1
            print(f"[{name}] resumed at epoch {ckpt['step']}.")

    def checkpoint_state():
        return {"params": params, "opt_state": opt_state}

    n = Ztr.shape[0]
    rng = np.random.default_rng(args.seed)
    for epoch in range(start_epoch, args.epochs):
        perm = rng.permutation(n)
        s_loss = s_correct = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            params, opt_state, loss, correct = update(
                params, opt_state, Ztr[idx], ytr[idx])
            s_loss += float(loss) * idx.shape[0]
            s_correct += float(correct)
        monitor.record(epoch, {"loss": s_loss / n, "train_acc": s_correct / n},
                       state=checkpoint_state,
                       force_log=epoch == args.epochs - 1)
    monitor.ckpt.save(args.epochs - 1, checkpoint_state)
    monitor.close()

    @jax.jit
    def predict(params, z):
        return jax.nn.softmax(model.apply(params, z))

    return params, predict


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    d = np.load(args.emb)
    Z, seeds, players, policies = d["z"], d["seeds"], d["players"], d["policies"]
    A = np.asarray(BiasedMatchingPennies(1.0).matrix)

    inputs, targets = best_response_targets(A, policies, players)

    # Seed-based train/test split (held-out seeds = unseen networks).
    uniq = np.unique(seeds)
    rng = np.random.default_rng(args.seed)
    test_seeds = set(rng.choice(uniq, size=args.n_test_seeds, replace=False).tolist())
    is_test = np.array([s in test_seeds for s in seeds])
    print(f"{len(uniq)} seeds: {len(uniq) - len(test_seeds)} train / "
          f"{len(test_seeds)} test (held out).")

    trained = {}
    for player in ("p1", "p2"):
        mask = inputs[player]                       # opponent networks for this BR net
        Zp = Z[mask]
        yp = targets[player]
        seeds_p = seeds[mask]
        opp_pol = policies[mask]                     # opponent's policy
        test_p = np.array([s in test_seeds for s in seeds_p])

        params, predict = train_one(
            player, Zp, yp, ~test_p, args)

        # Evaluate: argmax accuracy + best-response regret on the test split.
        pred = np.asarray(predict(params, jnp.asarray(Zp)))
        pred_action = pred.argmax(1)
        acc_tr = (pred_action[~test_p] == yp[~test_p]).mean()
        acc_te = (pred_action[test_p] == yp[test_p]).mean()

        # Regret: optimal BR value minus value achieved by predicted strategy.
        val_pred = best_response_value(A, pred, opp_pol, player)
        opt_strat = np.eye(A.shape[0])[yp]
        val_opt = best_response_value(A, opt_strat, opp_pol, player)
        regret = val_opt - val_pred
        print(f"\n[{player} best-response net]  "
              f"train acc {acc_tr:.3f} | test acc {acc_te:.3f}")
        print(f"    mean BR regret  train {regret[~test_p].mean():.4f} | "
              f"test {regret[test_p].mean():.4f}")
        trained[player] = params

    with open(out_dir / "br_params.pkl", "wb") as f:
        pickle.dump(trained, f)
    print(f"\nSaved best-response params to {out_dir / 'br_params.pkl'}")


if __name__ == "__main__":
    main()
