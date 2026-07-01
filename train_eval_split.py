#!/usr/bin/env python3
"""Train the joint hypernet-AE + best-response model on a fixed seed split.

Splits the training seeds deterministically: the first ``1 - test_frac`` of the
sorted seeds are the train set, the last ``test_frac`` are held out as the test
set (unseen networks). After training, reports best-response **accuracy** on the
test set (fraction of held-out networks for which the predicted best-response
action matches the exact best response), alongside train accuracy and regret.

Run:  uv run python train_eval_split.py --encoder equiv \
          --game brps \
          --data_dir data/biased_rock_paper_scissors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.best_response import BRNet, best_response_value
from src.games import GAME_CHOICES
from src.hypernet import (
    build_policy_net,
    infer_arch_config,
    load_sample_params,
    load_weight_dataset,
    make_game,
)
from src.sym_encoders import SymHyperAE, arch_spec_from_config, random_permute_batch
from src.trainlog import TrainMonitor
from train_joint import br_targets_per_sample, kl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data/biased_rps_8/biased_rock_paper_scissors")
    p.add_argument("--out_dir", default="data/biased_rps_8_hypernet")
    p.add_argument("--game", default="brps", choices=GAME_CHOICES,
                   help="mp=matching_pennies, bmp=biased_matching_pennies, "
                        "rps=rock_paper_scissors, brps=biased_rock_paper_scissors.")
    p.add_argument("--bias", type=float, default=1.0)
    p.add_argument("--activation", default="gelu")
    p.add_argument("--encoder", default="graph",
                   choices=["mlp", "canon", "deepsets", "graph", "equiv"])
    p.add_argument("--augment", action="store_true")
    p.add_argument("--test_frac", type=float, default=0.05,
                   help="Fraction of seeds (the last ones) held out for testing.")
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_coef", type=float, default=1.0)
    p.add_argument("--behaviour_coef", type=float, default=1.0)
    p.add_argument("--br_coef", type=float, default=1.0)
    p.add_argument("--br_target", default="action", choices=["action", "value"],
                   help="'action': classify the best-response action (cross-"
                        "entropy). 'value': regress the scalar best-response "
                        "value instead (MSE), skipping the action altogether.")
    p.add_argument("--log_every", type=int, default=1,
                   help="Print epoch stats every N epochs (1 = every epoch).")
    p.add_argument("--ckpt_every", type=int, default=1,
                   help="Checkpoint every N epochs (0 = only at the end). Each "
                        "save is a distinct <tag>_step<N>.pkl.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest <tag>_step<N>.pkl if present.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = infer_arch_config(load_sample_params(args.data_dir))
    net = build_policy_net(cfg, args.activation)
    game = make_game(args.game, args.bias)
    spec = arch_spec_from_config(cfg)
    A = np.asarray(game.matrix)

    ds = load_weight_dataset(args.data_dir, net, game, steps="all")
    print(f"[encoder={args.encoder}] arch {cfg['input_dim']}->"
          f"{'->'.join(map(str, cfg['hidden_dims']))}->{cfg['output_dim']} "
          f"norm={cfg['normalization']}; {ds.weights.shape[0]} weight vectors "
          f"of dim {ds.dim}.")

    Xraw = jnp.asarray(ds.weights)
    mean, std = jnp.asarray(ds.mean), jnp.asarray(ds.std)
    X = (Xraw - mean) / std
    true_pol = jnp.asarray(ds.policies)
    state_in = game.state_representation()
    unravel = ds.unravel

    br_tgt_np, is_p2net_np, is_p1net_np = br_targets_per_sample(
        A, ds.policies, ds.players)
    br_tgt = jnp.asarray(br_tgt_np)
    is_p2 = jnp.asarray(is_p2net_np.astype(np.float32))
    is_p1 = jnp.asarray(is_p1net_np.astype(np.float32))

    # Scalar best-response value target (value the BR action would achieve),
    # same sign convention as best_response_value: larger is better for the
    # responding player.
    br_val_np = np.zeros(len(ds.players), dtype=np.float32)
    br_val_np[is_p2net_np] = np.max(ds.policies[is_p2net_np] @ A.T, axis=1)
    br_val_np[is_p1net_np] = -np.min(ds.policies[is_p1net_np] @ A, axis=1)
    br_val = jnp.asarray(br_val_np)

    # --- Deterministic split: first (1-test_frac) seeds train, last test_frac test.
    uniq = np.unique(ds.seeds)                       # sorted ascending
    n_test = max(1, round(len(uniq) * args.test_frac))
    test_seeds = set(uniq[-n_test:].tolist())
    is_test = np.array([s in test_seeds for s in ds.seeds])
    train_idx = np.where(~is_test)[0]
    print(f"{len(uniq)} seeds: {len(uniq) - n_test} train (first) / "
          f"{n_test} test (last). Test seeds: "
          f"{sorted(test_seeds)[:3]}...{sorted(test_seeds)[-1]}")

    # --- Models ---
    ae = SymHyperAE(weight_dim=ds.dim, spec=spec, encoder_name=args.encoder,
                    latent_dim=args.latent_dim)
    is_value = args.br_target == "value"
    br1 = BRNet(n_actions=1 if is_value else A.shape[0])
    br2 = BRNet(n_actions=1 if is_value else A.shape[1])
    k_ae, k1, k2 = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    ae_params = ae.init(k_ae, X[:1], Xraw[:1])
    z0 = ae.apply(ae_params, X[:1], Xraw[:1], method=SymHyperAE.encode)
    params = {"ae": ae_params, "br_p1": br1.init(k1, z0), "br_p2": br2.init(k2, z0)}
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    def behaviour(theta_std):
        return jax.nn.softmax(net.apply(unravel(theta_std * std + mean), state_in))

    def forward(params, x_std, x_raw):
        x_hat, z = ae.apply(params["ae"], x_std, x_raw)
        return x_hat, z, br1.apply(params["br_p1"], z), br2.apply(params["br_p2"], z)

    def loss_fn(params, x_std, x_raw, tgt_pol, btgt, bval, m1, m2):
        x_hat, z, logits1, logits2 = forward(params, x_std, x_raw)
        recon = jnp.mean((x_hat - x_std) ** 2)
        behav = jnp.mean(kl(tgt_pol, jax.vmap(behaviour)(x_hat)))
        if is_value:
            # Regress the scalar best-response value instead of classifying
            # the action itself.
            l1 = (logits1[..., 0] - bval) ** 2
            l2 = (logits2[..., 0] - bval) ** 2
            metric = jnp.sum(l1 * m1 + l2 * m2)       # summed squared error
        else:
            l1 = optax.softmax_cross_entropy_with_integer_labels(logits1, btgt)
            l2 = optax.softmax_cross_entropy_with_integer_labels(logits2, btgt)
            ok1 = (jnp.argmax(logits1, -1) == btgt) * m1
            ok2 = (jnp.argmax(logits2, -1) == btgt) * m2
            metric = jnp.sum(ok1 + ok2)               # count correct
        br = jnp.sum(l1 * m1 + l2 * m2) / x_std.shape[0]
        total = args.recon_coef * recon + args.behaviour_coef * behav + args.br_coef * br
        return total, (recon, behav, br, metric)

    @jax.jit
    def update(params, opt_state, x_std, x_raw, tgt_pol, btgt, bval, m1, m2):
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_std, x_raw, tgt_pol, btgt, bval, m1, m2)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, total, aux

    @jax.jit
    def augment(x_raw, key):
        x_raw = random_permute_batch(x_raw, key, spec)
        return x_raw, (x_raw - mean) / std

    # Held-out test tensors, prepared once for the per-epoch accuracy probe.
    # (Monitoring only -- the test set is never used to update params.)
    te_idx = np.where(is_test)[0]
    Xte_std, Xte_raw = X[te_idx], Xraw[te_idx]
    btgt_te = br_tgt[te_idx]
    bval_te = br_val[te_idx]
    m1_te, m2_te = is_p2[te_idx], is_p1[te_idx]   # which head each test sample feeds
    n_te = int(te_idx.size)
    metric_name = "mse" if is_value else "acc"

    @jax.jit
    def test_metric(params):
        _, _, logits1, logits2 = forward(params, Xte_std, Xte_raw)
        if is_value:
            se1 = (logits1[..., 0] - bval_te) ** 2
            se2 = (logits2[..., 0] - bval_te) ** 2
            return jnp.sum(se1 * m1_te + se2 * m2_te) / n_te
        ok1 = (jnp.argmax(logits1, -1) == btgt_te) * m1_te
        ok2 = (jnp.argmax(logits2, -1) == btgt_te) * m2_te
        return jnp.sum(ok1 + ok2) / n_te

    tag = args.encoder + ("_aug" if args.augment else "") + (
        "_val" if is_value else "")
    monitor = TrainMonitor(args.out_dir, tag, log_every=args.log_every,
                           ckpt_every=args.ckpt_every)

    start_epoch = 0
    if args.resume:
        ckpt = monitor.resume()
        if ckpt is not None:
            params, opt_state = ckpt["params"], ckpt["opt_state"]
            start_epoch = ckpt["step"] + 1
            print(f"Resumed from {tag}_step{ckpt['step']}.pkl.")

    def checkpoint_state():
        return {"params": params, "opt_state": opt_state, "encoder": args.encoder}

    n = train_idx.shape[0]
    rng = np.random.default_rng(args.seed)
    aug_key = jax.random.PRNGKey(args.seed + 1)
    print(f"\nTraining for {args.epochs} epochs "
          f"({-(-n // args.batch_size)} batches/epoch, {n} train samples)\n")
    for epoch in range(start_epoch, args.epochs):
        perm = train_idx[rng.permutation(n)]
        # Sample-weighted sums over the epoch's batches.
        s_total = s_recon = s_behav = s_br = s_metric = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xr, xs = Xraw[idx], X[idx]
            if args.augment:
                aug_key, k = jax.random.split(aug_key)
                xr, xs = augment(xr, k)
            params, opt_state, total, aux = update(
                params, opt_state, xs, xr, true_pol[idx], br_tgt[idx],
                br_val[idx], is_p2[idx], is_p1[idx])
            recon, behav, br, metric = aux
            bs = idx.shape[0]
            s_total += float(total) * bs
            s_recon += float(recon) * bs
            s_behav += float(behav) * bs
            s_br += float(br) * bs
            s_metric += float(metric)
        last = epoch == args.epochs - 1
        monitor.record(epoch, {
            "loss": s_total / n, "recon": s_recon / n, "behav_kl": s_behav / n,
            "br_loss": s_br / n, f"train_br_{metric_name}": s_metric / n,
            f"test_br_{metric_name}": float(test_metric(params)),
        }, state=checkpoint_state, force_log=last)
    monitor.ckpt.save(args.epochs - 1, checkpoint_state)   # always save final
    monitor.close()

    # --- Test-set evaluation ---
    @jax.jit
    def eval_forward(params, x_std, x_raw):
        _, z, l1, l2 = forward(params, x_std, x_raw)
        if is_value:
            return z, l1[..., 0], l2[..., 0]
        return z, jax.nn.softmax(l1), jax.nn.softmax(l2)
    Z, pred1, pred2 = (np.asarray(a) for a in eval_forward(params, X, Xraw))

    print("\n========== TEST-SET RESULTS (held-out seeds) ==========")
    if is_value:
        for player, pred_val, headmask in (("p1", pred1, is_p2net_np),
                                           ("p2", pred2, is_p1net_np)):
            tr = headmask & ~is_test
            te = headmask & is_test
            err = pred_val - br_val_np
            mae_tr, mse_tr = np.abs(err[tr]).mean(), (err[tr] ** 2).mean()
            mae_te, mse_te = np.abs(err[te]).mean(), (err[te] ** 2).mean()
            print(f"[{player} BR-value]  train MAE {mae_tr:.4f} (MSE {mse_tr:.4f})"
                  f" | test MAE {mae_te:.4f} (MSE {mse_te:.4f})"
                  f"  (n_test={te.sum()})")
    else:
        correct_te = wrong_te = 0
        for player, pred, headmask in (("p1", pred1, is_p2net_np),
                                       ("p2", pred2, is_p1net_np)):
            tr = headmask & ~is_test
            te = headmask & is_test
            action = pred.argmax(1)
            acc_tr = (action[tr] == br_tgt_np[tr]).mean()
            acc_te = (action[te] == br_tgt_np[te]).mean()
            correct_te += int((action[te] == br_tgt_np[te]).sum())
            wrong_te += int((action[te] != br_tgt_np[te]).sum())
            val_pred = best_response_value(A, pred, ds.policies, player)
            val_opt = best_response_value(
                A, np.eye(A.shape[0])[br_tgt_np], ds.policies, player)
            regret = val_opt - val_pred
            print(f"[{player} BR]  train acc {acc_tr:.3f} | test acc {acc_te:.3f}"
                  f"   test regret {regret[te].mean():.4f}  (n_test={te.sum()})")

        overall = correct_te / (correct_te + wrong_te)
        print(f"\nOVERALL TEST ACCURACY (both players): {overall:.3f} "
              f"({correct_te}/{correct_te + wrong_te})")

    np.savez(out_dir / f"split_embeddings_{tag}.npz",
             z=Z, seeds=ds.seeds, steps=ds.steps, players=ds.players,
             policies=ds.policies, is_test=is_test)
    print(f"Saved final checkpoint ({tag}_step{args.epochs - 1}.pkl), CSV log "
          f"({tag}_log.csv), and embeddings to {out_dir}/")


if __name__ == "__main__":
    main()
