#!/usr/bin/env python3
"""End-to-end joint training of the hypernetwork auto-encoder + BR networks.

Unlike the two-stage pipeline (train_hypernet.py then train_best_response.py),
here the encoder, decoder, and both per-player best-response heads share a
single optimizer and are trained together.  Gradients from the best-response
loss flow *into the encoder*, so the latent embedding is shaped both to
reconstruct the weights and to be useful for predicting best responses.

Total loss per sample =
      recon_mse                              (weight-space reconstruction)
    + behaviour_coef * behaviour_kl          (policy of reconstructed weights)
    + br_coef        * best_response_ce       (BR head for the relevant player)

Each sample is one (seed, step, player) network.  A player-2 network's
embedding feeds the player-1 BR head (P1 best-responds to that P2 policy) and
vice-versa, so every sample trains exactly one BR head.

Run:  uv run python train_joint.py
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
from src.sym_encoders import (
    SymHyperAE,
    arch_spec_from_config,
    random_permute_batch,
)
from src.trainlog import TrainMonitor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data/biased_matching_pennies")
    p.add_argument("--out_dir", default="hypernet_out")
    p.add_argument("--game", default="bmp", choices=GAME_CHOICES,
                   help="mp=matching_pennies, bmp=biased_matching_pennies, "
                        "rps=rock_paper_scissors, brps=biased_rock_paper_scissors.")
    p.add_argument("--bias", type=float, default=1.0)
    p.add_argument("--activation", default="gelu",
                   help="Activation used when the checkpoints were trained.")
    p.add_argument("--encoder", default="mlp",
                   choices=["mlp", "canon", "deepsets", "graph", "equiv"])
    p.add_argument("--augment", action="store_true",
                   help="Apply random valid weight-space permutations each step.")
    p.add_argument("--steps_subset", default="all", choices=["all", "final"])
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_coef", type=float, default=1.0)
    p.add_argument("--behaviour_coef", type=float, default=1.0)
    p.add_argument("--br_coef", type=float, default=1.0,
                   help="Weight on the best-response loss (shapes the embedding).")
    p.add_argument("--n_test_seeds", type=int, default=10)
    p.add_argument("--log_every", type=int, default=50,
                   help="Print epoch stats every N epochs.")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="Checkpoint every N epochs (0 = only at the end). Each "
                        "save is a distinct <tag>_step<N>.pkl.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest <tag>_step<N>.pkl if present.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def kl(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    p = jnp.clip(p, 1e-8, 1.0)
    q = jnp.clip(q, 1e-8, 1.0)
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=-1)


def br_targets_per_sample(A, policies, players):
    """Best-response action target for every sample, from its own policy."""
    is_p2net = players == 1                    # policy q -> feeds P1 BR head
    is_p1net = players == 0                    # policy p -> feeds P2 BR head
    tgt = np.zeros(len(players), dtype=np.int32)
    tgt[is_p2net] = np.argmax(policies[is_p2net] @ A.T, axis=1)   # P1 BR
    tgt[is_p1net] = np.argmin(policies[is_p1net] @ A, axis=1)     # P2 BR
    return tgt, is_p2net, is_p1net


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = infer_arch_config(load_sample_params(args.data_dir))
    net = build_policy_net(cfg, args.activation)
    game = make_game(args.game, args.bias)
    spec = arch_spec_from_config(cfg)
    A = np.asarray(game.matrix)

    ds = load_weight_dataset(args.data_dir, net, game, steps=args.steps_subset)
    print(f"[encoder={args.encoder}] arch {cfg['input_dim']}->"
          f"{'->'.join(map(str, cfg['hidden_dims']))}->{cfg['output_dim']} "
          f"norm={cfg['normalization']}; {ds.weights.shape[0]} weight vectors "
          f"of dim {ds.dim}.")

    Xraw = jnp.asarray(ds.weights)                 # raw weights (sym encoders)
    mean = jnp.asarray(ds.mean)
    std = jnp.asarray(ds.std)
    X = (Xraw - mean) / std                         # per-dim standardized
    true_pol = jnp.asarray(ds.policies)
    state_in = game.state_representation()
    unravel = ds.unravel

    br_tgt_np, is_p2net_np, is_p1net_np = br_targets_per_sample(
        A, ds.policies, ds.players)
    br_tgt = jnp.asarray(br_tgt_np)
    is_p2 = jnp.asarray(is_p2net_np.astype(np.float32))   # mask -> P1 BR head
    is_p1 = jnp.asarray(is_p1net_np.astype(np.float32))   # mask -> P2 BR head

    # Seed-based train/test split (held-out = unseen networks).
    uniq = np.unique(ds.seeds)
    rng_np = np.random.default_rng(args.seed)
    test_seeds = set(rng_np.choice(uniq, args.n_test_seeds, replace=False).tolist())
    is_test = np.array([s in test_seeds for s in ds.seeds])
    train_idx = np.where(~is_test)[0]
    print(f"{len(uniq)} seeds: {len(uniq) - len(test_seeds)} train / "
          f"{len(test_seeds)} test.")

    # --- Models + combined parameter tree ---
    ae = SymHyperAE(weight_dim=ds.dim, spec=spec, encoder_name=args.encoder,
                    latent_dim=args.latent_dim)
    br1 = BRNet(n_actions=A.shape[0])
    br2 = BRNet(n_actions=A.shape[1])

    k_ae, k1, k2 = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    ae_params = ae.init(k_ae, X[:1], Xraw[:1])
    z0 = ae.apply(ae_params, X[:1], Xraw[:1], method=SymHyperAE.encode)
    params = {
        "ae": ae_params,
        "br_p1": br1.init(k1, z0),
        "br_p2": br2.init(k2, z0),
    }
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    def behaviour(theta_std):
        theta = theta_std * std + mean
        return jax.nn.softmax(net.apply(unravel(theta), state_in))

    def forward(params, x_std, x_raw):
        x_hat, z = ae.apply(params["ae"], x_std, x_raw)
        logits1 = br1.apply(params["br_p1"], z)
        logits2 = br2.apply(params["br_p2"], z)
        return x_hat, z, logits1, logits2

    def loss_fn(params, x_std, x_raw, tgt_pol, btgt, m_p1head, m_p2head):
        x_hat, z, logits1, logits2 = forward(params, x_std, x_raw)
        recon = jnp.mean((x_hat - x_std) ** 2)
        behav = jnp.mean(kl(tgt_pol, jax.vmap(behaviour)(x_hat)))
        ce1 = optax.softmax_cross_entropy_with_integer_labels(logits1, btgt)
        ce2 = optax.softmax_cross_entropy_with_integer_labels(logits2, btgt)
        # Each sample contributes to exactly one head; average over all samples.
        br = jnp.sum(ce1 * m_p1head + ce2 * m_p2head) / x_std.shape[0]
        total = args.recon_coef * recon + args.behaviour_coef * behav + args.br_coef * br
        return total, (recon, behav, br)

    @jax.jit
    def update(params, opt_state, x_std, x_raw, tgt_pol, btgt, m1, m2):
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_std, x_raw, tgt_pol, btgt, m1, m2)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, total, aux

    @jax.jit
    def augment(x_raw, key):
        x_raw = random_permute_batch(x_raw, key, spec)
        return x_raw, (x_raw - mean) / std

    tag = args.encoder + ("_aug" if args.augment else "")
    monitor = TrainMonitor(args.out_dir, "joint_" + tag, log_every=args.log_every,
                           ckpt_every=args.ckpt_every)

    start_epoch = 0
    if args.resume:
        ckpt = monitor.resume()
        if ckpt is not None:
            params, opt_state = ckpt["params"], ckpt["opt_state"]
            start_epoch = ckpt["step"] + 1
            print(f"Resumed from joint_{tag}_step{ckpt['step']}.pkl.")

    def checkpoint_state():
        return {"params": params, "opt_state": opt_state, "encoder": args.encoder}

    n = train_idx.shape[0]
    aug_key = jax.random.PRNGKey(args.seed + 1)
    for epoch in range(start_epoch, args.epochs):
        perm = train_idx[rng_np.permutation(n)]
        s_total = s_recon = s_behav = s_br = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xr, xs = Xraw[idx], X[idx]
            if args.augment:
                # Permutation leaves the policy (hence br_tgt + masks) unchanged.
                aug_key, k = jax.random.split(aug_key)
                xr, xs = augment(xr, k)
            params, opt_state, total, aux = update(
                params, opt_state, xs, xr, true_pol[idx], br_tgt[idx],
                is_p2[idx], is_p1[idx])
            recon, behav, br = aux
            bs = idx.shape[0]
            s_total += float(total) * bs
            s_recon += float(recon) * bs
            s_behav += float(behav) * bs
            s_br += float(br) * bs
        monitor.record(epoch, {
            "loss": s_total / n, "recon": s_recon / n,
            "behav_kl": s_behav / n, "br_ce": s_br / n,
        }, state=checkpoint_state, force_log=epoch == args.epochs - 1)
    monitor.ckpt.save(args.epochs - 1, checkpoint_state)
    monitor.close()

    # --- Evaluation ---
    @jax.jit
    def eval_forward(params, x_std, x_raw):
        _, z, logits1, logits2 = forward(params, x_std, x_raw)
        return z, jax.nn.softmax(logits1), jax.nn.softmax(logits2)
    Z, pred1, pred2 = (np.asarray(a) for a in eval_forward(params, X, Xraw))

    print("\n--- best-response quality (joint embedding) ---")
    for player, pred, headmask in (("p1", pred1, is_p2net_np),
                                   ("p2", pred2, is_p1net_np)):
        sel = headmask
        tr = sel & ~is_test
        te = sel & is_test
        action = pred.argmax(1)
        acc_tr = (action[tr] == br_tgt_np[tr]).mean()
        acc_te = (action[te] == br_tgt_np[te]).mean()
        opp_pol = ds.policies
        val_pred = best_response_value(A, pred, opp_pol, player)
        val_opt = best_response_value(A, np.eye(A.shape[0])[br_tgt_np], opp_pol, player)
        regret = val_opt - val_pred
        print(f"[{player} BR]  train acc {acc_tr:.3f} | test acc {acc_te:.3f}"
              f"   regret train {regret[tr].mean():.4f} | test {regret[te].mean():.4f}")

    np.savez(out_dir / f"joint_embeddings_{tag}.npz",
             z=Z, seeds=ds.seeds, steps=ds.steps, players=ds.players,
             policies=ds.policies)
    print(f"\nSaved final checkpoint (joint_{tag}_step{args.epochs - 1}.pkl), "
          f"CSV log, and embeddings to {out_dir}/")


if __name__ == "__main__":
    main()
