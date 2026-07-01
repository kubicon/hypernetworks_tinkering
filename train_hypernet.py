#!/usr/bin/env python3
"""Train a hypernetwork auto-encoder over trained policy-network weights.

Proof of concept.  Loads the 50 trained seeds for biased_matching_pennies,
flattens each checkpoint's policy params into a vector, and learns a small
latent embedding via a weight-space auto-encoder.

The encoder is selectable via ``--encoder`` (see src/sym_encoders.py and
HypernetArchitecture.md):
    mlp       plain MLP baseline (no symmetry awareness)
    canon     sort-canonicalize neurons, then an MLP            (symmetry: D)
    deepsets  neuron-token DeepSets with invariant pooling      (symmetry: B)
    graph     graph metanetwork / message passing               (symmetry: A)
    equiv     NFN/DWSNet-style equivariant-linear layers         (symmetry: C)

Loss = weight-space MSE + behaviour_coef * behavioural KL.

Outputs (in --out_dir): embeddings_<encoder>.npz, embeddings_<encoder>.png,
ae_params_<encoder>.pkl.

Run:  uv run python train_hypernet.py --encoder graph
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

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
    p.add_argument("--steps_subset", default="all", choices=["all", "final"])
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_coef", type=float, default=1.0)
    p.add_argument("--behaviour_coef", type=float, default=1.0)
    p.add_argument("--augment", action="store_true",
                   help="Apply random valid weight-space permutations each step.")
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = infer_arch_config(load_sample_params(args.data_dir))
    net = build_policy_net(cfg, args.activation)
    game = make_game(args.game, args.bias)
    spec = arch_spec_from_config(cfg)
    print(f"[encoder={args.encoder}] arch {cfg['input_dim']}->"
          f"{'->'.join(map(str, cfg['hidden_dims']))}->{cfg['output_dim']} "
          f"norm={cfg['normalization']}")

    ds = load_weight_dataset(args.data_dir, net, game, steps=args.steps_subset)
    print(f"loaded {ds.weights.shape[0]} weight vectors of dim {ds.dim}.")

    Xraw = jnp.asarray(ds.weights)              # raw weights (for sym encoders)
    mean = jnp.asarray(ds.mean)
    std = jnp.asarray(ds.std)
    Xstd = (Xraw - mean) / std                  # per-dim standardized (decoder target)
    true_pol = jnp.asarray(ds.policies)
    state_in = game.state_representation()
    unravel = ds.unravel

    model = SymHyperAE(weight_dim=ds.dim, spec=spec, encoder_name=args.encoder,
                       latent_dim=args.latent_dim)
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, Xstd[:1], Xraw[:1])
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    def behaviour(theta_std):
        theta = theta_std * std + mean
        return jax.nn.softmax(net.apply(unravel(theta), state_in))

    def loss_fn(params, x_std, x_raw, tgt_pol):
        x_hat, _ = model.apply(params, x_std, x_raw)
        recon = jnp.mean((x_hat - x_std) ** 2)
        behav = jnp.mean(kl(tgt_pol, jax.vmap(behaviour)(x_hat)))
        total = args.recon_coef * recon + args.behaviour_coef * behav
        return total, (recon, behav)

    @jax.jit
    def update(params, opt_state, x_std, x_raw, tgt_pol):
        (total, (recon, behav)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, x_std, x_raw, tgt_pol)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, total, recon, behav

    @jax.jit
    def augment(x_raw, key):
        x_raw = random_permute_batch(x_raw, key, spec)
        return x_raw, (x_raw - mean) / std

    tag = args.encoder + ("_aug" if args.augment else "")
    monitor = TrainMonitor(args.out_dir, "ae_" + tag, log_every=args.log_every,
                           ckpt_every=args.ckpt_every)

    start_epoch = 0
    if args.resume:
        ckpt = monitor.resume()
        if ckpt is not None:
            params, opt_state = ckpt["params"], ckpt["opt_state"]
            start_epoch = ckpt["step"] + 1
            print(f"Resumed from ae_{tag}_step{ckpt['step']}.pkl.")

    def checkpoint_state():
        return {"params": params, "opt_state": opt_state, "encoder": args.encoder}

    n = Xraw.shape[0]
    rng = np.random.default_rng(args.seed)
    aug_key = jax.random.PRNGKey(args.seed + 1)
    for epoch in range(start_epoch, args.epochs):
        perm = rng.permutation(n)
        s_total = s_recon = s_behav = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xr, xs, tp = Xraw[idx], Xstd[idx], true_pol[idx]
            if args.augment:
                aug_key, k = jax.random.split(aug_key)
                xr, xs = augment(xr, k)
            params, opt_state, total, recon, behav = update(
                params, opt_state, xs, xr, tp)
            bs = idx.shape[0]
            s_total += float(total) * bs
            s_recon += float(recon) * bs
            s_behav += float(behav) * bs
        monitor.record(epoch, {
            "loss": s_total / n, "recon": s_recon / n, "behav_kl": s_behav / n,
        }, state=checkpoint_state, force_log=epoch == args.epochs - 1)
    monitor.ckpt.save(args.epochs - 1, checkpoint_state)
    monitor.close()

    # --- Embeddings + diagnostics ---
    @jax.jit
    def encode_all(params, x_std, x_raw):
        return model.apply(params, x_std, x_raw, method=SymHyperAE.encode)
    Z = np.asarray(encode_all(params, Xstd, Xraw))

    @jax.jit
    def recon_pol(params, x_std, x_raw):
        x_hat, _ = model.apply(params, x_std, x_raw)
        return jax.vmap(behaviour)(x_hat)
    pred_pol = np.asarray(recon_pol(params, Xstd, Xraw))
    behav_mae = np.mean(np.abs(pred_pol - ds.policies))
    print(f"\nMean |policy - reconstructed policy|: {behav_mae:.4f}")

    pc = perm_consistency(model, params, Xstd, Xraw, mean, std, spec, args.seed)
    print(f"Permutation-consistency (lower=better): {pc:.4f}  "
          f"[mean ||enc(theta) - enc(perm.theta)|| / ||enc(theta)||]")

    np.savez(out_dir / f"embeddings_{tag}.npz",
             z=Z, seeds=ds.seeds, steps=ds.steps, players=ds.players,
             policies=ds.policies)
    _plot(Z, ds, out_dir / f"embeddings_{tag}.png", tag)
    print(f"Wrote outputs with tag '{tag}' to {out_dir}/")


def perm_consistency(model, params, Xstd, Xraw, mean, std, spec, seed, n_probe=512):
    """Mean relative embedding shift under a random valid weight permutation.

    0 == perfectly permutation-invariant encoder.
    """
    idx = np.random.default_rng(seed).choice(Xraw.shape[0],
                                             min(n_probe, Xraw.shape[0]), False)
    xr = Xraw[np.asarray(idx)]
    xs = Xstd[np.asarray(idx)]
    xr_p = random_permute_batch(xr, jax.random.PRNGKey(seed + 7), spec)
    xs_p = (xr_p - mean) / std

    @jax.jit
    def enc(xs_, xr_):
        return model.apply(params, xs_, xr_, method=SymHyperAE.encode)
    z = enc(xs, xr)
    z_p = enc(xs_p, xr_p)
    num = jnp.linalg.norm(z - z_p, axis=1)
    den = jnp.linalg.norm(z, axis=1) + 1e-8
    return float(jnp.mean(num / den))


def _plot(Z, ds, path, tag) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    pcs = Zc @ vt[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc0 = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=ds.policies[:, 0],
                          cmap="coolwarm", s=12)
    axes[0].set_title(f"[{tag}] latent PCA, P(action 0)")
    fig.colorbar(sc0, ax=axes[0], label="P(action 0)")
    sc1 = axes[1].scatter(pcs[:, 0], pcs[:, 1], c=ds.steps, cmap="viridis", s=12)
    axes[1].set_title(f"[{tag}] latent PCA, training step")
    fig.colorbar(sc1, ax=axes[1], label="training step")
    for ax in axes:
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
