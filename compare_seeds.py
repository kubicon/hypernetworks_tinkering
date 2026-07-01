#!/usr/bin/env python3
"""Compare trained networks across seeds.

For each pair of seeds, reports:
  - Policy similarity: how close the learned mixed strategies are (L1, KL, TV distance)
  - Weight similarity: cosine similarity of flattened parameter vectors
  - SV spectra similarity: per-layer singular value spectrum comparison (permutation-invariant)
  - Representational similarity: linear CKA between per-layer activations, evaluated
    on a shared set of probe inputs (invariant to neuron permutation, orthogonal
    rotation and isotropic scaling — so it compares the internal *function*, not the
    raw weights). Trained-vs-trained CKA is contrasted with init-vs-init CKA.

Policy comparison is the most meaningful metric here — weights can differ
substantially due to permutation symmetry of hidden layers while the policy
(the output mixed strategy) may be identical.
"""
import argparse
import copy
import pickle
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

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
from src.networks import GeneralMLP, get_activation
from src.training import PlayerState


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


def load_checkpoint(path: Path) -> tuple[PlayerState, PlayerState]:
    with open(path / "p1_state.pkl", "rb") as f:
        p1 = pickle.load(f)
    with open(path / "p2_state.pkl", "rb") as f:
        p2 = pickle.load(f)
    return p1, p2


def flatten_params(params: dict) -> np.ndarray:
    leaves = jax.tree.leaves(params)
    return np.concatenate([np.asarray(leaf).ravel() for leaf in leaves])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(float), b.astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def extract_sv_spectra(params: dict) -> dict[str, np.ndarray]:
    """Return sorted singular values for each Dense kernel in the network.

    Keys are layer names (e.g. 'Dense_0', 'Dense_1'). Only 2-D kernel arrays
    are included; biases and normalization scales are skipped.
    Sorting by layer name gives a consistent ordering across networks with the
    same architecture.
    """
    spectra: dict[str, np.ndarray] = {}
    layer_params = params.get("params", params)
    for layer_name in sorted(layer_params):
        layer = layer_params[layer_name]
        if not isinstance(layer, dict) or "kernel" not in layer:
            continue
        kernel = np.asarray(layer["kernel"])
        if kernel.ndim != 2:
            continue
        svs = np.linalg.svd(kernel, compute_uv=False)
        spectra[layer_name] = np.sort(svs)[::-1]  # descending
    return spectra


def sv_spectra_metrics(
    spectra_a: dict[str, np.ndarray],
    spectra_b: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Per-layer similarity metrics between two SV spectra.

    For each layer returns:
      cos   — cosine similarity of the SV vectors
      l2    — L2 distance between the SV vectors
      rel   — relative L2: l2 / mean(||sv_a||, ||sv_b||)
    """
    results: dict[str, dict[str, float]] = {}
    for layer in sorted(set(spectra_a) & set(spectra_b)):
        sa, sb = spectra_a[layer], spectra_b[layer]
        cos = cosine_similarity(sa, sb)
        l2 = float(np.linalg.norm(sa - sb))
        denom = (np.linalg.norm(sa) + np.linalg.norm(sb)) / 2.0
        rel = l2 / denom if denom > 0 else float("nan")
        results[layer] = {"cos": cos, "l2": l2, "rel": rel}
    return results


def align_params(params_a: dict, params_b: dict) -> dict:
    """Return a copy of params_a with hidden neurons permuted to best match params_b.

    For each hidden Dense layer we find the permutation of output neurons that
    maximises the cosine similarity to the corresponding neurons in params_b,
    using the Hungarian algorithm.  The permutation is applied consistently:
      - columns of the current layer's kernel and its bias
      - scale of the following Normalization layer (if present)
      - rows of the next layer's kernel (which read from those neurons)

    The output layer is never permuted — its outputs correspond to fixed actions.
    """
    aligned = copy.deepcopy(params_a)
    lp = aligned["params"]
    ref = params_b["params"]

    dense_layers = sorted(k for k in lp if k.startswith("Dense_"))
    hidden_dense = dense_layers[:-1]  # skip output layer

    for i, layer_name in enumerate(hidden_dense):
        kernel_a = np.asarray(lp[layer_name]["kernel"])   # (in, out)
        kernel_b = np.asarray(ref[layer_name]["kernel"])   # (in, out)

        # Cosine similarity between pairs of output neurons (columns)
        norm_a = kernel_a / (np.linalg.norm(kernel_a, axis=0, keepdims=True) + 1e-12)
        norm_b = kernel_b / (np.linalg.norm(kernel_b, axis=0, keepdims=True) + 1e-12)
        corr = norm_a.T @ norm_b  # (n_neurons_a, n_neurons_b)

        _, perm = linear_sum_assignment(-corr)  # maximise similarity

        # Permute this layer's outputs
        lp[layer_name]["kernel"] = lp[layer_name]["kernel"][:, perm]
        lp[layer_name]["bias"] = lp[layer_name]["bias"][perm]

        # Permute normalization scale if present
        norm_layer = f"Normalization_{i}"
        if norm_layer in lp and "scale" in lp[norm_layer]:
            lp[norm_layer]["scale"] = lp[norm_layer]["scale"][perm]

        # Permute the next layer's inputs accordingly
        next_layer = dense_layers[i + 1]
        lp[next_layer]["kernel"] = lp[next_layer]["kernel"][perm, :]

    return aligned


def aligned_distance(params_a: dict, params_b: dict) -> dict[str, float]:
    """Align params_a to params_b, then compute weight-space distances.

    Returns:
      l2        — L2 distance of all parameters after alignment
      cos       — cosine similarity of all parameters after alignment
      rel_l2    — l2 / mean(||params_a||, ||params_b||)
      l2_pre    — L2 distance before alignment (for comparison)
      cos_pre   — cosine similarity before alignment
    """
    flat_a = flatten_params(params_a)
    flat_b = flatten_params(params_b)
    l2_pre = float(np.linalg.norm(flat_a - flat_b))
    cos_pre = cosine_similarity(flat_a, flat_b)

    aligned_a = align_params(params_a, params_b)
    flat_aligned = flatten_params(aligned_a)
    l2 = float(np.linalg.norm(flat_aligned - flat_b))
    cos = cosine_similarity(flat_aligned, flat_b)
    denom = (np.linalg.norm(flat_aligned) + np.linalg.norm(flat_b)) / 2.0
    rel_l2 = l2 / denom if denom > 0 else float("nan")

    return {"l2": l2, "cos": cos, "rel_l2": rel_l2, "l2_pre": l2_pre, "cos_pre": cos_pre}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two activation matrices.

    X, Y have shape (n_samples, n_features_x/y) — the layer's activations on a
    shared set of probe inputs. CKA in [0, 1] measures how similar the two
    representations are, and is invariant to orthogonal transforms (hence neuron
    permutations) and isotropic scaling. 1.0 = identical up to those nuisances.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    denom = float(np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro"))
    return hsic_xy / denom if denom > 0 else float("nan")


def make_probe_inputs(
    game: NormalFormGame, n: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Build a shared batch of probe inputs for representational comparison.

    The network is only ever trained on the single fixed game state, so to probe
    the internal function we evaluate it elsewhere:
      local — a tight Gaussian cloud around the true game state (behaviour at the
              operating point the network actually learned).
      broad — uniform samples across input space (behaviour as a general function).
    """
    state = np.asarray(game.state_representation(), dtype=float)
    dim = state.shape[0]
    if mode == "local":
        return state[None, :] + rng.normal(0.0, 0.1, size=(n, dim))
    if mode == "broad":
        return rng.uniform(-2.0, 2.0, size=(n, dim))
    raise ValueError(f"Unknown probe mode: {mode!r}")


def get_hidden_reps(
    network: "GeneralMLP",
    params: dict,
    X: np.ndarray,
    hidden_dims: tuple[int, ...],
    activation: str,
    normalization: str | None,
) -> dict[str, np.ndarray]:
    """Return per-layer post-activation representations on probe inputs X.

    Keys are 'hidden_0', 'hidden_1', ..., and 'logits' (the network output).
    Hidden representations are the values that actually feed the next layer:
    activation(normalization(dense(.))).
    """
    act_fn = get_activation(activation)
    _, state = network.apply(
        params, jnp.asarray(X), capture_intermediates=True, mutable=["intermediates"]
    )
    inter = state["intermediates"]
    has_norm = normalization not in (None, "none")
    reps: dict[str, np.ndarray] = {}
    for i in range(len(hidden_dims)):
        key = f"Normalization_{i}" if has_norm else f"Dense_{i}"
        pre = inter[key]["__call__"][0]
        reps[f"hidden_{i}"] = np.asarray(act_fn(pre))
    reps["logits"] = np.asarray(inter[f"Dense_{len(hidden_dims)}"]["__call__"][0])
    return reps


def policy_metrics(
    p: np.ndarray, q: np.ndarray
) -> dict[str, float]:
    """Compute distances between two probability vectors p and q."""
    eps = 1e-12
    l1 = float(np.sum(np.abs(p - q)))
    tv = l1 / 2.0
    kl_pq = float(np.sum(p * np.log((p + eps) / (q + eps))))
    kl_qp = float(np.sum(q * np.log((q + eps) / (p + eps))))
    js = (kl_pq + kl_qp) / 2.0
    cosine = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + eps))
    return {"L1": l1, "TV": tv, "KL(p||q)": kl_pq, "KL(q||p)": kl_qp, "JS": js, "cos": cosine}


def get_policy(network: "GeneralMLP", params: dict, game: NormalFormGame) -> np.ndarray:
    state = game.state_representation()
    logits = network.apply(params, state)
    return np.asarray(jax.nn.softmax(logits))


def discover_seeds(data_root: Path, game: str) -> list[int]:
    game_dir = data_root / game
    if not game_dir.exists():
        return []
    seeds = sorted(int(d.name) for d in game_dir.iterdir() if d.is_dir() and d.name.isdigit())
    return seeds


def find_step(seed_dir: Path, step: int | None) -> int:
    available = sorted(int(d.name) for d in seed_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if not available:
        raise FileNotFoundError(f"No checkpoint directories in {seed_dir}")
    if step is None:
        return available[-1]
    if step not in available:
        raise FileNotFoundError(
            f"Step {step} not found in {seed_dir}. Available: {available}"
        )
    return step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare trained networks across seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--game", default="bmp", choices=GAME_CHOICES,
                   help="mp=matching_pennies, bmp=biased_matching_pennies, "
                        "rps=rock_paper_scissors, brps=biased_rock_paper_scissors.")
    p.add_argument("--bias", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Seeds to compare. Defaults to all seeds found under "
                        "data/<game_long_name>/.")
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load. Defaults to the latest available step.")
    p.add_argument("--data_dir", type=Path, default=Path("data"),
                   help="Root directory where checkpoints are stored.")
    p.add_argument("--params", type=str, default="params",
                   choices=["params", "ema_params", "periodic_params"],
                   help="Which parameter set to compare.")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 32])
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--normalization", type=str, default="rms_norm")
    p.add_argument("--cka_probe_n", type=int, default=256,
                   help="Number of shared probe inputs used for CKA representational similarity.")
    return p.parse_args()


def print_table(
    seeds: list[int],
    player: str,
    metric_name: str,
    matrix: np.ndarray,
) -> None:
    w = max(len(str(s)) for s in seeds)
    col_w = max(7, w)
    header = f"{'':>{w}} | " + " | ".join(f"{s:>{col_w}}" for s in seeds)
    print(f"\n{player} — {metric_name}")
    print(header)
    print("-" * len(header))
    for i, si in enumerate(seeds):
        row = f"{si:>{w}} | "
        row += " | ".join(
            f"{'---':>{col_w}}" if i == j else f"{matrix[i, j]:>{col_w}.4f}"
            for j in range(len(seeds))
        )
        print(row)


def main() -> None:
    args = parse_args()

    game = make_game(args)
    n_actions = game.matrix.shape[0]
    state_dim = int(game.state_representation().shape[0])

    network = GeneralMLP(
        hidden_dims=tuple(args.hidden_dims),
        output_dim=n_actions,
        activation=args.activation,
        normalization=args.normalization,
    )

    game_dir_name = GAME_LONG_NAME[args.game]

    seeds = args.seeds
    if seeds is None:
        seeds = discover_seeds(args.data_dir, game_dir_name)
        if not seeds:
            print(f"No seed directories found under {args.data_dir / game_dir_name}")
            return
        print(f"Auto-discovered seeds: {seeds}")

    # Load checkpoints
    records: dict[int, dict] = {}
    for seed in seeds:
        seed_dir = args.data_dir / game_dir_name / str(seed)
        if not seed_dir.exists():
            print(f"[skip] seed {seed}: directory {seed_dir} not found")
            continue
        step = find_step(seed_dir, args.step)
        ckpt_dir = seed_dir / str(step)
        p1_state, p2_state = load_checkpoint(ckpt_dir)
        p1_params = getattr(p1_state, args.params)
        p2_params = getattr(p2_state, args.params)

        init_dir = seed_dir / "0"
        p1_init = p2_init = None
        if init_dir.exists():
            p1_state_init, p2_state_init = load_checkpoint(init_dir)
            p1_init = getattr(p1_state_init, args.params)
            p2_init = getattr(p2_state_init, args.params)

        records[seed] = {
            "step": step,
            "p1_policy": get_policy(network, p1_params, game),
            "p2_policy": get_policy(network, p2_params, game),
            "p1_params": p1_params,
            "p2_params": p2_params,
            "p1_flat": flatten_params(p1_params),
            "p2_flat": flatten_params(p2_params),
            "p1_sv": extract_sv_spectra(p1_params),
            "p2_sv": extract_sv_spectra(p2_params),
            "p1_init": p1_init,
            "p2_init": p2_init,
        }
        print(f"  seed {seed}: loaded step {step}  |  "
              f"p1 policy={np.array2string(records[seed]['p1_policy'], precision=4, suppress_small=True)}  |  "
              f"p2 policy={np.array2string(records[seed]['p2_policy'], precision=4, suppress_small=True)}")

    loaded_seeds = [s for s in seeds if s in records]
    n = len(loaded_seeds)
    if n < 2:
        print("Need at least 2 loaded seeds to compare.")
        return

    print(f"\nLoaded {n} seeds using '{args.params}' at game '{args.game}'"
          + (f" bias={args.bias}" if args.game in BIASED_GAME_CHOICES else ""))
    print(f"Checkpoint step: {records[loaded_seeds[0]]['step']}")

    # Per-metric pairwise matrices
    policy_metric_keys = ["L1", "TV", "JS", "cos"]
    align_keys = ["cos", "l2", "rel_l2", "cos_pre", "l2_pre"]
    p1_metrics: dict[str, np.ndarray] = {k: np.full((n, n), np.nan) for k in policy_metric_keys + ["weight_cos"]}
    p2_metrics: dict[str, np.ndarray] = {k: np.full((n, n), np.nan) for k in policy_metric_keys + ["weight_cos"]}
    p1_align: dict[str, np.ndarray] = {k: np.full((n, n), np.nan) for k in align_keys}
    p2_align: dict[str, np.ndarray] = {k: np.full((n, n), np.nan) for k in align_keys}
    p1_align_init: dict[str, np.ndarray] = {k: np.full((n, n), np.nan) for k in align_keys}
    p2_align_init: dict[str, np.ndarray] = {k: np.full((n, n), np.nan) for k in align_keys}

    has_init = all(records[s]["p1_init"] is not None for s in loaded_seeds)

    for (i, si), (j, sj) in combinations(enumerate(loaded_seeds), 2):
        ri, rj = records[si], records[sj]

        pm1 = policy_metrics(ri["p1_policy"], rj["p1_policy"])
        pm2 = policy_metrics(ri["p2_policy"], rj["p2_policy"])
        wcos1 = cosine_similarity(ri["p1_flat"], rj["p1_flat"])
        wcos2 = cosine_similarity(ri["p2_flat"], rj["p2_flat"])
        ad1 = aligned_distance(ri["p1_params"], rj["p1_params"])
        ad2 = aligned_distance(ri["p2_params"], rj["p2_params"])

        for k in policy_metric_keys:
            p1_metrics[k][i, j] = p1_metrics[k][j, i] = pm1[k]
            p2_metrics[k][i, j] = p2_metrics[k][j, i] = pm2[k]
        p1_metrics["weight_cos"][i, j] = p1_metrics["weight_cos"][j, i] = wcos1
        p2_metrics["weight_cos"][i, j] = p2_metrics["weight_cos"][j, i] = wcos2
        for k in align_keys:
            p1_align[k][i, j] = p1_align[k][j, i] = ad1[k]
            p2_align[k][i, j] = p2_align[k][j, i] = ad2[k]

        if has_init:
            ai1 = aligned_distance(ri["p1_init"], rj["p1_init"])
            ai2 = aligned_distance(ri["p2_init"], rj["p2_init"])
            for k in align_keys:
                p1_align_init[k][i, j] = p1_align_init[k][j, i] = ai1[k]
                p2_align_init[k][i, j] = p2_align_init[k][j, i] = ai2[k]

    # SV spectra: pairwise matrices per layer and metric
    layers = sorted(records[loaded_seeds[0]]["p1_sv"].keys())
    sv_metrics_keys = ("cos", "l2", "rel")
    sv_matrices: dict[str, np.ndarray] = {
        f"{player}_{layer}_{metric}": np.full((n, n), np.nan)
        for player in ("p1", "p2")
        for layer in layers
        for metric in sv_metrics_keys
    }
    for (i, si), (j, sj) in combinations(enumerate(loaded_seeds), 2):
        ri, rj = records[si], records[sj]
        for player in ("p1", "p2"):
            per_layer = sv_spectra_metrics(ri[f"{player}_sv"], rj[f"{player}_sv"])
            for layer, metrics in per_layer.items():
                for metric, val in metrics.items():
                    key = f"{player}_{layer}_{metric}"
                    sv_matrices[key][i, j] = sv_matrices[key][j, i] = val

    # Summary stats
    def _off_diagonal(mat: np.ndarray) -> np.ndarray:
        idx = np.where(~np.isnan(mat) & (1 - np.eye(n, dtype=bool)))
        return mat[~np.isnan(mat) & ~np.eye(n, dtype=bool)]

    print("\n" + "="*60)
    print("POLICY SIMILARITY SUMMARY")
    print("="*60)
    for player_label, metrics in [("Player 1", p1_metrics), ("Player 2", p2_metrics)]:
        print(f"\n{player_label}")
        header = f"{'metric':>12} | {'mean':>8} | {'std':>8} | {'min':>8} | {'max':>8}"
        print(header)
        print("-" * len(header))
        for k in policy_metric_keys + ["weight_cos"]:
            vals = _off_diagonal(metrics[k])
            if len(vals) == 0:
                continue
            print(f"{k:>12} | {np.mean(vals):>8.4f} | {np.std(vals):>8.4f} | {np.min(vals):>8.4f} | {np.max(vals):>8.4f}")

    print("\n" + "="*60)
    print("SV SPECTRA SIMILARITY (permutation-invariant)")
    print("="*60)
    for player_label, player_key in [("Player 1", "p1"), ("Player 2", "p2")]:
        print(f"\n{player_label}")
        header = f"{'layer':>12} | {'cos mean':>9} | {'cos min':>9} | {'rel_l2 mean':>11} | {'rel_l2 max':>10}"
        print(header)
        print("-" * len(header))
        for layer in layers:
            cos_vals = _off_diagonal(sv_matrices[f"{player_key}_{layer}_cos"])
            rel_vals = _off_diagonal(sv_matrices[f"{player_key}_{layer}_rel"])
            print(
                f"{layer:>12} | "
                f"{np.mean(cos_vals):>9.4f} | "
                f"{np.min(cos_vals):>9.4f} | "
                f"{np.mean(rel_vals):>11.4f} | "
                f"{np.max(rel_vals):>10.4f}"
            )

    for layer in layers:
        for metric in ("cos", "rel"):
            for player_label, player_key in [("P1", "p1"), ("P2", "p2")]:
                print_table(
                    loaded_seeds,
                    f"{player_label} {layer}",
                    metric,
                    sv_matrices[f"{player_key}_{layer}_{metric}"],
                )

    # Pairwise tables for TV and weight_cos
    for metric in ["TV", "cos", "weight_cos"]:
        for player_label, metrics in [("P1", p1_metrics), ("P2", p2_metrics)]:
            print_table(loaded_seeds, player_label, metric, metrics[metric])

    print("\n" + "="*60)
    min_p1 = float(np.nanmin(_off_diagonal(p1_metrics["weight_cos"])))
    min_p2 = float(np.nanmin(_off_diagonal(p2_metrics["weight_cos"])))
    max_p1 = float(np.nanmax(_off_diagonal(p1_metrics["weight_cos"])))
    max_p2 = float(np.nanmax(_off_diagonal(p2_metrics["weight_cos"])))
    print(f"Min weight cosine similarity — P1: {min_p1:.4f}  |  P2: {min_p2:.4f}")
    print(f"Max weight cosine similarity — P1: {max_p1:.4f}  |  P2: {max_p2:.4f}")

    def _print_align_summary(label: str, p1: dict, p2: dict) -> None:
        print(f"\n{label}")
        header = f"{'metric':>10} | {'mean':>8} | {'std':>8} | {'min':>8} | {'max':>8}"
        for player_label, align in [("Player 1", p1), ("Player 2", p2)]:
            print(f"\n  {player_label}")
            print("  " + header)
            print("  " + "-" * len(header))
            for k in align_keys:
                vals = _off_diagonal(align[k])
                lab = k.replace("_pre", " (pre)")
                print(f"  {lab:>10} | {np.mean(vals):>8.4f} | {np.std(vals):>8.4f} | {np.min(vals):>8.4f} | {np.max(vals):>8.4f}")

    print("\n" + "="*60)
    print("PERMUTATION-ALIGNED WEIGHT DISTANCE")
    print("="*60)
    _print_align_summary("Trained weights", p1_align, p2_align)
    if has_init:
        _print_align_summary("Initial weights (step 0, sanity-check baseline)", p1_align_init, p2_align_init)
    else:
        print("\n[step 0 checkpoint not found — skipping initial weight baseline]")

    for metric in ("cos", "l2"):
        for player_label, align in [("P1", p1_align), ("P2", p2_align)]:
            print_table(loaded_seeds, player_label, f"aligned {metric}", align[metric])

    if has_init:
        print("\n" + "="*60)
        print("WEIGHT TRAVEL (init → final, per seed)")
        print("="*60)
        header = f"{'seed':>6} | {'P1 cos':>8} | {'P1 l2':>8} | {'P1 rel_l2':>10} | {'P2 cos':>8} | {'P2 l2':>8} | {'P2 rel_l2':>10}"
        print(header)
        print("-" * len(header))
        p1_travels, p2_travels = [], []
        for seed in loaded_seeds:
            r = records[seed]
            t1 = aligned_distance(r["p1_init"], r["p1_params"])
            t2 = aligned_distance(r["p2_init"], r["p2_params"])
            p1_travels.append(t1)
            p2_travels.append(t2)
            print(
                f"{seed:>6} | {t1['cos']:>8.4f} | {t1['l2']:>8.4f} | {t1['rel_l2']:>10.4f} | "
                f"{t2['cos']:>8.4f} | {t2['l2']:>8.4f} | {t2['rel_l2']:>10.4f}"
            )
        print("-" * len(header))
        for label, travels in [("mean", lambda v: np.mean(v)), ("std", lambda v: np.std(v))]:
            p1_cos = travels([t["cos"] for t in p1_travels])
            p1_l2  = travels([t["l2"]  for t in p1_travels])
            p1_rel = travels([t["rel_l2"] for t in p1_travels])
            p2_cos = travels([t["cos"] for t in p2_travels])
            p2_l2  = travels([t["l2"]  for t in p2_travels])
            p2_rel = travels([t["rel_l2"] for t in p2_travels])
            print(f"{label:>6} | {p1_cos:>8.4f} | {p1_l2:>8.4f} | {p1_rel:>10.4f} | {p2_cos:>8.4f} | {p2_l2:>8.4f} | {p2_rel:>10.4f}")

    # ----- Representational similarity (CKA) -----
    print("\n" + "="*60)
    print("REPRESENTATIONAL SIMILARITY (linear CKA) across seeds")
    print("="*60)
    print("CKA is permutation/rotation/scale-invariant: it compares the internal\n"
          "function on shared probe inputs, not the raw weights. Trained-vs-trained\n"
          "is contrasted with the init-vs-init baseline (1.0 = identical representation).")

    rep_layers = [f"hidden_{i}" for i in range(len(args.hidden_dims))] + ["logits"]
    rng = np.random.default_rng(0)

    def _cka_matrix(reps_by_seed: dict[int, dict], layer: str) -> np.ndarray:
        M = np.full((n, n), np.nan)
        for (i, si), (j, sj) in combinations(enumerate(loaded_seeds), 2):
            c = linear_cka(reps_by_seed[si][layer], reps_by_seed[sj][layer])
            M[i, j] = M[j, i] = c
        np.fill_diagonal(M, 1.0)
        return M

    for probe_label, probe_mode in [("local (state ± 0.1)", "local"),
                                    ("broad (uniform[-2, 2])", "broad")]:
        X = make_probe_inputs(game, args.cka_probe_n, probe_mode, rng)
        trained: dict[str, dict[int, dict]] = {"p1": {}, "p2": {}}
        init: dict[str, dict[int, dict]] = {"p1": {}, "p2": {}}
        for seed in loaded_seeds:
            r = records[seed]
            for pkey in ("p1", "p2"):
                trained[pkey][seed] = get_hidden_reps(
                    network, r[f"{pkey}_params"], X,
                    tuple(args.hidden_dims), args.activation, args.normalization)
                if has_init:
                    init[pkey][seed] = get_hidden_reps(
                        network, r[f"{pkey}_init"], X,
                        tuple(args.hidden_dims), args.activation, args.normalization)

        print(f"\nProbe: {probe_label}, N={args.cka_probe_n}")
        for player_label, pkey in [("Player 1", "p1"), ("Player 2", "p2")]:
            print(f"\n{player_label}")
            header = f"{'layer':>10} | {'trained mean':>12} | {'trained min':>11}"
            if has_init:
                header += f" | {'init mean':>9} | {'init min':>8}"
            print(header)
            print("-" * len(header))
            for layer in rep_layers:
                tv = _off_diagonal(_cka_matrix(trained[pkey], layer))
                row = f"{layer:>10} | {np.mean(tv):>12.4f} | {np.min(tv):>11.4f}"
                if has_init:
                    iv = _off_diagonal(_cka_matrix(init[pkey], layer))
                    row += f" | {np.mean(iv):>9.4f} | {np.min(iv):>8.4f}"
                print(row)

        # Full pairwise tables for the broad probe (general-function comparison).
        if probe_mode == "broad":
            for layer in rep_layers:
                for player_label, pkey in [("P1", "p1"), ("P2", "p2")]:
                    print_table(loaded_seeds, f"{player_label} {layer}",
                                "CKA (broad)", _cka_matrix(trained[pkey], layer))


if __name__ == "__main__":
    main()
