"""Symmetry-aware encoders for the hypernetwork auto-encoder.

A GeneralMLP's weight space has a hidden-neuron permutation symmetry -- the
units of each hidden layer can be permuted independently (one symmetric group
per hidden layer) -- plus, when RMSNorm is present, a per-layer global-scale
redundancy (see HypernetArchitecture.md). The encoders here exploit it.

Everything is driven by an ``ArchSpec`` inferred from the actual checkpoints, so
the same code works for any GeneralMLP regardless of input/output size, hidden
widths, depth, or whether normalization is present (e.g. biased_matching_pennies
4->32->32->2 with RMSNorm, or biased_rock_paper_scissors 9->8->8->3 without).

The encoders read the **raw** flat weight vector (per-dimension standardization
would break permutation symmetry) and apply a per-tensor, per-sample
normalization internally. The decoder still reconstructs the standardized
vector. Selectable encoders: "mlp", "canon", "deepsets", "graph", "equiv".
"""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Architecture spec: describes the weight-space structure and its symmetry
# ---------------------------------------------------------------------------

class ArchSpec(NamedTuple):
    """Structure of a GeneralMLP's flat weight vector and its permutation group.

    weight_spec lists (name, size, shape) in jax.flatten_util.ravel_pytree
    order: bias then kernel for Dense_0..Dense_H, then the per-hidden-layer
    norm scales s0..s_{H-1} (absent if has_norm is False). Hidden layer g has
    width hidden_dims[g]; its permutation acts on the columns of k{g}, the bias
    b{g}, the scale s{g}, and the rows of k{g+1}.
    """
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    has_norm: bool
    weight_spec: tuple[tuple[str, int, tuple[int, ...]], ...]

    @property
    def n_hidden(self) -> int:
        return len(self.hidden_dims)


def build_arch_spec(input_dim, output_dim, hidden_dims, has_norm) -> ArchSpec:
    dims = [input_dim, *hidden_dims, output_dim]
    H = len(hidden_dims)
    spec = []
    for h in range(H + 1):                       # Dense_0 .. Dense_H
        in_d, out_d = dims[h], dims[h + 1]
        spec.append((f"b{h}", out_d, (out_d,)))          # bias before kernel
        spec.append((f"k{h}", in_d * out_d, (in_d, out_d)))
    if has_norm:
        for h in range(H):
            spec.append((f"s{h}", hidden_dims[h], (hidden_dims[h],)))
    return ArchSpec(input_dim, output_dim, tuple(hidden_dims), has_norm,
                    tuple(spec))


def arch_spec_from_config(cfg: dict) -> ArchSpec:
    """Build an ArchSpec from hypernet.infer_arch_config output."""
    return build_arch_spec(cfg["input_dim"], cfg["output_dim"],
                           cfg["hidden_dims"], cfg["normalization"] != "none")


# ---------------------------------------------------------------------------
# Structured (un)flattening and the permutation group action
# ---------------------------------------------------------------------------

def split_flat(flat: jnp.ndarray, spec: ArchSpec) -> dict[str, jnp.ndarray]:
    """Split a (..., D) flat weight vector into named tensors."""
    out, i = {}, 0
    lead = flat.shape[:-1]
    for name, n, shape in spec.weight_spec:
        seg = jax.lax.dynamic_slice_in_dim(flat, i, n, axis=-1)
        out[name] = seg.reshape(lead + shape)
        i += n
    return out


def merge_flat(t: dict[str, jnp.ndarray], spec: ArchSpec) -> jnp.ndarray:
    name0, _, shape0 = spec.weight_spec[0]
    lead = t[name0].shape[:t[name0].ndim - len(shape0)]
    parts = [t[name].reshape(lead + (n,)) for name, n, _ in spec.weight_spec]
    return jnp.concatenate(parts, axis=-1)


def permute_params(flat, perms, spec: ArchSpec) -> jnp.ndarray:
    """Apply a hidden-neuron permutation (one perm per hidden layer) to a single
    flat weight vector. The function computed by the net is left unchanged."""
    t = split_flat(flat, spec)
    for g in range(spec.n_hidden):
        pg = perms[g]
        t[f"b{g}"] = t[f"b{g}"][pg]
        t[f"k{g}"] = t[f"k{g}"][:, pg]            # columns = outputs of Dense_g
        t[f"k{g + 1}"] = t[f"k{g + 1}"][pg]       # rows = inputs of Dense_{g+1}
        if spec.has_norm:
            t[f"s{g}"] = t[f"s{g}"][pg]
    return merge_flat(t, spec)


def random_permute_batch(flat, key, spec: ArchSpec) -> jnp.ndarray:
    """Independent random valid permutation for each row of a batch."""
    b = flat.shape[0]
    gkeys = jax.random.split(key, spec.n_hidden)
    perms = [
        jax.vmap(lambda k, w=spec.hidden_dims[g]: jax.random.permutation(k, w))(
            jax.random.split(gkeys[g], b))
        for g in range(spec.n_hidden)
    ]
    return jax.vmap(lambda f, *ps: permute_params(f, ps, spec),
                    in_axes=(0,) + (0,) * spec.n_hidden)(flat, *perms)


def _instance_norm(x: jnp.ndarray) -> jnp.ndarray:
    """Per-tensor, per-sample standardization over all non-batch axes."""
    axes = tuple(range(1, x.ndim))
    mean = x.mean(axes, keepdims=True)
    std = x.std(axes, keepdims=True) + 1e-6
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Shared structuring for the structural encoders (deepsets / graph / equiv)
# ---------------------------------------------------------------------------

def _node_raw_and_edges(flat_raw, spec: ArchSpec):
    """Per-hidden-layer raw node features, inter-hidden edge matrices, and a
    global feature (the output bias). All instance-normalized.

    node_raw[g]: (B, w_g, F_g) -- bias, scale, plus anchored incoming weights
                 for the first hidden layer and anchored outgoing weights for
                 the last.
    edges[g]:    (B, w_{g-1}, w_g) for g = 1..H-1 (the inter-hidden kernels).
    glob:        (B, output_dim) -- the output bias.
    """
    t = {k: _instance_norm(v) for k, v in split_flat(flat_raw, spec).items()}
    H = spec.n_hidden
    node_raw = []
    for g in range(H):
        parts = [t[f"b{g}"][..., None]]
        if spec.has_norm:
            parts.append(t[f"s{g}"][..., None])
        if g == 0:                                # incoming anchored to inputs
            parts.append(t["k0"].transpose(0, 2, 1))      # (B, w0, in)
        if g == H - 1:                            # outgoing anchored to outputs
            parts.append(t[f"k{H}"])                       # (B, w_{H-1}, out)
        node_raw.append(jnp.concatenate(parts, axis=-1))
    edges = {g: t[f"k{g}"] for g in range(1, H)}  # (B, w_{g-1}, w_g)
    glob = t[f"b{H}"]                             # output bias (anchored)
    return node_raw, edges, glob


# ---------------------------------------------------------------------------
# D. Canonicalization (sort neurons) -> MLP
# ---------------------------------------------------------------------------

def canonicalize(flat, spec: ArchSpec) -> jnp.ndarray:
    """Sort each hidden layer's neurons by an invariant key (incoming-weight L2
    norm) so functionally equivalent nets align. Single sample."""
    t = split_flat(flat, spec)
    perms = [jnp.argsort(jnp.linalg.norm(t[f"k{g}"], axis=0))
             for g in range(spec.n_hidden)]
    return permute_params(flat, tuple(perms), spec)


class CanonMLPEncoder(nn.Module):
    spec: ArchSpec
    latent_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)

    @nn.compact
    def __call__(self, flat_raw: jnp.ndarray) -> jnp.ndarray:
        canon = jax.vmap(lambda f: canonicalize(f, self.spec))(flat_raw)
        t = {k: _instance_norm(v) for k, v in split_flat(canon, self.spec).items()}
        x = merge_flat(t, self.spec)
        for h in self.hidden_dims:
            x = nn.gelu(nn.Dense(h)(x))
        return nn.Dense(self.latent_dim)(x)


# ---------------------------------------------------------------------------
# B. Neuron-token DeepSets
# ---------------------------------------------------------------------------

class DeepSetsEncoder(nn.Module):
    """Permutation-invariant set encoder: pool per-hidden-layer neuron tokens,
    plus doubly-pooled invariant summaries of the inter-hidden edge matrices."""
    spec: ArchSpec
    latent_dim: int
    feat: int = 64

    @nn.compact
    def __call__(self, flat_raw: jnp.ndarray) -> jnp.ndarray:
        node_raw, edges, glob = _node_raw_and_edges(flat_raw, self.spec)

        pooled = []
        for g, tok in enumerate(node_raw):
            phi = nn.Dense(self.feat, name=f"tok{g}_1")(tok)
            phi = nn.Dense(self.feat, name=f"tok{g}_2")(nn.gelu(phi))
            pooled.append(phi.mean(1))            # (B, feat)

        for g, w in edges.items():
            wexp = w[..., None]                   # (B, w_{g-1}, w_g, 1)
            psi = nn.Dense(self.feat, name=f"edge{g}_1")(wexp)
            psi = nn.Dense(self.feat, name=f"edge{g}_2")(nn.gelu(psi))
            pooled.append(psi.mean((1, 2)))       # invariant to both perms
            row_n = jnp.linalg.norm(w, axis=2)
            col_n = jnp.linalg.norm(w, axis=1)
            pooled.append(jnp.stack(
                [row_n.mean(1), row_n.std(1), col_n.mean(1), col_n.std(1)], -1))

        x = jnp.concatenate(pooled + [glob], axis=-1)
        x = nn.gelu(nn.Dense(self.feat * 2)(x))
        return nn.Dense(self.latent_dim)(x)


# ---------------------------------------------------------------------------
# A. Graph metanetwork (edge-gated message passing)
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """Message passing between adjacent hidden layers across the inter-hidden
    kernels. Messages are edge-gated MLPs; pooling over the permutable node
    axes gives a permutation-invariant embedding. Works for any depth."""
    spec: ArchSpec
    latent_dim: int
    feat: int = 64
    rounds: int = 3

    @nn.compact
    def __call__(self, flat_raw: jnp.ndarray) -> jnp.ndarray:
        node_raw, edges, glob = _node_raw_and_edges(flat_raw, self.spec)
        H = self.spec.n_hidden
        nodes = [nn.Dense(self.feat, name=f"in{g}")(f)
                 for g, f in enumerate(node_raw)]

        def gated(send, edge, agg_axis, name):
            # send broadcast over the target axis; edge weight concatenated.
            f = self.feat
            if agg_axis == 1:                     # send over rows -> target cols
                send_b = jnp.broadcast_to(send[:, :, None, :],
                                          edge.shape + (f,))
            else:                                 # send over cols -> target rows
                send_b = jnp.broadcast_to(send[:, None, :, :],
                                          edge.shape + (f,))
            m = nn.Dense(f, name=name)(
                jnp.concatenate([send_b, edge[..., None]], -1))
            return jax.nn.gelu(m).mean(agg_axis)

        for r in range(self.rounds):
            new = list(nodes)
            for g in range(H):
                inc = [nodes[g]]
                if g >= 1:                         # message from g-1 via edges[g]
                    inc.append(gated(nodes[g - 1], edges[g], 1, f"m{r}_{g}_dn"))
                if g <= H - 2:                     # message from g+1 via edges[g+1]
                    inc.append(gated(nodes[g + 1], edges[g + 1], 2, f"m{r}_{g}_up"))
                upd = nn.Dense(self.feat, name=f"u{r}_{g}_1")(
                    jnp.concatenate(inc, -1))
                upd = nn.Dense(self.feat, name=f"u{r}_{g}_2")(jax.nn.gelu(upd))
                new[g] = nodes[g] + upd
            nodes = new

        pool = [fn(n, 1) for n in nodes for fn in (jnp.mean, jnp.max)]
        g = jnp.concatenate(pool + [glob], -1)
        g = jax.nn.gelu(nn.Dense(self.feat * 2)(g))
        return nn.Dense(self.latent_dim)(g)


# ---------------------------------------------------------------------------
# C. NFN / DWSNet-style equivariant-linear layers
# ---------------------------------------------------------------------------

class EquivEncoder(nn.Module):
    """Stacked equivariant-linear layers over per-hidden-layer node features.

    Each layer mixes, for every hidden layer g: its own features, a global pool
    (broadcast), and edge-weighted linear pools from the adjacent layers
    (sum_k w[.,k] h_neighbour) -- all equivariant to independent permutations of
    every hidden layer. Final invariant pool over each layer's neurons."""
    spec: ArchSpec
    latent_dim: int
    feat: int = 64
    layers: int = 3

    @nn.compact
    def __call__(self, flat_raw: jnp.ndarray) -> jnp.ndarray:
        node_raw, edges, glob = _node_raw_and_edges(flat_raw, self.spec)
        H = self.spec.n_hidden
        nodes = [nn.Dense(self.feat, name=f"in{g}")(f)
                 for g, f in enumerate(node_raw)]

        for li in range(self.layers):
            new = []
            for g in range(H):
                parts = [nodes[g],
                         jnp.broadcast_to(nodes[g].mean(1, keepdims=True),
                                          nodes[g].shape)]
                if g >= 1:    # from g-1: sum over rows of edges[g]
                    e = edges[g]
                    parts.append(jnp.einsum("bij,bif->bjf", e, nodes[g - 1])
                                 / e.shape[1])
                if g <= H - 2:  # from g+1: sum over cols of edges[g+1]
                    e = edges[g + 1]
                    parts.append(jnp.einsum("bij,bjf->bif", e, nodes[g + 1])
                                 / e.shape[2])
                upd = nn.Dense(self.feat, name=f"l{li}_{g}")(
                    jnp.concatenate(parts, -1))
                new.append(nodes[g] + jax.nn.gelu(upd))
            nodes = new

        g = jnp.concatenate([n.mean(1) for n in nodes] + [glob], -1)
        g = jax.nn.gelu(nn.Dense(self.feat * 2)(g))
        return nn.Dense(self.latent_dim)(g)


# ---------------------------------------------------------------------------
# Plain MLP baseline (consumes standardized weights) and the factory
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    spec: ArchSpec                                 # unused; uniform interface
    latent_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)

    @nn.compact
    def __call__(self, flat_std: jnp.ndarray) -> jnp.ndarray:
        x = flat_std
        for h in self.hidden_dims:
            x = nn.gelu(nn.Dense(h)(x))
        return nn.Dense(self.latent_dim)(x)


_ENCODERS = {
    "mlp": MLPEncoder,
    "canon": CanonMLPEncoder,
    "deepsets": DeepSetsEncoder,
    "graph": GraphEncoder,
    "equiv": EquivEncoder,
}


def make_encoder(name: str, latent_dim: int, spec: ArchSpec) -> nn.Module:
    if name not in _ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Options: {sorted(_ENCODERS)}")
    return _ENCODERS[name](spec=spec, latent_dim=latent_dim)


# ---------------------------------------------------------------------------
# Pluggable auto-encoder
# ---------------------------------------------------------------------------

class SymHyperAE(nn.Module):
    """Auto-encoder with a selectable (optionally symmetry-aware) encoder.

    The encoder reads raw weights (except "mlp", which reads the standardized
    vector); the decoder always reconstructs the standardized weight vector.
    """
    weight_dim: int
    spec: ArchSpec
    encoder_name: str = "mlp"
    latent_dim: int = 8
    dec_dims: tuple[int, ...] = (128, 256)

    def setup(self) -> None:
        self.encoder = make_encoder(self.encoder_name, self.latent_dim, self.spec)
        self.dec_layers = [nn.Dense(h) for h in self.dec_dims]
        self.to_weights = nn.Dense(self.weight_dim)

    def encode(self, theta_std, theta_raw):
        inp = theta_std if self.encoder_name == "mlp" else theta_raw
        return self.encoder(inp)

    def decode(self, z):
        x = z
        for layer in self.dec_layers:
            x = nn.gelu(layer(x))
        return self.to_weights(x)

    def __call__(self, theta_std, theta_raw):
        z = self.encode(theta_std, theta_raw)
        return self.decode(z), z
