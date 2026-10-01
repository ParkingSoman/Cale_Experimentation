"""Turn activation statistics into neuron orderings and level definitions.

Neuron score (per layer, per neuron j), modality-balanced:
  rms_j   = sqrt( 0.5 * E_text[a_j^2] + 0.5 * E_img[a_j^2] )
  score_j = rms_j * ||down_proj[:, j]||        (structured Wanda-style)
FLAP-style score (what remains after a mean-compensation repair):
  var_j   = 0.5 * Var_text[a_j] + 0.5 * Var_img[a_j]
  flap_j  = var_j * ||down_proj[:, j]||^2
"""
import json

import mlx.core as mx
import numpy as np

GROUP = 64
N_LAYERS, WIDTH = 48, 15360


def load_stats(path="results/stats-bf16.npz"):
    return dict(np.load(path))


def _moments(st):
    nt, ni = float(st["n_text"]), max(float(st["n_img"]), 1.0)
    e2 = 0.5 * st["sq_text"] / nt + 0.5 * st["sq_img"] / ni
    e1 = 0.5 * st["sum_text"] / nt + 0.5 * st["sum_img"] / ni
    return e1, e2, nt, ni


def scores(st, kind="wanda"):
    e1, e2, nt, ni = _moments(st)
    dn = st["down_norm"].astype(np.float64)
    if kind == "wanda":
        return np.sqrt(e2) * dn
    if kind == "flap":
        var_t = st["sq_text"] / nt - (st["sum_text"] / nt) ** 2
        var_i = st["sq_img"] / ni - (st["sum_img"] / ni) ** 2
        return (0.5 * var_t + 0.5 * var_i) * dn ** 2
    raise ValueError(kind)


def perms(sc):
    """Neuron order per layer, most important first."""
    return {i: np.argsort(-sc[i], kind="stable").astype(np.int32) for i in range(sc.shape[0])}


def save_perms(p, path):
    np.savez(path, **{f"layer_{i}": v for i, v in p.items()})


def uniform_widths(frac):
    w = int(round(WIDTH * frac / GROUP)) * GROUP
    return [max(GROUP, w)] * N_LAYERS


def equal_loss_widths(sc, frac, iters=60):
    """Per-layer widths so every layer keeps the same share of its score energy,
    with the total width equal to frac * full width (rounded to groups of 64)."""
    sorted_e = -np.sort(-(sc ** 2), axis=1)
    cum = np.cumsum(sorted_e, axis=1) / sorted_e.sum(axis=1, keepdims=True)
    target_total = frac * WIDTH * N_LAYERS
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        keep = (lo + hi) / 2
        widths = [(int(np.searchsorted(cum[i], keep)) + GROUP) // GROUP * GROUP for i in range(N_LAYERS)]
        if sum(widths) > target_total:
            hi = keep
        else:
            lo = keep
    return [min(WIDTH, max(GROUP, w)) for w in widths]


def mean_comp(st, perm, widths, down_weights):
    """Bias compensation vectors: the mean output of the removed neurons, per layer.

    down_weights: {layer: np.ndarray (hidden, inter)} in ORIGINAL neuron order.
    Returns {layer: mx.array (hidden,)} for layers with width < full.
    """
    e1, _, _, _ = _moments(st)
    out = {}
    for i, w in enumerate(widths):
        if w >= WIDTH:
            continue
        removed = perm[i][w:]
        out[i] = mx.array((down_weights[i][:, removed] @ e1[i][removed]).astype(np.float32)).astype(mx.bfloat16)
    return out


def describe(widths, skip=()):
    total = sum(w for i, w in enumerate(widths) if i not in set(skip))
    return {"mlp_frac": total / (WIDTH * N_LAYERS), "skip": sorted(skip)}


def write_levels(levels, path):
    json.dump(levels, open(path, "w"), indent=1)
