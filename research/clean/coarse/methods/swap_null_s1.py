"""Null for diverse_cut (family "baseline-swap-null"): plain cut, but the same number of neurons that diverse_cut
changes are replaced by random neurons from ranks K..4K (seed 1); LS refit."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recipe_v4  # noqa: E402
from contract import mlp_io, orig_weights, SwiGLULayer, ls_down, hidden  # noqa: E402

METHOD = {"name": "swap_null_s1", "family": "baseline-swap-null", "minutes": 3.0}


def build(model, tok, L, K, cal, log=print):
    X, _ = mlp_io(model, cal[:64], L)
    G, U, D = orig_weights(model, L)
    h = hidden(X, G, U)
    order = np.argsort(-(np.linalg.norm(h, axis=0) * np.linalg.norm(D, axis=0)))
    pool = order[:min(4 * K, G.shape[0])]
    keys = recipe_v4.build(model, tok, L, K, cal, log=lambda *a: None, keys_for=pool)
    taken, keep = set(), []
    for j in order:
        k = keys.get(int(j), -1)
        if k < 0 or k not in taken:
            keep.append(int(j)); taken.add(k) if k >= 0 else None
        if len(keep) == K:
            break
    n_diff = len(set(keep) - set(order[:K].tolist()))
    rng = np.random.default_rng(1)
    out = list(order[:K - n_diff]) + list(rng.choice(order[K:min(4 * K, G.shape[0])], n_diff, replace=False))
    sel = np.sort(np.array(out))
    log(f"  swap_null L{L} K{K}: swapped {n_diff}")
    return SwiGLULayer(G[sel], U[sel], ls_down(X, G[sel], U[sel], h @ D.T))
