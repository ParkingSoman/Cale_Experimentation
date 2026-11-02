"""Family-diverse cut (family "diverse-cut"): keep at most one neuron per upstream-judgment family first (keys from
recipe_v4's both-pole IDF recipes), then fill by cut score; least-squares output refit. Essay: a coarse model keeps one
representative per earlier grouping and drops sibling specialists first, because their judgment survives upstream."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recipe_v4  # noqa: E402
from contract import mlp_io, orig_weights, SwiGLULayer, ls_down, hidden  # noqa: E402

METHOD = {"name": "diverse_cut", "family": "diverse-cut", "minutes": 3.0}


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
    for j in order:
        if len(keep) == K:
            break
        if int(j) not in keep:
            keep.append(int(j))
    keep = np.sort(np.array(keep))
    n_diff = len(set(keep.tolist()) - set(order[:K].tolist()))
    log(f"  diverse_cut L{L} K{K}: {n_diff} neurons differ from the plain cut")
    return SwiGLULayer(G[keep], U[keep], ls_down(X, G[keep], U[keep], h @ D.T))
