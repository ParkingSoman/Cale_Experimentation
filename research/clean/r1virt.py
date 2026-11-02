"""Downstream axis by virtual weights: which later components READ a unit's output direction?

Profile of a unit pole = for each of the next `span` layers: gate and up reads of every MLP neuron
(G_l @ (norm_l * d), U_l @ (norm_l * d)) and the attention query/key/value reads (Wq, Wk, Wv @ (norm_l * d)).
This is the "virtual weight" between two components (circuits practice). It needs no forward pass, so nothing is
averaged away over positions. Items are scaled by their spread over the old units, so hub items cannot dominate.
Then the same coverage test as r1cover.py.

  python3 research/clean/r1virt.py --layers 3,15,28 --K 48
Writes results/clean-r1virt.json (same row format as clean-r1cover.json).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, Weights  # noqa: E402
from r1 import load_w, rand_init, wpath  # noqa: E402
from r1cover import cover, TOP_C, CHILD_TOP  # noqa: E402


def reader_matrix(model, L, span):
    """Rows = later read directions (already multiplied by the RMSNorm weight). A unit's profile = R @ d."""
    rows = []
    layers = model.model.layers
    for l in range(L + 1, min(len(layers), L + 1 + span)):
        ly = layers[l]
        wm = np.array(ly.post_attention_layernorm.weight, dtype=np.float64)
        wa = np.array(ly.input_layernorm.weight, dtype=np.float64)
        for m, w in ((ly.mlp.gate_proj, wm), (ly.mlp.up_proj, wm),
                     (ly.self_attn.q_proj, wa), (ly.self_attn.k_proj, wa), (ly.self_attn.v_proj, wa)):
            rows.append(np.array(m.weight, dtype=np.float64) * w[None, :])
    return np.concatenate(rows, 0)


MIX_K, MIX_R2, MIX_SHARE = 5, 0.5, 0.10       # fixed before looking


def sparse_mix(p, C, exclude=None):
    """Linear-axis test: fit the parent profile p as a POSITIVE mix of at most MIX_K old pole profiles (columns of C,
    greedy, refit with non-negative least squares). Pass = R2 >= MIX_R2 and 2+ children with >= MIX_SHARE of the mix."""
    from scipy.optimize import nnls
    sel, r2, w = [], 0.0, np.zeros(0)
    res = p.copy()
    for _ in range(MIX_K):
        c = C.T @ res
        if exclude is not None:
            c[exclude] = -np.inf
        c[sel] = -np.inf
        j = int(np.argmax(c))
        if c[j] <= 0:
            break
        sel.append(j)
        w, _ = nnls(C[:, sel], p)
        res = p - C[:, sel] @ w
        r2 = 1 - (res @ res) / (p @ p)
    contrib = w * np.linalg.norm(C[:, sel], axis=0) if sel else w
    share = contrib / (contrib.sum() + 1e-12)
    kids = [int(s) for s, sh in zip(sel, share) if sh >= MIX_SHARE]
    return {"mix_r2": float(r2), "mix_children": kids, "mix_pass": bool(r2 >= MIX_R2 and len(kids) >= 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3,15,28")
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--tags", default="local_rand,global")
    ap.add_argument("--span", type=int, default=4)
    ap.add_argument("--reader", default="virtual", choices=["virtual", "dla"])
    args = ap.parse_args()
    model, _ = load_model()
    W = Weights(model)
    path = os.path.join(ROOT, "results", "clean-r1virt.json" if args.reader == "virtual" else "clean-r1dla.json")
    out = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    for L in [int(x) for x in args.layers.split(",")]:
        if args.reader == "dla":
            # Direct logit attribution: the unit's output direction through the final norm and the tied unembedding.
            Rm = np.array(model.model.embed_tokens.weight, dtype=np.float64) * np.array(model.model.norm.weight, dtype=np.float64)[None, :]
        else:
            Rm = reader_matrix(model, L, args.span)
        G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        prof = Rm @ D                                        # items x 1536 (pole +); pole - is the negative
        C = np.concatenate([prof, -prof], 1)
        zsd = C.std(1, keepdims=True) + 1e-9                 # the mean over +/- poles is exactly 0
        C = C / zsd
        child_top = np.argpartition(-C, TOP_C, axis=0)[:TOP_C]
        child_hot = C >= np.quantile(C, CHILD_TOP, axis=0)[None, :]
        child_sets = [set(child_top[:, c].tolist()) for c in range(C.shape[1])]

        def run(Dc):
            p = (Rm @ Dc) / zsd
            res = []
            for t in range(Dc.shape[1]):
                for pole, s in (("+", 1), ("-", -1)):
                    r = cover(s * p[:, t], child_top, child_hot, child_sets)
                    r.update(unit=int(t), pole=pole, **sparse_mix(s * p[:, t], C))
                    res.append(r)
            return res

        null_rand = run(rand_init(G, U, D, args.K, seed=3)[2])
        rng = np.random.default_rng(0)
        null_old = []
        for j in rng.choice(D.shape[1], 20, replace=False):
            for pole, c in (("+", j), ("-", j + D.shape[1])):
                r = cover(C[:, c], child_top, child_hot, child_sets)
                r["n_children"] -= 1
                r.update(unit=int(j), pole=pole, **sparse_mix(C[:, c], C, exclude=[j, j + D.shape[1]]))
                null_old.append(r)
        for tag in args.tags.split(","):
            if not os.path.exists(wpath(L, args.K, tag)):
                continue
            res = run(load_w(L, args.K, tag)[2])
            row = {"layer": L, "K": args.K, "tag": tag, "items": int(Rm.shape[0]), "units": res,
                   "null_random_units": null_rand, "null_old_units": null_old}
            out["rows"] = [r for r in out["rows"] if not (r["layer"] == L and r["K"] == args.K and r["tag"] == tag)]
            out["rows"].append(row)
            for name, rr in (("trained", res), ("random", null_rand), ("old", null_old)):
                print(f"L{L:2d} K={args.K} {tag:10s} {name:8s} items {Rm.shape[0]:6d} | median children "
                      f"{np.median([r['n_children'] for r in rr]):4.1f} | 2+ {np.mean([r['n_children'] >= 2 for r in rr]):.2f}"
                      f" | up-cover {np.mean([r['up_cover'] for r in rr]):.2f} | mix pass "
                      f"{np.mean([r['mix_pass'] for r in rr]):.2f} mix R2 {np.median([r['mix_r2'] for r in rr]):.2f}",
                      flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
