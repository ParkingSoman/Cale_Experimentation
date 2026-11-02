"""Coverage test: is a trained small-layer neuron a COARSE version of several old neurons (the essay's claim)?

A coarse parent fires wherever any of its children fires, and it fires on little else.
  down:  for each old neuron pole (a child candidate), take the child's top-50 tokens. Where does the parent's
         activation rank on those tokens (percentile over all tokens)? Child = percentile >= 0.95.
  up:    the parent's top-100 tokens. Share that sit in the top 1% of at least one child. Children needed to cover
         80% of them (greedy). Distinct children = low overlap of their top-50 token sets.
Baselines: untrained random neurons, and old neurons treated as parents of the other old neurons.
First position of every block is excluded (attention-sink token).

  python3 research/clean/r1cover.py --layers 3,15,28 --K 10 --tags local_rand
Writes results/clean-r1cover.json and research/clean/r1read/cover_L*_K*_*.md (children lists for reading).
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
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, load_w, rand_init, wpath  # noqa: E402

TOP_C, TOP_P, CHILD_PCT, CHILD_TOP = 50, 100, 0.95, 0.99


def pct_rank(a):
    """Percentile of every token's value within the column (0..1)."""
    r = np.empty_like(a)
    idx = np.argsort(a, axis=0)
    np.put_along_axis(r, idx, np.arange(a.shape[0])[:, None] / (a.shape[0] - 1), axis=0)
    return r


def cover(parent, child_top, child_hot, child_sets):
    """parent: tokens (one pole, sign applied). Returns stats and the children list."""
    pr = pct_rank(parent[:, None])[:, 0]
    down = pr[child_top].mean(0)                         # per child pole: parent percentile on child's top tokens
    kids = np.where(down >= CHILD_PCT)[0]
    ptop = np.argsort(-parent)[:TOP_P]
    hot = child_hot[ptop][:, kids]                       # parent top tokens x children: is the token a child's top 1%?
    up = float(hot.any(1).mean()) if len(kids) else 0.0
    # Greedy set cover of 80% of the parent's top tokens.
    covered, need, order = np.zeros(TOP_P, bool), 0, []
    while covered.mean() < 0.8 and len(kids):
        gain = (hot & ~covered[:, None]).sum(0)
        if gain.max() == 0:
            need = None
            break
        j = int(np.argmax(gain))
        covered |= hot[:, j]
        order.append(int(kids[j]))
        need += 1
    jac = []
    for a in range(min(len(order), 6)):
        for b in range(a + 1, min(len(order), 6)):
            sa, sb = child_sets[order[a]], child_sets[order[b]]
            jac.append(len(sa & sb) / len(sa | sb))
    by_down = kids[np.argsort(-down[kids])][:8]
    return {"n_children": int(len(kids)), "up_cover": up, "children_for_80pct": need,
            "children": [int(c) for c in by_down], "children_down_pct": [float(down[c]) for c in by_down],
            "cover_order": order, "sibling_overlap": float(np.mean(jac)) if jac else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3,15,28")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tags", default="local_rand")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    model, tok = load_model()
    W = Weights(model)
    held = token_blocks(tok, "test", 100)[:40]
    ids = np.array(held[:, :-1]).reshape(-1)
    XH = capture(model, held, layers)
    valid = np.arange(len(ids)) % SEQ != 0
    ids = ids[valid]
    path = os.path.join(ROOT, "results", "clean-r1cover.json")
    out = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    for L in layers:
        X = XH[L][valid].astype(np.float64)
        G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        H = hidden(X, G, U)
        C = np.concatenate([H, -H], 1)                   # child poles: old neuron j (+) and j (-)
        child_top = np.argpartition(-C, TOP_C, axis=0)[:TOP_C]
        thr = np.quantile(C, CHILD_TOP, axis=0)
        child_hot = C >= thr[None, :]
        child_sets = [set(child_top[:, c].tolist()) for c in range(C.shape[1])]
        name = lambda c: f"O{c % H.shape[1]}{'+' if c < H.shape[1] else '-'}"  # noqa: E731

        def run(P):
            A = hidden(X, P[0], P[1])
            res = []
            for t in range(A.shape[1]):
                for pole, s in (("+", 1), ("-", -1)):
                    r = cover(s * A[:, t], child_top, child_hot, child_sets)
                    r.update(unit=int(t), pole=pole)
                    res.append(r)
            return res

        def summ(res):
            return {"median_children": float(np.median([r["n_children"] for r in res])),
                    "mean_up_cover": float(np.mean([r["up_cover"] for r in res])),
                    "share_with_2plus_children": float(np.mean([r["n_children"] >= 2 for r in res])),
                    "share_reaching_80pct": float(np.mean([r["children_for_80pct"] is not None and r["n_children"] > 0
                                                           for r in res])),
                    "mean_sibling_overlap": float(np.nanmean([r["sibling_overlap"] if r["sibling_overlap"] is not None
                                                              else np.nan for r in res]))}

        null_rand = run(rand_init(G, U, D, args.K, seed=3))
        rng = np.random.default_rng(0)
        pick = rng.choice(H.shape[1], 20, replace=False)
        null_old = []
        for j in pick:
            for pole, s in (("+", 1), ("-", -1)):
                r = cover(s * H[:, j], child_top, child_hot, child_sets)
                r["n_children"] -= 1                     # it always covers itself
                r.update(unit=int(j), pole=pole)
                null_old.append(r)
        for tag in args.tags.split(","):
            if not os.path.exists(wpath(L, args.K, tag)):
                continue
            res = run(load_w(L, args.K, tag))
            row = {"layer": L, "K": args.K, "tag": tag, "trained": summ(res), "null_random": summ(null_rand),
                   "null_old_neurons": summ(null_old), "units": res,
                   "null_random_units": null_rand, "null_old_units": null_old}
            out["rows"] = [r for r in out["rows"] if not (r["layer"] == L and r["K"] == args.K and r["tag"] == tag)]
            out["rows"].append(row)
            for k in ("trained", "null_random", "null_old_neurons"):
                s = row[k]
                print(f"L{L:2d} K={args.K} {tag:10s} {k:16s} children median {s['median_children']:5.1f} | 2+ children "
                      f"{s['share_with_2plus_children']:.2f} | up-cover {s['mean_up_cover']:.2f} | children reach 80% "
                      f"{s['share_reaching_80pct']:.2f} | sibling overlap {s['mean_sibling_overlap']:.2f}", flush=True)
            md = [f"# Coverage children, layer {L}, K={args.K}, {tag}\n"]
            for r in res:
                if r["n_children"] >= 2:
                    md.append(f"- T{r['unit']}{r['pole']}: {r['n_children']} children, up-cover {r['up_cover']:.2f}, "
                              f"cover order {[name(c) for c in r['cover_order']]}, overlap {r['sibling_overlap']}")
            open(os.path.join(HERE, "r1read", f"cover_L{L}_K{args.K}_{tag}.md"), "w").write("\n".join(md) + "\n")
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
