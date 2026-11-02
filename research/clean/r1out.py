"""Output-side coverage: is a trained small-layer neuron a coarse version of old neurons by WHAT IT DOES?

Profile of a unit pole = steering effect: add (typical strong activation) x (its output column) to the residual
stream after layer L's MLP, at every position, and take the mean change in next-word log-probabilities over the
vocabulary. This is the essay's steering method, averaged over real text.
Then the same coverage test as r1cover.py, with vocabulary words in place of text positions:
  child = an old neuron pole whose top-50 promoted words get a parent percentile >= 0.95,
  up-cover = share of the parent's top-100 promoted words that are in some child's top 1%.
Baselines: untrained random neurons, and old neurons as parents of the other old neurons.

  python3 research/clean/r1out.py --layers 3,15,28 --K 10 --tags local_rand,global
Writes results/clean-r1out.json and research/clean/r1out_cache/L*_old.npy (old-neuron profiles, reused).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, load_w, rand_init, wpath  # noqa: E402
from r1cover import cover, TOP_C, CHILD_TOP  # noqa: E402

CACHE = os.path.join(HERE, "r1out_cache")
SEQ_S, BATCH = 256, 8


class Steer:
    """Wraps an MLP. Adds one steering vector per batch row to the MLP output (not at position 0)."""

    def __init__(self, inner):
        self.inner, self.delta = inner, None

    def __call__(self, x):
        y = self.inner(x)
        if self.delta is None:
            return y
        pos = (mx.arange(x.shape[1]) > 0).astype(y.dtype)[None, :, None]
        return y + self.delta[:, None, :] * pos


class Record:
    """Wraps a later MLP. Keeps the mean hidden activation over positions (not position 0), per batch row."""

    def __init__(self, inner):
        self.inner, self.mean = inner, None

    def __call__(self, x):
        g = self.inner.gate_proj(x)
        h = g * mx.sigmoid(g) * self.inner.up_proj(x)
        self.mean = mx.mean(h[:, 1:, :], axis=1)
        return self.inner.down_proj(h)


TARGET = {"name": "logits", "span": 4}


def readout(model, L, x, recs):
    """The profile items: next-word log-probs ("logits") or the next layers' MLP neurons ("downstream")."""
    lg = model(x)
    if TARGET["name"] == "logits":
        return mx.mean(lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=1)
    if TARGET["name"] == "probs":
        # Change in probability, not log-probability: a flatter distribution cannot fake "promotes rare junk".
        return mx.mean(mx.softmax(lg.astype(mx.float32), axis=-1), axis=1)
    return mx.concatenate([r.mean for r in recs], axis=-1)


def profiles(model, L, blocks, vecs):
    """vecs: (n, 576) steering vectors. Returns (items, n): mean change of the readout when steering."""
    steer = Steer(model.model.layers[L].mlp)
    model.model.layers[L].mlp = steer
    layers = model.model.layers
    later = range(L + 1, min(len(layers), L + 1 + TARGET["span"]))
    recs = [Record(layers[l].mlp) for l in later]
    for l, r in zip(later, recs):
        layers[l].mlp = r
    out = []
    base = []
    for b in blocks:
        base.append(readout(model, L, mx.array(b[None, :]), recs)[0])
    base = mx.mean(mx.stack(base), 0)
    for i in range(0, len(vecs), BATCH):
        v = mx.array(vecs[i:i + BATCH].astype(np.float32))
        acc = 0
        for b in blocks:
            steer.delta = v
            acc = acc + readout(model, L, mx.array(np.repeat(b[None, :], v.shape[0], 0)), recs)
        d = acc / len(blocks) - base[None, :]
        mx.eval(d)
        out.append(np.array(d, dtype=np.float32))
    steer.delta = None
    model.model.layers[L].mlp = steer.inner
    for l, r in zip(later, recs):
        layers[l].mlp = r.inner
    return np.concatenate(out, 0).T


def pole_vectors(H, Dcols):
    """Steering vectors for both poles: (99th percentile of the pole's activation) x output column."""
    a_pos = np.quantile(np.maximum(H, 0), 0.99, axis=0)
    a_neg = np.quantile(np.maximum(-H, 0), 0.99, axis=0)
    return np.concatenate([(Dcols * a_pos[None, :]).T, -(Dcols * a_neg[None, :]).T], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3,15,28")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tags", default="local_rand,global")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--target", default="logits", choices=["logits", "probs", "downstream"])
    ap.add_argument("--span", type=int, default=4)
    ap.add_argument("--z", action="store_true", help="scale each item over old units (removes hub items)")
    args = ap.parse_args()
    TARGET.update(name=args.target, span=args.span)
    suffix = {"logits": "", "probs": "-probs", "downstream": "-downstream"}[args.target]
    rsuffix = suffix + ("-z" if args.z and args.target == "logits" else "")
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    model, tok = load_model()
    W = Weights(model)
    held = token_blocks(tok, "test", 100)[:40]
    sb = [np.array(held[i, :SEQ_S]) for i in range(args.blocks)]
    XH = capture(model, held, layers)
    valid = np.arange(XH[layers[0]].shape[0]) % SEQ != 0
    path = os.path.join(ROOT, "results", f"clean-r1out{rsuffix}.json")
    out = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    for L in layers:
        X = XH[L][valid].astype(np.float64)
        G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        cp = os.path.join(CACHE, f"L{L}_old{suffix}.npy")
        if os.path.exists(cp):
            C = np.load(cp)
        else:
            C = profiles(model, L, sb, pole_vectors(hidden(X, G, U), D))     # vocab x 3072
            np.save(cp, C)
        print(f"L{L} old profiles {C.shape} [{time.time() - t0:.0f}s]", flush=True)
        # Downstream items: a few later neurons react to almost any push ("hubs") and fill every top list.
        # Scale each item by its spread over the old units, so a unit's profile shows what is special to it.
        zmu = zsd = None
        if args.target == "downstream" or args.z:
            zmu, zsd = C.mean(1, keepdims=True), C.std(1, keepdims=True) + 1e-9
            C = (C - zmu) / zsd
        child_top = np.argpartition(-C, TOP_C, axis=0)[:TOP_C]
        child_hot = C >= np.quantile(C, CHILD_TOP, axis=0)[None, :]
        child_sets = [set(child_top[:, c].tolist()) for c in range(C.shape[1])]

        def run(P, name):
            A = hidden(X, P[0], P[1])
            Pr = profiles(model, L, sb, pole_vectors(A, P[2]))              # vocab x 2K
            if zmu is not None:
                Pr = (Pr - zmu) / zsd
            K = P[0].shape[0]
            res = []
            for c in range(2 * K):
                r = cover(Pr[:, c], child_top, child_hot, child_sets)
                r.update(unit=int(c % K), pole="+" if c < K else "-")
                res.append(r)
            return res

        def summ(res):
            return {"median_children": float(np.median([r["n_children"] for r in res])),
                    "share_with_2plus_children": float(np.mean([r["n_children"] >= 2 for r in res])),
                    "mean_up_cover": float(np.mean([r["up_cover"] for r in res])),
                    "share_reaching_80pct": float(np.mean([r["children_for_80pct"] is not None and r["n_children"] > 0
                                                           for r in res]))}

        null_rand = run(rand_init(G, U, D, args.K, seed=3), "random")
        rng = np.random.default_rng(0)
        null_old = []
        n_old = C.shape[1] // 2
        for j in rng.choice(n_old, 20, replace=False):
            for pole, c in (("+", j), ("-", j + n_old)):
                r = cover(C[:, c], child_top, child_hot, child_sets)
                r["n_children"] -= 1                     # it always covers itself
                r.update(unit=int(j), pole=pole)
                null_old.append(r)
        for tag in args.tags.split(","):
            if not os.path.exists(wpath(L, args.K, tag)):
                continue
            res = run(load_w(L, args.K, tag), tag)
            row = {"layer": L, "K": args.K, "tag": tag, "trained": summ(res), "null_random": summ(null_rand),
                   "null_old_neurons": summ(null_old), "units": res,
                   "null_random_units": null_rand, "null_old_units": null_old}
            out["rows"] = [r for r in out["rows"] if not (r["layer"] == L and r["K"] == args.K and r["tag"] == tag)]
            out["rows"].append(row)
            for k in ("trained", "null_random", "null_old_neurons"):
                s = row[k]
                print(f"L{L:2d} K={args.K} {tag:10s} {k:16s} children median {s['median_children']:5.1f} | 2+ "
                      f"{s['share_with_2plus_children']:.2f} | up-cover {s['mean_up_cover']:.2f} | reach 80% "
                      f"{s['share_reaching_80pct']:.2f} [{time.time() - t0:.0f}s]", flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
