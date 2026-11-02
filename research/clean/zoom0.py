"""ZOOM-0: the smallest test of a zoom configuration (one layer, one dial, minutes per attempt).

Claim: ONE configuration gives a dial from K parents to the full layer. Every setting works without a refit,
quality improves steadily, it beats the cut at every size, and the full setting IS the original layer.

Configuration for layer L (default 28, so a loss check runs only the 2 layers after it on cached inputs):
  parents   K units from a short-tail fit (r2round5.fit_tail), started from the cut
  families  every original neuron belongs to exactly one parent: the one whose activation it matches best
  expand    remove a parent, put its original children back (original weights, no refit)
  order     families ranked by loss gained per added neuron (each expansion scored once, alone)
Pass criteria (fixed before the run):
  1 steady: loss never rises as the dial opens          2 beats a same-size cut (LS refit) at every test size
  3 full setting = original loss (to rounding)           4 within 25% of a layer fitted for that size (gap share)

  python3 research/clean/zoom0.py [--K 24] [--fit_secs 30] [--per_size_fits]
Writes results/clean-zoom0.json (one row per attempt).
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, SmallMLP  # noqa: E402
from r2round5 import cache_h, fit_tail, tail  # noqa: E402
from r2round3 import joint_out  # noqa: E402


class ZoomMLP:
    """Parents (trainable) + original neurons (fixed). mask_p switches parents on, mask_o switches originals on."""

    def __init__(self, Pp, G, U, D):
        import mlx.nn as nn

        class Parents(nn.Module):
            def __init__(self, P):
                super().__init__()
                self.g, self.u, self.d = (mx.array(P[0].astype(np.float32)), mx.array(P[1].astype(np.float32)),
                                          mx.array(P[2].astype(np.float32)))

        self.p = Parents(Pp)
        self.G, self.U, self.D = (mx.array(G.astype(np.float32)), mx.array(U.astype(np.float32)),
                                  mx.array(D.astype(np.float32)))
        self.mask_p = self.mask_o = None

    def __call__(self, x):
        gp = x @ self.p.g.T
        hp = gp * mx.sigmoid(gp) * (x @ self.p.u.T) * self.mask_p
        go = x @ self.G.T
        ho = go * mx.sigmoid(go) * (x @ self.U.T) * self.mask_o
        return hp @ self.p.d.T + ho @ self.D.T


def fit_nested(S, H, Pp, G, U, D, fam, K, secs, lr=3e-4, batch=4, seed=0):
    """Short-tail KL fit of the parents under RANDOM expansion states (the nesting constraint).
    Each step opens each family with probability q ~ U(0, 1): its children (original neurons) run, its parent is off."""
    import mlx.nn as nn
    import mlx.optimizers as optim
    rng = np.random.default_rng(seed)
    z = ZoomMLP(Pp, G, U, D)
    teacher = S.model.model.layers[S.L].mlp
    fam_mx = mx.array(fam)
    S.model.freeze()

    def loss_fn(params, h, tl, mp, mo):
        z.p.update(params)
        z.mask_p, z.mask_o = mp, mo
        lg = tail(S, h, z)
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        return (mx.exp(tl) * (tl - lg)).sum(-1)[:, 1:].mean()

    vg = mx.value_and_grad(loss_fn)
    opt = optim.Adam(learning_rate=lr)
    params = z.p.parameters()
    t0, steps = time.time(), 0
    while time.time() - t0 < secs:
        h = mx.array(H[rng.integers(0, H.shape[0], batch)])
        tl = tail(S, h, teacher)
        tl = mx.stop_gradient(tl - mx.logsumexp(tl, axis=-1, keepdims=True))
        q = rng.random()
        opened = rng.random(K) < q
        mp = mx.array((~opened).astype(np.float32))
        mo = mx.array(opened[fam].astype(np.float32))
        _, g = vg(params, h, tl, mp, mo)
        opt.update(params, g)
        mx.eval(params, opt.state)
        steps += 1
    z.p.update(params)
    S.model.unfreeze()
    return tuple(np.array(a, dtype=np.float64) for a in (z.p.g, z.p.u, z.p.d)), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--fit_secs", type=float, default=30)
    ap.add_argument("--n_cal", type=int, default=32)
    ap.add_argument("--n_held", type=int, default=20)
    ap.add_argument("--nested_secs", type=float, default=0, help="nested fit under random expansion states")
    ap.add_argument("--per_size_fits", action="store_true", help="criterion 4 (adds one short fit per test size)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    L, K = args.layer, args.K
    model, tok = load_model()
    W = Weights(model)
    S = types.SimpleNamespace(model=model, L=L)
    cal = token_blocks(tok, "train", args.n_cal)
    held = token_blocks(tok, "test", 100)[:args.n_held]
    Hcal, Hheld = cache_h(S, cal), cache_h(S, held)
    Xc = capture(model, cal, [L])[L]
    Xc = Xc[np.arange(len(Xc)) % SEQ != 0].astype(np.float64)
    G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
    n = G.shape[0]
    Hc = hidden(Xc, G, U)
    Yc = Hc @ D.T
    labels = np.array(held[:, 1:])
    Hh_mx = mx.array(Hheld)

    def ce(P):
        """Held-out CE with layer L's MLP = P, running only the layers after L on cached inputs."""
        mlp = SmallMLP(P)
        tot = 0.0
        for i in range(0, Hheld.shape[0], 4):
            lg = tail(S, Hh_mx[i:i + 4], mlp)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(lg, mx.array(labels[i:i + 4])[..., None], axis=-1).squeeze(-1)[:, 1:]
            tot += float(nll.sum().item())
        return tot / (labels.shape[0] * (labels.shape[1] - 1))

    base = ce((G, U, D))
    imp = np.abs(Hc).mean(0) * np.linalg.norm(D, axis=0)

    def cut(N):
        keep = np.sort(np.argsort(-imp)[:N])
        return G[keep], U[keep], joint_out(Xc, G[keep], U[keep], Yc)

    # 1. Parents: short-tail fit from the cut.
    P0 = cut(K)
    Pp, steps = fit_tail(S, Hcal, P0, args.fit_secs / 60)
    t_fit = time.time() - t0
    # 2. Families: each original neuron goes to the parent whose activation it matches best (|correlation|).
    A = hidden(Xc, Pp[0], Pp[1])
    Az = (A - A.mean(0)) / (A.std(0) + 1e-9)
    Hz = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-9)
    fam = np.argmax(np.abs(Hz.T @ Az) / len(Az), axis=1)          # n old neurons -> parent id
    sizes = np.bincount(fam, minlength=K)
    nsteps = 0
    if args.nested_secs > 0:
        Pp, nsteps = fit_nested(S, Hcal, Pp, G, U, D, fam, K, args.nested_secs)

    def layer(expanded):
        keep_p = [t for t in range(K) if t not in expanded]
        kids = np.where(np.isin(fam, list(expanded)))[0] if expanded else np.array([], int)
        Gl = np.concatenate([Pp[0][keep_p], G[kids]])
        Ul = np.concatenate([Pp[1][keep_p], U[kids]])
        Dl = np.concatenate([Pp[2][:, keep_p], D[:, kids]], axis=1)
        return Gl, Ul, Dl

    # 3. Order: score each single expansion once, rank by loss gained per added neuron.
    ce_parents = ce(layer(set()))
    gain = {}
    for t in range(K):
        if sizes[t] == 0:
            gain[t] = 0.0
            continue
        gain[t] = (ce_parents - ce(layer({t}))) / max(1, sizes[t] - 1)
    order = sorted(range(K), key=lambda t: -gain[t])
    # 4. The dial: open families in order, no refit.
    dial, expanded = [], set()
    dial.append({"neurons": K, "ce": ce_parents})
    for t in order:
        expanded.add(t)
        Pl = layer(expanded)
        dial.append({"neurons": int(Pl[0].shape[0]), "ce": ce(Pl)})
    full_ce = dial[-1]["ce"]
    # Criteria.
    steady = all(dial[i + 1]["ce"] <= dial[i]["ce"] + 1e-4 for i in range(len(dial) - 1))
    test_sizes = [s for s in (K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K) if s < n]

    def dial_at(N):
        """Loss of the dial at the largest setting with at most N neurons."""
        ok = [d for d in dial if d["neurons"] <= N]
        return ok[-1] if ok else dial[0]

    cmp = []
    for N in test_sizes:
        d = dial_at(N)
        c = ce(cut(d["neurons"]))
        row = {"N": N, "dial_neurons": d["neurons"], "dial": d["ce"] - base, "cut": c - base}
        if args.per_size_fits:
            Pf, _ = fit_tail(S, Hcal, cut(d["neurons"]), args.fit_secs / 60)
            row["fitted"] = ce(Pf) - base
            row["within_25pct"] = bool(row["dial"] <= row["fitted"] * 1.25 + 1e-4)
        cmp.append(row)
    res = {"note": args.note, "layer": L, "K": K, "fit_secs": args.fit_secs, "fit_steps": steps, "nested_secs": args.nested_secs, "nested_steps": nsteps,
           "base": base, "parents_only": ce_parents - base, "family_sizes": sizes.tolist(),
           "dial": [{"neurons": d["neurons"], "dce": d["ce"] - base} for d in dial],
           "compare": cmp, "criteria": {"1_steady": steady,
                                        "2_beats_cut": all(r["dial"] < r["cut"] for r in cmp),
                                        "3_full_equals_original": abs(full_ce - base) < 1e-3,
                                        "4_near_fitted": all(r.get("within_25pct", False) for r in cmp)
                                        if args.per_size_fits else None},
           "secs": time.time() - t0}
    path = os.path.join(ROOT, "results", "clean-zoom0.json")
    out = json.load(open(path)) if os.path.exists(path) else {"attempts": []}
    out["attempts"].append(res)
    json.dump(out, open(path, "w"), indent=1)
    print(f"base CE {base:.4f} | parents only (K={K}) +{ce_parents - base:.4f} | fit {steps} steps in {t_fit:.0f}s")
    print(f"family sizes: min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}")
    print("size   dial(no refit)   cut(LS refit)" + ("   fitted" if args.per_size_fits else ""))
    for r in cmp:
        print(f"{r['N']:5d}  +{r['dial']:.4f} ({r['dial_neurons']:4d})  +{r['cut']:.4f}"
              + (f"   +{r['fitted']:.4f}" if 'fitted' in r else ""))
    print(f"full setting: {dial[-1]['neurons']} neurons, +{full_ce - base:.5f}")
    print("criteria:", res["criteria"], f"[{res['secs']:.0f}s]")


if __name__ == "__main__":
    main()
