"""M1: which rule merges a group of SwiGLU neurons best, with only cheap fitting?

One group at a time. Target = the group's own contribution to the MLP output.
Rules H0..H6 are in the plan (~/.claude/plans/okay-i-have-sort-velvet-castle.md).

Run: python3 research/clean/m1.py [--smoke] [--groups N] [--layers 3,15,28]
Writes results/clean-m1.json. Imports tree_lib read-only.
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
from tree_lib import (load_model, token_blocks, Weights, Capture,  # noqa: E402
                      canonical_sign, neuron_features, silu)

RIDGE = 1e-4
RULES = ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]
CONFIGS = [(2, 1), (4, 1), (10, 1), (10, 2), (10, 3)]   # (children, parents)
SOURCES = ["random", "wt_nn", "val_cos", "two_pole", "out_cos"]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsilu(p):
    s = sig(p)
    return s * (1.0 + p * (1.0 - s))


def apply(X, P):
    """Parents P = (G, U, D), each (r x 576). Returns the parents' summed output."""
    G, U, D = P
    return (silu(X @ G.T) * (X @ U.T)) @ D


def relerr(Y, Yh):
    return float(np.linalg.norm(Y - Yh) / (np.linalg.norm(Y) + 1e-12))


def gram(F, W=None):
    F32 = F.astype(np.float32)
    A = F32.T @ (F32 if W is None else F32 * W[:, None].astype(np.float32))
    return A.astype(np.float64)


def solve_ridge(A, B):
    n = A.shape[0]
    A = A + np.eye(n) * RIDGE * np.trace(A) / n
    return np.linalg.solve(A, B)


def fit_M(X, g, Y, r):
    """Gate g fixed. Fit M (576x576) with Y ~ silu(Xg) * (X M). Keep rank r. Returns parents."""
    a = silu(X @ g)
    F = a[:, None] * X
    A = gram(F)
    B = (F.astype(np.float32).T @ Y.astype(np.float32)).astype(np.float64)
    M = solve_ridge(A, B)
    Uu, S, Vt = np.linalg.svd(M, full_matrices=False)
    s = np.sqrt(S[:r])
    U = (Uu[:, :r] * s).T
    D = Vt[:r] * s[:, None]
    G = np.tile(g, (r, 1))
    return G, U, D


def fit_gate(X, q, w):
    """Weighted least squares: X g ~ q."""
    A = gram(X, w)
    B = (X.astype(np.float32).T @ (q * w).astype(np.float32)).astype(np.float64)
    return solve_ridge(A, B)


def gn_gate(X, Y, g, M, cal_err):
    """One Gauss-Newton step on the gate with damping and a decrease check. Returns (g, err) or None."""
    p = X @ g
    a = silu(p)
    S = X @ M
    res = Y - a[:, None] * S
    sp = dsilu(p)
    w = sp * sp * (S * S).sum(1)
    A = gram(X, w)
    b = (X.astype(np.float32).T @ (sp * (S * res).sum(1)).astype(np.float32)).astype(np.float64)
    n = A.shape[0]
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        d = np.linalg.solve(A + np.eye(n) * lam * np.trace(A) / n, b)
        g2 = g + d
        e = relerr(Y, silu(X @ g2)[:, None] * S)
        if e < cal_err:
            return g2, e
    return None


def local_fit(Xc, Yc, init, steps=300, lr=2e-3):
    """H6: gradient descent on the parents' own (g, u, d) to match the group output. Local loss only."""
    P = [mx.array(a.astype(np.float32)) for a in init]
    X, Y = mx.array(Xc.astype(np.float32)), mx.array(Yc.astype(np.float32))
    denom = (Y * Y).sum()

    def loss(G, U, D):
        h = X @ G.T
        R = Y - ((h * mx.sigmoid(h)) * (X @ U.T)) @ D
        return (R * R).sum() / denom

    vg = mx.value_and_grad(loss, argnums=(0, 1, 2))
    m = [mx.zeros_like(p) for p in P]
    v = [mx.zeros_like(p) for p in P]
    best, bestP = None, P
    for t in range(1, steps + 1):
        l, gr = vg(*P)
        mx.eval(l, *gr)
        lv = l.item()
        if best is None or lv < best:
            best, bestP = lv, list(P)
        for i in range(3):
            m[i] = 0.9 * m[i] + 0.1 * gr[i]
            v[i] = 0.999 * v[i] + 0.001 * gr[i] * gr[i]
            P[i] = P[i] - lr * (m[i] / (1 - 0.9 ** t)) / (mx.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        mx.eval(*P)
    return tuple(np.array(p, dtype=np.float64) for p in bestP)


def run_rules(Xc, Xh, Gc, Uc, Dc, r, rules=RULES):
    """Children arrays Gc,Uc (m x 576) and Dc (m x 576, output rows). Returns {rule: (parents, seconds)}."""
    m = Gc.shape[0]
    hc = silu(Xc @ Gc.T) * (Xc @ Uc.T)           # T x m child activations (calibration)
    Yc = hc @ Dc
    dn = np.linalg.norm(Dc, axis=1)
    mass = np.abs(hc) * dn[None, :]              # per token, per child, output size
    out = {}

    def timed(name, fn):
        t0 = time.time()
        P = fn()
        out[name] = (P, time.time() - t0)

    # H0: keep the r children with the biggest output, refit their output rows.
    def h0():
        idx = np.argsort(-np.linalg.norm(hc * dn[None, :], axis=0))[:r]
        H = hc[:, idx]
        D = solve_ridge(gram(H), (H.astype(np.float32).T @ Yc.astype(np.float32)).astype(np.float64))
        return Gc[idx], Uc[idx], D
    timed("H0", h0)

    # H1: average gate and value (value sign-aligned), sum of outputs, then refit the output.
    if r == 1:
        def h1():
            s = np.sign(hc.mean(0))
            s[s == 0] = 1.0
            g = Gc.mean(0, keepdims=True)
            u = (Uc * s[:, None]).mean(0, keepdims=True)
            h = silu(Xc @ g.T) * (Xc @ u.T)
            D = solve_ridge(gram(h), (h.astype(np.float32).T @ Yc.astype(np.float32)).astype(np.float64))
            return g, u, D
        timed("H1", h1)

    # H2: fixed gate (best of: biggest child's gate, mean gate), then rank-r value/output fit.
    def h2():
        big = np.argmax(np.linalg.norm(hc * dn[None, :], axis=0))
        best = None
        for g in (Gc[big], Gc.mean(0)):
            P = fit_M(Xc, g, Yc, r)
            e = relerr(Yc, apply(Xc, P))
            if best is None or e < best[0]:
                best = (e, P)
        return best[1]
    timed("H2", h2)

    pre = Xc @ Gc.T
    wsum = mass.sum(1)

    # H3: OR gate. The gate input should follow the MAX of the children's gate inputs.
    def h3():
        w = wsum / wsum.mean() + 0.1
        g = fit_gate(Xc, pre.max(1), w)
        return fit_M(Xc, g, Yc, r)
    timed("H3", h3)

    # H5: gate input follows the output-weighted mean of the children's gate inputs.
    def h5():
        q = (mass * pre).sum(1) / (wsum + 1e-9)
        g = fit_gate(Xc, q, wsum / wsum.mean() + 0.1)
        return fit_M(Xc, g, Yc, r)
    timed("H5", h5)

    # H4: from H3, alternate: Gauss-Newton gate step, then refit value/output. Up to 10 rounds.
    def h4():
        P = out["H3"][0]
        g = P[0][0]
        err = relerr(Yc, apply(Xc, P))
        for _ in range(10):
            M = P[1].T @ P[2]
            step = gn_gate(Xc, Yc, g, M, err)
            if step is None:
                break
            g, _ = step
            P2 = fit_M(Xc, g, Yc, r)
            e2 = relerr(Yc, apply(Xc, P2))
            if e2 >= err:
                break
            P, err = P2, e2
        return P
    timed("H4", h4)

    # H6: local gradient fit, started from the best closed-form parent on calibration text.
    def h6():
        cand = [out[k][0] for k in ("H2", "H3", "H4", "H5")]
        best = min(cand, key=lambda P: relerr(Yc, apply(Xc, P)))
        init = (best[0], best[1], best[2])
        if r > 1:      # break the symmetry between parents that share one gate
            rng = np.random.default_rng(0)
            init = (best[0] + 0.01 * rng.standard_normal(best[0].shape), best[1], best[2])
        return local_fit(Xc, Yc, init)
    timed("H6", h6)

    return {k: v for k, v in out.items() if k in rules}


def oracle_error(Xc, Xh, Gc, Uc, Dc, r):
    """Best possible error for ANY r output directions with free per-token weights.
    Directions come from calibration text, error is scored on held-out text."""
    Yc = (silu(Xc @ Gc.T) * (Xc @ Uc.T)) @ Dc
    w, V = np.linalg.eigh(gram(Yc))
    V = V[:, -r:]
    Y = (silu(Xh @ Gc.T) * (Xh @ Uc.T)) @ Dc
    return relerr(Y, (Y @ V) @ V.T)


def measure(Xh, th, Gc, Uc, Dc, P):
    """Held-out error, concept survival, breadth ratio, coverage for one parent set."""
    hh = silu(Xh @ Gc.T) * (Xh @ Uc.T)
    Y = hh @ Dc
    Yh = apply(Xh, P)
    err = relerr(Y, Yh)
    dn = np.linalg.norm(Dc, axis=1)
    surv = []
    for i in range(Gc.shape[0]):
        top = np.argsort(-np.abs(hh[:, i]) * dn[i])[:100]
        dh = Dc[i] / (dn[i] + 1e-12)
        a, b = Yh[top] @ dh, Y[top] @ dh
        surv.append(float((a * b).sum() / ((b * b).sum() + 1e-12)))
    ch = [set(th[np.argsort(-np.abs(hh[:, i]) * dn[i])[:300]].tolist()) for i in range(Gc.shape[0])]
    pm = np.linalg.norm(Yh, axis=1)
    pset = set(th[np.argsort(-pm)[:300]].tolist())
    union = set().union(*ch)
    return {"err": err, "survival": float(np.mean(surv)),
            "breadth_ratio": len(pset) / (np.mean([len(c) for c in ch]) + 1e-9),
            "coverage": len(pset & union) / (len(union) + 1e-9)}


def pick_group(src, rng, m, feats):
    """feats: dict of per-layer arrays. Returns m neuron indices."""
    for _ in range(50):
        idx = _pick_once(src, rng, m, feats)
        c = feats["contrib"][idx]
        if c.min() >= 0.25 * c.max():      # every child matters, so the cut is not trivially good
            return idx
    return idx


def _pick_once(src, rng, m, feats):
    seed = rng.choice(feats["active"])
    ok = feats["is_active"].copy()
    ok[seed] = False
    if src == "random":
        rest = rng.choice(np.where(ok)[0], m - 1, replace=False)
        return np.concatenate([[seed], rest])
    if src == "wt_nn":
        d = ((feats["wfeat"] - feats["wfeat"][seed]) ** 2).sum(1)
    elif src == "val_cos":
        d = -feats["ucos"][seed]
    elif src == "out_cos":     # children that write nearly the same direction (shared reader)
        d = -feats["dcos"][seed]
    else:   # two_pole: similar gates and opposite value-times-output
        d = -(feats["gcos"][seed] - feats["ucos"][seed] * feats["dcos"][seed])
    d = np.where(ok, d, np.inf)             # never the seed itself, never a dead neuron
    return np.concatenate([[seed], np.argsort(d)[:m - 1]])


def unit_cos(A):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    return A @ A.T


def self_test(Xc, Xh, G, U, D):
    rng = np.random.default_rng(1)
    i = int(rng.integers(G.shape[0]))
    Gc, Uc, Dc = G[[i]], U[[i]], D[:, [i]].T
    for name, (Gx, Ux, Dx) in {"one neuron": (Gc, Uc, Dc),
                               "two identical": (np.vstack([Gc, Gc]), np.vstack([Uc, Uc]), np.vstack([Dc, Dc]))}.items():
        res = run_rules(Xc, Xh, Gx, Ux, Dx, 1, rules=["H0", "H1", "H2", "H3", "H5"])
        Yh = (silu(Xh @ Gx.T) * (Xh @ Ux.T)) @ Dx
        print(f"self-test {name:14s}", {k: round(relerr(Yh, apply(Xh, v[0])), 4) for k, v in res.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--layers", default="3,15,28")
    ap.add_argument("--held", type=int, default=40)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    groups = 2 if args.smoke else args.groups
    t0 = time.time()
    rng = np.random.default_rng(0)

    model, tok = load_model()
    cal = token_blocks(tok, "train", 32)
    held = token_blocks(tok, "test", 100)[: args.held]
    W = Weights(model)

    def cap(blocks):
        caps = {L: Capture(model.model.layers[L].mlp) for L in layers}
        for L in layers:
            model.model.layers[L].mlp = caps[L]
        for i in range(0, blocks.shape[0], 4):
            mx.eval(model(blocks[i:i + 4, :-1]))
        for L in layers:
            model.model.layers[L].mlp = caps[L].inner
        return {L: np.concatenate(caps[L].x, 0).astype(np.float64) for L in layers}

    XC, XH = cap(cal), cap(held)
    th = np.array(held[:, :-1]).reshape(-1)
    print(f"captured {XC[layers[0]].shape} cal, {XH[layers[0]].shape} held [{time.time() - t0:.0f}s]", flush=True)

    res = {"layers": layers, "groups_per_cell": groups, "records": [], "bias_check": {}}
    for L in layers:
        Xc, Xh = XC[L], XH[L]
        G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
        top = np.argsort(-np.abs(mu / sd))[:3]
        res["bias_check"][L] = [{"coord": int(j), "mean_over_std": float(mu[j] / sd[j])} for j in top]
        print(f"layer {L} near-constant input coordinates (mean/std):", res["bias_check"][L], flush=True)
        if L == layers[0]:
            self_test(Xc, Xh, G, U, D)

        hc = silu(Xc @ G.T) * (Xc @ U.T)
        sign = canonical_sign(Xc, G, U)
        contrib = np.linalg.norm(hc, axis=0) * np.linalg.norm(D, axis=0)
        Us, Ds = U * sign[:, None], D * sign[None, :]
        feats = {"n": G.shape[0], "active": np.where(contrib >= np.median(contrib))[0],
                 "is_active": contrib >= np.median(contrib), "contrib": contrib,
                 "wfeat": neuron_features("weights", Xc, G, U, D, sign),
                 "gcos": unit_cos(G), "ucos": unit_cos(Us), "dcos": unit_cos(Ds.T)}
        for (m, r) in CONFIGS:
            for src in SOURCES:
                for gi in range(groups):
                    idx = pick_group(src, rng, m, feats)
                    Gc, Uc, Dc = G[idx], U[idx], D[:, idx].T
                    out = run_rules(Xc, Xh, Gc, Uc, Dc, r)
                    rec = {"layer": L, "m": m, "r": r, "src": src, "idx": idx.tolist(), "rules": {},
                           "oracle": oracle_error(Xc, Xh, Gc, Uc, Dc, r),
                           "out_cos": float(np.mean(feats["dcos"][np.ix_(idx, idx)][np.triu_indices(m, 1)]))}
                    for k, (P, secs) in out.items():
                        rec["rules"][k] = {**measure(Xh, th, Gc, Uc, Dc, P), "secs": secs}
                    res["records"].append(rec)
                e = [x["rules"]["H0"]["err"] for x in res["records"][-groups:]]
                print(f"L{L} m={m} r={r} {src:9s} H0 err {np.median(e):.3f}  "
                      + " ".join(f"{k}={np.median([x['rules'][k]['err'] for x in res['records'][-groups:]]):.3f}"
                                 for k in RULES if k in res["records"][-1]["rules"])
                      + f"  [{time.time() - t0:.0f}s]", flush=True)

    # Aggregate over layers: median of (rule error / H0 error), share of groups where the rule wins.
    agg = {}
    for (m, r) in CONFIGS:
        for src in SOURCES:
            rs = [x for x in res["records"] if x["m"] == m and x["r"] == r and x["src"] == src]
            row = {}
            for k in RULES:
                v = [x["rules"][k] for x in rs if k in x["rules"]]
                if not v:
                    continue
                ratio = np.array([a["err"] / (x["rules"]["H0"]["err"] + 1e-12)
                                  for a, x in zip(v, [y for y in rs if k in y["rules"]])])
                boot = [np.median(ratio[rng.integers(0, len(ratio), len(ratio))]) for _ in range(200)]
                row[k] = {"median_ratio_to_H0": float(np.median(ratio)), "ratio_noise": float(np.std(boot)),
                          "win_share": float((ratio < 1).mean()),
                          "survival": float(np.mean([a["survival"] for a in v])),
                          "breadth_ratio": float(np.mean([a["breadth_ratio"] for a in v])),
                          "coverage": float(np.mean([a["coverage"] for a in v])),
                          "secs": float(np.mean([a["secs"] for a in v]))}
            row["oracle_err"] = float(np.median([x["oracle"] for x in rs]))
            row["H0_err"] = float(np.median([x["rules"]["H0"]["err"] for x in rs]))
            row["out_cos"] = float(np.mean([x["out_cos"] for x in rs]))
            agg[f"m{m}_r{r}_{src}"] = row
    res["aggregate"] = agg

    print("\nMEDIAN (rule error / H0 error), all layers. Below 1 = rule beats the cut.")
    print(f"{'group':22s}" + "".join(f"{k:>8s}" for k in RULES))
    for key, row in agg.items():
        print(f"{key:22s}" + "".join(f"{row[k]['median_ratio_to_H0']:8.2f}" if k in row else f"{'-':>8s}" for k in RULES))
    print("\nABSOLUTE ERROR: cut (H0) vs best possible rank-r bound (oracle), and children's mean output cosine")
    for key, row in agg.items():
        print(f"{key:22s} H0 {row['H0_err']:.3f}  oracle {row['oracle_err']:.3f}  H6 "
              f"{row['H6']['median_ratio_to_H0'] * row['H0_err']:.3f}  out_cos {row['out_cos']:+.2f}")
    print("\nCONCEPT SURVIVAL (1 = every child's signal kept, 0 = gone)")
    for key, row in agg.items():
        print(f"{key:22s}" + "".join(f"{row[k]['survival']:8.2f}" if k in row else f"{'-':>8s}" for k in RULES))

    out_path = os.path.join(ROOT, "results", "clean-m1-smoke.json" if args.smoke else "clean-m1.json")
    json.dump(res, open(out_path, "w"), indent=1)
    print("wrote", out_path, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
