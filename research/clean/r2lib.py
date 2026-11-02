"""Shared code for the HOW search (rounds R2.x): verified families, parent rules, scoring.

A family = one trained small-layer unit (its best pole) that passed the coverage test, plus its children (old neuron
poles, hub children removed). A rule builds ONE parent neuron (g, u, d) from the children and the old model's data,
never from the trained parent. Scores compare the rule's parent with (a) the children (does it fire where they fire,
does it reproduce their summed output) and (b) the trained parent (one valid answer, not the only one).
"""
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ, silu  # noqa: E402
from r1 import capture, load_w  # noqa: E402

HUB, N_KIDS = 0.20, 6


class Setup:
    """Model data for one layer: calibration and held-out MLP inputs (first position dropped), old and trained weights."""

    def __init__(self, L, K=48, tag="global", n_cal=64, n_held=40, families_from="clean-r1cover.json"):
        t0 = time.time()
        self.L, self.K, self.tag = L, K, tag
        self.model, self.tok = load_model()
        W = Weights(self.model)
        self.W = W
        cal = token_blocks(self.tok, "train", n_cal)
        held = token_blocks(self.tok, "test", 100)[:n_held]
        self.cal_blocks, self.held_blocks = cal, held
        XC, XH = capture(self.model, cal, [L])[L], capture(self.model, held, [L])[L]
        self.keep_c = np.arange(XC.shape[0]) % SEQ != 0
        self.keep_h = np.arange(XH.shape[0]) % SEQ != 0
        self.Xc, self.Xh = XC[self.keep_c].astype(np.float64), XH[self.keep_h].astype(np.float64)
        self.G, self.U, self.D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        self.n_old = self.G.shape[0]
        self.P = load_w(L, K, tag)
        self.Hc = hidden(self.Xc, self.G, self.U)
        self.Hh = hidden(self.Xh, self.G, self.U)
        self.families = self._families(families_from)
        print(f"Setup L{L} K={K} {tag}: {len(self.families)} families [{time.time() - t0:.0f}s]", flush=True)

    def _families(self, fname):
        row = next(r for r in json.load(open(os.path.join(ROOT, "results", fname)))["rows"]
                   if r["layer"] == self.L and r["K"] == self.K and r["tag"] == self.tag)
        cnt = collections.Counter(c for u in row["units"] for c in u["children"])
        hubs = {c for c, n in cnt.items() if n / len(row["units"]) > HUB}
        best = {}
        for u in row["units"]:
            kids = [c for c in u["children"] if c not in hubs]
            if len(kids) >= 2 and u["up_cover"] >= 0.5:
                if u["unit"] not in best or len(kids) > len(best[u["unit"]]["kids"]):
                    best[u["unit"]] = {"unit": int(u["unit"]), "pole": u["pole"], "kids": kids[:N_KIDS]}
        return [self.family(f) for f in sorted(best.values(), key=lambda f: f["unit"])]

    def family(self, f):
        """Children as neurons whose active direction is positive (flip value and output together for a '-' pole)."""
        n = self.n_old
        idx = np.array([c % n for c in f["kids"]])
        sg = np.array([1.0 if c < n else -1.0 for c in f["kids"]])
        s_T = 1.0 if f["pole"] == "+" else -1.0
        t = f["unit"]
        fam = dict(f)
        fam.update(idx=idx, sg=sg, s_T=s_T,
                   Gc=self.G[idx], Uc=self.U[idx] * sg[:, None], Dc=(self.D[:, idx] * sg[None, :]).T,
                   hc_cal=self.Hc[:, idx] * sg[None, :], hc_held=self.Hh[:, idx] * sg[None, :],
                   aT_held=s_T * hidden(self.Xh, self.P[0][[t]], self.P[1][[t]])[:, 0],
                   aT_cal=s_T * hidden(self.Xc, self.P[0][[t]], self.P[1][[t]])[:, 0],
                   gT=self.P[0][t], uT=s_T * self.P[1][t], dT=s_T * self.P[2][:, t])
        fam["Y_cal"] = fam["hc_cal"] @ fam["Dc"]            # the children's summed output (calibration)
        fam["Y_held"] = fam["hc_held"] @ fam["Dc"]
        return fam


def relerr(Y, Yh):
    return float(np.linalg.norm(Y - Yh) / (np.linalg.norm(Y) + 1e-12))


def score(S, fam, g, u, d):
    """g, u: (576,), d: (576,). Returns scores of one parent against the children and the trained parent."""
    a = silu(S.Xh @ g) * (S.Xh @ u)
    if np.corrcoef(a, fam["aT_held"])[0, 1] < 0:          # a single neuron's pole is free: use the matching pole
        a, d = -a, -d
    ref = np.sort(a)
    kid_tops = [np.argsort(-fam["hc_held"][:, i])[:50] for i in range(len(fam["idx"]))]
    topP = set(np.argsort(-a)[:100].tolist())
    topT = set(np.argsort(-fam["aT_held"])[:100].tolist())
    Yp = np.outer(a, d)
    return {"act_corr": float(np.corrcoef(a, fam["aT_held"])[0, 1]),
            "top_jaccard": len(topP & topT) / len(topP | topT),
            "child_cover": float(np.mean([(np.searchsorted(ref, a[k]) / len(ref)).mean() for k in kid_tops])),
            "kids_out_err": relerr(fam["Y_held"], Yp),
            "out_cos": float(d @ fam["dT"] / (np.linalg.norm(d) * np.linalg.norm(fam["dT"]) + 1e-12))}


def fit_value_out(X, g, Y, ridge=1e-4):
    """Gate fixed: fit u, d (rank 1) so that silu(Xg) * (X u) d^T ~ Y. Closed form (M1's fit_M)."""
    a = silu(X @ g)
    F = (a[:, None] * X).astype(np.float32)
    A = (F.T @ F).astype(np.float64)
    A += np.eye(A.shape[0]) * ridge * np.trace(A) / A.shape[0]
    B = (F.T @ Y.astype(np.float32)).astype(np.float64)
    M = np.linalg.solve(A, B)
    Uu, Sv, Vt = np.linalg.svd(M, full_matrices=False)
    return Uu[:, 0] * np.sqrt(Sv[0]), Vt[0] * np.sqrt(Sv[0])


def scale_gate(X, g_dir, Y, scales=(0.25, 0.5, 1, 2, 4, 8)):
    """Pick the gate length (it sets how sharp silu is) by the children-output error on calibration text."""
    base = np.std(X @ g_dir) + 1e-9
    best = None
    for s in scales:
        g = g_dir * s / base
        u, d = fit_value_out(X, g, Y)
        e = relerr(Y, np.outer(silu(X @ g) * (X @ u), d))
        if best is None or e < best[0]:
            best = (e, g, u, d)
    return best[1], best[2], best[3]


def union_labels(fam, q=0.99):
    """Calibration tokens where at least one child is in its own top 1%."""
    h = fam["hc_cal"]
    thr = np.quantile(h, q, axis=0)
    return (h >= thr[None, :]).any(1)


def summarize(rows, names):
    out = {}
    for k in names:
        v = [r[k] for r in rows if k in r]
        out[k] = {m: float(np.median([x[m] for x in v])) for m in v[0]}
    return out


def print_summary(summ, title):
    print(f"\n{title}")
    print(f"{'rule':14s} {'act_corr':>9s} {'top_jacc':>9s} {'child_cov':>10s} {'kids_err':>9s} {'out_cos':>8s}")
    for k, s in summ.items():
        print(f"{k:14s} {s['act_corr']:9.2f} {s['top_jaccard']:9.2f} {s['child_cover']:10.2f} "
              f"{s['kids_out_err']:9.2f} {s['out_cos']:8.2f}")
