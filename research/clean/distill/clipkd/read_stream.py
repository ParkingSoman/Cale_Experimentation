"""Step 2 of plan "read the CLIP-KD pair" (2026-11-25): is the KD student's stream a linear view of the teacher's?

Photos for reading only: reading photos, fit positions of each class (even positions, as in neuron/coarse/r18.py).
  FIT half   = the first fit photo of each class (1,000 photos)
  SCORE half = the second fit photo of each class (1,000 photos)
  These are not the test photos that score_clip.py scores.
Points: p0 = ln_pre output (the stream entering block 0), p1..p12 = the stream after block b. All 197 tokens.
Variants: raw stream, and per-token normalized (LayerNorm without its affine = what each block reads).
Per point, student s in (kd, baseline):
  R2      = Y (student, 192) from X (teacher, 768) + constant, ridge fit on FIT, scored on SCORE (pooled over dims).
  R2_shuf = the same with the student's photos rolled by one inside each batch (control, should be ~0).
  R2_pca  = Y from the teacher's top-192 principal directions only (FIT covariance): what a PCA cut keeps.
  pca_var = share of the teacher's stream variance in its top 192 directions.
Only the three image towers are kept in memory. Accumulators (Gram matrices) are saved, so later fits need no new pass.
Output: out/read_stream.json, out/read_stream_grams.npz.
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/read_stream.py [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_clip import MODELS, CLIP_MEAN, CLIP_STD, DEV, OUT, build, net  # noqa: E402

STUDENTS = ("kd", "baseline")
VARIANTS = ("raw", "norm")
NPT = 13


def photo_halves(ph, n_per_half):
    fit, score = [], []
    for c in range(1000):
        w = np.where(ph.lab == c)[0][0::2]
        fit.append(w[0]); score.append(w[1])
    return np.array(fit[:n_per_half]), np.array(score[:n_per_half])


def tower(name):
    arch, fname = MODELS[name]
    m = build(arch, fname)
    v = m.visual
    del m
    if DEV == "mps":
        torch.mps.empty_cache()
    return v.eval()


def streams(v, x):
    """Stream at the 13 points, each (B * 197, D), batch-first."""
    out = []
    hs = [v.ln_pre.register_forward_hook(lambda m, i, o: out.append(o))]
    hs += [b.register_forward_hook(lambda m, i, o: out.append(o)) for b in v.transformer.resblocks]
    with torch.no_grad():
        v(x)
    for h in hs:
        h.remove()
    res = []
    for o in out:
        if o.shape[0] == 197 and o.shape[1] == x.shape[0]:
            o = o.transpose(0, 1)
        assert o.shape[1] == 197, o.shape
        res.append(o.reshape(-1, o.shape[-1]).float())
    assert len(res) == NPT
    return res


class GX:
    """Teacher side, shared by both students: XtX of [X, 1] and n."""
    def __init__(self, dx):
        self.XtX, self.n = torch.zeros(dx + 1, dx + 1, dtype=torch.float64), 0

    def add(self, Xa):
        self.XtX += (Xa.T @ Xa).cpu().double(); self.n += len(Xa)


class Acc:
    """Student side: XtY, XtYs (rolled Y), sum Y, sum Y^2 (and the same for Ys)."""
    def __init__(self, gx, dx, dy):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)
        self.gx = gx
        self.XtY, self.XtYs = z(dx + 1, dy), z(dx + 1, dy)
        self.sy, self.syy, self.sys, self.syys = z(dy), z(dy), z(dy), z(dy)

    @property
    def XtX(self):
        return self.gx.XtX

    @property
    def n(self):
        return self.gx.n

    def add(self, Xa, Y, Ys):
        self.XtY += (Xa.T @ Y).cpu().double(); self.XtYs += (Xa.T @ Ys).cpu().double()
        self.sy += Y.sum(0).cpu().double(); self.syy += (Y * Y).sum(0).cpu().double()
        self.sys += Ys.sum(0).cpu().double(); self.syys += (Ys * Ys).sum(0).cpu().double()


def r2(fit, sc, shuf=False, P=None, lam=1e-4):
    XtX_f, XtY_f = fit.XtX, (fit.XtYs if shuf else fit.XtY)
    XtX_s, XtY_s = sc.XtX, (sc.XtYs if shuf else sc.XtY)
    sy, syy = (sc.sys, sc.syys) if shuf else (sc.sy, sc.syy)
    if P is not None:  # restrict X to P's columns (P includes the constant)
        XtX_f, XtY_f, XtX_s, XtY_s = P.T @ XtX_f @ P, P.T @ XtY_f, P.T @ XtX_s @ P, P.T @ XtY_s
    d = XtX_f.diagonal().mean()
    W = torch.linalg.solve(XtX_f + lam * d * torch.eye(len(XtX_f), dtype=torch.float64), XtY_f)
    ss_res = syy.sum() - 2 * (W * XtY_s).sum() + (W * (XtX_s @ W)).sum()
    ss_tot = (syy - sy ** 2 / sc.n).sum()
    return float(1 - ss_res / ss_tot)


def pca_P(fit, k):
    n = fit.n
    mu = fit.XtX[-1, :-1] / n
    C = fit.XtX[:-1, :-1] / n - torch.outer(mu, mu)
    ev, V = torch.linalg.eigh(C)
    ev, V = ev.flip(0), V.flip(1)
    D = len(C)
    P = torch.zeros(D + 1, k + 1, dtype=torch.float64)
    P[:-1, :k] = V[:, :k]
    P[-1, k] = 1.0
    return P, float(ev[:k].sum() / ev.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=25)
    a = ap.parse_args()
    t0 = time.time()
    ph = net.Photos("read")
    n = 50 if a.smoke else 1000
    halves = dict(zip(("fit", "score"), photo_halves(ph, n)))
    T = tower("teacher")
    S = {s: tower(s) for s in STUDENTS}
    print(f"towers loaded {time.time() - t0:.0f}s", flush=True)
    gx = {(h, p, v): GX(768) for h in halves for p in range(NPT) for v in VARIANTS}
    acc = {(h, p, s, v): Acc(gx[(h, p, v)], 768, 192) for h in halves for p in range(NPT) for s in STUDENTS for v in VARIANTS}
    for h, ids in halves.items():
        for i in range(0, len(ids), a.bs):
            raw = ph.raw(ids[i:i + a.bs])
            x = torch.from_numpy(raw.copy()).permute(0, 3, 1, 2).float() / 255.0
            x = ((x - CLIP_MEAN) / CLIP_STD).to(DEV)
            B = x.shape[0]
            XT = streams(T, x)
            XA = {}
            for p in range(NPT):
                for v in VARIANTS:
                    X = XT[p] if v == "raw" else F.layer_norm(XT[p], XT[p].shape[-1:])
                    XA[(p, v)] = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
                    gx[(h, p, v)].add(XA[(p, v)])
            for s in STUDENTS:
                YS = streams(S[s], x)
                for p in range(NPT):
                    for v in VARIANTS:
                        Y = YS[p] if v == "raw" else F.layer_norm(YS[p], YS[p].shape[-1:])
                        Ys = Y.reshape(B, 197, -1).roll(1, 0).reshape(B * 197, -1)
                        acc[(h, p, s, v)].add(XA[(p, v)], Y, Ys)
            print(f"{h} {i + B}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    res = {"photos": {h: int(len(ids)) for h, ids in halves.items()}, "tokens_per_photo": 197, "points": {}}
    for p in range(NPT):
        row = {}
        for v in VARIANTS:
            P, share = pca_P(acc[("fit", p, "kd", v)], 192)
            row[v] = {"teacher_top192_var_share": share}
            for s in STUDENTS:
                f, sc = acc[("fit", p, s, v)], acc[("score", p, s, v)]
                row[v][s] = {"R2": r2(f, sc), "R2_shuf": r2(f, sc, shuf=True), "R2_pca192": r2(f, sc, P=P)}
        res["points"][f"p{p}"] = row
        print(f"p{p}", json.dumps(row), flush=True)
    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if a.smoke else ""
    json.dump(res, open(os.path.join(OUT, f"read_stream{tag}.json"), "w"), indent=1)
    grams = {f"{h}_p{p}_{v}_XtX": gx[(h, p, v)].XtX.numpy() for h in halves for p in range(NPT) for v in VARIANTS}
    grams.update({f"{h}_p{p}_{s}_{v}_{k}": getattr(acc[(h, p, s, v)], k).numpy()
                  for h in halves for p in range(NPT) for s in STUDENTS for v in VARIANTS
                  for k in ("XtY", "XtYs", "sy", "syy", "sys", "syys")})
    grams.update({f"{h}_n": np.array(gx[(h, 0, "raw")].n) for h in halves})
    np.savez(os.path.join(OUT, f"read_stream_grams{tag}.npz"), **grams)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
