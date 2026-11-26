"""Analysis of the class-token files from read_cls.py (CPU only, no model). 2026-11-25.

Stream R^2: student class-token stream (192) from a source stream + constant, ridge, lambda picked on an inner split
of FIT, refit on all FIT, scored on SCORE (pooled over dims). Sources: teacher at the same depth, teacher at every
depth (cross-depth table), and the OTHER student (positive control: two same-shape networks on the same data).
Neuron match: best single source neuron by correlation on FIT, measured on SCORE (median over 768 student neurons).
Output: out/analyze_cls.json and a printed table.
"""
import json
import os

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def load(name):
    d = np.load(os.path.join(OUT, f"cls_{name}.npz"))
    return torch.from_numpy(d["stream"].astype(np.float64)), torch.from_numpy(d["fire"].astype(np.float64)), d["half"]


def ridge_r2(Xf, Yf, Xs, Ys, lam):
    mx, my = Xf.mean(0), Yf.mean(0)
    A, B = Xf - mx, Yf - my
    G = A.T @ A
    W = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(len(G), dtype=torch.float64), A.T @ B)
    E = (Ys - my) - (Xs - mx) @ W
    return float(1 - (E ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum())


def stream_r2(Xf, Yf, Xs, Ys):
    k = len(Xf) // 2
    lam = max(LAMS, key=lambda l: ridge_r2(Xf[:k], Yf[:k], Xf[k:], Yf[k:], l))
    return ridge_r2(Xf, Yf, Xs, Ys, lam), lam


def corr(A, B):
    A = (A - A.mean(0)) / A.std(0).clamp_min(1e-9)
    B = (B - B.mean(0)) / B.std(0).clamp_min(1e-9)
    return A.T @ B / (len(A) - 1)


def best_single(Sf, Tf, Ss, Ts):
    """median over student neurons of corr on SCORE with the source neuron picked on FIT."""
    pick = corr(Tf, Sf).argmax(0)
    cs = corr(Ts, Ss)
    return float(cs[pick, torch.arange(Sf.shape[1])].median())


def main():
    T, TF, half = load("teacher")
    K, KF, _ = load("kd")
    Bs, BF, _ = load("baseline")
    f, s = half == 0, half == 1
    res = {"photos": {"fit": int(f.sum()), "score": int(s.sum())}, "same_depth": {}, "cross_depth_stream_kd": [], "cross_depth_neuron_kd": []}
    print("point | KD<-teacher R2 | base<-teacher R2 | KD<-base R2 (control) | KD neuron<-teacher | base neuron<-teacher | KD neuron<-base (control)")
    for p in range(1, 13):  # p0: the class token has not attended yet, so it is the same for every photo
        r = {}
        r["kd_from_teacher"], _ = stream_r2(T[f, p], K[f, p], T[s, p], K[s, p])
        r["base_from_teacher"], _ = stream_r2(T[f, p], Bs[f, p], T[s, p], Bs[s, p])
        r["kd_from_base"], _ = stream_r2(Bs[f, p], K[f, p], Bs[s, p], K[s, p])
        if True:
            b = p - 1
            r["neuron_kd_from_teacher"] = best_single(KF[f, b], TF[f, b], KF[s, b], TF[s, b])
            r["neuron_base_from_teacher"] = best_single(BF[f, b], TF[f, b], BF[s, b], TF[s, b])
            r["neuron_kd_from_base"] = best_single(KF[f, b], BF[f, b], KF[s, b], BF[s, b])
        res["same_depth"][f"p{p}"] = r
        g = lambda k: f"{r[k]:.3f}" if k in r else "-"
        print(f"p{p} | {g('kd_from_teacher')} | {g('base_from_teacher')} | {g('kd_from_base')} | {g('neuron_kd_from_teacher')} | {g('neuron_base_from_teacher')} | {g('neuron_kd_from_base')}")
    print("\ncross-depth stream R2, KD point i (rows) from teacher point j (cols), best j marked *")
    for i in range(1, 13):
        row = [stream_r2(T[f, j], K[f, i], T[s, j], K[s, i])[0] for j in range(1, 13)]
        res["cross_depth_stream_kd"].append(row)
        jb = int(np.argmax(row))
        print(f"KD p{i:2d} (T p1..p12): " + " ".join(f"{v:.2f}{'*' if j == jb else ' '}" for j, v in enumerate(row)))
    print("\ncross-depth neuron match (median best single), KD block i (rows) from teacher block j (cols)")
    for i in range(12):
        row = [best_single(KF[f, i], TF[f, j], KF[s, i], TF[s, j]) for j in range(12)]
        res["cross_depth_neuron_kd"].append(row)
        jb = int(np.argmax(row))
        print(f"KD b{i + 1:2d}: " + " ".join(f"{v:.2f}{'*' if j == jb else ' '}" for j, v in enumerate(row)))
    json.dump(res, open(os.path.join(OUT, "analyze_cls.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
