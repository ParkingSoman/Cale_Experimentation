# scale: neurons=s3:69, B:n101, A:children; net=A, B128aug_s1; photos=saved; where=mac
"""Step 2b of "Back to one coarse neuron" (2026-11-23): the same split as Step 2, at n101's own cell. Description.

Data: results/oc_firesplit_pos.npz (546 strong firings, peak >= half of n101's max) and _neg.npz (545 photos with
n101's lowest minimum). split[:, c] = channel c's exact vote on n101's input at the cell; shift = n101's bn1 shift.
A_pre9 = A's inputs on the 3 x 3 around the cell, so T (children's joint amount there) = sum_i a_i relu(max_9 A_pre9[i]).
Output: printed.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
P, K = 69, 11


def corr(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def main():
    cw = np.load(os.path.join(RES, "oc_children_w.npz"))
    kids = cw["order"][:K]
    a = cw["score"][kids]
    w = np.load(os.path.join(RES, "oc_firesplit_w.npz"))
    sh = float(w["sh"])
    poles = {k: np.load(os.path.join(RES, f"oc_firesplit_{k}.npz")) for k in ("pos", "neg")}
    out = {}
    for k, d in poles.items():
        s = d["split"].astype(np.float64)
        pre = s.sum(1) + sh
        err = np.abs(pre - d["pre"]).max()
        par, rest = s[:, P], s.sum(1) - s[:, P]
        T = (np.maximum(d["A_pre9"][:, kids].max(2), 0) * a[None, :]).sum(1)
        out[k] = (s, pre, par, rest, T)
        pos_votes = np.maximum(s, 0).sum(1)
        rank69 = (s > s[:, [P]]).sum(1)
        print(f"[{k}] n={len(pre)} recon err {err:.1e}; n101 input median {np.median(pre):+.2f}; "
              f"ch69 vote median {np.median(par):+.3f}, share of positive votes median {np.median(par / pos_votes):.3f}; "
              f"ch69 rank among votes: median {np.median(rank69):.0f}, rank 0 on {np.mean(rank69 == 0):.2f}")
        print(f"      T-a corr(rest, parent) {corr(rest, par):+.3f}; T median {np.median(T):.2f}")
        top = np.argsort(-s.mean(0))[:10]
        bot = np.argsort(s.mean(0))[:6]
        print("      mean votes, largest up: " + ", ".join(f"s3:{c} {s[:, c].mean():+.3f}" for c in top))
        print("      mean votes, largest down: " + ", ".join(f"s3:{c} {s[:, c].mean():+.3f}" for c in bot))
        print(f"      sum of up votes median {np.median(pos_votes):.2f}, down votes {np.median(np.minimum(s, 0).sum(1)):.2f}, shift {sh:+.2f}")
        if k == "pos":
            alone = (par + sh > 0).mean()
            restalone = (rest + sh > 0).mean()
            print(f"      T-c parent vote + shift > 0 (parent alone turns n101 on): {alone:.2f}; "
                  f"rest + shift > 0 (rest alone turns it on): {restalone:.2f}; ch69 vote <= 0 on {(par <= 0).mean():.2f}")
    s = np.r_[out["pos"][0], out["neg"][0]]
    par = np.r_[out["pos"][2], out["neg"][2]]
    rest = np.r_[out["pos"][3], out["neg"][3]]
    T = np.r_[out["pos"][4], out["neg"][4]]
    pre = par + rest + sh
    c_par, c_full = corr(par, T), corr(par + rest, T)
    rng = np.random.default_rng(0)
    others = np.array([c for c in range(256) if c != P])
    # shuffle: keep each channel's votes but permute which photo they belong to (breaks the joint read)
    sg = []
    for _ in range(200):
        rs = np.zeros_like(rest)
        for c in others:
            rs += s[rng.permutation(len(s)), c]
        sg.append(corr(par + rs, T) - c_par)
    sg = np.array(sg)
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        return y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]
    print(f"[both poles, n={len(T)}] corr(parent, T) {c_par:+.3f}; corr(n101 input, T) {c_full:+.3f}; gain {c_full - c_par:+.3f}; "
          f"shuffled gain median {np.median(sg):+.3f}, 99th {np.percentile(sg, 99):+.3f}; "
          f"partial corr(rest, T | parent) {corr(resid(rest, par), resid(T, par)):+.3f}")
    for k in ("pos", "neg"):
        _, pr, pa, re, Tk = out[k]
        print(f"[{k} only] corr(parent, T) {corr(pa, Tk):+.3f}; corr(n101 input, T) {corr(pa + re, Tk):+.3f}; "
              f"partial corr(rest, T | parent) {corr(resid(re, pa), resid(Tk, pa)):+.3f}")


if __name__ == "__main__":
    main()
