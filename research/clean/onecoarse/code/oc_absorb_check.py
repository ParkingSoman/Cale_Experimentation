# scale: neurons=A:block0-all, B:block0-all; net=A, C-ladder, L-ladder, R4C-ladder, B128aug_s0, B128aug_s1; photos=saved, broden-held1; where=mac
# rung: Q=plan "Isolate where the spread comes from" (approved 2026-11-25), checks after the advisor review of VECMATH 12; EST=1-2 min Mac CPU; PERMIT=Shonu approved plan 2026-11-25
"""Checks on VECMATH 12 (oc_absorb.py), all from saved weights and the saved Broden stage-3 maps.

1. Power on real known answers: phi for the named take-over pairs (VECMATH 8.5 / 7) in C, L, R4C, next to jobtest_L.
   Planted partial widening: kept gate j OR a random 25% of cut neuron d's on-cells. How much does phi(d, j) move?
2. Strength, not on/off at zero: best match of each B neuron to any A neuron by (a) correlation of ReLU outputs per
   cell, (b) phi of top-10% cells. Same for L, R4C (as references).
3. Scope: d-side best phi at cuts 11, 64, 192 (C, L, R4C), not only 384.
4. N11's own wheel statistic on these photos: share of B0 n122's top-200 photos inside the union of the 8 wheel
   neurons' top-200s, against random 8-sets of A neurons.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oc_gate import LAD, VR, WA, patches, reads  # noqa: E402
from oc_absorb import R18, WHEEL, OUT  # noqa: E402

N_PH = 1000


def pre(X3, R, sh, n, bs=100):
    out = np.empty((n * 49, R.shape[0]), np.float32)
    for a in range(0, n, bs):
        b = min(n, a + bs)
        out[a * 49:b * 49] = patches(np.asarray(X3[a:b], np.float32)) @ R.T + sh
    return out


def phi_mat(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    N = a.shape[0]
    pa, pb = a.mean(0), b.mean(0)
    inter = a.T @ b / N
    return (inter - np.outer(pa, pb)) / (np.sqrt(np.outer(pa * (1 - pa), pb * (1 - pb))) + 1e-12)


def corr_mat(a, b):
    a = (a - a.mean(0)) / (a.std(0) + 1e-9)
    b = (b - b.mean(0)) / (b.std(0) + 1e-9)
    return a.T @ b / a.shape[0]


def top10(z):
    return z >= np.percentile(z, 90, axis=0)


def bread(s):
    w = np.load(os.path.join(R18, f"w_B128aug_s{s}.npz"))
    sc = w["bn1_0"].astype(np.float64)
    k = sc[0] / np.sqrt(sc[3] + 1e-5)
    return (w["W1_0"].reshape(128, -1) * k[:, None]).astype(np.float32), (sc[1] - sc[2] * k).astype(np.float32)


def main():
    X3 = np.load(os.path.join(VR, "held_s3.npy"), mmap_mode="r")
    RA, shA = reads(np.load(WA))
    zA = pre(X3, RA, shA, N_PH)
    onA = zA > 0
    out = {"n_photos": N_PH}
    rng = np.random.default_rng(0)

    # 1 + 3: per cut
    named = {64: [(339, 476)], 384: [(430, 501), (76, 374), (213, 501)], 11: [(5, 464)], 192: [(430, 114)]}
    for cut in (11, 64, 192, 384):
        jt = np.load(os.path.join(LAD, f"n{cut}", "jobtest_L.npz"))
        D, K = jt["D"], jt["K"]
        res = {}
        for b in ("C", "L", "R4C"):
            R, sh = reads(np.load(os.path.join(LAD, f"n{cut}", f"w_{b}.npz")))
            on = pre(X3, R[K], sh[K], N_PH) > 0
            ph = phi_mat(onA[:, D], on)                       # (|D|, |K|)
            best = ph.max(1)
            r = {"d_best_pct10_50_90_99": np.round(np.percentile(best, [10, 50, 90, 99]), 3).tolist(),
                 "n_ge_.4": int((best >= .4).sum()), "n_D": int(len(D))}
            for j, d in named.get(cut, []):
                ij, idd = int(np.nonzero(K == j)[0][0]), int(np.nonzero(D == d)[0][0])
                r[f"phi_n{d}_n{j}"] = round(float(ph[idd, ij]), 3)
                r[f"rank_of_n{j}_for_n{d}"] = int((ph[idd] > ph[idd, ij]).sum()) + 1
            res[b] = r
            del on
        for j, d in named.get(cut, []):
            ij = int(np.nonzero(K == j)[0][0])
            res[f"jobtest_L_n{j}"] = {"take": float(jt["take"][ij]), "best_d": int(D[jt["best_d"][ij]]),
                                     "best_corr": float(jt["best_corr"][ij]), "self": float(jt["self"][ij])}
        # planted partial widening at this cut: C gate of a kept j OR a random share of a cut d's cells
        Ri, shi = reads(np.load(os.path.join(LAD, f"n{cut}", "w_C.npz")))
        onC = onA[:, K]
        pw = []
        for _ in range(100):
            ij, idd = rng.integers(len(K)), rng.integers(len(D))
            base = phi_mat(onA[:, [D[idd]]], onC[:, [ij]])[0, 0]
            for share in (.1, .25, .5):
                cells = np.nonzero(onA[:, D[idd]])[0]
                add = rng.choice(cells, int(share * len(cells)), replace=False)
                g = onC[:, ij].copy()
                g[add] = True
                pw.append((share, base, phi_mat(onA[:, [D[idd]]], g[:, None])[0, 0]))
        pw = np.array(pw)
        res["planted_widening"] = {f"share_{s}": {"phi_before_median": round(float(np.median(pw[pw[:, 0] == s, 1])), 3),
                                                  "phi_after_median": round(float(np.median(pw[pw[:, 0] == s, 2])), 3)}
                                   for s in (.1, .25, .5)}
        out[f"cut{cut}"] = res
        print(f"cut {cut}:", json.dumps(res), flush=True)

    # 2: strength-based match of each small neuron to its best A neuron
    jt = np.load(os.path.join(LAD, "n384", "jobtest_L.npz"))
    K = jt["K"]
    fA = np.maximum(zA, 0)
    tA = top10(zA)
    st = {}
    for b in ("L", "R4C", "B0", "B1"):
        if b.startswith("B"):
            R, sh = bread(int(b[1]))
        else:
            Rf, shf = reads(np.load(os.path.join(LAD, "n384", f"w_{b}.npz")))
            R, sh = Rf[K], shf[K]
        z = pre(X3, R, sh, N_PH)
        c = corr_mat(fA, np.maximum(z, 0))                    # (512, J)
        t = phi_mat(tA, top10(z))
        st[b] = {"relu_corr_best_A_median": round(float(np.median(c.max(0))), 3),
                 "relu_corr_best_A_pct90": round(float(np.percentile(c.max(0), 90)), 3),
                 "top10_phi_best_A_median": round(float(np.median(t.max(0))), 3),
                 "top10_phi_best_A_pct90": round(float(np.percentile(t.max(0), 90)), 3),
                 "n_j_with_2plus_A_at_relu_corr_.4": int(((c >= .4).sum(0) >= 2).sum())}
        print(b, st[b], flush=True)
        del z
    out["strength"] = st

    # 4: N11's union statistic on these photos (per-photo peak, top 200 of N_PH... use all 2,000 photos)
    n2 = 2000
    pk = lambda R, sh: np.stack([(patches(np.asarray(X3[a:a + 100], np.float32)) @ R.T + sh).reshape(-1, 49, R.shape[0]).max(1)
                                 for a in range(0, n2, 100)]).reshape(n2, -1)
    PA = pk(RA, shA)
    RB, shB = bread(0)
    PB = pk(RB, shB)
    top = lambda v, k=200: set(np.argsort(-v)[:k].tolist())
    t122 = top(PB[:, 122])
    uni = set().union(*[top(PA[:, w]) for w in WHEEL])
    inside = len(t122 & uni) / 200
    rnd = []
    for _ in range(200):
        g = rng.choice(512, 8, replace=False)
        u = set().union(*[top(PA[:, w]) for w in g])
        rnd.append((len(t122 & u) / 200, len(u) / n2))
    rnd = np.array(rnd)
    best_single = max(len(t122 & top(PA[:, a])) / 200 for a in range(512))
    out["wheel_union"] = {"inside_wheel_union": inside, "wheel_union_size": len(uni) / n2,
                          "random8_inside_median": float(np.median(rnd[:, 0])), "random8_inside_95": float(np.percentile(rnd[:, 0], 95)),
                          "random8_union_size_median": float(np.median(rnd[:, 1])), "best_single_A": best_single,
                          "note": "top 200 of 2,000 Broden photos = 10%, N11 used top 200 of 19,000 ImageNet photos = 1%"}
    print("wheel union:", out["wheel_union"], flush=True)
    json.dump(out, open(os.path.join(OUT, "checks.json"), "w"))


if __name__ == "__main__":
    main()
