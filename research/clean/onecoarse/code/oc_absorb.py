# scale: neurons=A:block0-all, B:block0-all; net=A, C-ladder, L-ladder, R4C-ladder, B128aug_s0, B128aug_s1; photos=saved, broden-held1; where=mac
# rung: Q=plan "Isolate where the spread comes from" (approved 2026-11-25); N=cut 384, 5 builds, gate maps only; EST=few min Mac CPU; PERMIT=Shonu approved plan 2026-11-25
"""Where does each cut neuron's gate go? Same cut (384, 128 neurons left), three start points, one gate tool.

Gate map of neuron j = the cells (photo, 7 x 7 position) where j fires: read . patch + shift > 0, on the saved stage-3
maps of Broden held set 1 (vecread/results/held_s3.npy). Stage 3 is A's in every build here. Photos are for reading
only. Nothing is fitted.

Builds (block 0 of stage 4, 128 neurons each):
  C    = A's own gates of the 128 kept neurons (C and WC keep A's reads, checked in VECMATH 9)
  L    = trained from the cut network (A's weights minus the cut), lr ~1e-4, 4,000 steps
  R4C  = trained from C
  B0   = B128aug_s0 block 0, trained from random (blocks 0 and 1 trained), lr 3e-3, wd .05, 6,250 steps
  B1   = B128aug_s1, the same with seed 1

Measures (rec = share of A neuron d's on-cells that fall inside small neuron j's on-cells):
  null: j's map taken from another photo (photos rolled by half), same positions. Keeps j's rate and spatial habit,
        breaks the match with d. ex = rec - rec_null.
  d side (each of the 384 cut A neurons): top-1, top-2 union, top-5 union excess recall (top by ex).
  j side (each small neuron): how many of A's 512 neurons it covers (rec >= .5), minus the null count.
  P3 known answer: B0's n122 against A's 8 wheel neurons (NEURON-LOG N11).
Usage: python oc_absorb.py [--photos 2000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oc_gate import LAD, VR, WA, check_s4b0, patches, reads  # noqa: E402

ONE = os.path.dirname(HERE)
CLEAN = os.path.dirname(ONE)
R18 = os.path.join(CLEAN, "neuron", "results", "coarse", "r18")
OUT = os.path.join(ONE, "results", "oc_absorb")
CUT = 384
WHEEL = [79, 81, 37, 297, 496, 303, 458, 103]
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def gates(X3, R, sh, n, bs=100):
    """Bool on-maps (n*49, units), computed in chunks to keep memory small."""
    out = np.empty((n * 49, R.shape[0]), bool)
    for a in range(0, n, bs):
        b = min(n, a + bs)
        out[a * 49:b * 49] = patches(np.asarray(X3[a:b], np.float32)) @ R.T + sh > 0
    return out


def roll_photos(on, n):
    """Null: each cell gets the map value from the photo n/2 away, same position."""
    return np.roll(on.reshape(n, 49, -1), n // 2, axis=0).reshape(n * 49, -1)


def union_rec(onA_d, onJ, idx):
    u = onJ[:, idx].any(1)
    return float((onA_d & u).sum() / max(onA_d.sum(), 1))


def analyse(name, onA, onJ, onJn, D, n, labels):
    cntA = onA.sum(0).astype(np.float64)                       # (512,)
    Af = onA.astype(np.float32)
    inter = Af.T @ onJ.astype(np.float32)                      # (512, J)
    intern = Af.T @ onJn.astype(np.float32)
    rec = inter / np.maximum(cntA, 1)[:, None]
    recn = intern / np.maximum(cntA, 1)[:, None]
    ex = rec - recn
    # d side, cut neurons only
    dside = {"top1": [], "top2": [], "top5": [], "top1_null": [], "top2_null": [], "top5_null": [], "top1_j": []}
    for d in D:
        order = np.argsort(-ex[d])
        dside["top1_j"].append(int(labels[order[0]]))
        for k in (1, 2, 5):
            r = union_rec(onA[:, d], onJ, order[:k])
            rn = union_rec(onA[:, d], onJn, order[:k])
            dside[f"top{k}"].append(r - rn)
            dside[f"top{k}_null"].append(rn)
    # j side: how many A neurons each small neuron covers
    cov = (rec >= .5).sum(0)
    covn = (recn >= .5).sum(0)
    covered = {int(labels[j]): [int(d) for d in np.nonzero(rec[:, j] >= .5)[0]] for j in range(onJ.shape[1])}
    rate = onJ.mean(0)
    # rate-fair match (added after the smoke run: B fires at .40 vs A .17, so recall counts measured rate):
    # phi = correlation of the two on/off maps. 0 at chance whatever the rates.
    N = onA.shape[0]
    pA, pJ = cntA / N, onJ.sum(0) / N
    den = np.sqrt(np.outer(pA * (1 - pA), pJ * (1 - pJ))) + 1e-12
    phi = (inter / N - np.outer(pA, pJ)) / den                 # (512, J)
    phin = (intern / N - np.outer(pA, onJn.sum(0) / N)) / den
    pd = phi[D]                                                # cut neurons x small neurons
    jtop = phi.max(0)                                          # best A match of each small neuron
    res_phi = {"d_top1_phi_median": float(np.median(pd.max(1))),
               "d_n_phi_ge_.3_median": float(np.median((pd >= .3).sum(1))),
               "d_n_phi_ge_.2_median": float(np.median((pd >= .2).sum(1))),
               "j_top1_phi_median": float(np.median(jtop)),
               "j_nA_phi_ge_.3_median": float(np.median((phi >= .3).sum(0))),
               "j_nA_phi_ge_.2_median": float(np.median((phi >= .2).sum(0))),
               "j_share_copy_phi_ge_.8": float(np.mean(jtop >= .8)),
               "null_phi_abs_99": float(np.percentile(np.abs(phin), 99)),
               "per_d_top1_phi": pd.max(1).tolist(), "per_d_top1_j": [int(labels[x]) for x in pd.argmax(1)],
               "per_j_top1_phi": jtop.tolist(), "per_j_top1_A": phi.argmax(0).tolist(),
               "per_j_nA_phi_ge_.3": (phi >= .3).sum(0).tolist()}
    log(f"{name} phi: d top1 {res_phi['d_top1_phi_median']:.3f}, d n>=.3 {res_phi['d_n_phi_ge_.3_median']:.0f} | "
        f"j top1 {res_phi['j_top1_phi_median']:.3f}, j nA>=.3 {res_phi['j_nA_phi_ge_.3_median']:.0f}, "
        f"copies(>=.8) {res_phi['j_share_copy_phi_ge_.8']:.2f} | null |phi| 99th {res_phi['null_phi_abs_99']:.3f}")
    res = {"build": name, "phi": res_phi, "rate_median": float(np.median(rate)), "rate_A_median": float(np.median(onA.mean(0))),
           "d_top1_ex_median": float(np.median(dside["top1"])), "d_top2_ex_median": float(np.median(dside["top2"])),
           "d_top5_ex_median": float(np.median(dside["top5"])),
           "d_top2_null_median": float(np.median(dside["top2_null"])),
           "j_cover_median": float(np.median(cov)), "j_cover_null_median": float(np.median(covn)),
           "j_cover_ex_median": float(np.median(cov - covn)),
           "j_share_copy": float(np.mean(cov - covn <= 1)),
           "top1_j_distinct": int(len(set(dside["top1_j"]))),
           "top1_j_most_common": max(set(dside["top1_j"]), key=dside["top1_j"].count),
           "top1_j_most_common_count": max(dside["top1_j"].count(j) for j in set(dside["top1_j"])),
           "per_d": {k: v for k, v in dside.items()}, "per_j_cover": cov.tolist(), "per_j_cover_null": covn.tolist(),
           "per_j_rate": rate.tolist(), "labels": [int(x) for x in labels], "covered": covered}
    log(f"{name}: rate {res['rate_median']:.3f} (A {res['rate_A_median']:.3f}) | d top1/2/5 ex "
        f"{res['d_top1_ex_median']:.3f}/{res['d_top2_ex_median']:.3f}/{res['d_top5_ex_median']:.3f} | "
        f"j cover {res['j_cover_median']:.0f} (null {res['j_cover_null_median']:.0f}), copies {res['j_share_copy']:.2f} | "
        f"distinct top absorbers {res['top1_j_distinct']} of {len(labels)}")
    return res, rec, recn, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=int, default=2000)
    a = ap.parse_args()
    n = a.photos
    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if n < 2000 else ""
    wA = np.load(WA)
    X3 = np.load(os.path.join(VR, "held_s3.npy"), mmap_mode="r")
    S4 = np.load(os.path.join(VR, "held_s4b0.npy"), mmap_mode="r")
    err = check_s4b0(wA, X3, S4)
    log(f"check A block 0 vs held_s4b0: max rel err {err:.2e} (bar 1e-2)")
    assert err < 1e-2
    jt = np.load(os.path.join(LAD, f"n{CUT}", "jobtest_L.npz"))
    D, K = jt["D"], jt["K"]
    RA, shA = reads(wA)
    onA = gates(X3, RA, shA, n)
    log(f"A gates done, on share {onA.mean():.3f}")
    builds = {"C": (RA[K], shA[K], K)}
    for b in ("C", "L", "R4C"):
        R, sh = reads(np.load(os.path.join(LAD, f"n{CUT}", f"w_{b}.npz")))
        if b == "C":
            assert np.array_equal(R[K], RA[K]) and np.array_equal(sh[K], shA[K]), "C gates differ from A"
        builds[b] = (R[K], sh[K], K)
    for s in (0, 1):
        wB = np.load(os.path.join(R18, f"w_B128aug_s{s}.npz"))
        sc = wB["bn1_0"].astype(np.float64)
        scale = sc[0] / np.sqrt(sc[3] + 1e-5)
        RB = (wB["W1_0"].reshape(128, -1).astype(np.float64) * scale[:, None]).astype(np.float32)
        shB = (sc[1] - sc[2] * scale).astype(np.float32)
        builds[f"B{s}"] = (RB, shB, np.arange(128))
    keep_phi = {}
    out, keep = {"cut": CUT, "n_photos": n, "check_s4b0_rel_err": err, "D": D.tolist(), "K": K.tolist()}, {}
    for b, (R, sh, labels) in builds.items():
        onJ = gates(X3, R, sh, n)
        res, rec, recn, phi = analyse(b, onA, onJ, roll_photos(onJ, n), D, n, labels)
        out[b] = res
        keep[b] = (rec, recn)
        keep_phi[b] = phi
        del onJ
    # P3 known answer: B0 n122 against A's 8 wheel neurons
    rec, recn = keep["B0"]
    exw = rec[WHEEL] - recn[WHEEL]                                  # (8, 128)
    rank = [int((exw[i] > exw[i, 122]).sum()) + 1 for i in range(len(WHEEL))]
    out["P3"] = {"wheel": WHEEL, "rank_of_n122": rank, "rec_n122": rec[WHEEL, 122].tolist(),
                 "rec_null_n122": recn[WHEEL, 122].tolist(), "pass": int(sum(r <= 3 for r in rank)) >= 4}
    log(f"P3 wheel: n122 rank per wheel neuron {rank}; rec {np.round(rec[WHEEL, 122], 2).tolist()}; "
        f"pass {out['P3']['pass']} (bar: top 3 for >= 4 of 8)")
    ph = keep_phi["B0"][WHEEL]                                     # (8, 128)
    rk = [int((ph[i] > ph[i, 122]).sum()) + 1 for i in range(len(WHEEL))]
    out["P3_phi"] = {"rank_of_n122": rk, "phi_n122": ph[:, 122].tolist(), "best_B0_per_wheel": ph.argmax(1).tolist(),
                     "best_phi_per_wheel": ph.max(1).tolist(), "pass": int(sum(r <= 3 for r in rk)) >= 4}
    log(f"P3 (phi) wheel: n122 rank {rk}; phi {np.round(ph[:, 122], 2).tolist()}; best B0 neuron per wheel "
        f"{ph.argmax(1).tolist()} at {np.round(ph.max(1), 2).tolist()}; pass {out['P3_phi']['pass']}")
    np.savez_compressed(os.path.join(OUT, f"phi{tag}.npz"), **keep_phi)
    json.dump(out, open(os.path.join(OUT, f"absorb{tag}.json"), "w"))
    np.savez_compressed(os.path.join(OUT, f"rec{tag}.npz"), **{f"{b}_rec": r for b, (r, _) in keep.items()},
                        **{f"{b}_recnull": rn for b, (_, rn) in keep.items()})
    log("saved")


if __name__ == "__main__":
    main()
