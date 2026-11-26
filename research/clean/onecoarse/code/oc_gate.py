# scale: neurons=A:block0-all; net=A, C-ladder, WC-ladder, L-ladder, R4C-ladder; photos=saved, broden-held1; where=mac
# rung: Q=plan "Gates: where do trained kept neurons switch on" (approved 2026-11-25); N=4 cuts x 4 builds, gate maps only; EST=few min Mac CPU; PERMIT=Shonu approved plan 2026-11-25
"""Gate maps of stage-4 block-0 neurons, A against the ladder builds. Mac, numpy only, no torch, no photo pass.

Gate map of neuron j = the cells (photo, 7 x 7 position) where j fires: bn1(conv1(stage 3)) > 0.
Input: saved stage-3 maps of Broden held set 1 (vecread/results/held_s3.npy, fp16). Stage 3 is A's in every build used.
Read of j = bn1 scale x conv1 row (effective read). Threshold of j = bn1 shift (gate: read . x + shift > 0).

Per cut, per trained build m (L, R4C), per kept unit j:
  onA, new (A off, m on), lost (A on, m off), new cells where some deleted unit fired in A, off-in-A cells where some
  deleted unit fired (chance), the gate partner (deleted unit whose A on-cells hold most of j's new cells) and its lift,
  and the new-on count with only the threshold changed (A's read) and with only the read changed (A's threshold).
Checks: A's block 0 on held_s3 reproduces held_s4b0 (r + skip, before ReLU); C and WC gate maps equal A's.
Usage: python oc_gate.py [--cut 64] [--photos 200]
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONE = os.path.dirname(HERE)
CLEAN = os.path.dirname(ONE)
LAD = os.path.join(ONE, "results", "oc_ladder")
VR = os.path.join(CLEAN, "vecread", "results")
WA = os.path.join(CLEAN, "neuron", "results", "coarse", "r18", "w_A.npz")
OUT = os.path.join(ONE, "results", "oc_gate")
NAMED = {11: [(5, 464)], 64: [(339, 476)], 192: [(430, 114)], 384: [(430, 501), (76, 374)]}
EPS = 1e-5
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def bn(rows):
    w, b, m, v = rows.astype(np.float64)
    sc = w / np.sqrt(v + EPS)
    return sc, b - m * sc


def patches(x):
    """x (B, C, 14, 14) -> (B*49, C*9): 3 x 3 patches, stride 2, padding 1, channel-major (C, 3, 3) like conv weights."""
    B, C = x.shape[:2]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    P = np.stack([xp[:, :, di:di + 13:2, dj:dj + 13:2] for di in range(3) for dj in range(3)], 2)  # B,C,9,7,7
    return P.transpose(0, 3, 4, 1, 2).reshape(B * 49, C * 9)


def reads(w):
    """Effective reads (512, 2304) and shifts (512,) of block 0."""
    sc, sh = bn(w["bn1_0"])
    return (w["W1_0"].reshape(512, -1).astype(np.float64) * sc[:, None]).astype(np.float32), sh.astype(np.float32)


def check_s4b0(wA, X3, S4, n=20):
    """A's whole block 0 (r + skip, before ReLU) against the saved held_s4b0."""
    x = np.asarray(X3[:n], np.float32)
    R, sh = reads(wA)
    f = np.maximum(patches(x) @ R.T + sh, 0).reshape(n, 7, 7, 512).transpose(0, 3, 1, 2)
    s2, t2 = bn(wA["bn2_0"])
    W2 = (wA["W2_0"] * s2[:, None, None, None]).astype(np.float32)
    fp = np.pad(f, ((0, 0), (0, 0), (1, 1), (1, 1)))
    r = np.zeros((n, 512, 7, 7), np.float32)
    for di in range(3):
        for dj in range(3):
            r += np.einsum("bchw,oc->bohw", fp[:, :, di:di + 7, dj:dj + 7], W2[:, :, di, dj], optimize=True)
    r += t2[None, :, None, None].astype(np.float32)
    sd, td = bn(wA["bnd"])
    sk = np.einsum("bchw,oc->bohw", x[:, :, ::2, ::2], (wA["Wd"][:, :, 0, 0] * sd[:, None]).astype(np.float32))
    sk += td[None, :, None, None].astype(np.float32)
    got, want = r + sk, np.asarray(S4[:n], np.float32)
    return float(np.abs(got - want).max() / np.abs(want).max())


def pre(X3, R, sh, n, bs=200):
    """read . x + shift for all cells of the first n photos -> (n*49, 512) float32 (no shift: shift added later)."""
    out = np.empty((n * 49, R.shape[0]), np.float32)
    for a in range(0, n, bs):
        b = min(n, a + bs)
        out[a * 49:b * 49] = patches(np.asarray(X3[a:b], np.float32)) @ R.T
    return out


def run_cut(cut, wA, X3, n):
    jt = np.load(os.path.join(LAD, f"n{cut}", "jobtest_L.npz"))
    D, K = jt["D"], jt["K"]
    best_d = dict(zip(jt["K"].tolist(), D[jt["best_d"]].tolist()))   # jobtest stores an index into D (bug fixed after run 1)
    RA, shA = reads(wA)
    uA = pre(X3, RA, shA, n)
    onA = uA + shA > 0
    onD = onA[:, D].astype(np.float32)                       # A's deleted units' gates
    Dany = onD.max(1) > 0                                    # some deleted unit on at the cell
    Dsum = np.maximum(uA[:, D] + shA[D], 0).sum(1)           # A's summed deleted firing per cell (added after run 1)
    xn = np.concatenate([np.linalg.norm(patches(np.asarray(X3[a:a + 200], np.float32)), axis=1)
                         for a in range(0, n, 200)])         # input patch norm per cell (added after review)
    photo = np.repeat(np.arange(n), 49)
    res = {"cut": cut, "n_photos": n, "D": D.tolist(), "K": K.tolist(), "Dany_share": float(Dany.mean())}
    # check: calculated builds keep A's gates
    for b in ("C", "WC"):
        R, sh = reads(np.load(os.path.join(LAD, f"n{cut}", f"w_{b}.npz")))
        res[f"check_{b}_gates_equal_A"] = bool(np.array_equal(R[K], RA[K]) and np.array_equal(sh[K], shA[K]))
    onAK = onA[:, K]
    offAK = ~onAK
    named = {}
    for b in ("L", "R4C"):
        R, sh = reads(np.load(os.path.join(LAD, f"n{cut}", f"w_{b}.npz")))
        u = pre(X3, R[K], sh[K], n)
        on = u + sh[K] > 0
        new, lost = on & offAK, onAK & ~on
        thr_only = (uA[:, K] + sh[K] > 0) & offAK            # A's read, trained threshold
        read_only = (u + shA[K] > 0) & offAK                 # trained read, A's threshold
        newf = new.astype(np.float32)
        cnt = newf.T @ onD                                   # (K, D): new cells of j where d was on in A
        offc = offAK.astype(np.float32).T @ onD              # (K, D): off cells of j where d was on
        part = cnt.argmax(1)
        nnew = new.sum(0)
        noff = offAK.sum(0)
        lift = (cnt[np.arange(len(K)), part] / np.maximum(nnew, 1)) / np.maximum(offc[np.arange(len(K)), part] / np.maximum(noff, 1), 1e-12)
        # lift rank of the 8.5 / jobtest partner, and the best-lift partner (support >= 20 new cells)
        liftall = (cnt / np.maximum(nnew, 1)[:, None]) / np.maximum(offc / np.maximum(noff, 1)[:, None], 1e-12)
        liftall[cnt < 20] = 0
        per_photo = np.stack([np.bincount(photo, (new & Dany[:, None]).sum(1), n), np.bincount(photo, new.sum(1), n),
                              np.bincount(photo, (offAK & Dany[:, None]).sum(1), n), np.bincount(photo, offAK.sum(1), n),
                              np.bincount(photo, lost.sum(1), n), np.bincount(photo, onAK.sum(1), n),
                              np.bincount(photo, (new * Dsum[:, None]).sum(1), n),
                              np.bincount(photo, (offAK * Dsum[:, None]).sum(1), n)], 1)
        # near miss (added after run 1): percentile of j's A pre-activation on its new-on cells among its A-off cells
        nearpct = np.full(len(K), np.nan)
        zA = uA[:, K] + shA[K]
        # margin-matched control (added after run 2): off cells that STAYED off with A pre-activation at least the
        # 10th pct of j's new-on cells' A pre-activation. Mean Dsum on new-on vs on these near-miss stayed-off cells.
        mm = np.zeros((len(K), 4))                           # sum Dsum new, n new, sum Dsum stayed, n stayed
        # null (added after review): per-unit MEAN firing of deleted units vs of kept units (j left out), and input norm
        nul = np.zeros((len(K), 6))                          # Dmean new, Dmean stayed, Kmean new, Kmean stayed, xn new, xn stayed (means)
        fK = np.maximum(zA, 0)
        Ksum = fK.sum(1)
        # gain or tilt (added after review): trained read = alpha x A's read + leftover; gate with alpha x A's read + trained shift
        RAK = RA[K].astype(np.float64)
        alpha = (R[K].astype(np.float64) * RAK).sum(1) / (RAK ** 2).sum(1)
        new_gain = (uA[:, K] * alpha.astype(np.float32) + sh[K] > 0) & offAK
        gain_overlap = (new_gain & new).sum(0)
        for i in range(len(K)):
            if nnew[i]:
                off = np.sort(zA[offAK[:, i], i])
                zn = zA[new[:, i], i]
                nearpct[i] = np.mean(np.searchsorted(off, zn) / len(off))
                st = offAK[:, i] & ~new[:, i] & (zA[:, i] >= np.percentile(zn, 10))
                mm[i] = Dsum[new[:, i]].sum(), nnew[i], Dsum[st].sum(), st.sum()
                if st.any():
                    km = (Ksum - fK[:, i]) / (len(K) - 1)
                    nul[i] = (Dsum[new[:, i]].mean() / len(D), Dsum[st].mean() / len(D), km[new[:, i]].mean(),
                              km[st].mean(), xn[new[:, i]].mean(), xn[st].mean())
        res[b] = {"onA": onAK.sum(0).tolist(), "new": nnew.tolist(), "lost": lost.sum(0).tolist(),
                  "newD": (new & Dany[:, None]).sum(0).tolist(), "offA": noff.tolist(),
                  "offAD": (offAK & Dany[:, None]).sum(0).tolist(),
                  "partner": D[part].tolist(), "partner_share": (cnt[np.arange(len(K)), part] / np.maximum(nnew, 1)).tolist(),
                  "partner_lift": lift.tolist(), "lift_partner": D[liftall.argmax(1)].tolist(),
                  "lift_partner_lift": liftall.max(1).tolist(),
                  "jobtest_partner": [int(best_d[k]) for k in K.tolist()],
                  "new_thr_only": thr_only.sum(0).tolist(), "new_read_only": read_only.sum(0).tolist(),
                  "per_photo": per_photo.tolist(), "near_pct": nearpct.tolist(), "margin_matched": mm.tolist(), "null": nul.tolist(),
                  "alpha": alpha.tolist(), "gain_overlap": gain_overlap.tolist(),
                  "new_gain": new_gain.sum(0).tolist()}
        for j, d in NAMED.get(cut, []):
            ij, idd = int(np.nonzero(K == j)[0][0]), int(np.nonzero(D == d)[0][0])
            named[f"{b}_n{j}_n{d}"] = {"onA_j": onAK[:, ij].reshape(n, 49), "new_j": new[:, ij].reshape(n, 49),
                                       "onA_d": onA[:, d].reshape(n, 49),
                                       "rank_of_d_by_count": int((cnt[ij] > cnt[ij, idd]).sum()) + 1,
                                       "rank_of_d_by_lift": int((liftall[ij] > liftall[ij, idd]).sum()) + 1}
        log(f"cut {cut} {b}: median new {np.median(nnew / np.maximum(onAK.sum(0), 1)):.3f} "
            f"lost {np.median(lost.sum(0) / np.maximum(onAK.sum(0), 1)):.3f}")
        del u, on, new, lost, thr_only, read_only, newf
    return res, named


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--photos", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if a.photos < 2000 else ""
    wA = np.load(WA)
    X3 = np.load(os.path.join(VR, "held_s3.npy"), mmap_mode="r")
    S4 = np.load(os.path.join(VR, "held_s4b0.npy"), mmap_mode="r")
    err = check_s4b0(wA, X3, S4)
    log(f"check A block 0 vs held_s4b0: max rel err {err:.2e} (bar 1e-2)")
    assert err < 1e-2
    for cut in ([a.cut] if a.cut else [11, 64, 192, 384]):
        res, named = run_cut(cut, wA, X3, a.photos)
        res["check_s4b0_rel_err"] = err
        assert res["check_C_gates_equal_A"] and res["check_WC_gates_equal_A"], "calculated gates differ from A"
        json.dump(res, open(os.path.join(OUT, f"n{cut}{tag}.json"), "w"))
        np.savez_compressed(os.path.join(OUT, f"named_n{cut}{tag}.npz"),
                            **{f"{k}__{f}": v for k, d in named.items() for f, v in d.items()})
        log(f"cut {cut} saved; C, WC gates equal A: True")


if __name__ == "__main__":
    main()
