# scale: neurons=A:block0-all; net=A, C-ladder, L-ladder, R4C-ladder; photos=saved, broden-held1; where=mac
# rung: Q=plan "Gates, part 2: read the tilt as vectors, test one reason" (approved 2026-11-25); N=4 cuts x 2 builds, same scope as VECMATH 9; EST=few min Mac CPU; PERMIT=Shonu approved plan 2026-11-25
"""VECMATH 10. Read each kept neuron's read TILT (trained read minus its A part) with vecread, and test one rule for it.

Object: t_j = r_j - alpha_j a_j (a = A's effective read = C's, r = trained read of L or R4C), 2304 numbers on s3k.
Metric for everything: the vecread read metric <u, v> = u . cov . v (cov of s3k).
Rule: q_j = - sum over j's on-cells of g_j(cell) x patch(cell), g_j = d(1/2 |stream_C - stream_A|^2) / d f_j (block 0,
before the ReLU; transposed 3 x 3 conv of the stream error with C's writes). Built from A and C only.
Controls: shuffle null (g_j permuted among j's on-cells within each photo, 20 times); neighbour control (j's on-cells,
g of the kept unit whose A read is most similar to a_j).
Usage: python oc_gate_vec.py [--cut 64] [--photos 200] [--examples]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oc_gate import bn, patches, reads, LAD, VR, WA, NAMED, CLEAN  # noqa: E402

sys.path.insert(0, os.path.join(CLEAN, "vecread"))
import vecread as vr  # noqa: E402

ONE = os.path.dirname(HERE)
OUT = os.path.join(ONE, "results", "oc_gate_vec")
NSH = 20
BS = 200
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def w2(w):
    s2, t2 = bn(w["bn2_0"])
    return (w["W2_0"] * s2[:, None, None, None]).astype(np.float32), t2.astype(np.float32)


def conv(f, W):
    """f (n, Cin, 7, 7), W (Cout, Cin, 3, 3), padding 1 -> (n, Cout, 7, 7)."""
    fp = np.pad(f, ((0, 0), (0, 0), (1, 1), (1, 1)))
    r = np.zeros((f.shape[0], W.shape[0], 7, 7), np.float32)
    for di in range(3):
        for dj in range(3):
            r += np.einsum("bchw,oc->bohw", fp[:, :, di:di + 7, dj:dj + 7], W[:, :, di, dj], optimize=True)
    return r


def conv_t(e, W):
    """Adjoint of conv: <conv(f, W), e> = <f, conv_t(e, W)>."""
    ep = np.pad(e, ((0, 0), (0, 0), (1, 1), (1, 1)))
    g = np.zeros((e.shape[0], W.shape[1], 7, 7), np.float32)
    for di in range(3):
        for dj in range(3):
            g += np.einsum("bohw,oc->bchw", ep[:, :, 2 - di:9 - di, 2 - dj:9 - dj], W[:, :, di, dj], optimize=True)
    return g


def check_adjoint():
    g = np.random.default_rng(0)
    f, e, W = g.standard_normal((2, 5, 7, 7)), g.standard_normal((2, 4, 7, 7)), g.standard_normal((4, 5, 3, 3))
    f, e, W = (x.astype(np.float32) for x in (f, e, W))
    a, b = float((conv(f, W) * e).sum()), float((f * conv_t(e, W)).sum())
    return abs(a - b) / abs(a)


def shuffle_within(vals, on, rng):
    """vals (n, 49, K), zero off on. Permute each unit's values among its on-cells within each photo."""
    order_on = np.argsort(~on, axis=1, kind="stable")             # on-cells first, original order
    keys = rng.random(on.shape)
    keys[~on] = 2.0
    order_rand = np.argsort(keys, axis=1)                         # on-cells first, random order
    out = np.zeros_like(vals)
    np.put_along_axis(out, order_on, np.take_along_axis(vals, order_rand, 1), 1)
    return out


def dic_W(M):
    return M.W


class Metric:
    def __init__(self, cov, W=None):
        self.cov = cov
        self.W = W                                                # cov^-1/2: maps white noise into the read metric

    def ip(self, U, V):                                           # row-wise <u, v>
        return np.einsum("ij,ij->i", U @ self.cov, V)

    def perp(self, Q, A):                                         # Q minus its part along A, row-wise
        return Q - (self.ip(Q, A) / self.ip(A, A))[:, None] * A

    def cos(self, U, V):
        return self.ip(U, V) / np.sqrt(self.ip(U, U) * self.ip(V, V))

    def cosmat(self, U, V):
        G = U @ self.cov @ V.T
        nu, nv = np.sqrt(self.ip(U, U)), np.sqrt(self.ip(V, V))
        return G / nu[:, None] / nv[None, :]


def pass1(cut, wA, wC, K, X3, n, nb, rng):
    """Rule q (K, 2304), shuffle nulls (NSH, K, 2304), neighbour control (K, 2304), plus stream-error size."""
    RA, shA = reads(wA)
    RC, shC = reads(wC)
    W2A, t2A = w2(wA)
    W2C, t2C = w2(wC)
    nK = len(K)
    Q = np.zeros((nK, 2304))
    Qs = np.zeros((NSH, nK, 2304))
    Qn = np.zeros((nK, 2304))
    err2 = 0.0
    for a in range(0, n, BS):
        b = min(n, a + BS)
        m = b - a
        P = patches(np.asarray(X3[a:b], np.float32))                 # (m*49, 2304)
        fA = np.maximum(P @ RA.T + shA, 0)
        fC = np.maximum(P @ RC.T + shC, 0)
        img = lambda F: F.reshape(m, 7, 7, -1).transpose(0, 3, 1, 2)
        e = (conv(img(fC), W2C) + t2C[None, :, None, None]) - (conv(img(fA), W2A) + t2A[None, :, None, None])
        err2 += float((e ** 2).sum())
        df = conv_t(e, W2C).transpose(0, 2, 3, 1).reshape(m * 49, 512)[:, K]   # dLoss / d f_j per cell
        on = (P @ RA[K].T + shA[K]) > 0
        G = np.where(on, -df, 0).astype(np.float32)                 # descent values on j's on-cells
        Q += G.T @ P
        Qn += np.where(on, -df[:, nb], 0).astype(np.float32).T @ P   # fixed run 3: j's on-cells, neighbour's values
        G3, on3 = G.reshape(m, 49, nK), on.reshape(m, 49, nK)
        for s in range(NSH):
            Qs[s] += shuffle_within(G3, on3, rng).reshape(m * 49, nK).T @ P
        log(f"  cut {cut} pass 1: photos {b}/{n}")
    return Q, Qs, Qn, err2 / (n * 49)


def gate_counts(P, R, sh):
    return (P @ R.T + sh) > 0


def pass2(wA, builds, preds, K, X3, n):
    """Per unit: training's new / lost cells, and how many of them each predicted read also opens / closes."""
    RA, shA = reads(wA)
    nK = len(K)
    res = {b: {k: np.zeros(nK) for k in ("onA", "new", "lost")} for b in builds}
    for b in builds:
        for p in preds[b]:
            res[b].update({f"{p}_new": np.zeros(nK), f"{p}_lost": np.zeros(nK),
                           f"{p}_hit_new": np.zeros(nK), f"{p}_hit_lost": np.zeros(nK)})
    for a in range(0, n, BS):
        P = patches(np.asarray(X3[a:min(n, a + BS)], np.float32))
        onA = gate_counts(P, RA[K], shA[K])
        for b, (R, sh) in builds.items():
            on = gate_counts(P, R, sh)
            new, lost = on & ~onA, onA & ~on
            d = res[b]
            d["onA"] += onA.sum(0)
            d["new"] += new.sum(0)
            d["lost"] += lost.sum(0)
            for p, Rp in preds[b].items():
                onp = gate_counts(P, Rp, shA[K])
                pn, pl = onp & ~onA, onA & ~onp
                d[f"{p}_new"] += pn.sum(0)
                d[f"{p}_lost"] += pl.sum(0)
                d[f"{p}_hit_new"] += (pn & new).sum(0)
                d[f"{p}_hit_lost"] += (pl & lost).sum(0)
    return res


def readings(dic, V):
    out = []
    for v in V:
        g = vr.geometry(dic, dic.pattern(v), "whitened", 5, "read")
        out.append({"up": [x[0] for x in g["up"]], "down": [x[0] for x in g["down"]],
                    "top_z": g["up"][0][2], "leftover": g["leftover"]})
    return out


def run_cut(cut, wA, X3, n, dic, M, rng):
    jt = np.load(os.path.join(LAD, f"n{cut}", "jobtest_L.npz"))
    D, K = jt["D"], jt["K"]
    nK = len(K)
    wC = np.load(os.path.join(LAD, f"n{cut}", "w_C.npz"))
    RA, shA = reads(wA)
    RC, shC = reads(wC)
    assert np.array_equal(RC[K], RA[K]) and np.array_equal(shC[K], shA[K]), "C's kept gates differ from A's"
    AK = RA[K].astype(np.float64)
    # neighbour: the kept unit (not j) whose A read is most similar to a_j, read metric
    S = M.cosmat(AK, AK)
    np.fill_diagonal(S, -np.inf)
    nb = S.argmax(1)
    Q, Qs, Qn, err2 = pass1(cut, wA, wC, K, X3, n, nb, rng)
    Qp, Qnp = M.perp(Q, AK), M.perp(Qn, AK)
    Qsp = np.stack([M.perp(q, AK) for q in Qs])
    res = {"cut": cut, "n_photos": n, "K": K.tolist(), "D": D.tolist(), "neighbour": K[nb].tolist(),
           "neighbour_read_cos": S[np.arange(nK), nb].tolist(), "stream_err2_per_cell_C": err2,
           "q_along_a_share": (1 - M.ip(Qp, Qp) / M.ip(Q, Q)).tolist()}
    # known answer: a synthetic tilt with cosine ~.2 to the rule (plus random noise in the read metric) must score
    # above its own shuffle 95th for >= .9 of units, else the score cannot see a real effect of this size
    g = rng.standard_normal(Qp.shape) @ dic_W(M)
    g = M.perp(g, AK)
    un = lambda V: V / np.sqrt(M.ip(V, V))[:, None]
    syn = .2 * un(Qp) + np.sqrt(1 - .04) * un(M.perp(g, Qp))
    cs = np.stack([M.cos(syn, q) for q in Qsp])
    res["known_answer"] = {"median_cos": float(np.median(M.cos(syn, Qp))),
                           "share_above_own_p95": float(np.mean(M.cos(syn, Qp) > np.percentile(cs, 95, 0)))}
    log(f"cut {cut} known answer {res['known_answer']} (bar share >= .9)")
    builds, preds, tilts = {}, {}, {}
    for b in ("L", "R4C"):
        R, sh = reads(np.load(os.path.join(LAD, f"n{cut}", f"w_{b}.npz")))
        RK = R[K].astype(np.float64)
        alpha = M.ip(RK, AK) / M.ip(AK, AK)
        t = RK - alpha[:, None] * AK
        tilts[b] = t
        tn = np.sqrt(M.ip(t, t))
        c = M.cos(t, Qp)
        cs = np.stack([M.cos(t, q) for q in Qsp])                 # (NSH, K)
        cn = M.cos(t, Qnp)
        unit = lambda V: V / np.sqrt(M.ip(V, V))[:, None]
        preds[b] = {"rule": (AK + tn[:, None] * unit(Qp)).astype(np.float32),
                    "shuffle": (AK + tn[:, None] * unit(Qsp[0])).astype(np.float32),
                    "neighbour": (AK + tn[:, None] * unit(Qnp)).astype(np.float32)}
        # added after run 1: the read metric (centred cov) does not see a shift of the MEAN input, and q carried a
        # mean shift 10-20x training's tilt. _zm = the same predicted tilt with its mean shift removed (raw projection
        # off mu, the mean s3k patch), then scaled to training's tilt length again.
        zm = lambda V: V - np.outer(V @ MU / (MU @ MU), MU)
        rnd = M.perp(np.random.default_rng(7).standard_normal(AK.shape) @ M.W, AK)   # added run 3: random direction
        for nm, V in (("rule", Qp), ("shuffle", Qsp[0]), ("neighbour", Qnp), ("random", rnd)):
            preds[b][f"{nm}_zm"] = (AK + tn[:, None] * unit(zm(V))).astype(np.float32)
        ms_b = {"tilt": (np.abs(t @ MU) / np.sqrt(M.ip(AK, AK))).tolist(),
                                            "rule": (np.abs((tn[:, None] * unit(Qp)) @ MU) / np.sqrt(M.ip(AK, AK))).tolist()}
        builds[b] = (R[K], sh[K])
        res[b] = {"alpha": alpha.tolist(), "tilt_rel_len": (tn / np.sqrt(M.ip(AK, AK))).tolist(),
                  "cos_rule": c.tolist(), "cos_shuffle_p95": np.percentile(cs, 95, 0).tolist(),
                  "cos_shuffle_med": np.median(cs, 0).tolist(), "cos_neighbour": cn.tolist(),
                  "shift_change": (sh[K] - shA[K]).tolist(), "mean_shift_over_spread": ms_b}
        log(f"cut {cut} {b}: median cos(t, q) {np.median(c):+.3f}, shuffle med {np.median(cs):+.3f}, "
            f"share > own p95 {np.mean(c > np.percentile(cs, 95, 0)):.2f}, neighbour med {np.median(cn):+.3f}")
    # added run 3: does the tilt point at the DELETED units' reads more than at other kept units' reads?
    # cos(t_j, a_x with its a_j part removed), read metric. Set size matched: m = min(|D|, |K|-1) random members, 20 draws.
    AD = RA[D].astype(np.float64)
    nA2 = M.ip(AK, AK)
    rng2 = np.random.default_rng(3)
    m = min(len(D), nK - 1)
    for b in ("L", "R4C"):
        t = tilts[b]
        tn = np.sqrt(M.ip(t, t))
        def cosx(X):
            G = (t @ M.cov) @ X.T                                 # <t_j, a_x>, = <t_j, a_x perp> since t_j is perp to a_j
            Cj = (AK @ M.cov) @ X.T                               # <a_j, a_x>
            nx = np.einsum("ij,ij->i", X @ M.cov, X)
            return G / tn[:, None] / np.sqrt(np.maximum(nx[None, :] - Cj ** 2 / nA2[:, None], 1e-12))
        cD, cK = cosx(AD), cosx(AK)
        np.fill_diagonal(cK, -np.inf)
        bD, bK = np.zeros((20, nK)), np.zeros((20, nK))
        for s_ in range(20):
            for i in range(nK):
                bD[s_, i] = cD[i, rng2.choice(len(D), m, replace=False)].max()
                kk = rng2.choice(np.delete(np.arange(nK), i), m, replace=False)
                bK[s_, i] = cK[i, kk].max()
        res[b]["toward_D"] = {"best_D_med": np.median(bD, 0).tolist(), "best_K_med": np.median(bK, 0).tolist(),
                              "share_D_gt_K": float(np.mean(np.median(bD, 0) > np.median(bK, 0))),
                              "top_D_unit": D[cD.argmax(1)].tolist(), "top_D_cos": cD.max(1).tolist(), "m": m}
        log(f"cut {cut} {b}: toward deleted reads, best cos D {np.median(bD):.3f} vs kept {np.median(bK):.3f}, "
            f"share D > K {res[b]['toward_D']['share_D_gt_K']:.2f}")
    # P1: reproducibility, same unit L vs R4C against mismatched pairs
    Cm = M.cosmat(tilts["L"], tilts["R4C"])
    off = Cm[~np.eye(nK, dtype=bool)]
    res["repro"] = {"same": np.diag(Cm).tolist(), "mismatch_p99": float(np.percentile(off, 99)),
                    "mismatch_med": float(np.median(off))}
    log(f"cut {cut} P1: same-unit median {np.median(np.diag(Cm)):+.3f}, mismatched p99 {np.percentile(off, 99):+.3f}")
    # gate maps
    g = pass2(wA, builds, preds, K, X3, n)
    for b in builds:
        res[b]["gates"] = {k: v.tolist() for k, v in g[b].items()}
    # vecread readings (geometry only, every unit)
    rd = {"A": readings(dic, AK), "q": readings(dic, Qp), "D": readings(dic, AD)}   # D added run 3
    for b in builds:
        rd[f"r_{b}"] = readings(dic, builds[b][0].astype(np.float64))
        rd[f"t_{b}"] = readings(dic, tilts[b])
    res["readings"] = rd
    log(f"cut {cut} readings done")
    return res, {"Qp": Qp, "tL": tilts["L"], "tR4C": tilts["R4C"], "AK": AK,
                 "rL": builds["L"][0], "rR4C": builds["R4C"][0], "K": K}


def check_p0(dic, wA):
    RA, _ = reads(wA)
    rd = readings(dic, RA.astype(np.float64))
    tz, lf = np.median([r["top_z"] for r in rd]), np.median([r["leftover"] for r in rd])
    return {"top_z_median": float(tz), "leftover_median": float(lf),
            "pass": bool(abs(tz - 3.8) <= .1 and abs(lf - .80) <= .1)}


def examples_named(dic, vecs_by_cut):
    held = vr.Held()
    out = {}
    for cut, (V, K) in vecs_by_cut.items():
        for j, d in NAMED.get(cut, []):
            i = int(np.nonzero(K == j)[0][0])
            names = ["A", "rL", "rR4C", "tL", "tR4C", "q"]
            M = np.stack([V[nm][i] for nm in names]).astype(np.float32)
            vals = held.project("s3k", M)
            out[f"n{cut}_j{j}"] = {nm: vr.examples(held, dic, vals[:, :, k]) for k, nm in enumerate(names)}
            RA, _ = reads(np.load(WA))                            # added run 3: the named deleted partner's A read
            vd = held.project("s3k", RA[d].astype(np.float32))
            out[f"n{cut}_d{d}"] = {"read": vr.read_vector(dic, RA[d].astype(np.float64), "read"), "examples": vr.examples(held, dic, vd)}
            log(f"examples cut {cut} n{j} done")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--photos", type=int, default=2000)
    ap.add_argument("--examples", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if a.photos < 2000 else ""
    adj = check_adjoint()
    log(f"check conv adjoint rel err {adj:.1e} (bar 1e-4)")
    assert adj < 1e-4
    wA = np.load(WA)
    X3 = np.load(os.path.join(VR, "held_s3.npy"), mmap_mode="r")
    dic = vr.Dictionary("s3k")
    M = Metric(dic.cov, dic.W)
    global MU
    MU = dic.mu.astype(np.float64)
    p0 = check_p0(dic, wA)
    log(f"P0 {p0}")
    json.dump(p0, open(os.path.join(OUT, f"p0{tag}.json"), "w"))
    if not p0["pass"]:
        sys.exit("P0 failed: kernel layout or reader differs from K5b; stop")
    rng = np.random.default_rng(10)
    vecs = {}
    for cut in ([a.cut] if a.cut else [11, 64, 192, 384]):
        res, v = run_cut(cut, wA, X3, a.photos, dic, M, rng)
        json.dump(res, open(os.path.join(OUT, f"n{cut}{tag}.json"), "w"))
        np.savez_compressed(os.path.join(OUT, f"vec_n{cut}{tag}.npz"), **v)
        vecs[cut] = ({"A": v["AK"], "rL": v["rL"], "rR4C": v["rR4C"], "tL": v["tL"], "tR4C": v["tR4C"],
                      "q": v["Qp"]}, v["K"])
        log(f"cut {cut} saved")
    if a.examples:
        ex = examples_named(dic, vecs)
        json.dump(ex, open(os.path.join(OUT, f"examples{tag}.json"), "w"))


if __name__ == "__main__":
    main()
