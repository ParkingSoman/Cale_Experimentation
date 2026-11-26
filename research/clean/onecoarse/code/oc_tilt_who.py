# scale: neurons=A:block0-all; net=A, C-ladder, L-ladder, R4C-ladder; photos=saved, broden-held1; where=mac
# rung: Q=plan "Which kept neurons tilt, toward which cut neuron, parent reason" (approved 2026-11-25); same scope as VECMATH 10; EST=few min Mac
"""VECMATH 11. Which kept neurons AIM their read tilt at a cut neuron, at which one, and why.

aim(j, d) = <t_j, a_d>_M / (|t_j|_M |a_d - proj_{a_j} a_d|_M), M = cov of vecread s3k (section 10's metric).
Tilts come from section 10's saved vectors (results/oc_gate_vec/vec_n{cut}.npz).
Usage: python oc_tilt_who.py [--cut 64] [--examples]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oc_gate_vec import Metric, w2, CLEAN, LAD, VR, WA, reads, patches  # noqa: E402

sys.path.insert(0, os.path.join(CLEAN, "vecread"))
import vecread as vr  # noqa: E402

ONE = os.path.dirname(HERE)
GV = os.path.join(ONE, "results", "oc_gate_vec")
OUT = os.path.join(ONE, "results", "oc_tilt_who")
CUTS = [11, 64, 192, 384]
P = 69                     # the parent channel s3:69
NPERM = 1000
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def aim_matrix(M, T, AJ, X):
    """aim of each tilt T[i] (orthogonal to AJ[i]) at each read X[k], with AJ[i]'s part removed from X[k]."""
    tn = np.sqrt(M.ip(T, T))
    G = (T @ M.cov) @ X.T
    Cj = (AJ @ M.cov) @ X.T
    nA2 = M.ip(AJ, AJ)
    nx = np.einsum("ij,ij->i", X @ M.cov, X)
    return G / tn[:, None] / np.sqrt(np.maximum(nx[None, :] - Cj ** 2 / nA2[:, None], 1e-12))


def spear_perm(x, y, rng):
    r = spearmanr(x, y)[0]
    null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(NPERM)])
    return {"rho": float(r), "null_p99": float(np.percentile(null, 99)), "null_p1": float(np.percentile(null, 1))}


def channel_share(R):
    """(n, 2304) effective reads -> (n, 256) share of read energy per stage-3 channel."""
    e = (R.reshape(len(R), 256, 9).astype(np.float64) ** 2).sum(2)
    return e / e.sum(1, keepdims=True)


def firing_corr(RA, shA, X3, n=2000, bs=200):
    """Correlation of A's firings, all 512 block-0 neurons, over every cell of n saved photos."""
    S = np.zeros(512)
    SS = np.zeros((512, 512))
    N = 0
    for a in range(0, n, bs):
        f = np.maximum(patches(np.asarray(X3[a:min(n, a + bs)], np.float32)) @ RA.T + shA, 0).astype(np.float64)
        S += f.sum(0)
        SS += f.T @ f
        N += len(f)
    mu = S / N
    C = SS / N - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(C), 1e-12))
    return C / np.outer(sd, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--examples", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cuts = [a.cut] if a.cut else CUTS
    tag = "_smoke" if a.cut else ""
    rng = np.random.default_rng(11)
    dic = vr.Dictionary("s3k")
    M = Metric(dic.cov, dic.W)
    wA = np.load(WA)
    RA, shA = reads(wA)
    RA64 = RA.astype(np.float64)
    W2A, _ = w2(wA)
    wnorm = np.sqrt((W2A.astype(np.float64) ** 2).sum((0, 2, 3)))        # write norm per neuron (R1's write factor)
    p69 = RA64.reshape(512, 256, 9)[:, P, :].sum(1)                      # read weight on s3:69 (R1's read factor)
    share = channel_share(RA64)
    X3 = np.load(os.path.join(VR, "held_s3.npy"), mmap_mode="r")
    corrA = firing_corr(RA, shA, X3)
    log("A firing correlations done")
    readcos = M.cosmat(RA64, RA64)                                       # whole-read similarity, read metric

    V, res, aims = {}, {}, {}
    for c in CUTS:
        V[c] = np.load(os.path.join(GV, f"vec_n{c}.npz"))
    for c in cuts:
        K = V[c]["K"]
        jt = np.load(os.path.join(LAD, f"n{c}", "jobtest_L.npz"))
        D = jt["D"]
        assert np.array_equal(jt["K"], K)
        beta = np.load(os.path.join(LAD, f"n{c}", "beta_C.npy")).astype(np.float64)     # (D, K)
        old = json.load(open(os.path.join(GV, f"n{c}.json")))
        AK = V[c]["AK"]
        r = {"cut": c, "K": K.tolist(), "D": D.tolist()}
        for b in ("L", "R4C"):
            T = V[c][f"t{b}"]
            Am = aim_matrix(M, T, AK, RA64[D])                            # (K, D)
            aims[(c, b)] = Am
            best, part = Am.max(1), D[Am.argmax(1)]
            # check against section 10's saved numbers
            err = float(np.abs(best - np.array(old[b]["toward_D"]["top_D_cos"])).max())
            assert err < 1e-6, f"aim differs from section 10 by {err}"
            ki = {int(k): i for i, k in enumerate(K)}
            di = {int(d): i for i, d in enumerate(D)}
            # unit level (Q2, Q5 + reported)
            unit = {"best_aim": best.tolist(), "partner": part.tolist(),
                    "parent_read": spear_perm(p69[K], best, rng), "write_norm": spear_perm(wnorm[K], best, rng),
                    "beta_mass": spear_perm(np.sqrt((beta ** 2).sum(0)), best, rng),
                    "cofire_maxD": spear_perm(corrA[np.ix_(K, D)].max(1), best, rng),
                    "self": spear_perm(jt["self"], best, rng) if b == "L" else None}
            # pair level (Q3, Q4 + reported): is the partner the top-ranked d on each score?
            scores = {"shared_parent": np.minimum(share[K][:, None, :], share[D][None, :, :]).max(2),
                      "read_similarity": readcos[np.ix_(K, D)],
                      "beta_C": beta.T,
                      "cofire": corrA[np.ix_(K, D)]}
            pi = Am.argmax(1)
            pair = {}
            for nm, S in scores.items():
                hit = float(np.mean(S.argmax(1) == pi))
                rank = np.array([(S[i] > S[i, pi[i]]).sum() for i in range(len(K))])       # 0 = top
                pair[nm] = {"hit": hit, "chance": 1 / len(D), "ratio": hit * len(D),
                            "partner_rank_pct_median": float(np.median(1 - rank / max(len(D) - 1, 1)))}
            # Q4: shared parent within the partner's whole-read-similarity fifth
            RS, SP = scores["read_similarity"], scores["shared_parent"]
            h4, ch4 = [], []
            for i in range(len(K)):
                q = np.quantile(RS[i], [.2, .4, .6, .8])
                band = np.searchsorted(q, RS[i])
                same = np.nonzero(band == band[pi[i]])[0]
                if len(same) < 2:
                    continue
                h4.append(same[SP[i, same].argmax()] == pi[i])
                ch4.append(1 / len(same))
            pair["shared_parent_within_read_fifth"] = {"hit": float(np.mean(h4)), "chance": float(np.mean(ch4)),
                                                       "ratio": float(np.mean(h4) / np.mean(ch4)), "n": len(h4)}
            # known answer for the pair test: plant tilts toward each j's top shared-parent d at cosine .3
            dstar = SP.argmax(1)
            ad = RA64[D[dstar]]
            ad_perp = M.perp(ad, AK)
            un = lambda X: X / np.sqrt(M.ip(X, X))[:, None]
            noise = M.perp(M.perp(rng.standard_normal(T.shape) @ M.W, AK), ad_perp)
            Tsyn = .3 * un(ad_perp) + np.sqrt(1 - .09) * un(noise)
            ka = float(np.mean(aim_matrix(M, Tsyn, AK, RA64[D]).argmax(1) == dstar))
            pair["known_answer_planted_hit"] = ka
            r[b] = {"unit": unit, "pair": pair}
            log(f"cut {c} {b}: parent rho {unit['parent_read']['rho']:+.3f} (p99 {unit['parent_read']['null_p99']:+.3f}), "
                f"write rho {unit['write_norm']['rho']:+.3f}, beta rho {unit['beta_mass']['rho']:+.3f}, "
                f"cofire rho {unit['cofire_maxD']['rho']:+.3f} | partner hit x chance: "
                + ", ".join(f"{k} {v['ratio']:.1f}" for k, v in pair.items() if isinstance(v, dict))
                + f" | known answer {ka:.2f}")
        # vecread groups: A reads of the top / bottom aim tenth (R4C), from section 10's saved readings
        best = aims[(c, "R4C")].max(1)
        o = np.argsort(-best)
        n10 = max(1, len(K) // 10)
        rd = old["readings"]
        grp = {"top": o[:n10].tolist(), "bottom": o[-n10:].tolist()}
        r["groups"] = {g: {"units": K[idx].tolist(), "A_up": [rd["A"][i]["up"] for i in idx],
                           "best_aim": best[idx].tolist()} for g, idx in grp.items()}
        r["top5"] = [{"j": int(K[i]), "aim": float(best[i]), "partner": int(D[aims[(c, 'R4C')][i].argmax()]),
                      "A": rd["A"][i], "t_R4C": rd["t_R4C"][i],
                      "partner_A": rd["D"][int(aims[(c, 'R4C')][i].argmax())],
                      "self_L": float(jt["self"][i]), "p69": float(p69[K[i]]),
                      "shared_parent_channel": int(np.argmax(np.minimum(share[K[i]], share[D[aims[(c, 'R4C')][i].argmax()]]))),
                      "shared_parent_value": float(np.minimum(share[K[i]], share[D[aims[(c, 'R4C')][i].argmax()]]).max())}
                     for i in o[:5]]
        res[c] = r
        json.dump(r, open(os.path.join(OUT, f"n{c}{tag}.json"), "w"))
        log(f"cut {c} saved")

    # Q1: cross-cut test (needs all cuts)
    if not a.cut:
        cross = {}
        for c1, c2 in ((11, 64), (64, 192), (192, 384)):
            K1, K2 = V[c1]["K"], V[c2]["K"]
            D1, D2 = set(np.load(os.path.join(LAD, f"n{c1}", "jobtest_L.npz"))["D"].tolist()), \
                set(np.load(os.path.join(LAD, f"n{c2}", "jobtest_L.npz"))["D"].tolist())
            E = np.array(sorted(D2 - D1))
            i1 = {int(k): i for i, k in enumerate(K1)}
            J = K2
            out = {"n_j": len(J), "n_entering": len(E)}
            for b in ("L", "R4C"):
                T1 = V[c1][f"t{b}"][[i1[int(j)] for j in J]]
                T2 = V[c2][f"t{b}"]
                AJ = RA64[J]
                X = np.concatenate([RA64[E], RA64[J]])                      # entering, then staying (kept at both)
                rise = aim_matrix(M, T2, AJ, X) - aim_matrix(M, T1, AJ, X)  # (J, E + J)
                lab = np.r_[np.ones(len(E), bool), np.zeros(len(J), bool)]
                selfmask = np.zeros_like(rise, bool)
                selfmask[np.arange(len(J)), len(E) + np.arange(len(J))] = True

                def diff(l):
                    m_e = np.median(rise[:, l][~selfmask[:, l]])
                    m_s = np.median(rise[:, ~l][~selfmask[:, ~l]])
                    return m_e - m_s, m_e, m_s
                d0, me, ms = diff(lab)
                null = np.array([diff(rng.permutation(lab))[0] for _ in range(NPERM)])
                # per j: its best-aimed entering d at c2 - what was its aim at c1?
                a1, a2 = aim_matrix(M, T1, AJ, RA64[E]), aim_matrix(M, T2, AJ, RA64[E])
                k = a2.argmax(1)
                out[b] = {"median_rise_entering": float(me), "median_rise_staying": float(ms), "diff": float(d0),
                          "null_p99": float(np.percentile(null, 99)),
                          "partner_aim_c1_median": float(np.median(a1[np.arange(len(J)), k])),
                          "partner_aim_c2_median": float(np.median(a2[np.arange(len(J)), k]))}
                if c1 == 11 and 339 in J.tolist() and 476 in E.tolist():
                    jj, ee = J.tolist().index(339), E.tolist().index(476)
                    out[b]["n339_n476"] = [float(a1[jj, ee]), float(a2[jj, ee])]
                log(f"Q1 {c1}->{c2} {b}: rise entering {me:+.3f} staying {ms:+.3f} diff {d0:+.3f} (null p99 "
                    f"{np.percentile(null, 99):+.3f}); partner aim {out[b]['partner_aim_c1_median']:.3f} -> "
                    f"{out[b]['partner_aim_c2_median']:.3f}")
            cross[f"{c1}_{c2}"] = out
        json.dump(cross, open(os.path.join(OUT, "cross.json"), "w"))

    if a.examples:
        held = vr.Held()
        ex = {}
        for c in cuts:
            for t in res[c]["top5"]:
                j, d = t["j"], t["partner"]
                i = res[c]["K"].index(j)
                vecs = {"A": RA64[j], "t_R4C": V[c]["tR4C"][i], "partner_A": RA64[d]}
                ex[f"n{c}_j{j}"] = {k: vr.examples(held, dic, held.project("s3k", v.astype(np.float32)))
                                    for k, v in vecs.items()}
            log(f"examples cut {c} done")
        json.dump(ex, open(os.path.join(OUT, f"examples{tag}.json"), "w"))


if __name__ == "__main__" and "--checks" not in sys.argv:
    main()


def checks():
    """Added after the run (advisor review): (1) hub-matched null for the partner hit rates: score rows permuted across
    kept neurons (keeps which cut neurons are popular, breaks the pair link), 1,000 times; partner popularity.
    (2) overlap of the s3:69 read with beta_C mass and co-firing. (3) top-partner check (was inline: cross_matched.json),
    with the per-neuron paired share."""
    rng = np.random.default_rng(12)
    dic = vr.Dictionary("s3k")
    M = Metric(dic.cov, dic.W)
    wA = np.load(WA)
    RA, shA = reads(wA)
    RA64 = RA.astype(np.float64)
    p69 = RA64.reshape(512, 256, 9)[:, P, :].sum(1)
    share = channel_share(RA64)
    X3 = np.load(os.path.join(VR, "held_s3.npy"), mmap_mode="r")
    corrA = firing_corr(RA, shA, X3)
    readcos = M.cosmat(RA64, RA64)
    V = {c: np.load(os.path.join(GV, f"vec_n{c}.npz")) for c in CUTS}
    out = {}
    for c in CUTS:
        K = V[c]["K"]
        D = np.load(os.path.join(LAD, f"n{c}", "jobtest_L.npz"))["D"]
        beta = np.load(os.path.join(LAD, f"n{c}", "beta_C.npy")).astype(np.float64)
        scores = {"shared_parent": np.minimum(share[K][:, None, :], share[D][None, :, :]).max(2),
                  "read_similarity": readcos[np.ix_(K, D)], "beta_C": beta.T, "cofire": corrA[np.ix_(K, D)]}
        o = {}
        for b in ("L", "R4C"):
            pi = aim_matrix(M, V[c][f"t{b}"], V[c]["AK"], RA64[D]).argmax(1)
            cnt = np.bincount(pi, minlength=len(D))
            top = np.argsort(-cnt)[:3]
            ob = {"partner_popularity": [(int(D[t]), int(cnt[t])) for t in top], "n": len(K)}
            for nm, S in scores.items():
                tops = S.argmax(1)
                hit = float(np.mean(tops == pi))
                null = np.array([np.mean(tops[rng.permutation(len(K))] == pi) for _ in range(NPERM)])
                ob[nm] = {"hit": hit, "hub_null_median": float(np.median(null)), "hub_null_p99": float(np.percentile(null, 99)),
                          "ratio_to_hub_null": hit / max(float(np.median(null)), 1e-9)}
            o[b] = ob
            log(f"checks cut {c} {b}: popular partners {ob['partner_popularity']} | hit / hub-null median (p99): "
                + ", ".join(f"{k} {v['hit']:.3f}/{v['hub_null_median']:.3f} ({v['hub_null_p99']:.3f})" for k, v in ob.items() if isinstance(v, dict)))
        o["overlap"] = {"p69_vs_beta_mass": float(spearmanr(p69[K], np.sqrt((beta ** 2).sum(0)))[0]),
                        "p69_vs_cofire_maxD": float(spearmanr(p69[K], corrA[np.ix_(K, D)].max(1))[0]),
                        "beta_mass_vs_cofire_maxD": float(spearmanr(np.sqrt((beta ** 2).sum(0)), corrA[np.ix_(K, D)].max(1))[0])}
        log(f"checks cut {c} overlap {o['overlap']}")
        out[c] = o
    # top-partner check with paired share
    tp = {}
    for c1, c2 in ((11, 64), (64, 192), (192, 384)):
        K1, K2 = V[c1]["K"], V[c2]["K"]
        D1 = set(np.load(os.path.join(LAD, f"n{c1}", "jobtest_L.npz"))["D"].tolist())
        D2 = set(np.load(os.path.join(LAD, f"n{c2}", "jobtest_L.npz"))["D"].tolist())
        E = np.array(sorted(D2 - D1))
        i1 = {int(k): i for i, k in enumerate(K1)}
        m = min(len(E), len(K2) - 1)
        for b in ("L", "R4C"):
            T1 = V[c1][f"t{b}"][[i1[int(j)] for j in K2]]
            T2 = V[c2][f"t{b}"]
            AJ = RA64[K2]
            e1, e2 = aim_matrix(M, T1, AJ, RA64[E]), aim_matrix(M, T2, AJ, RA64[E])
            s1, s2 = aim_matrix(M, T1, AJ, RA64[K2]), aim_matrix(M, T2, AJ, RA64[K2])
            np.fill_diagonal(s1, -np.inf)
            np.fill_diagonal(s2, -np.inf)
            de, ds = np.zeros((20, len(K2))), np.zeros((20, len(K2)))
            for s_ in range(20):
                for i in range(len(K2)):
                    ce = rng.choice(len(E), m, replace=False) if m < len(E) else np.arange(len(E))
                    k = ce[e2[i, ce].argmax()]
                    de[s_, i] = e2[i, k] - e1[i, k]
                    cs = rng.choice(np.delete(np.arange(len(K2)), i), m, replace=False)
                    k = cs[s2[i, cs].argmax()]
                    ds[s_, i] = s2[i, k] - s1[i, k]
            dme, dms = np.median(de, 0), np.median(ds, 0)
            tp[f"{c1}_{c2}_{b}"] = {"rise_entering_median": float(np.median(dme)), "rise_staying_median": float(np.median(dms)),
                                    "paired_share_entering_gt_staying": float(np.mean(dme > dms)), "m": m}
            log(f"top partner {c1}->{c2} {b}: rise entering {np.median(dme):.3f} vs staying pick {np.median(dms):.3f}, "
                f"share of neurons entering > staying {np.mean(dme > dms):.2f}")
    json.dump({"pair_hub_null": {str(k): v for k, v in out.items()}, "top_partner": tp},
              open(os.path.join(OUT, "checks.json"), "w"))


if __name__ == "__main__" and "--checks" in sys.argv:
    checks()
