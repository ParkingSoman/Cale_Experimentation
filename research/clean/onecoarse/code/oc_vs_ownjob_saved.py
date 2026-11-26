# scale: neurons=A:block0-all; net=A, Z-ladder, C-ladder, WC-ladder, L-ladder, R4C-ladder; photos=saved; where=mac
# rung: Q=plan "Answer the two open questions from saved outputs", step 2+3; N=saved per-unit swap-back KL + saved firings + saved weights; EST=1 min Mac; PERMIT=Shonu approved plan 2026-11-25
"""Does morphing hurt a kept neuron's OWN job? Read from SAVED outputs only (Mac, numpy, no new forward pass).

Inputs (all saved):
  results/oc_vs_jobmap/n<cut>.npz (job 6994844): for every kept unit j and every photo p of a union set U
      <b>_full[j, p] = KL(A || build b, j alone swapped back to A) - KL(A || build b)
      (L, R4C also <b>_read / <b>_write: only that side swapped back). U = D's top photos + 500 random test photos.
  results/oc_step1.npz 'A_pos' (19,000 x 512): A's block-0 peak firing per photo. Rows 2000: are the 17,000 test
      photos in the job-map order (checked: labels equal on all 17,000; corr with job-map pkD_u .998).
  ~/uva-gpu/inbox/oc_vs_ownjob_smoke/n64.npz: 20 units at cut 64 on their TRUE top-50 own and clean photos (truth).
  results/oc_ladder/n<cut>/w_<b>.npz: weights (step 3).

Own photo sets of kept unit j, inside U (U holds only part of each unit's true top 50):
  top50-in-U  j's true top-50 test photos that are in U (coverage reported)
  rand-top    j's top 25 by A firing among the 500 random photos (D not selected for)
  clean-top   j's top 25 among U photos whose A summed D firing is below the test median
  h_j (nats/photo) = -mean <b>_full[j, own set]   (> 0: swapping j back helps its own photos = the morph hurt)
Validity bar (written before the read): a set stands in for own photos only if, over the 20 smoke units and 4 builds,
  sign agreement with the smoke's own h >= .8 and rank correlation >= .6.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oc_vs_ownjob_read import widened, unit_rows  # noqa: E402
from oc_vs_read import R18, LAD, eff  # noqa: E402
import vecread as vr  # noqa: E402

RES = os.path.join(HERE, "..", "results")
JM = os.path.join(RES, "oc_vs_jobmap")
SMOKE = os.path.expanduser("~/uva-gpu/inbox/oc_vs_ownjob_smoke/n64.npz")
BUILDS = ("C", "WC", "L", "R4C")
SETS = ("top50-in-U", "rand-top", "clean-top")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def own_sets(z, AP):
    """Per kept unit: positions in U of its three own photo sets; also true-top-50 coverage."""
    K, U, rand_u = z["K"], z["union"], z["rand_u"]
    D = z["D"]
    pos = {int(p): i for i, p in enumerate(U)}
    dsum = AP[:, D].sum(1)
    clean_u = np.nonzero(dsum[U] < np.median(dsum))[0]
    out, cov = {s: [] for s in SETS}, []
    for k in K:
        top = np.argsort(-AP[:, k])[:50]
        inu = np.array([pos[int(p)] for p in top if int(p) in pos], dtype=int)
        cov.append(len(inu))
        out["top50-in-U"].append(inu)
        out["rand-top"].append(rand_u[np.argsort(-AP[U[rand_u], k])[:25]])
        out["clean-top"].append(clean_u[np.argsort(-AP[U[clean_u], k])[:25]])
    return out, np.array(cov), len(clean_u)


def hvec(F, sets):
    return np.array([-F[i, s].mean() if len(s) else np.nan for i, s in enumerate(sets)])


def main():
    AP = np.load(os.path.join(RES, "oc_step1.npz"))["A_pos"][2000:].astype(np.float32)
    dk = vr.Dictionary("s3k")
    pat = lambda R: dk.metric(np.atleast_2d(R) @ dk.cov.T, True)
    VA, _, RA, _ = eff(np.load(os.path.join(R18, "w_A.npz")))
    res = {}
    md = ["# Does morphing hurt a kept neuron's own job? From saved outputs (`code/oc_vs_ownjob_saved.py`)", "",
          "h = KL (nats/photo) that swapping j back REMOVES on j's own photos. h > 0 means the morph hurt j's own photos.", ""]

    # 1. validity of each own set against the smoke truth (cut 64)
    z = np.load(os.path.join(JM, "n64.npz"))
    sets, cov, nclean = own_sets(z, AP)
    s = np.load(SMOKE)
    idx = np.array([{int(k): i for i, k in enumerate(z["K"])}[int(u)] for u in s["units"]])
    md += ["## 1. Which saved photo set stands in for a unit's own photos? (cut 64, the 20 smoke units)", "",
           "| Set | same sign as smoke own h (pooled 4 builds) | rank corr with smoke own h (pooled) | passes (>= .8 and >= .6) |",
           "|---|---|---|---|"]
    valid = {}
    for st in SETS:
        a, b = [], []
        for bd in BUILDS:
            h = hvec(z[f"{bd}_full"].astype(np.float64), [sets[st][i] for i in idx])
            a.append(h)
            b.append(s[f"{bd}_base_own"] - s[f"{bd}_full_own"])
        a, b = np.concatenate(a), np.concatenate(b)
        ok = ~np.isnan(a)
        sg, rc = float(np.mean(np.sign(a[ok]) == np.sign(b[ok]))), spearman(a[ok], b[ok])
        valid[st] = dict(sign=sg, rho=rc, n=int(ok.sum()), passes=bool(sg >= .8 and rc >= .6))
        md.append(f"| {st} | {sg:.2f} (n {int(ok.sum())}) | {rc:+.2f} | {'yes' if valid[st]['passes'] else 'no'} |")
    res["validity"] = valid
    md += ["", f"Smoke own h for scale (true own photos, cut 64): median over 20 units and 4 builds "
           f"{np.median(np.concatenate([s[f'{bd}_base_own'] - s[f'{bd}_full_own'] for bd in BUILDS])):+.5f} nats; "
           f"own-job size median {np.median(s['A_Adel_own']):.5f}.", ""]

    # 2. every kept unit, every cut, on the sets that pass
    md += ["## 2. Every kept unit (sets that passed section 1 only)", ""]
    for n in (11, 64, 192, 384):
        z = np.load(os.path.join(JM, f"n{n}.npz"))
        K = z["K"]
        sets, cov, nclean = own_sets(z, AP)
        wid, _ = widened(n, VA, RA, pat)
        W = np.array([wid[int(k)] for k in K])
        r = dict(top50_coverage_median=float(np.median(cov)), n_clean=nclean, n_widened=int(W.sum()), builds={})
        md += [f"### Cut {n}: {len(K)} kept units ({int(W.sum())} widened). True top-50 photos inside U: median "
               f"{np.median(cov):.0f} of 50", "",
               "| Build | Set | h median, widened / control (nats) | units with h > 0 | median h / build KL on the same photos | "
               "read / write side h, median (L, R4C) |", "|---|---|---|---|---|---|"]
        for bd in BUILDS:
            F = z[f"{bd}_full"].astype(np.float64)
            base = z[f"{bd}_base_u"].astype(np.float64)
            r["builds"][bd] = {}
            for st in SETS:
                if not valid[st]["passes"]:
                    continue
                h = hvec(F, sets[st])
                ok = ~np.isnan(h)
                bk = np.array([base[x].mean() if len(x) else np.nan for x in sets[st]])
                rel = np.nanmedian(h / bk)
                side = "-"
                row = dict(h_wid=float(np.nanmedian(h[W])) if W.any() else None,
                           h_ctl=float(np.nanmedian(h[~W])) if (~W).any() else None,
                           n_pos=int((h[ok] > 0).sum()), n=int(ok.sum()), rel=float(rel))
                if f"{bd}_read" in z.files:
                    hr = hvec(z[f"{bd}_read"].astype(np.float64), sets[st])
                    hw = hvec(z[f"{bd}_write"].astype(np.float64), sets[st])
                    row.update(h_read=float(np.nanmedian(hr)), h_write=float(np.nanmedian(hw)))
                    side = f"{row['h_read']:+.5f} / {row['h_write']:+.5f}"
                r["builds"][bd][st] = row
                hw_, hc_ = row["h_wid"], row["h_ctl"]
                md.append(f"| {bd} | {st} | {'-' if hw_ is None else f'{hw_:+.5f}'} / {'-' if hc_ is None else f'{hc_:+.5f}'} | "
                          f"{row['n_pos']} of {row['n']} | {rel:+.3f} | {side} |")
        # step 3: weights, every kept unit
        md += ["", "| Build | own-write share a: median / 10th pct | write cosine with A's: median | read cosine with A's: median |",
               "|---|---|---|---|"]
        for bd in BUILDS:
            Vn, _, Rn, _ = eff(np.load(os.path.join(LAD, f"n{n}", f"w_{bd}.npz")))
            a = np.array([np.linalg.lstsq(VA[k][:, None], Vn[k], rcond=None)[0][0] for k in K])
            wc = np.sum(unit_rows(Vn[K]) * unit_rows(VA[K]), 1)
            rc = np.sum(unit_rows(Rn[K]) * unit_rows(RA[K]), 1)
            r["builds"][bd]["weights"] = dict(a_med=float(np.median(a)), a_10=float(np.percentile(a, 10)),
                                              wcos=float(np.median(wc)), rcos=float(np.median(rc)))
            md.append(f"| {bd} | {np.median(a):.3f} / {np.percentile(a, 10):.3f} | {np.median(wc):.3f} | {np.median(rc):.3f} |")
        md.append("")
        res[str(n)] = r
    json.dump(res, open(os.path.join(JM, "ownjob_saved.json"), "w"), indent=1)
    open(os.path.join(JM, "ownjob_saved.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
