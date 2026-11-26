# scale: neurons=A:block0-all; net=A, L-ladder; photos=saved; where=mac
# rung: Q=plan "Does morphing hurt a kept neuron's own job?", read step; N=saved per-unit KL + weights; EST=1 min Mac; PERMIT=Shonu approved plan 2026-11-25
"""Reader for oc_vs_ownjob.py (Mac, numpy). Input: results/oc_vs/ownjob/n<cut>.npz (pulled from the cluster).

Per kept unit j, build b, photo set S (own, clean = own with D quiet, part = the partner's photos):
  harm  h = (KL base - KL j swapped back) / KL(A with j deleted)        on own / clean photos (a share of j's own job)
  split h_read (A's read side only back), h_write (A's write only back)
  gain  t = KL j swapped back - KL base                                  on the partner's photos (nats)
Groups: widened = some deleted partner in the top 5% of both j's write-change list and read-change list in L, and
a >= .8 (section 8.5 rule, oc_vs_widen_ctl.py). Control = the other kept units.
Bars (plan, before the run): free if widened median h <= .02 and 90th pct <= .10; costs if widened median h >= .10
and above control (bootstrap CI of the median difference excludes 0).
Output: results/oc_vs/ownjob.md, ownjob_read.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "vecread"))
import vecread as vr  # noqa: E402
from oc_vs_read import R18, LAD, OUT, eff  # noqa: E402

SRC = os.environ.get("OWNJOB_SRC", os.path.join(OUT, "ownjob"))
BUILDS = ("C", "WC", "L", "R4C")
NAMED = {11: [5], 64: [339], 192: [430], 384: [430, 76, 213]}
RNG = np.random.default_rng(3)


def unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-30)


def widened(n, VA, RA, pat):
    d = os.path.join(LAD, f"n{n}")
    jt = np.load(os.path.join(d, "jobtest_L.npz"))
    D, K = jt["D"], jt["K"]
    Vn, _, Rn, _ = eff(np.load(os.path.join(d, "w_L.npz")))
    dW = unit_rows(Vn[K] - VA[K])
    dRraw = Rn[K] - RA[K]
    moved = np.linalg.norm(dRraw, axis=1) > 1e-9
    CW = dW @ unit_rows(VA[D]).T
    CR = unit_rows(pat(dRraw)) @ unit_rows(pat(RA))[D].T
    a = np.array([np.linalg.lstsq(VA[k][:, None], Vn[k], rcond=None)[0][0] for k in K])
    q = max(1, int(np.ceil(.05 * len(D))))
    topW, topR = np.argsort(-CW, 1)[:, :q], np.argsort(-CR, 1)[:, :q]
    hit = np.array([len(set(topW[i]) & set(topR[i])) > 0 for i in range(len(K))])
    return dict(zip(K.tolist(), (hit & (a >= .8) & moved).tolist())), dict(zip(K.tolist(), a.tolist()))


def med_ci(x, y, B=2000):
    """Median of x minus median of y, bootstrap over units."""
    if len(x) == 0 or len(y) == 0:
        return float("nan"), (float("nan"), float("nan"))
    d = [np.median(RNG.choice(x, len(x))) - np.median(RNG.choice(y, len(y))) for _ in range(B)]
    return float(np.median(x) - np.median(y)), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def main():
    dk, dic = vr.Dictionary("s3k"), vr.Dictionary("s4b0")
    pat = lambda R: dk.metric(np.atleast_2d(R) @ dk.cov.T, True)
    A = np.load(os.path.join(R18, "w_A.npz"))
    VA, SA, RA, _ = eff(A)
    md = ["# Does morphing hurt a kept neuron's own job? (oc_vs_ownjob.py, read by oc_vs_ownjob_read.py)", "",
          "h = share of j's own job lost (KL that swapping j back removes, over KL of deleting j in A), on j's own top "
          "50 test photos. clean = the same, on photos where D is quiet. t = KL that j's morph removes on its "
          "partner's top 50 photos (nats). Widened = section 8.5 rule in L.", ""]
    res = {}
    for n in (11, 64, 192, 384):
        f = os.path.join(SRC, f"n{n}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        units = z["units"]
        wid, aa = widened(n, VA, RA, pat)
        W = np.array([wid[int(u)] for u in units])
        size = z["A_Adel_own"]
        sizec = z["A_Adel_clean"]
        big = size >= np.median(size)
        md += [f"## Cut {n}: {len(units)} kept units, {int(W.sum())} widened", "",
               f"D's summed firing on own photos / clean photos, as a multiple of its median over test photos: "
               f"{np.median(z['dfire_own'] / np.median(z['dfire'])):.2f}"
               f" / {np.median(z['dfire_clean'] / np.median(z['dfire'])):.2f}. "
               f"Own-job size KL(A, j deleted) on own photos: median {np.median(size):.2e}, 90th {np.percentile(size, 90):.2e}.",
               "",
               "| Build | Group | h own: median / 90th | h clean: median | h read only / write only (median) | "
               "pooled h own (sum/sum) | t on partner photos: sum (nats/photo) | own-job loss: sum (nats/photo) |",
               "|---|---|---|---|---|---|---|---|"]
        rc = {}
        for b in BUILDS:
            if f"{b}_base_own" not in z.files:
                continue
            base, full = z[f"{b}_base_own"], z[f"{b}_full_own"]
            loss = base - full
            h = loss / np.maximum(size, 1e-12)
            hc = (z[f"{b}_base_clean"] - z[f"{b}_full_clean"]) / np.maximum(sizec, 1e-12)
            hr = (base - z[f"{b}_read_own"]) / np.maximum(size, 1e-12)
            hw = (base - z[f"{b}_write_own"]) / np.maximum(size, 1e-12)
            t = z[f"{b}_full_part"] - z[f"{b}_base_part"]
            rb = {}
            for g, mk in (("widened", W), ("control", ~W), ("widened, big job", W & big), ("control, big job", ~W & big)):
                if mk.sum() == 0:
                    continue
                rb[g] = dict(n=int(mk.sum()), h_med=float(np.median(h[mk])), h_90=float(np.percentile(h[mk], 90)),
                             hc_med=float(np.median(hc[mk])), hr_med=float(np.median(hr[mk])),
                             hw_med=float(np.median(hw[mk])), h_pool=float(loss[mk].sum() / size[mk].sum()),
                             t_sum=float(t[mk].sum()), loss_sum=float(loss[mk].sum()))
                md.append(f"| {b} | {g} ({mk.sum()}) | {rb[g]['h_med']:.3f} / {rb[g]['h_90']:.3f} | {rb[g]['hc_med']:.3f} | "
                          f"{rb[g]['hr_med']:.3f} / {rb[g]['hw_med']:.3f} | {rb[g]['h_pool']:.3f} | "
                          f"{rb[g]['t_sum']:+.4f} | {rb[g]['loss_sum']:+.4f} |")
            dmed, ci = med_ci(h[W], h[~W])
            free = rb.get("widened", {}).get("h_med", 9) <= .02 and rb.get("widened", {}).get("h_90", 9) <= .10
            cost = rb.get("widened", {}).get("h_med", -9) >= .10 and ci[0] > 0
            rb["widened_minus_control_median"] = dict(d=dmed, ci=ci)
            rb["verdict"] = "free" if free else ("costs the own job" if cost else "neither bar")
            rb["per_unit"] = dict(units=units.tolist(), h=h.tolist(), hr=hr.tolist(), hw=hw.tolist(), t=t.tolist(),
                                  widened=W.tolist())
            rc[b] = rb
        md += ["", "| Build | widened - control, median h [95% CI] | Verdict |", "|---|---|---|"]
        for b, rb in rc.items():
            d_, ci = rb["widened_minus_control_median"]["d"], rb["widened_minus_control_median"]["ci"]
            md.append(f"| {b} | {d_:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {rb['verdict']} |")
        md += ["", "Named units:", "", "| Unit -> partner | Build | widened | a | h own | h read / write | h clean | t (nats/photo) | own-job size |",
               "|---|---|---|---|---|---|---|---|---|"]
        for j in NAMED[n]:
            if j not in units:
                continue
            i = list(units).index(j)
            for b, rb in rc.items():
                pu = rb["per_unit"]
                md.append(f"| n{j} -> n{int(z['partner'][i])} | {b} | {bool(W[i])} | {aa[j]:.2f} | {pu['h'][i]:.3f} | "
                          f"{pu['hr'][i]:.3f} / {pu['hw'][i]:.3f} | "
                          f"{(z[f'{b}_base_clean'][i] - z[f'{b}_full_clean'][i]) / max(sizec[i], 1e-12):.3f} | "
                          f"{pu['t'][i]:+.4f} | {size[i]:.2e} |")
        # the 3 most harmed units with a big own job, in L
        if "L" in rc:
            h = np.array(rc["L"]["per_unit"]["h"])
            cand = np.nonzero(big)[0]
            top = cand[np.argsort(-h[cand])[:3]]
            md += ["", "Most harmed units in L (own job at least the median size), own write read with vecread (s4b0):", ""]
            for i in top:
                j = int(units[i])
                c = vr.read_vector(dic, SA[j], "write")
                md.append(f"- n{j}: h {h[i]:.3f}, widened {bool(W[i])}, partner n{int(z['partner'][i])}; own write: " +
                          ", ".join(x for x, _, _ in c["up"][:3]) + " vs " + ", ".join(x for x, _, _ in c["down"][:2]) +
                          f" (leftover {c['leftover']:.2f})")
        md.append("")
        res[str(n)] = {b: {k: v for k, v in rb.items() if k != "per_unit"} for b, rb in rc.items()}
    open(os.path.join(OUT, "ownjob" + os.environ.get("OWNJOB_TAG", "") + ".md"), "w").write("\n".join(md))
    json.dump(res, open(os.path.join(OUT, "ownjob_read" + os.environ.get("OWNJOB_TAG", "") + ".json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
