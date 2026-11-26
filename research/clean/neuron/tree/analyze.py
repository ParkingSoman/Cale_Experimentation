"""N5 step 2 analysis: tables and paired bootstrap intervals from results/tree/shrink_<group>_f<f>.npz."""
import json
import os
import sys

import numpy as np

from common import OUT

f = sys.argv[1] if len(sys.argv) > 1 else "0.25"


def ci(d, n=2000):
    bs = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


out = {}
for g in ("dogs", "birds", "vehicles"):
    p = os.path.join(OUT, f"shrink_{g}_f{f}.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p)
    keys = [k.split("|") for k in z.files]
    vs = sorted(set(k[0] for k in keys), key=lambda v: list(dict.fromkeys(k[0] for k in keys)).index(v))
    depths = sorted(set(int(k[1]) for k in keys))
    dmax = depths[-1]
    print(f"\n== {g} (f = {f}), depths {depths}")
    print(f"{'variant':12s} " + "  ".join(f"grp@{d:<3d}" for d in depths) + f"   other@{dmax}  coarse@{dmax}")
    tab = {}
    for v in vs:
        row = [z[f"{v}|{d}|group"].mean() for d in depths]
        tab[v] = dict(group=[float(x) for x in row], other=float(z[f"{v}|{dmax}|other"].mean()),
                      coarse=float(z[f"{v}|{dmax}|group_coarse"].mean()))
        print(f"{v:12s} " + "  ".join(f"{x:.5f}" for x in row) +
              f"   {tab[v]['other']:.5f}   {tab[v]['coarse']:.5f}")
    G = lambda v, d=dmax: z[f"{v}|{d}|group"]
    pairs = {}
    for a, b in (("full", "bias"), ("full", "rewrite_ind"), ("full", "rewrite_seq"), ("parent", "full"),
                 ("ctl_absorb", "full"), ("drop_goal", "drop_all"), ("full", "drop_goal")):
        if f"{a}|{dmax}|group" in z.files and f"{b}|{dmax}|group" in z.files:
            pairs[f"{a} - {b}"] = dict(mean=round(float((G(a) - G(b)).mean()), 5), ci95=ci(G(a) - G(b)))
    if "full" in vs and "rewrite_ind" in vs:
        gf = G("full") - G("full", depths[0])
        gr = G("rewrite_ind") - G("rewrite_ind", depths[0])
        pairs["growth full - growth rewrite_ind"] = dict(mean=round(float((gf - gr).mean()), 5), ci95=ci(gf - gr))
    for k, v in pairs.items():
        print(f"  {k:36s} {v['mean']:+.5f}  95% {v['ci95']}")
    # share of the drop cost taken back, relative to drop_goal
    if "drop_goal" in tab:
        base = tab["drop_goal"]["group"][-1]
        print("  taken back vs drop_goal: " + ", ".join(
            f"{v} {100 * (1 - tab[v]['group'][-1] / base):.0f}%" for v in vs if v not in ("drop_goal", "ctl_tree")))
    out[g] = dict(table=tab, pairs=pairs, depths=depths)
json.dump(out, open(os.path.join(OUT, f"shrink_summary_f{f}.json"), "w"), indent=1)
