# scale: neurons=s3:69, A:children, A:block0-all; net=A, Z-ladder, C-ladder, W-ladder, L-ladder, R4-ladder, R-ladder; photos=saved; where=mac
"""Read the ladder (plan "Bigger cuts", 2026-11-24). Mac, numpy, saved results only.

Input: a folder with n11/, n64/, n192/, n384/ from oc_ladder.py (each: oc_ladder.json, kl_test_*.npy, strong.npy).
Output: recovery table, paired-bootstrap gap split (W-C, L-W, R4-L, R-R4), changed-job counts, P1-P5 verdicts.
Usage: python3 oc_ladder_read.py <folder>
"""
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1]
B = 2000
gen = np.random.default_rng(0)


def rec(k, k0, idx):
    return 1 - k[idx].mean() / k0[idx].mean()


def boot(ka, kb, k0, idx):
    """Paired bootstrap of rec(a) - rec(b)."""
    d = []
    for _ in range(B):
        j = idx[gen.integers(0, len(idx), len(idx))]
        d.append(rec(ka, k0, j) - rec(kb, k0, j))
    return rec(ka, k0, idx) - rec(kb, k0, idx), np.percentile(d, 2.5), np.percentile(d, 97.5)


rungs = sorted(int(d[1:]) for d in os.listdir(ROOT) if d.startswith("n") and
               os.path.exists(os.path.join(ROOT, d, "oc_ladder.json")))
table, gaps, summ = [], [], {}
for n in rungs:
    f = os.path.join(ROOT, f"n{n}")
    res = json.load(open(os.path.join(f, "oc_ladder.json")))
    strong = np.load(os.path.join(f, "strong.npy"))
    kl = {b: np.load(os.path.join(f, f"kl_test_{b}.npy")).astype(np.float64)
          for b in ["Z"] + list(res["builds"])}
    sets = {"test": np.arange(len(kl["Z"])), "strong": np.nonzero(strong)[0]}
    for b, row in res["builds"].items():
        table.append((n, b, row["held_kl"], row["rec_test"], row["rec_strong"], row.get("damping", ""),
                      row.get("lrs", ""), len(row.get("changed_job", [])) if "changed_job" in row else ""))
    s = {}
    for a, b_ in (("W", "C"), ("L", "W"), ("R4", "L"), ("R", "R4"), ("R4", "C"), ("WC", "C"), ("WC", "W"), ("R4C", "R4"), ("R4C", "WC")):
        if a in kl and b_ in kl:
            for sn, idx in sets.items():
                s[(a, b_, sn)] = boot(kl[a], kl[b_], kl["Z"], idx)
                gaps.append((n, f"{a} - {b_}", sn) + s[(a, b_, sn)])
    summ[n] = dict(res=res, s=s)

print("| Rung | Build | Held KL | Recovery test | Recovery strong | Damping | lrs | changed-job |")
print("|---|---|---|---|---|---|---|---|")
for r in table:
    print(f"| {r[0]} | {r[1]} | {r[2]:.5f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]} | {r[6]} | {r[7]} |")
print()
print("| Rung | Difference | Photos | Recovery diff | 95% CI |")
print("|---|---|---|---|---|")
for g in gaps:
    print(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]:+.3f} | [{g[4]:+.3f}, {g[5]:+.3f}] |")
print()
for n in rungs:
    r = summ[n]["res"]
    print(f"n{n}: nest order ok {r.get('nest_order_ok')}, W <= C {r.get('W_le_C')}, nest checks "
          f"{ {k: v['equal'] for k, v in r.get('nest_check', {}).items()} }, s3:69 skip change rank "
          f"{r.get('parent_skip_change', {}).get('rank69')}")
    for b in ("L", "R4", "R"):
        cj = r["builds"].get(b, {}).get("changed_job")
        if cj:
            top = sorted(cj, key=lambda u: -u["take"])[:8]
            print(f"  {b} changed-job: " + ", ".join(f"n{u['unit']} (self {u['self']:.2f}, take {u['take']:+.2f}, "
                                                    f"-> n{u['best_d']} {u['best_corr']:.2f})" for u in top))

# predictions
print()
if 11 in summ:
    s = summ[11]["s"]
    p1 = all(abs(s[("R4", "C", sn)][0]) <= .02 for sn in ("test", "strong"))
    print(f"P1 rung 11, |R4 - C| <= .02 on both sets: {p1} "
          f"({s[('R4', 'C', 'test')][0]:+.3f} test, {s[('R4', 'C', 'strong')][0]:+.3f} strong)")
gapRC = {n: summ[n]["s"][("R4", "C", "test")][0] for n in rungs}
print("P2 R4 - C by rung (test):", {n: round(v, 3) for n, v in gapRC.items()})
for n in rungs:
    s = summ[n]["s"]
    parts = {k: s[(a, b, "test")][0] for k, (a, b) in (("W-C", ("W", "C")), ("L-W", ("L", "W")), ("R4-L", ("R4", "L")))}
    print(f"P3 rung {n} gap parts (test):", {k: round(v, 3) for k, v in parts.items()})
for n in rungs:
    r = summ[n]["res"]["builds"]
    print(f"P4/P5 rung {n}: changed-job L {len(r['L'].get('changed_job', []))}, R4 {len(r['R4'].get('changed_job', []))}; "
          f"L - W test {summ[n]['s'][('L', 'W', 'test')][0]:+.3f}")
