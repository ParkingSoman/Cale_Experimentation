# scale: neurons=s3:69, A:children, s3:C1-groups; net=A; photos=saved; where=mac
"""C1 bar (LOG "C1: is coarsening local to a capability?"). Mac numpy, saved files only.
Inputs: results/oc_capability.npz (job output) and results/oc_dial.npz (set11 delete / morph KL per DEV photo).
"""
import os

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
c = np.load(os.path.join(R, "oc_capability.npz"))
d = np.load(os.path.join(R, "oc_dial.npz"))
P = 69
pk, wpos, kids = c["s3_peak"], c["wpos"], c["kids"]
assert np.array_equal(kids, d["order"][:11]), "children differ from oc_dial"
assert np.abs(pk[:, P] - d["dev_peak"]).max() < 1e-3, "parent peak differs from oc_dial"
de, mo = d["set11_delete"], d["set11_morph"]
weak = pk[:, P] < 1.5

w2 = wpos.copy(); w2[:, P] = -1
test = w2.argmax(1)
print("test channels (largest non-69 positive feeder of each child):", test.tolist())
med = np.median(wpos, 1, keepdims=True)
ok = (wpos <= med).all(0); ok[P] = False; ok[test] = False
cand = np.nonzero(ok)[0]
ctrl = np.random.default_rng(0).choice(cand, min(30, len(cand)), replace=False)
print(f"control candidates {len(cand)}; controls {sorted(ctrl.tolist())}")

base_left, base_del = mo[weak].mean(), de[weak].mean()


def row(ch):
    top = pk[:, ch] >= np.quantile(pk[:, ch], 0.95)
    g = top & weak
    return dict(n=int(g.sum()), coll=mo[g].mean() / base_left, dcoll=de[g].mean() / base_del,
                rec=1 - mo[g].mean() / de[g].mean(), ppk=float(pk[g, P].mean()))


T = {int(ch): row(ch) for ch in test}
C = {int(ch): row(ch) for ch in ctrl}
print(f"parent-weak DEV photos {weak.sum()}: KL left {base_left:.4f}, KL delete {base_del:.4f}, "
      f"recovery {1 - base_left / base_del:.2f}")
print("channel  n   collateral(left)  delete ratio  recovery  parent peak")
for name, S in (("TEST", T), ("CONTROL", C)):
    print(name)
    for ch, r in S.items():
        print(f"  s3:{ch:<4d} {r['n']:3d}   {r['coll']:.2f}   {r['dcoll']:.2f}   {r['rec']:.2f}   {r['ppk']:.2f}")
tc = np.median([r["coll"] for r in T.values()])
cc = np.array([r["coll"] for r in C.values()]) if C else np.array([np.nan])
print(f"C1a: median test collateral {tc:.2f} (bar >= 1.5) -> {'PASS' if tc >= 1.5 else 'FAIL'}")
print(f"C1b: controls median {np.median(cc):.2f}, 95th pct {np.percentile(cc, 95):.2f}; test median {tc:.2f} "
      f"-> {'PASS' if tc > np.percentile(cc, 95) else 'FAIL'}")
print(f"delete-ratio medians: test {np.median([r['dcoll'] for r in T.values()]):.2f}, "
      f"control {np.median([r['dcoll'] for r in C.values()]):.2f}")

# ADDED AFTER THE RUN (the pre-registered control set was empty: no channel is at or below the median weight into
# all 11 children). Description only, not a bar: the 30 channels with the smallest total positive weight into the 11.
tot = wpos.sum(0); tot[P] = np.inf; tot[test] = np.inf
low = np.argsort(tot)[:30]
CL = {int(ch): row(ch) for ch in low}
cl = np.array([r["coll"] for r in CL.values()])
print(f"post-hoc control (30 lowest-weight channels): collateral median {np.median(cl):.2f}, "
      f"5th-95th pct {np.percentile(cl, 5):.2f}-{np.percentile(cl, 95):.2f}; delete-ratio median "
      f"{np.median([r['dcoll'] for r in CL.values()]):.2f}; recovery median {np.median([r['rec'] for r in CL.values()]):.2f}")
print(f"test recovery median {np.median([r['rec'] for r in T.values()]):.2f}")
