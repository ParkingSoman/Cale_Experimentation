"""Score "Writes: a translation through the readers" (LOG). Mac, numpy only. Input: results/oc_effect.npz.
V1 per facet: mean per-photo correlation of n101's unit effect with each B seed-0 neuron's; named partners
   n122 (wheel facet f2) and n105 (nose and face facets f0, f1) against the other 127 (99th pct).
V2 the same after removing, per photo, the mean unit effect over all 138 scored neurons.
V3 half-mean agreement of each neuron's unit effect (even vs odd photos).
W1 per facet: n101 against each of the 9 A children (seed-0 neurons as a reference distribution).
W2 photos with 2+ children in their own strong half: firing-weighted n101 against the SUM of the children's
   firing-weighted effects, and against the best single child.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
E = np.load(os.path.join(R, "oc_effect.npz"))
I = np.load(os.path.join(R, "oc_effect_in.npz"))
S1 = np.load(os.path.join(R, "oc_step1.npz"))
assert (E["rows"] == I["rows"]).all()
e1 = E["s1_eff"].astype(np.float32)[:, 0]          # n x 1000
e0 = E["s0_eff"].astype(np.float32)                # n x 128 x 1000
eA = E["A_eff"].astype(np.float32)                 # n x 9 x 1000
kids = [int(k) for k in I["kids"]]
facet = I["facet"]
NAMES = {0: "f0 close nose", 1: "f1 whole face", 2: "f2 wheel hub", 3: "f3 coil"}
PART = {0: 105, 1: 105, 2: 122}


def pcorr(a, B):
    """per-photo correlation of a (n x 1000) with each of B (n x J x 1000) -> n x J"""
    a = a - a.mean(1, keepdims=True); B = B - B.mean(2, keepdims=True)
    return (B * a[:, None]).sum(2) / (np.linalg.norm(B, axis=2) * np.linalg.norm(a, axis=1)[:, None] + 1e-12)


def v1(a, B, tag):
    C = pcorr(a, B)
    ok = True
    for f in range(4):
        m = facet == f
        mc = C[m].mean(0); o = np.argsort(-mc)
        line = f"  {NAMES[f]} ({m.sum()} photos): top seed-0 " + " ".join(f"n{j}:{mc[j]:.2f}" for j in o[:5])
        if f in PART:
            p = PART[f]; others = np.delete(mc, p)
            q = np.quantile(others, .99); passed = mc[p] > q
            ok &= passed
            line += f" | partner n{p} {mc[p]:.2f} rank {int(np.where(o == p)[0][0])} vs others' 99th {q:.2f} -> {'PASS' if passed else 'FAIL'}"
        print(line)
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
    return ok


print("V1 (unit effect, n101 vs B seed-0 neurons)")
ok1 = v1(e1, e0, "V1")
allE = np.concatenate([e1[:, None], e0, eA], 1)
mu = allE.mean(1)
print("\nV2 (per-photo mean effect over 138 neurons removed)")
ok2 = v1(e1 - mu, e0 - mu[:, None], "V2")

print("\nV3 (half-mean agreement)")
h0, h1 = np.arange(0, len(e1), 2), np.arange(1, len(e1), 2)
hc = lambda X: float(np.corrcoef(X[h0].mean(0), X[h1].mean(0))[0, 1])
v3 = {"n101": hc(e1), "s0 n122": hc(e0[:, 122]), "s0 n105": hc(e0[:, 105])}
v3.update({f"A n{k}": hc(eA[:, i]) for i, k in enumerate(kids)})
allv = [hc(e0[:, j]) for j in range(128)]
print("  " + " ".join(f"{k}:{v:.2f}" for k, v in v3.items()) + f" | seed-0 median {np.median(allv):.2f}, min {np.min(allv):.2f}")
ok3 = all(v >= .8 for v in v3.values())
print(f"  V3: {'PASS' if ok3 else 'FAIL'}")

CI = json.load(open(os.path.join(HERE, "..", "..", "neuron", "results", "read", "imagenet_class_index.json")))
N = [CI[str(i)][1] for i in range(1000)]
print("\nwhat each unit effect pushes up (mean over n101's photos):")
for nm, v in [("B s1 n101", e1.mean(0)), ("B s0 n122", e0[:, 122].mean(0)), ("B s0 n105", e0[:, 105].mean(0))] + [(f"A n{k}", eA[:, i].mean(0)) for i, k in enumerate(kids)]:
    print(f"  {nm:10s} up: {', '.join(N[i] for i in np.argsort(-v)[:6])}")

print("\nW1 (n101 vs the A children, per facet; seed-0 distribution as reference)")
CA = pcorr(e1, eA); C0 = pcorr(e1, e0)
for f in range(4):
    m = facet == f
    mc = CA[m].mean(0); o = np.argsort(-mc); ref = np.quantile(C0[m].mean(0), [.5, .99])
    print(f"  {NAMES[f]}: " + " ".join(f"n{kids[i]}:{mc[i]:.2f}" for i in o) + f" | seed-0 median {ref[0]:.2f}, 99th {ref[1]:.2f}")

print("\nW2 (sum vs OR, firing-weighted, photos with 2+ children strong)")
ap = S1["A_pos"].astype(np.float32); mx = ap.max(0)
strong = ap[I["rows"]][:, kids] >= .5 * mx[kids]
two = strong.sum(1) >= 2
fwA = eA * (E["A_act"].astype(np.float32) / I["step_A"][None, :])[:, :, None]
fw1 = e1 * (E["s1_act"].astype(np.float32)[:, 0] / I["step_s1"][0])[:, None]
cs = pcorr(fw1[two], fwA[two].sum(1)[:, None])[:, 0]
ci = pcorr(fw1[two], fwA[two])
best = ci.max(1)
print(f"  {two.sum()} photos: corr with SUM of children median {np.median(cs):.2f}; with best single child median {np.median(best):.2f}; "
      f"sum better on {(cs > best).mean():.2f} of photos")
np.savez(os.path.join(R, "oc_effect_read.npz"), C0=C0, CA=CA, v3=np.array(list(v3.values())), two=two, cs=cs, best=best)
