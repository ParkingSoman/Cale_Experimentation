"""Score V1' (LOG "Writes: a translation through the readers"). Mac, numpy only. Input: results/oc_effall.npz.
Effect match = correlation of mean unit effects (1000 classes). Firing match = correlation of per-photo peaks over
the 19,000 held + test photos (never used by the translation).
V1'a seed 1 -> seed 0; V1'b seed 1 -> A; V1'c A -> seed 1. Then W1 again: n101's effect matches among all 512 A.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
RN = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
E = np.load(os.path.join(R, "oc_effall.npz"))
S1 = np.load(os.path.join(R, "oc_step1.npz"))
P = {"A": S1["A_pos"].astype(np.float32),
     "s0": np.load(os.path.join(RN, "read_B128aug_s0.npz"))["pos0"].astype(np.float32),
     "s1": np.load(os.path.join(RN, "read_B128aug_s1.npz"))["pos0"].astype(np.float32)}
rng = np.random.default_rng(0)
CI = json.load(open(os.path.join(HERE, "..", "..", "neuron", "results", "read", "imagenet_class_index.json")))
N = [CI[str(i)][1] for i in range(1000)]


def zc(X):
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def zcol(X):
    X = X - X.mean(0); return X / (np.linalg.norm(X, axis=0) + 1e-12)


EF = {k: zc(E[f"{k}_mean"]) for k in ("A", "s0", "s1")}
FZ = {k: zcol(v) for k, v in P.items()}
print("V3 (real check): half-mean agreement median per net:",
      {k: round(float(np.median((zc(E[f'{k}_h0']) * zc(E[f'{k}_h1'])).sum(1))), 3) for k in ("A", "s0", "s1")})


def test(src, dst, top, bar, tag):
    Ce = EF[src] @ EF[dst].T                       # effect corr, src x dst
    Cf = FZ[src].T @ FZ[dst]                       # firing corr, src x dst
    best = Ce.argmax(1)
    topf = np.argsort(-Cf, 1)[:, :top]
    hit = np.array([best[i] in topf[i] for i in range(len(best))])
    fm = Cf[np.arange(len(best)), best]
    rnd = [np.median(Cf[np.arange(len(best)), rng.integers(0, Cf.shape[1], len(best))]) for _ in range(1000)]
    q = np.quantile(rnd, .99)
    ok = hit.mean() >= bar and np.median(fm) > q
    chance = top / Cf.shape[1]
    print(f"{tag} {src}->{dst}: effect-best match is in the top {top} by firing for {hit.mean():.2f} (bar {bar}, chance {chance:.2f}); "
          f"median firing corr of matched pairs {np.median(fm):.2f} vs random 99th {q:.2f} -> {'PASS' if ok else 'FAIL'}")
    return Ce, Cf, ok


Ce10, Cf10, a = test("s1", "s0", 5, .30, "V1'a")
Ce1A, Cf1A, b = test("s1", "A", 5, .20, "V1'b")
CeA1, CfA1, c = test("A", "s1", 5, .20, "V1'c")

# effect vs firing over all pairs (how much the two measures agree in general)
print("\nall pairs: corr(effect match, firing match): s1-s0", round(float(np.corrcoef(Ce10.ravel(), Cf10.ravel())[0, 1]), 3),
      "| s1-A", round(float(np.corrcoef(Ce1A.ravel(), Cf1A.ravel())[0, 1]), 3))

for nm, i, Ce, Cf, dst in (("B s1 n101 -> s0", 101, Ce10, Cf10, "s0"), ("B s1 n101 -> A", 101, Ce1A, Cf1A, "A")):
    o = np.argsort(-Ce[i])[:10]
    print(f"\n{nm}: top effect matches (effect corr / firing corr):")
    for j in o:
        v = E[f"{dst}_mean"][j]
        print(f"   n{j}: {Ce[i, j]:.2f} / {Cf[i, j]:.2f}  up: {', '.join(N[k] for k in np.argsort(-v)[:5])}")
kids = [305, 297, 163, 97, 23, 108, 237, 275, 81]
rk = np.argsort(-Ce1A[101]); pos = {int(j): int(np.where(rk == j)[0][0]) for j in kids}
print("\nW1 again: the children's rank among 512 A neurons by effect match with n101:", pos)
print("   effect corr:", {k: round(float(Ce1A[101, k]), 2) for k in kids}, "| A 99th pct", round(float(np.quantile(Ce1A[101], .99)), 2))
np.savez(os.path.join(R, "oc_effall_read.npz"), Ce10=Ce10, Cf10=Cf10, Ce1A=Ce1A, Cf1A=Cf1A)
