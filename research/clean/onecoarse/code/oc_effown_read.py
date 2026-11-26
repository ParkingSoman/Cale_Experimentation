"""Score V1'' and W1'' (LOG). Mac, numpy only. Input: results/oc_effown.npz (own-context unit effects)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
RN = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
E = np.load(os.path.join(R, "oc_effown.npz"))
S1 = np.load(os.path.join(R, "oc_step1.npz"))
F = {"A": S1["A_pos"].astype(np.float32), "s1": np.load(os.path.join(RN, "read_B128aug_s1.npz"))["pos0"].astype(np.float32)}
rng = np.random.default_rng(0)
zc = lambda X: (X - X.mean(1, keepdims=True)) / (np.linalg.norm(X - X.mean(1, keepdims=True), axis=1, keepdims=True) + 1e-12)
zcol = lambda X: (X - X.mean(0)) / (np.linalg.norm(X - X.mean(0), axis=0) + 1e-12)
print("V3 own-context half agreement median: A", round(float(np.median((zc(E["A_h0"]) * zc(E["A_h1"])).sum(1))), 3),
      "| s1", round(float(np.median((zc(E["s1_h0"]) * zc(E["s1_h1"])).sum(1))), 3))
EA2 = np.load(os.path.join(R, "oc_effall.npz"))
print("same A neuron, own context vs centre cell: median corr",
      round(float(np.median((zc(E["A_mean"]) * zc(EA2["A_mean"])).sum(1))), 2))


def test(src, dst, bar, tag, top=5):
    Ce = zc(E[f"{src}_mean"]) @ zc(E[f"{dst}_mean"]).T
    Cf = zcol(F[src]).T @ zcol(F[dst])
    best = Ce.argmax(1); topf = np.argsort(-Cf, 1)[:, :top]
    hit = np.mean([best[i] in topf[i] for i in range(len(best))])
    fm = Cf[np.arange(len(best)), best]
    q = np.quantile([np.median(Cf[np.arange(len(best)), rng.integers(0, Cf.shape[1], len(best))]) for _ in range(1000)], .99)
    print(f"{tag} {src}->{dst}: hit {hit:.2f} (bar {bar}, chance {top / Cf.shape[1]:.3f}); matched firing corr {np.median(fm):.2f} "
          f"(random 99th {q:.2f}) -> {'PASS' if hit >= bar and np.median(fm) > q else 'FAIL'}")
    return Ce


Ce = test("s1", "A", .20, "V1''b")
test("A", "s1", .20, "V1''c")
A = E["A_mean"]; n = E["s1_mean"][101]
c = lambda a, b: float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])
sets = {"5": [305, 297, 163, 97, 23], "7": [305, 297, 163, 97, 23, 108, 237], "9": [305, 297, 163, 97, 23, 108, 237, 275, 81],
        "17": [23, 37, 71, 81, 97, 103, 108, 123, 163, 237, 252, 275, 297, 298, 305, 385, 496]}
for k, m in sets.items():
    v = c(n, A[m].sum(0)); r = [c(n, A[rng.choice(512, len(m), replace=False)].sum(0)) for _ in range(1000)]
    print(f"W1'' n101 vs SUM of {k} children (own context): {v:.2f} | random 99th {np.quantile(r, .99):.2f} -> {'PASS' if v > np.quantile(r, .99) else 'FAIL'}")
S3 = list(np.load(os.path.join(R, "oc_closure.npz"))["S3_members"])
r = [c(n, A[rng.choice(512, 25, replace=False)].sum(0)) for _ in range(1000)]
print(f"   (record) n101 vs SUM of co-firing S3: {c(n, A[S3].sum(0)):.2f} | random 99th {np.quantile(r, .99):.2f}")
o = np.argsort(-Ce[101])[:8]
CI = json.load(open(os.path.join(HERE, "..", "..", "neuron", "results", "read", "imagenet_class_index.json")))
N = [CI[str(i)][1] for i in range(1000)]
print("n101 (own context) up:", ", ".join(N[i] for i in np.argsort(-n)[:8]))
print("n101's top A effect matches (own context):", [(int(j), round(float(Ce[101, j]), 2)) for j in o])
kids = [305, 297, 163, 97, 23, 108, 237, 275, 81]
rk = np.argsort(-Ce[101]); print("children's ranks:", {k: int(np.where(rk == k)[0][0]) for k in kids})
for k in (305, 297, 97, 23, 108):
    print(f"   A n{k} (own context) up: {', '.join(N[i] for i in np.argsort(-A[k])[:6])}")
