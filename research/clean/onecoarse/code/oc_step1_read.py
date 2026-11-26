"""Read oc_step1.npz (numpy only, on the Mac). Prints the children by edge and the coarse candidates.
Usage: python3 oc_step1_read.py [n_children]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "..", "results", "oc_step1.npz"))
CI = json.load(open(os.path.join(HERE, "..", "..", "neuron", "results", "read", "imagenet_class_index.json")))
NAMES = [CI[str(i)][1] for i in range(1000)]
f = lambda k: D[k].astype(np.float32)
ppk = f("ppk")
top = np.argsort(-ppk)[:200]
print("parent s3:69 top-200 classes:", [NAMES[c] for c, _ in sorted(
    zip(*np.unique(D["lab"][top], return_counts=True)), key=lambda t: -t[1])[:12]])


def z(v):
    return (v - v.mean(0)) / (v.std(0) + 1e-6)


def corr(a, b):
    return (z(a) * z(b)[:, None]).mean(0) if a.ndim == 2 else float((z(a) * z(b)).mean())


def table(net, rows):
    c, pre, pos = f(f"{net}_c69"), f(f"{net}_pre"), f(f"{net}_pos")
    edge = c[top].mean(0)
    share = (c[top] / (np.abs(pre[top]) + np.abs(c[top]) + 1e-6)).mean(0)
    fire = (pre[top] > 0).mean(0)
    cp = corr(pos, ppk)
    o = np.argsort(-edge)
    print(f"\n{net}: edge from s3:69 on its top-200 photos (rank, neuron, edge, fires there, corr(pos, parent peak))")
    for r, j in enumerate(o[:rows]):
        tj = np.argsort(-pos[:, j])[:200]
        lab = [NAMES[c] for c, _ in sorted(zip(*np.unique(D["lab"][tj], return_counts=True)), key=lambda t: -t[1])[:4]]
        print(f"{r:3d} n{j:<4d} edge {edge[j]:+.3f}  fire {fire[j]:.2f}  corr {cp[j]:+.2f}  {lab}")
    print("  most negative edges:", [(f"n{j}", round(float(edge[j]), 3)) for j in o[-6:][::-1]])
    print("  edge quantiles 50/90/99:", np.round(np.quantile(edge, [.5, .9, .99]), 3))
    return edge, pos


eA, posA = table("A", 25)
nch = int(sys.argv[1]) if len(sys.argv) > 1 else 8
kids = np.argsort(-eA)[:nch]
print("\nchildren (top", nch, "by edge):", [f"n{j}" for j in kids])
kid_mean = z(posA[:, kids]).mean(1)
kid_max = z(posA[:, kids]).max(1)
union = set()
for j in kids:
    union |= set(np.argsort(-posA[:, j])[:200].tolist())
ptop = set(top.tolist())
for net in ("T0", "B1"):
    e, pos = table(net, 10)
    cp, cm, cx = corr(pos, ppk), corr(pos, kid_mean), corr(pos, kid_max)
    score = cp + cm
    print(f"  {net} best by corr(parent) + corr(children mean):")
    for j in np.argsort(-score)[:6]:
        tj = set(np.argsort(-pos[:, j])[:200].tolist())
        print(f"   n{j:<4d} edge {e[j]:+.3f} corr parent {cp[j]:+.2f} kids-mean {cm[j]:+.2f} kids-max {cx[j]:+.2f}"
              f"  top200 in kids union {len(tj & union) / 200:.2f}  in parent top200 {len(tj & ptop) / 200:.2f}")
print(f"\nkids union size {len(union)}; random 200-set expected in union {len(union) / len(ppk):.2f}")
