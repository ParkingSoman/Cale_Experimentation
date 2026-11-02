"""Print battery summary for neurons given as B,N args. max = frac of natural max; min = frac of |natural min|."""
import sys, os, json, numpy as np
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery.json"))); L = D["labels"]
def row(k, pref):
    M, N = np.array(D["resp"][k]["max"]), np.array(D["resp"][k]["min"])
    ix = [i for i, l in enumerate(L) if l.startswith(pref)]
    return " ".join(f"{L[i].split('-')[-1]}:{M[i]:+.2f}/{N[i]:+.2f}" for i in ix)
for k in sys.argv[1:]:
    print(f"==== {k}  (max/min)")
    M, N = np.array(D["resp"][k]["max"]), np.array(D["resp"][k]["min"])
    o = np.argsort(-M); print(" top:", ", ".join(f"{L[i]} {M[i]:.2f}" for i in o[:14]))
    o = np.argsort(N); print(" bottom(min):", ", ".join(f"{L[i]} {N[i]:.2f}" for i in o[:10]))
    for p in ("flat", "ramp-soft", "ramp-steep", "noise", "check", "gnoise", "lineend", "corner"):
        print(f" {p}:", row(k, p))
    for p in ("dline-w1", "dline-w3", "lline-w1", "edgeA", "edgeB", "grat4", "grat8", "grat16", "grat32", "sine32"):
        ix = [i for i, l in enumerate(L) if l.startswith(p + "-")]
        print(f" {p:8s} max:", " ".join(f"{M[i]:+.2f}" for i in ix), " | min:", " ".join(f"{N[i]:+.2f}" for i in ix))
    for nm in ("cross", "Tjunc", "dot12", "dot6", "ring40", "bar8-vert", "bar8-horz", "text-dark", "text-light"):
        i = L.index(nm); print(f" {nm} {M[i]:+.2f}/{N[i]:+.2f}", end="")
    print()
