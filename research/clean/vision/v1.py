"""V1: smaller = coarser? Top-100 analysis images per neuron; superclass purity, fine spread, breadth."""
import numpy as np
from vlib import *
import sys
POOL = sys.argv[1] if len(sys.argv) > 1 else "max"

d = load_data(); ana, sealed = split_test(); X = d["xte"][ana]; yf, yc = d["yte"][ana], d["cte"][ana]
out = {}
nets = {"S": (32, 128), "L": (128, 512), "L2": (128, 512), "LN": (128, 128)}
for name, (C, H) in list(nets.items()) + [("null_S", (32, 128)), ("null_L", (128, 512))]:
    if name.startswith("null"):
        mx.random.seed(99); m = Net(C, H); mx.eval(m.parameters())
    else:
        m = load_net(name, C, H, 25)
    out[name] = {"sealed_acc": {k: v for k, v in acc(m, d, sealed).items() if not k.endswith("correct")}}
    for b in (3, 5):
        A = neuron_max(m, X, b, POOL)
        top = np.argsort(-A, 0)[:100].T                     # (H, 100)
        alive = A.std(0) > 1e-6
        pur, spread, breadth = [], [], []
        for t in top[alive]:
            c = yc[t]; mc = np.bincount(c, minlength=20).argmax()
            pur.append((c == mc).mean()); spread.append(len(set(yf[t][c == mc]))); breadth.append(len(set(yf[t])))
        out[name][f"b{b}"] = {"purity_median": float(np.median(pur)), "spread_median": float(np.median(spread)),
                              "spread_mean": float(np.mean(spread)), "breadth_median": float(np.median(breadth)),
                              "alive": int(alive.sum())}
    print(name, out[name], flush=True)
s, l, ns, nl = out["S"]["b5"], out["L"]["b5"], out["null_S"]["b5"], out["null_L"]["b5"]
out["pass"] = bool(s["spread_median"] > l["spread_median"] and s["purity_median"] >= l["purity_median"] - .05
                   and s["purity_median"] >= ns["purity_median"] + .10 and l["purity_median"] >= nl["purity_median"] + .10)
print("V1 pass", out["pass"]); save("v1" if POOL == "max" else f"v1-{POOL}", out)
