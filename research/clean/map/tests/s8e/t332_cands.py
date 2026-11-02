"""2 n332 candidates on faint stimuli (contrast invariance + vertical vs horizontal). Value = map sd (phase-free
strength) and max, as a fraction of each neuron's natural max."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from early_lib import *
H = os.path.dirname(os.path.abspath(__file__)); DEEP = os.path.join(H, "..", "..", "deep")
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))["2_332"]
T = [(2, 332)] + [tuple(v) for v in C.values()]; names = ["T"] + list(C)
yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
S = {"flat158": np.full((224, 224), 158.)}
for p in (8, 16):
    for a in (1, 3, 60):
        S[f"v{p}-a{a}"] = 158 + a * np.sign(np.sin(2 * np.pi * xx / p)); S[f"h{p}-a{a}"] = 158 + a * np.sign(np.sin(2 * np.pi * yy / p))
X = np.stack([np.repeat(np.clip(v, 0, 255)[..., None], 3, -1).astype(np.uint8) for v in S.values()]); Z = multi(T, X)
print("stim      " + " ".join(f"{n:>11s}" for n in names)); print("          " + " ".join(f"{str(t):>11s}" for t in T))
for j, s in enumerate(S):
    row = []
    for t in T:
        mx_ = json.load(open(os.path.join(DEEP, f"b{t[0]}_n{t[1]}.json")))["max"]; k = Z[t].shape[1] // 4; z = Z[t][j, k:-k, k:-k]
        row.append(f"{z.max() / mx_:+.2f}/{z.std() / mx_:.2f}")
    print(f"{s:9s} " + " ".join(f"{r:>11s}" for r in row))
