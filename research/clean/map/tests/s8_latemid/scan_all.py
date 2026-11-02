"""Scan the 3000-image set (3 per class) once: fc1 pre-activation maps for the 5 latemid targets and all their
blind candidates, max-pooled to 14x14. Saves out/scan.npz (float16). Usage: python scan_all.py"""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "s6")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from judge_lib import maps
import gallery as G
T = ["14_1043", "12_575", "13_404", "12_245", "12_1342"]
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))
tg = sorted({tuple(map(int, k.split("_"))) for k in T} | {tuple(v) for k in T for v in C[k].values()})
G._init(); imgs = np.stack([np.asarray(G._X[i]) for i in G._idx]); out = {}
for s in range(0, len(tg), 12):
    Z = maps(imgs, tg[s:s + 12])
    for t, z in Z.items():
        f = z.shape[1] // 14
        if f > 1: z = z.reshape(len(z), 14, f, 14, f).max((2, 4))
        out[f"{t[0]}_{t[1]}"] = z.astype(np.float16)
    print(s, flush=True)
np.savez(os.path.join(H, "out", "scan.npz"), y=G._y[G._idx], idx=G._idx, **out)
