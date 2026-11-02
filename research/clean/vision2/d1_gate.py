"""D1 step 1f: do the tiny parts of one small neuron share a GATE in front (upstream neurons feeding many of them)?
Parts = top-K tiny neurons by correlation with the small neuron (same search as d1_bond, cached to d1/parts_*.npy).
For each part: its top-30 positive-edge parents (e1_edges). Count how many parts each upstream neuron feeds.
Null: 200 random same-block sets of K neurons. usage: python d1_gate.py SMALL sb sn K"""
import os, sys
import numpy as np
from collections import Counter
from convnext import *
small, sb, sn, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz")); H = [384] * 3 + [768] * 3 + [1536] * 9 + [3072] * 3; OFF = np.r_[0, np.cumsum(H)]
nid = lambda g: (int(np.searchsorted(OFF, g, side="right") - 1), int(g - OFF[np.searchsorted(OFF, g, side="right") - 1]))
parts = [tuple(v) for v in np.load(os.path.join(HF, "imagenet", "d1", f"parts_{small}_{sb}_{sn}.npy"))[:K]]
def feeders(nodes, k=30):
    cnt = Counter()
    for b, n in nodes:
        row = E[str(b)][n]; cnt.update(np.argsort(-row)[:k].tolist())
    return cnt
c = feeders(parts); top = c.most_common(8)
rng = np.random.default_rng(0); null = []
for _ in range(200):
    rs = [(b, int(rng.integers(H[b]))) for b, _ in parts]; null.append(feeders(rs).most_common(1)[0][1])
print(f"{small} {sb}/{sn}: {K} parts; blocks {Counter(b for b, _ in parts)}")
print("  most shared upstream feeders (neuron: #parts fed):", [(nid(g), v) for g, v in top])
print(f"  max #parts fed by one upstream neuron: {top[0][1]}  null 95th pct {np.percentile(null, 95):.0f}, median {np.median(null):.0f}")
