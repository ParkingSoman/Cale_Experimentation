# serves: H-map
"""Does one stream channel keep one job across a stage? It decides whether the map's unit is the stream channel or the block.
For each stage, over all channels: skip share of the variance at each block, over positions and photos; correlation of the
channel's mean firing, block 0 against block 1 (and stem against stage 1, block 0); the top 100 photos the two blocks share.
Predictions: results/PREDICTIONS.md, written before the first full pass. Reading photos only."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
READ = os.path.join(os.path.dirname(os.path.dirname(HERE)), "cache", "map18", "read")
st = np.load(os.path.join(READ, "skipstats.npz"))


def q(v):
    return [round(float(x), 2) for x in np.quantile(v, [.1, .5, .9])]


def load(name):
    return np.asarray(np.load(os.path.join(READ, name + ".npy"), mmap_mode="r"), dtype=np.float32)


out = {}
for s in (1, 2, 3, 4):
    row = {}
    for b in (0, 1):
        n = st[f"s{s}b{b}_n"]
        r, k = st[f"s{s}b{b}_r"] / n, st[f"s{s}b{b}_s"] / n
        vr, vk = st[f"s{s}b{b}_rr"] / n - r * r, st[f"s{s}b{b}_ss"] / n - k * k
        row[f"skip_share_b{b}"] = q(vk / (vr + vk + 1e-12))
    h0, h1 = load(f"s{s}b0_hm"), load(f"s{s}b1_hm")
    a, c = h0 - h0.mean(0), h1 - h1.mean(0)
    corr = (a * c).sum(0) / (np.sqrt((a * a).sum(0) * (c * c).sum(0)) + 1e-9)
    row["corr_b0_b1"] = q(corr)
    x0, x1 = load(f"s{s}b0_hx"), load(f"s{s}b1_hx")
    shared = [len(set(np.argpartition(-x0[:, u], 100)[:100].tolist()) & set(np.argpartition(-x1[:, u], 100)[:100].tolist())) for u in range(x0.shape[1])]
    row["top100_shared"] = q(np.array(shared, dtype=float))
    on = (h1 > .01 * h1.max(0)).mean(0)
    row["on_share_of_photos"] = q(on)
    out[f"s{s}"] = row
    print(f"s{s}", json.dumps(row), flush=True)
h0, h1 = load("s0b0_hm"), load("s1b0_hm")
a, c = h0 - h0.mean(0), h1 - h1.mean(0)
print("stem against s1 block 0, corr", q((a * c).sum(0) / (np.sqrt((a * a).sum(0) * (c * c).sum(0)) + 1e-9)))
json.dump(out, open(os.path.join(HERE, "results", "job_table.json"), "w"), indent=1)
