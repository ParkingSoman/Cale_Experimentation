"""S3: blind lineup. Rules: VISION-LOG.md S3."""
import json, os
import numpy as np
from convnext import *
import gallery as G
cards = [c for c in json.load(open(os.path.join(RES, "vision-s2-cards.json"))) if not c["held_out"]]
M = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz"))
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)])
rng = np.random.default_rng(21); key = {}; rows = []
for i, c in enumerate(cards):
    b, n = c["block"], c["neuron"]
    if b == 0 or not c["up"]: continue
    A = tuple(c["up"][0][:2]); rate = M[f"{b}_rate"][n]; best = (9, None)
    for v in ("view", "point"):
        for s in ("pos", "neg"):
            C = M[f"{b}_{v}_{s}_C"][n].astype(np.float32); br = M[f"{b}_{v}_{s}_br"]
            ok = (C >= .8) & (br > rate) & (br <= .5)
            if ok.any():
                j = int(np.argmin(np.where(ok, br, 9)))
                if br[j] < best[0]: best = (br[j], j)
    if best[1] is None: B = None
    else: g = best[1]; pb = int(np.searchsorted(OFF, g, side="right") - 1); B = (pb, int(g - OFF[pb]))
    ra = M[f"{A[0]}_rate"] if A[0] > 0 else None
    Cn = (A[0], int(rng.choice(np.where((ra > .001) & (ra < .5))[0]) if ra is not None else rng.integers(H[0])))
    cand = [("A", A), ("B", B), ("C", Cn)]; cand = [x for x in cand if x[1] is not None]; order = rng.permutation(len(cand))
    key[f"{b}_{n}"] = {"XYZ"[k]: cand[j][0] for k, j in enumerate(order)}
    key[f"{b}_{n}"]["cands"] = {cand[j][0]: cand[j][1] for j in order}
    rows.append((b, n, f"ANCHOR #{i}"))
    for k, j in enumerate(order): rows.append((cand[j][1][0], cand[j][1][1], f"   candidate {'XYZ'[k]} for #{i}"))
json.dump(key, open(os.path.join(RES, "vision-s3-lineup-key.json"), "w"), indent=1)
os.makedirs(os.path.join(HERE, "figs", "s3"), exist_ok=True)
from PIL import Image
fp = G.gallery(rows, "s3_lineup.png", "Blind lineup", k=6, hide=True); im = Image.open(fp); y0 = 20; i = 0
while y0 < im.height - 50:
    n = 1 + sum(1 for r in rows[i + 1:i + 4] if r[2].startswith("   cand")); h = n * 114
    im.crop((0, y0, im.width, y0 + h)).save(os.path.join(HERE, "figs", "s3", f"lineup_{i:02d}.png")); y0 += h; i += n
print(sorted(os.listdir(os.path.join(HERE, "figs", "s3"))))
