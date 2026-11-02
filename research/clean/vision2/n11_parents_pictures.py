"""Block-scale picture check: block-16 parents solved for block 17's removed children (n11_gn_17.npz).
Rows per parent: original top spots, solved top spots, solved on the strongest removed child's top spots vs original."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from PIL import Image, ImageDraw
import gallery_x as gx
from convnext import *
m = gx.model("tiny"); gx._init(); X, y, idx = gx._X, gx._y, gx._idx; names = [n.split(",")[0] for n in class_names()]
z = np.load(os.path.join(HF, "imagenet", "n11_gn_17.npz")); W1o = np.array(m.blocks[16].fc1.weight); b1o = np.array(m.blocks[16].fc1.bias); W2o = np.array(m.blocks[16].fc2.weight)
W1n, b1n, W2n = z["W1_16"], z["b1_16"], z["W2_16"]; ch = np.where(np.abs(W1n - W1o).sum(1) > 1e-6)[0]
dw = np.linalg.norm(W2n[:, ch] - W2o[:, ch], axis=0) / np.linalg.norm(W2o[:, ch], axis=0); du = np.linalg.norm(W1n[ch] - W1o[ch], axis=1) / np.linalg.norm(W1o[ch], axis=1)
print("solved parents", len(ch), "median |du|/|u| %.3f |dw|/|w| %.3f" % (np.median(du), np.median(dw)))
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]
W217 = np.array(m.blocks[17].fc2.weight); rem17 = set(np.where(np.abs(z["W2_17"]).sum(0) == 0)[0].tolist())
pick = ch[np.argsort(-(dw + du))[:3]]
pre16, h17 = [], []
for i in range(0, len(idx), 50):
    x = gx.mx.array(norm_img(np.asarray(X[idx[i:i + 50]]))); x = m.stem_norm(m.stem(x))
    for b in range(17):
        if b > 0 and m.stage_of[b] != m.stage_of[b - 1]: x = m.down[m.stage_of[b] - 1](m.down_norm[m.stage_of[b] - 1](x))
        if b == 16: pre16.append(np.array(m.blocks[16].pre(x)).reshape(x.shape[0], 49, -1))
        x = m.blocks[b](x)
    h17.append(np.array(nn.gelu(m.blocks[17].fc1(m.blocks[17].pre(x)))).reshape(x.shape[0], 49, -1))
pre16 = np.concatenate(pre16); h17 = np.concatenate(h17); gl = lambda v: v * 0.5 * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v ** 3)))
T = 88; k = 8; rows = []
for j in pick:
    ho = gl(pre16 @ W1o[j] + b1o[j]); hn = gl(pre16 @ W1n[j] + b1n[j])
    kids = [int(c) for c in rem17 if MAP[f"17_{c}"]["parent"] == [16, int(j)]]
    c = max(kids, key=lambda c: h17[..., c].max()) if kids else None
    sp = lambda a: [(i, a[i].argmax()) for i in np.argsort(-a.max(1))[:k]]
    rows += [(f"block 16 n{j} ORIGINAL (|du|/|u| {du[list(ch).index(j)]:.2f}, |dw|/|w| {dw[list(ch).index(j)]:.2f}, {len(kids)} removed children)", sp(ho), ho), (f"   n{j} SOLVED", sp(hn), hn)]
    if c is not None: hc = h17[..., c]; rows += [(f"   removed child 17 n{c}: its top spots", sp(hc), hc), ("   original parent on the child's spots", sp(hc), ho), ("   solved parent on the child's spots", sp(hc), hn)]
img = Image.new("RGB", (T * k + 16, (T + 30) * len(rows) + 30), "white"); dr = ImageDraw.Draw(img); yy = 30
dr.text((8, 6), "Block-scale solve: block-16 parents re-solved after half of block 17 was removed (closed form)", fill="black")
for cap, s_, val in rows:
    dr.text((8, yy - 12), cap, fill="black"); mv = max(val.max(), 1e-6)
    for q, (i, p) in enumerate(s_): img.paste(gx.crop(i, p, 3), (8 + q * T, yy)); dr.text((8 + q * T, yy + T), f"{names[y[idx[i]]][:9]} {100 * val[i, p] / mv:.0f}%", fill="darkgreen")
    yy += T + 30
fp = os.path.join(HERE, "figs", "l1", "n11_block_parents.png"); img.save(fp); print(fp)
