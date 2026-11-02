"""D5 which block-17 neurons did training change most (block 16 dropped), and what are they? Per neuron on the 2k
validation images: change size = std(h_trained - h_dropped) x |write|; read cos, bias shift (in units of the vote std),
write cos and norm ratio; firing rate before/after (rate law?). Then a sheet: for the top-6 changed neurons, the images
where the activation rose most and fell most (value dropped -> trained, % of max). usage: python d5_changed.py NPZ"""
import os, sys, copy
import numpy as np, mlx.core as mx, mlx.nn as nn
from PIL import Image, ImageDraw
from convnext import *
m = load_convnext(); X, y = load_images(); names = [n.split(",")[0] for n in class_names()]; D = os.path.join(HF, "imagenet", "d5cache")
z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0]; b16, b17 = m.blocks[16], m.blocks[17]; t17 = copy.deepcopy(b17); t17.load_weights([(k, mx.array(v)) for k, v in np.load(sys.argv[1]).items()])
Hd, Ht, Hi = [], [], []
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32)
    for L, blk, xx in ((Hd, b17, x), (Ht, t17, x), (Hi, b17, b16(x))): L.append(np.array(gelu(blk.fc1(blk.pre(xx)))).reshape(len(VA[a:a + 100]), 49, 3072).max(1))
Hd, Ht, Hi = (np.concatenate(v).astype(np.float64) for v in (Hd, Ht, Hi))   # per image: max over positions
g0 = np.array(b17.gamma); w0 = np.array(b17.fc2.weight) * g0[:, None]; w1 = np.array(t17.fc2.weight) * np.array(t17.gamma)[:, None]
u0, u1 = np.array(b17.fc1.weight), np.array(t17.fc1.weight); c0, c1 = np.array(b17.fc1.bias), np.array(t17.fc1.bias)
size = (Ht - Hd).std(0) * np.linalg.norm(w1, axis=0); top = np.argsort(-size)[:12]
cos = lambda a, b: (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
print("median over all neurons: read cos %.4f  write cos %.4f  write norm ratio %.3f" % (np.median(cos(u0, u1)), np.median(cos(w0.T, w1.T)), np.median(np.linalg.norm(w1, axis=0) / np.linalg.norm(w0, axis=0))))
print(f"{'neuron':>6s} {'size':>6s} {'readcos':>7s} {'dbias':>6s} {'wcos':>5s} {'wratio':>6s} {'mean intact/dropped/trained (img max)':>40s}")
for j in top: print(f"{j:6d} {size[j]:6.2f} {cos(u0[j], u1[j]):7.3f} {c1[j] - c0[j]:+6.2f} {cos(w0[:, j], w1[:, j]):5.2f} {np.linalg.norm(w1[:, j]) / np.linalg.norm(w0[:, j]):6.2f} {Hi[:, j].mean():12.2f} {Hd[:, j].mean():8.2f} {Ht[:, j].mean():8.2f}")
T = 80; k = 7; rows = []
for j in top[:6]:
    d = Ht[:, j] - Hd[:, j]; rows.append((j, "rose", np.argsort(-d)[:k])); rows.append((j, "fell", np.argsort(d)[:k]))
img = Image.new("RGB", (T * k + 16, (T + 28) * len(rows) + 10), "white"); dr = ImageDraw.Draw(img); yy = 14
for j, kind, ims in rows:
    mxv = max(Hi[:, j].max(), Hd[:, j].max(), Ht[:, j].max(), 1e-6); dr.text((8, yy - 12), f"block 17 n{j}: activation {kind} most (intact / dropped / trained, % of max)", fill="black")
    for q, i in enumerate(ims):
        gi = z["ana"][VA[i]]; img.paste(Image.fromarray(np.asarray(X[gi])).resize((T, T)), (8 + q * T, yy))
        dr.text((8 + q * T, yy + T), f"{names[y[gi]][:7]} {100 * Hi[i, j] / mxv:.0f}/{100 * Hd[i, j] / mxv:.0f}/{100 * Ht[i, j] / mxv:.0f}", fill="darkgreen")
    yy += T + 28
fp = os.path.join(HERE, "figs", "d", "d5_changed.png"); img.save(fp); print(fp)
