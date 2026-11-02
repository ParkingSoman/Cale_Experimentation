"""D6 the most damaged sealed images after dropping block 16: image + intact top-3 / dropped top-3 / trained (best
reference) top-3 with probabilities. Also: is the damage a confusion between NEIGHBOUR classes (intact top-2 contains the
dropped top-1) or a jump to an unrelated class?"""
import os, copy, numpy as np, mlx.core as mx
from PIL import Image, ImageDraw
from convnext import *
m = load_convnext(); X, y = load_images(); names = [n.split(",")[0] for n in class_names()]; D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz"))
b17 = m.blocks[17]; t17 = copy.deepcopy(b17); w = np.load(os.path.join(D, "tr5b_16_16_17_bs64_lr0.0001_cos0_sf0_s0_more1_feat1_lora0.npz"))
t17.load_weights([(k.replace("blocks.17.", ""), mx.array(v)) for k, v in w.items()], strict=False)
sm = lambda L: np.exp(L - L.max(1, keepdims=True)) / np.exp(L - L.max(1, keepdims=True)).sum(1, keepdims=True)
PD, PT = [], []
for i in range(0, 3000, 200):
    x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); PD.append(np.array(m.head(b17(x)))); PT.append(np.array(m.head(t17(x))))
PI, PD, PT = sm(z["LE"].astype(np.float64)), sm(np.concatenate(PD).astype(np.float64)), sm(np.concatenate(PT).astype(np.float64))
kl = (PI * (np.log(PI + 1e-12) - np.log(PD + 1e-12))).sum(1); o = np.argsort(-kl)
flip = PI.argmax(1) != PD.argmax(1); nb = np.array([PD[k].argmax() in np.argsort(-PI[k])[:5] for k in range(3000)])
print(f"images whose top answer changes: {flip.mean():.3f}; of those, the new answer was in the intact top-5: {nb[flip].mean():.3f}")
print(f"KL share from the 10% most damaged images: {np.sort(kl)[::-1][:300].sum() / kl.sum():.3f}")
T = 150; rows = o[:12]; img = Image.new("RGB", (T * 4 + 40, (T + 6) * 3 + 10), "white"); dr = ImageDraw.Draw(img)
fmt = lambda P, k: " | ".join(f"{names[c][:14]} {P[k, c]:.2f}" for c in np.argsort(-P[k])[:2])
lines = []
for q, k in enumerate(rows):
    gi = z["ev"][k]; c, rr = q % 4, q // 4; img.paste(Image.fromarray(np.asarray(X[gi])).resize((T, T)), (10 + c * (T + 8), 6 + rr * (T + 6)))
    lines.append(f"{names[y[gi]][:16]:16s} KL {kl[k]:.2f} | intact: {fmt(PI, k)} | dropped: {fmt(PD, k)} | trained: {fmt(PT, k)}")
fp = os.path.join(HERE, "figs", "d", "d6_damaged.png"); img.save(fp); print(fp); print("\n".join(lines))
