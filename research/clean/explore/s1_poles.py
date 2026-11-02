"""Move 1 and 2 of the exploration: a unit's strongest inputs (positive pole), its most negative inputs (negative pole,
on the pre-ReLU), and a sample of weak activations. Lab work (exploration plan, Phase 1).

usage: s1_poles.py <block> <channel> [n_images]
Saves out/<block>_<ch>_scan.npz (per image: max and min pre-ReLU over positions and where) and three contact sheets.
"""
import os
import sys

import numpy as np
import torch

from lib import DEV, OUT, class_names, load_images, load_model, prep, run, sheet

blk, ch = sys.argv[1], int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
tag = f"{blk}_{ch}"
m = load_model()
X, y = load_images()
names = class_names()
rng = np.random.default_rng(0)
ix = np.sort(rng.choice(len(y), n, replace=False))
mx, mn, amx, amn = np.zeros(n), np.zeros(n), np.zeros((n, 2), int), np.zeros((n, 2), int)
with torch.no_grad():
    for s in range(0, n, 64):
        b = ix[s:s + 64]
        z = run(m, prep(X[b]), want_pre=(blk,))["pre_" + blk][:, ch]  # B x H x W
        B, H, W = z.shape
        f = z.reshape(B, -1)
        mx[s:s + B], k = f.max(1)[0].cpu().numpy(), f.argmax(1).cpu().numpy()
        amx[s:s + B] = np.stack([k // W, k % W], 1)
        mn[s:s + B], k = f.min(1)[0].cpu().numpy(), f.argmin(1).cpu().numpy()
        amn[s:s + B] = np.stack([k // W, k % W], 1)
np.savez(os.path.join(OUT, f"{tag}_scan.npz"), ix=ix, mx=mx, mn=mn, amx=amx, amn=amn, grid=H)
stride = 224 / H


def box(rc, half=40):
    cy, cx = (rc[0] + 0.5) * stride, (rc[1] + 0.5) * stride
    return (cx - half, cy - half, cx + half, cy + half)


top = np.argsort(-mx)[:24]
bot = np.argsort(mn)[:24]
weak_pool = np.where((mx > 0.2 * mx.max()) & (mx < 0.4 * mx.max()))[0]
weak = rng.choice(weak_pool, min(24, len(weak_pool)), replace=False) if len(weak_pool) else []
for nm, sel, val, pos in [("top", top, mx, amx), ("bottom", bot, mn, amn), ("weak", weak, mx, amx)]:
    sheet([X[ix[i]] for i in sel], [f"{val[i]:.1f} {names[y[ix[i]]]}" for i in sel],
          os.path.join(OUT, f"{tag}_{nm}.png"), boxes=[box(pos[i]) for i in sel])
print(f"{tag}: grid {H}x{W}; max pre {mx.max():.2f}; share of images with max > 0: {(mx > 0).mean():.3f}; "
      f"min pre {mn.min():.2f}")
print("top classes:", [names[y[ix[i]]] for i in top[:12]])
print("bottom classes:", [names[y[ix[i]]] for i in bot[:12]])
print("weak pool size:", len(weak_pool))
