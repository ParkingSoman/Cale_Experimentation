"""Move 6 of the exploration: feature visualization with the transformation robustness the essay asks for (Lucent's
FFT, colour-decorrelated image, standard jitter / scale / rotate). Objective: the unit's pre-ReLU, either averaged
over all positions ("channel") or at the centre position ("neuron"); also the negative direction. Lab work.

usage: s4_fv.py <block> <channel>
"""
import os
import sys

import numpy as np
import torch
import lucent.optvis.param.color as _co
import lucent.optvis.param.spatial as _sp
import lucent.optvis.transform as _tr
from lucent.optvis import param, transform

from lib import DEV, OUT, load_model, run, sheet

for _mod in (_sp, _co, _tr):
    _mod.device = DEV
_CC = torch.tensor(_co.color_correlation_normalized.T).to(DEV)
_co._linear_decorrelate_color = lambda t: torch.matmul(t.permute(0, 2, 3, 1), _CC).permute(0, 3, 1, 2)

blk, ch = sys.argv[1], int(sys.argv[2])
m = load_model(redirected=True)


def visualize(kind, sign, seed, steps=512, size=224):
    torch.manual_seed(seed)
    params, img_f = param.image(size, fft=True, decorrelate=True)
    tf = transform.compose(transform.standard_transforms)
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(steps):
        x = tf(img_f()) * 255.0 - 117.0
        z = run(m, x, want_pre=(blk,))["pre_" + blk][0, ch]
        val = z.mean() if kind == "channel" else z[z.shape[0] // 2, z.shape[1] // 2]
        loss = -sign * val
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (img_f()[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


ims, caps = [], []
for kind in ("channel", "neuron"):
    for seed in (0, 1):
        ims.append(visualize(kind, +1, seed)); caps.append(f"{kind} + seed {seed}")
for seed in (0, 1):
    ims.append(visualize("channel", -1, seed)); caps.append(f"channel - seed {seed}")
print(sheet(ims, caps, os.path.join(OUT, f"{blk}_{ch}_fv.png"), cols=3, size=224))
