"""Move 4 of the exploration: a test that can fail. Synthetic shapes on a plain background, the unit's largest
pre-ReLU over positions for each. Lab work.

usage: s2_shapes.py <block> <channel>
Prediction for 4e:55 (written before the run): a filled shape that is wide at the top and narrows to a point at the
bottom (a downward triangle, a V of two strokes) scores higher than the same area as a circle or a square; the circle
scores lowest (negative, the "O" of the negative pole). An upward triangle (point at the top) scores lower than the
downward one if the unit cares about the direction of the point.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from lib import OUT, load_model, prep, run, sheet

blk, ch = sys.argv[1], int(sys.argv[2])
m = load_model()
S = 224


def canvas(bg, fg, draw_fn, size, jitter):
    im = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(im)
    cx, cy = S / 2 + jitter[0], S / 2 + jitter[1]
    draw_fn(d, cx, cy, size, fg)
    return np.asarray(im)


def tri_down(d, cx, cy, r, fg):
    d.polygon([(cx - r, cy - r * 0.8), (cx + r, cy - r * 0.8), (cx, cy + r)], fill=fg)


def tri_up(d, cx, cy, r, fg):
    d.polygon([(cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8), (cx, cy - r)], fill=fg)


def tri_left(d, cx, cy, r, fg):
    d.polygon([(cx + r * 0.8, cy - r), (cx + r * 0.8, cy + r), (cx - r, cy)], fill=fg)


def circle(d, cx, cy, r, fg):
    d.ellipse([cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85], fill=fg)


def square(d, cx, cy, r, fg):
    d.rectangle([cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75], fill=fg)


def v_strokes(d, cx, cy, r, fg):
    d.line([(cx - r, cy - r), (cx, cy + r), (cx + r, cy - r)], fill=fg, width=max(4, int(r / 6)))


def o_ring(d, cx, cy, r, fg):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=max(4, int(r / 6)))


def ears_face(d, cx, cy, r, fg):  # a crude "pointed face": two ear triangles on a downward triangle
    tri_down(d, cx, cy, r, fg)
    d.polygon([(cx - r, cy - r * 0.8), (cx - r * 0.5, cy - r * 0.8), (cx - r * 0.8, cy - r * 1.4)], fill=fg)
    d.polygon([(cx + r, cy - r * 0.8), (cx + r * 0.5, cy - r * 0.8), (cx + r * 0.8, cy - r * 1.4)], fill=fg)


shapes = dict(tri_down=tri_down, tri_up=tri_up, tri_left=tri_left, circle=circle, square=square,
              v_strokes=v_strokes, o_ring=o_ring, ears_face=ears_face)
colors = [((128, 128, 128), (235, 235, 235)), ((128, 128, 128), (30, 30, 30)), ((40, 90, 40), (200, 150, 90))]
rng = np.random.default_rng(1)
res = {k: [] for k in shapes}
examples = {}
with torch.no_grad():
    for name, fn in shapes.items():
        ims = []
        for bg, fg in colors:
            for size in (30, 45, 60, 80):
                for _ in range(3):
                    ims.append(canvas(bg, fg, fn, size, rng.integers(-20, 21, 2)))
        z = run(m, prep(np.stack(ims)), want_pre=(blk,))["pre_" + blk][:, ch]
        v = z.reshape(len(ims), -1).max(1)[0].cpu().numpy()
        res[name] = v
        examples[name] = (ims[int(np.argmax(v))], float(v.max()))
print(f"{blk}:{ch} largest pre-ReLU over positions, mean (min, max) over 36 images per shape:")
for name, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"  {name:10s} {np.mean(v):8.1f}  ({np.min(v):7.1f}, {np.max(v):7.1f})")
sheet([examples[k][0] for k in shapes], [f"{k} {examples[k][1]:.0f}" for k in shapes],
      os.path.join(OUT, f"{blk}_{ch}_shapes.png"), cols=4)
np.savez(os.path.join(OUT, f"{blk}_{ch}_shapes.npz"), **res)
