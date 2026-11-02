"""Shift test on each neuron's top-30 images: move content by 48 px (3 cells) up/down/left/right, fill the gap with
the ImageNet mean color (= what zero padding looks like). Peak 'stays' = same cell (+-1) as before (fixed frame
position); 'follows' = moved with the content (+-1). Also peak value ratio. Usage: python shift.py B,N [...]"""
import sys, os, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
import gallery as G
from judge_lib import maps
T = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]; W = {}
for b, n in T: W.setdefault(b, []).append(n)
G._init(); S = G.scan(W); MEAN = np.array([124, 116, 104], np.uint8); D = 48
def shift(im, dy, dx):
    out = np.empty_like(im); out[:] = MEAN
    ys, yd = (slice(0, 224 - dy), slice(dy, 224)) if dy >= 0 else (slice(-dy, 224), slice(0, 224 + dy))
    xs, xd = (slice(0, 224 - dx), slice(dx, 224)) if dx >= 0 else (slice(-dx, 224), slice(0, 224 + dx))
    out[yd, xd] = im[ys, xs]; return out
pk = lambda z: np.array(divmod(int(np.argmax(z)), z.shape[1]))
for t in T:
    vm = S[t][0]; top = np.argsort(-vm)[:30]; imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in top])
    z0 = maps(imgs, [t])[t]; res = {}
    for name, (dy, dx) in {"down": (D, 0), "up": (-D, 0), "right": (0, D), "left": (0, -D)}.items():
        z1 = maps(np.stack([shift(im, dy, dx) for im in imgs]), [t])[t]
        st = fo = 0
        for a, b in zip(z0, z1):
            p0, p1 = pk(a), pk(b); exp = p0 + np.array([dy, dx]) // 16
            st += np.abs(p1 - p0).max() <= 1; fo += np.abs(p1 - exp).max() <= 1
        res[name] = dict(stays=st / 30, follows=fo / 30, ratio=round(float(np.median(z1.max((1, 2)) / z0.max((1, 2)))), 2))
    print(f"b{t[0]} n{t[1]}", json.dumps(res))
