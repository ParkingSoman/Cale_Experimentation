"""D5 level law (derived from the trained map): after dropping block B, give each block-(B+1) neuron a bias shift so its
mean activation equals its intact mean (per-neuron 1-D solve by bisection on readings; no output loss).
Arms: level law (bias), rate law (bias: match firing rate), write-mean fix (fc2 bias restores the block's mean output, no
neuron changes). Readings: 3,000 analysis images (ranks 0-17); score: sealed 3k.
usage: python d5_levellaw.py [FRAC=1.0]  (FRAC scales the shift, description only)"""
import os, sys, copy
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu
FR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 3)[0]; b16, b17 = m.blocks[16], m.blocks[17]
Zi, Zd = [], []
for a in range(0, len(RD), 100):
    x = mx.array(z["XA"][RD[a:a + 100]]).astype(mx.float32)
    Zi.append(np.array(b17.fc1(b17.pre(b16(x)))).reshape(-1, 3072)); Zd.append(np.array(b17.fc1(b17.pre(x))).reshape(-1, 3072))
Zi, Zd = np.concatenate(Zi).astype(np.float64), np.concatenate(Zd).astype(np.float64)
gl = lambda v: np.array(gelu(mx.array(v.astype(np.float32)))).astype(np.float64)
Hi, Hd = gl(Zi), gl(Zd); mi, md = Hi.mean(0), Hd.mean(0); sub = Zd[::4]
def solve(target, f):          # per neuron: shift d so f(sub + d) = target (monotone in d), bisection
    lo, hi = -np.full(3072, 20.0), np.full(3072, 20.0)
    for _ in range(40):
        mid = (lo + hi) / 2; v = f(sub + mid); up = v < target; lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    return (lo + hi) / 2
d_level = solve(mi, lambda v: gl(v).mean(0)); d_rate = solve((Zi > 0).mean(0), lambda v: (v > 0).mean(0))
W2 = np.array(b17.fc2.weight) * np.array(b17.gamma)[:, None]
def sealed(dbias=None, dout=None):
    blk = copy.deepcopy(b17)
    if dbias is not None: blk.fc1.bias = blk.fc1.bias + mx.array((FR * dbias).astype(np.float32))
    T = mx.array(z["LE"]); out = []
    for a in range(0, 3000, 200):
        x = mx.array(z["XE"][a:a + 200]).astype(mx.float32); h = blk(x)
        if dout is not None: h = h + mx.array(dout.astype(np.float32))
        L = m.head(h); Tt = T[a:a + 200]; lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True)
        out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
print(f"bias shifts: level law median |d| {np.median(np.abs(d_level)):.3f}; rate law median |d| {np.median(np.abs(d_rate)):.3f}; corr(level, rate) {np.corrcoef(d_level, d_rate)[0, 1]:.2f}")
print(f"sealed KL: plain drop {sealed():.4f} | level law (x{FR}) {sealed(d_level):.4f} | rate law (x{FR}) {sealed(d_rate):.4f} | write-mean fix {sealed(dout=FR * W2 @ (mi - md)):.4f}   [strong training .0942]")
np.savez(os.path.join(D, "levellaw.npz"), d_level=d_level, d_rate=d_rate, mi=mi, md=md)
