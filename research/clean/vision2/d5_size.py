"""D5 size restore (essay: an unread stream direction sets the scale LayerNorm reads everything at). After dropping block 16,
add c * d to the stream entering block 17, where d = the zero-mean direction least read by block 17's fc1 (through its LN
weight) AND the head (fc through head-LN weight); c from one reading: block 17's mean LN scale returns to intact.
Arms: d = least-read direction; d = block 16's mean write direction (control); c * d scaled by FRAC for description."""
import os, sys, copy
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
RD = np.where(rk < 2)[0]
Rd = np.r_[np.array(b17.fc1.weight) * np.array(b17.norm.weight)[None], np.array(m.fc.weight) * np.array(m.head_norm.weight)[None]].astype(np.float64)
C0 = np.eye(768) - 1 / 768; Rc = Rd @ C0
w_, V = np.linalg.eigh(Rc.T @ Rc); cand = [V[:, i] for i in range(768) if abs(V[:, i].sum()) < 1e-3 * 768 ** .5][:3]
d_un = cand[0]; print("least-read direction: reader energy %.2e vs median direction %.2e" % (np.linalg.norm(Rc @ d_un) ** 2, np.median(w_)))
x = mx.array(z["XA"][RD]).astype(mx.float32); m16 = np.array(b16(x) - x).reshape(-1, 768).mean(0); d_m = C0 @ m16; d_m /= np.linalg.norm(d_m)
sig = lambda v: np.array(b17.conv_dw(v)).reshape(-1, 768).std(1)
target = sig(b16(x)).mean()
def cfor(d):
    lo, hi = 0.0, 500.0
    for _ in range(40):
        c = (lo + hi) / 2; s = sig(x + mx.array((c * d).astype(np.float32))).mean(); lo, hi = (c, hi) if s < target else (lo, c)
    return (lo + hi) / 2
def sealed(add=None):
    T = mx.array(z["LE"]); out = []
    for a in range(0, 3000, 200):
        xx = mx.array(z["XE"][a:a + 200]).astype(mx.float32)
        if add is not None: xx = xx + mx.array(add.astype(np.float32))
        L = m.head(b17(xx)); Tt = T[a:a + 200]; lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
print(f"plain drop {sealed():.4f}   [strong training .0942]")
for nm, d in (("least-read direction", d_un), ("block-16 mean-write direction", d_m)):
    c = cfor(d); print(f"{nm}: c {c:.1f} (|m16| {np.linalg.norm(m16):.1f}); sealed KL " + "  ".join(f"x{f}: {sealed(f * c * d):.4f}" for f in (0.5, 1.0)))
