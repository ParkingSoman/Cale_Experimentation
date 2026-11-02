"""D6 split block 16's non-proportional write d_perp = d - lambda s (lambda in block 17's readers' metric) into the
readers' subspace (top eigenvectors of M holding a share q of its trace) and the rest (unread). Add back each part alone
to the dropped stream (scaled 1/(1+lambda)), plus sizes: |part|^2 / |d|^2 (euclidean) and in M."""
import os, numpy as np, mlx.core as mx
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); b16, b17 = m.blocks[16], m.blocks[17]
Up = np.array(b17.fc1.weight, np.float64) * np.array(b17.norm.weight, np.float64)[None]; Mm = Up.T @ Up; Mm /= np.trace(Mm) / 768
ev, V = np.linalg.eigh(Mm); ev, V = ev[::-1], V[:, ::-1]; cum = np.cumsum(ev) / ev.sum()
print("readers' metric eigen-spectrum: dims holding 50/90/99/99.9% of trace:", [int(np.searchsorted(cum, q)) + 1 for q in (.5, .9, .99, .999)], f"; smallest/largest eigenvalue {ev[-1] / ev[0]:.2e}")
Mx = mx.array(Mm.astype(np.float32)); cen = lambda v: v - v.mean(-1, keepdims=True)
def parts(x, q):
    k = int(np.searchsorted(cum, q)) + 1; P = mx.array((V[:, :k] @ V[:, :k].T).astype(np.float32))
    s, d = cen(x), b16(x) - x; lam = ((s @ Mx) * cen(d)).sum(-1, keepdims=True) / ((s @ Mx) * s).sum(-1, keepdims=True)
    dp = (d - lam * s) / (1 + lam); return dp @ P, dp - dp @ P, d, lam
def sealed(addf):
    T = mx.array(z["LE"]); out = []
    for i in range(0, 3000, 200):
        x = mx.array(z["XE"][i:i + 200]).astype(mx.float32); L = m.head(b17(x + addf(x))); Tt = T[i:i + 200]
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = Tt - mx.logsumexp(Tt, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
x0 = mx.array(z["XA"][:200]).astype(mx.float32)
for q in (.9, .99, .999):
    r_, u_, d_, lam = parts(x0, q); n = lambda v: float((v * v).sum(-1).mean())
    print(f"q {q}: readers' part |.|^2/|d|^2 {n(r_) / n(d_):.3f}, unread part {n(u_) / n(d_):.3f} | add readers' part only {sealed(lambda x: parts(x, q)[0]):.4f} | add unread part only {sealed(lambda x: parts(x, q)[1]):.4f}", flush=True)
print("plain drop .1042")
