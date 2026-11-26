"""Diagnose K2 (stability) and K4 (reads) failures on one space. Description only, no bars.
K2: self-match rate by category; what a failing concept matches instead (name, category, cosine of the two build
directions).
K4: the per-cell LDA probe w for c. Compare four read scores on held half B: effect size (current tool), cosine of w
with the directions, cosine of the pattern Sigma w with the directions (Haufe et al. 2014: a linear read must be turned
into its forward pattern before it is interpreted), and a logistic probe's pattern.
Usage: python vr_diag.py s4b0
"""
import sys

import numpy as np

import vecread as vr
from vr_check import held_matrix

space = sys.argv[1]
rng = np.random.default_rng(1)
dic = vr.Dictionary(space)
held = vr.Held()
C, G = vr.SPACES[space]
X = held_matrix(held, space)
src = held.src
K = np.nonzero(dic.kept)[0]
Tsrc = {s: X[src == s].reshape(-1, C).sum(0) for s in np.unique(src)}
Msrc = {s: (src == s).sum() * G * G for s in np.unique(src)}


def hdir(k, imgs_mask=None):
    p, ce = held.cells(G, dic.rule[k])[int(dic.ids[k])]
    if imgs_mask is not None:
        ok = imgs_mask[p]
        p, ce = p[ok], ce[ok]
    if len(p) < 10:
        return None, len(p)
    xc = X[p, ce]
    d, w = 0.0, 0
    for s in np.unique(src[p]):
        m = src[p] == s
        d = d + m.sum() * (xc[m].mean(0) - (Tsrc[s] - xc[m].sum(0)) / (Msrc[s] - m.sum()))
        w += m.sum()
    return d / w, len(p)


Hd = np.array([hdir(k)[0] for k in K])
Db = dic.D[K] / np.linalg.norm(dic.D[K], axis=1, keepdims=True)
Dh = Hd / np.linalg.norm(Hd, axis=1, keepdims=True)
S = Db @ Dh.T
match = np.argmax(S, axis=1)
ok = match == np.arange(len(K))
cat = dic.cat[K]
print("K2 self-match by category:")
for c in vr.CATS:
    m = cat == c
    if m.any():
        print(f"  {c:9s} {ok[m].mean():.2f} of {m.sum()}")
bb = Db @ Db.T
print("  failing examples (concept -> matched; cos of the two build directions; build cells):")
for j in np.nonzero(~ok)[0][:15]:
    i = match[j]
    print(f"    {dic.names[K[j]]} ({cat[j]}) -> {dic.names[K[i]]} ({cat[i]}); {bb[j, i]:.2f}; n={int(dic.n[K[j]])}")
near = np.sort(bb - 2 * np.eye(len(K)), axis=1)[:, -1]
print(f"  nearest-other-concept cosine: failing median {np.median(near[~ok]):.2f}, passing median {np.median(near[ok]):.2f}")

# K4
imgs_a = np.zeros(held.n_img, bool)
imgs_a[::2] = True
XA = X[imgs_a].reshape(-1, C)
mu = XA.mean(0)
Sig = np.cov(XA, rowvar=False)
Ci = np.linalg.inv(Sig + 1e-3 * np.trace(Sig) / C * np.eye(C))
HdB = [hdir(k, ~imgs_a)[0] for k in K]
okB = np.array([h is not None for h in HdB])
HdB = np.array([h if h is not None else np.full(C, 1e-9) for h in HdB])
HdBu = HdB / np.linalg.norm(HdB, axis=1, keepdims=True)
sample = rng.choice(np.nonzero(okB)[0], size=min(120, okB.sum()), replace=False)
res = {"effect (current)": [], "cos(w, dirs)": [], "cos(Sigma w, dirs)": [], "logistic pattern": []}
for j in sample:
    k = K[j]
    p, ce = held.cells(G, dic.rule[k])[int(dic.ids[k])]
    a = imgs_a[p]
    if a.sum() < 10:
        continue
    xc = X[p[a], ce[a]]
    w = Ci @ (xc.mean(0) - mu)
    rd = vr.read_read(dic, held, w, imgs=np.nonzero(~imgs_a)[0], top=1)
    res["effect (current)"].append(rd["raise"][0][0] == dic.names[k])
    res["cos(w, dirs)"].append(np.argmax(HdBu @ (w / np.linalg.norm(w))) == j)
    pat = Sig @ w
    res["cos(Sigma w, dirs)"].append(np.argmax(HdBu @ (pat / np.linalg.norm(pat))) == j)
    # logistic probe: c cells against same-source other cells (half A), plain gradient descent
    srcs = np.unique(src[p[a]])
    neg_img = np.nonzero(imgs_a & np.isin(src, srcs))[0]
    ni = rng.choice(neg_img, size=min(len(neg_img), 3000))
    nc = rng.integers(0, G * G, size=len(ni))
    xn = X[ni, nc]
    xp = xc[rng.choice(len(xc), size=min(len(xc), 3000))]
    Z = np.concatenate([xp, xn]) - mu
    Z /= np.sqrt(np.diag(Sig))[None]
    y = np.concatenate([np.ones(len(xp)), np.zeros(len(xn))])
    beta, b0 = np.zeros(C), 0.0
    for _ in range(300):
        q = 1 / (1 + np.exp(-(Z @ beta + b0)))
        g = Z.T @ (q - y) / len(y) + 1e-3 * beta
        beta -= 0.5 * g
        b0 -= 0.5 * (q - y).mean()
    wl = beta / np.sqrt(np.diag(Sig))
    pat = Sig @ wl
    res["logistic pattern"].append(np.argmax(HdBu @ (pat / np.linalg.norm(pat))) == j)
print("K4 top-1 = c, on half B, by read score:")
for k_, v in res.items():
    print(f"  {k_:20s} {np.mean(v):.2f} (n={len(v)})")
