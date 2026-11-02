"""D2 toy library: data, nets, scores, consensus build, Bayes (LDA) write. from d2_lib import *"""
import os, sys
import numpy as np
from scipy.special import erf, logsumexp, softmax
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import os, json, time
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from scipy.special import erf
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf", "toy"); os.makedirs(OUT, exist_ok=True)
gl = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
D, NC, NF = 64, 8, 8; rng = np.random.default_rng(0)
Ac = rng.normal(0, 1.0, (NC, D)); Bf = rng.normal(0, 0.6, (NC, NF, D))          # coarse template, fine offset
def data(n, r):
    c = r.integers(NC, size=n); f = r.integers(NF, size=n); amp = r.uniform(.5, 1.5, (n, 1))
    sg = r.choice([-1.0, 1.0], size=(n, 1)); x = sg * amp * (Ac[c] + Bf[c, f]) + r.normal(0, 1.0, (n, D)); return x.astype(np.float32), (c * NF + f).astype(np.int32), c
Xtr, ytr, ctr = data(60000, np.random.default_rng(1)); Xte, yte, cte = data(20000, np.random.default_rng(2))
T = OUT; gl = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
def net(k): z = np.load(os.path.join(T, k + ".npz")); return {n: z[n].astype(np.float64) for n in z.files}
Xr = Xtr[:20000].astype(np.float64); Xe = Xte.astype(np.float64)
def logits(n, X): return gl(X @ n["W1"].T + n["b1"]) @ n["W2"].T + n["b2"]
def score(n, ref):
    L = logits(n, Xe); R = logits(ref, Xe); lp = L - logsumexp(L, 1, keepdims=True); rp = R - logsumexp(R, 1, keepdims=True)
    return float((L.argmax(1) == yte).mean()), float((L.argmax(1) // NF == cte).mean()), float((np.exp(rp) * (rp - lp)).sum(1).mean())
def build(W, groups, rate_mult=1.0, signs=None):
    Z = Xr @ W["W1"].T + W["b1"]; H = gl(Z); mu, sd = Z.mean(0), Z.std(0) + 1e-9; G = len(groups)
    W1 = np.zeros((G, D)); b1 = np.zeros(G); W2 = np.zeros((NC * NF, G)); b2 = W["b2"].copy()
    for g, idx in enumerate(groups):
        s = np.ones(len(idx)) if signs is None else signs[g]
        u = (s[:, None] * W["W1"][idx] / sd[idx, None]).sum(0); b = (s * (W["b1"][idx] - mu[idx]) / sd[idx]).sum()
        S = Xr @ u + b; rate = (Z[:, idx] > 0).mean() * rate_mult; th = np.quantile(S, 1 - min(max(rate, 1e-4), .9999))
        W1[g], b1[g] = u, b - th; hG = gl(S - th)
        hc = hG - hG.mean(); beta = (H[:, idx] - H[:, idx].mean(0)).T @ hc / (hc @ hc + 1e-12)
        W2[:, g] = W["W2"][:, idx] @ beta; b2 += W["W2"][:, idx] @ (H[:, idx].mean(0) - beta * hG.mean())
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
def groups_by(H, k, how):
    C = np.corrcoef(H.T); C = np.nan_to_num(C); alive = H.std(0) > 1e-6
    if how == "random": r = np.random.default_rng(0); lab = r.integers(k, size=H.shape[1])
    else:
        Dm = 1 - C; np.fill_diagonal(Dm, 0); lab = fcluster(linkage(squareform(np.clip(Dm, 0, 2), checks=False), how), k, "maxclust") - 1
    return [np.where((lab == g) & alive)[0] for g in range(lab.max() + 1) if ((lab == g) & alive).sum()]
def bayes_write(W, n, soft=True):
    Hn = gl(Xr @ n["W1"].T + n["b1"]); P = softmax(logits(W, Xr), 1) if soft else np.eye(NC * NF)[ytr[:len(Xr)]]
    pi = P.mean(0); M = (P.T @ Hn) / (P.sum(0)[:, None] + 1e-12)            # class means (C, k)
    R = Hn[:, None, :] - M[None]; S = np.einsum("nc,nci,ncj->ij", P, R, R) / len(Hn)   # pooled within-class covariance
    Si = np.linalg.inv(S + 1e-9 * np.trace(S) * np.eye(len(S))); Wt = M @ Si; b = -0.5 * np.einsum("ci,ci->c", Wt, M) + np.log(pi + 1e-12)
    return dict(n, W2=Wt, b2=b)

def build_w(W, groups, weights, rate_mult=1.0):
    Z = Xr @ W["W1"].T + W["b1"]; mu, sd = Z.mean(0), Z.std(0) + 1e-9; W1, b1 = [], []
    for idx, a in zip(groups, weights):
        u = (a[:, None] * W["W1"][idx] / sd[idx, None]).sum(0); b = (a * (W["b1"][idx] - mu[idx]) / sd[idx]).sum()
        S = Xr @ u + b; rate = (Z[:, idx] > 0).mean() * rate_mult; th = np.quantile(S, 1 - min(max(rate, 1e-4), .9999)); W1.append(u); b1.append(b - th)
    return {"W1": np.array(W1), "b1": np.array(b1)}
def pc1(A):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9); v = np.linalg.eigh(A.T @ A)[1][:, -1]; return v * np.sign(v.sum()) * np.sqrt(len(v))
