"""D2 read weights: equal (consensus) vs the group's top principal direction of z-scored votes or of activations
(loadings used as weights on the members' z-scored votes). Threshold by rate law; Bayes write. Plus global top-16 PCs of
the wide net's z-scored votes (no groups) as a structure-free baseline."""
import os, sys
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_lda.py")).read().split("for s in range(3):")[0])
def build_w(W, groups, weights, rate_mult=1.0):
    Z = Xr @ W["W1"].T + W["b1"]; mu, sd = Z.mean(0), Z.std(0) + 1e-9; W1, b1 = [], []
    for idx, a in zip(groups, weights):
        u = (a[:, None] * W["W1"][idx] / sd[idx, None]).sum(0); b = (a * (W["b1"][idx] - mu[idx]) / sd[idx]).sum()
        S = Xr @ u + b; rate = (Z[:, idx] > 0).mean() * rate_mult; th = np.quantile(S, 1 - min(max(rate, 1e-4), .9999)); W1.append(u); b1.append(b - th)
    return {"W1": np.array(W1), "b1": np.array(b1)}
def pc1(A):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9); v = np.linalg.eigh(A.T @ A)[1][:, -1]; return v * np.sign(v.sum()) * np.sqrt(len(v))
for s in range(3):
    W = net(f"gelu_256_s{s}"); Z = Xr @ W["W1"].T + W["b1"]; Hr = gl(Z); gs = groups_by(Hr, 16, "ward")
    for name, wts in (("equal", [np.ones(len(g)) for g in gs]), ("PC1 of votes", [pc1(Z[:, g]) for g in gs]), ("PC1 of activations", [pc1(Hr[:, g]) for g in gs])):
        print(f"seed {s} ward groups, {name:20s}: fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, build_w(W, gs, wts)), W))
    Zz = (Z - Z.mean(0)) / Z.std(0); ev, V = np.linalg.eigh(Zz.T @ Zz); V = V[:, ::-1][:, :16]
    U = (V.T / Z.std(0)) @ W["W1"]; b = (V.T * ((W["b1"] - Z.mean(0)) / Z.std(0))).sum(1)
    for mult in (1.0,):
        Sx = Xr @ U.T + b; rates = [(Z[:, np.argsort(-np.abs(V[:, k]))[:16]] > 0).mean() for k in range(16)]
        th = np.array([np.quantile(Sx[:, k], 1 - r) for k, r in enumerate(rates)])
        print(f"seed {s} global top-16 PCs of votes (no groups): fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, dict(W1=U, b1=b - th)), W))
