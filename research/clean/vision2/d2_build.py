"""D2 build: make a 16-neuron net from the 256-neuron wide net with NO training, by the D1 observations:
  group  = neurons that fire together (co-firing groups; several grouping rules shown)
  read   = sum of the members' z-scored votes  (u_G = sum u_i / sd_i, bias = sum (b_i - mean_i) / sd_i - theta)
  theta  = rate law: the new neuron fires as often as its members do on average
  write  = the group's expected output given the new neuron (projection of each member's activation on h_G)
Compare with trained 16-neuron nets (3 seeds): fine acc, coarse acc, KL(wide || net). Readings use the TRAIN inputs
(no labels, no output loss); scores use the test set."""
import os, sys
import numpy as np
from scipy.special import erf, logsumexp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
sys.argv = sys.argv[:1]; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_toy.py")).read().split("class Net")[0])
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
if __name__ == "__main__":
    for s in range(3):
        W = net(f"gelu_256_s{s}"); Hr = gl(Xr @ W["W1"].T + W["b1"])
        print(f"wide s{s}: fine/coarse acc, KL to wide = {score(W, W)}")
        for t in range(3):
            if s == 0: print(f"  trained 16 s{t}: fine %.3f coarse %.3f KL %.3f" % score(net(f"gelu_16_s{t}"), W))
        for how in ("average", "complete", "ward", "random"):
            gs = groups_by(Hr, 16, how); n = build(W, gs)
            print(f"  built 16 ({how:8s} co-firing groups; sizes {sorted(len(g) for g in gs)[::-1][:6]}...): fine %.3f coarse %.3f KL %.3f" % score(n, W))
