"""D2 joint write: same reads/thresholds as d2_build; the write is the JOINT expected wide output given all new neurons:
W2' = Cov(W2 h_wide, h_new) Cov(h_new)^-1, b2' = mean(wide output) - W2' mean(h_new). Readings on train inputs only."""
import os, sys
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_build.py")).read().split('if __name__ == "__main__":')[0])
def joint(W, n):
    Hw = gl(Xr @ W["W1"].T + W["b1"]); Hn = gl(Xr @ n["W1"].T + n["b1"]); O = Hw @ W["W2"].T
    Hc = Hn - Hn.mean(0); Oc = O - O.mean(0); W2 = np.linalg.solve(Hc.T @ Hc + 1e-9 * np.eye(Hc.shape[1]), Hc.T @ Oc).T
    return dict(n, W2=W2, b2=W["b2"] + O.mean(0) - W2 @ Hn.mean(0))
for s in range(3):
    W = net(f"gelu_256_s{s}"); Nn = net(f"gelu_16_s{s}"); Hr = gl(Xr @ W["W1"].T + W["b1"]); HN = gl(Xr @ Nn["W1"].T + Nn["b1"])
    zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); C = zs(Hr).T @ zs(HN) / len(Hr)
    print(f"seed {s}: trained 16: fine %.3f coarse %.3f KL %.3f" % score(Nn, W), "| trained reads + joint write: fine %.3f coarse %.3f KL %.3f" % score(joint(W, Nn), W))
    for K in (5, 10, 20):
        parts = [np.argsort(-C[:, j])[:K] for j in range(16)]; print(f"   oracle top-{K} groups + joint write: fine %.3f coarse %.3f KL %.3f" % score(joint(W, build(W, parts)), W))
    for how in ("average", "ward", "random"):
        print(f"   {how:8s} co-firing groups + joint write: fine %.3f coarse %.3f KL %.3f" % score(joint(W, build(W, groups_by(Hr, 16, how))), W))
