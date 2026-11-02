# serves: H-active
"""Part C, H-active: Theorem TQ10 on the ResNet-18 proxy (estimator fixed before the run; plan step 4).
Same 100 areas; full reads (the features of area w are its centred logits Y_w(x), so B_w = I and the linear-read claim
holds at eps = 0 by construction of the proxy). Q0 = top-k eigenvectors of Sigma; kept codes c0_x = Q0^T Y_w(x); dropped
part T_x = Y_w(x) - Q0 c0_x; H_x = H(s^0_x), s^0 = softmax(ybar_w + Q0 c0_x).
Phi = (1/2) min over (Delta: C x k, M_w: k x C, beta_w: C) of sum p_x |Delta c0_x + Q0 M_w Y_w(x) - T_x + beta_w|_(H_x)^2,
solved by conjugate gradients on this convex quadratic (matrix-free, torch). Reported, as shares of sum I:
  Phi (TQ10's leading term at e = 1), the KL of the TQ10 build (span Q0 + Delta*, codes c0 + M* Y, free terms
  ybar + beta*), next to the best build found (pc_span_search.py: 0.0249 at k = 38, 0.0974 at k = 14).
--own: the metric H_x = H(s_x) of N's own prediction (equal to leading order, Theorem TQ10's proof), instead of s^0.
Run with /opt/homebrew/bin/python3.11 (torch)."""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
Y = np.load(os.path.join(HERE, "cache/r18/y_train.npy")).astype(np.float64)
LAB = np.load(os.path.join(HERE, "cache/r18/active_labels.npz"))
n, C = Y.shape
logS = Y - Y.max(1, keepdims=True)
logS = logS - np.log(np.exp(logS).sum(1, keepdims=True))
S = np.exp(logS)
top = S.argmax(1)
lab = LAB["G100"]
areas = []
for g in range(lab.max() + 1):
    K = np.where(lab == g)[0]
    idx = np.where(np.isin(top, K))[0]
    if len(idx) < 2:
        continue
    P = Y[idx] - Y[idx].mean(1, keepdims=True)
    I = float(np.sum(S[idx] * (logS[idx] - np.log(S[idx].mean(0)))) / n)
    areas.append(dict(idx=idx, Yc=P - P.mean(0), ybar=P.mean(0), I=I))
sumI = sum(a["I"] for a in areas)
Sigma = sum(a["Yc"].T @ a["Yc"] / n for a in areas)
evS, US = np.linalg.eigh(Sigma)
pw = 1.0 / n


def run(k, iters=400):
    t0 = time.time()
    Q0 = torch.tensor(US[:, ::-1][:, :k].copy())
    data = []
    for a in areas:
        Yc = torch.tensor(a["Yc"]); c0 = Yc @ Q0; T = Yc - c0 @ Q0.T
        y0 = torch.tensor(a["ybar"]) + c0 @ Q0.T
        s0 = torch.softmax(y0, 1) if "--own" not in sys.argv else torch.tensor(S[a["idx"]])
        data.append((Yc, c0, T, s0, torch.tensor(a["ybar"]), torch.tensor(S[a["idx"]]), torch.tensor(logS[a["idx"]])))
    Delta = torch.zeros(C, k, requires_grad=True)
    Ms = [torch.zeros(k, C, requires_grad=True) for _ in areas]
    Bs = [torch.zeros(C, requires_grad=True) for _ in areas]

    def quad():
        tot = 0.0
        for j, (Yc, c0, T, s0, yb, s, ls) in enumerate(data):
            r = c0 @ Delta.T + (Yc @ Ms[j].T) @ Q0.T - T + Bs[j]
            Hr = s0 * r - s0 * (s0 * r).sum(1, keepdim=True)
            tot = tot + 0.5 * pw * (r * Hr).sum()
        return tot
    # convex quadratic: L-BFGS converges like CG here; capped
    opt = torch.optim.LBFGS([Delta] + Ms + Bs, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50)
    for _ in range(3):
        def cl():
            opt.zero_grad(); v = quad(); v.backward(); return v
        opt.step(cl)
    with torch.no_grad():
        Phi = float(quad())
        T0 = float(sum(0.5 * pw * (d[2] * (d[3] * d[2] - d[3] * (d[3] * d[2]).sum(1, keepdim=True))).sum() for d in data))
        kl = 0.0
        for j, (Yc, c0, T, s0, yb, s, ls) in enumerate(data):
            l = yb + Bs[j] + (c0 + Yc @ Ms[j].T) @ (Q0 + Delta).T
            lq = l - torch.logsumexp(l, 1, keepdim=True)
            kl += float((s * (ls - lq)).sum()) / n
    out = dict(k=k, Phi=Phi / sumI, untilted_quadratic=T0 / sumI, tq10_build_kl=kl / sumI, sum_I=sumI,
               seconds=time.time() - t0)
    print(out, flush=True)
    return out


res = [run(int(a)) for a in ([x for x in sys.argv[1:] if not x.startswith("--")] or ["38", "14"])]
json.dump(res, open(os.path.join(HERE, "results_pc_small_drop%s.json" % ("_own" if "--own" in sys.argv else "")), "w"), indent=1)
