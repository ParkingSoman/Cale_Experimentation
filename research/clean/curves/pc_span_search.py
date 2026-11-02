# serves: H-active
"""Part C, H-active, step 1 of plan i-think-we-are-agile-raccoon.md (estimator fixed before the run).
Same 100 proxy areas; full reads (area w's logits = t_w + V M_w Y_w(x), V shared C x k, M_w k x C, t_w free).
(a) Joint local search over (V, M_w, t_w), capped L-BFGS, from three starts: the shared span (top-k eigenvectors of
    Sigma = sum of area covariances), the canonical packed span (TQ1 part 2), and a random span. The best value found
    is an UPPER bound on the always-on least L^on_S; if it is close to the shared span's fixed-span value, the span is
    the object.
(b) At the shared-span fit: the share of the loss on the quarter of photos where N's top two probabilities are closest
    (the essay: cutting detail loses "fine distinctions between things the network is already confused about"), and
    the share carried by the 10 worst areas.
(c) The second-order objective of TI4 (area-mean Fisher metrics H_w): sum of psi_w(Q) at the shared span against a
    local minimum over Q (L-BFGS on Q, from the shared span and from a random span): TQ7's conclusion, measured
    where its premise fails.
Run with /opt/homebrew/bin/python3.11 (torch)."""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(1102)
rng = np.random.default_rng(1102)
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
    Yc = P - P.mean(0)
    I = float(np.sum(S[idx] * (logS[idx] - np.log(S[idx].mean(0)))) / n)
    vt = np.linalg.svd(Yc[:, K], full_matrices=False)[2]
    areas.append(dict(idx=idx, Yc=Yc, ybar=P.mean(0), I=I, K=K, vt=vt, sb=S[idx].mean(0)))
sumI = sum(a["I"] for a in areas)
Pp = np.eye(C) - np.ones((C, C)) / C
Sigma = sum(a["Yc"].T @ a["Yc"] / n for a in areas)
evS, US = np.linalg.eigh(Sigma)
data = [(torch.tensor(a["Yc"]), torch.tensor(S[a["idx"]]), torch.tensor(logS[a["idx"]])) for a in areas]


def span(kind, k):
    if kind == "shared":
        return US[:, ::-1][:, :k].copy()
    if kind == "packed":
        V = np.zeros((C, k))
        for a in areas:
            r = min(k, a["vt"].shape[0]); V[a["K"], :r] += a["vt"][:r].T
        return np.linalg.qr(Pp @ V)[0]
    return np.linalg.qr(Pp @ rng.normal(size=(C, k)))[0]


def lbfgs(params, f, rounds, iters):
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                            tolerance_change=1e-13, history_size=30)
    for _ in range(rounds):
        def cl():
            opt.zero_grad(); v = f(); v.backward(); return v
        opt.step(cl)
    with torch.no_grad():
        return float(f())


def kl_total(V, Ms, Ts, per_point=False):
    tot = 0.0; pts = []
    for j, (Yc, s, ls) in enumerate(data):
        L = (Yc @ Ms[j].T) @ V.T + Ts[j]
        lq = L - torch.logsumexp(L, 1, keepdim=True)
        kl = (s * (ls - lq)).sum(1)
        tot = tot + kl.sum() / n
        if per_point:
            pts.append(kl.detach().numpy())
    return (tot, pts) if per_point else tot


def fixed_span_fit(Q):
    """the convex fits at a fixed span, area by area (as pc_shared.py --fullread)."""
    V = torch.tensor(Q)
    Ms = [torch.tensor(np.ascontiguousarray(Q.T)).clone().requires_grad_(True) for _ in areas]
    Ts = [torch.tensor(a["ybar"]).clone().requires_grad_(True) for a in areas]
    for j, (Yc, s, ls) in enumerate(data):
        def f(j=j, Yc=Yc, s=s, ls=ls):
            L = (Yc @ Ms[j].T) @ V.T + Ts[j]
            lq = L - torch.logsumexp(L, 1, keepdim=True)
            return (s * (ls - lq)).sum() / n
        lbfgs([Ms[j], Ts[j]], f, 3, 150)
    return V, Ms, Ts


out = dict(sum_I=sumI)
ks = [int(a) for a in sys.argv[1:]] or [38, 14]
for k in ks:
    t0 = time.time(); row = dict(k=k)
    # (a)
    for start in ["shared", "packed", "random"]:
        Q = span(start, k)
        V, Ms, Ts = fixed_span_fit(Q)
        fixed = float(kl_total(V, Ms, Ts))
        if start == "shared":
            # (b) at the shared fixed-span fit, before the joint search moves it
            with torch.no_grad():
                tot, pts = kl_total(V, Ms, Ts, per_point=True)
            allpts = np.concatenate(pts); idx_all = np.concatenate([a["idx"] for a in areas])
            srt = np.sort(S[idx_all], axis=1); margin = srt[:, -1] - srt[:, -2]
            q = np.quantile(margin, 0.25)
            area_loss = np.array([p.sum() for p in pts])
            row["near_tie_quarter_share"] = float(allpts[margin <= q].sum() / allpts.sum())
            row["worst10_area_share"] = float(np.sort(area_loss)[::-1][:10].sum() / area_loss.sum())
        V = V.clone().requires_grad_(True)
        joint = lbfgs([V] + Ms + Ts, lambda: kl_total(V, Ms, Ts), 4, 100)
        row[start] = dict(fixed_span=fixed / sumI, joint=joint / sumI)
        print(k, start, row[start], flush=True)
    # (c) second-order objective with area-mean metrics
    Hs = [torch.tensor(a["sb"]) for a in areas]
    Ycs = [d[0] for d in data]; ms = [len(a["idx"]) / n for a in areas]

    def psi_sum(Q):
        tot = 0.0
        for sb, Yc in zip(Hs, Ycs):
            HQ = sb[:, None] * Q - sb[:, None] * (sb @ Q)[None, :]
            G = Q.T @ HQ
            A = Yc @ HQ
            HYY = (Yc * sb[None, :]) - (Yc @ sb)[:, None] * sb[None, :]   # rows: H Y(x)
            trHS = (Yc * HYY).sum() / n
            tot = tot + trHS - torch.trace(torch.linalg.solve(G, A.T @ A)) / n
        return tot
    with torch.no_grad():
        ps = float(psi_sum(torch.tensor(span("shared", k))))
    best = ps
    for start in ["shared", "random"]:
        Qv = torch.tensor(span(start, k)).requires_grad_(True)
        v = lbfgs([Qv], lambda: psi_sum(torch.linalg.qr(Qv)[0]), 3, 100)
        best = min(best, v)
    row["second_order_shared"] = ps; row["second_order_best"] = best; row["second_order_ratio"] = ps / best
    row["seconds"] = time.time() - t0
    print(row, flush=True)
    out[f"k{k}"] = row
json.dump(out, open(os.path.join(HERE, "results_pc_span_search.json"), "w"), indent=1)
