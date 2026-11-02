# serves: H-active
"""Part C, H-active, the shared span (estimator fixed before the run; COARSE-THEORY.md Part C).
The essay: "with less room ... neurons just cover coarser-grained concepts", "coarser, never fractured". Read for
row 3: the good squeeze keeps the directions the areas SHARE, not one folded unit per pair of unrelated details.
Test on the ResNet-18 proxy (same 100 areas as pc_active_scale.py, cache/r18/active_labels.npz):
  the output span Q is computed from N alone, in closed form, then every area gets its exact convex fit at that span
  (Theorem TI3 part 2: at a fixed span the least is a convex soft-label logistic fit; here the fit reads the area's
  own centred logits through Q^T, logits = t_w + Q M_w Q^T Y_w(x), M_w k x k, t_w free: convex in (M_w, t_w)).
  Spans: (a) D2: top-k eigenvectors of Sigma = sum over areas of p-weighted covariances of P_perp y on the area;
         (b) Fisher: Hbar = sum of m_w H_w (H_w = diag(sbar_w) - sbar_w sbar_w^T of the area's mean softmax),
             Q = span of Hbar^-1/2 times the top-k eigenvectors of Hbar^1/2 Sigma Hbar^1/2 (TI4 part 2 with one metric).
  --packed: the control, the canonical packed span of TQ1 part 2 (same convex fits), so only the span differs.
  --fullread: the area reads its whole centred movement (logits = t_w + Q M_w Y_w, M_w k x C), as a box may.
  Reported: loss / sum I next to the packed build with alignments (pc_active_free.py: 0.0966 at k = 38).
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
    Xk = (P - P.mean(0))[:, K]
    vt = np.linalg.svd(Xk, full_matrices=False)[2]
    areas.append(dict(idx=idx, Yc=P - P.mean(0), ybar=P.mean(0), I=I, K=K, vt=vt))
sumI = sum(a["I"] for a in areas)
Sigma = sum(a["Yc"].T @ a["Yc"] / n for a in areas)
Hbar = np.zeros((C, C))
for a in areas:
    sb = S[a["idx"]].mean(0); m = len(a["idx"]) / n
    Hbar += m * (np.diag(sb) - np.outer(sb, sb))
Pp = np.eye(C) - np.ones((C, C)) / C
ev, U = np.linalg.eigh(Pp @ Hbar @ Pp)
keep = ev > 1e-12 * ev.max()
Hh = (U[:, keep] * np.sqrt(ev[keep])) @ U[:, keep].T            # Hbar^1/2 on the image of P_perp
Hih = (U[:, keep] / np.sqrt(ev[keep])) @ U[:, keep].T           # its inverse there
evS, US = np.linalg.eigh(Sigma)
evF, UF = np.linalg.eigh(Hh @ Sigma @ Hh)


def span(kind, k):
    if kind == "packed":   # the canonical packed span of TQ1 part 2: column j = sum over areas of their j-th own direction
        V = np.zeros((C, k))
        for a in areas:
            r = min(k, a["vt"].shape[0]); V[a["K"], :r] += a["vt"][:r].T
        Q, _ = np.linalg.qr(Pp @ V)
        return Q
    if kind == "D2":
        Q = US[:, ::-1][:, :k]
    else:
        Q = Hih @ UF[:, ::-1][:, :k]
    Q, _ = np.linalg.qr(Q)
    return Q


FULL = "--fullread" in sys.argv


def fit(Q, iters=200):
    k = Q.shape[1]; Qt = torch.tensor(Q)
    if FULL:   # the area reads its whole centred movement: logits = t_w + Q M_w Y_w(x), M_w k x C (a legal box read)
        data = [(torch.tensor(a["Yc"]), torch.tensor(S[a["idx"]]), torch.tensor(logS[a["idx"]])) for a in areas]
        Ms = [torch.tensor(np.ascontiguousarray(Q.T)).clone().requires_grad_(True) for _ in areas]
    else:
        data = [(torch.tensor(a["Yc"] @ Q), torch.tensor(S[a["idx"]]), torch.tensor(logS[a["idx"]])) for a in areas]
        Ms = [torch.eye(k).clone().requires_grad_(True) for _ in areas]
    Ts = [torch.tensor(a["ybar"]).clone().requires_grad_(True) for a in areas]
    tot = 0.0
    for j, (c, s, ls) in enumerate(data):                     # areas are independent at a fixed span: fit each
        opt = torch.optim.LBFGS([Ms[j], Ts[j]], max_iter=iters, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-13)
        def f():
            opt.zero_grad()
            L = (c @ Ms[j].T) @ Qt.T + Ts[j]
            lq = L - torch.logsumexp(L, 1, keepdim=True)
            v = (s * (ls - lq)).sum() / n; v.backward(); return v
        for _ in range(3):
            opt.step(f)
        tot += float(f())
    return tot


res = []
for k in [int(a) for a in sys.argv[1:] if not a.startswith("--")] or [38, 14, 2]:
    for kind in (["packed"] if "--packed" in sys.argv else ["D2", "Fisher"]):
        t0 = time.time()
        v = fit(span(kind, k))
        r = dict(k=k, span=kind, fullread=FULL, loss_over_I=v / sumI, sum_I=sumI, seconds=time.time() - t0)
        print(r, flush=True); res.append(r)
json.dump(res, open(os.path.join(HERE, "results_pc_shared%s%s.json" % ("_full" if FULL else "", "_packed" if "--packed" in sys.argv else "")), "w"), indent=1)
