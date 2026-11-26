"""geometry step 2 (cached data only): is a Small read, mapped into Base's input coordinates, a simple function of
Base's read rows?

v_s = stream_V[s] = M^T r_S (M = C_SB C_BB^-1). All comparisons in the metric of u_B (cos = correlation of the two
linear pre-activations over all fit rows).  p = counterpart (pmap).  Neurons: the 2,017 agreeing pairs.
Rule families, each a function of Base weights + stats only (no Small information except which p):
  P      q = W1[p]
  COV_g  q = W1[p] C^g                     (g=1: p's "pattern" Cov(u, h_p) = matched filter, not decorrelating)
  NBk_t  q = sum_j softmax(cos(p,j)/t) n_j over the k nearest Base reads j of p (n_j unit-norm reads)
  USEk   q = sum_j usage_j n_j over the k nearest (usage = Base mean output after GELU)
Controls: same k but random Base rows added to p; and the oracle sparse fit (uses v_s, upper line).
Global parameters chosen on even neuron ids, reported on odd ids.
"""
import json
import os

import numpy as np
import torch

from cache import OUT

torch.set_grad_enabled(False)
GO = os.path.join(OUT, "geometry")
os.makedirs(GO, exist_ok=True)
T = torch.float64


def main():
    d = np.load(os.path.join(OUT, "stream_cov.npz"))
    C = torch.tensor(d["Cuu"], dtype=T)
    W = torch.tensor(d["W1"], dtype=T)
    agree, pmap = d["agree"], d["pmap"]
    V = torch.tensor(np.load(os.path.join(OUT, "stream_V.npy")), dtype=T)
    usage = torch.tensor(np.load(os.path.join(OUT, "grams.npz"))["fit_mb"], dtype=T)
    D = C.shape[0]
    C = C + 1e-6 * torch.eye(D, dtype=T)
    L = torch.linalg.cholesky(C)
    ev, U = torch.linalg.eigh(C)
    ev = ev.clamp_min(1e-9)

    def met(q):  # rows -> metric coordinates, unit norm
        m = q @ L
        return m / m.norm(dim=-1, keepdim=True)

    Wn = met(W)                      # (4096, D)
    Vn = met(V)
    S = agree
    P = torch.tensor(pmap[S])
    v = Vn[S]
    G = Wn @ Wn.T                    # read cosines among Base neurons (metric)
    res = {}

    def score(Q, name):
        c = (met(Q) * v).sum(1).numpy()
        res[name] = dict(even=float(np.median(c[0::2])), odd=float(np.median(c[1::2])),
                         odd_share=float(np.median(c[1::2] ** 2)))
        return c

    base = score(W[P], "P")
    for g in (0.25, 0.5, 0.75, 1.0, 1.5):
        Cg = (U * ev ** g) @ U.T
        score(W[P] @ Cg, f"COV_{g}")
    # neighbor smoothing in the metric
    Gp = G[P].clone()
    Gp[torch.arange(len(P)), P] = 1.0
    order = torch.argsort(-Gp, dim=1)
    Wun = W / (W @ L).norm(dim=1, keepdim=True)    # unit metric norm, Euclidean coords
    rng = np.random.default_rng(0)
    for k in (2, 3, 5, 10, 30, 100):
        idx = order[:, :k]
        cs = torch.gather(Gp, 1, idx)
        for t in (0.05, 0.1, 0.2, 1.0):
            w = torch.softmax(cs / t, dim=1)
            Q = (w[:, :, None] * Wun[idx]).sum(1)
            score(Q, f"NB{k}_t{t}")
        w = usage[idx] * (idx != P[:, None]).double() + (idx == P[:, None]).double() * usage[P][:, None]
        score((w[:, :, None] * Wun[idx]).sum(1), f"USE{k}")
        # control: p plus k-1 random rows, softmax weights as if they had p's neighbour cosines (t=0.1)
        ridx = torch.tensor(rng.integers(0, 4096, size=(len(P), k)))
        ridx[:, 0] = P
        w = torch.softmax(cs / 0.1, dim=1)
        score((w[:, :, None] * Wun[ridx]).sum(1), f"RAND{k}_t0.1")
    # oracle: least squares of v on p's k nearest reads (uses v; upper line for any neighbour rule)
    for k in (3, 10, 30):
        idx = order[:, :k]
        sh = []
        for i in range(len(P)):
            A = Wn[idx[i]].T
            coef = torch.linalg.lstsq(A, v[i][:, None]).solution
            sh.append(float(1 - ((A @ coef)[:, 0] - v[i]).norm() ** 2))
        res[f"ORACLE_nb{k}_share"] = float(np.median(sh[1::2]))
    # what is the residual of v after p? its cosine with p's covariance pattern and with p's neighbours
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(GO, "reads.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
