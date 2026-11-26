"""geometry: write side. Map Small's block-0 write vectors into Base's stream and compare with Base's writes.

Write of a neuron = layer_scale * W2[:, j] (a vector in the stream after block 0).
Map x_S -> x_B by least squares on the block-0 output streams (pass_cov.npz, fit): N = C_BS C_SS^-1 (1024 x 768).
Compared in two metrics: plain stream coordinates, and whitened by Base's stream covariance.
Agreeing pairs (fire best = write best). Rules: p alone; p + k nearest Base writes (equal weights after unit norm);
oracle NNLS on p's k nearest writes; random control. Prints only (disk is low).
"""
import os

import numpy as np
import torch
import torchvision
from scipy.optimize import nnls

from cache import OUT

T = torch.float64


def wr(m):
    b = m.features[7][0]
    return (b.layer_scale.detach().reshape(-1, 1) * b.block[5].weight.detach()).double().T  # (hidden, dim)


def main():
    P = np.load(os.path.join(OUT, "geometry", "pass_cov.npz"))
    Ws = wr(torchvision.models.convnext_small(weights="IMAGENET1K_V1"))
    Wb = wr(torchvision.models.convnext_base(weights="IMAGENET1K_V1"))
    Rf = np.load(os.path.join(OUT, "tools_peakR_b0.npy")).astype(np.float32)
    Rw = np.load(os.path.join(OUT, "writesR_b0.npy")).astype(np.float32)
    fb, wb = Rf.argmax(1), Rw.argmax(1)
    S = np.where(fb == wb)[0]
    p = fb[S]
    CSS = torch.tensor(P["fit_x_SS"], dtype=T)
    CSB = torch.tensor(P["fit_x_SB"], dtype=T)
    CBB = torch.tensor(P["fit_x_BB"], dtype=T)
    tSS, tSB, tBB = (torch.tensor(P[k], dtype=T) for k in ("test_x_SS", "test_x_SB", "test_x_BB"))
    N = torch.linalg.solve(CSS + 1e-6 * torch.eye(768, dtype=T), CSB).T        # x_B ~ N x_S
    res = 1 - (tBB - N @ tSB - tSB.T @ N.T + N @ tSS @ N.T).trace() / tBB.trace()
    print(f"x_B from x_S, test R^2 (total var): {float(res):.3f}")
    ev, U = torch.linalg.eigh(CBB)
    Wh = (U * ev.clamp_min(1e-9) ** -0.5) @ U.T
    rng = np.random.default_rng(0)
    pr = rng.integers(0, 4096, len(S))
    for mname, Mx in (("stream", torch.eye(1024, dtype=T)), ("whitened", Wh)):
        A = (Ws[S] @ N.T) @ Mx
        A = A / A.norm(dim=1, keepdim=True)
        B = Wb @ Mx
        B = B / B.norm(dim=1, keepdim=True)
        c = A @ B.T
        cp = c[np.arange(len(S)), p]
        print(f"[{mname}] cos(s write, p write) median {float(cp.median()):.3f}; random p' "
              f"{float(c[np.arange(len(S)), pr].median()):.3f}; top1 is p {(c.argmax(1).numpy() == p).mean():.3f}")
        G = B @ B.T
        G.fill_diagonal_(-2)
        order = torch.argsort(-G[p], dim=1)
        for k in (1, 3, 10):
            nb = order[:, :k]
            q = B[p] + B[nb].sum(1)
            q = q / q.norm(dim=1, keepdim=True)
            qr = B[p] + B[torch.tensor(rng.integers(0, 4096, (len(S), k)))].sum(1)
            qr = qr / qr.norm(dim=1, keepdim=True)
            sh = [1 - nnls(torch.cat([B[p[i]][None], B[nb[i]]]).T.numpy(), A[i].numpy())[1] ** 2
                  for i in range(0, len(S), 4)]
            sr = [1 - nnls(torch.cat([B[p[i]][None], B[rng.integers(0, 4096, k)]]).T.numpy(), A[i].numpy())[1] ** 2
                  for i in range(0, len(S), 4)]
            print(f"   p + {k} nearest writes: cos {float((q * A).sum(1).median()):.3f} (random {float((qr * A).sum(1).median()):.3f});"
                  f" oracle NNLS share {np.median(sh):.3f} (random {np.median(sr):.3f}; p alone {float((cp.clamp_min(0) ** 2).median()):.3f})")


if __name__ == "__main__":
    main()
