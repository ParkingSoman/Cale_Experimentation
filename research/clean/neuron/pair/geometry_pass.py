"""geometry step 1: how much of Small's block-0 input is linear in Base's, and the same for the block-0 output stream.

One pass of both models over the 4,000 photos of pos.photos (2 fit + 2 test per class).
Hooks, stage 4 block 0 of each model:
  u = features[7][0].block[2] output (LayerNorm after dwconv, channels last)  Base 1,024 / Small 768
  x = features[7][0] output (stream after block 0, channels first)            Base 1,024 / Small 768
  g = features[7][0].block[4] output (hidden neurons after GELU)              Base 4,096 / Small 3,072
  Cov(g, u_B) for both models, so any read rule q on u_B can be scored against a neuron's real output.
Sums only (float64 on CPU): per split, means and the covariance blocks BB, SS, SB for u and for x.
Saved as float32 in results/pair/geometry/pass_cov.npz.
Use the compute lock (results/pair/FORWARD.lock) around this run.
"""
import argparse
import os
import time

import numpy as np
import torch

from cache import DEV, OUT, load, net
from pos import photos

torch.set_grad_enabled(False)
GO = os.path.join(OUT, "geometry")
os.makedirs(GO, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="smoke: first n photos of each split")
    a = ap.parse_args()
    ph = net.Photos("read")
    tr, te = photos(ph)
    if a.n:
        tr, te = tr[:a.n], te[:a.n]
    ms, mb = load("small"), load("base")
    H = {}
    for name, m in (("S", ms), ("B", mb)):
        m.features[7][0].block[2].register_forward_hook(lambda mod, i, o, name=name: H.__setitem__("u" + name, o))
        m.features[7][0].register_forward_hook(lambda mod, i, o, name=name: H.__setitem__("x" + name, o))
        m.features[7][0].block[4].register_forward_hook(lambda mod, i, o, name=name: H.__setitem__("g" + name, o))
    z = lambda *s: torch.zeros(*s, dtype=torch.float64)
    A = {}
    for sp in ("fit", "test"):
        A[sp] = {}
        for v in ("u", "x"):
            A[sp][v] = dict(n=0, sB=z(1024), sS=z(768), BB=z(1024, 1024), SS=z(768, 768), SB=z(768, 1024))
        # GELU outputs against u_B: Small (3072) and Base (4096)
        A[sp]["g"] = dict(gS=z(3072), gB=z(4096), gSS=z(3072), gBB=z(4096), gSu=z(3072, 1024), gBu=z(4096, 1024))
    t0 = time.time()
    for sp, idx in (("fit", tr), ("test", te)):
        for k in range(0, len(idx), 50):
            ii = idx[k:k + 50]
            x = ph.tensor(ii).to(DEV)
            ms(x), mb(x)
            for v in ("u", "x"):
                if v == "u":
                    B = H["uB"].reshape(-1, 1024).float()
                    S = H["uS"].reshape(-1, 768).float()
                else:
                    B = H["xB"].permute(0, 2, 3, 1).reshape(-1, 1024).float()
                    S = H["xS"].permute(0, 2, 3, 1).reshape(-1, 768).float()
                d = A[sp][v]
                d["n"] += B.shape[0]
                d["sB"] += B.sum(0).cpu().double()
                d["sS"] += S.sum(0).cpu().double()
                d["BB"] += (B.T @ B).cpu().double()
                d["SS"] += (S.T @ S).cpu().double()
                d["SB"] += (S.T @ B).cpu().double()
            g = A[sp]["g"]
            uB = H["uB"].reshape(-1, 1024).float()
            gS = H["gS"].reshape(-1, 3072).float()
            gB = H["gB"].reshape(-1, 4096).float()
            g["gS"] += gS.sum(0).cpu().double()
            g["gB"] += gB.sum(0).cpu().double()
            g["gSS"] += (gS * gS).sum(0).cpu().double()
            g["gBB"] += (gB * gB).sum(0).cpu().double()
            g["gSu"] += (gS.T @ uB).cpu().double()
            g["gBu"] += (gB.T @ uB).cpu().double()
            if (k // 50) % 10 == 0:
                print(f"  {sp} {k + len(ii)}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    out = {}
    for sp in A:
        g = A[sp]["g"]
        n = A[sp]["u"]["n"]
        mu = A[sp]["u"]["sB"] / n
        for nm in ("S", "B"):
            m = g["g" + nm] / n
            out[f"{sp}_g{nm}_mean"] = m.numpy()
            out[f"{sp}_g{nm}_var"] = (g[f"g{nm}{nm}"] / n - m ** 2).numpy()
            out[f"{sp}_g{nm}_cov_uB"] = (g[f"g{nm}u"] / n - m[:, None] * mu[None]).float().numpy()
        for v in ("u", "x"):
            d = A[sp][v]
            n = d["n"]
            mB, mS = d["sB"] / n, d["sS"] / n
            out[f"{sp}_{v}_n"] = np.array(n)
            out[f"{sp}_{v}_mB"] = mB.numpy()
            out[f"{sp}_{v}_mS"] = mS.numpy()
            out[f"{sp}_{v}_BB"] = (d["BB"] / n - mB[:, None] * mB[None]).float().numpy()
            out[f"{sp}_{v}_SS"] = (d["SS"] / n - mS[:, None] * mS[None]).float().numpy()
            out[f"{sp}_{v}_SB"] = (d["SB"] / n - mS[:, None] * mB[None]).float().numpy()
    tag = f"_smoke{a.n}" if a.n else ""
    np.savez(os.path.join(GO, f"pass_cov{tag}.npz"), **out)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
