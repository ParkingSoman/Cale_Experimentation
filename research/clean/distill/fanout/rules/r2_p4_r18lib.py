"""P4 round 2, ResNet-18: OUTPUT-MATCHED conv write ("conv" form) against the round-1 per-position fold.

Round 1 fold: a cut inner unit's firing at position p is predicted from the kept firings at the SAME p, then pushed
through conv2. Mean-only null took most of its gain. Here the write W' (o x k x 3 x 3) of the kept inner units is solved
directly so that conv2'(a') matches the teacher's conv2(a) (pre-BN2, every output channel), using the kept firings at
all 9 taps (neighbours carry spatial information about the cut units, which the per-position fold cannot use).
  min_W' E || conv(W, a) - conv(W', a') ||^2 + lam || W' - W_K ||^2   (im2col rows, one Gram for all out channels)
  lam = tr(Ss) / N (as round 1). Derived from: round-1 objective with the 3x3 reader structure kept (the per-position
  fold is the special case W' = W G). The output-channel metric cancels (each channel solves alone, same Gram).
affine=True adds a constant column; its weight goes into bn2.bias (times gamma / sqrt(rv + eps): the network's own
inference map, not read as data). null="mean": cut-unit teacher firings replaced by their per-position mean over images.
Object (picked by): conv2 of every block (inner set cut by the floor's noise ranking).
PHOTO USE: none. Weights + generic input X only.
"""
import copy
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
from rules import noisecut, p4write as P  # noqa: E402

DEV = noisecut.DEV
BS = 16


@torch.no_grad()
def build(A, f, X=None, form="conv", null=None, affine=False, tag="r2r18"):
    import wide18
    t0 = time.time()
    X = P._x_r18(X)
    key = ("r18", id(X))
    Ad = copy.deepcopy(A).to(DEV).eval()
    if key not in P._C:
        Xd = X.to(DEV).float()
        P._C[key] = (X, Xd, wide18.channel_scores(Ad, Xd, bs=P.BS))
    _, Xd, sc = P._C[key]
    w = wide18.widths(f)
    S = copy.deepcopy(Ad).eval()
    inn = [[wide18.keep_idx(sc["inner"][s_][b], w[s_]) for b in range(2)] for s_ in range(4)]
    for s_ in range(4):
        for b in range(2):
            keep = torch.zeros(wide18.BASE[s_])
            keep[inn[s_][b]] = 1.0
            wide18.layers(S)[s_][b].conv2.weight.mul_(keep.to(DEV).view(1, -1, 1, 1))
    nX = len(Xd)
    sls = [slice(a, min(a + BS, nX)) for a in range(0, nX, BS)]
    hT = torch.cat([P._stem(Ad, Xd[sl]) for sl in sls])
    hS = torch.cat([P._stem(S, Xd[sl]) for sl in sls])
    offs = []
    for s_ in range(4):
        for b in range(2):
            blT, blS = wide18.layers(Ad)[s_][b], wide18.layers(S)[s_][b]
            n = wide18.BASE[s_]
            kept = inn[s_][b].cpu()
            k = len(kept)
            if k < n:
                kd = kept.to(DEV)
                cutc = torch.ones(n, dtype=torch.bool)
                cutc[kept] = False
                cutc = cutc.nonzero().flatten().to(DEV)
                W2 = blT.conv2.weight                               # (o, n, 3, 3)
                mu = None
                if null == "mean":
                    mu = sum(P._inner(blT, hT[sl])[:, cutc].sum(0) for sl in sls) / nX
                kk = k * 9 + (1 if affine else 0)
                Ss = torch.zeros(kk, kk, dtype=torch.float64)
                Sx = torch.zeros(kk, W2.shape[0], dtype=torch.float64)
                N = 0
                for sl in sls:
                    aT = P._inner(blT, hT[sl])
                    if mu is not None:
                        aT = aT.clone()
                        aT[:, cutc] = mu.unsqueeze(0).expand(aT.shape[0], -1, -1, -1)
                    z = F.conv2d(aT, W2, padding=1)                  # teacher pre-BN2 (o, H, W)
                    aS = P._inner(blS, hS[sl])[:, kd]
                    U = F.unfold(aS, 3, padding=1).transpose(1, 2).reshape(-1, k * 9)   # (rows, k*9) order (k, tap)
                    if affine:
                        U = torch.cat([U, torch.ones_like(U[:, :1])], 1)
                    Z = z.permute(0, 2, 3, 1).reshape(-1, W2.shape[0])
                    Ss += (U.T @ U).cpu().double()
                    Sx += (U.T @ Z).cpu().double()
                    N += U.shape[0]
                Ss /= N
                Sx /= N
                lam = float(torch.trace(Ss[:k * 9, :k * 9])) / N / 9.0   # per tap, as round 1 (tr over k units)
                WK = W2[:, kd].reshape(W2.shape[0], -1).cpu().double()   # (o, k*9) slice anchor
                R = Sx.clone()
                R[:k * 9] += lam * WK.T
                Wn = torch.linalg.solve(Ss + lam * torch.eye(kk, dtype=torch.float64), R)  # (kk, o)
                Wk = Wn[:k * 9].T.reshape(W2.shape[0], k, 3, 3).float().to(DEV)
                blS.conv2.weight.zero_()
                blS.conv2.weight[:, kd] = Wk
                if affine:
                    bn = blS.bn2
                    bn.bias.add_((Wn[k * 9].float().to(DEV)) * bn.weight / torch.sqrt(bn.running_var + bn.eps))
                offs.append(float((Wn[:k * 9].T - WK).norm() / WK.norm()))
            nT, nS = [], []
            for sl in sls:
                nT.append(P._out(blT, hT[sl], P._inner(blT, hT[sl])))
                nS.append(P._out(blS, hS[sl], P._inner(blS, hS[sl])))
            hT, hS = torch.cat(nT), torch.cat(nS)
    S = S.cpu().eval()
    del Ad, hT, hS
    if DEV == "mps":
        torch.mps.empty_cache()
    print(f"[{tag} f={f:g}] sites {len(offs)} mean rel change vs slice {sum(offs) / max(len(offs), 1):.4f}; "
          f"{time.time() - t0:.1f} s", flush=True)
    return S
