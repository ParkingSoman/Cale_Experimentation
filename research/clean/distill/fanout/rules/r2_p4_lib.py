"""P4 round 2 library (2026-11-26). Same fold as rules/p4write.py (slice anchor, lam = tr(Ss)/N), plus switches:

  null="mean"  : SAME-FORM null. The cut units' teacher firings are replaced by their mean over images at the same
                 token position (a deterministic first-moment message). Same closed form, same ridge, same anchor.
                 It cannot add random cross-image noise (the shuffled null can, and harmed heads in round 1).
  affine=True  : the kept firings get a constant column; its G column goes into the write bias. Derived from: the same
                 least squares with a' -> [a', 1] (the bias is part of the write). Ridge also on the bias column.
  planted      : handled in r2_p4_planted.py (uses build_vit2 with given keep sets).

Object (picked by): every write matrix of the cut set (CLIP o_w / pr_w). Kept units = the floor's ranking (noisecut on
the same X) unless keep sets are given.
PHOTO USE: none. Weights + generic input X only.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
import vitgen  # noqa: E402
from rules import noisecut, p4write as P  # noqa: E402

DEV = noisecut.DEV
BS = P.BS


def fold_aff(Sx, Ss, kept, n_samples, affine):
    """Sx (n x k[+1]), Ss (k[+1] x k[+1]). Slice anchor on the first k columns. Returns G (n x k[+1]), lam."""
    kk = Ss.shape[0]
    k = len(kept)
    lam = float(torch.trace(Ss[:k, :k])) / n_samples
    R = Sx.clone()
    R[torch.as_tensor(kept).long(), torch.arange(k)] += lam
    G = torch.linalg.solve(Ss + lam * torch.eye(kk, dtype=Ss.dtype), R.T).T
    return G, lam


@torch.no_grad()
def build_vit2(T, f, X=None, sets=("mlp",), null=None, affine=False, keep=None, tag="r2", verbose=True):
    """keep: optional dict(stream=idx, heads=[idx per block], mlp=[idx per block]) overriding the floor ranking."""
    t0 = time.time()
    X = P._x_vit(X)
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    if keep is None:
        sc = P._vit_scores(T, X)
        sd = noisecut.top(sc["stream"], d if "stream" in sets else D)
        hk = [noisecut.top(s, h if "heads" in sets else H) for s in sc["head"]]
        mk = [noisecut.top(s, m if "mlp" in sets else M) for s in sc["mlp"]]
    else:
        sd, hk, mk = keep["stream"], keep["heads"], keep["mlp"]
    Td = vitgen.to(T, DEV)
    Wd = vitgen.to(vitgen.cut(T, sd, hk, mk), DEV)
    nX = len(X)
    sls = P._slices(nX)
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, X[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s = emb(Td), emb(Wd)
    sd_dev = sd.to(DEV)
    res = []
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        for kind in ("heads", "mlp"):
            fire, wk, bk = (P._heads_fire, "o_w", "o_b") if kind == "heads" else (P._mlp_fire, "pr_w", "pr_b")
            n = BT[wk].shape[1]
            kept = P._head_rows(hk[bi]) if kind == "heads" else mk[bi]
            k = len(kept)
            kept_dev = kept.to(DEV)
            if kind in sets and k < n:
                cutc = torch.ones(n, dtype=torch.bool)
                cutc[kept] = False
                cutc = cutc.nonzero().flatten().to(DEV)
                mu = None
                if null == "mean":     # per-position mean of the cut firings over images
                    mu = sum(fire(t[sl], BT)[:, :, cutc].sum(0) for sl in sls) / nX     # (L, ncut)
                kk = k + (1 if affine else 0)
                Sx = torch.zeros(n, kk, dtype=torch.float64)
                Ss = torch.zeros(kk, kk, dtype=torch.float64)
                N = 0
                for sl in sls:
                    aT = fire(t[sl], BT)
                    if mu is not None:
                        aT = aT.clone()
                        aT[:, :, cutc] = mu.unsqueeze(0).expand(aT.shape[0], -1, -1)
                    aS = fire(s[sl], Bs)
                    aT, aS = aT.flatten(0, 1), aS.flatten(0, 1)
                    if affine:
                        aS = torch.cat([aS, torch.ones_like(aS[:, :1])], 1)
                    Sx += (aT.T @ aS).cpu().double()
                    Ss += (aS.T @ aS).cpu().double()
                    N += aS.shape[0]
                G, lam = fold_aff(Sx / N, Ss / N, kept, N, affine)
                Dsd = BT[wk][sd_dev].cpu().double()
                WG = Dsd @ G
                Bs[wk] = WG[:, :k].float().to(DEV)
                if affine:
                    Bs[bk] = Bs[bk] + WG[:, k].float().to(DEV)
                res.append(float((G[:, :k] - P._ek(kept, n)).norm() / G[:, :k].norm()))
            for sl in sls:
                t[sl] = t[sl] + F.linear(fire(t[sl], BT), BT[wk], BT[bk])
                s[sl] = s[sl] + F.linear(fire(s[sl], Bs), Bs[wk], Bs[bk])
    W = vitgen.to(Wd, "cpu")
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    if verbose:
        print(f"[r2_p4 {tag} f={f:g}] sites {len(res)} mean G off-slice {sum(res) / max(len(res), 1):.4f}; "
              f"{time.time() - t0:.1f} s", flush=True)
    return W
