"""TREE round 3 (2026-11-26): TILT, not sum. CLIP MLP-only; stream and heads at full width.

Object (picked by): the MLP read rows fc_w[j] of every KEPT unit j (kept set = magcut WEIGHT order, as r2_p4_wfold_mlp).
Base = plain deletion + P4 fold (r2_p4_lib: slice anchor, lam = tr(Ss)/N, closed loop), MLP sites only.

Derived from: VECMATH 11 (training tilts each kept read toward ONE cut unit = the kept unit's top fold coefficient).
At block b (student stream s, closed loop):
  pass A: the plain fold G (n x k) on the untilted student firings.
  partner c(j) = argmax over CUT c of |G[c, j]|   (part="fold")
               = top R1 partner                    (part="r1": cut unit with the same R1 parent and max cos of votes)
               = uniform random cut unit (seed 5)  (part="rand": same-form NULL, same size, sign from G[c, j])
  tilt:   z_j' = z_j + alpha * sgn * (sd_j / sd_c) * (z_c - mu_c)      (z = pre-activation on the student stream, X)
          i.e. fc_w[j] += a fc_w_T[c],  fc_b[j] += a (fc_b_T[c] - mu_c),  a = alpha sgn sd_j / sd_c.
          The partner's centred read is scaled to one noise sd of the kept unit; the kept unit's mean is unchanged.
  pass B: the fold again on the TILTED firings; write = D G.
alpha = 0 reproduces r2_p4_wfold_mlp exactly (off row).
PHOTO USE: none. Weights + generic input X.
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
from rules import noisecut, p4write as P, r2_p4_lib as L4  # noqa: E402
from rules.r2_p4_wfold_mlp import keep_w  # noqa: E402

DEV = noisecut.DEV


def _pre(x, B):
    return F.linear(vitgen.norm(x, B["ln2"]), B["fc_w"], B["fc_b"])


def _solve(t, s, BT, Bs, sls, kept):
    n = BT["pr_w"].shape[1]
    k = len(kept)
    Sx = torch.zeros(n, k, dtype=torch.float64)
    Ss = torch.zeros(k, k, dtype=torch.float64)
    N = 0
    for sl in sls:
        aT = P._mlp_fire(t[sl], BT).flatten(0, 1)
        aS = P._mlp_fire(s[sl], Bs).flatten(0, 1)
        Sx += (aT.T @ aS).cpu().double()
        Ss += (aS.T @ aS).cpu().double()
        N += aS.shape[0]
    G, _ = L4.fold_aff(Sx / N, Ss / N, kept, N, False)
    return G


@torch.no_grad()
def build(T, f, X=None, alpha=0.1, part="fold", r1=None, tag="r3tree", verbose=True):
    t0 = time.time()
    X = P._x_vit(X)
    D, H, M = vitgen.sizes(1.0, T)
    keep = keep_w(T, f)
    sd, hk, mk = keep["stream"], keep["heads"], keep["mlp"]
    Td = vitgen.to(T, DEV)
    Wd = vitgen.to(vitgen.cut(T, sd, hk, mk), DEV)
    nX = len(X)
    sls = P._slices(nX)
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, X[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s = emb(Td), emb(Wd)
    g = torch.Generator().manual_seed(5)
    stats = []
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        for sl in sls:   # heads: full width, teacher weights
            t[sl] = t[sl] + F.linear(P._heads_fire(t[sl], BT), BT["o_w"], BT["o_b"])
            s[sl] = s[sl] + F.linear(P._heads_fire(s[sl], Bs), Bs["o_w"], Bs["o_b"])
        kept = mk[bi]
        k = len(kept)
        if k < M:
            cutm = torch.ones(M, dtype=torch.bool)
            cutm[kept] = False
            cutc = cutm.nonzero().flatten()
            if alpha != 0:
                G = _solve(t, s, BT, Bs, sls, kept)
                Gc = G[cutc]                                   # (ncut, k)
                if part == "fold":
                    ci = Gc.abs().argmax(0)
                elif part == "rand":
                    ci = torch.randint(len(cutc), (k,), generator=g)
                elif part == "r1":
                    ci = r1(bi, kept, cutc, Gc)
                else:
                    raise ValueError(part)
                sgn = torch.sign(Gc[ci, torch.arange(k)])
                sgn[sgn == 0] = 1
                pc = cutc[ci]                                  # partner teacher index per kept unit
                # pre-activation moments on the student stream (all M teacher units read the full-width stream)
                s1 = torch.zeros(M, dtype=torch.float64)
                s2 = torch.zeros(M, dtype=torch.float64)
                N = 0
                for sl in sls:
                    z = _pre(s[sl], BT).flatten(0, 1)
                    s1 += z.sum(0).cpu().double()
                    s2 += z.pow(2).sum(0).cpu().double()
                    N += z.shape[0]
                mu = s1 / N
                sdv = (s2 / N - mu.pow(2)).clamp_min(1e-12).sqrt()
                a = (alpha * sgn.double() * sdv[kept] / sdv[pc])
                fcT, fbT = BT["fc_w"].cpu().double(), BT["fc_b"].cpu().double()
                Bs["fc_w"] = (fcT[kept] + a[:, None] * fcT[pc]).float().to(DEV)
                Bs["fc_b"] = (fbT[kept] + a * (fbT[pc] - mu[pc])).float().to(DEV)
                stats.append(float((Gc[ci, torch.arange(k)].abs()).mean()))
            G = _solve(t, s, BT, Bs, sls, kept)
            Dsd = BT["pr_w"][sd.to(DEV)].cpu().double()
            Bs["pr_w"] = (Dsd @ G).float().to(DEV)
        for sl in sls:
            t[sl] = t[sl] + F.linear(P._mlp_fire(t[sl], BT), BT["pr_w"], BT["pr_b"])
            s[sl] = s[sl] + F.linear(P._mlp_fire(s[sl], Bs), Bs["pr_w"], Bs["pr_b"])
    W = vitgen.to(Wd, "cpu")
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    if verbose:
        print(f"[r3_tree {tag} f={f:g} alpha={alpha} part={part}] mean |G partner| "
              f"{sum(stats) / max(len(stats), 1):.4f}; {time.time() - t0:.1f} s", flush=True)
    return W
