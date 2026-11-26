"""WRITE round 3 library (2026-11-26): the P4 unit fold on a CLIP cut, with switches to find why the fold
hurts the whole cut (r2: weight order .820 / 68.2% -> + MLP fold 1.074 / 61.7%, 95%).

Copied from rules/r2_joint_vit.py (JOINT round 2), same fold (p4write.fold_map, slice anchor), plus switches:
  cut    which sets are cut ("stream", "heads", "mlp"). A set not in cut stays at full width.
  fold   which sets get the unit fold (only "mlp" used here).
  loop   "closed": student firings from the student's own stream (r2 default).
         "open":   student firings from the TEACHER stream (kept stream coords, the student's own block weights).
  null   "roll": P4 shuffled pairing (image i takes image i+1's cut-unit firings); "mean": per-position mean.
  xfold  path of a cached input used ONLY for the fold moments (the kept sets still come from the board's X).
  tokw   weight of the CLS token's samples in the moments (1 = plain; patch tokens keep weight 1).
Kept sets: stream + heads by the floor activity on X, MLP by weight (P5 a = 0); = r2_joint_v_w.
Object (picked by): every pr_w of the cut CLIP blocks (weight order picks the kept MLP neurons).
Photo use: none. Weights + generic input only.
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
import vitgen  # noqa: E402
from rules import magcut, noisecut  # noqa: E402
from rules import p4write as P4  # noqa: E402
from rules import r2_p5_lib as P5L  # noqa: E402

DEV = noisecut.DEV


def keep_sets(T, f, X, cut=("stream", "heads", "mlp"), mlp_a=0.0, head_a=1.0):
    W, A = P5L.scores(T, X)
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    sd = noisecut.top(A["stream"], d if "stream" in cut else D)
    hk = [magcut.top(P5L.mixs(w, x, head_a), h if "heads" in cut else H) for w, x in zip(W["head"], A["head"])]
    mk = [magcut.top(P5L.mixs(w, x, mlp_a), m if "mlp" in cut else M) for w, x in zip(W["mlp"], A["mlp"])]
    return sd, hk, mk


def _load(p):
    return torch.load(p, weights_only=True).float()


@torch.no_grad()
def build(T, f, X=None, cut=("stream", "heads", "mlp"), fold=("mlp",), loop="closed", null=None, xfold=None,
          tokw=1.0, tag=""):
    X = P4._x_vit(X)
    sd, hk, mk = keep_sets(T, f, X, cut)
    Wc = vitgen.cut(T, sd, hk, mk)
    if not fold:
        return Wc
    XF = _load(xfold) if xfold else X
    Td = vitgen.to(T, DEV)
    Wd = vitgen.to(Wc, DEV)
    nX = len(XF)
    sls = P4._slices(nX)
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, XF[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s = emb(Td), emb(Wd)
    sdd = sd.to(DEV)
    offs = []
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        for kind in ("heads", "mlp"):
            fire, wk, bk = (P4._heads_fire, "o_w", "o_b") if kind == "heads" else (P4._mlp_fire, "pr_w", "pr_b")
            n = BT[wk].shape[1]
            kept = P4._head_rows(hk[bi]) if kind == "heads" else mk[bi]
            if kind in fold and len(kept) < n:
                cutc = torch.ones(n, dtype=torch.bool)
                cutc[kept] = False
                cutc = cutc.nonzero().flatten().to(DEV)
                mu = None
                if null == "mean":
                    mu = sum(fire(t[sl], BT)[:, :, cutc].sum(0) for sl in sls) / nX
                mom = P4._Mom(n, len(kept))
                for sl in sls:
                    aT = fire(t[sl], BT)
                    if null == "roll":
                        aT = aT.clone()
                        aT[:, :, cutc] = fire(t[P4._roll(sl, nX).to(DEV)], BT)[:, :, cutc]
                    elif null == "mean":
                        aT = aT.clone()
                        aT[:, :, cutc] = mu.unsqueeze(0).expand(aT.shape[0], -1, -1)
                    src = s[sl] if loop == "closed" else t[sl][..., sdd]
                    aS = fire(src, Bs)
                    if tokw != 1.0:          # sqrt-weight the CLS rows so the moments weight them by tokw
                        wv = torch.ones(aS.shape[1], device=aS.device)
                        wv[0] = tokw ** 0.5
                        aT = aT * wv[None, :, None]
                        aS = aS * wv[None, :, None]
                    mom.add(aT.flatten(0, 1), aS.flatten(0, 1))
                G, lam = P4.fold_map(mom.Sx / mom.N, mom.Ss / mom.N, kept, mom.N, "slice")
                Bs[wk] = (BT[wk][sdd].cpu().double() @ G).float().to(DEV)
                offs.append(float((G - P4._ek(kept, n)).norm() / max(G.norm(), 1e-30)))
            for sl in sls:
                t[sl] = t[sl] + F.linear(fire(t[sl], BT), BT[wk], BT[bk])
                s[sl] = s[sl] + F.linear(fire(s[sl], Bs), Bs[wk], Bs[bk])
    W = vitgen.to(Wd, "cpu")
    print(f"[r3_write {tag} f={f:g}] cut={cut} fold={fold} loop={loop} null={null} xfold={bool(xfold)} tokw={tokw} "
          f"G_offslice mean {sum(offs) / max(len(offs), 1):.3f}", flush=True)
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    return W


def make(**kw):
    def vit(T, f, X=None):
        return build(T, f, X, **kw)
    return vit
