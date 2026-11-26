"""JOINT (FANOUT round 2), CLIP: the whole cut with parts that gained, front to back. Separate switches per part.

Object (picked by): every stream dim, head and MLP neuron of the CLIP ViT-B/16 vision tower (the order part picks).
Switches:
  mlp_a   P5 r2 MLP order: s = W^(1-a) ACT^a (rules/r2_p5_lib). a = 1 = the floor (noisecut activity); a = 0 = weights.
  head_a  same form for heads (default 1 = floor).
  fold    sets that get the P4 reader-matched unit fold (rules/p4write.fold_map, slice anchor, A = 0, closed loop on
          the student, front to back). () = floor.
  null    P4's shuffled-pairing null for the folded sets (same form).
Stream: floor activity order (F1: the weight order kills the loud dims), norm mode "kept".
Derived from: round-1 P4 (MLP fold, the one real 50% gain) + round-2 P5 notes (weight order on MLP .202 quick).
Photo use: none. Weights + the generic input X only.
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import vitgen  # noqa: E402
from rules import magcut, noisecut  # noqa: E402
from rules import p4write as P4  # noqa: E402
from rules import r2_p5_lib as P5L  # noqa: E402

DEV = noisecut.DEV


def keep_sets(T, f, X, mlp_a=1.0, head_a=1.0):
    W, A = P5L.scores(T, X)
    d, h, m = vitgen.sizes(f, T)
    sd = noisecut.top(A["stream"], d)
    hk = [magcut.top(P5L.mixs(w, x, head_a), h) for w, x in zip(W["head"], A["head"])]
    mk = [magcut.top(P5L.mixs(w, x, mlp_a), m) for w, x in zip(W["mlp"], A["mlp"])]
    return sd, hk, mk


@torch.no_grad()
def build(T, f, X=None, mlp_a=1.0, head_a=1.0, fold=("heads", "mlp"), null=False):
    X = P4._x_vit(X)
    sd, hk, mk = keep_sets(T, f, X, mlp_a, head_a)
    if not fold:
        return vitgen.cut(T, sd, hk, mk)
    Td = vitgen.to(T, DEV)
    Wd = vitgen.to(vitgen.cut(T, sd, hk, mk), DEV)
    nX = len(X)
    sls = P4._slices(nX)
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, X[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s = emb(Td), emb(Wd)
    sdd = sd.to(DEV)
    offs = []
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        for kind in ("heads", "mlp"):
            fire, wk, bk = (P4._heads_fire, "o_w", "o_b") if kind == "heads" else (P4._mlp_fire, "pr_w", "pr_b")
            if kind in fold:
                n = BT[wk].shape[1]
                kept = P4._head_rows(hk[bi]) if kind == "heads" else mk[bi]
                cutc = torch.ones(n, dtype=torch.bool)
                cutc[kept] = False
                cutc = cutc.nonzero().flatten().to(DEV)
                mom = P4._Mom(n, len(kept))
                for sl in sls:
                    aT = fire(t[sl], BT).flatten(0, 1)
                    if null:
                        aT = aT.clone()
                        aT[:, cutc] = fire(t[P4._roll(sl, nX).to(DEV)], BT).flatten(0, 1)[:, cutc]
                    mom.add(aT, fire(s[sl], Bs).flatten(0, 1))
                G, lam = P4.fold_map(mom.Sx / mom.N, mom.Ss / mom.N, kept, mom.N, "slice")
                Bs[wk] = (BT[wk][sdd].cpu().double() @ G).float().to(DEV)
                offs.append(float((G - P4._ek(kept, n)).norm() / max(G.norm(), 1e-30)))
            for sl in sls:
                t[sl] = t[sl] + F.linear(fire(t[sl], BT), BT[wk], BT[bk])
                s[sl] = s[sl] + F.linear(fire(s[sl], Bs), Bs[wk], Bs[bk])
    W = vitgen.to(Wd, "cpu")
    print(f"[r2_joint vit f={f:g}] mlp_a={mlp_a} head_a={head_a} fold={fold} null={null} "
          f"G_offslice mean {sum(offs) / max(len(offs), 1):.3f}", flush=True)
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    return W


def make(**kw):
    def vit(T, f, X=None):
        return build(T, f, X, **kw)
    return vit
