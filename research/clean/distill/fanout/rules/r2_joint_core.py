"""JOINT (FANOUT round 2): the parts that gained in round 1, put together in one ResNet-18 calculation, front to back.

Object (picked by): every stream channel and inner channel of ResNet-18 (all of them; the order part picks which stay).
Parts and switches (each switch sets ONE part back to its floor):
  order  "floor" = noisecut / wide18.channel_scores on X (the floor);  "p5" = P5 read-aware greedy order
         (rules/p5rank.py, cached order on the same X);  "p5null" = P5's read-permuted null order.
  mean   "off" = floor;  "mean" = P6 mean fold of the cut STREAM channels into every reader's bias
         (rules/_p6diag_r18.py, teacher stream, open loop);  "perm" = its derangement null (same form).
  fold   "off" = floor;  "on" = P4 reader-matched unit fold of the INNER set into conv2 (rules/p4write.py, A = 0,
         slice anchor, closed loop on the student that already has the stream cut + mean fold);
         "null" = P4's shuffled-pairing null (same form).
Form: full-width masked network. A cut stream channel = every read of it set to 0 (block conv1 columns, downsample
columns, fc column). A cut inner channel = its conv2 column set to 0 (the fold writes only kept columns). This is
the same function as the narrow cut (checked: all switches at floor = the noisecut row).
Derived from: round-1 per-part results (round1/build.json): P5 ResNet whole 95% +3.7 agree; P6 meanfold ResNet 95%
+4.6 agree; P4 inner fold 95% +1.8 agree. The composition is plain: order -> cut -> mean fold -> unit fold.
Photo use: none. Weights + the generic input X only. BatchNorm running stats are never read as data (the mean fold
edits running_mean as an edit of the inference map, as in round 1).
"""
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
from rules import _p6diag_r18 as P6  # noqa: E402
from rules import noisecut  # noqa: E402
from rules import p4write as P4  # noqa: E402
from rules import p5rank as P5  # noqa: E402

DEV = noisecut.DEV
BS = 32


def _x(X):
    return X if X is not None else P6._input(None)


def keep_sets(A, f, X, order):
    import wide18
    w = wide18.widths(f)
    if order == "floor":
        sc = P6._scores(A, X)
        st = [wide18.keep_idx(sc["stream"][s], w[s]) for s in range(4)]
        inn = [[wide18.keep_idx(sc["inner"][s][b], w[s]) for b in range(2)] for s in range(4)]
    else:
        C = P5.order("r18", A.cpu(), X, seed=1, null=(order == "p5null"))
        st = [P5.keep(C, f"stream{s}", w[s]) for s in range(4)]
        inn = [[P5.keep(C, f"inner{s}{b}", w[s]) for b in range(2)] for s in range(4)]
    return st, inn


def _mask(n, keep):
    m = torch.ones(n, dtype=torch.bool)
    m[keep] = False
    return m


@torch.no_grad()
def build_r18(A, f, X=None, order="floor", mean="off", fold="off", sets=("stream", "inner"), verbose=True):
    """sets: which sets are cut ('stream', 'inner'); the other set stays at full width."""
    import wide18
    X = _x(X)
    st, inn = keep_sets(A, f, X, order)
    if "stream" not in sets:
        st = [torch.arange(c) for c in wide18.BASE]
    if "inner" not in sets:
        inn = [[torch.arange(c) for _ in range(2)] for c in wide18.BASE]
    cut = [_mask(wide18.BASE[s], st[s]) for s in range(4)]
    S = copy.deepcopy(A).cpu().eval()
    L = wide18.layers(S)
    readers = [(s, name, pt) for s in range(4) for name, pt in P6._readers(s)]

    # ---- P6 mean fold (teacher stream, open loop), computed before any weight changes
    contrib = {}
    if mean != "off" and any(bool(c.any()) for c in cut):
        perm = []
        for s in range(4):
            ci = torch.nonzero(cut[s]).flatten()
            src = torch.arange(wide18.BASE[s])
            src[ci] = ci[P6._derangement(len(ci), seed=s)]
            perm.append(src)
        Ad = copy.deepcopy(A).to(DEV).eval()
        n = 0
        for a in range(0, len(X), BS):
            pts = P6._points(Ad, X[a:a + BS].to(DEV).float())
            for s, name, pt in readers:
                if not cut[s].any():
                    continue
                h = pts[pt]
                if mean == "perm":
                    h = h[:, perm[s].to(DEV)]
                hc = h * cut[s].to(DEV).view(1, -1, 1, 1)
                mod, _ = P6._mods(Ad, name)
                contrib[name] = contrib.get(name, 0) + P6._mean_out(P6._apply(mod, name, hc))
            n += len(pts["stem"])
        contrib = {k: (v / n).cpu() for k, v in contrib.items()}
        del Ad

    # ---- the cut (stream reads zeroed, inner conv2 columns zeroed) + the mean fold
    for s, name, pt in readers:
        mod, bn = P6._mods(S, name)
        mod.weight[:, cut[s]] = 0
        if name in contrib:
            if name == "fc":
                mod.bias += contrib[name]
            else:
                bn.running_mean -= contrib[name]
    for s in range(4):
        for b in range(2):
            L[s][b].conv2.weight[:, _mask(wide18.BASE[s], inn[s][b])] = 0

    # ---- P4 inner fold, closed loop, front to back on the student
    resid = []
    if fold != "off" and "inner" in sets:
        Ad = copy.deepcopy(A).to(DEV).eval()
        Sd = S.to(DEV).eval()
        Xd = X.to(DEV).float()
        nX = len(Xd)
        sls = P4._slices(nX)
        hT = torch.cat([P4._stem(Ad, Xd[sl]) for sl in sls])
        hS = torch.cat([P4._stem(Sd, Xd[sl]) for sl in sls])
        for s_ in range(4):
            for b in range(2):
                blT, blS = wide18.layers(Ad)[s_][b], wide18.layers(Sd)[s_][b]
                nU = wide18.BASE[s_]
                kept = inn[s_][b].cpu()
                kd = kept.to(DEV)
                cutc = _mask(nU, kept).nonzero().flatten().to(DEV)
                mom = P4._Mom(nU, len(kept))
                for sl in sls:
                    aT = P4._rows(P4._inner(blT, hT[sl]))
                    if fold == "null":
                        aT = aT.clone()
                        aT[:, cutc] = P4._rows(P4._inner(blT, hT[P4._roll(sl, nX).to(DEV)]))[:, cutc]
                    aS = P4._rows(P4._inner(blS, hS[sl]))[:, kd]
                    mom.add(aT, aS)
                G, lam = P4.fold_map(mom.Sx / mom.N, mom.Ss / mom.N, kept, mom.N, "slice")
                W2T = blT.conv2.weight                                    # (o, n, 3, 3), all rows
                Wnew = torch.einsum("onhw,nk->okhw", W2T.cpu().double(), G).float().to(DEV)
                blS.conv2.weight.zero_()
                blS.conv2.weight[:, kd] = Wnew
                resid.append(float((G - P4._ek(kept, nU)).norm() / max(G.norm(), 1e-30)))
                nT, nS = [], []
                for sl in sls:
                    nT.append(P4._out(blT, hT[sl], P4._inner(blT, hT[sl])))
                    nS.append(P4._out(blS, hS[sl], P4._inner(blS, hS[sl])))
                hT, hS = torch.cat(nT), torch.cat(nS)
        S = Sd.cpu().eval()
        del Ad, hT, hS, Xd
    if verbose:
        print(f"[r2_joint r18 f={f:g}] order={order} mean={mean} fold={fold} sets={sets} n_contrib={len(contrib)}"
              + (f" G_offslice mean {sum(resid) / len(resid):.3f}" if resid else ""), flush=True)
    if DEV == "mps":
        torch.mps.empty_cache()
    return S.eval()


def make(order="floor", mean="off", fold="off", sets=("stream", "inner")):
    def r18(A, f, X=None):
        return build_r18(A, f, X, order=order, mean=mean, fold=fold, sets=sets)
    return r18
