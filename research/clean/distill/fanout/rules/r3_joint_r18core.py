"""JOINT round 3 copy of r2_joint_core.py with ONE new switch, fc (2026-11-26):
  fc "pool" = round 2 (fc fold regresses the POOLED stage-4 stream: 256 rows for 436-487 unknowns, ridge lam = tr/N);
     "pos"  = the same regression on every spatial position of the stage-4 stream (256 x 49 rows). fc reads the pooled
              mean, and a linear map commutes with the mean, so the per-position G applies to the pooled read.
     "off"  = no fold at fc (the cut fc columns are just dropped).
JOINT (FANOUT round 2): the parts that gained in round 1, put together in one ResNet-18 calculation, front to back.

Object (picked by): every stream channel and inner channel of ResNet-18 (all of them; the order part picks which stay).
Parts and switches (each switch sets ONE part back to its floor):
  order  "floor" = noisecut / wide18.channel_scores on X (the floor);  "p5" = P5 read-aware greedy order
         (rules/p5rank.py, cached order on the same X);  "p5null" = P5's read-permuted null order.
  mean   "off" = floor;  "mean" = P6 mean fold of the cut STREAM channels into every reader's bias
         (rules/_p6diag_r18.py, teacher stream, open loop);  "perm" = its derangement null (same form).
  fold   "off" = floor;  "on" = P4 reader-matched unit fold of the INNER set into conv2 (rules/p4write.py, A = 0,
         slice anchor, closed loop on the student that already has the stream cut + mean fold);
         "null" = P4's shuffled-pairing null (same form); "aff" = the same fold with a constant unit (bias into bn2).
  sfold  "off" = floor;  "on" = STREAM FOLD, closed loop, front to back: at every reader of a cut stream point
         (block conv1, downsample, fc), regress the teacher's full stream (n ch) on [the student's kept channels, 1]
         (p4write.fold_map, slice anchor, the constant unit added), W' = W G on the kept columns, the constant part
         into the reader's BN running_mean (fc: bias). Excludes mean (it contains the mean fold as its constant).
         "null" = the same with the cut channels' teacher values taken from the next image (shuffled pairing).
         Derived from: P4's stream fold A (round-1 proposal), switched ON here against the round-1 critic's "A = 0"
         verdict (A conflicted with P6 read maps on CLIP; this is ResNet, coordinate stream, no P6 basis).
         Honest status: no prediction was written before its first run, and it has no planted test. Its evidence
         is its same-form null only (full 1,000: beats the null by .070 +- .005 KL at 95%, .578 +- .025 at 85%,
         paired on the 800 photos outside the quick set).
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
def _sfold_G(aTs, aSs, kept, n):
    """Stream fold map with a constant unit: regress the teacher stream (n ch) on [kept student ch, 1]."""
    mom = P4._Mom(n + 1, len(kept) + 1)
    for aT, aS in zip(aTs, aSs):
        one = torch.ones(aT.shape[0], 1, device=aT.device)
        mom.add(torch.cat([aT, one], 1), torch.cat([aS, one], 1))
    ka = torch.cat([torch.as_tensor(kept).long(), torch.tensor([n])])
    G, _ = P4.fold_map(mom.Sx / mom.N, mom.Ss / mom.N, ka, mom.N, "slice")
    return G[:n, :len(kept)], G[:n, len(kept)]


def _sfold_conv(convT, bnT, convS, bnS, Gk, gc, kd):
    W = convT.weight.cpu().double()
    convS.weight.zero_()
    convS.weight[:, kd] = torch.einsum("onhw,nk->okhw", W, Gk).float().to(convS.weight.device)
    bnS.running_mean.copy_(bnT.running_mean - (W.sum((2, 3)) @ gc).float().to(bnS.running_mean.device))


def build_r18(A, f, X=None, order="floor", mean="off", fold="off", sets=("stream", "inner"), verbose=True, sfold="off", fc="pool"):
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
    assert sfold == "off" or mean == "off"
    if (fold != "off" and "inner" in sets) or sfold != "off":
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
                ss = s_ if (b == 1 or s_ == 0) else s_ - 1
                if sfold != "off" and cut[ss].any():                     # stream fold of this block's readers
                    ks = st[ss].cpu()
                    aTs, aSs = [], []
                    for sl in sls:
                        aT = P4._rows(hT[sl])
                        if sfold == "null":
                            aT = aT.clone()
                            cc = cut[ss].to(DEV)
                            aT[:, cc] = P4._rows(hT[P4._roll(sl, nX).to(DEV)])[:, cc]
                        aTs.append(aT)
                        aSs.append(P4._rows(hS[sl])[:, ks.to(DEV)])
                    Gk, gc = _sfold_G(aTs, aSs, ks, wide18.BASE[ss])
                    del aTs, aSs
                    _sfold_conv(blT.conv1, blT.bn1, blS.conv1, blS.bn1, Gk, gc, ks.to(DEV))
                    if b == 0 and s_ > 0:
                        _sfold_conv(blT.downsample[0], blT.downsample[1], blS.downsample[0], blS.downsample[1], Gk, gc, ks.to(DEV))
                if not (fold != "off" and "inner" in sets):
                    nT, nS = [], []
                    for sl in sls:
                        nT.append(P4._out(blT, hT[sl], P4._inner(blT, hT[sl])))
                        nS.append(P4._out(blS, hS[sl], P4._inner(blS, hS[sl])))
                    hT, hS = torch.cat(nT), torch.cat(nS)
                    continue
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
                if fold == "aff":                                         # affine inner fold: + a constant unit
                    aTs = [P4._rows(P4._inner(blT, hT[sl])) for sl in sls]
                    aSs = [P4._rows(P4._inner(blS, hS[sl]))[:, kd] for sl in sls]
                    G, gc = _sfold_G(aTs, aSs, kept, nU)
                    del aTs, aSs
                    _sfold_conv(blT.conv2, blT.bn2, blS.conv2, blS.bn2, G, gc, kd)
                else:
                    G, lam = P4.fold_map(mom.Sx / mom.N, mom.Ss / mom.N, kept, mom.N, "slice")
                    W2T = blT.conv2.weight                                # (o, n, 3, 3), all rows
                    Wnew = torch.einsum("onhw,nk->okhw", W2T.cpu().double(), G).float().to(DEV)
                    blS.conv2.weight.zero_()
                    blS.conv2.weight[:, kd] = Wnew
                resid.append(float((G - P4._ek(kept, nU)).norm() / max(G.norm(), 1e-30)))
                nT, nS = [], []
                for sl in sls:
                    nT.append(P4._out(blT, hT[sl], P4._inner(blT, hT[sl])))
                    nS.append(P4._out(blS, hS[sl], P4._inner(blS, hS[sl])))
                hT, hS = torch.cat(nT), torch.cat(nS)
        if sfold != "off" and fc != "off" and cut[3].any():               # fc reads the pooled stage-4 stream
            ks = st[3].cpu()
            if fc == "pos":
                pT = torch.cat([hT[sl].permute(0, 2, 3, 1).reshape(-1, hT.shape[1]) for sl in sls])
                pS = torch.cat([hS[sl].permute(0, 2, 3, 1).reshape(-1, hS.shape[1]) for sl in sls])
            else:
                pT = torch.cat([hT[sl].mean((2, 3)) for sl in sls])
                pS = torch.cat([hS[sl].mean((2, 3)) for sl in sls])
            if sfold == "null":
                cc = cut[3].to(DEV)
                pT = pT.clone()
                pT[:, cc] = torch.roll(pT, -(hT.shape[2] * hT.shape[3] if fc == "pos" else 1), 0)[:, cc]   # next image
            Gk, gc = _sfold_G([pT], [pS[:, ks.to(DEV)]], ks, 512)
            Wf = Ad.fc.weight.cpu().double()
            Sd.fc.weight.zero_()
            Sd.fc.weight[:, ks.to(DEV)] = (Wf @ Gk).float().to(DEV)
            Sd.fc.bias.copy_(Ad.fc.bias + (Wf @ gc).float().to(DEV))
        S = Sd.cpu().eval()
        del Ad, hT, hS, Xd
    if verbose:
        print(f"[r2_joint r18 f={f:g}] order={order} mean={mean} fold={fold} sfold={sfold} fc={fc} sets={sets} n_contrib={len(contrib)}"
              + (f" G_offslice mean {sum(resid) / len(resid):.3f}" if resid else ""), flush=True)
    if DEV == "mps":
        torch.mps.empty_cache()
    return S.eval()


def make(order="floor", mean="off", fold="off", sets=("stream", "inner"), sfold="off", fc="pool"):
    def r18(A, f, X=None):
        return build_r18(A, f, X, order=order, mean=mean, fold=fold, sets=sets, sfold=sfold, fc=fc)
    return r18
