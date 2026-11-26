"""P6 Stream diagnostic rows, ResNet-18 (FANOUT round 1, joint-order step 1). Builder: P6 builder, 2026-11-26.

Question: is the ResNet stream-only damage at 50% (floor 18.051) a MEAN problem (the cut channels' positive mean is
lost from every reader, BatchNorm stays fixed), not lost content?
Object (picked by): the bias of every reader of a cut stream channel (P6 theorist). The cut is the floor's stream-only
cut as in _split.py: stream order = wide18.channel_scores on the generic input X (top round(f * width) per stage),
inner channels at full width, a cut stream channel = every read of it set to 0 (block conv1 columns, downsample
columns, fc column). Params reported = the full network's (masked weights stay in the tensors).

Modes:
  plain  baseline on X: the cut, no fold (same as split_stream, order from X).
  mean   mean-only fold: each reader's bias gets the MEAN of the cut channels' contribution at its own output on X,
         c_r = mean over images and positions of conv_r(x restricted to the cut channels), teacher stream, open loop.
         The zero-padded borders are inside that mean (no weight x channel-mean shortcut).
         Conv readers: c_r folds into the following BatchNorm as running_mean -= c_r (an edit of the inference map,
         never read as a statistic). fc: bias += c_r.
  perm   NULL for mean: the same computation with the cut channels' activations permuted among the cut channels of
         the stage (a derangement: no channel keeps its own, seed 0 + stage).
Readers and the stream point each one reads (teacher, on X):
  stage s block 1 conv1 <- block 0 output of stage s;  stage s+1 block 0 conv1 and downsample <- block 1 output of
  stage s;  stage 1 block 0 conv1 <- stem output;  fc <- avgpool of the stage-4 block 1 output.
No photo is used in the calculation (X only). Photos: scoring only (board.py).

Implementation checks: f = 1 cuts nothing, so KL 0 there is vacuous (no fold runs). In-rule check (passes by
algebra, mode mean): on the teacher's stream, each folded reader (conv with cut columns zeroed + edited BN, or fc)
has the same output mean as the teacher's reader. Printed as the max abs difference over all readers.
"""
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import generic  # noqa: E402
from rules import noisecut  # noqa: E402

DEV = noisecut.DEV
BS = 32
_C = {}


def _input(X):
    return X if X is not None else generic.pink(noisecut.N_NOISE, seed=noisecut.SEED)


def _scores(A, X):
    import wide18
    key = id(X) if X is not None else "pink-default"
    if _C.get("key") != key:
        Ad = copy.deepcopy(A).to(DEV).eval()
        with torch.no_grad():
            _C["sc"] = wide18.channel_scores(Ad, _input(X).to(DEV).float(), bs=BS)
        _C["key"] = key
        del Ad
    return _C["sc"]


def _readers(s):
    """(name, point) for every reader of stage s's stream set. name = (stage, block, 'conv1' | 'ds') or 'fc'."""
    out = [((s, 1, "conv1"), (s, 0))]
    if s < 3:
        out += [((s + 1, 0, "conv1"), (s, 1)), ((s + 1, 0, "ds"), (s, 1))]
    if s == 0:
        out += [((0, 0, "conv1"), "stem")]
    if s == 3:
        out += [("fc", (3, 1))]
    return out


def _mods(M, name):
    import wide18
    if name == "fc":
        return M.fc, None
    s, b, kind = name
    bl = wide18.layers(M)[s][b]
    return (bl.conv1, bl.bn1) if kind == "conv1" else (bl.downsample[0], bl.downsample[1])


def _points(Ad, x):
    """Teacher forward, yields every stream point once: 'stem', (s, b)."""
    import wide18
    h = Ad.maxpool(Ad.relu(Ad.bn1(Ad.conv1(x))))
    pts = {"stem": h}
    for s, layer in enumerate(wide18.layers(Ad)):
        for b, bl in enumerate(layer):
            i = torch.relu(bl.bn1(bl.conv1(h)))
            skip = h if bl.downsample is None else bl.downsample(h)
            h = torch.relu(bl.bn2(bl.conv2(i)) + skip)
            pts[(s, b)] = h
    return pts


def _apply(mod, name, xin):
    """The reader's linear map (no BN): conv module, or fc weight on the avgpooled input."""
    if name == "fc":
        return xin.mean((2, 3)) @ mod.weight.T
    return mod(xin)


def _mean_out(y):
    return y.sum(0) if y.dim() == 2 else y.mean((2, 3)).sum(0)


def _derangement(n, seed):
    if n < 2:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    o = torch.randperm(n, generator=g)
    p = torch.empty_like(o)
    p[o] = torch.roll(o, -1)
    return p


@torch.no_grad()
def build(A, f, X, mode):
    import wide18
    assert mode in ("plain", "mean", "perm")
    sc = _scores(A, X)
    w = wide18.widths(f)
    S = copy.deepcopy(A).cpu().eval()
    cut, perm = [], []
    for s in range(4):
        keep = wide18.keep_idx(sc["stream"][s], w[s])
        m = torch.ones(wide18.BASE[s], dtype=torch.bool)
        m[keep] = False
        cut.append(m)
        ci = torch.nonzero(m).flatten()
        src = torch.arange(wide18.BASE[s])
        src[ci] = ci[_derangement(len(ci), seed=s)]               # perm: cut channel c takes the activation of src[c]
        perm.append(src)
    readers = [(s, name, pt) for s in range(4) for name, pt in _readers(s)]

    contrib = {}
    if mode != "plain" and any(bool(c.any()) for c in cut):
        Ad = copy.deepcopy(A).to(DEV).eval()
        x = _input(X)
        n = 0
        ptmean = {}
        for a in range(0, len(x), BS):
            pts = _points(Ad, x[a:a + BS].to(DEV).float())
            for key, h in pts.items():
                ptmean[key] = ptmean.get(key, 0) + h.mean((2, 3)).sum(0)
            for s, name, pt in readers:
                if not cut[s].any():
                    continue
                h = pts[pt]
                if mode == "perm":
                    h = h[:, perm[s].to(DEV)]
                hc = h * cut[s].to(DEV).view(1, -1, 1, 1)
                mod, _ = _mods(Ad, name)
                contrib[name] = contrib.get(name, 0) + _mean_out(_apply(mod, name, hc))
            n += len(pts["stem"])
        contrib = {k: (v / n).cpu() for k, v in contrib.items()}
        ptmean = {k: (v / n).cpu() for k, v in ptmean.items()}
        for s in range(4):
            if cut[s].any():
                pm = ptmean[(s, 1)]
                print(f"[p6diag r18 {mode}] f={f:g} stage {s + 1}: block-1 output channel mean on X, cut {float(pm[cut[s]].mean()):.4f}"
                      f" kept {float(pm[~cut[s]].mean()):.4f} (n cut {int(cut[s].sum())})", flush=True)

    for s, name, pt in readers:                                  # the cut (as _split.py) + the fold
        mod, bn = _mods(S, name)
        mod.weight[:, cut[s]] = 0
        if name in contrib:
            if name == "fc":
                mod.bias += contrib[name]
            else:
                bn.running_mean -= contrib[name]

    if mode == "mean" and contrib:                               # in-rule check, passes by algebra
        Ad = copy.deepcopy(A).to(DEV).eval()
        Sd = copy.deepcopy(S).to(DEV).eval()
        x = _input(X)
        dt, dsd, n = {}, {}, 0
        for a in range(0, len(x), BS):
            pts = _points(Ad, x[a:a + BS].to(DEV).float())
            for s, name, pt in readers:
                if name not in contrib:
                    continue
                for M, acc in ((Ad, dt), (Sd, dsd)):
                    mod, bn = _mods(M, name)
                    y = _apply(mod, name, pts[pt])
                    y = y + mod.bias if name == "fc" else bn(y)
                    acc[name] = acc.get(name, 0) + _mean_out(y)
            n += len(pts["stem"])
        diff = max(float((dt[k] - dsd[k]).abs().max()) / n for k in dt)
        scale = max(float(dt[k].abs().max()) / n for k in dt)
        print(f"[p6diag r18 mean] f={f:g} CHECK (algebra): max |folded reader mean - teacher reader mean| = {diff:.2e}"
              f" (teacher reader mean scale {scale:.2e})", flush=True)
    if DEV == "mps":
        torch.mps.empty_cache()
    return S.eval()
