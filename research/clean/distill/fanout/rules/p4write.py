"""P4 WRITE, round 1 build (2026-11-26). The kept writes carry the cut units' message, solved from the student's own
firings on the generic input X. A = 0 (no stream fold). Stream cut = the floor's slice.

Source: round1/talk.json turn2 "P4 Write" (revised_proposal, interface); critic verdict "build, with A = 0".
Object (picked by): every write matrix of a set this row cuts. CLIP: o_w and pr_w of each block. ResNet: conv2 of each
block. Kept units = the floor's sets (rules/noisecut scores on the same X).

THE FOLD (one closed form per writer site, front to back on the student)
  a  = teacher firings at the site, all n units, on the TEACHER's stream       (CLIP heads: the 64H head outputs before
  a' = student firings at the site, k kept units, on the STUDENT's own stream   o_w; MLP: gelu; ResNet: relu(bn1 conv1))
  S_x = E[a a'^T] (n x k), S_s = E[a' a'^T] (k x k), uncentred, pooled over images x positions, float64 on the CPU.
  lam = tr(S_s) / N_samples, N_samples = images x positions, read from the firing tensor (CLIP 256 x 197 = 50,432;
        the proposal's "x 50" token count is wrong for ViT-B/16 at 224, which has 197 tokens).
  G   = (S_x + lam E_K)(S_s + lam I)^-1  (n x k), E_K[kept_j, j] = 1   ("slice" anchor, the default)
  W'  = W G (rows on the kept stream), bias unchanged.
  Derived from: min over W' of E||W a - W' a'||^2 + lam ||W' - W E_K||^2. The ridge pulls toward the floor's plain
  slice, not toward 0. This is the form for which the proposal's own identities hold: with a' = a_K it gives
  W' = W_K + W_C P, P = S_CK (S_KK + lam I)^-1 (the proposal's stated special case); f = 1 gives W' = W; closed loop
  equals open loop when nothing upstream changed. The proposal's literal general form W S_x (S_s + lam I)^-1 ("zero"
  anchor, switch anchor="zero") also shrinks the KEPT writes by S_KK (S_KK + lam I)^-1, which breaks those identities
  and makes the output less sure (the critic's calibration confound). It is kept only as a switch.
  M (the reader metric) cancels from G: the solution is the same for any metric on the rows of W. So no reader
  weights, no M, no order from the end. The M = I check is NOT APPLICABLE with A = 0.

SWITCHES  sets: which sets are cut (and folded); fold=False: the plain slice through this code path (control);
          null=True: shuffled pairing (the cut units' teacher firings come from the NEXT image, same positions; the
          kept-to-kept pairing stays true). In expectation = the position-wise mean-only fold (calibration control).
          loop="open": a' = the teacher's kept firings (one switch row; used by the implementation check).

READOUTS (saved to out/p4write_<tag>_f<f>_n<N>.json, one line printed): per site lam, k, n, the message residual on X
  (err_floor = plain slice with the student's own firings, err_rule = the fold, both / the message energy;
  err_rule <= err_floor holds by algebra for the slice anchor in closed loop and is asserted), CLIP sigma'/sigma at
  every LN site for the rule's student and for the floor's student (same X).

PHOTO USE: none. Weights + the generic input X only. ResNet BatchNorm enters only as the network's own inference map.
Compute: Mac, MPS float32 forwards, float64 CPU solves. All images' streams are held on the device (CLIP 3 x 155 MB).
"""
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
import generic  # noqa: E402
import vitgen  # noqa: E402
from rules import noisecut  # noqa: E402

DEV = noisecut.DEV
BS = 32
HD = vitgen.HEAD_DIM
OUT = os.path.join(FAN, "out")
SLACK = 1e-4          # residual assert slack, as a fraction of the message energy (float32 forwards)
_C = {}               # per-process cache: default inputs and floor scores, keyed by the input object


# ---------------------------------------------------------------- the fold (generic, any architecture)
def fold_map(Sx, Ss, kept, n_samples, anchor="slice", lam=None):
    """G (n x k) with W' = W G. Sx = E[a a'^T] (n x k), Ss = E[a' a'^T] (k x k), float64.
    kept[j] = the teacher unit that student column j is. lam None = tr(Ss) / n_samples."""
    k = Ss.shape[0]
    lam = float(torch.trace(Ss)) / n_samples if lam is None else float(lam)
    R = Sx.clone()
    if anchor == "slice":
        R[torch.as_tensor(kept).long(), torch.arange(k)] += lam
    elif anchor != "zero":
        raise ValueError(anchor)
    Areg = Ss + lam * torch.eye(k, dtype=Ss.dtype)
    G = torch.linalg.solve(Areg, R.T).T            # Areg symmetric: R Areg^-1
    return G, lam


class _Mom:
    def __init__(self, n, k):
        self.Sx = torch.zeros(n, k, dtype=torch.float64)
        self.Ss = torch.zeros(k, k, dtype=torch.float64)
        self.N = 0

    def add(self, aT, aS):
        """aT (rows, n), aS (rows, k) on the device, float32. Per-batch products, summed in float64 on the CPU."""
        self.Sx += (aT.T @ aS).cpu().double()
        self.Ss += (aS.T @ aS).cpu().double()
        self.N += aS.shape[0]


def _slices(n):
    return [slice(a, min(a + BS, n)) for a in range(0, n, BS)]


def _roll(sl, n):
    """null pairing: image i takes the cut-unit firings of image i + 1 (no fixed point)."""
    return (torch.arange(sl.start, sl.stop) + 1) % n


def _resid_add(acc, aT, aS, Dfull, DK, DW, chunk=16384):
    """Message residual on X, row chunks: teacher message Dfull a, floor DK a', rule DW a'."""
    for a in range(0, aT.shape[0], chunk):
        mT = aT[a:a + chunk] @ Dfull.T
        acc["msg"] += float(mT.pow(2).sum())
        acc["rule"] += float((mT - aS[a:a + chunk] @ DW.T).pow(2).sum())
        acc["floor"] += float((mT - aS[a:a + chunk] @ DK.T).pow(2).sum())


def _check_resid(name, acc, strict):
    msg = acc["msg"]
    rec = dict(msg=msg, err_floor=acc["floor"] / max(msg, 1e-30), err_rule=acc["rule"] / max(msg, 1e-30))
    if strict and acc["rule"] > acc["floor"] + SLACK * msg:
        raise RuntimeError(f"p4write: residual check failed at {name}: rule {rec['err_rule']:.6f} > floor "
                           f"{rec['err_floor']:.6f} (orientation or index map is wrong)")
    return rec


def _save(info, tag, f, n):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"p4write_{tag}_f{f:g}_n{n}.json")
    json.dump(info, open(p, "w"), indent=1)
    return p


def _summary(info):
    s = [r for r in info["sites"] if "err_rule" in r]
    if not s:
        return ""
    ef = sum(r["err_floor"] for r in s) / len(s)
    er = sum(r["err_rule"] for r in s) / len(s)
    out = f"mean message residual on X: floor {ef:.4f} -> rule {er:.4f} over {len(s)} sites"
    sg = [r for r in info["sites"] if "sigma_rule" in r]
    if sg:
        out += (f"; sigma'/sigma mean over LN sites: floor {sum(r['sigma_floor'] for r in sg) / len(sg):.3f}, "
                f"rule {sum(r['sigma_rule'] for r in sg) / len(sg):.3f}")
    return out


# ---------------------------------------------------------------- CLIP (vitgen dict)
def _x_vit(X):
    if X is not None:
        return X
    if "vitX" not in _C:   # noisecut's default draw, batched seeds 100000 + a
        _C["vitX"] = torch.cat([generic.pink(BS, seed=noisecut.SEED * 100000 + a) for a in range(0, noisecut.N_NOISE, BS)])
    return _C["vitX"]


def _vit_scores(T, X):
    key = ("vit", id(X))
    if key not in _C:
        _C[key] = (X, noisecut.vit_scores(T, X=X))
    return _C[key][1]


def _head_rows(hk):
    return torch.cat([torch.arange(h * HD, (h + 1) * HD) for h in hk])


def _heads_fire(x, B):
    """The heads' outputs before o_w, (N, L, 64h), columns head-major as in vitgen.attn."""
    y = vitgen.norm(x, B["ln1"])
    N, L, _ = y.shape
    h = B["q_w"].shape[0] // HD
    q = F.linear(y, B["q_w"], B["q_b"]).view(N, L, h, HD).transpose(1, 2)
    k = F.linear(y, B["k_w"], B["k_b"]).view(N, L, h, HD).transpose(1, 2)
    v = F.linear(y, B["v_w"], B["v_b"]).view(N, L, h, HD).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, L, h * HD)


def _mlp_fire(x, B):
    return F.gelu(F.linear(vitgen.norm(x, B["ln2"]), B["fc_w"], B["fc_b"]))


def _sigma(x):
    return torch.sqrt(x.var(-1, unbiased=False) + vitgen.EPS)


@torch.no_grad()
def build_vit(T, f, X=None, sets=("stream", "heads", "mlp"), fold=True, null=False, loop="closed", anchor="slice",
              tag="full", keep_G=(), save=True):
    """Returns (student dict on the CPU, info). T is never changed."""
    t0 = time.time()
    X = _x_vit(X)
    sc = _vit_scores(T, X)
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    sd = noisecut.top(sc["stream"], d if "stream" in sets else D)
    hk = [noisecut.top(s, h if "heads" in sets else H) for s in sc["head"]]
    mk = [noisecut.top(s, m if "mlp" in sets else M) for s in sc["mlp"]]
    info = dict(rule="p4write", tag=tag, f=f, sets=list(sets), fold=fold, null=null, loop=loop, anchor=anchor,
                input=dict(n=len(X), sum=float(X.float().sum())), photo_use="none (weights + generic input X)", sites=[])
    if not fold:
        return vitgen.cut(T, sd, hk, mk), info
    Td = vitgen.to(T, DEV)
    Wd = vitgen.to(vitgen.cut(T, sd, hk, mk), DEV)       # the student (its writes get folded)
    W0d = vitgen.to(vitgen.cut(T, sd, hk, mk), DEV)      # the floor (plain slice), for the readouts only
    nX = len(X)
    sls = _slices(nX)
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, X[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s, s0 = emb(Td), emb(Wd), emb(W0d)
    sd_dev = sd.to(DEV)
    Gs = {}
    for bi in range(len(T["blocks"])):
        BT, Bs, B0 = Td["blocks"][bi], Wd["blocks"][bi], W0d["blocks"][bi]
        for kind in ("heads", "mlp"):
            fire, wk, bk = (_heads_fire, "o_w", "o_b") if kind == "heads" else (_mlp_fire, "pr_w", "pr_b")
            name = f"b{bi}.{kind}"
            n = BT[wk].shape[1]
            kept = _head_rows(hk[bi]) if kind == "heads" else mk[bi]
            k = len(kept)
            kept_dev = kept.to(DEV)
            do = kind in sets
            rec = dict(site=name, n=n, k=k, folded=do)
            Dsd = BT[wk][sd_dev]                                  # teacher write, kept stream rows (d x n)
            if do:
                cutc = torch.ones(n, dtype=torch.bool)
                cutc[kept] = False
                cutc = cutc.nonzero().flatten().to(DEV)            # integer index of the cut units
                mom = _Mom(n, k)
                for sl in sls:
                    aT = fire(t[sl], BT).flatten(0, 1)
                    if null:
                        aT = aT.clone()
                        aT[:, cutc] = fire(t[_roll(sl, nX).to(DEV)], BT).flatten(0, 1)[:, cutc]
                    aS = aT[:, kept_dev] if loop == "open" else fire(s[sl], Bs).flatten(0, 1)
                    mom.add(aT, aS)
                G, lam = fold_map(mom.Sx / mom.N, mom.Ss / mom.N, kept, mom.N, anchor)
                Bs[wk] = (Dsd.cpu().double() @ G).float().to(DEV)
                rec.update(lam=lam, n_samples=mom.N, G_offslice=float((G - _ek(kept, n)).norm() / max(G.norm(), 1e-30)))
                if name in keep_G:
                    Gs[name] = G
            acc = dict(msg=0.0, floor=0.0, rule=0.0)
            sig = dict(r=0.0, z=0.0, c=0)
            DK = B0[wk]
            for sl in sls:
                sT = _sigma(t[sl])
                sig["r"] += float((_sigma(s[sl]) / sT).sum())
                sig["z"] += float((_sigma(s0[sl]) / sT).sum())
                sig["c"] += sT.numel()
                aT, aS, a0 = fire(t[sl], BT), fire(s[sl], Bs), fire(s0[sl], B0)
                if do:
                    _resid_add(acc, aT.flatten(0, 1), aS.flatten(0, 1), Dsd, DK, Bs[wk])
                t[sl] = t[sl] + F.linear(aT, BT[wk], BT[bk])
                s[sl] = s[sl] + F.linear(aS, Bs[wk], Bs[bk])
                s0[sl] = s0[sl] + F.linear(a0, B0[wk], B0[bk])
            rec.update(ln="ln1" if kind == "heads" else "ln2", sigma_rule=sig["r"] / sig["c"], sigma_floor=sig["z"] / sig["c"])
            if do:
                rec.update(_check_resid(name, acc, strict=(not null and loop == "closed" and anchor == "slice")))
            info["sites"].append(rec)
    sT = _sigma(t[:, 0])
    info["sites"].append(dict(site="ln_post", ln="ln_post", sigma_rule=float((_sigma(s[:, 0]) / sT).mean()),
                              sigma_floor=float((_sigma(s0[:, 0]) / sT).mean())))
    W = vitgen.to(Wd, "cpu")
    info["seconds"] = round(time.time() - t0, 1)
    del Td, Wd, W0d, t, s, s0
    if DEV == "mps":
        torch.mps.empty_cache()
    if save:
        info["saved"] = _save(info, tag, f, nX)
    print(f"[p4write {tag} f={f:g}] {_summary(info)}; {info['seconds']} s", flush=True)
    info["G"] = Gs
    return W, info


def _ek(kept, n):
    E = torch.zeros(n, len(kept), dtype=torch.float64)
    E[torch.as_tensor(kept).long(), torch.arange(len(kept))] = 1.0
    return E


def vit(T, f, X=None):
    return build_vit(T, f, X, sets=("stream", "heads", "mlp"), tag="full")[0]


# ---------------------------------------------------------------- ResNet-18 (torchvision module)
def _x_r18(X):
    if X is not None:
        return X
    if "r18X" not in _C:   # noisecut's default draw for ResNet
        _C["r18X"] = generic.pink(noisecut.N_NOISE, seed=noisecut.SEED)
    return _C["r18X"]


def _stem(mdl, x):
    return mdl.maxpool(mdl.relu(mdl.bn1(mdl.conv1(x))))


def _inner(bl, h):
    return torch.relu(bl.bn1(bl.conv1(h)))


def _out(bl, h, i):
    skip = h if bl.downsample is None else bl.downsample(h)
    return torch.relu(bl.bn2(bl.conv2(i)) + skip)


def _rows(a):
    return a.permute(0, 2, 3, 1).reshape(-1, a.shape[1])


@torch.no_grad()
def build_r18(A, f, X=None, sets=("stream", "inner"), fold=True, null=False, loop="closed", anchor="slice",
              tag="full", keep_G=(), save=True):
    """Returns (student module on the CPU, eval, info). A is never changed (a device copy is used)."""
    import wide18
    t0 = time.time()
    X = _x_r18(X)
    key = ("r18", id(X))
    Ad = copy.deepcopy(A).to(DEV).eval()
    if key not in _C:
        Xd = X.to(DEV).float()
        _C[key] = (X, Xd, wide18.channel_scores(Ad, Xd, bs=BS))
    _, Xd, sc = _C[key]
    w = wide18.widths(f)
    if "stream" in sets:
        S, kp = wide18.cut_from_a(Ad, f, sc)
        inn, st = kp["inner"], kp["stream"]
    else:
        S = copy.deepcopy(Ad)
        inn = [[wide18.keep_idx(sc["inner"][s_][b], w[s_]) for b in range(2)] for s_ in range(4)]
        st = [torch.arange(c) for c in wide18.BASE]
        for s_ in range(4):                                # the floor: cut inner channel = zero conv2 column
            for b in range(2):
                keep = torch.zeros(wide18.BASE[s_])
                keep[inn[s_][b]] = 1.0
                wide18.layers(S)[s_][b].conv2.weight.mul_(keep.to(DEV).view(1, -1, 1, 1))
    S = S.eval()
    info = dict(rule="p4write", tag=tag, f=f, sets=list(sets), fold=fold, null=null, loop=loop, anchor=anchor,
                input=dict(n=len(X), sum=float(X.float().sum())), photo_use="none (weights + generic input X)",
                resid_metric="per tap, same position (the fold's own objective)", sites=[])
    if not fold or "inner" not in sets:
        del Ad
        return S.cpu().eval(), info
    nX = len(Xd)
    sls = _slices(nX)
    hT = torch.cat([_stem(Ad, Xd[sl]) for sl in sls])
    hS = torch.cat([_stem(S, Xd[sl]) for sl in sls])
    Gs = {}
    for s_ in range(4):
        for b in range(2):
            blT, blS = wide18.layers(Ad)[s_][b], wide18.layers(S)[s_][b]
            name = f"layer{s_ + 1}.{b}"
            n = wide18.BASE[s_]
            kept = inn[s_][b].cpu()
            k = len(kept)
            kept_dev = kept.to(DEV)
            selS = (torch.arange(k) if "stream" in sets else kept).to(DEV)   # student conv1 columns = the kept units
            W2T = blT.conv2.weight[st[s_].to(DEV)]                           # (o, n, 3, 3) teacher write, kept rows
            cutc = torch.ones(n, dtype=torch.bool)
            cutc[kept] = False
            cutc = cutc.nonzero().flatten().to(DEV)                          # integer index of the cut units
            mom = _Mom(n, k)
            for sl in sls:
                aT = _rows(_inner(blT, hT[sl]))
                if null:
                    aT = aT.clone()
                    aT[:, cutc] = _rows(_inner(blT, hT[_roll(sl, nX).to(DEV)]))[:, cutc]
                aS = aT[:, kept_dev] if loop == "open" else _rows(_inner(blS, hS[sl]))[:, selS]
                mom.add(aT, aS)
            G, lam = fold_map(mom.Sx / mom.N, mom.Ss / mom.N, kept, mom.N, anchor)
            Wnew = torch.einsum("onhw,nk->okhw", W2T.cpu().double(), G).float().to(DEV)
            if "stream" in sets:
                blS.conv2.weight.copy_(Wnew)
            else:
                blS.conv2.weight.zero_()
                blS.conv2.weight[:, selS] = Wnew
            if name in keep_G:
                Gs[name] = G
            rec = dict(site=name, n=n, k=k, lam=lam, n_samples=mom.N,
                       G_offslice=float((G - _ek(kept, n)).norm() / max(G.norm(), 1e-30)))
            Dfl = W2T.permute(0, 2, 3, 1).reshape(-1, n)                    # (o*9, n): one row per (out, tap)
            DK = Dfl[:, kept_dev]
            DW = Wnew.permute(0, 2, 3, 1).reshape(-1, k)
            acc = dict(msg=0.0, floor=0.0, rule=0.0)
            nT, nS = [], []
            for sl in sls:
                iT, iS = _inner(blT, hT[sl]), _inner(blS, hS[sl])
                _resid_add(acc, _rows(iT), _rows(iS)[:, selS], Dfl, DK, DW)
                nT.append(_out(blT, hT[sl], iT))
                nS.append(_out(blS, hS[sl], iS))
            hT, hS = torch.cat(nT), torch.cat(nS)
            rec.update(_check_resid(name, acc, strict=(not null and loop == "closed" and anchor == "slice")))
            info["sites"].append(rec)
    S = S.cpu().eval()
    info["seconds"] = round(time.time() - t0, 1)
    del Ad, hT, hS
    if DEV == "mps":
        torch.mps.empty_cache()
    if save:
        info["saved"] = _save(info, tag, f, nX)
    print(f"[p4write {tag} f={f:g}] {_summary(info)}; {info['seconds']} s", flush=True)
    info["G"] = Gs
    return S, info


def r18(A, f, X=None):
    return build_r18(A, f, X, sets=("stream", "inner"), tag="full")[0]
