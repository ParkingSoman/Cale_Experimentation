"""JOINT round 3 (FANOUT, 2026-11-26): the ONE calculation for the whole CLIP vision tower, as one rule.

Object (picked by): every stream direction, head and MLP neuron of the CLIP ViT-B/16 vision tower.
Parts, front to back (each has a switch; the floor value of each switch is the leave-one-out row):
  stream  "rG"    P6 round 2: one reader-weighted subspace Q = top-k eig of G^1/2 C G^1/2, divisor rho (r2_p6_core).
                  C from the input named by cx ("x" = the board input X, "self" = cache/clip_self_n256_k32_s1.pt).
          "floor" the floor's kept coordinates (noisecut stream RMS on X), plain slice, LayerNorm over kept dims.
  heads   "act"   floor activity on X (noisecut head score).  "w" = magcut weight score.
  mlp     "w"     magcut weight score (P5 r2_p5_w0).  "act" = floor activity.  "rand" = seeded random order (NULL).
  fold    True    P4 reader-matched write fold on the MLP writes (fold_aff, slice anchor, closed loop on the student,
                  front to back). The student write = (the stream part's full write, n columns) @ G.
          "mean"  P4's same-form mean null (cut firings replaced by their per-position mean over images).
          False   no fold.
  post    True    READOUT fold (round 3, the ResNet fc analogue): closed loop, after the MLP fold. Regress the teacher's
                  output ln_post(x_T) @ proj on [the student's rms-normed stream, 1], on EVERY token of X (256 x 197
                  rows; tokens of one image are not independent rows), ridge 1e-3 * mean eigenvalue. proj <- the map,
                  ln_post bias <- lstsq(map^T, constant). False = P6's readout (exact at f = 1).
Heads and MLP are cut AFTER the stream part, so the stream basis is computed with full heads and MLP (P6 as promoted).
Derived from: round 2 best rows (BOARD "What round 2 settled": P5 w0, P4 wfold_mlp, P6 rG_rho).
Photo use: none. Weights + generic input(s) only. Photos: scoring only (board.py).
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
from rules import magcut, noisecut, r2_p6_core  # noqa: E402
from rules import p4write as P4  # noqa: E402
from rules import r2_p4_lib as P4L  # noqa: E402

DEV = noisecut.DEV
HD = vitgen.HEAD_DIM
_C = {}
SELF = os.path.join(FAN, "cache", "clip_self_n256_k32_s1.pt")


def _input(name, X):
    if name == "x":
        return X
    if name not in _C:
        path = dict(self=SELF, null=os.path.join(FAN, "cache", "clip_null_n256_k32_s1.pt"))[name]
        _C[name] = torch.load(path, weights_only=True).float()
    return _C[name]


def _rand(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g)


def keep_units(T, f, X, heads="act", mlp="w", seed=0):
    d, h, m = vitgen.sizes(f, T)
    act = P4._vit_scores(T, X)
    wsc = magcut.vit_scores(T)
    hs = act["head"] if heads == "act" else wsc["head"]
    if mlp == "w":
        ms = wsc["mlp"]
    elif mlp == "act":
        ms = act["mlp"]
    elif mlp == "rand":
        ms = [_rand(len(s), 1000 * (seed + 1) + i) for i, s in enumerate(wsc["mlp"])]
    else:
        raise ValueError(mlp)
    return act["stream"], [magcut.top(s, h) for s in hs], [magcut.top(s, m) for s in ms]


def stream_part(T, f, X, stream, cx):
    """Full heads and MLP; the stream cut by the chosen form. Returns the dict."""
    if stream == "floor":
        d, _, _ = vitgen.sizes(f, T)
        D, H, M = vitgen.sizes(1.0, T)
        sd = noisecut.top(P4._vit_scores(T, X)["stream"], d)
        return vitgen.cut(T, sd, [torch.arange(H)] * len(T["blocks"]), [torch.arange(M)] * len(T["blocks"]))
    return r2_p6_core.build(T, f, _input(cx, X), stream, "rho")


def cut_units(W, hk, mk):
    """Slice heads and MLP units only; the stream (and every norm) stays as it is."""
    out = dict(W)
    out["blocks"] = []
    for B, h, m in zip(W["blocks"], hk, mk):
        rows = torch.cat([torch.arange(i * HD, (i + 1) * HD) for i in h])
        nb = dict(B)
        for k in ("q_w", "k_w", "v_w", "q_b", "k_b", "v_b"):
            nb[k] = B[k][rows]
        nb["o_w"] = B["o_w"][:, rows]
        nb["fc_w"], nb["fc_b"], nb["pr_w"] = B["fc_w"][m], B["fc_b"][m], B["pr_w"][:, m]
        out["blocks"].append(nb)
    return out


@torch.no_grad()
def fold_mlp(T, Wfull, Wcut, X, mk, null=None, mlp=True, post=False):
    """P4 fold on the MLP writes. Teacher T full; student Wcut (closed loop, front to back).
    Student write = Wfull pr_w (stream_k x n, the stream part's write of every teacher unit) @ G (n x k)."""
    X = P4._x_vit(X)
    Td, Wd = vitgen.to(T, DEV), vitgen.to(Wcut, DEV)
    nX = len(X)
    sls = P4._slices(nX)
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, X[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s = emb(Td), emb(Wd)
    offs = []
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        for sl in sls:
            t[sl] = t[sl] + F.linear(P4._heads_fire(t[sl], BT), BT["o_w"], BT["o_b"])
            s[sl] = s[sl] + F.linear(P4._heads_fire(s[sl], Bs), Bs["o_w"], Bs["o_b"])
        kept = mk[bi]
        n, k = BT["pr_w"].shape[1], len(kept)
        if mlp and k < n:
            cutc = torch.ones(n, dtype=torch.bool)
            cutc[kept] = False
            cutc = cutc.nonzero().flatten().to(DEV)
            mu = None
            if null == "mean":
                mu = sum(P4._mlp_fire(t[sl], BT)[:, :, cutc].sum(0) for sl in sls) / nX
            Sx = torch.zeros(n, k, dtype=torch.float64)
            Ss = torch.zeros(k, k, dtype=torch.float64)
            N = 0
            for sl in sls:
                aT = P4._mlp_fire(t[sl], BT)
                if mu is not None:
                    aT = aT.clone()
                    aT[:, :, cutc] = mu.unsqueeze(0).expand(aT.shape[0], -1, -1)
                aS = P4._mlp_fire(s[sl], Bs)
                aT, aS = aT.flatten(0, 1), aS.flatten(0, 1)
                Sx += (aT.T @ aS).cpu().double()
                Ss += (aS.T @ aS).cpu().double()
                N += aS.shape[0]
            G, lam = P4L.fold_aff(Sx / N, Ss / N, kept, N, False)
            Bs["pr_w"] = (Wfull["blocks"][bi]["pr_w"].double() @ G).float().to(DEV)
            offs.append(float((G - P4._ek(kept, n)).norm() / G.norm()))
        for sl in sls:
            t[sl] = t[sl] + F.linear(P4._mlp_fire(t[sl], BT), BT["pr_w"], BT["pr_b"])
            s[sl] = s[sl] + F.linear(P4._mlp_fire(s[sl], Bs), Bs["pr_w"], Bs["pr_b"])
    if post:
        kd = s.shape[-1]
        Szz = torch.zeros(kd + 1, kd + 1, dtype=torch.float64)
        Szy = torch.zeros(kd + 1, Td["proj"].shape[1], dtype=torch.float64)
        for sl in sls:
            y = (vitgen.norm(t[sl], Td["ln_post"]) @ Td["proj"]).flatten(0, 1)
            z = s[sl] * torch.rsqrt((s[sl] * s[sl]).sum(-1, keepdim=True) / Wd["ln_post"]["D"] + vitgen.EPS)
            z = torch.cat([z.flatten(0, 1), torch.ones_like(z.flatten(0, 1)[:, :1])], 1)
            Szz += (z.T @ z).cpu().double()
            Szy += (z.T @ y).cpu().double()
        lam = 1e-3 * float(torch.trace(Szz)) / (kd + 1)
        beta = torch.linalg.solve(Szz + lam * torch.eye(kd + 1, dtype=torch.float64), Szy)
        Mo, c = beta[:kd], beta[kd]
        b = torch.linalg.lstsq(Mo.T, c[:, None]).solution[:, 0]
        Wd["proj"] = Mo.float().to(DEV)
        Wd["ln_post"] = dict(g=torch.ones(kd, device=DEV), b=b.float().to(DEV), mode="rms", D=Wd["ln_post"]["D"])
    W = vitgen.to(Wd, "cpu")
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    return W, sum(offs) / max(len(offs), 1)


@torch.no_grad()
def build(T, f, X=None, stream="rG", cx="x", heads="act", mlp="w", fold=True, seed=0, post=False):
    X = P4._x_vit(X)
    _, hk, mk = keep_units(T, f, X, heads, mlp, seed)
    Wf = stream_part(T, f, X, stream, cx)
    W = cut_units(Wf, hk, mk)
    off = None
    if fold or post:
        W, off = fold_mlp(T, Wf, W, X, mk, null="mean" if fold == "mean" else None, mlp=bool(fold), post=post)
    print(f"[r3_joint clip f={f:g}] stream={stream} cx={cx} heads={heads} mlp={mlp} fold={fold} post={post} "
          f"G_offslice {off}", flush=True)
    return W


def make(**kw):
    def vit(T, f, X=None):
        return build(T, f, X, **kw)
    return vit
