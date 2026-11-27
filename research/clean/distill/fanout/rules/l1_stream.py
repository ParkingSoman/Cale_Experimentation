"""L1 test 4 (plan 2026-11-26 night, NODE.md): the stream size follows from the node cut (Shonu 2026-11-26).

Law: after the node cut, the stream keeps only the directions that KEPT nodes write AND KEPT nodes read.
No separate stream ranking, no separate stream error.

One CLIP calculation, three stream modes (everything else identical):
  full  node cut only. Stream at full width (768). The reference: "no extra error" means law ~= full.
  all   round-3 rG basis: Q = top-k eig of G^1/2 C G^1/2 with G from ALL readers and C from the FULL teacher
        stream (r2_p6_core, as in r3_joint_c_on). The off row (law v1: the stream ranked on its own).
  law   the same formula, with G from the KEPT readers only (q/k/v rows of kept heads, fc rows of kept MLP neurons,
        the readout) and C from the part of the teacher stream that KEPT writers made (embedding + kept heads'
        o_w writes + kept MLP pr_w writes, teacher firings, open loop on X). Each site trace-normalized, as rG.
  rand  NULL: random orthonormal basis of the same width.
Node cut (all modes): heads by floor activity on X, MLP by weight (r3_joint keep_units). Fold: the joint MLP fold
with kslice (kept units keep their own write; only cut units' messages fold), closed loop.
Object (picked by): every head, MLP neuron and stream direction of CLIP ViT-B/16 (kept sets by r3_joint keep_units).
Derived from: r3_joint_clip.py (build), r3_write_lib.py (kslice), r2_p6_core.py (rG).
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
from rules import r2_p6_core as P6  # noqa: E402
from rules import r3_joint_clip as J  # noqa: E402
from rules import p4write as P4  # noqa: E402
from rules import r2_p4_lib as P4L  # noqa: E402

DEV = J.DEV
HD = vitgen.HEAD_DIM
dt = torch.float64


def _rows(h):
    return torch.cat([torch.arange(i * HD, (i + 1) * HD) for i in h])


def kept_readers(T, hk, mk):
    out = []
    for B, h, m in zip(T["blocks"], hk, mk):
        r = _rows(h)
        W1 = torch.cat([B["q_w"][r], B["k_w"][r], B["v_w"][r]]).double() * B["ln1"]["g"].double()[None]
        out.append(W1.T @ W1)
        W2 = B["fc_w"][m].double() * B["ln2"]["g"].double()[None]
        out.append(W2.T @ W2)
    Wp = (T["proj"].double() * T["ln_post"]["g"].double()[:, None]).T
    out.append(Wp.T @ Wp)
    return out


@torch.no_grad()
def kept_written(T, X, hk, mk):
    """Second moment (centred over the D dims, as site_moments) of the KEPT-written stream at the 25 sites.
    Teacher firings on its own full stream (open loop); the kept stream = embedding + kept writes only."""
    Td = vitgen.to(T, DEV)
    D, nb = T["cls"].numel(), len(T["blocks"])
    M = torch.zeros(2 * nb + 1, D, D, device=DEV)
    ntok = ncls = 0
    rows = [_rows(h).to(DEV) for h in hk]
    mks = [torch.as_tensor(m).long().to(DEV) for m in mk]
    for a in range(0, len(X), P6.BS):
        x = X[a:a + P6.BS].to(DEV).float()
        s = vitgen.embed(Td, x)
        k = s.clone()

        def acc(j, v):
            c = (v - v.mean(-1, keepdim=True)).reshape(-1, D)
            M[j] += c.T @ c
        for i, B in enumerate(Td["blocks"]):
            acc(2 * i, k)
            hf = P4._heads_fire(s, B)                       # (b, t, D) head outputs before o_w
            s = s + F.linear(hf, B["o_w"], B["o_b"])
            k = k + F.linear(hf[..., rows[i]], B["o_w"][:, rows[i]], B["o_b"])
            acc(2 * i + 1, k)
            mf = P4._mlp_fire(s, B)
            s = s + F.linear(mf, B["pr_w"], B["pr_b"])
            k = k + F.linear(mf[..., mks[i]], B["pr_w"][:, mks[i]], B["pr_b"])
        acc(2 * nb, k[:, :1])
        ntok += x.shape[0] * s.shape[1]
        ncls += x.shape[0]
    n = torch.tensor([ntok] * (2 * nb) + [ncls], device=DEV, dtype=torch.float32)
    out = (M / n[:, None, None]).cpu().double()
    del Td
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def law_basis(T, X, k, hk, mk):
    Gs = kept_readers(T, hk, mk)
    G = sum(Gl / Gl.trace() for Gl in Gs)
    Mk = kept_written(T, X, hk, mk)
    C = (Mk / Mk.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]).sum(0)
    Gh = P6._sym_sqrt(G)
    e, V = torch.linalg.eigh(Gh @ C @ Gh)
    return P6._orth_centred(V[:, e.argsort(descending=True)[:k]]), None


@torch.no_grad()
def fold_kslice(T, Wfull, Wcut, X, mk):
    """r3_joint fold_mlp with kslice: G's kept rows reset to the slice."""
    X = P4._x_vit(X)
    Td, Wd = vitgen.to(T, DEV), vitgen.to(Wcut, DEV)
    sls = P4._slices(len(X))
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, X[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s = emb(Td), emb(Wd)
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        for sl in sls:
            t[sl] = t[sl] + F.linear(P4._heads_fire(t[sl], BT), BT["o_w"], BT["o_b"])
            s[sl] = s[sl] + F.linear(P4._heads_fire(s[sl], Bs), Bs["o_w"], Bs["o_b"])
        kept = mk[bi]
        n, kk = BT["pr_w"].shape[1], len(kept)
        if kk < n:
            Sx = torch.zeros(n, kk, dtype=dt)
            Ss = torch.zeros(kk, kk, dtype=dt)
            N = 0
            for sl in sls:
                aT = P4._mlp_fire(t[sl], BT).flatten(0, 1)
                aS = P4._mlp_fire(s[sl], Bs).flatten(0, 1)
                Sx += (aT.T @ aS).cpu().double()
                Ss += (aS.T @ aS).cpu().double()
                N += aS.shape[0]
            G, _ = P4L.fold_aff(Sx / N, Ss / N, kept, N, False)
            kt = torch.as_tensor(kept).long()
            G[kt] = P4._ek(kept, n)[kt].to(G.dtype)          # kslice
            Bs["pr_w"] = (Wfull["blocks"][bi]["pr_w"].double() @ G).float().to(DEV)
        for sl in sls:
            t[sl] = t[sl] + F.linear(P4._mlp_fire(t[sl], BT), BT["pr_w"], BT["pr_b"])
            s[sl] = s[sl] + F.linear(P4._mlp_fire(s[sl], Bs), Bs["pr_w"], Bs["pr_b"])
    W = vitgen.to(Wd, "cpu")
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    return W


@torch.no_grad()
def build(T, f, X=None, mode="law", fold=True, kfix=None):
    X = P4._x_vit(X)
    _, hk, mk = J.keep_units(T, f, X, "act", "w", 0)
    if mode == "full":
        Wf = T
    else:
        kind = dict(all="rG", law="LAW", rand="rand")[mode]
        old = P6.basis
        if mode == "law":
            P6.basis = lambda T_, X_, k_, kind_, seed=0, plant=None: law_basis(T_, X_, kfix or k_, hk, mk)  # noqa: E731
        try:
            Wf = P6.build(T, f, X, kind, "rho")
        finally:
            P6.basis = old
    W = J.cut_units(Wf, hk, mk)
    if fold:
        W = fold_kslice(T, Wf, W, X, mk)
    print(f"[l1_stream clip f={f:g}] mode={mode} fold={fold} kfix={kfix}", flush=True)
    return W


def make(**kw):
    def vit(T, f, X=None):
        return build(T, f, X, **kw)
    return vit
