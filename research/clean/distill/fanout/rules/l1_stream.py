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
def fold_kslice(T, Wfull, Wcut, X, mk, hk=None, hfold=False, mfold=True, drift=()):
    """r3_joint fold_mlp with kslice: G's kept rows reset to the slice."""
    X = P4._x_vit(X)
    Td, Wd = vitgen.to(T, DEV), vitgen.to(Wcut, DEV)
    sls = P4._slices(len(X))
    emb = lambda Wx: torch.cat([vitgen.embed(Wx, X[sl].to(DEV).float()) for sl in sls])  # noqa: E731
    t, s = emb(Td), emb(Wd)
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        nh = BT["o_w"].shape[1]
        if hfold and hk is not None and len(_rows(hk[bi])) < nh:     # head fold, kslice form, on head output dims
            hr = _rows(hk[bi])
            Sx = torch.zeros(nh, len(hr), dtype=dt)
            Ss = torch.zeros(len(hr), len(hr), dtype=dt)
            N = 0
            for sl in sls:
                aT = P4._heads_fire(t[sl], BT).flatten(0, 1)
                aS = P4._heads_fire(s[sl], Bs).flatten(0, 1)
                Sx += (aT.T @ aS).cpu().double()
                Ss += (aS.T @ aS).cpu().double()
                N += aS.shape[0]
            Gh, lamh = P4L.fold_aff(Sx / N, Ss / N, hr.tolist(), N, False)
            Gh[hr] = P4._ek(hr.tolist(), nh)[hr].to(Gh.dtype)
            Wn = Wfull["blocks"][bi]["o_w"].double() @ Gh
            if "heads" in drift:
                Dx = torch.zeros(Wn.shape[0], len(hr), dtype=dt)
                for sl in sls:
                    aS = P4._heads_fire(s[sl], Bs).flatten(0, 1)
                    Dx += ((t[sl] - s[sl]).flatten(0, 1).T @ aS).cpu().double()
                Wn = Wn + torch.linalg.solve(Ss / N + lamh * torch.eye(len(hr), dtype=dt), (Dx / N).T).T
            Bs["o_w"] = Wn.float().to(DEV)
        for sl in sls:
            t[sl] = t[sl] + F.linear(P4._heads_fire(t[sl], BT), BT["o_w"], BT["o_b"])
            s[sl] = s[sl] + F.linear(P4._heads_fire(s[sl], Bs), Bs["o_w"], Bs["o_b"])
        kept = mk[bi]
        n, kk = BT["pr_w"].shape[1], len(kept)
        if mfold and kk < n:
            Sx = torch.zeros(n, kk, dtype=dt)
            Ss = torch.zeros(kk, kk, dtype=dt)
            N = 0
            for sl in sls:
                aT = P4._mlp_fire(t[sl], BT).flatten(0, 1)
                aS = P4._mlp_fire(s[sl], Bs).flatten(0, 1)
                Sx += (aT.T @ aS).cpu().double()
                Ss += (aS.T @ aS).cpu().double()
                N += aS.shape[0]
            G, lamm = P4L.fold_aff(Sx / N, Ss / N, kept, N, False)
            kt = torch.as_tensor(kept).long()
            G[kt] = P4._ek(kept, n)[kt].to(G.dtype)          # kslice
            Wn = Wfull["blocks"][bi]["pr_w"].double() @ G
            if "mlp" in drift:                               # kept writes also repair the stream error so far
                Dx = torch.zeros(Wn.shape[0], kk, dtype=dt)
                for sl in sls:
                    aS = P4._mlp_fire(s[sl], Bs).flatten(0, 1)
                    Dx += ((t[sl] - s[sl]).flatten(0, 1).T @ aS).cpu().double()
                Wn = Wn + torch.linalg.solve(Ss / N + lamm * torch.eye(kk, dtype=dt), (Dx / N).T).T
            Bs["pr_w"] = Wn.float().to(DEV)
        for sl in sls:
            t[sl] = t[sl] + F.linear(P4._mlp_fire(t[sl], BT), BT["pr_w"], BT["pr_b"])
            s[sl] = s[sl] + F.linear(P4._mlp_fire(s[sl], Bs), Bs["pr_w"], Bs["pr_b"])
    W = vitgen.to(Wd, "cpu")
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    return W


@torch.no_grad()
def root_order(T, X, f, heads_too=False, want_M=False):
    """Law 'keep the roots, delete the leaves': per block, keep the MLP neurons from which the others can be rebuilt.
    Greedy column selection (QR with column pivoting, LAPACK geqp3) on the write-weighted firing second moment
    M = D E[a a^T] D, a = teacher MLP firings on X (open loop), D = diag(|pr_w column|). Weights + X only."""
    import numpy as np
    import scipy.linalg as sla
    Td = vitgen.to(T, DEV)
    _, _, m = vitgen.sizes(f, T)
    nb = len(T["blocks"])
    M = [torch.zeros(T["blocks"][0]["pr_w"].shape[1], T["blocks"][0]["pr_w"].shape[1], dtype=dt) for _ in range(nb)]
    for a0 in range(0, len(X), P6.BS):
        x = X[a0:a0 + P6.BS].to(DEV).float()
        s = vitgen.embed(Td, x)
        for i, B in enumerate(Td["blocks"]):
            s = s + F.linear(P4._heads_fire(s, B), B["o_w"], B["o_b"])
            a = P4._mlp_fire(s, B).flatten(0, 1)
            M[i] += (a.T @ a).cpu().double()
            s = s + F.linear(P4._mlp_fire(s, B), B["pr_w"], B["pr_b"])
    out = []
    for i, B in enumerate(T["blocks"]):
        d = B["pr_w"].double().norm(dim=0)
        Mw = d[:, None] * M[i] * d[None]
        e, V = torch.linalg.eigh(Mw)
        Ft = (V * e.clamp_min(0).sqrt()).T.numpy()          # Mw = Ft^T Ft, columns = units
        _, _, piv = sla.qr(Ft, pivoting=True, mode="economic")
        out.append(torch.as_tensor(np.sort(piv[:m])).long())
    del Td
    if DEV == "mps":
        torch.mps.empty_cache()
    return (out, M) if want_M else out


@torch.no_grad()
def build(T, f, X=None, mode="law", fold=True, kfix=None, cut=("heads", "mlp"), hfold=False, morder="w", drift=(), blocks=None, tilt=0.0):
    X = P4._x_vit(X)
    _, hk, mk = J.keep_units(T, f, X, "act", "w", 0)
    Mmom = None
    if morder == "root":
        mk, Mmom = root_order(T, X, f, want_M=True)
    H0, M0 = T["blocks"][0]["o_w"].shape[1] // HD, T["blocks"][0]["pr_w"].shape[1]
    H, M = H0, M0
    if blocks is not None:                                    # cut only these blocks; the others stay full
        hk = [h if i in blocks else torch.arange(H) for i, h in enumerate(hk)]
        mk = [m if i in blocks else torch.arange(M) for i, m in enumerate(mk)]
    if "heads" not in cut:
        hk = [torch.arange(H)] * len(T["blocks"])
    if "mlp" not in cut:
        mk = [torch.arange(M)] * len(T["blocks"])
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
    if tilt:                                                  # identity law: a kept node's read tilts toward
        assert Mmom is not None and mode == "full"            # the reads of the deleted nodes it carries
        Wf = dict(Wf)
        Wf["blocks"] = [dict(B) for B in Wf["blocks"]]
        for i, B in enumerate(Wf["blocks"]):
            kept = torch.as_tensor(mk[i]).long()
            n = B["fc_w"].shape[0]
            if len(kept) == n:
                continue
            cutm = torch.ones(n, dtype=torch.bool)
            cutm[kept] = False
            cutc = cutm.nonzero().flatten()
            dg = Mmom[i].diagonal().clamp_min(1e-30).sqrt()
            Cs = Mmom[i] / dg[:, None] / dg[None]               # uncentred firing cosine on X
            car = kept[Cs[kept][:, cutc].argmax(0)]             # carrier of each deleted node
            fw, fb = B["fc_w"].clone(), B["fc_b"].clone()
            for d, j in zip(cutc.tolist(), car.tolist()):
                sc = float(B["fc_w"][j].norm() / B["fc_w"][d].norm().clamp_min(1e-12))
                fw[j] += tilt * sc * B["fc_w"][d]
                fb[j] += tilt * sc * B["fc_b"][d]
            B["fc_w"], B["fc_b"] = fw, fb
    W = J.cut_units(Wf, hk, mk)
    if fold:
        assert not drift or mode == "full"
        W = fold_kslice(T, Wf, W, X, mk, hk=hk, hfold=hfold, drift=drift)
    print(f"[l1_stream clip f={f:g}] mode={mode} fold={fold} kfix={kfix} cut={cut} hfold={hfold} morder={morder} drift={drift} tilt={tilt}", flush=True)
    return W


def make(**kw):
    def vit(T, f, X=None):
        return build(T, f, X, **kw)
    return vit
