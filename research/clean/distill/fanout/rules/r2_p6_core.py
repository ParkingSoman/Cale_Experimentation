"""P6 Stream, round 2 (FANOUT). Builder: P6, 2026-11-26.

Question: does keeping a k-dim SUBSPACE Q of the centred stream (not k coordinates) let stream-only 50% leave chance?
Object (picked by): the stream basis Q (D x k, orthonormal, orthogonal to the ones vector), one Q for the whole
network (P6 round-1 revised proposal: "the stream has no fixed basis, so the cut there is a projection").
Heads and MLP stay at full width. Input: generic X only (no photos in the calculation). Photos: scoring only.

Fold (as _p6diag_clip, with E replaced by Q): writers left-multiplied by P = U^T Q^T J, readers by R = Q U,
ln_pre handled by the SVD of Q^T J diag(g_pre) J, every norm becomes rms with divisor D_l, ln_post bias by least squares
on the output (exact at f = 1).

Bases:
  coord  the floor's kept coordinates (stream RMS order on X)  = round-1 D/rho rows (sanity)
  pca    top-k eigenvectors of sum_l M_l, M_l = second moment of the centred stream at site l on X (raw energy)
  pcan   same, each M_l divided by its trace (every site counts equally)
  rand   NULL: random orthonormal Q of the same rank (orthogonal to ones), seed 0
  read   reader-weighted: eigvecs of C^1/2 G C^1/2 mapped back by C^1/2, C = pooled trace-normalized moment,
         G = sum_l g_l W_l^T W_l g_l / s_l^2 (W_l = the rows that read site l, s_l = mean teacher divisor)
Divisor modes: D (768 everywhere), rho (768 * kept energy share at that site, teacher open loop).
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import vitgen  # noqa: E402
from rules import noisecut, _p6diag_clip  # noqa: E402

DEV = noisecut.DEV
BS = 32
_C = {}
dt = torch.float64


@torch.no_grad()
def site_moments(T, X):
    """Second moment of the centred stream at the 25 stream sites (ln1 x12, ln2 x12 in order, post = cls only),
    plus the mean squared teacher divisor per site. Teacher, full width, on X."""
    key = id(X) if X is not None else "pink"
    if _C.get("key") == key:
        return _C["m"]
    Xs = X if X is not None else _p6diag_clip._pink_default()
    Td = vitgen.to(T, DEV)
    D, nb = T["cls"].numel(), len(T["blocks"])
    M = torch.zeros(2 * nb + 1, D, D, device=DEV)
    s2 = torch.zeros(2 * nb + 1, device=DEV)
    ntok = ncls = 0
    for a in range(0, len(Xs), BS):
        x = Xs[a:a + BS].to(DEV).float()
        s = vitgen.embed(Td, x)

        def acc(j, v):
            c = (v - v.mean(-1, keepdim=True)).reshape(-1, D)
            M[j] += c.T @ c
            s2[j] += (c.pow(2).sum(-1) / D).sum()
        for i, B in enumerate(Td["blocks"]):
            acc(2 * i, s)
            s = s + vitgen.attn(vitgen.norm(s, B["ln1"]), B)
            acc(2 * i + 1, s)
            s = s + F.linear(F.gelu(F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])), B["pr_w"], B["pr_b"])
        acc(2 * nb, s[:, :1])
        ntok += x.shape[0] * s.shape[1]
        ncls += x.shape[0]
    n = torch.tensor([ntok] * (2 * nb) + [ncls], device=DEV, dtype=torch.float32)
    M = (M / n[:, None, None]).cpu().double()
    s2 = (s2 / n).cpu().double()
    m = dict(M=M, s2=s2)
    _C["key"], _C["m"] = key, m
    return m


def readers(T):
    """Per site l (same order as site_moments): G_l = diag(g) W^T W diag(g), W = the rows that read the norm output."""
    out = []
    for B in T["blocks"]:
        W1 = torch.cat([B["q_w"], B["k_w"], B["v_w"]]).double() * B["ln1"]["g"].double()[None]
        out.append(W1.T @ W1)
        W2 = B["fc_w"].double() * B["ln2"]["g"].double()[None]
        out.append(W2.T @ W2)
    Wp = (T["proj"].double() * T["ln_post"]["g"].double()[:, None]).T
    out.append(Wp.T @ Wp)
    return out


def _orth_centred(Q):
    D = Q.shape[0]
    J = torch.eye(D, dtype=dt) - 1.0 / D
    q, _ = torch.linalg.qr(J @ Q)
    return q


def _sym_sqrt(A):
    e, V = torch.linalg.eigh(A)
    return (V * e.clamp_min(0).sqrt()) @ V.T


def basis(T, X, k, kind, seed=0, plant=None):
    """plant = (u, lam): add lam * u u^T to every site moment (the planted unread high-variance direction)."""
    D = T["cls"].numel()
    if kind == "coord":
        st = _p6diag_clip._stats(T, X)
        sd = noisecut.top(st["score"], k)
        return _orth_centred(torch.eye(D, dtype=dt)[:, sd]), sd
    m = site_moments(T, X)
    M = m["M"].clone()
    if plant is not None:
        u, lam = plant
        M = M + lam * torch.outer(u, u)[None]
    if kind == "rand":
        g = torch.Generator().manual_seed(seed)
        return _orth_centred(torch.randn(D, k, generator=g, dtype=dt)), None
    if kind == "pca":
        C = M.sum(0)
    else:
        C = (M / M.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]).sum(0)
    if kind in ("pca", "pcan"):
        e, V = torch.linalg.eigh(C)
        return _orth_centred(V[:, e.argsort(descending=True)[:k]]), None
    if kind == "read":
        Gs = readers(T)
        if plant is not None:
            u = plant[0]
            Pu = torch.eye(D, dtype=dt) - torch.outer(u, u)
            Gs = [Pu @ G @ Pu for G in Gs]
        G = sum(Gl / Gl.trace() for Gl in Gs)          # each reader site counts equally
        Ch = _sym_sqrt(C)
        e, V = torch.linalg.eigh(Ch @ G @ Ch)
        V = V[:, e.argsort(descending=True)[:k]]
        return _orth_centred(Ch @ V), None
    raise ValueError(kind)


@torch.no_grad()
def build(T, f, X, kind, dmode="D", seed=0):
    D = T["cls"].numel()
    nb = len(T["blocks"])
    d, _, _ = vitgen.sizes(f, T)
    k = d
    Q, _ = basis(T, X, k, kind, seed)
    k = Q.shape[1]
    J = torch.eye(D, dtype=dt) - 1.0 / D
    g = lambda v: v.to(dt)                                 # noqa: E731
    Kp = Q.T @ J @ torch.diag(g(T["ln_pre"]["g"])) @ J
    U, S, Vh = torch.linalg.svd(Kp, full_matrices=False)
    V = Vh.T
    P = U.T @ Q.T @ J
    R = Q @ U
    m = site_moments(T, X)
    rho_sites = [float(torch.trace(Q.T @ m["M"][j] @ Q) / torch.trace(m["M"][j])) for j in range(2 * nb + 1)]
    # embed site share (patch embedding, through V)
    st = _p6diag_clip._stats(T, X)
    rho_e = float(torch.trace(V.T @ st["Ct"] @ V) / torch.trace(st["Ct"]))
    rho = [rho_e] + rho_sites
    Dl = [float(D)] * len(rho) if dmode == "D" else [D * r for r in rho]
    c32 = lambda v: v.float().contiguous()                 # noqa: E731
    ones, zeros = torch.ones(k), torch.zeros(k)
    rmsn = lambda Dv, b=zeros: dict(g=ones.clone(), b=c32(b), mode="rms", D=Dv)  # noqa: E731
    pw = g(T["patch_w"]).reshape(D, -1)
    Vj = V.T @ J
    W = dict(patch_w=c32((Vj @ pw).reshape(k, *T["patch_w"].shape[1:])), cls=c32(Vj @ g(T["cls"])),
             pos=c32(g(T["pos"]) @ Vj.T),
             ln_pre=dict(g=c32(S), b=c32(P @ g(T["ln_pre"]["b"])), mode="rms", D=Dl[0]), blocks=[])
    for i, B in enumerate(T["blocks"]):
        A1 = torch.diag(g(B["ln1"]["g"])) @ R
        A2 = torch.diag(g(B["ln2"]["g"])) @ R
        b1, b2 = g(B["ln1"]["b"]), g(B["ln2"]["b"])
        nb_ = dict(ln1=rmsn(Dl[1 + 2 * i]), ln2=rmsn(Dl[2 + 2 * i]))
        for n in ("q", "k", "v"):
            nb_[f"{n}_w"] = c32(g(B[f"{n}_w"]) @ A1)
            nb_[f"{n}_b"] = c32(g(B[f"{n}_b"]) + g(B[f"{n}_w"]) @ b1)
        nb_["o_w"], nb_["o_b"] = c32(P @ g(B["o_w"])), c32(P @ g(B["o_b"]))
        nb_["fc_w"], nb_["fc_b"] = c32(g(B["fc_w"]) @ A2), c32(g(B["fc_b"]) + g(B["fc_w"]) @ b2)
        nb_["pr_w"], nb_["pr_b"] = c32(P @ g(B["pr_w"])), c32(P @ g(B["pr_b"]))
        W["blocks"].append(nb_)
    gp, bp = g(T["ln_post"]["g"]), g(T["ln_post"]["b"])
    Mo = R.T @ torch.diag(gp) @ g(T["proj"])               # k x 512
    W["proj"] = c32(Mo)
    tgt = bp @ g(T["proj"])
    bpost = torch.linalg.lstsq(Mo.T, tgt[:, None]).solution[:, 0]
    W["ln_post"] = rmsn(Dl[-1], bpost)
    print(f"[r2p6 {kind}/{dmode}] f={f:g} k={k} rho mean {sum(rho)/len(rho):.3f} min {min(rho):.3f} "
          f"embed {rho_e:.3f} post {rho[-1]:.3f}", flush=True)
    return W
