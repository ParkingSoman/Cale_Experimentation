"""P6 round 2: PER-SITE stream bases with a k x k map on every skip link (SliceGPT form, CALC section 1), CLIP.
Object (picked by): one basis Q_j per stream site j (25 sites: ln1 x12, ln2 x12, post), P6 round-1 proposal step 2.
Each sublayer writes into the NEXT site's basis (P_j = Q_j^T J); the skip carries S_j = P_j R_{j-1} (k x k, extra
parameters: 24 maps, counted in n_params). Readers at site j use R_j = Q_j (site 0: Q_0 U from the ln_pre SVD).
Divisor: rms, D_j = 768 * kept energy share at site j (teacher, open loop, on X). Input X only. Photos: scoring only.
Kinds: pca (top-k eigvecs of the site's own centred 2nd moment), rand (NULL: random Q_j of the same rank, per site).
Needs its own forward (vitgen has no skip map): forward(W, x) below. NOT a board.py rule; scored by r2_p6_run.py.
"""
import torch
import torch.nn.functional as F

from rules import r2_p6_core as core, _p6diag_clip
import vitgen

dt = torch.float64


def site_basis(M, k, kind, j, seed=0):
    D = M.shape[0]
    if kind == "rand":
        g = torch.Generator().manual_seed(seed * 1000 + j)
        return core._orth_centred(torch.randn(D, k, generator=g, dtype=dt))
    e, V = torch.linalg.eigh(M)
    return core._orth_centred(V[:, e.argsort(descending=True)[:k]])


@torch.no_grad()
def build(T, f, X, kind="pca"):
    D, nb = T["cls"].numel(), len(T["blocks"])
    k = vitgen.sizes(f, T)[0]
    m = core.site_moments(T, X)
    Qs = [site_basis(m["M"][j], k, kind, j) for j in range(2 * nb + 1)]
    J = torch.eye(D, dtype=dt) - 1.0 / D
    g = lambda v: v.to(dt)                                 # noqa: E731
    c32 = lambda v: v.float().contiguous()                 # noqa: E731
    Kp = Qs[0].T @ J @ torch.diag(g(T["ln_pre"]["g"])) @ J
    U, S, Vh = torch.linalg.svd(Kp, full_matrices=False)
    V = Vh.T
    Rs = [Qs[0] @ U] + Qs[1:]
    Ps = [U.T @ Qs[0].T @ J] + [Q.T @ J for Q in Qs[1:]]
    rho = [float(torch.trace(Q.T @ m["M"][j] @ Q) / torch.trace(m["M"][j])) for j, Q in enumerate(Qs)]
    st = _p6diag_clip._stats(T, X)
    rho_e = float(torch.trace(V.T @ st["Ct"] @ V) / torch.trace(st["Ct"]))
    ones, zeros = torch.ones(k), torch.zeros(k)
    rmsn = lambda Dv, b=zeros: dict(g=ones.clone(), b=c32(b), mode="rms", D=Dv)  # noqa: E731
    pw = g(T["patch_w"]).reshape(D, -1)
    Vj = V.T @ J
    W = dict(patch_w=c32((Vj @ pw).reshape(k, *T["patch_w"].shape[1:])), cls=c32(Vj @ g(T["cls"])),
             pos=c32(g(T["pos"]) @ Vj.T), ln_pre=dict(g=c32(S), b=c32(Ps[0] @ g(T["ln_pre"]["b"])), mode="rms", D=D * rho_e),
             blocks=[], skip=[])
    for i, B in enumerate(T["blocks"]):
        j1, j2, j3 = 2 * i, 2 * i + 1, 2 * i + 2
        A1 = torch.diag(g(B["ln1"]["g"])) @ Rs[j1]
        A2 = torch.diag(g(B["ln2"]["g"])) @ Rs[j2]
        b1, b2 = g(B["ln1"]["b"]), g(B["ln2"]["b"])
        nb_ = dict(ln1=rmsn(D * rho[j1]), ln2=rmsn(D * rho[j2]))
        for n in ("q", "k", "v"):
            nb_[f"{n}_w"] = c32(g(B[f"{n}_w"]) @ A1)
            nb_[f"{n}_b"] = c32(g(B[f"{n}_b"]) + g(B[f"{n}_w"]) @ b1)
        nb_["o_w"], nb_["o_b"] = c32(Ps[j2] @ g(B["o_w"])), c32(Ps[j2] @ g(B["o_b"]))
        nb_["fc_w"], nb_["fc_b"] = c32(g(B["fc_w"]) @ A2), c32(g(B["fc_b"]) + g(B["fc_w"]) @ b2)
        nb_["pr_w"], nb_["pr_b"] = c32(Ps[j3] @ g(B["pr_w"])), c32(Ps[j3] @ g(B["pr_b"]))
        W["blocks"].append(nb_)
        W["skip"].append((c32(Ps[j2] @ Rs[j1]), c32(Ps[j3] @ Rs[j2])))
    gp, bp = g(T["ln_post"]["g"]), g(T["ln_post"]["b"])
    Mo = Rs[-1].T @ torch.diag(gp) @ g(T["proj"])
    W["proj"] = c32(Mo)
    bpost = torch.linalg.lstsq(Mo.T, (bp @ g(T["proj"]))[:, None]).solution[:, 0]
    W["ln_post"] = rmsn(D * rho[-1], bpost)
    print(f"[r2p6 persite {kind}] f={f:g} k={k} rho mean {sum(rho)/len(rho):.3f} min {min(rho):.3f} embed {rho_e:.3f}", flush=True)
    return W


def forward(W, x):
    s = vitgen.embed(W, x)
    for B, (S1, S2) in zip(W["blocks"], W["skip"]):
        s = s @ S1.T + vitgen.attn(vitgen.norm(s, B["ln1"]), B)
        s = s @ S2.T + F.linear(F.gelu(F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])), B["pr_w"], B["pr_b"])
    return vitgen.norm(s[:, 0], W["ln_post"]) @ W["proj"]


def n_params(W):
    return vitgen.n_params(W) + sum(a.numel() + b.numel() for a, b in W["skip"])
