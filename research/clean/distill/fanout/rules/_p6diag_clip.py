"""P6 Stream diagnostic rows, CLIP (FANOUT round 1, joint-order step 1). Builder: P6 builder, 2026-11-26.

Question: is the CLIP stream-only damage at 50% (floor 11.104) a SCALE / MEAN problem of the norms, not lost content?
Object (picked by): the norm divisor at every stream reading site (P6 theorist). The cut itself is the floor's:
the kept stream dims = noisecut's stream RMS order on the generic input X (top round(f * 768)). Heads and MLP stay at
full width. No photo is used in the calculation (X only). Photos: scoring only (board.py).

Modes (one builder, they differ ONLY in the divisor D_l of each norm):
  plain  floor baseline on X: vitgen.cut, LayerNorm over the kept dims (re-centres over the kept dims, divisor k).
  k      rms form, D_l = k at every site.
  D      rms form, D_l = 768 at every site.
  rho    rms form, D_l = 768 * rho_l, rho_l = kept share of the centred stream energy at site l on X (teacher, open loop).
  grho   NULL for rho: one global rho = arithmetic mean of the 26 rho_l, the same D_l at every site.

How the stream is centred (SliceGPT form, CALC section 1). LayerNorm(x) = g * Jx / sqrt(|Jx|^2 / 768 + eps) + b,
J = I - 11^T / 768. Every writer into the stream is left-multiplied by J (patch embedding, cls, pos, o_w, o_b, pr_w,
pr_b), so the stream holds Jx and needs no mean subtraction. Each norm's gain and bias fold into its readers
(q/k/v/fc weights and biases; ln_post into proj). Every norm becomes rms with g = 1, b = 0.
The one non-coordinate step: ln_pre's output g_pre * u + b_pre is not centred. Its centred form J diag(g_pre) J u is a
full matrix. The student keeps the floor's coordinates E (768 x k) of the centred stream, in a basis rotated INSIDE
the kept subspace: z = U^T E^T J x, from the SVD  E^T J diag(g_pre) J = U S V^T  (U k x k). A rotation inside the
kept subspace does not change what is kept, and every rms norm is rotation invariant. The student's ln_pre then reads
t_s = V^T J t (k dims) with gain S: exact at f = 1. At f < 1 the ln_pre normalizer reads |V^T J t| (the embed site's
own divisor, rho_e = kept share of the centred patch-embedding energy on X).
ln_post bias: carried as the floor carries it (the kept dims' share b_post[kept] @ proj[kept], exactly).
Fold algebra in float64 on CPU, cast to float32.

Implementation check (passes by algebra): f = 1 gives the teacher (KL < 1e-3, agree >= 99%) in every rms mode.
Prints per f: rho_l at every site, and the realized divisor ratio per site (teacher divisor / student divisor,
median over tokens, on the first 32 images of X, closed loop). 1.00 = the student reads at the teacher's scale.
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import generic  # noqa: E402
import vitgen  # noqa: E402
from rules import noisecut  # noqa: E402

DEV = noisecut.DEV
BS = 32
_C = {}


def _pink_default():
    """Same images as noisecut's default ViT input (batched draws, seeds 100000 + a) = the cached input."""
    return torch.cat([generic.pink(BS, seed=noisecut.SEED * 100000 + a) for a in range(0, noisecut.N_NOISE, BS)])


def _centred_energy(s):
    c = s - s.mean(-1, keepdim=True)
    return c.pow(2).reshape(-1, c.shape[-1]).sum(0)


@torch.no_grad()
def _stats(T, X):
    """Teacher on X, full width (does not depend on f). Floor stream order + per-site centred energy per dim
    (ln1 and ln2 inputs of every block, all tokens; ln_post input, class token) + centred patch-embedding covariance."""
    key = id(X) if X is not None else "pink-default"
    if _C.get("key") == key:
        return _C["st"]
    score = noisecut.vit_scores(T, X=X)["stream"]
    Xs = X if X is not None else _pink_default()
    Td = vitgen.to(T, DEV)
    D, nb = T["cls"].numel(), len(T["blocks"])
    Ct = torch.zeros(D, D, device=DEV)
    e1 = torch.zeros(nb, D, device=DEV)
    e2 = torch.zeros(nb, D, device=DEV)
    ep = torch.zeros(D, device=DEV)
    ntok = ncls = 0
    for a in range(0, len(Xs), BS):
        x = Xs[a:a + BS].to(DEV).float()
        t = F.conv2d(x, Td["patch_w"], stride=16).flatten(2).transpose(1, 2)
        t = torch.cat([Td["cls"].expand(len(x), 1, -1), t], 1) + Td["pos"]
        tc = (t - t.mean(-1, keepdim=True)).reshape(-1, D)
        Ct += tc.T @ tc
        s = vitgen.norm(t, Td["ln_pre"])
        for i, B in enumerate(Td["blocks"]):
            e1[i] += _centred_energy(s)
            s = s + vitgen.attn(vitgen.norm(s, B["ln1"]), B)
            e2[i] += _centred_energy(s)
            s = s + F.linear(F.gelu(F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])), B["pr_w"], B["pr_b"])
        ep += _centred_energy(s[:, :1])
        ntok += tc.shape[0]
        ncls += len(x)
    st = dict(score=score, Ct=(Ct / ntok).cpu().double(), e1=(e1 / ntok).cpu().double(), e2=(e2 / ntok).cpu().double(),
              ep=(ep / ncls).cpu().double(), x32=Xs[:BS].float(), Td=Td)
    _C["key"], _C["st"] = key, st
    return st


def _site_inputs(W, x):
    """Yield (pre-norm input, norm dict) at the 26 stream reading sites of a vitgen dict: ln_pre, ln1/ln2 x 12, ln_post."""
    t = F.conv2d(x, W["patch_w"], stride=16).flatten(2).transpose(1, 2)
    t = torch.cat([W["cls"].expand(len(x), 1, -1), t], 1) + W["pos"]
    yield t, W["ln_pre"]
    s = vitgen.norm(t, W["ln_pre"])
    for B in W["blocks"]:
        yield s, B["ln1"]
        s = s + vitgen.attn(vitgen.norm(s, B["ln1"]), B)
        yield s, B["ln2"]
        s = s + F.linear(F.gelu(F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])), B["pr_w"], B["pr_b"])
    yield s[:, :1], W["ln_post"]


def _den(x, n):
    """The per-token divisor the norm applies."""
    if n["mode"] == "ln":
        return torch.sqrt(x.var(-1, correction=0) + vitgen.EPS)
    return torch.sqrt(x.pow(2).sum(-1) / n["D"] + vitgen.EPS)


@torch.no_grad()
def _realized_ratio(Td, W, x):
    Wd, x = vitgen.to(W, DEV), x.to(DEV)
    out = []
    for (xt, nt), (xs, ns) in zip(_site_inputs(Td, x), _site_inputs(Wd, x)):
        out.append(float((_den(xt, nt) / _den(xs, ns)).float().cpu().median()))
    return out


def _fmt(v):
    return " ".join(f"{a:.3f}" for a in v)


@torch.no_grad()
def build(T, f, X, mode):
    assert mode in ("plain", "k", "D", "rho", "grho")
    st = _stats(T, X)
    D = T["cls"].numel()
    d, _, _ = vitgen.sizes(f, T)
    sd = noisecut.top(st["score"], d)                          # the floor's kept stream dims (sorted)
    k = len(sd)
    nb = len(T["blocks"])
    allh = [torch.arange(B["q_w"].shape[0] // vitgen.HEAD_DIM) for B in T["blocks"]]
    allm = [torch.arange(B["fc_w"].shape[0]) for B in T["blocks"]]

    if mode == "plain":
        W = vitgen.cut(T, sd, allh, allm)
        rho = None
    else:
        dt = torch.float64
        J = torch.eye(D, dtype=dt) - 1.0 / D
        E = torch.eye(D, dtype=dt)[:, sd]                      # D x k
        g = lambda v: v.to(dt)                                 # noqa: E731
        Kp = E.T @ J @ torch.diag(g(T["ln_pre"]["g"])) @ J      # k x D
        U, S, Vh = torch.linalg.svd(Kp, full_matrices=False)  # U k x k, S k, Vh k x D
        V = Vh.T                                               # D x k
        assert U.shape == (k, k) and V.shape == (D, k) and S.shape == (k,)
        P = U.T @ E.T @ J                                      # writer map, k x D
        R = E @ U                                              # reader map, D x k (kept coordinates, z basis)

        # rho per site: embed, ln1 x nb, ln2 x nb, ln_post (teacher, open loop, on X)
        rho_e = float(torch.trace(V.T @ st["Ct"] @ V) / torch.trace(st["Ct"]))
        rho1 = [float(st["e1"][i][sd].sum() / st["e1"][i].sum()) for i in range(nb)]
        rho2 = [float(st["e2"][i][sd].sum() / st["e2"][i].sum()) for i in range(nb)]
        rho_p = float(st["ep"][sd].sum() / st["ep"].sum())
        rho = [rho_e] + [v for pair in zip(rho1, rho2) for v in pair] + [rho_p]   # site order = _site_inputs order
        rglob = sum(rho) / len(rho)
        if mode == "k":
            Dl = [float(k)] * len(rho)
        elif mode == "D":
            Dl = [float(D)] * len(rho)
        elif mode == "rho":
            Dl = [D * r for r in rho]
        else:
            Dl = [D * rglob] * len(rho)
        c32 = lambda v: v.float().contiguous()                 # noqa: E731
        ones, zeros = torch.ones(k), torch.zeros(k)
        rmsn = lambda Dv, b=zeros: dict(g=ones.clone(), b=c32(b), mode="rms", D=Dv)  # noqa: E731

        pw = g(T["patch_w"]).reshape(D, -1)
        Vj = V.T @ J                                           # k x D
        W = dict(patch_w=c32((Vj @ pw).reshape(k, *T["patch_w"].shape[1:])), cls=c32(Vj @ g(T["cls"])),
                 pos=c32(g(T["pos"]) @ Vj.T),
                 ln_pre=dict(g=c32(S), b=c32(P @ g(T["ln_pre"]["b"])), mode="rms", D=Dl[0]), blocks=[])
        for i, B in enumerate(T["blocks"]):
            A1 = torch.diag(g(B["ln1"]["g"])) @ R              # D x k
            A2 = torch.diag(g(B["ln2"]["g"])) @ R
            b1, b2 = g(B["ln1"]["b"]), g(B["ln2"]["b"])
            nb_ = dict(ln1=rmsn(Dl[1 + 2 * i]), ln2=rmsn(Dl[2 + 2 * i]))
            for n in ("q", "k", "v"):
                nb_[f"{n}_w"] = c32(g(B[f"{n}_w"]) @ A1)
                nb_[f"{n}_b"] = c32(g(B[f"{n}_b"]) + g(B[f"{n}_w"]) @ b1)
            nb_["o_w"], nb_["o_b"] = c32(P @ g(B["o_w"])), c32(P @ g(B["o_b"]))
            nb_["fc_w"], nb_["fc_b"] = c32(g(B["fc_w"]) @ A2), c32(g(B["fc_b"]) + g(B["fc_w"]) @ b2)
            nb_["pr_w"], nb_["pr_b"] = c32(P @ g(B["pr_w"])), c32(P @ g(B["pr_b"]))
            assert nb_["q_w"].shape == (B["q_w"].shape[0], k) and nb_["o_w"].shape == (k, B["o_w"].shape[1])
            assert nb_["fc_w"].shape == (B["fc_w"].shape[0], k) and nb_["pr_w"].shape == (k, B["pr_w"].shape[1])
            W["blocks"].append(nb_)
        gp, bp = g(T["ln_post"]["g"]), g(T["ln_post"]["b"])
        assert float(gp[sd].abs().min()) > 1e-8, "ln_post gain has a zero on a kept dim"
        W["proj"] = c32(R.T @ torch.diag(gp) @ g(T["proj"]))   # k x 512
        bpost = U.T @ (bp[sd] / gp[sd])                        # (bpost + zhat) @ proj' = floor's kept-share bias, exactly
        W["ln_post"] = rmsn(Dl[-1], bpost)
        assert W["proj"].shape == (k, T["proj"].shape[1]) and W["pos"].shape == (T["pos"].shape[0], k)

    ratio = _realized_ratio(st["Td"], W, st["x32"])
    print(f"[p6diag clip {mode}] f={f:g} k={k}", flush=True)
    if rho is not None:
        print(f"[p6diag clip {mode}]   rho_l (embed, ln1/ln2 x12, post): {_fmt(rho)}  mean {sum(rho) / len(rho):.3f}"
              f"  min {min(rho):.3f}  max {max(rho):.3f}", flush=True)
    print(f"[p6diag clip {mode}]   realized divisor ratio teacher/student per site (1 = teacher scale): {_fmt(ratio)}",
          flush=True)
    if DEV == "mps":
        torch.mps.empty_cache()
    return W
