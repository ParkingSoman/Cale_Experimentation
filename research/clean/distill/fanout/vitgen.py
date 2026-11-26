"""Generic ViT (CLIP vision tower) as a plain dict of tensors, any width per block (FANOUT plan 2026-11-26).

A calculation rule returns such a dict. `forward(W, x)` runs it. With the teacher's own weights it must give the
teacher's output (board.py --rule identity checks this: KL 0).

Dict layout (d = stream width, H = heads kept in a block, m = MLP neurons kept in a block; head dim stays 64):
  patch_w (d, 3, 16, 16)   cls (d,)   pos (197, d)   ln_pre: norm
  blocks[i]: ln1: norm, q_w/k_w/v_w (64H, d), q_b/k_b/v_b (64H,), o_w (d, 64H), o_b (d,),
             ln2: norm, fc_w (m, d), fc_b (m,), pr_w (d, m), pr_b (d,)
  ln_post: norm   proj (d, 512)
  norm = dict(g (d,), b (d,), mode "ln" | "rms", D)   "ln": LayerNorm over the d kept dims (open_clip's own form).
         "rms": x * g / sqrt(sum(x^2) / D + eps) + b, for a stream that is already centred (SliceGPT form);
         D = the divisor (the teacher's width when a projection keeps the norm's scale).
Calculation code may import this file. It must not import any photo loader.
"""
import torch
import torch.nn.functional as F

HEAD_DIM = 64
EPS = 1e-5


def from_open_clip(visual):
    """Teacher (or any open_clip ViT) -> dict. float32, CPU."""
    g = lambda t: t.detach().float().cpu().clone()  # noqa: E731
    nrm = lambda ln: dict(g=g(ln.weight), b=g(ln.bias), mode="ln", D=ln.weight.shape[0])  # noqa: E731
    W = dict(patch_w=g(visual.conv1.weight), cls=g(visual.class_embedding), pos=g(visual.positional_embedding),
             ln_pre=nrm(visual.ln_pre), ln_post=nrm(visual.ln_post), proj=g(visual.proj), blocks=[])
    for r in visual.transformer.resblocks:
        a = r.attn
        qw, kw, vw = g(a.in_proj_weight).chunk(3, 0)
        qb, kb, vb = g(a.in_proj_bias).chunk(3, 0)
        W["blocks"].append(dict(ln1=nrm(r.ln_1), q_w=qw, k_w=kw, v_w=vw, q_b=qb, k_b=kb, v_b=vb,
                                o_w=g(a.out_proj.weight), o_b=g(a.out_proj.bias), ln2=nrm(r.ln_2),
                                fc_w=g(r.mlp.c_fc.weight), fc_b=g(r.mlp.c_fc.bias),
                                pr_w=g(r.mlp.c_proj.weight), pr_b=g(r.mlp.c_proj.bias)))
    return W


def to(W, dev):
    if isinstance(W, dict):
        return {k: to(v, dev) for k, v in W.items()}
    if isinstance(W, list):
        return [to(v, dev) for v in W]
    return W.to(dev) if torch.is_tensor(W) else W


def norm(x, n):
    if n["mode"] == "ln":
        return F.layer_norm(x, (x.shape[-1],), n["g"], n["b"], EPS)
    return x * n["g"] * torch.rsqrt((x * x).sum(-1, keepdim=True) / n["D"] + EPS) + n["b"]


def attn(x, B):
    N, L, _ = x.shape
    h = B["q_w"].shape[0] // HEAD_DIM
    q = F.linear(x, B["q_w"], B["q_b"]).view(N, L, h, HEAD_DIM).transpose(1, 2)
    k = F.linear(x, B["k_w"], B["k_b"]).view(N, L, h, HEAD_DIM).transpose(1, 2)
    v = F.linear(x, B["v_w"], B["v_b"]).view(N, L, h, HEAD_DIM).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, L, h * HEAD_DIM)
    return F.linear(o, B["o_w"], B["o_b"])


def embed(W, x):
    N, C, Hh, Ww = x.shape                                 # patch conv as reshape + matmul (fast backward on MPS)
    p = x.reshape(N, C, Hh // 16, 16, Ww // 16, 16).permute(0, 2, 4, 1, 3, 5).reshape(N, (Hh // 16) * (Ww // 16), C * 256)
    t = p @ W["patch_w"].reshape(W["patch_w"].shape[0], -1).t()
    t = torch.cat([W["cls"].expand(len(x), 1, -1), t], 1) + W["pos"]
    return norm(t, W["ln_pre"])


def block(x, B):
    if B["q_w"].shape[0]:
        x = x + attn(norm(x, B["ln1"]), B)
    if B["fc_w"].shape[0]:
        x = x + F.linear(F.gelu(F.linear(norm(x, B["ln2"]), B["fc_w"], B["fc_b"])), B["pr_w"], B["pr_b"])
    return x


def forward(W, x, upto=None):
    """x: CLIP-normalized photos (N, 3, 224, 224). Returns the 512 output (before normalizing).
    upto=i returns the stream after block i (all tokens) instead."""
    x = embed(W, x)
    for i, B in enumerate(W["blocks"]):
        x = block(x, B)
        if upto == i:
            return x
    return norm(x[:, 0], W["ln_post"]) @ W["proj"]


def n_params(W):
    n = W["patch_w"].numel() + W["cls"].numel() + W["pos"].numel() + W["proj"].numel()
    n += 2 * (W["ln_pre"]["g"].numel() + W["ln_post"]["g"].numel())
    for B in W["blocks"]:
        n += sum(B[k].numel() for k in ("q_w", "k_w", "v_w", "q_b", "k_b", "v_b", "o_w", "o_b", "fc_w", "fc_b", "pr_w", "pr_b"))
        n += 2 * (B["ln1"]["g"].numel() + B["ln2"]["g"].numel())
    return n


def shape(W):
    return dict(d=W["cls"].numel(), heads=[B["q_w"].shape[0] // HEAD_DIM for B in W["blocks"]],
                mlp=[B["fc_w"].shape[0] for B in W["blocks"]])


def cut(T, sd, heads, mlp):
    """Plain slice of T. sd = kept stream dims (LongTensor). heads[i] / mlp[i] = kept head / neuron indices of block i.
    Norms stay "ln" over the kept dims. No weight is changed."""
    s = lambda n: dict(g=n["g"][sd], b=n["b"][sd], mode=n["mode"], D=len(sd))  # noqa: E731
    W = dict(patch_w=T["patch_w"][sd], cls=T["cls"][sd], pos=T["pos"][:, sd], ln_pre=s(T["ln_pre"]),
             ln_post=s(T["ln_post"]), proj=T["proj"][sd], blocks=[])
    for B, hk, mk in zip(T["blocks"], heads, mlp):
        rows = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in hk]) if len(hk) else torch.zeros(0, dtype=torch.long)
        W["blocks"].append(dict(
            ln1=s(B["ln1"]), q_w=B["q_w"][rows][:, sd], k_w=B["k_w"][rows][:, sd], v_w=B["v_w"][rows][:, sd],
            q_b=B["q_b"][rows], k_b=B["k_b"][rows], v_b=B["v_b"][rows], o_w=B["o_w"][sd][:, rows], o_b=B["o_b"][sd],
            ln2=s(B["ln2"]), fc_w=B["fc_w"][mk][:, sd], fc_b=B["fc_b"][mk], pr_w=B["pr_w"][sd][:, mk], pr_b=B["pr_b"][sd]))
    return W


def sizes(f, T):
    """Generic widths at fraction f of the teacher: stream, heads per block, MLP neurons per block (at least 1 each)."""
    d0, h0, m0 = T["cls"].numel(), T["blocks"][0]["q_w"].shape[0] // HEAD_DIM, T["blocks"][0]["fc_w"].shape[0]
    return max(1, round(f * d0)), max(1, round(f * h0)), max(1, round(f * m0))
