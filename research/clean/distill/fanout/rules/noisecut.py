"""Floor rung 2: Minitron's activation ranking (arXiv 2407.14679), with generic noise in place of calibration photos.
No weight is changed. Input: generic.pink(256, seed 1) only (no photos).

ViT:    stream dim i = RMS of the stream at i, over all tokens and all 12 block outputs (and the embedding)
        head h       = RMS of the head's write (o_w[:, h] @ head output), all tokens
        MLP neuron j = mean |GELU output|, all tokens
ResNet: wide18.channel_scores (Minitron scores as in the P start of distill/LOG.md), on the noise instead of photos.
Keep the top f of each set. Widths as in magcut.
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

N_NOISE, SEED, BS = 256, 1, 32
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def vit_scores(T, n=N_NOISE, seed=SEED, X=None):
    """X: optional cached generic input (N, 3, 224, 224). Default: pink(n, seed) in batches as before."""
    Td = vitgen.to(T, DEV)
    nb = len(T["blocks"])
    H = T["blocks"][0]["q_w"].shape[0] // vitgen.HEAD_DIM
    d = T["cls"].numel()
    st = torch.zeros(d, device=DEV)
    hd = torch.zeros(nb, H, device=DEV)
    ml = [0.0] * nb
    cnt = 0
    n = len(X) if X is not None else n
    for a in range(0, n, BS):
        x = (X[a:a + BS] if X is not None else generic.pink(BS, seed=seed * 100000 + a)).to(DEV).float()
        s = vitgen.embed(Td, x)
        st += s.pow(2).mean((0, 1))
        for i, B in enumerate(Td["blocks"]):
            y = vitgen.norm(s, B["ln1"])
            N, L, _ = y.shape
            q = F.linear(y, B["q_w"], B["q_b"]).view(N, L, H, 64).transpose(1, 2)
            k = F.linear(y, B["k_w"], B["k_b"]).view(N, L, H, 64).transpose(1, 2)
            v = F.linear(y, B["v_w"], B["v_b"]).view(N, L, H, 64).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v)                      # (N, H, L, 64)
            ow = B["o_w"].view(d, H, 64)
            per_head = torch.einsum("nhlc,dhc->nhld", o, ow)                 # each head's write
            hd[i] += per_head.pow(2).mean((0, 2, 3))
            s = s + per_head.sum(1) + B["o_b"]
            g = F.gelu(F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"]))
            ml[i] = ml[i] + g.abs().mean((0, 1))
            s = s + F.linear(g, B["pr_w"], B["pr_b"])
            st += s.pow(2).mean((0, 1))
        cnt += 1
    return dict(stream=(st / cnt).sqrt().cpu(), head=[(hd[i] / cnt).sqrt().cpu() for i in range(nb)],
                mlp=[(ml[i] / cnt).cpu() for i in range(nb)])


def top(s, k):
    return torch.sort(torch.topk(s, k).indices).values


def vit(T, f, X=None):
    sc = vit_scores(T, X=X)
    d, h, m = vitgen.sizes(f, T)
    return vitgen.cut(T, top(sc["stream"], d), [top(s, h) for s in sc["head"]], [top(s, m) for s in sc["mlp"]])


@torch.no_grad()
def r18(A, f, X=None):
    import wide18
    A = A.to(DEV).eval()
    x = (X if X is not None else generic.pink(N_NOISE, seed=SEED)).to(DEV).float()
    sc = wide18.channel_scores(A, x, bs=BS)
    S, _ = wide18.cut_from_a(A, f, sc)
    A.cpu()
    return S.cpu().eval()
