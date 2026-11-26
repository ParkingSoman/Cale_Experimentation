"""Floor rung: plain cut by weight magnitude, no weight changed (FANOUT step 0a).

Every unit is scored by (read size) x (write size), from weights only:
  ViT MLP neuron j:  |fc_w[j] * ln2.g| x |pr_w[:, j]|
  ViT head h:        |v_w[h rows] * ln1.g|_F x |o_w[:, h cols]|_F   (q and k are left out: they set where to look,
                     v and o carry what is written)
  ViT stream dim i:  sqrt(sum of squared writes into i) x sqrt(sum of squared reads from i, times each reader's norm gain)
  ResNet inner channel: |BN-folded conv1 row| x |BN-folded conv2 column|
  ResNet stream channel of a stage: writes = folded conv2 rows of both blocks (+ downsample, + stem for stage 1),
                     reads = conv1 columns of the blocks that read the stage, next downsample, fc.
BN folding uses gamma / sqrt(running_var + eps): that is the network's own inference map, not a use of the
statistics as data. Keep the top f of each set. Widths as in distill/wide18.py (ResNet) and vitgen.sizes (ViT).
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import vitgen  # noqa: E402


def vit_scores(T):
    d = T["cls"].numel()
    wr = T["patch_w"].flatten(1).pow(2).sum(1) + T["cls"].pow(2) + T["pos"].pow(2).sum(0)
    rd = (T["proj"] * T["ln_post"]["g"][:, None]).pow(2).sum(1)
    head, mlp = [], []
    for B in T["blocks"]:
        g1, g2 = B["ln1"]["g"], B["ln2"]["g"]
        wr = wr + B["o_w"].pow(2).sum(1) + B["pr_w"].pow(2).sum(1)
        rd = rd + sum((B[k] * g1).pow(2).sum(0) for k in ("q_w", "k_w", "v_w")) + (B["fc_w"] * g2).pow(2).sum(0)
        H = B["q_w"].shape[0] // vitgen.HEAD_DIM
        v = (B["v_w"] * g1).view(H, vitgen.HEAD_DIM, d).pow(2).sum((1, 2)).sqrt()
        o = B["o_w"].view(d, H, vitgen.HEAD_DIM).pow(2).sum((0, 2)).sqrt()
        head.append(v * o)
        mlp.append((B["fc_w"] * g2).norm(dim=1) * B["pr_w"].norm(dim=0))
    return dict(stream=wr.sqrt() * rd.sqrt(), head=head, mlp=mlp)


def top(s, k):
    return torch.sort(torch.topk(s, k).indices).values


def vit(T, f):
    sc = vit_scores(T)
    d, h, m = vitgen.sizes(f, T)
    return vitgen.cut(T, top(sc["stream"], d), [top(s, h) for s in sc["head"]], [top(s, m) for s in sc["mlp"]])


def _fold(conv, bn):
    return conv.weight.detach() * (bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)


def r18_scores(A):
    import wide18
    inner = [[None, None] for _ in range(4)]
    wr = [0.0] * 4
    rd = [0.0] * 4
    wr[0] = wr[0] + _fold(A.conv1, A.bn1).flatten(1).pow(2).sum(1)
    for s, layer in enumerate(wide18.layers(A)):
        for b, bl in enumerate(layer):
            c1, c2 = _fold(bl.conv1, bl.bn1), _fold(bl.conv2, bl.bn2)
            inner[s][b] = c1.flatten(1).norm(dim=1) * c2.transpose(0, 1).flatten(1).norm(dim=1)
            wr[s] = wr[s] + c2.flatten(1).pow(2).sum(1)
            src = s if b > 0 or s == 0 else s - 1          # which stage's stream this block reads
            rd[src] = rd[src] + bl.conv1.weight.detach().transpose(0, 1).flatten(1).pow(2).sum(1)
            if bl.downsample is not None:
                dw = _fold(bl.downsample[0], bl.downsample[1])
                wr[s] = wr[s] + dw.flatten(1).pow(2).sum(1)
                rd[s - 1] = rd[s - 1] + bl.downsample[0].weight.detach().transpose(0, 1).flatten(1).pow(2).sum(1)
    rd[3] = rd[3] + A.fc.weight.detach().pow(2).sum(0)
    return dict(inner=inner, stream=[wr[s].sqrt() * rd[s].sqrt() for s in range(4)])


@torch.no_grad()
def r18(A, f):
    import wide18
    S, _ = wide18.cut_from_a(A, f, r18_scores(A))
    return S.eval()
