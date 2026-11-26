"""Floor split (critic's step 0, round 1): the floor (noisecut) with ONE set cut and every other set at full width.
Exact removal: a cut ResNet inner channel = its conv2 input column set to 0; a cut ResNet stream channel = every
read of it set to 0 (block conv1 columns, downsample columns, fc column). Params reported = the full network's
(the masked weights stay in the tensor)."""
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import vitgen  # noqa: E402
from rules import noisecut  # noqa: E402

_SC = {}


def vit(T, f, which):
    if "vit" not in _SC:
        _SC["vit"] = noisecut.vit_scores(T)
    sc = _SC["vit"]
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    d, h, m = (d if which == "stream" else D), (h if which == "heads" else H), (m if which == "mlp" else M)
    return vitgen.cut(T, noisecut.top(sc["stream"], d), [noisecut.top(s, h) for s in sc["head"]], [noisecut.top(s, m) for s in sc["mlp"]])


@torch.no_grad()
def r18(A, f, which):
    import wide18
    if "r18" not in _SC:
        A2 = A.to(noisecut.DEV).eval()
        import generic
        _SC["r18"] = wide18.channel_scores(A2, generic.pink(noisecut.N_NOISE, seed=noisecut.SEED).to(noisecut.DEV), bs=noisecut.BS)
        A.cpu()
    sc = _SC["r18"]
    S = copy.deepcopy(A).cpu().eval()
    w = wide18.widths(f)
    L = wide18.layers(S)
    for s in range(4):
        if which == "inner":
            for b in range(2):
                keep = wide18.keep_idx(sc["inner"][s][b], w[s])
                cut = torch.ones(wide18.BASE[s], dtype=torch.bool)
                cut[keep] = False
                L[s][b].conv2.weight[:, cut] = 0
        else:
            keep = wide18.keep_idx(sc["stream"][s], w[s])
            cut = torch.ones(wide18.BASE[s], dtype=torch.bool)
            cut[keep] = False
            readers = [L[s][1].conv1]                     # block 1 of this stage reads the stage stream
            if s < 3:
                readers += [L[s + 1][0].conv1, L[s + 1][0].downsample[0]]
            if s == 0:
                readers += [L[0][0].conv1]                # stage-1 block 0 reads the stem output
            for r in readers:
                r.weight[:, cut] = 0
            if s == 3:
                S.fc.weight[:, cut] = 0
    return S
