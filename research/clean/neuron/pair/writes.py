"""Pair study, step 1 tool: match neurons by what they write.

Each stage-4 hidden neuron i writes layer_scale * W2[:, i] into the stream. The stream reaches the classifier through
avgpool, a LayerNorm (centre, divide by std, scale by ln_w) and the Linear W_cls. Its direct effect on the 1,000
class scores (up to the LayerNorm's per-photo std) is
    v_i = W_cls @ (ln_w * centre(layer_scale * W2[:, i]))
Small and Base have different stream widths, but v_i lives in the same 1,000-class space for both.
Score: cosine of the class-centred v vectors. Also saves V for later steps.
"""
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from cache import NB, OUT, load

torch.set_grad_enabled(False)


def write_vectors(m):
    ln = m.classifier[0]
    Wc = m.classifier[2].weight.detach().cpu().double()  # (1000, dim)
    out = []
    for blk in m.features[7]:
        W2 = blk.block[5].weight.detach().cpu().double()  # (dim, 4dim)
        ls = blk.layer_scale.detach().cpu().double().view(-1, 1)
        w = ls * W2
        w = w - w.mean(0, keepdim=True)
        w = ln.weight.detach().cpu().double().view(-1, 1) * w
        v = Wc @ w  # (1000, 4dim)
        v = v - v.mean(0, keepdim=True)
        out.append(v.T.contiguous())  # (4dim, 1000)
    return out


def main():
    Vs = write_vectors(load("small"))
    Vb = write_vectors(load("base"))
    res = {}
    for b in range(NB):
        a = Vs[b] / Vs[b].norm(dim=1, keepdim=True).clamp_min(1e-12)
        c = Vb[b] / Vb[b].norm(dim=1, keepdim=True).clamp_min(1e-12)
        R = (a @ c.T).numpy()
        best = R.max(1)
        r, cc = linear_sum_assignment(-R)
        res[f"b{b}"] = dict(best_mean=float(best.mean()), best_over={t: float((best > t).mean()) for t in (.4, .6, .8)},
                            hung_mean=float(R[r, cc].mean()),
                            norm_small_median=float(Vs[b].norm(dim=1).median()),
                            norm_base_median=float(Vb[b].norm(dim=1).median()))
        np.save(os.path.join(OUT, f"writesR_b{b}.npy"), R.astype(np.float16))
        print(b, res[f"b{b}"], flush=True)
    np.savez(os.path.join(OUT, "writes_V.npz"), **{f"s{b}": Vs[b].float().numpy() for b in range(NB)},
             **{f"b{b}": Vb[b].float().numpy() for b in range(NB)})
    json.dump(res, open(os.path.join(OUT, "writes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
