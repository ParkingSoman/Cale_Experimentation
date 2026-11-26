"""The no-training build on every block of stages 3 and 4 of convnext_base, scored end to end.

Per block b (hidden h_b, stream d_b): g = GELU(W1 u + b1), MLP output y = Wb g + bb (Wb, bb include layer_scale).
Calibration (1,000 fit photos, all positions, original network): mean m_g and covariance K of g.
Build at keep fraction f:
  value_i = sqrt(K_ii) * |Wb[:, i]|;  S = top f * h_b by value
  writes re-solved: Wnew^T = K_SS^-1 K_S,all Wb^T  (least squares of the full output on the kept outputs)
                    bias  = bb + Wb m_g - Wnew m_g[S]
All blocks are changed at once (dropped neurons get write 0). Test: 1,500 test photos of distill.py.
Variants: value+resolve, value unchanged writes (plain truncation), random+resolve, random unchanged.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from cache import DEV, load, net, split
from distill import D

torch.set_grad_enabled(False)


def blocks(m):
    return [(s, i, blk) for s in (5, 7) for i, blk in enumerate(m.features[s])]


def main():
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    te = np.load(os.path.join(D, "photos_test.npy"))
    m = load("base")
    BL = blocks(m)
    acc = {}
    hooks = []
    for key, (s, i, blk) in enumerate(BL):
        h = blk.block[3].out_features
        acc[key] = dict(n=0, s=torch.zeros(h, dtype=torch.float64), ss=torch.zeros(h, h, dtype=torch.float64))

        def f(mod, inp, out, key=key):
            g = out.reshape(-1, out.shape[-1]).float()
            a = acc[key]
            a["n"] += g.shape[0]
            a["s"] += g.sum(0).cpu().double()
            a["ss"] += (g.T @ g).cpu().double()
        hooks.append(blk.block[4].register_forward_hook(f))
    t0 = time.time()
    for k in range(0, len(cal), 25):
        m(ph.tensor(cal[k:k + 25]).to(DEV))
        if (k // 25) % 10 == 0:
            print(f"  calib {k + 25}/{len(cal)} {time.time() - t0:.0f}s", flush=True)
    for h_ in hooks:
        h_.remove()
    orig = {key: (blk.block[5].weight.detach().clone(), blk.block[5].bias.detach().clone()) for key, (s, i, blk) in enumerate(BL)}

    def apply(frac, how, resolve, seed=0):
        rng = np.random.default_rng(seed)
        for key, (s, i, blk) in enumerate(BL):
            a = acc[key]
            n = a["n"]
            mg = a["s"] / n
            K = a["ss"] / n - mg[:, None] * mg[None, :]
            W2, b2 = orig[key]
            ls = blk.layer_scale.detach().view(-1).cpu().double()
            Wb = ls[:, None] * W2.cpu().double()
            bb = ls * b2.cpu().double()
            h = Wb.shape[1]
            kk = int(round(frac * h))
            if how == "value":
                v = K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)
                S = torch.argsort(-v)[:kk]
            else:
                S = torch.from_numpy(rng.choice(h, kk, replace=False))
            Wnew = torch.zeros_like(Wb)
            if resolve:
                KSS = K[S][:, S] + 1e-6 * K.diagonal().mean() * torch.eye(kk, dtype=torch.float64)
                Wnew[:, S] = torch.linalg.solve(KSS, K[S] @ Wb.T).T
            else:
                Wnew[:, S] = Wb[:, S]
            bnew = bb + Wb @ mg - Wnew @ mg
            blk.block[5].weight.copy_((Wnew / ls[:, None]).float().to(DEV))
            blk.block[5].bias.copy_((bnew / ls).float().to(DEV))

    def restore():
        for key, (s, i, blk) in enumerate(BL):
            blk.block[5].weight.copy_(orig[key][0])
            blk.block[5].bias.copy_(orig[key][1])

    def evaluate():
        kl, c, n = 0.0, 0, 0
        for a in range(0, len(te), 50):
            ii = te[a:a + 50]
            x = ph.tensor(ii).to(DEV)
            lab = torch.tensor(ph.lab[ii], device=DEV)
            y1 = m(x)
            kl += float((F.softmax(ref[a], 1) * (F.log_softmax(ref[a], 1) - F.log_softmax(y1, 1))).sum())
            c += int((y1.argmax(1) == lab).sum())
            n += len(ii)
        return dict(kl=kl / n, top1=c / n)

    restore()
    ref = {}
    c0 = 0
    for a in range(0, len(te), 50):
        ii = te[a:a + 50]
        ref[a] = m(ph.tensor(ii).to(DEV))
        c0 += int((ref[a].argmax(1) == torch.tensor(ph.lab[ii], device=DEV)).sum())
    res = {"base_top1": c0 / len(te)}
    print(res, flush=True)
    for frac in (0.5, 0.25):
        for how, resolve in (("value", True), ("value", False), ("random", True), ("random", False)):
            apply(frac, how, resolve)
            r = evaluate()
            restore()
            res[f"keep{frac}_{how}_{'resolved' if resolve else 'unchanged'}"] = r
            print(frac, how, resolve, r, flush=True)
            json.dump(res, open(os.path.join(D, "allblocks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
