"""N5 step 0: why does the one-layer build fail end to end? (log N4 step 8: keep 50%, all 30 blocks: KL .37)

Same build as pair/allblocks.py (1,000 calibration photos, covariance of each block's GELU outputs, keep 50%,
writes re-solved). New:
  single  - swap ONE block at a time; value survivors and random survivors (3 seeds); per-photo KL kept
  cumul   - value survivors, blocks 0..j swapped, j = 0..29
Test: the first 500 photos of distill.py's test list. Saved: results/tree/diag.json, diag_rows.npz.
Reading: sum of single-block KLs well below the all-at-once KL -> errors pile up.
         value ~ random already at single blocks -> layer R^2 is the wrong yardstick.
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import DEV, OUT, blocks, kl_rows, load, net, split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pair"))
from distill import D  # noqa: E402

torch.set_grad_enabled(False)
FRAC = 0.5


def main():
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][7] for c in range(1000)]))
    te = np.load(os.path.join(D, "photos_test.npy"))[:500]
    m = load("base")
    BL = blocks(m)
    acc = {}
    hooks = []
    for key, blk in enumerate(BL):
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
    for h_ in hooks:
        h_.remove()
    print(f"calibration {time.time() - t0:.0f}s", flush=True)
    orig = {k: (blk.block[5].weight.detach().clone(), blk.block[5].bias.detach().clone()) for k, blk in enumerate(BL)}
    built = {}

    def build(key, how, seed):
        if (key, how, seed) in built:
            return built[(key, how, seed)]
        blk = BL[key]
        a = acc[key]
        mg = a["s"] / a["n"]
        K = a["ss"] / a["n"] - mg[:, None] * mg[None, :]
        W2, b2 = orig[key]
        ls = blk.layer_scale.detach().view(-1).cpu().double()
        Wb = ls[:, None] * W2.cpu().double()
        bb = ls * b2.cpu().double()
        h = Wb.shape[1]
        kk = int(round(FRAC * h))
        if how == "value":
            S = torch.argsort(-(K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)))[:kk]
        else:
            S = torch.from_numpy(np.random.default_rng(seed).choice(h, kk, replace=False))
        KSS = K[S][:, S] + 1e-6 * K.diagonal().mean() * torch.eye(kk, dtype=torch.float64)
        Wnew = torch.zeros_like(Wb)
        Wnew[:, S] = torch.linalg.solve(KSS, K[S] @ Wb.T).T
        bnew = bb + Wb @ mg - Wnew @ mg
        out = ((Wnew / ls[:, None]).float().to(DEV), (bnew / ls).float().to(DEV))
        if how == "value":
            built[(key, how, seed)] = out
        return out

    def put(key, wb):
        BL[key].block[5].weight.copy_(wb[0])
        BL[key].block[5].bias.copy_(wb[1])

    def restore():
        for k in range(len(BL)):
            put(k, orig[k])

    ref = []
    for a in range(0, len(te), 50):
        ref.append(m(ph.tensor(te[a:a + 50]).to(DEV)))

    def evaluate():
        out = []
        for i, a in enumerate(range(0, len(te), 50)):
            out.append(kl_rows(ref[i], m(ph.tensor(te[a:a + 50]).to(DEV))).cpu())
        return torch.cat(out).numpy()

    rows = {}
    res = {}
    # sanity: 100% kept equals Base
    restore()
    res["sanity_restore_kl"] = float(evaluate().mean())
    variants = [("value", 0), ("random", 0), ("random", 1), ("random", 2)]
    for key in range(len(BL)):
        for how, seed in variants:
            put(key, build(key, how, seed))
            r = evaluate()
            put(key, orig[key])
            rows[f"single_{key}_{how}{seed}"] = r
        v = rows[f"single_{key}_value0"].mean()
        rr = np.mean([rows[f"single_{key}_random{s}"].mean() for s in range(3)])
        print(f"block {key}: value {v:.4f}  random {rr:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        np.savez(os.path.join(OUT, "diag_rows.npz"), **rows)
    restore()
    for key in range(len(BL)):
        put(key, build(key, "value", 0))
        r = evaluate()
        rows[f"cumul_{key}"] = r
        print(f"cumulative 0..{key}: {r.mean():.4f}", flush=True)
    restore()
    np.savez(os.path.join(OUT, "diag_rows.npz"), **rows)
    sv = np.array([rows[f"single_{k}_value0"].mean() for k in range(len(BL))])
    sr = np.array([[rows[f"single_{k}_random{s}"].mean() for s in range(3)] for k in range(len(BL))])
    diff = np.stack([rows[f"single_{k}_value0"] - np.mean([rows[f"single_{k}_random{s}"] for s in range(3)], 0)
                     for k in range(len(BL))])  # blocks x photos
    tot = diff.sum(0)
    bs = [tot[np.random.default_rng(i).integers(0, len(tot), len(tot))].mean() for i in range(2000)]
    res.update(single_value=sv.tolist(), single_random=sr.tolist(), sum_single_value=float(sv.sum()),
               sum_single_random=float(sr.mean(1).sum()),
               sum_value_minus_random_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               cumulative=[float(rows[f"cumul_{k}"].mean()) for k in range(len(BL))])
    json.dump(res, open(os.path.join(OUT, "diag.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)


if __name__ == "__main__":
    main()
