"""Pair study, steps 0 and 1 (plan okay-last-session-within-scalable-beaver.md).

Two torchvision ConvNeXts trained with one recipe and the same depth: Small (coarse) and Base (fine).
Stage 4 has 3 blocks. Each block's hidden neurons are the outputs of its first Linear (pre-activation, before GELU):
Small 3,072 per block, Base 4,096 per block, at 7 x 7 positions.

This script, in one pass over the reading photos:
  - top-1 accuracy of each model on the test half (sanity check)
  - per block, the correlation of every Small neuron with every Base neuron, same photo, same position,
    on the fit half (running sums in float64)
  - per photo, the maximum pre-activation of every hidden neuron (fp16), for sheets and later steps
Smoke mode (--n 500) also checks: Base against Base gives correlation 1 on the diagonal, and Small against Base
with photos shuffled gives best matches near 0.

Photos: the reading photos of the ResNet-18 map (curves/map_agents/resnet18/net.py), 35 per class.
Inside each class, even positions fit, odd positions test. The sealed photos are never opened.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
R18 = os.path.join(HERE, "..", "..", "curves", "map_agents", "resnet18")
sys.path.insert(0, R18)
import net  # noqa: E402

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT = os.path.join(HERE, "..", "results", "pair")
os.makedirs(OUT, exist_ok=True)
NB = 3  # blocks in stage 4


def load(name):
    if name == "small":
        m = torchvision.models.convnext_small(weights="IMAGENET1K_V1")
    else:
        m = torchvision.models.convnext_base(weights="IMAGENET1K_V1")
    return m.eval().to(DEV)


def hook_stage4(m, store):
    hs = []
    for b, blk in enumerate(m.features[7]):
        lin1 = blk.block[3]
        assert isinstance(lin1, torch.nn.Linear)

        def f(mod, inp, out, b=b):
            store[b] = out  # (B, 7, 7, 4*dim), pre-GELU
        hs.append(lin1.register_forward_hook(f))
    return hs


def split(lab):
    pos = np.zeros(len(lab), int)
    for c in range(1000):
        w = np.where(lab == c)[0]
        pos[w] = np.arange(len(w))
    return pos % 2 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="photos to use (0 = all); spread over classes")
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    smoke = a.n > 0
    tag = f"smoke{a.n}" if smoke else "all"

    ph = net.Photos("read")
    lab = ph.lab
    fit = split(lab)
    idx = np.arange(len(lab))
    if smoke:
        idx = np.sort(np.random.default_rng(0).choice(len(lab), a.n, replace=False))
    print(f"photos {len(idx)}, fit {fit[idx].sum()}, device {DEV}", flush=True)

    ms, mb = load("small"), load("base")
    st_s, st_b = {}, {}
    hook_stage4(ms, st_s)
    hook_stage4(mb, st_b)
    Ds, Db = 4 * 768, 4 * 1024

    # running sums per block (fit photos only)
    S = {b: dict(n=0, sx=torch.zeros(Ds, dtype=torch.float64), sy=torch.zeros(Db, dtype=torch.float64),
                 sxx=torch.zeros(Ds, dtype=torch.float64), syy=torch.zeros(Db, dtype=torch.float64),
                 sxy=torch.zeros(Ds, Db, dtype=torch.float64)) for b in range(NB)}
    if smoke:  # controls: Base vs Base (block 0) and Small vs shuffled Base (block 0)
        C = dict(byy=torch.zeros(Db, Db, dtype=torch.float64), sh=torch.zeros(Ds, Db, dtype=torch.float64),
                 shy=torch.zeros(Db, dtype=torch.float64), shyy=torch.zeros(Db, dtype=torch.float64))
        shuffle_rows = []
    mx_s = np.zeros((len(idx), NB, Ds), np.float16)
    mx_b = np.zeros((len(idx), NB, Db), np.float16)
    correct = {"small": 0, "base": 0}
    ntest = 0
    t0 = time.time()
    for k in range(0, len(idx), a.batch):
        ii = idx[k:k + a.batch]
        x = ph.tensor(ii).to(DEV)
        ys, yb = ms(x), mb(x)
        te = ~fit[ii]
        y = torch.tensor(lab[ii], device=DEV)
        tt = torch.tensor(te, device=DEV)
        correct["small"] += int(((ys.argmax(1) == y) & tt).sum())
        correct["base"] += int(((yb.argmax(1) == y) & tt).sum())
        ntest += int(te.sum())
        fi = torch.tensor(fit[ii], device=DEV)
        for b in range(NB):
            hs, hb = st_s[b], st_b[b]
            mx_s[k:k + len(ii), b] = hs.amax((1, 2)).float().cpu().numpy().astype(np.float16)
            mx_b[k:k + len(ii), b] = hb.amax((1, 2)).float().cpu().numpy().astype(np.float16)
            X = hs[fi].reshape(-1, Ds).float()
            Y = hb[fi].reshape(-1, Db).float()
            if X.shape[0] == 0:
                continue
            s = S[b]
            s["n"] += X.shape[0]
            s["sx"] += X.sum(0).cpu().double()
            s["sy"] += Y.sum(0).cpu().double()
            s["sxx"] += (X * X).sum(0).cpu().double()
            s["syy"] += (Y * Y).sum(0).cpu().double()
            s["sxy"] += (X.T @ Y).cpu().double()
            if smoke and b == 0:
                C["byy"] += (Y.T @ Y).cpu().double()
                shuffle_rows.append((X.cpu(), Y.cpu()))
        if (k // a.batch) % 20 == 0:
            el = time.time() - t0
            print(f"  {k + len(ii)}/{len(idx)}  {el:.0f}s  eta {el / (k + len(ii)) * (len(idx) - k - len(ii)):.0f}s",
                  flush=True)

    res = dict(n_photos=int(len(idx)), n_test=ntest, top1={m: correct[m] / max(ntest, 1) for m in correct})
    corr = {}
    for b in range(NB):
        s = S[b]
        n = s["n"]
        mxs, mys = s["sx"] / n, s["sy"] / n
        vx = (s["sxx"] / n - mxs ** 2).clamp_min(1e-12)
        vy = (s["syy"] / n - mys ** 2).clamp_min(1e-12)
        cov = s["sxy"] / n - mxs[:, None] * mys[None, :]
        R = (cov / vx.sqrt()[:, None] / vy.sqrt()[None, :]).float().numpy()
        corr[b] = R
        best = R.max(1)
        res[f"block{b}"] = dict(rows=int(n), best_small_mean=float(best.mean()),
                                share_best_over={t: float((best > t).mean()) for t in (0.2, 0.4, 0.6, 0.8)})
        print(f"block {b}: rows {n}, best match per Small neuron: mean {best.mean():.3f}, "
              f">0.6 {np.mean(best > .6):.2f}, >0.8 {np.mean(best > .8):.2f}", flush=True)
    if smoke:
        s = S[0]
        n = s["n"]
        my = s["sy"] / n
        vy = (s["syy"] / n - my ** 2).clamp_min(1e-12)
        cyy = C["byy"] / n - my[:, None] * my[None, :]
        Ryy = cyy / vy.sqrt()[:, None] / vy.sqrt()[None, :]
        res["check_base_self_diag_min"] = float(Ryy.diagonal().min())
        # shuffled control: pair Small rows of photo i with Base rows of another photo
        Xs = torch.cat([r[0] for r in shuffle_rows])
        Yb = torch.cat([r[1] for r in shuffle_rows])
        perm = torch.randperm(Yb.shape[0] // 49, generator=torch.Generator().manual_seed(0))
        Yb = Yb.view(-1, 49, Db)[perm].reshape(-1, Db)
        Xc = (Xs - Xs.mean(0)) / Xs.std(0).clamp_min(1e-6)
        Yc = (Yb - Yb.mean(0)) / Yb.std(0).clamp_min(1e-6)
        Rsh = (Xc.T @ Yc) / Xs.shape[0]
        res["check_shuffled_best_mean"] = float(Rsh.max(1).values.mean())
        print(f"checks: Base self diag min {res['check_base_self_diag_min']:.4f}; "
              f"shuffled best match mean {res['check_shuffled_best_mean']:.3f}", flush=True)
    print("top1 on test half:", res["top1"], flush=True)
    json.dump(res, open(os.path.join(OUT, f"cache_{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, f"corr_{tag}.npz"), **{f"b{b}": corr[b].astype(np.float16) for b in corr})
    np.savez(os.path.join(OUT, f"max_{tag}.npz"), idx=idx, lab=lab[idx], fit=fit[idx], small=mx_s, base=mx_b)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
