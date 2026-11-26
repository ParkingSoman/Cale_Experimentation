"""N7 step 1: the coarsening lens. ConvNeXt-Base stages 3-4, keep a fraction of each block's MLP neurons.

Built in order, block by block, in closed form (no training):
  u_B = Base's block input (after dwconv + LayerNorm) on Base's stream
  u_C = the coarse network's block input on its own stream
  lens       L, m = ridge least squares  u_C -> u_B          (mode lens; identity for seq; mean shift for mean)
  activations a_C = GELU(W1_S (L u_C + m) + b1_S)            (kept reads unchanged, folded: W1' = W1_S L)
  writes          = least squares a_C -> Base's block output y_B (with intercept, ridge as the N6 build)
Modes:
  base   writes solved on Base's own activations on the fit photos (with --fit cal this is the N6 build B)
  seq    no lens, writes solved on C's activations (sequential build)
  mean   lens = identity + mean shift only (bias-only control)
  lens   the full lens
Options: --ds_lens (lens before the stage-4 downsample conv), --head_lens (lens before the classifier, fitted on all
train photos + flips by one row per photo).
Saved: results/lens/<tag>.json (test and held KL / top-1, per-block firing rates), <tag>_rows.npy (per-photo test KL),
<tag>.pt (the coarse stage 3-4 state, if --save).
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C  # noqa: E402

DEV = C.DEV
torch.set_grad_enabled(False)
OUT = os.path.join(HERE, "..", "results", "lens")
os.makedirs(OUT, exist_ok=True)


def log(msg, f=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if f is not None:
        f.write(line + "\n")
        f.flush()


def u_of(blk, x):
    """Block input to the first Linear: dwconv, permute, LayerNorm. (B, H, W, C)."""
    return blk.block[2](blk.block[1](blk.block[0](x)))


def mlp_rows(blk, u):
    ls = blk.layer_scale.view(-1)
    return ls * blk.block[5](blk.block[4](blk.block[3](u)))


class Gram:
    """Running sums for a least-squares fit with intercept: X rows -> Y rows. fp32 per batch, float64 on CPU."""

    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.xx = self.xy = self.yy = None

    def add(self, X, Y):
        X, Y = X.reshape(-1, X.shape[-1]).float(), Y.reshape(-1, Y.shape[-1]).float()
        sx, sy = X.sum(0).cpu().double(), Y.sum(0).cpu().double()
        xx, xy, yy = (X.T @ X).cpu().double(), (X.T @ Y).cpu().double(), (Y.T @ Y).cpu().double()
        if self.n == 0:
            self.sx, self.sy, self.xx, self.xy, self.yy = sx, sy, xx, xy, yy
        else:
            self.sx += sx
            self.sy += sy
            self.xx += xx
            self.xy += xy
            self.yy += yy
        self.n += X.shape[0]

    def solve(self, ridge_rel):
        mx, my = self.sx / self.n, self.sy / self.n
        Cxx = self.xx / self.n - mx[:, None] * mx[None, :]
        Cxy = self.xy / self.n - mx[:, None] * my[None, :]
        lam = ridge_rel * Cxx.diagonal().mean()
        A = torch.linalg.solve(Cxx + lam * torch.eye(len(mx), dtype=torch.float64), Cxy)   # (dx, dy)
        b = my - mx @ A
        return A.T.contiguous(), b                                                          # Y ~ X A^T... as (dy,dx)

    def ot(self, W, b):
        """Keep the fit's direction, restore the target's spread: compose the linear map Y_hat = W X + b with the
        optimal-transport map from N(mean, Cov(Y_hat)) to N(mean_Y, Cov(Y)). Returns (W', b', amplification stats)."""
        mx, my = self.sx / self.n, self.sy / self.n
        Cxx = self.xx / self.n - mx[:, None] * mx[None, :]
        Cyy = self.yy / self.n - my[:, None] * my[None, :]
        A = W @ Cxx @ W.T                                   # Cov(Y_hat)
        eps = 1e-9 * A.diagonal().mean()
        ea, Va = torch.linalg.eigh(A)
        ea = ea.clamp_min(eps)
        Ah = Va @ torch.diag(ea.sqrt()) @ Va.T
        Aih = Va @ torch.diag(ea.rsqrt()) @ Va.T
        ei, Vi = torch.linalg.eigh(Ah @ Cyy @ Ah)
        Mid = Vi @ torch.diag(ei.clamp_min(0).sqrt()) @ Vi.T
        M = Aih @ Mid @ Aih
        Wn = M @ W
        bn = my - Wn @ mx
        amp = torch.linalg.eigvalsh((M + M.T) / 2)
        return Wn, bn, dict(amp_median=float(amp.median()), amp_max=float(amp.max()),
                            trace_ratio=float(Cyy.trace() / A.trace()))


def fit_rows(cache, per_class, lab):
    """Rows of a cache: the first per_class photos of every class (or spread smoke rows)."""
    cl = lab[cache.ix]
    rows = []
    for c in range(1000):
        w = np.where(cl == c)[0]
        rows += w[:per_class].tolist()
    return np.array(sorted(rows))


def batched(n, bs):
    for a in range(0, n, bs):
        yield slice(a, min(a + bs, n))


def build_lens(base, calib, xs, mode, frac, ridge_lens, ridge_w, nblocks, ds_lens, fl, ot_lens=False, ot_writes=False):
    """Sequential build. xs: fit photos' stage-3 inputs (N, 512, 14, 14) fp16 on DEV. Returns the coarse model + stats."""
    mC = copy.deepcopy(base)
    BB, BC = C.blocks(base), C.blocks(mC)
    xB = xs.clone()
    xC = xs.clone()
    N = xs.shape[0]
    bs = 50 if xs.shape[-1] == 14 else 100
    stats = []
    for g in range(30):
        if g == 27:                                  # stage-4 downsample, for both streams
            xB, xC = downsample(base, mC, xB, xC, ds_lens and mode != "base", ridge_lens, fl)
            bs = 100
        bB, bC = BB[g], BC[g]
        t0 = time.time()
        if g >= nblocks:                              # unchanged block (smoke runs)
            for s in batched(N, bs):
                xB[s] = bB(xB[s].float()).half()
                xC[s] = bC(xC[s].float()).half()
            continue
        mg, K = calib[g]
        S = C.choose(bB, mg, K, frac, "value")
        Sd = S.to(DEV)
        W1, b1 = bB.block[3].weight[Sd], bB.block[3].bias[Sd]
        d = W1.shape[1]
        # lens
        if mode in ("lens", "mean"):
            G = Gram()
            for s in batched(N, bs):
                G.add(u_of(bB, xC[s].float()), u_of(bB, xB[s].float()))
            if mode == "lens":
                L, m = G.solve(ridge_lens)
                if ot_lens:
                    L, m, amp_l = G.ot(L, m)
            else:
                L = torch.eye(d, dtype=torch.float64)
                m = G.sy / G.n - G.sx / G.n
        else:
            L, m = torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
        W1n = (W1.cpu().double() @ L).float().to(DEV)
        b1n = (b1.cpu().double() + W1.cpu().double() @ m).float().to(DEV)
        # writes
        Gw = Gram()
        on_B = torch.zeros(len(S), device=DEV)
        on_C = torch.zeros(len(S), device=DEV)
        nrow = 0
        for s in batched(N, bs):
            uB = u_of(bB, xB[s].float())
            yB = mlp_rows(bB, uB)
            zB = uB @ W1.T + b1
            if mode == "base":
                zC = zB
            else:
                zC = u_of(bB, xC[s].float()) @ W1n.T + b1n
            Gw.add(F.gelu(zC), yB)
            on_B += (zB > 0).float().reshape(-1, len(S)).sum(0)
            on_C += (zC > 0).float().reshape(-1, len(S)).sum(0)
            nrow += zB.reshape(-1, len(S)).shape[0]
        Wn, bn = Gw.solve(ridge_w)                     # (dout, k), (dout,)
        amp_w = None
        if ot_writes:
            Wn, bn, amp_w = Gw.ot(Wn, bn)
        ls = bB.layer_scale.detach().view(-1).cpu().double()
        n1 = nn.Linear(d, len(S)).to(DEV)
        n2 = nn.Linear(len(S), Wn.shape[0]).to(DEV)
        n1.weight.copy_(W1n)
        n1.bias.copy_(b1n)
        n2.weight.copy_((Wn / ls[:, None]).float())
        n2.bias.copy_((bn / ls).float())
        bC.block[3], bC.block[5] = n1, n2
        # advance
        for s in batched(N, bs):
            xB[s] = bB(xB[s].float()).half()
            xC[s] = bC(xC[s].float()).half()
        rB, rC = (on_B / nrow).cpu(), (on_C / nrow).cpu()
        act = rB > 0
        ratio = float((rC[act] / rB[act]).median()) if act.any() else float("nan")
        lens_dev = float((L - torch.eye(d, dtype=torch.float64)).norm() / np.sqrt(d))
        stats.append(dict(g=g, on_base=float(rB.mean()), on_coarse=float(rC.mean()), on_ratio_median=ratio,
                          lens_dev=lens_dev, amp_writes=amp_w, sec=round(time.time() - t0, 1)))
        log(f"block {g}: on-rate Base {rB.mean():.5f} coarse {rC.mean():.5f} (median ratio {ratio:.2f}), "
            f"|L-I|/sqrt(d) {lens_dev:.3f}, write OT {amp_w}, {time.time() - t0:.0f}s", fl)
    return mC, stats


def downsample(base, mC, xB, xC, with_lens, ridge_lens, fl):
    dB, dC = base.features[6], mC.features[6]
    N = xB.shape[0]
    if with_lens:
        G = Gram()
        for s in batched(N, 100):
            G.add(dB[0](xC[s].float()).permute(0, 2, 3, 1), dB[0](xB[s].float()).permute(0, 2, 3, 1))
        L, m = G.solve(ridge_lens)
        conv = dC[1]
        W = conv.weight.detach().cpu().double()                     # (1024, 512, 2, 2)
        Wn = torch.einsum("ojhw,ji->oihw", W, L)
        bn = conv.bias.detach().cpu().double() + torch.einsum("ojhw,j->o", W, m)
        conv.weight.copy_(Wn.float().to(DEV))
        conv.bias.copy_(bn.float().to(DEV))
        log(f"downsample lens |L-I|/sqrt(d) {float((L - torch.eye(512, dtype=torch.float64)).norm() / np.sqrt(512)):.3f}", fl)
    nB = torch.empty(N, 1024, 7, 7, dtype=torch.float16, device=DEV)
    nC = torch.empty_like(nB)
    for s in batched(N, 100):
        nB[s] = dB(xB[s].float()).half()
        nC[s] = dC(xC[s].float()).half()
    return nB, nC


def head_feats(m, cache, rows, bs=50):
    """LayerNorm output of the head (N, 1024) for the given cache rows."""
    out = []
    for s in batched(len(rows), bs):
        x, _ = cache.get(rows[s])
        h = m.features[7](m.features[6](m.features[5](x)))
        out.append(m.classifier[0](m.avgpool(h)).flatten(1))
    return torch.cat(out)


def add_head_lens(base, mX, caches, ridge, fl):
    G = Gram()
    for cache in caches:
        rows = np.arange(len(cache))
        for s in batched(len(rows), 500):
            G.add(head_feats(mX, cache, rows[s]), head_feats(base, cache, rows[s]))
    L, m = G.solve(ridge)
    lin = mX.classifier[2]
    W = lin.weight.detach().cpu().double()
    lin.bias.copy_((lin.bias.detach().cpu().double() + W @ m).float().to(DEV))
    lin.weight.copy_((W @ L).float().to(DEV))
    log(f"head lens on {G.n} photos, |L-I|/sqrt(d) {float((L - torch.eye(1024, dtype=torch.float64)).norm() / 32):.3f}", fl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lens", choices=["base", "seq", "mean", "lens", "B"])
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--fit", default="train", help="train (per_class per class) or cal (the N6 calibration photos)")
    ap.add_argument("--per_class", type=int, default=4)
    ap.add_argument("--ridge_lens", type=float, default=1e-3)
    ap.add_argument("--ridge_w", type=float, default=1e-6)
    ap.add_argument("--ds_lens", action="store_true")
    ap.add_argument("--head_lens", action="store_true")
    ap.add_argument("--head_ridge", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true", help="100 fit photos, 3 changed blocks, 300 test photos")
    ap.add_argument("--nblocks", type=int, default=30)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--ot_lens", action="store_true")
    ap.add_argument("--ot_writes", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or f"{a.mode}_f{a.frac}_rl{a.ridge_lens}" + ("_ds" if a.ds_lens else "") + ("_head" if a.head_lens else "") \
        + ("_smoke" if a.smoke else "")
    fl = open(os.path.join(OUT, f"{tag}.log"), "w")
    log(f"{tag} on {DEV}: {vars(a)}", fl)
    lab = C.net.Photos("read").lab
    base = C.load("base")
    cal = C.Cache("cal")
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    log("calibration done", fl)
    te, held = C.Cache("test"), C.Cache("held")
    nblocks = 3 if a.smoke else a.nblocks
    if a.mode == "B":
        mC = copy.deepcopy(base)
        C.build(mC, calib, a.frac, "value")
        stats = []
    else:
        fitc = C.Cache(a.fit)
        rows = np.arange(0, 1000, 10) if a.smoke else (np.arange(len(fitc)) if a.fit == "cal" else fit_rows(fitc, a.per_class, lab))
        xs, _ = fitc.get(rows)
        xs = xs.half()
        log(f"fit photos {len(rows)}", fl)
        mC, stats = build_lens(base, calib, xs, a.mode, a.frac, a.ridge_lens, a.ridge_w, nblocks, a.ds_lens, fl,
                                a.ot_lens, a.ot_writes)
        del xs
    if a.head_lens:
        caches = [C.Cache("cal")] if a.smoke else [C.Cache("train"), C.Cache("trainflip")]
        add_head_lens(base, mC, caches, a.head_ridge, fl)
    te_rows = np.arange(0, 1500, 5) if a.smoke else None
    held_rows = np.arange(0, 2000, 10) if a.smoke else None
    r_te, kl_te = C.evaluate(mC, te, lab, rows_all=te_rows)
    r_he, _ = C.evaluate(mC, held, lab, rows_all=held_rows)
    log(f"RESULT {tag}: test KL {r_te['kl']:.4f} top-1 {r_te['top1']:.4f} | held KL {r_he['kl']:.4f}", fl)
    json.dump(dict(args=vars(a), test=r_te, held=r_he, blocks=stats), open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    np.save(os.path.join(OUT, f"{tag}_rows.npy"), kl_te)
    if a.save:
        torch.save(dict(args=vars(a), state=[mC.features[s].state_dict() for s in (5, 6, 7)],
                        head=mC.classifier.state_dict()), os.path.join(OUT, f"{tag}.pt"))


if __name__ == "__main__":
    main()
