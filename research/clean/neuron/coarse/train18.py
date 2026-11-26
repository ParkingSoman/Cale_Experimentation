"""N11 step 1: train network B on ResNet-18 (stage-4 inner layers with k neurons, from random) by KL to A.

Only the new conv1, bn1, conv2, bn2 of the two stage-4 blocks train (their BatchNorms in train mode while training).
AdamW, warmup + cosine, batches from train + flips. Learning rate from a short sweep on held photos. Stop point =
the evaluation with the least held KL. Checkpoints at --save_at steps.
Saved in results/coarse/r18/: B_k<k>_s<seed>.pt / .json / .log / _rows.npy (test KL per photo), ckpt/.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import r18


def set_train(m, on, blocks=(0, 1)):
    m.eval()
    if on:
        for mod in r18.new_modules(m, blocks):
            if isinstance(mod, nn.BatchNorm2d):
                mod.train()


def train(m, tr, trf, held, lr, steps, bs, wd, log, every, ckpt=None, save_at=()):
    ps = r18.new_params(m)
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    rng = np.random.default_rng(0)
    warm = max(1, steps // 20)
    h0, _ = r18.evaluate(m, held)
    curve, best = [dict(step=0, held_kl=h0["kl"])], (h0["kl"], r18.state(m), 0)
    t0 = time.time()
    for step in range(steps):
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for gr in opt.param_groups:
            gr["lr"] = lr * f
        x, z = (trf if rng.random() < 0.5 else tr).get(rng.choice(len(tr), bs, replace=False))
        set_train(m, True)
        with torch.enable_grad():
            y = r18.run4(m, x)
            loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
        opt.step()
        set_train(m, False)
        if ckpt and (step + 1) in save_at:
            os.makedirs(os.path.join(r18.OUT, "ckpt"), exist_ok=True)
            torch.save(dict(state=r18.state(m), step=step + 1, loss=float(loss)),
                       os.path.join(r18.OUT, "ckpt", f"{ckpt}_{step + 1}.pt"))
        if (step + 1) % every == 0 or step + 1 == steps:
            h, _ = r18.evaluate(m, held)
            curve.append(dict(step=step + 1, held_kl=h["kl"], loss=float(loss), sec=round(time.time() - t0)))
            log(f"  step {step + 1}/{steps} loss {float(loss):.4f} held KL {h['kl']:.4f} {time.time() - t0:.0f}s")
            if h["kl"] < best[0]:
                best = (h["kl"], r18.state(m), step + 1)
    for p in ps:
        p.requires_grad_(False)
    return curve, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lrs", default="1e-3,3e-3")
    ap.add_argument("--sweep_steps", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--evals", type=int, default=30)
    ap.add_argument("--save_at", default="1,3,10,30,100,300,1000,3000")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = f"B_k{a.k}_s{a.seed}" + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(r18.OUT, f"{tag}.log"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    lab = r18.net.Photos("read").lab
    tr, trf, held, te = r18.Cache("train", True), r18.Cache("trainflip", True), r18.Cache("held"), r18.Cache("test")
    A = r18.load_a()
    a_test, _ = r18.evaluate(A, te, lab)
    log(f"{tag}: train {len(tr)} (+flips), held {len(held)}, test {len(te)}, device {r18.DEV}. A on test: {a_test}")
    m = r18.make_b(A, a.k, a.seed)
    n_par = sum(p.numel() for p in r18.new_params(m))
    log(f"B at init: held KL {r18.evaluate(m, held)[0]['kl']:.4f}, trained numbers {n_par}")
    start = r18.state(m)
    sweep = {}
    for lr in [float(v) for v in a.lrs.split(",")]:
        r18.load_state(m, start)
        curve, _ = train(m, tr, trf, held, lr, 20 if a.smoke else a.sweep_steps, a.bs, a.wd, log, 10 ** 9)
        sweep[lr] = curve[-1]["held_kl"]
        log(f"sweep lr {lr}: held KL {sweep[lr]:.4f}")
    lr = min(sweep, key=sweep.get)
    steps = 60 if a.smoke else int(a.epochs * 2 * len(tr) / a.bs)
    save_at = tuple(int(v) for v in a.save_at.split(",") if v) + (steps,)
    log(f"main run: lr {lr}, {steps} steps")
    r18.load_state(m, start)
    curve, best = train(m, tr, trf, held, lr, steps, a.bs, a.wd, log, max(1, steps // a.evals), ckpt=tag, save_at=save_at)
    r18.load_state(m, best[1])
    t_test, t_rows = r18.evaluate(m, te, lab)
    fit, _ = r18.evaluate(m, tr, rows_all=np.arange(0, len(tr), 8))
    res = dict(k=a.k, seed=a.seed, lr=lr, sweep={str(k): v for k, v in sweep.items()}, wd=a.wd, steps=steps,
               stop_step=best[2], held_kl=best[0], fit_kl=fit["kl"], test=t_test, A_test=a_test, curve=curve,
               trained_numbers=n_par)
    log(f"B best: stop step {best[2]}, held KL {best[0]:.4f}, fit KL {fit['kl']:.4f}, test {t_test}")
    np.save(os.path.join(r18.OUT, f"{tag}_rows.npy"), t_rows)
    torch.save(dict(state=best[1], k=a.k, seed=a.seed, lr=lr), os.path.join(r18.OUT, f"{tag}.pt"))
    json.dump(res, open(os.path.join(r18.OUT, f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
