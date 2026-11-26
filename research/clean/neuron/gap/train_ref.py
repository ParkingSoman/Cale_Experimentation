"""N6 step 1: the trained reference for the whole network, same form as the build (keep 50% in all 30 blocks).

Runs (--run):
  T         start: build with value survivors. Trains the kept MLP numbers (reads, writes, biases), loss = KL to Base.
  T-random  the same, random survivors.         T-greedy  the same, greedy survivors.
  T-local   start as T. Each block trained against its OWN Base output on Base's inputs (the build's target),
            loss = sum over blocks of |y_student - y_base|^2 / var(y_base). No end-to-end signal.
  T-all     start as T. Also trains every other stage 3-4 weight (norms, depthwise convs, layer scales, downsample).
Learning rate: picked from --lrs by held-out KL after --sweep_steps steps each. Then --epochs with cosine decay.
The stop point = the epoch with the least held-out KL. Test photos are scored once, at the end.
Saved: results/gap/<run>.pt (survivors, kept weights), <run>.json (curve, fit and held-out KL, test KL and top-1),
       <run>_rows.npy (per-photo test KL). The build B itself: <run>_build.pt (same survivors, before training).
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C


def kept_params(m, run):
    if run == "T-all":
        return [p for s in (5, 6, 7) for p in m.features[s].parameters()]
    ps = []
    for blk in C.blocks(m):
        ps += [blk.block[3].weight, blk.block[3].bias, blk.block[5].weight, blk.block[5].bias]
    return ps


def mlp_state(m):
    """Every weight of stages 3-4 (the kept MLP numbers and, for T-all, the rest)."""
    return [{k: v.detach().cpu().clone() for k, v in m.features[s].state_dict().items()} for s in (5, 6, 7)]


def load_mlp_state(m, st):
    for s, d in zip((5, 6, 7), st):
        m.features[s].load_state_dict(d)


def local_loss(m, base, x3):
    """Sum over blocks of the relative squared error of the student MLP on Base's block inputs."""
    store = {}
    hs = []
    for g, blk in enumerate(C.blocks(base)):
        hs.append(blk.block[3].register_forward_hook(lambda mod, inp, out, g=g: store.__setitem__(("u", g), inp[0])))
        hs.append(blk.block[5].register_forward_hook(lambda mod, inp, out, g=g: store.__setitem__(("y", g), out)))
    with torch.no_grad():
        C.tail(base, x3)
    for h in hs:
        h.remove()
    loss = 0.0
    for g, blk in enumerate(C.blocks(m)):
        ls = blk.layer_scale.view(-1)
        u, y = store[("u", g)], store[("y", g)] * ls
        ys = blk.block[5](F.gelu(blk.block[3](u))) * ls
        loss = loss + (ys - y).pow(2).mean() / y.var().clamp_min(1e-12)
    return loss


SAVE_AT = ()          # steps after which the kept state is saved (N8 checkpoints); empty = off


def train(m, base, run, tr, trf, held, lr, steps, bs, log, every, ckpt_tag=None):
    ps = kept_params(m, run)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    n = len(tr)
    h0, _ = C.evaluate(m, held)
    curve, best = [dict(step=0, held_kl=h0["kl"])], (h0["kl"], mlp_state(m), 0)
    t0 = time.time()
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        rows = rng.choice(n, bs, replace=False)
        flip = rng.random() < 0.5
        x, z = (trf if flip else tr).get(rows)
        with torch.enable_grad():
            if run == "T-local":
                loss = local_loss(m, base, x)
            else:
                y = C.tail(m, x)
                loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
        opt.step()
        if ckpt_tag and (step + 1) in SAVE_AT:
            os.makedirs(os.path.join(C.OUT, "ckpt"), exist_ok=True)
            torch.save(dict(state=mlp_state(m), step=step + 1, loss=float(loss)),
                       os.path.join(C.OUT, "ckpt", f"{ckpt_tag}_{step + 1}.pt"))
        if (step + 1) % every == 0 or step + 1 == steps:
            h, _ = C.evaluate(m, held)
            curve.append(dict(step=step + 1, held_kl=h["kl"], loss=float(loss), sec=round(time.time() - t0)))
            log(f"  step {step + 1}/{steps} loss {float(loss):.4f} held KL {h['kl']:.4f} {time.time() - t0:.0f}s")
            if h["kl"] < best[0]:
                best = (h["kl"], mlp_state(m), step + 1)
    for p in m.parameters():
        p.requires_grad_(False)
    return curve, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="T")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    ap.add_argument("--sweep_steps", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--evals", type=int, default=12, help="held-out evaluations over the main run")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save_at", default="", help="N8: comma list of steps to save the kept state at (main run only)")
    a = ap.parse_args()
    global SAVE_AT
    SAVE_AT = tuple(int(v) for v in a.save_at.split(",") if v)
    if a.smoke:
        C.CACHE = os.path.join(C.OUT, "cache_smoke")
    tag = a.run + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(C.OUT, f"{tag}.log"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    ph_lab = C.net.Photos("read").lab
    tr, trf, held, cal, te = (C.Cache(n) for n in ("train", "trainflip", "held", "cal", "test"))
    log(f"{tag}: train {len(tr)} (+flips), held {len(held)}, cal {len(cal)}, test {len(te)}, device {C.DEV}")
    base = C.load("base")
    for p in base.parameters():
        p.requires_grad_(False)
    t0 = time.time()
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    log(f"calibration {time.time() - t0:.0f}s")
    how = {"T-random": "random", "T-greedy": "greedy"}.get(a.run, "value")   # T-ckpt: as T
    m = copy.deepcopy(base)
    t0 = time.time()
    Ss = C.build(m, calib, a.frac, how)
    log(f"build ({how} survivors) {time.time() - t0:.0f}s")
    torch.save(dict(S=Ss, state=mlp_state(m), how=how, frac=a.frac), os.path.join(C.OUT, f"{tag}_build.pt"))
    b_test, b_rows = C.evaluate(m, te, ph_lab)
    b_held, _ = C.evaluate(m, held)
    FIT = np.arange(0, len(tr), 8)
    b_fit, _ = C.evaluate(m, tr, rows_all=FIT)
    log(f"build: test {b_test}, held KL {b_held['kl']:.4f}, fit KL {b_fit['kl']:.4f}")
    res = dict(run=a.run, how=how, frac=a.frac, build_test=b_test, build_held=b_held["kl"], build_fit=b_fit["kl"])
    np.save(os.path.join(C.OUT, f"{tag}_build_rows.npy"), b_rows)

    start = mlp_state(m)
    sweep = {}
    for lr in [float(v) for v in a.lrs.split(",")]:
        load_mlp_state(m, start)
        log(f"sweep lr {lr}")
        curve, best = train(m, base, a.run, tr, trf, held, lr, a.sweep_steps, a.bs, log, a.sweep_steps)
        sweep[lr] = curve[-1]["held_kl"]
    lr = min(sweep, key=sweep.get)
    res["sweep"] = {str(k): v for k, v in sweep.items()}
    res["lr"] = lr
    steps = int(a.epochs * 2 * len(tr) / a.bs)
    log(f"main run: lr {lr}, {steps} steps ({a.epochs} epochs over train + flips)")
    load_mlp_state(m, start)
    curve, best = train(m, base, a.run, tr, trf, held, lr, steps, a.bs, log, max(1, steps // a.evals),
                        ckpt_tag=a.run if SAVE_AT else None)
    res["curve"] = curve
    res["stop_step"] = best[2]
    load_mlp_state(m, best[1])
    t_test, t_rows = C.evaluate(m, te, ph_lab)
    t_fit, _ = C.evaluate(m, tr, rows_all=FIT)
    res.update(trained_test=t_test, trained_held=best[0], trained_fit=t_fit["kl"])
    d = b_rows - t_rows
    bs_ = np.random.default_rng(1).choice(len(d), (2000, len(d)))
    res["build_minus_trained_kl_ci95"] = [float(np.percentile(d[bs_].mean(1), 2.5)), float(np.percentile(d[bs_].mean(1), 97.5))]
    log(f"trained: test {t_test}, held KL {best[0]:.4f}, fit KL {t_fit['kl']:.4f}, stop step {best[2]}")
    np.save(os.path.join(C.OUT, f"{tag}_rows.npy"), t_rows)
    torch.save(dict(S=Ss, state=best[1], how=how, frac=a.frac, lr=lr), os.path.join(C.OUT, f"{tag}.pt"))
    json.dump(res, open(os.path.join(C.OUT, f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
