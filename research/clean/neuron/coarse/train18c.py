"""N11 step 1 on the cluster: train network B (ResNet-18, stage-4 inner layers with k neurons, from random) by KL to A,
with fresh augmentation every step, so B cannot memorize its photos.

Train photos (r18.photo_sets "train", 16,000) are decoded once from the N6 JPEG bundle: the stored image (short side 256)
center-cropped to 256 x 256, kept as uint8 on the GPU. Each step: a random resized crop (area 25-100%, aspect 3/4-4/3)
to 224 (roi_align), a left-right flip half the time, and with probability --mix a blend of two photos (mixup,
weight ~ Uniform(0, 1)). A's frozen stages 1-3 give the stage-4 input; A's stage 4 gives the target on that same
input; B's stage 4 learns it (KL). Held KL and test KL use the r18 center-crop caches (cache18.py), as the reading does.
Saved in results/coarse/r18/: B<k>aug_s<seed>.pt / .json / .log / _rows.npy, ckpt/.
"""
import argparse
import glob
import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

import r18
from train18 import set_train

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_train_images(jpeg_dir, ids):
    want = {int(v): i for i, v in enumerate(ids)}
    X = torch.zeros((len(ids), 3, 256, 256), dtype=torch.uint8)
    got = 0
    for f in sorted(glob.glob(os.path.join(jpeg_dir, "jpeg_*.npz"))):
        z = np.load(f)
        zids, off, data = z["ids"], z["off"], z["data"]          # read each array once (np.load of an npz is lazy)
        for i, vi in enumerate(zids):
            if int(vi) in want:
                img = Image.open(io.BytesIO(data[off[i]:off[i + 1]].tobytes())).convert("RGB")
                w, h = img.size
                s = 256 / min(w, h)
                if s != 1:
                    img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
                    w, h = img.size
                l, t = (w - 256) // 2, (h - 256) // 2
                X[want[int(vi)]] = torch.from_numpy(np.asarray(img.crop((l, t, l + 256, t + 256))).copy()).permute(2, 0, 1)
                got += 1
    assert got == len(ids), (got, len(ids))
    return X


def batch(X, bs, g, mix):
    idx = torch.randint(0, len(X), (bs,), generator=g)
    img = X[idx.to(X.device)].float() / 255.0
    area = torch.empty(bs).uniform_(0.25, 1.0, generator=g) * 256 * 256
    logr = torch.empty(bs).uniform_(math.log(3 / 4), math.log(4 / 3), generator=g)
    w = (area * logr.exp()).sqrt().clamp(32, 256)
    h = (area / logr.exp()).sqrt().clamp(32, 256)
    x0 = torch.rand(bs, generator=g) * (256 - w)
    y0 = torch.rand(bs, generator=g) * (256 - h)
    boxes = torch.stack([torch.arange(bs).float(), x0, y0, x0 + w, y0 + h], 1).to(img.device)
    out = roi_align(img, boxes, output_size=224, spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flip = (torch.rand(bs, generator=g) < 0.5).to(img.device)
    out = torch.where(flip.view(-1, 1, 1, 1), out.flip(3), out)
    if torch.rand(1, generator=g).item() < mix:
        lam = torch.rand(bs, generator=g).to(img.device).view(-1, 1, 1, 1)
        out = lam * out + (1 - lam) * out[torch.randperm(bs, generator=g).to(img.device)]
    return (out - MEAN.to(img.device)) / STD.to(img.device), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg_dir", required=True)
    ap.add_argument("--lrs", default="1e-3,3e-3")
    ap.add_argument("--sweep_steps", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=100)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--evals", type=int, default=25)
    ap.add_argument("--save_at", default="1,3,10,30,100,300,1000,3000")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--blocks", default="0,1", help="stage-4 blocks rebuilt narrow (the others stay A's, frozen)")
    ap.add_argument("--suffix", default="", help="added to every output file name (e.g. _x2 for a 2x-steps rerun)")
    a = ap.parse_args()
    blocks = tuple(int(v) for v in a.blocks.split(","))
    tag = f"B{a.k}aug_s{a.seed}" + ("" if blocks == (0, 1) else "_b" + "".join(map(str, blocks))) + a.suffix + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(r18.OUT, f"{tag}.log"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    rows = sets["train"][::31] if a.smoke else sets["train"]      # smoke: spread over all classes (the list is class-sorted)
    ids = ph.ix[rows]
    t0 = time.time()
    X = load_train_images(a.jpeg_dir, ids).to(r18.DEV)
    log(f"{tag}: {len(X)} train photos decoded in {time.time() - t0:.0f}s, device {r18.DEV}")
    held, te, tr = r18.Cache("held"), r18.Cache("test"), r18.Cache("train")
    A = r18.load_a()
    a_test, _ = r18.evaluate(A, te, ph.lab)
    log(f"A on test (center crops): {a_test}")
    with torch.no_grad():                                         # check the crops: A's top-1 on augmented photos
        xa, ia = batch(X, 512, torch.Generator().manual_seed(7), 0.0)
        acc = float((r18.net.head(A, A.layer4(A.layer3(A.layer2(A.layer1(r18.net.stem(A, xa)))))).argmax(1).cpu()
                     == torch.as_tensor(ph.lab[rows][ia.numpy()])).float().mean())
    log(f"A top-1 on 512 augmented train crops (no mixup): {acc:.3f}")
    m = r18.make_b(A, a.k, a.seed, blocks)
    log(f"B at init: held KL {r18.evaluate(m, held)[0]['kl']:.4f}, trained numbers {sum(p.numel() for p in r18.new_params(m, blocks))}")
    start = r18.state(m)

    def run(lr, steps, every, ckpt=None, save_at=()):
        ps = r18.new_params(m, blocks)
        for p in ps:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(ps, lr=lr, weight_decay=a.wd)
        g = torch.Generator().manual_seed(1000 + a.seed)
        warm = max(1, steps // 20)
        best = (r18.evaluate(m, held)[0]["kl"], r18.state(m), 0)
        curve = [dict(step=0, held_kl=best[0])]
        t0 = time.time()
        for step in range(steps):
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for gr in opt.param_groups:
                gr["lr"] = lr * f
            with torch.no_grad():
                x, _ = batch(X, a.bs, g, a.mix)
                h = A.layer3(A.layer2(A.layer1(r18.net.stem(A, x))))
                z = r18.run4(A, h)
            set_train(m, True, blocks)
            with torch.enable_grad():
                y = r18.run4(m, h)
                loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
                opt.zero_grad(set_to_none=True)
                loss.backward()
            opt.step()
            set_train(m, False, blocks)
            if ckpt and (step + 1) in save_at:
                os.makedirs(os.path.join(r18.OUT, "ckpt"), exist_ok=True)
                torch.save(dict(state=r18.state(m), step=step + 1, loss=float(loss.detach())),
                           os.path.join(r18.OUT, "ckpt", f"{ckpt}_{step + 1}.pt"))
            if (step + 1) % every == 0 or step + 1 == steps:
                hk = r18.evaluate(m, held)[0]["kl"]
                curve.append(dict(step=step + 1, held_kl=hk, loss=float(loss.detach()), sec=round(time.time() - t0)))
                log(f"  step {step + 1}/{steps} loss {float(loss.detach()):.4f} held KL {hk:.4f} {time.time() - t0:.0f}s")
                if hk < best[0]:
                    best = (hk, r18.state(m), step + 1)
        for p in ps:
            p.requires_grad_(False)
        return curve, best

    sweep = {}
    for lr in [float(v) for v in a.lrs.split(",")]:
        r18.load_state(m, start)
        curve, _ = run(lr, 20 if a.smoke else a.sweep_steps, 10 ** 9)
        sweep[lr] = curve[-1]["held_kl"]
        log(f"sweep lr {lr}: held KL {sweep[lr]:.4f}")
    lr = min(sweep, key=sweep.get)
    steps = 60 if a.smoke else int(a.epochs * len(X) / a.bs)
    save_at = tuple(int(v) for v in a.save_at.split(",") if v) + (steps,)
    log(f"main run: lr {lr}, {steps} steps of {a.bs}")
    r18.load_state(m, start)
    curve, best = run(lr, steps, max(1, steps // a.evals), ckpt=tag, save_at=save_at)
    r18.load_state(m, best[1])
    t_test, t_rows = r18.evaluate(m, te, ph.lab)
    fit, _ = r18.evaluate(m, tr, rows_all=np.arange(0, len(tr), 8))
    res = dict(k=a.k, seed=a.seed, lr=lr, sweep={str(k): v for k, v in sweep.items()}, wd=a.wd, mix=a.mix, steps=steps,
               stop_step=best[2], held_kl=best[0], fit_kl=fit["kl"], test=t_test, A_test=a_test, curve=curve)
    log(f"B best: stop step {best[2]}, held KL {best[0]:.4f}, fit KL (center crops) {fit['kl']:.4f}, test {t_test}")
    np.save(os.path.join(r18.OUT, f"{tag}_rows.npy"), t_rows)
    torch.save(dict(state=best[1], k=a.k, seed=a.seed, lr=lr, blocks=blocks), os.path.join(r18.OUT, f"{tag}.pt"))
    json.dump(res, open(os.path.join(r18.OUT, f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
