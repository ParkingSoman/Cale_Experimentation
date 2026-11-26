"""Distill the whole of A (torchvision ResNet-18) into a student with every width times f (plan 2026-11-25).

Starts:  R = from random.  P = A cut to width f by Minitron activation scores (wide18.cut_from_a), then distilled.
Photos:  the reading photos only (35 per class; never the other set). Per class (sorted): fit = even positions,
         odd = odd positions. held = fit[8:10] (as before), test = odd[:5], train = the other 28.
Train:   JPEG bundle, 256 x 256 centre crop, then each step a random-resized crop + flip + mixup on half the batches
         (train18c.batch). A scores the same input. Loss KL(A || student), temperature 1. AdamW, wd .05, warm-up, cosine.
Score:   held and test = the 224 centre crops (val_224, as all earlier reads). A's logits computed once.
Saved:   neuron/results/coarse/r18/distill_whole/<tag>.pt / .json / .log / _testkl.npy, ckpt/<tag>_<step>.pt, and for P
         <tag>_scores.pt (scores + kept sets).
Usage:   python train_whole.py --f 0.5 --start P --seed 0 --jpeg_dir /bigtemp/fkm5yp/n6 [--epochs 300] [--smoke]
         python train_whole.py --check        (no training: shape, cut-at-100% and forward checks)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "neuron", "coarse"))
sys.path.insert(0, HERE)
import r18  # noqa: E402
from train18c import batch  # noqa: E402
import wide18  # noqa: E402

DEV = r18.DEV
OUT = os.path.join(HERE, "..", "neuron", "results", "coarse", "r18", "distill_whole")
LRS = {"R": "3e-4,1e-3,3e-3,1e-2", "P": "3e-5,1e-4,3e-4,1e-3"}


def split(lab):
    tr, held, te = [], [], []
    for c in range(1000):
        w = np.where(lab == c)[0]
        fit, odd = w[0::2], w[1::2]
        held += fit[8:10].tolist()
        te += odd[:5].tolist()
        tr += [i for k, i in enumerate(fit) if k not in (8, 9)] + odd[5:].tolist()
    return dict(train=np.array(tr), held=np.array(held), test=np.array(te))


def load_train(jpeg_dir, ph, rows):
    """256 x 256 uint8 train photos. From the JPEG bundle when it holds the photo (the old train photos), else from the
    224 centre crop (val_224) scaled up to 256 (the old test photos moved into training). Returns X and the JPEG count."""
    import glob
    import io
    from PIL import Image
    ids = ph.ix[rows]
    want = {int(v): i for i, v in enumerate(ids)}
    X = torch.zeros((len(ids), 3, 256, 256), dtype=torch.uint8)
    have = np.zeros(len(ids), bool)
    for f in sorted(glob.glob(os.path.join(jpeg_dir, "jpeg_*.npz"))):
        z = np.load(f)
        zids, off, data = z["ids"], z["off"], z["data"]
        for i, vi in enumerate(zids):
            j = want.get(int(vi))
            if j is None:
                continue
            img = Image.open(io.BytesIO(data[off[i]:off[i + 1]].tobytes())).convert("RGB")
            w, h = img.size
            s = 256 / min(w, h)
            if s != 1:
                img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
                w, h = img.size
            l, t = (w - 256) // 2, (h - 256) // 2
            X[j] = torch.from_numpy(np.asarray(img.crop((l, t, l + 256, t + 256))).copy()).permute(2, 0, 1)
            have[j] = True
    miss = np.where(~have)[0]
    for a in range(0, len(miss), 500):
        part = miss[a:a + 500]
        raw = torch.from_numpy(ph.raw(rows[part]).copy()).permute(0, 3, 1, 2).float()
        up = F.interpolate(raw, size=256, mode="bilinear", align_corners=False, antialias=True)
        X[torch.as_tensor(part)] = up.round().clamp(0, 255).to(torch.uint8)
    return X, int(have.sum())


def photos(ph, rows, dtype=torch.float16):
    out = []
    for a in range(0, len(rows), 500):
        out.append(ph.tensor(rows[a:a + 500]).to(DEV, dtype))
    return torch.cat(out).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def logits(m, x, bs=500):
    out = []
    for a in range(0, len(x), bs):
        with torch.autocast("cuda", torch.bfloat16, enabled=DEV == "cuda"):
            out.append(m(x[a:a + bs].float()).float())
    return torch.cat(out)


def kl_rows(z, y):
    lz, ly = F.log_softmax(z, 1), F.log_softmax(y, 1)
    return (lz.exp() * (lz - ly)).sum(1)


@torch.no_grad()
def score(m, x, z, lab=None):
    was = m.training
    m.eval()
    y = logits(m, x)
    m.train(was)
    k = kl_rows(z, y)
    out = dict(kl=float(k.mean()))
    if lab is not None:
        out["top1"] = float((y.argmax(1).cpu() == torch.as_tensor(lab)).float().mean())
    return out, k.cpu().numpy()


def check():
    A = r18.load_a()
    x = torch.randn(8, 3, 224, 224, device=DEV)
    S1 = wide18.make_student(1.0).to(DEV)
    sa, ss = A.state_dict(), S1.state_dict()
    assert sa.keys() == ss.keys(), set(sa) ^ set(ss)
    assert all(sa[k].shape == ss[k].shape for k in sa), "shape mismatch at 100%"
    print("CHECK 1 pass: make_student(1.0) has A's exact keys and shapes")
    sc = wide18.channel_scores(A, x)
    P1, _ = wide18.cut_from_a(A, 1.0, sc)
    P1.eval()
    with torch.no_grad():
        d = float((P1(x) - A(x)).abs().max())
    print(f"CHECK 2 {'pass' if d < 1e-4 else 'FAIL'}: cut_from_a at 100% vs A, max logit difference {d:.2e}")
    for f in (0.95, 0.5, 0.05):
        S = wide18.make_student(f).to(DEV).eval()
        P, kept = wide18.cut_from_a(A, f, sc)
        P.eval()
        with torch.no_grad():
            y, yp = S(x), P(x)
        print(f"CHECK 3 f={f}: widths {wide18.widths(f)}, neurons {wide18.n_neurons(S)} of {wide18.n_neurons(S1)}, "
              f"weights {sum(p.numel() for p in S.parameters())} of {sum(p.numel() for p in S1.parameters())}, "
              f"out {tuple(y.shape)}, P finite {bool(torch.isfinite(yp).all())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float)
    ap.add_argument("--start", choices=("R", "P"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg_dir", default="/bigtemp/fkm5yp/n6")
    ap.add_argument("--lrs", default="")
    ap.add_argument("--sweep_steps", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=300)
    ap.add_argument("--max_minutes", type=float, default=90, help="cap on the main run, from the speed seen in the sweep")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--evals", type=int, default=30)
    ap.add_argument("--save_at", default="1,3,10,30,100,300,1000,3000,10000")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    tag = f"W{round(a.f * 100)}{a.start}_s{a.seed}{a.suffix}" + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(OUT, f"{tag}.log"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    torch.backends.cudnn.benchmark = True
    ph = r18.net.Photos("read")
    sets = split(ph.lab)
    log(f"{tag}: photos per class train {len(sets['train']) / 1000:.1f}, held {len(sets['held']) / 1000:.1f}, "
        f"test {len(sets['test']) / 1000:.1f}")
    rows = sets["train"][::31] if a.smoke else sets["train"]
    t0 = time.time()
    X, n_jpeg = load_train(a.jpeg_dir, ph, rows)
    X = X.to(DEV)
    log(f"{len(X)} train photos in {time.time() - t0:.0f}s ({n_jpeg} from JPEG at 256, {len(X) - n_jpeg} from the 224 "
        f"centre crop scaled to 256), device {DEV}")
    A = r18.load_a().to(memory_format=torch.channels_last)
    xh, xt = photos(ph, sets["held"]), photos(ph, sets["test"])
    zh, zt = logits(A, xh), logits(A, xt)
    a_test = dict(kl=0.0, top1=float((zt.argmax(1).cpu() == torch.as_tensor(ph.lab[sets["test"]])).float().mean()))
    log(f"A on the new test set (centre crops): top-1 {a_test['top1']:.4f}")

    torch.manual_seed(a.seed)
    if a.start == "R":
        m = wide18.make_student(a.f, a.seed).to(DEV)
        extra = {}
    else:
        cal = photos(ph, sets["train"][::14], torch.float32)
        sc = wide18.channel_scores(A, cal)
        del cal
        m, kept = wide18.cut_from_a(A, a.f, sc)
        torch.save(dict(scores=sc, kept=kept), os.path.join(OUT, f"{tag}_scores.pt"))
        extra = dict(calibration_photos=len(sets["train"][::14]))
    m = m.to(memory_format=torch.channels_last)
    n_w = sum(p.numel() for p in m.parameters())
    log(f"student f={a.f} start {a.start}: widths {wide18.widths(a.f)}, neurons {wide18.n_neurons(m)}, weights {n_w}; "
        f"held KL at start {score(m, xh, zh)[0]['kl']:.4f}")
    start = {k: v.detach().clone() for k, v in m.state_dict().items()}

    def run(lr, steps, every, ckpt=None, save_at=()):
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=a.wd)
        g = torch.Generator().manual_seed(1000 + a.seed)
        warm = max(1, steps // 20)
        best = (score(m, xh, zh)[0]["kl"], {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, 0)
        curve = [dict(step=0, held_kl=best[0])]
        m.train()
        t0 = time.time()
        for step in range(steps):
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for gr in opt.param_groups:
                gr["lr"] = lr * f
            x, _ = batch(X, a.bs, g, a.mix)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=DEV == "cuda"):
                z = A(x).float()
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV == "cuda"):
                y = m(x).float()
            loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if not torch.isfinite(loss):
                log(f"  NON-FINITE loss at step {step + 1}, lr {lr}: stop this run")
                curve.append(dict(step=step + 1, held_kl=float("inf"), loss=float("nan"), sec=round(time.time() - t0)))
                break
            if ckpt and (step + 1) in save_at:
                torch.save(dict(state=m.state_dict(), step=step + 1, loss=float(loss.detach())),
                           os.path.join(OUT, "ckpt", f"{ckpt}_{step + 1}.pt"))
            if (step + 1) % every == 0 or step + 1 == steps:
                hk = score(m, xh, zh)[0]["kl"]
                sec = time.time() - t0
                curve.append(dict(step=step + 1, held_kl=hk, loss=float(loss.detach()), sec=round(sec)))
                log(f"  step {step + 1}/{steps} loss {float(loss.detach()):.4f} held KL {hk:.4f} {sec:.0f}s "
                    f"({(step + 1) * a.bs / sec:.0f} photos/s)")
                if hk < best[0]:
                    best = (hk, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, step + 1)
        return curve, best

    sweep, sec_per_step = {}, []
    for lr in [float(v) for v in (a.lrs or LRS[a.start]).split(",")]:
        m.load_state_dict(start)
        n_sw = 20 if a.smoke else a.sweep_steps
        ts = time.time()
        curve, _ = run(lr, n_sw, 10 ** 9)
        sec_per_step.append((time.time() - ts) / n_sw)
        sweep[lr] = curve[-1]["held_kl"] if math.isfinite(curve[-1]["held_kl"]) else float("inf")
        log(f"sweep lr {lr}: held KL {sweep[lr]:.4f} ({sec_per_step[-1]:.3f} s/step)")
    lr = min(sweep, key=sweep.get)
    want = int(a.epochs * len(X) / a.bs)
    fit = int(a.max_minutes * 60 / max(sec_per_step))           # plan: cut to fit if a run would exceed max_minutes
    steps = 60 if a.smoke else min(want, fit)
    save_at = tuple(int(v) for v in a.save_at.split(",") if v) + (steps,)
    log(f"main run: lr {lr}, {steps} steps of {a.bs} = {steps * a.bs / len(X):.0f} epochs of {len(X)} photos "
        f"(wanted {a.epochs:g} epochs = {want} steps; {a.max_minutes} min at {max(sec_per_step):.3f} s/step allows {fit})")
    m.load_state_dict(start)
    t0 = time.time()
    curve, best = run(lr, steps, max(1, steps // a.evals), ckpt=tag, save_at=save_at)
    m.load_state_dict({k: v.to(DEV) for k, v in best[1].items()})
    t_test, t_rows = score(m, xt, zt, ph.lab[sets["test"]])
    h_best = score(m, xh, zh, ph.lab[sets["held"]])[0]
    res = dict(f=a.f, start=a.start, seed=a.seed, widths=wide18.widths(a.f), neurons=wide18.n_neurons(m), weights=n_w,
               lr=lr, sweep={str(k): v for k, v in sweep.items()}, wd=a.wd, mix=a.mix, bs=a.bs, epochs=a.epochs,
               steps=steps, steps_wanted=want, sec_per_step=max(sec_per_step), stop_step=best[2], held_kl=best[0], held=h_best, test=t_test, A_test=a_test,
               train_photos=len(X), train_from_jpeg=n_jpeg, minutes=round((time.time() - t0) / 60, 1), curve=curve, **extra)
    log(f"best: stop step {best[2]}, held KL {best[0]:.4f}, test {t_test}")
    np.save(os.path.join(OUT, f"{tag}_testkl.npy"), t_rows)
    torch.save(dict(state=best[1], f=a.f, start=a.start, seed=a.seed, widths=wide18.widths(a.f), lr=lr),
               os.path.join(OUT, f"{tag}.pt"))
    json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
