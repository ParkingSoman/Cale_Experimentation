"""Distill the whole of A (torchvision ResNet-18) into a student with every width times f (plan 2026-11-25).

Starts:  R = from random.  P = A cut to width f by Minitron activation scores (wide18.cut_from_a), then distilled.
         Wave 3 (plan 2026-11-26) adds: C = a saved student cut to width f by its own scores (--source).
         W = a saved student at the same width, trained on (--source). G = two-step cut (--grad 0.75:N): A -> P at
         0.75, N steps, then that student cut to f, the rest of the steps.
Photos:  the reading photos only (35 per class; never the other set). Per class (sorted): fit = even positions,
         odd = odd positions. held = fit[8:10] (as before), test = odd[:5], train = the other 28.
         --extra <dir>: adds ImageNet-TRAIN photos (extra_photos.py, labels unused) to the train set only.
Train:   JPEG bundle, 256 x 256 centre crop, then each step a random-resized crop + flip + mixup on a share --mix of
         the batches (train18c.batch). A scores the same input. Loss KL(A || student), temperature 1. AdamW, warm-up,
         cosine. Wave 3: an EMA copy (weights and BatchNorm buffers) is kept in the main run and scored at every eval
         next to the raw weights; the best of both is kept. --clip = grad-norm clip.
         --multi: several students at one width on the same batches, one A forward per step (wave 3, 5% screen).
Score:   held and test = the 224 centre crops (val_224, as all earlier reads). A's logits computed once.
Saved:   neuron/results/coarse/r18/distill_whole/<tag>.pt / .json / .log / _testkl.npy, ckpt/<tag>_<step>.pt, and for
         P / C / G <tag>_scores.pt (scores + kept sets).
Usage:   python train_whole.py --f 0.5 --start P --seed 0 --jpeg_dir /bigtemp/fkm5yp/n6 [--epochs 300] [--smoke]
         python train_whole.py --f 0.5 --start P --extra /bigtemp/fkm5yp/distill/extra --steps 32812 --suffix _ph
         python train_whole.py --f 0.05 --multi R,Rwd0,Rclip,C50,P --suffix _m
         python train_whole.py --check        (no training: shape, cut-at-100% and forward checks)
The wave 1-2 trainer is kept as old/train_whole_wave12.py.
"""
import argparse
import copy
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
LRS = {"R": "3e-4,1e-3,3e-3,1e-2", "P": "3e-5,1e-4,3e-4,1e-3", "C": "3e-4,1e-3,3e-3,1e-2", "W": "3e-5,1e-4,3e-4,1e-3",
       "G": "3e-5,1e-4,3e-4,1e-3"}
# --multi variants (wave 3, 5% screen). Every R variant starts from the same random init (make_student(f, seed)).
VARIANTS = {
    "R": dict(start="R", lrs="3e-3,1e-2,3e-2"),
    "Rwd0": dict(start="R", lrs="3e-3,1e-2,3e-2", wd=0.0),
    "Rclip": dict(start="R", lrs="3e-3,1e-2,3e-2", clip=1.0),
    "C50": dict(start="C", source="W50P_s0_x2", lrs="3e-4,1e-3,3e-3,1e-2"),
    "P": dict(start="P", lrs="1e-3,3e-3,1e-2"),
}


def split(lab):
    tr, held, te = [], [], []
    for c in range(1000):
        w = np.where(lab == c)[0]
        fit, odd = w[0::2], w[1::2]
        held += fit[8:10].tolist()
        te += odd[:5].tolist()
        tr += [i for k, i in enumerate(fit) if k not in (8, 9)] + odd[5:].tolist()
    return dict(train=np.array(tr), held=np.array(held), test=np.array(te))


def crop256(raw):
    """JPEG bytes -> 256 x 256 x 3 uint8: short side scaled to 256, centre crop (as the n6 bundle reads)."""
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    if s != 1:
        img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
        w, h = img.size
    l, t = (w - 256) // 2, (h - 256) // 2
    return np.asarray(img.crop((l, t, l + 256, t + 256))).copy()


def load_train(jpeg_dir, ph, rows):
    """256 x 256 uint8 train photos. From the JPEG bundle when it holds the photo (the old train photos), else from the
    224 centre crop (val_224) scaled up to 256 (the old test photos moved into training). Returns X and the JPEG count."""
    import glob
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
            X[j] = torch.from_numpy(crop256(data[off[i]:off[i + 1]].tobytes())).permute(2, 0, 1)
            have[j] = True
    miss = np.where(~have)[0]
    for a in range(0, len(miss), 500):
        part = miss[a:a + 500]
        raw = torch.from_numpy(ph.raw(rows[part]).copy()).permute(0, 3, 1, 2).float()
        up = F.interpolate(raw, size=256, mode="bilinear", align_corners=False, antialias=True)
        X[torch.as_tensor(part)] = up.round().clamp(0, 255).to(torch.uint8)
    return X, int(have.sum())


def _decode_file(args):
    path, every = args
    z = np.load(path)
    off, data = z["off"], z["data"]
    idx = range(0, len(off) - 1, every)
    return np.stack([crop256(data[off[i]:off[i + 1]].tobytes()) for i in idx]), z["lab"][::every]


def load_extra(d, device, every=1, head=None):
    """The extra ImageNet-TRAIN photos (extra_photos.py) as (N, 3, 256, 256) uint8 on device, placed after the rows of
    head (CPU uint8) in ONE tensor (no concatenation copy on the GPU). every > 1 = smoke subset. Returns X, labels,
    file count, number of extra photos."""
    import glob
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(d, "extra_*.npz")))
    n = sum((len(np.load(f)["off"]) - 1 + every - 1) // every for f in files)
    h = 0 if head is None else len(head)
    pool = Pool(min(16, os.cpu_count() or 1))         # workers fork before this process touches CUDA
    X = torch.empty((h + n, 3, 256, 256), dtype=torch.uint8, device=device)
    if h:
        X[:h] = head.to(device)
    labs, a = [], h
    with pool:
        for x, lab in pool.imap(_decode_file, [(f, every) for f in files]):
            X[a:a + len(x)] = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
            labs.append(lab)
            a += len(x)
    assert a == h + n, (a, h, n)
    return X, np.concatenate(labs), len(files), n


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


class EMA:
    """EMA copy of a student: weights AND BatchNorm buffers averaged; integer buffers copied. Warm-up (1+n)/(10+n)."""

    def __init__(self, m, decay):
        self.m = copy.deepcopy(m).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, m):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, v in zip(self.m.state_dict().values(), m.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                e.copy_(v)


def load_student(src):
    p = src if os.path.isabs(src) else os.path.join(OUT, src + ".pt")
    ck = torch.load(p, map_location="cpu")
    s = wide18.Student(ck["widths"])
    s.load_state_dict(ck["state"])
    return s.to(DEV).eval(), ck


def cut_student(src, f, cal):
    """Cut a saved (or live) student to width f by its own Minitron scores. Returns (student, scores, kept)."""
    src = src.eval()
    sc = wide18.channel_scores(src, cal)
    m, kept = wide18.cut_from_a(src, f, sc)
    return m, sc, kept


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
    if os.path.exists(os.path.join(OUT, "W50P_s0_x2.pt")):
        s50, _ = load_student("W50P_s0_x2")
        c50, _, _ = cut_student(s50, 0.5, x)
        c50.eval()
        with torch.no_grad():
            d = float((c50(x) - s50(x)).abs().max())
        c5, _, _ = cut_student(s50, 0.05, x)
        with torch.no_grad():
            ok5 = bool(torch.isfinite(c5.eval()(x)).all())
        print(f"CHECK 4 {'pass' if d < 1e-4 else 'FAIL'}: W50P_s0_x2 cut to its own width vs itself, max logit "
              f"difference {d:.2e}; cut to 5%: widths {c5.w}, finite {ok5}")


class Ctx:
    """Photos, A, the batch source and the log of one process."""

    def __init__(self, a, tag):
        self.a = a
        self.logf = open(os.path.join(OUT, f"{tag}.log"), "w")
        ph = r18.net.Photos("read")
        self.ph = ph
        self.sets = split(ph.lab)
        self.log(f"{tag}: photos per class train {len(self.sets['train']) / 1000:.1f}, held "
                 f"{len(self.sets['held']) / 1000:.1f}, test {len(self.sets['test']) / 1000:.1f}")
        rows = self.sets["train"][::31] if a.smoke else self.sets["train"]
        t0 = time.time()
        X, n_jpeg = load_train(a.jpeg_dir, ph, rows)
        self.n_read, self.n_jpeg, self.n_extra = len(X), n_jpeg, 0
        self.log(f"{len(X)} train photos in {time.time() - t0:.0f}s ({n_jpeg} from JPEG at 256, {len(X) - n_jpeg} from "
                 f"the 224 centre crop scaled to 256), device {DEV}")
        if a.extra:
            t0 = time.time()
            X, lab_e, nf, self.n_extra = load_extra(a.extra, DEV, every=50 if a.smoke else 1, head=X)
            cnt = np.bincount(lab_e, minlength=1000)
            self.log(f"extra photos (ImageNet-train, labels unused): {self.n_extra} from {nf} files in "
                     f"{time.time() - t0:.0f}s; per class min {cnt.min()} median {np.median(cnt):.0f} max {cnt.max()}")
        else:
            X = X.to(DEV)
        self.X = X
        self.A = r18.load_a().to(memory_format=torch.channels_last)
        self.xh, self.xt = photos(ph, self.sets["held"]), photos(ph, self.sets["test"])
        self.zh, self.zt = logits(self.A, self.xh), logits(self.A, self.xt)
        self.a_test = dict(kl=0.0, top1=float((self.zt.argmax(1).cpu() == torch.as_tensor(
            ph.lab[self.sets["test"]])).float().mean()))
        self.log(f"A on the new test set (centre crops): top-1 {self.a_test['top1']:.4f}")
        if self.n_extra:                               # loader check: A on extra photos, centre 224 crop, saved labels
            from train18c import MEAN, STD
            idx = torch.arange(self.n_read, len(X), max(1, self.n_extra // 2000), device=X.device)
            xe = (X[idx][:, :, 16:240, 16:240].float() / 255 - MEAN.to(X.device)) / STD.to(X.device)
            ye = logits(self.A, xe.contiguous(memory_format=torch.channels_last))
            t1 = float((ye.argmax(1).cpu() == torch.as_tensor(lab_e[(idx - self.n_read).cpu().numpy()])).float().mean())
            self.log(f"LOADER CHECK: A top-1 on {len(idx)} extra photos (centre 224 crop) {t1:.4f} "
                     f"(A on val test {self.a_test['top1']:.4f})")
            del xe, ye

    def log(self, s):
        print(s, flush=True)
        self.logf.write(s + "\n")
        self.logf.flush()

    def cal(self):
        return photos(self.ph, self.sets["train"][::14], torch.float32)


def train(cx, ms, lrs, steps, every, wds, clips, ckpts=None, save_at=(), ema=None, names=None):
    """Train the students ms together on the same batches (one A forward per step). Each has its own lr, wd, clip and
    optimizer. A student with a non-finite loss stops; the others go on. ema = decay or None (main run only).
    Returns per student (curve, best) with best = (held KL, state on CPU, step, 'raw' or 'ema')."""
    a = cx.a
    opts = [torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd) for m, lr, wd in zip(ms, lrs, wds)]
    emas = [EMA(m, ema) for m in ms] if ema else [None] * len(ms)
    g = torch.Generator().manual_seed(1000 + a.seed)
    warm = max(1, steps // 20)
    names = names or [""] * len(ms)
    best, curves, alive = [], [], [True] * len(ms)
    for m in ms:
        k0 = score(m, cx.xh, cx.zh)[0]["kl"]
        best.append((k0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, 0, "raw"))
        curves.append([dict(step=0, held_kl=k0)])
        m.train()
    t0 = time.time()
    for step in range(steps):
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        x, _ = batch(cx.X, a.bs, g, a.mix)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=DEV == "cuda"):
            z = cx.A(x).float()
        lz = F.log_softmax(z, 1)
        losses = []
        for i, (m, opt) in enumerate(zip(ms, opts)):
            if not alive[i]:
                losses.append(float("nan"))
                continue
            for gr in opt.param_groups:
                gr["lr"] = lrs[i] * f
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV == "cuda"):
                y = m(x).float()
            loss = F.kl_div(F.log_softmax(y, 1), lz, log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clips[i]:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clips[i])
            opt.step()
            if emas[i]:
                emas[i].update(m)
            if not torch.isfinite(loss):
                cx.log(f"  NON-FINITE loss at step {step + 1}, {names[i]} lr {lrs[i]}: stop this student")
                curves[i].append(dict(step=step + 1, held_kl=float("inf"), loss=float("nan"), sec=round(time.time() - t0)))
                alive[i] = False
            losses.append(float(loss.detach()))
            if ckpts and (step + 1) in save_at:
                torch.save(dict(state=m.state_dict(), step=step + 1, loss=losses[-1]),
                           os.path.join(OUT, "ckpt", f"{ckpts[i]}_{step + 1}.pt"))
        if (step + 1) % every == 0 or step + 1 == steps:
            sec = time.time() - t0
            for i, m in enumerate(ms):
                if not alive[i]:
                    continue
                hk = score(m, cx.xh, cx.zh)[0]["kl"]
                row = dict(step=step + 1, held_kl=hk, loss=losses[i], sec=round(sec))
                cand = [(hk, m, "raw")]
                if emas[i]:
                    row["held_kl_ema"] = score(emas[i].m, cx.xh, cx.zh)[0]["kl"]
                    cand.append((row["held_kl_ema"], emas[i].m, "ema"))
                curves[i].append(row)
                kb, mb, src = min(cand, key=lambda c: c[0])
                if kb < best[i][0]:
                    best[i] = (kb, {k: v.detach().cpu().clone() for k, v in mb.state_dict().items()}, step + 1, src)
                cx.log(f"  {names[i]} step {step + 1}/{steps} loss {losses[i]:.4f} held KL {hk:.4f}"
                       + (f" ema {row['held_kl_ema']:.4f}" if emas[i] else "")
                       + f" {sec:.0f}s ({(step + 1) * a.bs / sec:.0f} photos/s)")
    return list(zip(curves, best))


def sweep(cx, starts, grids, wds, clips, names):
    """lr sweep, raw weights, sweep_steps steps from each start. All (student, lr) pairs train together on the same
    batches. Returns per student {lr: held KL}, and the seconds per step of ONE student (time / number of pairs)."""
    a = cx.a
    n_sw = 20 if a.smoke else a.sweep_steps
    ms, lrs, ws, cs, nm, who = [], [], [], [], [], []
    for i, (st, grid) in enumerate(zip(starts, grids)):
        for lr in grid:
            ms.append(copy.deepcopy(st).to(DEV))
            lrs.append(lr), ws.append(wds[i]), cs.append(clips[i]), nm.append(f"{names[i]} lr {lr:g}"), who.append(i)
    ts = time.time()
    out = train(cx, ms, lrs, n_sw, 10 ** 9, ws, cs, names=nm)
    sec = (time.time() - ts) / n_sw
    res = [dict() for _ in starts]
    for (curve, _), i, lr in zip(out, who, lrs):
        k = curve[-1]["held_kl"]
        res[i][lr] = k if math.isfinite(k) else float("inf")
        cx.log(f"sweep {names[i]} lr {lr:g}: held KL {res[i][lr]:.4f}")
    return res, sec


def plan_steps(cx, sec_per_step, n_students=1):
    a = cx.a
    want = a.steps if a.steps else int(a.epochs * len(cx.X) / a.bs)
    fit = int(a.max_minutes * 60 / max(sec_per_step, 1e-9))
    steps = 60 if a.smoke else min(want, fit)
    cx.log(f"main run: {steps} steps of {a.bs} = {steps * a.bs / len(cx.X):.1f} epochs of {len(cx.X)} photos (wanted "
           f"{want} steps; {a.max_minutes} min at {sec_per_step:.3f} s/step allows {fit})")
    if a.need_full and not a.smoke and steps < want:
        cx.log(f"STOP: --need_full and the time cap allows {steps} of {want} steps")
        sys.exit(3)
    return steps, want


def finish(cx, tag, m, curve, best, lr, sw, steps, want, sec, n_w, start, extra):
    a = cx.a
    m.load_state_dict({k: v.to(DEV) for k, v in best[1].items()})
    t_test, t_rows = score(m, cx.xt, cx.zt, cx.ph.lab[cx.sets["test"]])
    h_best = score(m, cx.xh, cx.zh, cx.ph.lab[cx.sets["held"]])[0]
    res = dict(f=a.f, start=start, seed=a.seed, widths=wide18.widths(a.f), neurons=wide18.n_neurons(m), weights=n_w,
               lr=lr, sweep={str(k): v for k, v in sw.items()}, mix=a.mix, bs=a.bs, epochs=a.epochs, steps=steps,
               steps_wanted=want, sec_per_step=sec, stop_step=best[2], best_from=best[3], held_kl=best[0], held=h_best,
               test=t_test, A_test=cx.a_test, train_photos=len(cx.X), read_photos=cx.n_read, train_from_jpeg=cx.n_jpeg,
               extra_photos=cx.n_extra, ema_decay=a.ema, minutes=round(curve[-1].get("sec", 0) / 60, 1), curve=curve,
               **extra)
    cx.log(f"best {tag}: stop step {best[2]} ({best[3]}), held KL {best[0]:.4f}, test {t_test}")
    np.save(os.path.join(OUT, f"{tag}_testkl.npy"), t_rows)
    torch.save(dict(state=best[1], f=a.f, start=start, seed=a.seed, widths=wide18.widths(a.f), lr=lr),
               os.path.join(OUT, f"{tag}.pt"))
    json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)


def fit_one(cx, m, grid, wd, clip, ckpt, steps_override=None):
    """Sweep + main run for one student. Returns (m at its best, curve, best, lr, sweep, steps, want, sec)."""
    a = cx.a
    start = {k: v.detach().clone() for k, v in m.state_dict().items()}
    sw, sec = sweep(cx, [m], [grid], [wd], [clip], [ckpt])
    sw = sw[0]
    lr = min(sw, key=sw.get)
    if steps_override is not None:
        steps = 60 if a.smoke else steps_override
        want = steps_override
        cx.log(f"main run {ckpt}: {steps} steps (fixed by the two-step plan)")
    else:
        steps, want = plan_steps(cx, sec)
    cx.log(f"main run {ckpt}: lr {lr:g}")
    m.load_state_dict(start)
    save_at = tuple(int(v) for v in a.save_at.split(",") if v) + (steps,)
    (curve, best), = train(cx, [m], [lr], steps, max(1, steps // a.evals), [wd], [clip], ckpts=[ckpt],
                           save_at=save_at, ema=a.ema or None, names=[ckpt])
    return m, curve, best, lr, sw, steps, want, sec


def main_single(a):
    tag = f"W{round(a.f * 100)}{a.start}_s{a.seed}{a.suffix}" + ("_smoke" if a.smoke else "")
    cx = Ctx(a, tag)
    torch.manual_seed(a.seed)
    extra = {}
    grid = [float(v) for v in (a.lrs or LRS[a.start]).split(",")]
    if a.start == "R":
        m = wide18.make_student(a.f, a.seed).to(DEV)
    elif a.start == "P":
        cal = cx.cal()
        sc = wide18.channel_scores(cx.A, cal)
        del cal
        m, kept = wide18.cut_from_a(cx.A, a.f, sc)
        torch.save(dict(scores=sc, kept=kept), os.path.join(OUT, f"{tag}_scores.pt"))
        extra = dict(calibration_photos=len(cx.sets["train"][::14]))
    elif a.start in ("C", "W"):
        src, ck = load_student(a.source)
        extra = dict(source=a.source, source_widths=list(ck["widths"]))
        if a.start == "W":
            assert tuple(ck["widths"]) == wide18.widths(a.f), (ck["widths"], wide18.widths(a.f))
            m = src
        else:
            cal = cx.cal()
            m, sc, kept = cut_student(src, a.f, cal)
            del cal
            torch.save(dict(scores=sc, kept=kept), os.path.join(OUT, f"{tag}_scores.pt"))
            extra["calibration_photos"] = len(cx.sets["train"][::14])
    elif a.start == "G":
        f1, n1 = a.grad.split(":")
        f1, n1 = float(f1), int(n1)
        cal = cx.cal()
        sc = wide18.channel_scores(cx.A, cal)
        m1, kept1 = wide18.cut_from_a(cx.A, f1, sc)
        m1 = m1.to(memory_format=torch.channels_last)
        cx.log(f"two-step cut, step 1: A -> {f1} (widths {wide18.widths(f1)}), held KL at start "
               f"{score(m1, cx.xh, cx.zh)[0]['kl']:.4f}, {n1} steps")
        m1, c1, b1, lr1, sw1, _, _, _ = fit_one(cx, m1, grid, a.wd, a.clip, f"{tag}_a", steps_override=n1)
        m1.load_state_dict({k: v.to(DEV) for k, v in b1[1].items()})
        m, sc2, kept2 = cut_student(m1, a.f, cal)
        del cal
        torch.save(dict(scores=sc, kept=kept1, scores2=sc2, kept2=kept2), os.path.join(OUT, f"{tag}_scores.pt"))
        extra = dict(grad=a.grad, step1=dict(widths=wide18.widths(f1), lr=lr1, sweep={str(k): v for k, v in sw1.items()},
                                             stop_step=b1[2], best_from=b1[3], held_kl=b1[0], curve=c1))
        cx.log(f"two-step cut, step 2: best {f1} student (held KL {b1[0]:.4f}) cut to {a.f}")
    m = m.to(memory_format=torch.channels_last)
    n_w = sum(p.numel() for p in m.parameters())
    cx.log(f"student f={a.f} start {a.start}: widths {wide18.widths(a.f)}, neurons {wide18.n_neurons(m)}, weights "
           f"{n_w}; held KL at start {score(m, cx.xh, cx.zh)[0]['kl']:.4f}")
    over = None
    if a.start == "G":
        total = a.steps if a.steps else int(a.epochs * len(cx.X) / a.bs)
        over = total - n1
    m, curve, best, lr, sw, steps, want, sec = fit_one(cx, m, grid, a.wd, a.clip, tag, steps_override=over)
    finish(cx, tag, m, curve, best, lr, sw, steps, want, sec, n_w, a.start, dict(extra, wd=a.wd, clip=a.clip))


def main_multi(a):
    names = a.multi.split(",")
    pct = round(a.f * 100)
    tags = [f"W{pct}{n}_s{a.seed}{a.suffix}" + ("_smoke" if a.smoke else "") for n in names]
    cx = Ctx(a, f"W{pct}multi_s{a.seed}{a.suffix}" + ("_smoke" if a.smoke else ""))
    cx.log(f"multi: {names} on the same batches, mix {a.mix}")
    cal = None
    starts, grids, wds, clips, extras = [], [], [], [], []
    for n, tag in zip(names, tags):
        v = VARIANTS[n]
        e = {}
        if v["start"] == "R":
            m = wide18.make_student(a.f, a.seed).to(DEV)
        elif v["start"] == "P":
            cal = cx.cal() if cal is None else cal
            sc = wide18.channel_scores(cx.A, cal)
            m, kept = wide18.cut_from_a(cx.A, a.f, sc)
            torch.save(dict(scores=sc, kept=kept), os.path.join(OUT, f"{tag}_scores.pt"))
        else:
            cal = cx.cal() if cal is None else cal
            src, ck = load_student(v["source"])
            m, sc, kept = cut_student(src, a.f, cal)
            torch.save(dict(scores=sc, kept=kept), os.path.join(OUT, f"{tag}_scores.pt"))
            e = dict(source=v["source"])
        m = m.to(memory_format=torch.channels_last)
        starts.append(m)
        grids.append([float(x) for x in (a.lrs or v["lrs"]).split(",")])
        wds.append(v.get("wd", a.wd))
        clips.append(v.get("clip", a.clip))
        extras.append(dict(e, variant=n, wd=wds[-1], clip=clips[-1]))
        cx.log(f"{tag}: start {v['start']}, wd {wds[-1]}, clip {clips[-1]}, lrs {grids[-1]}, held KL at start "
               f"{score(m, cx.xh, cx.zh)[0]['kl']:.4f}")
    del cal
    init = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in starts]
    sw, sec = sweep(cx, starts, grids, wds, clips, names)
    lrs = [min(s, key=s.get) for s in sw]
    n_pairs = sum(len(g) for g in grids)
    steps, want = plan_steps(cx, sec * len(names) / n_pairs)
    cx.log(f"main run lrs: {dict(zip(names, lrs))}")
    for m, st in zip(starts, init):
        m.load_state_dict(st)
    save_at = tuple(int(v) for v in a.save_at.split(",") if v) + (steps,)
    out = train(cx, starts, lrs, steps, max(1, steps // a.evals), wds, clips, ckpts=tags, save_at=save_at,
                ema=a.ema or None, names=names)
    for (curve, best), m, tag, lr, s, e, v in zip(out, starts, tags, lrs, sw, extras, names):
        finish(cx, tag, m, curve, best, lr, s, steps, want, sec, sum(p.numel() for p in m.parameters()),
               VARIANTS[v]["start"], e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float)
    ap.add_argument("--start", choices=("R", "P", "C", "W", "G"))
    ap.add_argument("--source", default="", help="C / W: a saved student tag in distill_whole/ or a path")
    ap.add_argument("--grad", default="0.75:12000", help="G: first width and its steps")
    ap.add_argument("--multi", default="", help="comma list of VARIANTS names, trained together")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg_dir", default="/bigtemp/fkm5yp/n6")
    ap.add_argument("--extra", default="", help="folder of extra_*.npz (ImageNet-train photos) added to the train set")
    ap.add_argument("--lrs", default="")
    ap.add_argument("--sweep_steps", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=300)
    ap.add_argument("--steps", type=int, default=0, help="main-run steps (overrides --epochs)")
    ap.add_argument("--max_minutes", type=float, default=90, help="cap on the main run, from the speed seen in the sweep")
    ap.add_argument("--need_full", action="store_true", help="stop if the cap cuts the run below the steps wanted")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.9998, help="EMA decay of the scored copy; 0 = off")
    ap.add_argument("--evals", type=int, default=30)
    ap.add_argument("--save_at", default="1,3,10,30,100,300,1000,3000,10000")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    torch.backends.cudnn.benchmark = True
    if a.multi:
        return main_multi(a)
    return main_single(a)


if __name__ == "__main__":
    main()
