"""P0 Input (FANOUT round 1, BUILD): the self-made generic input and its shuffled-weights null.

WHAT IT MAKES
  cache/{net}_self_n256_k32_s1.pt : the network's own input, made from the SHARED pink file
  cache/{net}_null_n256_k32_s1.pt : the same recipe, but a copy of the network with shuffled weights makes the images
  float16 [256, 3, 224, 224], per image per channel mean 0, std 1 (the normalization of generic.pink). A sidecar
  {same name}.json holds the recipe, the loss at step 1 and step K, and how far the images moved.

RECIPE (talk.json turn2, P0 revised proposal; every knob is fixed here and is NOT tuned on check_bn)
  Start: x0 = cache/input_pink_n256_s1.pt (the floor's own 256 pink images). Batch 32, weights frozen, MPS.
  Image parametrization (Fourier, 1/f amplitude A as in generic.pink, A[0,0] = 0):
      g(z) = chan_norm(irfft2(A * z, norm="ortho"))          chan_norm = mean 0, std 1 (unbiased) per image per channel
      x(z) = x0 + (g(z) - g(z0))                               so x(z0) = x0 EXACTLY (K = 0 gives the shared file)
      z0  = rfft2(x0, norm="ortho") / A, DC set 0, then scaled to RMS 1 per image per channel ("unit seed" scale)
    Why the unit scale: Adam moves each coordinate by about lr per step whatever the gradient size, so the scale of z
    sets how far the image can move. chan_norm makes g scale-free, so the scale of z0 is a free knob. It is fixed here
    to the scale of a unit white seed in an orthonormal Fourier basis (the seed of generic.pink). It is never retuned.
    The sidecar records the relative change |x_K - x0| / |x0| so the reader can see how far the images moved.
  Each step k = 1..K, with a random transform t (one draw per batch: roll up to 16 px both axes, scale .9-1.1 by
  bilinear resize then random crop or zero pad to 224; flip per image with p .5):
    ResNet: L = mean_i CE(A(t(x_i)), y_i), y_i ~ softmax(fc.bias), drawn once (generator seed 1) for all 256 images.
            Since that prior is near uniform, this is "make the network sure of a randomly drawn class".
    CLIP:   self-InfoNCE proxy, NOT CLIP's training loss. e_i = normalize(forward(T, t(x_i))), e'_j = the previous step's
            embeddings, detached (step 1: a second view under its own transform). L = mean_i -log softmax_j(100 e_i.e'_j)_i.
            Negatives j = the other images of the same batch of 32. Scale 100 = the stored logit scale of the teacher.
            Derived from: a contrastive tower trained image-to-text, with text banned, so the nearest self-supervised
            form of its objective is image-to-image at its stored scale (SimCLR-type).
    z <- Adam(z, lr .05). K = 32. The model runs in eval mode; no loss term reads BatchNorm running stats.
  Null: the same seeds, targets, transforms, K and lr, but every weight tensor of a linear map is shuffled inside itself
    (CLIP: patch_w, q/k/v/o, fc, pr, proj; ResNet: every conv and fc weight). Norms, biases, cls and pos are kept.
    The board and the checks then run on the REAL network with these images.
  Language model carry-over (not built here): 128 own samples of 512 tokens from BOS at temperature 1, no gradient.

PHOTO USE: none. The input comes from the weights and the pink seed only. This file imports no photo loader.

USAGE (from research/clean; ONE torch job on the Mac at a time, only the runner runs it):
  curves/.venv/bin/python -I distill/fanout/make_input.py --k0-check                 # no network loaded
  curves/.venv/bin/python -I distill/fanout/make_input.py --net r18 --kind null --smoke   # 1 batch, 2 steps, no file
  curves/.venv/bin/python -I distill/fanout/make_input.py --net r18 --kind self
  curves/.venv/bin/python -I distill/fanout/make_input.py --net clip --kind null
  --render-dev mps runs the Fourier render and transforms on MPS (default cpu: only the network runs on MPS).
"""
import argparse
import copy
import datetime
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board  # noqa: E402
import vitgen  # noqa: E402

CACHE = os.path.join(HERE, "cache")
PINK = os.path.join(CACHE, "input_pink_n256_s1.pt")
N, K, LR, BS, SEED = 256, 32, 0.05, 32, 1
SHIFT, SCALE_LO, SCALE_HI, FLIP_P = 16, 0.9, 1.1, 0.5
CLIP_SCALE = 100.0        # the teacher's stored logit scale (clipkd/out/teacher_*.npz logit_scale; check_input asserts it)
NULL_SEED = 2             # shuffle permutation seed of the null generator
SIZE = 224
DEV = board.DEV
VIT_LIN = ("q_w", "k_w", "v_w", "o_w", "fc_w", "pr_w")


def out_path(net, kind, k=K, n=N, seed=SEED):
    return os.path.join(CACHE, f"{net}_{kind}_n{n}_k{k}_s{seed}.pt")


# ---------------------------------------------------------------- image parametrization
def amp(size=SIZE):
    """1/f amplitude on the rfft2 grid, exactly as generic.pink (alpha 1). A[0, 0] = 0."""
    fy = torch.fft.fftfreq(size).view(-1, 1)
    fx = torch.fft.rfftfreq(size).view(1, -1)
    f = torch.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1.0
    a = 1.0 / f
    a[0, 0] = 0.0
    return a


def chan_norm(x):
    return (x - x.mean((2, 3), keepdim=True)) / x.std((2, 3), keepdim=True)   # unbiased std, as generic.pink


def render(z, a):
    """z: real view (B, 3, 224, 113, 2) -> chan_norm image (B, 3, 224, 224)."""
    return chan_norm(torch.fft.irfft2(torch.view_as_complex(z) * a, s=(SIZE, SIZE), norm="ortho"))


def z_start(x0, a):
    """Unit-seed-scale Fourier coordinates of x0 (DC dropped; chan_norm removes it anyway)."""
    Z = torch.fft.rfft2(x0.float(), norm="ortho")
    a_safe = a.clone()
    a_safe[0, 0] = 1.0
    Z = Z / a_safe
    Z[..., 0, 0] = 0
    c = Z.abs().pow(2).mean((2, 3), keepdim=True).sqrt()
    return torch.view_as_real(Z / c).clone()   # a fresh real leaf, not a view of the complex tensor


def transform(x, gen):
    """Random roll, flip (per image), zoom. Every branch draws the same random numbers, so the transform sequence
    depends only on the generator (self and null see the same transforms)."""
    B = x.shape[0]
    dy, dx = (int(v) for v in torch.randint(-SHIFT, SHIFT + 1, (2,), generator=gen))
    flip = torch.rand(B, generator=gen) < FLIP_P
    s = SCALE_LO + (SCALE_HI - SCALE_LO) * float(torch.rand(1, generator=gen))
    u = torch.rand(2, generator=gen)
    x = torch.roll(x, (dy, dx), (2, 3))
    x = torch.where(flip.to(x.device).view(B, 1, 1, 1), x.flip(3), x)
    m = int(round(SIZE * s))
    if m != SIZE:
        x = F.interpolate(x, size=(m, m), mode="bilinear", align_corners=False)
    if m >= SIZE:
        oy, ox = (int(v * (m - SIZE + 1)) for v in u)
        oy, ox = min(oy, m - SIZE), min(ox, m - SIZE)
        x = x[:, :, oy:oy + SIZE, ox:ox + SIZE]
    else:
        p = SIZE - m
        oy, ox = (min(int(v * (p + 1)), p) for v in u)
        x = F.pad(x, (ox, p - ox, oy, p - oy))
    return x


# ---------------------------------------------------------------- networks
def load_net(net, kind):
    if net == "r18":
        A = board.r18_A().eval()
        y = torch.multinomial(F.softmax(A.fc.bias.detach().float(), 0), N, replacement=True,
                              generator=torch.Generator().manual_seed(SEED))
        if kind == "null":
            A = shuffle_r18(A, NULL_SEED)
        for p in A.parameters():
            p.requires_grad_(False)
        return A.to(DEV).eval(), y
    T = board.clip_teacher()
    if kind == "null":
        T = shuffle_vit(T, NULL_SEED)
    return vitgen.to(T, DEV), None


def _perm(w, g):
    return w.flatten()[torch.randperm(w.numel(), generator=g)].view_as(w).clone()


def shuffle_r18(A, seed):
    S = copy.deepcopy(A)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in S.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                m.weight.copy_(_perm(m.weight.detach().cpu(), g))
    return S


def shuffle_vit(T, seed):
    g = torch.Generator().manual_seed(seed)
    S = dict(T)
    S["patch_w"] = _perm(T["patch_w"], g)
    S["proj"] = _perm(T["proj"], g)
    S["blocks"] = [dict(B, **{k: _perm(B[k], g) for k in VIT_LIN}) for B in T["blocks"]]
    return S


def shuffle_check(net, kind):
    """Implementation check (passes by algebra): every shuffled tensor holds the same values as the real one."""
    if kind != "null":
        return None
    if net == "r18":
        A = board.r18_A().eval()
        S = shuffle_r18(A, NULL_SEED)
        pairs = [(m1.weight, m2.weight) for m1, m2 in zip(A.modules(), S.modules())
                 if isinstance(m1, (torch.nn.Conv2d, torch.nn.Linear))]
        kept = [(m1.weight, m2.weight) for m1, m2 in zip(A.modules(), S.modules()) if isinstance(m1, torch.nn.BatchNorm2d)]
    else:
        T = board.clip_teacher()
        S = shuffle_vit(T, NULL_SEED)
        pairs = [(T["patch_w"], S["patch_w"]), (T["proj"], S["proj"])]
        pairs += [(B1[k], B2[k]) for B1, B2 in zip(T["blocks"], S["blocks"]) for k in VIT_LIN]
        kept = [(B1["ln1"]["g"], B2["ln1"]["g"]) for B1, B2 in zip(T["blocks"], S["blocks"])]
    same_vals = all(torch.equal(torch.sort(a.detach().flatten()).values, torch.sort(b.detach().flatten()).values) for a, b in pairs)
    moved = sum(int((a.detach() != b.detach()).sum()) for a, b in pairs) / sum(a.numel() for a, _ in pairs)
    kept_ok = all(torch.equal(a.detach(), b.detach()) for a, b in kept)
    res = dict(shuffled_tensors=len(pairs), same_values=same_vals, frac_entries_moved=round(moved, 4), norms_kept=kept_ok)
    print("[implementation check, passes by algebra] shuffle:", json.dumps(res))
    return res


# ---------------------------------------------------------------- the recipe
def make(net, model, y, x_all, k_steps=K, n_batches=None, render_dev="cpu"):
    """Returns (x float32 CPU (n, 3, 224, 224), per-batch log). model may be None when k_steps == 0."""
    a = amp().to(render_dev)
    gen = torch.Generator().manual_seed(SEED)
    outs, log = [], []
    for bi, s in enumerate(range(0, len(x_all), BS)):
        if n_batches is not None and bi >= n_batches:
            break
        x0 = x_all[s:s + BS].float().to(render_dev)
        z = z_start(x0, a).requires_grad_(True)
        with torch.no_grad():
            g0 = render(z, a)
        opt = torch.optim.Adam([z], lr=LR)
        e_prev = None
        if net == "clip" and k_steps > 0:
            with torch.no_grad():
                e_prev = F.normalize(vitgen.forward(model, transform(x0, gen).to(DEV)).float(), dim=-1)
        yb = y[s:s + BS].to(DEV) if y is not None else None
        losses, t0 = [], time.time()
        for _ in range(k_steps):
            x = x0 + (render(z, a) - g0)
            xt = transform(x, gen).to(DEV)
            if net == "r18":
                L = F.cross_entropy(model(xt).float(), yb)
            else:
                e = F.normalize(vitgen.forward(model, xt).float(), dim=-1)
                L = F.cross_entropy(CLIP_SCALE * e @ e_prev.T, torch.arange(len(e), device=DEV))
            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            if net == "clip":
                e_prev = e.detach()
            losses.append(float(L))
        with torch.no_grad():
            xk = (x0 + (render(z, a) - g0)).cpu()
            row = dict(batch=bi, seconds=round(time.time() - t0, 2),
                       sec_per_step=round((time.time() - t0) / max(k_steps, 1), 3),
                       loss_step1=losses[0] if losses else None, loss_stepK=losses[-1] if losses else None,
                       rel_change=float((xk - x0.cpu()).norm() / x0.cpu().norm()),
                       chan_std_dev=float((xk.std((2, 3)) - 1).abs().max()),
                       chan_mean_dev=float(xk.mean((2, 3)).abs().max()))
            if net == "r18" and k_steps > 0:
                row["frac_argmax_is_target"] = float((model(xk.to(DEV)).argmax(1) == yb).float().mean())
        log.append(row)
        print(json.dumps(row), flush=True)
        outs.append(xk)
    return torch.cat(outs), log


def k0_check():
    """Implementation check: with K = 0 the recipe returns the shared pink file bit for bit (passes by algebra:
    x = x0 + (g(z0) - g(z0))). The render round trip |g(z0) - x0| is printed for information."""
    x_all = torch.load(PINK, weights_only=True)
    res = {}
    for net in ("r18", "clip"):
        xk, _ = make(net, None, None, x_all, k_steps=0)
        res[net] = dict(equal_to_shared_file=bool(torch.equal(xk.half(), x_all)), shape=list(xk.shape))
    a = amp()
    x0 = x_all[:BS].float()
    g0 = render(z_start(x0, a), a)
    res["render_round_trip_max_abs"] = float((g0 - x0).abs().max())
    res["shared_file"] = dict(dtype=str(x_all.dtype), chan_mean_max=float(x_all.float().mean((2, 3)).abs().max()),
                              chan_std_dev_max=float((x_all.float().std((2, 3)) - 1).abs().max()))
    print("[implementation check, passes by algebra] K = 0:", json.dumps(res))
    if not all(res[n]["equal_to_shared_file"] for n in ("r18", "clip")):
        raise SystemExit("K = 0 does NOT reproduce the shared pink file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("r18", "clip"))
    ap.add_argument("--kind", choices=("self", "null"), default="self")
    ap.add_argument("--k0-check", action="store_true", help="implementation check, loads no network")
    ap.add_argument("--smoke", action="store_true", help="one batch, 2 steps, writes nothing, prints the time estimate")
    ap.add_argument("--render-dev", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing cache file")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.k0_check:
        k0_check()
        return
    if a.net is None:
        raise SystemExit("--net is required")
    path = out_path(a.net, a.kind)
    if not a.smoke and os.path.exists(path) and not a.force:
        raise SystemExit(f"{path} exists (use --force to overwrite)")
    x_all = torch.load(PINK, weights_only=True)
    assert tuple(x_all.shape) == (N, 3, SIZE, SIZE), x_all.shape
    sc = shuffle_check(a.net, a.kind) if a.smoke else None
    model, y = load_net(a.net, a.kind)
    t0 = time.time()
    if a.smoke:
        _, log = make(a.net, model, y, x_all, k_steps=2, n_batches=1, render_dev=a.render_dev)
        est = log[0]["sec_per_step"] * K * (N // BS) / 60
        print(f"SMOKE {a.net} {a.kind}: {log[0]['sec_per_step']:.3f} s per step (batch {BS}); "
              f"full run estimate {est:.1f} min; loss {log[0]['loss_step1']:.4f} -> {log[0]['loss_stepK']:.4f}; nothing written")
        return
    X, log = make(a.net, model, y, x_all, k_steps=K, render_dev=a.render_dev)
    X16 = X.half().contiguous()
    if not torch.isfinite(X16).all():
        raise SystemExit(f"non-finite values in the made input; {path} NOT written")
    torch.save(X16, path)
    meta = dict(net=a.net, kind=a.kind, file=os.path.basename(path), n=N, K=K, lr=LR, batch=BS, seed=SEED,
                null_shuffle_seed=NULL_SEED if a.kind == "null" else None, start=os.path.basename(PINK),
                z_scale="unit RMS per image per channel (orthonormal Fourier, 1/f amplitude divided out)",
                transforms=dict(roll_px=SHIFT, scale=[SCALE_LO, SCALE_HI], flip_p=FLIP_P, crop_or_zero_pad=SIZE),
                loss=("CE to y ~ softmax(fc.bias), seed 1" if a.net == "r18" else
                      "self-InfoNCE proxy vs previous step's detached embeddings, scale 100, batch negatives"),
                photo_use="none", minutes=round((time.time() - t0) / 60, 2),
                loss_step1_mean=sum(r["loss_step1"] for r in log) / len(log),
                loss_stepK_mean=sum(r["loss_stepK"] for r in log) / len(log),
                rel_change_mean=sum(r["rel_change"] for r in log) / len(log),
                saved_chan_std_dev_max=float((X16.float().std((2, 3)) - 1).abs().max()),
                when=datetime.datetime.now().isoformat(timespec="seconds"), batches=log)
    if a.net == "r18":
        meta["frac_argmax_is_target_mean"] = sum(r["frac_argmax_is_target"] for r in log) / len(log)
    json.dump(meta, open(path[:-3] + ".json", "w"), indent=1)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.0f} MB) in {meta['minutes']} min; loss {meta['loss_step1_mean']:.4f} -> "
          f"{meta['loss_stepK_mean']:.4f}; mean relative change of x {meta['rel_change_mean']:.3f}")


if __name__ == "__main__":
    main()
