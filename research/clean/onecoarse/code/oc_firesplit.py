"""Read, one firing at a time (LOG "Read, one firing at a time", 2026-11-23): exact input split of n101's firings.
Runs on a cluster CPU node in n6/clean/neuron/coarse (needs r18.py, capture18.py, the held + test caches, the photos).

Pass 1 (caches, all 19,000 held + test photos): n101's input before its ReLU on the 7 x 7 grid -> peak, its cell, minimum,
its cell; A's 512 stage-4 block-0 neurons -> each neuron's own max over all photos and cells.
Poles: strong = peak >= half of n101's own max; negative = the same number of photos with the lowest minimum.
Pass 2 (only those photos, from the raw photo): stem, stages 1-2 -> h2; stage 3 by hand. With the ReLU masks after each
add held fixed and batch norm written as scale + shift, n101's input at its cell is an exact sum over
  inner1 (stage-3 block-1 hidden neurons, 256), inner0 (stage-3 block-0 hidden neurons, 256, through the skip),
  h2 (stage-2 stream channels, 128, through the downsample), and 4 shifts (b1 bn2, b0 bn2, downsample bn, n101 bn1).
Contribution of a source = gradient x value (exact: the function is linear in the sources once the masks are fixed).
Saved per pole (results/coarse/r18/oc_firesplit_<pole>.npz):
  gix photo index in the 19,000 (held then test); cell (row*7+col); pre (n101, recomputed); pre_cache (from the cache)
  split (256: n101's per-channel vote at the cell), patch (stage-3 h, 256 x 7 x 7 around the cell, zero padded)
  src (644 = 256 inner1 + 256 inner0 + 128 h2 + 4 shifts), top (20 largest |votes|), src_top (20 x 644)
  A_pre9 (512 x 9: A's input on the 3 x 3 neighbourhood of the cell, row-major), err_sum, err_h3
Saved once (oc_firesplit_w.npz): n101 w (256 x 3 x 3), scale, shift; A w (512 x 256 x 3 x 3), scale, shift; A_max (512).
Usage: python -u oc_firesplit.py [n_photos_per_pole]   (a number = smoke run on that many photos per pole)
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

J = 101
NSRC = 256 + 256 + 128 + 4


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def aff(bn, x, on=True):
    sc, sh = bn_parts(bn)
    y = x * sc.view(1, -1, 1, 1)
    return y + sh.view(1, -1, 1, 1) if on else y


def pass1(B, A):
    bB, bA = B.layer4[0], A.layer4[0]
    pk, arg, mn, amn, gix, amax = [], [], [], [], [], torch.full((512,), -1e9)
    amin = torch.full((512,), 1e9)
    loc = []
    with torch.no_grad():
        g = 0
        for s in ("held", "test"):
            cache = r18.Cache(s)
            for a in range(0, len(cache), 250):
                rows = np.arange(a, min(a + 250, len(cache)))
                x, _ = cache.get(rows)
                pre = bB.bn1(bB.conv1(x))[:, J].flatten(1)
                p, ar = pre.max(1)
                m, am = pre.min(1)
                pk.append(p.cpu()); arg.append(ar.cpu()); mn.append(m.cpu()); amn.append(am.cpu())
                preA = bA.bn1(bA.conv1(x))
                amax = torch.maximum(amax, preA.amax((0, 2, 3)).cpu())
                amin = torch.minimum(amin, preA.amin((0, 2, 3)).cpu())
                loc += [(s, int(r)) for r in rows]
                gix.append(np.arange(g, g + len(rows))); g += len(rows)
    return (torch.cat(pk).numpy(), torch.cat(arg).numpy(), torch.cat(mn).numpy(), torch.cat(amn).numpy(),
            amax.numpy(), amin.numpy(), loc)


@torch.enable_grad()   # capture18 turns gradients off at import
def stage3_split(A, B, xin, cells):
    """xin: normalised photos (n x 3 x 224 x 224). cells: n101's cell per photo (n,). Returns a dict of tensors."""
    l3, bB = A.layer3, B.layer4[0]
    with torch.no_grad():
        h = A.layer2(A.layer1(A.maxpool(torch.relu(A.bn1(A.conv1(xin))))))           # h2: n x 128 x 28 x 28
        b0, b1 = l3[0], l3[1]
        in0 = torch.relu(b0.bn1(b0.conv1(h)))
        ds = b0.downsample[1](b0.downsample[0](h))
        z0 = b0.bn2(b0.conv2(in0)) + ds
        m0 = (z0 > 0).float()
        h0 = torch.relu(z0)
        in1 = torch.relu(b1.bn1(b1.conv1(h0)))
        z1 = b1.bn2(b1.conv2(in1)) + h0
        m1 = (z1 > 0).float()
        h3 = torch.relu(z1)
    sc1, sh1 = bn_parts(bB.bn1)
    w = bB.conv1.weight[J]                                               # 256 x 3 x 3
    n = len(xin)
    r, c = torch.as_tensor(cells // 7), torch.as_tensor(cells % 7)
    ar = torch.arange(n)

    def pre_of(i1, i0, hh, on):
        """on = which shifts are on: (b1 bn2, b0 bn2, downsample bn, n101 bn1). Masks fixed."""
        d = aff(b0.downsample[1], b0.downsample[0](hh), on[2])
        zz0 = aff(b0.bn2, b0.conv2(i0), on[1]) + d
        hh3 = m1 * (aff(b1.bn2, b1.conv2(i1), on[0]) + m0 * zz0)
        votes = F.conv2d(hh3, w.view(256, 1, 3, 3), stride=2, padding=1, groups=256) * sc1[J]   # n x 256 x 7 x 7
        v = votes[ar, :, r, c]                                                                  # n x 256
        return v, v.sum(1) + (sh1[J] if on[3] else 0.0), hh3

    i1 = in1.clone().requires_grad_(True)
    i0 = in0.clone().requires_grad_(True)
    hh = h.clone().requires_grad_(True)
    v, pre, hh3 = pre_of(i1, i0, hh, (True, True, True, True))
    with torch.no_grad():
        consts = [pre_of(torch.zeros_like(in1), torch.zeros_like(in0), torch.zeros_like(h),
                         tuple(k == q for q in range(4)))[1] for k in range(4)]
    top = v.detach().abs().topk(20, dim=1).indices                                               # n x 20

    def contrib(scalar_per_photo):
        g1, g0, gh = torch.autograd.grad(scalar_per_photo.sum(), (i1, i0, hh), retain_graph=True)
        return torch.cat([(g1 * in1).sum((2, 3)), (g0 * in0).sum((2, 3)), (gh * h).sum((2, 3))], 1)   # n x 640

    lin = contrib(pre)
    src = torch.cat([lin, torch.stack(consts, 1)], 1)                                            # n x 644
    src_top = []
    for k in range(20):
        vk = v[ar, top[:, k]]
        lk = contrib(vk)
        # a channel's vote holds no n101 bn1 shift; its other shifts come from the masked constant path
        with torch.no_grad():
            ck = []
            for q in range(3):
                vq = pre_of(torch.zeros_like(in1), torch.zeros_like(in0), torch.zeros_like(h),
                            tuple(t == q for t in range(4)))[0]
                ck.append(vq[ar, top[:, k]])
        src_top.append(torch.cat([lk, torch.stack(ck + [torch.zeros(n)], 1)], 1))
    src_top = torch.stack(src_top, 1)                                                            # n x 20 x 644
    with torch.no_grad():
        err_sum = (src.sum(1) - pre).abs() / pre.abs().clamp(min=1e-6)
        err_top = (src_top.sum(2) - v[ar.view(-1, 1), top]).abs().amax(1) / pre.abs().clamp(min=1e-6)
        hp = F.pad(h3, (3, 3, 3, 3))
        patch = torch.stack([hp[i, :, 2 * int(r[i]):2 * int(r[i]) + 7, 2 * int(c[i]):2 * int(c[i]) + 7] for i in range(n)])
        bA = A.layer4[0]
        preA = bA.bn1(bA.conv1(h3))                                                              # n x 512 x 7 x 7
        pp = F.pad(preA, (1, 1, 1, 1), value=float("-inf"))
        A_pre9 = torch.stack([pp[i, :, int(r[i]):int(r[i]) + 3, int(c[i]):int(c[i]) + 3].reshape(512, 9) for i in range(n)])
    return dict(pre=pre.detach(), split=v.detach(), patch=patch, src=src.detach(), top=top, src_top=src_top.detach(),
                A_pre9=A_pre9, err_sum=err_sum, err_top=err_top, h3=h3)


def main(nsmoke):
    torch.manual_seed(0)
    A = net_for("A").cpu().eval()
    B = net_for("B128aug_s1").cpu().eval()
    for p in list(A.parameters()) + list(B.parameters()):
        p.requires_grad_(False)
    pk, arg, mn, amn, amax, amin, loc = pass1(B, A)
    strong = np.where(pk >= .5 * pk.max())[0]
    neg = np.argsort(mn)[:len(strong)]
    print(f"n101 max {pk.max():.3f}; strong photos {len(strong)}; negative photos {len(neg)} (min {mn[neg].max():.3f} or lower)",
          flush=True)
    photos = r18.net.Photos("read")
    caches = {s: r18.Cache(s) for s in ("held", "test")}
    bB, bA = B.layer4[0], A.layer4[0]
    scB, shB = bn_parts(bB.bn1)
    scA, shA = bn_parts(bA.bn1)
    np.savez(os.path.join(r18.OUT, "oc_firesplit_w.npz"), w=bB.conv1.weight[J].numpy(), sc=float(scB[J]), sh=float(shB[J]),
             A_w=bA.conv1.weight.numpy(), A_sc=scA.numpy(), A_sh=shA.numpy(), A_max=amax, A_min=amin,
             n101_max=float(pk.max()), n101_min=float(mn.min()))
    for pole, sel, cellv in (("pos", strong, arg), ("neg", neg, amn)):
        sel = sel[:nsmoke] if nsmoke else sel
        out = {}
        for a in range(0, len(sel), 32):
            gi = sel[a:a + 32]
            set_rows = [loc[i] for i in gi]
            pix = np.array([caches[s].ix[r] for s, r in set_rows])
            xin = photos.tensor(pix)
            res = stage3_split(A, B, xin, cellv[gi])
            xc = torch.stack([caches[s].get(np.array([r]))[0][0].cpu() for s, r in set_rows])
            res["err_h3"] = (res.pop("h3") - xc).abs().amax((1, 2, 3)) / xc.abs().amax((1, 2, 3)).clamp(min=1e-6)
            with torch.no_grad():
                pc = bB.bn1(bB.conv1(xc))[:, J].flatten(1)
            res["pre_cache"] = pc[torch.arange(len(gi)), torch.as_tensor(cellv[gi])]
            for k, v in res.items():
                out.setdefault(k, []).append(v.cpu())
            print(f"  {pole}: {a + len(gi)}/{len(sel)}  err_sum max {float(res['err_sum'].max()):.2e}  "
                  f"err_top max {float(res['err_top'].max()):.2e}  err_h3 max {float(res['err_h3'].max()):.2e}", flush=True)
        o = {k: torch.cat(v).numpy() for k, v in out.items()}
        o["patch"] = o["patch"].astype(np.float32)
        f = os.path.join(r18.OUT, f"oc_firesplit_{pole}{'_smoke' if nsmoke else ''}.npz")
        np.savez(f, gix=sel, cell=cellv[sel], **o)
        print("saved", f, {k: v.shape for k, v in o.items()}, flush=True)
        dpre = np.abs(o["pre"] - o["pre_cache"]) / np.maximum(np.abs(o["pre_cache"]), 1e-6)
        print(f"  {pole}: exact sum err max {o['err_sum'].max():.2e}, top-20 err max {o['err_top'].max():.2e}, "
              f"h3 vs cache err max {o['err_h3'].max():.2e}, pre vs cache rel diff max {dpre.max():.3e}", flush=True)
        bad = [m for m, ok in (("exact sum", o["err_sum"].max() <= 1e-3), ("top-20 sum", o["err_top"].max() <= 1e-3),
                               ("h3 vs cache", o["err_h3"].max() <= 1e-2), ("pre vs cache", dpre.max() <= 1e-2)) if not ok]
        if bad:
            print(f"STOP: {', '.join(bad)} failed on pole {pole}. The photo run does not reproduce the cached firing, "
                  "or the split does not add up. Nothing else is read.", flush=True)
            sys.exit(3)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
