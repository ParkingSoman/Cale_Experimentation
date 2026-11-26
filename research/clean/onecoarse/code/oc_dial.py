# scale: neurons=s3:69, A:children; net=A; photos=held, test; where=cluster
# rung: Q=the dial: are the children right? per-child calculated morph along the parent's value; N=30 (top 30 R1 children, each alone); EST=3 min GPU; PREV=results/oc_morph.npz (11 children); PERMIT=Shonu approved plan "Reconcile the random-photo gap, and check the children with the dial" 2026-11-24 incl. its SCALE-UP REQUEST
"""The dial: are the children right? (plan 2026-11-24, LOG section "The dial: are the children right?").

Calculated build only (Shonu 2026-11-24: "you can just use the calculation one"). Nothing is fitted.
For each of the top 30 R1 children of s3:69 in A (score = channel-69 read x write norm, as oc_morph.py):
  delete_i: child i's value set to 0
  morph_i:  delete_i + the parent's direct write through child i's path, v_i applied to (k_i applied to p)
Per photo: KL(A || build) for both. Recovery_i by parent-peak bin is read on the Mac (oc_dial_read.py).
Also per photo and per child, at the child's own peak cell (where its input before the ReLU is largest):
  the input value, channel 69's part of it (k_i on p), and all 256 per-channel parts (fp16), plus the bias.
Set mode: the R1 top 11 together (same build as oc_morph.py "morph"), on DEV and on the oc_morph test random rows
(sanity: must reproduce oc_morph.npz delete and morph_cal rows).
DEV = all 2,000 held photos. Test photos: only the sanity set rows here.
Step 2 (argument step2; LOG "Step 2: the weak range itself"): on the set of 11, DEV photos, builds delete, morph,
delete_weak, delete_strong (strong = at least half the child's own max over DEV), delete_otherroute, delete_parentroute
(parent-driven cell = the parent's push k_i on p is at least half the child's input), child_relu (diagnostic:
ReLU(k_i on p + b_i)). Saves per photo KL and top-1 of each build.
Usage: python -u oc_dial.py [smoke] [step2]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
P, N_POOL, K_SET = 69, 30, 11
SMOKE = "smoke" in sys.argv
STEP2 = "step2" in sys.argv
STEP3 = "step3" in sys.argv
DEV = r18.DEV


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, t1 = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)                                   # 512 x 256 x 3 x 3
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)                                   # 512 x 512 x 3 x 3
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).cpu().numpy()
    order = np.argsort(-score)
    pool = order[:N_POOL].copy()
    print("pool:", pool.tolist(), flush=True)

    def run(x, kids, mode):
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        if mode != "base":
            inner = inner.clone(); inner[:, kids] = 0
        r = b0.bn2(b0.conv2(inner))
        if mode == "morph":
            p = x[:, P:P + 1]
            r = r + F.conv2d(F.conv2d(p, R[kids][:, P:P + 1], stride=2, padding=1), V[:, kids], padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    held, test = r18.Cache("held"), r18.Cache("test")
    dev_rows = np.arange(len(held))
    te_rows = np.sort(np.random.default_rng(2).choice(len(test), 2000, replace=False))   # = oc_morph test_random
    if SMOKE:
        dev_rows, te_rows = dev_rows[:40], te_rows[:40]
    print(f"held {len(held)}, test {len(test)}; dev {DEV}; {time.time()-t0:.0f} s", flush=True)

    def over(cache, rows, fn, bs=250):
        out = []
        for s in range(0, len(rows), bs):
            x, z = cache.get(rows[s:s + bs])
            out.append(fn(x, z))
        return out

    # per photo: parent peak, base margin and top-1, base KL (sanity)
    def base_info(x, z):
        y = run(x, None, "base")
        top2 = torch.topk(z, 2, 1).values
        return np.stack([x[:, P].amax((1, 2)).cpu().numpy(), (top2[:, 0] - top2[:, 1]).cpu().numpy(),
                         z.argmax(1).float().cpu().numpy(), r18.kl_rows(z, y).cpu().numpy()], 1)
    info = np.concatenate(over(held, dev_rows, base_info))
    assert info[:, 3].max() < 1e-4, f"base KL {info[:, 3].max():.2e}"
    print(f"base KL max {info[:, 3].max():.1e}; {time.time()-t0:.0f} s", flush=True)

    def kl_of(kids, mode, cache=held, rows=dev_rows):
        kt = torch.as_tensor(np.atleast_1d(kids), device=DEV)
        return np.concatenate(over(cache, rows, lambda x, z: r18.kl_rows(z, run(x, kt, mode)).cpu().numpy()))

    if STEP3:
        return step3(held, dev_rows, over, info, order, R, V, b0, b1, m, t0)
    if STEP2:
        return step2(run, held, dev_rows, over, info, order, R, V, t1, b0, b1, m, t0)

    res = dict(order=order, score=score, pool=pool, dev_rows=dev_rows, dev_peak=info[:, 0], dev_margin=info[:, 1],
               dev_top1=info[:, 2], bias=t1[torch.as_tensor(pool, device=DEV)].cpu().numpy())
    res["one_delete"] = np.stack([kl_of(i, "delete") for i in pool])               # 30 x photos
    res["one_morph"] = np.stack([kl_of(i, "morph") for i in pool])
    print(f"one-child builds done; {time.time()-t0:.0f} s", flush=True)
    top = order[:K_SET].copy()
    res["set11_delete"], res["set11_morph"] = kl_of(top, "delete"), kl_of(top, "morph")
    res["te_rows"] = te_rows
    res["te_set11_delete"] = kl_of(top, "delete", test, te_rows)
    res["te_set11_morph"] = kl_of(top, "morph", test, te_rows)
    print(f"[dev] set11 recovery {1 - res['set11_morph'].mean() / res['set11_delete'].mean():.3f}; "
          f"[test random] {1 - res['te_set11_morph'].mean() / res['te_set11_delete'].mean():.3f} "
          f"(oc_morph.npz: .432)", flush=True)

    # per child at its own peak cell: input before ReLU, channel 69's part, all 256 parts
    pt = torch.as_tensor(pool, device=DEV)
    Rp = R[pt]                                                                    # 30 x 256 x 3 x 3

    def parts(x, z):
        pre = b0.bn1(b0.conv1(x))[:, pt]                                          # B x 30 x 7 x 7
        B = pre.shape[0]
        flat = pre.flatten(2)
        val, arg = flat.max(2)                                                    # B x 30
        cols = F.unfold(x, 3, padding=1, stride=2)                                # B x (256*9) x 49
        cols = cols.view(B, 256, 9, 49)
        idx = arg.view(B, 1, 1, -1).expand(B, 256, 9, len(pool))
        patch = torch.gather(cols, 3, idx)                                        # B x 256 x 9 x 30
        per = torch.einsum("bcks,sck->bsc", patch, Rp.flatten(2))                 # B x 30 x 256
        e = float((per.sum(2) + t1[pt].view(1, -1) - val).abs().max())
        return val.cpu().numpy(), per.half().cpu().numpy(), e
    got = over(held, dev_rows, parts)
    res["peak_in"] = np.concatenate([g[0] for g in got])                          # photos x 30
    res["peak_parts"] = np.concatenate([g[1] for g in got])                       # photos x 30 x 256 (fp16)
    err = max(g[2] for g in got)
    print(f"parts sum check (fp32): max error {err:.2e} (bar 1e-3)", flush=True)
    assert err < 1e-3

    out = os.path.join(r18.OUT, "oc_dial" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


def step2(run, held, dev_rows, over, info, order, R, V, t1, b0, b1, m, t0):
    kt = torch.as_tensor(order[:K_SET].copy(), device=DEV)
    kP = R[kt][:, P:P + 1]

    def parts(x):
        pre = b0.bn1(b0.conv1(x))
        push = F.conv2d(x[:, P:P + 1], kP, stride=2, padding=1)                    # B x 11 x 7 x 7
        return pre, push
    mx = []
    for s in range(0, len(dev_rows), 250):
        x, _ = held.get(dev_rows[s:s + 250])
        mx.append(torch.relu(parts(x)[0][:, kt]).amax((0, 2, 3)))
    own_max = torch.stack(mx).amax(0)
    half = (own_max / 2).view(1, -1, 1, 1)
    print("own max of the 11:", [round(float(v), 2) for v in own_max], flush=True)

    def build(x, mode):
        pre, push = parts(x)
        inner = torch.relu(pre)
        c = inner[:, kt]
        if mode in ("delete", "morph"):                                          # morph: children deleted, parent writes
            c = torch.zeros_like(c)
        elif mode == "delete_weak":
            c = c * (c >= half)
        elif mode == "delete_strong":
            c = c * (c < half)
        elif mode in ("delete_otherroute", "delete_parentroute"):
            pdrv = push >= 0.5 * pre[:, kt]
            c = c * pdrv if mode == "delete_otherroute" else c * (~pdrv)
        elif mode == "child_relu":
            c = torch.relu(push + t1[kt].view(1, -1, 1, 1))
        inner = inner.clone(); inner[:, kt] = c
        r = b0.bn2(b0.conv2(inner))
        if mode == "morph":
            r = r + F.conv2d(push, V[:, kt], padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    def one(md):
        def f(x, z):
            y = build(x, md)
            return np.stack([r18.kl_rows(z, y).cpu().numpy(), y.argmax(1).float().cpu().numpy()], 1)
        return f

    modes = ["delete", "morph", "delete_weak", "delete_strong", "delete_otherroute", "delete_parentroute", "child_relu"]
    res = dict(order=order, dev_rows=dev_rows, dev_peak=info[:, 0], dev_margin=info[:, 1], dev_top1=info[:, 2],
               own_max=own_max.cpu().numpy())
    for md in modes:
        a = np.concatenate(over(held, dev_rows, one(md)))
        res[f"s2_{md}_kl"], res[f"s2_{md}_top1"] = a[:, 0], a[:, 1]
        print(f"{md}: mean KL {a[:, 0].mean():.5f}, top-1 changed {np.mean(a[:, 1] != info[:, 2]):.3f}; "
              f"{time.time()-t0:.0f} s", flush=True)
    out = os.path.join(r18.OUT, "oc_dial2" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


def step3(held, dev_rows, over, info, order, R, V, b0, b1, m, t0):
    """Exact route cut on the logit change (LOG "Step 3"). T = base - delete, Ro = base - routecut, O = routecut - delete,
    M = morph - delete; all centred over classes. Saves per-photo dot products."""
    kt = torch.as_tensor(order[:K_SET].copy(), device=DEV)
    kP = R[kt][:, P:P + 1]

    def build(x, mode):
        pre = b0.bn1(b0.conv1(x))
        push = F.conv2d(x[:, P:P + 1], kP, stride=2, padding=1)
        inner = torch.relu(pre)
        if mode != "base":
            inner = inner.clone()
            inner[:, kt] = torch.relu(pre[:, kt] - push) if mode == "routecut" else 0
        r = b0.bn2(b0.conv2(inner))
        if mode == "morph":
            r = r + F.conv2d(push, V[:, kt], padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    def f(x, z):
        c = lambda y: y - y.mean(1, keepdim=True)
        yb, yd, yr, ym = (c(build(x, md)) for md in ("base", "delete", "routecut", "morph"))
        T, Ro, O, M = yb - yd, yb - yr, yr - yd, ym - yd
        dot = lambda a, b: (a * b).sum(1).cpu().numpy()
        err = float((T - Ro - O).abs().max())
        return np.stack([dot(T, T), dot(Ro, T), dot(O, T), dot(Ro, Ro), dot(O, O), dot(M, T), dot(M, Ro), dot(M, O),
                         dot(M, M), r18.kl_rows(z, build(x, "routecut")).cpu().numpy(), np.full(len(x), err)], 1)
    a = np.concatenate(over(held, dev_rows, f))
    names = ["TT", "RoT", "OT", "RoRo", "OO", "MT", "MRo", "MO", "MM", "kl_routecut", "err"]
    res = dict(order=order, dev_rows=dev_rows, dev_peak=info[:, 0], **{f"s3_{n}": a[:, i] for i, n in enumerate(names)})
    print(f"additivity error max {a[:, 10].max():.2e}; route share {a[:, 1].sum() / a[:, 0].sum():.3f}; "
          f"morph on route {a[:, 6].sum() / a[:, 3].sum():.3f}; {time.time()-t0:.0f} s", flush=True)
    out = os.path.join(r18.OUT, "oc_dial3" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out)


if __name__ == "__main__":
    main()
