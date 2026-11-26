# scale: neurons=s3:69, A:children, A:block1-weights; net=A; photos=held, test; where=cluster
# rung: Q=reads: WHEN each sibling fires at the cell, and can a kept sibling carry a deleted child's job; N=11 R1 children, 3 named pairs; EST=5 min CPU; PREV=results/oc_dial.npz, results/oc_reads_w.npz; PERMIT=Shonu approved plan "Reads: why siblings split" 2026-11-24 incl. its SCALE-UP REQUEST
"""Part B of LOG "Reads: why siblings split" (2026-11-24). Calculated builds only, nothing trained.

B1 (DEV = 2,000 held): cell-level second moments of the 11 children's values a_k = ReLU(pre_k), the parent's push
   through each child (k_k on p), and each pair's woken carrier a_j' (below). Mismatch is read on the Mac.
B3 builds for 3 named pairs (i deleted, j kept), picked from oc_reads_w.py A2 before the run:
   del_i     child i deleted
   morph_i   del_i + the parent's push through i, written with v_i (the old one-child morph)
   hand_a    del_i + c_a * a_j written with v_i          (j's write takes i's message; j's read unchanged)
   hand_b    del_i + j's read also takes i's PRIVATE read (outside the span of the 10 siblings' reads, bias unchanged):
             a_j' = ReLU(pre_j + priv_i * x); j writes v_j * a_j' + c_b * a_j' * v_i
   del_j     child j deleted (what j's job is worth, for W3)
   c_a, c_b = least squares of a_i on the carrier over DEV cells (no intercept), one number per pair.
Photos: DEV, 2,000 random test (= oc_morph test_random), test parent-strong (peak >= 2.92, = oc_morph test_strong).
Usage: python -u oc_when.py [smoke]
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
P = 69
KIDS = [305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275]
PAIRS = [(97, 108), (163, 237), (23, 237)]          # (deleted i, kept j): top private-read overlap, oc_reads_w A2
SMOKE = "smoke" in sys.argv
DEV = r18.DEV


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)                                   # 512 x 256 x 3 x 3
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)                                   # 512 x 512 x 3 x 3
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    order = np.argsort(-(R[:, P].sum((1, 2)) * wnorm).cpu().numpy())
    assert order[:11].tolist() == KIDS, order[:11]
    kt = torch.as_tensor(KIDS, device=DEV)
    kP = R[kt][:, P:P + 1]                                                       # 11 x 1 x 3 x 3

    # private read of each child: outside the span of its 10 siblings' reads (flattened 256*9)
    Rk = R[kt].reshape(11, -1).double()
    priv = {}
    for a, n in enumerate(KIDS):
        Q, _ = torch.linalg.qr(torch.cat([Rk[:a], Rk[a + 1:]]).T)
        r = Rk[a]
        priv[n] = (r - Q @ (Q.T @ r)).float().view(1, 256, 3, 3)
        print(f"n{n}: private share {float((priv[n] ** 2).sum() / (r ** 2).sum()):.3f}", flush=True)
    ix = {n: a for a, n in enumerate(KIDS)}

    held, test = r18.Cache("held"), r18.Cache("test")
    dev_rows = np.arange(len(held))
    te_rand = np.sort(np.random.default_rng(2).choice(len(test), 2000, replace=False))
    pk = []
    for s in range(0, len(test), 1000):
        xs = test.x[s:s + 1000, P]
        xs = xs if torch.is_tensor(xs) else torch.as_tensor(np.ascontiguousarray(xs))
        pk.append(xs.float().amax((1, 2)).cpu().numpy())
    te_strong = np.nonzero(np.concatenate(pk) >= 2.92)[0]
    if SMOKE:
        dev_rows, te_rand, te_strong = dev_rows[:64], te_rand[:64], te_strong[:64]
    print(f"held {len(held)}, test {len(test)}, strong {len(te_strong)}; {time.time()-t0:.0f} s", flush=True)

    def over(cache, rows, fn, bs=250):
        out = []
        for s in range(0, len(rows), bs):
            x, z = cache.get(rows[s:s + bs])
            out.append(fn(x, z))
        return out

    def parts(x):
        pre = b0.bn1(b0.conv1(x))
        push = F.conv2d(x[:, P:P + 1], kP, stride=2, padding=1)                  # B x 11 x 7 x 7
        return pre, push

    def woken(x, pre, i, j):
        return torch.relu(pre[:, j:j + 1] + F.conv2d(x, priv[i], stride=2, padding=1))

    # B1: second moments over DEV cells. Variables: 11 children, 11 pushes, 3 woken carriers.
    def moments(x, z):
        pre, push = parts(x)
        a = torch.relu(pre[:, kt])
        wk = torch.cat([woken(x, pre, i, j) for i, j in PAIRS], 1)
        v = torch.cat([a, push, wk], 1).permute(1, 0, 2, 3).reshape(25, -1).double()
        y = finish(x, torch.relu(pre), 0)
        return v.sum(1).cpu().numpy(), (v @ v.T).cpu().numpy(), v.shape[1], \
            np.stack([x[:, P].amax((1, 2)).cpu().numpy(), r18.kl_rows(z, y).cpu().numpy()] +
                     [woken(x, pre, i, j).amax((1, 2, 3)).cpu().numpy() for i, j in PAIRS] +
                     [torch.relu(pre[:, j]).amax((1, 2)).cpu().numpy() for _, j in PAIRS], 1)

    def finish(x, inner, extra):
        r = b0.bn2(b0.conv2(inner)) + extra
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    got = over(held, dev_rows, moments)
    S1 = sum(g[0] for g in got); S2 = sum(g[1] for g in got); N = sum(g[2] for g in got)
    info = np.concatenate([g[3] for g in got])
    assert info[:, 1].max() < 1e-4, f"base KL {info[:, 1].max():.2e}"
    print(f"B1 moments over {N} cells; base KL max {info[:, 1].max():.1e}; {time.time()-t0:.0f} s", flush=True)

    c = {}
    for p_, (i, j) in enumerate(PAIRS):
        ai, aj, wj = ix[i], ix[j], 22 + p_
        c[(i, j, "a")] = S2[ai, aj] / S2[aj, aj]
        c[(i, j, "b")] = S2[ai, wj] / S2[wj, wj]
        print(f"pair n{i} -> n{j}: c_a {c[(i, j, 'a')]:.3f}, c_b {c[(i, j, 'b')]:.3f}", flush=True)

    def build(x, kind, i, j=None):
        pre, push = parts(x)
        inner = torch.relu(pre)
        if kind == "base":
            return finish(x, inner, 0)
        inner = inner.clone()
        if kind == "del_j":
            inner[:, j] = 0
            return finish(x, inner, 0)
        inner[:, i] = 0
        vi = V[:, i:i + 1]
        if kind == "del_i":
            return finish(x, inner, 0)
        if kind == "morph_i":
            return finish(x, inner, F.conv2d(push[:, ix[i]:ix[i] + 1], vi, padding=1))
        if kind == "hand_a":
            return finish(x, inner, F.conv2d(c[(i, j, "a")] * inner[:, j:j + 1], vi, padding=1))
        if kind == "hand_b":
            wj = woken(x, pre, i, j)
            inner[:, j:j + 1] = wj
            return finish(x, inner, F.conv2d(c[(i, j, "b")] * wj, vi, padding=1))
        raise ValueError(kind)

    res = dict(kids=np.array(KIDS), pairs=np.array(PAIRS), S1=S1, S2=S2, N=N, dev_rows=dev_rows, te_rand=te_rand,
               te_strong=te_strong, dev_peak=info[:, 0], dev_woken_peak=info[:, 2:5], dev_j_peak=info[:, 5:8],
               c=np.array([[c[(i, j, "a")], c[(i, j, "b")]] for i, j in PAIRS]))
    sets = {"dev": (held, dev_rows), "te_rand": (test, te_rand), "te_strong": (test, te_strong)}
    jobs = [("del_i", i, None) for i, _ in PAIRS] + [("morph_i", i, None) for i, _ in PAIRS] + \
           [("del_j", None, j) for j in sorted({j for _, j in PAIRS})] + \
           [(k, i, j) for k in ("hand_a", "hand_b") for i, j in PAIRS]
    seen = set()
    for kind, i, j in jobs:
        key = f"{kind}_{i}_{j}"
        if key in seen:
            continue
        seen.add(key)
        for sn, (cache, rows) in sets.items():
            kl = np.concatenate(over(cache, rows, lambda x, z: r18.kl_rows(z, build(x, kind, i, j)).cpu().numpy()))
            res[f"{sn}_{key}"] = kl
        print(f"{key}: dev {res['dev_' + key].mean():.5f}, te_rand {res['te_rand_' + key].mean():.5f}, "
              f"te_strong {res['te_strong_' + key].mean():.5f}; {time.time()-t0:.0f} s", flush=True)

    out = os.path.join(r18.OUT, "oc_when" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
