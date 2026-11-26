# scale: neurons=s3:69, s3:all, A:children, A:block0-all, A:block1-weights, A:head-weights; net=A; photos=held, test; where=cluster
# rung: Q=vector-math: write held to the parent's gate vs every gate, with the skip copy removed; N=1 deleted set of 11, 4 gate sets; EST=5 min CPU; PREV=results/oc_vm_rebuild.npz; PERMIT=Shonu approved plan "everywhere against the parent's gate, without the skip copy" 2026-11-24 incl. its SCALE-UP REQUEST
"""VECMATH section 5 (2026-11-24). Calculated builds only, exact write solve (least squares), nothing trained.

Same object as oc_vm_rebuild.py: network A, stage-4 block 0, D = the 11 R1 children of s3:69, deleted.
Gates (firing signals) j: 0..500 kept block-0 neurons, 501..756 stage-3 channels at the 1x1 skip positions, 757 const.
Gate sets, each with its own damping picked on a DEV split:
  all758   every gate (repeat of section 3's V-known, check row)
  b0p      the 501 kept block-0 neurons + the parent's own 1x1 column (s3:69). "Everywhere, no skip copy"
  b0       the 501 kept block-0 neurons only
  p        the parent's 1x1 column only (repeat of section 3's k=1 row)
Output: KL per test photo for base, V0 and each build; D's summed firing per photo; the photo groups.
Usage: python -u oc_vm_noskip.py [smoke]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

P = 69
KIDS = [305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275]
SMOKE = "smoke" in sys.argv
DEV = r18.DEV
BS = 125
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    order = np.argsort(-(R[:, P].sum((1, 2)) * wnorm).cpu().numpy())
    assert order[:11].tolist() == KIDS, order[:11]
    kt = torch.as_tensor(KIDS, device=DEV)
    kept = torch.as_tensor(np.setdiff1d(np.arange(512), KIDS), device=DEV)
    VD = V[:, kt].contiguous()
    UD = VD.sum((2, 3))
    NB, NS = 501, 256
    NJ = NB + NS + 1
    gsets = {"all758": np.arange(NJ), "b0p": np.r_[np.arange(NB), NB + P], "b0": np.arange(NB), "p": np.array([NB + P])}

    held, test = r18.Cache("held"), r18.Cache("test")
    dev_rows = np.arange(len(held))
    te_rand = np.sort(np.random.default_rng(2).choice(len(test), 2000, replace=False))

    def parts(x):
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        Fs = x[:, :, ::2, ::2]
        return inner, torch.cat([inner[:, kept], Fs, torch.ones_like(Fs[:, :1])], 1)

    def tail(x, r):
        return r18.net.head(m, r18.net.block_parts(b1, torch.relu(r + b0.downsample(x)))["h"])

    def stream(inner, beta=None, Fall=None):
        inner = inner.clone()
        inner[:, kt] = 0
        r = b0.bn2(b0.conv2(inner))
        if beta is not None:
            bt = torch.as_tensor(beta, dtype=torch.float32, device=DEV)
            ah_b = torch.einsum("dj,bjhw->bdhw", bt[:, :NB], Fall[:, :NB])
            ah_s = torch.einsum("dj,bjhw->bdhw", bt[:, NB:], Fall[:, NB:])
            r = r + F.conv2d(ah_b, VD, padding=1) + torch.einsum("od,bdhw->bohw", UD, ah_s)
        return r

    @torch.no_grad()
    def dsum(cache, rows):
        out = []
        for s in range(0, len(rows), 500):
            x, _ = cache.get(rows[s:s + 500])
            out.append(torch.relu(b0.bn1(b0.conv1(x)))[:, kt].sum((1, 2, 3)).cpu().numpy())
        return np.concatenate(out)

    dev_d = dsum(held, dev_rows)
    te_d = dsum(test, np.arange(len(test)))
    hi, lo = np.quantile(dev_d, .9), np.quantile(dev_d, .5)
    pk = []
    for s in range(0, len(test), 1000):
        xs = test.x[s:s + 1000, P]
        xs = xs if torch.is_tensor(xs) else torch.as_tensor(np.ascontiguousarray(xs))
        pk.append(xs.float().amax((1, 2)).cpu().numpy())
    sets = {"te_rand": te_rand, "te_strong": np.nonzero(np.concatenate(pk) >= 2.92)[0],
            "te_fired": np.nonzero(te_d >= hi)[0], "te_quiet": np.intersect1d(te_rand, np.nonzero(te_d <= lo)[0])}
    if SMOKE:
        dev_rows = dev_rows[:64]
        sets = {k: v[:48] for k, v in sets.items()}
    print({k: len(v) for k, v in sets.items()}, f"{time.time()-t0:.0f} s", flush=True)

    # DEV moments, two halves (fit half and check half for damping)
    perm = np.random.default_rng(7).permutation(dev_rows)
    halves = [perm[:len(perm) // 2], perm[len(perm) // 2:]]
    mom = []
    for rows_h in halves:
        a = dict(FF=np.zeros((NJ, NJ)), FA=np.zeros((NJ, 11)), AA=np.zeros(11))
        for s in range(0, len(rows_h), BS):
            x, _ = held.get(np.sort(rows_h[s:s + BS]))
            with torch.no_grad():
                inner, Fall = parts(x)
                Ff = Fall.permute(0, 2, 3, 1).reshape(-1, NJ).double()
                Af = inner[:, kt].permute(0, 2, 3, 1).reshape(-1, 11).double()
                a["FF"] += (Ff.T @ Ff).cpu().numpy(); a["FA"] += (Ff.T @ Af).cpu().numpy()
                a["AA"] += (Af ** 2).sum(0).cpu().numpy()
        mom.append(a)
    full = {k: mom[0][k] + mom[1][k] for k in mom[0]}
    print(f"DEV moments done; {time.time()-t0:.0f} s", flush=True)

    def solve(a, lam_rel, S):
        A_ = a["FF"][np.ix_(S, S)]
        lam = lam_rel * np.trace(A_) / len(S)
        b = np.linalg.solve(A_ + lam * np.eye(len(S)), a["FA"][S])
        beta = np.zeros((11, NJ)); beta[:, S] = b.T
        return beta

    def resid(a, beta):
        Bt = beta.T
        return float(np.trace(Bt.T @ a["FF"] @ Bt) - 2 * np.trace(Bt.T @ a["FA"]) + a["AA"].sum())

    lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .3, 1., 3., 10., 30., 100.]
    builds, res = {}, dict(kids=np.array(KIDS), dev_d=dev_d, te_d=te_d, hi=hi, lo=lo,
                           **{f"set_{k}": v for k, v in sets.items()})
    for g, S in gsets.items():
        vk = [resid(mom[1], solve(mom[0], l, S)) for l in lams]
        lam = lams[int(np.argmin(vk))]
        builds[g] = solve(full, lam, S)
        res[f"lam_{g}"], res[f"val_{g}"], res[f"beta_{g}"] = lam, np.array(vk), builds[g]
        print(f"{g}: {len(S)} gates, damping {lam} (held-out DEV half resid {min(vk):.1f}; "
              f"rebuilt share of D's firing energy {1 - resid(full, builds[g]) / full['AA'].sum():.3f})", flush=True)

    evl = np.unique(np.concatenate(list(sets.values())))
    kl = {b: [] for b in ["base", "V0"] + list(builds)}
    for s in range(0, len(evl), BS):
        rows = evl[s:s + BS]
        x, z = test.get(rows)
        with torch.no_grad():
            inner, Fall = parts(x)
            r0 = stream(inner)
            kl["base"].append(r18.kl_rows(z, tail(x, r0 + F.conv2d(inner[:, kt], VD, padding=1))).cpu().numpy())
            kl["V0"].append(r18.kl_rows(z, tail(x, r0)).cpu().numpy())
            for b, beta in builds.items():
                kl[b].append(r18.kl_rows(z, tail(x, stream(inner, beta, Fall))).cpu().numpy())
    kl = {b: np.concatenate(v) for b, v in kl.items()}
    assert kl["base"].max() < 1e-4, f"base KL {kl['base'].max():.2e}"
    res["evl"] = evl
    for b, v in kl.items():
        res[f"kl_{b}"] = v
    pos = {r_: i for i, r_ in enumerate(evl)}
    print(f"base KL max {kl['base'].max():.1e}", flush=True)
    for sn, rows in sets.items():
        ix = np.array([pos[r_] for r_ in rows])
        d0 = kl["V0"][ix].mean()
        print(f"[{sn}] n {len(rows)}, V0 {d0:.5f}; " + "; ".join(
            f"{b} {1 - kl[b][ix].mean() / d0:.3f}" for b in builds), flush=True)

    out = os.path.join(r18.OUT, "oc_vm_noskip" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
