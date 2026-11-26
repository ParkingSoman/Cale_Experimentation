# scale: neurons=A:block0-all; net=A, Z-ladder, C-ladder, WC-ladder, L-ladder, R4C-ladder; photos=test; where=cluster
# rung: Q=plan "Where did each deleted neuron's job go?", swap-back of each kept unit read on each deleted unit's photos; N=4 cuts x 4 builds x all kept units, forward only; EST=30 min GPU; PERMIT=Shonu approved plan 2026-11-25
"""Cluster job, forward passes only. Runs in n6/clean/neuron/coarse next to r18.py, oc_retrain.py and oc_vs_why.py.

Question (Shonu, 2026-11-25, option A of the broad concept): in the trained builds, where did each deleted neuron's
job go? To ONE kept neuron (merge concept), a few, or MANY (shared take-over)?

Swap-back, one kept unit j at a time, inside a build m (C, WC, L, R4C). Same exact operation as oc_vs_ownjob.py:
  stage-4 block 0 output (before the ReLU):  s = downsample(x) + bn2 shift + sum_j conv(h_j, V_j)
  full   s - conv(h_j^m, V_j^m) + conv(h_j^A, V_j^A)    j alone gets A's read, bn1 and write back
  read   s - conv(h_j^m, V_j^m) + conv(h_j^A, V_j^m)    A's read side only (L and R4C)
  write  s - conv(h_j^m, V_j^m) + conv(h_j^m, V_j^A)    A's write only (L and R4C)
Per photo in the union U of the photo sets: dkl[j, photo] = KL(A || m with j swapped back) - KL(A || m).
Photo sets: each deleted unit d's top 50 test photos by A's peak firing; plus 500 random test photos (control).
g[j, d] = mean of dkl[j] over d's photos. Reader (Mac): oc_vs_jobmap_read.py. Bars are in the plan.

Checks: the cached split forward equals A's logits; every build's mean test KL equals the ladder's; in A and Z a
swap-back changes nothing; in C, WC, L, swapping back EVERY kept unit plus A's bn2 shift gives Z exactly.
Helpers parts / hidden / contrib / full_kl / all_swap_check / a_stats are copied from oc_vs_ownjob.py (2026-11-25),
not imported, because that file belongs to a run of another session.
Usage: python -u oc_vs_jobmap.py [smoke]
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import oc_retrain as OR
import r18
from oc_vs_why import load_build, ranking, s4b0, stage3, tail

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
DEV = OR.DEV
CUTS = (11,) if SMOKE else (11, 64, 192, 384)
BUILDS = ("C", "WC", "L", "R4C")
VARIANTS = {"C": ("full",), "WC": ("full",), "L": ("full", "read", "write"), "R4C": ("full", "read", "write")}
NPH, NRAND = 50, 500
LAD = os.path.join(r18.OUT, "oc_ladder")
OUT = os.path.join(r18.OUT, "oc_vs_jobmap" + ("_smoke" if SMOKE else ""))
T0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


# ---- copied from oc_vs_ownjob.py ----
def parts(m):
    b0 = m.layer4[0]
    s2, t2 = OR.bn_parts(b0.bn2)
    return (b0.conv2.weight * s2.view(-1, 1, 1, 1)).detach(), t2.detach()


def hidden(m, x):
    b0 = m.layer4[0]
    return torch.relu(b0.bn1(b0.conv1(x)))


def contrib(h, v):
    return F.conv2d(h, v, padding=1)


@torch.no_grad()
def a_stats(A, X3, bs=250):
    pk = []
    for a in range(0, len(X3), bs):
        pk.append(hidden(A, X3[a:a + bs].float()).amax((2, 3)).cpu())
    return torch.cat(pk).numpy()


@torch.no_grad()
def full_kl(m, X3, ZA, bs=250):
    return torch.cat([r18.kl_rows(ZA[a:a + bs], tail(m, s4b0(m, X3[a:a + bs].float())))
                      for a in range(0, len(X3), bs)])


@torch.no_grad()
def all_swap_check(A, m, X3, ZA, Zm, K, n=1000):
    x = X3[:n].float()
    VA, tA = parts(A)
    Vm, tm = parts(m)
    hA, hm = hidden(A, x), hidden(m, x)
    s = s4b0(m, x)
    mk = torch.zeros(512, device=DEV)
    mk[K] = 1
    s = s - F.conv2d(hm * mk.view(1, -1, 1, 1), Vm, padding=1) + F.conv2d(hA * mk.view(1, -1, 1, 1), VA, padding=1)
    s = s + (tA - tm).view(1, -1, 1, 1)
    sZ = s4b0(Zm, x)
    return float((s - sZ).abs().max()), float(r18.kl_rows(tail(Zm, sZ), tail(m, s)).max())
# ---- end of copy ----


@torch.no_grad()
def swap_dkl(A, m, X3u, ZAu, units, variants, bs=500):
    """dkl[variant][i, p]: KL(A || m with units[i] swapped back) - KL(A || m), photo p of the union."""
    VA, _ = parts(A)
    Vm, _ = parts(m)
    out = {v: np.zeros((len(units), len(X3u)), np.float32) for v in variants}
    base_all = []
    for a in range(0, len(X3u), bs):
        x = X3u[a:a + bs].float()
        za = ZAu[a:a + bs]
        s = s4b0(m, x)
        hA, hm = hidden(A, x), hidden(m, x)
        base = r18.kl_rows(za, tail(m, s))
        base_all.append(base)
        for i, j in enumerate(units):
            cm = contrib(hm[:, j:j + 1], Vm[:, j:j + 1])
            for v in variants:
                if v == "full":
                    s2 = s - cm + contrib(hA[:, j:j + 1], VA[:, j:j + 1])
                elif v == "read":
                    s2 = s - cm + contrib(hA[:, j:j + 1], Vm[:, j:j + 1])
                else:
                    s2 = s - cm + contrib(hm[:, j:j + 1], VA[:, j:j + 1])
                out[v][i, a:a + len(x)] = (r18.kl_rows(za, tail(m, s2)) - base).cpu().numpy()
    return out, torch.cat(base_all).cpu().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    A = r18.load_a().eval()
    for p in A.parameters():
        p.requires_grad_(False)
    order = ranking(A)
    assert sorted(order[:11].tolist()) == sorted([305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275])
    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    U = OR.load_photos(ph, sets["test"])
    ZA0 = OR.logits(A, U)
    X3 = stage3(A, U)
    del U
    with torch.no_grad():
        ZA = torch.cat([tail(A, s4b0(A, X3[a:a + 250].float())) for a in range(0, len(X3), 250)])
    chk = float(r18.kl_rows(ZA0, ZA).max())
    log(f"test photos {len(X3)}; check: KL(A full || A via cached stage 3) max {chk:.1e}")
    pk = a_stats(A, X3)
    rand = np.sort(np.random.default_rng(0).choice(len(X3), NRAND, replace=False))
    res = dict(check_split_forward=chk, nph=NPH, nrand=NRAND, cuts={})

    for n in CUTS:
        d = os.path.join(LAD, f"n{n}")
        jt = np.load(os.path.join(d, "jobtest_L.npz"))
        D = sorted(order[:n].tolist())
        K = np.setdiff1d(np.arange(512), D)
        assert np.array_equal(jt["D"], D) and np.array_equal(jt["K"], K)
        top = np.stack([np.argsort(-pk[:, dd])[:NPH] for dd in D])          # D x 50 photo ids
        uni = np.unique(np.r_[top.ravel(), rand])
        pos = {int(p): i for i, p in enumerate(uni)}
        top_u = np.vectorize(pos.get)(top)                                   # D x 50, positions in the union
        rand_u = np.array([pos[int(p)] for p in rand])
        uidx = torch.as_tensor(uni, device=X3.device)
        X3u, ZAu = X3[uidx], ZA[uidx]
        Zm = copy.deepcopy(A)
        OR.apply_masks(Zm, OR.masks(Zm, D))
        Zm.eval()
        klZ_full = full_kl(Zm, X3, ZA).cpu().numpy()
        klZ_u = klZ_full[uni]
        units = [int(k) for k in K]
        r = dict(D=D, n_union=int(len(uni)), builds={})
        arrays = dict(D=np.asarray(D), K=K, union=uni, top_u=top_u, rand_u=rand_u, klZ_u=klZ_u.astype(np.float32),
                      pkD_u=pk[uni][:, D].astype(np.float32))
        # sanity: swap-back changes nothing in A and Z (first 5 units)
        san = []
        for mm in (A, Zm):
            o, _ = swap_dkl(A, mm, X3u[:200], ZAu[:200], units[:5], ("full", "read", "write"))
            san.append(float(max(np.abs(v).max() for v in o.values())))
        r["check_swap_noop_A_Z"] = max(san)
        log(f"n{n}: D {len(D)}, K {len(K)}, union photos {len(uni)}; check swap-back in A and Z: max {max(san):.1e}")
        for b in BUILDS:
            m = load_build(A, n, b)
            for p in m.parameters():
                p.requires_grad_(False)
            kl = full_kl(m, X3, ZA).cpu().numpy()
            row = dict(kl_test=float(kl.mean()), rec=float(1 - kl.mean() / klZ_full.mean()))
            lf = os.path.join(d, f"kl_test_{b}.npy")
            if os.path.exists(lf):
                row["check_vs_ladder"] = float(abs(np.load(lf).mean() - kl.mean()))
            if b in ("C", "WC", "L"):
                row["check_allswap_eq_Z"] = all_swap_check(A, m, X3, ZA, Zm, K)
            log(f"n{n} {b}: rec {row['rec']:.3f} checks " +
                ", ".join(f"{k} {v}" for k, v in row.items() if k.startswith("check")))
            dkl, base = swap_dkl(A, m, X3u, ZAu, units, VARIANTS[b])
            arrays[f"{b}_base_u"] = base.astype(np.float32)
            for v, M in dkl.items():
                arrays[f"{b}_{v}"] = M.astype(np.float16) if M.size > 4e7 else M
            g = np.stack([dkl["full"][:, top_u[i]].mean(1) for i in range(len(D))], 1)   # K x D
            R = klZ_u[top_u].mean(1) - base[top_u].mean(1)
            gp = np.clip(g, 0, None)
            share1 = gp.max(0) / np.maximum(gp.sum(0), 1e-12)
            ok = R >= .1 * klZ_u[top_u].mean(1)
            row.update(n_counted=int(ok.sum()), share1_median=float(np.median(share1[ok])) if ok.any() else None,
                       sumg_over_R_median=float(np.median(g.sum(0)[ok] / R[ok])) if ok.any() else None)
            log(f"n{n} {b}: counted d {int(ok.sum())}/{len(D)}; top-1 share median {row['share1_median']}; "
                f"sum g / R median {row['sumg_over_R_median']}" +
                (f"; n476's top unit {units[int(g[:, D.index(476)].argmax())]}" if 476 in D else ""))
            r["builds"][b] = row
            del m
            torch.cuda.empty_cache()
        res["cuts"][str(n)] = r
        np.savez(os.path.join(OUT, f"n{n}.npz"), **arrays)
        json.dump(res, open(os.path.join(OUT, "jobmap.json"), "w"), indent=1)
    log("saved", OUT)


if __name__ == "__main__":
    main()
