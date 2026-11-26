# scale: neurons=A:block0-all; net=A, Z-ladder, C-ladder, WC-ladder, L-ladder, R4C-ladder; photos=test; where=cluster
# rung: Q=plan "Does morphing hurt a kept neuron's own job?", swap-back of each kept unit; N=4 cuts x 4 builds x all kept units, forward only; EST=30 min GPU; PERMIT=Shonu approved plan 2026-11-25
"""Cluster job, forward passes only. Runs in n6/clean/neuron/coarse next to r18.py, oc_retrain.py and oc_vs_why.py.

Question (Shonu, 2026-11-25): the kept neurons morph (widen). Does that hurt each kept neuron's OWN job?

Swap-back, one kept unit j at a time, inside a build m (C, WC, L, R4C):
  stage-4 block 0 output (before the ReLU) is a sum over hidden units:
      s = downsample(x) + bn2 shift + sum_j conv(h_j, V_j)
      h_j = ReLU(bn1_j(conv1_j(x)))  (j's firing map, 7 x 7)    V_j = bn2 scale x conv2[:, j]  (j's effective write)
  variants, exact:
      full   s - conv(h_j^m, V_j^m) + conv(h_j^A, V_j^A)    j alone gets A's read, bn1 and write back
      read   s - conv(h_j^m, V_j^m) + conv(h_j^A, V_j^m)    A's read side only (gate and strength)
      write  s - conv(h_j^m, V_j^m) + conv(h_j^m, V_j^A)    A's write only
  The bn2 shift is shared by all units, so it stays the build's.
  Own-job size: KL(A || A with j deleted) on the same photos.

Photo sets per unit (test photos, 50 each):
  own    j's top 50 by A's peak firing
  clean  j's top 50 among photos where D's summed firing in A is below its median (D quiet)
  part   the partner's top 50, partner = the deleted unit whose A firing best matches j's firing in L
         (jobtest_L.npz best_d from oc_ladder.py)
Saved per unit and set: mean KL(A || .) for base, full, read, write, A-deleted, Z.
Reader (Mac): oc_vs_ownjob_read.py. Bars are in the plan.

Checks: the cached split forward equals A's logits; every build's mean test KL equals the ladder's; in Z and A a
swap-back changes nothing; in C, WC, L, swapping back EVERY kept unit plus A's bn2 shift gives Z exactly.
Usage: python -u oc_vs_ownjob.py [smoke]
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
CUTS = (64,) if SMOKE else (11, 64, 192, 384)
BUILDS = ("C", "WC", "L", "R4C")
NPH = 50
LAD = os.path.join(r18.OUT, "oc_ladder")
OUT = os.path.join(r18.OUT, "oc_vs_ownjob" + ("_smoke" if SMOKE else ""))
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def parts(m):
    """Effective writes V (512 out x 512 in x 3 x 3) and bn2 shift of stage-4 block 0."""
    b0 = m.layer4[0]
    s2, t2 = OR.bn_parts(b0.bn2)
    return (b0.conv2.weight * s2.view(-1, 1, 1, 1)).detach(), t2.detach()


def hidden(m, x):
    b0 = m.layer4[0]
    return torch.relu(b0.bn1(b0.conv1(x)))


def contrib(h, v):
    """One unit's write into the stream: h (B x 1 x 7 x 7) through v (512 x 1 x 3 x 3)."""
    return F.conv2d(h, v, padding=1)


@torch.no_grad()
def logits_of(m, S, bs=300):
    return torch.cat([tail(m, S[a:a + bs]) for a in range(0, len(S), bs)])


@torch.no_grad()
def a_stats(A, X3, bs=250):
    """A's peak and summed firing of every stage-4 block-0 unit, per test photo."""
    pk, sm = [], []
    for a in range(0, len(X3), bs):
        h = hidden(A, X3[a:a + bs].float())
        pk.append(h.amax((2, 3)).cpu())
        sm.append(h.sum((2, 3)).cpu())
    return torch.cat(pk).numpy(), torch.cat(sm).numpy()


@torch.no_grad()
def full_kl(m, X3, ZA, bs=250):
    return torch.cat([r18.kl_rows(ZA[a:a + bs], tail(m, s4b0(m, X3[a:a + bs].float())))
                      for a in range(0, len(X3), bs)]).cpu().numpy()


@torch.no_grad()
def all_swap_check(A, m, X3, ZA, Zm, K, n=1000):
    """Every kept unit back to A (read, bn1, write) and A's bn2 shift: must equal Z."""
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


@torch.no_grad()
def unit_run(A, m, j, idx, X3, ZA, with_A):
    """Mean KL per variant on the photo rows idx (one tensor of indices). Returns dict variant -> per-photo KL."""
    x = X3[idx].float()
    VA, _ = parts(A)
    Vm, _ = parts(m)
    hA = hidden(A, x)[:, j:j + 1]
    hm = hidden(m, x)[:, j:j + 1]
    vA, vm = VA[:, j:j + 1], Vm[:, j:j + 1]
    s = s4b0(m, x)
    cm = contrib(hm, vm)
    S = {"base": s,
         "full": s - cm + contrib(hA, vA),
         "read": s - cm + contrib(hA, vm),
         "write": s - cm + contrib(hm, vA)}
    out = {}
    for k, v in S.items():
        out[k] = r18.kl_rows(ZA[idx], tail(m, v))
    if with_A:
        sA = s4b0(A, x)
        out["Adel"] = r18.kl_rows(ZA[idx], tail(A, sA - contrib(hA, vA)))
    return out


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
    pk, sm = a_stats(A, X3)
    res = dict(check_split_forward=chk, nph=NPH, cuts={})

    for n in CUTS:
        d = os.path.join(LAD, f"n{n}")
        jt = np.load(os.path.join(d, "jobtest_L.npz"))
        D = sorted(order[:n].tolist())
        K = np.setdiff1d(np.arange(512), D)
        assert np.array_equal(jt["D"], D) and np.array_equal(jt["K"], K)
        units = K[jt["alive_A"]]
        partner = {int(K[i]): int(D[jt["best_d"][i]]) for i in range(len(K))}
        if SMOKE:
            units = np.unique(np.r_[units[:19], 339])
        dfire = sm[:, D].sum(1)
        quiet = np.nonzero(dfire < np.median(dfire))[0]
        sel = {}
        for j in units:
            j = int(j)
            own = np.argsort(-pk[:, j])[:NPH]
            clean = quiet[np.argsort(-pk[quiet, j])[:NPH]]
            part = np.argsort(-pk[:, partner[j]])[:NPH]
            sel[j] = dict(own=own, clean=clean, part=part)
        Zm = load_build(A, n, "Z") if os.path.exists(os.path.join(d, "Z.pt")) else None
        if Zm is None:
            Zm = copy.deepcopy(A)
            OR.apply_masks(Zm, OR.masks(Zm, D))
            Zm.eval()
        r = dict(D=D, units=[int(u) for u in units], partner={int(u): partner[int(u)] for u in units},
                 dfire_median=float(np.median(dfire)), builds={})
        arrays = dict(units=units, D=np.asarray(D), partner=np.array([partner[int(u)] for u in units]),
                      dfire=dfire.astype(np.float32))
        for name, ix in (("own", "own"), ("clean", "clean"), ("part", "part")):
            arrays[f"dfire_{name}"] = np.array([dfire[sel[int(u)][ix]].mean() for u in units], np.float32)
        # Z on the unit photo sets (the cut's own damage there), and swap-back sanity in A and Z
        idx_all = {j: torch.as_tensor(np.r_[s["own"], s["clean"], s["part"]], device=DEV) for j, s in sel.items()}
        klZ_full = full_kl(Zm, X3, ZA)
        for nm in ("own", "clean", "part"):
            arrays[f"Z_{nm}"] = np.array([klZ_full[sel[int(u)][nm]].mean() for u in units])
        san = []
        for j in [int(u) for u in units[:5]]:
            for mm in (A, Zm):
                o = unit_run(A, mm, j, idx_all[j], X3, ZA, False)
                san.append(float(max((o[k] - o["base"]).abs().max() for k in ("full", "read", "write"))))
        r["check_swap_noop_A_Z"] = max(san)
        log(f"n{n}: {len(units)} units; check swap-back in A and Z changes nothing: max {max(san):.1e}")
        first = True
        for b in BUILDS:
            m = load_build(A, n, b)
            kl = full_kl(m, X3, ZA)
            row = dict(kl_test=float(kl.mean()), rec=float(1 - kl.mean() / klZ_full.mean()))
            lf = os.path.join(d, f"kl_test_{b}.npy")
            if os.path.exists(lf) and not SMOKE:
                row["check_vs_ladder"] = float(abs(np.load(lf).mean() - kl.mean()))
            if b in ("C", "WC", "L"):
                row["check_allswap_eq_Z"] = all_swap_check(A, m, X3, ZA, Zm, K)
            log(f"n{n} {b}: rec {row['rec']:.3f} kl {row['kl_test']:.4f} checks " +
                ", ".join(f"{k} {v}" for k, v in row.items() if k.startswith("check")))
            acc = {}
            for j in [int(u) for u in units]:
                o = unit_run(A, m, j, idx_all[j], X3, ZA, first)
                for k, v in o.items():
                    v = v.cpu().numpy()
                    for t, (a0, a1) in (("own", (0, NPH)), ("clean", (NPH, 2 * NPH)), ("part", (2 * NPH, 3 * NPH))):
                        key = (f"A_{k}_{t}" if k == "Adel" else f"{b}_{k}_{t}")
                        acc.setdefault(key, []).append(float(v[a0:a1].mean()))
            for k, v in acc.items():
                arrays[k] = np.array(v)
            first = False
            own_h = (arrays[f"{b}_base_own"] - arrays[f"{b}_full_own"]) / np.maximum(arrays["A_Adel_own"], 1e-12)
            log(f"n{n} {b}: own-job harm h median {np.median(own_h):.3f}, 90th pct {np.percentile(own_h, 90):.3f}; "
                f"n339 " + (f"{own_h[list(units).index(339)]:.3f}" if 339 in units else "-"))
            r["builds"][b] = row
            del m
            torch.cuda.empty_cache()
        res["cuts"][str(n)] = r
        np.savez(os.path.join(OUT, f"n{n}.npz"), **arrays)
        json.dump(res, open(os.path.join(OUT, "ownjob.json"), "w"), indent=1)
    log("saved", OUT)


if __name__ == "__main__":
    main()
