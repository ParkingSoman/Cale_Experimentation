# scale: neurons=s3:69, s3:all, A:children, A:block0-all; net=A; photos=saved; where=mac
# rung: Q=read VECMATH step 3 (recovery with CIs, carriers that emerged); N=saved oc_vm_rebuild.npz; EST=10 s Mac; PERMIT=Shonu approved plan "New track: deletion as vector math" 2026-11-24
"""Reads results/oc_vm_rebuild.npz (pulled from the cluster). Mac, numpy."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "oc_vm_rebuild.npz")
d = np.load(f)
evl = d["evl"]
pos = {r: i for i, r in enumerate(evl)}
names = d["names"]
kids = d["kids"]
rng = np.random.default_rng(0)


def rec(b, rows, n_boot=2000):
    ix = np.array([pos[r] for r in rows])
    a, z = d[f"kl_{b}"][ix], d["kl_V0"][ix]
    pt = 1 - a.mean() / z.mean()
    bs = rng.integers(0, len(ix), (n_boot, len(ix)))
    boot = 1 - a[bs].mean(1) / z[bs].mean(1)
    return pt, np.percentile(boot, [2.5, 97.5])


print("damping: known", d["lam_known"], "KL", d["lam_kl"], "| top-16 class mass on DEV", round(float(d["dev_mass"]), 4))
print("held-out DEV check, known:", np.round(d["val_known"], 2).tolist())
print("held-out DEV check, KL   :", np.round(d["val_kl"], 5).tolist())
print()
print(f"{'set':10s} {'n':>5s} {'V0 KL':>8s} | {'V-known rec [CI]':>24s} {'pred':>6s} | {'V-KL rec [CI]':>24s} {'pred':>6s} | diff KL-known [CI]")
for sn in ("te_fired", "te_quiet", "te_strong", "te_rand"):
    rows = d[f"set_{sn}"]
    ix = np.array([pos[r] for r in rows])
    out = []
    for b in ("V-known", "V-KL"):
        pt, ci = rec(b, rows)
        pr = 1 - d[f"pred_{b}"][ix].mean() / d["pred_V0"][ix].mean()
        out.append(f"{pt:7.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {pr:6.3f}")
    bs = rng.integers(0, len(ix), (2000, len(ix)))
    z = d["kl_V0"][ix]
    diff = (d["kl_V-known"][ix][bs].mean(1) - d["kl_V-KL"][ix][bs].mean(1)) / z[bs].mean(1)
    dpt = (d["kl_V-known"][ix].mean() - d["kl_V-KL"][ix].mean()) / z.mean()
    print(f"{sn:10s} {len(rows):5d} {z.mean():8.5f} | {out[0]:>24s} | {out[1]:>24s} | {dpt:.3f} [{np.percentile(diff,2.5):.3f}, {np.percentile(diff,97.5):.3f}]")
    # harm check on quiet photos: build KL above V0?
    if sn == "te_quiet":
        for b in ("V-known", "V-KL"):
            print(f"   quiet photos: {b} mean KL {d[f'kl_{b}'][ix].mean():.5f} vs V0 {z.mean():.5f}; "
                  f"photos worse than V0: {(d[f'kl_{b}'][ix] > z).mean():.2f}")

print("\nstop curve (recovery, known / KL) by top-k carriers (order = V-KL shares):")
for sn in ("te_fired", "te_quiet", "te_strong", "te_rand"):
    rows = d[f"set_{sn}"]
    parts = []
    for k in list(d["ks"][:-1]):
        parts.append(f"k{k}: {rec(f'known_k{k}', rows, 1)[0]:.2f}/{rec(f'kl_k{k}', rows, 1)[0]:.2f}")
    parts.append(f"all: {rec('V-known', rows, 1)[0]:.2f}/{rec('V-KL', rows, 1)[0]:.2f}")
    print(f"  {sn:10s}", "  ".join(parts))
print("size curve (DEV photos):")
for sn in ("te_fired", "te_quiet", "te_strong", "te_rand"):
    rows = d[f"set_{sn}"]
    parts = [f"n{s}: {rec(f'known_n{s}', rows, 1)[0]:.2f}/{rec(f'kl_n{s}', rows, 1)[0]:.2f}" for s in d["sizes"][:-1]]
    parts.append(f"n{d['sizes'][-1]}: {rec('V-known', rows, 1)[0]:.2f}/{rec('V-KL', rows, 1)[0]:.2f}")
    print(f"  {sn:10s}", "  ".join(parts))

print("\ncarriers that emerged (share of the rebuilt signal):")
for nm in ("known", "kl"):
    sh = d[f"share_{nm}"]
    top = np.argsort(-sh)[:15]
    rank69 = int((sh > sh[list(names).index("s3:69")]).sum())
    grp = {"block-0": sh[:501].sum(), "stage-3 skip": sh[501:757].sum(), "constant": sh[757]}
    print(f"  {nm:5s} s3:69 rank {rank69} share {sh[list(names).index('s3:69')]:.3f} | by kind:",
          " ".join(f"{k} {v:.2f}" for k, v in grp.items()))
    print("        top:", " ".join(f"{names[j]} {sh[j]:.3f}" for j in top))
    # how concentrated: carriers needed for 50% / 80% of the positive share
    s = np.sort(np.clip(sh, 0, None))[::-1]
    c = np.cumsum(s) / s.sum()
    print(f"        carriers for 50% / 80% of positive share: {int((c < .5).sum()) + 1} / {int((c < .8).sum()) + 1}")
    pd = d[f"perd_{nm}"]
    for i, n in enumerate(kids):
        t = np.argsort(-pd[i])[:6]
        print(f"        n{n}: " + " ".join(f"{names[j]} {pd[i, j]:.2f}" for j in t))
