# scale: neurons=s3:69, A:children, A:all-units; net=A, R, R4; photos=train, held, test; where=cluster
# rung: Q=delete set D (the 11 R1 children of s3:69), retrain the whole network free (R) and stage 4 + head only (R4); N=1 net, 2 data orders; EST=30 min GPU; PERMIT=Shonu approved plan "Delete set D: train R, read it neuron by neuron" 2026-11-24
"""Rung 1 of the plan "Delete set D: train R, read it neuron by neuron, design the C-vs-R comparison" (2026-11-24).

Z  = A with D removed: D's read rows (conv1), bn1 scale and shift, and write columns (conv2) set to 0 and kept at 0.
R  = Z, then every other weight trained (BN scale and shift too; BN running statistics frozen, eval mode).
R4 = the same, but only stage 4 and the head free.
Loss: KL(A || net) on the same augmented photo (random zoom 0.6-1, shift, left-right flip). Weight decay 0.
KL is 0 at A, so every weight change is a response to the deletion.
Photos: the reading photos (r18.photo_sets): train for training, held for lr and checkpoint, test for the report.
Report: recovery = 1 - KL(net) / KL(Z), ratio of means, on test parent-strong (s3:69 peak >= 2.92) and all test.
Checks: R1a identity (per-unit activation correlation A vs R on held, every block's inner neurons and stream),
        R1c the two data orders agree (cosine of the weight-change vectors, whole and per layer).
Runs on the cluster in n6/clean/neuron/coarse. Saved: results/coarse/r18/oc_retrain/.
Usage: python -u oc_retrain.py [smoke]
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
CPUPHOTOS = "cpuphotos" in sys.argv      # keep photos in CPU memory (small GPUs); results identical
DEV = r18.DEV
P, K_N = 69, 11
STRONG = 2.92
LRS = (3e-6, 1e-5, 3e-5, 1e-4)
STEPS, BS, EVERY = (20, 32, 10) if SMOKE else (1000, 128, 100)
OUT = os.path.join(r18.OUT, "oc_retrain" + ("_smoke" if SMOKE else ""))
os.makedirs(OUT, exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def children(a):
    b0 = a.layer4[0]
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).detach().cpu().numpy()
    return np.argsort(-score)[:K_N].tolist()


def masks(m, D):
    """The weights that must stay 0 (D's read rows, bn1 scale and shift, write columns)."""
    b0 = m.layer4[0]
    out = {}
    for name, p in m.named_parameters():
        mk = torch.zeros_like(p, dtype=torch.bool)
        if p is b0.conv1.weight or p is b0.bn1.weight or p is b0.bn1.bias:
            mk[D] = True
        elif p is b0.conv2.weight:
            mk[:, D] = True
        if mk.any():
            out[name] = mk
    return out


def apply_masks(m, mk):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name in mk:
                p[mk[name]] = 0


def load_photos(ph, idx):
    """uint8 photos on the GPU, read in sorted file order."""
    rows = ph.ix[idx]
    order = np.argsort(rows)
    srt = rows[order]
    assert np.all(np.diff(srt) > 0)
    buf = np.empty((len(idx), 224, 224, 3), np.uint8)
    BLK = 1000                                         # read the file in sequential blocks (random reads on /bigtemp are slow)
    for a in range(int(srt[0]) // BLK * BLK, int(srt[-1]) + 1, BLK):
        sel = np.nonzero((srt >= a) & (srt < a + BLK))[0]
        if len(sel) == 0:
            continue
        block = np.asarray(ph.X[a:a + BLK])
        buf[order[sel]] = block[srt[sel] - a]
    t = torch.from_numpy(buf)
    return t.pin_memory() if CPUPHOTOS and DEV == "cuda" else (t if CPUPHOTOS else t.to(DEV))


def norm(u8):
    return ((u8.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0) - MEAN) / STD


def augment(x, g):
    n = x.shape[0]
    s = 0.6 + 0.4 * torch.rand(n, device=DEV, generator=g)
    tx = (torch.rand(n, device=DEV, generator=g) * 2 - 1) * (1 - s)
    ty = (torch.rand(n, device=DEV, generator=g) * 2 - 1) * (1 - s)
    f = torch.where(torch.rand(n, device=DEV, generator=g) < 0.5, -1.0, 1.0)
    th = torch.zeros(n, 2, 3, device=DEV)
    th[:, 0, 0], th[:, 0, 2], th[:, 1, 1], th[:, 1, 2] = s * f, tx, s, ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


@torch.no_grad()
def logits(m, U, bs=250):
    return torch.cat([m(norm(U[a:a + bs])).float() for a in range(0, len(U), bs)])


@torch.no_grad()
def parent_peak(a, U, bs=250):
    out = []
    for a0 in range(0, len(U), bs):
        h = a.layer3(a.layer2(a.layer1(r18.net.stem(a, norm(U[a0:a0 + bs])))))
        out.append(h[:, P].amax((1, 2)).float())
    return torch.cat(out)


def train(z, U, ZA_held, Uh, lr, seed, free, mk):
    m = copy.deepcopy(z).eval()
    for name, p in m.named_parameters():
        p.requires_grad_(free(name))
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator(device=DEV).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed)
    hk0 = float(r18.kl_rows(ZA_held, logits(m, Uh)).mean())
    best = (hk0, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0)
    curve = [(0, float("nan"), hk0)]
    for step in range(1, STEPS + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        rows = torch.randint(0, len(U), (BS,), generator=gc).to(U.device)
        x = augment(norm(U[rows]), g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            with torch.no_grad():
                za = A(x)
            zm = m(x)
        loss = r18.kl_rows(za.float(), zm.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        apply_masks(m, mk)
        if step % EVERY == 0 or step == STEPS:
            hk = float(r18.kl_rows(ZA_held, logits(m, Uh)).mean())
            curve.append((step, float(loss.detach()), hk))
            if hk < best[0]:
                best = (hk, {k: v.detach().clone() for k, v in m.state_dict().items()}, step)
    m.load_state_dict(best[1])
    return m, best[0], best[2], curve


@torch.no_grad()
def identity(a, m, Uh, D, n=1000, bs=100):
    """Pearson correlation per unit, A vs m, over (photo, cell), on n held photos. Every block's inner and stream h."""
    acc = {}
    for a0 in range(0, min(n, len(Uh)), bs):
        x = norm(Uh[a0:a0 + bs])
        _, oa = r18.net.forward(a, x)
        _, om = r18.net.forward(m, x)
        for key in oa:
            for part in ("inner", "h"):
                if part not in oa[key]:
                    continue
                u = oa[key][part].double().transpose(0, 1).flatten(1)
                v = om[key][part].double().transpose(0, 1).flatten(1)
                s = acc.setdefault(f"s{key[0]}b{key[1]}_{part}", [0] * 6)
                for i, t in enumerate((u.sum(1), v.sum(1), (u * u).sum(1), (v * v).sum(1), (u * v).sum(1))):
                    s[i] = s[i] + t
                s[5] += u.shape[1]
    out = {}
    for k, (su, sv, suu, svv, suv, N) in acc.items():
        cov = suv / N - su * sv / N ** 2
        r = cov / torch.sqrt((suu / N - (su / N) ** 2).clamp_min(1e-12) * (svv / N - (sv / N) ** 2).clamp_min(1e-12))
        r = r.cpu().numpy()
        if k == "s4b0_inner":
            r[D] = np.nan
        out[k] = r
    return out


def delta(m, z):
    sz = dict(z.named_parameters())
    return {n: (p.detach() - sz[n].detach()).flatten() for n, p in m.named_parameters()}


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    global A
    A = r18.load_a()
    D = children(A)
    log("D:", D)
    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    if SMOKE:
        sets = {k: v[:: {"train": 31, "held": 15, "test": 66}[k]] for k, v in sets.items()}
    U = {k: load_photos(ph, v) for k, v in sets.items()}
    log("photos:", {k: len(v) for k, v in U.items()})

    Z = copy.deepcopy(A)
    mk = masks(Z, D)
    apply_masks(Z, mk)
    ZA = {k: logits(A, U[k]) for k in ("held", "test")}
    kaa = float(r18.kl_rows(ZA["held"], logits(A, U["held"])).max())
    pk = parent_peak(A, U["test"])
    strong = pk >= STRONG
    kz = {k: r18.kl_rows(ZA[k], logits(Z, U[k])) for k in ("held", "test")}
    log(f"check KL(A||A) max {kaa:.2e}; Z KL held {kz['held'].mean():.4f}, test all {kz['test'].mean():.4f}, "
        f"test strong ({int(strong.sum())}) {kz['test'][strong].mean():.4f}")

    full = lambda n: True
    s4 = lambda n: n.startswith("layer4") or n.startswith("fc")
    res = dict(D=D, kl_AA=kaa, Z=dict(held=float(kz["held"].mean()), test=float(kz["test"].mean()),
                                    strong=float(kz["test"][strong].mean()), n_strong=int(strong.sum())), runs={})

    def report(tag, m, hk, st, curve, lr, seed):
        kt = r18.kl_rows(ZA["test"], logits(m, U["test"]))
        row = dict(lr=lr, seed=seed, held_kl=hk, best_step=st, curve=curve,
                   test=float(kt.mean()), strong=float(kt[strong].mean()),
                   rec_test=1 - float(kt.mean()) / res["Z"]["test"],
                   rec_strong=1 - float(kt[strong].mean()) / res["Z"]["strong"])
        res["runs"][tag] = row
        log(f"{tag}: lr {lr:g} seed {seed} held KL {hk:.5f} (step {st}); recovery strong {row['rec_strong']:.3f}, "
            f"all test {row['rec_test']:.3f}")
        np.save(os.path.join(OUT, f"kl_test_{tag}.npy"), kt.cpu().numpy())
        return row

    # sweep (order 0) for R and R4
    best = {}
    for name, free in (("R", full), ("R4", s4)):
        for lr in (LRS[1:2] if SMOKE else LRS):
            m, hk, st, curve = train(Z, U["train"], ZA["held"], U["held"], lr, 0, free, mk)
            report(f"{name}_lr{lr:g}_s0", m, hk, st, curve, lr, 0)
            if name not in best or hk < best[name][0]:
                best[name] = (hk, lr, m)
    R0 = best["R"][2]
    R1, hk1, st1, c1 = train(Z, U["train"], ZA["held"], U["held"], best["R"][1], 1, full, mk)
    report(f"R_lr{best['R'][1]:g}_s1", R1, hk1, st1, c1, best["R"][1], 1)
    res["chosen"] = dict(R=best["R"][1], R4=best["R4"][1])

    # R1a identity, R1c agreement
    ident = {tag: identity(A, m, U["held"], D) for tag, m in (("R_s0", R0), ("R_s1", R1), ("R4_s0", best["R4"][2]))}
    res["identity"] = {tag: {k: dict(median=float(np.nanmedian(v)), p05=float(np.nanpercentile(v, 5)),
                                     min=float(np.nanmin(v))) for k, v in d.items()} for tag, d in ident.items()}
    for tag, d in res["identity"].items():
        worst = min(d.items(), key=lambda kv: kv[1]["median"])
        log(f"R1a {tag}: lowest layer median {worst[0]} {worst[1]['median']:.3f}")
    d0, d1 = delta(R0, Z), delta(R1, Z)
    whole = cos(torch.cat([d0[n] for n in d0]), torch.cat([d1[n] for n in d1]))
    per = {n: cos(d0[n], d1[n]) for n in d0 if d0[n].norm() > 0}
    res["agree"] = dict(whole=whole, per_param=per)
    log(f"R1c whole weight-change cosine order 0 vs 1: {whole:.3f}")

    np.savez(os.path.join(OUT, "identity.npz"), **{f"{t}__{k}": v for t, d in ident.items() for k, v in d.items()})
    for tag, m in (("R_s0", R0), ("R_s1", R1), ("R4_s0", best["R4"][2])):
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, os.path.join(OUT, f"{tag}.pt"))
    np.save(os.path.join(OUT, "parent_peak_test.npy"), pk.cpu().numpy())
    json.dump(res, open(os.path.join(OUT, "oc_retrain.json"), "w"), indent=1)
    log("saved", OUT)


if __name__ == "__main__":
    main()
