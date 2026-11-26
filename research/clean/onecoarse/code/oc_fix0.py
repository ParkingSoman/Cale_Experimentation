# scale: neurons=s3:69, A:children, A:all-units; net=A, R, R4; photos=train, held, test; where=cluster
# rung: Q=step 0 of plan "Bigger cuts": fix the training bug (R ends worse than R4) with one lr per weight group; N=1 net, 9 runs at the 11-neuron cut; EST=15 min GPU; PERMIT=Shonu approved plan "Bigger cuts: calculation against training at each cut size" 2026-11-24 incl. its SCALE-UP REQUEST
"""Step 0 of the plan "Bigger cuts: calculation against training at each cut size" (2026-11-24).

Bug: R (every weight free) ended worse than R4 (stage 4 + head free) in oc_retrain.py (held KL .00731 vs .00725).
Hypothesis: one lr for every layer. Stage 4 wants 3e-5; at 3e-5 stages 0-3 wreck the features in the first steps.
Fix: one lr per group. inner = stage 4 + head, outer = stages 0-3 (stem, layer1-3).

Runs (same seed 0, same photos, same 1,000 steps x 128 as oc_retrain.py):
  R4ref         inner 3e-5, outer FROZEN (requires_grad False) = oc_retrain's chosen R4
  R_o0          inner 3e-5, outer lr 0 but requires_grad True  -> must equal R4ref exactly (the failing test)
  R_i{a}_o{b}   inner a in (1e-5, 3e-5), outer b in (1e-6, 3e-6, 1e-5)
Pass bar: (1) R_o0 == R4ref (held KL equal, stage-4 weights equal); (2) chosen R held KL <= R4ref;
          (3) test recovery of chosen R >= R4ref's within the paired bootstrap interval.
Usage: python -u oc_fix0.py [smoke]
"""
import copy
import json
import math
import os
import time

import numpy as np
import torch

import oc_retrain as OR
import r18

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SMOKE = OR.SMOKE
DEV = OR.DEV
STEPS, BS, EVERY = OR.STEPS, OR.BS, OR.EVERY
OUT = os.path.join(r18.OUT, "oc_fix0" + ("_smoke" if SMOKE else ""))
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def group_of(name):
    return "inner" if (name.startswith("layer4") or name.startswith("fc")) else "outer"


def train_g(z, U, ZA_held, Uh, lrs, seed, mk):
    """lrs: {group: lr or None}. None = frozen (requires_grad False). 0 = in the optimizer with lr 0."""
    m = copy.deepcopy(z).eval()
    groups = {}
    for name, p in m.named_parameters():
        g = group_of(name)
        p.requires_grad_(lrs.get(g) is not None)
        if lrs.get(g) is not None:
            groups.setdefault(g, []).append(p)
    opt = torch.optim.AdamW([dict(params=ps, lr=lrs[g], base=lrs[g]) for g, ps in groups.items()], weight_decay=0.0)
    g = torch.Generator(device=DEV).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed)
    hk0 = float(r18.kl_rows(ZA_held, OR.logits(m, Uh)).mean())
    best = (hk0, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0)
    curve = [(0, float("nan"), hk0)]
    for step in range(1, STEPS + 1):
        f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        for gr in opt.param_groups:
            gr["lr"] = gr["base"] * f
        rows = torch.randint(0, len(U), (BS,), generator=gc).to(U.device)
        x = OR.augment(OR.norm(U[rows]), g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            with torch.no_grad():
                za = OR.A(x)
            zm = m(x)
        loss = r18.kl_rows(za.float(), zm.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        OR.apply_masks(m, mk)
        if step % EVERY == 0 or step == STEPS:
            hk = float(r18.kl_rows(ZA_held, OR.logits(m, Uh)).mean())
            curve.append((step, float(loss.detach()), hk))
            if hk < best[0]:
                best = (hk, {k: v.detach().clone() for k, v in m.state_dict().items()}, step)
    m.load_state_dict(best[1])
    return m, best[0], best[2], curve


def main():
    OR.A = r18.load_a()
    A = OR.A
    D = OR.children(A)
    log("D:", D)
    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    if SMOKE:
        sets = {k: v[:: {"train": 31, "held": 15, "test": 66}[k]] for k, v in sets.items()}
    U = {k: OR.load_photos(ph, v) for k, v in sets.items()}
    log("photos:", {k: len(v) for k, v in U.items()})

    Z = copy.deepcopy(A)
    mk = OR.masks(Z, D)
    OR.apply_masks(Z, mk)
    ZA = {k: OR.logits(A, U[k]) for k in ("held", "test")}
    strong = OR.parent_peak(A, U["test"]) >= OR.STRONG
    kz = {k: r18.kl_rows(ZA[k], OR.logits(Z, U[k])) for k in ("held", "test")}
    zt, zs = float(kz["test"].mean()), float(kz["test"][strong].mean())
    log(f"Z KL held {kz['held'].mean():.5f}, test {zt:.5f}, strong ({int(strong.sum())}) {zs:.5f}")

    res = dict(D=D, Z=dict(held=float(kz["held"].mean()), test=zt, strong=zs), runs={})
    kts, models = {}, {}

    def run(tag, lrs):
        m, hk, st, curve = train_g(Z, U["train"], ZA["held"], U["held"], lrs, 0, mk)
        kt = r18.kl_rows(ZA["test"], OR.logits(m, U["test"]))
        row = dict(lrs=lrs, held_kl=hk, best_step=st, curve=curve, test=float(kt.mean()),
                   strong=float(kt[strong].mean()), rec_test=1 - float(kt.mean()) / zt,
                   rec_strong=1 - float(kt[strong].mean()) / zs)
        res["runs"][tag] = row
        kts[tag], models[tag] = kt, m
        log(f"{tag}: {lrs} held {hk:.5f} (step {st}); rec strong {row['rec_strong']:.3f}, test {row['rec_test']:.3f}")
        np.save(os.path.join(OUT, f"kl_test_{tag}.npy"), kt.cpu().numpy())

    inner_grid = (3e-5,) if SMOKE else (1e-5, 3e-5)
    outer_grid = (3e-6,) if SMOKE else (1e-6, 3e-6, 1e-5)
    run("R4ref", {"inner": 3e-5, "outer": None})
    run("R_o0", {"inner": 3e-5, "outer": 0.0})

    # the failing test: outer lr 0 must equal R4
    a, b = models["R4ref"].state_dict(), models["R_o0"].state_dict()
    wdiff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    hk_eq = res["runs"]["R4ref"]["held_kl"] == res["runs"]["R_o0"]["held_kl"]
    res["test_o0"] = dict(max_weight_diff=wdiff, held_equal=hk_eq)
    log(f"TEST outer lr 0 == R4: max weight diff {wdiff:.3e}, held KL equal {hk_eq}")

    for i in inner_grid:
        for o in outer_grid:
            run(f"R_i{i:g}_o{o:g}", {"inner": i, "outer": o})

    cands = [t for t in res["runs"] if t.startswith("R_")]
    chosen = min(cands, key=lambda t: res["runs"][t]["held_kl"])
    res["chosen"] = chosen

    # paired bootstrap of test recovery: chosen R minus R4ref
    gen = np.random.default_rng(0)
    kr, k4, k0 = (kts[chosen].cpu().numpy().astype(np.float64), kts["R4ref"].cpu().numpy().astype(np.float64),
                  kz["test"].cpu().numpy().astype(np.float64))
    sm = strong.cpu().numpy()
    boot = {}
    for nm, sel in (("test", np.ones_like(sm)), ("strong", sm)):
        idx = np.nonzero(sel)[0]
        diffs = []
        for _ in range(2000):
            j = idx[gen.integers(0, len(idx), len(idx))]
            diffs.append((1 - kr[j].mean() / k0[j].mean()) - (1 - k4[j].mean() / k0[j].mean()))
        d0 = (1 - kr[idx].mean() / k0[idx].mean()) - (1 - k4[idx].mean() / k0[idx].mean())
        boot[nm] = dict(diff=float(d0), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))
    res["chosen_minus_R4"] = boot
    hr, h4 = res["runs"][chosen]["held_kl"], res["runs"]["R4ref"]["held_kl"]
    res["bar"] = dict(o0_equal=bool(wdiff == 0.0 and hk_eq), held_ok=bool(hr <= h4),
                      test_ok=bool(boot["test"]["hi"] >= 0 and boot["strong"]["hi"] >= 0))
    log(f"chosen {chosen}: held {hr:.5f} vs R4ref {h4:.5f}; recovery minus R4ref: "
        f"test {boot['test']['diff']:+.4f} [{boot['test']['lo']:+.4f}, {boot['test']['hi']:+.4f}], "
        f"strong {boot['strong']['diff']:+.4f} [{boot['strong']['lo']:+.4f}, {boot['strong']['hi']:+.4f}]")
    log("BAR", res["bar"])
    torch.save({k: v.cpu() for k, v in models[chosen].state_dict().items()}, os.path.join(OUT, "R_fixed.pt"))
    json.dump(res, open(os.path.join(OUT, "oc_fix0.json"), "w"), indent=1)
    log("saved", OUT)


if __name__ == "__main__":
    main()
