# scale: neurons=s3:69, A:children, A:all-units, A:block0-all; net=A, Z-ladder, C-ladder, W-ladder, L-ladder, R4-ladder, R-ladder; photos=train, held, test; where=cluster
# rung: set in the job command
"""Plan "Bigger cuts: calculation against training at each cut size" (2026-11-24). Rungs run in sequence in one job.

D = the top n stage-4 block-0 neurons of A by the R1 score (read weight on channel 69 x write norm). n = 11 is the old D.
Builds (same cut; only the method differs; no weight of D survives in any):
  Z    A with D deleted (read rows, bn1, write columns at 0)
  C    exact least-squares write solve: each deleted firing regressed on the kept block-0 firings (all cells),
       folded into the kept write columns (VECMATH 5 "b0" form, no constant). Fit on train photos (no augmentation),
       damping picked on held photos. C_held: fit on held (VECMATH 5 form, check row)
  W    from Z, train block-0 write (conv2, bn2)                          gates and readers = A's
  L    W + block-0 read (conv1, bn1)                                     readers = A's
  R4   L + rest of stage 4 (downsample, block 1) + head
  R    R4 + stages 0-3 (rungs in --full_at)
  WC   from C, W's chosen setting (calculation, then training)
  R4C  from C, R4's chosen setting
Nested training (memory feedback-nested-training-order): one lr per group; each build fixes the lrs chosen for the
smaller build and sweeps the new group's lr over a grid that INCLUDES 0 (= the smaller build, checked equal).
Changes after step 0 (oc_fix0, job 6990921; logged as made after that result):
  - warm-up: lr rises linearly over the first 5% of steps, then cosine (step 0: freeing stages 0-3 did early damage)
  - tie rule: a new group's lr > 0 is taken only if its held KL is below the lr-0 run's by more than 1%
    (two data orders of old R differed by 1.1%); otherwise the smaller build stands
  - 2x rerun: W and R4's chosen settings always rerun at 2x steps (a cosine schedule always ends flat, so no
    flatness test can see under-training); the 2x result is taken if it is lower by more than 1%, and then every
    later build of the rung uses 2x steps (one doubling at most)
  - every grid run is scored on test and parent-strong too; held per-photo KL saved
  - changed-job test on every second held photo (held is class-sorted, 2 per class)
Scores: recovery = 1 - mean KL(build) / mean KL(Z) on all test photos and the parent-strong ones.
Changed-job test (bar in the plan): for kept block-0 neuron j in L and R4, self = corr(firing in net, firing in A),
take-over = max_d corr(firing in net, A's deleted d) - max_d corr(A's j, A's deleted d). Held photos, all cells.
Usage: python -u oc_ladder.py --n 11,64,192,384 --full_at 11,384 [smoke]
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="11,64,192,384")
ap.add_argument("--full_at", default="11,384")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("smoke", nargs="?", default="")
ARGS = ap.parse_args()
SMOKE = ARGS.smoke == "smoke"
DEV = OR.DEV
STEPS = 20 if SMOKE else ARGS.steps
BS = 32 if SMOKE else 128
EVERY = 10 if SMOKE else 100
TIE = 0.01
ROOT = os.path.join(r18.OUT, "oc_ladder" + ("_smoke" if SMOKE else ""))
T0 = time.time()

GRIDS = {                     # new group of each build -> lr grid (0 = keep the smaller build)
    "W": ("write", (3e-5, 1e-4, 3e-4)),
    "L": ("read", (0.0, 1e-5, 3e-5, 1e-4)),
    "R4": ("s4rest", (0.0, 1e-5, 3e-5)),
    "R": ("outer", (0.0, 1e-6, 3e-6)),
}
if SMOKE:
    GRIDS = {k: (v[0], v[1][:2]) for k, v in GRIDS.items()}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def group_of(name):
    if name.startswith("layer4.0.conv2") or name.startswith("layer4.0.bn2"):
        return "write"
    if name.startswith("layer4.0.conv1") or name.startswith("layer4.0.bn1"):
        return "read"
    if name.startswith("layer4") or name.startswith("fc"):
        return "s4rest"
    return "outer"


def ranking(a):
    b0 = a.layer4[0]
    s1, _ = OR.bn_parts(b0.bn1)
    s2, _ = OR.bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    score = (R[:, OR.P].sum((1, 2)) * torch.sqrt((V ** 2).sum((0, 2, 3)))).detach().cpu().numpy()
    return np.argsort(-score)


def train_g(start, U, ZA_held, Uh, lrs, seed, mk, steps):
    """lrs: {group: lr}; groups not named are frozen. lr 0 = in the optimizer, never moves. Warm-up 5%, then cosine."""
    m = copy.deepcopy(start).eval()
    groups = {}
    for name, p in m.named_parameters():
        g = group_of(name)
        p.requires_grad_(g in lrs)
        if g in lrs:
            groups.setdefault(g, []).append(p)
    opt = torch.optim.AdamW([dict(params=ps, lr=lrs[g], base=lrs[g]) for g, ps in groups.items()], weight_decay=0.0)
    gg = torch.Generator(device=DEV).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed)
    warm = max(1, steps // 20)
    hk0 = float(r18.kl_rows(ZA_held, OR.logits(m, Uh)).mean())
    best = (hk0, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0)
    curve = [(0, float("nan"), hk0)]
    for step in range(1, steps + 1):
        if step <= warm:
            f = step / warm
        else:
            f = 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for gr in opt.param_groups:
            gr["lr"] = gr["base"] * f
        rows = torch.randint(0, len(U), (BS,), generator=gc).to(U.device)
        x = OR.augment(OR.norm(U[rows]), gg)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            with torch.no_grad():
                za = OR.A(x)
            zm = m(x)
        loss = r18.kl_rows(za.float(), zm.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        OR.apply_masks(m, mk)
        if step % EVERY == 0 or step == steps:
            hk = float(r18.kl_rows(ZA_held, OR.logits(m, Uh)).mean())
            curve.append((step, float(loss.detach()), hk))
            if hk < best[0]:
                best = (hk, {k: v.detach().clone() for k, v in m.state_dict().items()}, step)
    m.load_state_dict(best[1])
    return m, best[0], best[2], curve


@torch.no_grad()
def block0_in(m, U, bs=250):
    for a0 in range(0, len(U), bs):
        yield m.layer3(m.layer2(m.layer1(r18.net.stem(m, OR.norm(U[a0:a0 + bs])))))


@torch.no_grad()
def firings(m, x):
    b0 = m.layer4[0]
    return torch.relu(b0.bn1(b0.conv1(x)))


@torch.no_grad()
def moments(a, U, D, K):
    nk = len(K)
    FF = torch.zeros(nk, nk, dtype=torch.float64, device=DEV)
    FA = torch.zeros(nk, len(D), dtype=torch.float64, device=DEV)
    AA = torch.zeros(len(D), dtype=torch.float64, device=DEV)
    for x in block0_in(a, U):
        inner = firings(a, x)
        Ff = inner[:, K].permute(0, 2, 3, 1).reshape(-1, nk).double()
        Af = inner[:, D].permute(0, 2, 3, 1).reshape(-1, len(D)).double()
        FF += Ff.T @ Ff
        FA += Ff.T @ Af
        AA += (Af ** 2).sum(0)
    return dict(FF=FF.cpu().numpy(), FA=FA.cpu().numpy(), AA=AA.cpu().numpy())


def solve(mo, lam_rel):
    lam = lam_rel * np.trace(mo["FF"]) / len(mo["FF"])
    return np.linalg.solve(mo["FF"] + lam * np.eye(len(mo["FF"])), mo["FA"]).T      # (nD, nK)


def resid(mo, beta):
    Bt = beta.T
    return float(np.trace(Bt.T @ mo["FF"] @ Bt) - 2 * np.trace(Bt.T @ mo["FA"]) + mo["AA"].sum())


def fold(z, a, beta, D, K):
    """Kept write column j += sum_d beta_dj * A's write column d (raw conv2 weight space; bn2 scale is per output)."""
    c = copy.deepcopy(z)
    W2a = a.layer4[0].conv2.weight.detach()
    with torch.no_grad():
        add = torch.einsum("dk,odhw->okhw", torch.as_tensor(beta, dtype=W2a.dtype, device=W2a.device), W2a[:, D])
        c.layer4[0].conv2.weight[:, K] += add
    return c


@torch.no_grad()
def job_test(a, m, U, D, K, bs=100):
    """Correlations over (photo, cell) on the given photos: kept firings in m vs A, and vs A's deleted firings."""
    acc = None
    for a0 in range(0, len(U), bs):
        x = OR.norm(U[a0:a0 + bs])
        xa = a.layer3(a.layer2(a.layer1(r18.net.stem(a, x))))
        xm = m.layer3(m.layer2(m.layer1(r18.net.stem(m, x))))
        fa, fm = firings(a, xa), firings(m, xm)
        P_ = fa[:, K].permute(0, 2, 3, 1).reshape(-1, len(K)).double()
        Q_ = fm[:, K].permute(0, 2, 3, 1).reshape(-1, len(K)).double()
        R_ = fa[:, D].permute(0, 2, 3, 1).reshape(-1, len(D)).double()
        parts = dict(sP=P_.sum(0), sQ=Q_.sum(0), sR=R_.sum(0), PP=(P_ ** 2).sum(0), QQ=(Q_ ** 2).sum(0),
                     RR=(R_ ** 2).sum(0), PQ=(P_ * Q_).sum(0), PR=P_.T @ R_, QR=Q_.T @ R_, N=P_.shape[0])
        acc = parts if acc is None else {k: acc[k] + v for k, v in parts.items()}
    N = acc["N"]

    def sd(s, ss):
        return torch.sqrt((ss / N - (s / N) ** 2).clamp_min(1e-12))

    dP, dQ, dR = sd(acc["sP"], acc["PP"]), sd(acc["sQ"], acc["QQ"]), sd(acc["sR"], acc["RR"])
    self_c = (acc["PQ"] / N - acc["sP"] * acc["sQ"] / N ** 2) / (dP * dQ)
    cQR = (acc["QR"] / N - torch.outer(acc["sQ"], acc["sR"]) / N ** 2) / torch.outer(dQ, dR)
    cPR = (acc["PR"] / N - torch.outer(acc["sP"], acc["sR"]) / N ** 2) / torch.outer(dP, dR)
    take = cQR.max(1).values - cPR.max(1).values
    return dict(self=self_c.cpu().numpy(), take=take.cpu().numpy(), best_d=cQR.argmax(1).cpu().numpy(),
                best_corr=cQR.max(1).values.cpu().numpy(), alive_A=(dP > 1e-5).cpu().numpy(),
                alive_m=(dQ > 1e-5).cpu().numpy())


def rung(n, full, U, ZA, strong, order):
    A = OR.A
    OUT = os.path.join(ROOT, f"n{n}")
    os.makedirs(OUT, exist_ok=True)
    D = sorted(order[:n].tolist())
    K = np.setdiff1d(np.arange(512), D).tolist()
    log(f"===== rung n {n}: deleted {len(D)}, kept {len(K)}, R included {full}")
    Z = copy.deepcopy(A)
    mk = OR.masks(Z, D)
    OR.apply_masks(Z, mk)
    kz = {k: r18.kl_rows(ZA[k], OR.logits(Z, U[k])) for k in ("held", "test")}
    zt, zs = float(kz["test"].mean()), float(kz["test"][strong].mean())
    log(f"Z held {kz['held'].mean():.5f}, test {zt:.5f}, strong ({int(strong.sum())}) {zs:.5f}")
    res = dict(n=n, D=D, steps=STEPS, Z=dict(held=float(kz["held"].mean()), test=zt, strong=zs,
                                              n_strong=int(strong.sum())), builds={})
    np.save(os.path.join(OUT, "kl_test_Z.npy"), kz["test"].cpu().numpy())
    np.save(os.path.join(OUT, "strong.npy"), strong.cpu().numpy())
    models = {}

    def score(m):
        kh = r18.kl_rows(ZA["held"], OR.logits(m, U["held"]))
        kt = r18.kl_rows(ZA["test"], OR.logits(m, U["test"]))
        return kh, kt, dict(held_kl=float(kh.mean()), test=float(kt.mean()), strong=float(kt[strong].mean()),
                            rec_test=1 - float(kt.mean()) / zt, rec_strong=1 - float(kt[strong].mean()) / zs)

    def keep(tag, m, extra):
        kh, kt, row = score(m)
        row.update(extra)
        res["builds"][tag] = row
        models[tag] = m
        np.save(os.path.join(OUT, f"kl_test_{tag}.npy"), kt.cpu().numpy())
        np.save(os.path.join(OUT, f"kl_held_{tag}.npy"), kh.cpu().numpy())
        log(f"{tag}: held {row['held_kl']:.5f}; recovery test {row['rec_test']:.3f}, strong {row['rec_strong']:.3f}")

    # ---- C: exact write solve ----
    lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .3, 1., 3., 10.]
    mo_tr, mo_he = moments(A, U["train"], D, K), moments(A, U["held"], D, K)
    for tag, fit, chk in (("C", mo_tr, mo_he), ("C_held", mo_he, mo_tr)):
        vk = [resid(chk, solve(fit, l)) for l in lams]
        li = int(np.argmin(vk))
        beta = solve(fit, lams[li])
        c = fold(Z, A, beta, D, K)
        OR.apply_masks(c, mk)
        keep(tag, c, dict(damping=lams[li], damping_interior=bool(0 < li < len(lams) - 1),
                          rebuilt_share_check=float(1 - resid(chk, beta) / chk["AA"].sum())))
        if tag == "C":
            np.save(os.path.join(OUT, "beta_C.npy"), beta.astype(np.float32))

    # ---- nested training from Z ----
    chosen, prev, steps_now, doubled = {}, None, STEPS, False
    builds = ["W", "L", "R4"] + (["R"] if full else [])
    for b in builds:
        grp, grid = GRIDS[b]
        runs = {}
        for lr in grid:
            lrs = dict(chosen, **{grp: lr})
            m, hk, st, curve = train_g(Z, U["train"], ZA["held"], U["held"], lrs, 0, mk, steps_now)
            _, _, sc = score(m)
            runs[lr] = dict(m=m, hk=hk, st=st, curve=curve, lrs=lrs, sc=sc)
            log(f"  {b} {grp} lr {lr:g}: held {hk:.5f} (step {st}); test {sc['rec_test']:.3f}, strong {sc['rec_strong']:.3f}")
        if 0.0 in runs:
            best_pos = min((k for k in runs if k > 0), key=lambda k: runs[k]["hk"])
            lr = best_pos if runs[best_pos]["hk"] < (1 - TIE) * runs[0.0]["hk"] else 0.0
            eq = runs[0.0]["hk"] == res["builds"][prev]["held_kl_train"]
            res.setdefault("nest_check", {})[b] = dict(lr0_held=runs[0.0]["hk"],
                                                       prev_held=res["builds"][prev]["held_kl_train"], equal=bool(eq))
            log(f"  nest check {b} lr 0 == {prev}: {eq}; tie rule picks lr {lr:g}")
        else:
            lr = min(runs, key=lambda k: runs[k]["hk"])
        r = runs[lr]
        rerun = None
        if b in ("W", "R4") and not doubled and not SMOKE:
            m2, hk2, st2, c2 = train_g(Z, U["train"], ZA["held"], U["held"], r["lrs"], 0, mk, 2 * steps_now)
            _, _, sc2 = score(m2)
            rerun = dict(held=hk2, step=st2, curve=c2, sc=sc2, taken=bool(hk2 < (1 - TIE) * r["hk"]))
            log(f"  {b} 2x steps: held {hk2:.5f} vs {r['hk']:.5f}; test {sc2['rec_test']:.3f}, strong "
                f"{sc2['rec_strong']:.3f}; taken {rerun['taken']}")
            if rerun["taken"]:
                r = dict(m=m2, hk=hk2, st=st2, curve=c2, lrs=r["lrs"], sc=sc2)
                steps_now, doubled = 2 * steps_now, True
                if prev is not None:
                    log("  NOTE: smaller builds ran at fewer steps; nesting from here is at the new length")
        chosen = dict(r["lrs"])
        keep(b, r["m"], dict(held_kl_train=r["hk"], lrs={k: float(v) for k, v in r["lrs"].items()},
                             best_step=r["st"], curve=r["curve"], steps=steps_now, rerun=rerun,
                             grid={str(k): dict(held=v["hk"], **{kk: vv for kk, vv in v["sc"].items()})
                                   for k, v in runs.items()}))
        prev = b

    # ---- from C: calculation, then training ----
    C = models["C"]
    for tag, src in (("WC", "W"), ("R4C", "R4")):
        lrs = res["builds"][src]["lrs"]
        m, hk, st, curve = train_g(C, U["train"], ZA["held"], U["held"], lrs, 0, mk, res["builds"][src]["steps"])
        keep(tag, m, dict(held_kl_train=hk, lrs=lrs, best_step=st, curve=curve, start="C"))

    order_ok = all(res["builds"][builds[i + 1]]["held_kl"] <= res["builds"][builds[i]]["held_kl"] + 1e-9
                   for i in range(len(builds) - 1))
    res["nest_order_ok"] = bool(order_ok)
    res["W_le_C"] = bool(res["builds"]["W"]["held_kl"] <= res["builds"]["C"]["held_kl"])
    log(f"nesting order on held ({' >= '.join(builds)}): {order_ok}; W <= C: {res['W_le_C']}")

    # ---- changed-job test ----
    Uj = U["held"][::2]
    for b in [x for x in ("L", "R4", "R") if x in models]:
        jt = job_test(A, models[b], Uj, D, K)
        changed = (jt["self"] < .8) & (jt["take"] >= .2) & jt["alive_A"]
        res["builds"][b]["changed_job"] = [dict(unit=int(K[i]), self=float(jt["self"][i]), take=float(jt["take"][i]),
                                                best_d=int(D[jt["best_d"][i]]), best_corr=float(jt["best_corr"][i]))
                                           for i in np.nonzero(changed)[0]]
        np.savez(os.path.join(OUT, f"jobtest_{b}.npz"), K=np.array(K), D=np.array(D), **jt)
        log(f"{b}: changed-job neurons {int(changed.sum())}; self median {np.median(jt['self'][jt['alive_A']]):.3f}; "
            f"top take-over {np.sort(jt['take'])[-5:][::-1].round(3).tolist()}")
    ds = (models["R4"].layer4[0].downsample[0].weight - A.layer4[0].downsample[0].weight).detach()
    cn = ds.flatten(2).norm(dim=(0, 2)).cpu().numpy()
    res["parent_skip_change"] = dict(norm69=float(cn[OR.P]), rank69=int((cn > cn[OR.P]).sum()) + 1,
                                     median=float(np.median(cn)), max=float(cn.max()))
    log(f"R4: s3:69 skip-column change norm {cn[OR.P]:.2e} (median {np.median(cn):.2e}), "
        f"rank {res['parent_skip_change']['rank69']} of 256")

    for b, m in models.items():
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, os.path.join(OUT, f"{b}.pt"))
    json.dump(res, open(os.path.join(OUT, "oc_ladder.json"), "w"), indent=1)
    log("saved", OUT)


def main():
    OR.A = r18.load_a()
    A = OR.A
    for p in A.parameters():
        p.requires_grad_(False)
    order = ranking(A)
    assert sorted(order[:11].tolist()) == sorted([305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275])
    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    if SMOKE:
        sets = {k: v[:: {"train": 31, "held": 15, "test": 66}[k]] for k, v in sets.items()}
    U = {k: OR.load_photos(ph, v) for k, v in sets.items()}
    log("photos:", {k: len(v) for k, v in U.items()})
    ZA = {k: OR.logits(A, U[k]) for k in ("held", "test")}
    kaa = float(r18.kl_rows(ZA["held"], OR.logits(A, U["held"])).max())
    strong = OR.parent_peak(A, U["test"]) >= OR.STRONG
    log(f"KL(A||A) max {kaa:.1e}")
    full_at = {int(v) for v in ARGS.full_at.split(",") if v}
    for n in [int(v) for v in ARGS.n.split(",")]:
        rung(n, n in full_at, U, ZA, strong, order)
    log("ALL RUNGS DONE")


if __name__ == "__main__":
    main()
