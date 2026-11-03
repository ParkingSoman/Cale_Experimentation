# serves: TG1
"""Step 3 of the plan, card 3-40: the goal read rule (Definition DG2) against trained reads, on real goals.

Object: the read y = V M z + t on the last block, z the mean pre-activation over positions. Trunk T6's line (a), (b), (c).
Goal = weights on photos, by class. Weights are normalised on each photo list, by the same class rule.
   dogcat     dogs 151-268 and cats 281-285 (123 classes), weight 1. The rest weight 0
   ctrl0-2    123 random classes (fixed seeds), weight 1. The rest weight 0
   dogcat90   the same dogs and cats hold .9 of the weight, every other class shares .1
   equal      all 1000 classes, weight 1

Data, fixed before the run. resnet18: cache/r18, 30 fit and 5 validation per class from its train list, 10 sealed per
class. resnet50: cache/map, 12 fit, 3 validation, 5 sealed per class. Rows are split inside each class, never by taking the
first rows of the file: the lists are sorted by class (LADDER 51, correction of 2026-11-03).
float64. Standardised under the goal's weights, on fit rows.

Methods.
   rule      DG2's span, then the convex fit of TG1. One eigendecomposition per goal, shared by both sizes.
   rule2     the same, then 2 alternating rounds of the two convex fits.
   trained   LBFGS on (M, V, t), chunks of 50 iterations, at most 30 chunks. Starts: logitpca (top directions of the logits
             with a least-squares read), rand0, rand1, rule. Keeps the chunk with the best validation KL, stops after 3
             chunks with no validation gain or a fit gain under 2e-5 in a chunk. The rule never uses validation photos.
             Sealed photos are scored only after all fitting.

Pass lines, fixed before the run.
   (a) sealed KL of rule (or rule2, stated separately) <= 1.3 x the lowest sealed KL over the four trained starts, every goal, k = 14 and 38.
       The reference uses its lowest sealed value, which is the hardest comparison for the rule.
   (b) rule time for both sizes < the mean time of the two random starts at k = 14 (one trained level).
   (c) running the rule twice gives the same sealed KL to 1e-9 (relative).
   (d) rule's sealed KL on dogcat < the mean over ctrl0-2, at each k.
Not a pass line, reported: agreement of top-1 with N on sealed goal photos, and sealed KL on photos outside the goal.

PREDICTION, written before the first run (2026-11-03). ResNet-18.
   equal: trained k = 14 about 1.5, rule/best 1.1 to 1.4. k = 38 trained about .6, rule/best 1.1 to 1.5.
   dogcat: trained k = 14 about .7, k = 38 about .35, rule/best 1.0 to 1.3 at both. Agreement .6 to .8 at k = 14.
   dogcat beats the controls at both sizes. (a) passes on most goals and fails on at least one. (b) and (c) pass.
Every number here is an upper value on what the read can do. Label ONE-BUILD until the lines are read; OPT-GOAL only if all pass on both nets.

ADDED 2026-11-03, AFTER the ResNet-18 run, because line (b) failed on 5 of 6 goals: rule_fast, the same rule with its convex fit stopped at a chunk gain under
1e-4 in place of 1e-6. This is one value, not tuned. On ResNet-18 it is a look after the fact. ResNet-50 is the clean test of it.
PREDICTION, written before any rule_fast run: sealed KL within 3% of rule on every goal, so equal at k = 14 goes over 1.3. Time about a third of rule. (b) passes on about half the goals.

Usage: pc_goal_read.py --net resnet18 [--smoke] [--goals dogcat,ctrl0] [--fast-only]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="resnet18", choices=["resnet18", "resnet50"])
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--goals", default="dogcat,ctrl0,ctrl1,ctrl2,dogcat90,equal")
ap.add_argument("--fast-only", action="store_true", help="add rule_fast to an existing result file, nothing else")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)
SIZES = [14, 38]
CHUNK, MAXCHUNK = (5, 3) if a.smoke else (50, 30)
OUT = os.path.join(HERE, "rounds/round5")
os.makedirs(OUT, exist_ok=True)
TAG = a.net + ("_smoke" if a.smoke else "")
PATH = os.path.join(OUT, f"goal_{TAG}.json")
DOGCAT = list(range(151, 269)) + list(range(281, 286))
assert len(DOGCAT) == 123


def per_class_split(lab, counts):
    """Row indices of consecutive slices inside each class, in file order within the class."""
    idx = [[] for _ in counts]
    for c in range(1000):
        rows = np.where(lab == c)[0]
        pos = 0
        for i, n in enumerate(counts):
            idx[i].append(rows[pos:pos + n])
            pos += n
    return [np.concatenate(v) for v in idx]


def load():
    if a.net == "resnet18":
        lab = np.load(os.path.join(HERE, "cache/r18/lab_train.npy"))
        labs = np.load(os.path.join(HERE, "cache/r18/lab_sealed.npy"))
        counts = (6, 2) if a.smoke else (30, 5)
        fit, val = per_class_split(lab, counts)
        seal = per_class_split(labs, (3 if a.smoke else 10,))[0]
        Zt = np.load(os.path.join(HERE, "cache/r18/z_train.npy"), mmap_mode="r")
        Yt = np.load(os.path.join(HERE, "cache/r18/y_train.npy"), mmap_mode="r")
        Zs = np.load(os.path.join(HERE, "cache/r18/z_sealed.npy"), mmap_mode="r")
        Ys = np.load(os.path.join(HERE, "cache/r18/y_sealed.npy"), mmap_mode="r")
        pool = lambda Z, ix: np.asarray(Z[np.sort(ix)], dtype=np.float32).mean(axis=2).astype(np.float64)
        get = lambda Z, Y, L, ix: (torch.from_numpy(pool(Z, ix)), torch.from_numpy(np.asarray(Y[np.sort(ix)]).astype(np.float64)), L[np.sort(ix)])
        return get(Zt, Yt, lab, fit), get(Zt, Yt, lab, val), get(Zs, Ys, labs, seal)
    meta = np.load(os.path.join(HERE, "cache/map/resnet50_meta.npz"), allow_pickle=True)
    lab = meta["labels"].astype(int)
    fit, val, seal = per_class_split(lab, (4, 2, 2) if a.smoke else (12, 3, 5))
    Z = np.load(os.path.join(HERE, "cache/map/resnet50_z.npy"), mmap_mode="r")
    Y = np.load(os.path.join(HERE, "cache/map/resnet50_y.npy"), mmap_mode="r")
    get = lambda ix: (torch.from_numpy(np.asarray(Z[np.sort(ix)]).astype(np.float64)), torch.from_numpy(np.asarray(Y[np.sort(ix)]).astype(np.float64)), lab[np.sort(ix)])
    return get(fit), get(val), get(seal)


def class_weights(goal):
    w = np.zeros(1000)
    if goal == "equal":
        w[:] = 1
    elif goal == "dogcat":
        w[DOGCAT] = 1
    elif goal == "dogcat90":
        w[:] = .1 / 877
        w[DOGCAT] = .9 / 123
    elif goal.startswith("ctrl"):
        w[np.random.default_rng(1000 + int(goal[4:])).choice(1000, 123, replace=False)] = 1
    return w


class Part:
    """One photo list under a goal. The whole list stays, and rows at weight 0 are dropped for the goal's terms."""

    def __init__(self, data, w, mu=None, sd=None):
        Z, Y, L = data
        p = torch.from_numpy(w[L])
        keep = p > 0
        self.p = p[keep] / p[keep].sum()
        if mu is None:
            mu = (self.p[:, None] * Z[keep]).sum(0)
            sd = ((self.p[:, None] * (Z[keep] - mu) ** 2).sum(0)).sqrt() + 1e-4
        self.mu, self.sd = mu, sd
        self.x = (Z[keep] - mu) / sd
        self.y = Y[keep]
        self.ls = torch.log_softmax(self.y, 1)
        self.s = self.ls.exp()
        off = ~keep
        self.off = None
        if off.any():
            self.off = ((Z[off] - mu) / sd, torch.log_softmax(Y[off], 1))


def wkl(P, y):
    return (P.p * (P.s * (P.ls - torch.log_softmax(y, 1))).sum(1)).sum()


def scores(P, M, V, t):
    """KL under the goal's weights, top-1 agreement with N, KL on photos outside the goal."""
    with torch.no_grad():
        y = P.x @ M.T @ V.T + t
        out = dict(kl=float(wkl(P, y)), agree=float((P.p * (y.argmax(1) == P.y.argmax(1))).sum()))
        if P.off is not None:
            x, ls = P.off
            out["off_kl"] = float((ls.exp() * (ls - torch.log_softmax(x @ M.T @ V.T + t, 1))).sum(1).mean())
    return out


def lbfgs(params, fn, maxchunk=None, tol=1e-6):
    """Deterministic chunks of LBFGS on a convex problem. Stops when a chunk gains under tol."""
    o = torch.optim.LBFGS(params, lr=1, max_iter=CHUNK, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def cl():
        o.zero_grad()
        v = fn()
        v.backward()
        return v

    last, chunks = float(fn().detach()), 0
    for chunks in range(1, (maxchunk or MAXCHUNK) + 1):
        o.step(cl)
        now = float(fn().detach())
        if last - now < tol:
            return now, chunks, "gain under tol"
        last = now
    return last, chunks, "chunk limit"


def fit_write(F, c, V0, t0, tol=1e-6):
    V, t = V0.detach().contiguous().clone().requires_grad_(True), t0.detach().contiguous().clone().requires_grad_(True)
    val, ch, stop = lbfgs([V, t], lambda: wkl(F, c @ V.T + t), tol=tol)
    with torch.no_grad():
        q = torch.softmax(c @ V.T + t, 1)
        w = F.p[:, None] * (q - F.s)
        g = float(((w.T @ c).norm() ** 2 + w.sum(0).norm() ** 2).sqrt())
    return V.detach(), t.detach(), dict(fit=val, chunks=ch, stop=stop, grad=g)


def fit_read(F, V, M0, t0):
    M, t = M0.detach().contiguous().clone().requires_grad_(True), t0.detach().contiguous().clone().requires_grad_(True)
    val, ch, stop = lbfgs([M, t], lambda: wkl(F, F.x @ M.T @ V.T + t))
    return M.detach(), t.detach()


def rule_span(F):
    t0 = time.time()
    ybar = (F.p[:, None] * F.y).sum(0)
    Yc = F.y - ybar
    Szz = (F.p[:, None] * F.x).T @ F.x
    Szy = (F.p[:, None] * F.x).T @ Yc
    sbar = (F.p[:, None] * F.s).sum(0)
    Fm = torch.diag(sbar) - (F.p[:, None] * F.s).T @ F.s
    ev, U = torch.linalg.eigh(Szz)
    keep = ev > 1e-10 * ev.max()
    W = (U[:, keep] / ev[keep].sqrt()) @ U[:, keep].T
    B = W @ Szy
    A = B @ Fm @ B.T
    evA, UA = torch.linalg.eigh((A + A.T) / 2)
    UA = UA.flip(1)
    sign = torch.sign(UA[UA.abs().argmax(0), torch.arange(UA.shape[1])])
    UA = UA * sign
    return dict(W=W, B=B, UA=UA, ybar=ybar, sec=time.time() - t0, top=[float(v) for v in evA.flip(0)[:5]])


def rule_level(F, sp, k, tol=1e-6):
    t0 = time.time()
    Uk = sp["UA"][:, :k]
    M = Uk.T @ sp["W"]
    V0 = sp["B"].T @ Uk
    V, t, info = fit_write(F, F.x @ M.T, V0, sp["ybar"], tol=tol)
    info["sec"] = time.time() - t0
    return M, V, t, info


def alternate(F, M, V, t, rounds=2):
    t0 = time.time()
    for _ in range(rounds):
        M, t = fit_read(F, V, M, t)
        V, t, info = fit_write(F, F.x @ M.T, V, t)
    info["sec"] = time.time() - t0
    return M, V, t, info


def trained(F, Va, start):
    M, V, t = [x.detach().contiguous().clone().requires_grad_(True) for x in start]
    o = torch.optim.LBFGS([M, V, t], lr=1, max_iter=CHUNK, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def cl():
        o.zero_grad()
        v = wkl(F, F.x @ M.T @ V.T + t)
        v.backward()
        return v

    t0 = time.time()
    best, keep, bad, lastfit, stop, chunks = 1e18, None, 0, float("inf"), "chunk limit", 0
    for chunks in range(1, MAXCHUNK + 1):
        o.step(cl)
        with torch.no_grad():
            fit = float(wkl(F, F.x @ M.T @ V.T + t))
            v = float(wkl(Va, Va.x @ M.T @ V.T + t))
        if v < best:
            best, keep, bad = v, (M.detach().clone(), V.detach().clone(), t.detach().clone()), 0
        else:
            bad += 1
        if bad >= 3:
            stop = "validation"
            break
        if lastfit - fit < 2e-5:
            stop = "fit gain"
            break
        lastfit = fit
    return keep, dict(val=best, fit=fit, chunks=chunks, stop=stop, sec=time.time() - t0)


def starts(F, k, rule_build, seed_base):
    C, d = F.y.shape[1], F.x.shape[1]
    D = F.y - (F.p[:, None] * F.y).sum(0)
    D = D - D.mean(1, keepdim=True)
    Cov = (F.p[:, None] * D).T @ D
    Q = torch.linalg.eigh(Cov)[1].flip(1)[:, :k]
    sq = F.p.sqrt()[:, None]
    M0 = torch.linalg.lstsq(sq * F.x, sq * (D @ Q)).solution.T
    ybar = (F.p[:, None] * F.y).sum(0)
    out = {"logitpca": (M0, Q.clone(), ybar)}
    for i in range(2):
        g = torch.Generator().manual_seed(seed_base + i)
        out[f"rand{i}"] = (torch.randn(k, d, generator=g) / d ** .5, torch.randn(C, k, generator=g) / k ** .5, ybar)
    out["rule"] = rule_build
    return out


if a.smoke:
    from torchvision.models import ResNet18_Weights
    names = ResNet18_Weights.DEFAULT.meta["categories"]
    print("class check", [(i, names[i]) for i in (150, 151, 268, 269, 280, 281, 285, 286)], flush=True)
data = load()
print("loaded", [d[0].shape for d in data], flush=True)
results = json.load(open(PATH)) if os.path.exists(PATH) and not a.smoke else {}
FAST = 1e-4


def add_fast(F, Se, sp, k, lv):
    M, V, t, info = rule_level(F, sp, k, tol=FAST)
    lv["rule_fast"] = dict(info, **scores(Se, M, V, t))
    lv["ratio_rule_fast"] = lv["rule_fast"]["kl"] / lv["best_trained"]


if a.fast_only:
    for goal in a.goals.split(","):
        w = class_weights(goal)
        F = Part(data[0], w)
        Se = Part(data[2], w, F.mu, F.sd)
        sp = rule_span(F)
        for k in SIZES:
            add_fast(F, Se, sp, k, results[goal][str(k)])
        res = results[goal]
        res["span_sec_fast"] = sp["sec"]
        zoom = sp["sec"] + sum(res[str(k)]["rule_fast"]["sec"] for k in SIZES)
        rand = np.mean([res["14"]["trained"][f"rand{i}"]["sec"] for i in range(2)])
        print(goal, "zoom sec", round(zoom, 1), "rand sec", round(rand, 1), "(b)", bool(zoom < rand), [(k, round(res[str(k)]["rule_fast"]["kl"], 3), round(res[str(k)]["ratio_rule_fast"], 3), res[str(k)]["rule_fast"]["chunks"]) for k in SIZES], flush=True)
        with open(PATH, "w") as f:
            json.dump(results, f, indent=1)
    raise SystemExit(0)

for goal in a.goals.split(","):
    w = class_weights(goal)
    F = Part(data[0], w)
    Va = Part(data[1], w, F.mu, F.sd)
    Se = Part(data[2], w, F.mu, F.sd)
    res = dict(fit_rows=F.x.shape[0], val_rows=Va.x.shape[0], sealed_rows=Se.x.shape[0])
    sp = rule_span(F)
    res["span_sec"] = sp["sec"]
    print(goal, "rows", res["fit_rows"], res["val_rows"], res["sealed_rows"], "span sec", round(sp["sec"], 2), flush=True)
    for k in SIZES:
        M, V, t, info = rule_level(F, sp, k)
        lv = dict(rule=dict(info, **scores(Se, M, V, t)))
        if goal in ("dogcat", "ctrl0"):
            M_, V_, t_, _ = rule_level(F, sp, k)
            lv["rule_repeat_rel_diff"] = abs(scores(Se, M_, V_, t_)["kl"] - lv["rule"]["kl"]) / lv["rule"]["kl"]
        M2, V2, t2, info2 = alternate(F, M, V, t)
        lv["rule2"] = dict(info2, **scores(Se, M2, V2, t2))
        print(goal, k, "rule", {x: round(lv["rule"][x], 4) for x in ("kl", "agree", "fit", "grad", "sec")}, "rule2", round(lv["rule2"]["kl"], 4), round(lv["rule2"]["sec"], 1), flush=True)
        lv["trained"] = {}
        for name, st in starts(F, k, (M, V, t), 7000 + k).items():
            kp, tr = trained(F, Va, st)
            lv["trained"][name] = dict(tr, **scores(Se, *kp))
            print(goal, k, name, {x: round(lv["trained"][name][x], 4) for x in ("kl", "agree", "fit", "val", "sec")}, lv["trained"][name]["stop"], flush=True)
        lv["best_trained"] = min(v["kl"] for v in lv["trained"].values())
        lv["ratio_rule"] = lv["rule"]["kl"] / lv["best_trained"]
        lv["ratio_rule2"] = lv["rule2"]["kl"] / lv["best_trained"]
        add_fast(F, Se, sp, k, lv)
        print(goal, k, "rule_fast", round(lv["rule_fast"]["kl"], 4), "ratio", round(lv["ratio_rule_fast"], 3), "sec", round(lv["rule_fast"]["sec"], 1), flush=True)
        res[str(k)] = lv
        results[goal] = res
        with open(PATH, "w") as f:
            json.dump(results, f, indent=1)
        print(goal, k, "RATIO rule", round(lv["ratio_rule"], 3), "rule2", round(lv["ratio_rule2"], 3), flush=True)

print("\nsummary (sealed KL)")
for goal, res in results.items():
    for k in SIZES:
        if str(k) not in res:
            continue
        lv = res[str(k)]
        rand = np.mean([lv["trained"][f"rand{i}"]["sec"] for i in range(2)])
        print(goal, k, "rule", round(lv["rule"]["kl"], 3), "rule2", round(lv["rule2"]["kl"], 3), "best", round(lv["best_trained"], 3), "ratio", round(lv["ratio_rule"], 3), round(lv["ratio_rule2"], 3), "agree", round(lv["rule"]["agree"], 3), "rule sec", round(res["span_sec"] + lv["rule"]["sec"], 1), "rand sec", round(rand, 1))
