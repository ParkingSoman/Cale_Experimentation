# serves: T9
"""Toy chain check Y9 of COARSE-THEORY.md Part C: does training converge to the formula of Theorem T9?
Tracks random-start students: excess D2 over the formula, distance of centred outputs to the formula's, and the
bound of T9 part 4, at logged steps. Usage: c3_converge.py [quick]
"""
import sys, json, time
import numpy as np
import torch
import chain as C

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
STEPS = 600 if quick else 12000
LOG = sorted(set([0, 100, 200, 500, 1000, 2000, 3000, 4500, 6000, 8000, 10000, 12000]))
LOG = [s for s in LOG if s <= STEPS] + ([STEPS] if STEPS not in LOG else [])
SEEDS = 1 if quick else 3

T = C.Teacher(seed=0); T.load_state_dict(torch.load("teacher_gated_sym.pt", weights_only=True)); T.eval()
for p in T.parameters():
    p.requires_grad_(False)
X, _ = C.sample(3000, seed=1, sym=True)
fm = C.formula(T, X)
y, g, mu = fm["y"], fm["g"], fm["mu"]


def centred(o):
    return (o - o.mean(0)) @ C.PPERP


def d2_best_bias(o):
    """D2 of a build with its best bias (the bias does not change centred outputs)."""
    return float(0.5 * ((centred(o) - centred(y)) ** 2).sum(1).mean())


res = {}
t0 = time.time()
for k, modes in C.strict_widths(fm):
    if k > 5:
        continue
    gap = float(mu[k - 1] - mu[k])
    const = (1 + np.sqrt(2 * float(mu[0]) / gap)) ** 2
    with torch.no_grad():
        yf = C.t10_student(T, fm, modes)(X, g)
    cf = centred(yf)
    dstar = d2_best_bias(yf)
    size = float((cf ** 2).sum(1).mean())
    runs = []
    for seed in range(SEEDS):
        S = C.Student(k, k, k, seed=seed)
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(S.parameters(), 3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        trace = []
        for it in range(STEPS + 1):
            if it in LOG:
                with torch.no_grad():
                    o = S(X, g)
                exc = d2_best_bias(o) - dstar
                dist2 = float(((centred(o) - cf) ** 2).sum(1).mean())
                trace.append({"step": it, "excess / formula": exc / dstar, "rel distance": float(np.sqrt(dist2 / size)),
                              "distance^2 / bound": dist2 / max(2 * exc * const, 1e-300), "excess >= 0": exc >= -1e-9 * dstar})
            if it == STEPS:
                break
            idx = torch.randint(len(X), (512,), generator=gen)
            loss = C.d2(y[idx], S(X[idx], g[:, idx]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        runs.append(trace)
        print(k, seed, trace[-1], f"{time.time() - t0:.0f}s", flush=True)
    res[k] = {"gap / mu_1": gap / float(mu[0]), "bound constant": const, "runs": runs}
json.dump(res, open("c3_converge_quick.json" if quick else "c3_converge.json", "w"), indent=1, default=str)
