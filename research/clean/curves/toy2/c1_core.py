# serves: T9
# serves: T10
# serves: T6
"""Toy chain checks Y1-Y6 of COARSE-THEORY.md Part C (toy chains). Predictions are in the paper.
Usage: c1_core.py [quick]   ('quick' = fewer student steps, for a smoke run)
"""
import sys, time, json
import numpy as np
import torch
import chain as C

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
STEPS = 400 if quick else 4000
torch.manual_seed(0)
T = C.Teacher(seed=0); T.load_state_dict(torch.load("teacher_gated_sym.pt", weights_only=True)); T.eval()
for p in T.parameters():
    p.requires_grad_(False)
X, _ = C.sample(3000, seed=1, sym=True)          # the training law (same seed as the teacher)
Xh, _ = C.sample(500, seed=3, sym=True)           # held-out law
fm = C.formula(T, X)
y, g = fm["y"], fm["g"]
with torch.no_grad():
    yh = T(Xh); gh = T.gates(Xh)
mu = fm["mu"]
res = {"mu (top 10)": [float(v) for v in mu[:10]], "D2 at k=0": float(0.5 * mu.sum())}
levels = C.strict_widths(fm)
res["strict-gap levels (width, modes)"] = [(k, m) for k, m in levels]


def full_d2(S, Xs, ys, gs):
    with torch.no_grad():
        return float(C.d2(ys, S(Xs, gs)))


# Y1
Y1 = {}
for k, modes in levels:
    if k > 8:
        continue
    target = float(0.5 * mu[k:].sum())
    S9, S10 = C.t9_student(T, fm, k), C.t10_student(T, fm, modes)
    with torch.no_grad():
        o9, o10 = S9(X, g), S10(X, g)
        c9 = (o9 - o9.mean(0)) @ C.PPERP; c10 = (o10 - o10.mean(0)) @ C.PPERP
    Y1[k] = {"T9 rel err": abs(full_d2(S9, X, y, g) - target) / max(target, 1e-300),
             "T10 rel err": abs(full_d2(S10, X, y, g) - target) / max(target, 1e-300),
             "outputs T9 vs T10": float((c9 - c10).abs().max() / c9.abs().max()),
             "formula D2": target, "fraction of k=0": target / res["D2 at k=0"]}
res["Y1"] = Y1

# Y5 nesting of the trained teacher's levels
E = fm["E"]; ks = [k for k, _ in levels if k < C.NY]
res["Y5 nesting error"] = max(float((E[:, :a] @ E[:, :a].T @ E[:, :b] @ E[:, :b].T - E[:, :a] @ E[:, :a].T).abs().max())
                              for a, b in zip(ks[:-1], ks[1:]))


def train(S, steps, lr, Xs, ys, gs, bs=512, seed=0):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(S.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        idx = torch.randint(len(Xs), (bs,), generator=gen)
        loss = C.d2(ys[idx], S(Xs[idx], gs[:, idx]))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return S


# Y2, Y4, Y6: random-start students at the first strict-gap widths <= 5
Y2 = {}
t0 = time.time()
for k, modes in levels:
    if k > 5:
        continue
    f_train = Y1[k]["formula D2"]
    S10 = C.t10_student(T, fm, modes)
    f_held = full_d2(S10, Xh, yh, gh)
    row = {"formula train": f_train, "formula held-out": f_held}
    for name, widths in [("kkk", (k, k, k)), ("k88", (k, 8, 8))]:
        best = None
        for seed in range(2):
            S = C.Student(*widths, seed=seed)
            train(S, STEPS, 3e-3, X, y, g, seed=seed)
            d_tr, d_he = full_d2(S, X, y, g), full_d2(S, Xh, yh, gh)
            if best is None or d_tr < best[0]:
                best = (d_tr, d_he)
        row[f"student {name} train / formula"] = best[0] / f_train
        row[f"student {name} held-out / formula held-out"] = best[1] / f_held
    Y2[k] = row
    print(k, row, f"{time.time() - t0:.0f}s", flush=True)
res["Y2/Y4/Y6"] = Y2

# Y3 local exactness at the second level
k, modes = levels[1]
S10 = C.t10_student(T, fm, modes)
loss = C.d2(y, S10(X, g)); grads = torch.autograd.grad(loss, list(S10.parameters()))
gn = float(torch.sqrt(sum((q ** 2).sum() for q in grads)))
Sr = C.Student(k, k, k, seed=5)
loss_r = C.d2(y, Sr(X, g)); gr = torch.autograd.grad(loss_r, list(Sr.parameters()))
gnr = float(torch.sqrt(sum((q ** 2).sum() for q in gr)))
before = full_d2(S10, X, y, g)
opt = torch.optim.Adam(S10.parameters(), 1e-4)
for it in range(300):
    l = C.d2(y, S10(X, g)); opt.zero_grad(); l.backward(); opt.step()
after = full_d2(S10, X, y, g)
res["Y3"] = {"k": k, "grad norm at formula": gn, "grad norm at a random start": gnr,
             "D2 before": before, "D2 after 300 full-batch Adam steps": after, "relative change": (after - before) / before}
print(json.dumps(res, indent=1, default=str))
json.dump(res, open("c1_core_quick.json" if quick else "c1_core.json", "w"), indent=1, default=str)
