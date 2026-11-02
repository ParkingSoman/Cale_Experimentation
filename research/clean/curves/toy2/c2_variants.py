# serves: T9
# serves: T7
"""Toy chain checks Y7 (H-symN broken by sampling) and Y8 (H-gate broken in the first family) of COARSE-THEORY.md
Part C (toy chains). Predictions are in the paper.
Usage: c2_variants.py [quick]
"""
import sys, json
import numpy as np
import torch
import chain as C

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
STEPS = 400 if quick else 4000
res = {}


def full_d2(S, Xs, ys, gs):
    with torch.no_grad():
        return float(C.d2(ys, S(Xs, gs)))


def train(S, steps, lr, Xs, ys, gs, bs=512, seed=0, cosine=True):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(S.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cosine else None
    for it in range(steps):
        idx = torch.randint(len(Xs), (bs,), generator=gen)
        loss = C.d2(ys[idx], S(Xs[idx], gs[:, idx]))
        opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()
    return S


def load(variant):
    T = C.Teacher(seed=0, variant=variant)
    T.load_state_dict(torch.load(f"teacher_{variant}_sym.pt", weights_only=True)); T.eval()
    for p in T.parameters():
        p.requires_grad_(False)
    return T


# Y7: gated teacher, unsymmetrised law
T = load("gated")
Xs, _ = C.sample(3000, seed=1, sym=True)
levels_sym = C.strict_widths(C.formula(T, Xs))
Xa, _ = C.sample(24000, seed=4, sym=False)
fa = C.formula(T, Xa)
ya, ga = fa["y"], fa["g"]
Y7 = {}
for k, modes in levels_sym[:3]:
    target = float(0.5 * fa["mu"][k:].sum())
    S9, S10 = C.t9_student(T, fa, k), C.t10_student(T, fa, modes)
    d9, d10 = full_d2(S9, Xa, ya, ga), full_d2(S10, Xa, ya, ga)
    Y7[k] = {"T9 rel err": abs(d9 - target) / target, "harmonic excess / T9": (d10 - d9) / d9}
k = levels_sym[1][0]
Sr = train(C.Student(k, k, k, seed=0), STEPS, 3e-3, Xa, ya, ga)
Y7["student / T9 at k=%d" % k] = full_d2(Sr, Xa, ya, ga) / (0.5 * float(fa["mu"][k:].sum()))
off = fa["F"].T @ fa["Chat"] @ fa["F"]
Y7["Chat off-diagonal in Fourier basis (relative)"] = float((off - torch.diag(torch.diag(off))).abs().max() / off.abs().max())
res["Y7"] = Y7
print("Y7", Y7, flush=True)

# Y8: relu1 teacher (each first-family member has its own ReLU)
T = load("relu1")
X, _ = C.sample(3000, seed=1, sym=True)
fm = C.formula(T, X)
y, g = fm["y"], fm["g"]
Y8 = {}
for k, modes in C.strict_widths(fm)[:2]:
    what = float(0.5 * fm["mu"][k:].sum())                       # T9 value over reads of member OUTPUTS (not a build)
    Uk = fm["E"][:, :k]; Lam = Uk.T @ C.PPERP @ fm["A"]
    naive = C.Student(k, k, k, relu1=True)
    with torch.no_grad():
        naive.K.copy_(torch.einsum("ai,ipd->apd", Lam, T.members().reshape(C.N, C.PN, C.D)))
        naive.w1.copy_(torch.eye(k)); naive.w2.copy_(torch.eye(k)); naive.H.copy_(Uk)
        o = naive(X, g); naive.c.copy_(naive.c + (y - o).mean(0))
    d_naive = full_d2(naive, X, y, g)
    train(naive, STEPS // 2, 1e-4, X, y, g, cosine=False)
    d_naive_tr = full_d2(naive, X, y, g)
    best = min(full_d2(train(C.Student(k, k, k, seed=s, relu1=True), STEPS, 3e-3, X, y, g, seed=s), X, y, g)
               for s in range(2))
    Y8[k] = {"WHAT optimum (reads of outputs, not buildable)": what, "read-then-one-ReLU build": d_naive,
             "after training from it": d_naive_tr, "relative drop from training": (d_naive - d_naive_tr) / d_naive,
             "best random-start student": best, "random student / read-then-ReLU": best / d_naive,
             "random student / WHAT optimum": best / what}
    print("Y8", k, Y8[k], flush=True)
res["Y8"] = Y8
json.dump(res, open("c2_variants_quick.json" if quick else "c2_variants.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
