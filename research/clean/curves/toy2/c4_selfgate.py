# serves: T11
# serves: T12
# serves: D6
"""Self-gated toy chain checks Y10-Y13 of COARSE-THEORY.md Part C (self-gated toy chain). Predictions are in the paper.
Usage: c4_selfgate.py lambda   (teacher's lambda_n and the Y12 case only; no students)
       c4_selfgate.py [quick]  (all checks; 'quick' = fewer student steps)
"""
import sys, json, time
import numpy as np
import torch
import chain as C

mode = sys.argv[1] if len(sys.argv) > 1 else "full"
STEPS = 400 if mode == "quick" else 4000
torch.manual_seed(0)
T = C.Teacher(seed=0, variant="selfgate")
T.load_state_dict(torch.load("teacher_selfgate_sym.pt", weights_only=True)); T.eval()
for p in T.parameters():
    p.requires_grad_(False)
X, _ = C.sample(3000, seed=1, sym=True)
fm = C.formula(T, X)
q = C.t11_quantities(fm)
y, g = fm["y"], fm["g"]
lam, labels = fm["lam"], fm["labels"]
lam_mode = {int(n): float(lam[labels == n][0]) for n in sorted(set(labels.tolist()))}
mu, nu = fm["mu"], q["nu"]
res = {"lambda by mode": lam_mode, "sigma (edge column sums)": [float(w[:, 0].sum()) for w in fm["w"]],
       "gate zero fraction": [float((g[l] == 0).double().mean()) for l in range(3)],
       "ga, gb": [T.ga.tolist(), T.gb.tolist()]}
lam_sorted = np.sort(lam.numpy())[::-1]
lam0 = lam_mode[0]
res["Y12 case: widths k <= 8 where lambda_0 is among the k largest"] = [
    k for k in range(1, 9) if lam0 >= lam_sorted[k - 1] - 1e-12 * lam_sorted[0]]
res["rank of lambda_0 among the 8 modes (1 = largest)"] = int(1 + np.sum(lam.numpy() > lam0 + 1e-12 * lam_sorted[0]))
if mode == "lambda":
    print(json.dumps(res, indent=1))
    sys.exit(0)


def full_d2(f):
    with torch.no_grad():
        return float(C.d2(y, f(X)))


def t11_value(k):
    return float(0.5 * nu[k - 1:].sum())


def t9_value(k):
    return float(0.5 * mu[k:].sum())


def gate_err(gb):
    return float((gb - g).abs().max() / g.abs().max())


def train(S, steps, lr, seed=0, cosine=True, log=None):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in S.parameters() if p.requires_grad], lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cosine else None
    for it in range(steps):
        idx = torch.randint(len(X), (512,), generator=gen)
        loss = C.d2(y[idx], S(X[idx]))
        opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()
        if log is not None and (it + 1) % max(steps // 8, 1) == 0:
            log.append(full_d2(S))
    return S


# Y10
lam_other = np.sort(np.array([lam_mode[n] for n in labels.tolist()])[labels != 0])[::-1]
res["Y10 nu vs lambda (n != 0), relative"] = float(np.abs(nu[:C.N - 1].numpy() - lam_other).max() / lam_other.max())
Y10 = {}
for k in range(1, C.N + 1):
    S = C.t11_student(T, fm, q, k)
    with torch.no_grad():
        yb, gb = S(X, parts=True)
    v, f = float(C.d2(y, yb)), t11_value(k)
    Y10[k] = {"gate err": gate_err(gb), "D2": v, "T11 value": f, "rel err": abs(v - f) / max(f, 1e-300)}
res["Y10"] = Y10
print("Y10", json.dumps(Y10), flush=True)

# Y12 and Y12b
Y12 = {}
for k in range(1, C.N):
    top = lam_sorted[:k].sum()
    other = lam0 + (lam_other[:k - 1].sum() if k > 1 else 0.0)
    cost = t11_value(k) - t9_value(k)
    Y12[k] = {"T11 value": t11_value(k), "T9 value": t9_value(k), "cost": cost,
              "cost - formula (relative to T9 value at k=0)": (cost - 0.5 * (top - other)) / float(0.5 * mu.sum())}
    if k <= 4:
        f = C.t11_1_build(T, fm, q, k)
        with torch.no_grad():
            yb, gb = f(X)
        Y12[k]["Y12b gate err"] = gate_err(gb)
        Y12[k]["Y12b rel err to T9 value"] = abs(float(C.d2(y, yb)) - t9_value(k)) / t9_value(k)
res["Y12"] = Y12
print("Y12", json.dumps(Y12), flush=True)

# Y11 faithful students
Y11 = {}
t0 = time.time()
for k in (1, 2, 3, 4):
    row = {"T11 value": t11_value(k), "T9 value": t9_value(k)}
    rs = []
    for seed in range(2):
        S = train(C.SelfStudent(T, k, seed=seed), STEPS, 3e-3, seed=seed)
        rs.append(full_d2(S) / t11_value(k))
    row["random start / T11 (2 seeds)"] = rs
    S = C.t11_student(T, fm, q, k)
    d0 = full_d2(S)
    train(S, 300, 1e-4, cosine=False)
    row["from T11 start: relative change after 300 small steps"] = (full_d2(S) - d0) / d0
    Y11[k] = row
    print("Y11", k, row, f"{time.time() - t0:.0f}s", flush=True)
res["Y11"] = Y11


# Y13 free-read students
def free_from_t11(k):
    Sf = C.t11_student(T, fm, q, k)
    S = C.SelfStudent(T, k, faithful=False)
    with torch.no_grad():
        S.K.copy_(Sf.kernels())
        w1, w2 = Sf.edges()
        S.w1.copy_(w1); S.w2.copy_(w2); S.H.copy_(Sf.H); S.c.copy_(Sf.c)
        S.s.zero_(); S.s[0, 0] = 1.0; S.s[1, 0] = 1.0; S.s[2, 0] = Sf.sig[0]
    return S


Y13 = {}
for k in (1, 2, 3):
    row = {"T11 value": t11_value(k), "T12 bound": t9_value(k)}
    S = free_from_t11(k)
    row["free-read copy of T11 / T11 (should be 1)"] = full_d2(S) / t11_value(k)
    log = []
    train(S, 300, 1e-4, cosine=False, log=log)
    row["from T11, 300 steps lr 1e-4: D2 / T11 (log)"] = [v / t11_value(k) for v in log]
    log = []
    train(S, STEPS, 1e-3, log=log)
    row["then STEPS lr 1e-3: D2 / T11 (log)"] = [v / t11_value(k) for v in log]
    row["final / T12 bound"] = full_d2(S) / t9_value(k)
    rs = []
    for seed in range(2):
        Sr = train(C.SelfStudent(T, k, seed=seed, faithful=False), STEPS, 3e-3, seed=seed)
        rs.append(full_d2(Sr))
    row["random start: D2 / T11"] = [v / t11_value(k) for v in rs]
    row["random start: D2 / T12 bound"] = [v / t9_value(k) for v in rs]
    Y13[k] = row
    print("Y13", k, row, f"{time.time() - t0:.0f}s", flush=True)
res["Y13"] = Y13
json.dump(res, open("c4_selfgate_quick.json" if mode == "quick" else "c4_selfgate.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
