# serves: T16
"""Toy check Y14 of COARSE-THEORY.md Part C (self-gated toy chain): why a free-read student beats the T11 formula at
width 1 (Theorem T16 part 6). Prediction in the paper."""
import json
import numpy as np
import torch
import chain as C

torch.manual_seed(0)
T = C.Teacher(seed=0, variant="selfgate")
T.load_state_dict(torch.load("teacher_selfgate_sym.pt", weights_only=True)); T.eval()
for p in T.parameters():
    p.requires_grad_(False)
X, _ = C.sample(3000, seed=1, sym=True)
fm = C.formula(T, X)
q = C.t11_quantities(fm)
y, g = fm["y"], fm["g"]
S = C.SelfStudent(T, 1, seed=0, faithful=False)
opt = torch.optim.Adam(S.parameters(), 3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
gen = torch.Generator().manual_seed(0)
for it in range(4000):
    idx = torch.randint(len(X), (512,), generator=gen)
    loss = C.d2(y[idx], S(X[idx]))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
with torch.no_grad():
    yb, gb = S(X, parts=True)
    d2 = float(C.d2(y, yb))
    Ks = S.K[0].reshape(-1)
    M = T.members().reshape(C.N, -1)
    coef = torch.linalg.lstsq(M.T, Ks[:, None]).solution[:, 0]
    in_span = float((M.T @ coef).norm() ** 2 / Ks.norm() ** 2)
    F = np.fft.fft(coef.numpy())
    energy = {n: float(abs(F[n]) ** 2 + (abs(F[C.N - n]) ** 2 if 0 < n < C.N // 2 else 0)) for n in range(C.N // 2 + 1)}
    tot = sum(energy.values())
    energy = {n: e / tot for n, e in energy.items()}
    gate_rel = [float((gb[l] - g[l]).norm() / g[l].norm()) for l in range(3)]
    gate_corr = [float(np.corrcoef(gb[l].reshape(-1).numpy(), g[l].reshape(-1).numpy())[0, 1]) for l in range(3)]
lam = {int(n): float(fm["lam"][fm["labels"] == n][0]) for n in sorted(set(fm["labels"].tolist()))}
t11 = float(0.5 * q["nu"].sum())
res = {"student D2": d2, "T11 value": t11, "T12 floor": float(0.5 * fm["mu"][1:].sum()), "student / T11": d2 / t11,
       "fraction of kernel norm^2 in member span": in_span,
       "member-mode energy of the in-span part (n: share)": energy, "teacher lambda_n": lam,
       "gate relative difference (3 gates)": gate_rel, "gate correlation with teacher (3 gates)": gate_corr}
print(json.dumps(res, indent=1))
json.dump(res, open("c5_why.json", "w"), indent=1)
