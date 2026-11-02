"""D2 look: in the toy, what is a narrow neuron in terms of the wide net's neurons? Seed ceiling, label content, parts,
form, lone-part response, rate law, co-firing rank, shared class vote. Description only."""
import os
import numpy as np
from scipy.special import erf
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf", "toy"); gl = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
d = np.load(os.path.join(T, "data.npz")); X, y, c = d["Xte"].astype(np.float64), d["yte"], d["cte"]
def net(k): z = np.load(os.path.join(T, k + ".npz")); return {n: z[n].astype(np.float64) for n in z.files}
def Z(n): return X @ n["W1"].T + n["b1"]
zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)
W = {s: net(f"gelu_256_s{s}") for s in range(3)}; N = {s: net(f"gelu_16_s{s}") for s in range(3)}
HW = {s: gl(Z(W[s])) for s in W}; HN = {s: gl(Z(N[s])) for s in N}
def best(A, B): C = zs(A).T @ zs(B) / len(A); return C.max(1)
print("seed ceiling (median best-match corr): narrow s0->narrow s1 %.3f  wide s0->wide s1 %.3f  narrow s0->wide s0 %.3f  narrow s0->wide s1 %.3f" % (
    np.median(best(HN[0], HN[1])), np.median(best(HW[0], HW[1])), np.median(best(HN[0], HW[0])), np.median(best(HN[0], HW[1]))))
def eta(h, lab):   # share of variance explained by a label (correlation ratio)
    m = np.array([h[lab == k].mean() for k in np.unique(lab)]); n_ = np.array([(lab == k).sum() for k in np.unique(lab)]); return float((n_ * (m - h.mean()) ** 2).sum() / (h.var() * len(h)))
sgn = np.sign(X @ np.zeros(X.shape[1]) + 1)   # placeholder
# sign of the shown pattern is hidden; use (class, sign of projection on its template) as the full label
proj = np.einsum("nd,nd->n", X, d["Ac"][c] + d["Bf"][c, y % 8]); s_ = (proj > 0).astype(int)
for nm, H in (("narrow s0", HN[0]), ("wide s0", HW[0])):
    e = np.array([[eta(h, c * 2 + s_), eta(h, y * 2 + s_)] for h in H.T]); alive = H.std(0) > 1e-3
    print(f"{nm}: variance explained by (coarse,sign) {np.median(e[alive, 0]):.2f}, by (fine,sign) {np.median(e[alive, 1]):.2f}; extra from fine {np.median(e[alive, 1] - e[alive, 0]):.2f}")
Cw = zs(HW[0]); out = []
for j in range(16):
    t = HN[0][:, j]; tz = zs(t[:, None])[:, 0]; cc = Cw.T @ tz / len(t); P = np.argsort(-cc)[:10]
    Zw = Z(W[0]); zp = zs(Zw[:, P]); sg = np.sign([np.corrcoef(Zw[:, p], t)[0, 1] for p in P]); S = (zp * sg).sum(1); hs = HW[0][:, P] / HW[0][:, P].std(0)
    cor = lambda v: float(np.corrcoef(v, t)[0, 1]); TH = np.arange(0, 25, 1.0); ct = [cor(gl(S - th)) for th in TH]; th = TH[int(np.argmax(ct))]
    on = zp > 2; one = on.sum(1) == 1; many = on.sum(1) >= 3
    pc = (zs(HW[0][:, P]).T @ zs(HW[0][:, P]) / len(t))[np.triu_indices(10, 1)].mean()
    co = Cw.T @ Cw[:, P[0]] / len(t); co[P[0]] = -9; rk = np.median(np.argsort(np.argsort(-co))[P[1:]])
    V = W[0]["W2"][:, P].T; V = V - V.mean(1, keepdims=True); V /= np.linalg.norm(V, axis=1, keepdims=True); vc = (V @ V.T)[np.triu_indices(10, 1)].mean()
    coarse_of_parts = [int(np.argmax([HW[0][c == k, p].mean() for k in range(8)])) for p in P]
    out.append((j, cc[P[0]], cor(hs.sum(1)), cor(hs.max(1)), cor(hs[:, 0]), max(ct), th, (S > th).mean() / (Zw[:, P] > 0).mean(), tz[one].mean() if one.sum() > 20 else np.nan,
                tz[many].mean() if many.sum() > 20 else np.nan, pc, rk, vc, len(set(coarse_of_parts)), eta(t, c * 2 + s_)))
print(f"{'n':>3s} {'best':>5s} {'sum':>5s} {'max':>5s} {'1prt':>5s} {'cons':>5s} {'th*':>4s} {'rate':>5s} {'1on':>6s} {'3+on':>5s} {'pcorr':>6s} {'rank':>5s} {'vote':>6s} {'#coarse':>7s} {'eta_c':>6s}")
for o in out: print(f"{o[0]:3d} " + " ".join(f"{v:5.2f}" if isinstance(v, float) else f"{v:5}" for v in o[1:]))
A = np.array([o[1:] for o in out], float); print("median: " + " ".join(f"{v:.2f}" for v in np.nanmedian(A, 0)))
