"""T4b: Theorem 10b child weights against simple rules. Cached vectors only, no images. Prediction in CURVES-LOG.md."""
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=150)
t4, t2, A = np.load("cache/t4.npz"), np.load("cache/t2.npz"), np.load("cache/t1_C.npz")["A"]
trained = {"twin L": np.array([.105, .15, .043, .125, .153, .147, .079, .071, .07, .057]),     # logs/t4_moved.log
           "twin R": np.array([.055, .156, .081, .142, .154, .195, .121, -.006, .067, .025])}
nrm = lambda v: v / np.abs(v).sum()
low = np.argsort(np.diag(A.T @ A))[:3]; keep = np.setdiff1d(np.arange(10), low)
d3 = np.zeros(10); d3[keep] = 1
d3mu = np.zeros(10); d3mu[keep] = t4["mum"][keep]
rules = {"uniform": np.ones(10), "drop-3 equal": d3, "drop-3, rest by mean": d3mu, "diag(Phi)": np.diag(t2["Phi"]),
         "lam_I (T2)": t2["lam_I"], "lam_F (T2)": t2["lam_F"], "free lam (T4)": t4["free"], "Theorem 10b (T4)": t4["lam"]}
sup = np.where(t4["lam"] > .01)[0]
pr = lambda a, b: np.nan if a.std() < 1e-12 or b.std() < 1e-12 else np.corrcoef(a, b)[0, 1]
cs = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("drop-3 members (lowest diag A^T A):", low, "| support of Theorem 10b:", sup)
print(f"{'rule':24s}" + "".join(f"{t + ' ' + s:>17s}" for t in trained for s in ("r10", "cos", "r-sup")))
for n, v in rules.items():
    v = nrm(v); print(f"{n:24s}" + "".join(f"{f(v[ix], lt[ix]):17.2f}" for lt in trained.values()
                                          for f, ix in ((pr, slice(None)), (cs, slice(None)), (pr, sup))))
a, b = trained.values(); print(f"{'twin vs twin':24s}{pr(a, b):17.2f}{cs(a, b):17.2f}{pr(a[sup], b[sup]):17.2f}")
for r, k in ((.82, 10), (.5, len(sup))):
    z = np.arctanh(r); print(f"r = {r} with {k} points: 95% interval {np.tanh(z - 1.96 / np.sqrt(k - 3)):.2f} .. {np.tanh(z + 1.96 / np.sqrt(k - 3)):.2f}")
