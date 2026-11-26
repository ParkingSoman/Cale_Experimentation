"""theory agent, step 5: ceiling of ANY linear rule over Base block-0 outputs (after GELU, all positions).

Ridge of each Small neuron on all 4,096 Base neurons: fit on fit rows, scored on test rows (grams.npz).
lambda picked on even Small ids, reported on odd ids. Also top-k principal directions of Base (fit), and the same
ceiling for predicting a held-out BASE neuron from the other 4,095 (noise floor: how well Base predicts itself).
"""
import json
import numpy as np

R = "../results/pair/"
g = np.load(R + "grams.npz")
CF, CT = g["fit_Cbb"].astype(np.float64), g["test_Cbb"].astype(np.float64)
cF, cT, vT = g["fit_Cbs"].astype(np.float64), g["test_Cbs"].astype(np.float64), g["test_vs"]
e, U = np.linalg.eigh(CF); e, U = e[::-1], U[:, ::-1]
Uc = U.T @ cF
UCTU = U.T @ CT @ U
UcT = U.T @ cT
tr = e.sum() / len(e)
even, odd = np.arange(0, 3072, 2), np.arange(1, 3072, 2)
out = {}


def score(f):  # f: filter over eigen-directions -> weights in eigen coords
    W = f[:, None] * Uc
    err = vT - 2 * (W * UcT).sum(0) + (W * (UCTU @ W)).sum(0)
    return 1 - err / vT


best = None
for lam in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
    r = score(1 / (e + lam * tr))
    out[f"ridge_{lam}"] = dict(even=float(np.median(r[even])), odd=float(np.median(r[odd])))
    print("ridge", lam, out[f"ridge_{lam}"], flush=True)
    if best is None or np.median(r[even]) > best[0]:
        best = (np.median(r[even]), lam, r)
lam = best[1]; rbest = best[2]
print(f"CEILING (lam {lam} chosen on even): odd median test R^2 {np.median(rbest[odd]):.3f}, q10 {np.quantile(rbest[odd],.1):.3f} q90 {np.quantile(rbest[odd],.9):.3f}")
for k in (16, 64, 256, 768, 1024, 2048, 4096):
    f = np.zeros(4096); f[:k] = 1 / (e[:k] + lam * tr)
    r = score(f); out[f"pca_{k}"] = float(np.median(r[odd])); print("top-k PCs", k, round(out[f"pca_{k}"], 3), flush=True)
np.save(R + "theory/ceiling_r2.npy", rbest.astype(np.float32))
# noise floor: each Base neuron j predicted from the other 4095 (ridge, same lambda), on 400 random Base neurons
rng = np.random.default_rng(0); J = rng.choice(4096, 400, replace=False)
rb = []
for j in J:
    o = np.setdiff1d(np.arange(4096), [j])
    A = CF[np.ix_(o, o)] + lam * tr * np.eye(4095)
    w = np.linalg.solve(A, CF[o, j])
    err = CT[j, j] - 2 * w @ CT[o, j] + w @ CT[np.ix_(o, o)] @ w
    rb.append(1 - err / CT[j, j])
    if len(rb) == 40:
        break
out["base_from_other_base_median"] = float(np.median(rb)); out["lam"] = lam
out["ceiling_odd_median"] = float(np.median(rbest[odd]))
print("noise floor: Base neuron from other 4095 Base, median test R^2", round(np.median(rb), 3), "(n=40)")
json.dump(out, open(R + "theory/ceiling.json", "w"), indent=1)
