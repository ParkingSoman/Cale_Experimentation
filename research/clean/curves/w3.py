"""Card W3: the Fisher write for KEPT members (k >= 2), unchanged from W2. Target = what the whole family gave the
368 4a readers; source = the kept members' own pooled outputs. Compared paired on dev-5000 with the Bayes write.
usage: w3.py curves|circles|both <n images>"""
import sys, json, numpy as np, torch
from inception import *
from family import Data
from p5_drop import fam_stats
from w_fisher import solve, real_kl, A_cols
CURVES = [330, 340, 324, 388, 343, 342, 385, 406, 379, 349]; CIRCLES = [462, 82, 300, 311, 446, 255, 352, 315, 363, 454, 323]
FAMS = {"curves": (CURVES, [330, 388, 385]), "circles": (CIRCLES, [82, 323]),
        "both": (CURVES + CIRCLES, [330, 388, 385, 82, 323])}


class Keep:
    def __init__(s, m, keep): s.m, s.keep = m, keep
    def pre(s, ins): return pre_3b(s.m, ins[0])[:, s.keep]


def bayes(m, fam, keep, st):
    Ki = [fam.index(c) for c in keep]; Ri = [i for i in range(len(fam)) if i not in Ki]; cols, cbs = [], []
    for j, A in enumerate(A_cols(m, fam)):
        mu, C = st[1] if j == 3 else st[0]; A = A.cpu().double().numpy()
        B = np.linalg.solve(C[np.ix_(Ki, Ki)], C[np.ix_(Ki, Ri)]); est = A[:, Ri] @ B.T
        cols.append(torch.tensor(A[:, Ki] + est, dtype=torch.float32, device=DEV))
        cbs.append(torch.tensor(A[:, Ri] @ mu[Ri] - est @ mu[Ki], dtype=torch.float32, device=DEV))
    return cols, cbs


if __name__ == "__main__":
    name, n = sys.argv[1], int(sys.argv[2]); fam, keep = FAMS[name]
    m = load_model(); D = Data(seed=0); fk = Keep(m, keep)
    idx = np.sort(np.random.default_rng(12).choice(D.tr, n, replace=False))
    cols, cbs, cov = solve(m, fam, D, fk, idx, 10, len(keep), bs=50, save=f"cache/fisher_ne_w3_{name}_{n}.pt")
    B = {"fisher": (fk, cols, cbs), "bayes": (fk, *bayes(m, fam, keep, fam_stats(m, fam, D)))}
    K = real_kl(m, fam, D, D.dv, B)
    for k, v in K.items(): print(f"{name} keep {keep} {k:6s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})", flush=True)
    d = K["fisher"] - K["bayes"]; print(f"  paired fisher - bayes: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
    np.savez(f"cache/w3_{name}_{n}.npz", **K)
    json.dump({k: float(v.mean()) for k, v in K.items()} | dict(n=n, keep=keep), open(f"../../../results/curves-w3-{name}-{n}.json", "w"), indent=1)
