# serves: DQ19
"""Claim form: log sbar_c[i] = alpha_c + beta_i + f(lca(c,i)) + sum_g a_cg M_ig   (off diagonal)
tree part = family position; M_ig = column i's membership in side group g (shared); a_cg = row c's link to group g (row numbers).
Fitted on rows A1 (half of A); scored on rows A2 (other half of A): alpha_c and a_cg refitted per held-out row, all else frozen."""
import sys, json, argparse
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--tree", default="/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3/t_avg_cos.json")
ap.add_argument("--G", type=int, nargs="+", default=[0, 2, 4, 8, 14])
ap.add_argument("--ridge", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--outer", type=int, default=6)
a = ap.parse_args()
sb, meta, lab = load()
K, C = sb.shape
A = splitA(K)
rng = np.random.default_rng(a.seed)
pa = rng.permutation(A)
A1, A2 = np.sort(pa[:len(A) // 2]), np.sort(pa[len(A) // 2:])
tree = json.load(open(a.tree))


def lca_nested(tree):
    lca = -np.ones((C, C), dtype=np.int64); cnt = [0]
    def rec(n):
        if isinstance(n, int): return [n]
        parts = [rec(x) for x in n]; t = cnt[0]; cnt[0] += 1
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                lca[np.ix_(parts[i], parts[j])] = t; lca[np.ix_(parts[j], parts[i])] = t
        return [x for p in parts for x in p]
    rec(tree); np.fill_diagonal(lca, 0)
    return lca, cnt[0]


lca, nn = lca_nested(tree)
Wfull = (sb * (~torch.eye(K, C, dtype=torch.bool))).numpy()
Lfull = sb.log().numpy()


def fit(rows, G, Mi=None):
    """fit alpha, beta, f, M, a on `rows`"""
    W = Wfull[rows]; L = Lfull[rows]; idx = lca[rows].reshape(-1)
    n = len(rows)
    beta = np.zeros(C); f = np.zeros(nn); al = np.zeros(n)
    M = np.random.default_rng(0).normal(size=(C, G)) * 0.1 if G else np.zeros((C, 0))
    Acoef = np.zeros((n, G))
    for outer in range(a.outer if G else 1):
        fv = f[idx].reshape(n, C)
        for _ in range(30):   # alpha, beta, f given side part
            side = Acoef @ M.T
            al = (W * (L - beta[None] - fv - side)).sum(1) / W.sum(1)
            beta = (W * (L - al[:, None] - fv - side)).sum(0) / np.maximum(W.sum(0), 1e-300)
            r = L - al[:, None] - beta[None] - side
            num = np.bincount(idx, (W * r).reshape(-1), nn); den = np.bincount(idx, W.reshape(-1), nn)
            f = num / np.maximum(den, 1e-300); fv = f[idx].reshape(n, C)
        if not G: break
        T = L - beta[None] - fv     # = alpha_c + side
        for _ in range(10):         # ALS for alpha_c, a_c, M
            X = np.hstack([np.ones((C, 1)), M])
            for c in range(n):
                Xw = X * W[c][:, None]
                sol = np.linalg.solve(X.T @ Xw + a.ridge * np.eye(G + 1) * W[c].sum(), Xw.T @ T[c])
                al[c], Acoef[c] = sol[0], sol[1:]
            Tm = T - al[:, None]
            for i in range(C):
                w = W[:, i]
                M[i] = np.linalg.solve(Acoef.T @ (Acoef * w[:, None]) + a.ridge * np.eye(G) * max(w.sum(), 1e-12), (Acoef * w[:, None]).T @ Tm[:, i])
    return dict(beta=beta, f=f, M=M, al=al, Acoef=Acoef)


def score(rows, P, G, refit=True):
    W = Wfull[rows]; L = Lfull[rows]; idx = lca[rows].reshape(-1)
    fv = P["f"][idx].reshape(len(rows), C)
    T = L - P["beta"][None] - fv
    X = np.hstack([np.ones((C, 1)), P["M"]])
    res = np.zeros_like(T)
    for c in range(len(rows)):
        Xw = X * W[c][:, None]
        sol = np.linalg.solve(X.T @ Xw + a.ridge * np.eye(G + 1) * W[c].sum(), Xw.T @ T[c])
        res[c] = T[c] - X @ sol
    return float(np.sqrt((W * res ** 2).sum() / W.sum())), float((W * (np.abs(res) <= np.log(3))).sum() / W.sum())


def score_cols(rows, P, G, seed=3):
    """row numbers fitted on a random half of the columns, scored on the other half"""
    W = Wfull[rows]; L = Lfull[rows]; idx = lca[rows].reshape(-1)
    fv = P["f"][idx].reshape(len(rows), C)
    T = L - P["beta"][None] - fv
    X = np.hstack([np.ones((C, 1)), P["M"]])
    cm = np.random.default_rng(seed).random(C) < 0.5
    num = 0.0; den = 0.0; w3 = 0.0
    for fit_m in (cm, ~cm):
        for c in range(len(rows)):
            wf = W[c] * fit_m
            Xw = X * wf[:, None]
            sol = np.linalg.solve(X.T @ Xw + a.ridge * np.eye(G + 1) * wf.sum(), Xw.T @ T[c])
            r = T[c] - X @ sol
            ws = W[c] * (~fit_m)
            num += (ws * r ** 2).sum(); den += ws.sum(); w3 += (ws * (np.abs(r) <= np.log(3))).sum()
    return float(np.sqrt(num / den)), float(w3 / den)


def score_tied(rows, P, B, G, F=None):
    W = Wfull[rows]; L = Lfull[rows]; idx = lca[rows].reshape(-1)
    fv = P["f"][idx].reshape(len(rows), C)
    side = ((P["M"][rows] if F is None else F[rows]) @ B.T) @ P["M"].T
    T = L - P["beta"][None] - fv - side
    al = (W * T).sum(1) / W.sum(1)
    res = T - al[:, None]
    return float(np.sqrt((W * res ** 2).sum() / W.sum()))


out = []
for G in a.G:
    P = fit(A1, G)
    tr = score(A1, P, G); te = score(A2, P, G)
    cs = score_cols(A2, P, G)
    d = dict(G=G, colheld_A2=cs[0], colheld_within3=cs[1], train_A1=tr[0], heldout_A2=te[0], within3_A2=te[1])
    print(json.dumps(d), flush=True)
    if G:
        lam = a.ridge * 10
        MA = P["M"][A1]
        B = np.linalg.solve(MA.T @ MA + lam * np.eye(G), MA.T @ P["Acoef"]).T   # a_c ~ B M_c
        d["tied_train_A1"] = score_tied(A1, P, B, G); d["tied_heldout_A2"] = score_tied(A2, P, B, G)
        Wc = meta["W"].astype(np.float64); Wc = Wc - Wc.mean(0)
        U_, S_, Vt_ = np.linalg.svd(Wc, full_matrices=False)
        for dd in (32, 128):
            F = U_[:, :dd] * S_[:dd]; F = F / F.std()
            FA = F[A1]
            for lamf in (1.0, 30.0):
                Bf = np.linalg.solve(FA.T @ FA + lamf * np.eye(dd), FA.T @ P["Acoef"]).T
                d[f"W{dd}_lam{lamf}_heldout"] = score_tied(A2, P, Bf, G, F)
        d["corr_a_vs_BM"] = float(np.corrcoef((MA @ B.T).ravel(), P["Acoef"].ravel())[0, 1])
    print(json.dumps(d), flush=True)
    out.append(d)
    if G == a.G[-1]:
        np.savez(sys.path[0] + f"/side_G{G}.npz", M=P["M"], beta=P["beta"], f=P["f"], A1=A1, A2=A2)
json.dump(out, open(sys.path[0] + "/side_results.json", "w"), indent=1)
