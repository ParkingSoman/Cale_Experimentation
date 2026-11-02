"""Derived k-member build (closed form): keep members K (chosen by symmetry: equally spaced in orientation), estimate the
removed members' reader input from the kept channels by multivariate regression on analysis-image statistics.
  zero  : kept members with their own columns, removed members dropped
  const : + removed members' reader-side mean
  bayes : kept columns += regression of removed contribution on kept channels (+ intercept)
  vmk   : bayes estimate scaled per reader to keep the removed contribution's variance (/ multiple correlation)
usage: d12_k3.py 330,388,385 [--eval dev|sealed]"""
import sys, json, numpy as np, torch
from inception import *
from family import Family, Data, evaluate, IN4A, pool3

def member_stats(m, fam, D, n=1000, seed=11):
    idx = np.sort(np.random.default_rng(seed).choice(D.tr, n, replace=False)); acc = [[np.zeros(10), np.zeros((10, 10)), 0] for _ in range(2)]
    with torch.no_grad():
        for i in range(0, n, 100):
            z = F.relu(pre_3b(m, to_3a(m, prep(D.X[idx[i:i + 100]])))[:, fam]); p1 = pool3(z)
            p2 = F.max_pool2d(F.pad(p1, (1, 1, 1, 1), value=-1e9), 3, 1)
            for a, p in zip(acc, [p1, p2]):
                x = p.permute(0, 2, 3, 1).reshape(-1, 10).cpu().numpy().astype(np.float64); a[0] += x.sum(0); a[1] += x.T @ x; a[2] += len(x)
    return [(a[0] / a[2], a[1] / a[2] - np.outer(a[0] / a[2], a[0] / a[2])) for a in acc]

def build(m, fam, keep, st, mode):
    K = [fam.index(k) for k in keep]; Rm = [i for i in range(10) if i not in K]; cols, cbs = [], []
    for j, name in enumerate(IN4A):
        mu, C = st[1] if j == 3 else st[0]
        A = getattr(m, name).weight.detach().cpu()[:, fam, 0, 0].double().numpy(); AK, AR = A[:, K], A[:, Rm]
        if mode == "zero": col, cb = AK, np.zeros(A.shape[0])
        elif mode == "const": col, cb = AK, AR @ mu[Rm]
        else:
            B = np.linalg.solve(C[np.ix_(K, K)], C[np.ix_(K, Rm)])          # (k, r): removed ~ B^T kept
            est = AR @ B.T                                                  # (out, k): reader coefficients on kept
            if mode == "vmk":
                varR = np.einsum("oi,ij,oj->o", AR, C[np.ix_(Rm, Rm)], AR); varE = np.einsum("oi,ij,oj->o", est, C[np.ix_(K, K)], est)
                est = est * np.sqrt(varR / np.maximum(varE, 1e-9))[:, None]
            col = AK + est; cb = AR @ mu[Rm] - est @ mu[K]
        cols.append(torch.tensor(col, dtype=torch.float32)); cbs.append(torch.tensor(cb, dtype=torch.float32).to(DEV))
    return cols, cbs

if __name__ == "__main__":
    keep = [int(t) for t in sys.argv[1].split(",")]; split_ = sys.argv[3] if len(sys.argv) > 3 else "dev"
    m = load_model(); D = Data(seed=0, ndev=1000 if split_ == "dev" else 0); fam = list(np.load("cache/s1_fine.npz")["fam"])
    st = member_stats(m, fam, D); out = {}
    fm = Family(m, fam, len(keep), init="member:" + ",".join(map(str, keep))).to(DEV)
    for mode in ["zero", "const", "bayes", "vmk"]:
        cols, cbs = build(m, fam, keep, st, mode)
        with torch.no_grad():
            for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
        fm.colbias = cbs; r = evaluate(m, fm, D, D.dv if split_ == "dev" else D.se); out[mode] = r
        print(f"keep {keep} write {mode:5s}: {split_} KL {r[0]:.4f} (se {r[1]:.4f}) top-1 {r[2]:.4f}", flush=True)
    json.dump(out, open(f"../../../results/curves-derived-keep{'-'.join(map(str, keep))}-{split_}.json", "w"), indent=1)
