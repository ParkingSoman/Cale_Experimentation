"""Write laws for a coarse channel c replacing the family (closed form from analysis-image statistics, no knob).
Reader b originally receives I_b = sum_theta col_b,theta * m_theta (m = pooled member outputs; pool branch: pooled
again). New input: col_b,c * m_c + bias_b.
  bayes : col = Cov(I_b, m_c)/Var(m_c)                      (regression: shrinks variance)
  vmg   : bayes x one global factor restoring sum_b Var(I_b) (variance kept in total)
  vmr   : bayes / rho_b per reader (each reader's variance kept)
bias_b always matches the mean of I_b.
usage: d4_write.py <read: ckpt tag | oracle0 | derived2 tag>"""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, evaluate, pool3, IN4A
from derive2 import moved_sum_t


def stats(m, fm, fam, D, n=1000, seed=11):
    idx = np.sort(np.random.default_rng(seed).choice(D.tr, n, replace=False))
    acc = [dict(S=np.zeros(11), SS=np.zeros((11, 11)), N=0) for _ in range(2)]
    with torch.no_grad():
        for i in range(0, n, 100):
            ins = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i + 100]])))
            v = torch.cat([F.relu(pre_3b(m, ins[0])[:, fam]), F.relu(fm.pre(ins))], 1)
            p1 = pool3(v); p2 = F.max_pool2d(F.pad(p1, (1, 1, 1, 1), value=-1e9), 3, 1)
            for a, p in zip(acc, [p1, p2]):
                x = p.permute(0, 2, 3, 1).reshape(-1, 11).cpu().numpy().astype(np.float64)
                a["S"] += x.sum(0); a["SS"] += x.T @ x; a["N"] += len(x)
    out = []
    for a in acc:
        mu = a["S"] / a["N"]; C = a["SS"] / a["N"] - np.outer(mu, mu); out.append((mu, C))
    return out


def writes(m, fam, st):
    res = {k: ([], []) for k in ["bayes", "vmg", "vmr"]}; tot_I = 0.0; tot_B = 0.0; tmp = []
    for j, name in enumerate(IN4A):
        mu, C = st[1] if j == 3 else st[0]
        A = getattr(m, name).weight.detach().cpu()[:, fam, 0, 0].double().numpy()      # (out, 10)
        Cmm, Cmc, vc = C[:10, :10], C[:10, 10], C[10, 10]
        covIc = A @ Cmc; varI = np.einsum("oi,ij,oj->o", A, Cmm, A); meanI = A @ mu[:10]
        bay = covIc / vc; rho = covIc / np.sqrt(varI * vc + 1e-12)
        tot_I += varI.sum(); tot_B += (bay ** 2 * vc).sum()
        tmp.append((bay, rho, meanI, mu[10]))
    g = np.sqrt(tot_I / tot_B)
    for bay, rho, meanI, mc in tmp:
        for k, col in [("bayes", bay), ("vmg", bay * g), ("vmr", bay / np.maximum(np.abs(rho), 1e-3))]:
            res[k][0].append(torch.tensor(col, dtype=torch.float32)[:, None]); res[k][1].append(torch.tensor(meanI - col * mc, dtype=torch.float32).to(DEV))
    return res, g


if __name__ == "__main__":
    m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"]); what = sys.argv[1]
    if what == "oracle0":
        class O(Family):
            def pre(self, ins): return moved_sum_t(F.relu(pre_3b(m, ins[0])[:, fam]), 0)[:, None]
        fm = O(m, fam, 1).to(DEV)
    else:
        ck = torch.load(f"cache/ckpt/{what}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV)
        fm.load_state_dict(ck["state"])
    st = stats(m, fm, fam, D); res, g = writes(m, fam, st)
    print(f"{what}: global variance factor g = {g:.2f}", flush=True)
    for k, (cols, cb) in res.items():
        with torch.no_grad():
            for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
        fm.colbias = cb
        print(f"  write {k:5s}: dev KL {evaluate(m, fm, D, D.dv)}", flush=True)
