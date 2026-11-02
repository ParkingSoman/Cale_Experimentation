"""Diagnostics read off the trained write (NOT pre-registered derivations): vmr x f; vmr for |rho|>=.15 and Bayes below."""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, evaluate, IN4A
from d4_write import stats
m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
for tag in sys.argv[1:]:
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
    st = stats(m, fm, fam, D); parts = []
    for j, name in enumerate(IN4A):
        mu, C = st[1] if j == 3 else st[0]
        A = getattr(m, name).weight.detach().cpu()[:, fam, 0, 0].double().numpy()
        covIc = A @ C[:10, 10]; varI = np.einsum("oi,ij,oj->o", A, C[:10, :10], A); rho = covIc / np.sqrt(varI * C[10, 10] + 1e-12)
        parts.append((covIc / C[10, 10], rho, A @ mu[:10], mu[10]))
    def setw(fn):
        with torch.no_grad():
            cb = []
            for p, s, (bay, rho, meanI, mc) in zip(fm.cols, fm.cscale, parts):
                col = fn(bay, rho); p.copy_(torch.tensor(col, dtype=torch.float32)[:, None].to(DEV) / s); cb.append(torch.tensor(meanI - col * mc, dtype=torch.float32).to(DEV))
        fm.colbias = cb
    for lab, fn in [("vmr x 1.0", lambda b, r: b / np.maximum(np.abs(r), 1e-3)), ("vmr x .7", lambda b, r: .7 * b / np.maximum(np.abs(r), 1e-3)),
                    ("vmr |rho|>=.15, Bayes below", lambda b, r: np.where(np.abs(r) >= .15, b / np.maximum(np.abs(r), 1e-3), b)),
                    ("vmr x .7 |rho|>=.15, Bayes below", lambda b, r: np.where(np.abs(r) >= .15, .7 * b / np.maximum(np.abs(r), 1e-3), b))]:
        setw(fn); print(f"{tag} {lab:34s}: dev KL {evaluate(m, fm, D, D.dv)}", flush=True)
