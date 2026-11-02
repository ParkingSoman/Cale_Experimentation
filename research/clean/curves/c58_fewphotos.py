"""32.3 follow-up: how many photos fix the frequency vector? c (bias column of g^cov) from n photos; reader-question
weights fitted to it (non-negative, half the readers); scored on the other half against the 8000-photo c."""
import numpy as np, torch, time
from scipy.optimize import nnls
SRC = open("c41_split.py").read(); exec(SRC.split("ref = torch.load")[0])
bc = lambda g: torch.cat([g[:304, 480], g[304:, 961]]).cpu().double().numpy()
P = bc(torch.load("cache/c40_photo8000_grad.pt", weights_only=False)["g"])
Z = torch.load("cache/c57_phi_368.pt"); phi = (Z["raw"] - Z["fact"]).double().numpy()
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
for n in [32, 64, 128]:
    X = D.X[np.sort(np.random.default_rng(58 + n).choice(D.tr, n, replace=False))]
    r = np.random.default_rng(41); DS = []
    while len(DS) < 4:
        pp = r.permutation(n)
        if (pp != np.arange(n)).all(): DS.append(pp)
    gc = push(X, X) - sum(push(X, X[d]) for d in DS) / 4; cn = bc(gc)
    out = []
    for rep in range(5):
        idx = np.random.default_rng(rep).permutation(368); fit, ho = idx[:184], idx[184:]
        w, _ = nnls(phi[:, fit].T, cn[fit] / np.linalg.norm(cn[fit]) * 100); out.append((cos((w @ phi)[ho], P[ho]), cos(cn[ho], P[ho])))
    o = np.array(out); print(f"n = {n:3d} photos: photo c alone vs 8000 {cos(cn, P):+.3f}; held-out: questions weighted by n photos {o[:,0].mean():+.3f}, the n photos' own c {o[:,1].mean():+.3f} ({time.time() - t0:.0f}s)", flush=True)
print("done")
