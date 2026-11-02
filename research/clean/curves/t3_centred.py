"""T3: Theorem 10b (per-reader constant) on the curve family. Prediction in CURVES-LOG.md (MIX, T3)."""
import numpy as np, torch
from inception import *
from family import pool3
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
t = np.load("cache/t2.npz"); Phi = t["Phi"]; ix = np.sort(np.random.default_rng(41).permutation(tr)[:1500])   # same images as C_gen in T1
mu = np.zeros(10); C = np.zeros((10, 10)); n = 0
with torch.no_grad():
    for i in range(0, len(ix), 50):
        z = pool3(F.relu(pre_3b(m, to_3a(m, prep(X[ix[i:i + 50]])))[:, fam]))[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(10, -1).cpu().double().numpy()
        mu += z.sum(1); C += z @ z.T; n += z.shape[1]
mu /= n; C /= n; S = C - np.outer(mu, mu); tot = np.trace(Phi @ C)
ev, V = np.linalg.eig(Phi @ S); o = np.argsort(-ev.real); ev = ev.real[o]; lam = V[:, o[0]].real; lam = lam * np.sign(lam.sum()); lam /= np.abs(lam).sum()
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"R1 beta share {mu @ Phi @ mu / tot:.3f} | R2 beta + k=1 share {(mu @ Phi @ mu + ev[0]) / tot:.3f} | + k=2 {(mu @ Phi @ mu + ev[:2].sum()) / tot:.3f} | + k=3 {(mu @ Phi @ mu + ev[:3].sum()) / tot:.3f}")
print("lam_10b", lam, "| min", lam.min())
for nm, lt in [("twin1", [0.118, 0.136, 0.107, 0.102, 0.118, 0.156, 0.068, 0.088, 0.074, 0.032]), ("twin2", [0.061, 0.156, 0.129, 0.13, 0.127, 0.192, 0.109, 0.049, 0.044, -0.002])]:
    print(f"R3 corr(lam_10b, {nm}) {np.corrcoef(lam, lt)[0, 1]:.2f}  (Theorem 10 uncentred: {np.corrcoef(t['lam_F'], lt)[0, 1]:.2f})")
print("normalised Sigma:\n", S / np.sqrt(np.outer(np.diag(S), np.diag(S))))
np.savez("cache/t3.npz", mu=mu, C=C, S=S, lam=lam, ev=ev)
