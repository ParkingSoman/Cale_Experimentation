"""I2 (Part VI): Kbar's top eigenmodes drawn as curve glyphs on the 14 x 14 grid (x-input block), and the share of level
0p's metric cost E r^T Kbar r (500 dev images) that each mode carries."""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from circ import *

Kbar = torch.load("cache/x5_kbar.pt").double()
lam, U = torch.linalg.eigh(Kbar); lam, U = lam.flip(0), U.flip(1)
print("top eigenvalue shares of trace:", [round(float(v), 3) for v in (lam[:10] / lam.sum())], flush=True)
# level 0p residual r = (h_F - Lambda z_R) on x and p inputs
fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(torch.load("cache/x2_b_level0_p.pt", map_location="cpu"))
cs = [float(c) for c in fm.cscale]
dWall = torch.cat([fm.dW[j].detach() * cs[j] for j in range(4)]).double(); cb = torch.cat([fm.cbp[j].detach() * 0.1 for j in range(4)]).double()
WFall = torch.cat([w.cpu() for w in WF]).double()
Y = torch.cat([dWall, cb[:, None]], 1); nx = sum(w.shape[0] for w in WF[:3])          # level 0p: Lambda_x for 1x1/3x3b/5x5b, Lambda_p for pool
Lams = [torch.linalg.lstsq(WFall[:nx], Y[:nx]).solution, torch.linalg.lstsq(WFall[nx:], Y[nx:]).solution]
print("Lambda fit residuals (x, p)", [round(float((WFall[s_] @ L - Y[s_]).norm() / Y[s_].norm()), 6) for s_, L in zip([slice(0, nx), slice(nx, None)], Lams)], flush=True)
D1 = Data(seed=0, ndev=1000); idx = np.sort(np.random.default_rng(3).choice(np.setdiff1d(D.dv, D1.dv), 500, replace=False))
proj = torch.zeros(3920, dtype=torch.float64); tot = 0.0
for i in range(0, 500, 50):
    x, p = family_inputs(idx[i:i + 50]); rs = []
    for t, Lam in zip((x, p), Lams):
        t = t.cpu().double().flatten(2)                                                       # (B, 480, 196)
        z = torch.cat([t[:, keep], torch.ones_like(t[:, :1])], 1)
        rs.append(t[:, fam] - torch.einsum("fa,bap->bfp", Lam, z))
    r = torch.cat(rs, 1).reshape(len(x), 3920)
    c = r @ U; proj += (lam * c ** 2).sum(0); tot += float(((r @ Kbar) * r).sum())
share = (proj / proj.sum()).numpy(); np.save("cache/c2_mode_share.npy", share)
print(f"check: sum over modes {float(proj.sum()):.4f} vs direct {tot:.4f}", flush=True)
print("cost share of top modes:", [round(float(s), 3) for s in share[:10]], f"top 3 {share[:3].sum():.3f}, top 20 {share[:20].sum():.3f}",
      f"; modes needed for 80%: {int(np.searchsorted(np.cumsum(np.sort(share)[::-1]), .8)) + 1}", flush=True)
order = np.argsort(-share)[:8]; print("modes ranked by cost share:", order.tolist(), [round(float(share[k]), 3) for k in order], flush=True)
# glyphs: member f at position p = an arc of the circle (radius 60 px = 3.75 cells) whose centre lies in direction ang[f]
Z = np.load("results_g0_tuning.npz"); th = Z["th"]; ang = np.array([th[Z["60"][:, n].argmax()] for n in fam], float)
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for ax, k in zip(axs.ravel(), order):
    v = U[:1960, k].reshape(10, 14, 14).numpy(); vm = np.abs(v).max()
    for y in range(14):
        for x_ in range(14):
            for f in range(10):
                a = abs(v[f, y, x_]) / vm
                if a < .2: continue
                t = np.deg2rad(ang[f]); cx, cy = x_ + 3.75 * np.cos(t), y - 3.75 * np.sin(t)      # image y points down
                phi0 = np.arctan2(y - cy, x_ - cx); ph = np.linspace(phi0 - .25, phi0 + .25, 12)
                ax.plot(cx + 3.75 * np.cos(ph), cy + 3.75 * np.sin(ph), color="C3" if v[f, y, x_] > 0 else "C0", lw=3 * a, alpha=min(1, a + .2))
    ax.set_xlim(-1, 14); ax.set_ylim(14, -1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"mode {k}: eig share {float(lam[k] / lam.sum()):.3f}, cost share {share[k]:.3f}")
plt.tight_layout(); plt.savefig("cache/c2_modes.png", dpi=80); print("saved cache/c2_modes.png")
