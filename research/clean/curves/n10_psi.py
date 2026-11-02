"""N10: child-scale Fisher Psi (10 x 10) on real unsure, boundary, and boundary moved by the SGD-imprint generator."""
import numpy as np, torch
import n9_imprint as N9   # patches max-pool once (twice-differentiable, same forward) and gives energy(), Dw
from inception import *
from family import Data, pool3, IN4A
from w_fisher import upper, intact, A_cols, fisher_grads
from scipy.special import softmax
m, D, fam = N9.m, N9.D, N9.fam; A = A_cols(m, fam)
def psi(X):
    Ps = torch.zeros(10, 10, dtype=torch.float64)
    for i in range(0, len(X), 16):
        with torch.no_grad(): ins = inputs_3b(m, to_3a(m, prep(X[i:i + 16]))); outs, z1, z2 = intact(m, fam, ins)
        G, _ = fisher_grads(m, outs, 10)
        for g in G:
            s = sum(torch.einsum("bohw,oi,bihw->bi", g[j], A[j], z1 if j < 3 else z2) for j in range(4)).cpu().double(); Ps += s.T @ s
    return Ps / len(X)
def norm(Pm): d = torch.sqrt(torch.diag(Pm)); return Pm / d[:, None] / d[None], d / d.sum()
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]
R = np.asarray(D.X[np.sort(np.random.default_rng(181).choice(uns, 128, replace=False))]); B0 = np.load("cache/e7_bnd_1024_128_0.npy")[:48]
B1 = []
for b in range(0, 48, 16):
    x = torch.tensor(B0[b:b + 16], device=DEV).permute(0, 3, 1, 2).float().requires_grad_(); opt = torch.optim.Adam([x], lr=1.0)
    for step in range(30):
        d = sum(N9.energy(x[i]) for i in range(len(x))); t = N9.Dw
        loss = -torch.dot(d - d.mean(), t - t.mean()) / (d - d.mean()).norm() / (t - t.mean()).norm(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): x.clamp_(0, 255)
    print(f"  batch {b // 16}: final train cos {-float(loss.detach()):+.3f}", flush=True); B1.append(x.detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8))
B1 = np.concatenate(B1); np.save("cache/n10_bnd_imprint48.npy", B1)
Pr, Pb, Pi = psi(R), psi(B0), psi(B1); iu = np.triu_indices(10, 1)
for k, Pm in [("boundary start", Pb), ("boundary + imprint", Pi)]:
    (Cn, dn), (Cr, dr) = norm(Pm), norm(Pr)
    print(f"  {k:20s}: Psi pattern corr vs real {np.corrcoef(Cn.numpy()[iu], Cr.numpy()[iu])[0, 1]:+.3f} | diagonal shape corr {np.corrcoef(dn.numpy(), dr.numpy())[0, 1]:+.3f} | scale ratio {float(torch.trace(Pm) / torch.trace(Pr)):.3f}")
