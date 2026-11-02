"""Statement 20.3 check: effective rank of E_x A_x^T A_x at 5b, A_x = Jacobian of the 5b output w.r.t. the family's 3920
signals with the true ReLU gates / max-pool choices of image x (Sigma = I). 8 train images (rng 5 set)."""
import numpy as np, torch
from torch.func import jvp, vmap
from circ import *
famt = torch.tensor(fam, device=DEV)
def f_from(x0, p0):
    def f(s):                                          # s: (n, 2, 10, 14, 14) added to the family channels of x and p
        x = x0.repeat(len(s), 1, 1, 1).index_add(1, famt, s[:, 0]); p = p0.repeat(len(s), 1, 1, 1).index_add(1, famt, s[:, 1])
        hh = cont4a(readers_post(x, p))
        for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: hh = mixed(m, n, hh)
        hh = pool3(hh)
        for n in ["mixed5a", "mixed5b"]: hh = mixed(m, n, hh)
        return hh.flatten(1)
    return f
def cont4a(post):
    a, b, cc, d = post
    return torch.cat([a, F.relu(C3(F.pad(b, (1, 1, 1, 1)))), F.relu(C5(F.pad(cc, (2, 2, 2, 2)))), d], 1)
K = torch.zeros(3920, 3920, device=DEV); eye = torch.eye(3920, device=DEV).reshape(3920, 2, 10, 14, 14)
er = lambda K: float(torch.exp(-(lambda p: (p[p > 0] * p[p > 0].log()).sum())(torch.linalg.eigvalsh(K.cpu().double()).clamp(min=0) / torch.linalg.eigvalsh(K.cpu().double()).clamp(min=0).sum())))
for n, i in enumerate(TR[:8]):
    x0, p0 = family_inputs(TR[i:i + 1] if False else np.array([i])); f = f_from(x0, p0); z = torch.zeros(2, 10, 14, 14, device=DEV)
    cols = []
    for c in range(0, 3920, 98):
        with torch.no_grad(): t = eye[c:c + 98] * 1e-3; o = (f(t) - f(-t)) / 2e-3
        cols.append(o)
    A = torch.cat(cols); Kx = A @ A.T; K += Kx
    print(f"image {n}: eff rank single {er(Kx):.1f}, running average {er(K):.1f}", flush=True)
print(f"TRUE-gated E_x A_x^T A_x at 5b (8 images): effective rank {er(K):.1f}; gate-free 5b was 23.5")
