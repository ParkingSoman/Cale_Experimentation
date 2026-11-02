"""C28: off-diagonal cos(Gram of 4a's input convs, Fisher at 3b's pooled output) as a weights-only score of an input set."""
import sys, numpy as np, torch
from inception import *
from family import Data, pool3
from w_fisher import upper, PAD
from family import IN4A
m = load_model(); D = Data(seed=0)
for p in m.parameters(): p.requires_grad_(False)
Ws = [getattr(m, f"mixed4a_{b}_pre_relu_conv").weight.detach()[:, :, 0, 0].cpu().double() for b in ["1x1", "3x3_bottleneck", "5x5_bottleneck", "pool_reduce"]]
G = sum(W.T @ W for W in Ws); mk = ~torch.eye(480, dtype=bool)
def fisher(X, nv=4, seed=0):
    gen = torch.Generator().manual_seed(seed); Fm = [torch.zeros(480, 480, dtype=torch.float64) for _ in range(2)]
    for i in range(0, len(X), 16):
        x = torch.as_tensor(X[i:i + 16], device=DEV).float()
        with torch.no_grad(): xb = pool3(F.relu(pre_3b(m, to_3a(m, x))))
        xb = xb.clone().requires_grad_(); p = PAD(xb); outs = [F.conv2d(xb if j < 3 else p, getattr(m, n).weight, getattr(m, n).bias) for j, n in enumerate(IN4A)]
        lp = F.log_softmax(upper(m, outs), 1); top = lp.detach().exp().topk(10, 1)
        for r in range(nv):
            eps = (torch.randint(0, 2, top.values.shape, generator=gen) * 2 - 1).float().to(DEV)
            g, = torch.autograd.grad((eps * top.values.sqrt() * lp.gather(1, top.indices)).sum(), xb, retain_graph=r < nv - 1)
            g = g.permute(0, 2, 3, 1).reshape(-1, 480).cpu().double(); Fm[int(i >= len(X) // 2)] += g.T @ g / nv
    return Fm
def score(X):
    F1, F2 = fisher(X); Ft = F1 + F2
    c = lambda A: float((A[mk] * G[mk]).sum() / A[mk].norm() / G[mk].norm())
    rel = float((F1[mk] * F2[mk]).sum() / F1[mk].norm() / F2[mk].norm())
    return c(Ft), c(F1), c(F2), rel
if __name__ == "__main__":
    from scipy.special import softmax
    P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]
    R = np.asarray(D.X[np.sort(np.random.default_rng(93).choice(uns, 128, replace=False))]).astype(np.float32).transpose(0, 3, 1, 2) - 117.0
    for s in [.5, .75, 1.0, 1.33]:
        sc = score(np.clip(R * s, -117, 138)); print(f"real unsure, contrast x{s:<4}: score {sc[0]:.4f} (halves {sc[1]:.4f} / {sc[2]:.4f}; split-half rel {sc[3]:.3f})", flush=True)
