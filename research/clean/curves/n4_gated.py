"""N4: gated co-sensitivity M_ij = E[sum_q 1(z_i>0) 1(z_j>0) g_i g_j] at 3b's 5x5 branch vs the kernel Gram K^T K."""
import numpy as np, torch
from inception import *
from family import Data, pool3
from w_fisher import upper, PAD
from family import IN4A
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); f5 = [i - 320 for i in fam]
for p in m.parameters(): p.requires_grad_(False)
K = m.mixed3b_5x5_pre_relu_conv.weight.detach().reshape(96, -1).cpu().double(); KK = K @ K.T
idx = np.sort(np.random.default_rng(131).choice(D.tr, 500, replace=False)); gen = torch.Generator().manual_seed(0)
acc = {k: torch.zeros(96, 96, dtype=torch.float64) for k in ["M_agop", "M_fisher", "ungated", "coact"]}
for i in range(0, 500, 20):
    x = prep(D.X[idx[i:i + 20]])
    with torch.no_grad(): a3 = to_3a(m, x); ins = inputs_3b(m, a3)
    z = m.mixed3b_5x5_pre_relu_conv(F.pad(ins[2], (2, 2, 2, 2))).detach().requires_grad_()
    pre = torch.cat([m.mixed3b_1x1_pre_relu_conv(ins[0]), m.mixed3b_3x3_pre_relu_conv(F.pad(ins[1], (1, 1, 1, 1))), z, m.mixed3b_pool_reduce_pre_relu_conv(ins[3])], 1)
    xb = pool3(F.relu(pre)); p = PAD(xb); outs = [F.conv2d(xb if j < 3 else p, getattr(m, n).weight, getattr(m, n).bias) for j, n in enumerate(IN4A)]
    lg = upper(m, outs); on = (z > 0).float().permute(1, 0, 2, 3).reshape(96, -1).cpu().double(); zr = F.relu(z).detach().permute(1, 0, 2, 3).reshape(96, -1).cpu().double()
    acc["coact"] += zr @ zr.T
    for r in range(4):
        v = torch.randn(lg.shape, generator=gen).to(DEV); g, = torch.autograd.grad((lg * v).sum(), z, retain_graph=True)
        g = g.permute(1, 0, 2, 3).reshape(96, -1).cpu().double(); acc["M_agop"] += (g * on) @ (g * on).T / 4; acc["ungated"] += g @ g.T / 4
    lp = F.log_softmax(lg, 1); top = lp.detach().exp().topk(10, 1)
    for r in range(4):
        eps = (torch.randint(0, 2, top.values.shape, generator=gen) * 2 - 1).float().to(DEV)
        g, = torch.autograd.grad((eps * top.values.sqrt() * lp.gather(1, top.indices)).sum(), z, retain_graph=r < 3)
        g = g.permute(1, 0, 2, 3).reshape(96, -1).cpu().double(); acc["M_fisher"] += (g * on) @ (g * on).T / 4
def off(A, B, sel=None):
    if sel is not None: A, B = A[sel][:, sel], B[sel][:, sel]
    mk = ~torch.eye(A.shape[0], dtype=bool); a, b = A[mk], B[mk]; return float((a * b).sum() / a.norm() / b.norm())
print("off-diagonal cosine with K^T K | all 96 channels | 10 curve channels")
for k in ["M_agop", "M_fisher", "ungated", "coact"]: print(f"  {k:9s} | {off(acc[k], KK):+.3f} | {off(acc[k], KK, f5):+.3f}")
torch.save(acc, "cache/n4_gated.pt")
