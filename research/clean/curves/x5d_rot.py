"""R5' part 1 (Statements 16.1, 16.4): average coupling metric Kbar = E_x Cov_c(g_c), g_c = W_F^T J_c (output J), on the
family's 10 signals x {x-input, p-input of the pool reader} x 196 positions. 2000 train images (rng 5). FITTED microscope."""
import numpy as np, torch
from inception import *
from family import Data, pool3, IN4A
exec(open("x2b_reason.py").read().split("names = {")[0])          # m, D, idx, convs, cont, reader_pre
fam = list(np.load("cache/s1_fine.npz")["fam"]); K = 10; WF = [c.weight.detach()[:, fam, 0, 0] for c in convs]
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))
Kb = torch.zeros(3920, 3920, device=DEV); Ksum = torch.zeros(3920, 3920, dtype=torch.float64)
for i in range(0, 500, 20):
    b = tr[i:i + 20]; NIMG = 500
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, torch.rot90(prep(D.X[b]), 1, (2, 3))))[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1); a0 = reader_pre(x, p)
    post = [F.relu(t).requires_grad_(True) for t in a0]; lg = cont(post)
    top = lg.detach().softmax(1).topk(K, 1); pk = top.values / top.values.sum(1, keepdim=True)
    Jo = [[] for _ in range(4)]
    for c in range(K):
        gr = torch.autograd.grad(lg.gather(1, top.indices[:, c:c + 1]).sum(), post, retain_graph=c < K - 1)
        for j in range(4): Jo[j].append(gr[j])
    with torch.no_grad():
        Jo = [torch.stack(t, 1) for t in Jo]
        gx = sum(torch.einsum("of,bkopq->bkfpq", WF[j], Jo[j]) for j in range(3)); gp = torch.einsum("of,bkopq->bkfpq", WF[3], Jo[3])
        g = torch.cat([gx, gp], 2).reshape(len(b), K, 3920)
        gc = (g - (pk[:, :, None] * g).sum(1, keepdim=True)) * pk[:, :, None].sqrt()
        Kb += gc.reshape(-1, 3920).T @ gc.reshape(-1, 3920)
    if (i // 20) % 10 == 9: Ksum += Kb.cpu().double(); Kb.zero_()
Ksum += Kb.cpu().double(); Kbar = (Ksum / 500).float()
torch.save(Kbar, "cache/x5_kbar_rot90.pt"); import sys; sys.exit()
# (c) nearest Kronecker product K_F (20 x 20) (x) K_P (196 x 196): Van Loan rearrangement, energy of the top singular value
Rr = Kbar.reshape(20, 196, 20, 196).permute(0, 2, 1, 3).reshape(400, 196 * 196)
s = torch.linalg.svdvals(Rr.to(DEV)).cpu()
print(f"(c) Kronecker: top singular value explains {float(s[0] ** 2 / (s ** 2).sum()):.3f} of Kbar's energy; top 3 {float((s[:3] ** 2).sum() / (s ** 2).sum()):.3f}")
# offset-averaged block of the x-signals: Koff[dy, dx, f, g] = mean over p of Kbar[(f, p), (g, p + d)]
Kx = Kbar[:1960, :1960].reshape(10, 14, 14, 10, 14, 14)
Koff = torch.zeros(27, 27, 10, 10); cnt = torch.zeros(27, 27)
for y0 in range(14):
    for x0 in range(14):
        blk = Kx[:, y0, x0]                                              # (10, 14, 14, 10) -> partner (g) at all q
        Koff[13 - y0:27 - y0, 13 - x0:27 - x0] += blk.permute(1, 2, 0, 3); cnt[13 - y0:27 - y0, 13 - x0:27 - x0] += 1
Koff /= cnt[:, :, None, None].clamp(min=1); torch.save(Koff, "cache/x5_koff.pt")
print("saved Kbar, Koff; diag energy share", float(Kbar.diagonal().sum() / Kbar.abs().sum()))
