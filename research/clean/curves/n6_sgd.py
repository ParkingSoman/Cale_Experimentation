"""N6: Theorem 31. Per-neuron imbalance D_i = ||u_i||^2 - ||v_i||^2 (weights) vs per-sample gradient energies (data check)."""
import numpy as np, torch
from scipy.stats import spearmanr
from inception import *
from family import Data, pool3, IN4A
m = load_model(); D = Data(seed=0)
convs = [m.mixed3b_1x1_pre_relu_conv, m.mixed3b_3x3_pre_relu_conv, m.mixed3b_5x5_pre_relu_conv, m.mixed3b_pool_reduce_pre_relu_conv]
outs4 = [getattr(m, n) for n in IN4A]
def u_norm2():
    return torch.cat([(c.weight.detach() ** 2).flatten(1).sum(1) + c.bias.detach() ** 2 for c in convs]).cpu().double()
def v_norm2():
    return sum((c.weight.detach()[:, :, 0, 0] ** 2).sum(0) for c in outs4).cpu().double()
Dw = (u_norm2() - v_norm2()).numpy(); ratio = (u_norm2() / v_norm2()).numpy()
for p in m.parameters(): p.requires_grad_(False)
for c in convs + outs4: c.weight.requires_grad_(True); c.bias.requires_grad_(True)
idx = np.sort(np.random.default_rng(151).choice(D.tr, 200, replace=False)); Eu = torch.zeros(480, dtype=torch.float64); Ev = torch.zeros(480, dtype=torch.float64); Eh = torch.zeros(480, dtype=torch.float64)
for i in idx:
    x = prep(D.X[i:i + 1]); y = torch.tensor([int(D.lab[D.y[i]])], device=DEV)
    a3 = to_3a(m, x); a, b, c, p = inputs_3b(m, a3)
    pre = torch.cat([convs[0](a), convs[1](F.pad(b, (1, 1, 1, 1))), convs[2](F.pad(c, (2, 2, 2, 2))), convs[3](p)], 1); h = F.relu(pre)
    xb = pool3(h); pp = F.max_pool2d(F.pad(xb, (1, 1, 1, 1), value=float('-inf')), 3, 1); o = [F.conv2d(xb if j < 3 else pp, oc.weight, oc.bias) for j, oc in enumerate(outs4)]
    from w_fisher import upper
    loss = F.cross_entropy(upper(m, o), y)
    gs = torch.autograd.grad(loss, [t for cc in convs for t in (cc.weight, cc.bias)] + [oc.weight for oc in outs4])
    gu = torch.cat([(gs[2 * k] ** 2).flatten(1).sum(1) + gs[2 * k + 1] ** 2 for k in range(4)]).detach().cpu().double()
    gv = sum((g[:, :, 0, 0] ** 2).sum(0) for g in gs[8:]).detach().cpu().double()
    Eu += gu; Ev += gv; Eh += (h.detach() ** 2).sum((0, 2, 3)).cpu().double()
Eu, Ev, Eh = Eu / len(idx), Ev / len(idx), Eh / len(idx)
print(f"imbalance in^2/out^2: median {np.median(ratio):.3f}; D_i sign: {np.mean(Dw > 0):.2f} positive")
print(f"N6 Spearman(D_i, E||g_u||^2 - E||g_v||^2) = {spearmanr(Dw, (Eu - Ev).numpy())[0]:+.3f}")
print(f"   Spearman(D_i / (||u||^2 + ||v||^2), (E||g_u||^2 - E||g_v||^2) / (E||g_u||^2 + E||g_v||^2)) = {spearmanr(Dw / (u_norm2() + v_norm2()).numpy(), ((Eu - Ev) / (Eu + Ev)).numpy())[0]:+.3f}")
print(f"   baseline Spearman(|D_i|, E[h^2]) = {spearmanr(np.abs(Dw), Eh.numpy())[0]:+.3f}; Spearman(D_i, -E||g_v||^2) = {spearmanr(Dw, -Ev.numpy())[0]:+.3f}; Spearman(D_i, E||g_u||^2) = {spearmanr(Dw, Eu.numpy())[0]:+.3f}")
np.savez("cache/n6_sgd.npz", D=Dw, Eu=Eu.numpy(), Ev=Ev.numpy(), Eh=Eh.numpy())
