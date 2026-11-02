"""N7: per-neuron Fisher-energy imbalance D_i(q) = E_q||g_u||^2 - E_q||g_v||^2 (network's own labels) vs D_i(weights)."""
import numpy as np, torch
from scipy.stats import spearmanr
from inception import *
from family import Data, pool3, IN4A
from w_fisher import upper
from scipy.special import softmax
m = load_model(); D = Data(seed=0); Dw = np.load("cache/n6_sgd.npz")["D"]
convs = [m.mixed3b_1x1_pre_relu_conv, m.mixed3b_3x3_pre_relu_conv, m.mixed3b_5x5_pre_relu_conv, m.mixed3b_pool_reduce_pre_relu_conv]; outs4 = [getattr(m, n) for n in IN4A]
for p in m.parameters(): p.requires_grad_(False)
params = [t for cc in convs for t in (cc.weight, cc.bias)] + [oc.weight for oc in outs4]
for t in params: t.requires_grad_(True)
def energies(X, R=10):
    Eu = torch.zeros(480, dtype=torch.float64); Ev = torch.zeros(480, dtype=torch.float64)
    for i in range(len(X)):
        x = prep(X[i:i + 1]); a3 = to_3a(m, x); a, b, c, p = inputs_3b(m, a3)
        h = F.relu(torch.cat([convs[0](a), convs[1](F.pad(b, (1, 1, 1, 1))), convs[2](F.pad(c, (2, 2, 2, 2))), convs[3](p)], 1))
        xb = pool3(h); pp = F.max_pool2d(F.pad(xb, (1, 1, 1, 1), value=float('-inf')), 3, 1)
        lp = F.log_softmax(upper(m, [F.conv2d(xb if j < 3 else pp, oc.weight, oc.bias) for j, oc in enumerate(outs4)]), 1); top = lp.detach().exp().topk(R, 1)
        for r in range(R):
            gs = torch.autograd.grad(-lp[0, top.indices[0, r]], params, retain_graph=r < R - 1); wr = float(top.values[0, r])
            Eu += wr * torch.cat([(gs[2 * k] ** 2).flatten(1).sum(1) + gs[2 * k + 1] ** 2 for k in range(4)]).cpu().double()
            Ev += wr * sum((g[:, :, 0, 0] ** 2).sum(0) for g in gs[8:]).cpu().double()
    return (Eu - Ev).numpy() / len(X)
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]
comb = np.concatenate([np.load("cache/e7_bnd_1024_128_0.npy")[:96], np.load("cache/c31_fv.npy")[:32]])
sets = {"real unsure": np.asarray(D.X[np.sort(np.random.default_rng(161).choice(uns, 128, replace=False))]),
        "real (all)": np.asarray(D.X[np.sort(np.random.default_rng(162).choice(D.tr, 128, replace=False))]),
        "class inversions": np.load("cache/e1_self_256.npy")[:128], "boundary": np.load("cache/e7_bnd_1024_128_0.npy")[:128],
        "reader pictures": np.load("cache/c31_fv.npy")[:104], "combined (boundary + readers)": comb}
out = {}
for k, X in sets.items():
    out[k] = energies(X); print(f"  {k:30s} Spearman(D weights, D set) = {spearmanr(Dw, out[k])[0]:+.3f}", flush=True)
np.savez("cache/n7_energy.npz", **{k.replace(' ', '_'): v for k, v in out.items()})
