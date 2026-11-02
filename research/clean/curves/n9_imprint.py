"""N9: move the pixels of 16 inputs (start: class inversions) so the set's per-neuron Fisher-energy imbalance vector lines
up with D(weights) on even neurons; score on odd neurons. Twice-differentiable max-pool (same forward)."""
import numpy as np, torch
import torch.nn.functional as F_
_mp = F_.max_pool2d
def _mp2(*a, **k):
    k = dict(k); k["return_indices"] = True; return _mp(*a, **k)[0]
F_.max_pool2d = _mp2
from scipy.stats import spearmanr
from inception import *
from family import Data, pool3, IN4A
from w_fisher import upper
m = load_model(); D = Data(seed=0); Dw = torch.tensor(np.load("cache/n6_sgd.npz")["D"].astype(np.float32), device=DEV); fam = list(np.load("cache/s1_fine.npz")["fam"])
convs = [m.mixed3b_1x1_pre_relu_conv, m.mixed3b_3x3_pre_relu_conv, m.mixed3b_5x5_pre_relu_conv, m.mixed3b_pool_reduce_pre_relu_conv]; outs4 = [getattr(m, n) for n in IN4A]
for p in m.parameters(): p.requires_grad_(False)
params = [t for cc in convs for t in (cc.weight, cc.bias)] + [oc.weight for oc in outs4]
for t in params: t.requires_grad_(True)
ev, od = torch.arange(0, 480, 2, device=DEV), torch.arange(1, 480, 2, device=DEV)
def energy(x, R=3, graph=True):
    a3 = to_3a(m, x[None] - 117.0); a, b, c, p = inputs_3b(m, a3)
    h = F.relu(torch.cat([convs[0](a), convs[1](F.pad(b, (1, 1, 1, 1))), convs[2](F.pad(c, (2, 2, 2, 2))), convs[3](p)], 1))
    xb = pool3(h); pp = F.max_pool2d(F.pad(xb, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    lp = F.log_softmax(upper(m, [F.conv2d(xb if j < 3 else pp, oc.weight, oc.bias) for j, oc in enumerate(outs4)]), 1); top = lp.detach().exp().topk(R, 1)
    d = 0
    for r in range(R):
        gs = torch.autograd.grad(-lp[0, top.indices[0, r]], params, create_graph=graph, retain_graph=True)
        d = d + float(top.values[0, r]) * (torch.cat([(gs[2 * k] ** 2).flatten(1).sum(1) + gs[2 * k + 1] ** 2 for k in range(4)]) - sum((g[:, :, 0, 0] ** 2).sum(0) for g in gs[8:]))
    return d
def score(X):
    d = sum(energy(torch.tensor(x, device=DEV).permute(2, 0, 1).float(), graph=False).detach() for x in X).cpu().numpy(); t = Dw.cpu().numpy()
    return spearmanr(d[0::2], t[0::2])[0], spearmanr(d[1::2], t[1::2])[0]
def cofire(X):
    with torch.no_grad():
        z = pool3(F.relu(pre_3b(m, to_3a(m, prep(X)))))[:, fam][:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(10, -1).cpu().numpy()
    return np.corrcoef(z)
X0 = np.load("cache/e1_self_256.npy")[:16]; R = np.asarray(D.X[np.sort(np.random.default_rng(171).choice(D.tr, 64, replace=False))]); iu = np.triu_indices(10, 1); Cr = cofire(R)
s0 = score(X0); print(f"start (inversions): Spearman even {s0[0]:+.3f} | odd (held out) {s0[1]:+.3f} | co-firing vs real {np.corrcoef(cofire(X0)[iu], Cr[iu])[0, 1]:+.3f}", flush=True)
x = torch.tensor(X0, device=DEV).permute(0, 3, 1, 2).float().requires_grad_(); opt = torch.optim.Adam([x], lr=1.0)
for step in range(40):
    d = sum(energy(x[i]) for i in range(len(x))); t = Dw[ev]; dv = d[ev]
    loss = -torch.dot(dv - dv.mean(), t - t.mean()) / (dv - dv.mean()).norm() / (t - t.mean()).norm()
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): x.clamp_(0, 255)
    if step % 10 == 9: print(f"  step {step + 1}: train cos {-float(loss):+.3f}", flush=True)
X1 = x.detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8); s1 = score(X1)
print(f"after: Spearman even {s1[0]:+.3f} | odd (held out) {s1[1]:+.3f} | co-firing vs real {np.corrcoef(cofire(X1)[iu], Cr[iu])[0, 1]:+.3f} | pixel change sd {float(np.abs(X1.astype(float) - X0).std()):.1f}")
np.save("cache/n9_x16.npy", X1)
