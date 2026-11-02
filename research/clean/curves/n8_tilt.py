"""N8: per-input Fisher-energy vectors d_s for the network-made pool; max-entropy tilt w ~ exp(mu . P d) on even neurons."""
import numpy as np, torch
from scipy.stats import spearmanr
from scipy.optimize import minimize
from inception import *
from family import pool3, IN4A
from w_fisher import upper
m = load_model(); Dw = np.load("cache/n6_sgd.npz")["D"]
convs = [m.mixed3b_1x1_pre_relu_conv, m.mixed3b_3x3_pre_relu_conv, m.mixed3b_5x5_pre_relu_conv, m.mixed3b_pool_reduce_pre_relu_conv]; outs4 = [getattr(m, n) for n in IN4A]
for p in m.parameters(): p.requires_grad_(False)
params = [t for cc in convs for t in (cc.weight, cc.bias)] + [oc.weight for oc in outs4]
for t in params: t.requires_grad_(True)
pool = np.concatenate([np.load("cache/e1_self_256.npy"), np.load("cache/e7_bnd_1024_128_0.npy"), np.load("cache/c31_fv.npy")[:104]]); gen = torch.Generator().manual_seed(0)
ds = np.zeros((len(pool), 480))
for i in range(len(pool)):
    x = prep(pool[i:i + 1]); a3 = to_3a(m, x); a, b, c, p = inputs_3b(m, a3)
    h = F.relu(torch.cat([convs[0](a), convs[1](F.pad(b, (1, 1, 1, 1))), convs[2](F.pad(c, (2, 2, 2, 2))), convs[3](p)], 1))
    xb = pool3(h); pp = F.max_pool2d(F.pad(xb, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    lp = F.log_softmax(upper(m, [F.conv2d(xb if j < 3 else pp, oc.weight, oc.bias) for j, oc in enumerate(outs4)]), 1); top = lp.detach().exp().topk(10, 1)
    for r in range(2):
        eps = (torch.randint(0, 2, (10,), generator=gen) * 2 - 1).float().to(DEV)
        gs = torch.autograd.grad((eps * top.values[0].sqrt() * lp[0, top.indices[0]]).sum(), params, retain_graph=r == 0)
        gu = torch.cat([(gs[2 * k] ** 2).flatten(1).sum(1) + gs[2 * k + 1] ** 2 for k in range(4)]); gv = sum((g[:, :, 0, 0] ** 2).sum(0) for g in gs[8:])
        ds[i] += (gu - gv).detach().cpu().double().numpy() / 2
    if i % 200 == 0: print(f"  {i}/{len(pool)}", flush=True)
np.save("cache/n8_ds.npy", ds)
ev, od = np.arange(0, 480, 2), np.arange(1, 480, 2)
def tilt(cols):
    t = Dw[cols] / np.linalg.norm(Dw[cols]); X = ds[:, cols]; X = X / np.abs(X).mean(); Pd = X - np.outer(X @ t, t)
    f = lambda mu: (lambda a: (np.log(np.exp(a - a.max()).sum()) + a.max(), (np.exp(a - a.max()) / np.exp(a - a.max()).sum()) @ Pd))(Pd @ mu)
    r = minimize(f, np.zeros(len(cols)), jac=True, method="L-BFGS-B", options=dict(maxiter=2000))
    a = Pd @ r.x; w = np.exp(a - a.max()); return w / w.sum(), r
w, r = tilt(ev); ess = 1 / (w ** 2).sum()
u = np.ones(len(pool)) / len(pool)
for tag, ww in [("unweighted", u), ("tilted (mu fit on even neurons)", w)]:
    dq = ww @ ds; print(f"  {tag:32s}: Spearman even (fit) {spearmanr(Dw[ev], dq[ev])[0]:+.3f} | odd (held out) {spearmanr(Dw[od], dq[od])[0]:+.3f}")
print(f"  effective sample size {ess:.0f} of {len(pool)} ({ess / len(pool):.2f}); weight by source: inversions {w[:256].sum():.3f}, boundary {w[256:1280].sum():.3f}, reader pictures {w[1280:].sum():.3f}; dual converged {r.success}")
np.save("cache/n8_w.npy", w)
