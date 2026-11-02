"""X5: adversarial consistent mixtures with and without the imprint constraint; pool = network-made unsure + real unsure."""
import numpy as np, torch
from scipy.optimize import linprog
from scipy.special import softmax
import x3_counter as X
from inception import *
from w_fisher import real_kl, intact, coarse, fisher_grads, upper
from family import pool3, IN4A
m, M, A, D, fam, P = X.m, X.M, X.A, X.D, X.fam, X.P
PS = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (PS ** 2).sum(1) > .1]
used = np.r_[np.sort(np.random.default_rng(81).choice(uns, 2000, replace=False))]
cand = np.sort(np.random.default_rng(191).choice(np.setdiff1d(uns, used), 512, replace=False)); REAL = np.asarray(D.X[cand])
# energy vectors for the real candidates (same estimator as N8: 2 random-sign projections)
convs = [m.mixed3b_1x1_pre_relu_conv, m.mixed3b_3x3_pre_relu_conv, m.mixed3b_5x5_pre_relu_conv, m.mixed3b_pool_reduce_pre_relu_conv]; outs4 = [getattr(m, n) for n in IN4A]
params = [t for cc in convs for t in (cc.weight, cc.bias)] + [oc.weight for oc in outs4]
for t in params: t.requires_grad_(True)
gen = torch.Generator().manual_seed(0); dR = np.zeros((len(REAL), 480))
for i in range(len(REAL)):
    x = prep(REAL[i:i + 1]); a3 = to_3a(m, x); a, b, c, p = inputs_3b(m, a3)
    h = F.relu(torch.cat([convs[0](a), convs[1](F.pad(b, (1, 1, 1, 1))), convs[2](F.pad(c, (2, 2, 2, 2))), convs[3](p)], 1)); xb = pool3(h)
    pp = F.max_pool2d(F.pad(xb, (1, 1, 1, 1), value=float('-inf')), 3, 1); lp = F.log_softmax(upper(m, [F.conv2d(xb if j < 3 else pp, oc.weight, oc.bias) for j, oc in enumerate(outs4)]), 1); top = lp.detach().exp().topk(10, 1)
    for r in range(2):
        eps = (torch.randint(0, 2, (10,), generator=gen) * 2 - 1).float().to(DEV)
        gs = torch.autograd.grad((eps * top.values[0].sqrt() * lp[0, top.indices[0]]).sum(), params, retain_graph=r == 0)
        dR[i] += (torch.cat([(gs[2 * k] ** 2).flatten(1).sum(1) + gs[2 * k + 1] ** 2 for k in range(4)]) - sum((g[:, :, 0, 0] ** 2).sum(0) for g in gs[8:])).detach().cpu().double().numpy() / 2
for t in params: t.requires_grad_(False)
POOL = np.concatenate([X.pool[X.keep], REAL]); Dpool = np.concatenate([X.ds[X.keep], dR]) / X.sc; src = np.r_[np.zeros(X.keep.sum()), np.ones(len(REAL))]
R = []
for i in range(0, len(POOL), 25):
    with torch.no_grad(): ins = inputs_3b(m, to_3a(m, prep(POOL[i:i + 25]))); outs, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(M, ins)
    G, _ = fisher_grads(m, outs, 10); rs = torch.zeros(len(ins[0]), P, dtype=torch.float64)
    with torch.no_grad():
        for g in G:
            feats, y = [], 0.0
            for j in range(4):
                c = c1 if j < 3 else c2; z = z1 if j < 3 else z2
                feats.append(torch.einsum("bohw,bkhw->bok", g[j], torch.cat([c, torch.ones_like(c[:, :1])], 1)).reshape(len(ins[0]), -1)); y = y + torch.einsum("bohw,oi,bihw->b", g[j], A[j], z)
            Xf = torch.cat(feats, 1).cpu().double(); rs += Xf * (y.cpu().double() - Xf @ X.w1)[:, None]
    R.append(rs)
R = torch.cat(R).numpy(); v = torch.linalg.inv(X.reg(X.N1)).numpy() @ R.mean(0); obj = R @ v
B = {"w*(p1)": (M, *X.unpack(X.w1))}
for tag, imprint in [("C1 only", False), ("C1 + C2", True)]:
    kw = dict(A_ub=np.vstack([Dpool.T, -Dpool.T]), b_ub=np.full(960, 0.002)) if imprint else {}
    lp = linprog(-obj, A_eq=np.ones((1, len(POOL))), b_eq=[1.0], bounds=(0, None), method="highs", **kw); q = lp.x
    print(f"{tag}: LP status {lp.status}; support {np.sum(q > 1e-9)}; share real {q[src == 1].sum():.2f}; imprint residual {np.linalg.norm(q @ Dpool):.3f}", flush=True)
    Nq, mq = X.wsolve(POOL, q)
    for e in [1.0, 10.0]:
        N2, m2 = (X.N1 + e * Nq) / (1 + e), (X.m1 + e * mq) / (1 + e); w2 = torch.linalg.solve(X.reg(N2), m2); d = X.w1 - w2
        bound = 0.5 * float(d @ torch.linalg.solve(torch.linalg.inv(X.reg(X.N1)) + torch.linalg.inv(X.reg(N2)), d)); print(f"   eps {e}: bound {bound:.4f}", flush=True)
        B[f"{tag}, eps {e}"] = (M, *X.unpack(w2))
K = real_kl(m, fam, D, D.dv, B); base = K["w*(p1)"]
for k, vv in K.items(): dd = vv - base; print(f"  {k:18s} dev-5000 KL {vv.mean():.4f} | paired vs w*(p1) {dd.mean():+.4f} (se {dd.std() / np.sqrt(len(dd)):.4f})")
