"""E9: the WHAT (children weights of the 3 coarse units, section 13) from network-made inputs only.
Phi (child-level output Fisher) from the boundary inputs + mirrors; Sigma, mu (pooled children) from inversions + boundary.
Then the contiguous 3-arc partition search of u5_what.py. usage: e9_what.py <boundary npy>"""
import sys, itertools, numpy as np, torch
from scipy.optimize import minimize
from inception import *
from family import IN4A, pool3
from w_fisher import upper, PAD
m = load_model(); fam = list(np.load("cache/s1_fine.npz")["fam"])
for p in m.parameters(): p.requires_grad_(False)
Bx = np.load(sys.argv[1]); Bx = np.concatenate([Bx, Bx[:, :, ::-1]]); INV = np.load("cache/e1_self_256.npy")
keep = torch.tensor([j for j in range(480) if j not in fam], device=DEV); famt = torch.tensor(fam, device=DEV)
Phi = np.zeros((10, 10)); npos = 0
for i in range(0, len(Bx), 25):
    with torch.no_grad(): x = pool3(F.relu(pre_3b(m, to_3a(m, prep(Bx[i:i + 25])))))
    zf = x[:, fam].clone().requires_grad_(); xx = torch.zeros_like(x).index_copy(1, keep, x[:, keep]).index_copy(1, famt, zf)
    p = PAD(xx); outs = [F.conv2d(xx if j < 3 else p, getattr(m, n).weight, getattr(m, n).bias) for j, n in enumerate(IN4A)]
    lp = F.log_softmax(upper(m, outs), 1); top = lp.detach().exp().topk(10, 1)
    for k in range(10):
        g = torch.autograd.grad((top.values[:, k].sqrt() * lp.gather(1, top.indices[:, k:k + 1])[:, 0]).sum(), zf, retain_graph=k < 9)[0]
        g = g[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(10, -1).cpu().double().numpy(); Phi += g @ g.T
    npos += x.shape[0] * 100
Phi /= npos
flat = lambda t: t[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(t.shape[1], -1).cpu().double().numpy(); Z = []
with torch.no_grad():
    for X in (INV, Bx):
        for i in range(0, len(X), 50): Z.append(flat(pool3(F.relu(pre_3b(m, to_3a(m, prep(X[i:i + 50])))))[:, fam]))
Z = np.concatenate(Z, 1); S = np.cov(Z, bias=True); mu = Z.mean(1); C = S + np.outer(mu, mu); tot = np.trace(Phi @ C)
real = np.load("cache/t4.npz"); Phi_r = np.load("cache/t2.npz")["Phi"]; iu = np.triu_indices(10, 1)
cn = lambda A: A / np.sqrt(np.outer(np.diag(A), np.diag(A)))
print(f"network-made vs real: Phi pattern corr {np.corrcoef(cn(Phi)[iu], cn(Phi_r)[iu])[0, 1]:+.3f}, Sigma pattern corr {np.corrcoef(cn(S)[iu], cn(real['Szz'])[iu])[0, 1]:+.3f}")
def share(L): return np.trace(np.linalg.solve(L @ S @ L.T, L @ S @ Phi @ S @ L.T)) / tot
def best_on(blocks, starts=4, seed=0):
    rng = np.random.default_rng(seed); best = (-1, None); n = sum(len(b) for b in blocks)
    def unpack(x):
        L = np.zeros((3, 10)); o = 0
        for j, b in enumerate(blocks): L[j, b] = x[o:o + len(b)] ** 2; o += len(b)
        return L
    for s in range(starts):
        x0 = np.ones(n) if s == 0 else rng.uniform(.2, 1.5, n); r = minimize(lambda x: -share(unpack(x) + 1e-9), x0, method="L-BFGS-B")
        if -r.fun > best[0]: best = (-r.fun, unpack(r.x))
    return best
rows = []
for cuts in itertools.combinations(range(10), 3):
    blocks = [[(cuts[j] + t) % 10 for t in range((cuts[(j + 1) % 3] - cuts[j]) % 10 or 10)] for j in range(3)]
    if sorted(sum(blocks, [])) != list(range(10)): continue
    eq = np.zeros((3, 10))
    for j, b in enumerate(blocks): eq[j, b] = 1
    rows.append((share(eq), blocks))
rows.sort(key=lambda r: -r[0]); res = sorted([(best_on(b)[0], b, best_on(b)[1]) for _, b in rows[:25]], key=lambda r: -r[0])
ANG = [10, 50, 70, 120, 170, 180, 250, 290, 320, 350]
for s_opt, b, L in res[:4]: print(f"  share {s_opt:.3f} arcs {[[ANG[i] for i in bb] for bb in b]}")
old = np.load("cache/u5_what.npz")["top_L"][1]; print("real-statistics arcs #2 (used so far):", [[ANG[i] for i in np.where(r > 1e-6)[0]] for r in old])
np.savez("cache/e9_what.npz", top_L=np.stack([r[2] for r in res[:4]]), Phi=Phi, S=S, mu=mu)
