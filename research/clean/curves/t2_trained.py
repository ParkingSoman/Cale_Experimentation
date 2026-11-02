"""T2: Theorem 10 against the trained k = 1 neuron. Prediction in CURVES-LOG.md (MIX, T2)."""
import numpy as np, torch
from inception import *
from family import IN4A, pool3
from w_fisher import load_build, upper, PAD
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
d = np.load("cache/t1_C.npz"); A = d["A"]; C = d["C_gen"]; At = [torch.tensor(getattr(m, n).weight.detach()[:, fam, 0, 0].cpu().numpy()) for n in IN4A]
r = np.random.default_rng(43); ix = np.sort(r.choice(tr, 500, replace=False)); ixf = np.sort(r.choice(tr, 1000, replace=False))
np.set_printoptions(precision=3, suppress=True, linewidth=150)
# child-level Fisher Phi (10 x 10)
Phi = np.zeros((10, 10)); npos = 0
for i in range(0, len(ixf), 25):
    with torch.no_grad():
        h = F.relu(pre_3b(m, to_3a(m, prep(X[ixf[i:i + 25]])))); x = pool3(h)
    zf = x[:, fam].clone().requires_grad_(); keep = [j for j in range(480) if j not in fam]
    xx = torch.zeros_like(x); xx = xx.index_copy(1, torch.tensor(keep, device=DEV), x[:, keep]).index_copy(1, torch.tensor(fam, device=DEV), zf)
    p = PAD(xx); outs = [F.conv2d(xx if j < 3 else p, getattr(m, n).weight, getattr(m, n).bias) for j, n in enumerate(IN4A)]
    lp = F.log_softmax(upper(m, outs), 1); top = lp.detach().exp().topk(10, 1)
    for k in range(10):
        g = torch.autograd.grad((top.values[:, k].sqrt() * lp.gather(1, top.indices[:, k:k + 1])[:, 0]).sum(), zf, retain_graph=k < 9)[0]
        g = g[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(10, -1).cpu().double().numpy(); Phi += g @ g.T
    npos += 25 * 100
Phi /= npos
def top_lam(M):
    ev, V = np.linalg.eig(M); v = V[:, np.argmax(ev.real)].real; v = v * np.sign(v.sum()); return v / np.abs(v).sum()
lam_I, lam_F = top_lam(A.T @ A @ C), top_lam(Phi @ C)
print("Phi normalised:\n", Phi / np.sqrt(np.outer(np.diag(Phi), np.diag(Phi))))
print("lam identity", lam_I, "\nlam Fisher  ", lam_F)
low = [fam.index(n) for n in (385, 406, 379)]; hi = [j for j in range(10) if j not in low]
for tag in ["L-k1-r-lr3e-2", "fam-curves-k1-lr0.03-s0-random"]:
    fm = load_build(m, fam, tag); Z, Cc = [], []
    with torch.no_grad():
        for i in range(0, len(ix), 50):
            ins = inputs_3b(m, to_3a(m, prep(X[ix[i:i + 50]]))); h = F.relu(pre_3b(m, ins[0]))
            Z.append(pool3(h[:, fam])[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(10, -1).cpu().numpy())
            Cc.append(pool3(F.relu(fm.pre(ins)))[:, 0, 2:12, 2:12].reshape(-1).cpu().numpy())
    Z = np.concatenate(Z, 1).T.astype(np.float64); c = np.concatenate(Cc).astype(np.float64)
    D = np.concatenate([Z, np.ones((len(Z), 1))], 1); w = np.linalg.lstsq(D, c, rcond=None)[0]; r2 = 1 - ((c - D @ w) ** 2).mean() / c.var()
    lt = w[:10] / np.abs(w[:10]).sum()
    print(f"\n{tag}: lam_trained {lt} | R^2 {r2:.2f} | low-3 / high-7 mean ratio {lt[low].mean() / lt[hi].mean():.2f}")
    print(f"   corr with lam identity {np.corrcoef(lt, lam_I)[0, 1]:.2f}, with lam Fisher {np.corrcoef(lt, lam_F)[0, 1]:.2f}, with uniform-ish (std) {lt.std():.3f}")
    cols = fm.weights()[2]
    for lamv, nm in [(lam_I, "identity"), (lam_F, "Fisher")]:
        b = [a.numpy() @ (C @ lamv) for a in At]
        print(f"   write cos (trained col, A C lam {nm}) per conv:", np.array([np.dot(cc[:, 0].cpu().numpy(), bb) / np.linalg.norm(cc[:, 0].cpu().numpy()) / np.linalg.norm(bb) for cc, bb in zip(cols, b)]))
np.savez("cache/t2.npz", Phi=Phi, lam_I=lam_I, lam_F=lam_F)
