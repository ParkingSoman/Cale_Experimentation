"""BEAT W3 at population scale: the same write (step0 plain Fisher, step1 one active-set step), estimated on many more
images. Estimator change only: the Fisher sum over the top-R classes is replaced by ONE backward per image of
sum_r eps_r sqrt(p_r) dlog p_r (eps = +-1 at random), whose outer product has expectation exactly sum_r p_r g_r g_r^T.
Same population moments, ~R x cheaper per image. Two reads share every gradient: M's trained read and the derived
arcs #2 read (u6_how.py). usage: u9_write.py <n images>"""
import sys, json, time, numpy as np, torch
from inception import *
from family import Data, Family
from w_fisher import load_build, real_kl, intact, A_cols, du, coarse, upper, upper_post
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); R = 10
M = load_build(m, fam, "L-k3-m-lr3e-2"); h = torch.load("cache/u6_how.pt", weights_only=False); ci = h["names"].index("arcs #2")
V = h["V"][3 * ci:3 * ci + 3].to(DEV); Ar = Family(m, fam, 3).to(DEV)
Ar.set_weights(V[:, :800].reshape(3, 32, 5, 5), V[:, 800], [c[:, :3] for c in M.weights()[2]])
Kp = Family(m, fam, 3, init="member:330,388,385").to(DEV)
reads = {"M": M, "arcs": Ar, "kept": Kp}; A = A_cols(m, fam); sizes = [a.shape[0] for a in A]; P = sum(sizes) * 4


def proj_grad(lp, wrt, gen):
    top = lp.detach().exp().topk(R, 1); eps = (torch.randint(0, 2, top.values.shape, generator=gen) * 2 - 1).float().to(DEV)
    s = (eps * top.values.sqrt() * lp.gather(1, top.indices)).sum()
    return [t.detach() for t in torch.autograd.grad(s, wrt)]


def pass_(idx, base=None, bs=50, seed=0):
    gen = torch.Generator().manual_seed(seed); acc = {k: [torch.zeros(P, P, dtype=torch.float64), torch.zeros(P, dtype=torch.float64)] for k in reads}
    t0 = time.time()
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); outs, z1, z2 = intact(m, fam, ins); cs = {k: coarse(fm, ins) for k, fm in reads.items()}
        if base is None:
            o = [x.detach().requires_grad_() for x in outs]; g = proj_grad(F.log_softmax(upper(m, o), 1), o, gen)
        else:
            hs = [F.relu(x).detach().requires_grad_() for x in outs]; g = proj_grad(F.log_softmax(upper_post(m, hs), 1), hs, gen)
        with torch.no_grad():
            for k in reads:
                c1, c2 = cs[k]; feats, y = [], 0.0
                if base is not None:
                    d = du(A, z1, z2, base[k][0], base[k][1], c1, c2); on = [((o - dj) > 0).float() for o, dj in zip(outs, d)]
                for j in range(4):
                    c = c1 if j < 3 else c2; z = z1 if j < 3 else z2; ge = g[j] if base is None else g[j] * on[j]
                    feats.append(torch.einsum("bohw,bkhw->bok", ge, torch.cat([c, torch.ones_like(c[:, :1])], 1)).reshape(len(b), -1))
                    y = y + torch.einsum("bohw,oi,bihw->b", ge, A[j], z)
                    if base is not None: y = y - (g[j] * (on[j] * outs[j] - F.relu(outs[j]))).sum((1, 2, 3))
                X = torch.cat(feats, 1).cpu().double(); yy = y.cpu().double(); acc[k][0] += X.T @ X; acc[k][1] += X.T @ yy
        if (i // bs) % 50 == 0: print(f"  pass {'step0' if base is None else 'step1'} {i + len(b)}/{len(idx)}  {time.time() - t0:.0f} s", flush=True)
    out = {}
    for k, (XtX, Xty) in acc.items():
        w = torch.linalg.solve(XtX + 1e-6 * torch.trace(XtX) / P * torch.eye(P, dtype=torch.float64), Xty).float(); cols, cbs, o = [], [], 0
        for s in sizes:
            W = w[o:o + 4 * s].reshape(s, 4); o += 4 * s; cols.append(W[:, :3].to(DEV)); cbs.append(W[:, 3].to(DEV))
        out[k] = (cols, cbs)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]); idx = np.sort(np.random.default_rng(12).choice(D.tr, n, replace=False))
    w0 = pass_(idx, seed=1); w1 = pass_(idx, base=w0, seed=2)
    torch.save({k: ([c.cpu() for c in w1[k][0]], [b.cpu() for b in w1[k][1]]) for k in w1}, f"cache/u9_write_{n}.pt")
    zc = [torch.zeros(c.shape[0], device=DEV) for c in M.weights()[2]]
    B = {"M read + trained write": (M, M.weights()[2], zc)}
    for k in reads:
        B[f"{k} read + step0"] = (reads[k], *w0[k]); B[f"{k} read + step1"] = (reads[k], *w1[k])
    K = real_kl(m, fam, D, D.dv, B); ref = K["M read + trained write"]
    for nm, v in K.items():
        d = v - ref; print(f"  {nm:24s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f}) | paired vs bar B {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
    np.savez(f"cache/u9_kl_{n}.npz", **{nm.replace(' ', '_'): v for nm, v in K.items()})
    json.dump({nm: float(v.mean()) for nm, v in K.items()} | dict(n=n), open(f"../../../results/curves-u9-{n}.json", "w"), indent=1)
