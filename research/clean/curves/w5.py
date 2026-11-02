"""Card W5: active-set Fisher write. With the set of readers that are ON after the change fixed (from the current
write), dh = on_new * (v - u) + (on_new * a - relu(a)) is linear in the write; the second-order damage at the 4a
outputs (post-ReLU Fisher) is then quadratic: one solve. Repeat with the new set.
usage: w5.py <n images> <steps> [read tag]"""
import sys, json, numpy as np, torch
from inception import *
from family import Data
from w_fisher import load_build, real_kl, intact, A_cols, du, coarse, upper_post


def grads_post(m, outs, R):
    hs = [F.relu(o).detach().requires_grad_() for o in outs]
    lp = F.log_softmax(upper_post(m, hs), 1); top = lp.detach().exp().topk(R, 1); G = []
    for r in range(R):
        g = torch.autograd.grad((top.values[:, r].sqrt() * lp.gather(1, top.indices[:, r:r + 1])[:, 0]).sum(), hs, retain_graph=r < R - 1)
        G.append([t.detach() for t in g])
    return G


def solve_active(m, fam, D, fm, idx, base, R=10, bs=50):
    A = A_cols(m, fam); sizes = [a.shape[0] for a in A]; P = sum(sizes) * 2
    XtX = torch.zeros(P, P, dtype=torch.float64); Xty = torch.zeros(P, dtype=torch.float64); woke = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); outs, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(fm, ins)
            d = du(A, z1, z2, base[0], base[1], c1, c2)
            on_new = [((o - dj) > 0).float() for o, dj in zip(outs, d)]
            kconst = [on * o - F.relu(o) for on, o in zip(on_new, outs)]
            woke.append(float(sum(((o < 0) & (on > 0)).float().sum() for o, on in zip(outs, on_new)) / sum((o < 0).float().sum() for o in outs)))
        G = grads_post(m, outs, R)
        with torch.no_grad():
            for g in G:
                feats, y = [], 0.0
                for j in range(4):
                    c = c1 if j < 3 else c2; z = z1 if j < 3 else z2; ge = g[j] * on_new[j]
                    feats.append(torch.einsum("bohw,bkhw->bok", ge, torch.cat([c, torch.ones_like(c[:, :1])], 1)).reshape(len(b), -1))
                    y = y + torch.einsum("bohw,oi,bihw->b", ge, A[j], z) - (g[j] * kconst[j]).sum((1, 2, 3))
                X = torch.cat(feats, 1).cpu().double(); yy = y.cpu().double(); XtX += X.T @ X; Xty += X.T @ yy
        if (i // bs) % 40 == 0: print(f"  solve {i + len(b)}/{len(idx)}", flush=True)
    w = torch.linalg.solve(XtX + 1e-6 * torch.trace(XtX) / P * torch.eye(P, dtype=torch.float64), Xty).float()
    cols, cbs, o = [], [], 0
    for s in sizes:
        W = w[o:o + 2 * s].reshape(s, 2); o += 2 * s; cols.append(W[:, :1].to(DEV)); cbs.append(W[:, 1].to(DEV))
    return cols, cbs, float(np.mean(woke))


if __name__ == "__main__":
    n, steps = int(sys.argv[1]), int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else "derived2-r1-ks7"
    m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); fm = load_build(m, fam, tag)
    c0, b0 = torch.load(f"cache/fisher_write_{tag}_{n}.pt"); cur = ([x.to(DEV) for x in c0], [x.to(DEV) for x in b0])
    idx = np.sort(np.random.default_rng(12).choice(D.tr, n, replace=False)); B = {"step0": (fm, *cur)}; hist = []
    for s in range(1, steps + 1):
        cols, cbs, woke = solve_active(m, fam, D, fm, idx, cur)
        print(f"step {s}: woken share under the previous write {woke:.4f}", flush=True)
        cur = (cols, cbs); B[f"step{s}"] = (fm, cols, cbs); hist.append(woke)
        torch.save(([c.cpu() for c in cols], [b.cpu() for b in cbs]), f"cache/fisher_active_{tag}_{n}_s{s}.pt")
    K = real_kl(m, fam, D, D.dv, B)
    for k, v in K.items(): print(f"  {k:6s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})", flush=True)
    for s in range(1, steps + 1):
        d = K[f"step{s}"] - K["step0"]; print(f"  paired step{s} - step0: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
    json.dump({k: float(v.mean()) for k, v in K.items()} | dict(n=n, woke=hist), open(f"../../../results/curves-w5-{tag}-{n}.json", "w"), indent=1)
