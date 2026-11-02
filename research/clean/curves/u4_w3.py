"""BEAT W3 (card written 2026-10-22, first run now): derived write for k units on a FIXED read.
step0 = plain Fisher write (w_fisher.solve), step1 = one active-set step (W5, generalised to k units).
Simple-rule write: multivariate least squares of each reader's input on the k units (identity metric, 'lsq').
Read = M's trained read (microscope). usage: u4_w3.py <n images> [read tag]"""
import sys, json, numpy as np, torch
from inception import *
from family import Data, pool3
from w_fisher import load_build, real_kl, intact, A_cols, du, coarse, solve, PAD
from w5 import grads_post


def solve_active_k(m, fam, D, fm, idx, base, k, R=10, bs=50):
    A = A_cols(m, fam); sizes = [a.shape[0] for a in A]; P = sum(sizes) * (k + 1)
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
        if (i // bs) % 40 == 0: print(f"  active {i + len(b)}/{len(idx)}", flush=True)
    w = torch.linalg.solve(XtX + 1e-6 * torch.trace(XtX) / P * torch.eye(P, dtype=torch.float64), Xty).float()
    cols, cbs, o = [], [], 0
    for s in sizes:
        W = w[o:o + (k + 1) * s].reshape(s, k + 1); o += (k + 1) * s; cols.append(W[:, :k].to(DEV)); cbs.append(W[:, k].to(DEV))
    return cols, cbs, float(np.mean(woke))


def lsq_write(m, fam, D, fm, idx, k):
    """Identity-metric least squares: col = A Cov(z, c) Cov(c, c)^-1, bias = A mu_z - col mu_c (per branch)."""
    acc = [dict(S=0, SS=0, N=0) for _ in range(2)]
    with torch.no_grad():
        for i in range(0, len(idx), 100):
            ins = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i + 100]]))); _, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(fm, ins)
            for a, (z, c) in zip(acc, [(z1, c1), (z2, c2)]):
                x = torch.cat([z, c], 1).permute(0, 2, 3, 1).reshape(-1, 10 + k).cpu().double().numpy()
                a["S"] = a["S"] + x.sum(0); a["SS"] = a["SS"] + x.T @ x; a["N"] += len(x)
    A = A_cols(m, fam); cols, cbs = [], []
    for j in range(4):
        a = acc[1] if j == 3 else acc[0]; mu = a["S"] / a["N"]; C = a["SS"] / a["N"] - np.outer(mu, mu)
        Aj = A[j].cpu().double().numpy(); col = Aj @ C[:10, 10:] @ np.linalg.inv(C[10:, 10:])
        cols.append(torch.tensor(col, dtype=torch.float32, device=DEV)); cbs.append(torch.tensor(Aj @ mu[:10] - col @ mu[10:], dtype=torch.float32, device=DEV))
    return cols, cbs


if __name__ == "__main__":
    n = int(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "L-k3-m-lr3e-2"
    m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); fm = load_build(m, fam, tag); k = fm.k
    idx = np.sort(np.random.default_rng(12).choice(D.tr, n, replace=False))
    zc = [torch.zeros(c.shape[0], device=DEV) for c in fm.weights()[2]]
    B = {"trained write": (fm, fm.weights()[2], zc)}
    B["lsq (simple rule)"] = (fm, *lsq_write(m, fam, D, fm, idx[:1000], k))
    c0, b0, _ = solve(m, fam, D, fm, idx, 10, k); B["step0 Fisher"] = (fm, c0, b0)
    torch.save(([c.cpu() for c in c0], [b.cpu() for b in b0]), f"cache/w3_fisher_{tag}_{n}.pt")
    c1, b1, woke = solve_active_k(m, fam, D, fm, idx, (c0, b0), k); B["step1 active"] = (fm, c1, b1)
    torch.save(([c.cpu() for c in c1], [b.cpu() for b in b1]), f"cache/w3_active_{tag}_{n}.pt")
    print(f"woken share under step0: {woke:.4f}", flush=True)
    K = real_kl(m, fam, D, D.dv, B)
    for nm, v in K.items(): print(f"  {nm:18s} dev-{len(v)} KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})", flush=True)
    for nm in ["lsq (simple rule)", "step0 Fisher", "step1 active"]:
        d = K[nm] - K["trained write"]; print(f"  paired {nm} - trained write: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
    d = K["step1 active"] - K["lsq (simple rule)"]; print(f"  paired step1 - lsq: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
    np.savez(f"cache/w3_{tag}_{n}.npz", **{nm.split()[0]: v for nm, v in K.items()})
    json.dump({nm: float(v.mean()) for nm, v in K.items()} | dict(n=n, woke=woke), open(f"../../../results/curves-w3-{tag}-{n}.json", "w"), indent=1)
