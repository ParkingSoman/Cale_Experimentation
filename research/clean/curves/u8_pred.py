"""BEAT W3b diagnosis (W2e style): is the k = 3 write gap estimation (too few images) or beyond second order?
For writes on M's read: real KL, plain second-order prediction (pre-ReLU Fisher), and the active-set model's own
objective with the 4a ReLUs exact (post-ReLU Fisher on dh = relu(a) - relu(a - du)). On solve images (in-sample) and
dev images. Prediction in CURVES-LOG.md (BEAT, W3b diagnosis)."""
import numpy as np, torch
from inception import *
from family import Data, kl
from w_fisher import load_build, intact, A_cols, du, coarse, fisher_grads, upper
from w5 import grads_post
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); fm = load_build(m, fam, "L-k3-m-lr3e-2")
L = lambda p: [x.to(DEV) for x in p]
zc = [torch.zeros(c.shape[0], device=DEV) for c in fm.weights()[2]]
c0, b0 = torch.load("cache/w3_fisher_L-k3-m-lr3e-2_4000.pt"); c1, b1 = torch.load("cache/w3_active_L-k3-m-lr3e-2_4000.pt")
builds = {"trained": (fm.weights()[2], zc), "step0": (L(c0), L(b0)), "step1": (L(c1), L(b1))}
solve_idx = np.sort(np.random.default_rng(12).choice(D.tr, 4000, replace=False))
sets = {"in-sample": np.sort(np.random.default_rng(5).choice(solve_idx, 500, replace=False)),
        "dev": np.sort(np.random.default_rng(5).choice(D.dv, 500, replace=False))}
A = A_cols(m, fam)
for sname, idx in sets.items():
    acc = {k: dict(real=[], pre=[], post=[]) for k in builds}
    for i in range(0, len(idx), 25):
        b = idx[i:i + 25]
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); outs, z1, z2 = intact(m, fam, ins); t = upper(m, outs); cc1, cc2 = coarse(fm, ins)
        Gpre, _ = fisher_grads(m, outs, 10); Gpost = grads_post(m, outs, 10)
        with torch.no_grad():
            for k, (cols, cb) in builds.items():
                d = du(A, z1, z2, cols, cb, cc1, cc2)
                acc[k]["real"].append(kl(t, upper(m, [o - dj for o, dj in zip(outs, d)])).cpu().numpy())
                acc[k]["pre"].append((sum(sum((g[j] * d[j]).sum((1, 2, 3)) for j in range(4)) ** 2 for g in Gpre) / 2).cpu().numpy())
                dh = [F.relu(o) - F.relu(o - dj) for o, dj in zip(outs, d)]
                acc[k]["post"].append((sum(sum((g[j] * dh[j]).sum((1, 2, 3)) for j in range(4)) ** 2 for g in Gpost) / 2).cpu().numpy())
    print(f"\n{sname} ({len(idx)} images): build | real | pred pre-ReLU | pred post-ReLU (active-set model) | real / post")
    for k, a in acc.items():
        r, p, q = (np.concatenate(a[x]).mean() for x in ["real", "pre", "post"])
        print(f"  {k:8s} {r:.4f} | {p:.4f} | {q:.4f} | {r / q:.2f}", flush=True)
