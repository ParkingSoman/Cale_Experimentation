"""Card W4: statistically linearized Fisher write. sigma_j^2 = Var(u_j)(1 - rho_j^2), the Bayes residual of each
reader (what the coarse neuron cannot predict). Exact-case check: sigma -> 0 gives the W2 gradients.
usage: w4.py <n images> [read tag]"""
import sys, json, numpy as np, torch
from inception import *
from family import Data, IN4A
from d4_write import stats
from w_fisher import load_build, solve, real_kl, intact, fisher_grads, fisher_grads_smooth, A_cols
n = int(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "derived2-r1-ks7"
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); fm = load_build(m, fam, tag)
st = stats(m, fm, fam, D); sig = []
for j, A in enumerate(A_cols(m, fam)):
    mu, C = st[1] if j == 3 else st[0]; A = A.cpu().double().numpy()
    covIc = A @ C[:10, 10]; varI = np.einsum("oi,ij,oj->o", A, C[:10, :10], A); rho2 = covIc ** 2 / (varI * C[10, 10] + 1e-12)
    sig.append(torch.tensor(np.sqrt(np.maximum(varI * (1 - rho2), 0)), dtype=torch.float32, device=DEV))
with torch.no_grad():
    outs, _, _ = intact(m, fam, inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:8])]))))
    print("sigma / sd(pre-activation) per conv (median):", [round(float((s / o.transpose(0, 1).reshape(o.shape[1], -1).std(1)).median()), 3) for s, o in zip(sig, outs)])
G0, _ = fisher_grads(m, outs, 3); G1, _ = fisher_grads_smooth(m, outs, 3, [s * 0 for s in sig])
print("exact case (sigma=0) max rel diff:", max(float((a - b).norm() / a.norm()) for g0, g1 in zip(G0, G1) for a, b in zip(g0, g1)), flush=True)
idx = np.sort(np.random.default_rng(12).choice(D.tr, n, replace=False))
cols, cbs, _ = solve(m, fam, D, fm, idx, 10, 1, bs=50, sig=sig)
torch.save(([c.cpu() for c in cols], [b.cpu() for b in cbs]), f"cache/fisher_smooth_{tag}_{n}.pt")
c0, b0 = torch.load(f"cache/fisher_write_{tag}_{n}.pt")
B = {"smooth": (fm, cols, cbs), "fisher": (fm, [x.to(DEV) for x in c0], [x.to(DEV) for x in b0])}
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:7s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})", flush=True)
d = K["smooth"] - K["fisher"]; print(f"  paired smooth - fisher: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
json.dump({k: float(v.mean()) for k, v in K.items()} | dict(n=n), open(f"../../../results/curves-w4-{tag}-{n}.json", "w"), indent=1)
