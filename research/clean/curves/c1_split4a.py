"""I1 (Prop. 18.5 transport identity) and A1-A3 (split of Gap A at 4a). 2000 train images = Kbar's images (rng 5).
Variants of K = E A^T Sigma A (gates on the 3x3 / 5x5 branches; Sigma = Cov_c of logit gradients at 4a outputs); each
gets the factorised build (circ.factor_build) and paired dev-5000 KL. FITTED microscopes."""
import numpy as np, torch, time
from circ import *

t0 = time.time()
# I1, part 1: 20 images, direct gradients at the reader outputs (as x5_kbar) vs transport from the 4a outputs
b = TR[:20]; x, p = family_inputs(b); post = [t.requires_grad_(True) for t in readers_post(x, p)]
a_, b_, c_, d_ = post
H = torch.cat([a_, F.relu(C3(F.pad(b_, (1, 1, 1, 1)))), F.relu(C5(F.pad(c_, (2, 2, 2, 2)))), d_], 1); lg = rest_from_4a(H)
top = lg.detach().softmax(1).topk(10, 1); pk = top.values / top.values.sum(1, keepdim=True); gd = []
for c in range(10):
    gr = torch.autograd.grad(lg.gather(1, top.indices[:, c:c + 1]).sum(), post, retain_graph=c < 9)
    gx = sum(torch.einsum("of,bopq->bfpq", WF[j], gr[j]) for j in range(3)); gp = torch.einsum("of,bopq->bfpq", WF[3], gr[3])
    gd.append(torch.cat([gx, gp], 1).reshape(20, 3920))
gd = torch.stack(gd, 1).detach()
post0 = [t.detach() for t in post]; G3, G5 = gates(post0); JH, pk2 = grads_4a(post0)
gt = transport(JH, G3, G5)
Kd_ = centred(gd, pk).T @ centred(gd, pk); Kt_ = centred(gt, pk2).T @ centred(gt, pk2)
print(f"I1 (20 images, float32): |g_t - g_d|/|g_d| = {float((gt - gd).norm() / gd.norm()):.2e};  |K_t - K_d|/|K_d| = {float((Kt_ - Kd_).norm() / Kd_.norm()):.2e}", flush=True)

# pass 1: firing rates of the 3x3 / 5x5 outputs
r3 = torch.zeros(N3, 14, 14, device=DEV); r5 = torch.zeros(N5, 14, 14, device=DEV)
for i in range(0, 2000, 50):
    x, p = family_inputs(TR[i:i + 50]); g3, g5 = gates(readers_post(x, p)); r3 += g3.sum(0); r5 += g5.sum(0)
r3 /= 2000; r5 /= 2000
print(f"rates: 3x3 mean {float(r3.mean()):.3f}, 5x5 mean {float(r5.mean()):.3f}  ({time.time() - t0:.0f}s)", flush=True)

# pass 2: full per-image transports (exact gates x, rate gates d, no gates g) + diagonal weights
acc = {k: torch.zeros(3920, 3920, device=DEV) for k in "xdg"}; tot = {k: torch.zeros(3920, 3920, dtype=torch.float64) for k in "xdg"}
S1 = torch.zeros(508, 14, 14, device=DEV); SD = torch.zeros(508, 14, 14, device=DEV)
one3, one5 = torch.ones(1, N3, 14, 14, device=DEV), torch.ones(1, N5, 14, 14, device=DEV)
for i in range(0, 2000, 20):
    x, p = family_inputs(TR[i:i + 20]); post = readers_post(x, p); G3, G5 = gates(post); JH, pk = grads_4a(post)
    with torch.no_grad():
        Jc = (JH - (pk[:, :, None, None, None] * JH).sum(1, keepdim=True)); sig = (pk[:, :, None, None, None] * Jc ** 2).sum(1)
        S1 += sig.sum(0); Dfull = torch.cat([torch.ones_like(sig[:, :N1]), G3, G5, torch.ones_like(sig[:, N1 + N3 + N5:])], 1); SD += (sig * Dfull).sum(0)
        for k, (g3, g5) in {"x": (G3, G5), "d": (r3[None], r5[None]), "g": (one3, one5)}.items():
            cg = centred(transport(JH, g3, g5), pk); acc[k] += cg.T @ cg
    if (i // 20) % 10 == 9:
        for k in acc: tot[k] += acc[k].cpu().double(); acc[k].zero_()
for k in acc: tot[k] += acc[k].cpu().double(); tot[k] /= 2000
Kbar = torch.load("cache/x5_kbar.pt").double()
print(f"I1 (2000 images): |K_x - Kbar(cached)|/|Kbar| = {float((tot['x'] - Kbar).norm() / Kbar.norm()):.2e}  ({time.time() - t0:.0f}s)", flush=True)

S1 = (S1 / 2000).cpu().double(); SD = (SD / 2000).cpu().double(); R3, R5 = r3.cpu().double(), r5.cpu().double()
blocks = lambda t: {"1x1": t[:N1], "3x3": t[N1:N1 + N3], "5x5": t[N1 + N3:N1 + N3 + N5], "pool": t[N1 + N3 + N5:]}
ones = torch.ones(508, 14, 14, dtype=torch.float64); rate2 = torch.cat([torch.ones(N1, 14, 14, dtype=torch.float64), R3 ** 2, R5 ** 2, torch.ones(NP, 14, 14, dtype=torch.float64)])
Ks = {"a": diag_metric(blocks(ones)), "b": diag_metric(blocks(rate2)), "c": diag_metric(blocks(S1)),
      "e": diag_metric(blocks(rate2 * S1)), "f": diag_metric(blocks(SD)), "g": tot["g"], "d": tot["d"], "x": tot["x"]}
torch.save({k: v.float() for k, v in Ks.items()}, "cache/c1_K.pt"); torch.save(dict(S1=S1, SD=SD, r3=R3, r5=R5), "cache/c1_w.pt")
print(f"metrics built ({time.time() - t0:.0f}s)", flush=True)
res = {}; refs = REFS()
for k in "abcefgdx":
    res[k] = factor_build(Ks[k], f"c1_build_{k}"); paired(k, res[k], refs)
np.savez("cache/c1_split4a.npz", **res)
for u, v in [("f", "x"), ("d", "x"), ("c", "e"), ("g", "x")]:
    dd = res[u] - res[v]; print(f"  {u} minus {v}: {dd.mean():+.4f} (paired se {dd.std() / np.sqrt(len(dd)):.4f})")
print(f"done ({time.time() - t0:.0f}s)")
