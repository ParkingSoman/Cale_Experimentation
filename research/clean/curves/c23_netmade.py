"""Gate patterns from network-made inputs: K = E_x A_x^T (I - 11^T/n) A_x over the 256 class inversions (DERIVED: weights +
fixed generation rule), carried to the logits. Control: 256 real train images. Activity moments as before."""
import numpy as np, torch, time
src = open("c20_depth.py").read().split("# pass 1: rates")[0]; exec(src)
def fam_in(X):
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(X)))[0]))); p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    return x, p
def K_from(X, rep=40):
    K = torch.zeros(3920, 3920, device=DEV)
    for i in range(0, len(X), 4):
        x, p = fam_in(X[i:i + 4]); rec, _ = record(x, p); B = len(x)
        xi = (lambda z: z - z.mean(1, keepdim=True))(torch.randn(B * rep, 1008, device=DEV))
        G, P = lin_true(rec, rep); g = fam_grads(G, P, xi); K += g.T @ g
    return (K / len(X)).cpu().double()
Xn = np.load("cache/e1_self_256.npy"); Xr = D.X[np.sort(np.random.default_rng(23).choice(TR, 256, replace=False))]
refs = REFS(); refs["4d"] = np.load("cache/c8_ladder_4b_4d_5b_logits.npz")["4d"]; refs["a3"] = np.load("cache/c20_depth_a1_a2_a3_b_b_uni.npz")["a3"]
for tag, X in [("real256", Xr), ("netmade256", Xn)]:
    K = K_from(X); torch.save(K.float(), f"cache/c23_K_{tag}.pt")
    e = torch.linalg.eigvalsh(K).clamp(min=0); pe = e / e.sum(); print(f"[{tag}] effective rank {float(torch.exp(-(pe[pe > 0] * pe[pe > 0].log()).sum())):.0f}", flush=True)
    r = factor_build(K, f"c23_build_{tag}"); paired(tag, r, refs); np.save(f"cache/c23_{tag}.npy", r)
print("done")
