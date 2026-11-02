"""One natural-gradient step from S on the uniform-metric per-image objective, gradient + curvature from 256 inputs
(network-made class inversions, or 256 real train images)."""
import numpy as np, torch
SRC21 = open("c21_newton.py").read(); exec(SRC21.split("vS = Vs[\"S\"] * mask")[0]); exec(SRC21[SRC21.index("def build(V):"):SRC21.index("g_all = marks[N]")])
src25 = open("c25_unigap.py").read(); exec(src25[src25.index("def reader_grads_uni"):src25.index("res = {")])
def fin(X):
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(X)))[0]))); p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    return x, p
family_inputs_saved = family_inputs
def gam(X):
    global family_inputs
    family_inputs = lambda b: fin(b); g = reader_grads_uni(X).reshape(len(X), 10, NRD, 196); x, p = fin(X); family_inputs = family_inputs_saved
    o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
    return torch.einsum("bcjp,bkp->bcjk", g, y) * mask
vS = Vs["S"] * mask; refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
Xn = np.load("cache/e1_self_256.npy"); Xr = D.X[np.sort(np.random.default_rng(26).choice(D.tr, 256, replace=False))]
for tag, X in [("real256", Xr), ("netmade256", Xn)]:
    g = torch.zeros(NRD, 962, device=DEV)
    for i in range(0, 256, 8):
        Gm = gam(X[i:i + 8]); g += torch.einsum("bc,bcjk->jk", (Gm * vS).sum((2, 3)), Gm)
    g = g * free / 256; d = -Pinv(g); curv = 0.0
    for i in range(0, 256, 8): curv += float(((gam(X[i:i + 8]) * d).sum((2, 3)) ** 2).sum())
    a = -float((g * d).sum()) / (curv / 256); fm = build(vS + a * d); r = per_image(fm); np.save(f"cache/c26_{tag}.npy", r)
    print(f"{tag}: step {a:.3g}; dev-5000 KL {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})", flush=True); paired(tag, r, {"kbar(S)": refs["kbar"], "T": refs["T"]})
