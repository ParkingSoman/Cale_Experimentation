"""R2c: where does the coupling live? Split the exact second-order KL term L3 = 1/2 Var_{c~p}(J_c e) into blocks, with
exact Jacobian rows for the top-K classes. Same 500 dev images as R2b."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A
exec(open("x2b_reason.py").read().split("acc = {k:")[0].split("names = {")[0])          # m, D, idx, convs, cont, reader_pre
fam = list(np.load("cache/s1_fine.npz")["fam"]); K = 10
def load(v):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    if v.startswith("x2_") or v.startswith("x3c_"): fm.load_state_dict(torch.load(f"cache/{v}.pt", map_location="cpu")); return fm.to(DEV)
    return b3_train.load_b3(m, v)
builds = {"level0": load("x2_a_level0_x"), "coupled35k": load("x3c_out_n35000"), "trained_from_level0": load("B3-k0-formulastart")}
Q = ["diag", "spatial", "position", "full"]; acc = {k: {q: [] for q in Q} for k in builds}; mass = []
for i in range(0, len(idx), 20):
    b = idx[i:i + 20]
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1); a0 = reader_pre(x, p)
        E = {k: torch.cat([(u - v).flatten(2) for u, v in zip(a0, reader_pre(x, p, fm))], 1) for k, fm in builds.items()}  # (B, 368, 196)
    pre = [t.clone().requires_grad_(True) for t in a0]; lg = cont([F.relu(t) for t in pre])
    pr = lg.detach().softmax(1); top = pr.topk(K, 1); pk = top.values / top.values.sum(1, keepdim=True); mass.append(top.values.sum(1).cpu().numpy())
    J = []
    for c in range(K):
        gr = torch.autograd.grad(lg.gather(1, top.indices[:, c:c + 1]).sum(), pre, retain_graph=c < K - 1)
        J.append(torch.cat([t.flatten(2) for t in gr], 1))
    J = torch.stack(J, 1)                                                    # (B, K, 368, 196)
    with torch.no_grad():
        for k in builds:
            T = J * E[k][:, None]                                            # (B, K, 368, 196): J_c,u e_u
            var = lambda s: (pk.view(*pk.shape, *([1] * (s.dim() - 2))) * s ** 2).sum(1) - (pk.view(*pk.shape, *([1] * (s.dim() - 2))) * s).sum(1) ** 2
            acc[k]["diag"].append(0.5 * var(T).flatten(1).sum(1).cpu().numpy())
            acc[k]["spatial"].append(0.5 * var(T.sum(3)).sum(1).cpu().numpy())
            acc[k]["position"].append(0.5 * var(T.sum(2)).sum(1).cpu().numpy())
            acc[k]["full"].append(0.5 * var(T.sum((2, 3))).cpu().numpy())
res = {k: {q: np.concatenate(v) for q, v in d.items()} for k, d in acc.items()}
np.savez("cache/x3d_blocks.npz", **{f"{k}_{q}": v for k, d in res.items() for q, v in d.items()})
print("top-K mass covered: mean", round(float(np.concatenate(mass).mean()), 3), " median", round(float(np.median(np.concatenate(mass))), 3))
fd = np.load("cache/x2b_reason_exact.npz")
for k, d in res.items(): print(f"{k:20s} " + "  ".join(f"{q} {d[q].mean():.4f}" for q in Q) + f"")
for a in ["trained_from_level0", "coupled35k"]:
    dd = {q: res[a][q] - res["level0"][q] for q in Q}
    print(f"{a} minus level0: " + "  ".join(f"{q} {v.mean():+.4f} ({v.std() / np.sqrt(len(v)):.4f})" for q, v in dd.items()))
