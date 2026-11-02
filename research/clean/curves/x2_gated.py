"""R2 (Statement 13.6, NEURON-MATH Part IV): level-1 formula, per-reader gated moments; paired dev-5000. MIX-DERIVED."""
import sys, numpy as np, torch
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
keep = [i for i in range(480) if i not in fam]; R = keep + [480]
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))
order = np.random.default_rng(11).permutation(tr); NSUB = 256
convs = [getattr(m, n) for n in IN4A]; outs = [c.weight.shape[0] for c in convs]
S0 = [torch.zeros(481, 481, dtype=torch.float64) for _ in range(2)]                 # level 0: x and p
A = [torch.zeros(o, 481, 481, dtype=torch.float64) for o in outs]; Asub = None
acc = [torch.zeros(o, 481, 481, device=DEV) for o in outs]
def flush():
    for j in range(4): A[j] += acc[j].cpu().double(); acc[j].zero_()
with torch.no_grad():
    for i in range(0, 2000, 16):
        b = np.sort(order[i:i + 16]); x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
        for t, inp in enumerate([x, p]):
            Z = torch.cat([inp.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(inp.shape[0] * 196, 1, device=DEV)], 1)
            S0[t] += (Z.T @ Z).cpu().double()
        for j, (c, inp) in enumerate(zip(convs, [x, x, x, p])):
            g = (F.conv2d(inp, c.weight, c.bias) > 0).float().permute(0, 2, 3, 1).reshape(-1, outs[j])
            Z = torch.cat([inp.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(inp.shape[0] * 196, 1, device=DEV)], 1)
            for r in range(0, outs[j], 16):
                acc[j][r:r + 16] += torch.einsum("pj,pa,pb->jab", g[:, r:r + 16], Z, Z)
        if (i // 16) % 8 == 7: flush()
        if i + 16 == NSUB: flush(); Asub = [a.clone() for a in A]
flush()
np.save("cache/x2_S0x.npy", S0[0].numpy()); np.save("cache/x2_S0p.npy", S0[1].numpy())
def solve(Sm, w):            # beta = (S_RR + ridge)^-1 S_RF w   for one reader row w (10,)
    Srr = Sm[R][:, R]; tr_ = torch.trace(Srr)
    if tr_ <= 0: return torch.zeros(471, dtype=torch.float64)
    return torch.linalg.solve(Srr + 1e-6 * tr_ / 471 * torch.eye(471, dtype=torch.float64), Sm[R][:, fam] @ w)
def build(per_reader):       # per_reader(j, r) -> moment matrix for reader r of conv j
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j, c in enumerate(convs):
            Wf = c.weight.detach()[:, fam, 0, 0].cpu().double()
            B = torch.stack([solve(per_reader(j, r), Wf[r]) for r in range(outs[j])]).float()
            fm.dW[j].copy_(B[:, :470] / fm.cscale[j]); fm.cbp[j].copy_(B[:, 470] / 0.1)
    return fm.to(DEV)
def per_image(fm, idx, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            out.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
    return np.concatenate(out)
builds = {"a_level0_x": lambda j, r: S0[0], "b_level0_p": lambda j, r: S0[1] if j == 3 else S0[0],
          "c_level1_n2000": lambda j, r: A[j][r], "d_level1_n256": lambda j, r: Asub[j][r]}
res = {}
for k, f in builds.items():
    fm = build(f); res[k] = per_image(fm, D.dv); torch.save(fm.state_dict(), f"cache/x2_{k}.pt")
    print(k, f"{res[k].mean():.4f} (se {res[k].std() / np.sqrt(len(res[k])):.4f})", flush=True)
np.savez("cache/x2_gated.npz", **res)
ref = res["a_level0_x"]
for k in res:
    d = res[k] - ref; print(f"{k:16s} minus level0  {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})")
for name in ["B3-k0-lr003-long", "B3-k0-formulastart"]:
    v = per_image(b3_train.load_b3(m, name), D.dv); d = res["c_level1_n2000"] - v
    print(f"level1 minus {name}: {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})  [{name} {v.mean():.4f}]")
gon = [float((A[j].diagonal(dim1=1, dim2=2)[:, 480] / S0[1 if j == 3 else 0][480, 480]).mean()) for j in range(4)]
print("mean on-fraction per conv:", np.round(gon, 3))
