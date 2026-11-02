"""R5' part 5 (Statement 16.1, Prop. 16.2): factorised coupled formula. H[(f,a),(g,b)] = E sum_{s,p,s',q}
Kbar[(s,f,p),(s',g,q)] z^s_a(p) z^s'_b(q); solve for Lambda (10 x 471). Kbar fixed (2000 images); activity moments from
the first 256 and all 2000 images (rng 5 set, rng 11 order). MIX-DERIVED (Kbar FITTED here as a check of 16.1)."""
import numpy as np, torch, time
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]
R = keep + [480]
Kb = torch.load("cache/x5_kbar.pt").reshape(2, 10, 196, 2, 10, 196).to(DEV)
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False)); order = np.random.default_rng(11).permutation(tr)
H = torch.zeros(10, 481, 10, 481, dtype=torch.float64); Hg = torch.zeros(10, 481, 10, 481, device=DEV); snap = None; t0 = time.time()
for i in range(0, 2000, 8):
    b = np.sort(order[i:i + 8])
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
        Z = torch.stack([torch.cat([t, torch.ones_like(t[:, :1])], 1).flatten(2).transpose(1, 2) for t in (x, p)], 1)   # (B, 2, 196, 481)
        for n in range(len(b)):
            T = torch.einsum("sfptgq,tqb->sfpgb", Kb, Z[n])                    # (2, 10, 196, 10, 481)
            Hg += torch.einsum("spa,sfpgb->fagb", Z[n], T)
    if (i // 8) % 8 == 7: H += Hg.cpu().double(); Hg.zero_()
    if i + 8 == 256: H += Hg.cpu().double(); Hg.zero_(); snap = (H.clone(), 256)
    if i == 0: print("first batch", round(time.time() - t0, 1), "s", flush=True)
H += Hg.cpu().double()
def solve(Hf, n):
    Hf = Hf / n; A = Hf[:, R][:, :, :, R].reshape(4710, 4710)
    v = sum(Hf[:, R, g, fam[g]] for g in range(10)).reshape(4710)
    return torch.linalg.solve(A + 1e-6 * torch.trace(A) / 4710 * torch.eye(4710, dtype=torch.float64), v).reshape(10, 471).float()
WF = [getattr(m, n).weight.detach()[:, fam, 0, 0].cpu() for n in IN4A]
def build(L):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j in range(4): fm.dW[j].copy_((WF[j] @ L[:, :470]) / fm.cscale[j]); fm.cbp[j].copy_((WF[j] @ L[:, 470]) / 0.1)
    return fm.to(DEV)
def per_image(fm, ix, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            bb = ix[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[bb]))))
            out.append(kl(torch.from_numpy(D.T[bb].astype(np.float32)).to(DEV), lg).cpu().numpy())
    return np.concatenate(out)
res = {}
for k, (Hf, n) in {"factor_n2000": (H, 2000), "factor_n256": snap}.items():
    fm = build(solve(Hf, n)); torch.save(fm.state_dict(), f"cache/x5f_{k}.pt"); res[k] = per_image(fm, D.dv)
    print(k, f"{res[k].mean():.4f} (se {res[k].std() / np.sqrt(len(res[k])):.4f})", flush=True)
ref = {"level0p": np.load("cache/x2_gated.npz")["b_level0_p"], "coupled35k": np.load("cache/x3c_pop.npz")["out_n35000"]}
np.savez("cache/x5f_factor.npz", **res)
for k in res:
    for r_, v in ref.items():
        d = res[k] - v; print(f"  {k} minus {r_:12s} {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})")
