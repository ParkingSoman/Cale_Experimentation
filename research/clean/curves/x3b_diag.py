"""Loop step 8 for R3': is the coupled formula's failure a bug or R-flat (non-coercive second-order objective)?"""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A
exec(open("x2b_reason.py").read().split("names = {")[0])          # m, D, idx, convs, cont, reader_pre
fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]; R = keep + [480]; K = 10
st = torch.load("cache/x3_coupled_stats.pt"); Mp, vp = st["M"]["pre"], st["v"]["pre"]
S0 = torch.from_numpy(np.load("cache/x2_S0x.npy")); Srr = S0[R][:, R]
L0 = torch.linalg.solve(Srr + 1e-6 * torch.trace(Srr) / 471 * torch.eye(471, dtype=torch.float64), S0[R][:, fam]).T     # (10, 471)
Lc = torch.linalg.solve(Mp + 1e-6 * torch.trace(Mp) / 4710 * torch.eye(4710, dtype=torch.float64), vp).reshape(10, 471)
Q = lambda L: 0.5 * (L.flatten() @ Mp @ L.flatten() - 2 * vp @ L.flatten()) / 2000
print(f"(i) stats: Q(coupled) - Q(level0) = {float(Q(Lc) - Q(L0)):+.4f}")
ev = torch.linalg.eigvalsh(Mp); print(f"(ii) M (pre-J) eigenvalues > 1e-6 max: {int((ev > 1e-6 * ev.max()).sum())} of 4710; > 1e-3 max: {int((ev > 1e-3 * ev.max()).sum())}")
print(f"(iii) |Lambda_coupled| / |Lambda_0| = {float(Lc.norm() / L0.norm()):.1f}; rest part {float(Lc[:, :470].norm() / L0[:, :470].norm()):.1f}")
WF = [c.weight.detach()[:, fam, 0, 0] for c in convs]
def build(L):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); L = L.float()
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j in range(4): w = WF[j].cpu(); fm.dW[j].copy_((w @ L[:, :470]) / fm.cscale[j]); fm.cbp[j].copy_((w @ L[:, 470]) / 0.1)
    return fm.to(DEV)
B = {"level0": build(L0), "coupled_pre": build(Lc)}
def terms(ix):
    out = {k: dict(KL=[], L3=[], L3p=[]) for k in B}
    for i in range(0, len(ix), 20):
        b = ix[i:i + 20]
        with torch.no_grad():
            x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
            p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1); a0 = reader_pre(x, p)
            T = torch.from_numpy(D.T[b].astype(np.float32)).to(DEV)
        pre = [t.clone().requires_grad_(True) for t in a0]; post = [F.relu(t) for t in pre]; [t.retain_grad() for t in post]
        lg = cont(post); top = lg.detach().softmax(1).topk(K, 1); pk = top.values / top.values.sum(1, keepdim=True)
        Jp, Jq = [], []
        for c in range(K):
            for t in pre + post: t.grad = None
            lg.gather(1, top.indices[:, c:c + 1]).sum().backward(retain_graph=c < K - 1)
            Jp.append(torch.cat([t.grad.flatten(1) for t in pre], 1)); Jq.append(torch.cat([t.grad.flatten(1) for t in post], 1))
        Jp, Jq = torch.stack(Jp, 1), torch.stack(Jq, 1)
        var = lambda s: (pk * s ** 2).sum(1) - (pk * s).sum(1) ** 2
        with torch.no_grad():
            for k, fm in B.items():
                a1 = reader_pre(x, p, fm); e = torch.cat([(u - v).flatten(1) for u, v in zip(a0, a1)], 1)
                dl = torch.cat([(F.relu(u) - F.relu(v)).flatten(1) for u, v in zip(a0, a1)], 1)
                out[k]["L3"].append((0.5 * var(torch.einsum("bku,bu->bk", Jp, e))).cpu().numpy())
                out[k]["L3p"].append((0.5 * var(torch.einsum("bku,bu->bk", Jq, dl))).cpu().numpy())
                out[k]["KL"].append(kl(T, cont([F.relu(v) for v in a1])).cpu().numpy())
    return {k: {q: np.concatenate(v) for q, v in d.items()} for k, d in out.items()}
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))
r = terms(tr); print(f"(i) direct on the same 2000 images: L3(coupled) - L3(level0) = {(r['coupled_pre']['L3'] - r['level0']['L3']).mean():+.4f}")
for k, d in r.items(): print(f"    train {k:12s} KL {d['KL'].mean():.4f}  L3 {d['L3'].mean():.4f}  L3' {d['L3p'].mean():.4f}")
r = terms(idx)
for k, d in r.items(): print(f"(iv) dev500 {k:12s} KL {d['KL'].mean():.4f}  L3 {d['L3'].mean():.4f}  L3' {d['L3p'].mean():.4f}")
