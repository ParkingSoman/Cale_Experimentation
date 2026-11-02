"""R3' (Statement 14.10, Prop. 14.9): in-span coupled formula. Lambda (10 x 471) minimises the exact second-order KL
term in the span, 1/2 E_x Var_{c~p}(t_c - <Lambda, A_c>). One linear solve. MIX-DERIVED (2000 train images, rng 5)."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A
exec(open("x2b_reason.py").read().split("names = {")[0])          # m, D, idx, convs, cont, reader_pre
fam = list(np.load("cache/s1_fine.npz")["fam"]); keep = [i for i in range(480) if i not in fam]; R = keep + [480]
K = 10; WF = [c.weight.detach()[:, fam, 0, 0] for c in convs]                       # (out_j, 10)
order = np.random.default_rng(11).permutation(D.tr); NTOT = len(order)
V = ["out", "pre"]; M = {v: torch.zeros(4710, 4710, dtype=torch.float64) for v in V}; vv = {v: torch.zeros(4710, dtype=torch.float64) for v in V}
Mg = {v: torch.zeros(4710, 4710, device=DEV) for v in V}; vg = {v: torch.zeros(4710, device=DEV) for v in V}; snap = None
def flush():
    for v in V: M[v] += Mg[v].cpu().double(); vv[v] += vg[v].cpu().double(); Mg[v].zero_(); vg[v].zero_()
for i in range(0, NTOT, 20):
    b = np.sort(order[i:i + 20])
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(b if False else D.X[b])))[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1); a0 = reader_pre(x, p)
        zx = torch.cat([x, torch.ones_like(x[:, :1])], 1); zp = torch.cat([p, torch.ones_like(p[:, :1])], 1)   # (B, 481, 14, 14)
    post = [F.relu(t).requires_grad_(True) for t in a0]; lg = cont(post)
    pr = lg.detach().softmax(1); top = pr.topk(K, 1); pk = top.values / top.values.sum(1, keepdim=True)
    Jo = [[] for _ in range(4)]
    for c in range(K):
        gr = torch.autograd.grad(lg.gather(1, top.indices[:, c:c + 1]).sum(), post, retain_graph=c < K - 1)
        for j in range(4): Jo[j].append(gr[j])
    with torch.no_grad():
        Jo = [torch.stack(t, 1) for t in Jo]                                         # (B, K, out_j, 14, 14)
        for v in V:
            J = Jo if v == "out" else [t * (a > 0).float()[:, None] for t, a in zip(Jo, a0)]
            gx = sum(torch.einsum("of,bkopq->bkfpq", WF[j], J[j]) for j in range(3)); gp = torch.einsum("of,bkopq->bkfpq", WF[3], J[3])
            A = torch.einsum("bkfpq,bzpq->bkfz", gx, zx) + torch.einsum("bkfpq,bzpq->bkfz", gp, zp)          # (B, K, 10, 481)
            t = sum(A[:, :, f, fam[f]] for f in range(10))                                                  # (B, K)
            a = A[:, :, :, R].reshape(len(b), K, -1)                                                        # (B, K, 4710)
            ac = (a - (pk[:, :, None] * a).sum(1, keepdim=True)) * pk[:, :, None].sqrt()
            tc = (t - (pk * t).sum(1, keepdim=True)) * pk.sqrt()
            Mg[v] += ac.reshape(-1, 4710).T @ ac.reshape(-1, 4710); vg[v] += ac.reshape(-1, 4710).T @ tc.reshape(-1)
    if (i // 20) % 10 == 9: flush()
    if i + 20 == 8000: flush(); snap = {v: (M[v].clone(), vv[v].clone()) for v in V}
flush()
torch.save(dict(M=M, v=vv, snap=snap, n=NTOT), "cache/x3c_stats_35k.pt")
def solve(Mm, v):
    return torch.linalg.solve(Mm + 1e-6 * torch.trace(Mm) / 4710 * torch.eye(4710, dtype=torch.float64), v).reshape(10, 471).float()
def build(L):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j in range(4):
            w = WF[j].cpu(); fm.dW[j].copy_((w @ L[:, :470]) / fm.cscale[j]); fm.cbp[j].copy_((w @ L[:, 470]) / 0.1)
    return fm.to(DEV)
def per_image(fm, ix, bs=100):
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            bb = ix[i:i + bs]; lgg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[bb]))))
            out.append(kl(torch.from_numpy(D.T[bb].astype(np.float32)).to(DEV), lgg).cpu().numpy())
    return np.concatenate(out)
L = {f"{v}_n35000": solve(M[v], vv[v]) for v in V}; L.update({f"{v}_n8000": solve(*snap[v]) for v in V})
res = {}
for k, l in L.items():
    fm = build(l); torch.save(fm.state_dict(), f"cache/x3c_{k}.pt"); res[k] = per_image(fm, D.dv)
    print(k, f"{res[k].mean():.4f} (se {res[k].std() / np.sqrt(len(res[k])):.4f})", flush=True)
fl = np.load("cache/x2_gated.npz"); ref = {"level0p": fl["b_level0_p"], "level1": fl["c_level1_n2000"]}
for nme in ["B3-k0-lr003-long", "B3-k0-formulastart"]: ref[nme] = per_image(b3_train.load_b3(m, nme), D.dv)
np.savez("cache/x3c_pop.npz", **res)
for k in res:
    for r_, v in ref.items():
        d = res[k] - v; print(f"  {k} minus {r_:20s} {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})")
