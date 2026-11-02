"""Statement 31.3 (Part VIII): PCG on the second-order objective with the EXACT uniform output metric (Prop. 31.2: one
JVP + one VJP per image, no xi sampling), from S. Train set: N images; held-out objective on 1000 other images decides
the stop. usage: c50_exactcg.py SOURCE N TAG   (SOURCE = photos | path to a network-made .npy)"""
import sys, numpy as np, torch, time
from torch.func import jvp
SRC = open("c46_readerstep.py").read(); exec(SRC.split("X = np.load(sys.argv[1])")[0])
SOURCE, N, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
GATED = len(sys.argv) > 5 and sys.argv[5] == "gated"          # reader gates D in the expansion (true 2nd-order Taylor of KL)
METRIC = sys.argv[4] if len(sys.argv) > 4 else "uni"   # uni: I - 11^T/n ; kl: diag(p) - p p^T (exact 2nd-order KL)
_mp = F.max_pool2d
def _mp_idx(*a, **k):                                   # index form: twice differentiable (needed for J v via double backward)
    k.pop("return_indices", None); return _mp(*a, return_indices=True, **k)[0]
F.max_pool2d = _mp_idx
refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
def upper(a, b, c, d):
    return rest_from_4a(torch.cat([a, F.relu(C3(F.pad(b, (1, 1, 1, 1)))), F.relu(C5(F.pad(c, (2, 2, 2, 2)))), d], 1))
def Fv_exact(V, X, bs=10):
    R = torch.zeros(NRD, 962, device=DEV); L = 0.0
    for i in range(0, len(X), bs):
        x, p = fin(X[i:i + bs]); o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
        with torch.no_grad(): post = [t.detach() for t in readers_post(x, p)]
        e = torch.einsum("jk,bkp->bjp", V * mask, y)
        if GATED: e = e * torch.cat([(t > 0).float().flatten(2) for t in post], 1)
        tans = [t.reshape(len(x), -1, 14, 14).contiguous() for t in torch.split(e, sizes, 1)]
        pp = [t.clone().requires_grad_(True) for t in post]; lg = upper(*pp); w = torch.zeros_like(lg, requires_grad=True)
        gw = torch.autograd.grad((lg * w).sum(), pp, create_graph=True)                       # J^T w, linear in w
        u = torch.autograd.grad(sum((a * b).sum() for a, b in zip(gw, tans)), w, retain_graph=True)[0]   # = J tans
        u = u.detach()
        if METRIC == "kl":
            pr = torch.softmax(lg.detach(), 1); pu = pr * u - pr * (pr * u).sum(1, keepdim=True)
        else:
            pu = u - u.mean(1, keepdim=True)
        gt = torch.cat([g.flatten(2) for g in torch.autograd.grad((lg * pu).sum(), pp)], 1)
        if GATED: gt = gt * torch.cat([(t > 0).float().flatten(2) for t in post], 1)
        R += torch.einsum("bjp,bkp->jk", gt, y) * mask; L += 0.5 * float((u * pu).sum())
    return R / len(X), L / len(X)
# self-check of Prop. 31.2 against the sampled form (50 xi) on 8 photos (uniform, gate-free only)
Xc = D.X[D.tr[:8]]; V0 = Vs["S"] * mask; Re, _ = Fv_exact(V0, Xc)
family_inputs = lambda b: fin(b); g = reader_grads_uni(Xc, rep=50).reshape(8, 50, NRD, 196); family_inputs = family_inputs_saved
x, p = fin(Xc); o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2); Gm = torch.einsum("bcjp,bkp->bcjk", g, y) * mask
Rs = torch.einsum("bc,bcjk->jk", (Gm * V0).sum((2, 3)), Gm) / 50 / 8
print(f"self-check: cos(exact, sampled-50) {float((Re * Rs).sum() / Re.norm() / Rs.norm()):.3f}, norm ratio {float(Rs.norm() / Re.norm()):.3f}", flush=True)
rng = np.random.default_rng(50)
if SOURCE == "photos":
    perm = rng.permutation(D.tr); Xtr, Xho = D.X[np.sort(perm[:N])], D.X[np.sort(perm[N:N + 1000])]
else:
    A = np.load(SOURCE); perm = rng.permutation(len(A)); Xtr, Xho = A[np.sort(perm[:N])], A[np.sort(perm[N:N + 1000])]
v = Vs["S"] * mask; Fvv, L = Fv_exact(v, Xtr); r = -(Fvv * free); z = Pinv(r); pd = z.clone(); rz = float((r * z).sum())
best = (Fv_exact(v, Xho)[1] if len(Xho) else L, 0, v.clone()); print(f"start: train L {L:.5g}, held-out L {best[0]:.5g} ({time.time() - t0:.0f}s)", flush=True)
for k in range(1, 16):
    Fp, _ = Fv_exact(pd, Xtr); al = rz / float((pd * Fp).sum()); v = v + al * pd; r = r - al * (Fp * free)
    Lho = Fv_exact(v, Xho)[1] if len(Xho) else float("nan"); res = per_image(build(v))
    print(f"  it {k}: held-out L {Lho:.5g}  dev-5000 KL {res.mean():.4f} (se {res.std() / np.sqrt(len(res)):.4f})  ({time.time() - t0:.0f}s)", flush=True)
    np.save(f"cache/c50_{TAG}_it{k}.npy", res); torch.save(v.cpu(), f"cache/c50_{TAG}_it{k}_V.pt")
    if Lho < best[0]: best = (Lho, k, v.clone())
    elif k - best[1] >= 2: print(f"  STOP at it {k}: held-out not improved for 2 its; best it {best[1]}", flush=True); break
    z = Pinv(r); rz2 = float((r * z).sum()); pd = z + (rz2 / rz) * pd; rz = rz2
rb = np.load(f"cache/c50_{TAG}_it{best[1]}.npy") if best[1] else None
if rb is not None: paired(f"{TAG}_it{best[1]}", rb, {"kbar(S)": refs["kbar"], "T": refs["T"]})
print(f"done ({time.time() - t0:.0f}s)")
