"""c24 with the FULL output covariance diag(p) - p p^T (sampled) instead of the top-10 renormalised metric."""
import numpy as np, torch, time
SRC21 = open("c21_newton.py").read(); exec(SRC21.split("vS = Vs[\"S\"] * mask")[0]); exec(SRC21[SRC21.index("def build(V):"):SRC21.index("g_all = marks[N]")])
def grads_full(b, rep=10):
    x, p = family_inputs(b); post = [t.requires_grad_(True) for t in readers_post(x, p)]
    a_, b_, c_, d_ = post
    lg = rest_from_4a(torch.cat([a_, F.relu(C3(F.pad(b_, (1, 1, 1, 1)))), F.relu(C5(F.pad(c_, (2, 2, 2, 2)))), d_], 1))
    pr = lg.detach().softmax(1); sp = pr.sqrt(); out = []
    for r in range(rep):
        u = sp * torch.randn_like(pr); xi = u - pr * u.sum(1, keepdim=True)
        out.append(torch.cat(torch.autograd.grad((lg * xi).sum(), post, retain_graph=r < rep - 1), 1))
    return torch.stack(out, 1).detach().reshape(len(b), rep, NRD, 196) / np.sqrt(rep) * np.sqrt(10)
def gam_full(b):
    g = grads_full(b); y = ybatch(b); return torch.einsum("bcjp,bkp->bcjk", g, y) * mask
perm = np.random.default_rng(28).permutation(D.tr); Itr, Iho = perm[:8000], perm[8000:12000]
def Fv(V, imgs):
    R = torch.zeros(NRD, 962, device=DEV); L = 0.0
    for i in range(0, len(imgs), 10):
        Gm = gam_full(imgs[i:i + 10]); sc = (Gm * V).sum((2, 3)); R += torch.einsum("bc,bcjk->jk", sc, Gm); L += float((sc ** 2).sum()) / 2
    return R * mask / len(imgs), L / len(imgs)
v = Vs["S"] * mask; refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
Fvv, L = Fv(v, Itr); r = -(Fvv * free); z = Pinv(r); p = z.clone(); rz = float((r * z).sum()); best = (Fv(v, Iho)[1], 0)
print(f"start: train L {L:.5f}, held-out L {best[0]:.5f} ({time.time() - t0:.0f}s)", flush=True)
for k in range(1, 11):
    Fp, _ = Fv(p, Itr); al = rz / float((p * Fp).sum()); v = v + al * p; r = r - al * (Fp * free)
    Lho = Fv(v, Iho)[1]; fm = build(v); res = per_image(fm)
    print(f"  it {k}: held-out L {Lho:.5f}  dev-5000 KL {res.mean():.4f} (se {res.std() / np.sqrt(len(res)):.4f})  ({time.time() - t0:.0f}s)", flush=True)
    if Lho < best[0]: best = (Lho, k); torch.save(fm.state_dict(), "cache/c28_best.pt"); np.save("cache/c28_best_dev.npy", res)
    elif k - best[1] >= 2: print(f"  STOP at it {k}: best held-out at it {best[1]}", flush=True); break
    z = Pinv(r); rz2 = float((r * z).sum()); p = z + (rz2 / rz) * p; rz = rz2
paired(f"krylov_full_it{best[1]}", np.load("cache/c28_best_dev.npy"), {"kbar(S)": refs["kbar"], "T": refs["T"]}); print("done")
