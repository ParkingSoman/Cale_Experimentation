"""PCG on the true second-order objective 1/2 v^T F v (reader layer, free coordinates), matvecs streamed over n = 8000
train images; held-out objective on 2000 other train images decides the stop. Start = S (factorised build)."""
import numpy as np, torch, time
SRC21 = open("c21_newton.py").read(); exec(SRC21.split("vS = Vs[\"S\"] * mask")[0])
perm = np.random.default_rng(24).permutation(D.tr); Itr, Iho = perm[:8000], perm[8000:10000]
def Fv(V, imgs):
    R = torch.zeros(NRD, 962, device=DEV); L = 0.0
    for i in range(0, len(imgs), 10):
        Gm, _, _ = gammas(imgs[i:i + 10]); sc = (Gm * V).sum((2, 3)); R += torch.einsum("bc,bcjk->jk", sc, Gm); L += float((sc ** 2).sum()) / 2
    return R * mask / len(imgs), L / len(imgs)
exec(SRC21[SRC21.index("def build(V):"):SRC21.index("g_all = marks[N]")])
v = Vs["S"] * mask; refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
Fvv, L = Fv(v, Itr); r = -(Fvv * free); z = Pinv(r); p = z.clone(); rz = float((r * z).sum()); best = (Fv(v, Iho)[1], 0, v.clone())
print(f"start: train L {L:.5f}, held-out L {best[0]:.5f} ({time.time() - t0:.0f}s)", flush=True)
for k in range(1, 13):
    Fp, _ = Fv(p, Itr); pFp = float((p * Fp).sum()); al = rz / pFp; v = v + al * p; r = r - al * (Fp * free)
    Lho = Fv(v, Iho)[1]; fm = build(v); res = per_image(fm)
    print(f"  it {k}: held-out L {Lho:.5f}  dev-5000 KL {res.mean():.4f} (se {res.std() / np.sqrt(len(res)):.4f})  ({time.time() - t0:.0f}s)", flush=True)
    if Lho < best[0]: best = (Lho, k, v.clone()); torch.save(fm.state_dict(), "cache/c24_best.pt"); np.save("cache/c24_best_dev.npy", res)
    elif k - best[1] >= 2: print(f"  STOP: held-out objective has not improved for 2 steps; best at it {best[1]}", flush=True); break
    z = Pinv(r); rz2 = float((r * z).sum()); p = z + (rz2 / rz) * p; rz = rz2
rb = np.load("cache/c24_best_dev.npy"); paired(f"krylov_it{best[1]}", rb, {"kbar(S)": refs["kbar"], "T": refs["T"]}); print("done")
