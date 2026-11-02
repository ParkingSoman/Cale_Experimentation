"""K1' (Part VIII): network-made same-image step from S along d = -M^-1 g^cov(class-256); step length (A) from the class
pictures' own curvature, (B) FITTED line search on photos. Paired dev-5000 KL vs S (kbar) and T."""
import numpy as np, torch, time
SRC = open("c40_errenergy.py").read(); exec(SRC.split("vS = Vs[\"S\"] * mask")[0])
SRC21 = open("c21_newton.py").read(); exec(SRC21[SRC21.index("def build(V):"):SRC21.index("g_all = marks[N]")])
vS = Vs["S"] * mask; refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
gc = torch.load("cache/c41_split_rand.pt", weights_only=False)["class256"][2].to(DEV); d = -Pinv(gc)
gP = torch.load("cache/c40_photo8000_grad.pt", weights_only=False)["g"].to(DEV)
def curv(X):
    c = 0.0
    for i in range(0, len(X), 8):
        torch.manual_seed(i); c += float(((gam(X[i:i + 8]) * d).sum((2, 3)) ** 2).sum())
    return c / len(X)
Xn = np.load("cache/e1_self_256.npy"); cq = curv(Xn)
Xp = D.X[np.sort(np.random.default_rng(43).choice(D.tr, 2000, replace=False))]; cp = curv(Xp)
aA = -float((gc * d).sum()) / cq; aB = -float((gP * d).sum()) / cp
print(f"cos(d, photo-8000 dir) {float((d * -Pinv(gP)).sum() / d.norm() / Pinv(gP).norm()):.3f}; step A {aA:.3g}, step B {aB:.3g}; curvature class/photo {cq / cp:.2f}", flush=True)
for tag, a in [("A_netmade_len", aA), ("B_photo_len_FITTED", aB)]:
    r = per_image(build(vS + a * d)); np.save(f"cache/c43_{tag}.npy", r)
    print(f"{tag}: dev-5000 KL {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})", flush=True); paired(tag, r, {"kbar(S)": refs["kbar"], "T": refs["T"]})
print(f"done ({time.time() - t0:.0f}s)")
