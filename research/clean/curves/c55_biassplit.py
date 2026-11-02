"""C5 (Part VIII): bias part vs weight part of the photo same-image step; plain-move bias share along the c50 path."""
import numpy as np, torch, time
SRC = open("c46_readerstep.py").read(); exec(SRC.split("X = np.load(sys.argv[1])")[0])
refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
Idv = np.sort(np.random.default_rng(54).choice(D.dv, 1000, replace=False)); pos = np.searchsorted(D.dv, Idv)
assert (np.asarray(D.dv)[pos] == Idv).all()
S0 = Vs["S"] * mask; bias = torch.zeros_like(S0); bias[:, 480] = 1; bias[:, 961] = 1
for k in [1, 4, 8, 10, 15]:
    dv_ = (torch.load(f"cache/c50_photo2000_it{k}_V.pt").to(DEV) - S0) * free
    print(f"c50 it {k}: bias share of plain move {float((dv_ * bias).norm() ** 2 / dv_.norm() ** 2):.3f}", flush=True)
g = torch.load("cache/c40_photo8000_grad.pt", weights_only=False)["g"].to(DEV); d = -Pinv(g)
X5 = D.X[np.sort(np.random.default_rng(55).choice(D.tr, 500, replace=False))]
kS, kT = refs["kbar"][pos], refs["T"][pos]; print(f"dev-1000: S {kS.mean():.4f}  T {kT.mean():.4f}  (T - S {float((kT - kS).mean()):+.4f})", flush=True)
for tag, dd in [("full", d), ("bias part", d * bias), ("weight part", d * (1 - bias))]:
    c = 0.0
    for i in range(0, 500, 8):
        torch.manual_seed(i); c += float(((gam(X5[i:i + 8]) * dd).sum((2, 3)) ** 2).sum())
    a = -float((g * dd).sum()) / (c / 500); r = per_image(build(S0 + a * dd), Idv)
    print(f"{tag:12s}: dev-1000 {r.mean():.4f}  vs S {float((r - kS).mean()):+.4f} (se {float((r - kS).std() / np.sqrt(1000)):.4f})", flush=True)
print(f"done ({time.time() - t0:.0f}s)")
