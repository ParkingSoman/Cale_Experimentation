"""K2-net (Part VIII): network-made KFC preconditioner (G, A from the reader questions, uniform metric) for the
same-image step; plus the 256-subset count check. Paired dev-5000."""
import numpy as np, torch, time
SRC = open("c46_readerstep.py").read(); exec(SRC.split("X = np.load(sys.argv[1])")[0])
refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
ref = torch.load("cache/c40_photo8000_grad.pt", weights_only=False)["g"].to(DEV); dP = -Pinv(ref)
XQ = np.load("cache/c45_readerq_368.npy")
Gn = torch.zeros(NRD, NRD, device=DEV); An = torch.zeros(962, 962, device=DEV)
for i in range(0, len(XQ), 8):
    torch.manual_seed(i); family_inputs = lambda b: fin(b)
    g = reader_grads_uni(XQ[i:i + 8]).reshape(-1, 10, NRD, 196); family_inputs = family_inputs_saved
    x, p = fin(XQ[i:i + 8]); o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
    Gn += torch.einsum("bcjp,bckp->jk", g, g); An += torch.einsum("bjp,bkp->jk", y, y) / 196
Gn /= len(XQ); An /= len(XQ)
Gni = torch.linalg.inv(Gn + 1e-6 * Gn.trace() / NRD * torch.eye(NRD, device=DEV)); Ani = torch.linalg.inv(An + 1e-6 * An.trace() / 962 * torch.eye(962, device=DEV))
Pnet = lambda R: (Gni @ (R * free) @ Ani) * free
print(f"network-made factors done ({time.time() - t0:.0f}s)", flush=True)
def cov_grad(X):
    n = len(X); r = np.random.default_rng(41); D_ = []
    while len(D_) < 4:
        pp = r.permutation(n)
        if (pp != np.arange(n)).all(): D_.append(pp)
    g = push(X, X); gf = sum(push(X, X[der]) for der in D_) / 4; return g - gf
def step(X, gc, P, tag):
    d = -P(gc); c = 0.0
    for i in range(0, len(X), 8):
        torch.manual_seed(i); c += float(((gam(X[i:i + 8]) * d).sum((2, 3)) ** 2).sum())
    a = -float((gc * d).sum()) / (c / len(X)); r = per_image(build(Vs["S"] * mask + a * d)); np.save(f"cache/c47_{tag}.npy", r)
    cosd = float((d * dP).sum() / d.norm() / dP.norm())
    print(f"{tag}: cos(d, photo dir) {cosd:.3f}; step {a:.3g}; dev-5000 KL {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})", flush=True)
    paired(tag, r, {"kbar(S)": refs["kbar"], "T": refs["T"], "readerq368_photoM": np.load("cache/c46_readerq368.npy")})
gc368 = torch.load("cache/c46_readerq368_grads.pt", weights_only=False)["gc"].to(DEV)
step(XQ, gc368, Pnet, "q368_netM")
sub = np.sort(np.random.default_rng(47).choice(368, 256, replace=False)); XS = XQ[sub]; gc256 = cov_grad(XS)
step(XS, gc256, Pinv, "q256_photoM"); step(XS, gc256, Pnet, "q256_netM")
torch.save(dict(Gn=Gn.cpu(), An=An.cpu(), gc256=gc256.cpu(), sub=sub), "cache/c47_netprec.pt"); print(f"done ({time.time() - t0:.0f}s)")
