"""K0b (Part VIII, statement 28.7): split each set's push at S into the factorised part (gates of input x, activity of
an independent input x' of the same set: 4 cyclic shifts) and the covariance part (the same-image term)."""
import numpy as np, torch, time
SRC = open("c40_errenergy.py").read(); exec(SRC.split("vS = Vs[\"S\"] * mask")[0])
vS = Vs["S"] * mask
def gam2(Xg, Xy):
    global family_inputs
    family_inputs = lambda b: fin(b); g = reader_grads_uni(Xg).reshape(len(Xg), 10, NRD, 196); family_inputs = family_inputs_saved
    x, p = fin(Xy); o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
    return torch.einsum("bcjp,bkp->bcjk", g, y) * mask
def push(Xg, Xy):
    g = torch.zeros(NRD, 962, device=DEV)
    for i in range(0, len(Xg), 8):
        torch.manual_seed(i); Gm = gam2(Xg[i:i + 8], Xy[i:i + 8]); sc = (Gm * vS).sum((2, 3))
        g += torch.einsum("bc,bcjk->jk", sc, Gm)
    return g * free / len(Xg)
ref = torch.load("cache/c40_photo8000_grad.pt", weights_only=False)["g"].to(DEV); dref = -Pinv(ref)
Mn = lambda g: float((g * Pinv(g)).sum()) ** 0.5
cos = lambda g: float((-Pinv(g) * dref).sum() / (-Pinv(g)).norm() / dref.norm())
rng = np.random.default_rng(26); sets = {"photo256": D.X[np.sort(rng.choice(D.tr, 256, replace=False))], "class256": np.load("cache/e1_self_256.npy")}
r41 = np.random.default_rng(41); DERS = []
while len(DERS) < 4:
    pp = r41.permutation(256)
    if (pp != np.arange(256)).all(): DERS.append(pp)
out = {}
for tag, X in sets.items():
    g = push(X, X); gf = sum(push(X, X[der]) for der in DERS) / 4; gc = g - gf
    out[tag] = (g, gf, gc)
    print(f"{tag}: |g_fact|/|g| (M-norm) {Mn(gf) / Mn(g):.3f}   |g_cov|/|g| {Mn(gc) / Mn(g):.3f}   cos vs photo-8000: raw {cos(g):.3f}  fact {cos(gf):.3f}  cov {cos(gc):.3f}  ({time.time() - t0:.0f}s)", flush=True)
gcP, gcC = out["photo256"][2], out["class256"][2]
print(f"cos(cov photo-256, cov class-256) {float((Pinv(gcP) * Pinv(gcC)).sum() / Pinv(gcP).norm() / Pinv(gcC).norm()):.3f}")
torch.save({k: [t.cpu() for t in v] for k, v in out.items()}, "cache/c41_split_rand.pt"); print(f"done ({time.time() - t0:.0f}s)")
