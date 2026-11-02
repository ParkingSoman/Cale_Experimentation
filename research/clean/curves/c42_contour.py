"""K0c (Part VIII, statement 28.8): split the push at S into same-position (p = p') and cross-position (p != p') parts,
raw and factorised (4 random derangements), for 256 photos and 256 class pictures."""
import numpy as np, torch, time
SRC = open("c40_errenergy.py").read(); exec(SRC.split("vS = Vs[\"S\"] * mask")[0])
vS = Vs["S"] * mask
def parts(Xg, Xy):
    global family_inputs
    tot = torch.zeros(NRD, 962, device=DEV); dia = torch.zeros(NRD, 962, device=DEV)
    for i in range(0, len(Xg), 8):
        torch.manual_seed(i); family_inputs = lambda b: fin(b)
        g = reader_grads_uni(Xg[i:i + 8]).reshape(-1, 10, NRD, 196); family_inputs = family_inputs_saved
        x, p = fin(Xy[i:i + 8]); o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
        Gm = torch.einsum("bcjp,bkp->bcjk", g, y) * mask; sc = (Gm * vS).sum((2, 3)); tot += torch.einsum("bc,bcjk->jk", sc, Gm)
        e = torch.einsum("jk,bkp->bjp", vS, y); s = torch.einsum("bcjp,bjp->bcp", g, e)          # per-position output error
        dia += torch.einsum("bcp,bcjp,bkp->jk", s, g, y) * mask
    n = len(Xg); return tot * free / n, dia * free / n
Mn = lambda g: float((g * Pinv(g)).sum()) ** 0.5
cs = lambda a, b: float((Pinv(a) * Pinv(b)).sum() / Pinv(a).norm() / Pinv(b).norm())
r41 = np.random.default_rng(41); DERS = []
while len(DERS) < 4:
    pp = r41.permutation(256)
    if (pp != np.arange(256)).all(): DERS.append(pp)
rng = np.random.default_rng(26); sets = {"photo256": D.X[np.sort(rng.choice(D.tr, 256, replace=False))], "class256": np.load("cache/e1_self_256.npy")}
R = {}
for tag, X in sets.items():
    t, d = parts(X, X); ft, fd = torch.zeros_like(t), torch.zeros_like(t)
    for der in DERS:
        a, b = parts(X, X[der]); ft += a / 4; fd += b / 4
    ct, cd = t - ft, d - fd; R[tag] = dict(same=cd, cross=ct - cd, cov=ct)
    print(f"{tag}: |cov same|/|cov| {Mn(cd) / Mn(ct):.3f}  |cov cross|/|cov| {Mn(ct - cd) / Mn(ct):.3f}  ({time.time() - t0:.0f}s)", flush=True)
for part in ["same", "cross", "cov"]:
    print(f"cos(class {part}, photo {part}) {cs(R['class256'][part], R['photo256'][part]):.3f}")
print(f"photo: cos(same, cross) {cs(R['photo256']['same'], R['photo256']['cross']):.3f}; class: {cs(R['class256']['same'], R['class256']['cross']):.3f}")
torch.save({k: {a: b.cpu() for a, b in v.items()} for k, v in R.items()}, "cache/c42_contour.pt"); print(f"done ({time.time() - t0:.0f}s)")
