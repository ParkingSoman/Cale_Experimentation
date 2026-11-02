"""K0d (Part VIII, statement 28.10): within-class vs between-class parts of the photos' same-image term."""
import numpy as np, torch, time
SRC = open("c41_split.py").read(); exec(SRC.split("ref = torch.load")[0])
Mn = lambda g: float((g * Pinv(g)).sum()) ** 0.5
cs = lambda a, b: float((Pinv(a) * Pinv(b)).sum() / Pinv(a).norm() / Pinv(b).norm())
rng = np.random.default_rng(44); ytr = D.y[D.tr]; cls = rng.choice(np.unique(ytr), 128, replace=False); idx = []
for c in cls: idx += list(rng.choice(D.tr[ytr == c], 2, replace=False))
X = D.X[np.array(idx)]; partner = np.arange(256) ^ 1                      # 2i <-> 2i+1, same class
g = push(X, X); gs = push(X, X[partner])
r41 = np.random.default_rng(41); gr = torch.zeros_like(g); k = 0
while k < 4:
    pp = r41.permutation(256)
    if (pp != np.arange(256)).all(): gr += push(X, X[pp]) / 4; k += 1
within, between = g - gs, gs - gr; cov = g - gr
gc_class = torch.load("cache/c41_split_rand.pt", weights_only=False)["class256"][2].to(DEV)
ref = torch.load("cache/c40_photo8000_grad.pt", weights_only=False)["g"].to(DEV)
print(f"|within|/|cov| {Mn(within) / Mn(cov):.3f}   |between|/|cov| {Mn(between) / Mn(cov):.3f}   cos(within, between) {cs(within, between):.3f}")
print(f"cos with photo-8000: cov {cs(cov, ref):.3f} within {cs(within, ref):.3f} between {cs(between, ref):.3f}")
print(f"cos(class-256 cov, photo within) {cs(gc_class, within):.3f}   cos(class-256 cov, photo between) {cs(gc_class, between):.3f}")
torch.save(dict(within=within.cpu(), between=between.cpu(), cov=cov.cpu()), "cache/c44_within.pt"); print(f"done ({time.time() - t0:.0f}s)")
