"""K2 test (Part VIII): for a network-made question set: push at S, factorised part (4 random derangements), same-image
part, direction cos with the 8000-photo direction, network-made step along -M^-1 g^cov with the set's own curvature,
paired dev-5000 KL. usage: c46_readerstep.py cache/SET.npy TAG"""
import sys, numpy as np, torch, time
SRC = open("c41_split.py").read(); exec(SRC.split("ref = torch.load")[0])
SRC21 = open("c21_newton.py").read(); exec(SRC21[SRC21.index("def build(V):"):SRC21.index("g_all = marks[N]")])
X = np.load(sys.argv[1]); TAG = sys.argv[2]; n = len(X); refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
ref = torch.load("cache/c40_photo8000_grad.pt", weights_only=False)["g"].to(DEV); dref = -Pinv(ref)
Mn = lambda g: float((g * Pinv(g)).sum()) ** 0.5
cs = lambda a, b: float((Pinv(a) * Pinv(b)).sum() / Pinv(a).norm() / Pinv(b).norm())
r41 = np.random.default_rng(41); DERS = []
while len(DERS) < 4:
    pp = r41.permutation(n)
    if (pp != np.arange(n)).all(): DERS.append(pp)
g = push(X, X); gf = sum(push(X, X[der]) for der in DERS) / 4; gc = g - gf
print(f"{TAG} (n = {n}): |g_fact|/|g| {Mn(gf) / Mn(g):.3f}; cos vs photo-8000: raw {cs(g, ref):.3f} fact {cs(gf, ref):.3f} cov {cs(gc, ref):.3f} ({time.time() - t0:.0f}s)", flush=True)
d = -Pinv(gc); c = 0.0
for i in range(0, n, 8):
    torch.manual_seed(i); c += float(((gam(X[i:i + 8]) * d).sum((2, 3)) ** 2).sum())
a = -float((gc * d).sum()) / (c / n); r = per_image(build(Vs["S"] * mask + a * d)); np.save(f"cache/c46_{TAG}.npy", r)
torch.save(dict(g=g.cpu(), gf=gf.cpu(), gc=gc.cpu()), f"cache/c46_{TAG}_grads.pt")
print(f"{TAG}: step {a:.3g}; dev-5000 KL {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})", flush=True); paired(TAG, r, {"kbar(S)": refs["kbar"], "T": refs["T"]})
print(f"done ({time.time() - t0:.0f}s)")
