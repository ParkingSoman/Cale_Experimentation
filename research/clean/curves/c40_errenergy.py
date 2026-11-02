"""K0 (Part VIII, statement 28.5): why the 256 class pictures taught nothing in c26. Uniform output metric (10 xi per
input), build S, shared preconditioner M = c17 KFC. Per set: error energy 2L_q(S) per input, top-5% share, Newton
decrement g.M^-1 g, direction cos of M^-1 g with the 8000-photo direction (K1 reference, cached)."""
import numpy as np, torch, time
SRC21 = open("c21_newton.py").read(); exec(SRC21.split("vS = Vs[\"S\"] * mask")[0])
src25 = open("c25_unigap.py").read(); exec(src25[src25.index("def reader_grads_uni"):src25.index("res = {")])
def fin(X):
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(X)))[0]))); p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    return x, p
family_inputs_saved = family_inputs
def gam(X):
    global family_inputs
    family_inputs = lambda b: fin(b); g = reader_grads_uni(X).reshape(len(X), 10, NRD, 196); x, p = fin(X); family_inputs = family_inputs_saved
    o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
    return torch.einsum("bcjp,bkp->bcjk", g, y) * mask
vS = Vs["S"] * mask
def stats(X, tag):
    g = torch.zeros(NRD, 962, device=DEV); en = []
    for i in range(0, len(X), 8):
        torch.manual_seed(i); Gm = gam(X[i:i + 8]); sc = (Gm * vS).sum((2, 3))
        g += torch.einsum("bc,bcjk->jk", sc, Gm); en.append((sc ** 2).sum(1).cpu().numpy())
        if i % 2000 == 0 and i: print(f"  {tag}: {i} ({time.time() - t0:.0f}s)", flush=True)
    en = np.concatenate(en); g = g * free / len(X); d = -Pinv(g)
    return dict(en=en, g=g, d=d, dec=float(-(g * d).sum()))
rng = np.random.default_rng(26); Xr = D.X[np.sort(rng.choice(D.tr, 256, replace=False))]
Xn = np.load("cache/e1_self_256.npy")
ref_idx = np.sort(np.random.default_rng(40).choice(D.tr, 8000, replace=False))
S = {"photo256": stats(Xr, "photo256"), "class256": stats(Xn, "class256")}
ref = torch.zeros(NRD, 962, device=DEV); en_ref = []
for i in range(0, 8000, 1000):
    s = stats(D.X[ref_idx[i:i + 1000]], f"photo8000[{i}]"); ref += s["g"] / 8; en_ref.append(s["en"])
S["photo8000"] = dict(en=np.concatenate(en_ref), g=ref, d=-Pinv(ref), dec=float((ref * Pinv(ref)).sum()))
torch.save(dict(g=ref.cpu(), idx=ref_idx), "cache/c40_photo8000_grad.pt")
dref = S["photo8000"]["d"]
print(f"{'set':10s} {'2L/input':>10s} {'top5% share':>11s} {'decrement':>11s} {'dec/2L':>9s} {'cos vs 8000':>11s}")
for k, s in S.items():
    en = s["en"]; top = np.sort(en)[::-1][:max(1, len(en) // 20)].sum() / en.sum()
    cos = float((s["d"] * dref).sum() / s["d"].norm() / dref.norm())
    print(f"{k:10s} {en.mean():10.4g} {top:11.3f} {s['dec']:11.4g} {s['dec'] / en.mean():9.3g} {cos:11.3f}")
print(f"ratio class/photo error energy: {S['class256']['en'].mean() / S['photo256']['en'].mean():.3f}")
np.savez("cache/c40_errenergy.npz", **{f"en_{k}": s["en"] for k, s in S.items()})
print(f"done ({time.time() - t0:.0f}s)")
