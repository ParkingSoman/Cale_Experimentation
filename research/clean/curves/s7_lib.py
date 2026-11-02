import numpy as np, torch
from inception import *
from family import Family
PREF = np.deg2rad(np.array([10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))
def load_maps(tag, n=400, seed=3, pre=False):
    m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu"); fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
    idx = np.sort(np.random.default_rng(seed).choice(tr, n, replace=False)); Zs, Cs = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 100):
            ins = inputs_3b(m, to_3a(m, prep(X[idx[i:i+100]])))
            z = pre_3b(m, ins[0])[:, fam]; Zs.append((z if pre else F.relu(z)).cpu().numpy()); Cs.append(fm.pre(ins).cpu().numpy())
    C = np.concatenate(Cs); return np.concatenate(Zs), (C[:, 0] if C.shape[1] == 1 else C)
def shift(a, dy, dx, fill=0.0):
    o = np.full_like(a, fill); H = a.shape[-1]
    ys, yd = slice(max(0, -dy), H - max(0, dy)), slice(max(0, dy), H - max(0, -dy))
    xs, xd = slice(max(0, -dx), H - max(0, dx)), slice(max(0, dx), H - max(0, -dx))
    o[..., ys, xs] = a[..., yd, xd]; return o
def moved(Z, r, th=PREF):
    """out[:, i, q] = Z[:, i, q - r u_i], u = (cos t, -sin t) in (x, y)."""
    return np.stack([shift(Z[:, i], int(round(r * np.sin(t))), int(round(-r * np.cos(t)))) for i, t in enumerate(th)], 1)
