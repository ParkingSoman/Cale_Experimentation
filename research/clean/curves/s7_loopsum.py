"""Function-space test: does the coarse neuron at q track L_r(q) = sum_theta relu(z_theta(q - r u(theta)))?
u(theta) = direction from the arc's middle to its centre of curvature (stimulus convention: (cos t, -sin t) in x,y,
y down). r in positions (8 px). r<0 = curves bulging the wrong way (control). Also: family sum / max at q itself."""
import sys, numpy as np, torch
from inception import *
from family import Family
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"])
pref = np.deg2rad(np.array([10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu"); fm = Family(m, ck["fam"], ck["k"]).to(DEV); fm.load_state_dict(ck["state"])
idx = np.sort(np.random.default_rng(3).choice(tr, 400, replace=False)); Zs, Cs = [], []
with torch.no_grad():
    for i in range(0, len(idx), 100):
        ins = inputs_3b(m, to_3a(m, prep(X[idx[i:i+100]])))
        Zs.append(F.relu(pre_3b(m, ins[0])[:, fam]).cpu().numpy()); Cs.append(fm.pre(ins)[:, 0].cpu().numpy())
Z = np.concatenate(Zs); C = np.concatenate(Cs)          # (N,10,28,28), (N,28,28)
def shift(a, dy, dx):   # out[q] = a[q + (dy,dx)], zero outside
    o = np.zeros_like(a); H = a.shape[-1]
    ys, yd = slice(max(0, -dy), H - max(0, dy)), slice(max(0, dy), H - max(0, -dy))
    xs, xd = slice(max(0, -dx), H - max(0, dx)), slice(max(0, dx), H - max(0, -dx))
    o[..., ys, xs] = a[..., yd, xd]; return o
M = (slice(None), slice(4, 24), slice(4, 24))           # ignore borders
c = C[M].ravel(); cr = np.maximum(c, 0)
def L(r):
    out = np.zeros_like(C)
    for i, t in enumerate(pref):
        dx, dy = -r * np.cos(t), r * np.sin(t)           # q - r u(theta), u = (cos t, -sin t)
        out += shift(Z[:, i], int(round(dy)), int(round(dx)))
    return out
print("corr(coarse pre, X):  [post-ReLU coarse in brackets]")
print(f"  family sum at q:  {np.corrcoef(c, Z.sum(1)[M].ravel())[0,1]:.3f} [{np.corrcoef(cr, Z.sum(1)[M].ravel())[0,1]:.3f}]")
print(f"  family max at q:  {np.corrcoef(c, Z.max(1)[M].ravel())[0,1]:.3f}")
for r in [-3, -2, -1, 1, 2, 3, 4, 5]:
    l = L(r)[M].ravel(); print(f"  loop sum r={r:+d} ({8*r:+d}px): {np.corrcoef(c, l)[0,1]:.3f} [{np.corrcoef(cr, l)[0,1]:.3f}]")
