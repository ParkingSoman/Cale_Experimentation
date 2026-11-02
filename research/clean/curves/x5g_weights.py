"""Statement 16.7 (weights only, no image): M_off(d)[f, g] = sum_u sum_delta k_u(f, delta) k_u(g, delta + d), with k_u the
effective kernel of each 4a 3x3 / 5x5 unit on the curve family (spatial kernel x bottleneck weights on the family)."""
import numpy as np, torch
from inception import *
m = load_model(); dev = "cpu"
fam = list(np.load("cache/s1_fine.npz")["fam"])
w = lambda n: getattr(m, n).weight.detach().cpu().double()
k3 = torch.einsum("ubyx,bf->ufyx", w("mixed4a_3x3_pre_relu_conv"), w("mixed4a_3x3_bottleneck_pre_relu_conv")[:, fam, 0, 0])   # (u, 10, 3, 3)
k5 = torch.einsum("ubyx,bf->ufyx", w("mixed4a_5x5_pre_relu_conv"), w("mixed4a_5x5_bottleneck_pre_relu_conv")[:, fam, 0, 0])   # (u, 10, 5, 5)
Moff = torch.zeros(9, 9, 10, 10, dtype=torch.float64)                    # offsets -4..4
for k, h in [(k3, 1), (k5, 2)]:
    s = 2 * h + 1
    for ay in range(s):
        for ax in range(s):
            for by in range(s):
                for bx in range(s):
                    dy, dx = by - ay, bx - ax                            # partner offset d = (position of g) - (position of f)
                    Moff[4 + dy, 4 + dx] += torch.einsum("uf,ug->fg", k[:, :, ay, ax], k[:, :, by, bx])
Moff = Moff.numpy(); np.save("cache/x5g_moff.npy", Moff)
Koff = np.load("cache/x5_koff.npy")[9:18, 9:18]                           # Kbar offsets -4..4
mask = np.ones((9, 9), bool)
c_all = np.corrcoef(Moff[mask].ravel(), Koff[mask].ravel())[0, 1]
off = mask.copy(); off[4, 4] = False
c_off = np.corrcoef(Moff[off].ravel(), Koff[off].ravel())[0, 1]
print(f"offset-pattern corr M vs Kbar (all offsets |d|<=4): {c_all:.3f}; excluding d = 0: {c_off:.3f}")
Z = np.load("results_g0_tuning.npz"); th = Z["th"]; tun = Z["60"]; ang = np.array([th[tun[:, n].argmax()] for n in fam], float)
STRIDE, R = 16.0, 60.0
def cc_pred(f, dy, dx):
    t = np.deg2rad(ang[f]); c = np.array([R * np.cos(t), R * np.sin(t)]); v = c - np.array([dx * STRIDE, -dy * STRIDE])
    return np.degrees(np.arctan2(v[1], v[0])) % 360
for name, A in [("M (weights)", Moff), ("Kbar (images)", Koff)]:
    sh = []
    for f in range(10):
        E = np.clip(A[:, :, f, :], 0, None); o, p_, q_, ww = [], [], [], []
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if (dy, dx) == (0, 0) or E[4 + dy, 4 + dx].sum() <= 0: continue
                pr = cc_pred(f, dy, dx)
                if abs((pr - ang[f] + 180) % 360 - 180) < 60: continue
                e = E[4 + dy, 4 + dx]; o.append(np.degrees(np.angle(np.sum(e * np.exp(1j * np.deg2rad(ang))))) % 360)
                p_.append(pr); q_.append(ang[f]); ww.append(e.sum())
        o, p_, q_, ww = map(np.array, (o, p_, q_, ww)); cs = lambda a, b: float(np.sum(ww * np.cos(np.deg2rad(a - b))) / ww.sum())
        sh.append((cs(o, p_), cs(o, q_)))
    print(f"{name}: sharp test (|d|<=4) cos co-circular {np.mean([s[0] for s in sh]):+.3f}  same-orientation {np.mean([s[1] for s in sh]):+.3f}  members co-circ > same: {sum(s[0] > s[1] for s in sh)}/10")
