"""R5' part 2 (Statement 16.4 a, b): offset-averaged coupling of the x-signals and the co-circularity test."""
import numpy as np, torch
Kbar = torch.load("cache/x5_kbar.pt")
Kx = Kbar[:1960, :1960].reshape(10, 14, 14, 10, 14, 14)
Koff = torch.zeros(27, 27, 10, 10); cnt = torch.zeros(27, 27)
for y0 in range(14):
    for x0 in range(14):
        Koff[13 - y0:27 - y0, 13 - x0:27 - x0] += Kx[:, y0, x0].permute(2, 3, 0, 1); cnt[13 - y0:27 - y0, 13 - x0:27 - x0] += 1
Koff = (Koff / cnt[:, :, None, None]).numpy(); np.save("cache/x5_koff.npy", Koff)
Z = np.load("results_g0_tuning.npz"); th = Z["th"]; tun = Z["60"]
fam = list(np.load("cache/s1_fine.npz")["fam"]); ang = np.array([th[tun[:, n].argmax()] for n in fam], float)
print("member orientations (deg, direction to centre, y up):", ang.astype(int).tolist())
STRIDE, R = 16.0, 60.0                                    # 224 / 14 px per cell; preferred radius (tuning radius)
def cc_pred(f, dy, dx):                                   # co-circular partner orientation at offset (dy rows down, dx cols right)
    t = np.deg2rad(ang[f]); c = np.array([R * np.cos(t), R * np.sin(t)])           # centre, y up
    q = np.array([dx * STRIDE, -dy * STRIDE]); v = c - q
    return np.degrees(np.arctan2(v[1], v[0])) % 360, abs(np.linalg.norm(v) - R)      # orientation, distance off the circle
def circ_corr(a, b, w):
    a, b = np.deg2rad(a), np.deg2rad(b); return float(np.sum(w * np.cos(a - b)) / np.sum(w))
rows = []
for f in range(10):
    E = np.clip(Koff[:, :, f, :], 0, None)                  # positive coupling of f (at centre) to each g at each offset
    pred, obs, w, offc, par = [], [], [], [], []
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            if dy == 0 and dx == 0: continue
            e = E[13 + dy, 13 + dx]; tot = e.sum()
            if tot <= 0: continue
            z = np.sum(e * np.exp(1j * np.deg2rad(ang))); obs.append(np.degrees(np.angle(z)) % 360)   # coupling-weighted partner orientation
            pr, off = cc_pred(f, dy, dx); pred.append(pr); offc.append(off); w.append(tot); par.append(ang[f])
    pred, obs, w, offc, par = map(np.array, (pred, obs, w, offc, par))
    on = offc < 0.5 * STRIDE * 2                              # offsets within ~16 px of the predicted circle
    rows.append((f, int(ang[f]), circ_corr(obs, pred, w), circ_corr(obs, par, w),
                 float(w[on].sum() / w.sum()), float(on.mean())))
print("f  orient  cos(obs - cocircular)  cos(obs - same orientation)  energy share on circle  area share on circle")
for r in rows: print(f"{r[0]}  {r[1]:4d}   {r[2]:+.3f}                {r[3]:+.3f}                      {r[4]:.3f}                 {r[5]:.3f}")
print("mean:", np.round(np.mean([r[2:] for r in rows], 0), 3))
