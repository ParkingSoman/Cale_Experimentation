"""R5' part 3: sharp co-circularity test (offsets where co-circular and same-orientation disagree by > 60 deg) and
pictures of each member's coupling field with the predicted circle drawn."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("x5b_offsets.py").read().split("rows = []")[0].split("Kbar = torch.load")[0])
import torch
Koff = np.load("cache/x5_koff.npy"); Z = np.load("results_g0_tuning.npz"); th = Z["th"]; tun = Z["60"]
fam = list(np.load("cache/s1_fine.npz")["fam"]); ang = np.array([th[tun[:, n].argmax()] for n in fam], float)
STRIDE, R = 16.0, 60.0
def cc_pred(f, dy, dx):
    t = np.deg2rad(ang[f]); c = np.array([R * np.cos(t), R * np.sin(t)]); v = c - np.array([dx * STRIDE, -dy * STRIDE])
    return np.degrees(np.arctan2(v[1], v[0])) % 360
sh = []
for f in range(10):
    E = np.clip(Koff[:, :, f, :], 0, None); o, p_, q_, w = [], [], [], []
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            if (dy, dx) == (0, 0) or E[13 + dy, 13 + dx].sum() <= 0: continue
            pr = cc_pred(f, dy, dx); dd = abs((pr - ang[f] + 180) % 360 - 180)
            if dd < 60: continue
            e = E[13 + dy, 13 + dx]; o.append(np.degrees(np.angle(np.sum(e * np.exp(1j * np.deg2rad(ang))))) % 360)
            p_.append(pr); q_.append(ang[f]); w.append(e.sum())
    o, p_, q_, w = map(np.array, (o, p_, q_, w)); cs = lambda a, b: float(np.sum(w * np.cos(np.deg2rad(a - b))) / w.sum())
    sh.append((cs(o, p_), cs(o, q_), len(w)))
    print(f"f{f} ({int(ang[f])} deg): sharp offsets n={len(w)}  cos(obs - cocircular) {sh[-1][0]:+.3f}  cos(obs - same) {sh[-1][1]:+.3f}")
print("mean sharp:", np.round(np.mean([s[:2] for s in sh], 0), 3))
fig, ax = plt.subplots(2, 5, figsize=(20, 8.5))
for f in range(10):
    a = ax.flat[f]; E = np.clip(Koff[:, :, f, :], 0, None).sum(-1)[7:20, 7:20]
    a.imshow(E, cmap="magma", extent=[-6.5, 6.5, 6.5, -6.5])
    t = np.deg2rad(ang[f]); cx, cy = R / STRIDE * np.cos(t), -R / STRIDE * np.sin(t)
    a.add_patch(plt.Circle((cx, cy), R / STRIDE, fill=False, color="cyan", lw=1.5, ls="--"))
    a.arrow(0, 0, 1.6 * np.cos(t), -1.6 * np.sin(t), color="lime", width=.08)
    for dy in range(-6, 7, 2):                                           # partner orientation ticks
        for dx in range(-6, 7, 2):
            e = np.clip(Koff[13 + dy, 13 + dx, f], 0, None)
            if e.sum() < .05 * E.max(): continue
            oo = np.angle(np.sum(e * np.exp(1j * np.deg2rad(ang)))); a.plot([dx, dx + .7 * np.cos(oo)], [dy, dy - .7 * np.sin(oo)], color="white", lw=1)
    a.set_title(f"member {fam[f]} ({int(ang[f])} deg)"); a.set_xlim(-6.5, 6.5); a.set_ylim(6.5, -6.5)
fig.suptitle("Kbar coupling energy of each curve member (at centre) to the family at each offset (cells of 16 px). Green: member's centre direction. Cyan: predicted circle (r = 60 px). White ticks: partners' mean centre direction.")
plt.tight_layout(); plt.savefig("cache/x5_maps.png", dpi=70); print("saved cache/x5_maps.png")
