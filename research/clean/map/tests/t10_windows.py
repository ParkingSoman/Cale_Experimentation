"""T10: 112 px windows around the peak (heat overlaid), top-32 of the 10k set, for 7 n604 and 10 n1316; plus the
windows around the strongest NEGATIVE spot (top-16)."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); names = [s.split(",")[0] for s in class_names()]; y = G._y; W = 112
for t in [(7, 604), (10, 1316)]:
    z = np.load(os.path.join(OUTD, f"t2_z_{t[0]}_{t[1]}.npy")).astype(np.float32); f = z.reshape(len(z), -1); cells = []
    for sign, k in ((1, 32), (-1, 16)):
        top = np.argsort(-sign * (f.max(1) if sign > 0 else f.min(1)))[:k]
        for j in top:
            pos = f[j].argmax() if sign > 0 else f[j].argmin(); r, c = divmod(pos, 14)
            r0 = int(np.clip((r + .5) * 16 - W / 2, 0, 224 - W)); c0 = int(np.clip((c + .5) * 16 - W / 2, 0, 224 - W))
            h = np.array(heat(G._X[ana[j]], z[j], size=224))
            cells.append((Image.fromarray(h[r0:r0 + W, c0:c0 + W]), f"{'+' if sign>0 else ''}{(f[j].max() if sign>0 else f[j].min()):.1f} r{r}c{c} {names[y[ana[j]]]}"))
    print(sheet(cells, f"t10_win_b{t[0]}_n{t[1]}.png", ncol=8, size=150, title=f"{t}: peak windows (top 32) | strongest-negative windows (16)"))
