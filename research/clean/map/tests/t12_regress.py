"""T12: what simple image statistics explain 7 n604 and 10 n1316? Ridge regression of z (each of 196 cells, 1500
random images of the 10k set) on: position (row, col one-hots), and local stats in the cell's 48 px window, the window
ABOVE it and the window BELOW it (luminance, contrast = luminance std, horizontal-edge energy |dI/dy|, vertical-edge
energy |dI/dx|, saturation). Report R^2 (held-out half) and standardized weights. Descriptive, no prediction."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(3)
pick = np.sort(rng.choice(len(ana), 1500, replace=False)); X = np.asarray(G._X[ana[pick]]).astype(np.float32) / 255
L = X @ np.array([.299, .587, .114], np.float32); sat = X.max(-1) - X.min(-1)
gy = np.abs(np.diff(L, axis=1, append=L[:, -1:])); gx = np.abs(np.diff(L, axis=2, append=L[:, :, -1:]))
P = np.pad  # pool 16x16 cells then 3x3 window sums


def cellpool(a, f=np.mean): return a.reshape(len(a), 14, 16, 14, 16).mean((2, 4))


Lc, L2c, syc, sxc, sc = cellpool(L), cellpool(L * L), cellpool(gy), cellpool(gx), cellpool(sat)


def win(a, dr):   # 3x3 cell window centred dr cells away vertically (edge-padded)
    a = np.pad(a, ((0, 0), (4, 4), (1, 1)), mode="edge"); out = 0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1): out = out + a[:, 4 + dr + i:4 + dr + i + 14, 1 + j:1 + j + 14]
    return out / 9


feats, fn = [], []
for dr, tag in ((0, "here"), (-3, "above"), (3, "below")):
    m = win(Lc, dr); v = np.sqrt(np.maximum(win(L2c, dr) - m * m, 0))
    for a, nm in ((m, "lum"), (v, "contrast"), (win(syc, dr), "hEdge"), (win(sxc, dr), "vEdge"), (win(sc, dr), "sat")): feats.append(a); fn.append(f"{nm}_{tag}")
S = np.stack(feats, -1).reshape(-1, len(fn)); S = (S - S.mean(0)) / S.std(0)
rr, cc = np.meshgrid(np.arange(14), np.arange(14), indexing="ij"); pos = np.concatenate([np.eye(14)[rr.ravel()], np.eye(14)[cc.ravel()]], 1)
POS = np.tile(pos, (len(pick), 1))


def ridge(A, yv, tr, te, lam=1.0):
    A1 = np.c_[A, np.ones(len(A))]; w = np.linalg.solve(A1[tr].T @ A1[tr] + lam * np.eye(A1.shape[1]), A1[tr].T @ yv[tr])
    p = A1[te] @ w; return 1 - ((yv[te] - p) ** 2).mean() / yv[te].var(), w


n = len(pick) * 196; img_of = np.repeat(np.arange(len(pick)), 196); tr, te = img_of < len(pick) // 2, img_of >= len(pick) // 2
for t in [(7, 604), (10, 1316)]:
    z = np.load(os.path.join(OUTD, f"t2_z_{t[0]}_{t[1]}.npy")).astype(np.float32)[pick].reshape(-1)
    for nm, A in (("position only", POS), ("stats only", S), ("both", np.c_[S, POS])):
        r2, w = ridge(A, z, tr, te); print(f"{t} {nm:14s} R2={r2:.3f}")
    _, w = ridge(S, z, tr, te); o = np.argsort(-np.abs(w[:-1]))
    print("   weights (stats only, z per 1 sd):", ", ".join(f"{fn[k]} {w[k]:+.2f}" for k in o[:10]))
    top = z > np.quantile(z, .99); print("   top-1% cells vs all, mean sd units:", ", ".join(f"{fn[k]} {S[top, k].mean():+.2f}" for k in range(len(fn))))
