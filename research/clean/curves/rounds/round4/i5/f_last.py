# serves: TQ31
"""Step 6. (a) ResNet-18: one-way against reciprocal links, by kind. (b) lambda_min of the reciprocal link graph. (c) Movement energy captured by a code that is a linear read of z, against a free code, at k = 14.

Fixed before the run. Links as in e_more.py: mass >= .01 of the off-own table, table from half B. Kind by cosW.
Read: the fit in cache/map/fit_resnet18_read_14.npz, if it holds an encoder.
PREDICTION: side links mostly one-way. Read capture far below free capture.
"""
import numpy as np

HERE = "/Users/shonusengupta/Cale_Experimentation/research/clean/curves"
Y = np.load(f"{HERE}/cache/map/resnet18_y.npy").astype(np.float64)
meta = np.load(f"{HERE}/cache/map/resnet18_meta.npz")
lab = meta["labels"].astype(int)
Wo = meta["W"].astype(np.float64)
K = 1000
e = np.exp(Y - Y.max(1, keepdims=True))
Sx = e / e.sum(1, keepdims=True)
gen = np.random.default_rng(7)
ib = [gen.permutation(np.where(lab == c)[0])[10:] for c in range(K)]
SB = np.stack([Sx[i].mean(0) for i in ib])
T = SB.copy()
np.fill_diagonal(T, 0)
T = T / T.sum(1, keepdims=True)
un = Wo / np.linalg.norm(Wo, axis=1, keepdims=True)
cosW = un @ un.T
H = T >= .01
R = H & H.T
for name, mask in (("family cosW>=.3", cosW >= .3), ("middle", (cosW < .3) & (cosW >= .1)), ("side cosW<.1", cosW < .1)):
    n = (H & mask).sum()
    r = (R & mask).sum()
    print(f"{name}: directed links {n}, reciprocated {r} = {r / max(n, 1):.3f}; mean T mass per row in these links {(T * (H & mask)).sum() / K:.3f}, of it reciprocated {(T * (R & mask)).sum() / K:.3f}")
print("mean T mass outside all links:", 1 - (T * H).sum() / K)
A = R.astype(float)
np.fill_diagonal(A, 0)
d = A.sum(1)
ok = d > 0
N = A[ok][:, ok] / np.sqrt(np.outer(d[ok], d[ok]))
lam = np.linalg.eigvalsh(N)
print("reciprocal link graph: nodes", int(ok.sum()), "edges", int(A.sum() / 2), "lambda_min", lam[0], "uncut lower (1+lambda_min)/2 =", (1 + lam[0]) / 2)
f = np.load(f"{HERE}/cache/map/fit_resnet18_read_14.npz")
print("read fit keys", {k: f[k].shape for k in f.files})

# (c) the read's capture on the same movement energy, metric = class row table, kept class offsets, one free constant per photo
Sall = np.stack([Sx[lab == c].mean(0) for c in range(K)])
Vr, Mr = f["V"].astype(np.float64), f["M"].astype(np.float64)
for nm in ("h", "z", "m"):
    Zc = np.load(f"{HERE}/cache/map/resnet18_{nm}.npy").astype(np.float64)
    code = Zc @ Mr
    Etot = Eres = 0.
    for c in range(K):
        ix = lab == c
        d = Y[ix] - Y[ix].mean(0)
        p = (code[ix] - code[ix].mean(0)) @ Vr.T
        w = Sall[c] / Sall[c].sum()
        dd = d - (d @ w)[:, None]
        r = d - p
        rr = r - (r @ w)[:, None]
        Etot += (dd ** 2 @ w).sum()
        Eres += (rr ** 2 @ w).sum()
    print(f"read k=14 code from {nm}: capture {1 - Eres / Etot:.3f}")
