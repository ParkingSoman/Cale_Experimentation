"""R2: parent diagnostics for block-17 neurons in ConvNeXt-Tiny (vision analog of TF-DIAG / TF-CTX).
Rules: VISION-LOG.md R2."""
import json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import get_acts, next_stream

C, P, K = 17, 16, 96
m = load_convnext(); X, y = load_images(); ana, sea = split(y); amax, _ = get_acts(m)
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
cal = np.where(rank < 2)[0]                                   # 2 analysis images per class = 2,000
tr_img, te_img = cal[::2], cal[1::2]


def hid(imgs, blocks, bs=100):
    out = {b: [] for b in blocks}
    for i in range(0, len(imgs), bs):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[imgs[i:i + bs]])))))
        for b in range(18):
            if b > 0: x = next_stream(m, x, b - 1)
            blk = m.blocks[b]
            if b in blocks: out[b].append(np.array(nn.gelu(blk.fc1(blk.pre(x)))).reshape(-1, blk.fc1.weight.shape[0]))
            x = blk(x)
            if b == max(blocks): break
    return {b: np.concatenate(v).astype(np.float64) for b, v in out.items()}


Htr, Hte = hid(tr_img, (15, 16, 17)), hid(te_img, (15, 16, 17))
blk = m.blocks[C]; Dw = (np.array(blk.fc2.weight, np.float64) * np.array(blk.gamma, np.float64)[:, None])  # (768, 3072) incl. gamma
en = np.linalg.norm(Htr[C], axis=0) * np.linalg.norm(Dw, axis=0); rem = np.sort(np.argsort(-en)[K:])
R_te = Hte[C][:, rem] @ Dw[:, rem].T; den = (R_te ** 2).sum(); Rtr = Htr[C][:, rem] @ Dw[:, rem].T

# per-image attribution of block-16 (and 15) hiddens to a child's spatially summed activation
def stream_in(imgs, b):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[imgs])))))
    for j in range(b):
        x = m.blocks[j](x); x = next_stream(m, x, j)
    return x


rng = np.random.default_rng(0); kids = rng.choice(rem, 100, replace=False)
modal, mass, attr_mean = [], [], {}
for c in kids:
    imgs = ana[np.argsort(-amax[C][:, c])[:20]]
    s16 = stream_in(imgs, P); b16 = m.blocks[P]; h16 = nn.gelu(b16.fc1(b16.pre(s16)))
    f = lambda h: nn.gelu(blk.fc1(blk.pre(next_stream(m, s16 + b16.gamma * b16.fc2(h), P)))[..., int(c)]).sum()
    a = np.array((h16 * mx.grad(f)(h16)).sum((1, 2)))          # (20 images, 3072)
    top = np.abs(a).argmax(1)
    modal.append(np.bincount(top).max() / len(top)); mass.append(float(np.median(np.abs(a).max(1) / np.abs(a).sum(1))))
    attr_mean[int(c)] = np.abs(a).mean(0)
ctx = {"modal_parent_share_median": float(np.median(modal)), "top_parent_share_of_attr_mass_median": float(np.median(mass)),
       "uniform_share": 1 / 3072, "lm_reference": {"modal": [.10, .13], "mass": [.006, .008]}}
print("CTX", ctx, flush=True)

# held-out R^2 of the removed output (only the 100 sampled kids, so top-n is comparable to dense on the same kids)
kix = np.searchsorted(rem, kids)
Rk_tr = Htr[C][:, kids] @ Dw[:, kids].T; Rk_te = Hte[C][:, kids] @ Dw[:, kids].T; denk = (Rk_te ** 2).sum()
G16 = Htr[P].T @ Htr[P]; X16 = Htr[C][:, kids].T @ Htr[P]


def r2_top(n):
    A = np.zeros((len(kids), 3072))
    for i, c in enumerate(kids):
        ps = np.argsort(-attr_mean[int(c)])[:n]; g = G16[np.ix_(ps, ps)] + np.eye(n) * 1e-3 * np.trace(G16) / 3072
        A[i, ps] = np.linalg.solve(g, X16[i, ps])
    Rh = (Hte[P] @ A.T) @ Dw[:, kids].T; return float(1 - ((Rk_te - Rh) ** 2).sum() / denk)


def r2_dense(F_tr, F_te):
    best = -9; g = F_tr.T @ F_tr
    for lam in (1e-3, 1e-2, 1e-1, .3, 1, 3, 10):
        W = np.linalg.solve(g + np.eye(len(g)) * lam * np.trace(g) / len(g), F_tr.T @ Rk_tr)
        best = max(best, float(1 - ((Rk_te - F_te @ W) ** 2).sum() / denk))
    return best


out = {"ctx": ctx, "L-1": {f"top{n}": r2_top(n) for n in (1, 3, 10, 30, 100)}}
out["L-1"]["dense"] = r2_dense(Htr[P], Hte[P])
out["pooled_15_16_dense"] = r2_dense(np.c_[Htr[15], Htr[P]], np.c_[Hte[15], Hte[P]])
print(json.dumps(out, indent=0))
json.dump(out, open(os.path.join(RES, "vision-r2-parents.json"), "w"), indent=1)
