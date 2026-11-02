"""D1 step 1c: how much of the real coarse neuron (atto/nano twin of the parent) is its tiny FAMILY, vs the whole
parent+child blocks? Description only: linear R^2 fitted on half the images, reported on the other half.
usage: python d1_span.py KEY [KEY ...]"""
import os, sys
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); gelu = nn.gelu; CD = os.path.join(HF, "imagenet", "n7cache"); D = os.path.join(HF, "imagenet", "d1")
def r2(Xa, ya, Xb, yb, lam=1e-3):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; A = np.c_[(Xa - mu) / sd, np.ones(len(Xa))]; B = np.c_[(Xb - mu) / sd, np.ones(len(Xb))]
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ ya); return 1 - ((B @ w - yb) ** 2).mean() / yb.var()
for key in sys.argv[1:]:
    z = np.load(os.path.join(CD, key + ".npz")); d = np.load(os.path.join(D, key + ".npz")); pb, pn = (int(v) for v in z["p"]); cb = pb + 1
    cs = [int(c[1]) for c in z["cs"]]; XT = mx.array(z["XT"]).astype(mx.float32); R = XT.shape[1] * XT.shape[2]
    rng = np.random.default_rng(0); npos = min(40, R); pick = np.stack([rng.choice(R, npos, replace=False) for _ in range(XT.shape[0])])
    Hp, Hc = [], []
    for a in range(0, XT.shape[0], 50):
        x = XT[a:a + 50]; B = x.shape[0]; idx = mx.array((np.arange(B)[:, None] * R + pick[a:a + B]).reshape(-1))
        Hp.append(np.array(gelu(m.blocks[pb].fc1(m.blocks[pb].pre(x))).reshape(B * R, -1)[idx]))
        Hc.append(np.array(gelu(m.blocks[cb].fc1(m.blocks[cb].pre(m.blocks[pb](x)))).reshape(B * R, -1)[idx]))
    Hp, Hc = np.concatenate(Hp).astype(np.float64), np.concatenate(Hc).astype(np.float64); flat = (np.arange(XT.shape[0])[:, None] * R + pick).reshape(-1)
    half = np.repeat(np.arange(XT.shape[0]) % 2 == 0, npos); Hall = np.c_[Hp, Hc]
    print(f"\n{key}  (positions {len(flat)}; R^2 on held-back half of images)")
    for tn in ("atto_zp_z", "nano_zp_z", "zg"):
        t = np.array(gelu(mx.array(d[tn]))).astype(np.float64)[flat]
        fam = np.c_[Hp[:, pn], Hc[:, cs]]; c = np.abs([np.corrcoef(Hall[half, i], t[half])[0, 1] if Hall[half, i].std() > 0 else 0 for i in range(Hall.shape[1])])
        top = lambda k: Hall[:, np.argsort(-np.nan_to_num(c))[:k]]
        res = {"parent": r2(Hp[half, pn:pn + 1], t[half], Hp[~half, pn:pn + 1], t[~half]), "family": r2(fam[half], t[half], fam[~half], t[~half])}
        for k in (len(cs) + 1, 10, 50): res[f"top{k} by corr"] = r2(top(k)[half], t[half], top(k)[~half], t[~half])
        res["both blocks (6144)"] = r2(Hall[half], t[half], Hall[~half], t[~half], 1e-2)
        print(f"  {tn:10s} " + "  ".join(f"{k} {v:.3f}" for k, v in res.items()))
