"""D1: look at the coarse neuron. Per family, at every position of the family's 1,500 analysis images:
tiny parent / children / solved (GN) parent / trained (L1) parent pre-activations, and the best-matching atto and nano
neurons (twins of the parent, of each child, of the solved parent). No fitting. Output: hf/imagenet/d1/<key>.npz
usage: python d1_look.py KEY [KEY ...]"""
import os, sys, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
t0 = time.time(); X, y = load_images(); gelu = nn.gelu
CD = os.path.join(HF, "imagenet", "n7cache"); OUT = os.path.join(HF, "imagenet", "d1"); os.makedirs(OUT, exist_ok=True)
CFG = {"atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
tiny = load_convnext(); ST = tiny.stage_of
SMALL = {k: load_convnext(k, *v) for k, v in CFG.items()}

def small_stage_acts(sm, I, stage):
    """generator over batches: (B*R*C, K) fc1 PRE-activations of every neuron in `stage` of model sm, plus ids."""
    S = sm.stage_of; ids = [(b, n) for b in range(len(sm.blocks)) if S[b] == stage for n in range(sm.blocks[b].fc1.weight.shape[0])]
    for a in range(0, len(I), 50):
        x = sm.stem_norm(sm.stem(mx.array(norm_img(np.asarray(X[I[a:a + 50]]))))); acc = []
        for b in range(len(sm.blocks)):
            if b > 0 and S[b] != S[b - 1]: x = sm.down[S[b] - 1](sm.down_norm[S[b] - 1](x))
            if S[b] == stage: z = sm.blocks[b].fc1(sm.blocks[b].pre(x)); acc.append(z.reshape(-1, z.shape[-1]))
            if S[b] > stage: break
            x = sm.blocks[b](x)
        yield np.array(mx.concatenate(acc, 1)).astype(np.float64), ids

for key in sys.argv[1:]:
    z = np.load(os.path.join(CD, key + ".npz")); g = np.load(os.path.join(CD, "gn_" + key + ".npz"))
    pb, pn = (int(v) for v in z["p"]); cs = [tuple(int(v) for v in c) for c in z["cs"]]; cb = pb + 1; tr = z["tr"]
    XT = mx.array(z["XT"]).astype(mx.float32); blk = tiny.blocks[pb]; stage = ST[pb]; assert ST[cb] == stage
    U = {"gn": (g["u"], float(g["b0"])), "L": (z["Lu"], float(z["Lb"]))}
    rec = {k: [] for k in ["zp", "zg", "zL"] + [f"zc{j}" for j in range(len(cs))]}
    for a in range(0, XT.shape[0], 50):
        x = XT[a:a + 50]; pre = blk.pre(x); zz = blk.fc1(pre)
        rec["zp"].append(np.array(zz[..., pn]).reshape(-1))
        for k, (u, b0) in U.items(): rec["z" + ("g" if k == "gn" else "L")].append(np.array(pre @ mx.array(u.astype(np.float32)) + b0).reshape(-1))
        x1 = blk(x); zc = tiny.blocks[cb].fc1(tiny.blocks[cb].pre(x1))
        for j, (_, cn) in enumerate(cs): rec[f"zc{j}"].append(np.array(zc[..., cn]).reshape(-1))
    rec = {k: np.concatenate(v).astype(np.float32) for k, v in rec.items()}
    R = XT.shape[1]; out = dict(rec, p=z["p"], cs=z["cs"], tr=tr, grid=R)
    targets = {k: np.array(gelu(mx.array(v))).astype(np.float64) for k, v in rec.items()}
    for name, sm in SMALL.items():
        # pass 1: streaming correlation of every small neuron with each target (post-GELU, all positions)
        n, s1, s2, st, ids = 0, 0, 0, {k: 0 for k in targets}, None
        for A, ids in small_stage_acts(sm, tr, stage):
            A = np.array(gelu(mx.array(A.astype(np.float32)))).astype(np.float64); r = slice(n, n + A.shape[0]); n += A.shape[0]
            s1 = s1 + A.sum(0); s2 = s2 + (A * A).sum(0)
            for k, t in targets.items(): st[k] = st[k] + t[r] @ A
        mu = s1 / n; sd = np.sqrt(s2 / n - mu ** 2) + 1e-9
        corr = {k: (st[k] / n - mu * targets[k].mean()) / (sd * targets[k].std()) for k in targets}
        pick = {k: int(np.argmax(corr[k])) for k in targets}
        cols = sorted(set(pick.values())); ztw = []
        for A, _ in small_stage_acts(sm, tr, stage): ztw.append(A[:, cols].astype(np.float32))
        ztw = np.concatenate(ztw)
        for k, j in pick.items():
            out[f"{name}_{k}_id"] = np.array(ids[j]); out[f"{name}_{k}_corr"] = corr[k][j]; out[f"{name}_{k}_z"] = ztw[:, cols.index(j)]
            out[f"{name}_{k}_top5"] = np.array([ids[i] + (float(corr[k][i]),) for i in np.argsort(-corr[k])[:5]])
        print(key, name, {k: (ids[j], round(float(corr[k][j]), 3)) for k, j in pick.items()}, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, key + ".npz"), **out)
print("done", f"{time.time() - t0:.0f}s")
