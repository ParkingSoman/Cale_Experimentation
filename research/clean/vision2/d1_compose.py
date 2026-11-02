"""D1 step 1d: what is a real coarse neuron made of? For a small-model neuron (atto/nano), find the tiny neurons in the
same stage that track it best (correlation over positions), and place each in map v2's tree relative to the tiny
family parent: self, ancestor, descendant, sibling (same map parent), cousin (same grandparent), other.
Then R^2 (held-back half of images) of the small neuron from groups defined by the tree.
usage: python d1_compose.py KEY [small=atto|nano] [target=zp]"""
import os, sys, json
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
X, y = load_images(); gelu = nn.gelu; m = load_convnext(); ST = m.stage_of
CFG = {"atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
key = sys.argv[1]; sname = sys.argv[2] if len(sys.argv) > 2 else "atto"; tgt = sys.argv[3] if len(sys.argv) > 3 else "zp"
z = np.load(os.path.join(HF, "imagenet", "n7cache", key + ".npz")); d = np.load(os.path.join(HF, "imagenet", "d1", key + ".npz"))
pb, pn = (int(v) for v in z["p"]); stage = ST[pb]; sb, sn = (int(v) for v in d[f"{sname}_{tgt}_id"]); sm = load_convnext(sname, *CFG[sname])
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]
par = {tuple(map(int, k.split("_"))): tuple(v["parent"]) for k, v in MAP.items()}
kids = {}
for c, p in par.items(): kids.setdefault(p, []).append(c)
def chain(n):
    out = []
    while n in par and len(out) < 40: n = par[n]; out.append(n)
    return out
P = (pb, pn); anc = chain(P); sub = set(); st = [P]
while st:
    n = st.pop()
    for c in kids.get(n, []): sub.add(c); st.append(c)
def rel(n):
    if n == P: return "self"
    if n in anc: return f"ancestor-{anc.index(n) + 1}"
    if n in sub: return "descendant"
    if par.get(n) == par.get(P): return "sibling"
    if len(anc) > 1 and par.get(par.get(n)) == anc[1]: return "cousin"
    return "other"
I = z["tr"][::3]; rng = np.random.default_rng(0)
blocks = [b for b in range(len(m.blocks)) if ST[b] == stage]; T, S = [], []
for a in range(0, len(I), 50):
    xs = mx.array(norm_img(np.asarray(X[I[a:a + 50]])))
    x = m.stem_norm(m.stem(xs)); acc = []
    for b in range(len(m.blocks)):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        if ST[b] == stage: acc.append(gelu(m.blocks[b].fc1(m.blocks[b].pre(x))))
        if ST[b] > stage: break
        x = m.blocks[b](x)
    h = mx.concatenate(acc, -1); B, R, C, K = h.shape; pick = np.stack([rng.choice(R * C, 30, replace=False) for _ in range(B)])
    idx = mx.array((np.arange(B)[:, None] * R * C + pick).reshape(-1)); T.append(np.array(h.reshape(-1, K)[idx]))
    x = sm.stem_norm(sm.stem(xs)); S2 = sm.stage_of
    for b in range(sb + 1):
        if b > 0 and S2[b] != S2[b - 1]: x = sm.down[S2[b] - 1](sm.down_norm[S2[b] - 1](x))
        if b == sb: S.append(np.array(gelu(sm.blocks[b].fc1(sm.blocks[b].pre(x))[..., sn]).reshape(-1)[idx]))
        x = sm.blocks[b](x)
T = np.concatenate(T).astype(np.float64); s = np.concatenate(S).astype(np.float64)
ids = [(b, n) for b in blocks for n in range(m.blocks[b].fc1.weight.shape[0])]
Tz = (T - T.mean(0)) / (T.std(0) + 1e-9); c = (Tz * ((s - s.mean()) / s.std())[:, None]).mean(0)
o = np.argsort(-c); half = np.repeat(np.arange(len(T) // 30) % 2 == 0, 30)
def r2(cols):
    A = np.c_[Tz[:, cols], np.ones(len(T))]; w = np.linalg.solve(A[half].T @ A[half] + 1e-3 * half.sum() * np.eye(A.shape[1]), A[half].T @ s[half])
    return 1 - ((A[~half] @ w - s[~half]) ** 2).mean() / s[~half].var()
print(f"{key}: small {sname} {sb}/{sn} (twin of {tgt}); tiny parent {P}, map chain {anc[:4]}; positions {len(s)}")
from collections import Counter
print("top 15 tiny neurons by corr:"); [print(f"  {ids[i]}  corr {c[i]:.3f}  {rel(ids[i])}") for i in o[:15]]
print("relation counts among top 50:", dict(Counter(rel(ids[i]).split('-')[0] for i in o[:50])))
pos = {n: i for i, n in enumerate(ids)}; grp = lambda S_: [pos[n] for n in S_ if n in pos]
fam = grp([P] + [n for n in sub if n[0] == pb + 1]); sibs = grp([n for n in kids.get(par.get(P), [])]); subtree = grp([P] + list(sub))
gp_sub = []
if anc: st = [anc[0]]; gp_sub = [anc[0]]
while st:
    n = st.pop()
    for c_ in kids.get(n, []): gp_sub.append(c_); st.append(c_)
gp_sub = grp(gp_sub)
for name, cols in [("parent", grp([P])), ("parent+children(next block)", fam), ("parent's whole subtree", subtree),
                   ("parent + siblings", sorted(set(sibs + grp([P])))), ("map-parent's whole subtree", gp_sub), ("top10 by corr", list(o[:10])), ("top50 by corr", list(o[:50]))]:
    if len(cols) and len(cols) < 3000: print(f"  R^2 {name:30s} n={len(cols):4d}  {r2(cols):.3f}")
