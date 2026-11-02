"""D1 checks for one small neuron (target): (1) cross-model ceiling: R^2 from top-k parts in tiny vs in nano (and atto
for a nano target); (2) search tiny stages s-1 and s (s-1 pooled 2x2) -- is the stage-3 "general vehicle" gate a part?;
(3) functional form on the top-10 tiny parts: sum of GELUs vs GELU of summed votes vs max (no fitting: parts z-scored,
sign from corr, uniform weights; a threshold sweep is shown as a description); (4) class-vote bond among stage-4 parts.
usage: python d1_checks.py SMALL sb sn"""
import os, sys, json
import numpy as np, mlx.core as mx, mlx.nn as nn
from scipy.special import erf
from convnext import *
X, y = load_images(); g = nn.gelu; gl = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
CFG = {"tiny": ((3, 3, 9, 3), (96, 192, 384, 768)), "atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
small, sb, sn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); M = {k: load_convnext(k, *v) for k, v in CFG.items()}
stage = M[small].stage_of[sb]; ana, _ = split(y); I = ana[::20]; rng = np.random.default_rng(0); R = 56 // 2 ** stage
pick = np.stack([rng.choice(R * R, 30, replace=False) for _ in I])
def collect(name, stages):
    m = M[name]; ST = m.stage_of; out = {s: [] for s in stages}; ids = {s: [] for s in stages}
    for b in range(len(m.blocks)):
        if ST[b] in stages: ids[ST[b]] += [(b, n) for n in range(m.blocks[b].fc1.weight.shape[0])]
    for a in range(0, len(I), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[I[a:a + 50]]))))); acc = {s: [] for s in stages}; B = x.shape[0]
        idx = mx.array((np.arange(B)[:, None] * R * R + pick[a:a + B]).reshape(-1))
        for b in range(len(m.blocks)):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if ST[b] in stages:
                z = m.blocks[b].fc1(m.blocks[b].pre(x))
                if ST[b] < stage: Bz, Rz, _, K = z.shape; z = z.reshape(Bz, Rz // 2, 2, Rz // 2, 2, K).mean((2, 4))
                acc[ST[b]].append(z.reshape(-1, z.shape[-1])[idx])
            if ST[b] > max(stages): break
            x = m.blocks[b](x)
        for s in stages: out[s].append(np.array(mx.concatenate(acc[s], 1)).astype(np.float32))
    Z = np.concatenate([np.concatenate(out[s]) for s in stages], 1); return Z, sum((ids[s] for s in stages), [])
tgt = collect(small, [stage])[0][:, [i for i, n in enumerate(collect.__defaults__ or [])] if False else None] if False else None
Zs, ids_s = collect(small, [stage]); t = gl(Zs[:, ids_s.index((sb, sn))].astype(np.float64)); del Zs
half = np.repeat(np.arange(len(I)) % 2 == 0, 30)
def r2(F):
    A = np.c_[(F - F[half].mean(0)) / (F[half].std(0) + 1e-6), np.ones(len(F))]; w = np.linalg.solve(A[half].T @ A[half] + 1e-3 * half.sum() * np.eye(A.shape[1]), A[half].T @ t[half])
    return 1 - ((A[~half] @ w - t[~half]) ** 2).mean() / t[~half].var()
def corrs(H): Hz = (H - H.mean(0)) / (H.std(0) + 1e-9); return (Hz * ((t - t.mean()) / t.std())[:, None]).mean(0)
print(f"target {small} {sb}/{sn} (stage {stage}); positions {len(t)}")
fines = [("tiny", [stage - 1, stage]), ("tiny", [stage]), ("nano" if small != "nano" else "atto", [stage])]
keep = {}
for name, sts in fines:
    Z, ids = collect(name, sts); H = gl(Z.astype(np.float64)); c = corrs(H); o = np.argsort(-np.nan_to_num(c))
    print(f"  fine = {name} stages {sts}: " + "  ".join(f"top{k} R^2 {r2(H[:, o[:k]]):.3f}" for k in (1, 3, 10, 50)))
    print("     top parts:", [(ids[i], round(float(c[i]), 2)) for i in o[:10]])
    if name == "tiny" and sts == [stage - 1, stage]: keep = dict(Z=Z, ids=ids, c=c, o=o)
# (3) functional forms on the top-10 tiny parts (both stages)
Z, ids, c, o = keep["Z"].astype(np.float64), keep["ids"], keep["c"], keep["o"]; P = o[:10]
Zp = Z[:, P]; zs = (Zp - Zp.mean(0)) / Zp.std(0); Hp = gl(Zp); hs = Hp / Hp.std(0)
cc = lambda v: float(np.corrcoef(v, t)[0, 1])
print("  functional form on top-10 tiny parts (uniform weights, no fitting):")
print(f"     sum of GELUs {cc(hs.sum(1)):.3f}   max of GELUs {cc(hs.max(1)):.3f}   best single part {cc(hs[:, 0]):.3f}")
print("     GELU(sum of z-scored votes - theta): " + "  ".join(f"theta {th}: {cc(gl(zs.sum(1) - th)):.3f}" for th in (0, 4, 8, 12, 16, 24)))
print("     count of parts on (z>1): %.3f   OR (any part z>2): %.3f" % (cc((zs > 1).sum(1).astype(float)), cc((zs > 2).any(1).astype(float))))
# where exactly one part is strongly on: does the coarse neuron fire?
on = zs > 2; one = on.sum(1) == 1; many = on.sum(1) >= 3; none = on.sum(1) == 0
print(f"     coarse (z) when 0 parts on {((t - t.mean()) / t.std())[none].mean():.2f} | exactly 1 {((t - t.mean()) / t.std())[one].mean():.2f} (n {one.sum()}) | >=3 {((t - t.mean()) / t.std())[many].mean():.2f} (n {many.sum()})")
# (4) class-vote bond (stage 4 parts only)
m = M["tiny"]; Wf = np.array(m.fc.weight) * np.array(m.head_norm.weight)[None]
def vote(n): b, j = n; return Wf @ (np.array(m.blocks[b].fc2.weight[:, j]) * np.array(m.blocks[b].gamma))
s4 = [ids[i] for i in o[:40] if m.stage_of[ids[i][0]] == 3][:20]
if len(s4) >= 5:
    V = np.stack([vote(n) for n in s4]); V = V - V.mean(1, keepdims=True); V /= np.linalg.norm(V, axis=1, keepdims=True); iu = np.triu_indices(len(V), 1)
    rn = [(b, int(rng.integers(3072))) for b, _ in s4]; Vr = np.stack([vote(n) for n in rn]); Vr = Vr - Vr.mean(1, keepdims=True); Vr /= np.linalg.norm(Vr, axis=1, keepdims=True)
    names = [n.split(",")[0] for n in class_names()]; mean_vote = (V * np.sign(V @ V.mean(0))[:, None]).mean(0)
    print(f"  class-vote cos among {len(s4)} stage-4 parts: |cos| {np.abs(V @ V.T)[iu].mean():.3f} signed {(V @ V.T)[iu].mean():+.3f};  random |cos| {np.abs(Vr @ Vr.T)[iu].mean():.3f} signed {(Vr @ Vr.T)[iu].mean():+.3f}")
    print("     shared vote, top classes:", [names[i] for i in np.argsort(-mean_vote)[:12]])
