"""D3: block B of tiny from 3072 to K neurons by the derived consensus merge (no training, no knobs):
  groups    = co-firing (ward on 1 - corr of GELU outputs; corr > 0 only matters through ward)
  read      = sum of members' z-scored votes (same block -> exact)
  threshold = rate law (the new neuron fires as often as its members on average)
  write     = expected stream output of the block given the new activations (joint projection, pinv): the Bayes answer
              under linear readers
Readings on CAL (1 image per class, all positions). Score: KL(intact || edited) on the 3,000 sealed images used by
N10/N11 (rank 20-22). References at K = 1536 (block 16): cut lowest-energy half .0826 KL; GN won back .226, training .216.
usage: python d3_merge.py BLOCK K [GROUPING=ward|energy_cut|random]"""
import os, sys, time, json
import numpy as np, mlx.core as mx, mlx.nn as nn
from scipy.cluster.hierarchy import linkage, fcluster
from convnext import *
t0 = time.time(); B, K = int(sys.argv[1]), int(sys.argv[2]); HOW = sys.argv[3] if len(sys.argv) > 3 else "ward"
m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18; gelu = nn.gelu
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
CAL = np.where(rank < 1)[0]; EV = np.where((rank >= 20) & (rank < 23))[0]
def to_block(xs, b):
    x = m.stem_norm(m.stem(xs))
    for i in range(b):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
    return x
def suffix(h, b):
    for i in range(b + 1, NB):
        if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
        h = m.blocks[i](h)
    return m.head(h)
blk = m.blocks[B]; W1 = np.array(blk.fc1.weight, np.float64); b1 = np.array(blk.fc1.bias, np.float64)
W2 = np.array(blk.fc2.weight, np.float64) * np.array(blk.gamma, np.float64)[:, None]; b2 = np.array(blk.fc2.bias, np.float64) * np.array(blk.gamma, np.float64)
# readings: pre (LN output) at all CAL positions
P = []
for a in range(0, len(CAL), 50): P.append(np.array(blk.pre(to_block(mx.array(norm_img(np.asarray(X[CAL[a:a + 50]]))), B))).reshape(-1, W1.shape[1]))
P = np.concatenate(P).astype(np.float64); Z = P @ W1.T + b1; Hh = np.array(gelu(mx.array(Z.astype(np.float32)))).astype(np.float64)
mu, sd = Z.mean(0), Z.std(0) + 1e-9; print(f"readings: {len(P)} positions, {time.time() - t0:.0f}s", flush=True)
H = W1.shape[0]
if HOW == "ward":
    sub = np.random.default_rng(0).choice(len(Hh), min(len(Hh), 20000), replace=False); Hz = Hh[sub]; Hz = (Hz - Hz.mean(0)) / (Hz.std(0) + 1e-9)
    Cm = Hz.T @ Hz / len(Hz); Dm = np.clip(1 - Cm, 0, 2); iu = np.triu_indices(H, 1)
    lab = fcluster(linkage(Dm[iu], "ward"), K, "maxclust") - 1; groups = [np.where(lab == g)[0] for g in range(lab.max() + 1)]
elif HOW == "random":
    lab = np.random.default_rng(0).permutation(np.arange(H) % K); groups = [np.where(lab == g)[0] for g in range(K)]
else: groups = None
sizes = np.bincount([len(g) for g in groups]) if groups else None
if groups: print(f"groups: {len(groups)}; size counts (1,2,3,...): {sizes[1:8].tolist()} max {max(len(g) for g in groups)}", flush=True)
if HOW == "energy_cut":    # reference: keep the K highest-energy neurons, others -> their mean (= N10 cut at K=1536)
    en = np.linalg.norm(Hh, axis=0) * np.linalg.norm(W2, axis=0); keep = np.argsort(-en)[:K]; drop = np.setdiff1d(np.arange(H), keep)
    nW1, nb1, nW2, nb2 = W1[keep], b1[keep], W2[:, keep], b2 + W2[:, drop] @ Hh[:, drop].mean(0)
else:
    nW1 = np.zeros((len(groups), W1.shape[1])); nb1 = np.zeros(len(groups))
    for g, idx in enumerate(groups):
        u = (W1[idx] / sd[idx, None]).sum(0); b = ((b1[idx] - mu[idx]) / sd[idx]).sum(); S = P @ u + b
        rate = (Z[:, idx] > 0).mean(); th = np.quantile(S, 1 - min(max(rate, 1e-5), 1 - 1e-5)); nW1[g], nb1[g] = u, b - th
    Hn = np.array(gelu(mx.array((P @ nW1.T + nb1).astype(np.float32)))).astype(np.float64); O = Hh @ W2.T
    Hc = Hn - Hn.mean(0); Oc = O - O.mean(0); nW2 = (np.linalg.pinv(Hc) @ Oc).T; nb2 = b2 + O.mean(0) - nW2 @ Hn.mean(0)
    r2 = 1 - ((Hc @ nW2.T - Oc) ** 2).sum() / (Oc ** 2).sum(); print(f"write: share of the block's stream output kept (CAL) {r2:.3f}", flush=True)
def lin(Wt, bt): f = nn.Linear(Wt.shape[1], Wt.shape[0]); f.weight = mx.array(Wt.astype(np.float32)); f.bias = mx.array(bt.astype(np.float32)); return f
orig = (blk.fc1, blk.fc2, blk.gamma)
def kl_eval(edited):
    out = []
    for a in range(0, len(EV), 50):
        xs = mx.array(norm_img(np.asarray(X[EV[a:a + 50]]))); x = to_block(xs, B); T = suffix(m.blocks[B](x), B)
        if edited: blk.fc1, blk.fc2, blk.gamma = lin(nW1, nb1), lin(nW2, nb2), mx.ones((W2.shape[0],))
        L = suffix(blk(x), B); blk.fc1, blk.fc2, blk.gamma = orig
        lp = L - mx.logsumexp(L, 1, keepdims=True); tp = T - mx.logsumexp(T, 1, keepdims=True)
        out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
    return float(np.concatenate(out).mean())
k = kl_eval(True); print(f"block {B} -> {len(nb1)} neurons, {HOW}: sealed KL {k:.4f}  (cut ref .0826 at 1536 in block 16; share vs that cut {(0.0826 - k) / 0.0826:+.3f})  {time.time() - t0:.0f}s")
json.dump({"block": B, "K": K, "how": HOW, "kl": k}, open(os.path.join(RES, f"vision-d3-merge-{B}-{K}-{HOW}.json"), "w"))
