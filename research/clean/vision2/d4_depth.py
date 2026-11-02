"""D4: depth merge. Remove block B; blocks B and B+1 (2 x 3072 neurons) become ONE block (B+1's dwconv + LN) with K
neurons, derived with no training and no knobs:
  member votes from block B+1's view of the stream that ENTERS block B (block B is gone): conditional expectation
           z_i ~ v_i . pre'(x) + c_i  (covariance readings; target = the member's intact vote)
  groups = co-firing of intact outputs (ward, K groups over all 6144 neurons)
  read   = sum of members' z-scored votes; threshold = rate law (fires as often as its members)
  write  = expected intact output of blocks B and B+1 together, given the new activations (joint projection)
Readings on NREAD images per class (default 5), all positions, streamed. Score: sealed KL on the N11 3,000 images.
References (N11, blocks 16+17 at 50% each = 3072 kept): cut .285 KL; GN won back .238, early-stopped training .214.
usage: python d4_depth.py B K [GROUPS=ward|energy|pairs]"""
import os, sys, time, json
import numpy as np, mlx.core as mx, mlx.nn as nn
from scipy.cluster.hierarchy import linkage, fcluster
from convnext import *
t0 = time.time(); B, K = int(sys.argv[1]), int(sys.argv[2]); HOW = sys.argv[3] if len(sys.argv) > 3 else "ward"
m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18; gelu = nn.gelu; assert ST[B] == ST[B + 1]
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
b0, b1_ = m.blocks[B], m.blocks[B + 1]
def mats(blk): g = np.array(blk.gamma, np.float64); return (np.array(blk.fc1.weight, np.float64), np.array(blk.fc1.bias, np.float64),
                                                              np.array(blk.fc2.weight, np.float64) * g[:, None], np.array(blk.fc2.bias, np.float64) * g)
A1, a1, A2, a2 = mats(b0); C1, c1, C2, c2 = mats(b1_)
D16 = []; NREAD = int(os.environ.get("NREAD", 5)); READ = np.where(rank < NREAD)[0]; rng = np.random.default_rng(0)
def batches_read():
    for a in range(0, len(READ), 50):
        x = to_block(mx.array(norm_img(np.asarray(X[READ[a:a + 50]]))), B); x1 = b0(x); pn = b1_.pre(x)
        z = np.c_[np.array(b0.fc1(b0.pre(x))).reshape(-1, A1.shape[0]), np.array(b1_.fc1(b1_.pre(x1))).reshape(-1, C1.shape[0])].astype(np.float64)
        D16.append(np.array(x1 - x).reshape(-1, A1.shape[1]).astype(np.float64).mean(0)); yield z, np.array(pn).reshape(-1, A1.shape[1]).astype(np.float64), np.array(b1_(x1) - x).reshape(-1, A1.shape[1]).astype(np.float64)
# pass 1: conditional-expectation votes (streamed normal equations), stats, subsample for groups/quantiles
AtA = 0; AtZ = 0; N = 0; sub_h, sub_p, sub_z = [], [], []; rate_on = 0
for z, pn, o in batches_read():
    Ap = np.c_[pn, np.ones(len(pn))]; AtA = AtA + Ap.T @ Ap; AtZ = AtZ + Ap.T @ z; N += len(z); rate_on = rate_on + (z > 0).sum(0)
    k = rng.choice(len(z), len(z) // 12, replace=False); sub_z.append(z[k].astype(np.float32)); sub_p.append(pn[k].astype(np.float32))
Vc = np.linalg.pinv(AtA, rcond=1e-9) @ AtZ                                   # min-norm (LN outputs have a fixed channel sum)
Zs_ = np.concatenate(sub_z).astype(np.float64); Ps_ = np.concatenate(sub_p).astype(np.float64); Hh = np.array(gelu(mx.array(Zs_.astype(np.float32)))).astype(np.float64)
H2 = Zs_.shape[1]; Zhat = np.c_[Ps_, np.ones(len(Ps_))] @ Vc; mu, sd = Zhat.mean(0), Zhat.std(0) + 1e-9
r2 = 1 - ((Zhat - Zs_) ** 2).mean(0) / Zs_.var(0); print(f"readings {N} positions ({len(READ)} images); member votes from the new view R^2 (subsample): block {B} {np.median(r2[:3072]):.3f}, block {B + 1} {np.median(r2[3072:]):.3f}  {time.time() - t0:.0f}s", flush=True)
if HOW == "ward":
    Hz = (Hh - Hh.mean(0)) / (Hh.std(0) + 1e-9); Dm = np.clip(1 - Hz.T @ Hz / len(Hz), 0, 2); iu = np.triu_indices(H2, 1)
    lab = fcluster(linkage(Dm[iu], "ward"), K, "maxclust") - 1; del Dm; groups = [np.where(lab == g)[0] for g in range(lab.max() + 1)]
elif HOW == "energy":
    en = np.linalg.norm(Hh, axis=0) * np.r_[np.linalg.norm(A2, axis=0), np.linalg.norm(C2, axis=0)]; groups = [np.array([i]) for i in np.argsort(-en)[:K]]
if HOW in ("keep17", "drop17orig", "drop_mean", "ce17"): groups = [np.array([3072 + i]) for i in range(3072)]
cnt = np.bincount([len(g) for g in groups]); print(f"groups {len(groups)}; size counts 1..6 {cnt[1:7].tolist()}; mixed-block groups {sum((g < 3072).any() and (g >= 3072).any() for g in groups)}", flush=True)
rate = rate_on / N; nW1 = np.zeros((len(groups), A1.shape[1])); nb1 = np.zeros(len(groups))
for g, idx in enumerate(groups):
    u = (Vc[:-1, idx] / sd[idx]).sum(1); b = ((Vc[-1, idx] - mu[idx]) / sd[idx]).sum(); S = Ps_ @ u + b
    th = np.quantile(S, 1 - min(max(rate[idx].mean(), 1e-5), 1 - 1e-5)); nW1[g], nb1[g] = u, b - th
if HOW in ("keep17", "drop17orig"): nW1, nb1 = C1.copy(), c1.copy()     # block 17 keeps its own reads (on the stream without block 16)
# pass 2: the write = expected intact output given the new activations (streamed normal equations)
new1f = lambda pn: np.array(gelu(mx.array((pn @ nW1.T + nb1).astype(np.float32)))).astype(np.float64)
HtH = 0; HtO = 0; sH = 0; sO = 0; n2 = 0
for z, pn, o in batches_read():
    hn = new1f(pn); HtH = HtH + hn.T @ hn; HtO = HtO + hn.T @ o; sH = sH + hn.sum(0); sO = sO + o.sum(0); n2 += len(hn)
mH, mO = sH / n2, sO / n2; Chh = HtH / n2 - np.outer(mH, mH); Cho = HtO / n2 - np.outer(mH, mO)
nW2 = (np.linalg.pinv(Chh, rcond=1e-9) @ Cho).T; nb2 = mO - nW2 @ mH
if HOW == "drop17orig": nW2, nb2 = C2.copy(), c2.copy()             # pure block drop: nothing re-derived
m16 = np.mean(D16[:len(D16) // 2], 0)
if HOW == "drop_mean": nW1, nb1, nW2, nb2 = C1.copy(), c1.copy(), C2.copy(), c2 + m16
if HOW == "ce17": nW1, nb1, nW2, nb2 = Vc[:-1, 3072:].T.copy(), Vc[-1, 3072:].copy(), C2.copy(), c2 + m16   # reads restored by conditional expectation
print(f"write solved from {n2} positions  {time.time() - t0:.0f}s", flush=True)
# fold the new block's LN-affine-free form into block B+1's module: pre = b1_.pre (its own dwconv + LN), fc1 = (nW1, nb1)
def lin(Wt, bt): f = nn.Linear(Wt.shape[1], Wt.shape[0]); f.weight = mx.array(Wt.astype(np.float32)); f.bias = mx.array(bt.astype(np.float32)); return f
new1, new2 = lin(nW1, nb1), lin(nW2, nb2)
def merged(x): return x + new2(gelu(new1(b1_.pre(x))))
out = []
for a in range(0, len(EV), 50):
    x = to_block(mx.array(norm_img(np.asarray(X[EV[a:a + 50]]))), B); T = suffix(b1_(b0(x)), B + 1); L = suffix(merged(x), B + 1)
    lp = L - mx.logsumexp(L, 1, keepdims=True); tp = T - mx.logsumexp(T, 1, keepdims=True); out.append(np.array((mx.exp(tp) * (tp - lp)).sum(1)))
k = float(np.concatenate(out).mean())
print(f"blocks {B}+{B + 1} -> one block of {len(groups)} ({HOW}): sealed KL {k:.4f}   [N11 refs at 3072 kept: cut .285, GN ~.217, TR ~.224]  {time.time() - t0:.0f}s")
json.dump({"B": B, "K": K, "how": HOW, "kl": k}, open(os.path.join(RES, f"vision-d4-depth-{B}-{K}-{HOW}.json"), "w"))
