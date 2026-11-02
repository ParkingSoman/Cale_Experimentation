# serves: TY2
"""The merge rule (Definition DY2, Theorem TY2, Proposition TY3) on torchvision ResNet-18's last block (Part C).

Map (found by a program, outside the trunk): one district = all 512 channels of the last block, region = P; plus a
tree of proportional claims (DM10) from weighted average-linkage merging (weights c_u, distance 1 - <zhat_u, zhat_v>),
every node a claim with its error eps_prop. Zoom: one group = the whole layer, target sizes k = 512 ... 1.
P = the train split (35 per class, uniform weights). Every reported number is on the sealed split (10 per class) unless
it says train.
Per k:
  merge      B^m: least-cost cut of the claim tree into k sets (TY3 part 3), closed-form weights (DY2)
  bound      E^2 / 4 (TY2 part 1), on train (P)
  merge*     B^m*: the same channels, logit weights and bias re-fit (convex, L-BFGS on train)
  prune*     keep the k channels with the largest c_u, logit weights re-fit (a class (b) build: one-member boxes)
  random*    a random partition into k sets of sizes as in the cut, merged as in DY2, re-fit
  trained    boxes and readers trained by SGD from merge* (an estimate of the least loss; a check, never the rule);
             only for the k in TRAIN_K
Writes results_rz_zoom.json and cache/r18/zoom_builds.npz (the stored weights per k: the one-time computation).
Usage: python rz_zoom.py [--smoke]
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "cache", "r18")
SMOKE = "--smoke" in sys.argv
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
KS = [512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1]
TRAIN_K = [256, 64, 16]
if SMOKE:
    KS = [512, 128, 16, 1]; TRAIN_K = [16]
rng = np.random.default_rng(5)
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)


def load(split, n=None):
    Z = np.load(os.path.join(C, f"z_{split}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(C, f"y_{split}.npy")); L = np.load(os.path.join(C, f"lab_{split}.npy"))
    ix = np.arange(len(Z)) if n is None else np.sort(rng.choice(len(Z), n, replace=False))
    return torch.from_numpy(np.ascontiguousarray(Z[ix])).to(DEV), torch.from_numpy(Y[ix]).to(DEV), L[ix]


Ztr, Ytr, Ltr = load("train", 3000 if SMOKE else None)
Zse, Yse, Lse = load("sealed", 1000 if SMOKE else None)
hd = np.load(os.path.join(C, "head.npz")); W = torch.from_numpy(hd["W"]).to(DEV); b = torch.from_numpy(hd["b"]).to(DEV)
n, Cn, T = Ztr.shape
log("loaded", tuple(Ztr.shape), tuple(Zse.shape))

# --- DM10 quantities on P (train, uniform p)
G = torch.zeros(Cn, Cn, device=DEV); ssq = torch.zeros(Cn, device=DEV); act = torch.zeros(Cn, device=DEV)
for i in range(0, n, 1000):
    z = Ztr[i:i + 1000].float()                       # B x C x T
    zz = z.permute(1, 0, 2).reshape(Cn, -1)
    G += zz @ zz.T; ssq += (zz ** 2).sum(1); act += torch.relu(z).mean(2).sum(0)
G /= n * T; ssq /= n * T
s = ssq.sqrt()
Wp = W - W.mean(0, keepdim=True)
c = s * Wp.norm(dim=0)
read = (c > 0) & (act > 0)
Gh = (G / (s[:, None] * s[None, :])).cpu().double().numpy()     # <zhat_u, zhat_v>
cn = c.cpu().double().numpy(); sn = s.cpu().double().numpy()
log("read members", int(read.sum()), "of", Cn)

# --- map finder: weighted average linkage over 1 - <zhat_u, zhat_v>
D = 1.0 - Gh.copy(); np.fill_diagonal(D, np.inf)
members = {u: [u] for u in range(Cn)}; wt = {u: cn[u] for u in range(Cn)}
active = list(range(Cn)); idx = {u: u for u in range(Cn)}
Dm = D.copy(); kids = {}; nxt = Cn; rows = {u: u for u in range(Cn)}
alive = np.ones(Cn, bool); slot_of = {u: u for u in range(Cn)}; node_of = {u: u for u in range(Cn)}
while alive.sum() > 1:
    sub = np.where(alive)[0]
    M = Dm[np.ix_(sub, sub)]
    a, bb = np.unravel_index(np.argmin(M), M.shape); i, j = sub[a], sub[bb]
    qi, qj = node_of[i], node_of[j]
    wi, wj = wt[qi], wt[qj]
    newrow = (wi * Dm[i] + wj * Dm[j]) / (wi + wj)
    Dm[i] = newrow; Dm[:, i] = newrow; Dm[i, i] = np.inf
    alive[j] = False; Dm[j] = np.inf; Dm[:, j] = np.inf
    kids[nxt] = (qi, qj); members[nxt] = members[qi] + members[qj]; wt[nxt] = wi + wj
    node_of[i] = nxt; nxt += 1
root = nxt - 1
log("tree built, nodes", nxt)


def eps_prop(Q):
    Q = np.array(Q); R = Q[cn[Q] > 0]
    om = cn[R] / cn[R].sum()
    ip = Gh[np.ix_(R, R)] @ om                       # <zhat_u, z_Q>
    zq2 = om @ Gh[np.ix_(R, R)] @ om
    d = np.sqrt(np.maximum(1 - 2 * ip + zq2, 0))
    return float((cn[R] * d).sum()) ** 2


eps = {q: (0.0 if q < Cn else eps_prop(members[q])) for q in range(nxt)}
f = {}; arg = {}
order = list(range(Cn)) + list(range(Cn, nxt))
for q in order:
    if q not in kids:
        f[q] = {1: 0.0}; arg[q] = {1: None}; continue
    l, r = kids[q]; t = {1: np.sqrt(eps[q])}; ta = {1: None}
    for k1, v1 in f[l].items():
        for k2, v2 in f[r].items():
            if v1 + v2 < t.get(k1 + k2, np.inf):
                t[k1 + k2] = v1 + v2; ta[k1 + k2] = (k1, k2)
    f[q] = t; arg[q] = ta


def cut(q, k):
    if arg[q][k] is None:
        return [members[q]]
    l, r = kids[q]; k1, k2 = arg[q][k]
    return cut(l, k1) + cut(r, k2)


log("DP done")


def merge_weights(parts):
    """DY2: in-weights M (C x k) on z, readers V (1000 x k)."""
    Mi = np.zeros((Cn, len(parts))); V = np.zeros((1000, len(parts)))
    Wn = W.cpu().double().numpy()
    for j, Q in enumerate(parts):
        Q = np.array(Q); R = Q[cn[Q] > 0]
        om = cn[R] / cn[R].sum()
        Mi[R, j] = om / sn[R]
        V[:, j] = (sn[R][None, :] * Wn[:, R]).sum(1)
    return torch.tensor(Mi, dtype=torch.float32, device=DEV), torch.tensor(V, dtype=torch.float32, device=DEV)


def feats(Z, Mi, beta=None):
    out = []
    for i in range(0, len(Z), 1000):
        z = Z[i:i + 1000].float().permute(0, 2, 1)            # B x T x C
        h = z @ Mi
        if beta is not None:
            h = h + beta
        out.append(torch.relu(h).mean(1))
    return torch.cat(out)


def logsm(y):
    return torch.log_softmax(y, 1)


def kl(yN, yB):
    a, bq = logsm(yN), logsm(yB)
    return float((a.exp() * (a - bq)).sum(1).mean())


def score(yN, yB, L):
    return {"kl": kl(yN, yB), "agree": float((yN.argmax(1) == yB.argmax(1)).float().mean()),
            "top1": float((yB.argmax(1).cpu().numpy() == L).mean())}


def refit(F, Y, V0, b0, iters=60):
    Vp = V0.clone().requires_grad_(True); bp = b0.clone().requires_grad_(True)
    tgt = torch.softmax(Y, 1)
    opt = torch.optim.LBFGS([Vp, bp], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -(tgt * logsm(F @ Vp.T + bp)).sum(1).mean()
        loss.backward(); return loss
    opt.step(closure)
    return Vp.detach(), bp.detach()


def train_boxes(Mi, V, b0, epochs=6, lr=3e-4):
    Mp = Mi.clone().requires_grad_(True); Vp = V.clone().requires_grad_(True); bp = b0.clone().requires_grad_(True)
    beta = torch.zeros(Mi.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Mp, Vp, bp, beta], lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 256):
            sel = perm[i:i + 256]
            z = Ztr[sel].float().permute(0, 2, 1)
            yb = torch.relu(z @ Mp + beta).mean(1) @ Vp.T + bp
            loss = (torch.softmax(Ytr[sel], 1) * (logsm(Ytr[sel]) - logsm(yb))).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return Mp.detach(), Vp.detach(), bp.detach(), beta.detach()


res = {"n_train": n, "n_sealed": len(Zse), "smoke": SMOKE, "levels": []}
store = {}
cn_t = torch.tensor(cn, dtype=torch.float32, device=DEV)
for k in KS:
    parts = cut(root, k)
    E = sum(np.sqrt(eps_prop(Q)) if len(Q) > 1 else 0.0 for Q in parts)
    Mi, V = merge_weights(parts)
    Ftr, Fse = feats(Ztr, Mi), feats(Zse, Mi)
    row = {"k": k, "bound_train": E ** 2 / 4, "sizes_max": max(len(Q) for Q in parts)}
    row["merge"] = score(Yse, Fse @ V.T + b, Lse); row["merge_train_kl"] = kl(Ytr, Ftr @ V.T + b)
    V2, b2 = refit(Ftr, Ytr, V, b)
    row["merge_refit"] = score(Yse, Fse @ V2.T + b2, Lse)
    store[f"k{k}_M"] = Mi.cpu().numpy(); store[f"k{k}_V"] = V2.cpu().numpy(); store[f"k{k}_b"] = b2.cpu().numpy()
    # prune: top-k channels by c_u
    keep = torch.argsort(cn_t, descending=True)[:k]
    Mp = torch.zeros(Cn, k, device=DEV); Mp[keep, torch.arange(k, device=DEV)] = 1.0
    Fp_tr, Fp_se = feats(Ztr, Mp), feats(Zse, Mp)
    V3, b3 = refit(Fp_tr, Ytr, W[:, keep], b)
    row["prune_refit"] = score(Yse, Fp_se @ V3.T + b3, Lse)
    # random partition with the cut's sizes
    perm = rng.permutation(Cn); sz = [len(Q) for Q in parts]; rp = []; o = 0
    for z_ in sz:
        rp.append(list(perm[o:o + z_])); o += z_
    Mr, Vr = merge_weights(rp)
    Fr_tr, Fr_se = feats(Ztr, Mr), feats(Zse, Mr)
    V4, b4 = refit(Fr_tr, Ytr, Vr, b)
    row["random_refit"] = score(Yse, Fr_se @ V4.T + b4, Lse)
    row["random_bound_train"] = sum(np.sqrt(eps_prop(Q)) if len(Q) > 1 else 0.0 for Q in rp) ** 2 / 4
    if k in TRAIN_K:
        Mt, Vt, bt, betat = train_boxes(Mi, V2, b2, epochs=1 if SMOKE else 6)
        row["trained"] = score(Yse, feats(Zse, Mt, betat) @ Vt.T + bt, Lse)
    res["levels"].append(row)
    log(f"k={k:3d} bound {row['bound_train']:.3g} | merge kl {row['merge']['kl']:.3f} | merge* {row['merge_refit']['kl']:.3f} "
        f"agree {row['merge_refit']['agree']:.3f} | prune* {row['prune_refit']['kl']:.3f} | random* {row['random_refit']['kl']:.3f}"
        + (f" | trained {row['trained']['kl']:.3f}" if 'trained' in row else ""))
res["N_sealed_top1"] = float((Yse.argmax(1).cpu().numpy() == Lse).mean())
res["tree"] = {"kids": {int(q): [int(a), int(bb)] for q, (a, bb) in kids.items()}, "eps": {int(q): v for q, v in eps.items() if q >= Cn}}
res["c_u"] = cn.tolist(); res["s_u"] = sn.tolist()
tag = "_smoke" if SMOKE else ""
json.dump(res, open(os.path.join(HERE, f"results_rz_zoom{tag}.json"), "w"))
if not SMOKE:
    np.savez(os.path.join(C, "zoom_builds.npz"), **store)
log("done")
