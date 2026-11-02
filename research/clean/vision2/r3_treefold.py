"""R3: tree fold on ConvNeXt-Tiny block 17, scored by sealed top-1. Rules: VISION-LOG.md R3 (+ deviation)."""
import json, sys
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
from acts import next_stream

C, P = 17, 16
m = load_convnext(); X, y = load_images(); ana, sea = split(y)
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
cal = np.where(rank < 2)[0]; ev = np.where((rank >= 20) & (rank < 25))[0]          # 2,000 cal, 5,000 sealed
b16, b17 = m.blocks[P], m.blocks[C]
g16, g17 = np.array(b16.gamma, np.float64), np.array(b17.gamma, np.float64)
W16, c16 = np.array(b16.fc2.weight, np.float64), np.array(b16.fc2.bias, np.float64)
W17, c17 = np.array(b17.fc2.weight, np.float64), np.array(b17.fc2.bias, np.float64)


def s_in(imgs, b, bs=100):
    out = []
    for i in range(0, len(imgs), bs):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[imgs[i:i + bs]])))))
        for j in range(b):
            x = m.blocks[j](x); x = next_stream(m, x, j)
        out.append(x)
    return out


S16c = s_in(cal, P)
H16 = np.concatenate([np.array(nn.gelu(b16.fc1(b16.pre(s)))).reshape(-1, 3072) for s in S16c]).astype(np.float64)
H17 = np.concatenate([np.array(nn.gelu(b17.fc1(b17.pre(next_stream(m, b16(s), P))))).reshape(-1, 3072) for s in S16c]).astype(np.float64)
D17 = W17 * g17[:, None]; Y = H17 @ D17.T + g17 * c17                    # block-17 MLP branch output (with gamma)
S16e = s_in(ev, P)
lab = y[ev]
base_acc = None


def evaluate(Dk_g, bk_g, keep, W16n):
    """Dk_g, bk_g: kept write (with gamma) and bias; W16n: block-16 fc2 weight (without gamma)."""
    b16.fc2.weight = mx.array(W16n.astype(np.float32))
    b17.fc1 = nn.Linear(768, len(keep)); b17.fc1.weight = mx.array(np.array(orig_fc1w)[keep]); b17.fc1.bias = mx.array(np.array(orig_fc1b)[keep])
    b17.fc2 = nn.Linear(len(keep), 768); b17.fc2.weight = mx.array((Dk_g / g17[:, None]).astype(np.float32)); b17.fc2.bias = mx.array((bk_g / g17).astype(np.float32))
    pred = np.concatenate([np.array(m.run_from(s, P)).argmax(1) for s in S16e])
    b16.fc2.weight = mx.array(W16.astype(np.float32)); b17.fc1, b17.fc2 = orig_fc1, orig_fc2
    return pred == lab


orig_fc1, orig_fc2 = b17.fc1, b17.fc2; orig_fc1w, orig_fc1b = b17.fc1.weight, b17.fc1.bias
full = evaluate(D17, g17 * c17, np.arange(3072), W16); print("intact acc %.4f" % full.mean(), flush=True)


def ls(F, T, lam=1e-4):
    A = np.c_[F, np.ones(len(F))]; G = A.T @ A; G += np.eye(len(G)) * lam * np.trace(G) / len(G)
    S = np.linalg.solve(G, A.T @ T); return S[:-1].T, S[-1]


def boot(d, n=2000):
    r = np.random.default_rng(0); mm = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(mm, .025)), float(np.quantile(mm, .975))]


Hz = (H16 - H16.mean(0)) / (H16.std(0) + 1e-9)
r2 = json.load(open(os.path.join(RES, "vision-r2-parents.json")))
res = {"intact": float(full.mean())}
for K in (96, 768):
    en = np.linalg.norm(H17, axis=0) * np.linalg.norm(D17, axis=0); order = np.argsort(-en)
    keep, rem = np.sort(order[:K]), np.sort(order[K:])
    Hc = (H17[:, rem] - H17[:, rem].mean(0)) / (H17[:, rem].std(0) + 1e-9)
    corr = Hc.T @ Hz / len(Hz); par = np.argsort(-np.abs(corr), 1)[:, :3]
    Rrem = H17[:, rem] @ D17[:, rem].T
    arms = {}
    Dk, bk = ls(H17[:, keep], Y); arms["cut_ls"] = (Dk, bk, W16)
    arms["cut_raw"] = (D17[:, keep], g17 * c17 + H17[:, rem].mean(0) @ D17[:, rem].T, W16)   # removed -> their mean (bias)
    for name, n in (("tree1", 1), ("tree3", 3)):
        A = np.zeros((len(rem), 3072))
        for i in range(len(rem)):
            ps = par[i, :n]; F = np.c_[H16[:, ps]]; A[i, ps] = np.linalg.lstsq(F, H17[:, rem[i]], rcond=None)[0]
        Rh = (H16 @ A.T) @ D17[:, rem].T
        W16n = W16 + (D17[:, rem] @ A) / g16[:, None]
        Dk, bk = ls(H17[:, keep], Y - Rh); arms[name] = (Dk, bk, W16n)
        arms[name + "_raw"] = (D17[:, keep], g17 * c17 + (Rrem - Rh).mean(0), W16n)
    # dense fold (RCF analog): ridge of [kept, H16] -> Y, lambda by held-out half of cal images
    F = np.c_[H17[:, keep], H16]; half = np.arange(len(F)) < len(F) // 2; best = None
    for lam in (1e-3, 1e-2, 1e-1, 1):
        Dt, bt = ls(F[half], Y[half], lam); e = ((Y[~half] - F[~half] @ Dt.T - bt) ** 2).sum()
        if best is None or e < best[0]: best = (e, lam)
    Dt, bt = ls(F, Y, best[1]); arms["dense"] = (Dt[:, :K], bt, W16 + Dt[:, K:] / g16[:, None])
    out = {"lam_dense": best[1], "arms": {}}; corr_ = {}
    for name, (Dk, bk, W16n) in arms.items():
        ok = evaluate(Dk, bk, keep, W16n); corr_[name] = ok; out["arms"][name] = float(ok.mean())
        print(K, name, "acc %.4f" % ok.mean(), flush=True)
    for reg, cut in (("refit", "cut_ls"), ("raw", "cut_raw")):
        for n in corr_:
            if (reg == "raw") != n.endswith("_raw") or n == cut: continue
            loss = full.mean() - corr_[cut].mean()
            out.setdefault("share", {})[n] = float((corr_[n].mean() - corr_[cut].mean()) / loss) if loss > 0 else None
            out.setdefault("vs_cut", {})[n] = boot(corr_[n].astype(float) - corr_[cut].astype(float))
    s = out["share"]; out["tree1_over_dense"] = s["tree1"] / s["dense"] if s.get("dense") else None
    print(K, "shares", s, "tree1/dense", out["tree1_over_dense"], flush=True)
    res[f"K{K}"] = out
json.dump(res, open(os.path.join(RES, "vision-r3-treefold.json"), "w"), indent=1)
