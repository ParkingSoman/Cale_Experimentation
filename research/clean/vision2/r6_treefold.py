"""R6: tree fold on block B (parent B-1, same stage), sealed accuracy. Rules: VISION-LOG.md R3 / R6."""
import json, sys
import numpy as np, mlx.core as mx, mlx.nn as nn
from scipy.special import erf
from convnext import *
from acts import next_stream

B = int(sys.argv[1]); P = B - 1
m = load_convnext(); X, y = load_images(); ana, _ = split(y)
assert m.stage_of[B] == m.stage_of[P]
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
cal = np.where(rank < 2)[0]; ev = np.where((rank >= 20) & (rank < 23))[0]
rng = np.random.default_rng(0); gelu = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
bp, bc = m.blocks[P], m.blocks[B]
gp, gc = np.array(bp.gamma, np.float64), np.array(bc.gamma, np.float64)
Wp2, Wc2 = np.array(bp.fc2.weight, np.float64), np.array(bc.fc2.weight, np.float64)
Dc = Wc2 * gc[:, None]; Dp = Wp2 * gp[:, None]; cbc = gc * np.array(bc.fc2.bias, np.float64)
Wc1, bc1 = np.array(bc.fc1.weight, np.float64), np.array(bc.fc1.bias, np.float64)
Hp, Hc = [], []
for i in range(0, len(cal), 100):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[cal[i:i + 100]])))))
    for j in range(P):
        x = m.blocks[j](x); x = next_stream(m, x, j)
    hp = nn.gelu(bp.fc1(bp.pre(x))); x2 = bp(x); hc = nn.gelu(bc.fc1(bc.pre(x2)))
    n, h, w, _ = hp.shape; pick = rng.integers(0, h * w, (n, 49))
    Hp.append(np.take_along_axis(np.array(hp).reshape(n, h * w, -1), pick[:, :, None], 1).reshape(-1, hp.shape[-1]))
    Hc.append(np.take_along_axis(np.array(hc).reshape(n, h * w, -1), pick[:, :, None], 1).reshape(-1, hc.shape[-1]))
Hp, Hc = np.concatenate(Hp).astype(np.float64), np.concatenate(Hc).astype(np.float64)
N = Hc.shape[1]; Y = Hc @ Dc.T + cbc
f1c, f2c, f2p = bc.fc1, bc.fc2, bp.fc2


def acc():
    return np.concatenate([np.array(m(mx.array(norm_img(np.asarray(X[ev[i:i + 100]]))))).argmax(1) for i in range(0, len(ev), 100)]) == y[ev]


def evaluate(Dk, bk, keep, Dpn):
    bp.fc2 = nn.Linear(N, Dp.shape[0]); bp.fc2.weight = mx.array((Dpn / gp[:, None]).astype(np.float32)); bp.fc2.bias = f2p.bias
    if len(keep):
        bc.fc1 = nn.Linear(Wc1.shape[1], len(keep)); bc.fc1.weight = mx.array(Wc1[keep].astype(np.float32)); bc.fc1.bias = mx.array(bc1[keep].astype(np.float32))
    else:
        bc.fc1 = nn.Linear(Wc1.shape[1], 1); bc.fc1.weight = mx.zeros((1, Wc1.shape[1])); bc.fc1.bias = mx.full((1,), -10.0); Dk = np.zeros((Dc.shape[0], 1))
    bc.fc2 = nn.Linear(Dk.shape[1], Dc.shape[0]); bc.fc2.weight = mx.array((Dk / gc[:, None]).astype(np.float32)); bc.fc2.bias = mx.array((bk / gc).astype(np.float32))
    r = acc(); bc.fc1, bc.fc2, bp.fc2 = f1c, f2c, f2p; return r


def ls(F, T, lam=1e-4):
    A = np.c_[F, np.ones(len(F))]; G = A.T @ A; G += np.eye(len(G)) * lam * np.trace(G) / len(G)
    S = np.linalg.solve(G, A.T @ T); return S[:-1].T, S[-1]


def boot(d, n=2000):
    r = np.random.default_rng(0); mm = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(mm, .025)), float(np.quantile(mm, .975))]


full = acc(); print("intact %.4f" % full.mean(), flush=True)
Hz = (Hp - Hp.mean(0)) / (Hp.std(0) + 1e-9)
en = np.linalg.norm(Hc, axis=0) * np.linalg.norm(Dc, axis=0); order = np.argsort(-en)
res = {"block": B, "intact": float(full.mean())}
for frac in (.03, .25):
    K = int(round(frac * N)); keep, rem = np.sort(order[:K]), np.sort(order[K:])
    Hcz = (Hc[:, rem] - Hc[:, rem].mean(0)) / (Hc[:, rem].std(0) + 1e-9)
    par = np.argsort(-np.abs(Hcz.T @ Hz / len(Hz)), 1)[:, :3]
    Rrem = Hc[:, rem] @ Dc[:, rem].T; arms = {}
    arms["drop"] = (None, Y.mean(0), np.array([], int), Dp)
    Dk, bk = ls(Hc[:, keep], Y); arms["cut_ls"] = (Dk, bk, keep, Dp)
    arms["cut_raw"] = (Dc[:, keep], cbc + Hc[:, rem].mean(0) @ Dc[:, rem].T, keep, Dp)
    for name, n in (("tree1", 1), ("tree3", 3)):
        A = np.zeros((len(rem), N))
        for i in range(len(rem)):
            ps = par[i, :n]; A[i, ps] = np.linalg.lstsq(Hp[:, ps], Hc[:, rem[i]], rcond=None)[0]
        Rh = (Hp @ A.T) @ Dc[:, rem].T; Dpn = Dp + Dc[:, rem] @ A
        Dk, bk = ls(Hc[:, keep], Y - Rh); arms[name] = (Dk, bk, keep, Dpn)
        arms[name + "_raw"] = (Dc[:, keep], cbc + (Rrem - Rh).mean(0), keep, Dpn)
    F = np.c_[Hc[:, keep], Hp]; half = np.arange(len(F)) < len(F) // 2; best = None
    for lam in (1e-3, 1e-2, 1e-1, 1):
        Dt, bt = ls(F[half], Y[half], lam); e = ((Y[~half] - F[~half] @ Dt.T - bt) ** 2).sum()
        if best is None or e < best[0]: best = (e, lam)
    Dt, bt = ls(F, Y, best[1]); arms["dense"] = (Dt[:, :K], bt, keep, Dp + Dt[:, K:])
    out = {"K": K, "arms": {}}; ok = {}
    for name, (Dk, bk, kp, Dpn) in arms.items():
        ok[name] = evaluate(Dk, bk, kp, Dpn); out["arms"][name] = float(ok[name].mean()); print(B, K, name, "%.4f" % ok[name].mean(), flush=True)
    for reg, cut in (("refit", "cut_ls"), ("raw", "cut_raw")):
        loss = full.mean() - ok[cut].mean()
        for n in ok:
            if n in ("drop", cut) or (reg == "raw") != n.endswith("_raw"): continue
            out.setdefault("share", {})[n] = float((ok[n].mean() - ok[cut].mean()) / loss)
            out.setdefault("vs_cut", {})[n] = boot(ok[n].astype(float) - ok[cut].astype(float))
    out["tree1_over_dense"] = out["share"]["tree1"] / out["share"]["dense"] if out["share"]["dense"] > 0 else None
    print(B, K, "shares", {k: round(v, 3) for k, v in out["share"].items()}, "tree1/dense", out["tree1_over_dense"], flush=True)
    res[f"frac{frac}"] = out
# attribution vs best-predictor parent, 100 sampled children of the 3% cut (top-20 analysis images each)
from acts import get_acts
amax, _ = get_acts(m) if B in (11, 12, 13, 14, 15, 16, 17) else (None, None)
K = int(round(.03 * N)); rem = np.sort(order[K:]); kids = np.random.default_rng(1).choice(rem, 100, replace=False)
Hcz = (Hc[:, kids] - Hc[:, kids].mean(0)) / (Hc[:, kids].std(0) + 1e-9); pred = np.argmax(np.abs(Hcz.T @ Hz / len(Hz)), 1)
agree, in_top10 = [], []
for k, c in enumerate(kids):
    imgs = ana[np.argsort(-amax[B][:, c])[:20]]
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[imgs])))))
    for j in range(P):
        x = m.blocks[j](x); x = next_stream(m, x, j)
    hp = nn.gelu(bp.fc1(bp.pre(x)))
    f = lambda h: nn.gelu(bc.fc1(bc.pre(x + bp.gamma * bp.fc2(h)))[..., int(c)]).sum()
    a = np.abs(np.array((hp * mx.grad(f)(hp)).sum((0, 1, 2)))); top = np.argsort(-a)
    agree.append(top[0] == pred[k]); in_top10.append(pred[k] in top[:10])
res["parent_agreement"] = {"top1_equal": float(np.mean(agree)), "pred_in_attr_top10": float(np.mean(in_top10))}
print("parent agreement", res["parent_agreement"])
json.dump(res, open(os.path.join(RES, f"vision-r6-treefold-b{B}.json"), "w"), indent=1)
