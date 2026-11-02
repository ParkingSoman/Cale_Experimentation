"""TF-DIAG: how many upstream parents does a removed child need? Rules: CHILDREN-LOG.md TREE FOLD."""
import json, os, sys
import numpy as np
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.join(CO, "methods")); sys.path.insert(0, os.path.dirname(HERE))
from contract import orig_weights, SEQ, ROOT  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from rcf_v1 import _capture, silu  # noqa: E402
from tf1 import dsilu  # noqa: E402

L, K = int(sys.argv[1]), 48; BACK = 6
model, tok = load_model()
cal = token_blocks(tok, "train", 64)
txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()
ids = np.array(tok.encode(txt[400000:1600000]), dtype=np.int32)[: 64 * 513]
calm = mx.concatenate([cal, mx.array(ids.reshape(64, 513))], 0)
lays = model.model.layers
mods = {"x": (lays[L], "mlp"), "s": (lays[L], "post_attention_layernorm")}
src = list(range(L - BACK, L))
for l in src: mods[f"x{l}"] = (lays[l], "mlp")
cap = _capture(model, calm, mods)
G, U, D = orig_weights(model, L); X = cap["x"].astype(np.float64)
Zg, Zu = X @ G.T, X @ U.T; H = silu(Zg) * Zu
en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); rem = np.sort(np.argsort(-en)[K:])
rms = np.sqrt((cap["s"].astype(np.float64) ** 2).mean(1) + 1e-5); gam = np.array(lays[L].post_attention_layernorm.weight, np.float64)
Hs, Ds = [], []
for l in src:
    Gl, Ul, Dl = orig_weights(model, l); Xl = cap[f"x{l}"].astype(np.float64)
    Hs.append(silu(Xl @ Gl.T) * (Xl @ Ul.T)); Ds.append(Dl)
P = np.concatenate(Hs, 1); DP = np.concatenate(Ds, 1); nP = Hs[0].shape[1]
lastL1 = np.arange(P.shape[1]) >= (BACK - 1) * nP          # columns of layer L-1
MG, MU = (G * gam) @ DP, (U * gam) @ DP
ntop = max(1, int(.005 * len(H))); Pr = P / rms[:, None]
blk = np.repeat(np.arange(len(calm)), SEQ - 1); tr, te = blk % 2 == 0, blk % 2 == 1
Gtr = P[tr].T @ P[tr]; Ctr = H[tr][:, rem].T @ P[tr]                  # Gram and cross terms
R_te = H[te][:, rem] @ D[:, rem].T; den = (R_te ** 2).sum()
attr = np.zeros((len(rem), P.shape[1]))
for i, c in enumerate(rem):
    t = np.argpartition(-np.abs(H[:, c]), ntop)[:ntop]
    attr[i] = np.abs(MG[c] * ((dsilu(Zg[t, c]) * Zu[t, c])[:, None] * Pr[t]).mean(0) + MU[c] * (silu(Zg[t, c])[:, None] * Pr[t]).mean(0))


def r2_top(n, mask):
    A = np.zeros((len(rem), P.shape[1]))
    for i in range(len(rem)):
        a = np.where(mask, attr[i], -1); ps = np.argsort(-a)[:n]
        g = Gtr[np.ix_(ps, ps)]; g += np.eye(n) * 1e-6 * np.trace(g) / n
        A[i, ps] = np.linalg.solve(g, Ctr[i, ps])
    Rh = (P[te] @ A.T) @ D[:, rem].T
    return float(1 - ((R_te - Rh) ** 2).sum() / den)


def r2_dense(mask):
    cols = np.where(mask)[0]; g = Gtr[np.ix_(cols, cols)].copy(); best = -9
    for lam in (1e-3, 1e-2, 1e-1):
        W = np.linalg.solve(g + np.eye(len(cols)) * lam * np.trace(g) / len(cols), (H[tr][:, rem] @ D[:, rem].T).T @ P[tr][:, cols] .T.T if False else P[tr][:, cols].T @ (H[tr][:, rem] @ D[:, rem].T))
        Rh = P[te][:, cols] @ W; best = max(best, float(1 - ((R_te - Rh) ** 2).sum() / den))
    return best


out = {"L": L, "K": K}
for name, mask in (("L-1", lastL1), ("L-6..L-1", np.ones(P.shape[1], bool))):
    out[name] = {f"top{n}": r2_top(n, mask) for n in (1, 3, 10, 30, 100)}
    out[name]["dense"] = r2_dense(mask)
    print(L, name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
top1_layers = np.bincount(np.argmax(attr, 1) // nP, minlength=BACK)
out["top1_parent_layer_counts"] = {str(src[i]): int(v) for i, v in enumerate(top1_layers)}
print("top-1 parent layer", out["top1_parent_layer_counts"])
p = os.path.join(ROOT, "results", "clean-tf-diag.json"); res = json.load(open(p)) if os.path.exists(p) else {}
res[f"L{L}"] = out; json.dump(res, open(p, "w"), indent=1)
