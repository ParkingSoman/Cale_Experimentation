"""Z2: cut vs merge on the feed-forward part of SmolLM2-135M. No training.

For each size and each way of shrinking, with and without a least-squares refit of the
output weights, report held-out cross-entropy and KL to the full model.
Controls: merging random groups; cutting by weight size (no data) and by activity (data).
"""
import json
import sys
import time
import numpy as np
import mlx.core as mx
from tree_lib import *

SIZES = [768, 384, 192]          # 50%, 25%, 12.5% of 1536
t0 = time.time()
model, tok = load_model()
teacher, _ = load_model()
cal = token_blocks(tok, "train", 32)
held = token_blocks(tok, "test", 100)
W = Weights(model)
X = capture_mlp_inputs(model, cal)
Xh = capture_mlp_inputs(model, held[:20])
base = eval_loss(model, held, teacher=teacher)
print("baseline", base, flush=True)

L = len(W.G)
pre = []
for i in range(L):
    G, U, D = W.G[i], W.U[i], W.D[i]
    sign = canonical_sign(X[i], G, U)
    Y = hidden(X[i], G, U) @ D.T
    Yh = hidden(Xh[i], G, U) @ D.T
    H = hidden(X[i], G, U)
    fw = neuron_features("weights", X[i], G, U, D, sign)
    ff = neuron_features("function", X[i], G, U, D, sign)
    pre.append(dict(sign=sign, Y=Y, Yh=Yh, fw=fw, ff=ff, Zw=ward_tree(fw), Zf=ward_tree(ff),
                    s_mag=np.linalg.norm(G, axis=1) * np.linalg.norm(U, axis=1) * np.linalg.norm(D, axis=0),
                    s_act=np.abs(H).mean(0) * np.linalg.norm(D, axis=0)))
print(f"prepared {time.time()-t0:.0f}s", flush=True)


def shrink(i, method, k, seed=0):
    p = pre[i]
    G, U, D, s = W.G[i], W.U[i], W.D[i], p["sign"]
    if method in ("cut_mag", "cut_act"):
        keep = cut_keep(p["s_mag" if method == "cut_mag" else "s_act"], k)
        return G[keep], U[keep] * s[keep, None], D[:, keep] * s[keep][None, :]
    if method == "merge_wt_ward":
        lab = cut_tree(p["Zw"], k)
    elif method == "merge_fn_ward":
        lab = cut_tree(p["Zf"], k)
    elif method == "merge_wt_kmeans":
        lab = kmeans(p["fw"], k, seed=seed)
    elif method == "merge_rand":
        lab = np.random.default_rng(seed + 100 * i).permutation(G.shape[0]) % k
    else:
        raise ValueError(method)
    return merge_groups(G, U, D, s, lab)


METHODS = ["cut_mag", "cut_act", "merge_wt_ward", "merge_fn_ward", "merge_wt_kmeans", "merge_rand"]
results = []
for k in SIZES:
    for method in METHODS:
        for ls in (False, True):
            for seed in ((0, 1) if method == "merge_rand" else (0,)):
                errs = []
                for i in range(L):
                    G2, U2, D2 = shrink(i, method, k, seed)
                    if ls:
                        D2 = ls_down(X[i], G2, U2, pre[i]["Y"])
                    W.set_layer(i, G2, U2, D2)
                    errs.append(rel_err(pre[i]["Yh"], hidden(Xh[i], G2, U2) @ D2.T))
                W.commit()
                r = eval_loss(model, held, teacher=teacher)
                r.update(size=k, method=method, ls=ls, seed=seed, layer_err_mean=float(np.mean(errs)),
                         layer_err=[round(float(e), 4) for e in errs])
                results.append(r)
                print(f"k={k:4d} {method:16s} ls={int(ls)} seed={seed} ce={r['ce']:.3f} "
                      f"(+{r['ce']-base['ce']:.3f}) kl={r['kl']:.3f} layer_err={r['layer_err_mean']:.3f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                W.restore()

out = {"baseline": base, "sizes": SIZES, "results": results}
json.dump(out, open("../../results/tree-z2.json", "w"), indent=1)
print("saved results/tree-z2.json")
