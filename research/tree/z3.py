"""Z3: do Ward merges nest, and what does stacking cost? No training."""
import json
import time
import numpy as np
import mlx.core as mx
from tree_lib import *
from tree_lib2 import parent_map, merge_parents

SIZES = [768, 384, 192]
t0 = time.time()
model, tok = load_model()
teacher, _ = load_model()
cal = token_blocks(tok, "train", 32)
held = token_blocks(tok, "test", 100)
W = Weights(model)
X = [a.astype(np.float16) for a in capture_mlp_inputs(model, cal)]
Xh = [a.astype(np.float16) for a in capture_mlp_inputs(model, held[:20])]
base = eval_loss(model, held, teacher=teacher)
print("baseline", base, flush=True)
L = len(W.G)

# ---- per-layer tree (calibration text only) ----
sign, Zw, labs, pms = [], [], [], []
viol = {"768>384": 0, "384>192": 0}
for i in range(L):
    G, U, D = W.G[i], W.U[i], W.D[i]
    Xi = X[i].astype(np.float32)
    s = canonical_sign(Xi, G, U)
    Z = ward_tree(neuron_features("weights", Xi, G, U, D, s))
    lab = {k: cut_tree(Z, k) for k in SIZES}
    for k in SIZES:
        assert len(np.unique(lab[k])) == k and np.bincount(lab[k]).min() >= 1
        assert np.bincount(lab[k], minlength=k).sum() == G.shape[0]
    p1, b1 = parent_map(lab[768], lab[384])
    p2, b2 = parent_map(lab[384], lab[192])
    viol["768>384"] += b1
    viol["384>192"] += b2
    sign.append(s); Zw.append(Z); labs.append(lab); pms.append({384: p1, 192: p2})
total_viol = sum(viol.values())
print(f"NESTING VIOLATIONS over {L} layers: {total_viol}  {viol}  [{time.time()-t0:.0f}s]", flush=True)


def build(i, variant, k):
    """Returns G,U,D for layer i. Targets are always the ORIGINAL layer output on calibration text."""
    G, U, D, s = W.G[i], W.U[i], W.D[i], sign[i]
    Xi = X[i].astype(np.float32)
    Y = hidden(Xi, G, U) @ D.T
    lab = labs[i]
    if variant.startswith("straight"):
        g, u, d = merge_groups(G, U, D, s, lab[k])
        if variant == "straight_ref":
            d = ls_down(Xi, g, u, Y)
        return g, u, d
    refit_first = variant in ("chain_ref", "chain_ref_once")
    refit_each = variant == "chain_ref"
    g, u, d = merge_groups(G, U, D, s, lab[768])
    cnt = np.bincount(lab[768], minlength=768)
    if refit_first:
        d = ls_down(Xi, g, u, Y)
    for lvl in SIZES[1:]:
        if lvl < k:
            break
        g, u, d, cnt = merge_parents(g, u, d, cnt, pms[i][lvl])
        if refit_each:
            d = ls_down(Xi, g, u, Y)
    return g, u, d


# ---- stacked vs straight weights (no refit) ----
maxdiff = {str(k): 0.0 for k in SIZES}
for i in range(L):
    for k in SIZES:
        a = build(i, "straight_noref", k)
        b = build(i, "chain_noref", k)
        maxdiff[str(k)] = max(maxdiff[str(k)], max(float(np.abs(x - y).max()) for x, y in zip(a, b)))
print("MAX ABS WEIGHT DIFF straight vs stacked (no refit), over all layers:", maxdiff, flush=True)

# ---- storage, from real shapes, layer 0 ----
G, U, D = W.G[0], W.U[0], W.D[0]
n_orig = G.size + U.size + D.size
n_tree = n_orig + Zw[0].size + sign[0].size       # leaves + merge list (linkage) + signs
extra = 0
for k in SIZES:
    extra += build(0, "straight_ref", k)[2].size  # one extra down matrix per level
n_trained = n_tree + extra
storage = {"original": int(n_orig), "tree_average_parent": int(n_tree), "tree_trained_parent": int(n_trained),
           "note": "average parent = original neurons + linkage table + signs; trained = that + one refit down matrix (576 x k) per level 768/384/192"}
print("STORAGE (layer 0, numbers stored):", storage, flush=True)

# ---- evaluation ----
VARIANTS = ["straight_noref", "straight_ref", "chain_noref", "chain_ref", "chain_ref_once"]
results = []
for k in SIZES:
    for v in VARIANTS:
        errs = []
        for i in range(L):
            g, u, d = build(i, v, k)
            assert g.shape[0] == k and u.shape[0] == k and d.shape[1] == k
            W.set_layer(i, g, u, d)
            Xhi = Xh[i].astype(np.float32)
            Yh = hidden(Xhi, W.G[i], W.U[i]) @ W.D[i].T
            errs.append(rel_err(Yh, hidden(Xhi, g, u) @ d.T))
        W.commit()
        r = eval_loss(model, held, teacher=teacher)
        r.update(size=k, variant=v, layer_err_mean=float(np.mean(errs)), layer_err=[round(float(e), 4) for e in errs])
        results.append(r)
        print(f"k={k:4d} {v:15s} ce={r['ce']:.3f} (+{r['ce']-base['ce']:.3f}) kl={r['kl']:.3f} "
              f"layer_err={r['layer_err_mean']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        W.restore()

print("\n  k   variant          CE     dCE    KL    layer_err")
for r in results:
    print(f"{r['size']:4d}  {r['variant']:15s} {r['ce']:.3f} +{r['ce']-base['ce']:.3f} {r['kl']:.3f}  {r['layer_err_mean']:.3f}")
out = {"baseline": base, "sizes": SIZES, "nesting_violations": total_viol, "nesting_detail": viol,
       "max_abs_weight_diff_straight_vs_stacked": maxdiff, "storage_layer0": storage, "results": results}
json.dump(out, open("../../results/tree-z3.json", "w"), indent=1)
print("saved results/tree-z3.json")
