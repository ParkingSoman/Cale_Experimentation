"""Z1: do natural groups exist, and are parents broader than children? No training.

Groups come from calibration text only (train). Readings use held-out blocks held[:40].
"""
import json
import time
import numpy as np
import mlx.core as mx
from tree_lib import *

KS = [768, 192]
TOP = 300
SAMPLE_LAYERS = [3, 15, 28]
t0 = time.time()
model, tok = load_model()
cal = token_blocks(tok, "train", 32)
held = token_blocks(tok, "test", 100)
W = Weights(model)
Xc = [x.astype(np.float16) for x in capture_mlp_inputs(model, cal)]
hb = held[:40]
Xh = [x.astype(np.float16) for x in capture_mlp_inputs(model, hb)]
tok_ids = np.array(hb[:, :-1]).reshape(-1)          # token id at each held-out position
del model, cal, held
print(f"captured {time.time()-t0:.0f}s", flush=True)


def topk_tokens(A):
    """A: tokens x neurons (sign-aligned). Distinct token ids among each column's top TOP positions."""
    idx = np.argpartition(-A, TOP, axis=0)[:TOP]
    return np.array([len(np.unique(tok_ids[idx[:, j]])) for j in range(A.shape[1])])


def mean_pair_corr(C, lab):
    tot, n = 0.0, 0
    for g in range(lab.max() + 1):
        m = np.where(lab == g)[0]
        if len(m) < 2:
            continue
        sub = C[np.ix_(m, m)]
        tot += sub.sum() - np.trace(sub)
        n += len(m) * (len(m) - 1)
    return tot / n if n else float("nan")


def random_like(lab, seed):
    """Random partition with exactly the group sizes of lab."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(lab))
    out = np.empty_like(lab)
    sizes = np.bincount(lab)
    out[perm] = np.repeat(np.arange(len(sizes)), sizes)
    return out


def ctx(i, n_tok=3):
    b, p = divmod(i, SEQ)
    left = [tok.decode([int(t)]) for t in tok_ids[b * SEQ + max(0, p - n_tok): b * SEQ + p]]
    return (("".join(left) + "[[" + tok.decode([int(tok_ids[i])]) + "]]")).replace("\n", "\\n")


def neuron_md(H, j):
    col = H[:, j]
    hi = np.argsort(-col)[:12]
    lo = np.argsort(col)[:12]
    s = f"  - neuron {j}\n    - highest: " + " | ".join(f"{ctx(i)} ({col[i]:.2f})" for i in hi) + "\n"
    s += "    - lowest: " + " | ".join(f"{ctx(i)} ({col[i]:.2f})" for i in lo) + "\n"
    return s


def group_md(title, H, members, rho):
    s = f"### {title} (size {len(members)}, within-group corr {rho:.3f})\n"
    for j in members:
        s += neuron_md(H, int(j))
    return s + "\n"


def group_rho(C, m):
    sub = C[np.ix_(m, m)]
    return (sub.sum() - np.trace(sub)) / (len(m) * (len(m) - 1))


L = len(W.G)
res = []
md = ["# Z1 readable groups\n",
      "Groups built on calibration text (weight-feature Ward, k=192). Activations are RAW `hidden` on held-out text "
      "(tokens in `[[ ]]` with 3 tokens of left context; number = activation). Both poles shown.\n"]
for i in range(L):
    G, U, D = W.G[i], W.U[i], W.D[i]
    X = Xc[i].astype(np.float32)
    sign = canonical_sign(X, G, U)
    fw = neuron_features("weights", X, G, U, D, sign)
    ff = neuron_features("function", X, G, U, D, sign)
    Zw, Zf = ward_tree(fw), ward_tree(ff)
    del X
    Xi = Xh[i].astype(np.float32)
    H = hidden(Xi, G, U)
    Hs = H * sign[None, :]
    Z = Hs - Hs.mean(0)
    Z /= (np.linalg.norm(Z, axis=0, keepdims=True) + 1e-9)
    C = Z.T @ Z
    del Z
    child_b = topk_tokens(Hs)
    row = {"layer": i, "child_breadth_all_neurons": float(child_b.mean())}
    parts_keep = {}
    for k in KS:
        labw, labf = cut_tree(Zw, k), cut_tree(Zf, k)
        parts = {"weights": labw, "function": labf,
                 "random": random_like(labw, 100 * i + k), "random_fn": random_like(labf, 100 * i + k + 1)}
        assert np.array_equal(np.sort(np.bincount(parts["random"])), np.sort(np.bincount(labw)))
        assert np.array_equal(np.sort(np.bincount(parts["random_fn"])), np.sort(np.bincount(labf)))
        for name, lab in parts.items():
            assert lab.max() + 1 == k and len(np.unique(lab)) == k, (name, k)
            G2, U2, D2 = merge_groups(G, U, D, sign, lab)
            par_b = topk_tokens(hidden(Xi, G2, U2))
            sizes = np.bincount(lab)
            big = np.where(sizes >= 2)[0]
            ch = np.array([child_b[lab == g].mean() for g in big])
            pb = par_b[big]
            row[f"{name}_k{k}"] = dict(
                corr=float(mean_pair_corr(C, lab)), n_groups_ge2=int(len(big)),
                mean_group_size_ge2=float(sizes[big].mean()),
                child_breadth=float(ch.mean()), parent_breadth=float(pb.mean()),
                ratio_of_means=float(pb.mean() / ch.mean()), mean_group_ratio=float((pb / ch).mean()))
        parts_keep[k] = parts
    res.append(row)
    if i in SAMPLE_LAYERS:
        lab = parts_keep[192]["weights"]
        sizes = np.bincount(lab)
        cand = [g for g in range(192) if 3 <= sizes[g] <= 8]
        rho = {g: group_rho(C, np.where(lab == g)[0]) for g in cand}
        top = sorted(cand, key=lambda g: -rho[g])[:4]
        rest = [g for g in cand if g not in top]
        rr = np.random.default_rng(i).choice(rest, 4, replace=False)
        md.append(f"## Layer {i}\n\n### Weight-Ward groups\n")
        for tag, gs in (("top-corr", top), ("random", list(rr))):
            for g in gs:
                md.append(group_md(f"{tag} group {g}", H, np.where(lab == g)[0], rho[g]))
        rl = parts_keep[192]["random"]
        rs = np.bincount(rl)
        rc = [g for g in range(192) if 3 <= rs[g] <= 8]
        md.append("### Random-control groups\n")
        for g in np.random.default_rng(i + 1).choice(rc, 2, replace=False):
            m = np.where(rl == g)[0]
            md.append(group_md(f"random-control group {g}", H, m, group_rho(C, m)))
    del H, Hs, C, Xi
    print(f"layer {i} done [{time.time()-t0:.0f}s]", flush=True)

json.dump({"ks": KS, "top": TOP, "held_tokens": int(len(tok_ids)), "layers": res},
          open("../../results/tree-z1.json", "w"), indent=1)
open("../../results/tree-z1-groups.md", "w").write("\n".join(md))

print("\nlayer | corr k768 w/f/r | corr k192 w/f/r | breadth ratio parent/child k768 w/f/r | k192 w/f/r")
for r in res:
    c = lambda k: "/".join(f"{r[f'{n}_k{k}']['corr']:.3f}" for n in ("weights", "function", "random"))
    b = lambda k: "/".join(f"{r[f'{n}_k{k}']['ratio_of_means']:.2f}" for n in ("weights", "function", "random"))
    print(f"{r['layer']:2d} | {c(768)} | {c(192)} | {b(768)} | {b(192)}")
print("saved results/tree-z1.json, results/tree-z1-groups.md")
