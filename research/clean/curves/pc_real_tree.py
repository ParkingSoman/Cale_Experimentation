# serves: DQ19
"""Part C. How near are real class-row tables to a tree weight claim (Definition DQ19)? ResNet-18 and ResNet-50, cache/map, validation crops.
Estimator, fixed before the run. Table as in pc_real_weights.py. Tree: Ward on N's class weight vectors (a measure of form, not a map), and as a control the same tree with its leaf labels shuffled. Fit: alpha_c + beta_i + f_lca(c, i) by alternating weighted means, weights sbar, off the diagonal. Reported, for the tree and the control, and for f = 0: the weighted rms log residual sqrt(sum s r^2 / sum s) off the diagonal; the share of mass within a factor 3; the share of the k = 14 table loss (entrywise form, saved fit) on entries within a factor 3; the same on strangers only, at J = 20. Also the error eps_tree without eps_const.
Prediction, written before the run: rms about 1.0 on the real tree against about 1.8 for f = 0 and for the control; share of mass within a factor 3 at least 0.7 on the real tree.
Usage: pc_real_tree.py"""
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)


def lca_matrix(Z, n):
    """lca[c, i] = index of the internal node of the linkage where c and i meet; diagonal -1."""
    lca = -np.ones((n, n), dtype=np.int64)
    leaves = {i: [i] for i in range(n)}
    for t, row in enumerate(Z):
        a_, b_ = int(row[0]), int(row[1])
        la, lb = leaves.pop(a_), leaves.pop(b_)
        lca[np.ix_(la, lb)] = t
        lca[np.ix_(lb, la)] = t
        leaves[n + t] = la + lb
    return torch.from_numpy(lca)


def fit_effects(L, S, lca=None, iters=60):
    K, C = L.shape
    off = ~torch.eye(K, C, dtype=torch.bool)
    w = S * off
    alpha, beta = torch.zeros(K), torch.zeros(C)
    fv = torch.zeros(K, C)
    if lca is not None:
        idx = lca.clamp_min(0).reshape(-1)
        n_nodes = int(lca.max()) + 1
    for _ in range(iters):
        alpha = (w * (L - beta[None, :] - fv)).sum(1) / w.sum(1)
        beta = (w * (L - alpha[:, None] - fv)).sum(0) / w.sum(0)
        if lca is not None:
            r = L - alpha[:, None] - beta[None, :]
            num = torch.zeros(n_nodes).index_add_(0, idx, (w * r).reshape(-1))
            den = torch.zeros(n_nodes).index_add_(0, idx, w.reshape(-1))
            f = num / den.clamp_min(1e-300)
            fv = f[idx].reshape(K, C)
    return L - alpha[:, None] - beta[None, :] - fv, off


out = []
rng = np.random.default_rng(19)
for net in ("resnet18", "resnet50"):
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    meta = np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))
    Yt = torch.from_numpy(np.asarray(Y).astype(np.float64))
    lab = torch.from_numpy(meta["labels"].astype(np.int64))
    W = meta["W"].astype(np.float64)
    S = torch.softmax(Yt, 1)
    ids = torch.unique(lab)
    sb = torch.stack([S[lab == c].mean(0) for c in ids]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    K, C = sb.shape
    L = sb.log()
    Z = linkage(W, "ward")
    lca = lca_matrix(Z, C)
    perm = torch.from_numpy(rng.permutation(C))
    lca_c = lca[perm][:, perm]
    fz = np.load(os.path.join(HERE, f"cache/map/fit_{net}_table_14.npz"))
    q = torch.softmax(torch.from_numpy(fz["t"].astype(np.float64)) + torch.from_numpy(fz["c"].astype(np.float64)) @ torch.from_numpy(fz["V"].astype(np.float64)).T, 1)
    e = (q / sb).log()
    ent = sb * (e.exp() - 1 - e) / K
    fam = torch.from_numpy(fcluster(Z, 20, "maxclust").astype(np.int64))
    str_ = fam[None, :] != fam[:, None]
    for name, l_ in (("f=0", None), ("tree", lca), ("shuffled tree", lca_c)):
        r, off = fit_effects(L, sb, l_)
        within = r.abs() <= np.log(3)
        m_off = sb[off].sum()
        row = dict(net=net, shape=name, rms=float(((sb * r ** 2)[off].sum() / m_off).sqrt()), mass_within3=float(sb[off & within].sum() / m_off), loss_within3=float(ent[off & within].sum() / ent[off].sum()),
                   eps_tree=float((sb * r ** 2)[off].sum() / K), rms_strangers=float(((sb * r ** 2)[str_].sum() / sb[str_].sum()).sqrt()), mass_within3_strangers=float(sb[str_ & within].sum() / sb[str_].sum()), loss_within3_strangers=float(ent[str_ & within].sum() / ent[str_].sum()))
        out.append(row)
        print(row, flush=True)
json.dump(out, open(os.path.join(HERE, "results_pc_real_tree.json"), "w"), indent=1)
