# serves: DQ19
"""Round 4, screen S4 (card 3-37): the error of a tree weight claim, on held-out rows. Code computes the error. Agents propose the tree.

Definition DQ19's fit: log sbar_c[i] = alpha_c + beta_i + f_(lca(c, i)), off the diagonal, weights sbar. Rows are split once, with a fixed
seed, into A and B. The agents may score on A only. beta and f_v are fitted on A's rows. On B's rows only alpha_c is refitted (a row scale). The
held-out number is the rms on B, off the diagonal.

Input: a JSON tree of nested lists whose leaves are class indices 0..999, each exactly once, with any fan-out. --tree ward gives the
Ward tree on N's class weight vectors, and --tree shuffled the same tree with shuffled leaves.
Screen line, fixed before the agents ran: held-out rms at most 1.0 on both nets. Ward held-out is the number to beat.
PREDICTION, written before the agents ran (2026-11-02): no agent's tree goes below 1.2 held out.

Usage: pc_r4_s4_eval.py --net resnet18 --tree path.json --split A   (agents)
       pc_r4_s4_eval.py --net resnet18 --tree path.json --split B   (the main session only, after the agents have finished)
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="resnet18")
ap.add_argument("--tree", required=True)
ap.add_argument("--split", default="A", choices=["A", "B"])
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
Y = np.load(os.path.join(HERE, f"cache/map/{a.net}_y.npy"), mmap_mode="r")
meta = np.load(os.path.join(HERE, f"cache/map/{a.net}_meta.npz"))
lab = torch.from_numpy(meta["labels"].astype(np.int64))
S = torch.softmax(torch.from_numpy(np.asarray(Y).astype(np.float64)), 1)
ids = torch.unique(lab)
sb = torch.stack([S[lab == c].mean(0) for c in ids]).clamp_min(1e-12)
sb = sb / sb.sum(1, keepdim=True)
K, C = sb.shape
L = sb.log()


def lca_from_nested(tree):
    lca = -np.ones((C, C), dtype=np.int64)
    count = [0]
    seen = []

    def rec(node):
        if isinstance(node, int):
            seen.append(node)
            return [node]
        parts = [rec(x) for x in node]
        t = count[0]
        count[0] += 1
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                lca[np.ix_(parts[i], parts[j])] = t
                lca[np.ix_(parts[j], parts[i])] = t
        return [x for p in parts for x in p]
    rec(tree)
    assert sorted(seen) == list(range(C)), "the tree must hold each class exactly once"
    np.fill_diagonal(lca, 0)
    return torch.from_numpy(lca), count[0]


def nested_from_linkage(Z, leaves):
    nodes = [int(x) for x in leaves]
    for x, y, _, _ in Z:
        nodes.append([nodes[int(x)], nodes[int(y)]])
    return nodes[-1]


if a.tree in ("ward", "shuffled"):
    Z = linkage(meta["W"].astype(np.float64), "ward")
    leaves = np.arange(C) if a.tree == "ward" else np.random.default_rng(19).permutation(C)
    tree = nested_from_linkage(Z, leaves)
else:
    tree = json.load(open(a.tree))
lca, nn = lca_from_nested(tree)
off = ~torch.eye(K, C, dtype=torch.bool)
w = sb * off
perm = np.random.default_rng(2026).permutation(K)
A = torch.from_numpy(perm[:K // 2])
B = torch.from_numpy(perm[K // 2:])
idx = lca.reshape(-1)
beta = torch.zeros(C)
fv = torch.zeros(K, C)
wA, LA = w[A], L[A]
idxA = lca[A].reshape(-1)
for _ in range(60):
    alpha = (wA * (LA - beta[None, :] - fv[A])).sum(1) / wA.sum(1)
    beta = (wA * (LA - alpha[:, None] - fv[A])).sum(0) / wA.sum(0).clamp_min(1e-300)
    r = LA - alpha[:, None] - beta[None, :]
    num = torch.zeros(nn).index_add_(0, idxA, (wA * r).reshape(-1))
    den = torch.zeros(nn).index_add_(0, idxA, wA.reshape(-1))
    f = num / den.clamp_min(1e-300)
    fv = f[idx].reshape(K, C)
rows = A if a.split == "A" else B
wR, LR = w[rows], L[rows]
alpha = (wR * (LR - beta[None, :] - fv[rows])).sum(1) / wR.sum(1)
r = LR - alpha[:, None] - beta[None, :] - fv[rows]
rms = float(((wR * r ** 2).sum() / wR.sum()).sqrt())
within = float((wR * (r.abs() <= np.log(3))).sum() / wR.sum())
print(json.dumps(dict(net=a.net, tree=a.tree, split=a.split, internal_nodes=nn, rms=rms, mass_within3=within)))
