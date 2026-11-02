# serves: DQ19
import numpy as np, json, os
import torch
from scipy.cluster.hierarchy import linkage
torch.set_num_threads(4)
HERE = "/Users/shonusengupta/Cale_Experimentation/research/clean/curves"


def load(net="resnet18"):
    Y = np.load(f"{HERE}/cache/map/{net}_y.npy", mmap_mode="r")
    meta = np.load(f"{HERE}/cache/map/{net}_meta.npz")
    lab = torch.from_numpy(meta["labels"].astype(np.int64))
    S = torch.softmax(torch.from_numpy(np.asarray(Y).astype(np.float64)), 1)
    ids = torch.unique(lab)
    sb = torch.stack([S[lab == c].mean(0) for c in ids]).clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    return sb.double(), meta, lab


def splitA(K=1000):
    perm = np.random.default_rng(2026).permutation(K)
    return np.sort(perm[:K // 2])  # A rows only; B is never taken
