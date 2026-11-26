"""N6 shared parts: the whole-network gap between the no-training build and training of the same form.

ConvNeXt-Base, stages 3-4 (30 blocks: global index g 0..26 = stage 3, 27..29 = stage 4). A block's MLP:
  u = LayerNorm(dwconv(stream)),  g = GELU(W1 u + b1),  y = layer_scale * (W2 g + b2),  stream += y.
A size keeps a set S of hidden neurons in every block. The build re-solves the kept writes once:
  Wb = layer_scale * W2,  Wnew_S^T = K_SS^-1 K_S,all Wb^T,  bias = bb + Wb m - Wnew_S m_S      (pair/allblocks.py)
Training starts from the build and changes the kept numbers only (or, mode "all", every stage 3-4 weight).

Photos (reading photos, 35 per class; even positions in a class = fit, odd = test):
  test  = pair/distill's photos_test.npy (1,500, 938 classes; sorted by class: never use a prefix)
  held  = 2 per class from the fit half (positions 8 and 9 in the class's fit list): stop point and learning rate
  cal   = position 7 in the class's fit list (1,000 photos, every class): the build's covariances
  train = every other fit photo
Cache (cache3.py): the stream at the input of stage 3 (fp16, 512 x 14 x 14) and Base's logits, for every photo above,
plus the left-right flip of each train photo.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pair"))
from cache import DEV, load, net, split  # noqa: E402,F401

OUT = os.path.join(HERE, "..", "results", "gap")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "cache")
TEST_FILE = os.path.join(HERE, "..", "results", "pair", "distill", "photos_test.npy")


def photo_sets(lab):
    fit = split(lab)
    tr, held, cal = [], [], []
    for c in range(1000):
        w = np.where((lab == c) & fit)[0]
        cal.append(w[7])
        held += w[8:10].tolist()
        tr += [i for k, i in enumerate(w) if k not in (8, 9)]
    te = np.load(TEST_FILE)
    assert not set(te) & set(tr) and not set(te) & set(held)
    return dict(train=np.array(sorted(tr)), held=np.array(sorted(held)), cal=np.array(sorted(cal)), test=te)


def blocks(m):
    return [blk for s in (5, 7) for blk in m.features[s]]


def tail(m, x3):
    """Stages 3-4 and the head, from the stream at the input of stage 3."""
    x = m.features[7](m.features[6](m.features[5](x3)))
    return m.classifier(m.avgpool(x))


def calibrate(m, x3_batches):
    """Mean and covariance (float64, on CPU) of each block's GELU outputs, all positions."""
    BL = blocks(m)
    acc = {}
    hooks = []
    for g, blk in enumerate(BL):
        h = blk.block[3].out_features
        acc[g] = dict(n=0, s=torch.zeros(h, dtype=torch.float64), ss=torch.zeros(h, h, dtype=torch.float64))

        def f(mod, inp, out, g=g):
            z = out.reshape(-1, out.shape[-1]).float()
            a = acc[g]
            a["n"] += z.shape[0]
            a["s"] += z.sum(0).cpu().double()
            a["ss"] += (z.T @ z).cpu().double()
        hooks.append(blk.block[4].register_forward_hook(f))
    with torch.no_grad():
        for x3 in x3_batches:
            tail(m, x3)
    for h_ in hooks:
        h_.remove()
    out = []
    for g in range(len(BL)):
        a = acc[g]
        mg = a["s"] / a["n"]
        out.append((mg, a["ss"] / a["n"] - mg[:, None] * mg[None, :]))
    return out


def write_mat(blk):
    ls = blk.layer_scale.detach().view(-1).cpu().double()
    return ls, ls[:, None] * blk.block[5].weight.detach().cpu().double(), ls * blk.block[5].bias.detach().cpu().double()


def choose(blk, mg, K, frac, how, seed=0, batch=64):
    """Survivors of one block: 'value' (std x |write|), 'random', or 'greedy' (batch OMP on covariances, pair/gap3.py)."""
    _, Wb, _ = write_mat(blk)
    h = Wb.shape[1]
    k = int(round(frac * h))
    if how == "value":
        v = K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)
        return torch.sort(torch.argsort(-v)[:k]).values
    if how == "random":
        rng = np.random.default_rng(seed)
        return torch.sort(torch.from_numpy(rng.choice(h, k, replace=False))).values
    assert how == "greedy"
    Q = K @ Wb.T                                   # covariance of each neuron's output with the block output y
    ridge = 1e-6 * K.diagonal().mean()
    S = []
    while len(S) < k:
        if S:
            KSS = K[np.ix_(S, S)] + ridge * torch.eye(len(S), dtype=torch.float64)
            KcS = K[:, S]
            Z = torch.linalg.solve(KSS, KcS.T)
            corr = Q - KcS @ torch.linalg.solve(KSS, Q[S])
            denom = K.diagonal() - (KcS * Z.T).sum(1)
        else:
            corr, denom = Q, K.diagonal().clone()
        gain = corr.pow(2).sum(1) / denom.clamp_min(1e-9)
        gain[S] = -1
        S += torch.argsort(-gain)[:min(batch, k - len(S))].tolist()
    return torch.sort(torch.tensor(S)).values


def shrink_block(blk, S, mg, K):
    """Replace the block's two MLP layers by narrow ones holding the kept neurons, writes re-solved (in place)."""
    ls, Wb, bb = write_mat(blk)
    k = len(S)
    KSS = K[S][:, S] + 1e-6 * K.diagonal().mean() * torch.eye(k, dtype=torch.float64)
    Wnew = torch.linalg.solve(KSS, K[S] @ Wb.T).T                      # (d, k)
    bnew = bb + Wb @ mg - Wnew @ mg[S]
    l1, l2 = blk.block[3], blk.block[5]
    n1 = nn.Linear(l1.in_features, k).to(DEV)
    n2 = nn.Linear(k, l2.out_features).to(DEV)
    with torch.no_grad():
        n1.weight.copy_(l1.weight[S.to(l1.weight.device)])
        n1.bias.copy_(l1.bias[S.to(l1.bias.device)])
        n2.weight.copy_((Wnew / ls[:, None]).float())
        n2.bias.copy_((bnew / ls).float())
    blk.block[3], blk.block[5] = n1, n2


def build(m, cal, frac, how, seed=0):
    """Shrink every block of m in place. Returns the survivor lists."""
    Ss = []
    for g, blk in enumerate(blocks(m)):
        mg, K = cal[g]
        S = choose(blk, mg, K, frac, how, seed)
        shrink_block(blk, S, mg, K)
        Ss.append(S)
    return Ss


def kl_rows(ref, out):
    p = F.log_softmax(ref.float(), 1)
    return (p.exp() * (p - F.log_softmax(out.float(), 1))).sum(1)


class Cache:
    """Memory-mapped cache of stage-3 inputs (fp16) and Base logits."""

    def __init__(self, name):
        self.x = np.load(os.path.join(CACHE, f"{name}_x3.npy"), mmap_mode="r")
        self.z = np.load(os.path.join(CACHE, f"{name}_logits.npy"))
        self.ix = np.load(os.path.join(CACHE, f"{name}_ix.npy"))

    def __len__(self):
        return len(self.ix)

    def get(self, rows):
        rows = np.sort(np.asarray(rows))
        x = torch.from_numpy(np.ascontiguousarray(self.x[rows])).to(DEV).float()
        return x, torch.from_numpy(self.z[rows]).to(DEV).float()

    def batches(self, bs=50):
        for a in range(0, len(self), bs):
            yield self.get(np.arange(a, min(a + bs, len(self))))


def evaluate(m, cache, labels=None, bs=50, rows_all=None):
    """Per-photo KL to Base and top-1 on a cached set (or on the rows rows_all of it)."""
    rows_all = np.arange(len(cache)) if rows_all is None else np.sort(rows_all)
    kl, hit = [], []
    with torch.no_grad():
        for a in range(0, len(rows_all), bs):
            rows = rows_all[a:a + bs]
            x, z = cache.get(rows)
            y = tail(m, x)
            kl.append(kl_rows(z, y).cpu())
            if labels is not None:
                hit.append((y.argmax(1).cpu() == torch.as_tensor(labels[cache.ix[rows]])).float())
    kl = torch.cat(kl)
    return dict(kl=float(kl.mean()), top1=float(torch.cat(hit).mean()) if hit else None), kl.numpy()
