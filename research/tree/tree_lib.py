"""Shared code for the tree-model $0 track (Z1-Z4): cut vs merge on SmolLM2-135M.

Everything runs on the Mac in float32 with MLX. No training.
Calibration text (wikitext-2 train) and held-out text (wikitext-2 test) never overlap.
"""
import glob
import os
import numpy as np
import mlx.core as mx
from mlx_lm import load
from scipy.cluster.hierarchy import linkage, fcluster

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = glob.glob(f"{HERE}/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*")[0]
SEQ = 512


def load_model():
    model, tok = load(MODEL_DIR)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model, tok


def token_blocks(tok, name, n_seq, skip_chars=0):
    """n_seq blocks of SEQ+1 token ids from a text file, as an mx int32 array."""
    text = open(f"{HERE}/data/wikitext2_{name}.txt").read()[skip_chars:]
    need = n_seq * (SEQ + 1)
    ids = []
    pos = 0
    while len(ids) < need and pos < len(text):
        ids += tok.encode(text[pos:pos + 200_000])
        pos += 200_000
    ids = np.array(ids[:need], dtype=np.int32)
    return mx.array(ids.reshape(n_seq, SEQ + 1))


def eval_loss(model, blocks, teacher=None, batch=4):
    """Held-out cross-entropy (nats/token) and, if given, KL(teacher || model)."""
    ce_sum, kl_sum, n = 0.0, 0.0, 0
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        x, y = b[:, :-1], b[:, 1:]
        logp = model(x)
        logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(logp, y[..., None], axis=-1).squeeze(-1)
        cnt = int(np.prod(y.shape))
        ce_sum += float(ce.sum().item())
        if teacher is not None:
            tl = teacher(x)
            tl = tl - mx.logsumexp(tl, axis=-1, keepdims=True)
            kl = (mx.exp(tl) * (tl - logp)).sum(-1)
            kl_sum += float(kl.sum().item())
        n += cnt
    return {"ce": ce_sum / n, "kl": kl_sum / n if teacher is not None else None, "tokens": n}


class Capture:
    """Wraps an mlp module and keeps its input (the real input from the intact model)."""

    def __init__(self, inner):
        self.inner = inner
        self.x = []

    def __call__(self, x):
        self.x.append(np.array(x.astype(mx.float32)).reshape(-1, x.shape[-1]))
        return self.inner(x)


def capture_mlp_inputs(model, blocks, batch=4):
    layers = model.model.layers
    caps = [Capture(l.mlp) for l in layers]
    for l, c in zip(layers, caps):
        l.mlp = c
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    for l, c in zip(layers, caps):
        l.mlp = c.inner
    return [np.concatenate(c.x, 0) for c in caps]


def silu(a):
    return a / (1.0 + np.exp(-a))


def hidden(X, G, U):
    return silu(X @ G.T) * (X @ U.T)


class Weights:
    """The original MLP weights as numpy, plus setters that patch the live model."""

    def __init__(self, model):
        self.model = model
        self.G, self.U, self.D = [], [], []
        for l in model.model.layers:
            self.G.append(np.array(l.mlp.gate_proj.weight))
            self.U.append(np.array(l.mlp.up_proj.weight))
            self.D.append(np.array(l.mlp.down_proj.weight))

    def set_layer(self, i, G, U, D):
        m = self.model.model.layers[i].mlp
        m.gate_proj.weight = mx.array(G.astype(np.float32))
        m.up_proj.weight = mx.array(U.astype(np.float32))
        m.down_proj.weight = mx.array(D.astype(np.float32))

    def restore(self):
        for i in range(len(self.G)):
            self.set_layer(i, self.G[i], self.U[i], self.D[i])
        mx.eval(self.model.parameters())

    def commit(self):
        mx.eval(self.model.parameters())


def canonical_sign(X, G, U):
    """+1/-1 per neuron. Flipping (up row, down column) together leaves a neuron's
    function unchanged, so merging must first put neurons on the same side.
    Rule: make the neuron's mean activation over the calibration text non-negative."""
    s = np.sign(hidden(X, G, U).mean(0))
    s[s == 0] = 1.0
    return s


def neuron_features(kind, X, G, U, D, sign, dim=128):
    """One row per neuron, used for grouping."""
    if kind == "weights":
        def nrm(a):
            return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        return np.concatenate([nrm(G), nrm(U * sign[:, None]), nrm(D.T * sign[:, None])], 1)
    if kind == "function":
        H = hidden(X, G, U) * sign[None, :]            # tokens x neurons
        H = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-9)
        u, s, vt = np.linalg.svd(H.T, full_matrices=False)
        return u[:, :dim] * s[:dim]
    raise ValueError(kind)


def ward_tree(feat):
    return linkage(feat, method="ward")


def cut_tree(Z, k):
    lab = fcluster(Z, t=k, criterion="maxclust") - 1
    assert lab.max() + 1 == k, (lab.max() + 1, k)
    return lab


def kmeans(feat, k, seed=0, iters=30):
    rng = np.random.default_rng(seed)
    n = feat.shape[0]
    centers = [feat[rng.integers(n)]]
    d2 = ((feat - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        centers.append(feat[rng.choice(n, p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((feat - centers[-1]) ** 2).sum(1))
    C = np.array(centers)
    for _ in range(iters):
        dist = (feat ** 2).sum(1)[:, None] - 2 * feat @ C.T + (C ** 2).sum(1)[None, :]
        lab = dist.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = feat[m].mean(0)
            else:
                C[j] = feat[rng.integers(n)]
    dist = (feat ** 2).sum(1)[:, None] - 2 * feat @ C.T + (C ** 2).sum(1)[None, :]
    lab = dist.argmin(1)
    # relabel to 0..k'-1 (empty clusters are dropped, so sizes can be < k)
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def merge_groups(G, U, D, sign, lab):
    """Average parent: mean of the members' gate and up rows (after sign alignment),
    sum of their down columns. This is exact when the members are identical."""
    k = lab.max() + 1
    Us, Ds = U * sign[:, None], D * sign[None, :]
    G2 = np.zeros((k, G.shape[1]), np.float32)
    U2 = np.zeros_like(G2)
    D2 = np.zeros((D.shape[0], k), np.float32)
    cnt = np.bincount(lab, minlength=k).astype(np.float32)
    np.add.at(G2, lab, G)
    np.add.at(U2, lab, Us)
    for j in range(k):
        D2[:, j] = Ds[:, lab == j].sum(1)
    return G2 / cnt[:, None], U2 / cnt[:, None], D2


def cut_keep(scores, k):
    return np.sort(np.argsort(-scores)[:k])


def ls_down(X, G, U, Ytarget, ridge=1e-4):
    """Least-squares output weights: choose D so that hidden(X,G,U) @ D.T matches Ytarget."""
    H = hidden(X, G, U).astype(np.float64)
    A = H.T @ H
    A += np.eye(A.shape[0]) * ridge * np.trace(A) / A.shape[0]
    B = H.T @ Ytarget.astype(np.float64)
    return np.linalg.solve(A, B).T.astype(np.float32)


def rel_err(Y, Yhat):
    return float(np.linalg.norm(Y - Yhat) / np.linalg.norm(Y))
