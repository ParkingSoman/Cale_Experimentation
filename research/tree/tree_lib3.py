"""Helpers for Z4 (query heads and depth). No training."""
import copy
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.base import create_attention_mask
from tree_lib import Capture, rel_err

NH, NKV, HD = 9, 3, 64
GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]   # query head h uses kv head h // 3 (checked in z4.py)


def cols(h):
    return slice(h * HD, (h + 1) * HD)


class Walk:
    """Runs the intact model layer by layer over calibration blocks so that each layer sees
    its original input. For one layer at a time it can give the o_proj input (all heads' outputs)."""

    def __init__(self, model, blocks, batch=4):
        self.model, self.blocks, self.batch = model, blocks, batch
        self.h = []
        self.mask = None
        for i in range(0, blocks.shape[0], batch):
            self.h.append(model.model.embed_tokens(blocks[i:i + batch, :-1]))
        self.mask = create_attention_mask(self.h[0], None)
        self.i = 0

    def layer(self):
        return self.model.model.layers[self.i]

    def original(self):
        """(A, Y, new h list): o_proj input and output of the intact layer, and advance state later by advance()."""
        l = self.layer()
        cap = Capture(l.self_attn.o_proj)
        l.self_attn.o_proj = cap
        self._next = []
        for hb in self.h:
            self._next.append(l(hb, self.mask))
        mx.eval(self._next)
        l.self_attn.o_proj = cap.inner
        A = np.concatenate(cap.x, 0)
        Wo = np.array(l.self_attn.o_proj.weight)
        return A, A @ Wo.T

    def modified_A(self):
        """o_proj input of this layer in its current (modified) state, upstream still intact."""
        l = self.layer()
        cap = Capture(l.self_attn.o_proj)
        l.self_attn.o_proj = cap
        for hb in self.h:
            mx.eval(l.self_attn(l.input_layernorm(hb), self.mask))
        l.self_attn.o_proj = cap.inner
        return np.concatenate(cap.x, 0)

    def advance(self):
        self.h = self._next
        self.i += 1


def ridge_cols(A, keep_cols, Y, ridge=1e-4):
    """Least squares: o_proj columns (576 x len(keep_cols)) so that A[:, keep_cols] @ C.T ~ Y."""
    S = A[:, keep_cols].astype(np.float64)
    M = S.T @ S
    M += np.eye(M.shape[0]) * ridge * np.trace(M) / M.shape[0]
    return np.linalg.solve(M, S.T @ Y.astype(np.float64)).T.astype(np.float32)


def head_stats(A, Wo):
    """Per head: activity (mean token norm of head output x Frobenius norm of its o_proj columns)
    and the 9x9 cosine similarity of head outputs (flattened over tokens)."""
    Hs = [A[:, cols(h)] for h in range(NH)]
    act = np.array([np.linalg.norm(H, axis=1).mean() * np.linalg.norm(Wo[:, cols(h)]) for h, H in enumerate(Hs)])
    nr = np.array([np.linalg.norm(H) for H in Hs]) + 1e-12
    cos = np.zeros((NH, NH))
    for a in range(NH):
        for b in range(NH):
            cos[a, b] = float((Hs[a] * Hs[b]).sum()) / (nr[a] * nr[b])
    return act, cos


def head_plan(method, size, act, cos, rng):
    """Clusters of heads. Each cluster is a list of heads; its first head is kept, the others vanish
    (merge) and heads in no cluster vanish (cut). size = heads left in total (6 or 3)."""
    per = size // 3
    out = []
    for g in GROUPS:
        if method == "cut":
            keep = sorted(sorted(g, key=lambda h: -act[h])[:per])
            out += [[h] for h in keep]
        elif per == 1:                      # whole group collapses to one head
            out.append(list(g))
        else:                               # per == 2: merge one pair, the third head stays alone
            if method == "merge":
                pairs = [(a, b) for a in g for b in g if a < b]
                a, b = max(pairs, key=lambda p: cos[p[0], p[1]])
            else:                           # random pair
                a, b = sorted(rng.choice(g, 2, replace=False).tolist())
            rest = [h for h in g if h not in (a, b)]
            out += [[a, b]] + [[h] for h in rest]
    return out


def apply_plan(Wq, Wo, plan):
    """New q_proj and o_proj (no refit): kept head gets mean of members' q rows and the SUM of members'
    o_proj columns (the analogue of the MLP rule); all other heads get zero o_proj columns."""
    Q, O = Wq.copy(), np.zeros_like(Wo)
    for cl in plan:
        k = cl[0]
        Q[cols(k)] = np.mean([Wq[cols(h)] for h in cl], 0)
        O[:, cols(k)] = np.sum([Wo[:, cols(h)] for h in cl], 0)
    return Q, O


def kept_cols(plan):
    return np.concatenate([np.arange(cl[0] * HD, (cl[0] + 1) * HD) for cl in plan])


def flat_layer(layer):
    return dict(tree_flatten(layer.parameters()))


def average_layers(la, lb):
    """New layer = copy of la with every weight replaced by the mean of la and lb."""
    new = copy.deepcopy(la)
    fa, fb = flat_layer(la), flat_layer(lb)
    new.update(tree_unflatten([(k, (fa[k] + fb[k]) / 2) for k in fa]))
    mx.eval(new.parameters())
    return new


def layer_distance(la, lb):
    fa, fb = flat_layer(la), flat_layer(lb)
    d = sum(float(((fa[k] - fb[k]) ** 2).sum()) for k in fa)
    n = sum(float((fa[k] ** 2).sum() + (fb[k] ** 2).sum()) for k in fa) / 2
    return (d / n) ** 0.5
