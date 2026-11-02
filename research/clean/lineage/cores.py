"""Round 2 helpers: top-k cores of pre-activations and coverage between child and parent cores."""
import numpy as np
import scipy.sparse as sp
import mlx.core as mx
from common import *


def big_blocks(tok):
    return mx.concatenate([blocks(tok, "train", 40, skip=400), blocks(tok, "code_test", 8, skip=60)], 0)


def preact(model, B, L):
    Wi, bi, _, _ = mlp_w(model, L)
    return mlp_in(model, B, L).astype(np.float32) @ Wi.T + bi


def core(Z, k):
    """(n_neurons, k) token indices of the top-k z per neuron."""
    return np.argpartition(-Z, k, axis=0)[:k].T


def indicator(C, N):
    n, k = C.shape
    return sp.csr_matrix((np.ones(n * k, np.float32), (np.repeat(np.arange(n), k), C.ravel())), shape=(n, N))


def coverage(Cchild, Cpar, N):
    """(n_child, n_par) share of each child's core inside each parent's core."""
    return (indicator(Cchild, N) @ indicator(Cpar, N).T).toarray() / Cchild.shape[1]


def split_stats(Cchild, cov, thr=.5, max_overlap=10):
    N = int(Cchild.max()) + 1
    anc = cov.argmax(1); best = cov.max(1); has = best >= thr
    I = indicator(Cchild, N); own = (I @ I.T).toarray()
    in_split = np.zeros(len(anc), bool); kids_per = []
    for a in np.unique(anc[has]):
        kids = np.where(has & (anc == a))[0]
        kids_per.append(len(kids))
        if len(kids) >= 2:
            sub = own[np.ix_(kids, kids)]; np.fill_diagonal(sub, 0)
            ok = (sub <= max_overlap).any(1)   # a kid with at least one sibling it does not overlap much
            in_split[kids[ok]] = True
    return {"has_parent": float(has.mean()), "split_share": float(in_split.mean()),
            "kids_per_parent_mean": float(np.mean(kids_per)) if kids_per else 0.0,
            "parents_used": int(len(kids_per)), "best_cov_median": float(np.median(best))}, anc, has, in_split
