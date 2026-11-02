"""Upstream-grouping parents, whole layer (family "upstream"). Base model + public text only.

Essay mechanism: a later neuron SPLITS a grouping that an earlier unit already computes ("a car neuron provides a
reusable judgment and a sheltered place for taxis, ambulances, and other specialists"). So:
1. Seeds: the cut's top K-P neurons stay as-is (detailed).
2. Candidates: the next 6K neurons by output energy. For each, a BOTH-POLE gate recipe: what drives its gate open at
   its positive-pole top positions AND at its negative-pole top positions (each minus random positions); shared part =
   min of the two where both are positive.
3. IDF over candidates (generic ingredients shared by many neurons are down-weighted). Family key = the neuron's top
   IDF-weighted upstream NEURON (its "judgment" neuron). Families = candidates with the same key (>= 2 members).
4. P parents for the P largest-energy families: gate reads the family's shared IDF-weighted ingredients (their writes
   at the family's both-pole positions); value + write fitted in closed form to the family's summed output.
5. Joint least-squares refit of all writes.
"""
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import SEQ, SwiGLULayer, ls_down, orig_weights  # noqa: E402
from recipe_v1 import _stream, _scale_gate, silu  # noqa: E402

DEBUG = {}
METHOD = {"name": "recipe_v4", "family": "upstream", "minutes": 6.0}
M_PER_K, N_HOT, N_BASE, N_TOP_IDF, N_SHARED, N_CAL = 6, 50, 500, 20, 20, 64


def build(model, tok, L, K, cal, log=print, n_parents=None, keys_for=None):
    n_parents = n_parents or K // 4
    t0 = time.time()
    rng = np.random.default_rng(0)
    m = model.model
    blocks = cal[:N_CAL]
    ids = np.array(blocks[:, :-1]).reshape(-1)
    P = ids.size
    G, U, D = orig_weights(model, L)
    n, d = G.shape
    w = np.array(m.layers[L].post_attention_layernorm.weight, dtype=np.float64)
    Wo = [np.array(m.layers[l].self_attn.o_proj.weight, dtype=np.float64) for l in range(L + 1)]
    Dl = [np.array(m.layers[l].mlp.down_proj.weight, dtype=np.float64) for l in range(L)]
    E = np.array(m.embed_tokens.weight, dtype=np.float32)
    nh, hd = Wo[0].shape[1] // 64, 64

    # ---- pass 1
    X = np.zeros((P, d)); rms = np.zeros(P)

    def p1(r0, ctx, h, res):
        r = np.sqrt((res ** 2).mean(1) + 1e-5)
        X[r0:r0 + len(res)] = res / r[:, None] * w; rms[r0:r0 + len(res)] = r
    _stream(model, blocks, L, p1)
    valid = np.arange(P) % SEQ >= 2
    gpre, upre = X @ G.T, X @ U.T
    H = silu(gpre) * upre
    Hv = H[valid]
    cut_score = np.linalg.norm(Hv, axis=0) * np.linalg.norm(D, axis=0)
    order = np.argsort(-cut_score)
    seeds = order[:K - n_parents]
    cand = order[K - n_parents:min(n, K - n_parents + M_PER_K * K)]
    if keys_for is not None:
        seeds, cand = np.array([], dtype=int), np.asarray(keys_for)
    M = len(cand)

    def top(score):
        o = np.argsort(-np.where(valid, score, -np.inf)); seen, out = Counter(), []
        for p in o:
            if not np.isfinite(score[p]):
                break
            if seen[ids[p]] < 3:
                out.append(p); seen[ids[p]] += 1
            if len(out) == N_HOT:
                break
        return np.array(out, dtype=int)
    POS, NEG = [], []
    for j in cand:
        g, u, h = gpre[:, j], upre[:, j], H[:, j]
        POS.append(top(np.where((g > 0) & (u > 0), h, -np.inf)))
        NEG.append(top(np.where((g > 0) & (u < 0), -h, -np.inf)))
    base = rng.choice(np.where(valid)[0], N_BASE, replace=False)

    def wmat(sets):
        rows, cols, vals = [], [], []
        for c, pos in enumerate(sets):
            if len(pos) == 0:
                continue
            rows += pos.tolist() + base.tolist(); cols += [c] * (len(pos) + N_BASE)
            vals += (1.0 / len(pos) / rms[pos]).tolist() + (-1.0 / N_BASE / rms[base]).tolist()
        return sp.csr_matrix((vals, (rows, cols)), shape=(P, M)).astype(np.float32)
    Wp, Wn = wmat(POS), wmat(NEG)
    log(f"  recipe_v4 L{L}: pass 1, {len(seeds)} seeds, {M} candidates [{time.time() - t0:.0f}s]")

    # ---- pass 2: both-pole gate recipes
    acc = {k: {"S": [np.zeros((M, d), np.float32) for _ in range(L + 1)],
               "A": [np.zeros((M, n), np.float32) for _ in range(L)], "E": np.zeros((M, E.shape[1]), np.float32)}
           for k in ("p", "n")}

    def p2(r0, ctx, h, res):
        sl = slice(r0, r0 + len(res))
        emb = E[ids[sl]]
        for k, Wm in (("p", Wp), ("n", Wn)):
            wt = Wm[sl].T.tocsr()
            if wt.nnz == 0:
                continue
            for l in range(L + 1):
                acc[k]["S"][l] += wt @ ctx[l]
            for l in range(L):
                acc[k]["A"][l] += wt @ h[l]
            acc[k]["E"] += wt @ emb
    _stream(model, blocks, L, p2)
    Gt = G[cand] * w
    labels = [("embed", 0, 0)] + [("head", l, i) for l in range(L + 1) for i in range(nh)] + \
             [("neuron", l, j) for l in range(L) for j in range(n)]

    def recipe(k):
        parts = [(Gt * acc[k]["E"]).sum(1)[:, None]]
        for l in range(L + 1):
            for i in range(nh):
                hs = slice(i * hd, (i + 1) * hd)
                parts.append((Gt * (acc[k]["S"][l][:, hs] @ Wo[l][:, hs].T)).sum(1)[:, None])
        for l in range(L):
            parts.append((Gt @ Dl[l]) * acc[k]["A"][l])
        return np.concatenate(parts, 1)
    Rp, Rn = recipe("p"), recipe("n")
    del acc
    shared = np.where((Rp > 0) & (Rn > 0), np.minimum(Rp, Rn), 0.0)
    del Rp, Rn
    topk = np.argsort(-shared, axis=1)[:, :N_TOP_IDF]
    df = np.maximum(np.bincount(topk.ravel(), minlength=shared.shape[1]) / M, 1.0 / M)
    score = shared * np.log(1.0 / df)[None, :]
    neuron0 = 1 + (L + 1) * nh
    keys = neuron0 + np.argmax(score[:, neuron0:], axis=1)
    keyed = np.where(score[np.arange(M), keys] > 0, keys, -1)
    if keys_for is not None:
        return {int(cand[c]): int(k) for c, k in enumerate(keyed)}
    groups = defaultdict(list)
    for c, k in enumerate(keyed):
        if k >= 0:
            groups[int(k)].append(c)
    fam_energy = {k: sum(cut_score[cand[c]] ** 2 for c in v) for k, v in groups.items() if len(v) >= 2}
    chosen = sorted(fam_energy, key=lambda k: -fam_energy[k])[:n_parents]
    fams = [np.array(groups[k]) for k in chosen]
    log(f"  recipe_v4 L{L}: {len(groups)} keys, {len(fam_energy)} families with >= 2, chosen sizes "
        f"{[len(f) for f in fams]} keys {[labels[k][1:] for k in chosen][:6]}... [{time.time() - t0:.0f}s]")

    # ---- pass 3: family view of ingredient writes at both-pole positions
    rows, cols, vals = [], [], []
    for f_i, f in enumerate(fams):
        pos = np.unique(np.concatenate([np.concatenate([POS[c], NEG[c]]) for c in f]))
        rows += pos.tolist(); cols += [f_i] * len(pos); vals += (1.0 / len(pos) / rms[pos]).tolist()
    Fw = sp.csr_matrix((vals, (rows, cols)), shape=(P, max(1, len(fams)))).astype(np.float32)
    FC = [np.zeros((max(1, len(fams)), d), np.float32) for _ in range(L + 1)]
    FH = [np.zeros((max(1, len(fams)), n), np.float32) for _ in range(L)]
    FE = np.zeros((max(1, len(fams)), E.shape[1]), np.float32)

    def p3(r0, ctx, h, res):
        sl = slice(r0, r0 + len(res))
        fw = Fw[sl].T.tocsr()
        if fw.nnz == 0:
            return
        for l in range(L + 1):
            FC[l] += fw @ ctx[l]
        for l in range(L):
            FH[l] += fw @ h[l]
        FE[:] += fw @ E[ids[sl]]
    if fams:
        _stream(model, blocks, L, p3)

    # ---- parents
    Y = H @ D.T
    units = [(G[j], U[j], D[:, j]) for j in seeds]
    Sn = score / (np.linalg.norm(score, axis=1, keepdims=True) + 1e-12)
    for f_i, f in enumerate(fams):
        sh = Sn[f].mean(0)
        agree = (Sn[f] > 0).mean(0)
        sc = np.where(agree >= 0.5, sh, 0)
        g_p = np.zeros(d)
        for j in np.argsort(-sc)[:N_SHARED]:
            if sc[j] <= 0:
                break
            kind, l, i = labels[j]
            if kind == "embed":
                v = FE[f_i]
            elif kind == "head":
                hs = slice(i * hd, (i + 1) * hd); v = FC[l][f_i, hs] @ Wo[l][:, hs].T
            else:
                v = FH[l][f_i, i] * Dl[l][:, i]
            v = v * w
            g_p += sc[j] * v / (np.linalg.norm(v) + 1e-12)
        if not np.any(g_p):
            g_p = G[cand[f[0]]].copy()
        members = cand[f]
        Yf = H[:, members] @ D[:, members].T
        units.append(_scale_gate(X[valid], g_p / np.linalg.norm(g_p), Yf[valid]))
    for j in order[K - n_parents:]:
        if len(units) >= K:
            break
        if j not in set(cand[np.concatenate(fams)].tolist()) if fams else True:
            units.append((G[j], U[j], D[:, j]))
    Gk = np.stack([u[0] for u in units]); Uk = np.stack([u[1] for u in units])
    Dk = ls_down(X[valid], Gk, Uk, Y[valid]).astype(np.float64)
    DEBUG.update(fams=fams, cand=cand, keys=[labels[k] for k in chosen], seeds=seeds)
    log(f"  recipe_v4 L{L}: built {len(units)} units [{time.time() - t0:.0f}s]")
    return SwiGLULayer(Gk, Uk, Dk)
