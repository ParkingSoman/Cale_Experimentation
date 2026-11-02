"""Recipe parents, whole layer (family "recipe"). Base model + public text only.

1. Rank the 3072 neuron poles of layer L by output energy on calibration text. Candidates = the top M = 6K poles.
2. Recipe of each candidate = first-order split of its activation over every earlier unit (embedding, heads of layers
   0..L, neurons of layers 0..L-1), mean at its top positions minus mean at random positions. Computed in one streaming
   pass with virtual weights (nothing per-position is stored).
3. Hub guard: an ingredient in the top-20 of more than 20% of the candidates is generic (e.g. idle heads that rest on
   the first token) and is dropped.
4. NMF (k = K) on the recipes gives families.
5. One parent per family: the gate reads the family's shared ingredients (their writes at the family's firing
   positions); value and write are fitted in closed form to the family's summed output. Then one joint least-squares
   refit of all writes to the whole layer's output.
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import scipy.sparse as sp
from sklearn.decomposition import NMF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import SEQ, SwiGLULayer, ls_down, orig_weights  # noqa: E402

DEBUG = {}
METHOD = {"name": "recipe_v1", "family": "recipe", "minutes": 6.0}
M_PER_K, N_HOT, N_BASE, N_TOP, HUB, N_SHARED, N_CAL = 6, 50, 500, 20, 0.20, 20, 64


def silu(a):
    return a / (1 + np.exp(-a))


def dsilu(a):
    s = 1 / (1 + np.exp(-a))
    return s * (1 + a * (1 - s))


class _Grab(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.out = inner, None

    def __call__(self, x):
        self.out = x
        return self.inner(x)


def _stream(model, blocks, L, fn, batch=4):
    """Forward passes; fn(row0, ctx[l], h[l], res) per batch with every layer's o_proj input and MLP hidden."""
    m = model.model
    gc = [None] * (L + 1)
    gh = [None] * L
    for l in range(L + 1):
        gc[l] = m.layers[l].self_attn.o_proj = _Grab(m.layers[l].self_attn.o_proj)
        if l < L:
            gh[l] = m.layers[l].mlp.down_proj = _Grab(m.layers[l].mlp.down_proj)
    gr = m.layers[L].post_attention_layernorm = _Grab(m.layers[L].post_attention_layernorm)
    try:
        for i in range(0, blocks.shape[0], batch):
            mx.eval(model(blocks[i:i + batch, :-1]))
            d = gr.out.shape[-1]
            fn(i * SEQ, [np.array(g.out, dtype=np.float32).reshape(-1, d) for g in gc],
               [np.array(g.out, dtype=np.float32).reshape(-1, g.out.shape[-1]) for g in gh],
               np.array(gr.out, dtype=np.float64).reshape(-1, d))
    finally:
        for l in range(L + 1):
            m.layers[l].self_attn.o_proj = gc[l].inner
            if l < L:
                m.layers[l].mlp.down_proj = gh[l].inner
        m.layers[L].post_attention_layernorm = gr.inner


def _fit_value_out(X, g, Y, ridge=1e-4):
    a = silu(X @ g)
    F = (a[:, None] * X).astype(np.float32)
    A = (F.T @ F).astype(np.float64)
    A += np.eye(A.shape[0]) * ridge * np.trace(A) / A.shape[0]
    Mx = np.linalg.solve(A, (F.T @ Y.astype(np.float32)).astype(np.float64))
    Uu, Sv, Vt = np.linalg.svd(Mx, full_matrices=False)
    return Uu[:, 0] * np.sqrt(Sv[0]), Vt[0] * np.sqrt(Sv[0])


def _scale_gate(X, g_dir, Y, scales=(0.25, 0.5, 1, 2, 4, 8)):
    base = np.std(X @ g_dir) + 1e-9
    best = None
    for s in scales:
        g = g_dir * s / base
        u, d = _fit_value_out(X, g, Y)
        e = np.linalg.norm(Y - np.outer(silu(X @ g) * (X @ u), d))
        if best is None or e < best[0]:
            best = (e, g, u, d)
    return best[1], best[2], best[3]


def build(model, tok, L, K, cal, log=print, family_mode="nmf", seed_cos=0.2, n_parents=None):
    t0 = time.time()
    rng = np.random.default_rng(0)
    m = model.model
    blocks = cal[:N_CAL]
    ids = np.array(blocks[:, :-1]).reshape(-1)
    P = ids.size
    G, U, D = orig_weights(model, L)
    n = G.shape[0]
    w = np.array(m.layers[L].post_attention_layernorm.weight, dtype=np.float64)
    Wo = [np.array(m.layers[l].self_attn.o_proj.weight, dtype=np.float64) for l in range(L + 1)]
    Dl = [np.array(m.layers[l].mlp.down_proj.weight, dtype=np.float64) for l in range(L)]
    E = np.array(m.embed_tokens.weight, dtype=np.float32)
    nh = Wo[0].shape[1] // 64
    hd = Wo[0].shape[1] // nh

    # ---- pass 1: layer-L input on every position
    X = np.zeros((P, G.shape[1])); rms = np.zeros(P)

    def p1(r0, ctx, h, res):
        r = np.sqrt((res ** 2).mean(1) + 1e-5)
        X[r0:r0 + len(res)] = res / r[:, None] * w; rms[r0:r0 + len(res)] = r
    _stream(model, blocks, L, p1)
    valid = np.arange(P) % SEQ >= 2
    gpre, upre = X @ G.T, X @ U.T
    H = silu(gpre) * upre                                             # (P, n)
    Hv = H[valid]
    energy = np.concatenate([(np.maximum(Hv, 0) ** 2).sum(0), (np.maximum(-Hv, 0) ** 2).sum(0)]) * \
        np.tile((D ** 2).sum(0), 2)
    M = min(M_PER_K * K, 2 * n)
    cand = np.argsort(-energy)[:M]                                    # pole p < n: +, >= n: -
    neur, sgn = cand % n, np.where(cand < n, 1.0, -1.0)
    hot = []
    for c in range(M):
        order = np.argsort(-np.where(valid, sgn[c] * H[:, neur[c]], -np.inf))
        seen, pick = Counter(), []
        for p in order:
            if seen[ids[p]] < 3:
                pick.append(p); seen[ids[p]] += 1
            if len(pick) == N_HOT:
                break
        hot.append(np.array(pick))
    base = rng.choice(np.where(valid)[0], N_BASE, replace=False)
    # weights: recipe = sum_t Wa[t,c] * (G~_c . comp_t) + Wb[t,c] * (U~_c . comp_t), with 1/rms in the weights
    rows, cols, va, vb = [], [], [], []
    for c in range(M):
        for pos, wt in ((hot[c], 1.0 / N_HOT), (base, -1.0 / N_BASE)):
            g_, u_ = gpre[pos, neur[c]], upre[pos, neur[c]]
            rows += pos.tolist(); cols += [c] * len(pos)
            va += (wt * sgn[c] * dsilu(g_) * u_ / rms[pos]).tolist()
            vb += (wt * sgn[c] * silu(g_) / rms[pos]).tolist()
    Wa = sp.csr_matrix((va, (rows, cols)), shape=(P, M)).astype(np.float32)
    Wb = sp.csr_matrix((vb, (rows, cols)), shape=(P, M)).astype(np.float32)
    log(f"  recipe_v1 L{L}: pass 1 done, {M} candidate poles [{time.time() - t0:.0f}s]")

    # ---- pass 2: streaming accumulation of Wa^T comp, Wb^T comp
    SA = [np.zeros((M, Wo[0].shape[1]), np.float32) for _ in range(L + 1)]
    SB = [np.zeros((M, Wo[0].shape[1]), np.float32) for _ in range(L + 1)]
    AA = [np.zeros((M, n), np.float32) for _ in range(L)]
    AB = [np.zeros((M, n), np.float32) for _ in range(L)]
    EA = np.zeros((M, E.shape[1]), np.float32); EB = np.zeros_like(EA)

    def p2(r0, ctx, h, res):
        sl = slice(r0, r0 + len(res))
        wa, wb = Wa[sl].T.tocsr(), Wb[sl].T.tocsr()
        if wa.nnz == 0:
            return
        for l in range(L + 1):
            SA[l] += wa @ ctx[l]; SB[l] += wb @ ctx[l]
        for l in range(L):
            AA[l] += wa @ h[l]; AB[l] += wb @ h[l]
        emb = E[ids[sl]]
        EA[:] += wa @ emb; EB[:] += wb @ emb
    _stream(model, blocks, L, p2)
    Gt, Ut = G[neur] * w, U[neur] * w                                  # (M, d)
    parts, labels = [((Gt * EA).sum(1) + (Ut * EB).sum(1))[:, None]], [("embed", 0, 0)]
    for l in range(L + 1):
        for hh in range(nh):
            hs = slice(hh * hd, (hh + 1) * hd)
            oa, ob = SA[l][:, hs] @ Wo[l][:, hs].T, SB[l][:, hs] @ Wo[l][:, hs].T
            parts.append(((Gt * oa).sum(1) + (Ut * ob).sum(1))[:, None]); labels.append(("head", l, hh))
    for l in range(L):
        parts.append((Gt @ Dl[l]) * AA[l] + (Ut @ Dl[l]) * AB[l]); labels += [("neuron", l, j) for j in range(n)]
    R = np.concatenate(parts, 1)
    del parts, AA, AB
    log(f"  recipe_v1 L{L}: recipes {R.shape} [{time.time() - t0:.0f}s]")

    # ---- hub guard, families
    top = np.argsort(-np.abs(R), axis=1)[:, :N_TOP]
    use = np.bincount(top.ravel(), minlength=R.shape[1]) / M
    hubs = use > HUB
    R[:, hubs] = 0
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    singles = []                                                       # neurons kept as-is (cut style)
    if family_mode == "nmf":
        F = np.concatenate([np.maximum(Rn, 0), np.maximum(-Rn, 0)], 1)
        keep_cols = F.sum(0) > 0
        Wn = NMF(n_components=K, init="nndsvda", random_state=0, max_iter=500).fit_transform(F[:, keep_cols])
        fam = Wn.argmax(1)
        fams = [np.where(fam == k)[0] for k in range(K) if (fam == k).any()]
    elif family_mode == "tail":                                        # top K-P neurons as-is, P parents for the tail
        P_ = n_parents
        seeds = np.argsort(-(np.linalg.norm(Hv, axis=0) * np.linalg.norm(D, axis=0)))[:K - P_]
        singles = [int(j_) for j_ in seeds]
        tail = np.array([c for c in range(M) if neur[c] not in set(singles)])
        F = np.concatenate([np.maximum(Rn[tail], 0), np.maximum(-Rn[tail], 0)], 1)
        keep_cols = F.sum(0) > 0
        Wn = NMF(n_components=P_, init="nndsvda", random_state=0, max_iter=500).fit_transform(F[:, keep_cols])
        fam = Wn.argmax(1)
        fams = [tail[fam == k] for k in range(P_) if (fam == k).any()]
    else:                                                              # "seeded": the cut's top-K neurons are seeds
        seeds = np.argsort(-(np.linalg.norm(Hv, axis=0) * np.linalg.norm(D, axis=0)))[:K]
        seed_c = []
        for j_ in seeds:
            cs = np.where(neur == j_)[0]
            seed_c.append(cs[np.argmax(energy[cand[cs]])] if len(cs) else -1)
        seed_set = set(seeds.tolist())
        members = {k: [] for k in range(K)}
        ok = [k for k in range(K) if seed_c[k] >= 0]
        S_ = Rn[[seed_c[k] for k in ok]]
        for c in range(M):
            if neur[c] in seed_set:
                continue
            cs = S_ @ Rn[c]
            b = int(np.argmax(cs))
            if cs[b] >= seed_cos:
                members[ok[b]].append(c)
        fams = []
        for k in range(K):
            if seed_c[k] >= 0 and members[k]:
                fams.append(np.array([seed_c[k]] + members[k]))
            else:
                singles.append(int(seeds[k]))
    log(f"  recipe_v1 L{L}: {int(hubs.sum())} hub ingredients dropped; {len(fams)} families, sizes "
        f"{sorted(len(f) for f in fams)[::-1][:10]}..., {len(singles)} kept as-is [{time.time() - t0:.0f}s]")

    # ---- pass 3: each family's view of its shared ingredients' writes (at the family's firing positions)
    rows, cols, vals = [], [], []
    for k, f in enumerate(fams):
        pos = np.unique(np.concatenate([hot[c] for c in f]))
        rows += pos.tolist(); cols += [k] * len(pos); vals += (1.0 / len(pos) / rms[pos]).tolist()
    Fw = sp.csr_matrix((vals, (rows, cols)), shape=(P, len(fams))).astype(np.float32)
    FC = [np.zeros((len(fams), Wo[0].shape[1]), np.float32) for _ in range(L + 1)]
    FH = [np.zeros((len(fams), n), np.float32) for _ in range(L)]
    FE = np.zeros((len(fams), E.shape[1]), np.float32)

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
    _stream(model, blocks, L, p3)

    # ---- parents
    Y = H @ D.T                                                        # whole-layer output (P, d)
    pole_out = lambda c: (np.maximum(sgn[c] * H[:, neur[c]], 0) * sgn[c])[:, None] * D[:, neur[c]][None, :]
    units = []
    for k, f in enumerate(fams):
        shared = Rn[f].mean(0)
        agree = (np.sign(Rn[f]) == np.sign(shared)[None, :]).mean(0)
        score = np.where(agree >= 0.5, shared, 0)
        g_p = np.zeros(G.shape[1])
        for j in np.argsort(-score)[:N_SHARED]:
            if score[j] <= 0:
                break
            kind, l, i = labels[j]
            if kind == "embed":
                v = FE[k]
            elif kind == "head":
                hs = slice(i * hd, (i + 1) * hd)
                v = FC[l][k, hs] @ Wo[l][:, hs].T
            else:
                v = FH[l][k, i] * Dl[l][:, i]
            v = v * w
            g_p += score[j] * v / (np.linalg.norm(v) + 1e-12)
        if not np.any(g_p):
            g_p = G[neur[f[0]]].copy()
        Yf = sum(pole_out(c) for c in f)
        units.append(_scale_gate(X[valid], g_p / np.linalg.norm(g_p), Yf[valid]))
    for j in singles:
        units.append((G[j], U[j], D[:, j]))
    # fill up to K with the strongest remaining neurons (cut) if NMF left empty families
    used = set(neur.tolist()) | set(singles)
    extra = [j for j in np.argsort(-(np.linalg.norm(Hv, axis=0) * np.linalg.norm(D, axis=0))) if j not in used]
    for j in extra[:K - len(units)]:
        units.append((G[j], U[j], D[:, j]))
    DEBUG.update(fams=fams, neur=neur, sgn=sgn, energy=energy, cand=cand, X=X, valid=valid, H=H, Y=Y, D=D, G=G, U=U,
                 units=units, pole_out=pole_out)
    Gk = np.stack([u[0] for u in units]); Uk = np.stack([u[1] for u in units])
    Dk = ls_down(X[valid], Gk, Uk, Y[valid]).astype(np.float64)
    log(f"  recipe_v1 L{L}: built {len(units)} units [{time.time() - t0:.0f}s]")
    return SwiGLULayer(Gk, Uk, Dk)
