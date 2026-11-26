"""Tree check, steps 4-5 (2026-11-25, plan pasted-content-id-0373-while-federated-coral; THEORY.md).

Does each KD student MLP neuron match a teacher BRANCH ROOT better than a single teacher neuron, a random-group root,
a co-firing-group root and a look-alike mean (model folding)?

Tree from weights only:
  parent p = teacher MLP neuron in block b-1. Its write w_p = W_proj[b-1][:, p], mean removed (LayerNorm centers).
  child score of neuron j in block b = cos(W_fc[b][j] * ln2_w[b], w_p). Children = top K = 8 by score.
  root = top right singular vector of the 8 unit-length child reads (x space), sign = positive mean dot.
Every candidate set has 3,072 units per teacher block, all built by the same operation except 'single' and 'fold':
  single    teacher neuron reads
  root      root of the parent's 8 weight-children (the concept)
  random    root of 8 random neurons of the same block
  cofire    root of j and its 7 most co-firing neurons (FIT half pre-activation correlation)
  fold      mean of j and its 7 nearest reads by cosine (look-alike, as model folding)
Match: pre-activation correlation on the class token. FIT half picks the best unit among teacher blocks b and b-1,
SCORE half measures it. Photos for reading only (4,000 fit-position photos). Output: out/tree_roots.json.
Usage: curves/.venv/bin/python -I distill/clipkd/tree_roots.py [--smoke]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
K = 8
SETS = ("single", "root", "random", "cofire", "fold", "proot", "voteA", "voteB", "voteW")


def z(a):
    a = a - a.mean(0)
    return a / (a.std(0) + 1e-8)


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def group_root(R):
    """R (K, D) unit reads -> (D,) top shared direction."""
    _, _, vt = np.linalg.svd(R, full_matrices=False)
    r = vt[0]
    return r if (R @ r).mean() >= 0 else -r


def proj_root(R, v):
    """Project v on the span of the rows of R (K, D), unit length."""
    q, _ = np.linalg.qr(R.T)
    r = q @ (q.T @ v)
    return r / (np.linalg.norm(r) + 1e-12)


def load(name, tag):
    d = np.load(os.path.join(OUT, f"cfc_{name}{tag}.npz"))
    return {k: d[k] for k in d.files}


def gelu(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = "_smoke" if a.smoke else ""
    rng = np.random.default_rng(0)
    T, S = load("teacher", tag), {n: load(n, tag) for n in ("kd", "baseline")}
    for n in S:
        assert (S[n]["photo_ids"] == T["photo_ids"]).all()
    fit, sco = T["half"] == 0, T["half"] == 1
    X = T["x"].astype(np.float32)                       # (N, 12, D)
    nb, H = T["W_fc"].shape[:2]
    pos = np.tile(np.arange(4), len(X) // 4)               # position of each photo in its class (0-3)
    gelu_t = lambda v: 0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v ** 3)))
    stats = {}
    res = {"K": K, "photos": int(len(X)), "sanity": {}, "blocks": {}}

    # sanity: recomputed pre-activation through GELU equals the saved class-token firing (read_cls.py)
    cls = os.path.join(OUT, f"cls_teacher{tag}.npz")
    if os.path.exists(cls):
        fire = np.load(cls)["fire"][:, 5, :64].astype(np.float32)
        pre = X[:, 5] @ T["W_fc"][5, :64].T + T["b_fc"][5, :64]
        res["sanity"]["max_abs_gelu_vs_saved_block6"] = float(np.abs(gelu(pre) - fire).max())

    # candidate directions per teacher block (x space), blocks 1..11 (roots need a parent block)
    cand = {}
    for b in range(1, nb):
        Wr = T["W_fc"][b]                                      # (H, D) reads in x space
        eff = unit(Wr * T["ln2_w"][b])
        wp = T["W_proj"][b - 1]                                # (D, H) writes of parents
        wraw = wp - wp.mean(0, keepdims=True)
        wp = unit((wp - wp.mean(0, keepdims=True)).T)          # (H, D)
        score = wp @ eff.T                                     # (parent, child)
        kids = np.argsort(-score, 1)[:, :K]
        ur = unit(Wr)
        pre_fit = z(X[fit, b] @ Wr.T)
        cof = (pre_fit.T @ pre_fit) / fit.sum()
        np.fill_diagonal(cof, -np.inf)
        cof_g = np.concatenate([np.arange(H)[:, None], np.argsort(-cof, 1)[:, :K - 1]], 1)
        lk = ur @ ur.T
        np.fill_diagonal(lk, -np.inf)
        lk_g = np.concatenate([np.arange(H)[:, None], np.argsort(-lk, 1)[:, :K - 1]], 1)
        rnd_g = np.stack([rng.choice(H, K, replace=False) for _ in range(H)])
        # 3.3 vote families (plan "find families by who reads the parent's output"): a neuron's vote list =
        # how much it reads from every parent of block b-1, times how loudly that parent fires (s_p).
        fire_par = gelu_t(X[:, b - 1] @ T["W_fc"][b - 1].T + T["b_fc"][b - 1])
        s_map = {"A": np.sqrt((fire_par[pos == 0] ** 2).mean(0)), "B": np.sqrt((fire_par[pos == 1] ** 2).mean(0)),
                 "W": np.ones(H, np.float32)}
        V0 = (Wr * T["ln2_w"][b]) @ wraw                       # (child, parent) raw votes
        vote_g = {}
        for m, sp in s_map.items():
            Vn = unit(V0 * sp[None, :])
            sim = Vn @ Vn.T
            np.fill_diagonal(sim, -np.inf)
            vote_g[m] = np.concatenate([np.arange(H)[:, None], np.argpartition(-sim, K - 1, 1)[:, :K - 1]], 1)
        ov = lambda g1, g2: np.median([len(set(a[1:]) & set(c[1:])) / (K - 1) for a, c in zip(g1, g2)])
        acof = np.abs((pre_fit.T @ pre_fit) / fit.sum())
        iu = np.triu_indices(K, 1)
        cof_in = lambda G: float(np.median([np.median(acof[np.ix_(g, g)][iu]) for g in G]))
        stats[b + 1] = {"M1_overlap_A_B": float(ov(vote_g["A"], vote_g["B"])),
                        "M5_overlap_W_A": float(ov(vote_g["W"], vote_g["A"])),
                        "M2_cofire_voteA": cof_in(vote_g["A"]), "M2_cofire_voteW": cof_in(vote_g["W"]),
                        "M2_cofire_oldkids": cof_in(kids), "M2_cofire_random": cof_in(rnd_g)}
        cand[b] = {
            "single": Wr,
            "root": np.stack([group_root(ur[g]) for g in kids]),
            "random": np.stack([group_root(ur[g]) for g in rnd_g]),
            "cofire": np.stack([group_root(ur[g]) for g in cof_g]),
            "fold": unit(ur[lk_g].mean(1)),
            # fix (plan: one fix if C1/C2 fail): the parent's write as the siblings read it =
            # g * w_p (where the write lands in x space), projected on the span of the 8 children's reads
            "proot": np.stack([proj_root(ur[g], T["ln2_w"][b] * wraw[:, p]) for p, g in enumerate(kids)]),
            "voteA": np.stack([group_root(ur[g]) for g in vote_g["A"]]),
            "voteB": np.stack([group_root(ur[g]) for g in vote_g["B"]]),
            "voteW": np.stack([group_root(ur[g]) for g in vote_g["W"]]),
            "_kids": kids,
            "_kid_score": score[np.arange(H)[:, None], kids],
        }
        print("block", b + 1, "built", flush=True)

    U = {b: {s: z(X[:, b] @ cand[b][s].T) for s in SETS} for b in cand}   # (N, H), standardised over all photos

    def best(Zs, s, b, shuffle=False):
        pools = [bb for bb in (b, b - 1) if bb in cand]
        P = np.concatenate([U[bb][s] for bb in pools], 1)
        Zs_ = Zs[rng.permutation(len(Zs))] if shuffle else Zs
        f, c = z(Zs_[fit]), z(P[fit])
        pick = np.argmax(f.T @ c / fit.sum(), 1)
        f2, c2 = z(Zs_[sco]), z(P[sco])
        return (f2 * c2[:, pick]).mean(0), pick

    # P0: planted known answer (root + equal-variance noise) must be matched back to its root
    hits = 0
    for _ in range(200):
        b = int(rng.integers(3, nb))
        k = int(rng.integers(0, H))
        fake = U[b]["root"][:, k] + rng.standard_normal(len(X))
        _, pick = best(fake[:, None], "root", b)
        hits += int(pick[0] == k)
    res["P0_hit_rate"] = hits / 200
    print("P0", res["P0_hit_rate"], flush=True)

    for b in range(1, nb):
        row = {}
        for n in S:
            Ps = z(S[n]["x"][:, b].astype(np.float32) @ S[n]["W_fc"][b].T)
            for s in SETS:
                r, _ = best(Ps, s, b)
                row[f"{n}_{s}"] = float(np.median(r))
            r, _ = best(Ps, "root", b, shuffle=True)
            row[f"{n}_root_shuffled"] = float(np.median(r))
        # G1: when a child is in its top 10%, is the root above its own median?
        c = cand[b]
        Ur, Us = U[b]["root"], U[b]["single"]
        med = np.median(Ur, 0)
        shares = []
        for p in range(H):
            hi = Us[:, c["_kids"][p]] >= np.percentile(Us[:, c["_kids"][p]], 90, axis=0)
            above = np.broadcast_to((Ur[:, p] > med[p])[:, None], hi.shape)
            shares.append(above[hi].mean())
        shares = np.array(shares)
        row["G1_median_share"] = float(np.median(shares))
        row["G1_frac_ge_75"] = float((shares >= .75).mean())
        row["G1_frac_lt_60"] = float((shares < .60).mean())
        row["kid_score_median"] = float(np.median(c["_kid_score"]))
        row.update(stats[b + 1])
        res["blocks"][b + 1] = row
        print(b + 1, {k: round(v, 3) for k, v in row.items()}, flush=True)

    with open(os.path.join(OUT, f"tree_roots{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
