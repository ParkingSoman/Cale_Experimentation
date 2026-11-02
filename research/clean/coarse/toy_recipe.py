"""TOY RECIPE (dev, L15): do sibling neurons share a recipe of earlier units, and can a parent be built from it?

Recipe of a child pole = exact direct-path split of its activation (first order) over every earlier component
(embedding, 144 heads of L0-15, 23,040 neurons of L0-14), mean at its top positions minus mean at random positions.
Pass rules: CHILDREN-LOG.md "TOY RECIPE".

  python3 research/clean/coarse/toy_recipe.py
Writes results/clean-toy-recipe.json and research/clean/coarse/toy_recipe.md (readable recipes).
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, os.path.join(ROOT, "research", "clean"))
from tree_lib import load_model, token_blocks, SEQ  # noqa: E402
from r1 import load_w  # noqa: E402  (dev only: trained parents used as a reference, never as input to the parent rule)

L = 15
GROUPS = {"dates": ["833+", "1077+", "116-", "50+", "940-", "994+"],
          "conversions": ["655+", "168-", "131+", "653+", "297-", "865-"],
          "punctuation": ["16+", "410+", "717-", "365+"],
          "names": ["1402+", "638+", "184-", "646-"],
          "decoy": ["15+", "69-", "235-"]}
TRAINED = {"dates": "14-", "conversions": "19+", "punctuation": "7+", "names": "44+"}
N_HOT, N_BASE, N_SHARED = 50, 500, 20
rng = np.random.default_rng(0)


def silu(a):
    return a / (1 + np.exp(-a))


def dsilu(a):
    s = 1 / (1 + np.exp(-a))
    return s * (1 + a * (1 - s))


class Grab(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.out = inner, None

    def __call__(self, x):
        self.out = x
        return self.inner(x)


def run_grab(model, blocks, fn, batch=4):
    """Run blocks; for each batch call fn(batch_index_offset, grabs) while hooks hold the batch's tensors."""
    m = model.model
    g = {}
    for l in range(L + 1):
        g[("ctx", l)] = m.layers[l].self_attn.o_proj = Grab(m.layers[l].self_attn.o_proj)
        g[("ain", l)] = m.layers[l].self_attn = Grab(m.layers[l].self_attn)
        if l < L:
            g[("h", l)] = m.layers[l].mlp.down_proj = Grab(m.layers[l].mlp.down_proj)
    g[("res", L)] = m.layers[L].post_attention_layernorm = Grab(m.layers[L].post_attention_layernorm)
    for l in range(L + 1):   # self_attn wrapper must see the normed input: wrap input_layernorm output instead
        m.layers[l].self_attn = g[("ain", l)].inner
        g[("ain", l)] = m.layers[l].input_layernorm = Grab(m.layers[l].input_layernorm)
    try:
        for i in range(0, blocks.shape[0], batch):
            mx.eval(model(blocks[i:i + batch, :-1]))
            fn(i, {k: (np.array(v.out) if v.out is not None else None) for k, v in g.items()})
    finally:
        for l in range(L + 1):
            m.layers[l].self_attn.o_proj = g[("ctx", l)].inner
            m.layers[l].input_layernorm = g[("ain", l)].inner
            if l < L:
                m.layers[l].mlp.down_proj = g[("h", l)].inner
        m.layers[L].post_attention_layernorm = g[("res", L)].inner


def pole(s):
    return int(s[:-1]), (1.0 if s[-1] == "+" else -1.0)


def main():
    model, tok = load_model()
    m = model.model
    W = {"G": np.array(m.layers[L].mlp.gate_proj.weight, dtype=np.float64),
         "U": np.array(m.layers[L].mlp.up_proj.weight, dtype=np.float64),
         "w": np.array(m.layers[L].post_attention_layernorm.weight, dtype=np.float64)}
    Wo = [np.array(m.layers[l].self_attn.o_proj.weight, dtype=np.float64) for l in range(L + 1)]
    Dn = [np.array(m.layers[l].mlp.down_proj.weight, dtype=np.float64) for l in range(L)]
    E = np.array(m.embed_tokens.weight, dtype=np.float64)
    cal = token_blocks(tok, "train", 64)
    held = token_blocks(tok, "test", 40)
    ids_cal = np.array(cal[:, :-1])
    names = [(g, s) for g, lst in GROUPS.items() for s in lst]
    neur = np.array([pole(s)[0] for _, s in names])
    sgn = np.array([pole(s)[1] for _, s in names])

    # ---- pass 1: each child's activation at L15 on calibration text -> hot positions
    X15 = np.zeros((ids_cal.size, 576), dtype=np.float32)

    def p1(i, g):
        n = g[("res", L)]
        x = n.reshape(-1, 576)
        rms = np.sqrt((x.astype(np.float64) ** 2).mean(1, keepdims=True) + 1e-5)
        X15[i * SEQ:(i * SEQ + x.shape[0])] = (x / rms * W["w"]).astype(np.float32)
    run_grab(model, cal, p1)
    valid = np.arange(ids_cal.size) % SEQ >= 2
    Hc = silu(X15 @ W["G"][neur].T) * (X15 @ W["U"][neur].T) * sgn
    hot = []
    for c in range(len(names)):
        order = np.argsort(-np.where(valid, Hc[:, c], -np.inf))
        seen, pick = Counter(), []
        for p in order:
            t = ids_cal.flat[p]
            if seen[t] < 3:
                pick.append(p); seen[t] += 1
            if len(pick) == N_HOT:
                break
        hot.append(np.array(pick))
    base = rng.choice(np.where(valid)[0], N_BASE, replace=False)
    sel = np.unique(np.concatenate(hot + [base]))
    pos_of = {p: j for j, p in enumerate(sel)}

    # ---- pass 2: components at selected positions
    S = len(sel)
    res = np.zeros((S, 576)); emb = np.zeros((S, 576))
    ctx = np.zeros((L + 1, S, 576)); hid = np.zeros((L, S, 1536), dtype=np.float32)

    def p2(i, g):
        lo, hi = i * SEQ, (i + 4) * SEQ
        js = [j for j, p in enumerate(sel) if lo <= p < hi]
        if not js:
            return
        loc = sel[js] - lo
        res[js] = g[("res", L)].reshape(-1, 576)[loc]
        emb[js] = E[ids_cal.reshape(-1)[sel[js]]]
        for l in range(L + 1):
            ctx[l, js] = g[("ctx", l)].reshape(-1, 576)[loc]
        for l in range(L):
            hid[l, js] = g[("h", l)].reshape(-1, 1536)[loc]
    run_grab(model, cal, p2)
    # exact identity check: embedding + heads + MLP outputs = residual
    recon = emb + sum(ctx[l] @ Wo[l].T for l in range(L + 1)) + sum(hid[l].astype(np.float64) @ Dn[l].T for l in range(L))
    ident_err = float(np.abs(recon - res).max() / np.abs(res).max())
    print(f"residual identity: max rel err {ident_err:.2e}", flush=True)

    # ---- recipes
    rms = np.sqrt((res ** 2).mean(1) + 1e-5)
    nrm = res / rms[:, None] * W["w"]
    labels = ["embed"] + [f"head {l}.{h}" for l in range(L + 1) for h in range(9)] + \
             [f"neuron {l}.{n}" for l in range(L) for n in range(1536)]

    def contrib(c, js):
        """(len(js), n_units) first-order contributions to child c's pole activation."""
        gpre = nrm[js] @ W["G"][neur[c]]; up = nrm[js] @ W["U"][neur[c]]
        coef = sgn[c] * (dsilu(gpre)[:, None] * up[:, None] * W["G"][neur[c]] + silu(gpre)[:, None] * W["U"][neur[c]])
        coef = coef * W["w"] / rms[js, None]                              # (n, 576) on the raw residual
        out = [np.einsum("nd,nd->n", coef, emb[js])[:, None]]
        for l in range(L + 1):
            hc = (coef @ Wo[l]).reshape(len(js), 9, 64) * ctx[l, js].reshape(len(js), 9, 64)
            out.append(hc.sum(-1))
        for l in range(L):
            out.append(hid[l, js] * (coef @ Dn[l]))
        return np.concatenate(out, 1)

    jb = np.array([pos_of[p] for p in base])
    R = []
    for c in range(len(names)):
        jh = np.array([pos_of[p] for p in hot[c]])
        R.append(contrib(c, jh).mean(0) - contrib(c, jb).mean(0))
    R = np.array(R)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    C = Rn @ Rn.T
    grp = np.array([g for g, _ in names])
    kids = grp != "decoy"
    within, across, decoy = [], [], []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if grp[a] == "decoy" or grp[b] == "decoy":
                decoy.append(C[a, b])
            elif grp[a] == grp[b]:
                within.append(C[a, b])
            else:
                across.append(C[a, b])
    per_group = {g: float(np.mean([C[a, b] for a in range(len(names)) for b in range(a + 1, len(names))
                                  if grp[a] == grp[b] == g])) for g in GROUPS if g != "decoy"}
    rule1 = bool(np.mean(within) >= 0.3 and np.mean(within) >= 3 * max(np.mean(across), 1e-9))
    print(f"recipe cosine within {np.mean(within):.3f}  across {np.mean(across):.3f}  with decoys {np.mean(decoy):.3f}  "
          f"per group {per_group}  RULE1 {'PASS' if rule1 else 'fail'}", flush=True)

    # ---- NMF on the children's recipes
    F = np.concatenate([np.maximum(Rn[kids], 0), np.maximum(-Rn[kids], 0)], 1)
    nmf = NMF(n_components=4, init="nndsvda", random_state=0, max_iter=2000)
    Wn = nmf.fit_transform(F)
    assign = Wn.argmax(1)
    ari = float(adjusted_rand_score(grp[kids], assign))
    print(f"NMF k=4 assignment {list(zip([s for g, s in names if g != 'decoy'], assign.tolist()))}  ARI {ari:.2f}  "
          f"RULE2 {'PASS' if ari >= 0.5 else 'fail'}", flush=True)

    # ---- shared ingredients per group, and a parent gate built from them only
    Xh = []

    def ph(i, g):
        x = g[("res", L)].reshape(-1, 576).astype(np.float64)
        Xh.append(x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5) * W["w"])
    run_grab(model, held, ph)
    Xh = np.concatenate(Xh)
    vh = np.arange(Xh.shape[0]) % SEQ >= 2
    Xh = Xh[vh]
    Hh = silu(Xh @ W["G"][neur].T) * (Xh @ W["U"][neur].T) * sgn
    nh = max(1, Xh.shape[0] // 100)

    def cover(a, members):
        srt = np.sort(a)
        return float(np.median([np.median(np.searchsorted(srt, a[np.argsort(-Hh[:, c])[:nh]]) / len(srt))
                                for c in members]))

    def union_jaccard(a, members):
        u = set()
        for c in members:
            u |= set(np.argsort(-Hh[:, c])[:nh].tolist())
        top = set(np.argsort(-a)[:len(u)].tolist())
        return len(top & u) / len(top | u)

    report = {"identity_rel_err": ident_err, "within": float(np.mean(within)), "across": float(np.mean(across)),
              "decoy": float(np.mean(decoy)), "per_group": per_group, "rule1": rule1, "nmf_assign": assign.tolist(),
              "ari": ari, "groups": {}}
    md = ["# Toy recipes (L15, dev)\n"]
    for gname in [g for g in GROUPS if g != "decoy"]:
        mem = np.where(grp == gname)[0]
        shared = Rn[mem].mean(0)
        agree = (np.sign(Rn[mem]) == np.sign(shared)[None, :]).mean(0)
        score = np.where(agree >= 0.5, shared, 0)
        top = np.argsort(-score)[:N_SHARED]
        # parent gate: read the shared ingredients' writes, in the MLP's normalized input space
        js = np.array([pos_of[p] for c in mem for p in hot[c]])
        g_p = np.zeros(576)
        for k in top:
            lab = labels[k]
            if lab == "embed":
                v = emb[js]
            elif lab.startswith("head"):
                l, h = map(int, lab.split()[1].split("."))
                v = ctx[l, js][:, h * 64:(h + 1) * 64] @ Wo[l][:, h * 64:(h + 1) * 64].T
            else:
                l, n = map(int, lab.split()[1].split("."))
                v = hid[l, js, n][:, None].astype(np.float64) * Dn[l][:, n][None, :]
            v = (v / rms[js, None] * W["w"]).mean(0)
            g_p += score[k] * v / (np.linalg.norm(v) + 1e-12)
        dom = max(mem, key=lambda c: cover(Xh @ W["G"][neur[c]], [d for d in mem if d != c]))
        tn, ts = pole(TRAINED[gname])
        P = load_w(L, 48, "global")
        cand = {"recipe_parent": Xh @ g_p,
                "avg_child_gate_H1": Xh @ W["G"][neur[mem]].mean(0),
                "dominant_child_gate": Xh @ W["G"][neur[dom]],
                "trained_parent_gate": Xh @ P[0][tn],
                "trained_parent_hidden": ts * silu(Xh @ P[0][tn]) * (Xh @ P[1][tn]),
                "random_gate": Xh @ rng.standard_normal(576)}
        sc = {k: {"cover": cover(a, mem), "union_jaccard": union_jaccard(a, mem)} for k, a in cand.items()}
        rule3 = bool(sc["recipe_parent"]["cover"] >= 0.9 and sc["recipe_parent"]["cover"] > sc["avg_child_gate_H1"]["cover"])
        print(f"{gname:12s} " + "  ".join(f"{k} {v['cover']:.2f}/{v['union_jaccard']:.2f}" for k, v in sc.items()) +
              f"  RULE3 {'PASS' if rule3 else 'fail'}", flush=True)
        report["groups"][gname] = {"members": [names[c][1] for c in mem], "scores": sc, "rule3": rule3,
                                   "dominant": names[dom][1], "g_p": g_p.tolist(), "cos_recipe_parent_vs_trained_gate":
                                   float(np.corrcoef(cand["recipe_parent"], cand["trained_parent_gate"])[0, 1]),
                                   "shared_top": [(labels[k], float(score[k]), float(agree[k])) for k in top]}
        md.append(f"\n## {gname}: children {', '.join(names[c][1] for c in mem)}\n")
        md.append("| ingredient | shared weight | share of siblings that agree |\n|---|---|---|")
        md += [f"| {labels[k]} | {score[k]:.3f} | {agree[k]:.2f} |" for k in top]
    json.dump(report, open(os.path.join(ROOT, "results", "clean-toy-recipe.json"), "w"), indent=1)
    open(os.path.join(HERE, "toy_recipe.md"), "w").write("\n".join(md) + "\n")
    np.savez(os.path.join(HERE, "toy_recipe_cache.npz"), R=R, sel=sel, hot=np.array(hot), base=base)


if __name__ == "__main__":
    main()
