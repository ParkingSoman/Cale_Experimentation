"""R1 reading step: what does each trained small-layer neuron do, and is it a coarse parent of old neurons?

Methods (from research/clean/R0b-interp-methods.md, the essay's own recipe and delphi):
  - Both poles, long contexts, two text sets (wikitext test, unseen wikitext train blocks).
  - Ablation: zero one unit, measure the per-token loss change (causal effect).
  - Direct logit effect: the output column through the final norm and the tied unembedding.
  - Parent test: candidate children = old neurons that carry the trained neuron's signal (R1 sparse regression,
    first-token excluded). Compare ablation effects (correlation over tokens) and coverage: on each child's top
    tokens, how high is the parent's activation (an absorption check, arXiv 2409.14507)?
  - Descriptions are written later by one subagent and scored blind by another (delphi detection score).

  python3 research/clean/r1read.py --layer 15 --K 10 --tag local_rand
Writes results/clean-r1read-L{L}-K{K}-{tag}.json, research/clean/r1read/L{L}_K{K}_{tag}_dossier.md (for the
explainer), ..._detect.md (shuffled held-out examples for the blind scorer) and ..._key.json (answer key).
"""
import argparse
import json
import os
import sys

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, load_w, SmallMLP, omp  # noqa: E402

OUT = os.path.join(HERE, "r1read")
CTX = 48


class Ablate:
    """Wraps an MLP and zeroes one of its hidden units."""

    def __init__(self, inner, j):
        self.inner, self.j = inner, j

    def __call__(self, x):
        m = self.inner
        g = m.gate_proj(x)
        h = g * mx.sigmoid(g) * m.up_proj(x)
        mask = (mx.arange(h.shape[-1]) != self.j).astype(h.dtype)
        return m.down_proj(h * mask)


def token_ce(model, blocks, batch=4):
    out = []
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        lg = model(b[:, :-1])
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        out.append(np.array(-mx.take_along_axis(lg, b[:, 1:, None], axis=-1).squeeze(-1)))
    return np.concatenate(out).reshape(-1)


def ctx(tok, ids, p):
    """Long context ending at position p. The token at p is marked [[ ]]."""
    a = max(0, p - CTX)
    return (tok.decode(ids[a:p].tolist()) + "[[" + tok.decode([int(ids[p])]) + "]]").replace("\n", " ")


def poles(tok, ids, act, valid, n=8):
    a = np.where(valid, act, np.nan)
    hi = np.argsort(-np.nan_to_num(a, nan=-np.inf))[:n]
    lo = np.argsort(np.nan_to_num(a, nan=np.inf))[:n]
    return ([(float(act[p]), ctx(tok, ids, p)) for p in hi], [(float(act[p]), ctx(tok, ids, p)) for p in lo])


def dla(P_d, normw, emb, tok, n=12):
    w = (P_d * normw) @ emb.T
    return ([tok.decode([int(i)]) for i in np.argsort(-w)[:n]], [tok.decode([int(i)]) for i in np.argsort(w)[:n]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tag", default="local_rand")
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--children_from", default="omp", choices=["omp", "cover"])
    args = ap.parse_args()
    L, K, tag = args.layer, args.K, args.tag
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    model, tok = load_model()
    W = Weights(model)
    emb = np.array(model.model.embed_tokens.weight, dtype=np.float32)
    normw = np.array(model.model.norm.weight, dtype=np.float32)
    cal = token_blocks(tok, "train", 64)
    setA = token_blocks(tok, "test", 100)[:40]                          # text set A: held-out test
    setB = token_blocks(tok, "train", 128)[64:104]                      # text set B: train blocks unused by fits
    ids = {s: np.array(b[:, :-1]).reshape(-1) for s, b in (("A", setA), ("B", setB))}
    X = capture(model, cal, [L])[L].astype(np.float64)
    XA, XB = [capture(model, b, [L])[L].astype(np.float64) for b in (setA, setB)]
    valid = {s: np.arange(len(ids[s])) % SEQ != 0 for s in ids}          # first position excluded
    keep_c = np.arange(X.shape[0]) % SEQ != 0
    X = X[keep_c]

    G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
    P = load_w(L, K, tag)
    H_old = {"C": hidden(X, G, U), "A": hidden(XA, G, U), "B": hidden(XB, G, U)}
    H_new = {"C": hidden(X, P[0], P[1]), "A": hidden(XA, P[0], P[1]), "B": hidden(XB, P[0], P[1])}

    # Causal effects on text set A: per-token loss change when one unit is zeroed.
    orig = model.model.layers[L].mlp
    base_old = token_ce(model, setA)
    small = SmallMLP(P)
    model.model.layers[L].mlp = small
    base_new = token_ce(model, setA)

    def ablate_new(j):
        model.model.layers[L].mlp = Ablate(small, j)
        d = token_ce(model, setA) - base_new
        model.model.layers[L].mlp = small
        return d

    def ablate_old(j):
        model.model.layers[L].mlp = Ablate(orig, j)
        d = token_ce(model, setA) - base_old
        model.model.layers[L].mlp = small
        return d

    size = np.linalg.norm(H_new["A"], axis=0) * np.linalg.norm(P[2], axis=0)
    order = np.argsort(-size)
    rec = {"layer": L, "K": K, "tag": tag, "base_ce_small_layer": float(base_new.mean()),
           "base_ce_original": float(base_old.mean()), "neurons": []}
    dossier = [f"# Layer {L}, small layer K={K} ({tag}): units to describe\n",
               "For each unit: activations at both poles on two text sets (the token at [[ ]] is where it fires),",
               "what it writes (direct effect on next-word scores), and which tokens get worse when it is zeroed.",
               "Units named T* are the trained small-layer units. Units named O* are original-layer units that carry",
               "part of the same signal.\n"]
    detect, key = [f"# Layer {L} K={K} ({tag}): blind detection set\n"], {}

    def describe(name, actA, actB, d_col, abl, valid_mask):
        pa, na = poles(tok, ids["A"], actA, valid["A"])
        pb, nb = poles(tok, ids["B"], actB, valid["B"], n=5)
        up, dn = dla(d_col, normw, emb, tok)
        hurt = np.argsort(-np.where(valid["A"], abl, -np.inf))[:6]
        lines = [f"\n## {name}\n", "Positive pole, set A:"] + [f"  - {v:+.2f} {c}" for v, c in pa]
        lines += ["Positive pole, set B:"] + [f"  - {v:+.2f} {c}" for v, c in pb]
        lines += ["Negative pole, set A:"] + [f"  - {v:+.2f} {c}" for v, c in na]
        lines += ["Negative pole, set B:"] + [f"  - {v:+.2f} {c}" for v, c in nb]
        lines += [f"Writes up (raises the score of): {up}", f"Writes down (lowers the score of): {dn}",
                  f"Zeroing it: mean loss change {abl[valid['A']].mean():+.4f} nats. Tokens hurt most:"]
        lines += [f"  - {abl[p]:+.2f} {ctx(tok, ids['A'], p)}" for p in hurt]
        return lines

    def detection_set(name, actA):
        """delphi-style detection: 6 held-out top-10% positive examples not shown in the dossier, 6 random others."""
        v = np.where(valid["A"], actA, np.nan)
        thr = np.nanquantile(v, 0.99)
        top = np.where(v >= thr)[0]
        top = top[np.argsort(-actA[top])][8:]                     # skip the ones shown in the dossier
        pos = rng.choice(top, 6, replace=False)
        neg = rng.choice(np.where(valid["A"] & (np.nan_to_num(v) < np.nanquantile(v, 0.5)))[0], 6, replace=False)
        items = [(int(p), True) for p in pos] + [(int(p), False) for p in neg]
        rng.shuffle(items)
        detect.append(f"\n## {name}\n")
        key[name] = []
        for i, (p, lab) in enumerate(items):
            detect.append(f"{i + 1}. {ctx(tok, ids['A'], p)}")
            key[name].append(lab)

    cover = None
    if args.children_from == "cover":
        rows = json.load(open(os.path.join(ROOT, "results", "clean-r1cover.json")))["rows"]
        cover = next(r for r in rows if r["layer"] == L and r["K"] == K and r["tag"] == tag)["units"]
        n_old = G.shape[0]
    for t in order:
        r2, sel, coef = omp(H_old["C"], H_new["C"][:, t], H_old["A"], H_new["A"][:, t], smax=10, report=(1, 3, 10))
        kids = sel[: args.children]
        if cover is not None:
            # Children from the coverage test: the parent pole with the most children, first in cover order.
            best = max((u for u in cover if u["unit"] == int(t)), key=lambda u: u["n_children"])
            ps = 1 if best["pole"] == "+" else -1
            kids = [c % n_old for c in best["cover_order"][: args.children + 1]]
            coef = [ps * (1 if c < n_old else -1) for c in best["cover_order"][: args.children + 1]]
            dossier.append(f"\n(T{t}: parent pole {best['pole']}, {best['n_children']} coverage children, "
                           f"children explain {best['up_cover']:.0%} of its top tokens)")
        abl_t = ablate_new(int(t))
        tname = f"T{t}"
        dossier += describe(tname, H_new["A"][:, t], H_new["B"][:, t], P[2][:, t], abl_t, valid)
        detection_set(tname, H_new["A"][:, t])
        n = {"unit": int(t), "share": float(size[t] / size.sum()), "act_r2_1_3_10": r2,
             "ablation_mean": float(abl_t[valid["A"]].mean()), "children": []}
        for j, cf in zip(kids, coef):
            abl_j = ablate_old(int(j))
            m = valid["A"]
            ha, hn = H_old["A"][:, j], H_new["A"][:, t]
            s = np.sign(cf)
            # The child's top 50 tokens by its contribution to the parent's signal (coef x activation).
            topk = np.argsort(-np.where(m, s * ha, -np.inf))[:50]
            # Percentile of the parent's own activation on those tokens. 1 = the parent fires fully there too.
            ref = np.sort(hn[m])
            pct = float(np.mean(np.searchsorted(ref, hn[topk]) / len(ref)))
            dcos = float(D[:, j] @ P[2][:, t] / (np.linalg.norm(D[:, j]) * np.linalg.norm(P[2][:, t]) + 1e-12))
            n["children"].append({"old": int(j), "coef": float(cf), "out_cos": dcos,
                                  "ablation_mean": float(abl_j[m].mean()),
                                  "ablation_corr": float(np.corrcoef(abl_t[m], abl_j[m])[0, 1]),
                                  "parent_pct_on_child_top50": pct})
            oname = f"O{j} (carries part of T{t})"
            dossier += describe(oname, ha, H_old["B"][:, j], D[:, j], abl_j, valid)
            detection_set(f"O{j}", ha)
        rec["neurons"].append(n)
        print(f"T{t:3d} share {n['share']:.2f} R2 1/3/10 {r2[1]:.2f}/{r2[3]:.2f}/{r2[10]:.2f} abl {n['ablation_mean']:+.4f} | "
              + " ".join(f"O{c['old']}: cos {c['out_cos']:+.2f} ablcorr {c['ablation_corr']:+.2f} pct {c['parent_pct_on_child_top50']:.2f}"
                         for c in n["children"]), flush=True)
    model.model.layers[L].mlp = orig

    stem = f"L{L}_K{K}_{tag}" + ("_cover" if cover is not None else "")
    open(os.path.join(OUT, f"{stem}_dossier.md"), "w").write("\n".join(dossier) + "\n")
    open(os.path.join(OUT, f"{stem}_detect.md"), "w").write("\n".join(detect) + "\n")
    json.dump(key, open(os.path.join(OUT, f"{stem}_key.json"), "w"))
    json.dump(rec, open(os.path.join(ROOT, "results", f"clean-r1read-{stem}.json"), "w"), indent=1)
    print("wrote", stem)


if __name__ == "__main__":
    main()
