"""TOY RECIPE, part 2: what ARE the shared ingredients? (dev, L15)

Neurons: top contexts on calibration text, on the pole they take at the group's firing positions.
Heads: at the group's firing positions, which earlier token the head reads most (attention weight), with the offset.
Also a specificity score: an ingredient's shared weight in this group minus its largest weight in another group,
so generic ingredients (used by every group) drop out.

  python3 research/clean/coarse/toy_label.py
Writes research/clean/coarse/toy_label.md.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from toy_recipe import (L, GROUPS, ROOT, run_grab, load_model, token_blocks, SEQ)  # noqa: E402

N_TOP = 8


def ctx_str(tok, ids, p, before=10):
    b, t = divmod(p, SEQ)
    a = max(0, t - before)
    return (tok.decode(ids[b, a:t].tolist()) + " [[" + tok.decode([int(ids[b, t])]) + "]]").replace("\n", " ")


def main():
    rep = json.load(open(os.path.join(ROOT, "results", "clean-toy-recipe.json")))
    z = np.load(os.path.join(HERE, "toy_recipe_cache.npz"))
    R, hot = z["R"], z["hot"]
    names = [(g, s) for g, lst in GROUPS.items() for s in lst]
    grp = np.array([g for g, _ in names])
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    gnames = [g for g in GROUPS if g != "decoy"]
    shared = {g: Rn[grp == g].mean(0) for g in gnames}
    labels = ["embed"] + [f"head {l}.{h}" for l in range(L + 1) for h in range(9)] + \
             [f"neuron {l}.{n}" for l in range(L) for n in range(1536)]
    pick = {}
    for g in gnames:
        others = np.max(np.stack([np.abs(shared[o]) for o in gnames if o != g]), 0)
        spec = shared[g] - others
        top_shared = [labels.index(x[0]) for x in rep["groups"][g]["shared_top"][:8]]
        top_spec = list(np.argsort(-spec)[:8])
        pick[g] = {"shared": top_shared, "specific": top_spec, "spec": spec}
    need_n = defaultdict(set)
    need_h = defaultdict(set)
    for g in gnames:
        for k in pick[g]["shared"] + pick[g]["specific"]:
            lab = labels[k]
            if lab.startswith("neuron"):
                l, n = map(int, lab.split()[1].split(".")); need_n[l].add(n)
            elif lab.startswith("head"):
                l, h = map(int, lab.split()[1].split(".")); need_h[l].add(h)

    model, tok = load_model()
    m = model.model
    cal = token_blocks(tok, "train", 64)
    ids = np.array(cal[:, :-1])
    gpos = {g: np.concatenate([hot[c] for c in np.where(grp == g)[0]]) for g in gnames}
    acts = {(l, n): np.zeros(ids.size, dtype=np.float32) for l in need_n for n in need_n[l]}
    heads = defaultdict(list)   # (g, l, h) -> list of (offset, src_pos, weight)

    def fn(i, gr):
        lo = i * SEQ
        nb = gr[("res", L)].shape[0]
        for l, ns in need_n.items():
            hcur = gr[("h", l)].reshape(-1, 1536)
            for n in ns:
                acts[(l, n)][lo:lo + hcur.shape[0]] = hcur[:, n]
        for l, hs in need_h.items():
            xn = mx.array(gr[("ain", l)])
            at = m.layers[l].self_attn
            q = at.rope(at.q_proj(xn).reshape(nb, SEQ, 9, 64).transpose(0, 2, 1, 3))
            k = at.rope(at.k_proj(xn).reshape(nb, SEQ, 3, 64).transpose(0, 2, 1, 3))
            q, k = np.array(q), np.array(k)
            for g in gnames:
                for p in gpos[g]:
                    b, t = divmod(int(p), SEQ)
                    if not (i <= b < i + nb):
                        continue
                    for h in hs:
                        s = k[b - i, h // 3, :t + 1] @ q[b - i, h, t] * at.scale
                        w = np.exp(s - s.max()); w /= w.sum()
                        j = int(np.argmax(w))
                        heads[(g, l, h)].append((t - j, b * SEQ + j, float(w[j])))
    run_grab(model, cal, fn)

    md = ["# What the shared ingredients are (L15 toy, dev)\n",
          "Neuron lines: top contexts on the pole the neuron takes when the group fires ([[x]] = the firing token).",
          "Head lines: the token the head reads most when the group fires. offset 0 = the current token.\n"]

    def describe(g, k):
        lab = labels[k]
        if lab.startswith("neuron"):
            l, n = map(int, lab.split()[1].split("."))
            a = acts[(l, n)]
            sg = 1.0 if a[gpos[g]].mean() >= 0 else -1.0
            valid = np.arange(a.size) % SEQ >= 2
            order = np.argsort(-np.where(valid, sg * a, -np.inf))
            seen, out = Counter(), []
            for p in order:
                tkn = ids.flat[p]
                if seen[tkn] < 2:
                    out.append(ctx_str(tok, ids, p)); seen[tkn] += 1
                if len(out) == 5:
                    break
            z_ = (a[gpos[g]].mean() - a[valid].mean()) / (a[valid].std() + 1e-9)
            return f"**{lab}** ({'+' if sg > 0 else '-'} pole; at group positions {z_:+.1f} sd)<br>" + "<br>".join(
                f"`{o[-90:]}`" for o in out)
        if lab.startswith("head"):
            l, h = map(int, lab.split()[1].split("."))
            rows = heads[(g, l, h)]
            offs = Counter(min(o, 6) if o < 6 else ("6-20" if o <= 20 else ">20") for o, _, _ in rows)
            srcs = Counter(tok.decode([int(ids.flat[s])]).strip() or repr(tok.decode([int(ids.flat[s])]))
                           for _, s, _ in rows)
            wmean = np.mean([w for _, _, w in rows])
            ex = [ctx_str(tok, ids, s, before=6) for _, s, _ in rows[:3]]
            return (f"**{lab}** reads offset {dict(offs.most_common(4))}, mean weight {wmean:.2f}; source tokens "
                    f"{dict(srcs.most_common(6))}<br>" + "<br>".join(f"`{e[-70:]}`" for e in ex))
        return f"**{lab}**"

    for g in gnames:
        mem = [names[c][1] for c in np.where(grp == g)[0]]
        md.append(f"\n## {g} (children {', '.join(mem)})\n")
        md.append("### Top shared ingredients (raw)\n")
        md += [f"- weight {Rn[grp == g].mean(0)[k]:.3f}: " + describe(g, k) for k in pick[g]["shared"]]
        md.append("\n### Top SPECIFIC ingredients (shared here minus the most any other group uses it)\n")
        md += [f"- specificity {pick[g]['spec'][k]:.3f}: " + describe(g, k) for k in pick[g]["specific"]]
    open(os.path.join(HERE, "toy_label.md"), "w").write("\n".join(md) + "\n")
    print("wrote toy_label.md")


if __name__ == "__main__":
    main()
