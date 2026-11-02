"""Blind semantic test: is a covering parent a BROADER CATEGORY of its children (coarse), or a grab bag (polysemantic)?

Builds, for one layer and one axis:
  units.md   every unit pole (parents, real children, decoy children) with anonymous ids, shuffled. Material on the
             layer's axis: input = long contexts where it fires (two text sets). output = words it promotes when
             steered, plus a few contexts.
  detect.md  per parent: 10 held-out items (5 true, 5 not), shuffled. input: contexts. output: words.
  key.json   which unit is which, families (parent, real children, decoy children), detection answers.
Agents then (1) describe every unit, (2) score parent descriptions blind on detect.md, (3) judge families blind.

  python3 research/clean/r1family.py --layer 3 --axis input
  python3 research/clean/r1family.py --layer 28 --axis output
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, load_w  # noqa: E402

OUT = os.path.join(HERE, "r1family")
CTX = 40
N_KIDS = 4


def ctx(tok, ids, p):
    a = max(0, p - CTX)
    return (tok.decode(ids[a:p].tolist()) + "[[" + tok.decode([int(ids[p])]) + "]]").replace("\n", " ")


def dedup_top(act, ids, n, skip=0):
    """Top positions by activation, at most 2 per distinct target token, so one word cannot fill the list."""
    seen, out = {}, []
    for p in np.argsort(-act):
        t = int(ids[p])
        if seen.get(t, 0) >= 2:
            continue
        seen[t] = seen.get(t, 0) + 1
        out.append(int(p))
        if len(out) >= n + skip:
            break
    return out[skip:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--axis", choices=["input", "output", "virtual"], required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tag", default="local_rand")
    ap.add_argument("--z", action="store_true", help="output axis with item scaling (clean-r1out-z.json)")
    args = ap.parse_args()
    L, K, tag, axis = args.layer, args.K, args.tag, args.axis
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(L * 10 + (axis == "output"))

    model, tok = load_model()
    W = Weights(model)
    setA = token_blocks(tok, "test", 100)[:40]
    setB = token_blocks(tok, "train", 128)[64:104]
    XA, XB = [capture(model, b, [L])[L].astype(np.float64) for b in (setA, setB)]
    ids = {s: np.array(b[:, :-1]).reshape(-1) for s, b in (("A", setA), ("B", setB))}
    vA, vB = [np.arange(len(ids[s])) % SEQ != 0 for s in ("A", "B")]
    G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
    P = load_w(L, K, tag)
    n_old = G.shape[0]
    old = {s: hidden(X, G, U) for s, X in (("A", XA), ("B", XB))}
    new = {s: hidden(X, P[0], P[1]) for s, X in (("A", XA), ("B", XB))}

    src = "clean-r1cover.json" if axis == "input" else ("clean-r1out-z.json" if args.z else "clean-r1out.json")
    if axis == "virtual":
        src = "clean-r1virt.json"
    row = next(r for r in json.load(open(os.path.join(ROOT, "results", src)))["rows"]
               if r["layer"] == L and r["K"] == K and r["tag"] == tag)
    if axis == "virtual":
        # Material: words each unit promotes directly (final norm + tied unembedding), scaled over the old units.
        E = np.array(model.model.embed_tokens.weight, dtype=np.float64) * np.array(model.model.norm.weight, dtype=np.float64)[None, :]
        C = E @ np.concatenate([D, -D], 1)
        zsd = C.std(1, keepdims=True) + 1e-9
        C = C / zsd
        Pr = (E @ np.concatenate([P[2], -P[2]], 1)) / zsd
    if axis == "output":
        from r1out import profiles, pole_vectors, SEQ_S
        C = np.load(os.path.join(HERE, "r1out_cache", f"L{L}_old.npy"))
        held = token_blocks(tok, "test", 100)[:40]
        sb = [np.array(held[i, :SEQ_S]) for i in range(2)]
        Xv = XA[vA]
        Pr = profiles(model, L, sb, pole_vectors(hidden(Xv, P[0], P[1]), P[2]))
        if args.z:
            zmu, zsd = C.mean(1, keepdims=True), C.std(1, keepdims=True) + 1e-9
            C, Pr = (C - zmu) / zsd, (Pr - zmu) / zsd

    # Pick each trained unit's best pole. Keep units that pass (2+ children, up-cover >= 0.5).
    best = {}
    for u in row["units"]:
        b = best.get(u["unit"])
        if b is None or (u["n_children"], u["up_cover"]) > (b["n_children"], b["up_cover"]):
            best[u["unit"]] = u
    fams = [b for b in best.values() if b["n_children"] >= 2 and b["up_cover"] >= 0.5]

    units = {}       # anon id -> (kind, which, sign)

    def add(kind, which, sign):
        uid = f"U{len(units) + 1:03d}"
        units[uid] = {"kind": kind, "which": which, "sign": sign}
        return uid

    # Parents may share children (one old pole gets one anonymous id). Decoys never come from any real child.
    families = []
    all_kids = {c % n_old for b in fams for c in b["children"]}
    pool = [c for c in range(2 * n_old) if c % n_old not in all_kids]
    pole_ids = {}

    def old_id(kind, c):
        if c not in pole_ids:
            pole_ids[c] = add(kind, c % n_old, 1 if c < n_old else -1)
        return pole_ids[c]

    for b in fams:
        t, s = b["unit"], (1 if b["pole"] == "+" else -1)
        kids = b["children"][:N_KIDS]
        decoys = [int(c) for c in rng.choice(pool, len(kids), replace=False)]
        pool = [c for c in pool if c not in decoys]
        families.append({"parent": add("parent", int(t), s), "trained_unit": int(t), "pole": b["pole"],
                         "children": [old_id("child", c) for c in kids],
                         "decoys": [old_id("decoy", c) for c in decoys]})

    def act(u, s_):
        a = new[s_][:, u["which"]] if u["kind"] == "parent" else old[s_][:, u["which"]]
        return u["sign"] * a

    def words(u, n, skip=0):
        if u["kind"] == "parent":
            col = Pr[:, u["which"] + (0 if u["sign"] > 0 else K)]
        else:
            col = C[:, u["which"] + (0 if u["sign"] > 0 else n_old)]
        return [int(i) for i in np.argsort(-col)[skip:skip + n]], col

    order = list(units)
    rng.shuffle(order)
    md = [f"# Units to describe (layer {L}, axis: {axis})\n",
          "Each unit is one direction of one neuron. Describe what it does, as a function, from the material.",
          "[[ ]] marks the token where the unit is strongly active." +
          (" 'Promotes' = words whose next-word probability it raises most." if axis in ("output", "virtual") else "") + "\n"]
    for uid in order:
        u = units[uid]
        a, b_ = np.where(vA, act(u, "A"), -np.inf), np.where(vB, act(u, "B"), -np.inf)
        md.append(f"\n## {uid}\n")
        if axis in ("output", "virtual"):
            w, _ = words(u, 30)
            md.append("Promotes: " + ", ".join(repr(tok.decode([i])) for i in w))
            md.append("Active in (set A):")
            md += [f"  - {ctx(tok, ids['A'], p)}" for p in dedup_top(a, ids["A"], 5)]
        else:
            md.append("Active in (set A):")
            md += [f"  - {ctx(tok, ids['A'], p)}" for p in dedup_top(a, ids["A"], 10)]
            md.append("Active in (set B):")
            md += [f"  - {ctx(tok, ids['B'], p)}" for p in dedup_top(b_, ids["B"], 5)]

    det, det_key = [f"# Detection set (layer {L}, axis {axis})\n",
                    "For each unit: given its description, pick the items where it is active (exactly 5 of 10).\n"], {}
    for f in families:
        u = units[f["parent"]]
        det.append(f"\n## {f['parent']}\n")
        if axis in ("output", "virtual"):
            w, col = words(u, 5, skip=30)                           # promoted words not shown in units.md
            low = [int(i) for i in rng.choice(np.argsort(col)[: len(col) // 2], 5, replace=False)]
            items = [(tok.decode([i]), True) for i in w] + [(tok.decode([i]), False) for i in low]
        else:
            a = np.where(vA, act(u, "A"), -np.inf)
            pos = dedup_top(a, ids["A"], 5, skip=10)                # active examples not shown in units.md
            neg = rng.choice(np.where(vA & (act(u, "A") < np.quantile(act(u, "A")[vA], 0.5)))[0], 5, replace=False)
            items = [(ctx(tok, ids["A"], p), True) for p in pos] + [(ctx(tok, ids["A"], int(p)), False) for p in neg]
        rng.shuffle(items)
        det_key[f["parent"]] = [lab for _, lab in items]
        det += [f"{i + 1}. {repr(x) if axis in ('output', 'virtual') else x}" for i, (x, _) in enumerate(items)]

    stem = f"L{L}_{axis}" + ("z" if args.z else "") + (f"_K{K}" if K != 10 else "")
    open(os.path.join(OUT, f"{stem}_units.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(OUT, f"{stem}_detect.md"), "w").write("\n".join(det) + "\n")
    json.dump({"layer": L, "axis": axis, "K": K, "tag": tag, "units": units, "families": families,
               "detect_key": det_key}, open(os.path.join(OUT, f"{stem}_key.json"), "w"), indent=1)
    print(f"{stem}: {len(families)} families, {len(units)} units")


if __name__ == "__main__":
    main()
