"""Z4: does cut vs merge work on query heads and on depth? No training.
Usage: python z4.py [heads|depth]   (no argument = both). Parts go to ../../results/tree-z4.json.
"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
from tree_lib import *
from tree_lib3 import *

part = sys.argv[1] if len(sys.argv) > 1 else "both"
OUT = "../../results/tree-z4.json"
t0 = time.time()
model, tok = load_model()
teacher, _ = load_model()
cal = token_blocks(tok, "train", 32)
held = token_blocks(tok, "test", 100)
base = eval_loss(model, held, teacher=teacher)
print("baseline", base, flush=True)
L = len(model.model.layers)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res["baseline"] = base
res.setdefault("results", [])
res["results"] = [r for r in res["results"] if part != "both" and r["axis"] != part]
rows = res["results"]


def add(axis, method, ls, size, seed, extra=None):
    r = eval_loss(model, held, teacher=teacher)
    r.update(axis=axis, method=method, ls=ls, size=size, seed=seed)
    if extra:
        r.update(extra)
    rows.append(r)
    print(f"{axis:6s} {method:12s} refit={'-' if ls is None else int(ls)} size={size:2d} seed={seed} "
          f"ce={r['ce']:.4f} (+{r['ce']-base['ce']:.3f}) kl={r['kl']:.4f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    return r


if part in ("heads", "both"):
    Q0 = [np.array(l.self_attn.q_proj.weight) for l in model.model.layers]
    O0 = [np.array(l.self_attn.o_proj.weight) for l in model.model.layers]

    def set_qo(i, Q, O):
        a = model.model.layers[i].self_attn
        a.q_proj.weight = mx.array(Q.astype(np.float32))
        a.o_proj.weight = mx.array(O.astype(np.float32))

    def restore():
        for i in range(L):
            set_qo(i, Q0[i], O0[i])
        mx.eval(model.parameters())

    # grouping check: perturb k_proj rows of kv head 1 in layer 0, see which query heads change
    walk = Walk(model, cal[:4])
    A_ref, _ = walk.original()
    kp = model.model.layers[0].self_attn.k_proj
    w0 = kp.weight
    w = np.array(w0); w[HD:2 * HD] *= 1.5
    kp.weight = mx.array(w)
    A_new = Walk(model, cal[:4]).modified_A()
    kp.weight = w0
    changed = [h for h in range(NH) if np.abs(A_new[:, cols(h)] - A_ref[:, cols(h)]).max() > 1e-4]
    print("query heads that react to kv head 1:", changed, flush=True)
    assert changed == [3, 4, 5], changed
    del walk

    # identity checks
    set_qo(0, Q0[0], O0[0]); mx.eval(model.parameters())
    ca = eval_loss(model, held)["ce"]
    O = O0[5].copy(); O[:, cols(4)] = 0
    set_qo(5, Q0[5], O); mx.eval(model.parameters())
    cz = eval_loss(model, held)["ce"]
    set_qo(5, Q0[5], O0[5]); mx.eval(model.parameters())
    cb = eval_loss(model, held)["ce"]
    print(f"identity heads: rebuilt {ca:.4f} | one head zeroed (layer 5, head 4) {cz:.4f} | restored {cb:.4f} | baseline {base['ce']:.4f}")
    assert abs(ca - base["ce"]) < 1e-4 and abs(cb - base["ce"]) < 1e-4 and cz != cb
    res["identity_heads"] = dict(rebuilt=ca, zeroed=cz, restored=cb)

    # pass 0: per-layer head statistics on calibration text
    walk = Walk(model, cal)
    STATS = []
    for i in range(L):
        A, Y = walk.original()
        STATS.append(head_stats(A, O0[i]))
        walk.advance()
    del walk, A, Y
    print(f"stats done [{time.time()-t0:.0f}s]", flush=True)

    def sweep(method, size, ls, seed):
        rng = np.random.default_rng(seed)
        walk = Walk(model, cal)
        new, errs = [], []
        for i in range(L):
            A, Y = walk.original()
            act, cos = STATS[i]
            plan = head_plan(method, size, act, cos, np.random.default_rng((seed, i)))
            Q, O = apply_plan(Q0[i], O0[i], plan)
            if ls:
                set_qo(i, Q, O0[i]); mx.eval(model.parameters())
                A2 = walk.modified_A()                       # kept heads' outputs with the NEW q
                kc = kept_cols(plan)
                O = np.zeros_like(O0[i])
                O[:, kc] = ridge_cols(A2, kc, Y)             # target: ORIGINAL attention output
                err = rel_err(Y, A2 @ O.T)
                set_qo(i, Q0[i], O0[i]); mx.eval(model.parameters())
            else:
                err = None
            new.append((Q, O)); errs.append(err)
            walk.advance()
        for i, (Q, O) in enumerate(new):
            set_qo(i, Q, O)
        mx.eval(model.parameters())
        r = add("heads", method, ls, size, seed,
                dict(layer_fit_err_cal=None if not ls else [round(float(e), 4) for e in errs]))
        restore()
        return r

    for size in (6, 3):
        for method in ("cut", "merge", "random"):
            if method == "random" and size == 3:
                continue          # at 3 heads every group collapses to one head: random = merge
            for ls in (False, True):
                for seed in ((0, 1) if method == "random" else (0,)):
                    sweep(method, size, ls, seed)
    del Q0, O0, STATS

if part in ("depth", "both"):
    layers0 = list(model.model.layers)
    model.model.layers = list(layers0)
    c = eval_loss(model, held)["ce"]
    print(f"identity depth: rebuilt list {c:.4f} | baseline {base['ce']:.4f}")
    assert abs(c - base["ce"]) < 1e-4
    res["identity_depth"] = dict(rebuilt=c)

    def cal_ce(drop):
        model.model.layers = [l for j, l in enumerate(layers0) if j not in drop]
        return eval_loss(model, cal)["ce"]

    # cut: rank layers by calibration CE after removing that layer alone (never 0 or 29)
    single = {j: cal_ce({j}) for j in range(1, L - 1)}
    order = sorted(single, key=single.get)
    print("calibration CE with one layer removed (best 8):", [(j, round(single[j], 3)) for j in order[:8]], flush=True)
    res["depth_single_removal_cal_ce"] = single
    for size in (28, 26, 22):
        drop = set(order[:L - size])
        model.model.layers = [l for j, l in enumerate(layers0) if j not in drop]
        add("depth", "cut", None, size, 0, dict(dropped=sorted(drop)))

    def merged_model(pairs):
        pm = {a: b for a, b in pairs}
        used = {a for a, b in pairs} | {b for a, b in pairs}
        out = []
        for j, l in enumerate(layers0):
            if j in pm:
                out.append(average_layers(l, layers0[pm[j]]))
            elif j not in used:
                out.append(l)
        return out

    dist = {j: layer_distance(layers0[j], layers0[j + 1]) for j in range(1, L - 2)}   # pairs (j, j+1), j+1 <= 28
    res["depth_pair_distance"] = dist
    chosen, taken = [], set()
    for j in sorted(dist, key=dist.get):
        if j in taken or j + 1 in taken:
            continue
        chosen.append((j, j + 1)); taken |= {j, j + 1}
    print("closest adjacent pairs:", [(p, round(dist[p[0]], 3)) for p in chosen[:8]], flush=True)
    for size in (28, 26, 22):
        pairs = chosen[:L - size]
        model.model.layers = merged_model(pairs)
        add("depth", "merge", None, size, 0, dict(pairs=pairs))
    for size in (28, 26, 22):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            pairs, taken = [], set()
            while len(pairs) < L - size:
                j = int(rng.integers(1, L - 2))
                if j in taken or j + 1 in taken:
                    continue
                pairs.append((j, j + 1)); taken |= {j, j + 1}
            model.model.layers = merged_model(sorted(pairs))
            add("depth", "random", None, size, seed, dict(pairs=sorted(pairs)))
    model.model.layers = list(layers0)

print()
print(f"baseline CE {base['ce']:.4f}")
print(f"{'axis':6s} {'method':8s} {'refit':5s} {'size':>4s} {'seed':>4s} {'CE':>8s} {'+CE':>7s} {'KL':>7s}")
for r in sorted(rows, key=lambda r: (r["axis"], -r["size"], r["method"], str(r["ls"]), r["seed"])):
    print(f"{r['axis']:6s} {r['method']:8s} {'-' if r['ls'] is None else ('yes' if r['ls'] else 'no'):5s} "
          f"{r['size']:4d} {r['seed']:4d} {r['ce']:8.4f} {r['ce']-base['ce']:7.3f} {r['kl']:7.4f}")
json.dump(res, open(OUT, "w"), indent=1)
print("saved results/tree-z4.json")
