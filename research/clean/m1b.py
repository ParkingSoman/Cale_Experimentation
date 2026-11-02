"""M1b: merge only neurons whose outputs look the same to the NEXT layer (shared reader).

Pairs (and triples) are picked by |cosine| of the output column after the next layer's
reading matrices (attention q/k/v and MLP gate/up, with the RMSNorm scales folded in).
Same rules and same measures as M1, plus the error as the next layer sees it.

Run: python3 research/clean/m1b.py [--per 30] [--layers 3,15,28]
Writes results/clean-m1b.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, Capture, silu  # noqa: E402
from m1 import run_rules, measure, oracle_error, apply, relerr, RULES  # noqa: E402


def reader_matrix(model, L):
    nxt = model.model.layers[L + 1]
    w1 = np.array(nxt.input_layernorm.weight).astype(np.float64)
    w2 = np.array(nxt.post_attention_layernorm.weight).astype(np.float64)
    att = np.vstack([np.array(nxt.self_attn.q_proj.weight), np.array(nxt.self_attn.k_proj.weight),
                     np.array(nxt.self_attn.v_proj.weight)]).astype(np.float64) * w1[None, :]
    mlp = np.vstack([np.array(nxt.mlp.gate_proj.weight), np.array(nxt.mlp.up_proj.weight)]).astype(np.float64) * w2[None, :]
    return np.vstack([att, mlp])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--layers", default="3,15,28")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    t0 = time.time()
    rng = np.random.default_rng(0)
    model, tok = load_model()
    W = Weights(model)
    cal, held = token_blocks(tok, "train", 32), token_blocks(tok, "test", 100)[:40]
    th = np.array(held[:, :-1]).reshape(-1)

    def cap(blocks):
        caps = {L: Capture(model.model.layers[L].mlp) for L in layers}
        for L in layers:
            model.model.layers[L].mlp = caps[L]
        for i in range(0, blocks.shape[0], 4):
            mx.eval(model(blocks[i:i + 4, :-1]))
        for L in layers:
            model.model.layers[L].mlp = caps[L].inner
        return {L: np.concatenate(caps[L].x, 0).astype(np.float64) for L in layers}

    XC, XH = cap(cal), cap(held)
    res = {"thresh": args.thresh, "records": []}
    for L in layers:
        Xc, Xh = XC[L], XH[L]
        G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        R = reader_matrix(model, L)
        Dr = R @ D
        A = Dr / (np.linalg.norm(Dr, axis=0, keepdims=True) + 1e-9)
        C = A.T @ A
        np.fill_diagonal(C, 0.0)
        hc = silu(Xc @ G.T) * (Xc @ U.T)
        contrib = np.linalg.norm(hc, axis=0) * np.linalg.norm(D, axis=0)
        active = contrib >= np.median(contrib)
        Ca = np.where(active[None, :] & active[:, None], np.abs(C), 0.0)
        pairs = [(i, int(Ca[i].argmax())) for i in np.where(Ca.max(1) > args.thresh)[0]]
        pairs = sorted({tuple(sorted(p)) for p in pairs})
        pairs = [p for p in pairs if min(contrib[list(p)]) >= 0.25 * max(contrib[list(p)])]
        print(f"layer {L}: {len(pairs)} aligned active pairs with |cos|>{args.thresh} (read space) [{time.time() - t0:.0f}s]", flush=True)
        if not pairs:
            continue
        sel = [pairs[i] for i in rng.permutation(len(pairs))[: args.per]]
        # control: random active pairs of similar size, same layer
        act_idx = np.where(active)[0]
        ctrl = [tuple(rng.choice(act_idx, 2, replace=False)) for _ in range(len(sel))]
        for kind, grp in (("aligned", sel), ("random", ctrl)):
            for p in grp:
                idx = np.array(p)
                Gc, Uc, Dc = G[idx], U[idx], D[:, idx].T
                out = run_rules(Xc, Xh, Gc, Uc, Dc, 1)
                Y = (silu(Xh @ Gc.T) * (Xh @ Uc.T)) @ Dc
                rec = {"layer": L, "kind": kind, "idx": idx.tolist(), "read_cos": float(C[p[0], p[1]]),
                       "oracle": oracle_error(Xc, Xh, Gc, Uc, Dc, 1), "rules": {}}
                for k, (P, secs) in out.items():
                    m = measure(Xh, th, Gc, Uc, Dc, P)
                    m["err_read"] = relerr(Y @ R.T, apply(Xh, P) @ R.T)
                    m["secs"] = secs
                    rec["rules"][k] = m
                res["records"].append(rec)
            rs = [r for r in res["records"] if r["layer"] == L and r["kind"] == kind]
            print(f"L{L} {kind:8s} n={len(rs)} median held error  "
                  + " ".join(f"{k}={np.median([r['rules'][k]['err'] for r in rs]):.3f}" for k in RULES if k in rs[0]["rules"])
                  + f"  oracle={np.median([r['oracle'] for r in rs]):.3f}  [{time.time() - t0:.0f}s]", flush=True)

    print("\nALL LAYERS. Median error; ratio = rule / H0 (below 1 = rule beats the cut)")
    agg = {}
    for kind in ("aligned", "random"):
        rs = [r for r in res["records"] if r["kind"] == kind]
        if not rs:
            continue
        agg[kind] = {"n": len(rs), "oracle": float(np.median([r["oracle"] for r in rs])),
                     "read_cos": float(np.mean([abs(r["read_cos"]) for r in rs]))}
        for k in RULES:
            if k not in rs[0]["rules"]:
                continue
            ratio = np.array([r["rules"][k]["err"] / (r["rules"]["H0"]["err"] + 1e-12) for r in rs])
            ratio_read = np.array([r["rules"][k]["err_read"] / (r["rules"]["H0"]["err_read"] + 1e-12) for r in rs])
            boot = [np.median(ratio[rng.integers(0, len(ratio), len(ratio))]) for _ in range(300)]
            agg[kind][k] = {"err": float(np.median([r["rules"][k]["err"] for r in rs])),
                            "err_read": float(np.median([r["rules"][k]["err_read"] for r in rs])),
                            "ratio": float(np.median(ratio)), "ratio_noise": float(np.std(boot)),
                            "ratio_read": float(np.median(ratio_read)), "win_share": float((ratio < 1).mean()),
                            "survival": float(np.mean([r["rules"][k]["survival"] for r in rs])),
                            "breadth_ratio": float(np.mean([r["rules"][k]["breadth_ratio"] for r in rs]))}
        print(f"\n{kind} pairs (n={agg[kind]['n']}, mean |read cos| {agg[kind]['read_cos']:.2f}, oracle rank-1 error {agg[kind]['oracle']:.3f})")
        print(f"{'rule':6s}{'err':>8s}{'err_read':>10s}{'ratio':>8s}{'noise':>8s}{'ratio_rd':>10s}{'wins':>7s}{'surv':>7s}{'breadth':>9s}")
        for k in RULES:
            if k in agg[kind]:
                a = agg[kind][k]
                print(f"{k:6s}{a['err']:8.3f}{a['err_read']:10.3f}{a['ratio']:8.2f}{a['ratio_noise']:8.2f}"
                      f"{a['ratio_read']:10.2f}{a['win_share']:7.2f}{a['survival']:7.2f}{a['breadth_ratio']:9.2f}")
    res["aggregate"] = agg
    out_path = os.path.join(ROOT, "results", "clean-m1b.json")
    json.dump(res, open(out_path, "w"), indent=1)
    print("wrote", out_path, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
