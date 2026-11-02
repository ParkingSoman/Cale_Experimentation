"""R1: train a small MLP layer (K neurons), then ask if its neurons are mixes of the old neurons.

Plan: ~/.claude/plans/okay-we-ve-been-experimenting-vectorized-salamander.md

  python3 research/clean/r1.py train   --layers 3,15,28 --ks 192,48,10       # cut, local fits, one-layer CE
  python3 research/clean/r1.py global  --layers 15 --ks 48 --minutes 20      # frozen rest, KL to the original
  python3 research/clean/r1.py analyze --layers 3,15,28 --ks 192,48,10       # sparse regression, weights, labels
  python3 research/clean/r1.py check                                         # K=1536 from the original keeps CE

Writes results/clean-r1-train.json, clean-r1-global.json, clean-r1-analyze.json, clean-r1-labels-L*.md.
Trained weights go to research/clean/r1_weights/ (small). Imports tree_lib and m1 read-only.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, Capture, hidden, ls_down, rel_err  # noqa: E402

WDIR = os.path.join(HERE, "r1_weights")
RES = os.path.join(ROOT, "results")
N_CAL, N_HELD_X, N_HELD = 64, 40, 100


# ---------- data ----------

def capture(model, blocks, layers):
    caps = {L: Capture(model.model.layers[L].mlp) for L in layers}
    for L in layers:
        model.model.layers[L].mlp = caps[L]
    for i in range(0, blocks.shape[0], 4):
        mx.eval(model(blocks[i:i + 4, :-1]))
    for L in layers:
        model.model.layers[L].mlp = caps[L].inner
    return {L: np.concatenate(caps[L].x, 0) for L in layers}


def block_ce(model, blocks, batch=4):
    """Mean cross-entropy per block, so differences can be bootstrapped over blocks."""
    out = []
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        lg = model(b[:, :-1])
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(lg, b[:, 1:, None], axis=-1).squeeze(-1).mean(1)
        out.append(np.array(ce))
    return np.concatenate(out)


def boot_diff(a, b, n=1000, seed=0):
    """Mean of a-b and its bootstrap standard error over blocks (paired)."""
    d = a - b
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.std(bs))


# ---------- small MLP ----------

def mlp_out(X, P):
    G, U, D = P            # G, U: K x 576.  D: 576 x K (like down_proj.weight)
    return hidden(X, G, U) @ D.T


def train_local(Xc, Yc, init, steps=3000, lr=1e-3, batch=4096, seed=0):
    """Adam on (G, U, D) of the small MLP so its output matches the original MLP output. Local loss only."""
    rng = np.random.default_rng(seed)
    P = {k: mx.array(v.astype(np.float32)) for k, v in zip("GUD", init)}
    X, Y = mx.array(Xc.astype(np.float32)), mx.array(Yc.astype(np.float32))

    def loss(P, x, y):
        h = x @ P["G"].T
        r = y - ((h * mx.sigmoid(h)) * (x @ P["U"].T)) @ P["D"].T
        return (r * r).sum() / (y * y).sum()

    vg = mx.value_and_grad(loss)
    opt = optim.Adam(learning_rate=optim.cosine_decay(lr, steps, lr * 0.05))
    for t in range(steps):
        idx = mx.array(rng.integers(0, X.shape[0], batch))
        l, g = vg(P, X[idx], Y[idx])
        opt.update(P, g)
        mx.eval(P, opt.state)
    return tuple(np.array(P[k], dtype=np.float64) for k in "GUD")


def cut_init(Xc, Yc, G, U, D, K):
    """Keep the K neurons with the biggest output on calibration text, least-squares refit of the outputs."""
    h = hidden(Xc, G, U)
    score = np.linalg.norm(h, axis=0) * np.linalg.norm(D, axis=0)
    idx = np.sort(np.argsort(-score)[:K])
    Dn = ls_down(Xc, G[idx], U[idx], Yc).astype(np.float64)
    return (G[idx], U[idx], Dn), idx


def rand_init(G, U, D, K, seed=0):
    rng = np.random.default_rng(seed)
    sg, su, sd = G.std(), U.std(), D.std()
    return (rng.standard_normal((K, G.shape[1])) * sg, rng.standard_normal((K, U.shape[1])) * su,
            rng.standard_normal((D.shape[0], K)) * sd)


def set_small(model, L, P):
    """Swap layer L's MLP for one with K neurons."""
    m = model.model.layers[L].mlp
    m.gate_proj = nn.Linear(P[0].shape[1], P[0].shape[0], bias=False)
    m.up_proj = nn.Linear(P[0].shape[1], P[0].shape[0], bias=False)
    m.down_proj = nn.Linear(P[0].shape[0], P[0].shape[1], bias=False)
    m.gate_proj.weight = mx.array(P[0].astype(np.float32))
    m.up_proj.weight = mx.array(P[1].astype(np.float32))
    m.down_proj.weight = mx.array(P[2].astype(np.float32))
    mx.eval(m.parameters())


class SmallMLP(nn.Module):
    """A separate K-neuron SwiGLU MLP, so the original module stays intact as the teacher."""

    def __init__(self, P):
        super().__init__()
        K, d = P[0].shape
        self.gate_proj = nn.Linear(d, K, bias=False)
        self.up_proj = nn.Linear(d, K, bias=False)
        self.down_proj = nn.Linear(K, d, bias=False)
        self.gate_proj.weight = mx.array(P[0].astype(np.float32))
        self.up_proj.weight = mx.array(P[1].astype(np.float32))
        self.down_proj.weight = mx.array(P[2].astype(np.float32))

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def wpath(L, K, tag):
    return os.path.join(WDIR, f"L{L}_K{K}_{tag}.npz")


def save_w(L, K, tag, P):
    os.makedirs(WDIR, exist_ok=True)
    np.savez(wpath(L, K, tag), G=P[0], U=P[1], D=P[2])


def load_w(L, K, tag):
    z = np.load(wpath(L, K, tag))
    return z["G"], z["U"], z["D"]


# ---------- train: cut and local fits ----------

def cmd_train(args):
    layers, ks = args.layers, args.ks
    t0 = time.time()
    model, tok = load_model()
    W = Weights(model)
    cal = token_blocks(tok, "train", N_CAL)
    heldx = token_blocks(tok, "test", N_HELD)[:N_HELD_X]
    held = token_blocks(tok, "test", N_HELD)
    base = block_ce(model, held)
    XC, XH = capture(model, cal, layers), capture(model, heldx, layers)
    print(f"base CE {base.mean():.4f}  captured {XC[layers[0]].shape} [{time.time() - t0:.0f}s]", flush=True)
    out = {"base_ce": float(base.mean()), "rows": []}
    for L in layers:
        G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        Xc, Xh = XC[L].astype(np.float64), XH[L].astype(np.float64)
        Yc, Yh = mlp_out(Xc, (G, U, D)), mlp_out(Xh, (G, U, D))
        for K in ks:
            row = {"layer": L, "K": K}
            # Best any K output directions can do with free per-token weights (not an MLP, a bound).
            _, V = np.linalg.eigh(Yc.T @ Yc)
            V = V[:, -K:]
            row["oracle_err"] = rel_err(Yh, Yh @ V @ V.T)
            Pc, idx = cut_init(Xc, Yc, G, U, D, K)
            save_w(L, K, "cut", Pc)
            cands = {"cut": Pc}
            for tag, init in (("local_cut", Pc), ("local_rand", rand_init(G, U, D, K))):
                t1 = time.time()
                P = train_local(Xc, Yc, init, steps=args.steps)
                row[f"{tag}_secs"] = time.time() - t1
                save_w(L, K, tag, P)
                cands[tag] = P
            ces = {}
            for tag, P in cands.items():
                row[f"{tag}_err"] = rel_err(Yh, mlp_out(Xh, P))
                set_small(model, L, P)
                ces[tag] = block_ce(model, held)
                row[f"{tag}_ce"] = float(ces[tag].mean())
                row[f"{tag}_dce"] = boot_diff(ces[tag], base)
            restore_mlp(model, W, L)
            # The gate: does training find something the cut does not? (mean, noise) of CE difference.
            row["gap_local_cut_vs_cut"] = boot_diff(ces["local_cut"], ces["cut"])
            row["gap_local_rand_vs_cut"] = boot_diff(ces["local_rand"], ces["cut"])
            print(f"L{L:2d} K={K:4d}  err oracle {row['oracle_err']:.3f} cut {row['cut_err']:.3f} "
                  f"local_cut {row['local_cut_err']:.3f} local_rand {row['local_rand_err']:.3f} | CE cut "
                  f"{row['cut_ce']:.3f} local_cut {row['local_cut_ce']:.3f} local_rand {row['local_rand_ce']:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            out["rows"].append(row)
            json.dump(out, open(os.path.join(RES, "clean-r1-train.json"), "w"), indent=1)
    print("wrote results/clean-r1-train.json")


def restore_mlp(model, W, L):
    """Put back an MLP with the original width and weights."""
    set_small(model, L, (W.G[L], W.U[L], W.D[L]))


# ---------- global: frozen rest, KL to the original ----------

def cmd_global(args):
    t0 = time.time()
    model, tok = load_model()
    W = Weights(model)
    cal = token_blocks(tok, "train", 256)
    held = token_blocks(tok, "test", N_HELD)
    base = block_ce(model, held)
    path = os.path.join(RES, "clean-r1-global.json")
    out = json.load(open(path)) if os.path.exists(path) else {"base_ce": float(base.mean()), "rows": []}
    rng = np.random.default_rng(0)
    for L in args.layers:
        for K in args.ks:
            start = "local_cut" if os.path.exists(wpath(L, K, "local_cut")) else "cut"
            P0 = load_w(L, K, start)
            orig = model.model.layers[L].mlp
            student = SmallMLP(P0)
            mx.eval(student.parameters())
            model.model.layers[L].mlp = student
            ce0 = block_ce(model, held)
            model.freeze()
            student.unfreeze()

            def loss_fn(model, x, tl):
                lg = model(x)
                lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                return (mx.exp(tl) * (tl - lg)).sum(-1).mean()

            vg = nn.value_and_grad(model, loss_fn)
            opt = optim.Adam(learning_rate=args.lr)
            steps, t1, kls = 0, time.time(), []
            while time.time() - t1 < args.minutes * 60:
                b = cal[mx.array(rng.integers(0, cal.shape[0], args.batch))][:, :-1]
                model.model.layers[L].mlp = orig
                tl = model(b)
                tl = mx.stop_gradient(tl - mx.logsumexp(tl, axis=-1, keepdims=True))
                mx.eval(tl)
                model.model.layers[L].mlp = student
                l, g = vg(model, b, tl)
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state)
                kls.append(l.item())
                steps += 1
                if steps % 100 == 0:
                    print(f"  L{L} K={K} step {steps} KL {np.mean(kls[-100:]):.4f} [{time.time() - t1:.0f}s]", flush=True)
            P = tuple(np.array(w, dtype=np.float64) for w in
                      (student.gate_proj.weight, student.up_proj.weight, student.down_proj.weight))
            save_w(L, K, "global", P)
            ce = block_ce(model, held)
            model.model.layers[L].mlp = orig
            model.unfreeze()
            row = {"layer": L, "K": K, "start": start, "steps": steps, "minutes": args.minutes,
                   "start_ce": float(ce0.mean()), "global_ce": float(ce.mean()),
                   "global_dce": boot_diff(ce, base), "gain_over_start": boot_diff(ce, ce0)}
            print(f"L{L:2d} K={K:4d}  start({start}) CE {ce0.mean():.3f} -> global {ce.mean():.3f} "
                  f"in {steps} steps [{time.time() - t0:.0f}s]", flush=True)
            out["rows"] = [r for r in out["rows"] if not (r["layer"] == L and r["K"] == K)] + [row]
            json.dump(out, open(path, "w"), indent=1)


# ---------- analyze ----------

_OMP_CACHE = {}


def _omp_prep(F_cal, F_held):
    """Standardized features and their Gram matrix, computed once per feature matrix."""
    key = (id(F_cal), id(F_held))
    if key not in _OMP_CACHE:
        _OMP_CACHE.clear()
        mu, sd = F_cal.mean(0), F_cal.std(0) + 1e-9
        A = ((F_cal - mu) / sd).astype(np.float32)
        Ah = ((F_held - mu) / sd).astype(np.float32)
        _OMP_CACHE[key] = (sd, A, Ah, (A.T @ A).astype(np.float64))
    return _OMP_CACHE[key]


def omp(F_cal, y_cal, F_held, y_held, smax=50, report=(1, 2, 3, 5, 10, 20, 50)):
    """Greedy sparse regression (orthogonal matching pursuit, Gram form). Fit on calibration, R2 on held-out.
    Returns {s: R2_held}, the chosen columns, and their coefficients at smax."""
    sd, A, Ah, Gm = _omp_prep(F_cal, F_held)
    ym = y_cal.mean()
    b = (A.T @ (y_cal - ym).astype(np.float32)).astype(np.float64)
    sel, r2, coef = [], {}, np.zeros(0)
    vh = ((y_held - y_held.mean()) ** 2).sum()
    for s in range(1, smax + 1):
        c = np.abs(b - (Gm[:, sel] @ coef if sel else 0))
        c[sel] = -1
        sel.append(int(np.argmax(c)))
        coef = np.linalg.solve(Gm[np.ix_(sel, sel)] + 1e-8 * np.eye(len(sel)), b[sel])
        if s in report:
            pred = Ah[:, sel].astype(np.float64) @ coef + ym
            r2[s] = float(1 - ((y_held - pred) ** 2).sum() / vh)
    return r2, sel, (coef / sd[sel]).tolist()


_RIDGE_CACHE = {}


def ridge_r2(F_cal, y_cal, F_held, y_held, lam=1e-3):
    """R2 on held-out text using ALL old neurons (dense). The factor is cached per feature matrix."""
    key = id(F_cal)
    if key not in _RIDGE_CACHE:
        _RIDGE_CACHE.clear()
        mu, sd = F_cal.mean(0), F_cal.std(0) + 1e-9
        A = (F_cal - mu) / sd
        Gm = A.T @ A
        Gm += np.eye(Gm.shape[0]) * lam * np.trace(Gm) / Gm.shape[0]
        _RIDGE_CACHE[key] = (mu, sd, A, np.linalg.cholesky(Gm))
    mu, sd, A, Lc = _RIDGE_CACHE[key]
    Ah = (F_held - mu) / sd
    ym = y_cal.mean()
    z = np.linalg.solve(Lc, A.T @ (y_cal - ym))
    w = np.linalg.solve(Lc.T, z)
    pred = Ah @ w + ym
    return float(1 - ((y_held - pred) ** 2).sum() / ((y_held - y_held.mean()) ** 2).sum())


def omp_vec(atoms, v, s):
    """Sparse fit of one weight vector by s atoms (rows). Returns R2 (no intercept) and support."""
    A = atoms / (np.linalg.norm(atoms, axis=1, keepdims=True) + 1e-9)
    res, sel = v.copy(), []
    for _ in range(s):
        c = np.abs(A @ res)
        c[sel] = -1
        sel.append(int(np.argmax(c)))
        coef, *_ = np.linalg.lstsq(A[sel].T, v, rcond=None)
        res = v - A[sel].T @ coef
    return float(1 - (res @ res) / (v @ v)), sel


def contexts(tok, ids, pos, n=12, before=6):
    out = []
    for p in pos[:n]:
        a = max(0, p - before)
        ctx = tok.decode(ids[a:p].tolist())
        out.append(f"{ctx[-40:]!r} [{tok.decode([int(ids[p])])!r}]")
    return out


def cmd_analyze(args):
    t0 = time.time()
    model, tok = load_model()
    W = Weights(model)
    emb = np.array(model.model.embed_tokens.weight, dtype=np.float32)      # tied: also the unembedding
    normw = np.array(model.model.norm.weight, dtype=np.float32)
    cal = token_blocks(tok, "train", N_CAL)
    heldx = token_blocks(tok, "test", N_HELD)[:N_HELD_X]
    XC, XH = capture(model, cal, args.layers), capture(model, heldx, args.layers)
    th = np.array(heldx[:, :-1]).reshape(-1)
    # Drop the first position of every block. Its activations are huge (an attention-sink token) and
    # dominate any R2 in late layers. Field practice also leaves it out.
    keep_c = np.arange(XC[args.layers[0]].shape[0]) % (cal.shape[1] - 1) != 0
    keep_h = np.arange(th.shape[0]) % (heldx.shape[1] - 1) != 0
    XC = {L: v[keep_c] for L, v in XC.items()}
    XH = {L: v[keep_h] for L, v in XH.items()}
    th = th[keep_h]
    path = os.path.join(RES, "clean-r1-analyze.json")
    out = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    for L in args.layers:
        G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
        Xc, Xh = XC[L].astype(np.float64), XH[L].astype(np.float64)
        Hc, Hh = hidden(Xc, G, U), hidden(Xh, G, U)          # old activations, tokens x 1536
        # Self-test: an old neuron regressed on all old neurons is found with one neuron.
        j = int(np.argmax(np.linalg.norm(Hc, axis=0)))
        st, sel, _ = omp(Hc, Hc[:, j], Hh, Hh[:, j], smax=1, report=(1,))
        print(f"L{L} self-test: old neuron {j} -> picked {sel[0]}, R2 {st[1]:.4f}", flush=True)
        md = [f"# R1 trained neurons, layer {L}\n", "Pole + = largest positive activation. Writes = output column "
              "through the final norm and the (tied) unembedding.\n"]
        for K in args.ks:
            for tag in args.tags:
                if not os.path.exists(wpath(L, K, tag)):
                    continue
                P = load_w(L, K, tag)
                Ac, Ah = hidden(Xc, P[0], P[1]), hidden(Xh, P[0], P[1])
                # Null: untrained random neurons of the same shape.
                R = rand_init(G, U, D, min(K, 20), seed=1)
                Rc, Rh = hidden(Xc, R[0], R[1]), hidden(Xh, R[0], R[1])
                size = np.linalg.norm(Ah, axis=0) * np.linalg.norm(P[2], axis=0)
                order = np.argsort(-size)
                n_read = min(K, args.max_neurons)
                neurons = []
                for c in order[:n_read]:
                    r2, sel, coef = omp(Hc, Ac[:, c], Hh, Ah[:, c])
                    full = ridge_r2(Hc, Ac[:, c], Hh, Ah[:, c])
                    w = {}
                    for name, vec, atoms in (("gate", P[0][c], G), ("value", P[1][c], U), ("out", P[2][:, c], D.T)):
                        r2v, sv = omp_vec(atoms, vec, 10)
                        w[name] = {"r2_10": r2v, "support": sv}
                    jac = {f"{a}-{b}": len(set(w[a]["support"]) & set(w[b]["support"])) / 10
                           for a, b in (("gate", "value"), ("gate", "out"), ("value", "out"))}
                    act_sup = set(sel[:10])
                    jac["act-gate"] = len(act_sup & set(w["gate"]["support"])) / 10
                    neurons.append({"neuron": int(c), "share": float(size[c] / size.sum()), "omp_r2": r2,
                                    "ridge_r2_all": full, "support": sel[:20], "coef": coef[:20],
                                    "weights": {k: v["r2_10"] for k, v in w.items()}, "overlap10": jac})
                nulls = []
                for c in range(Rc.shape[1]):
                    r2, _, _ = omp(Hc, Rc[:, c], Hh, Rh[:, c], smax=10, report=(1, 10))
                    rv = {n: omp_vec(a, v, 10)[0] for n, v, a in (("gate", R[0][c], G), ("value", R[1][c], U),
                                                                   ("out", R[2][:, c], D.T))}
                    nulls.append({"omp_r2": r2, "weights": rv})
                summ = {s: float(np.median([n["omp_r2"][s] for n in neurons])) for s in (1, 3, 10, 50)}
                summ["all"] = float(np.median([n["ridge_r2_all"] for n in neurons]))
                null = {s: float(np.median([n["omp_r2"][s] for n in nulls])) for s in (1, 10)}
                wsum = {k: float(np.median([n["weights"][k] for n in neurons])) for k in ("gate", "value", "out")}
                wnull = {k: float(np.median([n["weights"][k] for n in nulls])) for k in ("gate", "value", "out")}
                ov = {k: float(np.mean([n["overlap10"][k] for n in neurons])) for k in neurons[0]["overlap10"]}
                row = {"layer": L, "K": K, "tag": tag, "n_read": n_read, "act_r2_median": summ,
                       "act_r2_null_random_neurons": null, "weight_r2_10_median": wsum,
                       "weight_r2_10_null": wnull, "support_overlap_mean": ov, "neurons": neurons}
                out["rows"] = [r for r in out["rows"] if not (r["layer"] == L and r["K"] == K and r["tag"] == tag)]
                out["rows"].append(row)
                json.dump(out, open(path, "w"), indent=1)
                print(f"L{L:2d} K={K:4d} {tag:10s} act R2 (1/3/10/50/all neurons) "
                      + "/".join(f"{summ[s]:.2f}" for s in (1, 3, 10, 50, "all"))
                      + f"  null(1/10) {null[1]:.2f}/{null[10]:.2f} | weight R2@10 gate/value/out "
                      + "/".join(f"{wsum[k]:.2f}" for k in ("gate", "value", "out"))
                      + f" null " + "/".join(f"{wnull[k]:.2f}" for k in ("gate", "value", "out"))
                      + f" | overlap g-v {ov['gate-value']:.2f} act-g {ov['act-gate']:.2f} [{time.time() - t0:.0f}s]",
                      flush=True)
                if K <= args.label_k:
                    md.append(f"\n## K={K}, {tag}\n")
                    for nrec in neurons[: args.label_n]:
                        c = nrec["neuron"]
                        a = Ah[:, c]
                        wr = (P[2][:, c] * normw) @ emb.T
                        md.append(f"\n### neuron {c} (share of output {nrec['share']:.2f}, act R2 with 1/10/50 old "
                                  f"neurons {nrec['omp_r2'][1]:.2f}/{nrec['omp_r2'][10]:.2f}/{nrec['omp_r2'][50]:.2f})\n")
                        md.append("- Pole +: " + " | ".join(contexts(tok, th, np.argsort(-a))))
                        md.append("- Pole -: " + " | ".join(contexts(tok, th, np.argsort(a))))
                        md.append("- Writes +: " + ", ".join(repr(tok.decode([int(i)])) for i in np.argsort(-wr)[:12]))
                        md.append("- Writes -: " + ", ".join(repr(tok.decode([int(i)])) for i in np.argsort(wr)[:12]))
                        for j, cf in list(zip(nrec["support"], nrec["coef"]))[:4]:
                            md.append(f"  - old neuron {j} (coef {cf:+.2f}): + "
                                      + " | ".join(contexts(tok, th, np.argsort(-Hh[:, j]), n=6))
                                      + "  ;; - " + " | ".join(contexts(tok, th, np.argsort(Hh[:, j]), n=4)))
        with open(os.path.join(RES, f"clean-r1-labels-L{L}.md"), "w") as f:
            f.write("\n".join(md) + "\n")
    print("wrote", path)


def cmd_check(args):
    """K=1536 trained from the original weights must keep the original CE."""
    model, tok = load_model()
    W = Weights(model)
    L = args.layers[0]
    cal = token_blocks(tok, "train", N_CAL)
    held = token_blocks(tok, "test", N_HELD)
    base = block_ce(model, held)
    Xc = capture(model, cal, [L])[L].astype(np.float64)
    P0 = tuple(a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L]))
    P = train_local(Xc, mlp_out(Xc, P0), P0, steps=300, lr=1e-4)
    set_small(model, L, P)
    ce = block_ce(model, held)
    print(f"check L{L}: base {base.mean():.4f}  after K=1536 local training {ce.mean():.4f}  diff "
          f"{boot_diff(ce, base)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "global", "analyze", "check"])
    ap.add_argument("--layers", default="3,15,28")
    ap.add_argument("--ks", default="192,48,10")
    ap.add_argument("--tags", default="local_cut,local_rand,global,cut")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_neurons", type=int, default=48)
    ap.add_argument("--label_k", type=int, default=10)
    ap.add_argument("--label_n", type=int, default=10)
    args = ap.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    args.ks = [int(x) for x in args.ks.split(",")]
    args.tags = args.tags.split(",")
    {"train": cmd_train, "global": cmd_global, "analyze": cmd_analyze, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
