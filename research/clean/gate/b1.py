"""B1: channels on kept trunk gates vs equal-storage RCF (CHILDREN-LOG "RCF SCALE + CHANNELS", B1).
A channel = new (u, d) pair under a kept trunk's gate: silu(g_i x) (u . x) d. Stored 2d numbers (2/3 of a neuron).

  python3 research/clean/gate/b1.py --layers 12,15,16,22 --budgets 48:18,48:72,384:144
"""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import orig_weights, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot  # noqa: E402
from g10 import Lin  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from sink import SinkGuard  # noqa: E402
from a0 import grams, cv_ridge_g, LAMS, FORBID  # noqa: E402

PER_TRUNK = 4   # candidate channels per trunk (top singular directions)


def candidates(Xs, Ss, Rs, E, P=None, Xv=None, Sv=None, Rv=None):
    """Per trunk i: rank-r LS of residual Rs on F_i = s_i * x under the metric F_i^T F_i. Returns top-E (gain, i, u, d).
    P (d x p): restrict u to span(P). With Xv/Sv/Rv: gain = held-out residual reduction (write refit on the fit rows)."""
    cands = []
    Xr = Xs @ P if P is not None else Xs
    for i in range(Ss.shape[1]):
        F = Xr * Ss[:, i:i + 1]; M = (F.T @ F).astype(np.float64); M[np.diag_indices_from(M)] += 1e-3 * np.trace(M) / M.shape[0]
        C = np.linalg.cholesky(M); B = (F.T @ Rs).astype(np.float64)          # M = C C^T
        N = np.linalg.solve(M, B)                                           # full LS
        Pm, sv, Qt = np.linalg.svd(C.T @ N, full_matrices=False)
        for k in range(PER_TRUNK):
            ur = np.linalg.solve(C.T, Pm[:, k] * sv[k]); u = P @ ur if P is not None else ur
            gain = sv[k] ** 2
            if Xv is not None:
                a = (Ss[:, i] * (Xs @ u)).astype(np.float64); d = (a @ Rs) / max(a @ a, 1e-12)
                av = (Sv[:, i] * (Xv @ u)).astype(np.float64)
                gain = float(np.sum(Rv.astype(np.float64) ** 2) - np.sum((Rv - np.outer(av, d)) ** 2))
            cands.append((gain, i, u, Qt[k]))
    cands.sort(key=lambda c: -c[0])
    return cands[:E]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,15,16,22"); ap.add_argument("--budgets", default="48:18,48:72,384:144")
    ap.add_argument("--ncal", type=int, default=128); ap.add_argument("--screen_only", action="store_true"); ap.add_argument("--pdim", type=int, default=0); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    budgets = [tuple(int(v) for v in b.split(":")) for b in a.budgets.split(",")]
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    base = {"prose": block_ce(model, dev), "code": block_ce(model, devc)}
    path = os.path.join(ROOT, "results", "clean-b1.json" if not a.pdim else "clean-b1b.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for L in layers:
        lay = model.model.layers[L]; prev = model.model.layers[L - 1]
        cap = capture(model, cal, {"x": (lay, "mlp"), "z": (lay.self_attn, "o_proj"), "xp": (prev, "mlp")})
        G, U, D = orig_weights(model, L); X = cap["x"]
        Gf, Uf, Df = (w.astype(np.float32) for w in (G, U, D))
        S = silu(X @ Gf.T).astype(np.float32); H = (S * (X @ Uf.T)).astype(np.float32); Y = (H @ Df.T).astype(np.float32)
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); order = np.argsort(-en)
        Gp, Up, Dp = orig_weights(model, L - 1); Xp = cap["xp"]
        R = np.concatenate([(silu(Xp @ Gp.T.astype(np.float32)) * (Xp @ Up.T.astype(np.float32))).astype(np.float32), cap["z"]], 1)
        del Xp, cap
        n = X.shape[0]; half = n // 2; hold = int(0.2 * half)      # rows: prose half, then code half
        tr = np.r_[0:half - hold, half:n - hold]; te = np.r_[half - hold:half, n - hold:n]
        Wo = np.array(lay.self_attn.o_proj.weight, dtype=np.float64)
        o_mlp, o_op, o_prev = lay.mlp, lay.self_attn.o_proj, prev.mlp; orig_L = SwiGLULayer(G, U, D)

        def fit(keep, ch):
            """Joint CV ridge on train rows. ch = list of (trunk index in layer, u, d). Returns (layer weights, W split)."""
            Hk = H[:, keep]
            Ac = np.stack([S[:, i] * (X @ u.astype(np.float32)) for i, u, _ in ch], 1).astype(np.float32) if ch else np.zeros((n, 0), np.float32)
            A = np.concatenate([Hk, Ac, R], 1); W, lam = cv_ridge_g(grams(A[tr], Y[tr]), LAMS)
            err = float(np.sum((Y[te] - A[te] @ W.astype(np.float32)) ** 2) / np.sum(Y[te] ** 2)); del A
            k, e = len(keep), len(ch)
            Gl = np.concatenate([G[keep]] + [G[[i for i, _, _ in ch]]] if ch else [G[keep]], 0)
            Ul = np.concatenate([U[keep]] + [np.stack([u for _, u, _ in ch])] if ch else [U[keep]], 0)
            Dl = W[:k + e].T
            return (Gl, Ul, Dl), W[k + e:], err, lam

        def score(layerw, Wr):
            lay.mlp = SinkGuard(SwiGLULayer(*layerw), orig_L)
            Cp = Wr[:Gp.shape[0]]; prev.mlp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp))
            lay.self_attn.o_proj = SinkGuard(Lin(Wo + Wr[Gp.shape[0]:].T), Lin(Wo))
            r = {t: block_ce(model, b) for t, b in (("prose", dev), ("code", devc))}
            lay.mlp, lay.self_attn.o_proj, prev.mlp = o_mlp, o_op, o_prev
            return r

        for K, E in budgets:
            Kp = K + math.ceil(2 * E / 3); keep = np.sort(order[:K]); keepp = np.sort(order[:Kp])
            base_w, base_r, _, _ = fit(keep, [])
            A0r = np.concatenate([H[:, keep], R], 1)
            Wb = np.concatenate([base_w[2].T, base_r], 0).astype(np.float32)
            Res = (Y - A0r @ Wb).astype(np.float32); del A0r
            if a.pdim:   # B1b: low-dim up vectors, held-out gain
                sub, val = tr[0::4], tr[2::4]
                Xc = X[sub] - X[sub].mean(0); _, _, Vt = np.linalg.svd(Xc[::4], full_matrices=False); Pd = Vt[:a.pdim].T
                cands = candidates(X[sub], S[sub][:, keep], Res[sub], E, Pd, X[val], S[val][:, keep], Res[val])
            else:
                sub = tr[::4]
                cands = candidates(X[sub], S[sub][:, keep], Res[sub], E)
            ch = [(int(keep[i]), u, d) for _, i, u, d in cands]
            ch_w, ch_r, err_ch, lam_ch = fit(keep, ch)
            kp_w, kp_r, err_kp, lam_kp = fit(keepp, [])
            n_tr = len(set(i for i, _, _ in ch))
            row = {"K": K, "E": E, "Kp": Kp, "screen_err_ch": err_ch, "screen_err_kp": err_kp, "lam": [lam_ch, lam_kp],
                   "trunks_used": n_tr, "top_gain_share": float(sum(c[0] for c in cands) / np.sum(Res[val if a.pdim else sub] ** 2))}
            print(f"L{L} K{K} E{E} vs K'{Kp}: screen err ch {err_ch:.4f} kp {err_kp:.4f} trunks {n_tr} [{time.time()-t0:.0f}s]", flush=True)
            if not a.screen_only:
                rc, rk = score(ch_w, ch_r), score(kp_w, kp_r)
                for t in ("prose", "code"):
                    row[t] = {"ch": float((rc[t] - base[t]).mean()), "kp": float((rk[t] - base[t]).mean()), "ch_minus_kp": boot(rc[t] - rk[t])}
                print(f"   prose ch {row['prose']['ch']:.4f} kp {row['prose']['kp']:.4f} {np.round(row['prose']['ch_minus_kp'], 4)} | "
                      f"code ch {row['code']['ch']:.4f} kp {row['code']['kp']:.4f} {np.round(row['code']['ch_minus_kp'], 4)}", flush=True)
            out[f"L{L}_K{K}_E{E}"] = row; json.dump(out, open(path, "w"), indent=1)
        del H, S, X, Y, R


if __name__ == "__main__":
    main()
