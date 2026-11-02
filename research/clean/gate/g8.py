"""G8 (exploratory, then pre-registered if it helps): side-effect-aware fold. Penalize folded write directions that the
layers BETWEEN the route and layer L read. Sink guard on all changed layers.

  python3 research/clean/gate/g8.py --L 12 --K 48 --mult 30 --mus 0,1,10,100
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import mlp_io, orig_weights, resid_in, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot, FORBID  # noqa: E402
from g7 import ridge, cv_ridge, BACK  # noqa: E402
from sink import SinkGuard  # noqa: E402


def reader_matrix(model, cal, lo, L, keep):
    """Sum over in-between readers (layers lo+1..L) of (row * norm weight / rms)^T (same). Layer L: attention + kept MLP rows."""
    d = model.model.layers[0].mlp.gate_proj.weight.shape[1]; P = np.zeros((d, d))
    for m in range(lo + 1, L + 1):
        lay = model.model.layers[m]; h = resid_in(model, cal[:8], m).reshape(-1, d)
        rms = float(np.sqrt(np.mean(h[np.arange(len(h)) % h.shape[0] != 0] ** 2)))
        a = np.array(lay.input_layernorm.weight, dtype=np.float64); w = np.array(lay.post_attention_layernorm.weight, dtype=np.float64)
        rows = [np.array(getattr(lay.self_attn, p).weight, dtype=np.float64) * a for p in ("q_proj", "k_proj", "v_proj")]
        Gm, Um, _ = orig_weights(model, m)
        if m == L:
            Gm, Um = Gm[keep], Um[keep]
        rows += [Gm * w, Um * w]
        R = np.concatenate(rows, 0) / rms
        P += R.T @ R / R.shape[0]
    return P


def penal_ls(A, Y, nk, P, mu, lam):
    """min ||Y - A W||^2 + lam*ridge + mu * sum_{route rows r} W_r P W_r^T (rows nk.. are routes)."""
    M = (A.T @ A).astype(np.float64); n = M.shape[0]; M += np.eye(n) * lam * np.trace(M) / n
    B = (A.T @ Y).astype(np.float64)
    if mu == 0:
        return np.linalg.solve(M, B)
    ev, V = np.linalg.eigh(P); ev = ev / ev.mean()
    Bv = B @ V; Wv = np.zeros_like(Bv); s = np.zeros(n); s[nk:] = 1.0; scale = mu * np.trace(M) / n
    for k in range(V.shape[1]):
        Wv[:, k] = np.linalg.solve(M + np.diag(s * scale * ev[k]), Bv[:, k])
    return Wv @ V.T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=12); ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--mult", type=int, default=30); ap.add_argument("--mus", default="0,0.1,1,10"); ap.add_argument("--back", type=int, default=BACK); ap.add_argument("--code", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); L, K = a.L, a.K; assert L not in FORBID
    model, tok = load_model(); cal = token_blocks(tok, "train", 64); dev = token_blocks(tok, "test", 40)
    if a.code:
        import mlx.core as mx
        txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()
        ids = np.array(tok.encode(txt[:400000]), dtype=np.int32); nb = min(40, len(ids) // 513)
        dev = mx.array(ids[:nb * 513].reshape(nb, 513))
    base = block_ce(model, dev); t0 = time.time()
    X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
    H = (silu(X @ G.T) * (X @ U.T)).astype(np.float32); Y = (H @ D.T).astype(np.float32)
    en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); keep = np.argsort(-en)[:K]
    lays = list(range(max(0, L - a.back), L)); acts, lab = [], []
    for l in lays:
        Xl, _ = mlp_io(model, cal, l); Gl, Ul, _ = orig_weights(model, l)
        acts.append((silu(Xl @ Gl.T) * (Xl @ Ul.T)).astype(np.float32)); lab += [(l, j) for j in range(Gl.shape[0])]
    Aall = np.concatenate(acts, 1); del acts
    orig_L = SwiGLULayer(G, U, D)
    def run(layer):
        o = model.model.layers[L].mlp; model.model.layers[L].mlp = SinkGuard(layer, orig_L)
        ce = block_ce(model, dev); model.model.layers[L].mlp = o; return ce
    Wc = ridge(H[:, keep], Y, 1e-4); ce_cut = run(SwiGLULayer(G[keep], U[keep], Wc.T))
    Yres = Y - H[:, keep] @ Wc
    sc = np.sum((Aall.T @ Yres) ** 2, 1) / (np.sum(Aall ** 2, 0) + 1e-9); routes = np.argsort(-sc)[:min(a.mult * K, Aall.shape[1])]
    Afit = np.concatenate([H[:, keep], Aall[:, routes]], 1)
    _, lam = cv_ridge(Afit, Y)
    P = reader_matrix(model, cal, min(lays), L, keep)
    print(f"L{L} K{K}: ready, lam {lam}, cut {float((ce_cut-base).mean()):.4f} [{time.time()-t0:.0f}s]", flush=True)
    path = os.path.join(ROOT, "results", "clean-g8.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    row = {"cut": float((ce_cut - base).mean())}
    for mu in map(float, a.mus.split(",")):
        W = penal_ls(Afit, Y, K, P, mu, lam); Dk, C = W[:K].T, W[K:]
        newD = {l: orig_weights(model, l)[2].copy() for l in lays}
        for ri, r in enumerate(routes):
            l, j = lab[r]; newD[l][:, j] += C[ri]
        origs = {}
        for l in lays:
            Gl, Ul, Dl = orig_weights(model, l); origs[l] = model.model.layers[l].mlp
            model.model.layers[l].mlp = SinkGuard(SwiGLULayer(Gl, Ul, newD[l]), SwiGLULayer(Gl, Ul, Dl))
        ce = run(SwiGLULayer(G[keep], U[keep], Dk))
        for l, mm in origs.items():
            model.model.layers[l].mlp = mm
        fit_err = float(np.sum((Y - Afit @ W) ** 2) / np.sum(Yres ** 2))
        row[f"mu{mu}"] = float((ce - base).mean()); row[f"mu{mu}_minus_cut"] = boot(ce - ce_cut); row[f"mu{mu}_fit_resid_share"] = fit_err
        print(f"  mu {mu}: {row[f'mu{mu}']:.4f} vs cut {[round(x, 4) for x in row[f'mu{mu}_minus_cut']]} fit resid share {fit_err:.3f}", flush=True)
    out[f"L{L}_K{K}_R{a.mult}K_b{a.back}{'_code' if a.code else ''}{a.tag}"] = row; json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
