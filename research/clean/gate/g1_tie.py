"""G1 diagnostic: how equal are the most-shared gates? Tie error for the best partner of each neuron."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "coarse"))
from contract import mlp_io, orig_weights, ROOT
from tree_lib import load_model, token_blocks
def silu(z): return z / (1 + np.exp(-z))
model, tok = load_model(); cal = token_blocks(tok, "train", 32)
out = {}
for L in (15, 28):
    X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
    X = X[np.random.default_rng(0).choice(len(X), 12000, replace=False)]
    GP = X @ G.T; S = silu(GP); UP = X @ U.T; H = S * UP
    Z = (S - S.mean(0)) / (S.std(0) + 1e-9); C = Z.T @ Z / len(Z); np.fill_diagonal(C, 0)
    best = np.argmax(np.abs(C), 1); bc = np.abs(C)[np.arange(len(C)), best]
    # tie error: replace silu(g_b) by c*silu(g_a)+e (LS) inside b's hidden; also pre-activation corr on rows where either is open
    tie, opencorr = [], []
    for b in range(len(C)):
        a = best[b]; A = np.stack([S[:, a] * UP[:, b], UP[:, b]], 1)
        coef = np.linalg.lstsq(A, H[:, b], rcond=None)[0]
        tie.append(np.sum((H[:, b] - A @ coef) ** 2) / (np.sum(H[:, b] ** 2) + 1e-12))
        m = (GP[:, a] > 0) | (GP[:, b] > 0)
        opencorr.append(abs(np.corrcoef(GP[m, a], GP[m, b])[0, 1]) if m.sum() > 50 else np.nan)
    tie = np.array(tie); opencorr = np.array(opencorr)
    q = lambda v: [round(float(np.nanquantile(v, p)), 3) for p in (.1, .5, .9, .99)]
    out[L] = {"best_silu_corr_q10_50_90_99": q(bc), "tie_err_q": q(tie), "open_preact_corr_q": q(opencorr),
              "share_corr_ge_.8": float(np.mean(bc >= .8)), "share_corr_ge_.9": float(np.mean(bc >= .9)),
              "share_tie_err_le_.1": float(np.mean(tie <= .1)), "share_tie_err_le_.25": float(np.mean(tie <= .25))}
    print(L, out[L], flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "clean-g1-tie.json"), "w"), indent=1)
