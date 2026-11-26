"""geometry step 1-3 on the pass covariances (results/pair/geometry/pass_cov.npz).

1. How linear is Small's block-0 input u_S in Base's u_B: per-dim test R^2, canonical correlations rho (fit, and test
   correlations along the fit canonical pairs).
2. Canonical coordinates. Whiten u_B and u_S (fit covariances), K = cross-correlation (768 x 1024) = A diag(rho) B^T.
   Small read a_s (whitened) -> alpha_s = A^T a_s; Base read w_p (whitened) -> beta_p = B^T w_p.
   cos(alpha_s, beta_p) inside rho bands: if high where rho is high, the gap is upstream; if flat, it is this layer.
3. Transports of p's read into Small's input, scored as the correlation of linear pre-activations in Small's input
   (cos in Small's whitened coordinates):
     T0: a = A beta_p           (same canonical coefficients; assumes perfect channel)
     T1: a = A rho beta_p       (= regression of h_p on u_S: the best Small-linear copy of p)
     T2: a = A rho^2 beta_p
   Controls: random Base p', p's nearest read neighbour (in u_B metric), and top-1 identification among all 4,096.
4. Reliability: s's best linear read of u_B from fit vs from test covariances.
5. Post-GELU score on test rows: corr(q . u_B, g_s) for rule reads q on u_B; ceiling = s's own regression on u_B (fit).
Neurons: agreeing pairs (fire best = write best). Outputs results/pair/geometry/cca.json.
"""
import json
import os

import numpy as np
import torch

from cache import OUT

T = torch.float64
GO = os.path.join(OUT, "geometry")


def t(x):
    return torch.tensor(np.asarray(x), dtype=T)


def main():
    P = np.load(os.path.join(GO, "pass_cov.npz"))
    d = np.load(os.path.join(OUT, "stream_cov.npz"))
    W = t(d["W1"])          # (4096, 1024)
    Ws = t(d["W1s"])        # (3072, 768)
    Rf = np.load(os.path.join(OUT, "tools_peakR_b0.npy")).astype(np.float32)
    Rw = np.load(os.path.join(OUT, "writesR_b0.npy")).astype(np.float32)
    fb, wb = Rf.argmax(1), Rw.argmax(1)
    agree = np.where(fb == wb)[0]
    res = {}
    eps = 1e-6
    I = lambda n: torch.eye(n, dtype=T)
    CBB, CSS, CSB = t(P["fit_u_BB"]) + eps * I(1024), t(P["fit_u_SS"]) + eps * I(768), t(P["fit_u_SB"])
    tBB, tSS, tSB = t(P["test_u_BB"]), t(P["test_u_SS"]), t(P["test_u_SB"])

    # 1. per-dim R^2 of u_S from u_B (fit map, test score)
    M = torch.linalg.solve(CBB, CSB.T).T                       # u_S ~ M u_B  (768 x 1024)
    resid = tSS - M @ tSB.T - tSB @ M.T + M @ tBB @ M.T
    r2 = 1 - resid.diagonal() / tSS.diagonal()
    res["uS_from_uB_test_R2_per_dim"] = dict(median=float(r2.median()), q10=float(r2.quantile(.1)),
                                             q90=float(r2.quantile(.9)))
    res["uS_from_uB_test_R2_total_var"] = float(1 - resid.trace() / tSS.trace())

    # 2. canonical coordinates
    Lb, Ls = torch.linalg.cholesky(CBB), torch.linalg.cholesky(CSS)
    K = torch.linalg.solve_triangular(Ls, torch.linalg.solve_triangular(Lb, CSB.T, upper=False).T, upper=False)
    A, rho, Bh = torch.linalg.svd(K, full_matrices=False)     # A (768x768), Bh (768 x 1024)
    # test canonical correlations along fit directions
    Lbi = torch.linalg.solve_triangular(Lb, I(1024), upper=False)
    Lsi = torch.linalg.solve_triangular(Ls, I(768), upper=False)
    xb = Bh @ Lbi          # rows: canonical directions in u_B coords
    xs = A.T @ Lsi         # rows: canonical directions in u_S coords
    num = ((xs @ tSB) * xb).sum(1)
    den = ((xb @ tBB) * xb).sum(1).sqrt() * ((xs @ tSS) * xs).sum(1).sqrt()
    rho_test = num / den
    res["rho_fit_quantiles"] = {q: float(rho.quantile(q)) for q in (.1, .25, .5, .75, .9)}
    res["rho_test_quantiles"] = {q: float(rho_test.quantile(q)) for q in (.1, .25, .5, .75, .9)}
    res["n_rho_over"] = {th: int((rho_test > th).sum()) for th in (.95, .9, .8, .7, .5, .3)}

    a_s = Ws @ Ls                                              # whitened Small reads
    a_s = a_s / a_s.norm(dim=1, keepdim=True)
    w_p = W @ Lb
    w_p = w_p / w_p.norm(dim=1, keepdim=True)
    al = a_s @ A                                               # (3072, 768) canonical coefs of Small reads
    be = w_p @ Bh.T                                            # (4096, 768) canonical coefs of Base reads (shared part)
    res["base_read_energy_in_shared_768"] = float((be ** 2).sum(1).median())
    # nearest read neighbour of p (u_B metric), and a random p'
    G = w_p @ w_p.T
    G.fill_diagonal_(-2)
    nn = G.argmax(1).numpy()
    rng = np.random.default_rng(0)
    S = agree
    p = fb[S]
    pr = rng.integers(0, 4096, len(S))
    bands = [(.9, 1.01), (.8, .9), (.7, .8), (.5, .7), (.3, .5), (0, .3)]
    bandres = {}
    for lo, hi in bands:
        m = (rho_test >= lo) & (rho_test < hi)
        if m.sum() < 3:
            continue

        def bc(q):
            x, y = al[S][:, m], be[q][:, m]
            return float(((x * y).sum(1) / x.norm(dim=1) / y.norm(dim=1)).median())
        bandres[f"{lo}-{hi}"] = dict(n_dims=int(m.sum()), cos_p=bc(p), cos_nn=bc(nn[p]), cos_rand=bc(pr),
                                     energy_s=float((al[S][:, m] ** 2).sum(1).median()),
                                     energy_p=float((be[p][:, m] ** 2).sum(1).median()))
    res["bands"] = bandres

    # 3. transports into Small's whitened input
    def score_tr(k):
        Tall = (be * rho ** k) @ A.T                           # (4096, 768)
        Tall = Tall / Tall.norm(dim=1, keepdim=True)
        c = a_s[S] @ Tall.T                                     # (|S|, 4096)
        cp = c[np.arange(len(S)), p]
        out = dict(cos_p=float(cp.median()), cos_nn=float(c[np.arange(len(S)), nn[p]].median()),
                   cos_rand=float(c[np.arange(len(S)), pr].median()),
                   top1_is_p=float((c.argmax(1).numpy() == p).mean()),
                   rank_p_median=float((c > cp[:, None]).sum(1).double().median()))
        return out, c
    for k in (0, 1, 2):
        res[f"T{k}"], _ = score_tr(k)

    # 4. reliability of s's best linear read on u_B (fit vs test)
    def Vread(BB, SB):
        return torch.linalg.solve(BB + eps * I(1024), (Ws[S] @ SB).T).T   # (|S|, 1024)
    Vf, Vt = Vread(CBB, CSB), Vread(tBB, tSB)
    mf, mt = Vf @ Lb, Vt @ Lb
    res["V_fit_vs_test_cos_median"] = float(((mf * mt).sum(1) / mf.norm(dim=1) / mt.norm(dim=1)).median())
    # raw (no transport) cos in u_B metric, same pass, for continuity with stream_V
    res["raw_cos_Vfit_p_median"] = float(((mf / mf.norm(dim=1, keepdim=True)) * w_p[p]).sum(1).median())

    # 5. post-GELU test score of linear reads on u_B
    gCov = t(P["test_gS_cov_uB"])[S]                           # (|S|, 1024)
    gvar = t(P["test_gS_var"])[S]

    def post(Q):   # Q rows in u_B coords
        num = (Q * gCov).sum(1)
        den = ((Q @ tBB) * Q).sum(1).clamp_min(1e-30).sqrt() * gvar.sqrt()
        return float((num / den).median())
    ceil = torch.linalg.solve(CBB, t(P["fit_gS_cov_uB"])[S].T).T
    res["post_test_corr"] = dict(
        own_regression_ceiling=post(ceil),
        s_pre_read_V=post(Vf),
        p_read=post(W[p]),
        p_read_T1=post(((be[p] * rho ** 2) @ Bh) @ Lbi),      # K^T A rho beta = B rho^2 beta in zB -> u_B coords
        p_read_T2=post(((be[p] * rho ** 3) @ Bh) @ Lbi),
        nn_read=post(W[nn[p]]), rand_read=post(W[pr]))
    # also p's own regression of its output on u_B (fit) as a read, against s's output
    gBc = t(P["fit_gB_cov_uB"])
    res["post_test_corr"]["p_output_regression"] = post(torch.linalg.solve(CBB, gBc[p].T).T)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(GO, "cca.json"), "w"), indent=1)
    np.savez(os.path.join(GO, "cca_basis.npz"), rho=rho.numpy(), rho_test=rho_test.numpy())


if __name__ == "__main__":
    main()
