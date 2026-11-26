"""geometry step 3 analysis: score read rules on ON rows (s or p in its top 1%) vs on all rows.

For a read q on u_B, corr(h_s, q . u_B) = q . c_s / sqrt(q C q * var h_s), with c_s = Cov(h_s, u_B), C = Cov(u_B),
both taken over the chosen rows (onpass.npz per pair, or all fit rows from pass_cov.npz).
Rules: p's read W1[p]; p's nearest Base read; random Base read; write-best Base read P2 (non-agreeing pairs);
p's read transported through the u_B -> u_S canonical channel (T1); ceiling = ridge regression of h_s on u_B in the
same rows (in-sample, optimistic). Prints only (disk is low).
"""
import numpy as np
import torch

from cache import OUT
import os

T = torch.float64
GO = os.path.join(OUT, "geometry")


def main():
    O = np.load(os.path.join(GO, "onpass.npz"))
    P = np.load(os.path.join(GO, "pass_cov.npz"))
    d = np.load(os.path.join(OUT, "stream_cov.npz"))
    W = torch.tensor(d["W1"], dtype=T)
    Ws = torch.tensor(d["W1s"], dtype=T)
    S, Pp, P2 = O["S"], O["P"], O["P2"]
    na = int(O["n_agree"])
    I = torch.eye(1024, dtype=T)
    CBB = torch.tensor(P["fit_u_BB"], dtype=T) + 1e-6 * I
    CSS = torch.tensor(P["fit_u_SS"], dtype=T) + 1e-6 * torch.eye(768, dtype=T)
    CSB = torch.tensor(P["fit_u_SB"], dtype=T)
    Lb = torch.linalg.cholesky(CBB)
    Wm = W @ Lb
    Wm = Wm / Wm.norm(dim=1, keepdim=True)
    G = Wm @ Wm.T
    G.fill_diagonal_(-2)
    nn = G.argmax(1).numpy()
    # T1 transport read on u_B: q = C_BB^-1 C_BS C_SS^-1 C_SB w_p  (p's read -> best Small-linear copy -> back to u_B)
    Mt = torch.linalg.solve(CBB, CSB.T @ torch.linalg.solve(CSS, CSB))      # (1024,1024)
    rng = np.random.default_rng(2)
    pr = rng.integers(0, 4096, len(S))
    rows = []
    for j, s in enumerate(S):
        n = O["n"][j]
        mu = O["su"][j] / n
        C = torch.tensor(O["Con"][j], dtype=T)
        C = C + 1e-4 * C.diagonal().mean() * I
        ms_ = O["shs"][j] / n
        vs = O["shs2"][j] / n - ms_ ** 2
        cs = torch.tensor(O["shsu"][j] / n - ms_ * mu, dtype=T)
        # all rows (fit): c_s = W1s[s] C_SB, var = W1s C_SS W1s
        ca = Ws[s] @ CSB
        va = float(Ws[s] @ CSS @ Ws[s])

        def sc(q, C, c, v):
            return float(q @ c / torch.sqrt((q @ C @ q) * v))
        wp = W[Pp[j]]
        rules = dict(p=wp, nn=W[nn[Pp[j]]], rand=W[pr[j]], p2=W[P2[j]], T1=Mt.T @ wp if False else wp @ Mt.T)
        r = dict(agree=j < na, frac_both=O["n_b"][j] / n, frac_s_only=O["n_s"][j] / n, frac_p_only=O["n_p"][j] / n)
        for k, q in rules.items():
            r["on_" + k] = sc(q, C, cs, vs)
            r["all_" + k] = sc(q, CBB, ca, va)
        qon = torch.linalg.solve(C, cs)
        r["on_ceiling"] = sc(qon, C, cs, vs)
        qall = torch.linalg.solve(CBB, ca)
        r["all_ceiling"] = sc(qall, CBB, ca, va)
        r["on_allread"] = sc(qall, C, cs, vs)     # s's all-row read, scored on on-rows
        # functional, direct: corr(h_s, h_p) on on rows
        mp = O["shp"][j] / n
        r["on_hs_hp"] = (O["shshp"][j] / n - ms_ * mp) / np.sqrt(vs * (O["shp2"][j] / n - mp ** 2))
        rows.append(r)
    keys = [k for k in rows[0] if k != "agree"]
    for name, sel in (("agreeing", [r for r in rows if r["agree"]]), ("non-agreeing", [r for r in rows if not r["agree"]])):
        print(f"{name} (n={len(sel)}) medians:")
        for k in keys:
            print(f"  {k:16s} {np.median([r[k] for r in sel]):.3f}")


if __name__ == "__main__":
    main()
