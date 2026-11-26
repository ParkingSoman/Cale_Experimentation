"""Pair study, open search: rebuild each Small neuron from the fewest Base neurons, then find what picks them.

Input: grams.npz (outputs after GELU, stage-4 block 0, all positions).
1. For each of the 3,072 Small neurons s: forward selection of Base neurons with weights >= 0 (least squares on the
   fit covariances; a term that turns negative is dropped), up to 12 terms. R^2 on test rows after each step.
2. For each s: term 1 (its best single Base neuron) and partners (terms 2-5). Base-only relations of each partner to
   term 1, against all other Base neurons (percentile rank, 100 = closest):
     act    correlation of outputs (fit rows)
     write  cosine of class-score effects (writes_V.npz)
     wout   cosine of stream write vectors (layer_scale * W2 column)
     read   cosine of read vectors (W1 rows)
3. Coverage: how often each Base neuron is used, and by how many Small neurons.
"""
import json
import os

import numpy as np

from cache import OUT, load


def nn_forward(C, c, v, Ct, ct, vt, kmax=12):
    S, w, r2s, r2t = [], None, [], []
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    for _ in range(kmax):
        r = c - (C[:, S] @ w if S else 0)
        sc = r / d
        sc[S] = -np.inf
        j = int(np.argmax(sc))
        if sc[j] <= 0:
            break
        S.append(j)
        while True:
            w = np.linalg.solve(C[np.ix_(S, S)] + 1e-8 * np.eye(len(S)), c[S])
            if (w >= 0).all():
                break
            S = [s for s, x in zip(S, w) if x >= 0]
        r2s.append(float((2 * w @ c[S] - w @ C[np.ix_(S, S)] @ w) / v))
        r2t.append(float((2 * w @ ct[S] - w @ Ct[np.ix_(S, S)] @ w) / vt))
    return S, w, r2s, r2t


def pct_rank(score, j, ref):
    s = score.copy()
    s[ref] = -np.inf
    return float((s < s[j]).sum() / (len(s) - 1) * 100)


def main():
    g = np.load(os.path.join(OUT, "grams.npz"))
    C, Cs, v = g["fit_Cbb"].astype(np.float64), g["fit_Cbs"].astype(np.float64), g["fit_vs"]
    Ct, Cst, vt = g["test_Cbb"].astype(np.float64), g["test_Cbs"].astype(np.float64), g["test_vs"]
    m = load("base")
    blk = m.features[7][0]
    W1 = blk.block[3].weight.detach().cpu().double().numpy()
    Wo = (blk.layer_scale.detach().cpu().double().view(-1, 1) * blk.block[5].weight.detach().cpu().double()).numpy().T
    Vw = np.load(os.path.join(OUT, "writes_V.npz"))["b0"].astype(np.float64)
    nrm = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    W1n, Won, Vwn = nrm(W1), nrm(Wo), nrm(Vw)
    sd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    R = C / sd[:, None] / sd[None, :]
    rows = []
    for s in range(3072):
        S, w, r2s, r2t = nn_forward(C, Cs[:, s], v[s], Ct, Cst[:, s], vt[s])
        if not S:
            rows.append(dict(s=s, terms=[], r2_test=[]))
            continue
        order = np.argsort(-w * sd[S])  # by contribution size
        terms = [S[i] for i in order]
        t1 = terms[0]
        rel = []
        for j in terms[1:5]:
            rel.append(dict(j=int(j), act=pct_rank(R[t1], j, t1), write=pct_rank(Vwn @ Vwn[t1], j, t1),
                            wout=pct_rank(Won @ Won[t1], j, t1), read=pct_rank(W1n @ W1n[t1], j, t1),
                            act_raw=float(R[t1, j]), write_raw=float(Vwn[j] @ Vwn[t1]),
                            wout_raw=float(Won[j] @ Won[t1]), read_raw=float(W1n[j] @ W1n[t1])))
        rows.append(dict(s=s, terms=[int(t) for t in terms], weights=[float(w[i] * sd[S[i]]) for i in order],
                         first_selected=int(S[0]), r2_fit=r2s, r2_test=r2t, partners=rel))
    ok = [r for r in rows if r["terms"]]
    rk = lambda k: [r["r2_test"][min(k, len(r["r2_test"])) - 1] for r in ok]
    summ = {f"r2_test_k{k}_median": float(np.median(rk(k))) for k in (1, 2, 3, 5, 8, 12)}
    summ["n_terms_median"] = float(np.median([len(r["terms"]) for r in ok]))
    share2 = [r["weights"][1] / r["weights"][0] for r in ok if len(r["weights"]) > 1]
    summ["weight2_over_weight1_median"] = float(np.median(share2))
    P = [p for r in ok for p in r["partners"][:2]]
    for key in ("act", "write", "wout", "read"):
        summ[f"partner_rank_{key}_median"] = float(np.median([p[key] for p in P]))
        summ[f"partner_{key}_raw_median"] = float(np.median([p[f"{key}_raw"] for p in P]))
        summ[f"partner_top1pct_{key}"] = float(np.mean([p[key] >= 99 for p in P]))
    # random-pair baselines for raw values
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 4096, 5000), rng.integers(0, 4096, 5000)
    summ["random_raw"] = dict(act=float(np.median(R[a, b])), write=float(np.median((Vwn[a] * Vwn[b]).sum(1))),
                              wout=float(np.median((Won[a] * Won[b]).sum(1))),
                              read=float(np.median((W1n[a] * W1n[b]).sum(1))))
    used = np.zeros(4096, int)
    for r in ok:
        for t in r["terms"][:5]:
            used[t] += 1
    summ["base_used_in_top5_share"] = float((used > 0).mean())
    summ["base_used_count_q50_q90_max"] = [float(np.quantile(used, .5)), float(np.quantile(used, .9)), int(used.max())]
    t1s = [r["terms"][0] for r in ok]
    summ["distinct_term1"] = len(set(t1s))
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, "greedy.json"), "w"))


if __name__ == "__main__":
    main()
