"""Pair study, step 4 at photo positions: is the Small neuron s its counterpart p plus p's children?

20 pairs from kids.py (p in stage-4 block 0, children found by removing p). Photos: 4 per class from the reading
photos, 2 fit and 2 test (by position inside the class). Rows = photo x 49 positions, pre-GELU.

Pass 1 (fit photos): running sums give, for each pair, the correlation of s, of p, and of the residual of s after p,
with every Base stage-4 neuron. These choose the control sets:
  nbr    for each child, the Base neuron in the same block whose correlation with p is closest to the child's,
         among neurons p barely moves (child score below its 50th largest)
  resid  the 6 Base neurons (any block) that best correlate with s's residual after p   (a data-chosen ceiling)
  rand   6 random Base neurons in blocks 1-2
Pass 2 (all photos): stores the chosen neurons at all positions. Linear fits on fit rows, R^2 on test rows.
P4: for each child c, on c's top 1% test rows: the mean residual of s after p (in s's std units), against the same
for c's neighbourhood control. Top-1 of both models on the test photos is recorded.
"""
import json
import os
import time

import numpy as np
import torch

from cache import DEV, OUT, hook_stage4, load, net

torch.set_grad_enabled(False)


def photos(ph):
    tr, te = [], []
    for c in range(1000):
        w = np.where(ph.lab == c)[0]
        tr += w[[0, 2]].tolist()
        te += w[[1, 3]].tolist()
    return np.array(sorted(tr)), np.array(sorted(te))


def run(ph, ms, mb, st_s, st_b, idx, batch, fn):
    t0 = time.time()
    for k in range(0, len(idx), batch):
        ii = idx[k:k + batch]
        x = ph.tensor(ii).to(DEV)
        ys, yb = ms(x), mb(x)
        S0 = st_s[0].reshape(-1, 3072).float()
        B = torch.cat([st_b[b].reshape(-1, 4096).float() for b in range(3)], 1)  # rows x 12288
        fn(ii, ys, yb, S0, B)
        if (k // batch) % 10 == 0:
            print(f"  {k + len(ii)}/{len(idx)} {time.time() - t0:.0f}s", flush=True)


def main():
    kj = json.load(open(os.path.join(OUT, "kids.json")))
    ksc = np.load(os.path.join(OUT, "kids_scores.npy"))
    ph = net.Photos("read")
    tr, te = photos(ph)
    ms, mb = load("small"), load("base")
    st_s, st_b = {}, {}
    hook_stage4(ms, st_s)
    hook_stage4(mb, st_b)
    S_ix = torch.tensor([q["s"] for q in kj], device=DEV)
    P_ix = torch.tensor([q["p"] for q in kj], device=DEV)  # block 0 -> flat index = p
    nP = len(kj)
    acc = dict(n=0, sb=torch.zeros(12288, dtype=torch.float64), sbb=torch.zeros(12288, dtype=torch.float64),
               ss=torch.zeros(nP, dtype=torch.float64), sss=torch.zeros(nP, dtype=torch.float64),
               sxs=torch.zeros(nP, 12288, dtype=torch.float64), sxp=torch.zeros(nP, 12288, dtype=torch.float64))

    def f1(ii, ys, yb, S0, B):
        s = S0[:, S_ix]
        p = B[:, P_ix]
        acc["n"] += B.shape[0]
        acc["sb"] += B.sum(0).cpu().double()
        acc["sbb"] += (B * B).sum(0).cpu().double()
        acc["ss"] += s.sum(0).cpu().double()
        acc["sss"] += (s * s).sum(0).cpu().double()
        acc["sxs"] += (s.T @ B).cpu().double()
        acc["sxp"] += (p.T @ B).cpu().double()

    run(ph, ms, mb, st_s, st_b, tr, 50, f1)
    n = acc["n"]
    mb_, vb = acc["sb"] / n, (acc["sbb"] / n - (acc["sb"] / n) ** 2).clamp_min(1e-12)
    ms_, vs = acc["ss"] / n, (acc["sss"] / n - (acc["ss"] / n) ** 2).clamp_min(1e-12)
    cov_sb = acc["sxs"] / n - ms_[:, None] * mb_[None, :]
    cov_pb = acc["sxp"] / n - mb_[P_ix.cpu()][:, None] * mb_[None, :]
    rng = np.random.default_rng(0)
    sets = []
    for i, q in enumerate(kj):
        p = q["p"]
        corr_p = (cov_pb[i] / (vb.sqrt() * vb[p].sqrt())).numpy()
        beta = float(cov_sb[i, p] / vb[p])
        cov_r = cov_sb[i] - beta * cov_pb[i]
        var_r = vs[i] - beta * cov_sb[i, p]
        corr_r = (cov_r / (vb.sqrt() * var_r.clamp_min(1e-12).sqrt())).numpy()
        corr_r[p] = -np.inf
        kids = [4096 * d["block"] + d["c"] for d in q["kids"]]
        thr = np.sort(ksc[i])[-50]
        moved = {4096 + j for j in np.where(ksc[i] >= thr)[0]}
        nbr = []
        for c in kids:
            blk = c // 4096
            cand = np.array([j for j in range(blk * 4096, (blk + 1) * 4096) if j not in moved and j not in nbr])
            nbr.append(int(cand[np.argmin(np.abs(corr_p[cand] - corr_p[c]))]))
        resid = [int(j) for j in np.argsort(-corr_r)[:6]]
        rand = [int(j) for j in rng.choice(np.arange(4096, 12288), 6, replace=False)]
        sets.append(dict(kids=kids, nbr=nbr, resid=resid, rand=rand,
                         kid_corr_p=[float(corr_p[c]) for c in kids], nbr_corr_p=[float(corr_p[c]) for c in nbr],
                         resid_child_score=[float(ksc[i][j - 4096]) if j >= 4096 else None for j in resid],
                         resid_child_rank=[int((ksc[i] > ksc[i][j - 4096]).sum()) if j >= 4096 else None
                                           for j in resid]))
    need = sorted({j for i, q in enumerate(kj) for k in ("kids", "nbr", "resid", "rand") for j in sets[i][k]}
                  | {q["p"] for q in kj})
    col = {j: t for t, j in enumerate(need)}
    need_t = torch.tensor(need, device=DEV)
    allix = np.concatenate([tr, te])
    is_te = np.concatenate([np.zeros(len(tr), bool), np.ones(len(te), bool)])
    order = np.argsort(allix)
    allix, is_te = allix[order], is_te[order]
    store_s, store_b, rows_te, correct = [], [], [], {"small": 0, "base": 0}

    def f2(ii, ys, yb, S0, B):
        store_s.append(S0[:, S_ix].cpu().numpy())
        store_b.append(B[:, need_t].cpu().numpy())
        t = np.isin(ii, te)
        rows_te.append(np.repeat(t, 49))
        y = torch.tensor(ph.lab[ii], device=DEV)
        tt = torch.tensor(t, device=DEV)
        correct["small"] += int(((ys.argmax(1) == y) & tt).sum())
        correct["base"] += int(((yb.argmax(1) == y) & tt).sum())

    run(ph, ms, mb, st_s, st_b, allix, 50, f2)
    Sx = np.concatenate(store_s)
    Bx = np.concatenate(store_b)
    T = np.concatenate(rows_te)
    F = ~T

    def r2(y, cols):
        X = np.c_[np.ones(len(y)), Bx[:, [col[c] for c in cols]]]
        w, *_ = np.linalg.lstsq(X[F], y[F], rcond=None)
        e = y[T] - X[T] @ w
        return float(1 - e.var() / y[T].var()), w

    out = []
    for i, q in enumerate(kj):
        y = Sx[:, i]
        p = q["p"]
        r = dict(s=q["s"], p=p, fire=q["fire"], write=q["write"], kid_scores=[d["score"] for d in q["kids"]],
                 **{k: v for k, v in sets[i].items()})
        for name in ("p", "kids", "nbr", "resid", "rand"):
            cols = [p] if name == "p" else [p] + sets[i][name]
            r[f"r2_{name}"] = r2(y, cols)[0]
        r["r2_kids_resid"] = r2(y, [p] + sets[i]["kids"] + sets[i]["resid"])[0]
        _, w = r2(y, [p])
        res_ = y - np.c_[np.ones(len(y)), Bx[:, col[p]]] @ w
        res_ = res_ / y[F].std()
        p4 = []
        for c, c2 in zip(sets[i]["kids"], sets[i]["nbr"]):
            k_ = max(int(T.sum() * .01), 50)
            tc = np.where(T)[0][np.argsort(-Bx[T, col[c]])[:k_]]
            tn = np.where(T)[0][np.argsort(-Bx[T, col[c2]])[:k_]]
            p4.append(dict(kid=float(res_[tc].mean()), nbr=float(res_[tn].mean())))
        r["p4"] = p4
        out.append(r)
        print(f"s{q['s']}~p{p} fire {q['fire']:.2f}: R2 p {r['r2_p']:.3f} kids {r['r2_kids']:.3f} "
              f"nbr {r['r2_nbr']:.3f} rand {r['r2_rand']:.3f} resid {r['r2_resid']:.3f} | P4 kid "
              f"{np.mean([d['kid'] for d in p4]):+.3f} nbr {np.mean([d['nbr'] for d in p4]):+.3f}", flush=True)
    keys = ["r2_p", "r2_kids", "r2_nbr", "r2_rand", "r2_resid", "r2_kids_resid"]
    summ = {k: float(np.median([r[k] for r in out])) for k in keys}
    g = lambda k: np.array([r[k] - r["r2_p"] for r in out])
    summ["kids_beat_nbr"] = int((g("r2_kids") > g("r2_nbr")).sum())
    summ["kids_beat_rand"] = int((g("r2_kids") > g("r2_rand")).sum())
    summ["p4_kid_mean"] = float(np.mean([d["kid"] for r in out for d in r["p4"]]))
    summ["p4_nbr_mean"] = float(np.mean([d["nbr"] for r in out for d in r["p4"]]))
    summ["p4_kid_beats_nbr"] = int(sum(d["kid"] > d["nbr"] for r in out for d in r["p4"]))
    summ["p4_n"] = int(sum(len(r["p4"]) for r in out))
    summ["top1_test"] = {k: v / len(te) for k, v in correct.items()}
    rk = [x for r in out for x in r["resid_child_rank"] if x is not None]
    summ["resid_in_blocks_1_2_child_rank_median"] = float(np.median(rk)) if rk else None
    summ["resid_in_block0_share"] = float(np.mean([j < 4096 for r in out for j in r["resid"]]))
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, pairs=out), open(os.path.join(OUT, "pos.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
