"""Distilled pair analysis: how is each coarse neuron built from Base's 4,096 neurons? (test rows, exact same input u)

Coarse MLP (k = 1,024, seeds 0 and 1) and Base block-0 MLP read the same u. On test rows:
  A_b = Base pre-activations (4,096), A_n = coarse pre-activations (1,024). "on" = pre-activation > 0.
1. Counterpart: each coarse neuron's best Base match by pre-activation correlation; and seed 0 vs seed 1 matches.
2. Members: Base neuron b is a member of coarse n if n covers b's firing: cover(n, b) = P(n on | b on).
   Each Base neuron's host = argmax_n cover. Report how many Base neurons each coarse neuron hosts, how complete the
   cover is, and which Base neurons have no good host.
3. Read rule: r_n against r_counterpart, against the mean of its members' reads (each read scaled to unit
   covariance-metric norm, weighted by the member's firing rate x write norm), and against random members.
4. Write rule: w_n against the sum of its members' writes weighted by each member's mean output on n's on-rows.
"""
import json
import os

import numpy as np
import torch

from cache import OUT, load

D = os.path.join(OUT, "distill")


def main():
    U = np.load(os.path.join(D, "U_test.npy")).astype(np.float32)
    m = load("base")
    blk = m.features[7][0]
    W1 = blk.block[3].weight.detach().cpu().double().numpy()
    b1 = blk.block[3].bias.detach().cpu().double().numpy()
    Wo = (blk.layer_scale.detach().cpu().double().view(-1, 1) * blk.block[5].weight.detach().cpu().double()).numpy().T
    C = np.cov(U[::5].T.astype(np.float64))
    L = np.linalg.cholesky(C + 1e-6 * np.eye(1024))
    nets = []
    for s in (0, 1):
        st = torch.load(os.path.join(D, f"mlp_k1024_s{s}.pt"), map_location="cpu")
        nets.append(dict(W1=st["l1.weight"].double().numpy(), b1=st["l1.bias"].double().numpy(),
                         W2=st["l2.weight"].double().numpy().T))
    Ab = U @ W1.T.astype(np.float32) + b1.astype(np.float32)
    onb = Ab > 0
    rate_b = onb.mean(0)
    gelu = lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    res = {}
    z = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-6)
    Zb = z(Ab)
    for si, nt in enumerate(nets):
        An = U @ nt["W1"].T.astype(np.float32) + nt["b1"].astype(np.float32)
        onn = An > 0
        Zn = z(An)
        R = Zn.T @ Zb / len(U)                       # (1024, 4096) pre-activation correlations
        cp = R.argmax(1)
        # cover(n, b) = P(n on | b on)
        cov_nb = (onn.T.astype(np.float32) @ onb.astype(np.float32)) / np.maximum(onb.sum(0), 1)
        host = cov_nb.argmax(0)
        hostc = cov_nb.max(0)
        live = onb.sum(0) >= 20                      # Base neurons with at least 20 on-rows in the test set
        members = {n: np.where((host == n) & live)[0] for n in range(1024)}
        nmem = np.array([len(members[n]) for n in range(1024)])
        # read rule
        unitM = lambda r: (L.T @ r) / np.linalg.norm(L.T @ r)
        cosM = lambda a, b: float(unitM(a) @ unitM(b))
        wnorm = np.linalg.norm(Wo, axis=1)
        rng = np.random.default_rng(si)
        rr = dict(cp=[], mem=[], mem_w=[], rand=[], cp_in_members=[])
        wr = dict(cp=[], mem=[], rand=[])
        for n in range(1024):
            M = members[n]
            if len(M) < 2:
                continue
            rn = nt["W1"][n]
            rr["cp"].append(cosM(rn, W1[cp[n]]))
            avg = sum(unitM(W1[b]) for b in M)
            rr["mem"].append(float(unitM(rn) @ avg / np.linalg.norm(avg)))
            wts = rate_b[M] * wnorm[M]
            avgw = sum(w * unitM(W1[b]) for w, b in zip(wts, M))
            rr["mem_w"].append(float(unitM(rn) @ avgw / np.linalg.norm(avgw)))
            Rm = rng.choice(np.where(live)[0], len(M), replace=False)
            avr = sum(unitM(W1[b]) for b in Rm)
            rr["rand"].append(float(unitM(rn) @ avr / np.linalg.norm(avr)))
            rr["cp_in_members"].append(int(cp[n] in set(M.tolist())))
            # write rule: members' writes weighted by their mean output on n's on-rows
            rows = onn[:, n]
            if rows.sum() < 20:
                continue
            g = gelu(Ab[rows][:, M].astype(np.float64)).mean(0)
            wsum = (g[:, None] * Wo[M]).sum(0)
            wn = nt["W2"][n]
            cs = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            wr["cp"].append(cs(wn, Wo[cp[n]]))
            wr["mem"].append(cs(wn, wsum))
            gr = gelu(Ab[rows][:, Rm].astype(np.float64)).mean(0)
            wr["rand"].append(cs(wn, (gr[:, None] * Wo[Rm]).sum(0)))
        res[f"seed{si}"] = dict(
            coarse_on_rate_median=float(np.median(onn.mean(0))), base_on_rate_median=float(np.median(rate_b)),
            counterpart_corr_median=float(np.median(R.max(1))),
            distinct_counterparts=int(len(set(cp.tolist()))),
            base_live=int(live.sum()), base_host_cover_median=float(np.median(hostc[live])),
            base_host_cover_over_05=float(np.mean(hostc[live] > .5)),
            members_per_coarse_q=[float(np.quantile(nmem, q)) for q in (.1, .5, .9)],
            coarse_with_no_member=int((nmem == 0).sum()),
            read_cos_median={k: float(np.median(v)) for k, v in rr.items()},
            write_cos_median={k: float(np.median(v)) for k, v in wr.items()},
            n_scored=len(rr["cp"]))
        np.savez(os.path.join(D, f"an_s{si}.npz"), cp=cp, host=host, hostc=hostc, live=live, Rmax=R.max(1),
                 nmem=nmem)
        if si == 0:
            An0, on0 = Zn, onn
        else:
            S = An0.T @ Zn / len(U)
            res["seed0_vs_seed1_best_corr_median"] = float(np.median(S.max(1)))
            res["seed0_vs_seed1_best_corr_q10_q90"] = [float(np.quantile(S.max(1), .1)), float(np.quantile(S.max(1), .9))]
            h0 = np.load(os.path.join(D, "an_s0.npz"))["host"]
            # do two Base neurons that share a host in seed 0 also share one in seed 1? (pairs among live)
            lv = np.where(live)[0]
            rng = np.random.default_rng(5)
            a, b = rng.choice(lv, 20000), rng.choice(lv, 20000)
            same0 = h0[a] == h0[b]
            same1 = host[a] == host[b]
            res["cohost_pairs_seed0"] = int(same0.sum())
            res["cohost_seed1_given_seed0"] = float(same1[same0].mean()) if same0.any() else None
            res["cohost_seed1_base_rate"] = float(same1.mean())
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(D, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
