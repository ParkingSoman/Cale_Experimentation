"""Write-projection rule: is a coarse neuron's output = Base's outputs weighted by write overlap?

Base block-0 MLP output: y(u) = sum_b w_b g_b(u) (+ bias), g_b = GELU(r_b . u + c_b), w_b = layer_scale * W2[:, b].
Coarse layer with write matrix Wc (1024 x k). If coarse neuron n carries the part of y along its write, then
    g_n(u)  ~  T_n(u) = [pinv(Wc) y(u)]_n = sum_b coef[n, b] g_b(u),   coef = pinv(Wc) @ W_base   (k x 4096)
i.e. each Base neuron's output enters coarse neuron n in proportion to how its write projects on n's dual write.

On test rows, for the distilled coarse layers (k = 1024, seeds 0 and 1):
  corr(g_n, T_n)                    the rule (writes only; no fitting to g_n)
  corr(g_n, g_counterpart)          one Base neuron
  corr(g_n, T_n with Wc columns shuffled across neurons)   control
How many Base neurons carry the rule: the number needed for 50% / 90% of sum_b |coef[n,b]| * std(g_b).
Then a Base-only build: choose k write directions = writes of the k Base neurons with the largest
std(g_b) * |w_b| ("value"), set T = pinv(W_sel) y, fit each new neuron's read by least squares of T_n on u, and
score the layer: R^2 of y from W_sel GELU(read), against keeping the same k Base neurons unchanged.
"""
import json
import os

import numpy as np
import torch

from cache import OUT, load

D = os.path.join(OUT, "distill")
gelu = lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def corr_cols(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return (A * B).mean(0)


def main():
    U = np.load(os.path.join(D, "U_test.npy")).astype(np.float32)
    Ut = np.load(os.path.join(D, "U_train.npy"))[::4].astype(np.float32)
    Y = np.load(os.path.join(D, "Y_test.npy")).astype(np.float32)
    m = load("base")
    blk = m.features[7][0]
    W1 = blk.block[3].weight.detach().cpu().float().numpy()
    b1 = blk.block[3].bias.detach().cpu().float().numpy()
    Wb = (blk.layer_scale.detach().cpu().float().view(-1, 1) * blk.block[5].weight.detach().cpu().float()).numpy()  # (1024, 4096)
    G = gelu(U @ W1.T + b1)                          # Base outputs, test rows (n, 4096)
    res = {}
    rng = np.random.default_rng(0)
    for s in (0, 1):
        st = torch.load(os.path.join(D, f"mlp_k1024_s{s}.pt"), map_location="cpu", weights_only=True)
        Wc = st["l2.weight"].float().numpy()         # (1024, k)
        gn = gelu(U @ st["l1.weight"].float().numpy().T + st["l1.bias"].float().numpy())
        coef = np.linalg.pinv(Wc) @ Wb               # (k, 4096)
        T = G @ coef.T
        c_rule = corr_cols(gn, T)
        an = np.load(os.path.join(D, f"an_s{s}.npz"))
        c_cp = corr_cols(gn, G[:, an["cp"]])
        Wsh = Wc[:, rng.permutation(Wc.shape[1])]
        c_sh = corr_cols(gn, G @ (np.linalg.pinv(Wsh) @ Wb).T)
        sg = G.std(0)
        contrib = np.abs(coef) * sg[None, :]
        srt = -np.sort(-contrib, 1)
        cum = np.cumsum(srt, 1) / srt.sum(1, keepdims=True)
        n50 = (cum < .5).sum(1) + 1
        n90 = (cum < .9).sum(1) + 1
        top1 = contrib.argmax(1)
        res[f"seed{s}"] = dict(corr_rule_median=float(np.median(c_rule)), corr_rule_q10_q90=[float(np.quantile(c_rule, .1)), float(np.quantile(c_rule, .9))],
                               corr_counterpart_median=float(np.median(c_cp)), corr_shuffled_median=float(np.median(c_sh)),
                               rule_beats_counterpart=float(np.mean(c_rule > c_cp)),
                               base_for_50pct_median=float(np.median(n50)), base_for_90pct_median=float(np.median(n90)),
                               top_contributor_is_counterpart=float(np.mean(top1 == an["cp"])))
        print(s, res[f"seed{s}"], flush=True)

    # Base-only build at k = 1024
    k = 1024
    Gt = gelu(Ut @ W1.T + b1)
    value = Gt.std(0) * np.linalg.norm(Wb, axis=0)
    sel = np.argsort(-value)[:k]
    Wsel = Wb[:, sel]
    Yt = Gt @ Wb.T                                    # Base MLP output without its bias (train rows)
    Ttr = Yt @ np.linalg.pinv(Wsel).T                 # targets (n, k)
    X = np.c_[Ut, np.ones(len(Ut), np.float32)].astype(np.float64)
    R = np.linalg.solve(X.T @ X + 1e-3 * np.eye(1025), X.T @ Ttr.astype(np.float64))  # linear reads of the targets
    Yte = G @ Wb.T
    yv = ((Yte - Yte.mean(0)) ** 2).sum(1).mean()
    def r2(Yhat):
        e = Yte - Yhat
        e = e - e.mean(0)
        return float(1 - (e ** 2).sum(1).mean() / yv)
    Xte = np.c_[U, np.ones(len(U), np.float32)]
    pre = Xte @ R.astype(np.float32)
    res["build_k1024"] = dict(
        keep_top_value_unchanged=r2(G[:, sel] @ Wsel.T),
        keep_random_unchanged=r2(G[:, (rs := rng.choice(4096, k, replace=False))] @ Wb[:, rs].T),
        write_rule_linear_targets=r2(pre @ Wsel.T),
        write_rule_gelu_of_fitted_reads=r2(gelu(pre) @ Wsel.T),
        distilled_trained=[json.load(open(os.path.join(D, f"eval_k1024_s{s}.json")))["test_r2"] for s in (0, 1)])
    print(json.dumps(res["build_k1024"], indent=1))
    json.dump(res, open(os.path.join(D, "write_rule.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
