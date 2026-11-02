# serves: TQ18
"""Part C. How far are real networks from the test of Theorem TQ18, and from a pole claim (Definition DQ13)? cache/map: 20000 validation crops, not N's training photos.
Estimator, fixed before the run. Nets: ResNet-18, ResNet-50. k = 14, 38.
1. Always-on read of mean z, L-BFGS to 1500 iterations, float64, from the logit PCA start. Rows are photos, p_x = 1/n. Gauge: V orthonormal, codes centred. Numbers g = |P_Z G|_op, sigma, rho, the stationarity errors, and the test ratio g / (rho min(1, sigma/2)). Also |G|_op, so that the share of the residual the read cannot reach shows. Also the value rho would need: g / min(1, sigma/2); and the share of all (photo, class) pairs with p q below that value.
2. Free codes on the class-row table (one row per class, mean softmax on the photos with that label), same fit, same numbers.
3. Pole form at K = C = 1000, k = 500, on the class-row table. A pairing is needed. This is a measure of the form, not a map: pair each class with the unpaired class it is most confused with, greedy, from the table. Fit at that pole span with free codes against the free fit of the same k, both from L-BFGS. Also the table's distance from Definition DQ13: at the span's fit, |s - q|_1 per row, and the same after removing the best single mu Pi* pattern.
Prediction, written before the run: the test ratio is above 1e3 at each row of 1 and 2, because rho is tiny on 1000 classes. The read's P_Z G is much smaller than G. At 3, the pole span is well above the free fit, and the pattern removes under a fifth of the residual.
Usage: pc_real_cert.py [--smoke] [--threads N]"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(a.threads)
torch.set_default_dtype(torch.float64)
ITERS = 30 if a.smoke else 1500


def fit(params, fn, iters):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)

    def cl():
        opt.zero_grad()
        v = fn()
        v.backward()
        for p in params:
            p.grad = p.grad.contiguous()
        return v
    opt.step(cl)
    return float(fn().detach())


def numbers(S, p, t, V, c, Qz):
    """Certificate numbers in the gauge of Definition DQ14. Qz: orthonormal basis of Z, or None for free codes."""
    mean = c.mean(0, keepdim=True)
    t = t + (mean @ V.T)[0]
    c = c - mean
    t = t - t.mean()
    Vc = V - V.mean(0, keepdim=True)
    t = t + (c.mean(0) * 0).sum()
    Q, Rr = torch.linalg.qr(Vc)
    c = c @ Rr.T
    q = torch.softmax(t + c @ Q.T, 1)
    G = p[:, None] * (S - q)
    if Qz is None:
        PG = G
    else:
        PG = Qz @ (Qz.T @ G)
    g = float(torch.linalg.matrix_norm(PG, 2))
    Gop = float(torch.linalg.matrix_norm(G, 2))
    sv = torch.linalg.svdvals(c)
    sigma = float(sv.min())
    rhoall = p[:, None] * q
    rho = float(rhoall.min())
    need = g / min(1.0, sigma / 2)
    return dict(g=g, G_op=Gop, read_share_of_G=g / Gop, sigma=sigma, rho=rho, ratio=g / (rho * min(1.0, sigma / 2)), rho_needed=need, share_pairs_below_needed=float((rhoall < need).double().mean()), q_min=float(q.min()), stat_mean=float(G.sum(0).abs().sum()), stat_c=float((c.T @ G).norm()), stat_V=float((PG @ Q).norm()), kl=float((p * (S * (S.clamp_min(1e-300).log() - q.log())).sum(1)).sum()), gap_premise=g < rho)


out = dict(read=[], table=[], pole=[])
for net in ("resnet18", "resnet50"):
    Z = np.load(os.path.join(HERE, f"cache/map/{net}_z.npy"), mmap_mode="r")
    Y = np.load(os.path.join(HERE, f"cache/map/{net}_y.npy"), mmap_mode="r")
    lab = np.load(os.path.join(HERE, f"cache/map/{net}_meta.npz"))["labels"]
    n = 1500 if a.smoke else Z.shape[0]
    Zt = torch.from_numpy(np.asarray(Z[:n]).astype(np.float64))
    Yt = torch.from_numpy(np.asarray(Y[:n]).astype(np.float64))
    lab = torch.from_numpy(lab[:n].astype(np.int64))
    Zt = (Zt - Zt.mean(0)) / (Zt.std(0) + 1e-4)
    ls = torch.log_softmax(Yt, 1)
    S = ls.exp()
    D = Yt - Yt.mean(1, keepdim=True)
    ybar = D.mean(0)
    D = D - ybar
    U = torch.linalg.eigh(D.T @ D / n)[1].flip(1)
    Qz = torch.linalg.qr(torch.cat([torch.ones(n, 1), Zt], 1))[0]
    p = torch.ones(n) / n
    for k in ((14,) if a.smoke else (14, 38)):
        t0 = time.time()
        Q0 = U[:, :k].contiguous()
        M = torch.linalg.lstsq(Zt, D @ Q0).solution.contiguous().requires_grad_(True)
        V = Q0.clone().requires_grad_(True)
        t = ybar.clone().requires_grad_(True)
        fit([V, M, t], lambda: (S * (ls - torch.log_softmax(t + (Zt @ M) @ V.T, 1))).sum(1).mean(), ITERS)
        row = dict(net=net, k=k, **numbers(S, p, t.detach(), V.detach(), (Zt @ M).detach(), Qz), sec=time.time() - t0)
        out["read"].append(row)
        print(row, flush=True)
    # class-row table
    ids = torch.unique(lab)
    sb = torch.stack([S[lab == c].mean(0) for c in ids])
    sb = sb.clamp_min(1e-12)
    sb = sb / sb.sum(1, keepdim=True)
    K = len(ids)
    pi = torch.ones(K) / K
    lb = sb.log()
    for k in ((14,) if a.smoke else (14, 38)):
        t0 = time.time()
        torch.manual_seed(k)
        V = (torch.randn(1000, k) * .3).requires_grad_(True)
        c = (torch.randn(K, k) * .3).requires_grad_(True)
        t = lb.mean(0).clone().requires_grad_(True)
        fit([V, c, t], lambda: (pi * (sb * (lb - torch.log_softmax(t + c @ V.T, 1))).sum(1)).sum(), ITERS)
        row = dict(net=net, k=k, **numbers(sb, pi, t.detach(), V.detach(), c.detach(), None), sec=time.time() - t0)
        out["table"].append(row)
        print(row, flush=True)
    if K == 1000:
        t0 = time.time()
        # the pole form at k = K/2
        conf = (sb + sb.T).clone()
        conf.fill_diagonal_(-1)
        order = torch.argsort(conf.flatten(), descending=True)
        pair = -np.ones(K, dtype=int)
        left = K
        for f in order.tolist():
            i, j = f // K, f % K
            if pair[i] < 0 and pair[j] < 0 and i != j:
                pair[i], pair[j] = j, i
                left -= 2
                if left == 0:
                    break
        k = K // 2
        Vp = torch.zeros(K, k)
        seen = {}
        for c_ in range(K):
            if c_ < pair[c_]:
                j = len(seen)
                seen[c_] = j
                Vp[c_, j], Vp[pair[c_], j] = 1, -1
        cp = (torch.zeros(K, k)).requires_grad_(True)
        tp = lb.mean(0).clone().requires_grad_(True)
        vp = fit([cp, tp], lambda: (pi * (sb * (lb - torch.log_softmax(tp + cp @ Vp.T, 1))).sum(1)).sum(), ITERS)
        q = torch.softmax(tp + cp @ Vp.T, 1).detach()
        Ps = torch.zeros(K, K)
        Ps[torch.arange(K), torch.from_numpy(pair)] = 1
        Pi = .5 * (torch.eye(K) + Ps) - torch.ones(K, K) / K
        res = sb - q
        mu = float((res * Pi).sum() / (Pi * Pi).sum())
        l1 = float(res.abs().sum(1).mean())
        l1after = float((res - mu * Pi).abs().sum(1).mean())
        torch.manual_seed(1)
        Vf = Vp.clone().requires_grad_(True)
        cf = cp.detach().clone().requires_grad_(True)
        tf = tp.detach().clone().requires_grad_(True)
        vf = fit([Vf, cf, tf], lambda: (pi * (sb * (lb - torch.log_softmax(tf + cf @ Vf.T, 1))).sum(1)).sum(), ITERS)
        Vr = (torch.randn(K, k) * .3).requires_grad_(True)
        cr = (torch.randn(K, k) * .3).requires_grad_(True)
        tr_ = lb.mean(0).clone().requires_grad_(True)
        vr = fit([Vr, cr, tr_], lambda: (pi * (sb * (lb - torch.log_softmax(tr_ + cr @ Vr.T, 1))).sum(1)).sum(), ITERS)
        row = dict(net=net, k=k, pole_span_fit=vp, free_from_pole=vf, free_random=vr, residual_l1=l1, mu_fit=mu, residual_l1_after_pattern=l1after, sec=time.time() - t0)
        out["pole"].append(row)
        print(row, flush=True)
    if not a.smoke:
        json.dump(out, open(os.path.join(HERE, "results_pc_real_cert.json"), "w"), indent=1)
print("done", flush=True)
