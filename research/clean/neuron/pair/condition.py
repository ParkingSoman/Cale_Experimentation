"""Base-only rule test: is the coarse read = the Base counterpart's read with its "condition" removed?

Finding to test (N4 step 5): Small s = Base p generalized one step: same local pattern, p's object/class condition
dropped. If so, s's read should be p's read minus the part that reads "p's classes".

Pass (Base only, fit photos, 10 per class): per-class mean of u (Base block-0 input, LayerNorm output) and per-class
mean of every Base block-0 neuron's output after GELU.
Whitened coordinates use stream_cov.npz (C = L L^T): a read r becomes L^T r; s's read is v_s from stream_V.npy.
Rules, scored by cosine to L^T v_s (= covariance-metric cosine), over the 2,017 counterpart pairs:
  p          L^T r_p
  cond(l)    remove l x the component along p's condition direction delta_p = L^-1 sum_c w_c (mu_c - mu),
             w_c = p's mean output on class c (its own classes)        l = 0.5, 1
  lda(k)     remove the top-k between-class directions (all classes) from L^T r_p        k = 1, 5, 20, 100
Controls:   other(l)  remove another random pair's condition direction;   rand(k)  remove k random directions.
"""
import json
import os
import time

import numpy as np
import torch

from cache import DEV, OUT, load, net, split

torch.set_grad_enabled(False)


def main():
    ph = net.Photos("read")
    fit = split(ph.lab)
    ix = []
    for c in range(1000):
        w = np.where((ph.lab == c) & fit)[0]
        ix += w[:10].tolist()
    ix = np.array(sorted(ix))
    mb = load("base")
    H = {}
    mb.features[7][0].block[2].register_forward_hook(lambda m, i, o: H.__setitem__("u", o))
    mb.features[7][0].block[4].register_forward_hook(lambda m, i, o: H.__setitem__("g", o))
    su = torch.zeros(1000, 1024, dtype=torch.float64)
    sg = torch.zeros(1000, 4096, dtype=torch.float64)
    cnt = torch.zeros(1000, dtype=torch.float64)
    t0 = time.time()
    for k in range(0, len(ix), 50):
        ii = ix[k:k + 50]
        mb(ph.tensor(ii).to(DEV))
        lab = torch.tensor(ph.lab[ii])
        u = H["u"].float().mean((1, 2)).cpu().double()
        g = H["g"].float().mean((1, 2)).cpu().double()
        su.index_add_(0, lab, u)
        sg.index_add_(0, lab, g)
        cnt.index_add_(0, lab, torch.ones(len(ii), dtype=torch.float64))
        if (k // 50) % 40 == 0:
            print(f"  {k + len(ii)}/{len(ix)} {time.time() - t0:.0f}s", flush=True)
    mu_c = (su / cnt[:, None]).numpy()
    g_c = (sg / cnt[:, None]).numpy()
    np.savez(os.path.join(OUT, "condition_means.npz"), mu_c=mu_c, g_c=g_c)

    d = np.load(os.path.join(OUT, "stream_cov.npz"))
    C, W1, agree, pmap = d["Cuu"], d["W1"], d["agree"], d["pmap"]
    V = np.load(os.path.join(OUT, "stream_V.npy")).astype(np.float64)
    L = np.linalg.cholesky(C + 1e-6 * np.eye(1024))
    Linv = np.linalg.inv(L)
    mu = mu_c.mean(0)
    Dc = (mu_c - mu) @ Linv.T                      # class means in whitened coordinates (1000, 1024)
    B = Dc.T @ Dc / 1000                            # between-class covariance, whitened
    ev, E = np.linalg.eigh(B)
    E = E[:, ::-1]
    unit = lambda x: x / np.linalg.norm(x)
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    rng = np.random.default_rng(0)

    def cond_dir(p):
        w = np.clip(g_c[:, p], 0, None)
        w = w / max(w.sum(), 1e-12)
        return unit(w @ Dc)

    out = {k: [] for k in ["p", "cond_0.5", "cond_1", "other_0.5", "other_1"]
           + [f"lda_{k}" for k in (1, 5, 20, 100)] + [f"rand_{k}" for k in (1, 5, 20, 100)]}
    for s in agree:
        p = pmap[s]
        a = L.T @ W1[p]
        v = L.T @ V[s]
        out["p"].append(cos(a, v))
        dp = cond_dir(p)
        do = cond_dir(pmap[rng.choice(agree)])
        for lam in (0.5, 1):
            out[f"cond_{lam}"].append(cos(a - lam * (a @ dp) * dp, v))
            out[f"other_{lam}"].append(cos(a - lam * (a @ do) * do, v))
        for k in (1, 5, 20, 100):
            Ek = E[:, :k]
            out[f"lda_{k}"].append(cos(a - Ek @ (Ek.T @ a), v))
            R = np.linalg.qr(rng.standard_normal((1024, k)))[0]
            out[f"rand_{k}"].append(cos(a - R @ (R.T @ a), v))
    base = np.array(out["p"])
    res = {k: dict(median=float(np.median(x)), mean_gain=float(np.mean(np.array(x) - base)),
                   beats_p=float(np.mean(np.array(x) > base))) for k, x in out.items()}
    res["between_class_var_share_top"] = {k: float(ev[::-1][:k].sum() / ev.sum()) for k in (1, 5, 20, 100)}
    # how strongly p's read loads on its own condition direction (whitened cosine), against another's
    res["read_on_own_condition_median"] = float(np.median([abs(unit(L.T @ W1[pmap[s]]) @ cond_dir(pmap[s]))
                                                          for s in agree[:500]]))
    res["read_on_other_condition_median"] = float(np.median([abs(unit(L.T @ W1[pmap[s]]) @ cond_dir(pmap[rng.choice(agree)]))
                                                            for s in agree[:500]]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "condition.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
