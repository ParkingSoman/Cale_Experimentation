"""geometry step 3: compare reads on the rows where the pair is ON (s or p in its top 1%), not on all rows.

64 sampled pairs: 48 agreeing (p = fire best = write best), 16 non-agreeing (p = fire best; write best = p2 kept).
One pass, all 4,000 pos.photos photos (fit + test rows together), stage 4 block 0.
Per pair, over rows where gelu(h_s) > thr_s or gelu(h_p) > thr_p (thresholds = top-1% values from topsets fit):
  n, sum u_B, sum u_B u_B^T, sums of h_s, h_s^2, h_s u_B, h_p, h_p^2, h_s h_p, and the same over the rows split by
  who is on (s only, p only, both) as counts.
Saved: results/pair/geometry/onpass.npz (on-region covariance per pair in float16, ~135 MB).
"""
import os
import time

import numpy as np
import torch

from cache import DEV, OUT, load, net
from pos import photos

torch.set_grad_enabled(False)
GO = os.path.join(OUT, "geometry")


def main():
    ph = net.Photos("read")
    tr, te = photos(ph)
    allix = np.sort(np.concatenate([tr, te]))
    allix = allix[:int(os.environ.get("NPH", len(allix)))]
    Rf = np.load(os.path.join(OUT, "tools_peakR_b0.npy")).astype(np.float32)
    Rw = np.load(os.path.join(OUT, "writesR_b0.npy")).astype(np.float32)
    fb, wb = Rf.argmax(1), Rw.argmax(1)
    rng = np.random.default_rng(1)
    ag = rng.choice(np.where(fb == wb)[0], 48, replace=False)
    di = rng.choice(np.where(fb != wb)[0], 16, replace=False)
    S = np.concatenate([ag, di])
    Pp = fb[S]
    ts = np.load(os.path.join(OUT, "topsets.npz"))
    thr = ts["fit_vals"][:, -1].astype(np.float32)          # (7168,) Small 0..3071, Base 3072..
    thr_s = torch.tensor(thr[S], device=DEV)
    thr_p = torch.tensor(thr[3072 + Pp], device=DEV)
    ms, mb = load("small"), load("base")
    H = {}
    ms.features[7][0].block[3].register_forward_hook(lambda m, i, o: H.__setitem__("hs", o))
    mb.features[7][0].block[3].register_forward_hook(lambda m, i, o: H.__setitem__("hp", o))
    mb.features[7][0].block[2].register_forward_hook(lambda m, i, o: H.__setitem__("u", o))
    n = len(S)
    f = lambda *s: torch.zeros(*s, device=DEV, dtype=torch.float32)
    A = dict(n=f(n), su=f(n, 1024), suu=f(n, 1024, 1024), shs=f(n), shs2=f(n), shsu=f(n, 1024), shp=f(n),
             shp2=f(n), shshp=f(n), n_s=f(n), n_p=f(n), n_b=f(n))
    St, Pt = torch.tensor(S, device=DEV), torch.tensor(Pp, device=DEV)
    t0 = time.time()
    for k in range(0, len(allix), 50):
        ii = allix[k:k + 50]
        x = ph.tensor(ii).to(DEV)
        ms(x), mb(x)
        u = H["u"].reshape(-1, 1024).float()
        hs = H["hs"].reshape(-1, 3072).float()[:, St]       # (rows, n)
        hp = H["hp"].reshape(-1, 4096).float()[:, Pt]
        on_s = torch.nn.functional.gelu(hs) > thr_s
        on_p = torch.nn.functional.gelu(hp) > thr_p
        m = (on_s | on_p).float()                           # (rows, n)
        A["n"] += m.sum(0)
        A["n_s"] += (on_s & ~on_p).float().sum(0)
        A["n_p"] += (on_p & ~on_s).float().sum(0)
        A["n_b"] += (on_s & on_p).float().sum(0)
        A["su"] += m.T @ u
        for j in range(n):
            r = m[:, j].nonzero()[:, 0]
            if len(r):
                ur = u[r]
                A["suu"][j] += ur.T @ ur
        A["shs"] += (m * hs).sum(0)
        A["shs2"] += (m * hs * hs).sum(0)
        A["shsu"] += (m * hs).T @ u
        A["shp"] += (m * hp).sum(0)
        A["shp2"] += (m * hp * hp).sum(0)
        A["shshp"] += (m * hs * hp).sum(0)
        if (k // 50) % 10 == 0:
            print(f"  {k + len(ii)}/{len(allix)} {time.time() - t0:.0f}s", flush=True)
    out = {k: v.cpu().double().numpy() for k, v in A.items() if k != "suu"}
    nn = A["n"].cpu().double()
    mu = A["su"].cpu().double() / nn[:, None]
    C = A["suu"].cpu().double() / nn[:, None, None] - mu[:, :, None] * mu[:, None, :]
    out["Con"] = C.half().numpy()                           # on-region covariance of u_B, per pair
    out["S"], out["P"], out["P2"], out["n_agree"] = S, Pp, wb[S], np.array(48)
    np.savez(os.path.join(GO, "onpass.npz"), **out)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
