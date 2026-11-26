# scale: neurons=s3:69, A:children; net=A; photos=held; where=cluster
# rung: Q=what the trained read added: does the new parent value follow the children's condition channels?; N=1; EST=3 min CPU; PREV=results/oc_morph_rw4.npz
"""Read the trained coarse read (LOG "Rung 2: train the read too").
On the 2,000 held photos, at every stage-3 cell: p_old = A's channel 69, p_new = the trained read's channel 69
(seed 0 and 1 of the given file). dp = p_new - p_old.
1. Correlation over cells of dp with each of the 256 stage-3 output channels (A's own values). Top 10 named.
2. The condition channels = each R1 child's largest non-69 positive feeder by weight (LOG "Children by the parent's
   write": s3:86, 139, 134, 92, 174, 71, 49, ...) taken from A's weights here (rows of stage-4 block-0 conv1).
   Their mean rank among the 256 by corr with dp, against the mean rank of 1,000 random sets of the same size.
3. Block-1 hidden neurons whose read weight changed most: which stage-3 channels they write to most (block-1 conv2).
Usage: python -u oc_rw_read.py <npz name in results/coarse/r18>
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import r18
import oc_morph_rw as rw
from capture18 import net_for

torch.set_grad_enabled(False)
P, K_N = 69, 11
DEV = r18.DEV


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "oc_morph_rw4.npz"
    d = np.load(os.path.join(r18.OUT, name))
    m = net_for("A").to(DEV).eval()
    b0 = m.layer4[0]
    s1, _ = rw.bn_parts(b0.bn1)
    s2, _ = rw.bn_parts(b0.bn2)
    R = (b0.conv1.weight * s1.view(-1, 1, 1, 1)).cpu().numpy()
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    wn = torch.sqrt((V ** 2).sum((0, 2, 3))).cpu().numpy()
    order = np.argsort(-(R[:, P].sum((1, 2)) * wn))
    kids = order[:K_N]
    feed = R[kids].sum((2, 3)) * wn[kids, None]                                   # 11 x 256
    feed[:, P] = -np.inf
    cond = sorted(set(int(np.argmax(f)) for f in feed))
    print("children", kids.tolist(), "-> top non-69 positive feeder of each:", [int(np.argmax(f)) for f in feed])
    held = rw.TCache("held")
    w0 = torch.as_tensor(d["w69_0"], device=DEV); b0_ = torch.as_tensor(d["b69_0"], device=DEV)
    for s in (0, 1):
        w = torch.as_tensor(d[f"rw{s}_w69"], device=DEV); bb = torch.as_tensor(d[f"rw{s}_b69"], device=DEV)
        num = np.zeros(256); xx = np.zeros(256); xs = np.zeros(256); dd = 0.0; ds = 0.0; n = 0
        po_x = np.zeros(256); po_d = 0.0; po_s = 0.0; po_p = 0.0
        for a in range(0, len(held), 250):
            x, _, t, s69 = held.get(np.arange(a, min(a + 250, len(held))))
            pold = torch.relu(F.conv2d(t, w0, padding=1) + b0_.view(1, 1, 1, 1) + s69)
            pnew = torch.relu(F.conv2d(t, w, padding=1) + bb.view(1, 1, 1, 1) + s69)
            dp = (pnew - pold).flatten().double().cpu().numpy()
            X = x.permute(1, 0, 2, 3).reshape(256, -1).double().cpu().numpy()
            num += X @ dp; xx += (X ** 2).sum(1); xs += X.sum(1); dd += (dp ** 2).sum(); ds += dp.sum(); n += dp.size
            po = pold.flatten().double().cpu().numpy()
            po_x += X @ po; po_d += po @ dp; po_s += po.sum(); po_p += po @ po
        cov = num / n - xs / n * ds / n
        corr = cov / np.sqrt((xx / n - (xs / n) ** 2) * (dd / n - (ds / n) ** 2) + 1e-12)
        corr[P] = np.nan
        # partial correlation with the old parent value removed (control: channels that simply co-fire with p)
        vx = xx / n - (xs / n) ** 2; vd = dd / n - (ds / n) ** 2; vp = po_p / n - (po_s / n) ** 2
        cxp = po_x / n - xs / n * po_s / n; cdp = po_d / n - ds / n * po_s / n
        rxp, rdp = cxp / np.sqrt(vx * vp + 1e-12), cdp / np.sqrt(vd * vp)
        pcorr = (corr - rxp * rdp) / np.sqrt((1 - rxp ** 2) * (1 - rdp ** 2) + 1e-12)
        pcorr[P] = np.nan
        prk = np.argsort(np.argsort(-np.nan_to_num(pcorr, nan=-9)))
        print(f"seed {s}: corr(dp, old p) {rdp:.2f}; partial (old p removed) top 10: "
              f"{[(int(c), round(float(pcorr[c]), 2)) for c in np.argsort(-np.nan_to_num(pcorr, nan=-9))[:10]]}")
        print(f"   condition ranks, partial: {[int(prk[c]) for c in cond]}, mean {prk[cond].mean():.1f}; "
              f"corr of each with old p: {[round(float(rxp[c]), 2) for c in cond]}")
        rk = np.argsort(np.argsort(-np.nan_to_num(corr, nan=-9)))
        top = np.argsort(-np.nan_to_num(corr, nan=-9))[:10]
        rng = np.random.default_rng(0)
        pool = np.array([c for c in range(256) if c != P])
        null = [rk[rng.choice(pool, len(cond), replace=False)].mean() for _ in range(1000)]
        print(f"seed {s}: top 10 stage-3 channels by corr with dp: {[(int(c), round(float(corr[c]), 2)) for c in top]}")
        print(f"   condition channels {cond}: ranks {[int(rk[c]) for c in cond]}, mean {rk[cond].mean():.1f}; "
              f"random sets: median {np.median(null):.1f}, 1st pct {np.percentile(null, 1):.1f}")
        dw = (d[f"rw{s}_w69"][0] - d["w69_0"][0])
        hid = np.argsort(-(dw ** 2).sum((1, 2)))[:6]
        b3 = m.layer3[1]
        s3s, _ = rw.bn_parts(b3.bn2)
        W2 = (b3.conv2.weight * s3s.view(-1, 1, 1, 1)).sum((2, 3)).cpu().numpy()  # 256 out x 256 hidden
        for h in hid:
            col = W2[:, h].copy(); col[P] = -np.inf
            print(f"   hidden h{int(h)} (read change sum {dw[h].sum():+.3f}): writes most to "
                  f"{[(int(c), round(float(W2[c, h]), 3)) for c in np.argsort(-col)[:4]]}")


if __name__ == "__main__":
    main()
