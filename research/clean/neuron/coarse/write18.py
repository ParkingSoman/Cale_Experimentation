"""N11 investigation: what does B's block-0 write consist of? (cluster; measurements only, no build)
B = B k128 seed 0. Block 0 residual branch (what the block adds to the stream): r = bn2(conv2(g)), g = ReLU(bn1(conv1 x)).
A's and B's block 0 see the same input x and add to the same skip (A's frozen downsample), so r_A and r_B live in the
same 512 stream channels.
(a) copy: R^2 of r_B against r_A (pooled over channels, positions, photos), and per channel.
(b) the copying ruler: the least-squares map from B's firing g_B (128 x 3 x 3 neighbourhood) to A's conv2 output
    (pre-bn2, as B's conv2 produces pre-bn2B), solved on calibration photos and used only as a ruler: per B neuron,
    cosine between B's actual write (bn2B-scaled conv2 column, 512 x 9) and the ruler's column (bn2A-scaled), and the
    R^2 of r_A the ruler reaches from g_B.
(c) end: B with its block-0 writes replaced by the ruler (B's reads, B's block 1 kept): KL to A and top-1 on test photos,
    against B itself.
(d) readers: B's block-1 pre-activations, regressed on the stream that A's block 0 would give (h_A) versus on B's
    actual stream (h_B): share of variance of B's block-1 input explained by h_A alone (a linear check, per neuron
    median).
Saved: results/coarse/r18/write.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
NCAL = 3000
A = r18.load_a()
TAG = sys.argv[1] if len(sys.argv) > 1 else "B128aug_s0"
B = net_for(TAG)
a0, b0, b1 = A.layer4[0], B.layer4[0], B.layer4[1]
cal = r18.Cache("train")
crow = np.random.default_rng(0).choice(len(cal), NCAL, replace=False)


def parts(bl, x):
    g = torch.relu(bl.bn1(bl.conv1(x)))
    c = F.conv2d(g, bl.conv2.weight, padding=1)
    return g, c, bl.bn2(c)


def bn_scale(bn):
    return bn.weight / torch.sqrt(bn.running_var + bn.eps)


def main():
    res = {}
    # (b) the ruler: LS from unfold(g_B) to A's conv2 output c_A
    k = b0.conv2.in_channels
    nf = k * 9 + 1
    XtX = torch.zeros(nf, nf, dtype=torch.float64, device=r18.DEV)
    XtY = torch.zeros(nf, 512, dtype=torch.float64, device=r18.DEV)
    sAA = torch.zeros(512, dtype=torch.float64, device=r18.DEV); sA = torch.zeros_like(sAA); sBB = torch.zeros_like(sAA)
    sAB = torch.zeros_like(sAA); sB = torch.zeros_like(sAA); n = 0
    CAA = torch.zeros(512, 512, dtype=torch.float64, device=r18.DEV); CBB = torch.zeros_like(CAA)
    for a in range(0, NCAL, 100):
        x, _ = cal.get(crow[a:a + 100])
        gB, cB, rB = parts(b0, x)
        gA, cA, rA = parts(a0, x)
        U = F.unfold(gB, 3, padding=1).transpose(1, 2).reshape(-1, k * 9).double()
        U = torch.cat([U, torch.ones(len(U), 1, dtype=torch.float64, device=r18.DEV)], 1)
        Y = cA.permute(0, 2, 3, 1).reshape(-1, 512).double()
        XtX += U.T @ U; XtY += U.T @ Y
        ra = rA.permute(0, 2, 3, 1).reshape(-1, 512).double(); rb = rB.permute(0, 2, 3, 1).reshape(-1, 512).double()
        CAA += ra.T @ ra; CBB += rb.T @ rb
        sA += ra.sum(0); sAA += (ra * ra).sum(0); sB += rb.sum(0); sBB += (rb * rb).sum(0); sAB += (ra * rb).sum(0); n += len(ra)
    mA, mB = sA / n, sB / n
    varA = sAA / n - mA ** 2; varB = sBB / n - mB ** 2; cov = sAB / n - mA * mB
    corr_ch = (cov / (varA * varB).sqrt().clamp_min(1e-12)).cpu().numpy()
    resid = varA + varB - 2 * cov
    res["a_copy"] = dict(R2_pooled=float(1 - resid.sum() / varA.sum()), corr_per_channel_median=float(np.median(corr_ch)),
                         varB_over_varA=float(varB.sum() / varA.sum()))
    CA = CAA / n - torch.outer(mA, mA); CB = CBB / n - torch.outer(mB, mB)
    eA, VA = torch.linalg.eigh(CA); eB, VB = torch.linalg.eigh(CB)
    eA, VA, eB, VB = eA.flip(0), VA.flip(1), eB.flip(0), VB.flip(1)
    share = lambda e, k: float(e[:k].sum() / e.sum())
    PA = VA[:, :32] @ VA[:, :32].T
    res["e_spectrum"] = dict(A_top1=share(eA, 1), A_top10=share(eA, 10), A_top32=share(eA, 32),
                             B_top1=share(eB, 1), B_top10=share(eB, 10), B_top32=share(eB, 32),
                             B_var_inside_A_top32=float(torch.trace(PA @ CB) / torch.trace(CB)),
                             A_var_inside_A_top32=float(torch.trace(PA @ CA) / torch.trace(CA)),
                             mean_shift_norm_over_A_std=float((mB - mA).norm() / torch.trace(CA).sqrt()))
    lam = 1e-4 * float(XtX.diagonal().mean())
    sol = torch.linalg.solve(XtX + lam * torch.eye(nf, dtype=torch.float64, device=r18.DEV), XtY)
    Wr = sol[:-1].T.reshape(512, k, 3, 3).float(); br = sol[-1].float()
    # ruler R^2 on held-out calibration (last third) and per-neuron cosine between B's write and the ruler's
    sres = torch.zeros(512, dtype=torch.float64, device=r18.DEV); stot = torch.zeros_like(sres)
    hold = r18.Cache("held")
    for a in range(0, len(hold), 200):
        x, _ = hold.get(np.arange(a, min(a + 200, len(hold))))
        gB, _, _ = parts(b0, x); _, cA, rA = parts(a0, x)
        pred = a0.bn2(F.conv2d(gB, Wr, padding=1) + br.view(1, -1, 1, 1))
        sres += ((rA - pred) ** 2).sum((0, 2, 3)).double(); stot += ((rA - rA.mean((0, 2, 3), keepdim=True)) ** 2).sum((0, 2, 3)).double()
    res["b_ruler_R2_heldout"] = float(1 - sres.sum() / stot.sum())
    WB = (bn_scale(b0.bn2).view(512, 1, 1, 1) * b0.conv2.weight).flatten(2)          # 512 x k x 9
    WR = (bn_scale(a0.bn2).view(512, 1, 1, 1) * Wr).flatten(2)
    cosn = F.cosine_similarity(WB.permute(1, 0, 2).reshape(k, -1), WR.permute(1, 0, 2).reshape(k, -1), 1).cpu().numpy()
    res["b_cos_B_write_vs_ruler"] = dict(median=float(np.median(cosn)), n122=float(cosn[122]), q10=float(np.percentile(cosn, 10)),
                                         q90=float(np.percentile(cosn, 90)))
    # norm ratio (does B write bigger or smaller than the copy?)
    res["b_norm_ratio_B_over_ruler_median"] = float(np.median((WB.norm(dim=(0, 2)) / WR.norm(dim=(0, 2)).clamp_min(1e-9)).cpu().numpy()))
    # (c) end: B with block-0 writes replaced by the ruler (expressed in B's bn2 units: conv2_B' so that bn2_B(conv2_B' g) = bn2_A(ruler g))
    sB2, sA2 = bn_scale(b0.bn2), bn_scale(a0.bn2)
    Wswap = (sA2 / sB2).view(512, 1, 1, 1) * Wr
    bswap = (sA2 * (br - a0.bn2.running_mean) + a0.bn2.bias - b0.bn2.bias) / sB2 + b0.bn2.running_mean

    def B_swap(x):
        g = torch.relu(b0.bn1(b0.conv1(x)))
        r = b0.bn2(F.conv2d(g, Wswap, padding=1) + bswap.view(1, -1, 1, 1))
        h = torch.relu(r + b0.downsample(x))
        return r18.net.head(B, b1(h))

    lab_all = r18.net.Photos("read").lab
    te = r18.Cache("test")
    out = {"B": [0.0, 0], "B_ruler_writes": [0.0, 0]}
    hit = {"B": 0, "B_ruler_writes": 0}
    for a in range(0, len(te), 250):
        rr = np.arange(a, min(a + 250, len(te)))
        x, z = te.get(rr); lab = torch.as_tensor(lab_all[te.ix[rr]], device=r18.DEV)
        lp = F.log_softmax(z, 1); p = lp.exp()
        for name, fn in (("B", lambda x: r18.run4(B, x)), ("B_ruler_writes", B_swap)):  # "B" = the network under study (TAG)
            y = fn(x); out[name][0] += float((p * (lp - F.log_softmax(y, 1))).sum()); out[name][1] += len(x)
            hit[name] += int((y.argmax(1) == lab).sum())
    res["c_end"] = {k2: dict(kl=v[0] / v[1], top1=hit[k2] / v[1]) for k2, v in out.items()}
    # (d) readers: B's block-1 pre-activation explained by A's stream h_A vs B's own stream h_B (linear, center position)
    def stream_feats(m_block0, x):
        return torch.relu(m_block0(x))
    XA, XB, YB = [], [], []
    for a in range(0, len(te), 250):
        x, _ = te.get(np.arange(a, min(a + 250, len(te))))
        hA = a0(x); hB = b0(x)
        preB1 = b1.bn1(b1.conv1(hB))
        XA.append(F.unfold(hA, 3, padding=1)[:, :, 24].cpu()); XB.append(F.unfold(hB, 3, padding=1)[:, :, 24].cpu())
        YB.append(preB1[:, :, 3, 3].cpu())
    XA, XB, YB = torch.cat(XA).double(), torch.cat(XB).double(), torch.cat(YB).double()
    def r2(X, Y):
        nfit = len(X) // 2
        Xc = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1)
        G = Xc[:nfit].T @ Xc[:nfit]
        W = torch.linalg.solve(G + 1e-3 * G.diagonal().mean() * torch.eye(len(G), dtype=torch.float64), Xc[:nfit].T @ Y[:nfit])
        e = Y[nfit:] - Xc[nfit:] @ W
        return (1 - e.var(0) / Y[nfit:].var(0)).numpy()
    rA, rBs = r2(XA, YB), r2(XB, YB)
    res["d_readers"] = dict(B_block1_from_A_stream_median_R2=float(np.median(rA)), B_block1_from_B_stream_median_R2=float(np.median(rBs)),
                            note="block-1 pre-activation at the centre position, linear in the 3x3 x 512 stream patch; fit on half the test photos, scored on the other half")
    json.dump(res, open(os.path.join(r18.OUT, f"write_{TAG}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
