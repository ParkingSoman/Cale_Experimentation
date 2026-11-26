"""N11: read T0's dominant write direction, and whether B's block 1 passes B's large block-0 message to the class scores.
T0 = B128aug_s0_b0 (block 0 at 128 trained, A's block 1 fixed). r = block-0 residual branch output (512 x 7 x 7).
(1) v1 = top principal direction of r_T0 over calibration positions. Per held + test photo: projection p = mean over
    positions of (r_T0 - mean) . v1. Report: share of variance; cosine of v1 with A's top principal direction and with
    the mean of r_A; correlation of p with A's top-1 confidence and entropy; the classes with the highest and lowest mean p;
    direct effect of v1 on the class scores (final layer x v1, block 1 skipped): top and bottom classes; and the same
    direction's share in the difference r_T0 - r_A.
(2) B (both blocks trained): R^2 of B's final stream (input of the head, after block 1) against A's, and of B's
    block-0 stream h1 against A's h1, pooled over channels, positions and photos.
Saved: results/coarse/r18/dominant.json
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
A = r18.load_a(); T0 = net_for("B128aug_s0_b0"); B = net_for("B128aug_s0")
NAMES = r18.net.class_names()
cal = r18.Cache("train"); crow = np.random.default_rng(0).choice(len(cal), 3000, replace=False)


def resid(m, x):
    bl = m.layer4[0]
    return bl.bn2(bl.conv2(torch.relu(bl.bn1(bl.conv1(x)))))


def main():
    res = {}
    C = torch.zeros(512, 512, dtype=torch.float64, device=r18.DEV); s = torch.zeros(512, dtype=torch.float64, device=r18.DEV)
    CD = torch.zeros_like(C); sD = torch.zeros_like(s); CA = torch.zeros_like(C); sA = torch.zeros_like(s); n = 0
    for a in range(0, 3000, 150):
        x, _ = cal.get(crow[a:a + 150])
        rT, rA = resid(T0, x), resid(A, x)
        v = rT.permute(0, 2, 3, 1).reshape(-1, 512).double(); va = rA.permute(0, 2, 3, 1).reshape(-1, 512).double(); d = v - va
        C += v.T @ v; s += v.sum(0); CD += d.T @ d; sD += d.sum(0); CA += va.T @ va; sA += va.sum(0); n += len(v)
    m = s / n; Cv = C / n - torch.outer(m, m)
    mD = sD / n; CDv = CD / n - torch.outer(mD, mD); mA = sA / n; CAv = CA / n - torch.outer(mA, mA)
    e, V = torch.linalg.eigh(Cv); v1 = V[:, -1]
    eA, VA = torch.linalg.eigh(CAv); a1 = VA[:, -1]
    res["share_T0"] = float(e[-1] / e.sum())
    res["share_of_v1_in_T0_minus_A"] = float(v1 @ CDv @ v1 / torch.trace(CDv))
    res["cos_v1_A_top_direction"] = float(abs(v1 @ a1))
    res["cos_v1_mean_rA"] = float(v1 @ mA / mA.norm()); res["cos_v1_mean_rT0"] = float(v1 @ m / m.norm())
    fc = A.fc.weight.double()
    eff = (fc @ v1).cpu().numpy(); eff = eff - eff.mean()
    res["direct_up"] = [NAMES[i] for i in np.argsort(-eff)[:10]]; res["direct_down"] = [NAMES[i] for i in np.argsort(eff)[:10]]
    lab_all = r18.net.Photos("read").lab
    P, conf, ent, labs = [], [], [], []
    for sname in ("held", "test"):
        c = r18.Cache(sname)
        for a in range(0, len(c), 250):
            rr = np.arange(a, min(a + 250, len(c)))
            x, z = c.get(rr)
            rT = resid(T0, x).permute(0, 2, 3, 1).double()
            P.append(((rT - m) @ v1).mean((1, 2)).cpu().numpy())
            pz = F.softmax(z, 1); conf.append(pz.max(1).values.cpu().numpy()); ent.append(-(pz * pz.clamp_min(1e-12).log()).sum(1).cpu().numpy())
            labs.append(lab_all[c.ix[rr]])
    P, conf, ent, labs = map(np.concatenate, (P, conf, ent, labs))
    sign = 1.0 if np.corrcoef(P, conf)[0, 1] >= 0 else -1.0
    res["corr_p_with_A_confidence"] = float(np.corrcoef(P, conf)[0, 1]); res["corr_p_with_A_entropy"] = float(np.corrcoef(P, ent)[0, 1])
    cm = np.zeros(1000); np.add.at(cm, labs, P); cm /= np.bincount(labs, minlength=1000)
    res["classes_high_p"] = [NAMES[i] for i in np.argsort(-cm)[:12]]; res["classes_low_p"] = [NAMES[i] for i in np.argsort(cm)[:12]]
    # (2) B's streams against A's
    def r2(sx, sy):
        return float(1 - ((sx - sy) ** 2).sum() / ((sy - sy.mean((0, 2, 3), keepdim=True)) ** 2).sum())
    num = {"h1": [0.0, 0.0], "h2": [0.0, 0.0]}
    te = r18.Cache("test")
    for a in range(0, 4000, 250):
        x, _ = te.get(np.arange(a, a + 250))
        h1A, h1B = A.layer4[0](x), B.layer4[0](x); h2A, h2B = A.layer4[1](h1A), B.layer4[1](h1B)
        for k, (sb, sa) in (("h1", (h1B, h1A)), ("h2", (h2B, h2A))):
            num[k][0] += float(((sb - sa) ** 2).sum()); num[k][1] += float(((sa - sa.mean((0, 2, 3), keepdim=True)) ** 2).sum())
    res["B_stream_R2_vs_A"] = {k: 1 - v[0] / v[1] for k, v in num.items()}
    json.dump(res, open(os.path.join(r18.OUT, "dominant.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
