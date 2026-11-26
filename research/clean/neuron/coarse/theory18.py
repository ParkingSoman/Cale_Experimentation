"""N11 step 4: the theory check on two groups of A's stage-4 block-0 neurons (ResNet-18), on the cluster.

Groups (from A): G1 wheel = [79, 81, 37, 297, 496, 303, 458, 103] (step 2 reading); G2 furry face = the 8 children of
stage-3 channel 198 by A's own path data = [71, 440, 399, 252, 489, 57, 97, 162].
Block 0 of stage 4: pre = bn1(conv1 x), g = relu(pre) (512 neurons), r = bn2(conv2 g), h = relu(r + down x).
Versions of A, per group set S (G1, G2, G1 + G2):
  DEL  the children's g is replaced by its mean (per neuron and position) over calibration photos.
  R    the essay's coarse neuron, one per group, no fit:
         pre_R = sum of the children's pre + t, with t set so that P(pre_R > 0) = P(any child > 0) on calibration
               positions (one quantile);
         write_R = sum_c a_c * write_c, a_c = E[g_c] / E[g_R] over calibration positions (ratio of means);
         the children are removed (their g -> 0, their writes -> 0) and g_R writes write_R.
  LS   our old build form: children removed; every other block-0 neuron's write (conv2 column, 3 x 3) and a bias are
       re-solved by least squares to reproduce A's conv2 output on calibration photos (ridge 1e-4 x mean diagonal).
Calibration: 4,000 train photos (center crops). Scoring: the 19,000 held + test photos (no network trained on them).
Per photo and version: KL(A || X); e = centred logits(X) - centred logits(A); cos(e, centred logits(A)); gained mass
g_c = max(q_c - p_c, 0) and its share in the group's classes (classes where the children's mean class-profile z > 1.5).
Alignment of the two groups' errors per photo: |e1 + e2|^2 / (|e1|^2 + |e2|^2) and the real both-at-once error e12.
Saved: results/coarse/r18/theory.json (+ theory_rows.npz).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18

torch.set_grad_enabled(False)
G1 = [79, 81, 37, 297, 496, 303, 458, 103]
G2 = [71, 440, 399, 252, 489, 57, 97, 162]
NCAL = 4000


def parts(A):
    b0, b1 = A.layer4[0], A.layer4[1]
    return b0, b1


def run_from_g(A, x, g, W2=None, extra=None):
    """Stage 4 + head from x, with block 0's g given (and optionally a replacement conv2 weight and an added map)."""
    b0, b1 = parts(A)
    c = F.conv2d(g, b0.conv2.weight if W2 is None else W2, padding=1)
    if extra is not None:
        c = c + extra
    r = b0.bn2(c)
    h = torch.relu(r + b0.downsample(x))
    return r18.net.head(A, b1(h))


def main():
    A = r18.load_a()
    b0, _ = parts(A)
    ph = r18.net.Photos("read")
    lab_all = ph.lab
    cal = r18.Cache("train")
    rows = np.random.default_rng(0).choice(len(cal), NCAL, replace=False)

    # calibration statistics
    pre_l, g_l = [], []
    sum_g = torch.zeros(512, 7, 7, device=r18.DEV)
    for a in range(0, NCAL, 250):
        x, _ = cal.get(rows[a:a + 250])
        pre = b0.bn1(b0.conv1(x))
        sum_g += torch.relu(pre).sum(0)
    mean_g = sum_g / NCAL

    def coarse(G):
        """t (threshold shift) and a_c for the group's coarse neuron, from calibration photos."""
        anyon, sums = 0, []
        Ssum = []
        for a in range(0, NCAL, 250):
            x, _ = cal.get(rows[a:a + 250])
            pre = b0.bn1(b0.conv1(x))
            anyon += float((pre[:, G] > 0).any(1).float().sum())
            Ssum.append(pre[:, G].sum(1).flatten().cpu())
        S = torch.cat(Ssum)
        rate = anyon / S.numel()
        t = -float(torch.quantile(S[torch.randperm(len(S))[:2_000_000]], 1 - rate))
        Eg = torch.zeros(len(G), device=r18.DEV)
        Er = 0.0
        for a in range(0, NCAL, 250):
            x, _ = cal.get(rows[a:a + 250])
            pre = b0.bn1(b0.conv1(x))
            Eg += torch.relu(pre[:, G]).mean((0, 2, 3)) * len(x)
            Er += float(torch.relu(pre[:, G].sum(1) + t).mean()) * len(x)
        acoef = (Eg / Er)
        Wr = (b0.conv2.weight[:, G] * acoef.view(1, -1, 1, 1)).sum(1, keepdim=True)   # 512 x 1 x 3 x 3
        return dict(t=t, a=acoef.cpu().tolist(), Wr=Wr, rate=rate)

    def ls_weights(S):
        keep = [j for j in range(512) if j not in set(S)]
        nf = len(keep) * 9 + 1
        XtX = torch.zeros(nf, nf, dtype=torch.float64, device=r18.DEV)
        XtY = torch.zeros(nf, 512, dtype=torch.float64, device=r18.DEV)
        for a in range(0, NCAL, 100):
            x, _ = cal.get(rows[a:a + 100])
            g = torch.relu(b0.bn1(b0.conv1(x)))
            y = F.conv2d(g, b0.conv2.weight, padding=1)                       # target: A's conv2 output
            U = F.unfold(g[:, keep], 3, padding=1)                             # n x (k*9) x 49
            U = U.transpose(1, 2).reshape(-1, len(keep) * 9).double()
            U = torch.cat([U, torch.ones(len(U), 1, dtype=torch.float64, device=r18.DEV)], 1)
            Y = y.permute(0, 2, 3, 1).reshape(-1, 512).double()
            XtX += U.T @ U
            XtY += U.T @ Y
        lam = 1e-4 * float(XtX.diagonal().mean())
        sol = torch.linalg.solve(XtX + lam * torch.eye(nf, dtype=torch.float64, device=r18.DEV), XtY)
        W = sol[:-1].T.reshape(512, len(keep), 3, 3).float()                    # unfold order: channel-major, then 3 x 3
        W2 = torch.zeros_like(b0.conv2.weight)
        W2[:, keep] = W
        return W2, sol[-1].float().view(1, 512, 1, 1)

    versions = {}
    for name, S, Gs in (("G1", G1, [G1]), ("G2", G2, [G2]), ("G12", G1 + G2, [G1, G2])):
        versions[f"DEL_{name}"] = ("del", S, None)
        versions[f"R_{name}"] = ("R", S, [coarse(G) for G in Gs])
        versions[f"LS_{name}"] = ("ls", S, ls_weights(S))
        print("built", name, flush=True)

    def forward(kind, S, info, x):
        pre = b0.bn1(b0.conv1(x))
        g = torch.relu(pre)
        if kind == "del":
            g = g.clone(); g[:, S] = mean_g[S][None]
            return run_from_g(A, x, g)
        if kind == "ls":
            W2, bias = info
            g = g.clone(); g[:, S] = 0
            return run_from_g(A, x, g, W2=W2, extra=bias)
        g2 = g.clone(); g2[:, S] = 0
        extra = 0
        Gs = [G1, G2] if len(S) == 16 else [S]
        for G, c in zip(Gs, info):
            gR = torch.relu(pre[:, G].sum(1, keepdim=True) + c["t"])
            extra = extra + F.conv2d(gR, c["Wr"], padding=1)
        return run_from_g(A, x, g2, extra=extra)

    # class groups (children mean class-profile z > 1.5), from A's capture
    d = np.load(os.path.join(r18.OUT, "read_A.npz"))
    v = d["pos0"].astype(np.float32)
    cm = np.zeros((1000, 512), np.float32)
    np.add.at(cm, d["lab"], v)
    cm /= np.bincount(d["lab"], minlength=1000)[:, None]
    pz = (cm - cm.mean(0)) / cm.std(0)
    cls = {"G1": np.where(pz[:, G1].mean(1) > 1.5)[0], "G2": np.where(pz[:, G2].mean(1) > 1.5)[0]}

    out = {k: [] for k in ["lab"] + [f"{v}_{m}" for v in versions for m in ("kl", "cos", "in1", "in2", "en")]}
    E = {k: [] for k in ("e1R", "e2R", "e12R", "e1L", "e2L", "e12L", "e1D", "e2D", "e12D")}
    c1 = torch.zeros(1000, dtype=torch.bool, device=r18.DEV); c1[torch.as_tensor(cls["G1"])] = True
    c2 = torch.zeros(1000, dtype=torch.bool, device=r18.DEV); c2[torch.as_tensor(cls["G2"])] = True
    stats = {k: [] for k in ("R", "L", "D")}
    for s in ("held", "test"):
        c = r18.Cache(s)
        for a in range(0, len(c), 250):
            x, _ = c.get(np.arange(a, min(a + 250, len(c))))
            zA = r18.run4(A, x)
            p = F.softmax(zA, 1); lp = F.log_softmax(zA, 1)
            ca = zA - zA.mean(1, keepdim=True)
            out["lab"].append(lab_all[c.ix[a:a + len(x)]])
            errs = {}
            for vname, (kind, S, info) in versions.items():
                zX = forward(kind, S, info, x)
                lq = F.log_softmax(zX, 1)
                e = (zX - zX.mean(1, keepdim=True)) - ca
                errs[vname] = e
                gm = (lq.exp() - p).clamp_min(0)
                tot = gm.sum(1).clamp_min(1e-9)
                out[f"{vname}_kl"].append((p * (lp - lq)).sum(1).cpu().numpy())
                out[f"{vname}_cos"].append(F.cosine_similarity(e, ca, 1).cpu().numpy())
                out[f"{vname}_in1"].append((gm[:, c1].sum(1) / tot).cpu().numpy())
                out[f"{vname}_in2"].append((gm[:, c2].sum(1) / tot).cpu().numpy())
                out[f"{vname}_en"].append(e.pow(2).sum(1).cpu().numpy())
            for k, tag in (("R", "R"), ("L", "LS"), ("D", "DEL")):
                e1, e2, e12 = errs[f"{tag}_G1"], errs[f"{tag}_G2"], errs[f"{tag}_G12"]
                stats[k].append(torch.stack([e1.pow(2).sum(1), e2.pow(2).sum(1), (e1 + e2).pow(2).sum(1),
                                             e12.pow(2).sum(1), (e12 - e1 - e2).pow(2).sum(1),
                                             F.cosine_similarity(e1, e2, 1)], 1).cpu().numpy())
        print("scored", s, flush=True)
    res = {"groups": {"G1": G1, "G2": G2, "G1_classes": cls["G1"].tolist(), "G2_classes": cls["G2"].tolist()},
           "coarse": {k: {kk: vv for kk, vv in versions[k][2][0].items() if kk != "Wr"} for k in versions if k.startswith("R_G1") or k.startswith("R_G2")}}
    lab = np.concatenate(out["lab"])
    on1, on2 = np.isin(lab, cls["G1"]), np.isin(lab, cls["G2"])
    other = ~(on1 | on2)
    rows_out = {}
    for vname in versions:
        r = {}
        for m in ("kl", "cos", "in1", "in2", "en"):
            arr = np.concatenate(out[f"{vname}_{m}"])
            rows_out[f"{vname}_{m}"] = arr
            for sname, mask in (("G1photos", on1), ("G2photos", on2), ("other", other)):
                r[f"{m}_{sname}"] = float(np.median(arr[mask])) if m in ("cos",) else float(arr[mask].mean())
        res[vname] = r
    for k, tag in (("R", "R"), ("L", "LS"), ("D", "DEL")):
        st = np.concatenate(stats[k])
        n1, n2, n12add, n12, inter, cs = st.T
        ratio = n12add / np.clip(n1 + n2, 1e-9, None)
        res[f"align_{tag}"] = dict(ratio_mean=float(np.mean(n12add) / np.mean(n1 + n2)), ratio_median=float(np.median(ratio)),
                                   cos_e1_e2_median=float(np.median(cs)), real_over_added=float(np.mean(n12) / np.mean(n12add)),
                                   interaction_share=float(np.mean(inter) / np.mean(n12)))
        rows_out[f"align_{tag}"] = st
    np.savez(os.path.join(r18.OUT, "theory_rows.npz"), lab=lab, **rows_out)
    json.dump(res, open(os.path.join(r18.OUT, "theory.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("R_", "LS_", "DEL_", "align"))}, indent=0))


if __name__ == "__main__":
    main()
