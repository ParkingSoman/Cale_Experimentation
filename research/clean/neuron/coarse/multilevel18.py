"""N11 M1, M3-M5: coarsen A's stage 4 (ResNet-18) to K = 128 neurons per block, one level or two, on the cluster.

Families (from A alone, results/coarse/r18/): "cof" = co-firing families (complete linkage on 1 - corr of peak inputs,
both blocks); "rec" = shared-feeder families for block 0 (complete linkage on 1 - cosine of path feed0), co-firing for
block 1. 128 families per block.
Rule coarse neuron for family f (no fit):
  pre_f = sum of the members' pre (bn1 outputs): exactly linear in the block input, no new threshold;
  write W2_f = sum_c a_c W2[:, c], a_c = E[g_c] / E[g_f] (calibration means; g = ReLU output);
  one mean map per output channel and position keeps the block's mean conv2 output (conv2 is linear).
LS (old build form): keep the K neurons with the largest std(g_c) x |W2[:, c]|, re-solve their conv2 weights and a bias
  by least squares to A's conv2 output on A's own block input (ridge 1e-4 x mean diagonal).
Trained references: B128aug_s0_b0 (block 0 at K trained, block 1 = A) and B128aug_s0 (both blocks trained).
Calibration: 4,000 train photos. Scoring: 19,000 held + test photos.
Also: per-family errors at level 0 (rule and DEL) on the 2,000 held photos, to measure whether the 128 families' errors
add up or cancel at the class scores; the rule neurons' peak inputs (for matching with B on the Mac); firing rates.
Saved: results/coarse/r18/multilevel.json, multilevel_rows.npz, rule_pos_<fam>.npz
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
K, NCAL = 128, 4000
A = r18.load_a()
BL = list(A.layer4)
cal = r18.Cache("train")
crow = np.random.default_rng(0).choice(len(cal), NCAL, replace=False)
fa = np.load(os.path.join(r18.OUT, "families_A.npz"))
FAM = {"cof": (fa["b0_comp"], fa["b1_comp"]),
       "rec": (np.load(os.path.join(r18.OUT, "families_recipe_b0_complete.npy")), fa["b1_comp"])}


def skip_of(bl, h):
    return h if bl.downsample is None else bl.downsample(h)


def finish(bl, c, h):
    return torch.relu(bl.bn2(c) + skip_of(bl, h))


def cal_inputs(b):
    """A's own input to block b, for calibration photos, in batches."""
    for a in range(0, NCAL, 200):
        x, _ = cal.get(crow[a:a + 200])
        yield x if b == 0 else BL[0](x)


def rule_block(b, lab):
    bl = BL[b]
    M = torch.zeros(K, 512, device=r18.DEV)
    M[torch.as_tensor(lab, device=r18.DEV), torch.arange(512, device=r18.DEV)] = 1
    Eg = torch.zeros(512, 7, 7, device=r18.DEV)
    Egf = torch.zeros(K, 7, 7, device=r18.DEV)
    on = torch.zeros(K, device=r18.DEV)
    n = 0
    for h in cal_inputs(b):
        pre = bl.bn1(bl.conv1(h))
        g = torch.relu(pre)
        pf = torch.einsum("kc,nchw->nkhw", M, pre)
        Eg += g.sum(0); Egf += torch.relu(pf).sum(0); on += (pf > 0).float().mean((2, 3)).sum(0); n += len(h)
    Eg, Egf, rate = Eg / n, Egf / n, on / n
    ec, ef = Eg.mean((1, 2)), Egf.mean((1, 2))
    a = ec / ef[torch.as_tensor(lab, device=r18.DEV)].clamp_min(1e-6)
    W2 = bl.conv2.weight                                          # 512 x 512 x 3 x 3
    W2f = torch.einsum("ocij,kc->okij", W2, M * a[None, :])       # 512 x K x 3 x 3
    corr = F.conv2d(Eg[None], W2, padding=1)[0] - F.conv2d(Egf[None], W2f, padding=1)[0]
    return dict(M=M, W2f=W2f, corr=corr, a=a, Eg=Eg, Egf=Egf, rate=rate.cpu().numpy())


def ls_block(b):
    bl = BL[b]
    s1 = torch.zeros(512, device=r18.DEV, dtype=torch.float64); s2 = torch.zeros_like(s1); n = 0
    for h in cal_inputs(b):
        g = torch.relu(bl.bn1(bl.conv1(h))).double()
        s1 += g.sum((0, 2, 3)); s2 += (g * g).sum((0, 2, 3)); n += g.shape[0] * 49
    std = (s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt()
    val = std * bl.conv2.weight.flatten(1).norm(dim=1).double()
    keep = torch.argsort(val, descending=True)[:K]
    nf = K * 9 + 1
    XtX = torch.zeros(nf, nf, dtype=torch.float64, device=r18.DEV)
    XtY = torch.zeros(nf, 512, dtype=torch.float64, device=r18.DEV)
    for h in cal_inputs(b):
        g = torch.relu(bl.bn1(bl.conv1(h)))
        y = F.conv2d(g, bl.conv2.weight, padding=1)
        U = F.unfold(g[:, keep], 3, padding=1).transpose(1, 2).reshape(-1, K * 9).double()
        U = torch.cat([U, torch.ones(len(U), 1, dtype=torch.float64, device=r18.DEV)], 1)
        Y = y.permute(0, 2, 3, 1).reshape(-1, 512).double()
        XtX += U.T @ U; XtY += U.T @ Y
    lam = 1e-4 * float(XtX.diagonal().mean())
    sol = torch.linalg.solve(XtX + lam * torch.eye(nf, dtype=torch.float64, device=r18.DEV), XtY)
    return dict(keep=keep, W=sol[:-1].T.reshape(512, K, 3, 3).float(), bias=sol[-1].float().view(1, 512, 1, 1))


def fwd_block(b, h, kind, P):
    bl = BL[b]
    if kind == "A":
        return bl(h)
    pre = bl.bn1(bl.conv1(h))
    g = torch.relu(pre)
    if kind == "rule":
        gf = torch.relu(torch.einsum("kc,nchw->nkhw", P["M"], pre))
        return finish(bl, F.conv2d(gf, P["W2f"], padding=1) + P["corr"][None], h)
    if kind == "ls":
        return finish(bl, F.conv2d(g[:, P["keep"]], P["W"], padding=1) + P["bias"], h)
    raise ValueError(kind)


def main():
    R = {f: [rule_block(0, FAM[f][0]), rule_block(1, FAM[f][1])] for f in FAM}
    L = [ls_block(0), ls_block(1)]
    print("built rule and LS blocks", flush=True)
    T0, BB = net_for("B128aug_s0_b0"), net_for("B128aug_s0")
    head = lambda h: r18.net.head(A, h)
    V = {
        "L0_rule_cof": lambda x: head(BL[1](fwd_block(0, x, "rule", R["cof"][0]))),
        "L0_rule_rec": lambda x: head(BL[1](fwd_block(0, x, "rule", R["rec"][0]))),
        "L0_LS": lambda x: head(BL[1](fwd_block(0, x, "ls", L[0]))),
        "L0_trained": lambda x: r18.run4(T0, x),
        "L01_rule_cof": lambda x: head(fwd_block(1, fwd_block(0, x, "rule", R["cof"][0]), "rule", R["cof"][1])),
        "L01_rule_rec": lambda x: head(fwd_block(1, fwd_block(0, x, "rule", R["rec"][0]), "rule", R["rec"][1])),
        "L01_LS": lambda x: head(fwd_block(1, fwd_block(0, x, "ls", L[0]), "ls", L[1])),
        "L01_trained_B": lambda x: r18.run4(BB, x),
        "L1only_rule_cof": lambda x: head(fwd_block(1, BL[0](x), "rule", R["cof"][1])),
        "L1only_LS": lambda x: head(fwd_block(1, BL[0](x), "ls", L[1])),
    }
    th = json.load(open(os.path.join(r18.OUT, "theory.json")))
    groups = {"cars": r18.CARS, "wheel": th["groups"]["G1_classes"], "face": th["groups"]["G2_classes"]}
    lab_all = r18.net.Photos("read").lab
    rows = {k: {m: [] for m in ("kl", "hit", "cos", "spread")} for k in V}
    labs = []
    rule_pos = {f: [] for f in FAM}
    for s in ("held", "test"):
        c = r18.Cache(s)
        for a in range(0, len(c), 250):
            rr = np.arange(a, min(a + 250, len(c)))
            x, _ = c.get(rr)
            lab = lab_all[c.ix[rr]]; labs.append(lab)
            zA = r18.run4(A, x); p = F.softmax(zA, 1); lp = F.log_softmax(zA, 1); ca = zA - zA.mean(1, keepdim=True)
            for f in FAM:
                pre = BL[0].bn1(BL[0].conv1(x))
                rule_pos[f].append(torch.einsum("kc,nchw->nkhw", R[f][0]["M"], pre).amax((2, 3)).half().cpu().numpy())
            for k, fn in V.items():
                z = fn(x); lq = F.log_softmax(z, 1); cz = z - z.mean(1, keepdim=True)
                rows[k]["kl"].append((p * (lp - lq)).sum(1).cpu().numpy())
                rows[k]["hit"].append(z.argmax(1).cpu().numpy() == lab)
                rows[k]["cos"].append(F.cosine_similarity(cz - ca, ca, 1).cpu().numpy())
                rows[k]["spread"].append((cz.std(1) / ca.std(1)).cpu().numpy())
        print("scored", s, flush=True)
    labs = np.concatenate(labs)
    res = {"rates": {f: dict(block0_median=float(np.median(R[f][0]["rate"])), block1_median=float(np.median(R[f][1]["rate"]))) for f in FAM}}
    # B's block-0 firing rate on calibration photos, for the rule neurons' rate to compare with
    onB, nB = torch.zeros(K, device=r18.DEV), 0
    for h in cal_inputs(0):
        onB += (BB.layer4[0].bn1(BB.layer4[0].conv1(h)) > 0).float().mean((2, 3)).sum(0); nB += len(h)
    res["rates"]["B128_block0_median"] = float((onB / nB).median())
    out = {"lab": labs}
    for k in V:
        d = {m: np.concatenate(v) for m, v in rows[k].items()}
        out.update({f"{k}_{m}": v for m, v in d.items()})
        r = dict(kl=float(d["kl"].mean()), top1=float(d["hit"].mean()), cos_median=float(np.median(d["cos"])),
                 spread_median=float(np.median(d["spread"])))
        res[k] = r
    # in-group accuracy needs logits; recompute compactly for each version on group photos only
    ing = {k: {g: [0, 0] for g in groups} for k in V}
    for s in ("held", "test"):
        c = r18.Cache(s)
        lab = lab_all[c.ix]
        for g, cls in groups.items():
            idx = np.where(np.isin(lab, cls))[0]
            ci = torch.as_tensor(cls, device=r18.DEV)
            yi = torch.as_tensor([list(cls).index(int(v)) for v in lab[idx]], device=r18.DEV)
            for a in range(0, len(idx), 250):
                x, _ = c.get(idx[a:a + 250])
                for k, fn in V.items():
                    ing[k][g][0] += int((fn(x)[:, ci].argmax(1) == yi[a:a + 250]).sum()); ing[k][g][1] += len(x)
    for k in V:
        res[k]["in_group_acc"] = {g: v[0] / v[1] for g, v in ing[k].items()}
    # per-family errors at level 0 (rule, DEL), held photos: do the 128 families' errors add up or cancel?
    held = r18.Cache("held")
    for f in ("cof", "rec"):
        P = R[f][0]
        W2 = BL[0].conv2.weight
        acc = {m: [0.0, 0.0, 0.0] for m in ("rule", "del")}
        for a in range(0, len(held), 100):
            x, _ = held.get(np.arange(a, min(a + 100, len(held))))
            zA = r18.run4(A, x); ca = zA - zA.mean(1, keepdim=True)
            pre = BL[0].bn1(BL[0].conv1(x)); g = torch.relu(pre); cfull = F.conv2d(g, W2, padding=1)
            gf = torch.relu(torch.einsum("kc,nchw->nkhw", P["M"], pre))
            sums = {m: torch.zeros_like(ca) for m in acc}; sq = {m: torch.zeros(len(x), device=r18.DEV) for m in acc}
            for fam in range(K):
                mem = torch.where(P["M"][fam] > 0)[0]
                cm = F.conv2d(g[:, mem], W2[:, mem], padding=1)
                for m in acc:
                    if m == "rule":
                        corr_f = F.conv2d(P["Eg"][mem][None], W2[:, mem], padding=1)[0] - F.conv2d(P["Egf"][fam:fam + 1][None], P["W2f"][:, fam:fam + 1], padding=1)[0]
                        cnew = cfull - cm + F.conv2d(gf[:, fam:fam + 1], P["W2f"][:, fam:fam + 1], padding=1) + corr_f[None]
                    else:
                        cnew = cfull - cm + F.conv2d(P["Eg"][mem][None], W2[:, mem], padding=1)
                    z = r18.net.head(A, BL[1](finish(BL[0], cnew, x)))
                    e = (z - z.mean(1, keepdim=True)) - ca
                    sums[m] += e; sq[m] += e.pow(2).sum(1)
            for m in acc:
                if m == "rule":
                    zall = r18.net.head(A, BL[1](fwd_block(0, x, "rule", P)))
                else:
                    zall = r18.net.head(A, BL[1](finish(BL[0], F.conv2d(P["Eg"][None].expand(len(x), -1, -1, -1), W2, padding=1), x)))
                eall = (zall - zall.mean(1, keepdim=True)) - ca
                acc[m][0] += float(sums[m].pow(2).sum()); acc[m][1] += float(sq[m].sum()); acc[m][2] += float(eall.pow(2).sum())
        res[f"align_L0_{f}"] = {m: dict(added_over_singles=v[0] / v[1], real_over_added=v[2] / v[0]) for m, v in acc.items()}
        print("aligned", f, flush=True)
    np.savez(os.path.join(r18.OUT, "multilevel_rows.npz"), **out)
    for f in FAM:
        np.savez(os.path.join(r18.OUT, f"rule_pos_{f}.npz"), pos0=np.concatenate(rule_pos[f]))
    json.dump(res, open(os.path.join(r18.OUT, "multilevel.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
