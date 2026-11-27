"""ANSKEY (round L1, plan 2026-11-26 night): read one node law in the saved trained students.

Answer key = what training did. P students keep the A index of every neuron (<tag>_scores.pt, field kept),
so a kept node is compared with ITSELF in A. Photo use: reading only (test photos), no calculation uses them.

  gate   law "the gate opens wider when children are deleted". Per inner neuron (conv1 -> bn1 output of each
         block): firing rate (share of photo x cell > 0), mean firing, and the per-photo max correlation with the
         same neuron in A (does it keep its job?). Student vs A at the same index.

Usage: curves/.venv/bin/python -I distill/fanout/anskey.py gate W50P_s0_ph4 [--n 1000]
       tag "A" = A against itself (identity check: ratio 1, corr 1).
Output: fanout/out/anskey_gate_<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.dirname(HERE)
sys.path.insert(0, DIST)
import train_whole as TW  # noqa: E402

DEV = TW.DEV
torch.set_grad_enabled(False)


def inner_fire(m, x, bs=100):
    """Per block (s, b): bn1 output -> rate (C,), mean relu (C,), per-photo max (N, C)."""
    names = [(s, b) for s in range(4) for b in range(2)]
    acc = {k: dict(pos=0, cnt=0, sum=0, mx=[]) for k in names}
    hooks = []
    for (s, b) in names:
        blk = getattr(m, f"layer{s + 1}")[b]

        def hk(mod, inp, out, key=(s, b)):
            o = out.float()
            a = acc[key]
            a["pos"] = a["pos"] + (o > 0).float().sum((0, 2, 3))
            a["cnt"] += o.shape[0] * o.shape[2] * o.shape[3]
            a["sum"] = a["sum"] + o.clamp(min=0).sum((0, 2, 3))
            a["mx"].append(o.amax((2, 3)).cpu())
        hooks.append(blk.bn1.register_forward_hook(hk))
    for a0 in range(0, len(x), bs):
        m(x[a0:a0 + bs].float())
    for h in hooks:
        h.remove()
    out = {}
    for k, a in acc.items():
        out[k] = dict(rate=(a["pos"] / a["cnt"]).cpu(), mean=(a["sum"] / a["cnt"]).cpu(), mx=torch.cat(a["mx"]))
    return out


def corr_cols(u, v):
    u = u - u.mean(0)
    v = v - v.mean(0)
    return (u * v).sum(0) / (u.norm(dim=0) * v.norm(dim=0) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("law", choices=["gate", "carrier", "identity", "rate", "write"])
    ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--kept_from", default=None, help="with tag A: use this student's kept sets (A-in-A baseline)")
    a = ap.parse_args()
    if a.law == "write":
        return write_read(a)
    ph = TW.r18.net.Photos("read")
    test = TW.split(ph.lab)["test"]
    rows = test[::max(1, len(test) // a.n)][:a.n]
    x = TW.photos(ph, rows)
    A = TW.r18.load_a().to(DEV).eval()
    fA = inner_fire(A, x)
    if a.tag == "A" or a.law in ("identity", "rate"):
        kept = None
    if a.tag == "A":
        S = A
        if a.kept_from:
            kept = torch.load(os.path.join(TW.OUT, f"{a.kept_from}_scores.pt"), map_location="cpu", weights_only=False)["kept"]
    elif a.law in ("identity", "rate"):
        S, _ = TW.load_student(a.tag)
    else:
        S, _ = TW.load_student(a.tag)
        kept = torch.load(os.path.join(TW.OUT, f"{a.tag}_scores.pt"), map_location="cpu", weights_only=False)["kept"]
    fS = inner_fire(S, x)
    if a.tag == "A" and kept is not None:          # A-in-A baseline: A's own kept neurons as the "student"
        for (s_, b_), v in fS.items():
            ii = torch.as_tensor(kept["inner"][s_][b_]).long()
            fS[(s_, b_)] = dict(rate=v["rate"][ii], mean=v["mean"][ii], mx=v["mx"][:, ii])
    if a.law == "carrier":
        return carrier(a, fA, fS, kept, len(rows))
    if a.law == "identity":
        return identity(a, fA, fS, len(rows))
    if a.law == "rate":
        print(f"{'block':6} {'A n':>5} {'A rate':>7} {'S n':>5} {'S rate':>7}")
        out = []
        for (s_, b_), sa in fA.items():
            ss = fS[(s_, b_)]
            out.append(dict(block=f"s{s_ + 1}b{b_}", A=float(sa["rate"].mean()), S=float(ss["rate"].mean()), nS=len(ss["rate"])))
            print(f"s{s_ + 1}b{b_:<4} {len(sa['rate']):>5} {float(sa['rate'].mean()):7.3f} {len(ss['rate']):>5} {float(ss['rate'].mean()):7.3f}")
        json.dump(out, open(os.path.join(HERE, "out", f"anskey_rate_{a.tag}.json"), "w"), indent=1)
        return
    res = dict(tag=a.tag, photos=len(rows), photo_use="reading only (test photos)", blocks=[])
    for (s, b), sa in fA.items():
        idx = torch.arange(len(sa["rate"])) if kept is None else torch.as_tensor(kept["inner"][s][b]).long()
        cut = torch.ones(len(sa["rate"]), dtype=torch.bool)
        cut[idx] = False
        ss = fS[(s, b)]
        rA, rS = sa["rate"][idx], ss["rate"]
        ratio = (rS + 1e-6) / (rA + 1e-6)
        q1, q2 = rA.quantile(1 / 3), rA.quantile(2 / 3)
        terc = [float(ratio[rA <= q1].median()), float(ratio[(rA > q1) & (rA <= q2)].median()), float(ratio[rA > q2].median())]
        cc = corr_cols(sa["mx"][:, idx], ss["mx"])
        # full student x A correlation of per-photo max: own index vs best other A neuron vs a random pair
        za = sa["mx"] - sa["mx"].mean(0)
        za = za / (za.norm(dim=0) + 1e-12)
        zs = ss["mx"] - ss["mx"].mean(0)
        zs = zs / (zs.norm(dim=0) + 1e-12)
        C = zs.T @ za                                   # (kept, width_A)
        best = C.max(1)
        Ck, Cc = C[:, idx], C[:, cut]
        own = Ck[torch.arange(len(idx)), torch.arange(len(idx))]
        Ck_other = Ck.clone()
        Ck_other[torch.arange(len(idx)), torch.arange(len(idx))] = -2
        rnd = C.abs().flatten()
        res["blocks"].append(dict(
            block=f"s{s + 1}b{b}", kept=len(idx), width_A=len(sa["rate"]),
            rate_A_kept=float(rA.mean()), rate_S=float(rS.mean()), ratio_by_A_tercile=terc,
            rate_A_cut=float(sa["rate"][cut].mean()) if cut.any() else None,
            ratio_median=float(ratio.median()), share_wider=float((rS > rA).float().mean()),
            mean_A_kept=float(sa["mean"][idx].mean()), mean_S=float(ss["mean"].mean()),
            job_corr_median=float(cc.median()), job_corr_p10=float(cc.quantile(.1)),
            best_is_own=float((best.indices == idx).float().mean()),
            best_other_kept_median=float(Ck_other.max(1).values.median()),
            best_cut_median=float(Cc.max(1).values.median()) if cut.any() else None,
            share_cut_beats_own=float((Cc.max(1).values > own).float().mean()) if cut.any() else None,
            random_pair_abs_median=float(rnd.median())))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    p = os.path.join(HERE, "out", f"anskey_gate_{a.tag}{'_kept_' + a.kept_from if a.kept_from else ''}.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"{'block':6} {'kept':>9} {'rateA':>6} {'rateS':>6} {'rateCut':>7} {'ratio':>6} {'wider':>6} {'jobCorr':>7} {'p10':>6}")
    for r in res["blocks"]:
        rc = "" if r["rate_A_cut"] is None else f"{r['rate_A_cut']:.3f}"
        print(f"{r['block']:6} {r['kept']:>4}/{r['width_A']:<4} {r['rate_A_kept']:.3f}  {r['rate_S']:.3f}  {rc:>7} "
              f"{r['ratio_median']:6.2f} {r['share_wider']:6.2f} {r['job_corr_median']:7.3f} {r['job_corr_p10']:6.3f}")
    print(f"{'block':6} {'bestIsOwn':>9} {'own':>6} {'bestOtherKept':>13} {'bestCut':>7} {'cut>own':>7} {'|rand|':>6}")
    for r in res["blocks"]:
        bc = "" if r["best_cut_median"] is None else f"{r['best_cut_median']:.3f}"
        sc = "" if r["share_cut_beats_own"] is None else f"{r['share_cut_beats_own']:.2f}"
        print(f"{r['block']:6} {r['best_is_own']:9.2f} {r['job_corr_median']:6.3f} {r['best_other_kept_median']:13.3f} {bc:>7} {sc:>7} {r['random_pair_abs_median']:6.3f}")
    print("->", p)


def _z(m):
    m = m - m.mean(0)
    return m / (m.norm(dim=0) + 1e-12)


def carrier(a, fA, fS, kept, nph):
    """Law 'a deleted node's job goes to ONE kept node (its main parent / carrier)'.
    For each cut A neuron d: gain_j = corr(trained kept j, A d) - corr(A j, A d), per-photo max firing.
    Reports: share of cut neurons with a clear carrier (top gain > .1), how many kept neurons gain > .05,
    top1 / top2 gain, and whether the top carrier is d's nearest kept neuron already in A (the fold partner)."""
    res = dict(tag=a.tag, law="carrier", photos=nph, photo_use="reading only (test photos)", blocks=[])
    print(f"{'block':6} {'cut':>4} {'clear':>6} {'n>.05':>6} {'top1':>6} {'top2':>6} {'top=nearA':>9} {'hub%':>6} {'nullTop1':>8} {'null n>.05':>10}")
    for (s, b), sa in fA.items():
        idx = torch.as_tensor(kept["inner"][s][b]).long()
        cut = torch.ones(len(sa["rate"]), dtype=torch.bool)
        cut[idx] = False
        za = _z(sa["mx"])
        zs = _z(fS[(s, b)]["mx"])
        CS = zs.T @ za[:, cut]                          # (kept, cut) trained
        CA = za[:, idx].T @ za[:, cut]                  # (kept, cut) in A
        G = CS - CA
        g2 = G.topk(2, dim=0)
        top1, top2 = g2.values[0], g2.values[1]
        near = CA.argmax(0)
        same = (g2.indices[0] == near).float().mean()
        cnt = torch.bincount(g2.indices[0], minlength=len(idx)).float()
        hub = float(cnt.max() / cut.sum())
        GK = zs.T @ za[:, idx] - za[:, idx].T @ za[:, idx]      # null: same gain toward other KEPT neurons
        GK.fill_diagonal_(-2)
        gk2 = GK.topk(2, dim=0).values
        r = dict(block=f"s{s + 1}b{b}", cut=int(cut.sum()), clear=float((top1 > .1).float().mean()),
                 null_top1_median=float(gk2[0].median()), null_n_gain_05=float((GK > .05).float().sum(0).median()),
                 n_gain_05=float((G > .05).float().sum(0).median()), top1_median=float(top1.median()),
                 top2_median=float(top2.median()), top_is_nearest_in_A=float(same), hub_share=hub)
        res["blocks"].append(r)
        print(f"{r['block']:6} {r['cut']:>4} {r['clear']:6.2f} {r['n_gain_05']:6.1f} {r['top1_median']:6.3f} "
              f"{r['top2_median']:6.3f} {r['top_is_nearest_in_A']:9.2f} {100 * r['hub_share']:6.1f} {r['null_top1_median']:8.3f} {r['null_n_gain_05']:10.1f}")
    p = os.path.join(HERE, "out", f"anskey_carrier_{a.tag}.json")
    json.dump(res, open(p, "w"), indent=1)
    print("->", p)


def identity(a, fA, fS, nph, ks=(1, 2, 3, 5)):
    """Law 'a coarse node = ONE big-network node (morph) or the UNION of several (merge)'.
    Per student neuron j (same block position in A): match = corr of per-photo max firing.
    union_k = corr(student j, elementwise MAX of its top-k A matches, each standardized).
    Null: top-1 plus k-1 random A neurons of the same block (seed 0)."""
    g = torch.Generator().manual_seed(0)
    res = dict(tag=a.tag, law="identity", photos=nph, photo_use="reading only (test photos)", blocks=[])
    print(f"{'block':6} {'n':>4} " + " ".join(f"{'U' + str(k):>6}" for k in ks) + " " + " ".join(f"{'null' + str(k):>6}" for k in ks[1:]) + f" {'share U3>U1+.05':>15}")
    for (s, b), sa in fA.items():
        A_ = sa["mx"]
        A_ = (A_ - A_.mean(0)) / (A_.std(0) + 1e-12)
        za = _z(sa["mx"])
        zs = _z(fS[(s, b)]["mx"])
        C = zs.T @ za
        n, nA = C.shape
        top = C.topk(max(ks), dim=1).indices
        U = {}
        for k in ks:
            u = torch.stack([A_[:, top[j, :k]].amax(1) for j in range(n)], 1)
            U[k] = (zs * _z(u)).sum(0)
        Nl = {}
        for k in ks[1:]:
            cols = []
            for j in range(n):
                r = torch.randperm(nA, generator=g)[:k - 1]
                cols.append(A_[:, torch.cat([top[j, :1], r])].amax(1))
            Nl[k] = (zs * _z(torch.stack(cols, 1))).sum(0)
        r = dict(block=f"s{s + 1}b{b}", n=n, **{f"U{k}": float(U[k].median()) for k in ks},
                 **{f"null{k}": float(Nl[k].median()) for k in ks[1:]},
                 share_U3_beats_U1=float((U[3] > U[1] + .05).float().mean()))
        res["blocks"].append(r)
        print(f"{r['block']:6} {n:>4} " + " ".join(f"{r['U' + str(k)]:6.3f}" for k in ks) + " " +
              " ".join(f"{r['null' + str(k)]:6.3f}" for k in ks[1:]) + f" {r['share_U3_beats_U1']:15.2f}")
    p = os.path.join(HERE, "out", f"anskey_identity_{a.tag}.json")
    json.dump(res, open(p, "w"), indent=1)
    print("->", p)


def _wcol(blk, j_out=None):
    """BN-folded conv2 write of a block: (out, in, 3, 3) * bn2 scale per out channel."""
    bn = blk.bn2
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return blk.conv2.weight.detach() * sc[:, None, None, None]


def write_read(a):
    """Law 'a kept node carries its deleted siblings' messages' in WEIGHTS (no photos).
    Per block: dW_j = student write of kept inner j - A's write of j (rows = the kept stream channels, BN folded).
    share_cut = energy of dW_j inside the span of the CUT neurons' A writes; null share_kept = inside the span of an
    equal number of OTHER kept neurons' A writes; chance = span dim / column dim."""
    A = TW.r18.load_a().cpu().eval()
    S, _ = TW.load_student(a.tag)
    S = S.cpu().eval()
    kept = torch.load(os.path.join(TW.OUT, f"{a.tag}_scores.pt"), map_location="cpu", weights_only=False)["kept"]
    g = torch.Generator().manual_seed(0)
    res = dict(tag=a.tag, law="write", photo_use="none (weights)", blocks=[])
    print(f"{'block':6} {'|dW|/|W|':>8} {'shareCut':>8} {'shareKept':>9} {'chance':>6}")
    with torch.no_grad():
        for s in range(4):
            for b in range(2):
                bA, bS = getattr(A, f"layer{s + 1}")[b], getattr(S, f"layer{s + 1}")[b]
                ki = torch.as_tensor(kept["inner"][s][b]).long()
                ks = torch.as_tensor(kept["stream"][s]).long()
                WA = _wcol(bA)[ks]                       # (kept stream, inner A, 3, 3)
                WS = _wcol(bS)                           # (kept stream, kept inner, 3, 3)
                colA = WA.permute(1, 0, 2, 3).reshape(WA.shape[1], -1)     # (inner A, dim)
                colS = WS.permute(1, 0, 2, 3).reshape(WS.shape[1], -1)
                dW = colS - colA[ki]
                cut = torch.ones(colA.shape[0], dtype=torch.bool)
                cut[ki] = False
                nc = int(cut.sum())
                def share(Bset, dW=dW):
                    Q, _ = torch.linalg.qr(Bset.T.double())
                    pr = (dW.double() @ Q)
                    return (pr.pow(2).sum(1) / dW.double().pow(2).sum(1).clamp_min(1e-30))
                sc = share(colA[cut])
                # null: an equal number of other kept neurons (excluding j itself is skipped: one shared set)
                perm = ki[torch.randperm(len(ki), generator=g)[:min(nc, len(ki))]]
                sk = share(colA[perm])
                chance = min(nc, colA.shape[1]) / colA.shape[1]
                r = dict(block=f"s{s + 1}b{b}", rel_change=float(dW.norm() / colA[ki].norm()),
                         share_cut=float(sc.median()), share_kept_null=float(sk.median()), chance=chance)
                res["blocks"].append(r)
                print(f"{r['block']:6} {r['rel_change']:8.3f} {r['share_cut']:8.3f} {r['share_kept_null']:9.3f} {chance:6.3f}")
    p = os.path.join(HERE, "out", f"anskey_write_{a.tag}.json")
    json.dump(res, open(p, "w"), indent=1)
    print("->", p)


if __name__ == "__main__":
    main()
