"""Pair study, steps 2 and 3: orphans, and each matched parent's children in Base.

Counterpart (from agree.py): a Small neuron s and the Base neuron p that is BOTH its best firing match (per-photo peaks)
and its best writing match (class-score effect). Chance of that agreement: 1/4096.
Orphan: a Base neuron that is neither the best firing match nor the best writing match of any Small neuron.

Child score (Base, mechanical, no photos): parent p in block k writes w_p = layer_scale_k * W2_k[:, p] into the stream.
A neuron c in a later stage-4 block k' reads LN(dwconv(stream)) through W1_k'. Per channel the dwconv scales a
spatially flat signal by its kernel sum, and the LayerNorm centres and scales by ln_w (and divides by a per-position std,
common to all channels). So the virtual weight is
    VW[p -> c] = W1_k'[c] . (ln_w_k' * ksum_k' * centre(w_p))
Children of p = later-block neurons with the largest positive VW, relative to all VW into c (its share).

Outputs results/pair/tree.json: per block, orphan counts and properties (P2), and for each confident pair the
top children with their VW, share, and whether each child is itself matched or an orphan.
"""
import json
import os

import numpy as np
import torch

from cache import NB, OUT, load

torch.set_grad_enabled(False)


def base_parts(m):
    P = []
    for blk in m.features[7]:
        dw = blk.block[0].weight.detach().cpu().double()  # (dim, 1, 7, 7)
        P.append(dict(ksum=dw.sum((1, 2, 3)), ln_w=blk.block[2].weight.detach().cpu().double(),
                      W1=blk.block[3].weight.detach().cpu().double(),  # (4dim, dim)
                      w=(blk.layer_scale.detach().cpu().double().view(-1, 1)
                         * blk.block[5].weight.detach().cpu().double())))  # (dim, 4dim) write columns
    return P


def vw(P, k, k2):
    w = P[k]["w"]
    w = w - w.mean(0, keepdim=True)
    r = P[k2]["ln_w"].view(-1, 1) * P[k2]["ksum"].view(-1, 1) * w  # (dim, 4dim parents)
    return (P[k2]["W1"] @ r).T  # (parents, children)


def main():
    pk = np.load(os.path.join(OUT, "tools_peaks_3000.npz"))
    peaks_s = pk["small"].astype(np.float32)
    peaks_b = pk["base"].astype(np.float32)
    P = base_parts(load("base"))
    res = {}
    match = {}
    for b in range(NB):
        Rf = np.load(os.path.join(OUT, f"tools_peakR_b{b}.npy")).astype(np.float32)
        Rw = np.load(os.path.join(OUT, f"writesR_b{b}.npy")).astype(np.float32)
        af, aw = Rf.argmax(1), Rw.argmax(1)
        agree = np.where(af == aw)[0]
        picked = set(af.tolist()) | set(aw.tolist())
        orphan = np.array(sorted(set(range(4096)) - picked))
        uniq = len(set(af[agree].tolist()))
        # selectivity: share of photos whose peak is above half the neuron's 99th percentile
        pb = peaks_b[:, b]
        q = np.quantile(pb, .99, axis=0)
        sel = (pb > .5 * q).mean(0)
        matched_b = np.array(sorted(set(af[agree].tolist())))
        match[b] = dict(agree=agree, p=af[agree], fire=Rf.max(1)[agree], write=Rw.max(1)[agree], orphan=orphan)
        res[f"b{b}"] = dict(n_agree=int(len(agree)), distinct_base_in_agree=uniq, n_orphan=int(len(orphan)),
                            breadth_orphan_median=float(np.median(sel[orphan])),
                            breadth_matched_median=float(np.median(sel[matched_b])))
        # breadth of the Small neuron against its counterpart
        ps = peaks_s[:, b]
        qs = np.quantile(ps, .99, axis=0)
        sels = (ps > .5 * qs).mean(0)
        res[f"b{b}"]["breadth_small_over_base_median"] = float(np.median(sels[agree] / np.maximum(sel[af[agree]], 1e-6)))
        print(b, res[f"b{b}"], flush=True)

    # P2: are orphans children of matched parents? Compare their strongest incoming share from a matched parent
    for k2 in (1, 2):
        shares = {}
        for k in range(k2):
            V = vw(P, k, k2).numpy()  # (4096 parents, 4096 children)
            pos = np.clip(V, 0, None)
            share = pos / pos.sum(0, keepdims=True).clip(1e-12)
            shares[k] = share
        orphan = match[k2]["orphan"]
        matched = np.array(sorted(set(match[k2]["p"].tolist())))
        best_from_matched = np.zeros(4096)
        for k in range(k2):
            mp = np.array(sorted(set(match[k]["p"].tolist())))
            best_from_matched = np.maximum(best_from_matched, shares[k][mp].max(0))
        res[f"children_in_b{k2}"] = dict(
            orphan_top_share_from_matched_parent_median=float(np.median(best_from_matched[orphan])),
            matched_top_share_from_matched_parent_median=float(np.median(best_from_matched[matched])))
        print(k2, res[f"children_in_b{k2}"], flush=True)

    # step 3: children of each confident pair with a parent in block 0 or 1
    pairs = []
    for k in (0, 1):
        Vs = {k2: vw(P, k, k2).numpy() for k2 in range(k + 1, NB)}
        orph = {k2: set(match[k2]["orphan"].tolist()) for k2 in range(k + 1, NB)}
        matched_c = {k2: set(match[k2]["p"].tolist()) for k2 in range(k + 1, NB)}
        for s, p, f, w in zip(match[k]["agree"], match[k]["p"], match[k]["fire"], match[k]["write"]):
            kids = []
            for k2, V in Vs.items():
                pos = np.clip(V, 0, None)
                share = pos[p] / pos.sum(0).clip(1e-12)
                top = np.argsort(-V[p])[:5]
                kids += [dict(block=k2, c=int(c), vw=float(V[p, c]), share=float(share[c]),
                              orphan=int(c) in orph[k2], matched=int(c) in matched_c[k2]) for c in top]
            kids = sorted(kids, key=lambda d: -d["vw"])[:6]
            pairs.append(dict(block=k, s=int(s), p=int(p), fire=float(f), write=float(w), kids=kids))
    res["pairs"] = pairs
    res["pairs_summary"] = dict(n=len(pairs),
                                kid_orphan_share=float(np.mean([d["orphan"] for q in pairs for d in q["kids"]])),
                                kid_share_median=float(np.median([d["share"] for q in pairs for d in q["kids"]])))
    print(res["pairs_summary"], flush=True)
    json.dump(res, open(os.path.join(OUT, "tree.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
