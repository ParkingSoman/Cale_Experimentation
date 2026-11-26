"""N6 E2: what does training's read turn reach for, in the essay's terms?

For a block b and the trained net T (same survivors as the build B): each survivor s has a Base read W1_B[s] and a
trained read W1_T[s]. Both are applied to BASE's block inputs u (so no upstream drift), peak over positions,
3 train photos per class.
  turn(s)      = 1 - cos(W1_T[s], W1_B[s])
  gained(s)    = top-200 photos of s under T's read that are not in its top-200 under Base's read
  owner d*     = the dropped neuron whose Base top-200 holds the largest share of gained(s)
Relations of d* to s (and of a random dropped neuron r, the control):
  co-firing    correlation of Base peaks (s, d*); sign < 0 = d* sits where s is off (negative-pole side)
  tree         s is among d*'s 8 tree parents (tree/abstree.py), i.e. the build already hands d*'s write to s
  gate         cosine of upstream attributions of s and d* (activation x gradient, 7 blocks back, on d*'s top photos)
  read move    cosine of (W1_T[s] - W1_B[s]) with W1_B[d*]
The essay reading (written before the run in NEURON-LOG): the turn merges siblings under one gate, so d* shares s's gate
(high attribution cosine), with no preference for co-firing, and the tree's own children are not the main owners.
Usage: python e2_reads.py --run T [--blocks 8,15,22,28] [--top 20]
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from e1_parents import Hooks, attribution
from train_ref import load_mlp_state

sys.path.insert(0, os.path.join(C.HERE, "..", "tree"))
from abstree import parents  # noqa: E402

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="T")
    ap.add_argument("--blocks", default="8,15,22,28")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    base = C.load("base")
    cal, tr = C.Cache("cal"), C.Cache("train")
    lab = C.net.Photos("read").lab
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    T = torch.load(os.path.join(C.OUT, f"{a.run}.pt"), weights_only=False)
    mT = copy.deepcopy(base)
    C.build(mT, calib, T["frac"], T["how"])
    load_mlp_state(mT, T["state"])
    S = T["S"]
    BLK = [int(b) for b in a.blocks.split(",")]
    cls = lab[tr.ix]
    rows = np.sort(np.concatenate([np.where(cls == c)[0][:3] for c in range(1000)]))
    BLb, BLt = C.blocks(base), C.blocks(mT)
    U = {}
    hs = [BLb[b].block[3].register_forward_hook(lambda mod, i, o, b=b: U.__setitem__(b, (i[0], o))) for b in BLK]
    PB = {b: [] for b in BLK}
    PT = {b: [] for b in BLK}
    for s0 in range(0, len(rows), 50):
        x, _ = tr.get(rows[s0:s0 + 50])
        C.tail(base, x)
        for b in BLK:
            u, o = U[b]
            PB[b].append(o.amax((1, 2)).cpu())
            PT[b].append(BLt[b].block[3](u).amax((1, 2)).cpu())
    for h in hs:
        h.remove()
    PB = {b: torch.cat(v).double() for b, v in PB.items()}
    PT = {b: torch.cat(v).double() for b, v in PT.items()}
    H = Hooks(base)
    out = dict(rows=[])
    for b in BLK:
        Sb = S[b].numpy()
        h = PB[b].shape[1]
        drop = np.setdiff1d(np.arange(h), Sb)
        WB = BLb[b].block[3].weight.detach().cpu().double()
        WT = BLt[b].block[3].weight.detach().cpu().double()
        dA = PT[b] - PB[b][:, Sb]                                     # change of each survivor's peak, per photo
        turn = dA.std(0) / PB[b][:, Sb].std(0).clamp_min(1e-9)
        mg, K = calib[b]
        _, Wb, _ = C.write_mat(BLb[b])
        v = K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)
        rank = torch.empty(h, dtype=torch.long)
        rank[torch.argsort(-v)] = torch.arange(h)
        par, _ = parents(K, rank)[8]
        Z = lambda A: (A - A.mean(0)) / A.std(0).clamp_min(1e-9)
        Zd = Z(PB[b][:, drop])                                        # dropped neurons' Base peaks, standardized
        topB = torch.argsort(-PB[b], 0)[:200].T.numpy()
        print(f"block {b}: turn (std change / std) median {float(turn.median()):.3f}, top-10% {float(turn.quantile(.9)):.3f}",
              flush=True)
        for k in torch.argsort(-turn)[:a.top].tolist():
            s = int(Sb[k])
            corr = (Z(dA[:, k:k + 1]) * Zd).mean(0)                   # corr of the change with each dropped neuron
            perm = torch.from_numpy(rng.permutation(len(rows)))
            null_max = float((Z(dA[perm, k:k + 1]) * Zd).mean(0).max())
            j = int(corr.argmax())
            dstar = int(drop[j])
            r = int(rng.choice(drop))
            row = dict(block=b, s=s, turn=float(turn[k]), dstar=dstar, owner_corr=float(corr[j]), null_max=null_max,
                       owner_corr_top5=float(corr.topk(5).values.mean()),
                       r=r, r_corr=float(corr[int(np.where(drop == r)[0][0])]),
                       self_corr=float((Z(dA[:, k:k + 1])[:, 0] * Z(PB[b][:, s:s + 1])[:, 0]).mean()))
            for name, d in (("dstar", dstar), ("r", r)):
                row[f"{name}_corr"] = float(np.corrcoef(PB[b][:, s].numpy(), PB[b][:, d].numpy())[0, 1])
                row[f"{name}_tree"] = bool(s in par[d].tolist())
                row[f"{name}_readmove"] = float(F.cosine_similarity((WT[k] - WB[s])[None], WB[d][None]).item())
                x30, _ = tr.get(rows[topB[d][:30]])
                att = attribution(base, H, x30, b, list(range(max(0, b - 7), b)), (s, d))
                row[f"{name}_gate"] = float(att[s] @ att[d] / (att[s].norm() * att[d].norm()).clamp_min(1e-12))
            out["rows"].append(row)
            print(f"b{b} s{s} turn {row['turn']:.3f} owner d*{dstar} corr {row['owner_corr']:+.2f} (random "
                  f"{row['r_corr']:+.2f}, chance max {null_max:+.2f}, self {row['self_corr']:+.2f}) base corr s-d* {row['dstar_corr']:+.2f} "
                  f"tree {row['dstar_tree']} gate {row['dstar_gate']:+.2f} (random {row['r_gate']:+.2f}) "
                  f"readmove {row['dstar_readmove']:+.2f}", flush=True)
            json.dump(out, open(os.path.join(C.OUT, f"e2_reads_{a.run}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
