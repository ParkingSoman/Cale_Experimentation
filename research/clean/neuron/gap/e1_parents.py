"""N6 E1: who are the absorption tree's parents, in the essay's terms?

For a dropped neuron d (keep 50% by value) and its top tree parent p (same block, higher rank, largest share
|c| x sd in the 8-parent OMP fit of tree/abstree.py), and a control q (a random kept neuron of higher rank), classify:
  (a) shared gate   an upstream neuron j (blocks b-7..b-1) whose removal (GELU output filled with its mean) drops d's
                    peak on d's top photos above the noise floor, AND drops p's peak on the same photos above the floor
  (b) poles         the sign of c, and where p's pre-activation peak sits on d's top photos (percentile of p's own)
                    c < 0 or p in its bottom 20% there = p keeps d's photos out (negative-pole sibling)
  (c) lineage       p broader than d on the same ground: share of d's top-50 photos inside p's top-200, and breadth
                    (classes in the top-200) of p against d
  (d) none of these
Photos: 3 train photos per class (3,000), peaks = max over positions of the pre-activation (block[3] output).
Removal tests on d's top 30 photos. Candidates j: top 20 by direct-path score |read_d . write_j| x mean g_j on those photos.
Noise floor: drops of d when 10 random upstream neurons are removed; floor = max(.25 std, their 99th percentile).
Usage: python e1_parents.py [--blocks 8,15,22,28] [--per 12] [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C

sys.path.insert(0, os.path.join(C.HERE, "..", "tree"))
from abstree import parents  # noqa: E402

torch.set_grad_enabled(False)


class Hooks:
    def __init__(self, m):
        self.BL = C.blocks(m)
        self.fill = None      # list of (block, neuron, value)
        self.keep, self.kept = None, {}
        self.rec = set()
        self.peaks = {}
        self.hs = []
        for g, blk in enumerate(self.BL):
            self.hs.append(blk.block[3].register_forward_hook(lambda mod, i, o, g=g: self._pre(g, o)))
            self.hs.append(blk.block[4].register_forward_hook(lambda mod, i, o, g=g: self._post(g, o)))

    def _pre(self, g, o):
        if g in self.rec:
            self.peaks[g] = o.amax((1, 2))
        return None

    def _post(self, g, o):
        if self.keep is not None and g in self.keep:
            self.kept[g] = o
        if self.fill:
            mine = [(j, v) for gg, j, v in self.fill if gg == g]
            if mine:
                o = o.clone()
                for j, v in mine:
                    o[..., j] = v
                return o
        return None


def attribution(m, H, x, b, UP, units):
    """For each unit n of block b: sum over photos of (g_j - mean) x d(peak_n)/d(g_j), for every upstream neuron j."""
    H.keep, H.kept, H.rec = set(UP), {}, {b}
    with torch.enable_grad():
        C.tail(m, x)
        outs = [H.kept[g] for g in UP]
        res = {}
        for n in units:
            gr = torch.autograd.grad(H.peaks[b][:, n].sum(), outs, retain_graph=True)
            res[n] = torch.cat([((o - o.mean((0, 1, 2))) * gg).sum((0, 1, 2)).detach().cpu().double()
                                for o, gg in zip(outs, gr)])
    H.keep, H.kept = None, {}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="8,15,22,28")
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        C.CACHE = os.path.join(C.OUT, "cache_smoke")
    rng = np.random.default_rng(0)
    m = C.load("base")
    cal, tr = C.Cache("cal"), C.Cache("train")
    lab = C.net.Photos("read").lab
    calib = C.calibrate(m, (x for x, _ in cal.batches(25)))
    # 3 train photos per class
    cls = lab[tr.ix]
    rows = np.sort(np.concatenate([np.where(cls == c)[0][:3] for c in range(1000)])) if not a.smoke else np.arange(len(tr))
    H = Hooks(m)
    BLK = [int(b) for b in a.blocks.split(",")]
    H.rec = set(BLK)
    P = {b: [] for b in BLK}
    for s in range(0, len(rows), 50):
        x, _ = tr.get(rows[s:s + 50])
        C.tail(m, x)
        for b in BLK:
            P[b].append(H.peaks[b].cpu())
    P = {b: torch.cat(v).double() for b, v in P.items()}          # (n photos, h) pre-activation peaks
    pcls = lab[tr.ix[rows]]
    out = dict(blocks={}, rows=[])
    t0 = time.time()
    for b in BLK:
        mg, K = calib[b]
        blk = H.BL[b]
        _, Wb, _ = C.write_mat(blk)
        h = K.shape[0]
        v = K.diagonal().clamp_min(0).sqrt() * Wb.norm(dim=0)
        order = torch.argsort(-v)
        rank = torch.empty(h, dtype=torch.long)
        rank[order] = torch.arange(h)
        par, coef = parents(K, rank)[8]
        sd = K.diagonal().clamp_min(1e-12).sqrt()
        dropped = order[h // 2:]
        ds = dropped[:a.per * 2][torch.from_numpy(rng.permutation(a.per * 2)[:a.per])]   # among the most valuable dropped
        W1 = blk.block[3].weight.detach().cpu().double()
        Pb = P[b]
        psd = Pb.std(0)
        for d in ds.tolist():
            share = coef[d].abs() * sd[par[d]]
            k = int(share.argmax())
            p, c = int(par[d, k]), float(coef[d, k])
            higher = order[:rank[d]].numpy()
            higher = higher[rank[higher].numpy() < h // 2]
            q = int(rng.choice([j for j in higher if j != p]))
            top_d = torch.argsort(-Pb[:, d])[:50].numpy()
            top30 = rows[top_d[:30]]
            row = dict(block=b, d=d, p=p, q=q, c=c, share=float(share[k] / share.sum().clamp_min(1e-12)))
            for name, j in (("p", p), ("q", q)):
                tj = set(torch.argsort(-Pb[:, j])[:200].numpy().tolist())
                row[f"{name}_overlap"] = float(np.mean([t in tj for t in top_d]))
                pct = (Pb[top_d, j][:, None] > Pb[:, j][None, :]).double().mean(1)
                row[f"{name}_pct_on_d"] = float(pct.median())
                row[f"{name}_breadth"] = len(set(pcls[list(tj)]))
            row["d_breadth"] = len(set(pcls[torch.argsort(-Pb[:, d])[:200].numpy()]))
            # removal tests
            x30, _ = tr.get(top30)
            H.rec = {b}
            H.fill = None
            C.tail(m, x30)
            base_pk = H.peaks[b].cpu().double()

            UP = list(range(max(0, b - 7), b))
            att = attribution(m, H, x30, b, UP, (d, p, q))              # {unit: (n upstream,) activation x gradient}
            cos = lambda u, v: float(u @ v / (u.norm() * v.norm()).clamp_min(1e-12))
            row["att_cos_dp"] = cos(att[d], att[p])
            row["att_cos_dq"] = cos(att[d], att[q])
            index = [(g, j) for g in UP for j in range(H.BL[g].block[3].out_features)]

            def group_drop(sel):
                H.fill = [(index[i][0], index[i][1], float(calib[index[i][0]][0][index[i][1]])) for i in sel]
                C.tail(m, x30)
                H.fill = None
                pk = H.peaks[b].cpu().double()
                return [float((base_pk[:, n] - pk[:, n]).mean() / psd[n]) for n in (d, p, q)]
            top = torch.argsort(-att[d])[:20].tolist()                   # d's 20 strongest positive feeders
            gd = group_drop(top)
            rnd = [group_drop(rng.choice(len(index), 20, replace=False).tolist()) for _ in range(3)]
            floor = max(.25, max(abs(r[0]) for r in rnd), max(abs(r[1]) for r in rnd), max(abs(r[2]) for r in rnd))
            row["gate_drop"] = dict(d=gd[0], p=gd[1], q=gd[2])
            row["random_drop"] = rnd
            row["floor"] = floor
            row["d_has_gate"] = gd[0] > floor
            row["p_shares_gate"] = gd[0] > floor and gd[1] > floor
            row["q_shares_gate"] = gd[0] > floor and gd[2] > floor
            out["rows"].append(row)
            print(f"b{b} d{d} p{p} c{c:+.2f} overlap p {row['p_overlap']:.2f} q {row['q_overlap']:.2f} "
                  f"att cos p {row['att_cos_dp']:+.2f} q {row['att_cos_dq']:+.2f} gate drop d/p/q {gd[0]:.2f}/{gd[1]:.2f}/{gd[2]:.2f} floor {floor:.2f} "
                  f"{time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(C.OUT, "e1_parents" + ("_smoke" if a.smoke else "") + ".json"), "w"),
                      indent=1)


if __name__ == "__main__":
    main()
