"""N6 step 3 (M3): where does each network put its error? Fisher directions of each block, from Base only.

Fisher of block g (Base, 1,000 calibration photos, 2 sampled classes per photo): the stream after block g is
z_g (d x H x W). For a class c drawn from Base's own probabilities, v = d log p_c / d z_g. Per position, the
d x d matrix F_g = E[ sum over positions of v v^T ]. Then, to second order, adding a small change e to the stream at
block g costs KL ~ 1/2 sum over positions of e^T F_g e. F_g is fixed by Base: no chain.
For B and T, on Base's block inputs (no drift), the block's output error e = y_student - y_base per position:
  plain     sum |e|^2 / sum |y_base - mean|^2              (the local error of the ledger)
  fisher    1/2 sum e^T F_g e  per photo                    (the KL this block's error would cost alone, to 2nd order)
  share     fraction of |e|^2 in the top 32 eigen-directions of F_g
And the drift of the stream (student network on its own inputs) at block g, split the same way.
W1 (prediction): T puts a smaller share of its error into the Fisher directions than B, in most blocks, even where
T's plain error is larger. Saved: results/gap/where_<run>.json.
Usage: python where.py --run T [--smoke]
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from train_ref import load_mlp_state

TOPK = 32


def fisher(base, cal, nsamp=2, limit=None):
    BL = C.blocks(base)
    Z = {}
    hs = [blk.register_forward_hook(lambda mod, i, o, g=g: (o.retain_grad(), Z.__setitem__(g, o))[1])
          for g, blk in enumerate(BL)]
    Fm = {g: torch.zeros(BL[g].block[5].out_features, BL[g].block[5].out_features, dtype=torch.float64)
          for g in range(len(BL))}
    n = 0
    gen = torch.Generator().manual_seed(0)
    for bi, (x, _) in enumerate(cal.batches(10)):
        if limit and bi >= limit:
            break
        for _ in range(nsamp):
            with torch.enable_grad():
                x_ = x.clone().requires_grad_(True)
                y = C.tail(base, x_)
                lp = F.log_softmax(y, 1)
                c = torch.multinomial(lp.exp().detach().cpu(), 1, generator=gen).to(y.device)
                lp.gather(1, c).sum().backward()
            for g in range(len(BL)):
                v = Z[g].grad.permute(0, 2, 3, 1).reshape(-1, Z[g].shape[1]).float()
                Fm[g] += (v.T @ v).cpu().double()
        n += x.shape[0] * nsamp
    for h in hs:
        h.remove()
    return {g: Fm[g] / n for g in Fm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="T")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    base = C.load("base")
    cal, te = C.Cache("cal"), C.Cache("test")
    path = os.path.join(C.OUT, "fisher.pt")
    if os.path.exists(path) and not a.smoke:
        Fm = torch.load(path)
    else:
        Fm = fisher(base, cal, limit=3 if a.smoke else None)
        if not a.smoke:
            torch.save(Fm, path)
    eig = {g: torch.linalg.eigh(Fm[g]) for g in Fm}
    V = {g: eig[g][1][:, -TOPK:].float().to(C.DEV) for g in Fm}
    Fd = {g: Fm[g].float().to(C.DEV) for g in Fm}
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    B = torch.load(os.path.join(C.OUT, f"{a.run}_build.pt"), weights_only=False)
    T = torch.load(os.path.join(C.OUT, f"{a.run}.pt"), weights_only=False)
    m = copy.deepcopy(base)
    C.build(m, calib, B["frac"], B["how"])
    BLb, BLm = C.blocks(base), C.blocks(m)
    res = dict(fisher_top_share={g: float(eig[g][0][-TOPK:].sum() / eig[g][0].sum()) for g in Fm})
    for name, st in (("B", B["state"]), ("T", T["state"])):
        load_mlp_state(m, st)
        U, Y, Sb, Sm = {}, {}, {}, {}
        hs = [blk.block[3].register_forward_hook(lambda mod, i, o, g=g: U.__setitem__(g, i[0])) for g, blk in enumerate(BLb)]
        hs += [blk.block[5].register_forward_hook(lambda mod, i, o, g=g: Y.__setitem__(g, o)) for g, blk in enumerate(BLb)]
        hs += [blk.register_forward_hook(lambda mod, i, o, g=g: Sb.__setitem__(g, o)) for g, blk in enumerate(BLb)]
        hs += [blk.register_forward_hook(lambda mod, i, o, g=g: Sm.__setitem__(g, o)) for g, blk in enumerate(BLm)]
        acc = {k: np.zeros(30) for k in ("e2", "y2", "efe", "etop", "d2", "dfd", "dtop", "s2")}
        nphot = 0
        with torch.no_grad():
            for bi, (x, _) in enumerate(te.batches(50)):
                if a.smoke and bi >= 2:
                    break
                C.tail(base, x)
                C.tail(m, x)
                nphot += x.shape[0]
                for g in range(30):
                    ls = BLb[g].layer_scale.view(-1)
                    yb = Y[g] * ls
                    e = (BLm[g].block[5](F.gelu(BLm[g].block[3](U[g]))) * ls - yb).reshape(-1, yb.shape[-1])
                    dd = (Sm[g] - Sb[g]).permute(0, 2, 3, 1).reshape(-1, yb.shape[-1])
                    acc["e2"][g] += float(e.pow(2).sum())
                    acc["y2"][g] += float((yb - yb.mean((0, 1, 2))).pow(2).sum())
                    acc["efe"][g] += float(((e @ Fd[g]) * e).sum()) / 2
                    acc["etop"][g] += float((e @ V[g]).pow(2).sum())
                    acc["d2"][g] += float(dd.pow(2).sum())
                    acc["dfd"][g] += float(((dd @ Fd[g]) * dd).sum()) / 2
                    acc["dtop"][g] += float((dd @ V[g]).pow(2).sum())
                    acc["s2"][g] += float(Sb[g].pow(2).sum())
        for h in hs:
            h.remove()
        res[name] = dict(plain=(acc["e2"] / acc["y2"]).tolist(), fisher_kl=(acc["efe"] / nphot).tolist(),
                         share_top=(acc["etop"] / acc["e2"]).tolist(), drift=np.sqrt(acc["d2"] / acc["s2"]).tolist(),
                         drift_fisher_kl=(acc["dfd"] / nphot).tolist(), drift_share_top=(acc["dtop"] / acc["d2"]).tolist())
        r = res[name]
        print(name, "plain", np.round(r["plain"], 3).tolist(), flush=True)
        print(name, "fisher_kl", np.round(r["fisher_kl"], 4).tolist(), flush=True)
        print(name, "share_top", np.round(r["share_top"], 3).tolist(), flush=True)
        print(name, "drift_fisher_kl", np.round(r["drift_fisher_kl"], 4).tolist(), flush=True)
    json.dump(res, open(os.path.join(C.OUT, f"where_{a.run}" + ("_smoke" if a.smoke else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
