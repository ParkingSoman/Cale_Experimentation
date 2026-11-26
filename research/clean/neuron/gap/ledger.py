"""N6 step 2: the gap ledger. Paste the trained network T's numbers into the build B one part at a time (and take them
out of T one part at a time), and score each hybrid by KL to Base on the 1,500 test photos.

Parts per block (same survivors, same shapes): reads W1, thresholds b1, writes W2, write bias b2.
Rows:
  B, T                         sanity: hybrid "none of T" = B, "all of T" = T
  reads / writes / b1 / b2     B + that part of T in every block;  T - that part (B's put back) in every block
  reads, writes re-solved      B + T's reads, writes re-solved by least squares on Base's inputs (the build's rule)
  block j                      B + all of T's block j;  T with B's block j
  local error (L1)             per block: |y - y_base|^2 / var(y_base) of B's and T's block on Base's inputs
  distance curve               after each block: |stream - Base stream| / |Base stream|, for B and T
Usage: python ledger.py --run T
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

torch.set_grad_enabled(False)
PARTS = {"reads": ("block.3.weight",), "b1": ("block.3.bias",), "writes": ("block.5.weight",), "b2": ("block.5.bias",)}


def block_keys(st, g):
    """st: [features5, features6, features7] state dicts. Return (dict, prefix) for global block g."""
    return (st[0], f"{g}.") if g < 27 else (st[2], f"{g - 27}.")


def mix(stB, stT, take):
    """take(g, part) -> True: use T's tensor. Other keys: B's."""
    out = [dict(d) for d in stB]
    for g in range(30):
        d, pre = block_keys(out, g)
        dT, _ = block_keys(stT, g)
        for part, keys in PARTS.items():
            if take(g, part):
                for k in keys:
                    d[pre + k] = dT[pre + k]
    return out


ROWS = None


def score(m, st, te, lab):
    load_mlp_state(m, st)
    r, rows = C.evaluate(m, te, lab, rows_all=ROWS)
    return r, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="T")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global ROWS
    NB = 2 if a.smoke else 30
    if a.smoke:
        ROWS = list(range(0, 1500, 15))
    lab = C.net.Photos("read").lab
    te, cal = C.Cache("test"), C.Cache("cal")
    base = C.load("base")
    B = torch.load(os.path.join(C.OUT, f"{a.run}_build.pt"), weights_only=False)
    T = torch.load(os.path.join(C.OUT, f"{a.run}.pt"), weights_only=False)
    m = copy.deepcopy(base)
    # give m the narrow shapes
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    C.build(m, calib, B["frac"], B["how"])
    stB, stT = B["state"], T["state"]
    res, rows = {}, {}

    def row(name, st):
        r, rw = score(m, st, te, lab)
        res[name] = r
        rows[name] = rw
        print(f"{name:32s} KL {r['kl']:.4f} top-1 {r['top1']:.3f}", flush=True)

    row("B", stB)
    row("T", stT)
    row("B+all(T)", mix(stB, stT, lambda g, p: True))
    for part in PARTS:
        row(f"B+{part}(T)", mix(stB, stT, lambda g, p, part=part: p == part))
        row(f"T-{part}", mix(stB, stT, lambda g, p, part=part: p != part))
    for g in range(NB):
        row(f"B+block{g}(T)", mix(stB, stT, lambda gg, p, g=g: gg == g))
        row(f"T-block{g}", mix(stB, stT, lambda gg, p, g=g: gg != g))

    # reads from T, writes re-solved by the build's rule on Base's inputs
    load_mlp_state(m, stT)
    U, Y = {}, {}
    hs = []
    BLb = C.blocks(base)
    acc = {g: None for g in range(30)}
    for g, blk in enumerate(BLb):
        hs.append(blk.block[3].register_forward_hook(lambda mod, i, o, g=g: U.__setitem__(g, i[0])))
        hs.append(blk.block[5].register_forward_hook(lambda mod, i, o, g=g: Y.__setitem__(g, o)))
    BLm = C.blocks(m)
    for x, _ in cal.batches(25):
        C.tail(base, x)
        for g in range(30):
            ls = BLb[g].layer_scale.view(-1)
            z = F.gelu(BLm[g].block[3](U[g])).reshape(-1, BLm[g].block[3].out_features)
            y = (Y[g] * ls).reshape(-1, ls.shape[0])
            zc = torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype, device=z.device)], 1)
            if acc[g] is None:
                acc[g] = [torch.zeros(zc.shape[1], zc.shape[1], dtype=torch.float64), torch.zeros(zc.shape[1], y.shape[1], dtype=torch.float64)]
            acc[g][0] += (zc.T @ zc).cpu().double()
            acc[g][1] += (zc.T @ y).cpu().double()
    for h in hs:
        h.remove()
    st = [dict(d) for d in stB]
    for g in range(30):
        d, pre = block_keys(st, g)
        dT, _ = block_keys(stT, g)
        A, Bm = acc[g]
        A = A + 1e-6 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
        W = torch.linalg.solve(A, Bm)                         # (k+1, d)
        ls = BLb[g].layer_scale.detach().view(-1).cpu().double()
        d[pre + "block.3.weight"] = dT[pre + "block.3.weight"]
        d[pre + "block.3.bias"] = dT[pre + "block.3.bias"]
        d[pre + "block.5.weight"] = (W[:-1].T / ls[:, None]).float()
        d[pre + "block.5.bias"] = (W[-1] / ls).float()
    row("B+reads,b1(T), writes re-solved", st)

    # L1 local error and the distance curve, on the test photos
    loc = {"B": np.zeros(30), "T": np.zeros(30)}
    dist = {"B": np.zeros(30), "T": np.zeros(30)}
    for name, stx in (("B", stB), ("T", stT)):
        load_mlp_state(m, stx)
        S_m, S_b = {}, {}
        hs = [blk.register_forward_hook(lambda mod, i, o, g=g: S_m.__setitem__(g, o)) for g, blk in enumerate(BLm)]
        hs += [blk.register_forward_hook(lambda mod, i, o, g=g: S_b.__setitem__(g, o)) for g, blk in enumerate(BLb)]
        U, Y = {}, {}
        hs += [blk.block[3].register_forward_hook(lambda mod, i, o, g=g: U.__setitem__(g, i[0])) for g, blk in enumerate(BLb)]
        hs += [blk.block[5].register_forward_hook(lambda mod, i, o, g=g: Y.__setitem__(g, o)) for g, blk in enumerate(BLb)]
        num_l, den_l, num_d, den_d = np.zeros(30), np.zeros(30), np.zeros(30), np.zeros(30)
        for bi, (x, _) in enumerate(te.batches(50)):
            if a.smoke and bi >= 2:
                break
            C.tail(base, x)
            C.tail(m, x)
            for g in range(30):
                ls = BLb[g].layer_scale.view(-1)
                yb = Y[g] * ls
                ys = BLm[g].block[5](F.gelu(BLm[g].block[3](U[g]))) * ls
                num_l[g] += float((ys - yb).pow(2).sum())
                den_l[g] += float((yb - yb.mean((0, 1, 2))).pow(2).sum())
                num_d[g] += float((S_m[g] - S_b[g]).pow(2).sum())
                den_d[g] += float(S_b[g].pow(2).sum())
        for h in hs:
            h.remove()
        loc[name] = num_l / den_l
        dist[name] = np.sqrt(num_d / den_d)
        print(name, "local error", np.round(loc[name], 3).tolist(), flush=True)
        print(name, "distance", np.round(dist[name], 3).tolist(), flush=True)
    res["local_error"] = {k: v.tolist() for k, v in loc.items()}
    res["distance"] = {k: v.tolist() for k, v in dist.items()}
    tag = a.run + ("_smoke" if a.smoke else "")
    json.dump(res, open(os.path.join(C.OUT, f"ledger_{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(C.OUT, f"ledger_{tag}_rows.npz"), **{k.replace(" ", "_"): v for k, v in rows.items()})


if __name__ == "__main__":
    main()
