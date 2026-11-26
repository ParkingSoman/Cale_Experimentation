"""N7 P4 and the first direct test of the gate reading: do kept neurons of the vehicle tree change their gate decisions?

For each network X (B, T, and any lens builds given with --lens TAG) against Base, on vehicle photos (held + test, which
T never trained on) and on the same number of non-vehicle photos:
  members  = vehicle-tree neurons (N5, blocks 18-29) that X keeps (the same kept set in every network: value ranking)
  control  = the same number of random kept non-members per block (seed 0)
  For each neuron, Base's 90th and 99th percentile of its pre-activation over all rows (photo x position) of the photo set.
  cross90 / cross99: share of rows where X and Base fall on different sides of that level (X on its own stream)
  lost99: share of Base's rows above the 99th percentile where X is below it; gained99 the other way
  dpost:  mean |GELU(z_X) - GELU(z_Base)| / std(GELU(z_Base))
Saved: results/lens/gates.json, gates.log.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C  # noqa: E402
from train_ref import load_mlp_state  # noqa: E402

DEV = C.DEV
torch.set_grad_enabled(False)
OUT = os.path.join(HERE, "..", "results", "lens")
TREES = os.path.join(HERE, "..", "results", "tree", "trees.json")


def vehicle_ids():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcommon", os.path.join(HERE, "..", "tree", "common.py"))
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    return set(tc.groups()["vehicles"])


def photo_rows(lab, veh, want_vehicle, n_other=None, seed=0):
    out = {}
    for name in ("held", "test"):
        c = C.Cache(name)
        isv = np.isin(lab[c.ix], list(veh))
        out[name] = np.where(isv)[0] if want_vehicle else np.where(~isv)[0]
    if not want_vehicle:
        rng = np.random.default_rng(seed)
        out = {k: np.sort(rng.choice(v, n_other[k], replace=False)) for k, v in out.items()}
    return out


def preacts(m, idx_by_block, rows_by_cache, bs=25):
    """Pre-activations of chosen neurons (block -> narrow or Base indices), rows = photo x position. dict block -> (R, n)."""
    BL = C.blocks(m)
    store, hs = {}, []
    for g, ix in idx_by_block.items():
        t = torch.as_tensor(ix, device=DEV).long()
        hs.append(BL[g].block[3].register_forward_hook(
            lambda mod, i, o, g=g, t=t: store.setdefault(g, []).append(o[..., t].reshape(-1, len(t)).cpu())))
    for name, rows in rows_by_cache.items():
        c = C.Cache(name)
        for a in range(0, len(rows), bs):
            x, _ = c.get(rows[a:a + bs])
            C.tail(m, x)
    for h in hs:
        h.remove()
    return {g: torch.cat(v) for g, v in store.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", nargs="*", default=[], help="tags of saved lens builds (results/lens/<tag>.pt)")
    a = ap.parse_args()
    lab = C.net.Photos("read").lab
    base = C.load("base")
    calib = C.calibrate(base, (x for x, _ in C.Cache("cal").batches(25)))
    tree = json.load(open(TREES))["vehicles"]["levels"]
    nets = {}
    mB = copy.deepcopy(base)
    Ss = C.build(mB, calib, 0.5, "value")
    nets["B"] = mB
    ck = torch.load(os.path.join(C.OUT, "T.pt"), weights_only=False)
    mT = copy.deepcopy(base)
    C.build(mT, calib, ck["frac"], ck["how"])
    load_mlp_state(mT, ck["state"])
    nets["T"] = mT
    for tag in a.lens:
        ckl = torch.load(os.path.join(OUT, f"{tag}.pt"), weights_only=False)
        mL = copy.deepcopy(base)
        C.build(mL, calib, 0.5, "value")
        load_mlp_state(mL, ckl["state"])
        mL.classifier.load_state_dict(ckl["head"])
        nets[tag] = mL
    rng = np.random.default_rng(0)
    base_idx, narrow_idx, groups = {}, {}, {}
    for lv in tree:
        g = lv["block"]
        S = Ss[g].numpy()
        pos = {int(n): i for i, n in enumerate(S)}
        mem = [n for n in lv["ids"] if n in pos]
        others = [n for n in S if n not in set(lv["ids"])]
        ctl = rng.choice(others, len(mem), replace=False).tolist()
        base_idx[g] = mem + ctl
        narrow_idx[g] = [pos[n] for n in mem + ctl]
        groups[g] = len(mem)
    veh = vehicle_ids()
    vrows = photo_rows(lab, veh, True)
    orows = photo_rows(lab, veh, False, n_other={k: len(v) for k, v in vrows.items()})
    res = dict(kept_members={int(g): groups[g] for g in groups},
               tree_members={int(lv["block"]): len(lv["ids"]) for lv in tree})
    for pname, prow in (("vehicle", vrows), ("other", orows)):
        zB = preacts(base, base_idx, prow)
        res[pname] = {"photos": int(sum(len(v) for v in prow.values()))}
        q90 = {g: torch.quantile(z.float(), .90, dim=0) for g, z in zB.items()}
        q99 = {g: torch.quantile(z.float(), .99, dim=0) for g, z in zB.items()}
        for xname, mX in nets.items():
            zX = preacts(mX, narrow_idx, prow)
            out = {}
            for part in ("members", "control"):
                acc = {k: [] for k in ("cross90", "cross99", "lost99", "gained99", "dpost")}
                for g in zB:
                    k = groups[g]
                    sl = slice(0, k) if part == "members" else slice(k, 2 * k)
                    b, x = zB[g][:, sl].float(), zX[g][:, sl].float()
                    if b.shape[1] == 0:
                        continue
                    t90, t99 = q90[g][sl], q99[g][sl]
                    acc["cross90"].append(((b > t90) != (x > t90)).float().mean(0))
                    acc["cross99"].append(((b > t99) != (x > t99)).float().mean(0))
                    on = b > t99
                    acc["lost99"].append(((on & (x <= t99)).float().sum(0) / on.float().sum(0).clamp_min(1)))
                    acc["gained99"].append((((~on) & (x > t99)).float().sum(0) / (~on).float().sum(0)))
                    pb, px = F.gelu(b), F.gelu(x)
                    acc["dpost"].append((px - pb).abs().mean(0) / pb.std(0).clamp_min(1e-6))
                out[part] = {kk: float(torch.cat(v).median()) for kk, v in acc.items()}
                out[part]["n_neurons"] = int(sum(len(v) for v in acc["cross90"]))
            res[pname][xname] = out
            print(pname, xname, json.dumps(out), flush=True)
    json.dump(res, open(os.path.join(OUT, "gates.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
