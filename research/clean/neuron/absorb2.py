"""Phase 2 of plan part 2: absorption that follows the subcircuits of Phase 1 (results/mix.json).

Recipes (children removed, filled with their all-photo means):
  245: children 243, 73, 377        418: children 317, 408
Forms, scored as KL to N on test photos:
  drop      children removed, readout unchanged
  bias      + the KL-best bias
  A1        + bias + the readout column of the parent (pooled s4b1:p, its relay through the skip)
  A2        + bias + the readout columns of every input that drives a subcircuit of a removed child (top 2 per subcircuit)
  rand      + bias + the readout columns of random channel sets of the same size as A2 (5 draws)
  A3r       children KEPT, each reading only the top-10 inputs of its parent's subcircuit (other inputs at their means). No fit
  A3all     children KEPT, each reading only the top-10 inputs of each of its subcircuits (union). No fit
"""
import json
import os

import numpy as np
import torch

from absorb import net, DEV, BATCH, OUT
from controls import boot
from fit import kl_rows, lbfgs

RECIPES = {245: [243, 73, 377], 418: [317, 408]}


def solve_multi(yN, yD, G):
    """KL optimum of yD + a + G @ B. G: (n, m) features, or None."""
    A = torch.zeros(yN.shape[1], dtype=torch.float64, requires_grad=True)
    ps = [A]
    if G is not None:
        B = torch.zeros(G.shape[1], yN.shape[1], dtype=torch.float64, requires_grad=True)
        ps.append(B)
    f = (lambda: kl_rows(yN, yD + A)) if G is None else (lambda: kl_rows(yN, yD + A + G @ B))
    with torch.enable_grad():
        lbfgs(ps, lambda: f().mean(), iters=200)
    return [p.detach() for p in ps]


def main():
    mix = json.load(open(os.path.join(OUT, "mix.json")))
    cache = np.load(os.path.join(OUT, "absorb_cache.npz"))
    torch.set_default_dtype(torch.float32)  # controls.py sets float64 on import; MPS needs float32
    m = net.load().to(DEV)
    ph = net.Photos("read")
    lab, fit = cache["lab"], cache["fit"]
    means = torch.tensor(cache["means"]).to(DEV)
    h_mean = net.block_means(os.path.join(net.STORE, "read"))[4][0].to(DEV)
    b0, b1 = m.layer4[0], m.layer4[1]

    # inputs kept by the coarse children (A3) and drivers for A2
    keep_r, keep_all, drivers = {}, {}, {}
    for p, kids in RECIPES.items():
        drv = set()
        for j in kids:
            subs = mix[str(j)]["subcircuits"]
            ref = min(subs, key=lambda s: s["parent_rank"])
            keep_r[j] = ref["top10"]
            keep_all[j] = sorted(set(t for s in subs for t in s["top10"]))
            for s in subs:
                drv.update(s["top10"][:2])
        drivers[p] = sorted(drv)
    print("drivers", drivers, flush=True)

    mean_map = h_mean.view(1, -1, 1, 1).expand(1, 512, 7, 7).contiguous()

    def coarse_inner(h40, keep):
        """block-1 inner pre-activation, with child j reading only the channels keep[j] (others at their means)."""
        u = b1.bn1(b1.conv1(h40))
        for j, chans in keep.items():
            mask = torch.zeros(512, device=DEV)
            mask[chans] = 1
            hm = mean_map + (h40 - mean_map) * mask.view(1, -1, 1, 1)
            u[:, j] = b1.bn1(b1.conv1(hm))[:, j]
        return u

    N = len(lab)
    feats = {f"{p}_{v}": np.zeros((N, 512), np.float32) for p in RECIPES for v in ("drop", "A3r", "A3all")}
    for i in range(0, N, BATCH):
        sl = slice(i, min(i + BATCH, N))
        x = ph.tensor(np.arange(sl.start, sl.stop)).float().to(DEV)
        with torch.no_grad():
            h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
            h40 = net.block_parts(b0, h3)["h"]
            for p, kids in RECIPES.items():
                for v in ("drop", "A3r", "A3all"):
                    if v == "drop":
                        inner = torch.relu(b1.bn1(b1.conv1(h40))).clone()
                        inner[:, kids] = means[kids].view(1, -1, 1, 1)
                    else:
                        keep = {j: (keep_r if v == "A3r" else keep_all)[j] for j in kids}
                        inner = torch.relu(coarse_inner(h40, keep))
                    h41 = torch.relu(b1.bn2(b1.conv2(inner)) + h40)
                    feats[f"{p}_{v}"][sl] = h41.mean((2, 3)).cpu().numpy()
        if (i // BATCH) % 56 == 0:
            print(f"  pass {i}/{N}", flush=True)
    np.savez_compressed(os.path.join(OUT, "absorb2_cache.npz"), **feats)

    torch.set_default_dtype(torch.float64)
    W = torch.tensor(cache["fcW"]).double()
    bias = torch.tensor(cache["fcb"]).double()
    yN_all = torch.tensor(cache["gN"]).double() @ W.T + bias
    parents = [245, 418, 237]
    rng = np.random.default_rng(2)
    res = []
    for p in RECIPES:
        pi = parents.index(p)
        par = cache["par"][:, pi]
        cm = np.array([par[(lab == c) & fit].mean() for c in range(1000)])
        goals = {"all": np.ones(N, bool), "own20": np.isin(lab, np.argsort(-cm)[:20])}
        g = torch.tensor(feats[f"{p}_drop"]).double()
        yD_all = g @ W.T + bias
        pool = [q for q in range(512) if q not in drivers[p]]
        rsets = [sorted(rng.choice(pool, len(drivers[p]), replace=False).tolist()) for _ in range(5)]
        for gn, gm in goals.items():
            F, T = np.where(gm & fit)[0], np.where(gm & ~fit)[0]
            yN, yD = yN_all[F], yD_all[F]
            row = dict(parent=p, goal=gn, children=RECIPES[p], drivers=drivers[p], n_test=len(T))
            per = {}
            per["drop"] = kl_rows(yN_all[T], yD_all[T])
            a, = solve_multi(yN, yD, None)
            per["bias"] = kl_rows(yN_all[T], yD_all[T] + a)
            for name, cols in (("A1", [p]), ("A2", drivers[p])):
                a, B = solve_multi(yN, yD, g[F][:, cols])
                per[name] = kl_rows(yN_all[T], yD_all[T] + a + g[T][:, cols] @ B)
            rv = []
            for rs in rsets:
                a, B = solve_multi(yN, yD, g[F][:, rs])
                rv.append(kl_rows(yN_all[T], yD_all[T] + a + g[T][:, rs] @ B))
            per["rand_median"] = sorted(rv, key=lambda t: float(t.mean()))[2]
            row["rand_all"] = [float(t.mean()) for t in rv]
            for v in ("A3r", "A3all"):
                yv = torch.tensor(feats[f"{p}_{v}"]).double()[T] @ W.T + bias
                per[v] = kl_rows(yN_all[T], yv)
            for k_, t in per.items():
                row[k_] = float(t.mean())
            row["boot_A2_minus_bias"] = boot(per["A2"] - per["bias"])
            row["boot_A2_minus_rand"] = boot(per["A2"] - per["rand_median"])
            row["boot_A1_minus_bias"] = boot(per["A1"] - per["bias"])
            res.append(row)
            print(json.dumps(row), flush=True)
    json.dump(res, open(os.path.join(OUT, "absorb2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
