"""Second review checks (2026-11-13).
A. Bootstrap intervals for the goal-"all" rows: parent column against bias only (per-photo test losses).
B. Parent 237's map children WITHOUT child 243 (shared with the bird parent): drop cost on every goal.
C. Own20 rows with each child filled by its mean on the GOAL's fit photos (not all-class means): drop, bias only,
   parent column, 20 random channels, bootstrap parent minus bias.
"""
import json
import os

import numpy as np
import torch

from absorb import net, stage4, DEV, BATCH, OUT
from controls import solve, boot
from fit import kl_rows

torch.set_default_dtype(torch.float64)


@torch.no_grad()
def pooled(m, ph, idx, drop=None, means=None, inner_mean=False):
    """Pooled final stream for photos idx. inner_mean: also return block-1 inner channel means over these photos."""
    g = np.zeros((len(idx), 512))
    acc = torch.zeros(512, device=DEV, dtype=torch.float32)
    for i in range(0, len(idx), BATCH):
        x = ph.tensor(idx[i:i + BATCH]).float().to(DEV)
        h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
        _, u, hd = stage4(m, h3, drop=drop, means=means)
        g[i:i + len(x)] = hd.mean((2, 3)).cpu().numpy()
        if inner_mean:
            acc += torch.relu(u).mean((2, 3)).sum(0)
    return (g, (acc / len(idx))) if inner_mean else g


def main():
    d = np.load(os.path.join(OUT, "absorb_cache.npz"))
    meta = json.load(open(os.path.join(OUT, "absorb_edges.json")))
    W, bias = torch.tensor(d["fcW"]).double(), torch.tensor(d["fcb"]).double()
    lab, fit = d["lab"], d["fit"]
    yN = torch.tensor(d["gN"]).double() @ W.T + bias
    parents = [245, 418, 237]
    res = dict(A=[], B={}, C=[])
    rng = np.random.default_rng(1)
    rand_q = sorted(rng.choice([q for q in range(512) if q not in parents], 20, replace=False).tolist())

    # A. goal "all" rows with bootstrap
    F, T = np.where(fit)[0], np.where(~fit)[0]
    for p in ([] if os.environ.get("SKIP_A") else parents):  # part A already ran: results/extra_A.log
        for rc in ("map", "edge"):
            g = torch.tensor(d[f"gD_{p}_{rc}"]).double()
            yD = g @ W.T + bias
            a0, _ = solve(yN[F], yD[F], None)
            bt = kl_rows(yN[T], yD[T] + a0)
            a, b = solve(yN[F], yD[F], g[F, p])
            pt = kl_rows(yN[T], yD[T] + a + b * g[T, p][:, None])
            row = dict(parent=p, recipe=rc, drop=float(kl_rows(yN[T], yD[T]).mean()), bias=float(bt.mean()),
                       parent_col=float(pt.mean()), boot_parent_minus_bias=boot(pt - bt))
            res["A"].append(row)
            print("A", json.dumps(row), flush=True)

    torch.set_default_dtype(torch.float32)  # MPS has no float64: build the network and photo tensors in float32
    m = net.load().to(DEV)
    ph = net.Photos("read")
    means_all = torch.tensor(d["means"]).to(DEV)
    torch.set_default_dtype(torch.float64)

    # B. 237's map children without 243, all photos, global means
    ch = [c for c in meta["parents"]["237"]["map_children"] if c != 243]
    gB = torch.tensor(pooled(m, ph, np.arange(len(lab)), drop=ch, means=means_all))
    yB = gB @ W.T + bias
    goals = {"all": np.ones(len(lab), bool)}
    for pi, p in enumerate(parents):
        par = d["par"][:, pi]
        cm = np.array([par[(lab == c) & fit].mean() for c in range(1000)])
        goals[f"own20_{p}"] = np.isin(lab, np.argsort(-cm)[:20])
    res["B"] = {"children": ch, **{gn: float(kl_rows(yN[gm & ~fit], yB[gm & ~fit]).mean()) for gn, gm in goals.items()}}
    print("B", json.dumps(res["B"]), flush=True)

    # C. own20 rows with goal-mean fill
    for p in parents:
        gm = goals[f"own20_{p}"]
        idx = np.where(gm)[0]
        Fi, Ti = np.where(fit[idx])[0], np.where(~fit[idx])[0]
        _, gmeans = pooled(m, ph, idx[Fi], inner_mean=True)
        yNg = yN[idx]
        for rc in ("map", "edge"):
            chs = meta["recipes"][f"{p}_{rc}"]
            g = torch.tensor(pooled(m, ph, idx, drop=chs, means=gmeans))
            yD = g @ W.T + bias
            a0, _ = solve(yNg[Fi], yD[Fi], None)
            bt = kl_rows(yNg[Ti], yD[Ti] + a0)
            chan = {}
            per = {}
            for q in [p] + rand_q:
                a, b = solve(yNg[Fi], yD[Fi], g[Fi, q])
                per[q] = kl_rows(yNg[Ti], yD[Ti] + a + b * g[Ti, q][:, None])
                chan[q] = float(per[q].mean())
            others = [chan[q] for q in rand_q]
            row = dict(parent=p, recipe=rc, drop=float(kl_rows(yNg[Ti], yD[Ti]).mean()), bias=float(bt.mean()),
                       parent_col=chan[p], rand_median=float(np.median(others)),
                       parent_rank=1 + sum(v < chan[p] for v in others), boot_parent_minus_bias=boot(per[p] - bt))
            res["C"].append(row)
            print("C", json.dumps(row), flush=True)
    json.dump(res, open(os.path.join(OUT, "extra.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
