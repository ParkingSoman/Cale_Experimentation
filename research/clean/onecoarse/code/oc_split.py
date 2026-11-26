"""One coarse neuron, step 2 (cluster, in n6/clean/neuron/coarse).
(a) Input-only negative pole. A neuron's value = its inputs + a fixed offset (the bn shift). On a plain area with no
    input it sits at the offset: that is rest, not a negative. So per photo (19,000): the min over positions of
    (value - offset), and its position. For the parent s3:69 the value is z (layer3[1] residual sum, before the ReLU).
(b) Bands: the same rule as oc_bands.py (top 10; 10 of ranks 100-400; 10 of ranks 1,000-3,000; seeds j and j+1),
    positive by the max, negative by the input-only min of (a).
(c) Exact input split at the max cell (positive rows) and the min cell (negative rows):
    children (A b0) and n101 (B1 b0): one contribution per stage-3 channel (256) = bn1 scale * conv1 weights on the
    3 x 3 window of that channel; they add up to value - offset.
    parent: one contribution per layer3[1] inner neuron (256) = bn2 scale * conv2 weights (69, k) on the 3 x 3
    window of inner neuron k, plus the skip (stage-3 block-0 output channel 69 at the cell); they add up to z - offset.
(d) Maps (as oc_maps.py) on every photo in the new bands.
Saved: results/coarse/r18/oc_split.npz
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
KIDS = [305, 97, 23, 297, 108, 163, 275]
NAMES = ["p"] + [f"k{j}" for j in KIDS] + ["c101"]


def bands(v, seed):
    o = np.argsort(-v)
    rng = np.random.default_rng(seed)
    return np.concatenate([o[:10], np.sort(rng.choice(o[100:400], 10, replace=False)),
                           np.sort(rng.choice(o[1000:3000], 10, replace=False))])


def bn_affine(bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return s, bn.bias - bn.running_mean * s


def main():
    A, B1 = net_for("A"), net_for("B128aug_s1")
    ph = r18.net.Photos("read")
    held, test = r18.Cache("held"), r18.Cache("test")
    nh = len(held)
    ix_all = np.concatenate([held.ix, test.ix])
    a0, c0, l31 = A.layer4[0], B1.layer4[0], A.layer3[1]
    sA, tA = bn_affine(a0.bn1); sC, tC = bn_affine(c0.bn1); s2, t2 = bn_affine(l31.bn2)
    off = {"p": float(t2[69])}
    for j in KIDS:
        off[f"k{j}"] = float(tA[j])
    off["c101"] = float(tC[101])

    def stage3(rows):                       # parent z map, inner of layer3[1], skip, stage-3 output
        img = ph.tensor(ix_all[rows]).to(r18.DEV)
        h = A.layer3[0](A.layer2(A.layer1(r18.net.stem(A, img))))
        p = r18.net.block_parts(l31, h)
        return p["z"][:, 69], p["inner"], h[:, 69], p["h"]

    def kid_maps(x):
        out = {f"k{j}": v for j, v in zip(KIDS, a0.bn1(a0.conv1(x))[:, KIDS].unbind(1))}
        out["c101"] = c0.bn1(c0.conv1(x))[:, 101]
        return out

    # (a) input-only min and the max, over all photos
    mx = {n: [] for n in NAMES}; mn = {n: [] for n in NAMES}; amx = {n: [] for n in NAMES}; amn = {n: [] for n in NAMES}
    for a in range(0, len(ix_all), 250):
        rows = np.arange(a, min(a + 250, len(ix_all)))
        z, _, _, x = stage3(rows)
        maps = {"p": z, **kid_maps(x)}
        for n, m in maps.items():
            f = (m - off[n]).flatten(1)
            mx[n].append(f.max(1)[0].cpu()); amx[n].append(f.argmax(1).cpu())
            mn[n].append(f.min(1)[0].cpu()); amn[n].append(f.argmin(1).cpu())
    cat = lambda d: {n: torch.cat(v).numpy() for n, v in d.items()}
    mx, mn, amx, amn = cat(mx), cat(mn), cat(amx), cat(amn)
    res = {"ix": ix_all, "lab": ph.lab[ix_all]}
    seeds = {"p": 69, **{f"k{j}": j for j in KIDS}, "c101": 101}
    allrows = set()
    for n in NAMES:
        res[f"{n}_off"] = off[n]
        res[f"{n}_max"], res[f"{n}_amax"], res[f"{n}_min"], res[f"{n}_amin"] = mx[n], amx[n], mn[n], amn[n]
        res[f"{n}_bpos"], res[f"{n}_bneg"] = bands(mx[n], seeds[n]), bands(-mn[n], seeds[n] + 1)
        allrows |= set(res[f"{n}_bpos"].tolist()) | set(res[f"{n}_bneg"].tolist())
    rows = np.array(sorted(allrows))
    res["rows"] = rows
    # (c) + (d) on the band photos
    z, inner, skip, x = stage3(rows)
    xc = torch.cat([held.get(rows[rows < nh])[0], test.get(rows[rows >= nh] - nh)[0]])
    print("check: stage-3 output vs cache, max abs diff", float((x - xc).abs().max()))
    km = kid_maps(x)
    res["map_p"] = z.cpu().numpy()
    for n in NAMES[1:]:
        res[f"map_{n}"] = km[n].cpu().numpy()
    where = {int(r): i for i, r in enumerate(rows)}
    xp = F.pad(x, (1, 1, 1, 1)); ip = F.pad(inner, (1, 1, 1, 1))
    W2 = l31.conv2.weight[69]                                   # 256 x 3 x 3
    for n in NAMES:
        for side, b, am in (("pos", res[f"{n}_bpos"], amx[n]), ("neg", res[f"{n}_bneg"], amn[n])):
            sp, tot = [], []
            for r in b:
                i, p = where[int(r)], int(am[r])
                if n == "p":
                    rr, cc = p // 14, p % 14
                    win = ip[i, :, rr:rr + 3, cc:cc + 3]
                    con = s2[69] * (win * W2).sum((1, 2))
                    con = torch.cat([con, skip[i, rr, cc].view(1)])     # 257: last = skip
                    tot.append(float(z[i, rr, cc] - off[n]))
                else:
                    rr, cc = p // 7, p % 7
                    win = xp[i, :, 2 * rr:2 * rr + 3, 2 * cc:2 * cc + 3]
                    if n == "c101":
                        con = sC[101] * (win * c0.conv1.weight[101]).sum((1, 2))
                    else:
                        j = int(n[1:])
                        con = sA[j] * (win * a0.conv1.weight[j]).sum((1, 2))
                    tot.append(float(km[n][i, rr, cc] - off[n]))
                sp.append(con.cpu().numpy())
            res[f"{n}_split_{side}"] = np.stack(sp); res[f"{n}_tot_{side}"] = np.array(tot)
            err = np.abs(res[f"{n}_split_{side}"].sum(1) - res[f"{n}_tot_{side}"]).max()
            print(n, side, "split adds up, max abs err", round(float(err), 4))
    np.savez(os.path.join(r18.OUT, "oc_split.npz"), **res)
    print("saved rows", len(rows), "offsets", {k: round(v, 3) for k, v in off.items()})


if __name__ == "__main__":
    main()
