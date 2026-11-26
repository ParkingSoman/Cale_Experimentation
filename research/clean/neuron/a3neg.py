"""Phase 2 follow-up: does the coarse child (A3) fail because it lost its suppressors?
A3r  : child reads only the top-10 inputs of its parent's subcircuit (as in absorb2.py)
A3rn : the same, plus the child's 5 strongest NEGATIVE inputs (from mix.json "neg_inputs")
A3alln: top-10 inputs of every subcircuit, plus the 5 negative inputs
No fit. KL to N on test photos, goals "all" and own20 (same goals as absorb2.py).
"""
import json
import os

import numpy as np
import torch

from absorb import net, DEV, BATCH, OUT

RECIPES = {245: [243, 73, 377], 418: [317, 408]}


def main():
    torch.set_default_dtype(torch.float32)
    mix = json.load(open(os.path.join(OUT, "mix.json")))
    cache = np.load(os.path.join(OUT, "absorb_cache.npz"))
    m = net.load().to(DEV)
    ph = net.Photos("read")
    lab, fit = cache["lab"], cache["fit"]
    h_mean = net.block_means(os.path.join(net.STORE, "read"))[4][0].to(DEV)
    b0, b1 = m.layer4[0], m.layer4[1]
    mean_map = h_mean.view(1, -1, 1, 1).expand(1, 512, 7, 7).contiguous()
    keeps = {}
    for p, kids in RECIPES.items():
        for j in kids:
            subs = mix[str(j)]["subcircuits"]
            ref = min(subs, key=lambda s: s["parent_rank"])["top10"]
            union = sorted(set(t for s in subs for t in s["top10"]))
            neg = [int(n[0].split(":")[1]) for n in mix[str(j)]["neg_inputs"]]
            keeps[("A3r", j)] = ref
            keeps[("A3rn", j)] = sorted(set(ref) | set(neg))
            keeps[("A3alln", j)] = sorted(set(union) | set(neg))
    forms = ("A3r", "A3rn", "A3alln")
    N = len(lab)
    feats = {f"{p}_{v}": np.zeros((N, 512), np.float32) for p in RECIPES for v in forms}
    feats["N"] = np.zeros((N, 512), np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl = slice(i, min(i + BATCH, N))
            x = ph.tensor(np.arange(sl.start, sl.stop)).float().to(DEV)
            h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
            h40 = net.block_parts(b0, h3)["h"]
            u0 = b1.bn1(b1.conv1(h40))
            feats["N"][sl] = torch.relu(b1.bn2(b1.conv2(torch.relu(u0))) + h40).mean((2, 3)).cpu().numpy()
            for p, kids in RECIPES.items():
                for v in forms:
                    u = u0.clone()
                    for j in kids:
                        mask = torch.zeros(512, device=DEV)
                        mask[keeps[(v, j)]] = 1
                        hm = mean_map + (h40 - mean_map) * mask.view(1, -1, 1, 1)
                        u[:, j] = b1.bn1(b1.conv1(hm))[:, j]
                    h41 = torch.relu(b1.bn2(b1.conv2(torch.relu(u))) + h40)
                    feats[f"{p}_{v}"][sl] = h41.mean((2, 3)).cpu().numpy()
    W = torch.tensor(cache["fcW"]).double()
    bias = torch.tensor(cache["fcb"]).double()
    yN = torch.tensor(cache["gN"]).double() @ W.T + bias
    check = float((torch.tensor(feats["N"]).double() @ W.T + bias - yN).abs().max())
    print("check against cached N logits (max abs):", check, flush=True)
    parents = [245, 418, 237]
    res = []
    for p in RECIPES:
        par = cache["par"][:, parents.index(p)]
        cm = np.array([par[(lab == c) & fit].mean() for c in range(1000)])
        for gn, gm in (("all", np.ones(N, bool)), ("own20", np.isin(lab, np.argsort(-cm)[:20]))):
            T = np.where(gm & ~fit)[0]
            row = dict(parent=p, goal=gn)
            for v in forms:
                y = torch.tensor(feats[f"{p}_{v}"][T]).double() @ W.T + bias
                lpN, lpB = torch.log_softmax(yN[T], 1), torch.log_softmax(y, 1)
                row[v] = float((lpN.exp() * (lpN - lpB)).sum(1).mean())
            res.append(row)
            print(json.dumps(row), flush=True)
    json.dump(dict(rows=res, keeps={f"{k[0]}_{k[1]}": v for k, v in keeps.items()}),
              open(os.path.join(OUT, "a3neg.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
