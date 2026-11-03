# serves: TC1
"""Realism check of Theorem TC1's premise on real cuts of ResNet-18, on SEALED photos of a goal. It also gives the first whole-network numbers.
Layers are blocks. A cut keeps the same channels through both blocks of a stage. The stem and the inner channels are kept.

Goal: dogs and cats (classes 151-268 and 281-285), weight 1, the rest weight 0.
Cuts, all fixed from the map before the run:
   dogs      keep ONLY the channels of the cards of agents dogface, furbody and gate, at stages 3 and 4. Stages 1, 2 stay whole.
   novehicle keep all channels, except those read only by agents roadcar and wheels at stages 3 and 4.
Controls for each cut: the same number of channels per stage, (r) at random, 3 seeds, (s) dropping the lowest spread of mean firing, so
keeping the strongest channels.

For each kept channel of each layer: e = E (z - z^K)^2 with layer below at N's values and the complement of K filled (Definition DC2).
Reported: eps_par = e / V (V over all goal photos and positions) and e / E z^2. Also KL of N_K against N, agreement with N's top class.
PREDICTION, written before the run. dogs cut: median eps_par above .5, KL above 1, far above the premise's 0. novehicle: median eps_par
under .1 at s3, under .2 at s4. KL under .1. novehicle beats random controls by 1.5 times or more on this goal. Against strongest channels it is close.
Label ONE-BUILD. Usage: cut_check.py [--smoke]"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import HERE, STORE, WIDTH, Photos, block_means, block_parts, head, load, stages, stem  # noqa: E402

smoke = "--smoke" in sys.argv
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
READ = os.path.join(STORE, "read")
P = Photos("sealed")
GOAL = list(range(151, 269)) + list(range(281, 286))
idx = np.where(np.isin(P.lab, GOAL))[0]
if smoke:
    idx = idx[:64]
m = load().to(DEV)
hm = {s: np.asarray(np.load(os.path.join(READ, f"s{s}b1_hm.npy"), mmap_mode="r"), dtype=np.float32) for s in (3, 4)}
MEAN = {s: tuple(v.to(DEV) for v in block_means(READ)[s]) for s in (3, 4)}
SPREAD = {s: hm[s].std(0) for s in (3, 4)}
by = {}
for a in ("dogface", "furbody", "gate", "roadcar", "wheels"):
    for line in open(os.path.join(HERE, "cards", a + ".jsonl")):
        if line.strip():
            by.setdefault(a, set()).add(json.loads(line)["channel"])


def chans(names, s):
    return {int(c.split(":")[1]) for a in names for c in by[a] if c.startswith(f"s{s}:")}


dog = {s: chans(("dogface", "furbody", "gate"), s) for s in (3, 4)}
veh = {s: chans(("roadcar", "wheels"), s) - dog[s] for s in (3, 4)}
cuts = {"dogs": {s: sorted(dog[s]) for s in (3, 4)},
        "novehicle": {s: sorted(set(range(WIDTH[s])) - veh[s]) for s in (3, 4)}}


def controls(K):
    out = {}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        out[f"random{seed}"] = {s: sorted(rng.choice(WIDTH[s], len(K[s]), replace=False).tolist()) for s in (3, 4)}
    out["strongest"] = {s: sorted(np.argsort(-SPREAD[s])[:len(K[s])].tolist()) for s in (3, 4)}
    return out


def fill(h, s, b, K):
    if s not in K:
        return h
    h = h.clone()
    out = torch.tensor(sorted(set(range(WIDTH[s])) - set(K[s])), dtype=torch.long, device=DEV)
    if len(out):
        h[:, out] = MEAN[s][b][out].view(1, -1, 1, 1)
    return h


def run(K, want_eps):
    """Returns KL, agreement, and per-layer eps for kept channels."""
    kl, agree, n = 0.0, 0.0, 0
    acc = {}
    with torch.no_grad():
        for b in range(0, len(idx), 64):
            x = P.tensor(idx[b:b + 64]).to(DEV)
            h, hk = stem(m, x), None
            hk = h
            for s, lay in stages(m).items():
                for bn, bl in enumerate(lay):
                    pN = block_parts(bl, h)
                    pK = block_parts(bl, hk)
                    hk = fill(pK["h"], s, bn, K)
                    if want_eps and s in K:
                        zQ = block_parts(bl, fill(h, s, 0, K) if bn == 1 else fill(h, s - 1, 1, K))["z"]
                        kept = torch.tensor(K[s], dtype=torch.long, device=DEV)
                        d = acc.setdefault((s, bn), dict(e=0.0, z2=0.0, z1=0.0, n=0))
                        zc, q = pN["z"][:, kept], zQ[:, kept]
                        d["e"] = d["e"] + ((zc - q) ** 2).sum((0, 2, 3)).float().cpu().numpy()
                        d["z2"] = d["z2"] + (zc ** 2).sum((0, 2, 3)).float().cpu().numpy()
                        d["z1"] = d["z1"] + zc.sum((0, 2, 3)).float().cpu().numpy()
                        d["n"] += zc.shape[0] * zc.shape[2] * zc.shape[3]
                    h = pN["h"]
            y0, y1 = head(m, h), head(m, hk)
            ls = F.log_softmax(y0, 1)
            kl += float((ls.exp() * (ls - F.log_softmax(y1, 1))).sum())
            agree += float((y0.argmax(1) == y1.argmax(1)).sum())
            n += len(x)
    res = dict(kl=kl / n, agree=agree / n, photos=n, kept={s: len(K[s]) for s in K})
    for key, d in acc.items():
        e, v = d["e"] / d["n"], d["z2"] / d["n"] - (d["z1"] / d["n"]) ** 2
        res[f"s{key[0]}b{key[1]}"] = dict(eps_median=float(np.median(e / (v + 1e-9))), share_median=float(np.median(e / (d["z2"] / d["n"] + 1e-9))))
    return res


out = {}
for name, K in cuts.items():
    out[name] = run(K, True)
    print(name, json.dumps(out[name]), flush=True)
    for cn, CK in controls(K).items():
        r = run(CK, False)
        out[f"{name}/{cn}"] = r
        print(f"  {cn} kl {r['kl']:.4f} agree {r['agree']:.3f}", flush=True)
json.dump(out, open(os.path.join(HERE, "results", "cut_check" + ("_smoke" if smoke else "") + ".json"), "w"), indent=0)
