# serves: H-map
"""Line "Parents act" of Step 4, on SEALED photos, for every link in results/links.json. Fixed before the run.

For a link parent -> child: take the child's 64 strongest sealed photos, at its own peak position on each. Set the parent to its mean
(over the reading photos) through its whole stage, and record the fall in the child's z. Sign "+" expects a fall, "-" a rise.
Controls: 3 channels of the parent's stage with the closest spread of largest firing, outside the child's top 30 feeders.
PASS for a link: the signed effect is at least 2 times the mean absolute control effect, and above 0.
Also recorded: the effect as a share of the child's mean z, and how the child falls when ALL its listed parents go together.

Second measure, for the question "how many parents does a child need": z of the child when every channel of the stage below EXCEPT
its k strongest feeders goes to its mean, for k = 4, 16, 64. Report error = E (z_new - z)^2 / Var z over the same photos.
PREDICTION, written before the run: at stage 4 one parent's effect is 2 to 20% of z (from the agents' reading photos); pass rate over
all links 70 to 90%. Error at k = 4 above .5, at k = 16 about .2 to .5, at k = 64 under .2. Stages 1 to 3 need fewer parents than stage 4.
Usage: score_acts.py [--smoke]"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import HERE, STORE, WIDTH, Photos, block_means, forward, load, parse  # noqa: E402

smoke = "--smoke" in sys.argv
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
READ, SEAL, EDGES = (os.path.join(STORE, x) for x in ("read", "sealed", "edges"))
P = Photos("sealed")
m = load().to(DEV)
links = json.load(open(os.path.join(HERE, "results", "links.json")))
if smoke:
    links = links[:3]


def arr(root, name):
    return np.load(os.path.join(root, name + ".npy"), mmap_mode="r")


def bk(s):
    return 0 if s == 0 else 1


MEAN = block_means(READ)
SPREAD = {s: np.asarray(arr(READ, f"s{s}b{bk(s)}_hx"), dtype=np.float32).std(0) for s in range(5)}


def photos_of(cs, cc):
    v = np.asarray(arr(SEAL, f"s{cs}b1_hx")[:, cc], dtype=np.float32)
    idx = np.sort(np.argsort(-v)[:64])
    return idx, np.asarray(arr(SEAL, f"s{cs}b1_hp")[idx, cc]).astype(np.int64)


def child_z(idx, pos, cs, cc, fill):
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), 32):
            sel = idx[b:b + 32]
            f = None if fill is None else {s: (c.to(DEV), MEAN[s][0][c].to(DEV), MEAN[s][1][c].to(DEV)) for s, c in fill.items()}
            _, o = forward(m, P.tensor(sel).to(DEV), f)
            z = o[(cs, 1)]["z"][:, cc].flatten(1)
            out.append(z[torch.arange(z.shape[0], device=DEV), torch.as_tensor(pos[b:b + 32], device=DEV)].float().cpu().numpy())
    return np.concatenate(out)


rows = []
children = {}
for child, parent, sign in links:
    cs, cc = parse(child)
    ps, pc = parse(parent)
    if ps != cs - 1:
        continue
    idx, pos = photos_of(cs, cc)
    base = child_z(idx, pos, cs, cc, None)
    eff = float((base - child_z(idx, pos, cs, cc, {ps: torch.tensor([pc])})).mean())
    eff = eff if sign != "-" else -eff
    par = np.load(os.path.join(EDGES, f"s{cs}_top_par.npy"), mmap_mode="r")[cc]
    barred = set(np.argsort(-np.abs(par))[:30].tolist()) | {pc}
    ctrl = [int(q) for q in np.argsort(np.abs(SPREAD[ps] - SPREAD[ps][pc])) if int(q) not in barred][:3]
    cf = [abs(float((base - child_z(idx, pos, cs, cc, {ps: torch.tensor([q])})).mean())) for q in ctrl]
    ok = eff > 0 and eff >= 2 * float(np.mean(cf))
    rows.append(dict(child=child, parent=parent, sign=sign, effect=eff, control=float(np.mean(cf)), z=float(base.mean()), share=eff / max(abs(float(base.mean())), 1e-6), ok=bool(ok)))
    children.setdefault(child, []).append(pc)
    print(f"{parent} -> {child} {sign} effect {eff:+.3f} control {np.mean(cf):.3f} z {base.mean():.2f} {'PASS' if ok else 'fail'}", flush=True)

need = []
for child, pcs in children.items():
    cs, cc = parse(child)
    ps = cs - 1
    idx, pos = photos_of(cs, cc)
    base = child_z(idx, pos, cs, cc, None)
    var = float(base.var()) + 1e-9
    together = float((base - child_z(idx, pos, cs, cc, {ps: torch.tensor(pcs)})).mean())
    par = np.load(os.path.join(EDGES, f"s{cs}_top_par.npy"), mmap_mode="r")[cc]
    o = np.argsort(-np.abs(par))
    err = {}
    for k in (4, 16, 64):
        rest = torch.tensor([int(q) for q in o[k:]], dtype=torch.long)
        new = child_z(idx, pos, cs, cc, {ps: rest}) if len(rest) else base
        err[k] = float(((new - base) ** 2).mean() / var)
        err[f"share{k}"] = float(((new - base) ** 2).mean() / (float((base ** 2).mean()) + 1e-9))
    need.append(dict(child=child, z=float(base.mean()), together=together, **{f"err{k}": v for k, v in err.items()}))
    print(child, "together", round(together, 2), "z", round(float(base.mean()), 2), {k: round(v, 3) for k, v in err.items()}, flush=True)
tag = "_smoke" if smoke else ""
json.dump(dict(links=rows, need=need), open(os.path.join(HERE, "results", f"acts_sealed{tag}.json"), "w"), indent=0)
for s in (1, 2, 3, 4):
    r = [x for x in rows if x["child"].startswith(f"s{s}:")]
    n = [x for x in need if x["child"].startswith(f"s{s}:")]
    if r:
        print(f"s{s}: links {len(r)}, pass {np.mean([x['ok'] for x in r]):.2f}, median share {np.median([x['share'] for x in r]):.3f}; "
              f"parents together / z median {np.median([x['together'] / max(abs(x['z']), 1e-6) for x in n]):.2f}; "
              f"err4 {np.median([x['err4'] for x in n]):.3f} err16 {np.median([x['err16'] for x in n]):.3f} err64 {np.median([x['err64'] for x in n]):.3f}; "
              f"share4 {np.median([x['errshare4'] for x in n]):.3f} share16 {np.median([x['errshare16'] for x in n]):.3f} share64 {np.median([x['errshare64'] for x in n]):.3f}")
print(f"all links {len(rows)}: pass {np.mean([x['ok'] for x in rows]):.3f}")
