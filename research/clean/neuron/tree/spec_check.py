"""N5 step 1 check: are group-specific neurons linked to the root? Remove the root alone; drops at ALL later blocks;
compare with each neuron's group concentration (profile). Also count group-specific neurons per block."""
import json, os
import numpy as np, torch
from common import DEV, OUT, Fills, Peaks, blocks, groups, load, net, split
from build_tree import root_scores, photos
torch.set_grad_enabled(False)
prof = dict(np.load(os.path.join(OUT, "profile_P10.npz"))); G = groups(); gn = list(prof["groups"])
T = json.load(open(os.path.join(OUT, "trees.json")))
ph = net.Photos("read"); fit = split(ph.lab); m = load("base"); BL = blocks(m)
fills = Fills(m); pk = Peaks(m); out = {}
for name in ("dogs", "birds", "vehicles"):
    grp = G[name]; gi = gn.index(name); l, r = T[name]["root"]["block"], T[name]["root"]["neuron"]
    x = ph.tensor(photos(ph, grp, fit, 10, max(1, -(-120 // len(grp))), 120)).to(DEV)
    def run(d):
        fills.set(d); st = {}
        for a in range(0, len(x), 60):
            m(x[a:a+60]); [st.setdefault(g, []).append(v.clone()) for g, v in pk.store.items()]
        fills.clear(); return {g: torch.cat(v) for g, v in st.items()}
    b0 = run({}); b1 = run({l: ([r], prof[f"gm{l}"][gi][[r]])})
    rows = []
    for g in range(l + 1, 30):
        sd = torch.from_numpy(prof[f"sd{g}"]).to(DEV)
        s = ((b0[g] - b1[g]) / sd).mean(0).cpu().numpy()
        conc, br = root_scores(prof, g, grp)
        spec = (conc > .5) & (br > .3)
        top = np.argsort(-s)[:64]
        rows.append(dict(block=g, n_specific=int(spec.sum()), size=len(s),
                         corr_drop_conc=float(np.corrcoef(s, conc)[0, 1]),
                         drop_specific_mean=float(s[spec].mean()) if spec.any() else None,
                         drop_other_mean=float(s[~spec].mean()), top64_specific=int(spec[top].sum()),
                         max_drop=float(s.max())))
    out[name] = rows
    print(name, f"root b{l}:{r}")
    for q in rows:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
json.dump(out, open(os.path.join(OUT, "spec_check.json"), "w"), indent=1)
