"""N10 step 2d: specificity. The set's part (b) and the 3 pilot neurons' single B-order gains, on held NON-bird photos (1,882),
as shares of the non-bird gap, next to the bird shares. Similar shares = general neurons, not bird neurons."""
import sys, os, json, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcommon import *
import s2_set_helpers as H
t0 = time.time(); say = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
nn_ = nets(); P = photo_sets(); H.init(nn_, P); base, mB, mT, Ss, calib = nn_
held = RCache("held"); nb = np.where(~np.isin(LAB[held.ix], BIRDS))[0]; NB = Rows([(held, nb)])
S = json.load(open(os.path.join(OUT, "s2_set.json")))
items = S["set"]["items"]; surv = [(i[1], i[2]) for i in items if i[0] == "s"]; dele = [(i[1], i[2]) for i in items if i[0] == "d"]
res = dict(set_nonbird=H.test(surv, dele, NB)); say("set, non-birds", res["set_nonbird"])
kB, kT = NB.kl(mB), NB.kl(mT); gap = (kB - kT).mean(); res["gap_nonbird"] = gap; res["pilot"] = {}
for name, kind, g, b in (("n1", "d", 22, 295), ("n2", "s", 19, 1961), ("n3", "s", 25, 1229)):
    m, ref = H.b_plus([(g, b)] if kind == "s" else [], [(g, b)] if kind == "d" else [])
    gain = NB.kl(ref) - NB.kl(m); res["pilot"][name] = dict(gain_share=gain.mean() / gap, ci=(np.array(boot_ci(gain)) / gap).tolist())
    say(name, res["pilot"][name])
json.dump(conv(res), open(os.path.join(OUT, "s2_spec.json"), "w"), indent=1)
