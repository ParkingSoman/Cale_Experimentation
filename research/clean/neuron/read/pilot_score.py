"""N10 pilot: score the blind judge's picks (rules in pilot_sheets.py, fixed before any card was read).
Values: Base on the JUDGE slice only (odd positions within each class's test-half list); class mean of pooled max (high picks)
and pooled min (low picks), ranked 0..1 among the 40 candidates.
pos = mean rank of HIGH picks - .5;  neg = .5 - mean rank of LOW picks. Chance 0.
Drop rule: a pick whose class name appears in the card as a whole word (case-insensitive) is dropped; unmatched labels counted.
Chance band: 5 random candidates, 10,000 draws, 95% band of the mean. Reading rule (fixed now): a card 'works' on a pole if its
own score is above the chance band's top AND above its mismatched score by .1 or more. own ~ mismatched > chance = a generic card."""
import sys, os, json, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "read"); PD = os.path.join(OUT, "pilot")
sys.path.insert(0, os.path.join(HERE, "..", "gap")); import common as C
LAB = C.net.Photos("read").lab; fit = C.split(LAB); RAW = C.net.class_names()
judge = np.zeros(len(LAB), bool)
for c in range(1000):
    w = np.where((LAB == c) & ~fit)[0]; judge[w[1::2]] = True
key = json.load(open(os.path.join(PD, "judge_key.json"))); meta = json.load(open(os.path.join(OUT, "pilot_meta.json")))
keys = [tuple(k[:2]) for k in meta["keys"]]; D = np.load(os.path.join(OUT, "pilot_base.npz")); lab = LAB[judge]
def ranks(name, cand):
    n = [x for x in meta["neurons"] if x["name"] == name][0]; c = keys.index((n["g"], n["base_index"]))
    mx, mn = D["mx"][judge, c].astype(np.float32), D["mn"][judge, c].astype(np.float32)
    hi = np.array([mx[lab == k].mean() for k in cand]); lo = np.array([mn[lab == k].mean() for k in cand])
    return hi.argsort().argsort() / (len(cand) - 1), lo.argsort().argsort() / (len(cand) - 1)
rng = np.random.default_rng(0); ch = np.array([rng.choice(40, 5, replace=False) for _ in range(10000)]) / 39.
band = np.percentile(ch.mean(1) - .5, [2.5, 97.5]).tolist()
rows = []
for pid, name in key["packets"].items():
    picks = json.load(open(os.path.join(PD, "judge", pid, "picks.json")))
    card = open(os.path.join(PD, f"reader_{name}", "card.md")).read().lower()
    cand = key["candidates"][name]; labels = [x.lower() for x in key["labels"][name]]
    def idx(lst):
        out, unmatched, dropped = [], 0, 0
        for x in lst:
            x = x.strip().lower()
            if x not in labels: unmatched += 1; continue
            base_name = RAW[cand[labels.index(x)]].lower()
            if any(re.search(r"\b" + re.escape(t.strip()) + r"\b", card) for t in base_name.split(",")): dropped += 1; continue
            out.append(labels.index(x))
        return out, unmatched, dropped
    hi, uh, dh = idx(picks["high"]); lo, ul, dl = idx(picks["low"])
    out = dict(neuron=name, packet=pid, n_high=len(hi), n_low=len(lo), unmatched=uh + ul, dropped=dh + dl)
    for tag, who in (("own", name), ("mismatched", key["pairs"][name])):
        rh, rl = ranks(who, cand)
        out[f"{tag}_pos"] = float(rh[hi].mean() - .5) if hi else None
        out[f"{tag}_neg"] = float(.5 - rl[lo].mean()) if lo else None
    rows.append(out); print(out)
print("chance band (5 random picks):", [round(x, 3) for x in band])
json.dump(dict(rows=rows, chance_band=band), open(os.path.join(PD, "judge_scores.json"), "w"), indent=1)
