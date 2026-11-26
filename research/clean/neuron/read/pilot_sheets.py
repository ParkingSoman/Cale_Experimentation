"""N10 pilot: reader packets (photo sheets + text evidence) and blind judge packets. Rules fixed before any card is read.
Photo split (reading photos, 35 per class):
  Base evidence = fit half (even positions in a class).
  Test half (odd positions) is split again by alternation inside each class: 'ev' slice and 'judge' slice.
  B and T evidence = photos T never trained on, from held (fit half, not in T's training) and the 'ev' slice only.
  The judge is scored ONLY on Base values on the 'judge' slice. No reader evidence comes from it.
Judge test: per neuron, 40 candidate classes drawn by strata from FIT-half Base class ranks, after removing every class
named anywhere in the reader's evidence: 8 from ranks 16-60 by class mean pooled max ('high'), 8 from ranks 16-60 by class
mean pooled min ('neg'), 24 at random from the rest. Shown shuffled, with unique labels. The judge picks 5 HIGHEST and 5 MOST
NEGATIVE. Scores in pilot_score.py. Mismatched control: the same picks scored against the paired neuron (n1->n2->n3->n1)."""
import sys, os, json, re, shutil
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "read"); PD = os.path.join(OUT, "pilot")
sys.path.insert(0, os.path.join(HERE, "..", "gap")); import common as C
if os.path.exists(PD): shutil.rmtree(PD)
os.makedirs(PD)
ph = C.net.Photos("read"); LAB = ph.lab; fit = C.split(LAB)
RAW = C.net.class_names(); cnt = {}
for n in RAW: cnt[n] = cnt.get(n, 0) + 1
NAMES = [n if cnt[n] == 1 else f"{n} (class {i})" for i, n in enumerate(RAW)]          # unique labels
judge = np.zeros(len(LAB), bool)
for c in range(1000):
    w = np.where((LAB == c) & ~fit)[0]; judge[w[1::2]] = True
NOTES = {"deleted": "Deleted when Base was shrunk to B (B keeps half the neurons). Re-adding it to B lowers B's error.",
         "survivor": "Kept in B. T's trained numbers for it, put into B, lower B's error."}
meta = json.load(open(os.path.join(OUT, "pilot_meta.json"))); keys = [tuple(k[:2]) for k in meta["keys"]]
D = {w: np.load(os.path.join(OUT, f"pilot_{w}.npz")) for w in ("base", "B", "T")}
def col(g, b): return keys.index((g, b))
def crop(i, p, g):
    img = Image.fromarray(np.asarray(ph.raw([int(i)])[0])); grid = 14 if g < 27 else 7; cell = 224 // grid
    r, c = divmod(int(p), grid); cy, cx = r * cell + cell // 2, c * cell + cell // 2; h = 48 if g < 27 else 64
    return img.crop((max(0, cx - h), max(0, cy - h), min(224, cx + h), min(224, cy + h))).resize((112, 112))
def sheet(rows, path):
    im = Image.new("RGB", (8 * 116 + 170, len(rows) * 130), "white"); d = ImageDraw.Draw(im)
    for r, (label, items) in enumerate(rows):
        d.text((4, r * 130 + 50), label, fill="black")
        for k, (i, p, g) in enumerate(items[:8]): im.paste(crop(i, p, g), (170 + k * 116, r * 130 + 8))
    im.save(path)
def bands(v, ii, pos, g, neg=False, seed=0):
    rng = np.random.default_rng(seed); o = np.argsort(v); n = len(v); out = []
    spec = [("most negative 0.1%", o[:max(8, n // 1000)]), ("negative p0.1-1", o[n // 1000:n // 100])] if neg else \
           [("top 0.1%", o[-max(8, n // 1000):][::-1]), ("p99-99.9", o[-n // 100:-n // 1000]), ("p90-99", o[-n // 10:-n // 100]), ("p50-90", o[n // 2:-n // 10])]
    for lab, rows in spec:
        sel = rows[:8] if lab.startswith(("top", "most")) else rng.choice(rows, min(8, len(rows)), replace=False)
        out.append((lab, [(ii[s], pos[s], g) for s in sel]))
    return out
def cmeans(ix, v):
    lab = LAB[ix]; return np.array([v[lab == c].mean() if (lab == c).any() else np.nan for c in range(1000)])
def prof(m, k=15, low=False):
    o = np.argsort(m) if low else np.argsort(-np.nan_to_num(m, nan=-1e9)); o = [c for c in o if not np.isnan(m[c])][:k]
    return o, "; ".join(f"{NAMES[c]} {m[c]:.2f}" for c in o)
key = dict(pairs={}, candidates={}, labels={}, strata={}, packets={})
for n_i, NE in enumerate(meta["neurons"]):
    name, g, b, kind = NE["name"], NE["g"], NE["base_index"], NE["kind"]; c = col(g, b)
    pdir = os.path.join(PD, f"reader_{name}"); os.makedirs(pdir); printed = set()
    txt = [f"Neuron {name}: block {g} of 30 ({'stage 3, 14x14 positions' if g < 27 else 'stage 4, 7x7 positions'}). Values are PRE-activations (before GELU), pooled per photo: max over positions (positive pole) and min (negative pole).",
           f"Kind: {kind}. {NOTES[kind]}"]
    vals = {}
    for w in ("base", "B", "T"):
        d = D[w]; ix = d["ix"]
        sel = fit[ix] if w == "base" else ~judge[ix]
        mx, mn = d["mx"][sel, c].astype(np.float32), d["mn"][sel, c].astype(np.float32); ii = ix[sel]; vals[w] = (ii, mx, mn, d["amx"][sel, c], d["amn"][sel, c])
        if w == "base":
            sheet(bands(mx, ii, vals[w][3], g) + bands(mn, ii, vals[w][4], g, neg=True), os.path.join(pdir, "sheet_base.png"))
            base_mx_c, base_mn_c = cmeans(ii, mx), cmeans(ii, mn)
        else:
            sheet(bands(mx, ii, vals[w][3], g)[:2] + bands(mn, ii, vals[w][4], g, neg=True)[:1], os.path.join(pdir, f"sheet_{w}.png"))
        virt = " (VIRTUAL: Base's read row on this network's block input; the neuron itself is deleted)" if (w != "base" and kind == "deleted") else ""
        o1, s1 = prof(cmeans(ii, mx)); o2, s2 = prof(cmeans(ii, mn), low=True); printed |= set(o1) | set(o2)
        txt += [f"\n== {w}{virt}: {len(ii)} photos. pooled max: median {np.median(mx):.2f}, p99 {np.percentile(mx, 99):.2f}, top {mx.max():.2f}; pooled min: median {np.median(mn):.2f}, p1 {np.percentile(mn, 1):.2f}",
                "top classes by mean pooled max: " + s1, "bottom classes by mean pooled min: " + s2]
    iiB, mxB = vals["B"][0], vals["B"][1]; iiT, mxT = vals["T"][0], vals["T"][1]; assert (iiB == iiT).all()
    if kind == "deleted":
        bm = D["base"]["mx"][iiB, c].astype(np.float32)        # base npz ix = arange(35000)
        txt.append(f"\n== does Base's information reach this block? per-photo correlation of pooled max, Base vs virtual: B {np.corrcoef(bm, mxB)[0,1]:.3f}, T {np.corrcoef(bm, mxT)[0,1]:.3f}")
    else:
        dlt = mxT - mxB; o = np.argsort(dlt)
        sheet([("largest rise T-B", [(iiB[s], vals["T"][3][s], g) for s in o[::-1][:8]]), ("largest fall T-B", [(iiB[s], vals["B"][3][s], g) for s in o[:8]])], os.path.join(pdir, "sheet_change.png"))
        o1, s1 = prof(cmeans(iiB, dlt), 10); o2, s2 = prof(cmeans(iiB, dlt), 10, low=True); printed |= set(o1) | set(o2)
        txt += [f"\n== change T minus B (pooled max, same photos): mean {dlt.mean():+.3f}, sd {dlt.std():.3f}; correlation B vs T {np.corrcoef(mxB, mxT)[0,1]:.3f}",
                "classes that RISE most in T: " + s1, "classes that FALL most in T: " + s2, "sheet_change.png: crops of the largest rises (at T's peak) and falls (at B's peak)"]
    for role, lst, gg in (("feeders (block before, approx direct edge)", NE["feeders"], g - 1), ("readers (block after, approx direct edge)", NE["readers"], g + 1)):
        txt.append(f"\n== {role}:")
        d = D["base"]; sel = fit[d["ix"]]
        for k, e in lst:
            cc = col(gg, k); o1, _ = prof(cmeans(d["ix"][sel], d["mx"][sel, cc].astype(np.float32)), 6); o2, _ = prof(cmeans(d["ix"][sel], d["mn"][sel, cc].astype(np.float32)), 4, low=True)
            printed |= set(o1) | set(o2)
            txt.append(f"  neuron {k} (edge {e:+.3f}): top classes " + ", ".join(NAMES[x] for x in o1) + " | most negative " + ", ".join(NAMES[x] for x in o2))
    open(os.path.join(pdir, "evidence.txt"), "w").write("\n".join(txt))
    shutil.copy(os.path.join(HERE, "pilot_reader_brief.md"), os.path.join(pdir, "BRIEF.md"))
    # candidates, by strata, from classes never printed
    rng = np.random.default_rng(1000 + n_i)
    okc = lambda c_: c_ not in printed and not np.isnan(base_mx_c[c_])
    hi_r = [c_ for c_ in np.argsort(-base_mx_c)[15:60] if okc(c_)]; hi = rng.choice(hi_r, 8, replace=False).tolist()
    ng_r = [c_ for c_ in np.argsort(base_mn_c)[15:60] if okc(c_) and c_ not in hi]; ng = rng.choice(ng_r, 8, replace=False).tolist()
    rest = [c_ for c_ in range(1000) if okc(c_) and c_ not in hi and c_ not in ng]; mid = rng.choice(rest, 24, replace=False).tolist()
    cand = [int(x) for x in rng.permutation(hi + ng + mid)]
    key["candidates"][name] = cand; key["labels"][name] = [NAMES[x] for x in cand]; key["strata"][name] = dict(high=hi, neg=ng)
key["pairs"] = {a["name"]: meta["neurons"][(k + 1) % len(meta["neurons"])]["name"] for k, a in enumerate(meta["neurons"])}
ids = np.random.default_rng(7).permutation(["p" + str(k) for k in range(len(meta["neurons"]))]).tolist()
for idd, a in zip(ids, meta["neurons"]): key["packets"][idd] = a["name"]
key["judge_slice"] = "odd positions inside each class's test-half list"
json.dump(key, open(os.path.join(PD, "judge_key.json"), "w"), indent=1)
print("reader packets written; judge packets need cards. pairs", key["pairs"], "packet ids", key["packets"])
