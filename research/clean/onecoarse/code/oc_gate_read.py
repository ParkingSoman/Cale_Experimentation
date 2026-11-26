# scale: neurons=A:block0-all; net=A, C-ladder, WC-ladder, L-ladder, R4C-ladder; photos=saved, broden-held1; where=mac
"""Read results/oc_gate/ (from oc_gate.py) -> results/oc_gate/READ.md. Mac, numpy only. Bars from the approved plan."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONE = os.path.dirname(HERE)
OUT = os.path.join(ONE, "results", "oc_gate")
sys.path.insert(0, os.path.join(os.path.dirname(ONE), "vecread"))
import vecread as vr  # noqa: E402

CUTS = (11, 64, 192, 384)
rng = np.random.default_rng(0)
L = []


def p(*a):
    L.append(" ".join(str(x) for x in a))


def boot_ratio(pp, nb=2000):
    """pooled (new & D) / new  over  (offA & D) / offA, photos resampled."""
    a, b, c, d = pp[:, 0], pp[:, 1], pp[:, 2], pp[:, 3]
    f = lambda i: (a[i].sum() / b[i].sum()) / (c[i].sum() / d[i].sum())
    n = len(pp)
    bs = [f(rng.integers(0, n, n)) for _ in range(nb)]
    return f(np.arange(n)), np.percentile(bs, [2.5, 97.5])


R = {c: json.load(open(os.path.join(OUT, f"n{c}.json"))) for c in CUTS}

p("# Gate maps: A (= C, WC) against the trained builds L and R4C")
p("")
p(f"2,000 Broden held-set-1 photos x 49 cells. Checks: A's block 0 on held_s3 = held_s4b0 (max rel err "
  f"{R[11]['check_s4b0_rel_err']:.1e}); C and WC gate maps = A's at every cut: "
  f"{all(R[c]['check_C_gates_equal_A'] and R[c]['check_WC_gates_equal_A'] for c in CUTS)}.")
p("")
p("## 1. How much the gates change, and where the new-on cells fall")
p("")
p("new share = new-on cells / A's on cells; lost share = lost cells / A's on cells (per kept unit, median; pooled in brackets).")
p("D-cells = cells where at least one deleted unit fired in A. Ratio = share of new-on cells on D-cells / share of all "
  "A-off cells on D-cells (pooled over units; 95% CI, photos resampled).")
p("")
p("| Cut | Build | Kept | D-cells share of all cells | new share median [pooled] | lost share median [pooled] | new-on on D-cells | chance | ratio [CI] |")
p("|---|---|---|---|---|---|---|---|---|")
V = {}
for c in CUTS:
    r = R[c]
    for b in ("L", "R4C"):
        x = r[b]
        onA, new, lost = (np.array(x[k], float) for k in ("onA", "new", "lost"))
        pp = np.array(x["per_photo"], float)
        rat, ci = boot_ratio(pp)
        sD = pp[:, 0].sum() / pp[:, 1].sum()
        ch = pp[:, 2].sum() / pp[:, 3].sum()
        V[(c, b)] = dict(new_med=np.median(new / np.maximum(onA, 1)), lost_med=np.median(lost / np.maximum(onA, 1)),
                         rat=rat, ci=ci, new_pool=new.sum() / onA.sum(), lost_pool=lost.sum() / onA.sum())
        v = V[(c, b)]
        p(f"| {c} | {b} | {len(r['K'])} | {r['Dany_share']:.2f} | {v['new_med']:.3f} [{v['new_pool']:.3f}] | "
          f"{v['lost_med']:.3f} [{v['lost_pool']:.3f}] | {sD:.3f} | {ch:.3f} | {rat:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] |")
p("")

p("### 1b. Graded measures (ADDED AFTER RUN 1: at cuts 64+ some deleted unit fires on ~every cell, so the yes/no "
  "D-cell test above has no room to show anything)")
p("")
p("D-strength ratio = mean of A's summed deleted firing on new-on cells / the same on j's A-off cells (pooled; 95% CI, "
  "photos resampled). Near miss = mean percentile of j's A pre-activation on its new-on cells among all its A-off "
  "cells (per unit, median; chance .50, 1.00 = the new cells are the ones j almost fired on in A).")
p("")
p("Margin-matched (ADDED AFTER RUN 2; a small read change opens near-miss cells first by construction): D-strength on "
  "new-on cells / on near-miss cells that STAYED off (A pre-activation >= the 10th pct of the new-on cells'). "
  "Per unit; median and share of units > 1. Above 1 = among cells equally close to firing, training opened the ones "
  "where the deleted units fired more.")
p("")
p("| Cut | Build | D-strength ratio [CI] | near miss: median [10th pct] | margin-matched ratio: median [share > 1] |")
p("|---|---|---|---|---|")
G = {}
for c in CUTS:
    for b in ("L", "R4C"):
        pp = np.array(R[c][b]["per_photo"], float)
        q = pp[:, [6, 1, 7, 3]]
        rat, ci = boot_ratio(q)
        nm = np.array(R[c][b]["near_pct"], float)
        nm = nm[~np.isnan(nm)]
        m = np.array(R[c][b]["margin_matched"], float)
        ok = (m[:, 1] >= 20) & (m[:, 3] >= 20)
        mr = (m[ok, 0] / m[ok, 1]) / np.maximum(m[ok, 2] / m[ok, 3], 1e-12)
        G[(c, b)] = (rat, ci, np.median(nm), np.median(mr), np.mean(mr > 1))
        p(f"| {c} | {b} | {rat:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {np.median(nm):.3f} [{np.percentile(nm, 10):.3f}] | "
          f"{np.median(mr):.2f} [{np.mean(mr > 1):.2f}] ({ok.sum()} units) |")
p("")

p("## 2. Which knob moved the gate: threshold (bn1 shift) or read (bn1 scale x conv1)")
p("")
p("New-on cells, pooled over kept units, as a share of the full build's new-on cells.")
p("")
p("| Cut | Build | threshold only (A's read) | read only (A's threshold) |")
p("|---|---|---|---|")
K5 = {}
for c in CUTS:
    for b in ("L", "R4C"):
        x = R[c][b]
        nf = np.sum(x["new"])
        t, rd = np.sum(x["new_thr_only"]) / nf, np.sum(x["new_read_only"]) / nf
        K5[(c, b)] = (t, rd)
        p(f"| {c} | {b} | {t:.2f} | {rd:.2f} |")
p("")

p("### 2b. Gain or tilt? (ADDED AFTER REVIEW; bar fixed before the run)")
p("")
p("alpha = the trained read's projection on A's read / |A's read|^2 (one number per unit). Gain-only gate = alpha x A's "
  "read + trained shift. Overlap = new-on cells the gain-only gate also opens / the build's new-on cells (pooled). "
  "Bar: >= .5 = the knob is one number per neuron; <= .2 = a real tilt; between = both.")
p("")
p("| Cut | Build | alpha median [10th, 90th] | gain-only overlap | gain-only new-on / build new-on | verdict |")
p("|---|---|---|---|---|---|")
GT = {}
for c in CUTS:
    for b in ("L", "R4C"):
        x = R[c][b]
        al = np.array(x["alpha"])
        ov = np.sum(x["gain_overlap"]) / np.sum(x["new"])
        GT[(c, b)] = ov
        v = "one number" if ov >= .5 else ("tilt" if ov <= .2 else "both")
        p(f"| {c} | {b} | {np.median(al):.3f} [{np.percentile(al, 10):.3f}, {np.percentile(al, 90):.3f}] | {ov:.2f} | "
          f"{np.sum(x['new_gain']) / np.sum(x['new']):.2f} | {v} |")
p("")
p("### 2c. Null for 'opened where the deleted units fired' (ADDED AFTER REVIEW; bar fixed before the run)")
p("")
p("Per unit j, on its new-on cells vs its near-miss cells that stayed off: ratio of the MEAN firing per deleted unit, and "
  "of the mean firing per kept unit (j left out), and of the input patch norm. Bar: deleted-specific only if median "
  "(deleted ratio / kept ratio) > 1.05 and > 1 for >= 60% of units; otherwise 'busy cells open'.")
p("")
p("| Cut | Build | deleted ratio (median) | kept ratio (median) | input-norm ratio (median) | deleted / kept: median [share > 1] | verdict |")
p("|---|---|---|---|---|---|---|")
NL = {}
for c in CUTS:
    for b in ("L", "R4C"):
        m = np.array(R[c][b]["null"], float)
        ok = (m[:, 1] > 0) & (m[:, 3] > 0) & (np.array(R[c][b]["new"]) >= 20)
        dr, kr, xr = m[ok, 0] / m[ok, 1], m[ok, 2] / m[ok, 3], m[ok, 4] / m[ok, 5]
        q = dr / kr
        spec = np.median(q) > 1.05 and np.mean(q > 1) >= .6
        NL[(c, b)] = (np.median(q), np.mean(q > 1), spec)
        p(f"| {c} | {b} | {np.median(dr):.2f} | {np.median(kr):.2f} | {np.median(xr):.2f} | {np.median(q):.2f} "
          f"[{np.mean(q > 1):.2f}] | {'deleted-specific' if spec else 'busy cells open'} |")
p("")

p("## 3. Gate partners: whose cells do a kept unit's new-on cells fall on?")
p("")
p("count partner = the deleted unit on whose A on-cells most of j's new-on cells fall. lift = that share / the same "
  "share on j's A-off cells. lift partner = the deleted unit with the largest lift (>= 20 new cells). Units with >= 20 new-on cells only.")
p("Agreement = the partner equals the 8.5 / jobtest partner (deleted unit whose firing j's trained firing matches best, L).")
p("")
p("| Cut | Build | Units | count partner: median share | median lift | count partner = jobtest | lift partner = jobtest | chance of a match (1 / #D) |")
p("|---|---|---|---|---|---|---|---|")
for c in CUTS:
    for b in ("L", "R4C"):
        x = R[c][b]
        ok = np.array(x["new"]) >= 20
        jp = np.array(x["jobtest_partner"])
        p(f"| {c} | {b} | {ok.sum()} | {np.median(np.array(x['partner_share'])[ok]):.2f} | "
          f"{np.median(np.array(x['partner_lift'])[ok]):.2f} | {np.mean(np.array(x['partner'])[ok] == jp[ok]):.2f} | "
          f"{np.mean(np.array(x['lift_partner'])[ok] == jp[ok]):.2f} | {1 / len(R[c]['D']):.3f} |")
p("")

p("Note: the jobtest partner comes from L's firing in both rows (no R4C jobtest exists at 11 and 64).")
p("")
p("## 4. Named units (L), read from the data with vecread.gate_card")
p("")
held = vr.Held()
imgs = np.arange(2000)
# known-answer test for gate_card (added after review): a label's own cells must read back as that label first;
# random cells must give top lifts near 1
lc = held.label_cells(7, "half")
names = vr._labels()
hits, tot = 0, 0
for l in sorted(lc, key=lambda l: -len(lc[l][0]))[:60]:
    im, ce = lc[l]
    sel = im < 2000
    if sel.sum() < 100 or names[int(l)][1] in vr.PHOTO_LEVEL:
        continue
    msk = np.zeros((2000, 49), bool)
    msk[im[sel], ce[sel]] = True
    cc = vr.gate_card(held, msk, imgs, top=1)
    tot += 1
    hits += bool(cc["cell_level"]) and cc["cell_level"][0][0] == names[int(l)][0]
rnd = rng.random((2000, 49)) < 0.05
rc = vr.gate_card(held, rnd, imgs, top=3)
p(f"gate_card known-answer test: a label's own cells read back as that label first for {hits} of {tot} common cell-level "
  f"labels (bar .8); random 5% of cells give top lifts {[li for _, _, li, _ in rc['cell_level']]} (should be near 1). "
  f"Low power: cards on a few hundred cells (n5's new-on cells: 284) are weak.")
p("")
P4 = {}
for c in CUTS:
    z = np.load(os.path.join(OUT, f"named_n{c}.npz"))
    keys = sorted({k.split("__")[0] for k in z.files if k.startswith("L_")})
    for key in keys:
        _, j, d = key.split("_")
        K = R[c]["K"]
        ij = K.index(int(j[1:]))
        x = R[c]["L"]
        p(f"### Cut {c}: {j} -> {d}")
        p("")
        p(f"A on-cells {x['onA'][ij]}, new-on {x['new'][ij]}, lost {x['lost'][ij]}. Count partner n{x['partner'][ij]} "
          f"(share {x['partner_share'][ij]:.2f}, lift {x['partner_lift'][ij]:.2f}); lift partner n{x['lift_partner'][ij]} "
          f"(lift {x['lift_partner_lift'][ij]:.2f}). Rank of {d} among {len(R[c]['D'])} deleted: by count "
          f"{int(z[key + '__rank_of_d_by_count'])}, by lift {int(z[key + '__rank_of_d_by_lift'])}.")
        p("")
        cards = {}
        for nm, f in (("old on-cells of " + j, "onA_j"), ("NEW-ON cells of " + j, "new_j"), (f"A on-cells of {d}", "onA_d")):
            cards[f] = vr.gate_card(held, z[f"{key}__{f}"], imgs)
            cc = cards[f]
            p(f"- {nm} ({cc['cells']} cells, {cc['photos']} photos): "
              + ", ".join(f"{n} {li}" for n, _, li, _ in cc["cell_level"][:6])
              + " | photo-level: " + ", ".join(f"{n} {li}" for n, _, li, _ in cc["photo_level"][:4]))
        top = lambda f: {n for n, _, _, _ in cards[f]["cell_level"][:5]}
        shared = top("new_j") & top("onA_d")
        P4[(c, j, d)] = (int(z[key + "__rank_of_d_by_count"]), shared)
        p(f"- shared top-5 cell-level concepts, NEW-ON of {j} and on-cells of {d}: {sorted(shared) or 'none'}; "
          f"with {j}'s old cells: {sorted(top('new_j') & top('onA_j')) or 'none'}")
        p("")

p("## 5. Predictions (written in the plan before the run)")
p("")
p1 = all(V[(c, "L")]["ci"][0] > 1 for c in CUTS)
p2 = V[(384, "L")]["new_med"] > V[(11, "L")]["new_med"]
p3 = all(V[(c, b)]["lost_med"] <= .1 for c in CUTS for b in ("L", "R4C"))
r339 = P4.get((64, "n339", "n476"), (0, set()))
p4 = r339[0] == 1 and len(r339[1]) > 0
p5 = all(K5[k][1] > K5[k][0] for k in K5)
p("| # | Prediction | Result | Verdict |")
p("|---|---|---|---|")
p(f"| P1 | new-on cells fall on D-cells above chance at every cut (L) | ratios "
  + ", ".join(f"{V[(c, 'L')]['rat']:.2f}" for c in CUTS) + f" | {'pass' if p1 else 'FAIL'} |")
p(f"| P2 | new share grows with the cut (L, median 384 > 11) | " + ", ".join(f"{V[(c, 'L')]['new_med']:.3f}" for c in CUTS)
  + f" | {'pass' if p2 else 'FAIL'} |")
p(f"| P3 | lost share median <= .1 at every cut, both builds | L " + ", ".join(f"{V[(c, 'L')]['lost_med']:.3f}" for c in CUTS)
  + "; R4C " + ", ".join(f"{V[(c, 'R4C')]['lost_med']:.3f}" for c in CUTS) + f" | {'pass' if p3 else 'FAIL'} |")
p(f"| P4 | n339's count partner is n476 and the cards share a top-5 concept | rank {r339[0]}, shared {sorted(r339[1])} | "
  f"{'pass' if p4 else 'FAIL'} |")
p(f"| P5 | the read moves the gate more than the threshold (every cut and build) | read-only / threshold-only: "
  + ", ".join(f"{c}{b[0]} {K5[(c, b)][1]:.2f}/{K5[(c, b)][0]:.2f}" for c, b in K5) + f" | {'pass' if p5 else 'FAIL'} |")

open(os.path.join(OUT, "READ.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
