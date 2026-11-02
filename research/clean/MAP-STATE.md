# Map track: current state (read this first)

Goal for now (Shonu, 2026-10-21): an ACCURATE coarse-to-fine map of ConvNeXt-Tiny (ImageNet). Every neuron as part
of a family (parent, siblings, children). Later sessions build on the map (coarsening, LM transfer). Do not stop to
ask for decisions inside this goal. Understand the study neurons deeply before generalizing.

## Salient concepts (from the essay and our tests)
- A neuron is a DISTINCTION (positive pole vs negative pole), described by what its inputs share, not by nouns (V vs O).
- Gate in front: a parent is on for a broad region; its children split inside it. Children can also sit on the
  parent's negative/off side (dog specialists behind a car-vs-dog gate).
- Weak activations radiate outward semantically; strong ones define the job. Trust raw activation shape.
- Early layers are dials (graded), late layers are detectors.
- Families are better found from CONNECTIONS than from activation groupings (essay: edge motifs).
- Recursive generalization: a neuron has a core plus offshoots that arrive by their own route.

## What we measured (VISION-LOG.md R8-R11, M0-M3, E1, S1-S3)
- Image-level class territory: a nested car chain exists (all vehicles 49 cls -> road vehicles 25 -> cars 15 -> cars
  vs dogs 13). The one-image "evidence chain" (top contributor) is NOT a taxonomy (follows color, background).
- Whole-network position-level containment: 60% get a parent (null 18%), but parents are weak (lift ~2). Ruler too weak.
- Exact direct contributions (all norms frozen, intermediate MLPs skipped) agree with integrated gradients.
- Hand study S2/S3 (17 neurons): families are visible at every depth. Connection finds the parent early; territory
  finds the broader parent in the last stage (the top input there is often a near-twin). 5/17 neurons unreadable
  with 6 crops (blocks 5-14).
- Link types: refinement of the same pattern; color dial -> thing of that color; near-twin relay; sibling competitor
  as strongest negative input; child on the parent's off side.
- Edit fingerprints mark families: grayscale kills color members, blur kills edge members, rot90 kills upright-world
  late members.

## Tools (research/clean/vision2/)
| Script | What |
|---|---|
| convnext.py, acts.py | model port, data (all 50k val images; per class 20 analysis / 30 sealed) |
| gallery.py | crop viewer: strongest and most negative spots |
| s2_study.py | batched: exact upstream inputs, downstream readers, edit tests |
| view.py | deep view of ONE neuron (activation bands, context, heatmaps, edits, inputs) |

## Progress (2026-10-21, later)
- S4 deep study: all 17 key neurons readable with tests (cards in map/cards/). Many first looks were wrong: neurons are
  parts and patterns (road wheels in a track, upright arch over a gap, line ends, frame position, rims), not nouns.
- S5 rule: parent score = C * log(lift) (containment at the child's on positions x log of lift over chance, parent
  broader). Median rank of the hand parent 3 of ~10k; top-5 for 9/16.
- S6 held-out blind test PASSED: judge's first pick was the rule's #1 for 3/4, #2 for 1/4; random never picked.
- Deep study found a parent for 20 of 21 sampled neurons (exception: a frame-position code).
- S7 map v1: `map/network_map.json` (top-3 parents per neuron, blocks 1-17). Its null-based "confident" flag (41%)
  is too strict: verified parents fail it in the last stage. Ranking good, certification weak.
- Map grammar: refinement chains; near-twin relays; complex-cell pooling (broader in one axis); sibling competitors as
  negative inputs; part inside whole via inhibition; "where" axes; color dials -> things of that color.

- S8: second blind sample: 19/20 have a parent; random never picked. Total 39/41 (95%) by deep study.
- S9/S10: chooser = C*log(lift) x log(1 + 1e4/edge rank) among rule top-3 + edge top-3 -> inside the judge's
  top-3 for 22/23. Map v2 `map/network_map_v2.json`: one parent per neuron, a tree; chains ~9 links at block 17.

- Atlas (private page): https://claude.ai/artifact/T9KzsaBtmbiqbWRyDNyh4M

## Tree fold (T1-T7, VISION-LOG)
- Family fold: removed child -> LS on <= 6 kept family members in the block right before it. Block 16 at 50%: wins
  back 20% of the cut's damage (dense ceiling 25%); blocks 12-17 together at 50%: 9%. Random parents: 0. Nested.
- Leaves-first removal is cheapest up to 50%. Cross-downsample folds are not solved (block 15 cannot fold).

## Small = parent + child? (2026-10-21 evening, VISION-LOG E1-E3, A1-A6)
Second reference: real small ConvNeXts (timm atto .70, nano .76 at our 224 crop; tiny .78). Same 56/28/14/7 grids, so
neurons match by activation correlation per position. Loader: `load_convnext("atto", (2,2,6,2), (40,80,160,320))`.
- One neuron holding parent + child votes, GELU(z_p + lam z_c), works on a pair (29%) but loses to the family fold
  in real weights, also with readers refit (E2, E2b, E3). Closed.
- Real small models KEEP the map's trunk neurons and LACK its leaves: a tiny neuron with more map descendants has a
  closer small twin and is better rebuilt from the whole small model (last stage R^2 leaf .06 vs >10 descendants .38;
  stage 3 .21 vs .41). Clean of energy in stage 3; in stage 4 partly tied to energy (A1, A1b, A1e).
- "The small neuron has both" is weak: a child shares its parent's small twin mostly because it resembles the parent
  (A1d, matched null).
- The tree as a removal order: no better than energy inside a block (A5, A5b). A2's big gain was cutting block 17
  wholesale (all leaves by construction). After 1 pass of distillation any order recovers the same (A6b).
- Pictures: `vision2/figs/a1/` (gallery_x.py: cross-model family sheets). They caught one invalid analysis (A1c split).
- Lesson: with 1 image per class, never split train/test by image for class-specific neurons. Use >= 3 per class.

## Learned coarse neuron and its closed form (2026-10-22, VISION-LOG L1-N10)
- L1: retrain ONLY a parent's ~1.5k weights after cutting its children -> wins back 13-65% (random-cut control ~0),
  follows the children (partial corr .2-.46) and moves TOWARD the real small model's (atto) neuron (6/6). Pictures:
  `vision2/figs/l1/`.
- Closed form = Gauss-Newton on the parent's weights against the output (true gradient, Fisher curvature), damping and
  step chosen on held-out TRAIN images. Per family it matches or beats training on held-out images (12 families,
  `n7_lib.py`, cache `hf/imagenet/n7cache/`). All 12 families at once: CF .00288 vs trained .00289 mean KL (cut .00531).
- Pitfalls found: unfair train images (10x weaker family signal) made CF look worse (N5); K-FAC curvature is 6x too
  small (positions in one image share the output gradient); per-parent solves overshoot when many parents share the
  same output error -> solve all parents JOINTLY (matrix-free CG on a gradient bank); exact top-k-class Fisher beats
  random samples.
- Whole block (N10): joint GN on block-16 children's parents (769): 8 iterations .226 vs training 80 epochs .216 (cut
  .0826 KL). Joint = all parents solved together by matrix-free CG on a gradient bank; exact top-k-class Fisher.
- Across blocks (N11): blocks 17 + 16 at 50%, deep -> shallow, PER-WEIGHT diagonal damping (crucial: one damping per
  group stalls): closed form .238 vs early-stopped training .214; top-1 .733 vs .730 (cut .715, intact .784); same top
  answer on 15/16 worst-hit images. Training overfits block-15 parents (best epoch 0); the closed form does not.
- Three blocks (17, 16, 15 at 50%, cut KL .482): closed form .236-.238 vs training .245-.247 (two seeds): a real ~.01
  gap, from compounding (block 15 alone: .128 vs .132). Report: `research/clean/NUDGE-REPORT-2026-10-22.md`.
- Code: `vision2/n11_multi.py` (MODE gn|tr, child blocks, iterations; env DIAG=1 LMS ALS TOPK SEED STOP1).

## Derive track (2026-10-22 D-series, current): `DERIVE-REPORT-2026-10-22.md`
- Coarse neuron = consensus of a co-firing group (siblings + depth), rate preserved; toy build works for coarse classes.
- ConvNeXt: dropping block 16 (.104) beats thinning 16+17 + training (.224); derived linear repairs hurt a clean drop.
- Next: training reference after a block drop; which block; readers'-view repair.

## Next (older)
0. Tree fold: cross-downsample fold fitted on data (block 15); output-aware fit; compare with dense fold composed.
1. Certification: a per-neuron check that its chosen parent is real (not just best of ~20k), so the map can say
   "has a parent" network-wide without a deep study each time.
2. Automatic link types network-wide (edit fingerprints per neuron; twin vs refinement by breadth ratio).
3. Grow the studied set where the map is weakest (middle blocks, position codes).
2. Better certification of a parent (agreement of containment rank with connection, or a per-child local null).
3. Browsable map for Shonu (families with crops), then link-type labels network-wide (edit fingerprints).
