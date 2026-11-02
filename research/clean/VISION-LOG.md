# Vision log (vision-first rule, 2026-10-21)

Rule (Shonu): reproduce each mechanism in a well-trained vision model first, then confirm on a well-trained language
model. Pretrained models only for these tests: seconds per block, no training.

Model: timm `convnext_tiny.fb_in1k` (28M weights, 18 blocks: 3/3/9/3, widths 96/192/384/768, MLP 4x), ported to
MLX (`vision2/convnext.py`). Same size and depth as the essay's "large" ConvNeXt (28.6M, 18 blocks).
Data: ImageNet val shard 0 (25,000 images, mirror evanarlian/imagenet_1k_resized_256), center crop 224.
Check: top-1 .850 on the first 2,000 images (timm reports .821 on all 50k; small sample). Port is correct.
Split: BUG FIXED before any result. The val shards are sorted by class, so an index split separates classes. Now: all
50,000 val images (2 shards); per class, images 0-19 = analysis + calibration (20,000), images 20-49 = sealed (30,000).
The 85.0% check above used the first 2,000 images = classes 0-39 only.
Blocks are numbered 0-17. Neurons = fc1 units. Poles = fc1 pre-activation z (max = positive pole, min = negative).

## R1. Sports-car chain (descriptive, the essay's figure)
For correctly classified sports-car images (class 817) in the analysis set: the block-17 neuron with the largest
activation x gradient contribution to the sports-car logit, then the block-16 neuron that contributes most to that
neuron (its spatially summed GELU activation), and so on back to block 13. For each chain neuron: its top-6 and bottom-6
analysis images by spatial max / min of z, with class names. Rendered to a PNG for Shonu. Coarseness per neuron:
distinct classes among its top-50 images. Expectation from the essay: later = finer (fewer distinct classes).

## R2. Parent diagnostics (same readings as CHILDREN-LOG TF-DIAG / TF-CTX)
Child block 17 (last stage = blocks 15-17, 3,072 neurons each), K = 96 kept (3%) for the removed set; parents =
block 16 (L-1) and blocks 15-16 pooled (block 14 is in the previous stage, another width, behind a downsample).
(Corrected before running: first draft named block 16 as child.)
Attribution of parent p to child c on an image: activation x gradient of c's spatially summed activation w.r.t. p's
hidden, through the real block (dwconv, LayerNorm). Fixed readings, as TF-DIAG:
- TF-CTX analog: modal-parent share over a child's top-50 images, top parent's share of |attribution| mass.
- Held-out R^2 of the removed output from top-n parents (n = 1, 3, 10, 30, 100) vs dense, same position.
Comparison numbers from SmolLM2: modal share .10-.13, top share of mass .006-.008, top-10 = 19-27% of dense.

## R3. Tree fold on block 17 (rules of CHILDREN-LOG TF1, vision version)
K = 96 and 768. Calibration: 2,000 analysis images, all 49 positions. Arms: cut_ls, tree1 (one parent by
attribution, alpha by LS, fold into block 16's fc2), dense fold (all block-16 neurons, CV ridge; RCF analog), and raw
(no kept refit) versions. Score: sealed top-1 accuracy (first 5,000 sealed images), share of the cut's accuracy loss
won back. Pass: tree1 wins back >= 50% of the dense fold's gain at K = 96.

## Results so far

Port check (fixed split): top-1 .778 on 1,000 sealed images (every 30th), .775 on 2,000. Flipped kernels .364, BGR
.688, so orientation and color order are right. The gap to timm's .821 is most likely the mirror's resize and JPEG
re-encode. Well-trained model; accepted.

R1 (`results/vision-r1-chain-817.json`, figure `vision2/figs/chain_817.png`): 14 correct sports-car analysis images.
Chain for image 40850 (yellow sports car): block 13 n540 and block 14 n1056 broad (48-49 distinct classes in top-50);
block 15 n2031 = yellow things (yellow cab, canoes, whistle) -> specific to this image's color; block 16 n1522 = cars
and car interiors (20 distinct classes); block 17 n1594 = chrome / shiny metal parts (grille, oil filter, chain),
negative pole = dingoes, coyotes, dogs (the essay's car-vs-dog gate, found unprompted). Across 8 images the top
block-17 neuron is 1594 or 2283 (2 neurons); block 16 repeats 1522 in 3 of 8. Near the output the parent is stable
across images; earlier it follows each image's color and context.

R2 (`results/vision-r2-parents.json`), child block 17, K = 96, 100 sampled removed children:
| Measure | ConvNeXt-Tiny | SmolLM2-135M |
|---|---|---|
| top parent share of |attribution| mass | .083 (uniform .0003) | .006-.008 |
| modal top parent over a child's top inputs | .25 | .10-.13 |
| held-out R^2 of removed output: top1 / top10 / top100 / dense | .104 / .191 / .188 / .171 | (L16) .040 / .128 / .242 / .480 |
Dense ridge was re-run with a wider ridge grid (up to 10): .171; pooled blocks 15-16 dense .166. Fixed reading of
TF-DIAG: top-n <= 10 reaches >= 50% of dense -> a few-parent fold is viable in vision. Absolute R^2 is low: the
same-position linear fit captures only ~1/5 of the removed output (the 7x7 depthwise conv mixes positions).

R3 deviation (logged before running): parent = best single same-position predictor (|corr|), not attribution, for
all 2,976 removed children (attribution for all would take ~10 min). Agreement between the two on the 100 sampled
children is reported. Everything else as pre-registered.

R3 first result (`results/vision-r3-treefold.json`): tree1 / dense = .66 (K96) and .60 (K768), above the .50 bar.
BUT the baseline is broken. Sanity check (`vision2/r3_sanity.py`), block-17 cut accuracy by K kept (cut_ls):
3072 .780, 2304 .775, 1536 .764, 768 .725, 384 .709, 96 .726, 16 .755, 1 .766. NOT monotone: keeping 1 neuron beats
keeping 384. (1) Block 17's MLP is nearly unneeded (-1.4 points with almost all of it gone). (2) Keeping a partial set
of the highest-energy neurons hurts more than dropping them all. Suspect (essay, Residual Stream): an unread saliency
direction that the head's LayerNorm uses to rescale everything; big writers push on it, a partial set unbalances it.
R3's pass is NOT accepted until the baseline is sound. Next: block importance (each block's MLP replaced by its mean
output), and a saliency check of the stage-4 stream.

R4 (`results/vision-r4-blocks.json`): replacing ONE block's MLP branch with its mean output costs 1-4 points for
blocks 1-17 (block 0: 34 points; blocks 14-16: 3.7-4.4; block 17: 1.9). Saliency coordinates exist (stream coords 97
and 159: mean |x| 19.4 and 14.2 entering the head; median coord .6). But they do NOT explain the broken cut: block-17
neurons put a median .3-.6% of their write on them, and ranking without them changes nothing (`r3_sanity2.py`).
New hypothesis (pre-registered before the check): the timm ConvNeXt-T was trained with stochastic depth (whole blocks
randomly skipped, rate rising linearly to .1 at the last block). The net is used to losing WHOLE blocks, never part of
one. Prediction: partial-cut curves are non-monotone (worse than dropping the block) in LATE blocks and monotone in
EARLY blocks. Check: cut_ls accuracy vs share kept on blocks 2, 8, 14, 16 (3,000 sealed images), plus whole-branch
mean replacement.
R5 (`results/vision-r5-curves.json`), cut_ls accuracy by share kept (.75/.5/.25/.125/.03/0 = branch -> mean):
block 2 .783/.780/.764/.742/.725/.721; block 8 .781/.778/.774/.771/.766/.768; block 14 .779/.769/.762/.754/.753/.748;
block 16 .768/.757/.739/.736/.737/.747. Monotone in blocks 2, 8 (tie at the end), 14. NOT monotone in block 16 (and 17):
in the last stage, a partial cut is worse than dropping the block. Consistent with stochastic depth only in part (the
rate rises smoothly, but the effect appears only in the last stage). R3 on block 17 is therefore void.
R6 (pre-registered): R3 repeated on block 14 (parent block 13, same stage, 1,536 neurons), K = 46 (3%) and 384
(25%), 3,000 sealed images (per class images 20-22), calibration 2,000 images x 49 random positions (same positions
for parent and child). Arms as R3 plus "drop" (branch -> mean). Same pass rule (tree1 >= 50% of dense's gain at the
3% size). Also: agreement of best-predictor parents with attribution parents on 100 sampled children.
R6 result (`results/vision-r6-treefold-b14.json`): block 14, sealed top-1 (3,000 images). Drop .748, intact .784.
3%: cut_ls .752, tree1 .755, tree3 .756, dense .759 -> tree1 / dense = .43 (rule .50: FAIL, barely). 25%: cut_ls .759,
tree1 .761, tree3 .764, dense .770 -> .15. Arms differ by tenths of a point: top-1 on 3,000 images is too coarse.
Parent agreement: best-predictor parent = attribution top parent for 18% of children; in attribution top-10 for 61%.
So R3/R6 did not test the essay's attribution parent.
R7 (pre-registered): same as R6 but (1) parent = top attribution parent (mean |activation x gradient| over the child's
top-20 analysis images), for EVERY removed child, as originally planned; (2) primary score = mean KL(intact || arm) on
the 3,000 sealed images (the vision analog of dCE), accuracy secondary; share = (KL cut - KL arm) / KL cut. Blocks 14
and 12 (parent 13, 11). Arms: drop, cut_ls, tree1, tree3, tree1_pred, dense, and nested (raw) tree1, tree3, dense.
Pass: tree1 >= 50% of dense's KL gain at 3%, on both blocks.
R7 result (`results/vision-r7-treefold-b{14,12}.json`), KL share of the cut's KL won back:
block 14, 3%: dense .33, dense_raw .31, tree1 -.02, tree3 .04, tree1_pred .05; 25%: dense .31, tree1 .005, tree3 .02.
block 12, 3%: dense .10, tree1 -.01, tree3 -.02; 25%: dense .05, tree1 -.02, tree3 -.05.
FAIL on both blocks. R6's "tree1 = .43 of dense" was top-1 noise. In vision, attribution is 10x more concentrated
than in SmolLM2-135M (R2), but one parent (or three) cannot carry a removed neuron. The dense fold (RCF analog) works
at block 14 (~1/3 of the cut's KL), weakly at block 12.

## Verdict (2026-10-21)
- Essay chain reproduced in part (R1): late-block neurons are semantic (chrome car parts vs dogs at block 17); near the
  output the parent is stable across images of one class; earlier blocks follow each image's color and context.
- One-fixed-parent "tree fold": FAIL in vision too (R7, KL). Not a weak-model artifact (SmolLM2-1.7B matches 135M).
- Dense fold (RCF analog) works in vision at block 14 (~1/3 of the cut's KL).
- Side finding: in the last stage, a partial MLP cut is worse than dropping the block (R5); any single block's MLP
  can be dropped for 1-4 points (R4). Depth looks like a natural dial in this model ("ends early"). Next test candidate.

## R8. Is the R1 chain nested? (Shonu's question, 2026-10-21, rules written before the run)
Shonu: block 15 n2031 is "yellow things", but block 16 n1522 (cars) is not only yellow things. So 15 -> 16 is not a
coarse-to-fine split. Note: R1 used the essay's own method ("the neuron that contributes most to the next one on this
image"). That method does not force nesting. This step measures nesting directly.
- Territory of a neuron: classes whose mean (over 20 analysis images) of spatial-max z is high:
  t = (class mean - median over classes) / (max - median) >= .5. Robustness at .3.
- Containment of a link (parent at block b, child at b+1): share of the child's territory classes inside the
  parent's territory. Breadth = territory size.
- Null: same containment for 1,000 random parent neurons of block b.
- A link is NESTED if containment >= .7, parent breadth > child breadth, and containment > the null's 95th percentile.
- Applied to all 32 links of the 8 R1 chains.
- Search: for each R1 child, the block-b neuron with the best Jaccard among those with containment >= .8
  (a nested parent, if one exists). Report its contribution rank to the child on the chain's image (grad x act).
  If nested parents exist but rank low, the chain "on this image" and the territory tree are different things.

R8 result (`results/vision-r8-nesting.json`): only 7 of 32 R1 links are nested at threshold .5 (8 of 32 at .3).
Shonu's link: block 15 n2031 (yellow birds, cab, canoe) -> block 16 n1522 (cars, car parts): containment .09, BELOW
the random null (.18). Several R1 chains run through bird or dog neurons (e.g. image 40858: all of blocks 13-16 are
birds / dogs / spiders, then block 17 is cars). Nested vehicle neurons exist (block 13 n417, block 14 n1227, block 15
n286, block 16 n2886), but on a single image they often rank low as contributors.

## R8b. A territory chain and why R1 picks off-territory neurons (rules before the run)
1. Sign check: for each R1 link, is the parent ON (summed GELU > 0) on that image, or OFF (negative)? A neuron that is
   OFF can still get a large positive grad x act (small negative activation x negative gradient): "being off helps".
2. Territory chain: start at block 17 n2283 and n1594. At each earlier block (16 down to 11), take the nested parent
   (containment >= .8 of the child's territory, broader than the child, best Jaccard). Report breadth per block.
   Coarse-to-fine predicts breadth that grows going back.
3. For each territory link, the parent's mean contribution share to the child over all 14 correct sports-car images
   (grad x act), and its rank. Null: mean share of 200 random parents.

R8b result (`results/vision-r8b-territory.json`): all 32 R1 parents are ON on their image (sign worry ruled out).
They contribute because they fire on that photo (its yellow color, its background), not because they hold the car's
category. Territory chain from block 17 n2283 (13 classes, cars): block 16 n2886 (15, cars + karts) <- block 15 n1539
(25, road vehicles) <- block 14 n1072 (49, all vehicles incl. trains, scooters, canoes) <- block 13 none. Breadth grows
going back: 13 -> 15 -> 25 -> 49. Wiring: block 14 n1072 is the #1 mean contributor to n1539 over 14 sports-car images;
block 15 n1539 pushes n2886 DOWN (rank 3067/3072); block 16 n2886 -> n2283 tiny (rank 34). n1594 has no nested parent.
R8c (`results/vision-r8c-scale.json`), all neurons, share with a nested parent one block back vs class-shuffled null:
block 12 .14 (.01), 13 .25 (.01), 14 .35 (.01), 15 .62 (.13), 16 .69 (.20), 17 .67 (.27). Median child territory
shrinks with depth: 44-45 classes (blocks 12-13), 40 (14), 16 (15), 10 (16), 6 (17).
Figures: `vision2/figs/territory_chain_817.png` (nested), `vision2/figs/evidence_chain_817_classes.png` (R1, class view).
Verdict: Shonu is right. R1's chain is an evidence path for one photo, not a coarse-to-fine tree (7/32 links nested;
block 14 -> 15 is a "yellow" path). A territory tree does exist and is far above chance, growing coarser going back.
R7 (tree fold) used evidence parents (attribution), i.e. the R1 kind. Territory parents were never tested as fold parents.

## R9. Label-free territory (Shonu, 2026-10-21: find the chain without anything vision-specific). Rules before the run.
Same as R8 with "class" replaced by "input". The only vision-specific line is "per-image spatial max of z" (for a
language model it becomes "per-token activation").
- Territory = analysis images with a >= median + .5 (max - median), per neuron. Robustness: .3, and max replaced by
  the 99.9th percentile (one outlier image can set the max).
- Containment, breadth > child, best Jaccard: as R8. Null: parent-block activations with the image order shuffled.
Pass (structure, not exact neurons):
 (a) share of neurons with a nested parent one block back is far above the shuffled null, and rises with depth (R8c);
 (b) from block 17 n2283, a nested chain exists with breadth growing going back;
 (c) agreement with class-level: correlation of input-level vs class-level containment over all pairs of blocks 16->17,
     and how often the top nested parent is the same neuron (vs chance 1/3072). Exact match is a bonus only.

## R10. Are territory parents wired as gates? (rules before the run)
For blocks 13->14, 15->16, 16->17 (same stage, no downsample), sample up to 150 children that have a class-level
nested parent (R8c).
- Attribution: grad x act of all block-b neurons on the child's summed GELU activation, over the child's top-20
  territory images. Report the nested parent's mean share and its percentile among all block-b neurons.
- Static: virtual weight = child fc1 row . (LN weight * (dwconv kernel sum * parent write - its channel mean)),
  parent write = gamma * fc2 column. Percentile of the nested parent among all block-b neurons.
- Null: random parents with breadth within 20% of the nested parent's.
Readings, fixed now:
- Median percentile >= .9 on both -> gate in front holds: territory parents feed their children. They are the
  candidate fold parents.
- Median near .5 (.4-.6) -> the tree is co-firing without direct wiring. Folding into a territory parent then has
  no mechanism behind it. The link must go through the stream (many writers), not one parent.
- Median <= .1 -> territory parents inhibit their children (competitors, not gates).

R9 result (`results/vision-r9-inputs.json`): per-image hard sets give nested parents for only 0-5% of neurons
(shuffled null 0.00), all three threshold variants. Pass (a) fails on size (far above null, but not the class-level
62-69%, and not rising cleanly with depth). (b) fails: no nested parent for n2283 at block 16. (c) partial:
containment correlates with the class level at .48-.55; same top parent 9-18% (chance .0003).
Reading: the structure is there but per-image activations are noisy (one wheel in a non-car photo), so hard
containment >= .8 breaks. Class averaging was a smoothing step.
## R9b. Label-free smoothing (rules before the run)
Groups from the model itself: k-means (k = 1000, also 300) on the final pooled feature (head LayerNorm output) of
the 20k analysis images. Language analog: k-means on the final hidden state of tokens. Then R8 exactly, with groups
in place of classes, and a group-shuffled null. Pass: (a)-(c) of R9. Also report how well groups match classes (NMI).

R9b result (`results/vision-r9b-groups.json`): k = 1000 groups (NMI .88 with classes) PASS (a)-(c): nested parent
.13/.26/.37/.61/.64/.63 (blocks 12-17) vs null .02-.23; chain n2283 <- 16 n2886 <- 15 n2182 <- 14 n1072 (11 -> 15 ->
25 -> 48 groups), same neurons as R8b except block 15 (n2182, also road vehicles); same top parent as class level
36% (chance .0003). k = 300 is too coarse (groups merge the leaf classes; the chain degrades). Label-free recipe:
average activations inside the model's own input groups, groups finer than the finest neurons.

## R11. Is it ALL neurons? (Shonu: "a pattern of ALL neurons ... if not, we are looking at the network wrong")
Rules before the run. k = 1000 model groups, blocks 12-17, threshold .5. For every neuron, which of these holds:
 1. nested parent one block back (R9b);
 2. else nested parent in ANY earlier cached block (11 .. b-2): skip connections through the stream;
 3. else nested in the union of 2 parents of block b-1 (greedy: best single, then best add), containment >= .8;
 4. else ROOT: broader than the median neuron of block b-1 (a tree needs roots; a root has no parent by definition);
 5. else ORPHAN: narrow and no parent by 1-3.
Nulls: 1-3 with group-shuffled parents. A tree for ALL neurons predicts orphans near 0 after nulls.
Also report the best single containment of orphans (near-misses .6-.8 vs none).

R11 result (`results/vision-r11-all.json`), share of neurons [shuffled null]:
| block | parent b-1 | parent earlier | 2 parents | root | orphan |
| 12 | .13 [.02] | .00 | .29 [.06] | .44 [.63] | .14 [.29] |
| 13 | .26 [.03] | .01 | .41 [.06] | .22 [.48] | .11 [.42] |
| 14 | .37 [.02] | .04 | .37 [.07] | .13 [.46] | .08 [.43] |
| 15 | .61 [.12] | .09 [.04] | .16 [.17] | .06 [.19] | .08 [.47] |
| 16 | .64 [.18] | .10 [.09] | .16 [.12] | .09 [.36] | .01 [.25] |
| 17 | .63 [.23] | .13 [.14] | .11 [.11] | .13 [.37] | .01 [.16] |
Late stage: ~all neurons are a nested child or a broad root; orphans 1-8% vs null 16-47%. Earlier: two-parent cover
(child spans two parents' territories) is far above null (.29-.41 vs .06), so blocks 12-14 MERGE branches more than
split them. The semantic ruler (final-feature groups) may also be the wrong ruler for early blocks.

R10 result (`results/vision-r10-wiring.json`), nested parent, median percentile among all parents [random parent]:
| link | grad x act | top 10% / bottom 10% | virtual weight | positive |
| 13->14 | .15 [.46] | .25 / .47 | .24 [.58] | .37 |
| 15->16 | .01 [.43] | .29 / .65 | .70 [.49] | .61 |
| 16->17 | .02 [.44] | .39 / .59 | .89 [.59] | .75 |
Bimodal: territory parents are rarely neutral. In the last stage, static weights say "feeds" but the local gradient
says "pushes down". Suspect: LayerNorm saturation. On the child's top images the stream already points along what
the parent writes; more of it mostly grows the norm, which LN removes, so the LOCAL slope is negative.
## R10b. Total effect instead of local slope (rules before the run)
Integrated gradients (16 steps, all block-b hiddens scaled 0 -> 1 together) and single-neuron ablation (parent
hidden set to 0) on the child's summed GELU, child's top-20 images. Same children, same random parents.
- IG percentile median >= .9 and ablation drop > random -> parents FEED; R10's negative is a norm-saturation artifact
  of grad x act (which R1, R7 and the essay's figure method all use).
- IG still <= .1 -> parents truly inhibit at the operating point.

R10b result (`results/vision-r10b-total.json`), IG percentile of the nested parent, median [random]; top 10% / bottom 10%:
13->14 .28 [.47] .28/.43 | 15->16 .15 [.47] .41/.48 | 16->17 .96 [.42] .53/.47. Single-parent ablation changes the
child by ~0 (median), random also ~0: one parent alone barely matters (many writers share the stream).
None of the fixed readings applies. Finding: territory parents are BIMODAL. About half strongly feed the child, about
half strongly inhibit it, almost none are neutral (random parents sit at the middle). The norm-saturation suspicion
explains part of R10 (16->17 median moved .02 -> .96) but not the inhibitors.
Feeders vs inhibitors (scratch check, not pre-registered): same breadth ratio (parent 3-4x the child), same small
positive co-firing inside the child's territory (+.14 to +.23). They differ in static weight sign (feeders 74-90%
positive; inhibitors 14-57%). Open: what an inhibiting parent is (the essay's negative-pole sibling? an
"explained elsewhere" signal?). Not resolved.

## R10c. Correction to R10/R10b (rules before the run)
The R10b "bimodal" reading is a null artifact: breadth-matched random parents are mostly OFF on the child's top
images, so their IG sits near zero (middle percentile). The nested parent is ON by construction, so it lands in a tail
either way. At 16->17, top10 + bottom10 = 1.00. That is magnitude, not wiring. Withdrawn.
Rerun: random parent = a neuron whose mean activation on the child's top-20 images is within 20% of the nested
parent's (activation-matched). Report the share with POSITIVE IG (signed), nested vs random.
Readings: nested >= .75 positive and random about .5 -> gate in front holds (territory parents feed).
Both about .5 -> territory tree = co-firing, no privileged feeder; "fold into the territory parent" has no mechanism.
Also R11 check: in the two-parent cover, is the first greedy pick broader than the child, or a near-twin (Jaccard
>= .6 with the child)?
Notes owned: R9's per-image territory (my design) failed; k-means smoothing fixed it. grad x act on OUR R1/R7 is
fooled by LayerNorm saturation. The essay does not say which method its figure used.

R10c result (`results/vision-r10c-signed.json`): share with positive IG, nested vs activation-matched random:
13->14 .43 vs .50 | 15->16 .45 vs .47 | 16->17 .53 vs .54. Pre-registered reading: no privileged POSITIVE feeder.
Extra (not pre-registered): median |IG| nested vs random 3.3 vs 1.0, 10.5 vs 4.0, 27.8 vs 5.7. Territory parents are
3-5x more strongly coupled to their child than equally active neurons, but the sign is a coin flip.
R11 two-parent check: the first pick is broader than the child in 96-100%, near-twin in 0-2%, covers a median 2/3 of
the child; breadth ratio 3.7-6.1. So "mostly inside one broad parent, the rest inside a second".

# MAP TRACK (Shonu, 2026-10-21): map EVERY neuron as coarse-to-fine; "maybe nouns is the wrong ruler"
Essay reread (full text + microscope page source). What the essay gives as method: a neuron is a DISTINCTION (verb),
positive pole vs negative pole (V vs O); describe it by what its inputs share, not by nouns; rank inputs by the
strongest response anywhere in the image and crop at that spot (positions); look at signed connections; gate in front:
the gate is positive on cars AND dogs (shared pose), the child splits inside it. The microscope (medium ConvNeXt) shows
top crops, negative-pole crops, FV, signed adjacent-block effects, logit ablations, RelP peers. It has NO tree or
family labels, so there is no scheme to copy.

## M0. Two-pole gate check on the R10c pairs (rules before the run)
Child's negative-pole territory: groups (k = 1000, R9b) whose mean spatial-min z is far below the median:
(median - mean_min) / (median - min over groups) >= .5. Parent's positive territory as R9b.
Split R10c pairs: BOTH-POLE (parent positive territory covers >= .6 of the child's negative territory too) vs
POS-ONLY (covers < .3). Prediction (essay, gate in front): BOTH-POLE pairs have positive IG far more than .5; POS-ONLY
pairs are the coin flip. Also: search blocks 14-16 for a neuron whose positive territory covers both poles of block 17
n2283 (cars vs dogs). The essay says a shared-pose neuron like this exists in ConvNeXts. Report found or absent.
M0 result (`results/vision-m0-twopole.json`): inconclusive. R10c pairs were chosen on the positive pole only, so few
cover both poles (BOTH-POLE n = 11 / 4 / 4; IG positive .36 / .25 / .25 vs POS-ONLY .45 / .47 / .55). No block 14-16
neuron covers both poles of n2283 (best: block 14 n520, vehicles + dogs, covers + .73, - .23). n2283's image-level
negative pole is broad (40 groups: dogs, birds, scorpions). Image-level min is crude: one spot sets it.

## M1. Position-level map, last stage first (rules before the run)
Inputs = image positions (essay microscope: strongest spot, crop there; also the per-token analog).
- Images: 5 per class from the analysis split (5,000) for the map; 2 per class from the sealed split (2,000) to check.
- Child block b in 15-17 (7x7 positions). Parent blocks 12-16. Parent value at a child position = max of parent z over
  the child's view: 2x2 max-pool across a stage boundary, then 7x7 max-pool (the child's dwconv window).
- Groups: k-means (K = 2000) on what the child reads (its LayerNorm(dwconv) input), per child block.
- Territories over groups as R9b: positive (mean z), negative (mean z, low side), parent (mean pooled z), rule .5.
- Two tests per child: POS (child + inside parent +, containment >= .8, parent broader) and GATE (child + AND child -
  inside parent +, containment >= .8 on each pole).
- NULL (pre-registered, decides if this tests neurons at all): shuffled-weight readouts of the same input (each fc1
  row permuted, same bias), same pipeline. If null readouts nest about as often as real neurons, the result is stream
  geometry, not neuron structure.
- Sanity: in the last stage the 7x7 window covers nearly the whole image, so POS one-block-back rates should be close
  to R9b (.64 / .63 for blocks 16 / 17). A large gap = a pooling bug.
- Sealed: containment of the chosen parent on sealed group means within .1 of the analysis value.
Estimated run time 5-10 min.
M1 result (`results/vision-m1-positions-15-16-17.json`): FAILED its sanity check. One-block-back POS rate .03 / .05 /
.12 (R9b .61-.64), sealed within .1 only 9-31%. By the rule above: a pipeline problem, not the network.
Diagnosis (`m1_diag.py`, block 17): k-means groups (K = 2000) on what the child reads explain only 37% of a neuron's
z variance (median), and a neuron's top-1% positions spread over ~263 groups. No dominant stream coordinate (top
coordinate = 1.5% of variance). The groups are far coarser than the neurons at position level, so every territory blurs.
Grouping is dropped.
Natural "on" point: z > 0 (GELU gate open). Median on-rate per position: block 1 .18, 4 .08, 8 .11, 13 .05, 16 .005,
17 .006. Breadth shrinks with depth.

## M2. Whole network, natural on-regions, no groups (rules before the run)
- Images: 3 per class from the analysis split (3,000), 1 per class from the sealed split (1,000). 49 sampled positions per
  image at each child block's resolution (all 49 in stage 4).
- Child on-region: z_c > 0. Weighted containment C_w(p|c) = sum over the child's on positions of z_c * [parent on] /
  sum z_c (strong activations count most: essay, weak activations radiate). Unweighted C also reported.
- Parent on at a child position, two variants: VIEW = max of parent z over the child's 7x7 window (after 2x2 max-pool
  across each stage boundary) > 0; POINT = same position only (2x2 max-pool across stages, no 7x7).
- Parent of c = the neuron in ANY earlier block (0 .. b-1) with the smallest breadth among those with C_w >= .8 and
  breadth > c's on-rate. INFORMATIVE only if the parent's breadth <= .5 (a parent on at half of all positions says little).
- Coverage = share of blocks 1-17 neurons with an informative parent. Block 0 = roots by construction.
- Sealed: the chosen parent's C_w on sealed images >= .7.
- Null: 256 shuffled-weight readouts per child block, each with its on-rate matched to a real neuron, same pipeline.
- Shonu's claim (strong form) holds if coverage >= .99 on analysis, sealed holds, and the null is parented at most half
  as often. Also reported: nearest block that has a parent; breadth chain (parent breadth / child on-rate).
- If stage 1-2 fail set containment, the pre-registered fallback is the graded "dial" test (essay: early layers are
  dials). Not built yet.
Estimated run time 10-20 min.
M2 result (`results/vision-m2-onregions.json`, parents in `hf/imagenet/m2_parents.npz`), 9 min.
Whole network blocks 1-17, share with an informative parent: VIEW .425, POINT .442. The strong form FAILS with this ruler.
| block | VIEW coverage [null] | POINT coverage [null] | sealed holds (view/point) |
| 1-2 | .04/.01 [.02/.00] | .18/.06 [.04/.02] | ok |
| 3-6 | .16/.31/.10/.25 | .51/.61/.41/.70 [.04-.27] | .84-.97 |
| 7-11 | .02-.13 [.00-.02] | .07-.34 [.00-.03] | .96-1.0 |
| 12-14 | .50/.45/.24 [.02-.08] | .39/.35/.20 [.04-.10] | .96-1.0 |
| 15-17 | .82/.87/.85 [.18-.32] | .73/.71/.53 [.30-.51] | .83-.95 |
Most unparented neurons have only a TRIVIAL parent (on at more than half of all positions). The nearest parent is mostly
one block back. Parents are ~10x broader than the child (view), 4-8x (point) in early blocks, 20-60x in the last stage.
Point beats view in early blocks (view pooling makes early parents too broad). Late stage: the null is high in POINT.

## M3. Other rulers, from the essay (rules before the run)
1. NEGATIVE-POLE parents. Essay: the gate is positive on cars and negative on dogs, and "the dogs go on to their own
   area". So dog specialists should sit inside the gate's NEGATIVE pole. Parent negative region: z_p <= .5 x (its most
   negative value), measured as the min over the child's view (VIEW) or at the point (POINT). Same containment rule.
2. Either pole, either variant counts ("EITHER"); the null gets the same four chances.
3. Threshold curve: coverage at C_w >= .8 / .7 / .6 (near-misses vs none), real vs null.
Pass rule unchanged: strong form needs >= .99 coverage, sealed holds, null at most half. Later (M4, only after this):
core + offshoot (child inside the union of two parents; essay: offshoots "arrive by their own route") and the graded
dial test for early blocks.
Estimated run time 12-18 min. Matrices saved temporarily (~3 GB, float16) to `hf/imagenet/m3_*.npz`, deleted after M4.
M3 result (`results/vision-m3-poles.json`, 12.5 min). Whole network blocks 1-17, share with an informative parent
(either pole, either variant) [null]: C >= .8: .600 [.179]; .7: .862 [.495]; .6: .986 [.934]. At .6 the null catches
up, so low thresholds say nothing. Real minus null: .42 / .37 / .05.
Per block at .8: 1-2 .27/.07; 3-6 .61/.67/.45/.76; 7-11 .35/.34/.35/.17/.20; 12-14 .65/.59/.36; 15-17 .91/.93/.89
(null .41-.66 in the last stage). Negative-pole parents add .17 overall (M2 POS-only .43). POINT wins in stages 1-2
and early stage 3; VIEW-positive wins in late stage 3 and stage 4. Weakest zones: blocks 1-2 and mid stage 3 (7-11).
Strong form still FAILS with this ruler. Next: look at unparented neurons (essay method: what do the inputs share).

M3 audit (`results/vision-m3-audit.json`, figs `parented_pairs_late.png`, `unparented_b2.png`, `unparented_b9.png`):
- Accepted parents are weak: median lift (containment / parent breadth) 1.9-3.3; p10 1.7-1.9. Parents 3-8x broader
  than the child in blocks 1-14, 21-40x in the last stage. With a lift floor of 3: whole network .246 (vs .600).
- Every unparented neuron HAS some parent with containment >= .8, but that parent is on at more than half of all
  positions (the breadth cap binds). Bucket "C < .8" = 0 and "only narrower" = 0 in every block.
- Six random late pairs, by eye: 2 clear families (block 17 n2394 circular metal parts <- block 10 n675 circular shiny
  objects; block 13 n956 dotted textures <- block 3 n271 negative pole = fine speckle), 1-2 plausible, 2-3 not visible.
Reading: on-region set containment is a weak ruler. Most neurons sit inside SOME broad region, few inside a narrow one.
The essay's own hint: "edge motifs [connections] predicted where ablation would matter much better than comparable
groupings of activations". R10c: territory parents are 3-5x more strongly connected than equally active neurons.
Candidate next ruler: families from connections (total-effect edges), then check containment and visual sense.

## E1. Families from connections (Shonu: go ahead, 2026-10-21). Rules before the run.
Essay: edge motifs beat activation groupings. Edge from parent p (any earlier block) to child c = p's DIRECT
contribution to c's pre-activation through the residual stream, averaged over c's on positions weighted by z_c:
edge(c,p) = V(c,p) * E_w[h_p / sigma_c], where h_p = GELU(z_p) at the child's position (2x2 average across a stage
boundary), sigma_c = the child's LayerNorm scale at that position, and V = static virtual weight: child fc1 row x
LN weight x (dwconv kernel sum * parent write - channel mean), parent write = gamma * fc2 column, passed through
each downsample (its LayerNorm weight, mean removal, mean 1/sigma from data, 2x2 kernel sum). Indirect paths ignored.
Checks and readings, fixed now:
- E1a validity: Spearman correlation of edge vs R10c's integrated gradients over each child's 11 parents (nested + 10
  activation-matched random). Median >= .5 -> edges usable. Else stop and fix.
- E1b concentration: top positive parent's share of the child's total positive direct input.
- E1c family check: the child's TOP positive-edge parent. Is it broader than the child, and does it contain the child
  (M3 containment, positive pole, point or view, >= .8)? Lift = containment / breadth. Null: a random parent from the
  same block with breadth within 20%.
- E1d by eye: 6 random child / top-edge-parent pairs, blocks 3-17.
Runtime estimate 6-10 min.
E1 edges (`hf/imagenet/e1_edges.npz`, 1.3 GB temporary, ~1 min). E1a validity vs IG, Spearman median / top-20 overlap:
13->14 .72/.45, 15->16 .46/.38, 16->17 .31/.53. FAILS the .5 bar in the last stage. Stop and fix.
E1a-diag (rules before the run): same 40 children per link, three measures over all previous-block parents:
IG (real LayerNorm), FROZEN = gradient x activation with the LayerNorm scale held fixed (exact direct contribution,
exact 7x7 dwconv), and the EDGE. Spearman(edge, frozen) low -> my spatial / weighting approximation is the problem.
Spearman(frozen, IG) low -> LayerNorm nonlinearity is the gap (then the edge is the standard frozen-norm attribution
and IG is measuring something else; attribution graphs freeze norms).
E1a-diag (`results/vision-e1a-diag.json`): frozen-norm vs IG .99 / .84 / .79 (LayerNorm is NOT the gap); edge vs
frozen .73 / .55 / .38. The gap is my spatial shortcut (kernel sum at one point), worst in the last stage where the
7x7 window covers the whole 7x7 map. Fix if needed: exact frozen-norm direct contributions.

# DIRECTION CHANGE (Shonu, 2026-10-21): "you're moving in reverse. The powerful method was hyper-specific first, then
# find a method that generalizes, taking advantage of the vision model's ease of interpretability."
Whole-network rulers (M2, M3, E1) paused. New order: understand ONE family closely the way the essay did the V neuron
(top crops, negative pole, upstream connections, shared pattern, test the hypothesis on other inputs), then the next
node up the chain, then siblings. Only then write the general method from what worked.

## S1. The car neuron's family, by hand (block 17 n2283, cars vs dogs)
Tool: exact direct contribution with all norms frozen (no spatial shortcut): for each earlier neuron p, h_p x the
gradient of the child's pre-activation along the direct stream path only (residual + downsamples, intermediate MLPs
skipped), at the child's strongest spot, on its top-50 images. Then look at the top contributors' crops (both poles).

## S2. Hand study of 20 neurons + the car neuron (Shonu: specific first, ~5 per depth band, fast). Rules before the run.
Sample (seed 0): 5 random neurons per band among active ones (on-rate .001-.5, from M3): early 1-5, middle 6-11,
late-middle 12-14, last 15-17. In each band the 5th is HELD OUT (not used to choose the general method). Plus 17 n2283.
Per neuron, all batched:
- crops: strongest spots and most negative spots (top images, 3 per class analysis set);
- upstream: exact direct contributions (all norms frozen, intermediate MLPs skipped) from every earlier neuron to its
  pre-activation at its strongest spot, top-20 images. Top-2 positive and top-1 negative inputs are shown;
- downstream: readers = later neurons it drives most, weighted by where each reader is on, on its top-20 images. Top-2;
- edit tests on its top-20 images: max response after edit / before (median) for left-right flip, 90-degree rotation,
  grayscale, blur, 4x4 patch shuffle (keeps texture, breaks shape).
Then I write each description without nouns BEFORE looking at the sealed-set crops, and judge the family links.
S2 run (`results/vision-s2-cards.json`, panels `vision2/figs/s2/*.png`, 1 min compute). Reading of the 16 key anchors:
| anchor | my description (no nouns where possible) | parent I see (rank among direct inputs) | children seen |
| 1 n351 | thin vertical line on light ground; - oblique hatching; blur kills | 0 n245 vertical edges (#1) | - |
| 3 n623 | muted cool blue-gray; grayscale kills | 0 n346 / 0 n364 color opponents (#1/#2) | - |
| 4 n499 | warm neutral textures where blue is absent; - saturated blue | 1 n365 blue, via its OFF side (#1) | - |
| 5 n135 | parallel curved bands? | unclear (shares .017) | - |
| 7 n604 | unclear | unclear | - |
| 8 n4 | horizontal edges/bands; rot90 .05, blur kills | unclear | 11 n676 / n745 stacked horizontal lines |
| 10 n1316 | object rim against bright ground? | unclear | - |
| 11 n1031 | long silky fur; - wiry/curly dog fur (both poles dogs) | 9 n1226 pale fur / 9 n80 dog fur (#1/#2) | 16 n630 dog scenes |
| 12 n136 | unclear | unclear | 16 n2246 clustered round bumps |
| 14 n1326 | unclear | unclear | - |
| 14 n33 | warm brown fuzzy; gray .54 | unclear | 17 n819 baked dough (pretzel) |
| 14 n831 | tawny/olive color; grayscale kills | 0 n346 purple-green opponent (#1) | 15 n2323 artichoke/cardoon |
| 15 n16 | upright cylindrical vessel; rot90 .15 | unclear | 16 n1328 bulbous vessels (teapot, barrel) |
| 16 n384 | half-tracks/tanks; rot90 -.01 | 15 n1162 boxy heavy machines (#1) <- 14 n103 vehicles+machines (#2) | - |
| 16 n26 | carousel / festive ornate scenes | 15 n2722 carousel (#1, near twin) | 17 n2795 carousel (twin) |
| 16 n2062 | handheld gadgets; - input 15 n3005 cameras (sibling) | 15 n2065 round mech parts (#1) / 15 n2584 twin (#2) | - |
| 17 n2283 | cars; - dogs | 16 n2886 cars (#1) | - |
Link types seen: refinement of the same pattern; color dial -> region/object of that color; near-twin relay across
blocks; sibling competitor as the strongest NEGATIVE input (cameras vs phones = essay's taxi vs ambulance); child on
a parent's OFF side. Edit fingerprints mark families: grayscale kills color members, blur kills edge members, rot90
kills upright-world late members. Direct-input shares are tiny past block 5 (.004-.02): no single dominant parent.
BIAS: I only displayed the top-2 inputs, so the "parent I see" is always one of them. Next: blind lineup.
## S3. Blind lineup (rules before the run)
For each of the 16 key anchors, three candidates in shuffled order, block numbers hidden: A = top direct input,
B = M3 parent (narrowest informative containment parent, either pole/variant), C = random active neuron from A's
block. I judge which candidate (or none) is a coarser version containing the anchor's pattern, then unblind.
A rule wins if it is picked clearly more often than C.
S3 result (`results/vision-s3-picks.json` saved BEFORE unblinding, key `results/vision-s3-lineup-key.json`).
17 key anchors (16 random + car). My pick of the coarser containing parent, by source:
A top direct input 7 | B territory parent 4 (B existed for 11) | C random 1 | none 5.
- Early blocks (1-4) and block 8, 11, 14 n33, 17: A. Late blocks 16 (x3) and 14 n831: B. In 16 n384 and 16 n26 the A
  candidate was a near-twin (same breadth); B was the broader family (all vehicles; outdoor scenes). In 16 n2062, B
  (handheld electronics) is the family but it was the strongest NEGATIVE input in S2 (sibling-like).
- Random won once (15 n16: lotion bottle / lipstick looked like "upright cylinders"): my eye can find a pattern in noise.
- No pick for 5 anchors (5 n135, 7 n604, 10 n1316, 12 n136, 14 n1326): I could not describe the anchor itself.
- Caveat: partly unblind, I had seen the A candidates in S2.
Reading: connection finds the parent in early blocks; territory finds the broader parent in the last stage, where
the top input is often a twin. A general rule needs both. The main failure is not the rule: 5/17 anchors (mostly
middle blocks) are unreadable with 6 crops.

## S4. Deep study of the 17 key neurons (Shonu: goal = accurate map; do not stop until deeply understood)
State file: `MAP-STATE.md`. Memory consolidated (62 -> 17 files). Deep viewer `vision2/view.py` (activation bands
100/75/50/25/10%, negative poles, heat maps, 13-edit fingerprint); 135 deep views (anchors + candidates) in
`map/deep/`. Four study agents (one per band) follow `map/STUDY-BRIEF.md`: prediction before test, synthetic and
edit tests, parent + link type, children, confidence. Cards in `map/cards/`. Held-out 4 not studied.
S4 result: 17 cards in `map/cards/` (agents' tests in `map/tests/`). All 17 now readable (5 were not). Verdicts:
yes 7 (5 n135, 8 n4, 14 n1326, 14 n831, 16 n384, 16 n26, 16 n2062), partly 9, no 1 (7 n604, a frame-position code).
Big corrections of first looks: 15 n16 = thin upright arch over a gap (not vessels); 16 n384 = road wheels inside a
track; 16 n26 = ribbed peaked canopy edge; 16 n2062 = dials/ports on small devices; 14 n33 = small red-orange detailed
part; 11 n1031 = hanging fur at a body's left rim; 4 n499 = blue-above/yellow-below boundary.
Map grammar seen: refinement chains (16 n384: machines .18 > wheels .034 > heavy boxy .016 > treads .0034 breadth);
near-twin relays across blocks; complex-cell pooling (child broader in ONE axis, e.g. polarity, same region); sibling
competitors as negative inputs (end-stopping, cameras vs phones); part inside whole via inhibition; non-content
"where" axes (frame position, rims); color dials feeding things of that color. Essay car+dog shape neurons exist
(15 n72, 16 n2117) but do not gate 17 n2283.
## S5. Rules vs the answer key (`results/vision-s5-rules.json`), rank of the hand parent among all earlier neurons:
C*log(lift) median 3, top-5 9/16, top-20 14/16 | |edge| median 8, top-5 7/16 | containment alone median ~5000.
Bias: the key used both containment scans and top inputs. Fair test: held-out 4, blind lineup with tests (S6).
## S6. Held-out blind lineup (rules before the run)
For 2 n194, 9 n851, 12 n122, 15 n1613: candidates = rule top-3 (C*log(lift)) + |edge| top-3 + 3 random active
neurons from the same blocks as the rule's picks; shuffled, labels hidden. A study agent (same brief, tests allowed)
names the coarser container or none, blind. Pass: the rule's top-3 holds the agent's pick for >= 3 of 4, and random
is picked at most once.
## S7. Whole-network map v1 (`map/network_map.json`, `vision2/build_map.py`)
Parent of every neuron (blocks 1-17) by C*log(lift); "confident" if above the 95th pct of null readouts' best score
(null from `m3_null.npz`, 10 min). Confident: whole network .407 (null .05 by construction). Per block .18-.60.
Parent mostly 1-3 blocks back (block 12: 6). Confident parents are 1.5-7x broader than the child; twins (ratio < 1.5)
10-47% of confident links. PROBLEM: the null bar is very high in the last stage (block 16: 2.27, block 17: 1.92)
and the hand-verified parents there score 1.4-2.0, so they would count as "not confident". The score ranks the true
parent near the top (S5) but cannot certify a parent against chance when there are ~20k candidates.
## S6 result (held-out blind lineup). PASS.
Judge's first pick came from the rule's top-3 for 4/4 (rule rank 1 for 3, rank 2 for 1); all 12 picks: rule 10,
edge 2, random 0. Held-out cards: 2 n194 line ends / corners (medium), 9 n851 beside steep figure outlines (medium),
12 n122 "nothing here" background vs object, child on the parent's off side (low-medium), 15 n1613 band of far
terrain in an upright outdoor scene (medium). 3 of these 4 are "not confident" under the S7 null bar: the bar is
too strict, the ranking is good.
Sample so far: 21 neurons (16 random + car + 4 held-out random). Deep study found a parent for 20 (7 n604, a frame-
position code, has only a position relay). Strictly finer: 7 of 17 key + 3 of 4 held-out.
## S8. Second random sample, 20 neurons, blind lineup protocol (rules before the run)
Seed 1, 5 per band (early 1-5, middle 6-11, late-middle 12-14, last 15-17), active neurons, none studied before.
Per neuron: understand it (prediction before test), then a blind lineup of 9 candidates (rule top-3, |edge| top-3,
3 random from the rule picks' blocks), first-3 picks or none. Estimates: (a) share of neurons with a parent;
(b) rule top-1 / top-3 agreement with the judge; (c) link-type mix.
## S8 result (second random sample, 20 neurons, blind). Picks unblinded (`results/vision-s8-key.json`).
Parent found for 19/20 (17 n966, a coin-purse-clasp detector, has only siblings among the candidates). Random never
picked (0 of 56 picks). First pick: rule top-3 13, |edge| top-3 6. Over both samples: 39/41 neurons (95%) have a
parent by deep blind study. Link mix (S8): refinement 8, near-twin relay 2, part -> whole 2, color dial -> thing 3,
off-side 1, position relay 2, none 1. Middle blocks: 4/5 have a frame-position component.
## S9. Choosing among the 6 candidates (`results/vision-s9-choose.json`)
Judged parent inside rule top-3 + |edge| top-3: 33/39. Exact first-pick match: rule 13/33, |edge| 9/33, rank-sum 16/33,
rule x log(1 + 1e4/edge rank) 16/33. Pick inside the judge's top-3 (S6+S8, n = 23): rule 19, rank-sum 19,
rule x edge 22. Chooser picked post hoc among 3 options: mildly optimistic. Several valid ancestors per neuron are
common (nested chains), so top-3 agreement is the fair metric.
## S10. Map v2 (`map/network_map_v2.json`, `vision2/build_map2.py`, 16 s)
Every neuron in blocks 1-17 gets one parent (rule x edge among 6 candidates). Tree: median chain length to block 0
grows with depth (block 1: 1, block 9: 4, block 14: 7, block 17: 9). 216 of 384 block-0 neurons are roots. Median
2 children per parent, max 90 (7 n828). Big families include 10 n1172 (whole animal body, 44 children).
## S11. Browsable atlas: https://claude.ai/artifact/T9KzsaBtmbiqbWRyDNyh4M (private)
41 studied neurons: tested distinction, judged parent with images, link type, verdict, confidence, automatic map
parent, chain to block 0. Built from map/cards + network_map_v2.json; images = compressed deep views.

# TREE FOLD WITH FAMILY PARENTS (Shonu, 2026-10-21: "make them a composable tree: a child disappears and a parent
# takes up some of the error")
## T1. Rules before the run
Fold: removed child c is replaced by a*h_p + b (LS fit of the child's hidden on the parent's hidden, same positions;
parent average-pooled to the child's resolution across stages). Implemented as weights only: the parent's write
column += a * (child write mapped to the parent's stage), the child's block bias += b * child write, child column = 0.
Across a downsample the child write is mapped back by least squares through the frozen downsample (pinv).
No other refit (one nested file). Composable: process removals deepest block first; a removed parent passes on
everything it absorbed to its own parent.
Score: KL(intact || edited) on 3,000 sealed images (3 per class); share = (KL cut - KL arm) / KL cut, 95% bootstrap.
Arms: cut (child -> its mean), fold_family (judged parent for studied neurons, map v2 parent otherwise),
fold_random (random parent from the same block with on-rate within 2x), and for single blocks the R7 dense_raw.
Experiments:
 A. The 39 studied children with a judged parent, removed all at once.
 B. One block at a time (14 and 16): remove 75% and 97% of neurons (lowest output energy, as R7), map v2 parents.
    Reference R7 block 14: dense_raw .31 (3% kept), attribution tree1 -.02.
 C. Composable: blocks 12-17 at once, 50% of each block removed, folded through the tree.
Pass (per experiment): fold_family share > 0 with the 95% interval above fold_random's mean.
T1 result A (`results/vision-t1-fold-A.json`) and diag (`vision2/t1_diag.py`): INVALID by a bug, plus a real finding.
- Bug: folding across a downsample (pinv back-map through the frozen downsample norm) blows the write up (|w| 4.7 ->
  1641 for 14 n831 <- 3 n92). That made fold_family KL 2.36 vs cut .0027. Cross-stage fold is not valid this way.
- Finding: same-stage folds recover ~nothing. R^2 of the child's hidden from its parent's (linear) is .002-.14
  (median ~.03). Per pair, KL fold = KL cut. A child is a split INSIDE the parent's region (both poles live there),
  so its average given the parent is near its overall average. The parent cannot carry a distinction it does not make.
  Fits the essay: a coarse model "ends early"; it lacks the fine split, it does not hold it elsewhere.
Fix: fold only into same-stage parents; cross-stage pairs are cut.
## T2. Does the tree give the removal ORDER? (rules before the run)
Blocks 14 and 16, remove 25/50/75% of neurons (cut to the mean, no refit), orders: (i) LEAVES first: fewest
descendants in map v2, ties by low energy; (ii) low energy first (R7 order); (iii) random; (iv) PARENTS first (most
descendants) as the reverse control. KL on the 3,000 sealed images. Pass: leaves-first < random at every size, and
parents-first is the worst. Also T1 B (block 16, same-stage folds) rerun with the fix, for the record.
T2 result (`results/vision-t2-order.json`), KL after cutting a share of a block (no refit), lower is better:
block 14: 25% leaves .016 | energy .017 | random .020 | parents-first .025; 50% .035 | .042 | .045 | .048; 75% .073 |
.068 | .070 | .072. Block 16: 25% .027 | .029 | .060 | .081; 50% .084 | .083 | .117 | .134; 75% .152 | .150 | .168 |
.171. ~55% of neurons are leaves. PARTIAL: leaves-first is best or tied up to 50% and parents-first worst, but at 75%
(parents must go too) all orders converge; leaves-first ~ low-energy except block 14 at 50% (-15%).
T1 B (fixed, block 16, `results/vision-t1-fold-B.json`): share of the cut's KL won back by folding into the map v2
parent: 50% removed .115 [.094, .136]; 75% removed .129 [.108, .149]. Random parent .001 / -.000. PASS.
Per child the gain is too small to see (R^2 ~.03), but summed over 1,500-2,300 children it is clean. Two numbers per
child (a, b): any subset of removals is a sum of per-child edits, so the dial nests by construction.
T1 C (fixed, blocks 12-17, 50% each, 6,912 neurons, composable): cut KL .730, family fold .707 (share .031 [.013,
.048]), random -.003. PASS by the rule, but small. Many block 15-17 parents sit in blocks 12-14 (across a downsample),
which now fall back to a cut.
## T3. Fold variants (rules before the run)
Block 16 at 50% and the T1 C setting. Arms: (a) map v2 parent (T1); (b) nearest same-stage ancestor when the map
parent is across a downsample (walk up the map chain); (c) FAMILY-k: LS of the child on its same-stage candidates
(map v1 rule top-3 + |edge| top-3, same stage only), each candidate's column += a_j * child write; (d) block 16
only: dense_raw ceiling (LS on all block-15 neurons, R7 style). Pass for (c): share above (a) with the 95% interval
clear of (a)'s mean.
T3 result (`results/vision-t3-variants.json`): block 16, 50% removed, share of cut KL won back: map parent .113
[.093, .133]; ancestor same-stage = same (block-16 map parents are already same-stage); FAMILY-k (<= 6 same-stage
candidates) .202 [.175, .232]; dense_raw ceiling .255 [.214, .296]. PASS: family-k reaches ~80% of the dense fold
with <= 6 numbers per child, nested by construction. Blocks 12-17 at 50% (composable): map parent .032, ancestor .033
(useless: an ancestor across a downsample is always in an earlier stage), family-k .046 [.028, .065]. Block 15 has
no same-stage earlier block, so its children cannot fold at all.
## T4. Sequential composable fold (rules before the run)
Blocks 12-17, 50% each. Process blocks shallow to deep. For each block: fit its removed children on the hidden values
of the CURRENTLY edited model (shallower edits applied), fold only into KEPT same-stage family members (removed
neurons are gone; no writes into their columns). Arms: T3 family-k (reference), family-k sequential. Pass: sequential
share above T3 family-k with the 95% interval clear of .046.
T4 result (`results/vision-t4-seq.json`): sequential .051 [.030, .073] vs T3 family-k .046. FAIL (interval includes .046). Diagnostic next: each block alone at 50% (cut vs family-k), to see where the multi-block damage sits.
T5 result (`results/vision-t5-perblock.json`), each block alone at 50%: cut KL / family-k share:
12 .014 / -.093 [-.136, -.050]; 13 .023 / .031; 14 .041 / .067; 15 .079 / 0 (no same-stage earlier block);
16 .083 / .202; 17 .135 / .045. Sum of single-block cut KL .375 vs .730 together (compounding ~2x).
Suspect: folds into far-back family members (block 12's family is in blocks 6-11) write into the stream early, and
every block in between reads the new write. Block 16 folds one block back and works best.
## T6. Adjacent-only family (rules before the run)
Same as T5 but the family is restricted to block b-1 (the 6 candidates filtered to block b-1; if none, the best
block-(b-1) neuron by |edge|). Blocks 12, 13, 14, 16, 17. Pass: block 12 share >= 0 and mean share over the 5 blocks
above T5's.
T6 result (`results/vision-t6-adjacent.json`), adjacent-only family, share: 12 .017 [.005, .029]; 13 .088; 14 .114;
16 .202; 17 .131 [.102, .162]. PASS (block 12 >= 0; mean .11 vs T5 .05). Folding far back disturbs the blocks in
between; families should sit in the block right before the child.
## T7. Composable, adjacent families (rules before the run)
Blocks 12-17 at 50% each, together. Sequential shallow to deep, fit on the edited model, fold only into KEPT block
b-1 family members (filtered; if none kept, top-3 kept block-(b-1) neurons by |edge|). Block 15 has no same-stage
block before it: cut. Pass: share above T4 (.051) with the interval clear of it.
T7 result (`results/vision-t7-compose.json`): blocks 12-17, 50% each (6,912 removed, 5,376 folded), cut KL .730,
fold .665, share .090 [.074, .106]. PASS (T4 .051). Block 15 (first block of stage 4) is still cut.
## Tree-fold summary (2026-10-21)
- A child's hidden is predicted poorly by its parent (R^2 ~.03): it is a split inside the parent's region. Per child
  the fold recovers ~nothing; summed over a block it is real.
- Best fold: each removed child -> LS on <= 6 family members in the block right before it (kept ones only), two to
  seven numbers per child, nested by construction. Block 16 alone: .20 of the cut's damage (dense ceiling .25).
  Six blocks together: .09. Random parents: 0.
- Folding far back hurts the blocks in between. Cross-downsample folds are invalid as built (pinv blow-up).
- Removal order: leaves first is cheapest up to 50% of a block.

# SMALL = PARENT + CHILD (Shonu, 2026-10-21): "small models with coarse neurons still attempt finer distinctions ...
# the small neuron has both; the large model has one and then the other. Find the equation small = parent + child."
## E1. One neuron for a parent-child pair (rules before the run)
Pairs: parent in block b-1, child in block b (same stage). 7 hand pairs (5 n135<-4 n212, 10 n1316<-9 n686,
11 n1031<-10 n1172, 14 n1326<-13 n78, 16 n384<-15 n1162, 16 n26<-15 n2142, 17 n2283<-16 n2886) + 200 random
block-16 children and 200 random block-13 children with their map v2 parent (when it is in block b-1).
Target at each position: the pair's joint write T = w_p h_p + w_c h_c. One replacement neuron s at the parent's
block writes w_s h_s. Score: R^2 of T on held-out positions (half the calibration images).
Forms:
 F1 parent only: w_p h_p + w_c * mean(h_c) (child cut).
 F2 linear: w_p h_p + w_c (a h_p + b) (T1 fold).
 F3 VOTE MERGE (reading B): h_s = GELU(z_p + lam * zc_hat), write nu w_p + mu w_c, plus bias; zc_hat = the child's
    pre-activation predicted linearly from the parent block's input (its "votes" in the parent's frame, LS fit).
    3 numbers per pair (lam, nu, mu) + bias.
 F0 FREE one neuron: read vector, bias and write vector all fitted (upper bound for one neuron).
Readings fixed now: F3 >> F2 and near F0 -> the merged-votes equation is the small-model form. F0 low -> one neuron
cannot hold both; the small model must spread the split over several neurons.
E1 result (`results/vision-e1-small.json`, `vision2/e1_fast.py`, 52 s; closed-form errors in the pair's 2-direction
write space, all positions, train/test split by image). Share of the child-cut error won back (median):
| pairs | linear fold | VOTE MERGE GELU(z_p + lam*zc_hat + c), write nu*w_p + mu*w_c | oracle (true z_c) |
| 7 hand | .061 | .292 | .558 |
| block 16 (100) | .018 | .036 | .160 |
| block 13 (100) | .052 | .154 | .271 |
| block 8 (100) | .047 | .086 | .335 |
Hand fits: lam 2-4, c ~1, nu .05-.4, mu .2-.56: the merged neuron reads parent + strong child votes and writes a mix
leaning to the child. 16 n384 <- 15 n1162: linear .004, merge .36, oracle .64. zc_hat (child pre-activation predicted
from the parent block's input) R^2 .6-.88; the gap to the oracle = evidence the child gets from block b-1's outputs.
Earlier slow attempts (Adam, subsampling) were invalid: optimizer drift and overfitting. Shonu: iterate fast.
## E2. Vote merge in the real weights (rules before the run)
Block 16, the 50% lowest-energy neurons removed (as T3). For each block-15 parent with removed map-v2 children, merge
its ONE highest-energy removed child: parent fc1 row += lam*u, bias += lam*u0 + c, parent write -> nu*w_p + mu*w_c
(+ constant to the block bias); other removed children use the T3 family fold. Arms: cut, family fold (T3), merge +
family. KL on 3,000 sealed images. Pass: merge + family above family fold, interval clear.
E2 result (`results/vision-e2-merge.json`): FAIL. merge+family .126 [.023, .190] vs family .202. The pair-local fit shrinks the parent's own write (nu .05-.4) and breaks its other readers. E2b: same, but the fit weighs the parent's write error by omega (1, 5, 20, 100); report KL per omega.
E2b result (`results/vision-e2b-merge.json`, 4 min): share won back, block 16 at 50%: family fold .202; merge (one
child per parent) + family for the rest: omega 1 .126, omega 5 .152, 20 .151, 100 .151. FAIL: the merge loses to
the family fold even with the parent protected. Local (pair) gains do not transfer: the parent serves many readers,
and the merged child loses its 6-member family fold. Block 16 was also the weakest block for the merge locally (.036).
Reading: "small = parent + child" holds as a one-neuron FORM (sum of votes, mixed write: GELU(z_p + lam z_c)) and
recovers 29% (56% with the child's true votes) of a clear family's error locally. In an untrained edit, changing the
parent's votes costs its other readers. A real small model retrains those readers around the coarse neuron.
## E3. Does the merge win when its readers adapt? (rules before the run)
E2b failed because the parent's other readers break. A real small model retrains its readers. Cheapest stand-in: after
the edit, ridge-refit the fc1 of the reading blocks (block 16 kept rows if block 15 changed, then block 17) so their
pre-activations on the calibration images match the intact model's. Closed form, sequential. Arms: cut, family fold,
merge (omega 5) + family, each with and without the reader refit. Block 16 at 50% (as E2). KL on 3,000 sealed images.
Pass: merge+refit above fold+refit, interval clear. cut+refit = how much the refit buys alone.
Readings fixed now: PASS -> the equation is two parts: GELU(z_p + lam z_c + c) AND readers that adapt. FAIL -> one
neuron cannot hold both even with adapted readers; a small model must spread the split. Note: the refit is per-size,
so a pass reopens the nesting question.
E3 result (`results/vision-e3-refit.json`, 3.5 min). Share of the cut's damage won back, block 16 at 50%:
| arm | no refit | readers refit |
| cut | 0 | .007 |
| family fold | .202 | .166 |
| merge (omega 5) + family | .152 | .137 |
FAIL. merge+refit - fold+refit = -.029 [-.043, -.015]. The refit buys almost nothing and even hurts the fold: a
reader refit to reproduce the intact pre-activations can only decode what the edited stream still holds. Readers
are not the bottleneck. The lost child information is. Note: a real small model's readers are trained for the task,
not to copy the big model, so this is not the same as training.

# A REAL SMALL MODEL AS THE SECOND REFERENCE (2026-10-21, Shonu away: "plug away at the composable tree")
Small models: timm convnext_atto.d2_in1k (dims 40-320, depths 2/2/6/2, top-1 .70 at our 224 crop) and
convnext_nano.d1h_in1k (80-640, 2/2/8/2, .76). Tiny .78. Same 56/28/14/7 grids, so a position in one model is the
same image patch in the other. Match neurons by correlation of hidden activation over positions (1,000 cal images).
## A1. Does a small model keep the tree's coarse neurons and drop its leaves? (rules before the run)
For each tiny neuron in stages 3-4: its best match r = max correlation with any small-model neuron in the same stage.
Tree height = number of map-v2 descendants. Predictions (dropdown reading: a small model is the collapsed tree):
 P1. Neurons with descendants match better than leaves, within block, at matched firing rate (partial Spearman of r
     on log(1+descendants) given block and rate > 0, bootstrap interval clear of 0). Atto and nano each.
 P2. The leaf gap (r of trunk minus r of leaves) is larger for atto than for nano (more collapsed).
 P3 "the small neuron has both": for a tiny child c whose parent p is in the same stage, take s* = p's best small
     match. Partial correlation of s* with h_c given h_p, versus a null child (random same-block tiny neuron at matched
     firing rate, not in p's subtree). Pass: median family > median null, interval clear.
Readings: P1 pass, P3 fail -> the small model ends early (keeps the parent, drops the child). P1 and P3 pass -> the
small model's coarse neuron carries the child's split too (small = parent + child). P1 fail -> the small model is not
a collapsed tiny tree at the neuron level.
Visual check after: crops of p, c, s* for the 3 strongest and 3 weakest P3 cases.
A1 result (`results/vision-a1-small.json`, 5 min). Best single-neuron match r of each tiny neuron in a small model:
| tiny neuron (map descendants) | atto stage 3 | atto stage 4 | nano stage 3 | nano stage 4 |
| leaf (0) | .24 | .26 | .30 | .33 |
| 1-2 | .30 | .31 | .35 | .38 |
| 3-10 | .34 | .38 | .39 | .43 |
| >10 | .41 | .48 (n 50) | .45 | .44 (n 50) |
P1 PASS: partial Spearman (given block and firing rate) .34 [.32, .35] stage 3 and .22 [.20, .24] stage 4, both
models. P2 weak: atto's leaf gap is about nano's (stage 3 .16 vs .16; stage 4 3-10 bin .12 vs .10).
P3 PASS but small: partial corr of the parent's small match with the child (given the parent) .052 vs null .009 (atto),
diff .028 [.025, .032]; nano .031. Stronger in stage 4 (.036/.041) than stage 3 (.017/.020). In atto 25% of
children have the same best small match as their parent (nano 19%); null rate not yet measured.
Caveats before reading it: matches are weak overall (r .24-.48), so information may be spread over several small
neurons. Descendant count may track activation energy. A1b checks both.
## A1b. Checks (rules before the run)
 1. P1 again with log energy (std of h x write norm) added to the controls. Pass: partial Spearman stays > .1.
 2. Null for "child shares its parent's best match": same rate for the A1 null children. Pass: family rate > 2x null.
 3. Spread decode: ridge R^2 of each tiny neuron from ALL small neurons in its stage (train/test split by image).
    Reading: leaves low R^2 -> the small model lacks the leaf split. Leaves high R^2 but low single r -> the small
    model holds the split spread over many neurons (it "attempts" the fine distinction without a neuron for it).
Picture check, hand pairs (`vision2/figs/a1/hand_*.png`, rows: tiny parent, tiny child, atto match of the parent,
that atto neuron on the child's spots and on parent-only spots):
- 10 n1172 (animal faces) -> 11 n1031 (fur texture without a face). Atto 7 n143 (r .54) = mammal faces incl. people.
  On the child's fur spots it is ~0% (one kuvasz 26%). The small model kept the parent and dropped the child.
- 13 n78 -> 14 n1326. Atto 6 n143 (r .39): 47-76% on parent-only spots, 26-45% on the child's spots. Partly "both".
- 16 n2886 -> 17 n2283. Atto 11 n1137 (r .30) = vehicles on a track; 38-65% on some child spots. Partly "both".
- 15 n1162 -> 16 n384: atto match r .14 (piano keys). Not a real match; only trust matches with r > .4.
## A2. Does collapsing the tree leaves-first make tiny behave like a real small model? (rules before the run)
Blocks 12-17, cut (mean replacement) 25/50/75%. Orders: TREE (map leaves first, then fewest descendants; ties by
lowest energy), ENERGY (lowest first), RANDOM. On 3,000 sealed images: top-1; agreement with atto's top-1; and on
the images tiny gets right and atto gets wrong, the fraction where the collapsed model gives atto's wrong class.
Compare orders at matched top-1 (interpolated). Pass: TREE gives more atto-like errors than ENERGY and RANDOM at the
same accuracy (bootstrap over images, interval clear). Reading: pass -> the dropdown collapse is how a real small
model differs from a big one.
## A3. Is a small-model neuron = its tiny twin + the twin's family? Which equation? (rules before the run)
Atto neurons in stages 3-4 whose best tiny match t has r > .4. Family F(t) = t's map parent + up to 4 map children
(same stage; strongest by correlation with t). Null N = the same number of tiny neurons from the same blocks, not in
t's family, matched on correlation with t. Fits on half the images, R^2 on the other half:
 S1  h_s ~ a h_t + c
 F-out  h_s ~ linear in the outputs h of {t} + F       N-out: same with N
 F-vote h_s ~ alpha * GELU(linear in the votes z of {t} + F) + d   (one neuron; Shonu's "has both")  N-vote: with N
Pass 1 (family): median R^2 gain over S1 of F-out > N-out, interval clear. Pass 2 (equation): F-vote > F-out.
Readings: 1 pass -> a small neuron is a collapsed family (small = parent + children). 2 pass -> the collapse is in the
votes (sum of votes through one GELU), not a sum of outputs.
A2 result (`results/vision-a2-attolike.json`, 5 min). Blocks 12-17 cut (mean replacement). 63% of these neurons are
map leaves. Tiny .784, atto .711 top-1; 331 sealed images where tiny is right and atto wrong.
| order | removed | top-1 | agrees with atto | gives atto's wrong class |
| tree (leaves first) | 25% | .738 | .773 | .178 |
| tree | 50% | .623 | .665 | .227 |
| tree | 75% | .482 | .537 | .254 |
| energy (lowest first) | 25 / 50 / 75% | .729 / .593 / .300 | .761 / .635 / .330 | .160 / .199 / .148 |
| random | 25 / 50 / 75% | .671 / .508 / .289 | .687 / .528 / .303 | .163 / .160 / .109 |
Atto-like errors: tree beats random clearly (tree 50% vs random 25%, higher accuracy and +.063 [.015, .115]).
Against energy order at about the same accuracy the gain is +.018 to +.027, intervals include 0. FAIL as written
(the tree and energy orders overlap a lot: leaves are low energy).
Side result, strong: collapsing by tree order (fewest descendants first) keeps far more accuracy at deep cuts:
75% of blocks 12-17 removed, no fold: tree .482, energy .300, random .289. The tree tells what can go.
## A2b. Is the tree order more than "few readers"? (rules before the run)
Same setup as A2 at 50% and 75%. Extra orders: OUT (lowest total |exact edge| to all later neurons first, from
e1_edges) and CHILDREN (fewest direct map children first, ties by energy). Pass: tree top-1 above OUT at 75%, by more
than .03. Reading: fail -> the map's tree order is mostly connection strength.
A2b result (`results/vision-a2b-orders.json`): Spearman(descendants, out-edge strength) .51. Top-1 at 50 / 75% removed:
tree .623 / .482; children .623 / .495; OUT .541 / .322. PASS: the tree order is not just "few readers".
## A2c. Is the tree order more than "fires rarely"? (rules before the run)
Orders: RATE (lowest firing rate first, M3 rates), RATE x ENERGY (product, lowest first). Pass: children-order top-1
above both at 75% by more than .03.
A2c result (`results/vision-a2c-orders.json`): top-1 at 50 / 75% removed: RATE .209 / .067; RATE x ENERGY .358 / .038
(children order .623 / .495). PASS by a wide margin. Removing rare-firing neurons first is the worst order. The
tree's "fewest children first" is not breadth in disguise.
Summary A2-A2c: collapsing blocks 12-17 leaves-first (by the map) keeps the most accuracy at every depth tested; at
75% removed (no fold, mean replacement) .48-.50 top-1 vs .30-.32 energy/out-edges, .29 random, .04-.07 rate orders.
## A4. Dropdown collapse: tree order + the child's error folded into its kept family (rules before the run)
Blocks 12-17, map-children order (fewest first, ties by energy), 50% and 75% removed. Arms: cut (mean) vs fold
(T7: each removed child -> LS on kept family members in block b-1, sequential shallow to deep; block 15 cut). Also
energy order + fold at the same sizes. Report top-1 and KL. Pass: tree+fold top-1 above tree+cut at 75% by > .02, and
tree+fold above energy+fold.
A1b/A1c/A1d results (`results/vision-a1b-checks.json`, `vision-a1c-decode.json`, `vision-a1d-p3.json`):
1. P1 with energy also controlled: stage 3 .22 / .21 (atto / nano) PASS; stage 4 .08 / .09 (below .1) FAIL. In stage 4
   descendants anti-correlate with energy (-.22).
2. Spread decode (ridge from ALL small neurons, penalty chosen on held-out images), median test R^2 by descendants:
   | | leaf | 1-2 | 3-10 | >10 |
   | atto stage 3 | .14 | .20 | .23 | .31 |
   | atto stage 4 | .00 | .01 | .06 | .25 |
   | nano stage 3 | .18 | .25 | .29 | .37 |
   | nano stage 4 | .02 | .02 | .03 | .12 |
   Last-stage leaves are not in the small models at all, not even spread over many neurons. Trunks are. (The first
   decode run overfit: negative R^2; discarded.)
3. "Child shares its parent's small match", with a null searched over the whole block and matched on correlation with
   the parent (family .22, null .17; perfect matching is impossible since children are unusually close to parents):
   atto 25% vs 10%, nano 19% vs 7%. Within bins of correlation with the parent the gap mostly closes (e.g. atto stage
   4, corr .3-.5: .41 vs .38; .2-.3: .19 vs .16; stage 3 .2-.3: .17 vs .09). Partial correlation (child beyond parent)
   .007 / .011 over null. Reading: "the small neuron has both" is WEAK. Most of it is the child's similarity to the
   parent. The strong result is the other one: a small model keeps the trunk and lacks the leaves.
Picture check (`vision2/figs/a1/leaves_vs_trunks.png`) CAUGHT A FLAW in A1c: leaf 17 n2432 (lorikeet/parrot colours)
has an atto twin with r .83 that shows the same parrots, yet its decode R^2 was -.66. Cause: 1 image per class, split
by image, so a class-specific leaf's classes sit only in train or only in test. A1c's decode is INVALID for sparse
class-specific neurons (most last-stage leaves). Trunk examples looked right (15 n2676 dogs <-> atto 10 n199 dogs r .68;
15 n562 white fluffy animals <-> atto 10 n295 r .64). Leaf 17 n2764 is a mixed bag (grey parrot, sulphur butterfly,
washbasin, ...) with no atto twin.
A1e: redo the decode with 4 images per class (ranks 0-1 train, 2 choose the penalty, 3 test), 25 positions each,
2,000 random tiny neurons per stage.
A1e result (`results/vision-a1e-decode.json`, 26 min under load; 2,000 random tiny neurons per stage, 4 images per class).
Median test R^2 of a tiny neuron rebuilt from ALL small-model neurons (n leaf / 1-2 / 3-10 / >10 = 1363/438/183/16
in stage 4; 1286/347/232/135 in stage 3):
| | leaf | 1-2 | 3-10 | >10 |
| atto stage 4 | .06 | .08 | .18 | .38 |
| nano stage 4 | .07 | .11 | .19 | .40 |
| atto stage 3 | .21 | .26 | .30 | .41 |
| nano stage 3 | .28 | .35 | .38 | .49 |
The trunk-kept / leaves-missing gradient holds on a valid split. Last-stage leaves are mostly absent from both small
models. Nano is not much fuller than atto in the last stage (P2 stays weak). A3 was stopped: same split flaw;
rerun as A3b with 4 images per class (ranks 0-1 train, 3 test).
A4 result (`results/vision-a4-dropdown.json`, 35 min under load). Blocks 12-17:
| order, removed | cut top-1 / KL | fold top-1 / KL |
| tree 50% | .623 / .727 | .638 / .635 |
| tree 75% | .496 / 1.73 | .458 / 1.53 |
| energy 50% | .593 / .763 | .604 / .709 |
| energy 75% | .300 / 2.83 | .136 / 3.91 |
FAIL (fold not above cut at 75%). The fold moves the output toward tiny's (KL down) but costs accuracy at deep cuts.
With most of a family gone, the fold falls back to edge neighbours and the small LS writes pile up. The order matters
far more than the fold: at 75%, tree cut .496 vs energy fold .136.
## A5. One dropdown slider for the whole network (rules before the run)
All 18 blocks, the same fraction per block (10-80%), cut with mean replacement. Orders within a block: TREE (fewest map
children first, ties by lowest energy) vs ENERGY vs RANDOM. Top-1 and KL on 3,000 sealed images. Pass: tree top-1 >=
energy at every fraction, and > by .05 at 50%+.
CONFOUND in A2-A2c and A4 (advisor, checked): the orders were global across blocks 12-17. Every block-17 neuron is a
map leaf (nothing after it to be a parent of), so the tree order removes ALL of block 17 first (3,072 of the 10,368
at 75%), then leaves of blocks 12-16. The A2 gap (.48 vs .30) is partly "which blocks lose neurons", not tree
knowledge. A5 (same fraction per block; inside block 17 tree = energy by construction) is the clean test. Do not
quote A2's gap without this caveat.
A5 result (`results/vision-a5-slider.json`, 7 min). All 18 blocks, same fraction per block, cut, top-1:
| removed | 10% | 25% | 40% | 50% | 60% | 75% |
| tree | .765 | .692 | .544 | .314 | .091 | .003 |
| energy | .762 | .699 | .555 | .387 | .199 | .007 |
| random | .722 | .566 | .224 | .045 | .003 | .004 |
FAIL. Whole-network, the tree order is no better than energy and worse at 50-60%.
A5b (`results/vision-a5b-late-equal.json`): blocks 12-17, same fraction per block: tree .637 vs energy .613 at 50%,
.374 vs .370 at 75%. Tie. With global allocation (A2) the tree order got .496 at 75%: that gain is ALLOCATION (remove
all of block 17, which is all leaves by construction), not tree knowledge inside a block.
CORRECTED READING of A2-A4: the map's tree does not give a better removal order than energy within a block. What
helped was cutting the last block wholesale before spreading cuts, which needs no map.
## A6. Does the tree mark what RETRAINING can do without? (rules before the run)
Reading so far: small models lack tiny's leaves (A1e), yet removing leaves from tiny is no cheaper than removing low-
energy neurons (A5b). A small model retrains its readers around the trunk; an edit cannot. Test with brief training:
blocks 12-17, 75% removed, same fraction per block, orders TREE / ENERGY / RANDOM (A5b), mean replacement. Then train
the kept weights of blocks 12-17 + head for 1 pass over the 20k analysis images (batch 50, Adam 1e-4, distill: KL to
intact tiny's logits; removed neurons stay zero). Top-1 on 3,000 sealed images before and after.
Pass: tree after-training top-1 above energy after-training by > .02 (bootstrap over images, interval clear).
Readings: pass -> the tree marks what a retrained small model can drop (small = trunk + retrained readers). Fail -> the
tree adds nothing even with retraining.
A6 first run (`results/vision-a6-retrain.json`, 18 min): tree 75% before .374 -> after 1 pass .637 (valid, it ran
first). BUG: energy and random arms reused the depthwise-conv and norm weights trained in the tree arm (only fc1/fc2/
head were reset), visible in their "before" (.467 vs .370 in A5b). Rerun as A6b with a fresh model per arm.
A6b result (`results/vision-a6b-retrain.json`, fresh model per arm, 12 min). Blocks 12-17, 75% removed (same fraction
per block), then 1 pass of distillation over the 20k analysis images:
| order | before | after |
| tree | .374 | .637 |
| energy | .370 | .648 |
| random | .286 | .636 |
FAIL. tree - energy after = -.011 [-.023, .001]; tree - random +.001 [-.014, .017]. One short pass of retraining
erases the difference between orders, even random. The tree does not mark what retraining can drop.
A3 was stopped twice (split flaw, then 27 GB memory: all pre-activations of both stages plus full-size copies;
swap hit 36 GB). A3c: one stage at a time, 3 images per class (ranks 0-1 train, 2 test), peak 9 GB, 10 min.
A3c result (`results/vision-a3c-rebuild.json`), 1,943 atto neurons with a tiny twin r > .4 (stage 3 1,049, stage 4
894). Median test R^2 of the atto neuron:
| | twin only | twin + family (outputs) | twin + null (outputs) | twin + family (one GELU of votes) | twin + null (votes) |
| stage 3 | .225 | .264 | .258 | .309 | .298 |
| stage 4 | .245 | .258 | .253 | .205 | .186 |
Pass 1 (family > null) is technically clear but tiny: +.004 R^2 [.003, .005]. An atto neuron is NOT "its twin + the
twin's family"; any neurons as similar to the twin help about as much. Pass 2 (votes > outputs): stage 3 +.031, stage 4
-.017. The stage-4 vote fit is under-optimized (twin's own votes .18 < twin output .24, which a correct fit cannot
give), so pass 2 is unresolved there.
Verdict on "small = parent + child" at the neuron level: no. A small model's neuron is its big-model twin (a trunk
neuron) plus a little of everything similar, not a merge of a parent with its children.
A3d (`results/vision-a3d-rebuild.json`, identity init: every vote fit starts at the twin-only fit; peak 9.2 GB, 8 min).
| | twin | +family out | +null out | twin votes | +family votes | +null votes |
| stage 3 | .225 | .264 | .258 | .253 | .308 | .298 |
| stage 4 | .245 | .258 | .253 | .228 | .235 | .225 |
Stage 3: one GELU over summed votes beats a sum of outputs by about the same for family (+.044) and null (+.040). The
FORM fits small neurons better; the FAMILY does not matter (+.003). Stage 4: test R^2 of vote fits stays below the
twin-only line even from an identity start. Likely cause: the vote fit uses an 8,000-row subsample that misses the
rare firings of class-specific neurons. Unresolved, no gain shown.
SESSION VERDICT (2026-10-21 evening): "small = parent + child" does not hold as a neuron equation, in tiny or in real
small models. Real small models hold the trunk and drop the leaves. The fine split a small model still makes comes from
retrained readers, not from a merged neuron. The dropdown picture is right about WHICH nodes a small model has and
wrong about the header absorbing its children.

# LEARN THE COARSE NEURON (Shonu, 2026-10-21 night): "a naive cut from the large model's children isn't enough to find
# whatever the coarse small neuron would've been." Stop guessing the formula. Let training find it, then look at it.
## L1. One free neuron replaces a parent and its children (rules before the run)
Families: 6 hand parents (15 n1162, 15 n2142, 16 n2886, 13 n78, 10 n1172, 9 n686) with their map children in the next
block (up to 5, by energy). Remove the children (mean replacement). Put ONE free neuron (read vector, bias, write
vector) in the parent's place, starting as a copy of the parent. Train only that neuron (~1.5k numbers) to match the
intact model's output (KL), 3 passes over 3,000 analysis images where the family is active (2,000) or random (1,000).
Everything else frozen. Evaluate on 1,000 held-out sealed images chosen the same way.
Arms: children cut, parent kept (= naive cut); parent + children cut; learned neuron.
Pass: (a) learned neuron wins back > 25% of the naive cut's KL; (b) partial corr of the learned neuron with the child
given the parent > .2 (it holds both). Pictures: parent, child, learned neuron, and the learned neuron on the child's
spots and on parent-only spots.
Readings: pass -> a coarse neuron that holds both exists given these readers; read its form off the weights.
(a) fail -> with the readers frozen, one neuron cannot carry the child; a small model needs its readers trained too.
L1 result (`results/vision-l1-learn.json`, 6 min; hand child added to each family since 15 n1162 has no map-v2 child
in block 16). Held-out sealed images where the family is active:
| parent (children) | KL naive cut | KL learned | won back [CI] | partial corr w/ children | read cos | write change ~ child write |
| 15 n1162 (1) | .00178 | .00063 | .65 [.03, 1.43] | .28 | .98 | .54 (null .20) |
| 15 n2142 (1) | .00101 | .00088 | .13 [.03, .26] | .38 | .99 | .56 (null .19) |
| 16 n2886 (2) | .00584 | .00228 | .61 [.42, .84] | .23 | .97 | .01, -.05 |
| 13 n78 (3) | .00044 | .00024 | .47 [.17, .84] | .46 | .98 | .19-.24 (null .51) |
| 10 n1172 (5 of 28) | .00054 | .00030 | .45 [.27, .70] | .06 | .98 | .05-.37 (null .12) |
| 9 n686 (5 of 7) | .00021 | .00014 | .32 [.20, .47] | .28 | .97 | .08-.41 (null .18) |
PASS on 4 of 6 families on both parts ((a) fails 15 n2142, (b) fails 10 n1172). The learned neuron stays close to the
parent (read change ~20% of its length) but now also follows the children and writes some of the child's direction.
Pictures (`vision2/figs/l1/sheet_*.png`): 13 n78: on all 8 of the child's top spots the learned neuron fires more than
the parent (45 vs 34, 59 vs 49, 72 vs 64 ...) and a little less on parent-only spots (60 vs 73). 15 n1162: parent =
tracked vehicles + pianos/bassoon (slat pattern); learned neuron's top crops are only tracked vehicles, and it fires
46-62% on the child's half-track spots where the parent fired 14-47%.
Second reference: correlation (held-out positions) of the best atto neuron with tiny's parent vs with the learned
neuron: .24 -> .32, .36 -> .41, .34 -> .42, .49 -> .53, .61 -> .63, .63 -> .68. 6 of 6 move TOWARD the real small model.
## L1b. Control: cut random non-family neurons instead (rules before the run)
Same as L1, but the cut neurons are random block-(b+1) neurons outside the parent's map subtree (same count, matched
mean activation within 2x). Pass for L1's reading: the atto-correlation gain and the partial corr are clearly smaller
than in L1 (median gain < half of L1's .045).
L1b result (`results/vision-l1b-control.json`, 7 min). Random non-family neurons cut (rate-matched), parent retrained:
won back -.03 to .07 (L1: .13-.65); partial corr with the cut neurons .00-.05 (L1: .06-.46); correlation with the
parent's atto twin (twin fixed as the parent's best match): median change -.005 (L1: +.046, 5 of 6 up).
PASS. A parent can absorb ITS OWN children, not arbitrary neurons, and absorbing them moves it toward the real small
model's neuron. This is the first direct support for "small = parent + child": the coarse neuron a small model would
have is the parent nudged by its children (read weights move ~20% of their length; the write gains part of the child's
write direction), found by training against the output, not by a local formula or a cut.
Why E1-E3 failed and L1 works: E1's local fit made big changes (nu .05-.4, lam 2-4) that broke the parent's other
readers. Training against the real output keeps the parent nearly intact and adds only what the readers can use.
Open: (1) a closed form for the nudge (no training); (2) many families at once (a whole block's children absorbed by
their parents); (3) more families, chosen at random, for the atto-gain statistic (n = 6 hand families now).

# CLOSED-FORM NUDGE (Shonu, 2026-10-21 night: "quick iteration to find the closed form nudge ... analysis on both the
# quantitatives and the images ... creativity and freedom"). Target: reproduce L1's trained neuron without training.
Harness `vision2/n_harness.py`: per family, cached block-input tensors (train 3,000 / the same 1,000 held-out images
as L1) in `hf/imagenet/l1cache/`, teacher logits, child cut, evalkl(neuron). Score = share of the naive cut's KL won
back on L1's held-out images (L1 trained: .65 .13 .61 .47 .45 .32).
## N0. What did training change? (no pass rule; description)
Learned minus parent: read change dz (as activations on train positions) regressed on [z_p, child votes as seen from
the parent's block (ridge from the parent's input), 1]; write change dw regressed on [w_p, w_c...].
N0 result (`vision2/n0_analyze.py`, 5 min). Learned minus parent:
- Read change dz is positive almost everywhere and largest where the CHILD is on (15 n1162: +.76 where child on /
  parent off, +.61 both on, +.15 both off; 13 n78: .40 / .28 vs .05). It is only partly in the span of [z_p, child
  votes seen from the parent's block]: R^2 .05 / .16 / .05 / .32 / .13 / .46. Child votes are well predicted from the
  parent's input (R^2 .62-.83; 10 n1172's children only .15-.25).
- Write change: |dw|/|w_p| .24-.66; only 16-50% lies in span(w_p, w_c); there it shrinks w_p a little (-.02 to -.06,
  16 n2886 -.49) and adds ~.02-.09 of each child's write.
Essay check (Shonu asked): "Keeping roughly the strongest quarter of positive activations in Inception's final layer
preserves 69.47% accuracy, versus 70.08% originally. ConvNeXt's late MLP branches are similarly tolerant of weak-
response removal." What is lost is "fine distinctions between things the network is already confused about ... the
faint activations that tip them come from the semantically related neurons you'd expect." Applied here: the nudge
should come from the CHILDREN (local), and only the child's STRONG firings should need absorbing.
## N1. Which part of the trained change matters? (rules before the run)
Arms (share of naive-cut KL won back, L1 held-out): learned; read change only (w = w_p); write change only (u = u_p);
learned projected onto the child span (read: span of [u_p, child vote vectors]; write: span of [w_p, w_c]).
Reading: if the child-span projection keeps most of the gain, a closed form needs only a few numbers per child.
N1 result (`results/vision-n1-parts.json`, 33 s with cache). Share won back:
| family | learned | write only | read only | write proj. on span(w_p,w_c) | read proj. + full write |
| 15 n1162 | .65 | .42 | .09 | .03 | .45 |
| 15 n2142 | .13 | .07 | .03 | .02 | .10 |
| 16 n2886 | .61 | .24 | -.19 | .00 | .48 |
| 13 n78 | .46 | .30 | .06 | .00 | .40 |
| 10 n1172 | .45 | .46 | -.01 | .00 | .47 |
| 9 n686 | .32 | .24 | .03 | .09 | .31 |
The WRITE change carries most of the gain, and the useful part is NOT along w_c (projection keeps ~nothing). Hypothesis:
the readers see the stream through LayerNorm, which ignores changes along each position's current stream direction,
so "write the child's direction" must be rotated into "what the readers can see", position by position.
## N2. Closed-form write nudge: weighted least squares (rules before the run)
Keep the parent's read. Solve dw (C numbers) + a bias vector (C) so that the parent's extra write dw*h_p + bias
replaces the children's lost write sum_c w_c (h_c - mean_c), with the error measured by a metric:
 M0 plain distance; M1 CHILD-LOCAL: change in the next block's neuron pre-activations (random-projection Jacobian of
 block cb+1's fc1 input, from block pb's output through block cb); M2 WHOLE NETWORK: Fisher of the output (sampled
 y ~ p, v = e_y - p; VJP to the parent's and children's write points).
One linear solve per family (rows = train images x samples). Score on L1's held-out images. Pass: M1 or M2 >= 80% of
L1's write-only share (.42 .07 .24 .30 .46 .24).
N2 result (`results/vision-n2-write.json`, 8.5 min; ridge chosen on held-out train rows, step on train KL). Share won
back on L1's held-out images (write only; parent's read unchanged):
| family | L1 write-only | M0 plain (= T1 linear fold + bias) | M1 next block, all neurons | M2 whole-network Fisher |
| 15 n1162 | .42 | .23 | .02 | .46 |
| 15 n2142 | .07 | .06 | .01 | .01 |
| 16 n2886 | .24 | .03 | .26 (head) | .39 |
| 13 n78 | .30 | .24 | .30 | .33 |
| 10 n1172 | .46 | .23 | .22 | .42 |
| 9 n686 | .24 | .21 | .14 | .18 |
PASS for M2 on 4/6 (>= 80% of trained). M1 weak: all next-block neurons weighted equally, most do not read the family.
(First N2 run had ridge too weak: M2 -30 on 15 n2142. Fixed.)
## N3. Family-local metric + read nudge (rules before the run)
M3 FAMILY-LOCAL: error measured only in the pre-activations of the family's own READERS: top-24 downstream neurons by
|exact edge| (e1_edges) from the parent and from each child, any later block (Shonu: the nudge comes from the children,
not whole-network effects). Plus a READ nudge: z_s = z_p + lam * sum_c zc_hat_c (child votes seen from the parent's
input, scaled to z_p's spread); lam in {0, .1, .2, .4} chosen on TRAIN KL; the write is re-solved for each lam.
Pass: M3 (best lam) >= 80% of L1 FULL learned (.65 .13 .61 .46 .45 .32) on 4/6 families.
N3 result (`results/vision-n3-family.json`, 5 min; first run left out readers inside block cb and the head; fixed).
Family-local metric (top-24 readers of the parent and of each child by exact edge), read nudge lam chosen on train:
| family | lam 0 | .1 | .2 | .4 | chosen (on train) | L1 learned |
| 15 n1162 | .08 | .10 | .13 | .18 | .10 | .65 |
| 15 n2142 | .02 | .05 | .05 | -.06 | .02 | .13 |
| 16 n2886 | .05 | .12 | .22 | .17 | .22 | .61 |
| 13 n78 | .32 | .35 | .38 | .40 | .38 | .46 |
| 10 n1172 | .17 | .15 | .13 | .09 | .15 | .45 |
| 9 n686 | .21 | .24 | .25 | .26 | .26 | .32 |
FAIL (2/6 at >= 80%: 13 n78, 9 n686). The read nudge toward the children's votes helps in 4/6. The family-local ruler
is worse than the whole-network ruler (N2 M2) for the write.
## N4. Whole-network ruler for the write + read nudge (rules before the run)
As N3 but the error ruler is the output Fisher (N2 M2), lam in {0, .1, .2, .4} chosen on train. Pass: >= 80% of L1
full learned on 4/6.
N4 result (`results/vision-n4-fisher-read.json`, 6 min). Whole-network write ruler + read nudge (lam on train):
| family | lam 0 | .1 | .2 | .4 | chosen | L1 learned | % of learned |
| 15 n1162 | .48 | .49 | .51 | .55 | .51 | .65 | 78% |
| 15 n2142 | .02 | .05 | .04 | -.07 | .02 | .13 | 15% |
| 16 n2886 | .37 | .43 | .50 | .54 | .54 | .61 | 89% |
| 13 n78 | .33 | .36 | .38 | .37 | .36 | .46 | 79% |
| 10 n1172 | .28 | .26 | .38 | .31 | .38 | .45 | 85% |
| 9 n686 | .17 | .19 | .19 | .19 | .19 | .32 | 60% |
Strict pass 2/6 (>= 80%), 4/6 at >= 78%. One linear solve (C numbers) + a 4-way choice reaches most of training.
15 n2142 resists every closed form (its trained gain is also small and noisy, CI .03-.26).
Pictures (`vision2/figs/l1/closed_*.png`; rows: parent, child, trained, closed form; the three on the child's top
spots; bottom: held-out images the cut hurt most, with intact / cut / closed form / trained answers):
- 16 n2886 read side: on the child's top spots the closed form fires 20 22 60 60 23 11%, trained 20 21 62 50 19 5%,
  parent -1 2 57 48 2 -1%. The closed form lands where training landed. Write side: racer .32 -> cut "mobile home" ->
  closed form racer .34 (KL .36 -> .03); moving van restored (.49 -> cut .66 -> .48). Not restored: sports car ->
  convertible (both fixes stay on convertible), limousine -> jeep. Worst-8 KL: cut .216, closed .068, trained .059.
- 15 n1162: closed form is halfway between parent and trained on the child's spots (11 34 6 .. 42 56 vs trained 25 46
  16 .. 51 62) and keeps the bassoon crop that training dropped. Half-track .79 -> cut amphibian .43 -> closed form
  half-track .59 (trained .60). Worst-8 KL: cut .214, closed .099, trained .070.
- 13 n78: worst-8 KL cut .027, closed .011, trained .006.
Reading: the closed-form nudge = (read) add a little of the children's votes, as seen from the parent's input, and
(write) solve the parent's write so the readers get the children's lost signal, measured by how much the output cares
at each position. The TARGET is purely the children. Only the RULER (what the readers can see) is network-wide, and it
can be computed once per block and shared by every family in it.
## N5. Why does the closed form not EXACTLY match training? (Shonu: "be a little unfair"; rules before the run)
N4 simplifies three things: (1) one linearization, (2) read change restricted to lam * (child votes), (3) a proxy
target (children's lost write) instead of the true output error. N5 removes all three: Gauss-Newton on ALL of the
neuron's numbers (read u, bias b0, write w; exactly L1's parameters), true gradient J^T (p - p*), curvature J^T F J
from sampled Fisher directions, Levenberg damping and step chosen on held-out TRAIN images. Report after 1, 2, 3 GN
steps vs L1 trained. Readings: GN-1 ~ trained -> the N4 gap was the proxy/read restriction. GN-1 < GN-3 ~ trained ->
the gap is linearization; a few closed-form steps suffice. GN-3 < trained -> something non-quadratic matters; dissect.
N5 first result (`results/vision-n5-gn.json`, ~10 min): GN on (u, b0, w), share after 1/2/3 steps vs trained:
15 n1162 .37/.41/.45 vs .65; 15 n2142 .05/.05/.05 vs .13; 16 n2886 -12.6/-3.5/-1.3 (unstable) vs .61; 13 n78
.28/.41/.43 vs .46; 10 n1172 .33/.44/.48 vs .45; 9 n686 .22/.28/.31 vs .32. Iterating closes the gap (linearization
is part of it). Damping always hit the top of the grid (badly conditioned read side).
FLAW FOUND (unfair to the closed form): the harness picked its train images from 8,000 random analysis images; L1
picked the family's top 2,000 from all 20,000. On harness images the cut's KL is ~10x smaller, so the solve fits mostly
noise. Fix (N6): harness v2 with L1's selection, and the trained reference retrained INSIDE the harness on the same
images.
## N6. Fair comparison, 24 families (rules before the run)
Families: the 6 hand families + 18 random parents with >= 1 map child in the next block (9 in blocks 9-13, 9 in 15-16),
up to 5 children each. Same train images (top 2,000 of all 20k analysis by family activity + 1,000 random) and the
same held-out images (top 500 + 500 random of sealed ranks 20-29) for every arm. Arms: TRAINED (L1 recipe, 3 epochs
Adam 3e-4); GN-k (k = 1..5, damping {.01..100}, step {.25, .5, 1} chosen on held-out train images); W|trained-u (GN
on the write only, read fixed at trained) and U|trained-w (read only, write fixed at trained) to locate the gap.
Gap measure: (trained - GN5) / trained, per family; median and worst.
## N7. Fast-iteration set (Shonu: loops of minutes, not an hour; the N6 sweep was stopped before any family finished)
Cache `hf/imagenet/n7cache/` (12 families: 6 hand + 3 random parents in blocks 9-13 + 3 in 15-16; train 1,500 =
top 1,000 of all 20k analysis by family activity + 500 random; held-out 500 = top 250 + 250 random of sealed ranks
20-29; trained reference = L1 recipe, 4 epochs). `vision2/n7_lib.py`: load, run, share, gn, train. 16 s per family.
Loop 1 (`n7_loop1.py`, 5 min, 3 families). Held-out share; trained on the 2/3 GN split for fairness:
| family | trained 4 ep | trained 12 ep | GN 4 steps | train-val KL x1e4: cut / tr4 / tr12 / GN4 |
| 15 n1162 | .655 | .708 | .700 | 6.53 / 3.81 / 5.35 / 3.19 |
| 16 n2886 | .682 | .726 | .707 | 97.3 / 39.3 / 32.2 / 26.1 |
| 13 n78 | .469 | .602 | .599 | 3.53 / 2.27 / 2.15 / 2.06 |
THE GAP CLOSES on a fair setup: 4 Gauss-Newton steps on the full neuron = 12 epochs of training on held-out images,
and lower train-val KL than any training run. N5's gap came from the unfair train images (10x weaker family signal)
plus too few, over-damped steps. Training is not a ceiling either (12 ep > 4 ep on held-out).
Loop 2 (`n7_loop2.py`, 6 min, 2 Fisher samples, 3 GN steps; trained4 used all 1,500 train images incl. the GN
validation third, a small edge for training). Held-out share:
| family | trained4 | GN step 1 / 2 / 3 |
| 9 n686 | .35 | .24 / .29 / .35 |
| 10 n1172 | .11 | .07 / .09 / .15 |
| 11 n409 | .23 | .18 / .25 / .31 |
| 12 n768 | .37 | .27 / .29 / .48 |
| 13 n78 | .53 | .33 / .46 / .58 |
| 13 n131 | .00 | .10 / -.21 / .01 (nothing to recover; step 2 overfit the 500-image check) |
| 15 n1162 | .65 | .57 / .60 / .63 |
| 15 n1773 | .34 | .15 / .31 / .34 |
| 15 n2142 | .16 | .06 / .10 / .12 |
| 15 n2454 | .64 | .59 / .61 / .61 |
| 16 n2886 | .70 | .47 / .63 / .68 |
| 16 n76 | .54 | .40 / .50 / .52 |
Stage 3: GN-3 >= trained on all 6. Stage 4: GN-3 is .02-.04 below and still rising. Single step: 40-90% of trained.
Guard to add: accept a step only if the held-out-train KL drops by a clear margin (13 n131).
## N8. Compounding: all 12 families collapsed at once (`vision2/n8_run.py`; 4,510 held-out images = union of the
families' held-out sets). Mean KL vs intact (share of the cut's damage won back):
| arm | mean KL | won back |
| cut (all children of all 12 families) | .00531 | 0 |
| CF independent (each family's GN-3 solved alone, applied together) | .00302 | .43 |
| TR independent (each family's trained neuron, applied together) | .00289 | .46 |
| CF sequential (shallow -> deep, GN-2 refine in the edited model) | .00309 | .42 |
| TR sequential (shallow -> deep, 4 epochs from the trained init in the edited model) | .00289 | .46 |
Families barely interact: TR seq = TR indep. Independent fixes compose (good for "liquid": solve families in
parallel). CF seq slightly worse than CF indep: the shallow families got worse (9 n686 .00168 -> .00183, 12 n768 .00197
-> .00226) and deep ones better. Reading: a shallow parent solved while deeper families are cut but not yet repaired
learns to cover damage that is repaired later. Order deep -> shallow, or just use independent solutions.
Remaining CF-vs-TR gap (.00013 = 2.4% of the cut) sits mostly in stage-4 families, where GN-3 had not converged.
Loop 3 (`n7_loop3.py`, 2.5 min): stage-4 GN with S=4, up to 8 steps, step kept only if held-out-train KL drops > 1%:
15 n1162 .70 (trained .65); 15 n1773 .44 (.34); 15 n2142 .13 (.16, guard stopped at 2); 15 n2454 .57 (.64, guard
stopped at 2; unguarded 3 steps gave .61); 16 n2886 .72 (.70); 16 n76 .54 (.54). Guard too strict; use keep-best.
N8 v2 (`results/vision-n8-indep_v2.json`): all 12 collapsed at once, CF independent .00288 vs TR independent .00289.
THE COMPOUNDING GAP IS CLOSED at 12 families (CF better on 6 families, worse on 4).
## N9. A shared (K-FAC) closed form so a whole block can be solved in one pass (rules before the run)
Full GN needs a (2C+1)^2 matrix per parent; a block has hundreds. K-FAC: write curvature for parent j ~ E[h_j^2] x G,
G = E[g g^T] (C x C, shared by all parents of the block); read curvature ~ E[(g.gamma w_j)^2 gelu'(z_j)^2] x E[a a^T]
(shared input covariance). Gradients for all parents come from the same backward pass. Test per family on the cache:
K-FAC 3-4 steps vs full GN. Pass: K-FAC within .03 of full GN on 10/12.
N9 result: K-FAC FAILS on 12/12 (no step improves held-out-train KL). Diagnosis (`n9_debug.py`, 16 n2886): gradients
match exactly, but K-FAC curvature is ~6x too small (trace ratio full/K-FAC 6.4 write, 5.9 read): within one image
the output gradients at different positions point the same way, and K-FAC treats positions as independent. Its
steps overshoot (|dw| 3-10x the full GN step, cos .13 at low damping).
Fix without approximation: a GRADIENT BANK. One backward pass stores the output gradient at every position of every
image (per Fisher sample). For parent j, its exact GN rows are phi_w = sum_pos h_j,pos * gamma g_pos and
phi_u = sum_pos (gamma g_pos . w_j) gelu'(z_j,pos) a_pos, i.e. the shared bank reweighted by the parent's own
activations. Exactly full GN, but one backward pass serves every parent of a block.
## N10. Whole block with the gradient bank (rules before the run)
Block 16, 50% of neurons removed (lowest energy, as T3; family fold won back .202). Each removed child whose map
parent is in block 15 is cut; its parent gets a bank-GN solve (all such parents at once, one global damping and step
chosen on held-out train images). Other removed children: plain cut. Reference: TRAINED = the same parents fine-tuned
jointly with Adam on the same images. Eval: KL on the 3,000 sealed images (T-series set). Debug first on 100 parents
(~3 min), then all. Pass: bank-GN >= trained - .02 share, and > family fold .202.
N10 debug (100 parents of block 16's removed children; eval KL cut .0826 = T-series value):
- Per-parent block-diagonal GN: no improving step. Every parent sees the same output error and tries to fix all of it
  alone; together they overshoot. -> Joint GN over all parents, matrix-free conjugate gradients (products computed
  from the bank, the big matrix never formed).
- Joint GN with 2 random Fisher samples: needs heavy damping (lm 100, step .3); 1 iteration .057. Gradient checked by
  finite differences (correct; surface steep). Fisher with 2 samples x 1,000 images has rank <= 2,000 for 153k numbers.
- Exact Fisher on each image's top-6 classes (v = sqrt(p_k)(e_k - p)): 4 iterations .046 / .064 / .095 / .112 vs
  TRAINED (Adam, same images, same parameters) .105 (4 epochs) / .107 (8 epochs). Joint GN >= training on 100 parents.
Full block run (all 1,073 parents), damping {100, 1000}, step {.3, .6}, 4 iterations, trained 8 epochs: ~12 min.
N10 full block (`results/vision-n10-block-0.json`, 5 min): 769 block-15 parents of 1,073 removed block-16 children
(1,536 removed in all). Held-out share of the cut's KL (.0826) won back:
| arm | share |
| joint GN it 1 / 2 / 3 / 4 | .108 / .162 / .193 / .209 |
| trained (Adam 3e-4, same params, same 1,000 images) 4 / 8 epochs | .081 / .122 |
| family fold T3 (other params: up to 6 family members per child) | .202 |
Joint GN beats training of the same weights by a wide margin and passes the family fold. Fairness check next:
training may be under-converged at lr 3e-4 x 8 epochs for 769 neurons -> 30 epochs at lr 1e-3.
Fairness (training of the same 769 parents): lr 1e-3 diverges (share -.62 at 4 epochs, -.42 at 30). lr 3e-4: 4 / 8 /
16 / 30 epochs = .085 / .126 / .160 / .189, still rising. Joint GN 4 iterations .209, also still rising. Running both to
convergence (GN 8 iterations; training 80 epochs) to compare end points and watch for overfitting.
N10 convergence (`results/vision-n10-block-0_gn8.json`, `..._tr80.json`). Block 16 at 50%, 769 parents:
| joint GN iterations 1-8 | .108 .162 .193 .209 .216 .223 .224 .226 (7.9 min) |
| training epochs 4 / 8 / 16 / 30 / 50 / 80 | .085 .126 .160 .189 .207 .216 (3.6 min) |
Same end region (~.22); the closed form ends at or slightly above training. No held-out overfitting seen in either
(eval share rises monotonically). Wall-clock: training is faster here; the result is equivalence, not speed.
## N11. Compounding across blocks (rules before the run)
Child blocks 17 then 16 (deep -> shallow), 50% removed each (lowest energy). For each, the kept map parents in the
block before (16, then 15) are solved jointly: GN (4 iterations, top-4 Fisher, 1,200 random analysis images, intact
teacher) vs TR (Adam 3e-4, 30 epochs, same images and parameters). Eval KL on 3,000 sealed images. Pass: GN >= TR - .02.
N11 result, child blocks 17 + 16 at 50% (cut KL .285; `results/vision-n11-*-17_16-*.json`). Share won back:
| arm | after block-16 parents (440) | after block-15 parents (769) |
| GN, 4 iterations per block (3 min) | .144 | .229 |
| TR, Adam 30 epochs per block (2 min) | .214 | .187 |
TR got WORSE on the second block (.214 -> .187): 1.2M free numbers, 800 images, no held-out stop -> overfits. GN picks
every step on held-out train images, so it cannot drift. GN was under-converged on block 16 (4 iterations; N10 needed
~8). Fair rerun: TR with early stopping on the held-out train third (keep best epoch); GN with 8 iterations per block.
N11 fair rerun (`n11b`): TR with early stopping on the held-out train third: block-16 parents best epoch 29/30 (.214);
block-15 parents best epoch 0 (every epoch made held-out-train KL worse) -> final .214. GN 8 iterations: stopped
early ("no improving step") at the same points as 4 iterations -> .144 / .229.
Wider grid (damping 10..1e4, step .1..6): child block 17 alone (773 parents): GN .238 (6 iterations) vs TR .217.
Two blocks: GN .144 after block-16 parents (stuck at iteration 3), .234 final; TR .214 / .214.
Remaining gap: the block-16 stage in the two-block case (GN .144 vs TR .214). Suspect the damping: joint GN damps each
group with ONE scale; Adam normalizes per weight; the per-family GN that matched training damped by diag(A).
Next: exact per-weight diagonal damping from the bank.
Per-weight diagonal damping (DIAG=1, damping .1..100, step .3..1, 6 iterations per block): two blocks 17 + 16:
after block-16 parents .194 (was stuck at .144; TR .214; still rising at iteration 6), final .238 (TR .214).
The closed form beats early-stopped training on the two-block collapse.
Pictures (`vision2/figs/l1/n11_two_blocks.png`, 16 held-out images the two-block cut hurt most): closed form (F, DIAG
GN) and trained (L, early-stopped) give the SAME top answer on 15/16; they fix the same images (mountain bike ->
bicycle-built-for-two) and fail on the same (buckeye stays "snail", vase stays "hourglass": fine distinctions lost
with the children). Top-1 on 3,000 sealed: intact .784, cut .715, F .733, L .730; agreement with intact .844 / .862 /
.861. Worst-16 KL: cut 4.01, F 3.04, L 2.96. The closed form lands on functionally the same model as training.
## N12. Three blocks (17, 16, 15 at 50% each; parents in 16, 15, 14 -- the last across a downsample). GN DIAG 6
iterations per block vs TR early-stopped 30 epochs. Pass: GN >= TR - .02.
N12 result (`results/vision-n11-*-17_16_15-*.json`; cut KL .482). Share won back:
| stage | GN DIAG (8.2 min) | TR early-stopped (3.6 min) |
| after block-16 parents | .193 (stalled at iteration 5: no improving step) | .220 |
| after block-15 parents | .233 | .239 |
| after block-14 parents (across the downsample) | .238 | .247 |
PASS (GN >= TR - .02), gap .009. The gap comes from the stalled first stage again. Hypotheses: top-4 Fisher misses
curvature when the heavily cut model's output is spread (KL .48); smallest step (.3) too large near the optimum.
Stalled stage (block-16 parents, three blocks cut), GN DIAG, STOP after the first stage:
| setting | iterations | share | TR |
| steps >= .3 (N12) | stalled at 5 | .193 | .220 |
| steps .1-1, damping .1-1000, top-4 | 8, still improving | .212 | .220 |
| top-12 Fisher, steps >= .3 | stalled at 5 | .202 | .220 |
The stall was steps too coarse near the optimum. Every gap so far closed with enough well-sized GN steps: an
optimization-budget issue, not a structural one. Final confirmation: three blocks, steps .1-1, top-6, 10 iterations.
N12b final (steps .1-1, damping .1-1000, top-6, up to 10 iterations, 11.5 min): stages .205 / .236 / .236 vs TR
.220 / .239 / .247. The block-14 stage (parents across the downsample, stage-3 grid) added nothing (stalled at 3) and
the first stage stays ~.01 below training. Three blocks: closed form .236-.238 vs training .247 = a .01 gap (4%
relative) that is NOT closed yet. Runs vary ~+-.01 (first stage .212 vs .205 under similar settings).
Open: (1) why GN stalls on the block-14 parents (stage-3 grid, 196 positions, C = 384) while Adam keeps descending;
(2) the first-stage .01.
Stall isolation: child block 15 ALONE (339 parents in block 14, across the downsample; cut KL .0791): GN converges
(held-out-train KL .06213 -> .06210 -> no improving step) at .128; TR early-stopped (best epoch 14 of 30) .132.
Gap .004. The cross-downsample block is fine alone; the three-block stall is a compounding effect (by the third stage
the remaining error is small next to the noise of 400 check images). GN on stage-3 banks is slow (~110 s/iteration).
Block-scale pictures (`vision2/figs/l1/n11_block_parents.png`; block-16 parents solved jointly for block 17's removed
children, 773 parents): median change is small (|du|/|u| .017, |dw|/|w| .022). The 3 most-changed parents:
n461 moves toward its removed child (on the child's top spots: African grey 67 -> 78%, lynx 74 -> 87%); n2248 barely
(jigsaw 1 -> 10%); n100 not at all on its child's spots (~0%) while its own top crops reshuffle (golf ball stays first;
king crab, toucan enter). Reading: in a JOINT block solve the parents act partly as a shared pool that repairs the
total error. The clean "parent nudged toward its own children" picture holds per family (L1), not neuron by neuron
at block scale.
Training seed check (SEED=12, other train images and order): three blocks TR stages .235 / .235 (best epoch 0) / .245
vs seed 11 .220 / .239 / .247. Final TR .245-.247 is stable; GN final .236-.238. The ~.01 three-block gap is REAL,
not run noise. Stage paths differ a lot between seeds (stage 1 .220 vs .235), so per-stage comparisons are noisy.

# D-SERIES: DISCOVER WHAT A COARSE NEURON IS, THEN DERIVE IT (Shonu, 2026-10-22: "investigate the circle to derive pi";
# no fitting, no knobs on images; training is an approximation of the true equation, not a ceiling). Plan:
# ~/.claude/plans/i-want-you-to-floating-gray.md. Study set: 6 hand families; the 6 random n7 families stay untouched.
## D1. Look at the object (no conjectures). Tools: vision2/d1_look.py (per-position family + solved + trained + atto/nano
## twins, cache hf/imagenet/d1/), d1_plane.py, d1_scatter.py, d1_span.py, d1_compose.py, d1_sheet.py, d1_bond.py, d1_gate.py.
D1a family plane (figs/d/scatter_*.png). Lean toward the child = slope(child)/slope(parent), raw units, on positions where
either is on (description only):
| family/child | atto twin of parent | nano twin | solved GN | trained L1 |
| 16_2886 c0 | .19 | .13 | .03 | -.07 |
| 15_2142 c0 | 1.31 | .59 | .07 | .09 |
| 13_78 c0 / c2 | .55 / .71 | .38 / .64 | .14 / .08 | .12 / .06 |
| 9_686 c2 | .59 | 1.14 | .24 | .19 |
| 10_1172 c3 | .34 | .16 | -.02 | .01 |
| 15_1162 c0 | -.29 (twin corr only .19) | -.29 | .17 | .16 |
The real small model's neuron leans ~5x more toward the child than the solved/trained parent. Matched check (same parent
level): its value rises with the child at every parent stratum (+.5 to +1.6 std), roughly additive.
CAVEAT: the atto twin shares only ~4-43% of its variance with the tiny parent (D1b), so the 'lean' compares different
objects. It is NOT a target for a parent nudge.
D1b how much of the small neuron is its tiny family (linear R^2, half images held back; description):
| family | parent | family | top-10 tiny by corr | top-50 |
| 16_2886 atto | .10 | .12 | .41 | .46 |
| 15_2142 atto | .12 | .19 | .42 | .47 |
| 13_78 atto | .26 | .32 | .37 | .44 |
| 9_686 atto | .43 | .52 | .62 | .66 |
| 10_1172 atto | .36 | .38 | .60 | .62 |
| 15_1162 atto | .04 | .04 | .29 | .32 |
The solved parent is ~.85-.96 the parent. The REAL coarse neuron is mostly NOT the map family.
D1c composition (d1_compose.py, whole stage searched): atto 10/320's top-50 tiny parts: 45 "other" in map v2, 4 siblings,
1 self. Parts sit in blocks 15, 16, 17 (stage 4 all blocks); atto 6/143's parts in blocks 6, 7, 8, 9, 12, 13, 14.
Pictures (figs/d/sheet_atto_10_320.png): atto 10/320 = side/rear of road vehicles (beach wagon, cab, pickup, sports car,
forklift). Its tiny parts are specialists: 16/2558 minivan interiors and vans, 16/2429 beach wagons (near-twin, atto
82-96% on its spots), 17/164 mixed vehicles + cabinets, 15/72 dogs + one jeep. The tiny parent 16/2886 corr .22.
Reading: the coarse neuron is a UNION of several fine specialists that share one pattern, across siblings AND across
depth (the essay's "one curves & loops neuron instead of ten curve detectors"). Map v2 (one depth parent per neuron) does
not group them. atto 6/143 (figs/d/sheet_atto_6_143.png): slanted edges/folds, parts from 5 blocks, same picture.
D1d weight bonds among the top-20 parts (d1_bond.py, d1_gate.py): read |cos| .079 / .069 vs random .049 / .042; write
|cos| .111 / .084 vs .051 / .083; no dominant shared direction (top singular share .10-.14). Shared upstream feeder:
max parts fed by one neuron 16 / 13 / 16 vs null 95th pct 15 / 11 / 15. Bonds in raw weight space are WEAK.
D1e checks (vision2/d1_checks.py; 1,000 analysis images, 1 per class, 30 random positions each; NOT family-selected).
On generic images the small neuron is only loosely matched by ANY model: R^2 from top-10 parts, tiny vs nano as the
fine side: atto 10/320 .13 vs .14; atto 10/311 .21 vs .09; atto 6/143 .39 vs .41. Tiny's parts are at the cross-model
ceiling, and the ceiling is low (two separately trained models share only a loose neuron basis). D1b's higher numbers came
from family-selected images. Adding tiny's previous stage (pooled) does not add the "general vehicle" gate (top part
14/430 for 10/320, R^2 .17 alone; rest same stage).
Form on the top-10 tiny parts (z-scored, uniform weights, sign from corr; correlation with the small neuron):
| small neuron | sum of GELUs | max | best part | GELU(sum of z-votes - theta), theta 0/4/8/12/16 | coarse z when 1 part on / >=3 on |
| atto 10/320 | .49 | .41 | .34 | .08 .19 .32 (theta 0/4/8; rising) | -.37 / +1.84 |
| atto 10/311 | .51 | .43 | .30 | .29 .43 .54 .55 .47 | -.06 / +2.97 |
| atto 6/143 | .64 | .51 | .38 | .57 .63 .64 .58 .46 | +.18 / +2.64 |
Class-vote bond (stage-4 parts, vote = fc x head-norm weight x write): |cos| .093 vs random .042 (10/320; shared vote =
minibus, beach wagon, minivan, jeep, police van, cab, pickup); .057 vs .041 (10/311; golfcart, Model T, jinrikisha,
oxcart: old/odd vehicles).
READING (new): a coarse neuron is a CONSENSUS detector over a group of fine neurons that share their downstream vote. It
fires when several agree (>=3 on: +1.8 to +3.0 std) and stays near zero when one specialist fires alone. It is not an OR.
One GELU of the summed, scaled votes with a high threshold expresses it (.54-.64), as well as the sum of GELUs. Earlier
merges (E1-E3, ESSAY-IDEAS "one half-space wall") tried to make one neuron fire whenever ANY member fires; the real small
model does not do that. Open: what fixes the threshold, and how the group is found from tiny's weights alone.
D1f many neurons (vision2/d1_consensus.py; 40 random alive atto neurons per stage; 1,000 images x 25 positions; top-10
tiny parts; results hf/imagenet/d1/consensus_atto_s*.json). Medians, all / best-part corr > .3:
| atto stage (0-based) | sum of GELUs | max | best part | consensus GELU(S - theta*) | 1 part on | >=3 on | theta*/sd(S) |
| 3 (last) | .50/.57 | .37/.41 | .34/.42 | .47/.57 | -.11/-.14 | +2.02/+2.05 | 1.96/2.50 |
| 2 | .55/.59 | .43/.47 | .33/.40 | .55/.61 | +.26/+.22 | +2.31/+2.49 | .76/.76 |
Consensus form fails (theta*=0, corr < .2) on 8/40 (stage 3) and 6/40 (stage 2). Rate law (where it works): the coarse
neuron's firing rate at theta* vs the parts' mean firing rate: stage 3 median ratio .88 (IQR .50-1.40), 30 neurons;
stage 2 median 1.43 (IQR 1.24-1.60), 34 neurons; across neurons log-rate corr .34 / .79. Candidate constant: a coarse
neuron fires about as often as its parts (late) or ~1.4x as often (mid, broader, more dial-like: lone part +.26).
D1g co-firing in TINY alone (vision2/d1_cofire.py): mean pairwise corr among the 10 parts .170 (stage 3) / .185 (stage 2)
vs random same-block .003 / .003. Median rank of parts 2-10 among part 1's co-firing neighbors (of 23,040): 20 / 28.
-> The group can be found from tiny alone by co-firing; the small model is not needed to find it.
## D2. Toy where the truth is known (vision2/d2_toy.py; 8 coarse x 8 fine classes, each pattern shown with a random
## sign so a linear net fails; 1 hidden GELU layer; wide 256 vs narrow 16, 3 seeds each, same data/recipe; nets in hf/toy/)
Accuracy fine/coarse: wide .948/.996; narrow .69/.95 (coarse by force); linear nets .09-.16 / .24-.37.
D2 look (d2_look.py): narrow neuron vs its top-10 wide parts (median of 16): sum .90, max .76, best part .75, consensus
GELU(S - theta*) .91 (theta* ~ 0), firing rate / parts' mean rate 1.03, parts' pairwise co-firing .42, co-firing rank 9
of 256. Lone part on: +1.0 std (toy is softer than vision's last stage). Seed ceiling: best-match corr narrow s0 ->
narrow s1 .63, wide -> wide .64. Same object as in vision: consensus sum of a co-firing group, rate preserved.
D2 build (d2_build.py, d2_diag.py, d2_joint.py, d2_fisher.py, d2_lda.py): make a 16-neuron net from the wide net with no
training. Read = sum of members' z-scored votes; threshold = rate law; write = ? Scores (seed 0; seeds 1-2 alike):
| write rule | reads | fine | coarse | KL to wide |
| trained from scratch | trained | .694 | .955 | .865 |
| per-group projection | co-firing groups | .16 | .71 | 3.85 |   (groups overlap -> outputs counted many times)
| joint logit projection (MSE) | trained reads | .24 | .93 | 2.39 |  (MSE treats all 64 logits equally)
| Fisher (2nd-order KL at the wide model) | trained reads | .09 | .32 | 6.15 |  (wide is near one-hot -> F ~ 0; coarse is FAR from fine)
| Bayes / LDA with wide's soft labels | trained reads | .612 | .950 | 1.11 |
| Bayes / LDA | oracle part lists (trained neurons' top-10) | .41 | .94 | 1.73 |
| Bayes / LDA | co-firing groups (ward / average) | .43 / .42 | .93 / .93 | 1.64 / 1.67 |
| Bayes / LDA | random groups | .26 | .56 | 2.64 |
Diag: with oracle groups each built neuron correlates .98 with its group's sum and .91 with the trained narrow neuron, yet
the write rule decided everything. Reading: the coarse net cannot sit near the fine net in output space (it cannot make
the fine split), so the write must be the BAYES answer (expected wide probabilities given what the coarse neurons see),
not a local expansion around the fine model. Fully derived (co-firing groups + consensus read + rate-law threshold +
Bayes write) reaches coarse .93 vs trained .95 with zero training; fine .43 vs .69. Gaps: write (Gaussian assumption,
~.08 with trained reads) and read (~.19, even with oracle part lists).
D2 read checks (d2_reads.py, d2_pc.py, d2_rate.py, d2_scale.py; library d2_lib.py), seed 0 unless noted:
- Ideal pure-coarse reads (one per coarse class and sign) + Bayes write: coarse .997 (better than trained .955) but fine
  .30, KL 1.86. The trained narrow net gives up coarse accuracy to keep fine information (its reads tilt toward fine
  classes: fine-template cos .65; covers 12/16 coarse patterns). Consensus reads are "too coarse" (coarse cos .71, 15/16).
  This is "the small neuron has both" in the toy.
- Member weights: group PC1 vs equal: .435 vs .429 fine (seeds 0-2 alike). Structure-free control, global top-16 PCs of
  the wide votes: coarse .34-.42, fine .13-.16. NEURON GROUPS are what make the derivation work.
- Threshold: trained narrow neurons fire on 47.9% of inputs, their parts 48.5% (rate law observed). Built net vs rate
  multiplier x.25/.5/1/2: fine .17/.29/.43/.09 -> peak exactly at the rate law (rate law confirmed functionally).
- Scale of the summed vote: x.25-x2 all within .01 (no effect in this regime). Trained vote std 11.9 vs sum of z-votes 10.3.
- Vote decomposition of trained reads into wide votes is not identifiable here (64-dim input < 256 neurons). Dropped.
Toy summary: derived pieces that hold: group = co-firing; read = sum of members' z-scored votes; threshold = rate law
(fire as often as the members); write = Bayes (expected wide probabilities given the coarse activations; LDA form).
Fully derived net: coarse .93 / fine .43 / KL 1.64 vs trained .95 / .69 / .87. Open: the "tilt" that lets trained
coarse neurons also carry fine information (~.19 fine), and the Gaussian-assumption write gap (~.08).
Note before the ConvNeXt test: the toy's "Bayes write, not a local expansion" holds at EXTREME compression (16 of 256),
where the coarse net sits far from the fine one. At 50% of one ConvNeXt block (cut KL .08) the local output-Fisher ruler
already worked (N2 M2). Both write rules are derived (conditional expectation under the readers' view); the test runs both.
Ideal pure-coarse reads in the toy beat training on the COARSE task (.997 vs .955) while training still wins on fine and
KL: a first, narrow instance of a derived net beating training.
## D3. ConvNeXt, width merge inside one block (vision2/d3_merge.py, d3_width.py, d3_diag.py)
Width or depth (d3_width.py, 600 images): block-16 neurons' best co-firing partner: in block 16 median .270, in block
15 .269, in block 17 .344 (share > .5: 5% / 9% / 24%). 79% have their best partner in another block; within-block
partners with corr > .3: median 0. Redundancy runs along DEPTH (relay twins), not width.
Block 16, 3072 -> 1536 (rules fixed before the run: ward co-firing groups, consensus read, rate law, joint stream
projection write; readings on CAL; sealed KL on the N10 3,000 images). Cut lowest-energy half .0826 (reproduces N10).
Consensus merge .1222 (48% MORE damage than the cut). FAIL.
Diag: merged groups' members co-fire only .19 (toy .42); h_G tracks the members' sum at .79; the write keeps .58 of the
block's output vs the cut .82 (readers' view .57 vs .76; the unread top direction holds 6.7% and is not the issue).
Reading: inside a ConvNeXt block there is little redundancy; forcing pairs merges different detectors. The theory's groups
(co-firing, consensus) live across blocks. Next: depth merge (remove a block, merge its neurons into the next block).
## D4. Depth merge (vision2/d4_transplant.py, d4_ce.py, d4_depth.py)
- Read transplant: block B's reads computed through block B+1's dwconv+LN: vote corr -.08 (B=16), -.09 (B=15); per-
  channel dwconv kernels of adjacent blocks are unrelated (cos -.09 / .005). The spatial filter is part of the read.
- Conditional expectation instead: block B's votes from block B+1's view of the SAME stream, held-out R^2: B=16 .907,
  B=15 .949, B=12 .964. A block's votes survive the removal of its own filter.
- Depth merge, blocks 16+17 -> one block of 3,072 (block 17's dwconv+LN); readings on 5,000 analysis images (5 per class),
  streamed; sealed KL on the N11 3,000 images. Two bugs fixed first: (1) LN outputs have a fixed channel sum -> exact solve
  gave huge cancelling coefficients that float32 broke (KL 5.3) -> min-norm solve; (2) 1,000 reading images -> write kept
  .77 of output on readings but .56 on new images -> 5,000 images.
| arm (3,072 neurons in blocks 16-17) | sealed KL |
| N11: cut 50% of each block | .285 |
| N11: GN / early-stopped training on parents | ~.217 / ~.224 |
| derived depth merge, co-firing groups (ward) | .206 |
| derived depth merge, top-energy neurons as singletons | .162 |
| drop block 16, block 17 unchanged | .104 |
| drop block 16, block 17's reads, write re-derived (joint stream projection) | .1225 |
HONEST READING: the derived merge beats GN/training only against the N11 removal pattern. Simply dropping block 16 is
better than everything (.104). The N11 benchmark (50% of each block) was a weak structure: "end one block early" costs
half as much as thinning two blocks and repairing with training. The stream-MSE write HURTS a clean drop (.104 ->
.1225): the stream metric is the wrong ruler (as in the toy, where logit MSE failed). Co-firing groups lose to singletons
here (pairs barely co-fire inside ConvNeXt). No claim of "derived beats training" stands for ConvNeXt yet.
Derived repairs of the clean drop of block 16 (no output loss): + block 16's mean output as a constant bias: .1249 (worse
than .104; a constant shift of the stream changes LayerNorm's scale for every reader, the essay's unread "size" axis);
+ block 17's reads restored to their intact votes by conditional expectation (R^2 .83): .3809 (much worse: projected
estimates pull activations toward the mean; the network handles the plain drop better on its own).
Open for D5: (1) the training reference AT THE RIGHT STRUCTURE (drop block 16, then train block 17 early-stopped) -- the
fair "beat training" target; (2) which block to drop (15/16/17, stage-3 blocks); (3) a derived repair that respects the
readers' view and the LayerNorm scale instead of stream MSE.
## D5. Strong training reference at the right structure, then map what training changed (Shonu: "we need more powerful
## references ... map the approximation itself ... if the slope is 2.96, maybe it's 3"; "interpret every number")
Cache: stream entering block 16 for 20k analysis + 3k sealed images (vision2/d5_cache.py, hf/imagenet/d5cache/x16.npz).
Training (vision2/d5_train.py): drop block 16, train block 17 to the intact logits (KL), init = original block 17; train
ranks 0-17, early stop on ranks 18-19, score sealed. Plain drop .1042.
| params, lr, 20 passes | sealed KL |
| all, 1e-3 / 3e-4 | never below start (overshoot in pass 0) -> .1042 |
| all, 1e-4 | .0942 (best; still creeping down) |
| fc1+fc2 only, 1e-4 | .0948 |
| all, 3e-5 | .0952 |
Strong reference: .0942 = 10% of the drop's damage won back.
Map (d5_map.py, d5_changed.py, d5_level.py, d5_readrank.py, d5_agree.py, d5_parent.py, d5_self.py):
- Training does NOT rebuild the lost parent votes: slope of vote change on lost vote .002 (corr .009); change std .30 vs
  lost std 1.08. Write side: slope .08, corr .24 on the lost output.
- It restores each neuron's average OUTPUT level: across 3,072 neurons, slope (trained - dropped) on (intact - dropped):
  mean activation .646 (corr .81), firing rate .41 (corr .73), mean vote -.05. Example 689: 2.43 intact -> 1.37 dropped ->
  2.58 trained; 2137: .09 -> .16 -> .09.
- Through small read turns, not thresholds: read cos .995, bias change ~+-.01, write cos .994. The change is spread out
  (top-10 directions 6.5%), ~10% of the weight norm, and REPRODUCIBLE across runs (read change cos .975 / .916, write
  .984 / .911; same image order in all runs, so partly inflated). The spatial filter moved 6-9%, LN/gamma <1%.
- Not explained by: the parents' activations (best block-16 match corr .31; it is a top-5 wiring parent 57/300 = 100x
  chance, but 5 wiring parents give R^2 .009), the neuron's own vote (R^2 .006), the LN size ratio (.022), all together .037.
- Fact: without block 16, block 17's input is ~27% smaller, so LayerNorm reads it ~1.36-1.41x louder.
Derived one-number fixes (no output loss), sealed KL vs plain drop .1042 / training .0942:
| level law by bias (match mean activation) | 1.62 (needs median |shift| 1.5; training moved biases ~.01) |
| rate law by bias | .116 |
| block mean-write fix | .1045 |
| size restore along least-read direction (reaches the head LN too) | 1.88 (x.5: .348) |
| same along block 16's mean-write direction | .579 (x.5: .253) |
| vote gain k on block 17 (1/ratio = .708 / .9 / 1.1) | .730 / .163 / .158 |
The plain drop is a local optimum for every one-number fix tried. Training's 10% comes from a spread-out, reproducible
change whose meaning is not read yet. Its clearest interpretable trace: neurons' average output levels move 65% back.
D5 pooled view (advisor: block 17 is the last block; the head averages over the 49 positions, so it only sees each
neuron's per-image POOLED activation). d5_pooled.py, strong run: slope (trained - dropped) on (intact - dropped):
pooled activation per image x neuron .097 (corr .23); per-neuron median slope .104 (corr .33); pooled block output .084
(corr .27). Reading: training takes a ~10% step toward giving each neuron its intact per-image pooled output (matching its
10% KL gain); the global average level moves faster (.65).
Pooled restore, slope 1 (d5_pooledfix.py; 300 most important neurons; one linearized min-norm solve per neuron, target =
intact pooled activations, no output loss): read change |du|/|u| 1.31 (training: ~.1) -> sealed .1860 (half step .1107),
worse than the plain drop. The intact pooled pattern cannot be reached by a small read turn; forcing it breaks the
neuron. Stopping the run of one-off fixes here (6 tried); next is reading, not fixing.
Reproducibility with a DIFFERENT image order (SEED=7; same init and data): sealed .0934 (seed 0: .0942). Change seed 0 vs
seed 7: read cos .926 (per neuron .930), write cos .945 (.948), same size. The trained change is a determined object
(signal), so mapping it is meaningful. Strong reference: .093-.094.
Pictures (vision2/figs/d/d5_changed.png; intact / dropped / trained, % of max, validation images): n689 (wolves, huskies,
malamutes) LOSES its territory when block 16 is dropped (timber wolf 43/6/32, green snake 13/4/30) and training restores
it; n2831 starts firing on dogs it ignored (Border collie 0/25/6, dhole 0/18/0) and training silences it; n2756 starts on
birds/monkeys (kite 0/34/18, squirrel monkey 0/18/3). READING: block 16 supplied GATES in front of block 17 (essay: the
positive side lets a neuron into its territory, the negative side keeps others out). Dropping it opens wrong gates and
closes right ones; training rebuilds the gates ~10%. D6 target: derive the gate each block-17 neuron lost from its
block-16 gate neurons (which ones, by wiring sign), as an interpretable object, before any new fix.
## D5b. Is the 10% reference real? (Shonu: "suspicious it only recovers 10%"). vision2/d5b_train.py, d5_cache2.py, d5_acc.py
Instrumented (block 17 only, 18k images): train KL .113 -> .027 while val .110 -> .104: OVERFITTING (4.7M weights, 18k
images). Feature loss (match the intact pre-head stream): .0958 (6 passes). LoRA r64 (0.49M): best pass 0, .1015 (overfits
even sooner -> block 17 alone lacks generalizable information). Blocks 12-15+17 on 18k: worse (train .067, val .126).
MORE DATA (ranks 23-49 + mirrored ranks 0-17/23-49 = 90k images; sealed scoring set ranks 20-22 untouched) + feature loss,
block 17, 4 passes: sealed .0860 (17.5% won back), still improving. The reference was data-limited.
Accuracy on the sealed 3k: intact 78.40%; drop block 16 76.17% (agree 89.7%); best reference 76.73% (agree 91.1%).
Literature (R0d-block-removal-literature.md; PRACTISE 2303.00972 read in full): block dropping + feature mimicking of
ALL remaining layers recovers most accuracy with 50-1000 images (ResNet-34 16% latency cut: 66.5 -> 70.3 at 50 images,
68.6 -> 72.5 at 500; original 73.3); "finetune convexity hypothesis". Expectation: much more is recoverable than 10-17%.
## D6. The self-healing law (vision2/d6_*.py)
Exact split for a block-17 vote when block 16 is removed: z_I = c + r (z_D - c) + G, r = sigma_D/sigma_I (block 17 LN).
- d6_decomp: r median .737. On each neuron's intact ON spots (300 most affected neurons): gate from block 16 G +5.59,
  extra native vote from LN amplification (1-r)(z_D-c) +5.13 -> LN replaces 89% of the lost gate; net lost vote -.63. G >
  0 on ON spots for 100% of neurons. WRONG spots (fire only after the drop): G +2.78 vs native +5.45 (over-amplified).
- d6_prop (the "2.96 -> 3"): lost vote = [(1-r)(z_I-c) - G]/r; zero iff G = (1-r)(z_I-c). Slope of G on (1-r)(z_I-c):
  .932 all spots, .911 ON spots, .790 near threshold; variance explained .41 / .63 / .32. Equivalent: u.d / u.s =
  sigma_I/sigma_D - 1 for every reader: a late block raises each reader's evidence by the same fraction as it raises the
  stream size. Pure proportionality would make the block invisible to LayerNorm. Its real content = the NOVELTY (the
  non-proportional part); removal damage = novelty / r.
- One-number fixes explained: restoring the stream size (mean output, size direction, gain 1/r) removes the healing and
  leaves the gates missing; that is why every such fix was worse than the plain drop.
- d6_bussgang: block-16 neurons' Bussgang slope E[GELU'(z)] median .000 (p90 .003); 96% of the output is nonlinear;
  linearized block .220 (keep 768/1536 real: .159/.117). Late blocks cannot be coarsened by linearizing (detectors).
- d6_novel / d6_ratio / d6_heal: per-neuron cos(write, stream) median .07 (proportionality is collective, not per
  neuron); keep-k orders and a global gain rho all worse than the full drop for k < 1536.
- d6_partial: keeping 96 high-energy neurons returns 10% of the gate but halves the healing (r .735 -> .861).
- d6_roles: VOLUME (share of the size block 16 adds) vs GATE (share of readers' input): top-96 by gate hold 37.6% of gate
  and 1.8% of volume; 59% of gate is negative (inhibition). Pictures (figs/d/d6_volume.png, d6_openers.png,
  d6_inhibitors.png): volume neurons (16/100, 1315, 2043, 2335) fire on salient man-made objects (golf ball, panpipe, gas
  mask, steel drum, traffic light, flagpole) with DOGS on their negative pole; inhibitors are specific (snakes/echidna
  16/2984 closes 17/1365 sports balls; lamps/matches 16/191).
- d6_byclass: amplification ~1.41 in every image group (uniform, not image-specific); damage: man-made .126 vs dogs .074,
  birds .057. Prediction "dogs get no healing" FAILED.
- d6_damaged (figs/d/d6_damaged.png): 10.3% of top answers change; 94% go to a class in the intact top-5 (neighbour
  confusions). Worst images = objects out of context recognized from local shape (Windsor tie hanging on a wall .58 -> .02,
  saltshakers close-up, ice lollies in kids' mouths, camp stove; coat hangers -> "whippet"). Training does not fix them.
  The 10% most damaged images hold 47% of the KL.
- d6_gatekeep: keep top-|gate| neurons: still worse than the full drop; ANY 32 neurons push r .735 -> .86 -> confound found.
- d6_bias: block 16's constant output bias (|13.6| centered vs stream 30.7; cos .84 with the block's mean output) is the
  confound. Full drop .1042 (r .736); full drop + bias only .2051 (r .865). Keep-k with bias / without: 32: .193/.125;
  96: .193/.137; 384: .177/.169; 768: .148/.170; 1536: .080/.130; 2304: .026/.082. Every partial cut so far (this
  session and the N/T-series 50% cuts) kept the bias and paid for it. Neurons and bias form a balanced pair: thresholds
  downstream were set with both present.
- d5b final reference: block 17, 90k images, feature loss, cosine, 10 passes: sealed .0836 (19.7% won back; train .017 vs
  val .090). Best training reference for "drop block 16".
- d6_balance: law-balanced bias (keep the bias fraction that makes the size leave with the gate) beats bias kept / dropped
  at k = 384 (.147 vs .177/.169) and 768 (.135 vs .148/.170) but small keeps still lose to the full drop. |gate|-keep is
  worse than energy-keep at large k (2304: .061 vs .026): the gate measure ignores the HEAD, which also reads block 16.
- d6_law_all (every block with a same-stage next block): LN covers .51-1.16 of the lost gate on territory in every block;
  the law's novelty/r predicts which blocks hurt most when dropped (corr .87 across 10 blocks; r alone -.57).
  Stage 3 drop KL .019-.038; blocks 15/16 .128/.107.
- d6_robust: the law slope depends on the neuron set; same set, two image sets agree: ON territory per-neuron slope 1.07-1.10
  (bootstrap 1.01-1.14) -> the law holds on territory; all spots 2.2 -> off territory block 16 adds ~2x proportional
  inhibition (source of false firing). The earlier .93 used a gamma-weighted neuron set.
- d6_affine (per-neuron affine estimate of the intact vote): beta ~.5 (regression to the mean) -> .69-.94. d6_transport
  (distribution-preserving per neuron): moment matching .946, upper-tail .137, exact quantile map .149 -- all worse
  than the plain drop. Reading: per-neuron fixes break the JOINT pattern across neurons at a spot; the plain drop scales
  every vote at a spot by the same LN factor and keeps the pattern. MSE estimates also collapse variance (REPAIR, I6).
## D6 KEY RESULT (d6_orth.py): in block 17's READERS' metric (M = U'^T U', U' = block-17 reads x LN weight; vectors
## centered and after block 17's dwconv), block 16's write d is 92.9% proportional to the stream s (euclidean: 50.5%);
## lambda = <s,d>_M/<s,s>_M median 1.073 (p10 .935, p90 1.283): block 16 adds ~ONE MORE COPY of what the readers see.
Identity: d = lambda s + d_perp  =>  LN(s + d) = LN(s + d_perp/(1 + lambda)). Test (d_perp from the intact block, in stream
space with the readers' metric): plain drop .1042; add d_perp/(1+lambda) .0037; add d_perp .0040; full block 0.
RETRACTED (same day): the test is circular. x + d_perp/(1+lambda) = mu + (s + d)/(1+lambda) exactly, which LayerNorm maps to the
intact input for ANY lambda. It does not show where the information is. What stands: in the readers' metric d is 93%
proportional to s (lambda ~1.07). In euclidean terms d_perp also holds -lambda x the stream's UNREAD mass (size axis).
Next test splits d_perp into its readers'-subspace part and its unread part.
- d6_split: readers' metric spectrum: 90% / 99% / 99.9% of its trace in 342 / 580 / 702 of 768 dims (gradual). Add back
  only the readers'-subspace part of d_perp/(1+lambda) (oracle from block 16): 99.9% subspace .0769 (below training .0836),
  99% .1284, 90% .1419; unread part only .133-.524. The two parts interact through LN's size; neither alone is enough.
  CORRECTION (advisor): near-circular like the .0037 test. Adding P d_perp/(1+lambda) gives the intact stream (rescaled) in the
  702-dim readers' subspace; only the ~66 least-read dims keep their dropped values. Honest reading: leaving the 66
  least-read dims (the size axis) at their dropped values costs .077. It does not locate block 16's information.
- d6_check (advisor): (1) lambda / proportional share with per-spot FLUCTUATIONS only (mean over spots removed): lambda .166,
  share .075 (pooled .048) vs raw 1.073 / .929. In the readers' metric 91% of |s|^2 and 96% of |d|^2 lie along one shared
  mean direction (cos_M .992). Correct statement: block 16 adds ~one more copy of the stream's COMMON direction (the size
  axis LayerNorm divides by; the block's bias is aligned with it, cos .84 with its mean output), not a copy of what
  readers see in detail. (2) novelty/r vs drop KL within stage 3 (8 blocks): corr .21 (all 10: .87) -> a stage effect,
  not a per-block predictor. The per-block "law slope" column in d6_law_all is the all-spots slope, not the territory
  slope; only the healed-share column tests the law per block.
