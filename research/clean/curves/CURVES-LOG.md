# Curves track log (started 2026-10-22)

Plan: `~/.claude/plans/so-there-s-another-session-compressed-nest.md`. Testbed: lucid InceptionV1 (lucent port), layer 3b
(480 neurons, no residual stream, no LayerNorm). Essay experiment reproduced locally: everything up to 3a frozen, 3b
replaced by k neurons, only those and their reader trained. The essay retrained the whole network from scratch.

## G0. Gate (passed)
- venv `research/clean/curves/.venv` (torch 2.14.1, MPS). My split path (to_3a -> 3b -> from_3b) equals the full model exactly.
- 333 img/s forward. Dev top-1 61.9% with a rough empirical label map (921 unique of 1000; lucid model has 1008 outputs).
- Curve neurons (Distill list): 379, 406, 385, 343, 342, 388, 324, 340, 330, 349. All in the 5x5 branch (320-415).
- Radial tuning on synthetic arcs (radius 40, 36 orientations, centre 5x5 max): each neuron prefers one orientation
  (290, 300, 240, 180, 150, 100, 110, 50, 50, 20 deg), peak/mean 3.4-5.0, best radius 60 of {20,40,60,100}, straight-line
  response 4-40% of arc response. They are 10 of the top 12 of 480 neurons on arcs (others: 380, 359, 348, 363, 367...).

## G1. Whole-layer crush into frozen later layers: not usable as a reference
- Replacing 3b inside the frozen network is extremely damaging (1000 dev images): zeros KL 7.57, per-channel mean 8.66,
  top-240 of 480 by variance (rest mean) KL 9.1 / top-1 20.7%, rank-120 PCA copy KL 1.07 / 54.6%, rank-10 PCA KL 15.9 / 1.2%.
- So a 10-channel 3b cannot feed the frozen 4a. Training only the crushed 3b + 4a: KL 6.9 after 300 steps, top-1 0%.
  Training everything after 3b (6.4M weights, lucid scale, no normalization): unstable (dev KL 128 at step 150).
- Bug found and fixed on the way: mirrored-image targets were recomputed with the network being trained.
- Decision: the essay's whole-layer k=10 crush needs from-scratch training. Our local reproduction crushes ONE FAMILY
  ("ten curve detectors -> one curves & loops neuron"), the essay's own sentence, with everything else intact.

## G2. Curve family removal (2000 dev images)
- Curves -> zero: KL .665 (se .039), top-1 61.6%. Curves -> mean: KL .589 (.036), 62.1%. Intact top-1 65.1%.
- Ten random other 5x5 neurons -> mean: KL .128 / .121. The curve family carries ~5x a random group.
- `family.py`: exact smaller network (3b = 470 intact + k coarse channels reading the 5x5 bottleneck, 4a gets k new
  columns after the same max-pool). Copy init at k=10 gives KL 6e-6. Zero columns give .567 (= removal).

## S1. The fine curve family (no training involved)
Order by preferred orientation (radius 60): 330(10) 340(50) 324(70) 388(120) 343(170) 342(180) 385(250) 406(290) 379(320) 349(350).
- Kernels (32x5x5 on the 5x5 bottleneck): norms 1.05-1.17 (equal), bias .13-.27. Cosines near 0; neighbours +.15-.24,
  opposite orientations down to -.49. Only 5-16% of each kernel's energy lies along the family mean kernel.
- Natural images (1000 train images, all positions): each on at 9-15% of positions. Members on at one spot:
  0: 27%, 1: 38%, 2: 26%, 3: 7%, 4: 1%, 5+: <0.1%. Post-ReLU correlation: neighbours +.10-.24, others about -.05.
- 4a reads (1x1 conv columns): tight pairs with cosine .93-.96: (330,342), (340,388), (343,349), (385,379). Some
  pairs are curves 180 deg apart. The pool branch also reads tight pairs (.89-.96).

## G3. Training harness fixes (all found by direct checks)
- Dead coarse neuron (random kernel on non-negative input never fires): identical dev KL to 7 digits across lrs.
  Fix: bias calibrated so the neuron is on at 12% of positions (the family's rate) on one train batch.
- Unit mismatch: bias parameter ~ -3600 vs kernel entries ~ .035, so Adam rescrambled the kernel at 30%/step.
  Fix: every parameter stored in units where its entries are O(1). Copy init still exact (KL 6e-6).
- Fact found on the way: curve-neuron biases are ~0 (.13-.27 vs pre-activation std ~500). They fire from the kernel alone.
- fp16 overflows at lucid scale (KL 15); bf16 floor .006 for 22% speed. Full precision kept (.36 s/step).

## R1. Short references (600 steps = ~1 epoch, dev 1000 images; removal = .64 on this dev set)
| run | dev KL |
|---|---|
| k=1 random lr .003 / .01 / .03 | .456 / .260 / .177 |
| k=10 random lr .01 (harness check) | .197 (far from 0: not passed at 1 epoch) |
Long runs (3000 steps): k=1 random lr .03 and member:379 start both at .185 after 500 steps (start-independent so far).

## Q1. Qualitative reproduction of the essay (PASSED), k=1, lr .03, 600 steps
- Feature visualization (Lucent, FFT + decorrelated colour + standard transforms, 3b is an early layer): ONE LOOP,
  concentric curves around a centre, not an oriented curve. The ten fine neurons render as ten rotated curves
  (`cache/fv-fine-*.png`), matching Distill.
- Top natural crops (3000 train images): whole round things: record-player centre, camera lenses, wheel, dial, speaker,
  hubcap, whitewall tire, stethoscope loop. Most negative crops: rims and arcs off-centre or partly out of view (unread).
- Its kernel is NOT a combination of the ten curve kernels (1.5% of its energy in their span; 15% in the span of all
  96 5x5-branch kernels). It is on at 98% of positions (a graded dial, not a sparse detector) in the 600-step version.
- Essay rules applied: raw activations (no percentiles), no SAEs or simulation scores, labels from strong activations
  and both poles, feature visualization always shown with natural crops.

## S2. Reading the k=1 coarse neuron (600-step, lr .03 version; long runs pending)
- Circles vs arcs (centre response, raw): centred full circle radius 15-30 px: 3,500-4,600, peak exactly at the circle
  centre, weak negative surround; best fine curve neuron on the same circle 1,800-3,200. Single arc (mean of 12
  orientations): coarse 1,500-2,200 vs fine 1,600-2,700. Circle radius 60: coarse -190 at centre. A LOOP detector.
- Falsified: coarse kernel = combination of SHIFTED fine kernels (lstsq over 250 shifted kernels): share .364 vs null
  .356 (shifted kernels of 10 random other neurons). Unfair test: shifts of 2-4 positions leave the 5x5 window.
- Readers (4a input columns on the 10 curves, 368 x 10). Harmonics over preferred orientation (uneven, lstsq basis):
  all convs: h0 20%, h1 39% (null h1 17%), h2 14%, h3 9%, h4 15%. 3x3 bottleneck and pool branch read the plain sum
  (cos with uniform .94 / .91); 5x5 bottleneck reads h1 (59%, null 22%); 1x1 reads h1 (37%, null 19%).
  Meaning: the downstream mostly wants HOW MUCH curve (h0) and WHICH WAY it faces as one wave (h1).
- Function space (400 train images, inner 20x20 positions). corr(coarse, X):
  family sum at q .33 | loop sum L_r(q) = sum_theta relu(z_theta(q - r u_theta)), u = toward the centre of curvature:
  r=8px .51, 16px .52, 24px .39, 32px .25 | reversed direction (control) .07-.15.
  Meaning: the coarse neuron = the fine family with each member moved to its own centre of curvature, then summed.
  This is the average of the family over rotations about ONE point (the group average).
- Weight space, the same group average of kernels (cropped to 5x5): cos with coarse kernel r=1 +.10, r=2 +.18,
  r=3 +.09; reversed -.02/-.09/-.09; shuffled-orientation null ~0 (sd .04-.05). Real but small: cropping and
  sum-of-ReLU vs ReLU-of-sum are the two named gaps.

## S3. Mechanism checks (advisor items 2, 3) on the 600-step k=1 neuron
- Co-firing at the coarse neuron's strongest spots (400 train images): members on at q: all spots 1.1, coarse top 0.5%
  2.2. Members on at their MOVED positions (r=2): all 1.1, top 5% 3.0, top 0.5% 4.4 (often 4-6 of 10).
  Meaning: at one spot the curve members exclude each other; moved to their centres of curvature they become a
  co-firing group. A single ReLU can express "several agree" (consensus, as in the D-series), not "one of ten".
  (The 600-step neuron still favours orientations 70-180 deg: an incompletely symmetric solution. Firing rate 86%.)
- Whose radius (arcs 8-110 px, circles 8-110 px): fine members prefer arcs of 50-75 px (broad). The coarse prefers
  centred circles of 25 px (3,869-4,996 at 16-30 px) and is ~0 from 50 px. The 5x5 window is ~±16 px plus input reach.
- r* from images alone (1000 train images, moved members): mean pairwise post-ReLU corr r=0 .003, 8px .044, 16px .073,
  24px .086, 32px .097, 40px .082; reversed -16px .003. P(>=3 on | >=1 on): .10 -> .23 at 32px.
  PRE-REGISTERED derived r* = 4 positions (32 px, argmax of both measures). Needs a 13x13 window.

## Derivation (written before any derived number exists)
Derived coarse neuron, no fitted knob:
1. Read: K = sum over the ten fine kernels, each moved by r* toward its own centre of curvature (13x13 window).
   Meaning: the family averaged over rotations about one point.
2. Bias: sum of the ten fine biases (~0). Meaning: positive only when the agreeing members outweigh the silent ones
   (each silent member votes ~ -500): a consensus threshold with no knob.
3. Write (4a columns): the Bayes write. For each member, beta = Cov(pooled member, pooled coarse)/Var(pooled coarse) on
   analysis (train) images; coarse column = sum_theta beta_theta * original column_theta.
Comparisons: trained k=1 at 5x5 (native) and trained k=1 at 13x13 (same window as the derived one); removal .59.

## R2. Long references (3000 steps, cosine lr, early stop on dev 1000; dev removal = .640, se ~.05)
| run | best dev KL (se) | note |
|---|---|---|
| k=1 start = neuron 379, lr .03 | .139 (.015) at step 2000, flat to 3000 | converged |
| k=1 random start, lr .03 | .142 (.016) at step 2250 (running) | same as member start |
| k=1 random, lr .1 | .640 flat | neuron died: lr edge check passes (.03 is the best working lr) |
| k=3 random, lr .03 | .143 (.015) at step 2250 (running) | NO better than k=1 |
| k=10 random, lr .01, 600 steps | .197 | harness: ten random neurons cannot find the copy solution (copy = 6e-6) |
Meaning: training finds the k=1 solution from any start, but does not find multi-neuron solutions from random starts.

## S4. The converged k=1 neuron (start 379): same law
- Circle r=20 centre 5,081 (fine best 2,400); loop sum r=+1 corr .52 (.57 post-ReLU), r=+2 .48; reversed r=-2 -.15;
  firing rate 70-71%; bias +74 (in pre units, ~0). Moved members on at its top 0.5%: 4.1.
- Write side (advisor item 4, harmonic prediction): cos(trained column, mean original column): 3x3 bottleneck .81,
  pool .91 (h0 readers: match), 5x5 bottleneck .19 (h1 reader: no match), 1x1 .66. Norm ratio 2.2-3.7.

## D1-D4. Derivation attempts and exactly how each failed (dev 1000; removal .640; constant-only .391)
| build | dev KL (se) | what it shows |
|---|---|---|
| constant only: 4a biases get the family's mean as the readers see it (after max-pool) | .391 (.040) | mean-after-pool fill is far better than mean-before-pool (.589) |
| D1 sum of moved fine kernels, r*=4, bias = sum of biases (pre-registered) | .366 (.037) | fires 1.6%: each silent member votes ~-500, so consensus is too strict |
| D1 same, dial bias (never clips), r=2 / 4 | .370 / .381 | not the bias: the kernel sum carries little (Bayes betas ~0) |
| D1 r=2 / r=3 (sparse) | .359 / .360 | |
| D2 projection of the moved ReLU sum onto one neuron (whitened STA), r*=4, 13x13 | .355 (.036) | projection R^2 .74, rate .73 (trained .70), but readers gain little |
| oracle (exact quantity, not a neuron) + Bayes write: plain family sum r=0 / moved r=1 / 2 / 4 | .291 / .297 / .321 / .354 | even the exact quantity loses to the trained neuron (.139) |
| trained read + Bayes write | .307 (.033) | THE WRITE IS THE MAIN GAP (same read: trained write .139) |
Bayes write vs trained write (same read): cos 1x1 .62, 3x3b .83, 5x5b .28, pool .91; trained norm 1.2-2.5x larger.
Meaning: regression shrinks the write (the D-series lesson); in the h1 branches the direction also differs.

## D4. Variance-keeping writes (closed form, no knob): vmg = Bayes x one factor restoring total reader variance;
vmr = Bayes / rho per reader (each reader's variance restored). Means matched by biases.
| read | Bayes | vmg | vmr |
|---|---|---|---|
| trained (start 379) | .307 | .239 | .216 |
| oracle plain family sum | .291 | .236 | .240 |
| D2 r*=4 13x13 (pre-registered) | .355 | .353 | .301 |
| D2 r=0 5x5 | .312 | .261 | .256 |
| D2 r=1 7x7 | .311 | .262 | **.242 (.025)** |
Best fully derived neuron (no training): .242. Trained: .139. Removal: .640. The derived neuron wins back 80% of the
trained neuron's gain over removal (59% of its gain over the constant-only fill). Remaining gap: write direction in
the h1 branches (5x5 bottleneck, 1x1) - the trained write uses 4a's later spatial convs, a per-position regression
cannot see that. Pre-registered r*=4 is worse than r=1: the co-firing rule for r is NOT the readers' r.

## S5. Is the trained neuron one object? (advisor items 1, 5; 300 train images)
- Trained k=1 random start (dev .1405) vs start 379 (.1392): activation corr .80, kernel cos .51. Not unique, close.
- Derived (D2 r=1 7x7) vs trained random .73, vs trained 379 .64: the derived neuron is almost as close to a trained
  twin as the twins are to each other. Loop sum r=1 explains .60 / .53 of the trained twins (ceiling ~.80).
- Pictures (`cache/sheet-derived-vs-trained.png`, `cache/fv-derived2-r1-ks7-0.png`): derived and trained top crops are
  the same kind of thing, several the very same spots (wheels, whitewall tires, hubcaps, stethoscope loop, dial).
  Derived feature vis = clean concentric rings (an eye); trained = a loop/spiral. Negative poles of both: curved
  things off-centre or too big (lemon edge, barrel rims, letters, fabric dots). Circle r=20 at centre: derived
  6,971, trained 5,081, both peak exactly at the centre.
- k=3 random start: .1389 (same as k=1).

## D6-D7. Where exactly the derived write fails (trained random-start read; dev 1000)
| swap | dev KL |
|---|---|
| trained write / vmr write (all branches) | .140 / .234 |
| vmr write, only 1x1 branch from training | .166 |
| trained write, only 1x1 branch vmr | .201 |
| same swaps for 3x3b, pool | changes of .003-.006 |
| same swaps for 5x5b | changes of ~.02 |
The 1x1 branch (192 pointwise 4a neurons; they read the h1 wave) holds the gap. Per neuron:
- vmr divides by rho; for h1 readers rho ~ 0 (neuron 50: h1 .41, h0 .02, rho -.10) so vmr makes a huge column with an
  arbitrary sign (-341; trained +112). Sign agreement trained vs vmr: 52% where |rho|<.1 (chance), 77% where |rho|>.3.
- 60 of 192 neurons get a strongly negative trained column although their family weights are ~0: training uses the
  loop signal to GATE them off. The derived write never does this.
- Where |rho| > .3 the trained column is ~1.47x the Bayes column (regression shrinkage again).
Pre-registered fix from this: Bayes column (adds no unpredictable variance) + each reader's bias shifted so it fires
as often as in the intact network (reader rate law, analysis images).

## D8-D10. More write laws, and how they failed (dev 1000)
- Reader rate law (Bayes column, each 4a input-conv reader's bias shifted to its intact firing rate), PRE-REGISTERED:
  derived read .311 -> .307; trained read .308 -> .307; on top of vmr it HURTS (.242 -> .262; .234 -> .260).
  Meaning: shrinkage does not hurt through 4a thresholds. It hurts through AMPLITUDE further down: later layers weigh
  this evidence against all other evidence, so a shrunk vote counts for less even when every threshold still fires.
- Reading the trained write like a best-fit line (readers with |rho|>.15 and agreeing signs): trained = .68 x Bayes x
  |rho|^-.89 (random start), .75 x Bayes x |rho|^-.80 (start 379); fit corr only -.37/-.32. Exponent ~ -1 = the vmr
  form. Writes read off this (vmr x .7; vmr above |rho| .15, Bayes below) are all worse than plain vmr
  (.264/.256 derived read). The trained 1x1 column works as a whole vector, not as a per-reader law.
- Write-only training on the derived k=1 read (measuring tool, start = vmr): .242 -> .160 (trained k=1 .140).
  So the derived read costs ~.02, the write ~.08.
- The 60 gated 4a 1x1 neurons (pictures `cache/sheet-gated-4a.png`): straight lines and textures (140 parallel lines:
  window grids, wood grain, mesh; 48 vertical bars; 81 fur; 94 printed letters). Their family weights are ~0 and the
  family input spread small (140: sd 28), but training gives them a loop-driven inhibition ~4x anything the family
  wrote: "a curve here, so not a straight line or texture". A NEW gate, not a copy (open: why it pays).

## R3 / D11-D14. k = 3
| build | dev KL (se) | top-1 |
|---|---|---|
| keep 3 members 330, 388, 385 (10, 120, 250 deg), others dropped | .311 (.029) | .628 |
| + others' constant fill | .162 (.018) | .628 |
| **+ Bayes write (derived, NO training)** | **.133 (.015)** | **.634** |
| + variance-keeping write (vmk) | .138 (.014) | .631 |
| trained k=3 random start (converged) | .139 (.015) | |
| trained k=1 (both starts) | .139-.141 | |
| trained k=3 start = 330, 388, 385 | .073 (.008) at step 250 (running) | |
- RANDOM-START TRAINING WAS A BAD REFERENCE for k=3: it converges to the k=1 level. The member start finds a far
  better solution. The derived k=3 beats both random-start trained references with no training.
- Why three at 120 deg: readers read h0 + h1 (S2); three equally spaced members are the smallest set that spans
  h0 + cos h1 + sin h1.
- Spacing control (Bayes write): 10/120/250 .133 | 50/170/290 (also even) .164 | 170/180/250 .159 | 10/50/70 (adjacent)
  .208 | 10/180/290 .143. Adjacent is clearly worst; spacing is not the whole story.
- Two statistics scores to choose members both FAIL among good triplets: share of reader-input variance carried
  (ranks 50/170/290 over 10/120/250) and Fisher-weighted leftover error (same mistake; Fisher on the family channels:
  diag nearly flat .84-1.15, top eigen shares .31/.22/.18). Both rank adjacent triplets worst. Meaning: second-order
  (linear) metrics cannot separate good triplets; the ReLU readers decide.
- Bayes beats vmk for k=3 (reverse of k=1): three kept channels predict the removed ones far better, so shrinkage costs
  less and variance inflation adds noise.

## D15. Derived k-curve (closed form, NO training; dev 1000)
| k | build | dev KL (se) | top-1 |
|---|---|---|---|
| 0 | family removed | .640 | .613 |
| 1 | keep member 330 / 342 + Bayes | .291 / .285 | .617 / .623 |
| 1 | coarse loop neuron (D2 r=1 7x7) + vmr | **.242 (.025)** | .633 |
| 2 | 330, 342 (10, 180 deg) + Bayes | .219 (.024) | .621 |
| 3 | 330, 388, 385 (10, 120, 250) + Bayes | .133 (.015) | .634 |
| 5 | 330, 324, 343, 385, 406 (10, 70, 170, 250, 290) + Bayes | .082 (.011) | .633 |
| 10 | intact | 0 | .651 |
At k=1 the real coarse neuron (curves & loops) beats keeping any single fine neuron: coarsening is not pruning.
Trained k=3 from members: .073 (step 250) -> .063 (step 500).

## R4. Final references (dev 1000)
| run | best dev KL (se) |
|---|---|
| k=3 trained, start = derived members 330/388/385 | .0422 at step 3000 (still improving slowly) |
| k=10 trained from random (harness; copy init = 6e-6) | .0438 at step 2750 |
| k=3 read frozen at the members, write trained only (measuring tool) | .0775 |
| k=1 13x13 window, random start | stopped at step 750 (.179) to free the GPU; not used |
Meaning: three neurons started from the derivation match ten neurons trained from random. Ten random neurons with
all the data do NOT get back to the exact copy (.044 vs .000006): random-start training is an approximation.
k=3 accounting: derived .133 -> trained write .0775 (write costs ~.055) -> trained read+write .042 (read ~.035).

## COLD TEST (sealed 10k, ONE run; `results/curves-cold.json`, per-image KL `cache/cold_per_image.npz`)
intact top-1 .6577. KL (se) / top-1:
| build | sealed KL | top-1 | selection |
|---|---|---|---|
| family removed | .684 (.019) | .625 | - |
| constant only | .392 (.014) | .641 | - |
| simple cut: keep member 342 (largest variance) + mean | .325 (.012) | .646 | - |
| D1 moved kernel sum r*=4 + Bayes | .375 (.013) | .642 | PRE-REGISTERED (failed) |
| D2 projection r*=4 13x13 + vmr | .292 (.010) | .647 | r pre-registered |
| D2 loop neuron r=1 7x7 + vmr | .250 (.009) | .649 | r and write chosen on dev (9 variants) |
| D2 loop neuron r=1 7x7 + Bayes | .311 (.011) | .646 | |
| keep 330 / 342 + Bayes | .291 / .301 | .646 | |
| k=2 10/180 + Bayes | .220 (.008) | .650 | |
| k=3 10/120/250 + const / Bayes / vmk | .171 / .139 / .144 | .654-.656 | first pick (made for the training run) |
| k=3 50/170/290 + const / Bayes / vmk | .197 / .150 / .128 | .652-.656 | control |
| k=3 10/50/70 + const / Bayes / vmk | .199 / .180 / .162 | .651-.655 | control (adjacent) |
| k=3 naive (largest variance: 342, 340, 343) + const / Bayes / vmk | .192 / .172 / .163 | .650-.651 | baseline |
| k=5 + const / Bayes / vmk | .129 / .088 / .074 | .655-.659 | |
| k=3 members, write trained only (tool) | .076 (.003) | .660 | |
| trained k=1 start 379 / random | .1205 / .1215 | .657 | |
| trained k=3 random start | .1217 (.0045) | .656 | |
| trained k=3 start = derived members | **.0373 (.0014)** | .658 | |
| trained k=10 random (harness) | .0406 (.0015) | .658 | |
PAIRED (same images): derived k=3 Bayes - trained k=3 random = +.0171 (.0044) -> RETRACTION: on sealed the derived k=3
does NOT beat random-start training (dev .133 vs .139 was inside dev noise). 10/120/250 - naive = -.033 (.005);
10/120/250 - 50/170/290 (Bayes) = -.0115 (.0049); Bayes - vmk (10/120/250) = -.005 (.005), not separated;
loop neuron - keep 342 = -.052 (.007); loop neuron - trained k=1 = +.128 (.007); trained-from-derived k=3 - derived
k=3 = -.10 (.005). Sealed numbers for trained models are ~.02 lower than dev (dev is a noisier 1000-image subset).

## P5. Second family: the circles in 3b (the essay: "two circle neurons instead of several circle types")
- Scout (centred full circles 6-60 px vs single arcs): circle detectors respond ~2x more to a full circle than to any
  arc. They spread over ALL four branches and tile RADIUS: 6 px 462, 82, 300 | 10 px 311, 446, 255, 352, 315 |
  15 px 363, 454 | 20 px 323 (also 158, 230, 28, 276 weaker). Feature vis (`cache/fv-fine-462_..._323.png`): small
  dots, eyes (dark centre, bright ring), concentric rings, double circles, one swirl (454).
- Generic member-keep code `p5_drop.py` (rewrites only 4a input weights/biases). Gate: keep all 10 curves -> 6e-6.
- Choice rule fixed before testing: evenly spaced in preferred radius; inside a bin the largest-variance member.
  Picks: k=1 446 (10 px) | k=2 82, 323 (6, 20 px) | k=3 82, 363, 323 (= naive top-3, so k=3 cannot test the rule) |
  k=4 82, 446, 363, 323. Naive (largest variance): k=1 363, k=2 363, 323.
| circle family (11), dev 1000 | const | Bayes | vmk |
|---|---|---|---|
| removed (zero) | .309 | | |
| removed + constant fill | .133 | | |
| k=1 rule 446 / naive 363 | .111 / .118 | **.078** / .090 | .089 / .108 |
| k=2 rule 82,323 / naive 363,323 | .095 / .110 | **.063** / .084 | .073 / .104 |
| k=3 82,363,323 | .079 | .048 | .051 |
| k=4 82,446,363,323 | .065 | .043 | .046 |
Bayes beats vmk again for kept members (as for curves k=3).
- References (`p5_train.py`): kept members' 3b kernels + their 4a columns + 4a biases trainable inside the real
  network; derived start reproduces .0634 exactly; random start begins at .293.
- Dev references: trained k=2 (82, 323) from the derived start .0326; from random .0501.
### P5 COLD TEST (sealed 10k, one run; `results/curves-p5-cold.json`)
| circle family (11) | sealed KL (se) | top-1 |
|---|---|---|
| removed / + constant | .335 (.011) / .159 (.007) | .642 / .652 |
| k=1 rule 446 / naive 363, Bayes | .085 / .090 | .656 |
| k=2 rule 82,323 / naive 363,323, Bayes | **.062** / .082 | .657 / .656 |
| k=3 82,363,323 Bayes | .043 (.002) | .657 |
| k=4 rule Bayes | .039 (.002) | .657 |
| trained k=2 from derived start | **.031 (.001)** | .660 |
| trained k=2 from random | .052 (.002) | .660 |
PAIRED: k=2 rule - naive -.0196 (.0031) | k=1 rule - naive -.005 (.003) not separated | derived k=2 - trained random
+.0104 (.0020) | trained derived-start - trained random -.0207 (.0014) | trained derived-start - derived k=2 -.031 (.002).
Meaning: the second family repeats the curve result exactly. Derived alone loses to random-start training at the
same size; the derived START beats the random start; the even-spacing rule beats the naive pick (k=2).

# NEW METHOD (2026-10-22 night). Plan: `~/.claude/plans/okay-i-want-you-optimized-perlis.md`. Theory page: `THEORY.md`.
Labels: DERIVED (principle before number) / CHOSEN (menu on dev, with menu size) / READ-OFF (from a trained solution).
Dev comparisons: paired, all 5000 dev images. Training = ruler or microscope only.

## Card W0/W2: the Fisher write (written before any number)
- Observation: read costs ~.02, write ~.08 (k=1). Training (a) votes larger than Bayes, (b) gates 60 line/texture 4a
  1x1 neurons whose family weight is ~0.
- Theorem: KL(intact||crushed) ~ 1/2 du^T G(x) du, G = output Fisher pulled back to the 368 4a reader inputs at the
  intact net. If G were constant, E[u|c] is optimal for ANY G, and its linear form is Bayes, which never writes to
  readers with zero family weight. So Bayes failing + the gate imply G(x) varies with the image and couples readers,
  or the quadratic picture breaks.
- W0 (ruler check) prediction: 1/2 du^T G du ranks trained < vmr < Bayes; per-image corr with real KL > .6.
- W2 derivation: B*, b* = argmin E_x E_c~p(x) [(g_c^T (u - Bc - b))^2] with g_c = sqrt(p_c) dlog p_c/du at the intact
  net (exact Fisher, top classes). One linear solve, 368(k+1) unknowns. Exact case: k=10 members -> original columns.
  Label: DERIVED. Prediction: dev KL <= .19 with the derived read (vmr .242 on dev-1000, trained write .160);
  >= 40 of the 60 gated neurons get negative 1x1 columns.

### BUG FOUND (W0c): lucent's RedirectedReLU leaks 0.1 x the gradient through switched-off units
- `lucent.modelzoo.inceptionv1(pretrained=True)` defaults to `redirected_ReLU=True` (a feature-vis trick; its own
  docstring: "the gradient is technically wrong"). Forward is a true ReLU. Found because the autograd slope of the
  logits was 1.8x the finite-difference slope (cos .93) at steps down to 1e-4 in float64.
- Fixed: `load_model(redirected=False)` is now the default; `fv.py` keeps `redirected=True`. After the fix autograd
  = finite difference (cos 1.000, norm ratio 1.000).
- NOT affected: every forward-only number (all derived builds, all cold-test numbers of derived builds).
- AFFECTED: every TRAINING run in this track (k=1, k=3, k=10, write-only, circles P5 references) used the leaky
  gradient. Open: whether "k=3 random stuck at the k=1 level", "k=10 random misses the copy" and the 60-neuron gate
  are partly artifacts of this. The references must be rechecked with true gradients.

### W0 result (true gradients, dev 1000, seed-5 subset)
| build | predicted 1/2 du^T G du | real KL | per-image corr |
|---|---|---|---|
| derived read + Bayes | .339 | .351 | .48 |
| derived read + vmr | .312 | .284 | .53 |
| trained read + trained write | .123 | .140 | .41 |
- Ranking trained < vmr < Bayes: HOLDS. Means match within ~10%. Per-image corr > .6: FAILS (.41-.53).
- W0b (scale and expansion point): for Bayes, vmr and trained, real/(t^2 pred) ~.8-.9 for t = .1 ... 1, and ~1.0
  when the Fisher is taken after 4a. The second-order picture is good on average for these builds.
  (Before the bug fix the prediction was 3-4x too large: that was the leaky gradient.)

### W2 result: Fisher write (DERIVED, no knob), first solve on 1500 train images
| write (derived read) | dev-5000 KL (se) |
|---|---|
| Bayes | .315 (.015) |
| vmr | .248 (.012) |
| Fisher | .241 (.010) |
| (trained read + trained write, ruler) | .126 (.007) |
- Paired: Fisher - vmr -.007 (se .011): NOT separated. Fisher - Bayes -.074 (.014). Prediction <= .19: FAILED.
- Gate: Fisher write gives negative 1x1 columns on 42 of the 63 gated neurons (prediction >= 40: HOLDS); corr with
  the trained 1x1 column .61 (trained read: 49/63, .65). So the gate IS a second-order Fisher effect: the output's
  sensitivity couples curve readers to line/texture readers on loop images. Bayes cannot see this (theorem).
- Why W2 failed (W2b): the solve OVER-FITS. Predicted damage on its own 1500 images .024, on fresh images .112
  (trained write: .089 / .105). Each image gives only ~2-3 informative class rows for 736 unknowns. So W2 did not
  test the theory; it tested a noisy estimate. The false assumption was statistical: "1500 images are enough".
- Re-derivation (not a knob): the estimator must converge. Rule fixed before running: fresh-image predicted damage
  within 10% of the solve-image value. W2c: 12000 train images.

### Literature (done after W2, not before as planned: noted as a slip)
- W* is in the OBS family: second-order loss increase, Fisher as Hessian (Hassibi 1993; WoodFisher, Singh &
  Alistarh 2020). Closest closed form: GFWSVD (Matrix-free Fisher Factorization, ICML 2026 poster), a Fisher-weighted
  layer decomposition under a Kronecker (matrix-normal) assumption. Its Kronecker structure is also a variance
  fix for our row-poor estimate (G ~ position factor x channel factor), if W2c does not converge.
- Fisher merging (Matena & Raffel 2022): per-parameter Fisher-weighted average, closed form. They warn the Fisher
  is LOCAL and fails when models sit far apart: the same limit as our "write moves where the quadratic breaks".

### W2c result: Fisher write, 12000 train images (DERIVED, no knob)
- Convergence: predicted damage on solve images .043, on fresh .055 (gap 30%; rule asked <= 10%: NOT converged yet).
  Real KL on fresh .099: real is still ~1.8x the second-order prediction for this write (higher-order terms).
- dev-5000: Fisher .200 (se .009) | vmr .248 | Bayes .315. Paired Fisher - vmr -.048 (se .011); - Bayes -.114 (.014).
- Prediction (<= .19): MISSED by .01, but the first derived write that clearly beats vmr (READ-OFF-confirmed form).
  1500 -> 12000 images: .241 -> .200. The estimate is still moving. Next: all 35000 train images (the estimator's
  natural limit, no new choice), normal equations saved.

### W2d result: Fisher write, all 35000 train images (DERIVED, no knob, no dev choice)
- dev-5000: Fisher .177 (se .008) | vmr .248 | Bayes .315 | (ruler: trained k=1 .126, leaky-gradient training).
  Paired Fisher - vmr -.071 (se .010); - Bayes -.138 (.013). PREDICTION (<= .19) HOLDS.
- Trend with analysis images: 1500 .241 -> 12000 .200 -> 35000 .177. Still data-limited (not converged).
- Share of the vmr-to-ruler gap closed: (.248-.177)/(.248-.126) = 58%. With no training and no knob.
- Cost note (honest): the write needs one pass over the analysis images WITH gradients per coarsening choice
  (G is fixed, but the features sum g x c, and c depends on the choice). Not yet "seconds per choice".

## Card W3 (written before running): the same Fisher write for kept members
- Derivation unchanged: target = what the whole family gave the 368 readers; source = kept members' pooled outputs;
  one solve. Label DERIVED (the kept members are the earlier even-spacing rule: curves 330/388/385, circles 82/323).
- Prediction: curves k=3 Fisher beats Bayes paired on dev-5000 by >= .02; circles k=2 by >= .01. 12000 images
  (1472 / 1104 unknowns: may be more data-limited than k=1).
- Ruler check running: k=3 random start with TRUE gradients, same settings as L-k3-r-lr3e-2, to step 1000.

### Ruler check (true gradients): k=3 random start, same settings, dev 1000
| step | 250 | 500 | 750 | 1000 |
|---|---|---|---|---|
| old, leaky gradient (L-k3-r-lr3e-2) | .217 | .182 | .165 | .161 |
| new, true gradient (T-k3-r-lr3e-2) | .208 | .181 | .162 | .157 |
Meaning: the leaky gradient did not change random-start training measurably (differences << se .016). The
rulers stand (probably). "k=3 random stuck near the k=1 level" is NOT explained by the bug (checked to step 1000).

### Microscope: trained read + Fisher write (12000 images), dev-5000
- trained read + Fisher .207 | derived read + Fisher .200 (same 12000-image solve size) | trained read + trained write .126.
  Meaning: with the Fisher write, the DERIVED read is as good as the trained read. At k=1 the whole remaining gap is
  in the WRITE.
### W2e: is the write gap data or higher order? (800 dev images, same images)
| write | second-order predicted | real | real/pred |
|---|---|---|---|
| derived read + Fisher (35000) | .090 | .174 | 1.93 |
| trained read + trained write | .108 | .113 | 1.05 |
- The Fisher write already beats the trained write IN the second-order model. Real damage is 2x its prediction.
  So the gap is beyond second order, not mainly data.
- Conjecture (from the ReLU, before testing): a reader OFF at the intact network has zero gradient, so the Fisher
  model charges nothing for pushing it up. The solve parks error there; the reader turns ON in reality: damage the
  model cannot see (one-sided kink). Falsifier: forcing intact-OFF readers to stay off does not bring the Fisher
  build's real KL near its prediction (~.09).
- RESULT (800 dev): Fisher build real .174 -> off readers kept off .094 (its prediction .090); only-off-readers-changed
  .047 (trained build: .113 -> .099; .007). Woken share 3.7% vs 3.0%. CONJECTURE HOLDS: the extra damage of the
  Fisher write is false waking of readers that are off in the intact network. Training learned not to wake them.

## Card W4 (written before running): statistically linearized Fisher write
- Assumption that broke: the reader's ReLU slope is 1[a > 0] (exact only for infinitesimal changes).
- Re-derivation: the write leaves a random error delta on reader j, with std sigma_j that the coarse neuron cannot
  predict: sigma_j^2 = Var(u_j)(1 - rho_j^2) (Bayes residual, closed form). Averaged over that error, the ReLU's slope
  at pre-activation a is E[1(a + delta > 0)] = Phi(a / sigma_j) (statistical linearization; Bussgang). So
  g_pre = g_post * Phi(a / sigma_j) replaces g_post * 1[a > 0], at the four 4a input convs only. Same single solve.
  Label DERIVED. Exact case: sigma -> 0 gives the W2 write back.
- Prediction: 12000 images, paired against the W2 Fisher write at 12000 (.200): <= .16; woken share < 3.7%.
- W4 RESULT: smoothed .2015 vs Fisher .2002 (12000 images, dev-5000), paired +.0013 (se .0008). FAILED (<= .16).
  sigma / sd(pre) only .07-.13, so Phi(a/sigma) ~ step. Diagnosis (W4b, 400 train images): the write's own error is
  1.25-1.57x the Bayes sigma, and woken readers are pushed 2-3x past their threshold (1x1: push 95, depth 41).
  False assumption: "waking is small random error near the threshold". It is a large systematic push, which a
  mean-slope linearization cannot charge.

## Card W5 (written before running): active-set Fisher write (exact one-sided ReLU at the 4a readers)
- For a fixed set of readers that are ON after the change, dh = on_new * (v - u) + (on_new * a - relu(a)) is linear
  in the write, so the second-order damage at the 4a outputs is quadratic: one linear solve. Alternate: set from
  the current write -> solve -> new set. Standard active-set method for piecewise-quadratic problems. Start: the W2
  write (12000 images). Label DERIVED (no knob; iteration count = until the set stops changing, report each step).
- Prediction: step 1 on dev-5000 <= .17 (W2 at 12000: .200); woken share falls below 3%.
- W5 RESULT (12000 images, dev-5000): step0 (W2) .200 -> step1 .150 (paired -.050, se .005) -> step2 .154
  (-.047, .005). Woken share 4.1% -> 3.1%. Prediction <= .17 HOLDS; woken < 3% just missed (3.09%).
  The active set swings, it does not settle (step 2 no better). Step 1 is the pre-registered build.
  Gap closed vs vmr-to-ruler: (.248 - .150)/(.248 - .126) = 80%, no training, no knob.
- Next: step 1 from the 35000-image Fisher write on 35000 images; then ONE sealed test.
- W5 at 35000 images (dev-5000): step0 .177 -> step1 .133 (paired -.044, se .004). Woken share 3.7%.

## COLD TEST of the write cards (sealed 10000, ONE run; `results/curves-w-cold.json`, `cache/w_cold_per_image.npz`)
| build (derived read r=1 7x7, no training) | sealed KL (se) |
|---|---|
| Bayes write | .311 (.011) |
| vmr write (old best) | .250 (.009) |
| Fisher write W2 (35000 images) | .176 (.007) |
| active-set Fisher write W5 step 1 (35000) | **.135 (.005)** |
| trained k=1 (ruler) | .122 (.005) |
PAIRED: Fisher - vmr -.073 (.008) | active - Fisher -.041 (.003) | active - vmr -.114 (.007) | active - trained
+.014 (.003). The derivation still LOSES to training at k=1, by .014. It closes 89% of the vmr-to-training gap and
95% of the constant-fill (.392) to training gap. Labels: write DERIVED (no knob; step 1 pre-registered); read r CHOSEN
(best of {0,1,4} on dev); window 7x7 follows from r=1.

# WEIGHTS-ONLY (2026-10-23). Plan: `~/.claude/plans/first-thing-i-want-enchanted-sutherland.md`. Paper: `PAPER.md`.
Rule (Shonu): the final formula reads the WEIGHTS only; images only CHECK. Labels DERIVED (weights only) / FITTED
(least squares or statistics on images, the old "derived") / TRAINED. Under this rule the .135 build is FITTED.
Budget: <= 6 image runs plus one cold test. Used so far: C2, C3, C4, C5 (C1 used no images).

## Theorem 1 (balance), weights only (`scratchpad balance.py`)
- Rescaling symmetry + weight decay => ||in||^2 = ||out||^2 per ReLU neuron at a stationary point (proof in PAPER.md).
- 3b, 480 neurons: ratio median 1.02 (IQR .95-1.09, log10 sd .05). 3a, 256: .996 (.94-1.05). Curves 1.0-1.17.
  3b 5x5 bottleneck (bias not counted) ~.75: open. HOLDS: lucid InceptionV1 sits at a weight-decay balance point.

## C1 (weights only): shifts that maximise ||sum T_s K_i||^2 (Corollary 2: = error-weighted co-firing)
- Prediction: ring toward each member's centre of curvature, radius ~1. Result: energy x1.59; mean direction right
  (cos .997) but scattered (resultant .67; two members 80-95 deg off); radius 1.5. PARTLY HOLDS. (`c1_interference.py`)

## C2 (8000 val images, true labels): is the mean loss gradient = -lam theta?
- Prediction cos >= .5: FAILED. cos(-g, theta) -.08 ... -.00 on 9 layers; split-half cos .66-.95 (stable signal).
- Reading: identity (B) is about TRAINING images; on unseen images a stable gradient is the generalisation gap
  (plus the rough label map). We cannot check Theorem 2 directly. It rests on its proof + Theorem 1. (`c2_stationary.py`)

## C3 (100 images): the "1.5% in the curve span" fact
- In the input metric E[XX^T]: random-start trained k=1 kernel 61% in the curve span (raw 3%), 93% in the 96-kernel
  span (raw 28%). Member-start: 25% / 85%. RETRACTED: the raw metric is meaningless for an un-decayed trained kernel.

## C4 (100 images): trained k=1 vs weights-only moved sum (centred pre-activation corr)
- Prediction >= .8: FAILED. C1 shifts .42, r=1 .43, unmoved .15-.22; trained twins .79. Moved sum mean -5,900
  (sd 2,400), on 0.7% at bias 0; trained mean +350 (sd 620), on ~70% (a dial).

## C5 (dev-5000, paired; FITTED Fisher write on 1500 images used as a fixed instrument)
| read | on-rate | dev KL (se) |
|---|---|---|
| der2 (FITTED least-squares read) | .81 | .241 (.010) = W2, reproduced |
| C1 moved sum, never clips (DERIVED read) | 1.00 | .357 (.014) |
| C1 moved sum, bias = sum of biases (consensus) | .006 | .845 (.049) |
- Prediction (dial within .02 of der2): FAILED, paired +.116 (se .010). The read is NOT already derived.
- Meaning: the least-squares read = E[XX^T]^-1 E[X sum relu(z_moved)]. It differs from the linear moved sum because
  E[X relu(z_i)] is the input covariance ON member i's gate applied to K_i, not the full covariance (Bussgang fails
  for non-negative, sparse inputs). That gated plain moment is exactly what Theorem 2 does NOT store (Theorem 5).

# MIX (2026-10-23). Plan: `~/.claude/plans/i-d-like-you-to-squishy-yao.md`. Paper: `PAPER.md` sections 8+.
Decision (Shonu): the image mix is a declared input. Generalist = the training mix (stand-in: train split). Developer
= a list of capabilities (images, what the app reads from the output, weight). Labels: DERIVED (weights only),
MIX-DERIVED (weights + closed-form averages over the declared mix), FITTED (chosen on dev / iterative on the output
loss), TRAINED. Budget: <= 7 image runs + 1 cold test.

## B0 (premise, no build). Does the curve family see two natural mixes differently?
- Mixes fixed by a rule that does not look at the family: ANIMALS = ImageNet classes 0-397, REST = 398-999.
  Train split only. Each mix split into two disjoint halves of 1750 images (A, B), by a fixed seed.
- Measured per half: member fire rate and mean (10 each), post-ReLU co-activation corr (45 pairs), and the
  MIX-DERIVED least-squares read of the C1 moved sum (the read Phi would build). Read compared by cos of kernels and
  by corr of pre-activations on 200 common dev images.
- PREDICTION (written before the run): between-mix differences exceed within-mix split-half differences for the
  rates and the co-activations by >= 2x. For the read: between-mix pre-activation corr is lower than split-half corr
  (1 - corr at least 2x larger). If the read shows no mix effect beyond noise, the curve family is the wrong testbed.
- RESULT (`b0_premise.py`, 42 s, 7000 train images). HOLDS.
| quantity | within-mix (split-half) | between mixes | ratio |
|---|---|---|---|
| fire rate, mean abs diff over 10 | .0018 | .0215 | 11.7 |
| mean activation, relative diff | .029 | .252 | 8.7 |
| co-activation corr, mean abs diff over 45 | .0022 | .0082 | 3.7 |
| read kernel, 1 - cos (raw metric) | .041 | .065 | 1.6 |
| read pre-activation on 200 dev images, 1 - corr | .0061 | .0184 | 3.0 |
- Meaning: the family's statistics move with the mix far beyond noise. Man-made images light member 324 (3rd in
  s1 order) 1.5x more (.156 vs .095). But the READ barely moves (corr .98 between mixes). So in 3b the mix lever, if it
  matters, will act mostly through the WRITE (which classes the output Fisher cares about) and the bias/level.

## Paper: PAPER.md Part II written (Def. 6, Lemma 8.1, Cor. 8.1b, Theorem 7, Lemmas 8.2-8.3, Props. 8-9)
- Cor. 8.1b: a list of capabilities is ONE measure on (image, output metric) pairs. Answers Shonu's "one Q" worry.
- Lemma 8.3: write moments compiled under the generalist read lose only 2nd order in how far a mix moves the read.
- Prop. 8 corrected on paper before any run: tilt by child i's SHARE, not its activity; member consistency (E2)
  forces a gated read.

## B1 (data hunger). Phi from 3500 images, twice, on disjoint halves of the train split (generalist mix)
- Build (MIX-DERIVED): read = least squares of the C1 moved sum (ks 11, moments.py), write = one Fisher solve
  (w_fisher.solve, R = 10), both from the same 3500 images. No knob.
- PREDICTION: (1) both halves land at dev KL .20 - .25 (the old curve: .241 at 1500, .200 at 12000 for the der2 read).
  (2) paired dev difference between the halves |mean| < .01. (3) KL on the build's OWN images is lower than dev KL by
  > .02, i.e. the hunger is variance (over-fit), so the noise cut (B2) is the right fix, not more images.
- RESULT B1: (1) HOLDS: .200 and .233 (se .009/.011). (2) FAILED: paired h1 - h2 = -.034 (se .006). (3) invalid as
  first run (class-sorted file: "first 1000" = all animals); redone on random 1000: own .184/.216 vs unused train
  .218/.250, so over-fit to own images is weak. Read stable (kernel cos .98); write unstable (column cos .59-.88,
  column biases cos .02-.22). Damage is heavy-tailed: effective sample ~8% of images; 48 of the 50 most damaged dev
  images are man-made. Generalist build: animals dev KL .10, rest .27-.31.
- PROCESS NOTE: after B1 I ran 4 diagnostic probes with no theorem (Shonu stopped it: "engineering, not the
  mathematical method"). Back to the paper. Phi is defined on POPULATION moments; estimation noise is set aside.

## Paper: Theorems 10-11 + Corollary 11.1 (PAPER.md section 11). WHAT the coarse neuron computes.
- k = 1: children's weights lam = top eigenvector of A^T F A C(nu); write b = A C lam / (lam^T C lam). The mix enters
  through the 10 x 10 co-activity C (55 numbers). Perron: if both factors are positive, lam > 0 and the ReLU is exact.
- k coarse: top-k SVD of F^(1/2) A C^(1/2); exact with ReLUs if the span has a non-negative basis. Rotation family:
  k = 3 = three bumps 1 + cos(theta - phi_j), 120 deg apart, NOT three kept members (new prediction).

## T1 (weights + C only, F = identity). Grounding Theorems 10-11 on the curve family
- C(nu) = E[z z^T] of the pooled children (14 x 14 interior) on 1500 train images per mix (generalist, animals, rest).
- PREDICTIONS (written first):
  P1 A^T A and C are entrywise positive, so lam > 0 (Perron applies).
  P2 lam is close to h0: min/max of lam > .5.
  P3 the 2nd and 3rd right-singular vectors of A C^(1/2) are the h1 pair: >= 70% of their energy in span{cos, sin} of
     the preferred angles (PREF).
  P4 mix lever: lam(rest) puts more weight on member 324 than lam(animals); 1 - cos(lam_ani, lam_rest) is > 3x the
     split-half value.
  P5 sigma_1^2 is >= 60% of the total (one coarse neuron carries most of the family's message).
- RESULT T1 (`t1_svd.py`, 20 s; signed check afterwards: all lam > 0, the abs() hid nothing).
| | prediction | result | verdict |
|---|---|---|---|
| P1 | A^T A and C entrywise > 0 | C yes (min .055). A^T A NO: opposite-facing members overlap down to -.43 | partly |
| P2 | lam near h0 (min/max > .5) | min/max .10: members at 250, 290, 320 deg get .01-.02, the other seven .12-.15 | FAILED |
| P3 | Lam2, Lam3 in span{cos, sin} >= 70% | .85-.88 | HOLDS |
| P4 | mix moves lam beyond noise; rest weighs 324 more | 1-cos ani/rest .0052 vs split-half .0011 (4.9x); 324: .103 ani, .121 rest | HOLDS |
| P5 | sigma_1^2 >= 60% | .44 (then .21, .15, .11) | FAILED |
- Meaning: the readers' overlap A^T A is NOT a positive circulant. It is h1-like: neighbours +.4 to +.6, facing-
  opposite members -.2 to -.4. The readers read which way a curve faces. A single coarse neuron that included the
  250-320 members would cancel part of its own message, so Theorem 10 drops them. Even the top singular value is only
  44%: in the identity metric one neuron cannot carry the family. Perron fails, but lam > 0 anyway here.
- NEW PREDICTION from Theorem 10 (sharp, not seen before): the TRAINED k = 1 neuron also nearly ignores members
  385, 406, 379 (250, 290, 320 deg). Test = T2.

## T2: does the TRAINED k = 1 neuron compute what Theorem 10 says?
- Trained twins (random start): L-k1-r-lr3e-2, fam-curves-k1-lr0.03-s0-random. lam_trained = regression of the
  trained neuron's pooled output on the 10 pooled children (+ intercept), 500 train images, interior positions.
- Theory, two metrics: identity (T1), and the mix's child-level Fisher Phi = E sum_p sum_c p_c g_c g_c^T,
  g_c = d log p_c / d z(p) (10 x 10, 1000 train images, top-10 classes as in w_fisher).
- PREDICTIONS (written first):
  Q1 both twins put < 1/3 of the other seven's mean weight on members 385, 406, 379.
  Q2 corr(lam_trained, lam_theory) > .8, and the Fisher metric is closer than the identity metric.
  Q3 trained write vs b = A C lam: cos > .7 on each of the four 4a input convs.
- RESULT T2 (`t2_trained.py`). ALL THREE FAILED.
| | prediction | result |
|---|---|---|
| Q1 | low-3 / high-7 < 1/3 | .70 and .59 (lower, but far from 0) |
| Q2 | corr(lam_trained, lam_theory) > .8, Fisher closer | identity .52 / .46, Fisher .53 / .51 (twins agree .83) |
| Q3 | write cos > .7 per conv | 1x1 .47-.57, 3x3 .64-.80, 5x5 .39-.49, pool .81-.90 |
- Also: a linear combination of the pooled children explains R^2 .46-.49 of the trained neuron.
- Captured message fraction in Theorem 10's model (free, from saved matrices): k = 1..4 = .44, .67, .82, .92 (Fisher
  metric). Trained k = 1 removes ~80% of the removal damage. So the model class is too small.
- FALSE ASSUMPTION (named): Theorem 10 has no per-reader constant beta. Every real build has one, and the constant
  alone (constant fill, .391 vs removal .62) removes ~37% of the damage. Theorem 10 spends its one neuron on the
  family's MEAN. Re-derive with beta (Theorem 10b), then re-test. No patch to the build.

## T3: Theorem 10b (with beta) against the trained neuron and against the measured damage
- mu = E z of the pooled children (generalist, the T2 images and positions). Phi from T2. Sigma = C - mu mu^T.
- PREDICTIONS (written first):
  R1 the beta share is .30 - .45 (measured: constant fill removes ~37% of the damage).
  R2 beta share + one-neuron share is .70 - .90 (trained k = 1 removes ~80%).
  R3 corr(lam_10b, lam_trained) > .7 for both twins (twins agree .83).
- RESULT T3 (`t3_centred.py`). FAILED (R1 narrowly, R2, R3).
| | prediction | result |
|---|---|---|
| R1 beta share | .30 - .45 | .26 |
| R2 beta + one neuron | .70 - .90 | .53 (k = 2: .70, k = 3: .82) |
| R3 corr with twins | > .7 | .57, .50 |
- New fact: the centred optimum has NEGATIVE weights on the three opposite-facing members (-.09 to -.11). So the best
  LINEAR neuron reports "which way the curve faces" (h1 in the readers' metric). The trained neuron keeps every weight
  >= 0 and reports "how much curve" (h0-tilted). Lemma 10.1 does not apply to the centred optimum.
- Calibration of the second-order shares: constant fill carries .26 predicted vs .37 of the damage measured. So shares
  under-state KL gains by ~1.4x. Even so, .53 x 1.4 = .74 < trained ~.80, and the WHICH-children answer is wrong.
- Three cards (T1-T3) without the theory matching training. Rule: rewrite the theory page, look at pictures and the
  essay, then a new conjecture.

## T2/T3 instrument fix (not a new card): moved children, and lam >= 0 as part of the model
- Flaw (advisor): T2 and T3 used UNMOVED pooled children, but the coarse neuron tracks children MOVED to their centres
  of curvature, and Theorem 10's "shift < one pooled cell" fails for ~4 of 10 members (shifts up to 3 at 28 x 28; one
  pooled cell = 2). Also the centred optimum with negative weights is outside the model (Lemma 10.1 needs lam >= 0).
- Fix: each child's 28 x 28 map moved by its C1 shift, then pooled (z_m). Readers still read the unmoved z.
  Loss reduction (L2, centred) = lam^T S_mz Phi S_zm lam / lam^T S_mm lam; maximised over lam >= 0 exactly by
  enumerating supports (1023 subsets of 10).
- PREDICTIONS: S1 the trained neuron on moved children: R^2 > .65 (unmoved .46-.49).
  S2 corr(constrained lam, trained lam on moved children) > .7 for both twins.
- Budget: image runs so far B0, B1 (+fix), T1, T2, T3 = 6 of 7. This fix re-runs T2's instrument.
- RESULT (`t4_moved.py`):
| | prediction | result | verdict |
|---|---|---|---|
| S1 R^2 of trained neuron on moved children | > .65 | .51-.52 (unmoved .42 on the same images) | FAILED |
| S2 corr(constrained lam, trained lam) | > .7 | .82 and .81; the twins agree with each other at .78 | HOLDS |
- So: with the model's own constraint (lam >= 0) and moved children, Theorem 10b predicts WHICH children the trained
  neuron weighs, and how much, as well as two trained twins agree with each other. Weights-only + mix: no training.
- Still open: (a) half of the trained neuron is not a combination of its children (R^2 .52); (b) the moved neuron
  carries only .13 of the message in the readers' metric vs .17 free and .27 unmoved: the readers read the children
  UNMOVED, so moving helps the HOW (one ReLU) but costs in the WHAT. A real trade-off inside the theory.
- Budget: 6 of 7 runs used. STOP here and report (direction change after 3 failed cards).

## T4b: does Theorem 10b beat simple rules? (Shonu's question, 2026-10-23; no image run, cached vectors only)
- Target: the two trained lam vectors on moved children, as printed in `logs/t4_moved.log`.
- Rules, fixed before the numbers: uniform; drop-3 (the 3 lowest diag(A^T A)) with the rest equal; drop-3 with the
  rest by mean activity; diag(Phi); lam_I and lam_F from T2; free lam (T4); constrained lam (Theorem 10b, T4).
- Scores: Pearson on all 10, cosine on all 10, Pearson on the support of the constrained lam (members with lam > .01).
- PREDICTIONS (written first):
  U1 drop-3-equal scores >= .70 (Pearson, all 10) on both twins, so the .82 is mostly "which three to drop".
  U2 within the support, the constrained lam vs trained is >= .5 on both twins (the theory's extra weighting is real).
  U3 se of a 10-point correlation near .8 is ~.15, so no rule separates from another by more than 2 se on all 10.
- RESULT T4b (`t4b_rules.py`, `logs/t4b_rules.log`). Drop-3 picked members [2, 8, 7] (lowest diag A^T A).
| rule | twin L r10 | twin L cos | twin L r-sup | twin R r10 | twin R cos | twin R r-sup |
|---|---|---|---|---|---|---|
| uniform | - | .93 | - | - | .85 | - |
| drop-3 equal | .64 | .91 | .90 | .55 | .87 | .46 |
| drop-3, rest by mean | .73 | .92 | .94 | .65 | .90 | .60 |
| diag(Phi) | .57 | .94 | .68 | .39 | .86 | .45 |
| lam_I (T2) | .57 | .91 | .75 | .51 | .87 | .79 |
| lam_F (T2) | .50 | .88 | .11 | .50 | .85 | .49 |
| free lam (T4) | .63 | .29 | .42 | .64 | .39 | .49 |
| Theorem 10b (T4) | .82 | .85 | .68 | .81 | .89 | .86 |
| twin vs twin | .78 | .94 | .75 | | | |
- U1 FAILED (drop-3 equal .64 / .55). U2 HOLDS (.68 / .86). U3 HOLDS: r = .82 on 10 points has 95% interval .39 - .96,
  so the theory's lead over the best simple rule (.09 - .16) is under 1 se.
- Plain reading: the theory is first on the headline score, but not by a margin this instrument can see. On cosine,
  uniform is as good as the theory (the trained neuron is close to an even mix). Shonu's doubt is right at this level.
- Retraction: the T4 sentence "as well as two trained twins agree" overclaims precision. Kept the numbers, dropped the claim.
- Next (needs the 7th run): the functional test, Theorem 10b read vs uniform read, same Fisher write, paired KL on
  dev-5000. That se is ~.006, so it can separate rules that the 10-point correlation cannot.

# BEAT (2026-10-23 afternoon). Plan: `~/.claude/plans/okay-i-d-like-you-enumerated-wreath.md`. Paper: PAPER.md Part III.
Goal: a formula that beats the BEST KNOWN trained k = 3 (bar B, sealed .037; Shonu's choice). Random-start k = 3
(.122) is reported next to it, never the headline. Every prediction names simple-rule baselines and a 2-se margin.

## I1-I2 (saved models only, no training). What did training make at k = 3?
- Models: R = L-k3-r-lr3e-2 (random start, stuck, sealed .122); M = L-k3-m-lr3e-2 (start = members 330/388/385 at
  10/120/250 deg, sealed .037). Instrument check first: the k = 1 twin L-k1-r-lr3e-2 on the T4 images must give
  R^2 unmoved ~.42, moved ~.51 (t4_moved.log).
- Regression of each unit's pooled output on the 10 pooled children (+ intercept), 500 train images (T4 set).
- PREDICTIONS (written first):
  I1a R's three units are near copies: every pairwise corr of pooled outputs > .8 (invariant-manifold proposition:
      they fell onto the duplicate set). Null: three units of M (expected pairwise corr < .3).
  I1b each R unit's lam correlates > .7 with the k = 1 trained lam (they are the loop neuron).
  I2a each M unit's lam is a broad bump around its start member: corr(lam_j, 1 + cos(theta - phi_j)) > .7, and it
      beats both nulls (one-hot start member; start member + its two neighbours equal) by > .1 on all three units.
  I2b M units are combinations of children far better than the k = 1 neuron: R^2 > .7 (k = 1: .42-.52).
- RESULT I1-I2 (`u1_k3.py`, `logs/u1_k3.log`, 3 s). Instrument reproduces T4 (k = 1 twin R^2 unmoved .42, moved .52).
| | prediction | result | verdict |
|---|---|---|---|
| I1a | R's 3 units are near copies | R has TWO DEAD units (on at 0.0% of positions) and one live unit (on 98%) | FAILED, for a sharper reason |
| I1b | R's live unit ~ k = 1 lam | corr .58 with the k = 1 lam | weak |
| I2a | M units are 1 + cos bumps, beat both nulls by .1 | corr bump .82 / .82 / .87; member+nbrs .94 / .68 / .63; one-hot .70 / .74 / .64 | FAILED as written |
| I2b | M units R^2 > .7 on children | .81 / .80 / .75 UNMOVED (moved only .38-.61) | HOLDS |
- What the M units really are: a PARTITION of the circle into three contiguous arcs, each unit a positive mix of its
  arc's members, about 0 elsewhere. Unit 0: 320-50 deg; unit 1: 70-180 deg; unit 2: 170-320 deg. On 55-62% of pooled
  positions, pairwise output corr -.18 to -.01. They read the children UNMOVED (at k = 3 no centre-of-curvature move).
- Meaning of I1: bar A (random start, .122) is not a k = 3 network. It is a k = 1 network with two dead units. A dead
  ReLU has zero gradient in its read AND its write, so it stays dead forever (exact; PAPER section 12). That is the
  whole reason random-start k = 3 sits at the k = 1 level (.1217 vs .1205).

## I3 (saved models, 200 train images): is the trained read in the MEMBER SPAN?
- Regress each unit's 28 x 28 PRE-activation on the 10 members' pre-activations (+ constant). If R^2 is high, the
  unit's read is exactly realisable from the large network alone: kernel sum_i w_i K_i, bias sum_i w_i b_i + b0.
- RESULT (`u2_prespan.py`, `logs/u2_prespan.log`): k = 3 M units: R^2 .77 / .85 / .78 unmoved (moved .35-.55).
  k = 1 twins: .07 / .15 unmoved, .25 moved. So the k = 3 reads are mostly in the member span; the k = 1 loop
  neuron is NOT (it is a new neuron, as the essay says). Arc weights again, plus a POSITIVE offset b0 (63-162 per unit
  of |w|): each unit is on at 21-27% of positions, about twice a member's rate.
- Card I4 (microscope, written first): M with its read replaced by the member-span projection (w, b0 from I3),
  trained write kept. PREDICTION: dev-5000 KL within .01 of M itself.
- RESULT I4 (`u3_span.py`, 22 s): M trained .0361 (se .0017) | member-span read + trained write .0807 (.0044) |
  paired +.0446 (se .0036). FAILED. The 15-25% of the trained read outside the member span carries .045 of KL,
  about as much as the whole read gap measured on 2026-10-22 (members + trained write, .076 sealed).
  Meaning: Shonu's "read from the large network only" fails here in the same way as C5 at k = 1. The best ReLU read
  of a sum of member ReLUs uses input directions outside the members' kernels, because the inputs are non-negative
  and sparse (no Bussgang). That needs the mix's gated input moments (Theorem 5 says the weights cannot hold them).

## Paper: PAPER.md Part III sections 12-13 written (Theorem 12 gap split; Theorem 13 dead units stay dead, PROVED;
Statement 13.1: the WHAT theorem fixes only the span, M's arcs are a basis chosen by something else). Plan step 3
(random-start sweep) is VOID: .122 is a k = 1 network with two dead units.

## I4b (instrument fix of I4) and I5 (is M's partition intrinsic or inherited from its start?)
- I4b: refit the member-span projection on M's ON-set only (positions where the unit's pre-activation > 0).
  PREDICTION: paired cost vs M stays >= .03 (then "read from the large network only fails at k = 3" stands).
- I5 (microscope, TRAINED): train k = 3 from members 340/343/406 (50/170/290 deg), same settings as M, 1000 steps.
  PREDICTION: arcs re-form around the NEW starts (each unit's largest weight within one member of its start), and dev
  KL at step 1000 within .02 of M's run at step 1000 (L-k3-m-lr3e-2.log). If yes, the partition is a free choice.
- Note on I5: its cosine schedule spans 1000 steps; M's spanned 3000 (M at step 1000: dev-1000 .068). So I5 at step
  1000 has a fully decayed lr; compare loosely (the prediction's .02 band allows for it).
- RESULT I4b (`u3b_span_on.py`, 22 s): on-set fit: .0697 (se .0037) vs M .0361, paired +.0337 (se .0030). The
  prediction (>= .03) HOLDS, narrowly. Statement stands: at k = 3 a read made only of the members' kernels loses
  ~.034 of KL; the trained read uses input directions outside the members' span.

## W3 (card from 2026-10-22, first run): the derived write for k = 3, on M's TRAINED read (microscope)
- Builds on M's read: trained write | lsq (simple rule: identity-metric least squares) | step0 plain Fisher write |
  step1 one active-set step (W5 generalised to k units, pre-registered as THE build). 12000 train images.
- PREDICTIONS (written first): W3a step1 beats lsq by > 2 paired se. W3b step1 within .015 of the trained write
  (.0361 dev-5000), i.e. the write is closed-form at k = 3 too. If W3b fails, the k = 3 write has a gap k = 1 did not.
- Run size: 4000 images (smoke test: ~0.35 s per image for the two passes, so ~25 min; 12000 would be ~70 min).

## U5 (no images; cached Phi, Sigma of the UNMOVED pooled children): the WHAT at k = 3, second-order shares
- No prediction was needed for the ordering of known builds; the new number is the arc partitions.
- RESULT (`u5_what.py`, `logs/u5_what.log`, 1 s). Shares of the family's message (beta carries .260 on top):
| WHAT (3 units) | share | label |
|---|---|---|
| Eckart-Young ceiling (any sign) | .563 | MIX-DERIVED bound |
| 1 + cos bumps (any rotation) | .532 | MIX-DERIVED |
| M's trained Lam (regression) | .512 | TRAINED (read-off) |
| best contiguous 3-arc partition, optimal weights >= 0 | .508 | MIX-DERIVED |
| M's start cells (320-50 / 70-180 / 250-290) | .480 | |
| best kept triple / kept 10-120-250 | .339 / .303 | simple rule |
- Meaning: the WHAT barely separates bumps, arcs and M (.51-.53, near the .56 ceiling). It separates all of them
  from kept members (.30-.34) by a lot. That matches the measured story: kept members + trained write .076 vs M .036.
  So at k = 3 the WHAT is essentially settled by any broad basis. The choice between bases, and the remaining KL, is
  decided by the HOW (Statement 13.1, candidate 1) and the write.

## H (PAPER section 14): the HOW read for each WHAT candidate (gated least squares on S_j = {t_j > 0})
- Candidates: kept members (E2 check), 1 + cos bumps, M's start cells, the top 4 arc partitions of U5. Fit on 2000
  train images; realized share on 500 others; dev-5000 KL under the lsq simple-rule write (1000 images), paired against
  M's trained read under the same write (smoke: .0798).
- PREDICTIONS (written first):
  H0 (instrument, Prop. 14.2) kept members come back exactly: kernel cos > .999.
  H1 (Prop. 14.3) the HOW reverses the WHAT order: the best arc partition's REALIZED share beats the bumps' realized
      share, although the bumps' WHAT share is higher (.532 vs .508).
  H2 the best candidate's read is within .015 of M's trained read under the lsq write, and beats kept members under
      the same write by > 2 paired se.
- RESULT I5 (`u7_i5.py`, `logs/u7_i5.log`): from start 50/170/290 the arcs re-form around the NEW starts (each unit's
  largest weight = its start member; R^2 .74-.87). Dev-1000 at step 1000 .048 (M at step 1000: .068 with a slower
  schedule; M's final .042). HOLDS: the partition is inherited from the start. Which partition is a free choice,
  matching U5's flat top (many partitions within .01 of the best share).
- RESULT W3 (`u4_w3.py 4000`, `logs/w3_4000.log`, ~25 min). Writes on M's trained read, dev-5000:
| write | KL (se) | paired vs trained write |
|---|---|---|
| trained (M) | .0361 (.0017) | 0 |
| lsq simple rule | .0836 (.0043) | +.0475 (.0036) |
| step0 plain Fisher | .0750 (.0038) | +.0389 (.0030) |
| step1 active set (THE build) | .0682 (.0033) | +.0321 (.0025) |
  W3a HOLDS (step1 - lsq -.0154, se .0039). W3b FAILED: the derived write is .032 short of the trained write on the same
  read (at k = 1 the gap was .013). Woken share under step0 4.8%. Only 4000 images for 1472 unknowns (data hunger at
  k = 1 was ~.017 between 12000 and 35000), so part of this may be estimation, not theory. Open.
- RESULT H (`u6_how.py`, `logs/u6_how.log`, ~3 min). H0 HOLDS: kept members come back exactly (kernel cos 1.0000).
| read (lsq write on all) | realized share | false-on | dev-5000 KL | paired vs M's trained read |
|---|---|---|---|---|
| M trained read | .423 | | .0784 | 0 |
| arcs #2 (MIX-DERIVED) | .438 | .26 | .0748 | -.0035 (.0022) |
| arcs #1 | .437 | .25 | .0805 | +.0022 (.0030) |
| arcs #4 / #3 | .429 / .431 | .30 | .0800 / .0837 | +.0016 / +.0053 |
| M start cells | .426 | .22 | .0817 | +.0033 (.0023) |
| 1 + cos bumps | .410 | .70 | .0859 | +.0075 (.0023) |
| kept 10/120/250 | .317 | 0 | .1325 | +.0542 (.0046) |
  H1 HOLDS: the HOW reverses the WHAT order (bumps WHAT .532 > arcs .508, but realized .410 < .437; bumps fire falsely
  on 70% of their off-target positions, as Prop. 14.3 and the named failure said). H2 HOLDS: the derived arc reads
  equal M's trained read under the same write (all four within .006; the share-rule pick #2 is -.0035, 1.6 se) and
  beat kept members by .05-.06 (> 10 se).
- MEANING: at k = 3 the READ is now closed-form (MIX-DERIVED, gated least squares, no knob) and as good as training's.
  The whole remaining gap to bar B is the WRITE (.032 on M's read), as at k = 1.

## W3b diagnosis (W2e style; `u8_pred.py`): estimation or beyond second order?
- For the writes on M's read: real KL, plain second-order prediction, and the active-set model's own objective
  (4a ReLUs exact, post-ReLU Fisher), on 500 of the solve's own images and on 500 dev images.
- PREDICTIONS (written first): P1 in-sample, step1's post-ReLU objective is BELOW the trained write's (the solve wins
  inside its own model, as in W2e). P2 on dev, step1's post objective rises by > 30% over in-sample (over-fit: 4000
  images for 1472 unknowns), while the trained write's changes < 15%. If P1 and P2 hold, the gap is mainly estimation.
  If P1 fails, the model's optimum itself is worse than training's write: a theory gap.
- RESULT (`logs/u8_pred.log`, ~4 min):
| images | build | real | pred pre-ReLU | pred post-ReLU (active-set model) | real / post |
|---|---|---|---|---|---|
| in-sample 500 | trained write | .0211 | .0242 | .0222 | .95 |
| in-sample 500 | step1 (THE build) | **.0208** | .0190 | .0114 | 1.82 |
| dev 500 | trained write | .0406 | .0440 | .0423 | .96 |
| dev 500 | step1 | .0642 | .0741 | .0707 | .91 |
  P1 HOLDS: on its own images the derived write beats the trained write in its model (.0114 vs .0222) AND in real KL
  (.0208 vs .0211). P2 as worded FAILED (the trained write also rises, x1.9, train -> dev), but step1 rises x3.1 in
  real KL and x6 in its own objective. And on dev the active-set model is accurate for every write (real/post .91-.98).
- MEANING: the k = 3 write gap is ESTIMATION, not theory. The second-order active-set model is right on new images; the
  write fitted to 4000 images over-fits them. Defined on population moments (the rule), the build is fine. To test it
  against bar B we need the population moments better estimated: many more images per unknown.

## W4: the same write at population scale (`u9_write.py 35000`), and THE FORMULA end to end
- Estimator change only (no change of the build): one backward per image of sum_r eps_r sqrt(p_r) dlog p_r (eps = +-1)
  replaces the 10 per-class backwards. E[outer product] = the same top-10 Fisher sum, so the same population moments,
  ~10x cheaper. All 35000 train images. Three reads share the gradients: M's trained read, the derived arcs #2 read
  (section 14), kept members 330/388/385 (simple rule). step0 = plain Fisher, step1 = one active-set step (THE build).
- THE FORMULA = arcs #2 read + step1 write: no training anywhere. Labels: read MIX-DERIVED, write MIX-DERIVED.
- PREDICTIONS (written first):
  W4a M read + step1 is within .01 of bar B (.0361 dev-5000): with enough images the write is closed-form at k = 3.
  W4b THE FORMULA lands at <= .05 dev-5000, and does NOT beat bar B (gap .005 - .015).
  W4c kept members + step1 is worse than THE FORMULA by > 2 paired se (simple-rule bar).
- RESULT W4 (`u9_write.py 35000`, `logs/u9_35000.log`, ~5 min with the cheap estimator). Dev-5000, paired vs bar B:
| read + write | KL (se) | paired vs bar B (se) |
|---|---|---|
| M read + trained write (bar B) | .0361 (.0017) | 0 |
| M read + step0 | .0514 (.0024) | +.0153 (.0016) |
| M read + step1 | .0443 (.0021) | +.0083 (.0012) |
| arcs read + step0 | .0563 (.0027) | +.0202 (.0024) |
| arcs read + step1 (THE FORMULA) | .0507 (.0025) | +.0147 (.0022) |
| kept read + step0 | .1124 (.0054) | +.0763 (.0050) |
| kept read + step1 | .0934 (.0045) | +.0574 (.0041) |
  W4a HOLDS (+.0083 < .01). W4b FAILED as written: .0507 > .05 by .0007 (well inside one se). Its second half holds:
  it does not beat bar B, and the gap .0147 is inside the predicted .005-.015 band.
  W4c HOLDS (.0934 vs .0507). Data hunger was real: 4000 -> 35000 images took M read + step1 from .0682 to .0443.

# DATA-FREE (2026-10-23 evening, plan `~/.claude/plans/i-want-you-to-majestic-milner.md`)
Decision change (Shonu): the GENERALIST build uses NO image data, in the read, the write or the mix. Images only
CHECK statements. Developer capabilities may still use images later. This reverses the morning MIX decision for the
generalist. Every Fisher / least-squares / gated-lsq build above is FITTED for the headline (incl. THE FORMULA .0507).
Labels: DERIVED (weights + file constants) / LAW-DERIVED (weights + a natural-image law, needs Shonu's OK) / FITTED /
TRAINED. Network: InceptionV1 only. Fallback order: weights (conjectures A-D), then inputs the network makes itself
(E), then a natural-image law only with Shonu's OK.
File inventory (checked): 58 weight + 58 bias tensors, architecture constants (LRN size 9, alpha 1e-4, beta .5, k 1),
input offset 117 (= the ImageNet mean pixel, the ONE stored plain training average). No BatchNorm statistics.

## D1 (card, written before running): the strict baseline, zero images (`d1_strict.py`)
- Correction to the plan before running: Theorem 1 balance cannot set the write gain. Scaling the read by a and the
  write by 1/a leaves the function unchanged (relu is positively homogeneous). Balance fixes only that gauge. The
  product (read scale x write scale) is set by Prop. 4: c = sum of moved members, B = sum of member columns. So the
  strict build has NO free number. The balance ratio of the coarse unit is printed only as a consistency check.
- Builds (all DERIVED, weights only): moved = C1 shifts, consensus bias sum b_i, write = sum of member columns;
  unmoved = same with zero shifts (simple rule); removed = family deleted (k = 0, trivial).
- PREDICTIONS: D1a moved lands in .50 - .62 dev-5000 (the consensus unit is on ~0.7% of positions, so it is close to
  removal). D1b moved beats removal by more than 2 paired se. D1c moved beats unmoved by more than 2 paired se.
- RESULT D1 (`d1_strict.py`, `logs/d1_strict.log`, ~1 min). Dev-5000: removed .6642 (se .0254) | unmoved .6642 (never
  fires: on-rate 0.0000) | moved .7542 (.0317), on 0.57% of positions. Paired moved - removed +.0900 (se .0242).
  D1a FAILED (above the .50-.62 band), D1b FAILED (worse than removal, not better), D1c FAILED.
  Side fact (weights only): the moved coarse unit is balanced like a trained neuron (in^2/out^2 1.094; members
  1.0-1.17); the unmoved sum is not (.753). So the C1 shifts make the unit consistent with Theorem 1.
  WHY it failed (paper section 16): where the unit fires it writes the SUM of 10 columns times the sum of 10 moved
  pre-activations, so the readers get cross-talk W_i x member j for every pair. The WHAT (Theorem 10) says the write
  is the activity-weighted MEAN of the columns, not the sum. That needs the members' activity shares: conjecture A.

## A (card, written before running): do the weights carry each neuron's activity share? (no new images)
- Theorem 2 corollary 1: lam ||W_.i||^2 = -E[sum e_i f_i] (error-weighted importance). Conjecture A (Shonu's idea in
  testable form): across neurons, this tracks the PLAIN activity. Plain activity from the cache (2000 train images,
  post-ReLU, `mean3b.npy`, `var3b.npy`): E[f^2] = var + mean^2, and E[f].
- Two readings, both weights-only if true: A1 (share) E[f_i^2] proportional to ||W_.i||^2 (log-log slope ~1).
  A2 (equal energy) E[f_i^2] about the same for all neurons in the balanced gauge (slope ~0); then the share of the
  downstream message is E[f^2] ||W_.i||^2, proportional to ||W_.i||^2 anyway.
- PREDICTION: Spearman(||W_.i||^2, E[f_i^2] ||W_.i||^2) >= .7 over the 480 neurons of 3b (the downstream energy share is
  readable from weights). Within the 10 members, the same Spearman >= .6. Simple-rule baseline: equal shares (rank
  correlation 0 by construction); the check is whether weights beat "all equal".
- RESULT A (no new images, 1 s). Spearman(||W||^2, downstream energy E[f^2]||W||^2): all 3b +.29, curves +.50; plain
  Spearman(||W||^2, E[f^2]) -.02 (all 3b), slopes -.79 to +.70 by branch. PREDICTION FAILED (>= .7 / >= .6).
  Neither reading holds in general: A1 slope ~0 overall; A2 fails (CV of E[f^2] .68). So the weight norm (= the
  error-weighted importance) does not give the plain activity share. Inside the curve family both are near uniform:
  ||W||^2 shares .092-.110, true downstream-energy shares .062-.128 (CV of E[f^2] .17). For this orbit, "equal shares"
  is what the weights say, and it is close to true.

## B (card, written before running): does kernel overlap give co-firing? (no new images; cached t4 Szz)
- Corollary 2: <K_i, T_s K_j> = error-weighted co-activation. Conjecture B: the max over shifts |s| <= 2 of the kernel
  cosine ranks member pairs like their plain pooled correlation (from Szz, 45 pairs).
- PREDICTION: Spearman >= .6 over the 45 pairs. Simple-rule baseline: angle difference between members' preferred
  orientations (cos of the angle gap), also weights-visible.
- RESULT B (no new images, 2 s). Spearman(max kernel cosine over shifts, plain pooled corr), 45 pairs: shift 0 +.44,
  <= 1 +.72, <= 2 +.76 (the pre-registered one). Baseline cos(angle gap) +.48. HOLDS, and beats the simple rule.
  Meaning: the weights carry WHO fires with whom (the rank pattern of co-firing), but conjecture A says they do not
  carry HOW OFTEN each fires. Rank structure transfers from error-weighted to plain; magnitudes do not.

## E (card, written before running): inputs the network makes for itself (`e1_selfmade.py`, `e2_compare.py`)
- Generation rule (stated once, never tuned): one input per class, classes spread evenly over 1000; maximize that
  class's logit on a PLAIN pixel image (no 1/f law, no ImageNet colour matrix), lucent's standard transforms, Adam lr
  .05, 256 steps, lucid's feature-vis ReLU. 256 inputs (smallest test; ~20 min). Label: DERIVED (weights + rule).
- Compared with the generalist images (FITTED references): arcs #2 read of section 14 (cached real read, u6_how.pt),
  the children statistics (t4 Szz), and the lsq simple-rule write (1000 real images, as in H).
- PREDICTIONS:
  E1 (statistics) the pooled children correlation pattern on self-made inputs matches the real one: Pearson over the
     45 pairs >= .6; every member's on-rate within a factor 2 of its real on-rate.
  E2 (read) Lemma 14.1 on self-made inputs gives a read whose kernel cosine with the real-image read is >= .8 per unit.
  E3 (read swap, same real lsq write) self read - real read, dev-5000 paired: <= +.010.
  E4 (fully self-made: read AND lsq write from self-made inputs) dev-5000 within +.02 of the same pipeline on real
     images (.0748 in H), and far below removal (.664) and D1 (.754).
- Conjecture C (readers' Gram as the window statistics) is deferred: conjecture A shows error-weighted magnitudes do
  not give plain ones, and a rank-96 Gram cannot stand in for an 801-dim window moment. E tests the same need directly.
- RESULT E1 generation (`logs/e1_self_256.log`, 15 min): the network gives its own class to 94-97% of the self-made
  inputs (top-1 on the intended class).
- RESULT E (`e2_compare.py 256`, `logs/e2_compare_256.log`, ~4 min). Self-made (256, DERIVED) vs real (FITTED):
| check | prediction | result | verdict |
|---|---|---|---|
| E1 children corr pattern (45 pairs) | Pearson >= .6 | .951 | HOLDS |
| E1 member on-rates | within x2 | ratios .89 - 1.35 | HOLDS |
| E2 read kernel cos per unit | >= .8 | .916 / .875 / .900 | HOLDS |
| E3 self read vs real read, same real lsq write | <= +.010 | +.0074 (se .0013) | HOLDS |
| E4 self read + self lsq write vs real read + real lsq | within +.02 | .0696 vs .0760: -.0064 (se .0018) | HOLDS, and BETTER |
  Caveats named now: one generation rule, 256 inputs, dev images (no cold test yet); the real lsq write used 1000
  images, the self one 256; the read bias of unit 1 differs (63 vs 10). E4 being better than real is not explained yet.
- Deviations from the card, named: (1) the plan named per-neuron feature visualizations; the run used CLASS
  inversions (one per class). Both read only the weights; class inversions were chosen because the classes are the
  training labels and the biases suggest even class shares. (2) The rule has chosen numbers (256 steps vs lucent's
  default 512, lr .05, plain pixels, standard transforms). None was tuned on images, but they are a fixed rule, not a
  theorem. Label for Shonu to decide: "DERIVED if a fixed published generation rule counts as part of the formula".
## E5 (control, card written before running): matched real build (read AND lsq write from the SAME 256 real images)
- PREDICTION: matched-real lands within 2 paired se of the self build (.0696). Then the finding is "self-made equals
  real at equal count", and the E4 advantage was same-set consistency. If matched-real stays near .0760, the self-made
  advantage is real and needs a statement.
- Target check: 255 of the 256 class targets are real ImageNet classes (one is an extra lucid output). The targets are
  evenly spaced logit indices, not evenly spaced ImageNet class numbers. Still one input per class.
- RESULT E5 (`e5_matched.py 256`, `logs/e5_matched_256.log`): matched real (read + lsq write from the same 256 real
  images) .0740 (se .0040) vs self .0696: paired +.0044 (se .0017). PREDICTION FAILED narrowly (2.6 se, not < 2), in
  the self-made direction. So about a third of E4's .0064 was count/consistency; at equal count the self-made inputs
  are at least as good as real ones for this pipeline (slightly better, 2.6 se).

## E6 (card, written before running): the Fisher write (step0, step1) from self-made vs real inputs, equal count 256
- Each side uses its own read (E5 builds: self read from self inputs, real read from the same 256 real images) and
  fits step0/step1 (u9 estimator) on its own 256 inputs. Dev-5000 paired.
- PREDICTION E6a: self step1 within 2 paired se of real step1. E6b (named, not a test of the source): with 1472
  unknowns and 256 inputs both writes over-fit (4000 images already did), so step1 may lose to the lsq write here.
- RESULT E6 (`e6_fisher.py 256`, `logs/e6_fisher_256.log`, ~6 min). Dev-5000:
| write (256 inputs, own read) | self-made | real | paired self - real |
|---|---|---|---|
| lsq | .0696 | .0740 | -.0044 (.0017) |
| step0 Fisher | .5497 | .1884 | +.3613 (.0177) |
| step1 active set | .5239 | .1740 | +.3499 (.0174) |
  E6a FAILED badly. E6b HOLDS (at 256 inputs both Fisher writes over-fit and lose to lsq).
  WHY (checked, `logs` inline): the network is certain on its own inputs. Mean max-prob .997 (real .910); mean
  1 - sum p^2 (the trace scale of the output Fisher) .0049 vs .125 real: 25x smaller. The Fisher write learns only from
  inputs where the network is unsure, and class inversions are the images it explains perfectly. This is Theorem 5
  itself: self-labeled, perfectly explained inputs carry no output-metric information.
- Budget note: D1, A, B, E, E5, E6 = 6 checks. Stop here and report (plan budget ~6 + cold test). No cold test spent:
  the write question is open, and the cold test is reserved for a build with the final write.
- Retraction: the plan predicted the read would be the hard part without data and pushed back on Shonu's view. E1-E3
  show he was right about the read. The write (output metric) is the part that fails.
- Not done (named): k = 1 never tested data-free; no `phi.py` yet (the build lives in `e2_compare.py` / `e5_matched.py`,
  which import the data loader for scoring; split build from score once the rule is accepted); sealed cold test unspent.
- Two open items, kept apart: (a) self-made inputs carry no output-metric information (cause known; boundary
  inversions proposed, NOT run: a second generation rule is Shonu's call). (b) Cost: the Fisher write needed 35000 real
  images; generation runs ~3.6 s per input (~35 h for 35000). Any data-free Fisher write needs ~100x cheaper
  generation or a lower-variance write. That is engineering, not theory.

# GENERATE (2026-10-23 night; plan `~/.claude/plans/i-want-you-to-majestic-milner.md`, rewritten)
Two problems (Shonu): P1 = a formula that makes inputs from any trained network; P2 = the coarse network from those
inputs. A density from the network + a standard sampler counts, if doubling the sampler's steps changes no reported
number by > 2 paired se (else CHOSEN). Theory: PAPER.md Part V (Theorem 19, Corollary 19.1).

## I1 (card, written before running): is the network's confidence honest? (cached teacher logits, train split)
- For a calibrated network, E[p(true class)] = E[sum_y p_y^2] and accuracy = mean confidence in every confidence bin.
- PREDICTION: InceptionV1 is mildly over-confident on these held-out images: ECE between .02 and .08, and mean
  confidence above accuracy. Then part 1 of Theorem 19 holds to a few percent.
- RESULT I1 (`i1_calib.py`, `logs/i1_calib.log`). 35000 train-split images (held out from the network's training):
  top-1 .656, mean confidence .912, E[p true] .648 vs E[sum p^2] .876, ECE .256. Confidence > .93 on 72% of images,
  accuracy there .787. The label map is clean (1000 unique targets; only 3 classes under 5% accuracy; ECE without
  them .256). PREDICTION FAILED (.02-.08): the network is strongly over-confident on unseen images.
  Meaning: part 1 of Theorem 19 (network conditional = data conditional) is only rough here. It does not touch P2:
  the coarse network's objective is KL to the NETWORK's answers, so the network's own conditional is the target.
- Correction to the plan's Corollary 19.1 (found while writing the proof, before any run): sampling x | y from
  p_theta(y|x) q(x) gives inputs whose marginal is q (when the class shares match q), so their uncertainty is the
  network's uncertainty UNDER q. It is not the uncertainty on real images "for any q". The true part: maximizing gives
  certainty for EVERY q. So sampling is necessary, and q decides the rest.

## I2 + I3 (card, written before running): does the Fisher write need unsure inputs, and how many? (`i2_share.py`)
- Write = step1 active-set Fisher (u9 estimator) on M's trained read (isolates the write; bar B = M's own write
  .0361). Subsets of the train split, chosen by the network's own u = 1 - sum p^2 (no labels):
  nat-N = random N; low-4000 = random from u below the median; high-N = random from u > .1 (30% of images, 10500).
- PREDICTIONS:
  I2a high-4000 beats nat-4000 (paired, > 2 se): the output metric lives on unsure inputs.
  I2b low-4000 is worse than nat-4000 by > .02.
  I3 high-10500 lands within .005 of the 35000 natural write (.0443): unsure inputs carry the Fisher mass, so ~3x fewer
     inputs suffice. Reference counts: nat-4000, nat-12000, high-1000.
- RESULT I2 + I3 (`i2_share.py`, `logs/i2_share.log`, ~8 min). Step1 write on M's read, dev-5000:
| subset | KL (se) | paired vs nat-4000 |
|---|---|---|
| nat-4000 | .1710 (.0083) | 0 |
| low-4000 (u below median) | .2295 (.0100) | +.0585 (.0089) |
| high-4000 (u > .1) | .0698 (.0033) | -.1012 (.0073) |
| high-1000 | 1.169 (.040) | too few: the solve blows up |
| high-10500 (all unsure) | .0454 (.0021) | -.1256; vs nat-35000 +.0010 (se .0012); vs bar B +.0093 |
| nat-12000 | .0633 (.0030) | -.1077; vs nat-35000 +.0189 (.0023) |
  I2a HOLDS, I2b HOLDS, I3 HOLDS. The 70% of images where the network is sure add nothing to the Fisher write: the 10498
  unsure images alone equal all 35000. (nat-4000 here, .171, is worse than W3's .0682 at 4000 because the cheap +-1
  estimator is noisier at small counts; all rows here share it.)

## I4 (card, written before running): what does the network say about p(x)? (`i4_prior.py`, 256 images as checks)
- (a) The logit level LSE(x) = log sum_y exp(logit_y(x)) is the one number softmax ignores ("your classifier is an
  energy model"). PREDICTION: mean LSE orders real > pixel-shuffled real > uniform noise, and class inversions sit
  ABOVE real (maximizing drives the logits up). If so, the network's own energy ranks natural above unnatural, but
  sampling exp(LSE) at temperature 1 would drift to inversion-like, sure inputs (bad for the write, Prop. 19.2).
- (b) Theorem 2 at the first layer: conv1 kernels are error-weighted pixel-patch moments. PREDICTION: the conv1
  filters' energy along the eigenvectors of the REAL 7x7x3 patch covariance follows a power law in the eigenvalue
  (|log-log slope| > .3, Spearman > .7). Then conv1 carries the pixel spectrum, a candidate prior from the weights.
- RESULT I4 (`i4_prior.py`, `logs/i4_prior.log`). (a) mean LSE: real 55.4 (sd 17.9) > shuffled 25.8 > noise 17.6;
  self-made inversions 491 (sd 119). 1 - sum p^2: real .122, shuffled .230, noise .524, self-made .005. (a) HOLDS: the
  logit level ranks natural above unnatural, but it is unbounded above, so exp(LSE) at temperature 1 would be ruled by
  inversion-like inputs ~9x the real level. Not a usable prior as it stands.
  (b) conv1 energy along the real patch eigenvectors: slope -.04, Spearman .20; cos(K^T K, C^a) best at a = 0 (.30).
  FAILED: conv1 spreads its energy almost evenly over pixel directions. It does not carry the pixel spectrum.

## I5 (card, written before running): a toy where the true input distribution is known (`toy/`)
- Truth: 32x32 grey images of 4 shape classes (disc, square, triangle, ring) with random size, position, brightness,
  background level and pixel noise. A small CNN (conv 16, conv 32, linear) trained full-batch on N = 1000 samples with
  weight decay until near-stationary (Theorem 2's premise; checked by the residual of grad + lam theta).
- Generators (no truth used): G0 class inversion (maximize the logit; rule of check E), G1 sampling p(y|x) q(x) with a
  flat q (Langevin; Lemma 19.1a), G2 conjecture G (inputs + labels + weights c_s >= 0 whose gradients reproduce
  -lam theta). Reference: fresh samples from the truth.
- P2 proxy: delete half of the conv2 channels and refit the linear layer's compensation (the "write") from generated
  inputs: lsq (input statistics only) and Fisher (output metric). Score KL(intact || repaired) on fresh true samples.
- PREDICTIONS: I5a (instrument) G2 recovers some actual training images (>= 10% of its outputs have a training
  image with pixel corr > .8). I5b G0's inputs are sure (mean 1 - sum p^2 < .1 x truth); G2's are not (>= .5 x truth).
  I5c Fisher write fit on G2 inputs is within 2 paired se of the same write fit on true samples (equal count), and
  beats G0 and G1 by > 2 paired se.
- RESULT I5 (`toy/i5_run.py`, `logs/i5_run_x1.log`, `logs/i5_run_g2b.log`). Toy network: train acc 1.00, fresh acc
  .838, 1 - sum p^2 fresh .122, balance .996 / 1.000, stationarity residual of the TRUE train set .241 (the floor).
  P2 proxy (delete conv2 channels 24-31; no repair KL 10.2), KL on 5000 fresh true samples, 1000 inputs per fit:
| inputs | 1 - sum p^2 | co-firing corr | lsq write | Fisher write |
|---|---|---|---|---|
| truth (fresh) | .116 | 1 | .0341 | .0159 |
| train set | .015 | 1.000 | .0341 | .0162 |
| G0 inversion | .000 | .664 | 2.10 | 10.2 (= no repair: zero metric) |
| G1 Langevin, flat prior | .187 | .598 | 1.08 | .160 |
| G2 stationary (not converged, residual 138) | .069 | .440 | .451 | .587 |
| G2 warm-started from G1, refine x | .008 | .597 | 2.41 | .775 |
| G2 from G1, reweight only (ESS 75) | .187 | .598 | 1.12 | .478 |
  I5a FAILED (no training image recovered; G2 never reached the floor). I5b: G0 sure (HOLDS), G2 .069 = .6 x truth.
  I5c FAILED: every generator is far from truth (best Fisher: G1, 10x truth).
  SLIP (named): after G2 failed I ran four optimizer tweaks in a row (lr; start weights near 0 -> collapsed to c = 0;
  weights summing to 1 -> collapsed to sure inputs, residual 1.0; warm start from G1 -> again residual 1.0). Shonu:
  "errors means you have to figure out the mathematical reason why you are wrong". The reason is now in PAPER.md
  section 21 (the plateau theorem); the next run follows from section 22.
- RESULT X1 (toy, inline): E_train[f_y - E_p f] = .0331 (se .0018) vs kappa from weights .0331: HOLDS (ratio 1.000).
  Fresh samples: -.267. X2 NOT RUN: Corollary 22.2 proves the first-order tilt changes density by at most K = 4, so
  with a flat prior it must fail. Next object (from the math): the full max-ent family's multipliers, layer by layer
  from conv1 (the only constraints that touch pixel patches).

## T1-T6 (cards, written before running): free-coefficient reconstruction on the toy (`toy/t_kkt.py`)
- Build (Theorem 23 part 3): m = 1000 candidates, labels balanced, c = softplus(a) free (no sum-to-1: lam and N live
  in c), residual over all parameters (T6: also fc only). Same optimizer as G2 (Adam lr .05), 3000 steps; doubling
  test on steps before a number counts.
- PREDICTIONS (pass conditions from the plan): T1 true train set with c = p_k/(N lam) gives residual .241.
  T2 residual < .5, never resting at 1.0, c not collapsing. T3 >= 10% of candidates have a training image with pixel
  corr > .8, and those images are more unsure than the average training image. T4 corr(log c ratios, log p ratios) > .7.
  T5 Fisher write within 2x of truth (.016) and simple write within 2x (.034); beats G1 (.160). T6 report both.
- RESULT T1 (first run, `logs/t_kkt_all_3000.log`): FAILED, residual 1027 instead of .241. Reason (algebra, not
  theory): the code aimed the sum at lam theta but scaled c as p_k / (N lam), which belongs to the target theta.
  Off by 1/lam = 1000. Theorem 23 part 3 says the target is theta with c = p_k / (N lam) (lam and N inside c). The G3
  run used the same wrong target, so its T2-T5 numbers are void (noted only: Fisher .146, co-firing .81). Fix: target
  theta in both. No other change.
- RESULT T1-T5 (fixed target, `logs/t_kkt_all_3000.log`, 5 min). T1 .2411 HOLDS. T2 FAILED (final 1.047, min .937;
  c did not collapse: sum 12.2). T3 FAILED (0 training images rebuilt; median best corr .31). T4 FAILED (-.06).
  T5: lsq .0803 (2.4x truth; best generator so far, G1 1.08); Fisher .189 (worse than G1 .160). Co-firing vs truth .824
  (best generator so far; G0 .66, G1 .60). Train-unsure reference void (only 27 unsure training images).
- Decomposition (no new generation, `toy/t_decomp.py`, `logs/t_decomp.log`): true set .241 (true c), .047 (NNLS c);
  top 10 / 30 / 100 / 300 contributing training points alone: 3.26 / 4.83 / 7.53 / 3.75; rebuilt inputs: own c 1.106,
  NNLS c for the same inputs .302 (2702 of 3000 c nonzero). Biases fit worst in every case (c2.bias 1.06 even for truth).
  Statements in PAPER.md section 23 (retraction of part 1's reading; Statements 23.2-23.4). No run before them.
- Corrections (advisor review, before any run): Statement 23.3 compared .302 (rebuilt, best c) with .241 (true, TRUE
  c); on equal terms it is .302 vs .047, so the equation PREFERS the training set 6x. Non-identifiability withdrawn.
  The partial-sum "retraction" of Theorem 23 part 1 broke the cancellation by construction: withdrawn, support open.
  T4 retired: best c is not unique even at the true inputs.

## S (card, written before running): support of theta with FREE c, restricted subsets (`toy/t_support.py`, no generation)
- For a subset of training images, R*(subset) = min over c >= 0 of || theta - G_subset^T c || / || theta ||. With no
  images R = 1. Subsets: the 27 unsure (u > .1); top 10 / 30 / 100 / 300 contributors; all sure (u <= .1, 973);
  random subsets of size 27 / 30 / 100 / 300 (5 draws each) as the simple-rule baseline.
- PREDICTION (Shonu's conjecture, "the network knows what it is unsure about"): unsure-27 beats random-27 by > .1 (mean of
  5 draws), and top-30 beats random-30 by > .1. Sure-973 alone reaches within .05 of all 1000 (.047) if the sure inputs
  are redundant; above that if the unsure ones are needed.
- RESULT S (`toy/t_support.py`, `logs/t_support.log`). Best-c residual: all 1000 .047; unsure-27 .856 vs random-27
  .943 (sd .013); top 10 / 30 / 100 / 300 contributors .982 / .917 / .623 / .169 vs random .990 / .951 / .774 / .535;
  sure-973 alone .249. PREDICTION FAILED narrowly at the pre-registered sizes (unsure-27 beats random by .087, top-30 by
  .035; threshold .1). At larger sizes the effect is strong: the top 30% reproduce theta 3x better than a random 30%
  (.169 vs .535), and removing the 27 most unsure images raises the residual 5x (.047 -> .249). Reading: Shonu's
  conjecture holds in a GRADED form (theta is carried mostly by the less-sure part of the data), not as a tiny support.

## V (card, written before running): variable projection (Statement 23.2) on the toy (`toy/t_varpro.py`)
- Exact c by NNLS (direct; the Gram was singular in float32 at the grey start) every K = 300 x-steps (one solve ~55 s; K = 100 would take ~33 min), Adam on x only in between (block coordinate descent:
  monotone in exact arithmetic; K is numerics, doubling test later). m = 1000, 3000 x-steps. Estimated ~12 min.
- PREDICTIONS (outcomes fixed in PAPER.md section 23): (a) R* <= .10 and training images rebuilt (T3 >= 10% at corr
  > .8); (b) R* <= .10 without training images; (c) R* stays near .3. I expect (b) or (c), not (a): the x-landscape
  from a grey start is hard, and m = N leaves no spare candidates (Buzaglo used m = 2N).
  Also T5: simple write and Fisher write from the result, against truth and against G3 (.080 / .189).
- RESULT V (`toy/t_kkt.py 3000 varpro`, `logs/t_varpro_3000.log`, 21 min). R* .726 -> .258 (stalls from step ~2100;
  true inputs .047). No training image rebuilt (median best corr .354). Co-firing .860 (best so far). T5: lsq .110,
  Fisher .468, both WORSE than G3 (.080 / .189) although R* is lower. Outcome (c) as pre-written: the x-landscape is
  the obstacle, not c. New fact: lowering R* made the P2 writes worse. Reason written in PAPER.md (Theorem 24) before
  any further run.

## W (card, written before running): is the worse write concentration or junk candidates? (no generation)
- Refit the G3 and V writes (a) on candidates with any c > 0 only, uniform; (b) all candidates weighted by sum_k c_{s,k}.
- PREDICTION under Statement 24.1 (concentration): restricting to c > 0 makes the Fisher write WORSE than uniform over
  all 1000. If it gets BETTER, the cause was junk candidates and 24.1's reading is wrong.
- RESULT W (`logs/t_refit.log`). G3: all 1000 have c > 0; top 10% hold 17.7% of c mass; Fisher uniform .189, weighted by
  c .239. V: 845 of 1000 have c > 0; top 10% hold 28.9%; Fisher uniform-all .468, c > 0 only .300, weighted by c .358.
  PREDICTION FAILED (c > 0 only made V's Fisher write BETTER: junk candidates cost .17). But concentration also hurts:
  weighting by c worsens the Fisher write in both runs, and V (more concentrated) is still worse than G3 after junk
  removal (.300 vs .189). Both causes are real. Statement 24.1 revised in PAPER.md and labelled a hypothesis.

## F1, F2 (cards, written before running): does "how often" live in the network? (`f1_freq.py`, ~2 min)
- Class decomposition (exact): P(neuron fires) = sum_y P(y) P(fires | y). P(y) from the last layer (near uniform);
  P(fires | y) from the network's own class-y inputs (256 class inversions, check E). Real: 256 train images (check only).
- F1 PREDICTION: per-neuron on-rate and mean activity, self-made vs real: Spearman >= .7 in 3b; agreement rises with
  depth over 3a, 3b, 4a, 4d, 5b. Simple-rule baseline: weight size (check A: .29 in 3b).
- F2 PREDICTION: 480 x 480 co-firing correlation matrices of 3b, self-made vs real: Pearson >= .6.
- RESULT F1, F2 (`logs/f1_freq.log`). On-rate Spearman self vs real: 3a .594, 3b .454, 4a .527, 4d .526, 5b .325;
  median ratio 1.02-1.52; 78-91% of neurons within x2. F1 FAILED (3b .454 < .7; depth trend reversed). F2 HOLDS (3b
  co-firing Pearson .780). Two mathematical reasons, written before any rerun: (1) attenuation: an observed correlation
  is at most sqrt(reliability_self x reliability_real); with 256 images per side the ceiling was never measured, so
  .454 cannot be read yet. Split-half (128/128) on both sides gives it, no new data. (2) The reversed depth trend: the
  inversion objective acts on the logits directly, so the deepest layers are pushed furthest from natural levels
  (I4: logit level 9x; 5b over-fires 1.52x). Prototypes are least natural where the maximization acts.
- RESULT ceiling (split-half 128/128, Spearman-Brown to 256): ceiling 3a .983, 3b .973, 4a .950, 4d .899, 5b .696.
  Observed / ceiling .47-.60 at every depth. Attenuation does NOT explain F1's failure. Reason (PAPER.md section 25):
  a class inversion is the MODE of class y; in high dimensions the mode is not a typical sample, and a firing frequency
  is an average over typical samples.

# INTERROGATE (2026-10-23 night; plan rewritten): assume the network IS enough; question each gap's assumption
## N1-N3 (cards, written before running): Neural Feature Ansatz (`n1_nfa.py`, ~5 min)
- For each reading layer: Gram of its 1x1 input convs' weights, W^T W summed over branches, vs (a) AGOP of the logits
  at its input (random projections), (b) the output Fisher at its input (u9 estimator), (c) the plain second moment of
  the input; baseline the identity. 500 train images, split halves for reliability. Matrix cosine; powers via eigen.
- N1 (4a reading 3b, 480 ch) PREDICTION: cos(Gram, AGOP^a) >= .8 for a in {1/2, 1}, and above cos(Gram, plain moment)
  and cos(Gram, identity). N3 (3b reading 3a, 4b reading 4a, 4e reading 4d): the same at every depth.
- N2 (the 5x5 branch kernels on the bottleneck window) folded into the same script only if N1 holds.
- RESULT N1, N3 (`n1_nfa.py`, `logs/n1_nfa.log`, ~6 min). Matrix cosine of the input-conv weight Gram with:
| site | AGOP^.5 | AGOP | Fisher^.5 | Fisher | plain^.5 | plain | identity | reliability AGOP / Fisher |
|---|---|---|---|---|---|---|---|---|
| 3a->3b | .939 | .879 | .939 | .882 | .471 | .106 | .541 | .990 / .971 |
| 3b->4a | .918 | .905 | .915 | .896 | .450 | .086 | .509 | .948 / .856 |
| 4a->4b | .923 | .835 | .920 | .843 | .515 | .156 | .550 | .959 / .867 |
| 4d->4e | .901 | .917 | .899 | .894 | .648 | .256 | .616 | .975 / .868 |
  Off-diagonal entries only: AGOP^.5 .902-.937, Fisher^.5 .896-.935, plain^.5 .203-.347 (unit-diagonal normalised: same).
  N1 HOLDS, N3 HOLDS (every depth). The weights carry a plain second moment of the data's output sensitivity, and the
  Fisher at each layer's input, to cos ~.9; plain activity statistics are not what the Gram holds.
## N-controls (card, written before running): is N1 about the DATA, or the weight sandwich?
- The AGOP at x is W^T G_u W, so cos(W^T W, AGOP^.5) may be high from structure alone. Controls: (1) G_u = I:
  cos(W^T W, (W^T W)^.5), zero images; (2) the same N1 script on pixel-shuffled real images and on uniform noise
  (250 each).
- PREDICTION: (1) clearly below N1 (< .85 at 3b->4a); (2) real > shuffled > noise, with real beating noise by > .05.
  If noise matches real, the Gram reflects the network's forward structure, not training data: Statement 26.2 changes.
- Lemma 26.3 (proved, no run): with an unconstrained write and a CONSTANT metric, the optimal write is the
  identity-metric least squares (identical regressors for every reader). So no constant metric (true average Fisher
  or NFA) can beat the lsq write; the Fisher write's gain is entirely per-image variation of the metric. The planned
  "NFA write" build is cancelled by this lemma.
- RESULT N-controls (`logs/n1_controls.log`). (1) G_u = I: cos(W^T W, (W^T W)^.5) = .926 / .932 / .940 / .948, as high
  as N1 or higher. PREDICTION (1) FAILED: N1's full-matrix cosine is explained by the sandwich structure; it is not
  evidence of data content. (2) Real .918 > shuffled .800 > noise .790 at 3b->4a (real beats noise by .128): HOLDS.
  The discriminating comparison (target = the REAL-image Fisher, off-diagonal): weights alone (Gram or Gram^2)
  .80-.89; the network's Fisher on noise .32-.74, on shuffled .38-.82. From 3b on, the weights predict the real metric
  ~2x better than running the network on unnatural inputs. Statement 26.2 corrected accordingly in PAPER.md.

# DISMANTLE (2026-10-23 night; Shonu: continue until the need for outside data is dismantled or precisely proved)
## S1 (card, written before running): does the Fisher write depend on WHICH unsure inputs? (`s1_mix.py`)
- Step1 write (u9 estimator) on M's read, 4000 unsure train images (u > .1) per fit: animal classes only (ImageNet
  0-397), non-animal only, and two disjoint random unsure sets (noise baseline). Dev-5000 (all classes), paired.
- PREDICTION (outcome 1, mix-robust): |animal - non-animal| < .01 and within 3x of |randA - randB|. If animal-only or
  non-animal-only is worse than random-unsure by > .02, the write depends on which inputs (outcome 2).
- RESULT S1 (`logs/s1_mix.log`, ~8 min). animal (3642) .0791, non-animal .0837, randA .0674, randB .0649. Paired:
  animal - non-animal -.0045 (se .0041); randA - randB +.0025 (.0028); animal - randA +.0117 (.0033); non-animal - randA
  +.0162 (.0034). HOLDS (outcome 1): the write barely depends on which natural unsure inputs (a write fit on half the
  world transfers to all classes at a cost of .012-.016; a broad mix helps a little).
## S2 (card): how many inputs does the EXACT Fisher write need? (real unsure, cost planning for generation)
- Exact top-10 Fisher (u4_w3 solve + solve_active_k) on M's read, from 1000 and 2000 random unsure real images.
- PREDICTION: exact 2000 <= .07 dev-5000 (the cheap estimator needed 4000 for .067).
- RESULT S2 (`logs/s2_count.log`): exact step1 on M's read from random unsure real images: 1000 -> .0824, 2000 -> .0582
  (~1 min per 2000). HOLDS (<= .07). Target for generation: ~2000 network-made unsure inputs.
## E7 (card, written before running): boundary inputs made by the network (`e7_boundary.py`)
- Rule (fixed once): class a spread evenly over the 1000 ImageNet classes; partner b = the real class whose last-layer
  weight row has the highest cosine with a's (weights only). Minimise cross-entropy to the soft target 1/2 e_a + 1/2 e_b
  (bounded: optimum p_a = p_b = 1/2), plain pixels, lucent standard transforms, Adam lr .05, 128 steps.
- Smallest test (64 inputs). PREDICTIONS: mean 1 - sum p^2 in .35-.55 (real unsure pool .39); mean logit level LSE
  within x2 of real (55), not x9 like inversions; curve-member co-firing pattern vs real Pearson >= .8 (inversions .95).
- RESULT E7 smallest test (64 inputs, 117 s): 1 - sum p^2 .381 (HOLDS; real unsure .39), LSE 23.4 (real 55: x2.4 low,
  just outside the x2 band; inversions 473), top-2 mass .889, member co-firing vs real .932 (HOLDS; inversions .944).
- Scale-up: 1024 boundary inputs, 128 steps (~31 min), plus mirror images (the network was trained flip-invariant,
  a property of the network) = 2048 inputs.
## E8 (card, written before running): P2 with network-made inputs only (`e8_p2.py`)
- (1) M's read + exact step1 write from the 2048 boundary inputs. (2) THE DATA-FREE FORMULA: self read (arcs #2 by
  Lemma 14.1 on the 256 class inversions, check E) + exact step1 write from the boundary inputs. References: real unsure
  2000 exact on M's read .0582; lsq on M's read .0836; self read + self lsq .0696; THE FORMULA (FITTED) .0507; bar B .0361.
- PREDICTIONS: (1) <= .070 and beats lsq (.0836) by > 2 paired se. (2) <= .065 and beats .0696 by > 2 paired se.
## E7-toy (card, written before running): boundary inputs on the toy (quick loop while the long run finishes)
- Same rule on the toy: class a, partner = nearest class by fc weight rows; cross-entropy to 1/2 e_a + 1/2 e_b; sigmoid
  pixels, small jitter; 256 steps; 1000 inputs (+ mirrors). P2 proxy (delete conv2 24-31), KL on 5000 fresh truth.
- PREDICTION: the Fisher write from boundary inputs beats every earlier generator (best .160, G1) by > 2 paired se and
  lands within 3x of truth (.016). Mirrors: the toy network was NOT trained with flips, so a mirror row is reported
  separately (no invariance is assumed).
- RESULT E7-toy (`logs/e7_toy.log`, 11 s generation): boundary Fisher .550 (mirrors .314; truth .016; G1 .160),
  co-firing .567. FAILED. Reason: on the toy even class inversions lose natural internal statistics (co-firing .66;
  InceptionV1 .94). A 4-class network carries at most log2 4 = 2 bits of label per input, so the set of inputs it maps
  to any answer is huge and mostly unnatural; a 1000-class deep network pins its inputs far more. So the toy cannot
  test generators whose quality depends on the network constraining natural structure ("no verdicts from weak
  models"). Quick loops for this question move to 64-input InceptionV1 slices.
- RESULT E9 (`logs/e9_what.log`): network-made Phi pattern vs real .990, Sigma .903; best arcs [170,180 | 250,290,320 |
  350..120] (share .588); real-statistics arcs #2 ranks 3rd (.578). Flat top, as in U5.
- RESULT E8 (`logs/e8_p2.log`): M read + boundary step1 2.41; self read + boundary step1 13.97; M read + lsq on boundary
  .124; self read + self lsq (e9 WHAT) .0777. FAILED catastrophically (predicted <= .07 / .065).
  Hypothesis H (written before any diagnostic): the Fisher write trusts the metric fully; in write directions where the
  generated inputs carry no metric mass the solve is unconstrained and fills them with noise. Boundary inputs put their
  uncertainty on 2 classes (top-2 mass .889), real unsure images on several. Predictions: (a) the boundary N has many
  more near-zero eigen-directions than the real-unsure N at equal count; (b) step0 alone already blows up.
- RESULT E10 (`logs/e10_diag.log`): step0 from 512 boundary 10.08 vs 512 real unsure .152 -> (b) HOLDS. N spectrum:
  boundary 448 / 1048 eigenvalues above 1e-3 / 1e-5 x top (real 646 / 1104); trace 77.7 vs 229.5; boundary mass along the
  real top-200 directions: median .30, 10th pct .17 of real -> (a) only partly: weaker, not empty. A uniform 3x
  weakening cannot explain a blow-up. Hypothesis H2 (from Prop. 27.1, written before checking): the write lives on the
  CROSS moment E[c c^T F]; on boundary inputs the coarse units may be silent where the output is sensitive, so the B
  columns rest on few samples and extrapolate on real images. Check from the saved N: per-reader ratio of the
  c-weighted metric mass (B block) to the plain metric mass (constant block), boundary vs real.
- RESULT H2 check: c-weighted / plain metric mass per reader: boundary median 1.9e5, real 8.3e5 (4.3x lower); plain
  mass boundary 3.3e-4 vs real 2.5e-4 (boundary MORE unsure), c-weighted 77.7 vs 229.5. H2 holds only partly (weaker,
  not absent); it cannot explain KL 10 either.
- Hypothesis H3 (Theorem 10b, written before checking): the constant beta = A mu_z - B mu_c is a PLAIN first moment of the
  input set; an error in it hits every position of every real image. Test: keep the boundary step0 columns B, recompute
  beta from (i) real train images (diagnostic only), (ii) the class inversions (data-free), (iii) boundary (as used).
  PREDICTION: (i) and (ii) bring the KL to the sane range (< .15); (iii) stays blown up.
- RESULT H3 (`logs/e11_beta.log`): with boundary B, beta from real (diagnostic) 5.05, from inversions 3.81, plain-mean
  boundary 6.42 (as used 10.08). FAILED: beta is not the cause; the columns B are wrong.
- Hypothesis H4 (written first): B fitted on boundary inputs is much LARGER than B fitted on real unsure (the units fire
  ~2x weaker there), direction similar. PREDICTION: ||B_bnd|| / ||B_real|| > 3 and column cosine > .5.
- RESULT H4 (`logs/e12_norm.log`): |B_bnd| / |B_real| = 2.4-2.6 (pred > 3: narrowly FAILED); cos(bnd, real) .07-.16
  (pred > .5: FAILED); cos(real, trained) .46-.52. Coarse units on boundary inputs fire at ~40% of real strength (means
  40-56 vs 89-110), family mean 14.8 vs 35.8. Reason: with ReLUs, weaker inputs change WHICH units fire, not only how
  much, so the (z, c, F) relation the write regresses on is different. S1's robustness held across natural unsure
  images, which all fire at natural strength. Boundary inputs do not. Next: is it pixel contrast? (seconds)
- RESULT contrast: per-image pixel sd real 61.4, inversions 44.0, boundary 21.3; neighbour-difference sd 19.7 / 38.6 /
  20.7. Reason (math): the boundary objective is bounded, so the gradient vanishes at p_a = p_b = 1/2 and the image stays
  near its grey start: its contrast is set by the start and step count (a CHOSEN setting), not by the network.
## E13 (card, quick slice): unsure inputs between two class modes (forward passes only, 64 pairs)
- x(alpha) = alpha inv_a + (1 - alpha) inv_b, b = a's nearest class by last-layer weights; bisect alpha until p_a = p_b.
- PREDICTION: family mean activity within x1.5 of real unsure (35.8; boundary 14.8), coarse-unit means within x1.5 of
  real, 1 - sum p^2 >= .3, member co-firing vs real >= .85.
- RESULT E13 (`logs/e13_blend.log`): blend at 50/50: 1 - sum p^2 .377 (HOLDS), family mean 26.9 vs real unsure 41.8
  (x1.55: narrowly FAILED), coarse units 62-88 vs 118-125 (FAILED for unit 0), co-firing .897 (HOLDS), pixel sd 31.5.
  Averaging two images lowers contrast. Open question, narrowed: is the natural input contrast in the network?
## C28 (card): the ansatz as a weights-only score of an input distribution (`c28_score.py`, quick slice)
- Conjecture 28.1: the next-layer Gram matches the input-side Fisher best on the TRAINING distribution (noise .38 vs
  real .87 at 3b->4a supports it). Test: real unsure images (128) with contrast scaled by s in {.5, .75, 1, 1.33}
  (x' = 117 + s (x - 117)); score = off-diagonal cos(Gram_4a, Fisher at 3b output).
- PREDICTION: the score peaks at s = 1, and falls by > .02 at s = .5 and s = 1.33 (split-half noise reported).
- RESULT C28 (`logs/c28_score.log`): score .856 / .852 / .854 / .863 at contrast x.5 / .75 / 1 / 1.33 (split-half .82-.86).
  FAILED: flat. Reason: the network is near positively homogeneous (ReLU, small biases), so contrast mostly rescales every
  activity by one factor, and the cosine ignores scale. Contrast is close to a gauge direction for shape statistics.
## C29 (card): is contrast the cause of the boundary write's failure? (quick: 512 real unsure at contrast x1 and x.5)
- If the network is near homogeneous, z and c scale together and the Fisher write (a ratio) barely moves.
- PREDICTION: step0 columns from real unsure at x.5 vs x1: cosine > .8 per unit, and dev KL within .03. If so, contrast
  is NOT why boundary inputs failed (cos .07-.16); the cause is their structure.
- RESULT C29 (`logs/c29_contrast.log`): real unsure x.5 vs x1: column cos .54-.64 (pred > .8: FAILED), norm ratio
  .94-1.02, dev KL .1548 vs .1517 (HOLDS). The write direction is loosely determined (flat directions), so column cosines
  are a poor diagnostic; and contrast is NOT the cause of the boundary blow-up.
- Hypothesis H5 (coverage, written first): boundary inputs put their uncertainty on 2 classes, so each image constrains
  the write along ONE logit direction; some write directions that real images weight heavily get almost no boundary
  weight, and N_bnd^-1 amplifies noise there. PREDICTION: the largest generalised eigenvalue of (N_real, N_bnd) exceeds
  1e3, and most of the boundary solution's real-metric energy lies in the top-20 generalised directions.
- RESULT H5 (`logs/h5_cover.log`): generalised eigenvalues of (N_real, N_bnd): top 1.4e7, 10th 3.3e4, 100th 412, median 1.2.
  61% of the real-metric error energy of (w_bnd - w_real) lies in the top-20 directions, 91% in the top-100. HOLDS.
  Mechanism: ~100 write directions that real images weight are nearly empty under boundary inputs; the solve divides by
  ~0 there. Requirement: a generated set must COVER every write direction the real metric weights.
## C30 (card): coverage by multi-class uncertainty (blends of class inversions, forward passes only)
- 2-way centroids (anchor + its 1st or 2nd nearest class among the 256 inversions, by last-layer weights) vs 4-way
  centroids (anchor + nearest 1-3, or 2-4); 512 each, plus mirrors = 1024. Step0 Fisher write on M's read.
- PREDICTION (from H5): 4-way: top generalised eigenvalue vs N_real (512) < 1e4 and dev KL < .3; 2-way: top > 1e5 and
  KL > 1 (blows up like boundary inputs).
- RESULT C30 (`logs/c30_cover.log`): 2-way: top / 10th / 100th generalised eigenvalue 4.6e6 / 1.7e5 / 1.2e4, KL 1.12
  (blows up: HOLDS). 4-way: 4.2e6 / 3.5e4 / 858, KL 2.16. FAILED: multi-class uncertainty improves the middle
  directions but not the worst ones, and the KL got worse.
- Hypothesis H6 (reader coverage, written first): the write's metric sits at the readers' PRE-ReLU inputs, so it carries
  each reader's gate. A reader that never fires on generated inputs has zero metric, so its column and constant are
  unconstrained; real images do switch it on. PREDICTION: the top-20 generalised directions put > 50% of their weight
  on readers whose on-rate on the generated set is < 10% of their real on-rate.
- RESULT H6 (`logs/h6_readers.log`): readers silent under generated inputs (< 10% of real on-rate): 23 (4-way), 45
  (boundary); they fire 35-37% on real. Top-20 generalised directions: 54% of weight on readers silent under 4-way
  blends (HOLDS, > 50%), 45% on constant terms. Requirement: READER coverage, not class coverage.
## C31 (card): complete the set with the network's own picture of each silent reader (`c31_readers.py`)
- For every reader silent under the boundary or 4-way sets: feature visualisation (same plain-pixel rule, maximise the
  reader's mean pre-ReLU over positions), 2 seeds, 128 steps, plus mirrors. Sets: boundary (2048) + reader inputs;
  4-way blends (1024) + reader inputs. Step0 write on M's read.
- PREDICTION: top generalised eigenvalue vs real drops > 100x for both, and dev KL < .3 (real unsure 512: .152).
- RESULT C31 (`logs/c31_readers.log`, ~10 min): 52 silent readers; their feature visualisations switch all 52 on.
  boundary + readers (2256): top / 10th / 100th generalised eigenvalue 1.2e3 / 481 / 53 (was 1.4e7 / 3.3e4 / 412), step0
  dev KL .169 (was 10.08; real unsure 512: .152). 4-way + readers: 3.4e3 / 1.6e3 / 235, KL .361. Coverage prediction
  HOLDS for both (>100x); KL < .3 HOLDS for boundary + readers, FAILED narrowly for 4-way. Reader coverage was the gap.
## E14 (card): THE DATA-FREE FORMULA, full pipeline (boundary + reader inputs; WHAT recomputed on them)
- e9 on the combined set (WHAT), then e8: M read + exact step1; self read (Lemma 14.1 on class inversions, e9 WHAT) +
  exact step1. PREDICTIONS: M read <= .07 (real unsure 2000 exact .0582); data-free formula <= .065 and beats self read +
  self lsq by > 2 paired se.
- RESULT E14 (`logs/e14_p2.log`): WHAT on the combined set: Phi .991, Sigma .904 vs real (same top arcs). M read + step1
  .1397; data-free formula (self read + step1) .1655; M read + lsq on the set .0965; self read + self lsq .0771.
  FAILED (<= .07 / <= .065): the write is sane now but loses to lsq.
- Before any fix, two numbers (no generation): (i) the coverage baseline between two INDEPENDENT real unsure sets of 512
  (what finite-sample coverage looks like); (ii) composition of the combined set's top generalised directions: B
  (per coarse unit) vs constant. Hypothesis H7 (written first): a column B_rj is constrained only where unit j is active
  AND reader r is sensitive; reader pictures switch readers on where the coarse units may be silent. PREDICTION: real vs
  real top eigenvalue < 1e2; the combined set's top-20 directions are > 60% B-part.
- RESULT H7 (`logs/h7_cover.log`): real(512) vs independent real(512): top / 10th / 100th 301 / 192 / 49. real vs combined
  network-made: 1212 / 481 / 53. Top-20 directions: 98.8% on the per-reader constants, 1.2% on unit columns. Both
  predictions FAILED (baseline is 301, not < 1e2; B-part is ~0, not > 60%). Meaning: coverage is within 4x of
  real-vs-real at the worst direction and equal by the 100th. No direction is structurally missing. The remaining gap
  is effective sample size (2256 network-made ~ 512 real: .169 vs .152), and the Fisher write loses to lsq below ~1000
  real-equivalent inputs (real 1000: .082 vs lsq .084).
## E15 (card): more samples from the network's own invariances (no new generation)
- Each input of the combined set (1128) x 4 random shifts (up to 16 px, reflect padding) x mirror = 9024 inputs; exact
  step0 + step1 on M's read and on the self read (e14 WHAT). (Shift and mirror invariance are properties of this network:
  convolutions + global pooling, trained on random crops and flips.)
- PREDICTION (effective sample size): M read + step1 <= .08 and beats M read + lsq on the set (.0965) by > 2 paired se;
  data-free formula (self read + step1) beats self read + self lsq (.0771).
- RESULT E15 (`logs/e15_aug.log`, ~15 min): with shifts x4 and mirrors (9024 inputs): M read + step1 .0976 (was .1397);
  data-free formula .1280 (was .1655); M read + lsq on the set .1028; paired step1 - lsq -.0052 (se .0050). FAILED (<= .08;
  beat lsq by > 2 se; beat .0771). More samples help, but the write only ties lsq (real 2000 beats lsq by .025).
## BV (card): bias or variance? (step0 on two halves of the augmented set + real unsure 2000; real metric)
- Variance proxy: (w_A - w_B)^T N_real (w_A - w_B) / 4 (the full-set write's variance ~ a quarter of the half-difference).
  Bias proxy: (w_full - w_real)^T N_real (w_full - w_real) minus the variance proxy and minus the real write's own noise.
- PREDICTION (if count is the gap): variance > bias. If bias > variance: the network-made inputs differ systematically
  in what the write uses, and more of them will not close it.
- RESULT BV (`logs/bv_split.log`): in the real metric: gap .4285 = network-made variance .1246 + real variance .0678 +
  bias .2361 (real write energy 4.18). FAILED (variance > bias): bias dominates; more inputs of this kind remove <= ~30%.
## BV2 (card): is the bias in N (coarse activity x metric) or in m (family effect x metric)? (cached, seconds)
- Swaps: w(N_real, m_gen) and w(N_gen, m_real), gap to w_real in the real metric.
- PREDICTION: none written as a direction; this is a decomposition. Reading: the swap that keeps the generated part with
  the smaller gap is where the network-made inputs are right.
- RESULT BV2 (`logs/bv2_swap.log`): swaps explode (real N + gen m 26.2; gen N + real m 19507); m_gen vs m_real cosine
  .073 (scale x6), N cosine .327. RETIRED as a probe: N and m are moments of the same distribution and the write depends
  only on their ratio, so mixing them is meaningless (should have been seen before running).
## B16 (card): is the bias in the per-reader constants? (the worst directions were 99% constants; step0 aug write)
- Keep the network-made step0 columns B (full augmented set), recompute beta = A mu_z - B mu_c from: class inversions
  (data-free), the augmented set (plain mean), real unsure 2000 (diagnostic). Compare with the solve's own beta.
- PREDICTION: if beta carries the bias, beta from real (diagnostic) cuts the dev KL by > .02 and beta from inversions
  sits between.
- RESULT B16 (`logs/b16_beta.log`): network-made step0 B with beta from: the solve .1282; inversions .1131; plain mean
  over the set .0932; real (diagnostic) .0938. HOLDS: the bias sat in the constants. Reason: the Fisher-weighted constant
  E_q[F (Az - Bc)] / E_q[F] puts its weight on the set's most sensitive positions, which are its least natural parts
  (reader pictures, boundary regions); the plain mean is a first moment of activity, which transfers across input sets
  (like the read's statistics, E1/B0). Rule that follows: columns from the Fisher metric, constant from the plain mean.
## E17 (card): THE DATA-FREE FORMULA with the plain-mean constant (step1 columns; both reads; augmented set)
- PREDICTIONS: M read + step1 B + plain beta <= .085 and beats M read + lsq on the set (.1028) by > 2 paired se.
  Data-free formula (self read, e14 WHAT) + step1 B + plain beta beats self read + self lsq (.0771) by > 2 paired se.
- RESULT E17 (`logs/e17_formula.log`, ~15 min): M read: step0 + plain beta .0942, step1 + plain beta .0807, lsq on the set
  .1070; paired step1 - lsq -.0263 (se .0042): HOLDS (<= .085; > 2 se). First network-made Fisher write that beats lsq,
  by the same margin real data gives (~.025). Self read: step1 + plain beta .0984, lsq on the set .1048, lsq on class
  inversions .0771: FAILED (paired +.0213 vs .0771). Reason: the self read was fit (Lemma 14.1) on class inversions only;
  on boundary and reader inputs its units fire elsewhere (the false-on failure named in section 14), so a write fit there
  does not transfer. M's read stays sane on those inputs.
## E18 (card): fit the self read on the SAME network-made set the write uses
- Lemma 14.1 (e14 WHAT) on class inversions + every 4th input of the augmented set (+ mirrors); then step0/step1 Fisher
  columns + plain beta on the full set; lsq on the set as baseline.
- PREDICTION: data-free formula <= .085 and beats self read (inversions) + lsq on inversions (.0771) by > 2 paired se.
- RESULT E18 (`logs/e18_formula.log`): self read fit on the set: step1 + plain beta .0984 (unchanged), lsq on the set
  .1104, lsq on inversions .0800. FAILED: the read's fitting set was not the reason. Reason (from both reads): the
  Fisher write always beats lsq ON THE SAME INPUTS (M: .081 vs .107; self: .098 vs .110), but the network-made set's
  plain activity statistics are biased (its lsq is ~.03 worse than lsq on class inversions); the Fisher write inherits it.
## E19 (card): Prop. 27.1's split with two sources (difference estimator)
- write = lsq(class inversions) + [Fisher step1 (set) - lsq (set)], all on the self read fit on class inversions (E4's
  read); constant = plain mean on class inversions for the combined columns. The bracket is the cross-moment correction
  measured where both terms share the set's bias, so the bias cancels to first order.
- PREDICTION: <= .068 and beats E4 (.0771) by > 2 paired se.
- RESULT E19 (`logs/e19_split.log`): self read: lsq(inv) + [Fisher(set) - lsq(set)] .0935 (vs E4 +.0163): FAILED.
  M read: same .0882; M read + lsq(inv) .0661 (!), M read + Fisher(set) .0807. New fact: a simple write fit on 256
  class inversions beats the same write on 1000 real images (.0836) and beats the network-made Fisher write. Reason
  (math): identity-metric regression weights positions by activity; inversions fire the family strongly (logit level
  9x real), so the fit is driven by clear, strong curve positions (an implicit activity weighting). The [Fisher - lsq]
  correction measured on the boundary set is not the real correction (BV: biased cross moment).
## E20 (card): the class inversions with the network's metric at a temperature
- Inversions are the best network-made set for activity statistics; their only flaw for the Fisher write is p ~ one-hot
  (Lemma 19.1b). Use p_T = softmax(f / T) in the metric. T0 = the T at which the inversions' mean 1 - sum p_T^2 equals
  the network-made boundary set's (.38): fixed by the network. Doubling test: T0/2, T0, 2 T0. Inputs: 256 inversions x 4
  shifts x mirror = 2048. Step0 columns + plain-mean constant (on the inversions). Both reads.
- PREDICTION: at T0, M read <= .060 and self read <= .070 (beat lsq(inv) .0661 / .0771 by > 2 paired se); T0/2 and 2T0
  within 2 paired se of T0 (then T is numerics, not a knob).
- RESULT E20 (`logs/e20_temp.log`): T0 = 44.3. M read: Fisher(inv, T) + plain beta .152 / .099 / .111 at T0/2, T0, 2T0;
  self read .117 / .097 / .093; lsq(inv) .0661 / .0771. FAILED: never beats lsq, and the doubling test fails (T is a knob).
- DRIFT (Shonu: "You're again drifting from math based stuff"): E17-E20 were four patch-variants, not consequences of
  written statements. Stopped. PAPER.md section 29 states what the results show mathematically before any further run.
## N4 (card; derived in PAPER.md section 30): does K^T K give the gated co-sensitivity M? (`n4_gated.py`)
- PREDICTION: off-diagonal cos(M, K^T K) > .7 over the 96 channels of 3b's 5x5 branch, above ungated E[g g^T] and plain
  co-activation; the same on the 10-curve block.
- RESULT N4 (`logs/n4_gated.log`): cos(M, K^T K) .320 (96 ch) / .542 (curves); ungated identical (pre-activation
  gradients are already gated); co-activation .469 / .448. FAILED. Statement 30.3: the ansatz pins sandwiched matrices
  only, not the inner gated metric; (iii) is given by neither stationarity nor the ansatz.
## N5 (card; Conjecture 30.4): the weights' naturalness score S_l on the network-made sets (`n5_score.py`)
- PREDICTION: real unsure >= .8 at 3b->4a, 4a->4b, 4d->4e; boundary and inversions < .6 at 4a->4b and 4d->4e.
- RESULT N5 (`logs/n5_score.log`): S_l real .880/.803/.868; inversions .838/.824/.819; boundary .709/.717/.796; reader
  pictures .832/.752/.827. FAILED (network-made sets pass it): S_l cannot remove the bias.
## N6 (card; Theorem 31): does each neuron's imbalance store per-sample gradient energies? (`n6_sgd.py`)
- PREDICTION: Spearman(D_i, E||g_in||^2 - E||g_out||^2) > .5 over 3b's 480 neurons; above the activity-only baseline.
- RESULT N6 (`logs/n6_sgd.log`): Spearman .621 (normalised .613); input side .417, output side .175, activity -.265.
  HOLDS: the per-neuron imbalance stores per-sample gradient energies (Theorem 31, SGD imprint).
## N7 (card; Corollary 31.2): do network-made sets meet the weights' second-moment constraints? (`n7_energy.py`)
- PREDICTION: Spearman(D(weights), D(q)): real unsure ~ .6; class inversions, boundary, reader pictures, combined < .4.
- RESULT N7 (`logs/n7_energy.log`): real unsure .704, real .665; inversions .213, boundary .264, reader pictures .239,
  combined .310. HOLDS: the weights' constraints separate real from network-made inputs.
## N8 (card; Construction 31.3): max-entropy reweighting of the network-made pool, held-out neurons (`n8_tilt.py`)
- Pool: 256 inversions + 1024 boundary + 104 reader pictures; d_s by 2 random-sign projections per input.
- PREDICTION: held-out (odd neurons) Spearman > .5 (unweighted ~ .3); effective sample size > 20% of the pool.
- RESULT N8 (`logs/n8_tilt.log`): held-out Spearman tilted .322 vs unweighted .316; fitted half .182 (worse); ESS 19%,
  weight 92% on inversions. FAILED. Hypothesis (math, written first): infeasibility. A reweighting can only reach the
  cone spanned by the pool's energy vectors d_s; if D(weights) lies outside it, no weights meet the constraints.
  Check: best non-negative combination (NNLS) of the d_s to D(weights): residual and cosine.
- RESULT N8-feasibility (`logs/n8_feas.log`): best NNLS cone combination cos .424 (Spearman .437, residual .906, 21
  images); real unsure mean vector cos .533; inversions alone .177. Infeasible: reweighting cannot meet the target.
## N9 (card; Construction 31.4): generate inputs against the SGD-imprint target (`n9_imprint.py`, smallest test)
- PREDICTION: held-out (odd neurons) Spearman from ~.2 to > .5; member co-firing vs real > .85.
- RESULT N9 (`logs/n9_imprint.log`): held-out .165 -> .337, fitted .194 -> .533, co-firing .912 -> .909, pixel sd 3.4.
  FAILED (> .5) but doubled. Under-determined fit (16 inputs, 240 constraints).
## N10 (card; Definition 31.6): does the imprint generator repair (iii)? (`n10_psi.py`)
- PREDICTION: Psi pattern correlation with real unsure: imprint (from boundary) beats its boundary start by > .2.
- RESULT N10 (`logs/n10_psi.log`): (the import re-ran N9: held-out .174 this time vs .337 before -> N9's held-out gain
  does NOT replicate; RETRACTED.) Psi pattern corr vs real: boundary start .498 -> boundary + imprint .395; diagonal shape
  -.158 -> -.084; scale ratio .135 -> .054. FAILED: the imprint generator makes (iii) worse. Reason: 480 scalar gradient
  energies are second-order quantities that react strongly to tiny pixel changes; the optimizer met them with an
  adversarial perturbation (sd 3.5), not a move toward natural inputs. Section 32 written before any further run.
## S3 (card; Prop. 32.2): does a scalar sensitivity weighting carry the Fisher gain? (`s3_scalar.py`, ~3 min)
- PREDICTION: > 70% of the lsq -> step0 gap recovered (then per-unit info suffices) or < 30% (shape needed).
- RESULT S3 (`logs/s3_scalar.log`): real unsure 2000, M read: identity lsq .0798, s-weighted lsq .0754, Fisher step0
  .0681: the scalar weighting recovers 38% of the gap (between the 30/70 thresholds). ~40% of the Fisher gain is per-unit
  importance (channel-2 type), ~60% is the per-image SHAPE of the metric (pairwise; no channel carries it). Note: the
  network-made M read + lsq on 256 class inversions (.0661) beats real-unsure Fisher step0 at 2000 (.0681).

# IMPOSSIBILITY (PAPER.md section 34): no weights-only map can be optimal for every distribution consistent with theta
## X3 (card): demonstrate the premise for channels C1 + C2 (3b) on InceptionV1 (`x3_counter.py`)
- Imprint-neutral self-labeled q from the network-made pool (QP on the N8 energy vectors); p2 = real unsure 2000 + q.
- PREDICTION: q exists (residual < 1% of typical |d_s|); w*(p2) costs > .02 dev KL more than w*(p1); bound > 0.
- RESULT X3 first run (`logs/x3_counter.log`): the QP put all weight on 12 class inversions (sure inputs: zero per-sample
  gradients), so q is imprint-neutral trivially and changes the write by exactly 0 (bound 0; dev KL .0532 both).
  Correct but degenerate: inputs the network is sure about are invisible to BOTH the weights and the write (Prop. 19.2).
  The construction must require write mass: restrict q to UNSURE inputs. Open question this decides: can a positive
  mixture of unsure inputs cancel the imprint at all? If not, channel 2 constrains the unsure mass and Theorem 34.3's
  premise fails there.
- RESULT X3 unsure (`logs/x3_unsure.log`): 924 unsure inputs in the pool; an imprint-neutral q exists among them
  (residual .0021, 4 boundary inputs, 1 - sum p^2 .50): channel 2 does NOT forbid unsure mass. But this q barely moves the
  write (+.0001; bound .0003): the QP asked only for neutrality.
## X4 (card): the adversarial consistent mixture (linear program; first-order shift of the write)
- delta w ~ 1/2 N1^-1 sum_s q_s r_s, r_s = m_s - N_s w1 (input s's own gradient of the write loss at w1). Maximise
  v . sum q_s r_s with v = N1^-1 rbar, over q >= 0, sum q = 1, sum q d = 0 (unsure pool). Then p2 = p1 + eps q,
  eps = 1 and 10; exact w*(p2) by solve; dev KL and the bound.
- PREDICTION: at eps = 1 the dev KL of w*(p2) exceeds w*(p1)'s by > .02; at eps = 10 by > .1.
- RESULT X4 (`logs/x4_adv.log`): adversarial imprint-neutral unsure mixture (9 inputs, residual .012, 1 - sum p^2 .46):
  bound .0005 (eps 1) / .0010 (eps 10); dev KL change -.0002 / -.0000. PREDICTION FAILED badly, and in the OTHER
  direction from Prop. 32.1: within the pool, imprint-neutral mixtures are write-neutral even at 10x the real mass.
  Reason (math, written first): the output-side energy in each neuron's imprint, || sum_p delta(p) h_n(p) ||^2, is the
  diagonal block of the write's own moment matrix for that neuron; 480 imprint constraints may pin most of what the
  write uses.
## X5 (card): control (C1 only) vs C1 + C2, pool = network-made unsure + 512 real unsure images not in p1
- PREDICTION (if the imprint is what pins the write): C1-only adversarial mixture moves the write by > .02 dev KL at
  eps = 1; C1 + C2 moves it by < .005.
- RESULT X5 (`logs/x5_control.log`): C1 only (support 1 input, imprint residual 5.7): dev change +.0008 / +.0009 at
  eps 1 / 10 (bounds .0023 / .0031). C1 + C2 (25 inputs, 99% real): +.0002 / +.0016. The control ALSO barely moves the
  write, so X4's "imprint pins the write" is not supported: something limits every mixture's effect. Diagnostic next
  (no conclusion drawn): per-input write-matrix trace tr(N_s) vs the real average tr(N1).
- RESULT X5-diagnostic (`logs/x5_diag.log`): per-input tr N_s / tr N1: network-made unsure median .21 (max 8.6), real
  unsure .63 (max 14.4): the mixtures were not small. Reason (math): an addition q moves the write by
  r_q = m_q - N_q w1, how badly the real-data write fits q's OWN objective; if w1 already fits q, the optimum barely moves.
## X6 (card): are consistent ADDITIONS harmless? The worst known addition: the whole network-made boundary + readers set
- Its Fisher-form moments are self-labeled expectations, so adding them keeps C1 exactly. Alone its write scored 2.41.
- PREDICTION (additions harmless): w*(p1 + eps q) within .005 dev KL of w*(p1) at eps = 1, within .02 at eps = 10.
- RESULT X6 (`logs/x6_add.log`): w*(p1) .0532; + 1 q .0533 (+.0000); + 10 q .0703 (+.0170); + 100 q .1059 (+.0527);
  q alone .1691. HOLDS (harmless at eps = 1; +.017 at 10). As the self-labeled (C1-consistent) mass grows, the optimal
  write moves continuously from the real one to q's.
## X7 (card): the same with the IMPRINT-NEUTRAL q of X4 (C1 + C2 exact), large eps
- PREDICTION: w*(p1 + eps q_neutral) degrades steadily with eps; > .02 worse than w*(p1) at eps = 1000.
- RESULT X7 (`logs/x7_neutral.log`): imprint-neutral q (9 inputs): change -.0000 / +.0002 / -.0001 at eps 10 / 100 / 1000.
  PREDICTION FAILED (> .02 at 1000). Two readings: (1) channel C2 pins the write's moments (the weights carry what the
  write needs); (2) confound: neutrality met by low-gradient inputs whose write mass is negligible.
## X8 (card): separate the readings: require write mass
- LP over the unsure pool: maximise the first-order write shift, s.t. q >= 0, sum q = 1, per-neuron imprint residual
  <= 2% of q's own mean |d| (relative, so small-gradient inputs gain nothing), and write mass sum_s q_s tr(N_s) >= tr(N1).
- PREDICTION under reading (1): change < .005 at eps = 1000; under (2): > .02.
- RESULT X8 (`logs/x8_mass.log`): INFEASIBLE: no mixture of the unsure pool is imprint-neutral (2% relative) and carries
  the real average write mass (tau >= 1; median per-input tau .36). Write mass and imprint neutrality pull against each other.
- CORRECTION (math): Theorem 31 fixes the imbalance only up to the unknown factor eta / (2 lam B), so the weights pin the
  DIRECTION of D, not its length. The right C2 is: the combined imbalance stays PARALLEL to D. X3-X8 used "equal", which
  is stricter than the theorem. Natural inputs already point along D (N7 .70), so natural additions are nearly consistent;
  unnatural, write-moving ones point elsewhere (.21-.31).
## X9 (card): direction-only C2: how much write mass can a consistent network-made mixture carry?
- LP: maximise sum q tau s.t. |P_perp(sum q d)| <= rel * sum q |d| componentwise (P_perp: orthogonal to D), q >= 0, sum 1;
  rel in {.02, .05, .1, .2}. Then, at the smallest rel that allows tau >= 1, maximise the write shift and test eps = 10-1000.
- PREDICTION: write mass >= 1 is feasible only at rel >= .1 (the imprint direction does constrain unnatural mass).
- RESULT X9 (`logs/x9_dir.log`): direction-only C2: infeasible at every tolerance (rel .02-.2), even without a write-mass
  floor: no mixture of the network-made unsure pool keeps the imbalance parallel to D. PREDICTION (feasible at >= .1)
  FAILED: the imprint direction excludes every write-moving network-made addition found.

# EXACTNESS (2026-10-24; plan ~/.claude/plans/okay-for-this-session-twinkly-gosling.md; NEURON-MATH Part IV)
Rule: no real-image statistic inside a generalist build. The real-moment formula (.0647) is a FITTED microscope.
Lean: `lean/coarse/Coarse/Exact.lean` PROVED `quad_moment_suff`, `gated_write_in_span`, `exact_no_gain` (Prop. 13.5,
Cor. E.1 logic). Build passes, no sorry.

## R1 (card): E-small. Is the level-0 statistic S estimated by a few hundred images?
- Build: D_n = F(S-hat_n), level 0 (Prop. 11.1, as R3d: unpooled x for all readers, ridge 1e-6 tr/471), nested random
  subsets of the train split (rng 5 order, same as R3d), n = 16 / 64 / 256 / 1024 / 2000. Paired dev-5000. MIX-DERIVED.
- Baselines: removal .664; G (weights only) .177; n = 2000 .0647.
- PREDICTION: n = 16 .09-.12; n = 64 ~.075; n = 256 ~.068; n = 1024 within .003 of n = 2000. Knee (smallest n within
  .003 of n = 2000): 256. If the knee is > 1024, E-small fails for level 0 and m must be re-stated (e.g. a smaller
  object than the full 481 x 481 S).
- RESULT R1 (`x1_ladder.py`, `logs/x1_ladder.log`): n=16 .0749 (+.0102, paired se .0021); n=64 .0668 (+.0022, .0009);
  n=256 .0669 (+.0022, .0004); n=1024 .0645 (-.0002); n=2000 .0647. Knee = 64 (predicted 256). E-small HOLDS at level 0,
  better than predicted. Meaning: the developer's re-mix at level 0 needs tens of images, not thousands. 16 images already
  beat the capped 2000-step from-zero ruler (.0738). The level-0 statistic is small in the sense of Remark 13.3.
- Toy check of Prop. 13.5 (gated formula vs brute-force weighted lstsq, 5000 samples, 30% gate): max diff 4e-16.

## R2 (card): Statement 13.6. Do the readers' gates (level 1) carry most of the .020 exactness gap?
- Builds, same 2000 train images as R3d (rng 5), ridge 1e-6 tr/471 per reader, paired dev-5000, MIX-DERIVED:
  (a) level 0 as R3d (reproduces .0647); (b) level 0 with the pool reader fed its true input p (instrument fix);
  (c) level 1: per-reader gated moments S^(j) (gate = reader on in the intact network, per position);
  (d) level 1 from the first 256 of those images (E-small at level 1).
- Baselines: level 0 .0647; converged from-zero training .0564; training from level 0 .0446 (best known).
- PREDICTION: (a) .0647 +- .0005. (b) within .002 of (a). (c) <= .055 (closes >= half of the .020), and beats the
  converged from-zero ruler .0564 by > 2 paired se. (d) within .006 of (c).
- If (c) > .060: the gates are NOT the main reason for E-KL0; the reason R (Prop. 7.5) is refuted for B3 and the
  next candidate is level 2 (per-image metric) or gate flips (beyond first order).
- RESULT R2 (`x2_gated.py`, `logs/x2_gated.log`): (a) .0647 HOLDS; (b) pool-input fix .0633 (-.0014, se .0006) HOLDS;
  (c) level 1 .0603 (-.0043 vs (a), se .0020; +.0039 vs converged from-zero .0564, se .0028; +.0157 vs training from
  level 0 .0446, se .0027) PREDICTION (<= .055) FAILS; (d) level 1 at n=256 .0628 (+.0025 vs (c)) HOLDS.
  Reader on-fractions per conv: .42 / .65 / .81 / .29. Statement 13.6 REFUTED: the gates carry ~.004 of the .020.
## R2b (card): loop step 8 for the R2 failure. Which reason (R-a metric / R-b coupling / R-c higher order)?
- Six builds (removal, level 0, level 0p, level 1, trained from zero long, trained from level 0) on 500 dev images
  outside dev-1000. L2, L3, L3' as in NEURON-MATH section 14 (8 labels per image drawn from the intact network).
- PREDICTION (my prior, R-a): L2 and L3 both rank trained-from-level-0 below level 1, with ratios near the KL ratio
  (~.74); L3 tracks KL across builds (per-image corr > .8) except removal (large errors, R-c).
- RESULT R2b (`x2b_reason.py`, `logs/x2b_reason.log`, 500 dev images): KL / L2 / L3 / L3':
  removal .658/.039/.720/.714; level0 .077/.0092/.083/.090; level1 .073/.0079/.062/.080; trained-zero .058/.020/.068/.072;
  trained-from-level0 .050/.0123/.064/.068. Trained-from-level0 minus level1: KL -.023 (se .011); L2 +.0045 (se .0005,
  ratio 1.57); L3 +.0026 (se .011); L3' -.012 (se .014).
  R-a (per-image diagonal metric) REFUTED as the main reason: L2 ranks the pair the wrong way by 9 se. The trained
  build has LARGER per-reader errors but lower KL, so its errors cancel downstream (R-b or R-c).
  Instrument flaw: L3 and L3' from 8 sampled labels have se ~.011, too noisy to separate R-b from R-c.
  Fix (same statement, exact instrument): L3 = 1/2 Var_{c ~ p}(J_c e), with J e from one forward-mode JVP per build.
- RESULT R2b-exact (`x2b_exact.py`, `logs/x2b_exact.log`): trained-from-level0 minus level0: KL -.0269, L2 +.0032
  (se .0004), L3 -.0256, L3' -.0270. Level1 minus level0: KL -.0042, L3 -.0206, L3' -.0087; flip term level1 .019 vs
  .005-.007 others. VERIFIED REASONS: (R-b) training's gain is coupling (cancellation through later layers), its diagonal
  part has the wrong sign; (R-c) level 1 fails by gate flips. R-a refuted. Prop. 14.8 (Lean via gated_write_in_span):
  a reader-reader coupling cannot move the write unless it varies per image; a position coupling can.

## R2c (card): where does the coupling live?
- Exact Fisher rows J_c (gradient of logit c w.r.t. reader pre-activations) for the top-K classes (K = 10, renormalised;
  report the mass covered). Split L3 of (level 0, trained from level 0) into blocks: diagonal; same reader channel
  across positions (spatial); same position across readers; full. Same 500 dev images.
- PREDICTION (from Prop. 14.8): the spatial block (same channel, all positions) carries >= 60% of the trained-minus-
  level-0 L3 difference; the same-position block carries < 25%. Consistency: full top-K L3 within 15% of the FD L3.
- RESULT R2c (`x2c_blocks.py`, `logs/x2c_blocks.log`): top-10 mass ~1.0; top-K L3 = FD L3 (.0797 vs .0801) consistency
  HOLDS. trained-from-level0 minus level0: diag +.0032, same-channel (spatial) +.0206 (se .0021), same-position +.0018,
  full -.0252. PREDICTION (spatial >= 60% of the gain) REFUTED: both blocks say trained is WORSE. Cross-cross term
  (different readers AND positions) = full - spatial - position + diag: level0 +.046, trained +.0014.
## R3' (card, Statement 14.10): in-span coupled formula (Prop. 14.9), predictions in NEURON-MATH section 14
- RESULT (`x3_coupled.py`, `logs/x3_coupled.log`): output-J n=2000 .2486 (se .0106); pre-J n=2000 .2433; output-J
  n=260 93.8. All far WORSE than level 0p (.0633). Statement 14.10 REFUTED on (a), (b), (e).
- Loop step 7, candidate reason R-flat (written before any diagnostic): the second-order objective is not coercive. It
  sees an error only through the output's local sensitivity, so directions of Lambda that change no logit difference are
  free (rank of M <= 9 per image; at n=260 the rank is < 2340 < 4710: the ridge alone holds it, hence 93.8). The
  minimiser moves far along nearly flat directions, where the second-order model is invalid (gate flips, curvature).
  Training never does this: it starts at level 0 and moves locally (an implicit trust region). Alternative: a bug.
- Loop step 8 checks (x3b_diag.py), predictions under R-flat: (i) identity: the stats' predicted L3 difference between
  the coupled and level-0 builds on the 2000 images equals the directly measured one within 10% (else: bug);
  (ii) M has effective rank << 4710 (eigenvalues > 1e-6 x max: < 3000 at n=2000); (iii) |Lambda_coupled| >> |Lambda_0|
  (> 5x); (iv) on dev, L3 of the coupled build is BELOW level 0's while its KL is far above (second-order breakdown).
- RESULT x3b_diag (`logs/x3b_diag.log`): (i) identity HOLDS exactly (stats -.0754 = direct -.0754): no bug.
  (ii) eigenvalues of M > 1e-6 max: 2478 / 4710; > 1e-3 max: 309. (iii) |Lambda| ratio 4.2 (rest part 12.6) HOLDS.
  Train 2000: level0 KL .079 L3 .0755; coupled KL .139, L3 .0000 (!), L3' .055. Dev500: coupled KL .209, L3 .212.
  (iv) PREDICTION (dev L3 below level 0) FAILS: R-flat as stated is refuted. VERIFIED REASON: over-fitting. The solve
  interpolates the 2000 images (train L3 = 0) because each image gives <= 9 constraints (the classes the network is
  unsure about), against 196 x 10 for level 0. Plus a second-order breakdown on train (L3 0, KL .139).
## R3'' (card): Statement 14.12, the same statistic on the POPULATION scale (all 35000 train images; snapshot 8000)
- Run time ~25 min (stated before running). Builds: output-J and pre-J, n = 8000 and 35000. Paired dev-5000.
- PREDICTION: n = 35000 dev L3 below level 0 (.080 on dev500) and KL in [.050, .100]; n = 8000 between n = 2000 and
  35000. If n = 35000 still > .0633 (level 0p): the second-order breakdown, not the sample, is the obstacle.
- RESULT R3'' (`x3c_population.py`, `logs/x3c_population.log`, dev-5000): output-J n=35000 .0552 (se .0027): vs level0p
  -.0081 (se .0034), vs converged from-zero .0564 -.0012 (se .0026, tie), vs best known .0446 +.0106 (se .0021).
  pre-J n=35000 .0561. n=8000: .1019 / .1086 (worse than level 0). PREDICTION: KL in [.05, .10] HOLDS; n=8000 between
  2000 and 35000 HOLDS; dev L3 below level 0 NOT YET MEASURED. 14.10(d) (gated-J worse) REFUTED (tie). The sample curve is still steep at 35000: not yet the population value.
## R3''-terms (card): close the unmeasured half. dev500 KL, L3, L3' and cross term of the n=35000 output-J build
- PREDICTION: L3 below level 0's .080; |KL - L3| <= .006 (Taylor holds, as for the trained builds); cross term < .015.
  If L3 << KL: the remaining .0106 is second-order breakdown; if L3 ~ KL: estimation or out-of-span.
- RESULT R3''-terms (`x3d_terms.py`, `x3d_blocks.py`, dev500): coupled35k KL .0702, L3 .0620, L3' .0632, cross term
  .0144 (level0 .046, trained .0014). L3 below level 0 HOLDS; cross < .015 HOLDS; |KL - L3| = .008 (pred <= .006) FAILS
  narrowly. Split of the dev500 gap to trained-from-level0 (.0702 - .0501 = .020): second-order objective .0075
  (estimation or out-of-span); beyond second order ~.0126 (coupled sits +.008 above its L3, trained -.004 below).
## X-coupled (card): THE exactness test on the best formula. Train B3 k=0 started at coupled35k (X-B3 recipe:
   2000 steps, lr .003, eval 125, patience 4, early stop on dev-1000)
- PREDICTION: training improves it by > 2 paired se (the formula's A > 0, 14.11(2) breakdown + 1/3 second-order part);
  it ends at .043-.047 dev-5000, i.e. the same basin as training from level 0 (.0446). If it ends clearly BELOW .0446
  (> 2 se): the formula is a better starting point and the best known network moves.
- RESULT X-coupled (`b3_train.py state:x3c_out_n35000 B3-k0-coupledstart`, `logs/b3_coupledstart.log`): dev-5000
  .0542 (se .0026). Paired: minus coupled start -.0010 (se .0015); minus best known (from level 0) +.0096 (se .0016);
  minus converged from-zero -.0023 (se .0022). PREDICTION (gain > 2 se, ends .043-.047) FAILS both halves.
  Curve: dev-1000 .057 at start -> .09-.12 at steps 125-500 -> back to .058 at step 1625 (best); train KL .091 -> .030.
  Reading: Adam's first steps knock the formula out of a narrow region and it re-converges near where it started.
  "No gain" here is NOT local optimality: a better network (.0446) exists that this run does not reach.

# ROUTE A (2026-10-24, Shonu chose A: the coupling metric from the network; plus "understand it on an image basis")
## R5' part 1-3 (Statements 16.1 premise, 16.4): Kbar from 2000 train images (`x5_kbar.py`, `x5b_offsets.py`, `x5c_maps.py`)
- (c) Kronecker: top term .669 of Kbar's energy (top 3 .726). R5' line (>= .70) FAILS narrowly; 16.4(c) (< .50) FAILS.
- (a) co-circularity, all offsets: cos .715 vs same-orientation .633 (weak test: near offsets agree). SHARP test (offsets
  where the rules differ > 60 deg): co-circular .696 vs same-orientation .211; 9 / 10 members. (a) HOLDS.
- (b) ring: REFUTED. Picture `cache/x5_maps.png`: every member has the SAME horizontal band of coupling energy (~±3 cells
  wide, ±1 tall), independent of its orientation. The "on circle" share .41 vs area .22 came from the circle passing
  through the bright centre.
- Reading: Kbar = (shared spatial envelope, horizontal) x (co-circular orientation rule). The envelope needs a reason.
- Step 8 for the envelope (`x5d_rot.py`, 500 images rotated 90 deg): horizontal/vertical neighbour energy 1.41 -> 1.15.
  The band weakens but stays horizontal in the network's frame. "Image statistics alone" REFUTED; the envelope is part
  photos, part network (weights learned from horizontal photos, W7). Not decided further.
## R5' part 4 (16.4 d): which images carry the level-0 cross term, and where in them (`x5e_images.py`)
- PREDICTION (written before): the top-12 dev images by level-0 cross term show round things (wheels, dials, eyes,
  loops, rims) in >= 7 of 12; their per-position contribution maps sit on curved contours, and the trained build's map
  is much weaker there.
- RESULT R5' part 4 (`x5e_images.py`, picture `cache/x5_images.png`): the top 5% of dev images carry 95.5% of the
  level-0 cross term. Top 12: bow tie, fig/melon halves, washbasin, teddy bear, snake on sand, two dogs, pineapple,
  combination lock, baby in bathtub, hatchet, mashed potato, poncho. Clearly round: 5 (melon, basin, teddy, lock dial,
  tub); curved shapes: 2 (bow tie, snake); not: 5. PREDICTION (>= 7 of 12 round) BORDERLINE (7 only if curved shapes
  count). Maps: level-0 error runs ALONG the curved contour (rind, rim, dial ring); training removes it there
  (dial 2.47 -> .15, melon 11.1 -> 2.9, basin 5.1 -> 2.2). Hatchet: trained worse (1.5 -> 3.9).
## R5' part 5 (Statement 16.1 premise): factorised build with Kbar (2000 images) and activity moments (n = 2000, 256)
- Predictions as in NEURON-MATH section 16 (<= .058 at n = 2000; n = 256 within .004).
- RESULT R5' part 5 (`x5f_factor.py`, `logs/x5f_factor.log`, dev-5000): factorised n=2000 .0491 (se .0029); n=256 .0534.
  Paired: n2000 vs level0p -.0142 (se .0013); vs coupled35k -.0061 (.0026); vs converged from-zero .0564 -.0074 (.0023,
  3.2 se); vs best known .0446 +.0045 (.0020). n256 vs n2000 +.0043 (pred within .004: FAILS by .0003).
  16.1 HOLDS, stronger than predicted: averaging the metric removes the per-image over-fit and IMPROVES on the 35000-image
  per-image formula. The coupling object for route A is Kbar (a network-side 3920 x 3920 metric).
## X-factor (card): exactness test on the factorised build (|DeltaW| 1.33x level 0, vs 2.80x for coupled35k)
- PREDICTION (from 15.1: a broad point is not kicked out): dev-1000 never rises more than .01 above its start; training
  gains > 2 paired se; it ends at or below the best known .0446 (the better start reaches a better network).
## 16.7 (card): co-circular part of Kbar from the first downstream spatial kernels, WEIGHTS ONLY (`x5g_weights.py`)
- Predictions in NEURON-MATH section 16 (sharp co-circular cos >= .5 and above same-orientation; offset-pattern corr
  with Kbar >= .5, offsets |d| <= 4 cells).
- RESULT 16.7 (`x5g_weights.py`, weights only): offset-pattern corr M vs Kbar .718 (.699 without d = 0). Sharp test:
  M co-circular .780 vs same-orientation .117, 10/10 members (Kbar from images: .802 vs .210, 10/10). HOLDS. The
  contour-integration geometry is in 4a's weights.
## 16.8 (card): build with the WEIGHTS metric (`x5h_wbuild.py`). Metric = Gram of 4a's effective kernels on the family's
   signals over all four branches (1x1: centre; 3x3/5x5: spatial conv x bottleneck; pool: W_F / 9 over 3x3), no image.
   Activity moments from the same n = 2000 / 256 images (developer form). Lambda is scale-free in the metric: no knob.
- PREDICTION: n = 2000 between factorised .0491 and level0p .0633, <= .058; n = 256 within .005 of n = 2000.
- RESULT 16.8 (`x5h_wbuild.py`, `logs/x5h_wbuild.log`, dev-5000): weights metric n=2000 .0570 (se .0033): vs level0p
  -.0063 (se .0009); vs Kbar-factorised +.0079 (se .0008); vs coupled35k +.0018 (.0029). n=256 .0594 (+.0024).
  PREDICTION (<= .058, between, n=256 within .005) HOLDS. The 4a-weights metric carries 44% of the image metric's gain
  ((.0633 - .0570) / (.0633 - .0491)), with no image in the metric.
- RESULT X-factor (`logs/b3_factorstart.log`): dev-5000 .0430 (se .0021). Paired: minus factorised start -.0061 (se .0019);
  minus old best known .0446 -.0016 (se .0006, 2.7 se: NEW BEST KNOWN k = 0 network); minus converged from-zero -.0135.
  PREDICTIONS: no kick-out FAILS (dev-1000 .056 -> .103 at step 250); gain > 2 se HOLDS; ends <= .0446 HOLDS.
  Old log `b3_fstart.log` (training from level 0) shows the same kick (.076 -> .113 at step 250): the kick is the fresh
  Adam steps at lr .003, on EVERY start. Statement 15.1's mechanism (large coefficients = sharp point) REFUTED.
## X-factor-local (card, Statement 15.2 instrument): local exactness test. Start at the factorised build, lr 3e-4 (1/10),
   500 steps, eval every 125, patience 10.
- PREDICTION: no kick (dev-1000 never above start + .005); a real local defect: dev-5000 gain >= .003 (> 2 paired se),
  because the formula minimises a second-order surrogate, not the KL (gate flips, out-of-span part).
- RESULT X-factor-local (`logs/b3_factorlocal.log`): no kick (dev-1000 .056 -> .047 at step 125, then down). dev-5000
  .0404 (se .0021): -.0087 vs its start (se .0013). PREDICTION (no kick; gain >= .003) HOLDS. NEW BEST KNOWN k = 0.
  Same recipe from level 0 (`logs/b3_level0local.log`): .0452. Factorised start beats level-0 start by .0048 (se .0006,
  8 se) under the identical recipe. lr .003 was hurting every earlier run (factor .0430 vs local .0404).

# CIRCUITS (2026-10-24, plan ~/.claude/plans/okay-last-session-landed-velvet-wren.md, NEURON-MATH Part VI)
Unit = edge between the network's own families (weight kernel + gate); Kbar = metric transport (Prop. 18.5).
Gap A = weights metric .0570 -> Kbar build .0491 (.0079). Gap B = Kbar build -> best known .0404 (.0087).
## I1 + A1-A3 (card): transport identity and the split of Gap A at 4a (`c1_split4a.py`, 2000 train images rng 5 =
   Kbar's images; activity moments as x5f, n = 2000; FITTED microscopes, dev-5000)
- Identity predictions: float64 on 20 images, |K_transport - K_direct| / |K| < 1e-6; float32 on 2000 images vs cached
  `x5_kbar.pt` < 1e-3 and its build = .0491 within .0005.
- Variants of K = E A^T Sigma A at 4a (gates on the 3x3 / 5x5 branches; Sigma = Cov_c of logit gradients at 4a outputs):
  | Var | gates | Sigma | prediction (dev-5000 KL) |
  |---|---|---|---|
  | a | 1 | I | .0570 (= 16.8, M) |
  | b | firing rate | I | .0565 (gates alone matter little) |
  | c | 1 | diag of E Sigma (per-unit importance) | .053 |
  | e | rate | diag E Sigma | .053 |
  | f | exact per image | diag Sigma(x) per image | .052 |
  | g | 1 | full Sigma(x) per image | .050 |
  | d | rate | full Sigma(x) per image (P3) | .050 |
  | x | exact | full (= Kbar) | .0491 |
- A1 (diag carries > half of Gap A): f <= .0530. A2: x - f carries > half. A3 (P3 at 4a): d within 2 paired se of x.
  My belief before the run: A1 fails narrowly (coupling across 4a units matters, the envelope is long range), A3 holds.
## B1-B2 (card): what training's last local step changed (`c5_gapb.py`). S = Kbar build (`x5f_factor_n2000`, .0491),
   T = trained locally from S (`B3-k0-factorlocal`, .0404). Delta = T - S in raw reader-weight units.
- Split: (i) in-span = W_F Delta-Lambda with ONE shared Delta-Lambda (10 x 470) fitted by least squares over all 368
  readers (Prop. 11.1 form); (ii) reader-specific remainder = Delta-W minus (i); (iii) reader-bias change.
- Builds S+(i), S+(ii), S+(iii), S+(i)+(iii), T on dev-5000.
- PREDICTION: (ii) carries >= half of the .0087 gain (readers' gates, Prop. 7.5), (i) <= half, (iii) < .001. Simple-rule
  baseline: proportional to squared norm of each part. Non-additivity reported.
- B2 PREDICTION: on the 500 x2b images the second-order term L3 shows >= 60% of the KL difference S -> T (a local step
  stays in the Taylor region); flip term changes < .002.
## I2 (card): Kbar's top eigenmodes as pictures (`c2_modes.py`, picture `cache/c2_modes.png`)
- PREDICTION: the top modes are arcs or rings of co-circular curves (not a plain horizontal band); the top 3 modes carry
  > 50% of level 0p's metric cost E r^T Kbar r on 500 dev images; the top 20 carry > 80%.
- RESULT I1 (`logs/c1_split4a.log`): transport identity |K_t - K_d|/|K_d| = 3.8e-7 on 20 images (float32; prediction said
  1e-6 in float64: met in float32), and the 2000-image transport reproduces the cached Kbar to 2.0e-7. Prop. 18.5 CHECKED.
- RESULT I2 (`logs/c2_modes.log`, `cache/c2_modes.png`): PREDICTION FAILS. Top 3 modes carry .166 of level 0p's cost, top
  20 .456; 108 modes needed for 80%. Picture: the top modes are NOT arcs or rings. They are low spatial harmonics of a
  CENTRED window (blob, top/bottom split, left/right, centre-surround), each mixing all ten orientations.
  Per-position energy of Kbar (trace over members): a centred blob, peak at the image centre, ~5-20x weaker at the
  corners. The 4a weights metric M (16.8) is FLAT (border effects only). Reading: the output cares about the centre
  (where ImageNet objects are); M has no such weight. Candidate for Gap A: the spatial importance window (readout).
- PARTIAL B1 (`logs/c5_gapb.log`): |Delta-W|^2 split: in-span .08, reader-specific remainder .92. S .0491, S+span .0488,
  S+rem .0407 (T .0404). The top-10 readers by remainder energy are 9 of the 16 5x5-bottleneck readers (global 289-303)
  and one 1x1 reader; top 10% of readers carry 42%.
## B4 (card): is training's reader-specific move an EDGE-level (per-branch) re-routing? (`c6_branch.py`)
- Statement: the four edges from the family to 4a (1x1, 3x3 bottleneck, 5x5 bottleneck, pool) have different spatial
  kernels downstream (1, 3x3, 5x5, max-pool), so the reader x position metric is not a Kronecker product across
  branches and Prop. 14.8 no longer forces one shared Lambda. Prediction: one Delta-Lambda_b per branch (4 x 10 x 470),
  least squares on T - S, explains >= 70% of |Delta-W|^2 and S + per-branch part reaches <= .0425 (>= 75% of the .0087).
  Simple-rule baseline: random 4-way split of the 368 readers (same parameter count) explains far less.
- RESULT A1-A3 (`logs/c1_split4a.log`, dev-5000; Kbar build .0491): a (no gates, Sigma = I; = M with the exact pool
  input) .0552; b (rate gates, I) .0574; c (no gates, diag E Sigma) .0557; e (rate, diag E Sigma) .0576; f (exact gates,
  diag Sigma(x)) .0564; g (no gates, full Sigma(x)) .0488 (vs Kbar -.0003, se .0003); d (rate gates, full Sigma(x)) .0493
  (+.0002, se .0002). A1 (diag carries > half) REFUTED: every diagonal metric stays at .055-.058, no better than M.
  A2 HOLDS: the whole of Gap A is the coupling BETWEEN 4a units. A3 HOLDS: gates do not matter at 4a (no gates at all
  ties Kbar). My pre-run belief (A1 fails narrowly) was right in sign, wrong in size: the diagonal carries ~0.
## A4 (card): weights-only circuit metric composed to depth L (`c8_ladder.py`): linearised network above the readers
   (gates = 1, which A3/g licensed at 4a; max-pool -> average), Sigma_L = I at layer L (logits: I - 11^T/n). DERIVED
   metric; activity moments as before.
- PREDICTION: 4b improves on 4a's .0552 by > 2 se; the best depth reaches <= .0520 (closes >= 40% of the .0061 from
  .0552 to Kbar's .0491); the logits (whole linearised net) is WORSE than the best middle depth (deep products of
  ungated weights collapse onto a few directions). Per-position energy becomes centred with depth (border falloff).
## B5 (card): per-reader minimiser of the averaged second-order objective (`c7_reader.py`; NEURON-MATH Statement B5).
- Reader-level metric FITTED (2000 Kbar images): window w(p) x stationary K_jj'(d), |d| <= 6. One linear system by CG.
- PREDICTION: (a) in-span restriction within .003 of the Kbar build (.0491); (b) per-reader minimiser <= .0430.
- RESULT A4 part 4b (`logs/c8_ladder.log`): weights-only metric composed to 4b (linearised, Sigma = I) .0512 (se .0030):
  vs Kbar +.0021 (se .0004); vs 16.8 weights metric -.0058 (se .0006); vs level 0p -.0121. PREDICTION (4b beats 4a by
  > 2 se; best <= .0520) HOLDS already at 4b. Gap A (.0570 -> .0491): 4b closes 76%, with no image in the metric.
  Per-position energy at 4b: corner/centre .45 (borders only; Kbar's window is much sharper).
- RESULT A4 part 4d: weights-only metric composed to 4d .0489 (se .0027): vs Kbar -.0001 (se .0005). GAP A CLOSED by a
  DERIVED metric (linearised weights 4a-4d, no gates, no image). Per-position energy corner/centre .24.
- RESULT A4 complete (`logs/c8_ladder.log`): 4a .0552, 4b .0512, 4d .0489, 5b .0516 (+.0025 vs Kbar, se .0008), logits
  .0579 (+.0088, se .0014). PREDICTIONS: 4b > 4a HOLDS; best <= .0520 HOLDS (4d .0489, ties Kbar); logits worse than the
  best middle depth HOLDS. Effective rank (entropy) of K: 4a 1264, 4b 732, 4d 210, 5b 24, logits 8; Kbar 324. Cosine with
  Kbar .56 / .68 / .62 / .46 / .36. Window grows with depth (corner/centre .45 / .24 / .06 / .08).
  LABEL: depth 4d is CHOSEN (best of a 4-item menu on dev-5000), until a weights-only rule fixes it.
## 20.3 verify (card): effective rank of the TRUE-gated metric E_x A_x^T A_x at 5b (Sigma = I), 8 train images
   (`c9_gaterank.py`). PREDICTION (only the gate-noise reason predicts it): effective rank >> 24 (gate-free), >= 150.
   Control: the same with gates frozen to 1 reproduces ~24.
- RESULT 20.3 verify (`logs/c9_gaterank.log`, central difference eps 1e-3): true-gated A_x^T A_x at 5b, effective rank
  per image 1566-1931; average of 8 images 3223 (still rising). Gate-free 5b: 24. PREDICTION (>> 24, >= 150) HOLDS.
  Statement 20.3's reason is VERIFIED: per-image gate patterns keep the deep metric's rank up; the gate-free product
  collapses. (Kbar's 324 is lower than 3223 because Kbar also carries the output's class weighting.)
## X-w4d-local (card): local exactness test on the DERIVED-metric build `c8_build_4d` (.0489): lr 3e-4, 500 steps, eval
   every 125, patience 10 (Statement 15.2 recipe).
- PREDICTION: gain ~ .008-.009 (Gap B is the in-span restriction, independent of where the metric comes from); ends
  .040-.041. A gain clearly larger than Gap B (> .0105) would mean the weights metric has a defect the Kbar build lacks.
- RESULT band check (cached tensors, no run): horizontal / vertical neighbour energy (|d| <= 3): Kbar 1.40 (old 1.41),
  Kbar on rotated images 1.21, weights-only metric 4a 1.01, 4b 1.54, 4d 1.28, 5b 1.11. The horizontal band of 16.5 is
  written in the weights of 4b and up (none at 4a). Section 16's open question (part photos, part network) closes on
  the network side: the weights carry it; the rotated-image Kbar keeps 1.21 because the weights do not rotate.
- RESULT B5 (a) (`logs/c7_reader.log`): in-span restriction of the stationary + window reader-level objective .0507
  (se .0030): vs Kbar build +.0016 (se .0002). PREDICTION (within .003) HOLDS: the approximation is admissible.
  CG: in-span ended rel. residual 2.4e-4 (rose from 4e-6 at it 400: lost conjugacy, ran to cap); per-reader 3.6e-4.
- RESULT B5 (b): per-reader minimiser .0543 (se .0031): vs its own in-span restriction +.0036; vs T +.0139. PREDICTION
  (<= .0430) REFUTED: per-reader freedom made it WORSE under the averaged objective.
## B5 reason (card, loop step 7-8): three candidate reasons, one measurement (`c10_objective.py`)
- R1 (estimation): 173k unknowns exploit noise in K (2000 images, <= 9 directions each at reader level) -> under an
  INDEPENDENT K estimate (2000 other train images) the per-reader solution loses its advantage over in-span.
- R2 (model): stationarity + window + |d| <= 6 are admissible after projection on the span but not per reader.
- R3 (factorisation): averaging the metric over images (E K_x (x) E zz^T instead of E[K_x (x) zz^T]) removes what
  training uses: the averaged objective does not credit training's reader-specific move.
- Measurement: value of the averaged objective L_avg for S, S+rem, T, c7_span, c7_reader under K_a (the c7 estimate)
  and K_b (independent 2000 images), with the same Y(d).
- PREDICTIONS: R1 -> L_a(c7_reader) << L_a(c7_span) but L_b(c7_reader) >= L_b(c7_span). R3 -> L_a(S+rem) >= L_a(S) - 10%
  of the S -> c7_reader drop (training's move is invisible to the averaged objective). R2 -> neither: L_b ranks
  c7_reader best, and S+rem is credited, yet the KL disagrees.
- RESULT X-w4d-local (`logs/b3_w4dlocal.log`): no kick (dev-1000 .0557 -> .0477 -> ... -> .0455). dev-5000 .0411 (se
  .0021): -.0079 (se .0013) from its start .0489; vs best known T (.0404) +.0006 (se .0004, tie). PREDICTION (gain
  .008-.009, ends .040-.041) HOLDS (gain at the low edge). The DERIVED-metric build has the same local defect as the
  Kbar build: Gap B only, no extra defect from replacing the image metric by the weights.
- RESULT B5 reason (`logs/c10_objective.log`): cos(K_a, K_b) .867. Averaged objective L_avg:
  | | S | S+rem | T | c7_span | c7_reader |
  |---|---|---|---|---|---|
  | K_a | .0460 | .0615 | .0609 | .0457 | .0411 |
  | K_b (independent) | .0419 | .0560 | .0560 | .0417 | .0386 |
  R1 (estimation noise) REFUTED: c7_reader stays best under K_b. The averaged objective PENALISES training's move
  (+34%) while the true KL and the per-image L3 credit it (B2). So either R2 (stationary + window model) or R3
  (averaging the metric over images). Separate them next.
## B5 reason, step 8 (card): exact (non-stationary) averaged objective by samples (`c11_exactavg.py`)
- L_fact(B) = 1/2 E_x E_x' sum_c (G_c(x') . e_B(x))^2: metric images x' (500, rng 6 set), activity images x (500, rng
  11 order) independent, no stationarity, no window, no offset cut. Also the per-image L3 on the SAME activity images
  (x' = x) as the control.
- PREDICTIONS: R3 -> L_fact(S+rem) > L_fact(S) and L_fact ranks c7_reader best, while per-image L3 ranks S+rem below S.
  R2 -> L_fact(S+rem) < L_fact(S) (exact averaging credits training; only the stationary model failed).
- RESULT B5 step 8 (`logs/c11_exactavg.log`; 500 metric images x', 500 activity images x, independent):
  | | S | S+rem | T | c7_span | c7_reader |
  |---|---|---|---|---|---|
  | L_fact (exact averaged metric) | .0381 | .0523 | .0520 | .0381 | .0356 |
  | L_img (same image's metric) | .1059 | .0931 | .0923 | .1101 | .0951 |
  R3 HOLDS, R2 REFUTED: the exact averaged objective (no stationarity, window or offset cut) also penalises training's
  move (+37%); the per-image objective credits it (-12%). Gap B lives in the CORRELATION between each image's metric and
  that image's errors. Note L_img > L_fact for every build: on each image the output cares most where the errors are.
## C.3 (a) (card): two circuit states (`c12_states.py`, FITTED split, same 500 + 500 images as c11).
- State = top 5% of images by the per-image second-order cost of S (L_img(S; x)) vs the rest; same rule on metric and
  activity images. L_2s(B) = 1/2 sum_s pi_s E_{x in s} E_{x' in s} sum_c (G_c(x') . e_B(x))^2. Also top 10%.
- PREDICTION: L_2s credits training's move as L_img does: L_2s(S+rem) < L_2s(S) (by >= 5%), while one state (no split)
  = L_fact penalises it.
- RESULT C.3 (a) (`logs/c12_states.log`): top 5% states: S .0405, S+rem .0532 (+31%); top 10%: .0411 / .0545.
  PREDICTION (2-state objective credits training) REFUTED. The top 5% carry 65-76% of the per-image cost, yet splitting
  by image does not restore the link. Reason (to verify next): the link is WITHIN each image, position by position.
## C.3' (card): local window model (`c13_window.py`, same 500 + 500 images).
- K_x[(j,p),(j',q)] ~ a_x(p) a_x(q) Kt[(j,p),(j',q)], Kt = averaged metric after whitening each metric image by ITS own
  per-position window. Windows tried: (o) oracle a_x(p)^2 = sum_{c,j} G_c(x)(j,p)^2 (FITTED, the image's own output
  sensitivity); (h) a_x(p) = |h_F(x,p)| (the family's own signal: network activity, no gradient); (r) a_x(p) = norm of
  all 368 reader outputs at p.
- PREDICTION: (o) credits training's move (L(S+rem) < L(S) by >= 5%), proving the link is a per-image spatial window;
  (h) also credits it (the circuit is on where the curves are).
- RESULT C.3' (`logs/c13_window.log`; absolute scale arbitrary, ratios matter): S+rem / S = 1.368 (oracle window),
  1.139 (family signal), 1.416 (reader outputs); per-image objective .879 (c11). PREDICTION REFUTED for all three, even
  the oracle. Windows from activity correlate weakly with the oracle (.08, .10). Reason (statement 22.2): the per-image
  link is not a spatial weight but the per-image DIRECTION of the metric (rank <= 9: the gradients of the classes that
  compete on that image), i.e. the per-image gate pattern of the downstream circuit.
## I3 (card): which clean 4d units carry the coupling (`c14_units.py`, picture `cache/c14_units.png`)
- Rank 4d channels u by E_x |A_u r_x|^2 (level 0p residual r on 300 dev images, A_u = the weights-only linearised map
  from the family to channel u, all positions). Top 12: feature visualization (lucent, centre position) next to the 4
  strongest natural crops (2000 train images) and the 2 most negative pre-activation crops.
- PREDICTION: the top 12 carry > 30% of the total (of 528 channels), and >= 6 of them are round-shape units (circles,
  rims, wheels, eyes, dials, loops) in their feature visualization and crops.
- RESULT I3 (`logs/c14_units.log`, `cache/c14_units.png`): top-12 4d channels [229, 129, 228, 461, 115, 177, 390, 304,
  284, 244, 266, 220] carry .107 of the level-0 error energy that reaches 4d (PREDICTION > .30 FAILS: the coupling is
  spread over many units). Pictures (fv | 4 top crops | 2 most negative): round-shape units 229 (eyes), 129 (round
  knob, eyes), 228 (golf ball, round heads), 461 (eyes, ring), 220 (round speaker / dial / seal), 115 (spots), 390
  (snouts, an ammonite spiral) = 6-7 of 12 (PREDICTION >= 6 HOLDS). Also straight / box units 284 (parallel lines),
  244 (rectangular furniture), 304 (radial street lines): the round-vs-straight distinction. Negatives are mostly dog
  faces or cars (the essay's gate-in-front pattern: these units keep faces / vehicles out).
## COLD (card): sealed split (10 per class, 10000 images), one look. Builds: level 0p, 16.8 weights metric, 4a (exact
   pool), 4b, 4d, 5b, Kbar build, X-w4d-local, best known T.
- PREDICTION: 4d ties the Kbar build within 2 paired se; order 4a > 4b > 4d < 5b holds; X-w4d-local ties T.
- RESULT COLD (`logs/c15_cold.log`, sealed 10000): level0p .0585, 16.8 .0528, 4a .0509, 4b .0474, 4d .0456, 5b .0496,
  Kbar .0455, X-w4d-local .0390, T .0385. 4d - Kbar +.0000 (se .0003); 4a-4b +.0035; 4b-4d +.0018; 5b-4d +.0041;
  X-w4d-local - T +.0006 (se .0003). PREDICTIONS HOLD (the last one at 2 se: a tie at the edge). The CHOSEN depth 4d
  was picked on dev and holds on sealed.
## 22.2 positive check (card): class-matched metric (`c16_matched.py`). For each of the 500 activity images x, the metric
   comes from 2 OTHER train images of the same label (same competing classes); control: 2 random train images.
- PREDICTION (if 22.2 is right): matched pairing credits training's move, L(S+rem)/L(S) < 1; random pairing > 1 (as
  L_fact, 1.37). If matched > 1 too, 22.2 is wrong as stated.
- RESULT 22.2 check (`logs/c16_matched.log`): S+rem / S = 1.472 with same-label metric images, 1.275 with random ones
  (per-image own metric .879). PREDICTION (matched < 1) REFUTED. Statement 22.2 ("direction = which classes compete")
  is WRONG as stated. Only the image's OWN metric credits training's move. Gap B is located (second order, reader-
  specific, same-image) but its compact form is OPEN.

# GAP B + DEPTH (2026-10-24, Shonu: "figure out Gap B; make depth 4d calculated"; NEURON-MATH Part VII)
## F0 + F2 (card, `c17_fisher.py`): factors from the 2000 Kbar train images (TR); L_img on 500 of them; 5 builds + 4d.
- F0 PREDICTION: |grad_R L_img(T)| / |grad_R L_img(S)| < .3 (training ends near the full-Fisher OBS point).
- F2 PREDICTION: plain KFC ratio L(S+rem)/L(S) > 1 (like every averaged metric). EKFAC ratio < 1 (my belief: ~50/50).
  E-small: rho_mn from 256 images vs 2000: Spearman >= .8 over the top 1000 pairs.
- RESULT F0 / F2 (`logs/c17_fisher.log`, 2000 TR images for factors, L_img on 500 of them):
  | ratio to S | L_img | plain KFC | EKFAC (2000) | EKFAC (256) |
  |---|---|---|---|---|
  | S+rem | .908 | 1.073 | 1.021 | 1.019 |
  | T | .917 | 1.072 | 1.022 | 1.020 |
  | c7_reader (in-sample for L_img) | .936 | 1.070 | 1.138 | 1.130 |
  F2 REFUTED: the EKFAC diagonal correction does not credit training's move. E-small for rho: Spearman .982 (256 vs
  2000): s_mn is well estimated; it is the wrong object. F0 REFUTED as stated: |grad L_img(T)| / |grad L_img(S)| = .893;
  cos(-grad(S), T - S) = .070. Training's 500 small Adam steps do not reach the full-Fisher OBS point, and they do not
  follow the raw gradient (Adam rescales each coordinate).
## F2' (card, `c18_modes.py`): exact F on the top-K reader (circuit) modes x all inputs, KFC elsewhere; K = 5, 10, 20,
   40, 80; control: top-K activity modes exact. Same 500 images, same builds.
- PREDICTION: reader-mode ratio L(S+rem)/L(S) < 1 at K <= 20; activity-mode control needs a larger K.
- RESULT F2' (`logs/c18_modes.log`): ratio L(S+rem)/L(S) with top-K reader modes exact: 1.12 / 1.16 / 1.28 / 1.16 / 1.08
  (K = 5 / 10 / 20 / 40 / 80); activity modes 1.13-1.24. All 368 (per-image F): .908. PREDICTION REFUTED.
  Where the move lives (G-eigenbasis): modes 0-20 .10, 20-80 .26, 80-160 .29, 160-240 .20, 240-368 .15 of its energy;
  77% of the 5x5-bottleneck readers sit in modes 80-240. The credit needs the per-image COUPLING of mid modes with the top
  modes; no top-K subspace carries it.
## F4 (card, `c19_pcg.py`): exact Krylov solve of the true second-order objective (MICROSCOPE: Gauss-Newton family).
- min 1/2 v^T F_n v over the free reader coordinates, start S, F_n = per-image Fisher from n = 2000 TR images (g and y
  stored); preconditioner = plain KFC (G^-1 . A^-1, masked). Stop when the preconditioned gradient is within its own
  noise: |g|^2_P < tr(P^-1 Cov(g_x)) / n (calculated, no held-out data). Dev-5000 KL recorded along the path (not used).
- PREDICTION: the stop comes after 3-15 steps; the stopped build reaches <= .0420 (>= 80% of Gap B); the path keeps
  falling past T's .0404 in L but the dev KL turns up after the stop (over-fit beyond the noise line).
- RESULT F4 (`logs/c19_pcg.log`, n = 2000, stopped by me at it 10 to free the GPU): start |g|_P^2 / noise line = 1.7
  (true signal ~.7 of the noise). Dev-5000 KL: it 1 .0470 (S .0491: -.0021), it 2 .0479, it 3 .0480, it 5 .0489, it 8
  .0478, while L on the 2000 images falls .0999 -> .0192 (fits the sample). PREDICTION (<= .0420, stop after 3-15) REFUTED.
  The stop rule never fired (signal / noise stayed 1.3-2.2): as written it compares an in-sample gradient with the
  noise and is not a valid rule after the solve starts fitting the sample.
## DEPTH (card, `c20_depth.py`): gated transport to the logits. Code check: on 2 images the true-gate linear copy now
   matches the direct gradient to 2e-7 (first version 6%: max-pool ties at 0 pass gradient because J is taken at the
   reader OUTPUTS, gate-free; fixed by using torch's own pool indices).
- Variants (all to the logits, no depth cut): a1 true gates + per-image output metric (= Kbar, code check at scale);
  a2 true gates + image-averaged output metric; a3 true gates + uniform I - 11^T/n; b Bernoulli(image rates) + uniform
  pool choice + averaged output metric; b_uni the same with the uniform output metric.
- PREDICTIONS: a1 = Kbar (< 1e-5). a2 within 2 se of Kbar (.0491): per-image output weighting not needed once gates are
  true. a3 within .002 of Kbar. b within 2 se of the 4d build (.0489) and better than 4b (.0512); b worse than 4b
  REFUTES 26.2(b). Effective rank of b in the hundreds (not 8 like gate-free logits).
## 25.9 test (card, `c21_newton.py`): one KFC-preconditioned step from S along the gradient of the true second-order
   objective, exact line search (curvature from all 35000 train images); gradient from n = 2000 / 8000 / 35000.
- PREDICTION (from Prop. 25.8, n* ~ 2900): the gain grows with n; n = 2000 gains ~.002 (as F4 it 1); n = 35000 gains
  >= .004; the 2000-image direction has cos < .7 with the population direction, the 8000 one > .8.
- RESULT DEPTH (`logs/c20_depth.log`): a1 code check 5.5e-6 (= Kbar), build .0491. a2 (true gates + averaged output
  metric) .0491 (+.0000). a3 (true gates + UNIFORM output metric) .0490 (-.0001 vs Kbar, se .0001). b (Bernoulli at
  image rates, averaged metric) .0520 (+.0030 vs 4d, se .0006); b_uni .0519. Effective ranks: a1 324, a2 565, a3 568,
  b 918, b_uni 916 (gate-free logits 8).
  PREDICTIONS: a1 HOLDS; a2 HOLDS; a3 HOLDS; b within 2 se of 4d REFUTED, and b is worse than 4b (.0512): 26.2(b)
  REFUTED. Reading: the output metric needs no image information once the gates are true; independent gates at the
  right rates over-whiten (rank 918 vs 568): the CORRELATIONS of the gates carry the information.
## GATE CORRELATIONS (card, `c22_shuffle.py`): true gate patterns, each gate layer (and pool layer) taken from an
   independent random image (within-layer co-firing kept exactly, cross-layer coherence destroyed); uniform output
   metric. Second variant: whole 4a-4e stage from one image, 5a-5b from another (coherence kept within stages).
- PREDICTION: layer-shuffle lands between Bernoulli (.0520) and true (.0490), about .0505: both within-layer co-firing
  and cross-layer path coherence matter. Stage-shuffle within .001 of true.
## NETMADE GATES (card, `c23_netmade.py`): Kbar's object per 26.3 = E_x A_x^T (I - 11^T/n) A_x over gate patterns. Take
   the gate patterns from the 256 cached class inversions (`e1_self_256`, fixed published rule, no photo) and carry to
   the logits: no depth cut. Control: the same with 256 real train images (separates count from source).
- Why it can work now: 26.3 shows the output metric can be uniform, so the network's certainty on its own inputs (the
  failure of the network-made Fisher write, P-Thm 24) no longer enters; only which gates open does.
- PREDICTION: real-256 within .001 of a3 (.0490); netmade-256 within 2 se of the 4d build (.0489) and better than 4b
  (.0512). If netmade is worse than 4b, gate patterns of class inversions are not typical (section 25's "modes are not
  typical") and the depth stays CHOSEN.
- RESULT 25.9 test (`logs/c21_newton.log`): one KFC-preconditioned step, exact line search on all 35000: n = 2000 .0469
  (-.0021 vs S, se .0006; cos with population direction .80), n = 8000 .0461 (-.0030; cos .94), n = 35000 .0460 (-.0031;
  vs T +.0055). PREDICTIONS: gain grows with n HOLDS (saturates by 8000); n = 35000 gain >= .004 REFUTED (.0031 = 36% of
  Gap B); cos(2000) < .7 REFUTED (.80); cos(8000) > .8 HOLDS. Reading: the gradient direction needs ~8000 images, not
  35000; one step is limited by the Kronecker curvature model (the step is along P^-1 g, not F^-1 g).
## KRYLOV-POP (card, `c24_krylov.py`): PCG on the true second-order objective with STREAMED matvecs over n = 8000 train
   images (no storage), KFC preconditioner, start S. Stop rule (calculated, no dev data): objective on 2000 OTHER train
   images stops decreasing. Dev-5000 KL recorded at each step.
- PREDICTION: the held-out objective falls for 3-6 steps; the stopped build reaches <= .0425 (>= 75% of Gap B);
  if it reaches <= .0404 it ties or beats the best known network with no gradient training.
- RESULT GATE CORRELATIONS (`logs/c22_layer.log`, `logs/c22_stage.log`, uniform output metric): layer-shuffled true gates
  .0499 (+.0009 vs true a3, se .0001; rank 643); stage-shuffled (4a-4e one image, 5a-5b another) .0491 (+.0001, se
  .0000; rank 543). PREDICTION (layer ~.0505; stage within .001) HOLDS (layer slightly better than predicted).
  Split of what gate correlations carry (Bernoulli .0520 -> true .0490 = .0030): within-layer co-firing .0021 (70%),
  cross-layer coherence inside 4a-4e .0009 (30%), coherence across the 4e -> 5a boundary .0000.
- RESULT NETMADE GATES (`logs/c23_netmade.log`, uniform output metric, carried to the logits, no cut): real-256 .0492
  (+.0002 vs a3); NETMADE-256 (class inversions, no photo) .0491: vs Kbar +.0000 (se .0002), vs 4d +.0001 (se .0004).
  PREDICTIONS HOLD. The depth is no longer chosen: the metric is carried to the logits through the gate patterns of the
  network's own inputs. Metric DERIVED (weights + the fixed e1 generation rule); activity moments still MIX-DERIVED.
## 26.3 for Gap B (card, `c25_unigap.py`): per-image objective with the UNIFORM output metric (10 random centred xi per
   image instead of the top-10 class covariance), 500 train images, builds S / S+rem / T.
- PREDICTION: it credits training's move like the per-image class metric (ratio S+rem/S < .95). If it does, the
  network's own inputs can supply Gap B's gradient (next check); if not, Gap B needs the image's class uncertainty.
- RESULT 26.3 for Gap B (`logs/c25_unigap.log`): per-image objective, ratio S+rem/S: class metric .908, UNIFORM metric
  .863 (T: .917 / .859). PREDICTION HOLDS: Gap B's objective needs no class uncertainty; it needs the same-image link of
  gate patterns and activity.
## NETMADE GAP-B STEP (card, `c26_netstep.py`): one KFC-preconditioned step from S, gradient AND line-search curvature of
   the uniform-metric per-image objective computed on network-made inputs only (256 class inversions; also 256 real
   images as the count control). Fully network-made direction and step length; activity enters only through those
   inputs.
- PREDICTION: real-256 gains ~.0015 (n = 2000 gave .0021); netmade-256 gains > 0 and >= half of real-256. If netmade
  makes it worse, class inversions do not carry the activity-gate 4th moment (modes are not typical, section 25).
- RESULT KRYLOV-POP (`logs/c24_krylov.log`): dev-5000 it 1 .0458, it 2 .0458, it 3 .0451, it 4 .0449 (held-out objective
  .1034 -> .0913 -> .0905 -> .0926 -> .0911); stop rule fired at it 2 (best by rule .0458, -.0033 vs S, se .0008).
  PREDICTION (<= .0425) REFUTED. Slow convergence: the KFC preconditioner is poor in Gap B's directions (it is Gap B).
- RESULT NETMADE GAP-B STEP (`logs/c26_netstep.log`): real-256 uniform-metric step .0465 (-.0026, se .0005; better than
  the n = 2000 top-10-class step, -.0021); netmade-256 .0489 (-.0002, se .0003). PREDICTION (netmade >= half of real)
  REFUTED: class inversions do not carry Gap B's activity-gate fourth moment.
## COLD-2 (card): sealed split, one look: netmade-256 metric build (26.6), real-256 metric build, Krylov it 2, the 35000
   one-step, the real-256 uniform step, plus 4d / Kbar / T references.
- PREDICTION: netmade ties Kbar on sealed (within 2 se); the Gap B steps keep their dev gains within 2 se.
## QUEUE (cards; predictions before runs)
- X-netmade-local: local exactness from `c23_build_netmade256` (lr 3e-4, 500 steps). PREDICTION gain .008-.009, ends
  .040-.042 (same as from 4d). More gain = a defect the dev tie hides.
- NETMADE-32: same metric from the 32 cached class inversions. PREDICTION within .001 of netmade-256 (E-small for the
  derived metric). The generator STEP-doubling check (e1 at 512 steps) stays OPEN.
- KRYLOV-FULL (`c28_krylov_full.py`): c24 with the FULL output covariance (10 sampled xi per image), 8000 train images,
  stop = 2-step patience on 4000 other train images. PREDICTION: closes >= 70% of Gap B on dev (<= .0430); if it stalls
  near 50% again, 25.10 (a) dominates and the remainder needs a better preconditioner, not more data.
- RESULT X-netmade-local (`logs/b3_netmadelocal.log`): .0491 -> .0403 (-.0088, se .0013); vs T -.0001 (se .0002).
  PREDICTION HOLDS: the network-made, no-cut metric has the same local defect as the image metric: Gap B only. The
  locally trained build from it ties the best known network.
- RESULT NETMADE-32 (`logs/c29_netmade32.log`): .0489 (-.0002 vs netmade-256, se .0002). PREDICTION HOLDS (E-small for
  the derived metric: 32 class inversions suffice). Step-doubling of the e1 rule: OPEN.
- RESULT KRYLOV-FULL (`logs/c28_krylov_full.log`): dev it 1-6 .0458 / .0460 / .0456 / .0462 / .0473 / .0469; stop at it 6,
  best held-out it 4 (.0462: -.0029 vs S, se .0013). PREDICTION (>= 70% of Gap B) REFUTED. 25.10(b) (truncation) is NOT
  the cause; 25.10(a) (Kronecker conditioning in Gap B's directions) stands. The sampled full-covariance objective is
  also noisier (held-out L jumps), so 8000 images x 10 sampled xi are too few for its curvature.

# NETWORK IS ENOUGH (Part VIII of NEURON-MATH, 2026-10-24 evening; plan `last-session-made-quite-lively-flute.md`)
Frame: coarse-graining whose target is the distillation optimum; training = approximate solver. "Gap B" renamed the
same-image term. Theory written first: 28.1 (moment equation), 28.2 (gain <= own loss), 28.3 (transfer bound:
coverage x leftover), 28.4 (root condition: q must reproduce the DIRECTION of g_P(S) = Cov(K_x, e_x y_R^T)).
## K0 (card, `c30_errenergy.py`): verify 28.5, the reason c26's class pictures taught nothing. Uniform metric (10 xi per
   input, as c26), build S, shared preconditioner M = c17 KFC (MIX-DERIVED tool, same for all sets). Sets: 256 class
   pictures (`e1_self_256`), 256 photos (c26 seed), 8000 photos (reference direction = K1, cached).
   Measures per set: error energy 2L_q(S) per input; share of the top 5% of inputs; Newton decrement g.M^-1 g; direction
   cos of M^-1 g with the 8000-photo direction.
- PREDICTION: (a) class pictures' error energy < 20% of photos'. (b) photos' top-5% share > 25%, class pictures' lower.
  (c) cos(photo-256, photo-8000) ~ .4 (noise ceiling at n = 256). Verdict rule, fixed now: reason "easy questions"
  (28.5) CONFIRMED if (a) holds and cos(class-256) >= half of cos(photo-256). Reason "wrong direction" if cos(class-256)
  < half of cos(photo-256). Both can hold.
- RESULT K0 (`logs/c40_errenergy.log`, 141 s). Uniform metric, build S, M = c17 KFC.
  | set | 2L per input | top-5% share | decrement g.M^-1 g | cos with photo-8000 direction |
  |---|---|---|---|---|
  | photo-256 | 1.20e4 | .243 | 1.63e13 | .938 |
  | class-256 | 7.08e4 | .169 | 6.84e14 | .313 |
  | photo-8000 | 1.19e4 | .258 | 9.29e12 | 1 |
  PREDICTION (a) REFUTED: class pictures carry 5.9x MORE error energy than photos, not < 20%. (b) photo top-5% .24-.26
  (< my .25 guess at 8000: borderline), class lower .17 (holds). (c) photo-256 cos .938, far above my ~.4: with the
  uniform metric 256 photos already fix the direction (25.8's "8000 images" was the top-10 metric). Verdict rule:
  "easy questions" (28.5) REFUTED; "wrong direction" CONFIRMED (.313 < .469).
## K0b (card, `c41_split.py`): verify 28.7 (paper). Shuffle estimator: gates/Jacobian from input x, activity from input
   x' = 4 cyclic shifts of the same 256 (unbiased for the factorised part g^fact). g^cov = g - g^fact. Sets: photo-256
   (c26 seed), class-256. Reference: cached photo-8000 raw direction (`c40_photo8000_grad.pt`).
- PREDICTION: (i) photos |g^fact|_M/|g|_M < .3; (ii) class > .7; (iii) cos(M^-1 g^cov_class, photo-8000) > .6.
  If (iii) fails while (ii) holds: the class pictures' same-image covariance itself differs -> write that reason next.
- RESULT K0b (`logs/c41_split.log`, cyclic shifts 1-4):
  | set | |g_fact|/|g| | |g_cov|/|g| | cos raw | cos fact | cos cov (vs photo-8000) |
  |---|---|---|---|---|---|
  | photo-256 | .181 | 1.041 | .938 | -.198 | .929 |
  | class-256 | .478 | .904 | .313 | -.425 | .589 |
  cos(cov photo-256, cov class-256) .634. (i) HOLDS; (ii) REFUTED (.478, not > .7); (iii) narrowly fails (.589 vs .6).
  The factorised part is ANTI-aligned on class pictures (-.425); removing it doubles the alignment (.31 -> .59).
  ESTIMATOR FLAW found before the verdict: both sets are sorted by class (e1 labels 0, 4, 8, ...; photos sorted index),
  so shifts 1-4 pair similar classes: the pairs are not independent as Prop. 28.6 requires.
## K0b' (card): same, with 4 random derangements (rng 41) instead of shifts.
- PREDICTION: if class adjacency leaked, class-256 cov cos moves by > .05 (direction unknown); photo numbers move < .03.
- RESULT K0b' (`logs/c41_split_rand.log`, random derangements): photo-256 fact share .146, cov cos .939 (fact cos .070);
  class-256 fact share .370, fact cos -.315, cov cos .492; cos(cov photo, cov class) .539. PREDICTION HOLDS: class cov cos
  moved by -.10 (adjacency leaked); photo moved < .03. Final verdict 28.7: (i) HOLDS (.146), (ii) REFUTED (.370),
  (iii) REFUTED (.492 < .6). The factorised contamination is real and anti-aligned; removing it lifts .31 -> .49. The
  class pictures' same-image covariance itself only half-matches the photos'.
## K0c (card, `c42_contour.py`): verify 28.8 (paper): split each push into same-position (p = p') and cross-position
   (p != p') parts, raw and factorised (random derangements), cov = raw - fact, per part. Reference: photo-256 parts.
- PREDICTION: (a) photos: the cross-position part carries > 50% of |g_cov|_M. (b) class pictures: cross-part cov cos
  with the photo cross part < .4, while the same-position cov cos with the photo same-position part > .6.
- RESULT K0c (`logs/c42_contour.log`): cov share in cross-position part: photos .973, class .984. cos(class, photo):
  same-position .761, cross-position .531, total .539. (a) HOLDS (photos' term is 97% cross-position = contour coupling,
  as 16.6). (b) REFUTED: class pictures HAVE the cross-position part (98%); it is half-aligned (.53), not missing.
  28.8 REFUTED. Three reason-checks in a row -> stop probing; write the paper statement (rule).
## K1' (card, `c43_covstep.py`): the first network-made same-image step (28.4 + 28.6): direction d = -M^-1 g^cov_class
   (class-256, random derangements, `c41_split_rand.pt`). Step length (A) network-made: a = -g^cov_q.d / d^T F_q d on the
   256 class pictures; (B) FITTED microscope: line search on photos (8000-photo gradient, 2000-photo curvature).
- PREDICTION: (B) gain ~ cos^2 x photo-direction gain = .24 x .0031 ~ -.0007 (range -.0004 to -.0012). (A) smaller
  than (B) (class pictures' curvature is inflated by their 5.9x energy), so -.0002 to -.0007.
- RESULT K1' (`logs/c43_covstep.log`): cos(d, photo dir) .492. (A) network-made length: dev .0483, -.0008 vs S (se
  .0003); (B) photo length (FITTED): -.0009 (se .0003). Step lengths almost equal (1.04e-11 vs 1.16e-11), although the
  class pictures' curvature along d is 22x the photos' (the gradient scales with it). PREDICTION (B ~ -.0007; A -.0002 to
  -.0007) HOLDS for B, A slightly better than predicted. The gain follows cos^2 (.24 x .0031 = .0007). Network-made
  same-image step: 4x the c26 step (-.0002), 31% of the photo-256 step (-.0026). The gap is direction only.
## K0d (card, `c44_within.py`): verify 28.10. 256 photos = 128 classes x 2 (train split, rng 44). raw g; same-class
   pairing (gates of one photo, activity of its class partner) = E_y E[K|y]E[Z|y]; 4 random derangements = E[K]E[Z].
   within = raw - same; between = same - random. Compare with class-256 cov (c41_split_rand).
- PREDICTION: |within|_M / |within + between|_M > .6; cos(class cov, photo between) > cos(class cov, photo within).
- RESULT K0d (`logs/c44_within.log`): |within|/|cov| .957, |between|/|cov| .468, cos(within, between) -.04. cos with
  photo-8000: within .846, between .089. cos(class-256 cov, photo within) .305, (.., photo between) .096.
  PREDICTION part 1 HOLDS (within > 60%); part 2 REFUTED (class pictures align with WITHIN, not between).
  Reading: the photos' same-image term is class-INDEPENDENT (between-class part orthogonal to the target). The class
  pictures carry part of it through their content, not through their label. Questions need no class at all.
## K2 (card, `c45_readerq.py` -> `cache/c45_readerq_368.npy`; then `c46_readerstep.py`): reader questions (paper K2 rule).
   Then the same pipeline as K0b'/K1': cov part (4 random derangements), direction cos with photo-8000, network-made step
   length, paired dev-5000.
- PREDICTION: cov cos > .6 (class pictures .49); dev gain better than the class-picture step (-.0008), target <= -.0013
  (half of photo-256). Falsifier: cov cos < .49 (no better than class pictures): then "one object, readers" is not the
  missing content; write why.
- RESULT K2 (`logs/c45_readerq.log` 998 s; `logs/c46_readerq368.log`): reader questions (368): |g_fact|/|g| .837 (fact cos
  .180); cos vs photo-8000: raw .430, cov .609. Network-made step 6.34e-11: dev .0479, -.0012 vs S (se .0002), +.0075 vs T.
  PREDICTION cov cos > .6 HOLDS (.609); dev <= -.0013 narrowly REFUTED (-.0012). Better than class pictures on every
  measure (cov .49 -> .61, dev -.0008 -> -.0012 = 46% of the photo-256 step). Gain again ~ cos^2 (.37 x .0031 = .0011).
  Pictures (scratchpad sheet): each reader question = one central patch (round, eye-like, spiral) in flat gray; class
  pictures = dense full-frame textures.
## K4 (card): step-doubling of the K2 rule: `c45_readerq.py 512` -> `c45_readerq_368_s512.npy`, then c46.
- PREDICTION: cov cos and dev gain within 2 se of the 256-step set (|dev diff| < .0004).
## K2-net (card, `c47_netprec.py`; advisor review): until now the step direction d = -M^-1 g^cov used M = c17 KFC from
   2000 PHOTOS (G and A factors). Fix: build both KFC factors from the reader questions themselves (uniform metric,
   same 1e-6 regularisation), then the same step. Also the count fairness check: a fixed 256-subset (rng 47) of the
   reader questions, with the photo M (comparable to the class-256 / photo-256 rows) and with the network-made M.
   Guard restated: the START S is still the Kbar build on 2000 photos' activity moments (MIX-DERIVED).
- PREDICTION: network-made M keeps the gain within 2 se of -.0012 (all 368). If it drops, say which factor (G or A)
  still needs photos (A = the open activity-moment object). Subset-256 with photo M: cov cos .55-.61, dev -.0009 to -.0012.
- RESULT K2-net (`logs/c47_netprec.log`): network-made KFC factors (from the 368 reader questions): dev -.0014 vs S (se
  .0003), -.0002 vs the photo-M step (se .0001). 256-subset: photo M -.0010, network-made M -.0012 (se .0003).
  PREDICTION HOLDS (within 2 se; the network-made metric is if anything better). The same-image step is now network-made
  end to end (gradient, preconditioner, length); the START S remains MIX-DERIVED (2000 photos' activity moments).
  Count fairness: at 256 the reader questions still beat class pictures (-.0012 vs -.0008, both network-made length).
## K5 (card, `c48_cold.py`): sealed split, ONE look. Builds: photo-256 step (c26 recipe), class-256 cov step (c43 A),
   reader-368 cov step with photo M (c46), reader-368 cov step with network-made M (c47). References Kbar (S) and T from
   `c27_cold.npz` (same sealed order).
- PREDICTION: every step keeps its dev gain within 2 se on sealed; reader-368 netM about -.0012 to -.0016 vs Kbar .0455.
- RESULT K5 (`logs/c48_cold.log`, sealed 10000, one look): vs Kbar (S, .0455): photo-256 step -.0019 (se .0004);
  class-256 cov -.0008 (.0002); reader-368 photo M -.0010 (.0002); reader-368 network-made M -.0011 (.0002) = .0445;
  T .0385. PREDICTION: all within 2 se of dev HOLDS (reader netM dev -.0014 vs sealed -.0011); the stated range
  -.0012 to -.0016 for reader netM narrowly missed (-.0011). Sealed ratio network-made / photo-256 = 58% (dev 54%).
## K6 (card, `c49_unitq.py mixed4e 256 256` -> `cache/c49_q_mixed4e_256.npy`, then `c46_readerstep.py`): statement 30.2
   (paper): questions at the top of the coherent circuit stage (4e), 256 of 832 units (rng 49), centre position.
- PREDICTION: cov cos > .75 (reader questions .61); network-made step (photo M, comparable to c46) better than -.0012 on
  dev. Falsifier: cov cos <= .61: context through the circuit's depth is not what reader questions miss; write why.
- K4 PAUSED (96/368 done): GPU shared with K6; step-doubling will run on whichever question rule wins K6.
## C1 (card, `c50_exactcg.py photos 2000 photo2000`): statement 31.3 (paper section 31). PCG from S on the second-order
   objective with the EXACT uniform output metric (JVP + VJP, Prop. 31.2), 2000 train photos, held-out 1000 photos
   decides the stop; preconditioner = c17 KFC. MICROSCOPE (FITTED: photos) to learn whether gap C is a metric-rank
   artifact. Self-check first: exact vs 50 sampled xi on 8 photos, cos > .9.
- PREDICTION: held-out objective falls for >= 5 iterations (c24 rose after 2); best dev beats the c24 stall (-.0033 vs S)
  and reaches <= .0440 (at least half of S -> T, which is .0491 -> .0403). Falsifier: stalls by it 3 again: then rank is
  not the cause; write the next reason.
- INTERIM C1 (`logs/c50_photo2000.log`): self-check exact vs 50 sampled xi cos .905, norm .949 (passes). Held-out L
  576 -> 540 -> 521 -> 504 -> 492 -> 484 -> 474 -> 468 -> 464 (still falling at it 8); dev .0463 / .0458 / .0446 / .0436 /
  .0431 / .0430 / .0427 / .0425. PREDICTION HOLDS so far (>= 5 falling its; <= .0440 by it 4). It 8 = 75% of S -> T
  (.0491 -> .0403) with 2000 photos and NO over-fit: 31.1/31.3 reading CONFIRMED (the stall was the rank-10 metric).
- RESULT C1 (final): it 9-15 dev .0425 / .0423 / .0428 / .0428 / .0435 / .0441 / .0452 while the held-out UNIFORM objective
  keeps falling (458 -> 447). Best dev at it 10: .0423 = 77% of S -> T, closed-form solve, no training. The uniform-metric
  stop rule picked it 15 (-.0039 vs S). PREDICTION (>= 5 falling its, <= .0440) HOLDS; new fact: the uniform objective
  and the true KL part ways after it 10.
## C1b (card, `c51_whichobj.py`): reason check. For each saved iterate V_k (k = 0..15) on 1000 held-out TRAIN photos
   (the c50 held-out set): true KL; second-order objective with the KL's own metric (top-10 class Fisher, c17 gammas);
   uniform objective (exact, c50). Reason (a) metric mismatch predicts: the class-metric objective bottoms near it 10 like
   KL. Reason (b) second-order breakdown predicts: the class-metric objective keeps falling past it 10 while KL rises.
- FAST CHECK C1-path (`logs/c52_path.log`, cached iterates): |V_k - S| / |T - S| (plain) 8.6 -> 51 (it 10) -> 73 (it 15);
  cos with T - S: plain .02-.03, cost metric (G (x) A) .58-.60 (its 1-4) falling to .32 (it 15); projection on T - S
  .20 -> 1.41 (it 10) -> 1.53. The solve heads toward training's move (and goes 1.4x as far by it 10) but adds HUGE moves
  in near-flat directions (50x training's whole move in plain norm).
## C2 (card, `c50_exactcg.py photos 2000 photo2000_klg kl gated`): statement 31.5 (paper). PCG with the TRUE second-order
   Taylor expansion of KL: exact softmax metric diag(p) - p p^T (all 1000 classes) and the readers' gates D in the
   tangent and the gradient. Same 2000 / 1000 photos, same start, same preconditioner. MICROSCOPE (photos).
- PREDICTION: no late dev rise (held-out objective and dev KL fall together until the stop); plain move size at the
  best iterate < 10x |T - S| (gate-free path: 51x); best dev <= .0423 (gate-free best). Falsifier: dev still turns up
  while the held-out objective falls: then the breakdown is beyond second order (gate flips), write that.
- RESULT C1b (`logs/c51_whichobj.log`, 1000 held-out train photos): true KL .0421 (S) -> .0394 (it 8, lowest) -> .0446
  (it 15); class-metric 2nd-order objective .0790 -> .0688 (it 8) -> flat .0677-.0682 (its 11-15); uniform keeps falling.
  Reason (a) metric mismatch REFUTED (the KL's own metric does not see the rise either); the rise is outside every
  gate-free 2nd-order model -> gates (31.5 mechanism).
- PARTIAL C2 (stopped after it 2, Shonu: one job at a time, smaller tests): gated + exact-softmax solve: held-out gated
  objective .0450 -> .0429 -> .0424 but dev .0493 / .0504 (S .0491): WORSE from the first step. 31.5's prediction ("no
  late rise") REFUTED in its fixed-gate form.
- Process change (Shonu, 2026-10-24 night): smaller test sizes, CPU math first, ONE GPU job at a time. 4e question
  generation stopped at 224/256 (K6 not scored).
## C3 (card, `c53_lex.py`, SMALL: 300 held-out train photos, cached path, one GPU job): statement 31.8. For V_0..V_15:
   true KL, L_ex (Def. 31.7, uniform output metric, exact reader ReLU), and the gate-free uniform model, same 300 photos.
- PREDICTION: argmin of L_ex within 1 iterate of the true-KL argmin (it 8 on the 1000-photo set); L_ex rises from it 11
  to 15 by > 5% while the gate-free model keeps falling. Run time < 4 min.
- RESULT C3 (`logs/c53_lex.log`, 300 photos, 5 min): true KL noisy (lowest its 1-6, rises from it 9: .0367 -> .0423);
  L_ex 311 -> 263.7 (it 8, lowest) -> 288 (it 15, +7.9% from it 11); gate-free keeps falling (575 -> 441).
  PREDICTION HOLDS: L_ex tracks the late rise; the gate-free model does not. 31.8 CHECKED.
## C4 (card, `c54_gn.py`, SMALL): statement 31.9 (paper). Preconditioned nonlinear CG (Polak-Ribiere+, Pinv = c17 KFC)
   on L_ex (uniform metric, exact reader gates; its gradient uses the CURRENT gates), 500 train photos; step length =
   gated-quadratic estimate, halved until L_ex falls. Held-out L_ex on 300 photos; dev KL on a fixed 1000-image dev
   subset (paired with S on the same subset). 12 outer steps. MICROSCOPE (photos).
- PREDICTION: dev KL falls monotonically (no late rise) and passes the gate-free path's best on the same subset;
  plain move size stays < 15x |T - S|.
- CPU CHECK (cached directions, no GPU): the KFC-preconditioned step is 99.9% BIAS (constant input column): A[const] = 1
  vs median channel 7.6e4, so A^-1 magnifies bias moves ~7.6e4x. The reported "direction cos" was mostly the bias part.
  Split: bias part / non-bias part (preconditioned) / non-bias (plain gradient): photo-256 .94 / .88 / .86; photo
  within-class .85 / .74 / .77; reader questions .61 / .42 / .48; class pictures .49 / .35 / .32. Branch split: energy
  shares match photos (1x1 .76, pool .20); 5x5 bottleneck aligned for all sets (.92-.97); mismatch spread over 1x1 and pool.
- RESULT C4 (`logs/c54_ph500.log`, stopped after it 6: held-out L_ex rising). dev-1000 vs S: -.0037 / -.0040 / -.0076 /
  -.0076 / -.0077 / -.0072 (se .002-.004); held-out L_ex 275 -> 251 (its 4-5) -> 252; train keeps falling (over-fit at
  n = 500); |v-S|/|T-S| 17 -> 49. PREDICTION: monotone dev HOLDS within noise up to the held-out stop; "move < 15x"
  REFUTED (45x at the best): exact gates do not stop the drift along flat directions. Held-out L_ex is a usable stop.
## C5 (card, `c55_biassplit.py`, SMALL): which part of the same-image step gives the KL gain? Photo-8000 direction
   d = -M^-1 g (cached), split into bias part (constant columns) and weight part; step length for each from the uniform
   curvature on 500 photos (c26 recipe); dev-1000 (same subset as C4) with S and T on the same images. Also: plain-norm
   share of the bias columns in |V_k - S| for the c50 gate-free path (CPU, cached).
- PREDICTION: the bias part gives > 60% of the full step's dev gain; > 90% of the c50 path's plain move is bias.
- RESULT C5 (`logs/c55_biassplit.log`, 35 s): c50 path plain move is 99.5-99.9% BIAS (its 1-15). dev-1000: S .0517, T .0426
  (T - S -.0091). Photo step: full -.0028, bias part alone -.0025, weight part alone -.0028 (se .0011). PREDICTION HOLDS
  (bias part 89% of the gain; path > 90% bias). Bias and weight parts are nearly redundant in output space.
## C6 (card, `c54_gn.py ... centered`, SMALL): statement 31.10 (paper): the drift is the uncentered parametrisation. Same as
   C4 (exact-gate solve, 500 photos, dev-1000) with the CENTERED KFC preconditioner (means from A's constant column).
- PREDICTION: plain move at the best iterate < 5x |T - S| (C4: 45x); held-out L_ex falls for more steps than C4 (4-5);
  best dev-1000 at least as good as C4's -.0077 (T - S = -.0091 on this subset).
- RESULT C6 (`logs/c54_ph500c.log`, stopped at it 6): centered KFC: dev-1000 vs S -.0025 / -.0025 / -.0046 / -.0047 / -.0052 /
  -.0046; held-out L_ex falls steadily 282 -> 257 (no over-fit by it 6); plain move 1500x -> 3000x |T - S|.
  PREDICTION REFUTED on every count (move size, speed). Reading: plain norm depends on the parametrisation (a centred
  bias carries W.mu), so "50x drift" (C1-path, C4) was never a meaningful measure; the damage measure is gate flips / KL.
  Slower per step than uncentered (C4 -.0077 at it 5). Stop chaining solver variants: 31.10 goes back to the paper.
- RESULT C-score (`logs/c56_score.log`, dev-5000, derived stops): C1 it 8 (2000 photos, gate-free exact uniform) .0425,
  -.0066 vs S (se .0013) = 75% of S -> T; C6 it 6 (500 photos, exact gates, centred) .0437, -.0054 (61%). C4's iterates
  were overwritten by C6 (argv bug in c54_gn.py: TAG reset by the borrowed header; fixed).
## 32.2 checks (CPU on cached tensors + one 32-s GPU pass `c57_phi.py 368`, smoke 16 first)
- Imprint vs photo |c|: Spearman .08 (branches -.29 to +.15). Per-question phi (368 x 368), self-check vs cached readerq c
  .945. Weighted sums vs c_photo8000: uniform .504; D/E (32.2) -.055; D .142; |u|^2 .499; 1/|phi_j| .404; 1/energy .516.
  32.2 REFUTED; reason visible: question j's contribution is NOT about reader j (own-reader share of |phi_j|^2 .004).
- DIAGNOSTIC (photo-fitted, non-negative weights over the 368 questions): fit on half the readers, score on the other
  half (5 splits): .781 (sd .048) vs uniform .544 vs photo-256 ceiling .907. Fitted weights spread (80% nonzero, top-10
  hold 14%), uncorrelated with every weights-only property tried (bias, |u|^2, D, energies: |rho| < .1), mildly against
  loud questions (-.24 / -.31).
- RESULT few-photos (`c58_fewphotos.py`, 30 s): photo c from n photos vs 8000: n 32 .55, 64 .60, 128 .84. Questions
  weighted by those n photos (held-out readers): .32 / .36 / .56, worse than the n photos' own c. Reader questions at
  uniform weight (.54-.58) are worth about 32-64 photos for this term. Calibrating frequencies with few photos does not
  beat using the photos directly.
