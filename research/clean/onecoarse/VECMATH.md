# VECMATH: deletion as vector math

**Index of all results and how they connect: `INDEX.md` (read it first).**

Track opened 2026-11-24 by Shonu. Plan: `~/.claude/plans/pasted-content-id-87d9-okay-i-mutable-cerf-of-c4219106-a6df-4a36-9135-da85396fb9b4.md` (approved, with its SCALE-UP REQUEST).

| Rule | Meaning |
|---|---|
| One principle | One operation (rebuild D's write from kept firings), one error (KL to A). No stacked methods. |
| Emergent carriers | Every kept neuron may carry. Nothing hand-picked. Read afterwards who got the weight. |
| KL on two photo groups | "D fired most" (top 10% of D's summed firing) and "D barely fired" (bottom 50%). A rise on the second = harm to unrelated jobs. |
| Fold must shrink | No weight of D survives. Only existing weight slots of kept units change. |
| Essay when stuck | `../curves/map_agents/essay.md` |

D = the 11 R1 children of s3:69 in A (stage-4 block-0 neurons n305, n97, n23, n163, n385, n297, n464, n237, n108, n510, n275), so earlier numbers stay comparable.

## 1. What others found (full texts read 2026-11-24)

| Paper | What it deletes | How it repairs | Error it minimizes | Before fine-tuning |
|---|---|---|---|---|
| ZipLM (Kurtic, Frantar, Alistarh, NeurIPS 2023, arXiv 2302.04089) | whole hidden neurons (FC2 columns), one at a time | all kept neurons' writes, by regressing on kept activations | stream size at that layer (curvature = activations x activations, damped) | BERT, 2x: 84.1 F1 vs 88.5 dense. With 4 calibration samples: 48.4 |
| Interpolative decomposition (Chee, Flynn, Damle, De Sa, arXiv 2108.00065) | many neurons per layer | each deleted firing = sum of kept firings (pivoted QR), folded into the next layer | the layer's own output after ReLU | "no loss up to 35% of FLOPs" (figures only). Theory for one hidden layer. No ResNet |
| Channel pruning (He, Zhang, Sun 2017) | input channels | LASSO picks, least squares refits all kept filters | that conv's output. Target from the original net | VGG-16 top-5 drop 2.7 / 7.9 / 22.0 at 2x / 4x / 5x. ResNet-50 2x: 4.0 |
| ThiNet (Luo, Wu, Lin 2017) | channels | one scale number per kept channel | next layer's pre-ReLU output | fine-tunes after every layer; almost no numbers without |
| OBC (Frantar, Alistarh 2022) | single weights | all kept weights in the row | layer output (H = 2XX^T) | ResNet-50 4x unstructured: 74.05 vs 76.13 |
| Optimal Brain Surgeon (Hassibi, Stork 1993) | single weights | all kept weights: dw = -(w_q / [H^-1]_qq) H^-1 e_q | the final output; H = sum over inputs of output-gradient outer products | small nets (MONK, XOR), exact |
| Neuron Merging (Kim et al., NeurIPS 2020) | neurons | the one most cosine-similar kept neuron, by weights, no data | none | ResNet-34 30%: ~37 vs 73.3 |

| Part of this track | Already done? |
|---|---|
| V-known: regress each deleted firing on all kept firings, add to kept writes | Yes. ZipLM's structured step, interpolative decomposition. Our baseline |
| V-KL: the same rebuild, error = final KL, curvature per photo | Not found. Only OBS uses output curvature per input, for single weights in small nets |
| Every kept neuron allowed, then read which carriers emerge | Not found |
| The skip path as a repair route | Not found. He et al. leave block outputs alone. Neuron Merging prunes only inner layers |
| A small cut (2% of one layer) against full retraining at the same cut | Not found. Every paper cuts 30-75% and fine-tunes after |

Why the method is not standard (Minitron, Sheared LLaMA, Liu et al. 2019, Blalock et al. 2020, SliceGPT): the target is final accuracy after retraining; retraining is cheap (Sheared LLaMA: ~3% of from-scratch compute); the loss right after a cut does not predict the loss after retraining (Minitron); for structured cuts, training the small net from zero matches pruning + fine-tuning (Liu et al.); closed-form repair is used where retraining is impossible (OBC, SliceGPT). Cale's case: a size switch must be instant with no stored extra networks, so a repair that reaches training without training is what Cale needs.

Lessons used here: damping in every solve; remove the 11 as one group; check the fit-set size (ZipLM collapsed at 4 samples); compare with R, not only with delete.

## 2. The parent as the children's one shared read direction (`code/oc_vm_parent.py`, Mac, weights only)

Shonu's definition, in vectors. Unit reads r_i, Gram G_ij = r_i . r_j. A unit p with r_i . p = c_i gives q_i = r_i - c_i p and q_i . q_j = G_ij - c_i c_j. So "removing the parent leaves the reads orthogonal" means: G off the diagonal = c c^T. Fit c by least squares on the off-diagonal. The part of p inside the children's span is fixed by c. The part outside is free.

Control: 200 random 11-sets of block-0 neurons outside the top 30 R1 (seed 0).

| Statistic | Siblings | Random median | Random 99th | Sibling percentile |
|---|---|---|---|---|
| Reads: mean overlap (cos), before | .050 | .023 | .044 | 99.5 |
| Reads: shared part (mean c_i c_j) | .050 | .020 | .043 | 99.5 |
| Reads: overlap left after removing p (median) | -.002 | .001 | .020 | 21.5 |
| Reads: top channel's share of p | .199 (s3:69) | .029 | .069 | 100 |
| Writes: mean overlap, before | .038 | .007 | .023 | 100 |
| Writes: shared part | .037 | .004 | .021 | 100 |
| Writes: overlap left after removing the shared write (median) | .002 | .001 | .011 | 60 |

Each child's share of the read direction c: n305 .30, n97 .27, n23 .22, n163 .24, n385 .26, n297 .33, n464 .14, n237 .23, n108 .11, n510 .19, n275 .18. Of the write direction: n305 .30, n97 .15, n23 .04, n163 .23, n385 .31, n297 .23, n464 .11, n237 .33, n108 .10, n510 .18, n275 .15.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | p's largest channel is s3:69, with at least half of p's energy | s3:69 is first, but with .199 of p's energy. Next: s3:35 .022, s3:92 .017, s3:248 .017, s3:95 .016, s3:86 .013. Random sets' top channel: .029 median | **fail** on the share; s3:69 emerges as the top channel, 7x the random level |
| P2 | after removing p, sibling overlaps fall to the random level | .045 -> -.002 (random sets after their own fit: .001) | **pass** |
| P3 | shared part above the 99th pct of random sets | .050 vs .043 (99.5th pct) | **pass**, narrowly |
| P4 | no shared write direction above random | shared write .037 vs random 99th .021 (100th pct) | **fail**: the siblings DO share a write direction |

Reading, at true strength:
1. Shonu's definition works on the reads. One direction explains the siblings' extra read overlap. Removing it leaves their reads as orthogonal as random neurons' reads after the same fit. The direction found without being told has s3:69 as its largest channel, 7 times the random level.
2. But s3:69 is only a fifth of that direction. The other four-fifths are spread thin over many channels (none above .022), among them condition channels found before (s3:92, s3:248, s3:86). So "the parent" as a vector is wider than the one channel s3:69. Whether that wide part is real or noise in an 11-vector fit is open: the random sets' top channels show a fit of 11 random reads also concentrates by chance (top share .029, 99th .069).
3. The writes have a shared direction too, well above random. It is close to the plain mean of the 11 unit writes (cos .92). It is NOT the parent's own skip write (cos .17 with s3:69's 1x1 skip column, rank 9 of 256). Removing it leaves the writes orthogonal (median .002).
4. So the family has one shared read and one shared write. The coarse neuron's write in the morph (the children's summed write) points along this shared write.
5. Limit: one family, weights only. The random 99th row for "writes after, mean" is 108: one random set has a child almost entirely along p (c_i near 1), so the normalisation blows up. Medians are used.

## 3. Rebuild D from every kept firing (`code/oc_vm_rebuild.py`, cluster CPU)

Written before the run.

```
deleted neuron d sent:     V_d * a_d                      (conv2 write column, 3x3)
carriers F_j (758):        501 kept block-0 firings, 256 stage-3 channels at the skip positions, the constant 1
rebuild:                   a_hat_d = sum_j beta_dj F_j
kept slots that carry it:  block-0 carrier c: conv2 column += sum_d beta_dc V_d
                           skip carrier k:    1x1 skip column += sum_d beta_dk u_d   (u_d = V_d summed over 3x3)
                           constant:          bn2 bias += sum_d beta_d,const u_d
```

| Build | beta from |
|---|---|
| V0 | none (delete) |
| V-known | plain least squares of each a_d on all 758 F_j over DEV cells. The interpolative-decomposition / ZipLM-style fold |
| V-KL | the lowest KL by the square rule: near A, KL ~ 1/2 sum_k p_k (dz_k - p.dz)^2 with dz = J e. Exact normal equations over DEV (top 16 classes per photo, mass reported), solved directly |

Damping for each is chosen on a DEV split (fit 1,000 photos, check 1,000). Final fit on all 2,000 DEV photos. Scores on test photos only.

| Photo group | Rule (cut on DEV) |
|---|---|
| D fired most | test photos with D's summed firing above DEV's 90th percentile |
| D barely fired | photos of the 2,000 random test set below DEV's 50th percentile |
| old sets | test parent-strong (peak >= 2.92), 2,000 random test (seed 2) |

Carrier share (for emergence): share_j = sum_d |V_d|^2 beta_dj E[F_j a_hat_d] / sum_d |V_d|^2 E[a_hat_d^2] (sums to 1). Stop curve: keep the top k carriers by V-KL share, k = 1, 3, 10, 30, 100, 300, 758, solve again. Size curve: DEV = 250, 500, 1,000, 2,000 photos.

Predictions Q1-Q6 as in the plan: Q1 square rule within .05 of measured recovery; Q2 V-KL beats V-known on "D fired most"; Q3 no build above V0 on "D barely fired"; Q4 V-KL reaches the trained write-only morph (.73 parent-strong); Q5 s3:69's skip column is the largest carrier; Q6 n57 or n233 among n305's top carriers.

Limits written now: the skip carriers write a 1x1 collapse of V_d, so they cannot copy a 3x3 write exactly. The rebuild cannot leave the span of D's write columns (the trained write-only morph left the children's paths with ~75% of its write).

### Result (`code/oc_vm_rebuild.py`, cluster CPU job 6989688, 7 min, script md5 19af143a; `results/oc_vm_rebuild.npz` md5 8cc73ccf; read `code/oc_vm_rebuild_read.py`, Mac)

Sanity: base KL max 1.7e-6. V0 equals the old delete on the same test photos (.02297 random, .0571 strong vs `oc_morph2.npz` .02297, .05712). Top-16 class mass on DEV .94. Damping picked on the DEV split: known .01 (inside the grid), KL .1 (the grid's EDGE; held-out error still falling there, so the KL build is under-damped; rerun below).

Recovery = 1 - KL(build) / KL(V0), mean over photos, bootstrap 95% CI. "pred" = the square rule's prediction, from the DEV fit only.

| Photo group | n | V0 KL | V-known | pred | V-KL | pred | KL minus known |
|---|---|---|---|---|---|---|---|
| D fired most | 1,760 | .0647 | .775 [.758, .791] | .753 | .730 [.715, .745] | .712 | -.044 [-.053, -.035] |
| D barely fired | 1,018 | .0088 | .584 [.540, .625] | .599 | .464 [.409, .514] | .491 | -.120 [-.156, -.086] |
| parent-strong | 342 | .0571 | .774 [.741, .807] | .747 | .739 [.700, .777] | .698 | -.035 [-.056, -.015] |
| random 2,000 | 2,000 | .0230 | .711 [.685, .733] | .699 | .655 [.630, .676] | .646 | -.056 [-.070, -.042] |

Against the old builds, same photos:

| Build | Kept slots changed | Random | Parent-strong |
|---|---|---|---|
| calculated morph (keeps D's ch-69 kernels: not size-legal) | parent write | .432 | .678 |
| trained write-only morph | parent write | .516 | .734 |
| trained read + write morph | parent read + write | .564 | .758 |
| **V-known (no training)** | all kept write columns, skip columns, bias | **.711** | **.774** |
| V-KL (no training, under-damped) | the same | .655 | .739 |

Stop curve (recovery known / KL, top-k carriers by V-KL share): rises with k on every group, including "D barely fired". k=1: .30/.34 (fired), .19/.19 (quiet); k=30: .58/.53, .35/.33; k=300: .73/.69, .55/.43; all 758: .77/.73, .58/.46. No k where the quiet photos get worse: the stop point is not reached inside 758 carriers.

Size curve (DEV photos 250 / 500 / 1,000 / 2,000), random test: known .68 / .70 / .71 / .71 (flat from 500); KL .53 / .56 / .61 / .65 (still rising).

Carriers that emerged (V-known; share of the rebuilt signal): s3:69 first (.064), then the constant (.034), s3:134 .024, s3:71 .018, s3:249 .017, b0:n86 .016, s3:216, b0:n113, b0:n487. By kind: block-0 neurons .58, stage-3 skip channels .38, constant .03. 66 carriers hold half the positive share, 189 hold 80%.

Each child's own top carriers (V-known), against the private-read channels found from weights in LOG "Reads: why siblings split" (A2):

| Child | Top carriers (share of its rebuilt firing) | Its private read (A2) |
|---|---|---|
| n305 | s3:69 .14, s3:86 .12, b0:n245 .12, b0:n113 .07 | s3:86 |
| n97 | s3:69 .09, s3:198 .06, s3:139 .06, s3:134 .05 | s3:139 |
| n23 | b0:n86 .10, s3:134 .09, s3:69 .07, s3:86 .05 | s3:174 (down), s3:134 |
| n163 | s3:92 .10, s3:151 .05, b0:n487 .05 | s3:92, s3:184 |
| n385 | s3:174 .09, s3:216 .07, s3:132 .06 | s3:174, s3:184 (down) |
| n297 | const .18, s3:69 .12, s3:71 .10 | s3:174, s3:71 |
| n464 | b0:n154 .07, s3:107 .06, s3:192 .05, s3:69 .05 | s3:49, s3:250 |
| n237 | s3:69 .08, s3:141 .07, s3:85 .06 | s3:141 |
| n108 | b0:n496 .11, s3:67 .09, s3:150 .05 | s3:23, s3:150, s3:67 |
| n510 | s3:60 .10, s3:11 .07, s3:65 .06 | s3:65, s3:60 |
| n275 | const .26, s3:248 .11, s3:71 .11 | s3:248 |

| # | Prediction | Result | Verdict |
|---|---|---|---|
| Q1 | square rule within .05 of measured recovery | every row within .05 (largest gap .041, V-KL parent-strong) | **pass** |
| Q2 | V-KL beats V-known on "D fired most" | V-KL is LOWER by .044 [.035, .053] | **fail** (under-damped; rerun below) |
| Q3 | on "D barely fired", no build above V0 | mean KL .00365 (known) and .00471 (KL) vs V0 .00878. But the group was not quiet (V0 .0088). On the photos the deletion barely touched, both builds are worse than delete (C3) | **fail** on a truly quiet set, small in nats |
| Q4 | V-KL reaches the trained write-only morph (.73 strong) | .739 [.700, .777]; V-known .774 [.741, .807] | **pass** (reaches, not clearly beats; V-known beats it) |
| Q5 | s3:69's skip column is the largest carrier | first for both builds, but only 6.4% (known) / 2.4% (KL) of the signal | **pass**, weakly: the parent is first, the rebuild is spread |
| Q6 | n57 or n233 among n305's top carriers | not in its top 6. s3:86 itself (the condition they read) is n305's 2nd carrier | **fail** as written; the condition channel carries directly through the skip |

Reading, at true strength:
1. The known simple method, with every kept unit allowed, rebuilds 71% of the loss on random photos and 77% on parent-strong photos, with no training and no weight of D kept. That is above the parent-only trained builds (.52-.56 random). Those had one carrier and this has 758, so it is not a same-form comparison: no same-form trained reference exists yet (R, still queued, also retrains reads). It is size-legal by count of weights; see C1 for what the skip part stores.
2. Emergence (CORRECTED by the checks below): the parent s3:69 is the largest single carrier, and each child's top skip carriers are its own condition channels. But this is mostly built in. D was picked as the top readers of channel 69, and a least-squares fit of a neuron's firing on its own input channels returns roughly its own read weights (check C1: cos .76). The part selection does not explain is the block-0 carriers (check C2).
3. But the rebuild is spread: the parent is 6% of it, and 66 carriers hold half. The constant (bias) is large for n297 and n275: part of their job was a steady average output.
4. The square rule (KL near A = a quadratic in the stream error) predicts every measured recovery within .05 on unseen photos. So a deleted set's loss, and any linear rebuild's loss, can be predicted without running the network. Only the DEV fit is needed.
5. The KL version lost to the stream-size version here. The wider damping grid (rerun below) confirms damping is not the cause: .1 is a true minimum. Its fitting-set curve is still rising at 2,000 photos, so it is short of data: 2,000 photos x 16 class rows do not pin down 8,338 numbers, while the stream fit has 98,000 cells for each child's 758. Open, not a verdict on the KL objective.
6. Unrelated jobs (CORRECTED by check C3): the "D barely fired" group was not quiet. D's deletion costs .0088 there. On truly unaffected photos the rebuild DOES hurt: see C3.
7. Not yet compared with R (full retraining at the same cut): job `oc_retrain` is still queued.

### Rerun with a wider damping grid (job 6989769, script md5 bd546eb5)

Grid 1e-6 to 100 (times the mean diagonal). Held-out DEV error for the KL solve: ... 8.70 (1e-3), 6.65 (1e-2), **6.40 (.1)**, 6.61 (.3), 7.12 (1), 7.88 (3) ... 10.51 (100). The pick stays .1, now an interior minimum. The known solve stays at .01. The builds are therefore the same as in the first run. The KL loss is a data limit, not a damping limit.

### Checks after the advisor review (`code/oc_vm_check.py`, Mac, saved data)

C1, is the skip part of the rebuild the child's own read? Cosine between each child's 256 skip coefficients and its own read kernel:

| | Own read, centre tap | Own read, 3x3 summed | A sibling's read | A random kept neuron's read |
|---|---|---|---|---|
| V-known | .71 (.53-.76) | .76 | .05 | .00 |
| V-KL | .37 | .45 | .01 | .00 |

So the skip part of V-known is close to a ReLU-free copy of each child: its read (centre tap of a 3x3 kernel) times its write (summed over 3x3), added to the existing 1x1 skip matrix. It uses an existing slot and adds no weights. But it carries the child's read-write pair in linear form. Shonu rejected a design that kept the children's write columns; this one stores a linear copy of them in the skip. He should judge whether it is allowed.

C2, the block-0 carriers (.58 of the rebuilt signal), which selection does not explain. For each child, the read similarity (channel 69 zeroed) of its top 5 block-0 carriers, as a percentile among all 501 kept neurons: median .98. Examples: n305 -> n245 (.99), n113 (.98), n364 (1.00); n23 -> n86 (1.00), n191 (.99); n464 -> n154 (1.00), n5 (1.00); n108 -> n496 (1.00). So the carriers are kept neurons that read what the child reads besides the parent, and so fire when it fires. Q6 named n57 / n233 for n305 and was wrong on the names; the principle (carriers read the child's condition) holds for all 11 children.

C3, photos the deletion barely touched:

| Random test photos, binned by delete harm | V0 KL | V-known | V-KL | share worse than delete (known) |
|---|---|---|---|---|
| lowest quarter | .00017 | .00027 | .00032 | .39 |
| 2nd quarter | .0024 | .0020 | .0025 | .26 |
| 3rd quarter | .0116 | .0050 | .0067 | .10 |
| 75-90% | .0343 | .0100 | .0122 | .01 |
| top 10% | .143 | .033 | .037 | .01 |

Lowest decile of D's summed firing (217 photos): V-known .0030 vs V0 .0040 on the mean, but 44% of photos worse (mean rise .0025).

So the rebuild DOES write where it should not: on the quarter of photos the deletion barely touched, KL rises from .00017 to .00027 (known) and .00032 (KL). Small in nats, but real, and it is exactly Shonu's "BAD thing". The stop curve did not show it because it averaged over groups that were not quiet.

C4, absolute KL left: random .0066 (known) / .0079 (KL) vs delete .0230; parent-strong .0129 / .0149 vs .0571.

Corrected verdicts: Q3 **fail** on a truly quiet set. Q5 **pass, but built in** (selection). Q6 **fail on names, principle holds** (carriers read the child's condition, 98th percentile).

## 4. Essay check and what is left

Essay line 617: "established routes have a head start in capturing residual error." Line 777: "what wins is whatever captures the residual error first ... what usually wins is the right area for the thing." The rebuild agrees in one part and not in another:
- Agrees: the deleted job lands on routes that already exist and already fire at the right times: kept neurons that read the child's own condition (C2, 98th percentile), and the condition channels and the parent through the existing skip. Nothing new is grown. These are "the right area for the thing".
- Does not yet agree: the rebuild is spread (66 carriers hold half) and it leaks onto photos the deletion did not touch (C3). The essay's picture is a local niche. The spread may be the ReLU: a carrier's firing matches the child's only in part, so least squares mixes many partial matches. That is a guess, not a measurement.

Left for the next plan:
1. The leak (C3). The simplest fix inside one principle: fit only where it matters is wrong (it ignores harm), so the fix must come from the error itself. Option: weight the fit by how much each photo's answer depends on the stream (the KL version does this, but it is data-limited). Shonu's call.
2. The skip part (C1) stores a linear copy of each child. Shonu decides if that is allowed.
3. The part of D's firing that no kept firing covers needs a change of some kept neuron's READ. Not started.
4. R, when it runs: compare unit by unit.
5. The KL version needs more fitting photos than held DEV (2,000). The train cache is outside this plan's photo scope.

## 5. Every gate against the parent's gate, skip copy removed (`code/oc_vm_noskip.py`, cluster CPU job 6990459, 2.4 min, script md5 643e3ded; `results/oc_vm_noskip.npz` md5 3c97cf46; read `code/oc_vm_noskip_read.py`, Mac)

Shonu's question: "compare the actual accuracy of writing if you let it choose everywhere versus if you constrain it to a gate and if the difference is noise or not." Every build uses the exact write solve (least squares), damping picked again per gate set on the DEV split. Same test photos as section 3. Recovery = 1 - KL(build) / KL(delete). 95% CI from a paired bootstrap over test photos (2,000 resamples).

| Gate set | Gates | Parent-strong (342) | D fired most (1,760) | Random (2,000) | D barely fired (1,018) |
|---|---|---|---|---|---|
| all758 (section 3, with skip copy) | 758 | .774 [.737, .807] | .775 [.758, .790] | .711 [.686, .734] | .584 [.539, .627] |
| b0p: kept block-0 + parent's 1x1 column | 502 | .757 [.715, .793] | .741 [.726, .756] | .702 [.682, .722] | .629 [.593, .663] |
| b0: kept block-0 only | 501 | .760 [.724, .792] | .738 [.723, .752] | .704 [.685, .723] | .634 [.598, .667] |
| p: parent's 1x1 column only | 1 | .316 [.256, .374] | .300 [.278, .322] | .229 [.199, .259] | .192 [.162, .223] |

| Difference | Parent-strong | D fired most | Random | D barely fired |
|---|---|---|---|---|
| b0 minus p (every gate minus the parent's gate) | .445 [.395, .499] | .437 [.419, .457] | .475 [.449, .500] | .442 [.404, .479] |
| all758 minus b0p (what the skip copy adds) | .018 [.005, .029] | .033 [.026, .040] | .009 [-.002, .019] | -.045 [-.069, -.023] |
| b0p minus b0 (what the parent's column adds) | -.004 [-.018, .010] | .003 [-.001, .007] | -.002 [-.008, .003] | -.005 [-.011, .002] |

Harm by D's summed firing (random test photos, DEV quantile bins). Mean KL, and the share of photos worse than delete:

| Bin | n | delete | all758 | b0p | b0 | p | worse than delete: all758 / b0p / b0 / p |
|---|---|---|---|---|---|---|---|
| 0-10% | 217 | .00399 | .00299 | .00239 | .00244 | .00357 | .44 / .35 / .38 / .31 |
| 10-25% | 299 | .00863 | .00377 | .00349 | .00345 | .00708 | .24 / .19 / .20 / .26 |
| 25-50% | 502 | .01095 | .00386 | .00349 | .00341 | .00863 | .19 / .15 / .14 / .28 |
| 50-75% | 473 | .02706 | .00715 | .00767 | .00766 | .02120 | .14 / .11 / .10 / .23 |
| 75-90% | 305 | .03138 | .00889 | .00885 | .00862 | .02429 | .11 / .07 / .07 / .24 |
| 90-100% | 204 | .07172 | .01702 | .01977 | .01991 | .05264 | .06 / .06 / .06 / .21 |

Checks: base KL max 1.7e-6. Delete KL equals section 3 (.02297 random, .05711 strong). all758 equals section 3 exactly (.711 / .774). p equals section 3's k=1 row (.229 / .316).

Reading, at true strength:
1. Holding the write to the parent's gate loses .44-.48 of recovery on every photo group. The CIs are far from 0. Not noise. The parent's one gate carries about a tenth of D's firing energy (.100 against .442 for the 501 kept neurons, DEV).
2. Removing the skip copy costs almost nothing: .009 on random photos (CI includes 0), .018-.033 where D fired. On the photos where D barely fired, the no-skip build does better by .045. The cause is not separated: all758 used damping .01 and holds the constant, b0 used damping .1 and has no constant.
3. The parent's own 1x1 column adds nothing once the 501 kept block-0 neurons are allowed (every difference within +-.02, CI includes 0).
4. So on this one 2% cut, the language-model form (gates from kept neurons of the same layer only, no skip, no constant) keeps .70 random / .76 parent-strong with no training.
5. The leak stays: in the lowest firing decile, 38% of photos are still worse than delete (44% with the skip copy). On the mean, no bin is worse than delete (lowest decile .00244 against .00399). The leak is a gate mismatch (section 6, limit 1), not a write problem.
6. The .44-.48 gap is for the parent seen at ONE position (A's 1x1 slot). The trained parent build with a 3x3 write (`oc_morph.py`, 9 positions, a slot A does not have, trained on KL so not like-for-like): .734 strong (inside b0's CI) / .516 random (.19 below b0). The like-for-like row (9 parent positions as gates, same exact solve) is not run.

## 6. Roadmap (reset 2026-11-24)

Shonu: "what we are ultimately aiming for is something which is as good as training but straight from calculations." Rule: bring prior art and any mathematical ceiling first (memory `feedback-ceiling-first`).

Terms used from here on:

| Term | Meaning |
|---|---|
| Read | the pattern a neuron looks for |
| Gate (when) | on or off. Set by the read |
| Strength (how much) | how strongly it fires while on. Also set by the read |
| Firing | the output: zero when the gate is off, the strength when on |
| Write | the direction the firing goes into the stream |

Math limits, stated before any new method:
1. Writes only (sections 3 and 5): every kept gate stays fixed. The part of a deleted firing that no mix of kept firings matches is lost. No write solve recovers it.
2. One neuron has one write. A neuron that takes a second job with an orthogonal write writes the mix on both jobs' photos. Training's R has the same 501 neurons and pays this too.
3. Readers read the stream, not a neuron. A piece of a deleted write added to any carrier lands along the deleted write, so the same readers feel it whichever carrier is used. The carrier sets only WHEN it goes out.
4. A small read change alters strength only where a neuron is already on. It cannot switch a neuron on in new places. The gate choice is yes/no per neuron, photo and position.

| # | Item | State |
|---|---|---|
| W | Write: is the exact solve as good as training for the SAME gates? | ANSWERED section 7: no. Trained writes with A's gates beat the solve by .054-.162 (test, from C), the gap grows with the cut |
| E | Every gate against the parent's gate (section 5) | done: parent's gate loses .44-.48, not noise; skip copy worth ~.01 on random |
| G | Gate ("when"): how kept neurons can switch on where a deleted neuron did. Main open problem. Prior art first, read in full: LLM Surgeon (van der Ouderaa et al. 2023), Toy Models of Superposition (Elhage et al. 2022) | next |
| S | Strength ("how much"): read changes where a neuron is already on | later |
| Rule | Skip route dropped: it stores a linear copy of each child (C1) and does not exist in a language model | standing |
| A | Does morphing hurt a kept neuron's own job? (2026-11-25) | PARKED by Shonu. No saved photo set passed the stand-in bar (`results/oc_vs_jobmap/ownjob_saved.md`, section 1). Only the 20-unit smoke at cut 64 (`oc_vs/ownjob_smoke.md`) has true own photos |
| B | Where did each deleted neuron's job go? (2026-11-25) | PARKED by Shonu. Data pulled (`results/oc_vs_jobmap/`, job 6994844). `code/oc_vs_jobmap_read.py` not yet run |
| G1 | Gates first: how do trained gate maps differ from the calculated build's (A's) gate maps? | section 9, Mac only |

## 7. Bigger cuts: calculation against training at each cut size (plan approved 2026-11-24; `code/oc_ladder.py`, read `code/oc_ladder_read.py`)

Shonu: "cut a bunch of neurons so to the point where we know that a coarse neuron is going to be created, and then just retrain ... what is the best strategy in general to kind of get as much capability back from that cut." Cut = s3:69's readers in stage-4 block 0 by the R1 score, strongest first (his choice): 11 (old D), 64, 192, 384 of 512. Nested: each cut holds the smaller ones.

Jobs: 6991221 (cuts 11 and 64, RTX 4000 Ada, cancelled after saving 64), 6991623 (192), 6991624 (384), script md5 12d5fce1. Step 0 (training bug) is in LOG.md, "Step 0 of plan Bigger cuts".

| Build | What changes | Start |
|---|---|---|
| Z | delete only | |
| C | kept write columns, exact least-squares solve (VECMATH 5 "b0" form), fit on train photos, damping on held | Z |
| W | kept write columns + bn2, trained on KL | Z |
| L | W + block-0 reads | Z |
| R4 | L + block 1, skip, head | Z |
| R | R4 + stages 0-3 (cuts 11 and 384) | Z |
| WC | W's setting | C |
| R4C | R4's setting: block-0 reads + writes at cuts 11 and 64 (R4 chose lr 0 for block 1, skip, head there); also block 1, skip and head at 192 and 384 | C |

Training: one lr per group, nested, each new group's grid includes 0 (= the smaller build; equal bit for bit at every cut, checked), tie rule 1%, warm-up 5%, 2,000 steps with a forced 2x rerun of W (taken at every cut, so 4,000 steps from W on). The planned 2x rerun of R4 never ran: the script skips it once W has doubled. L, R4 and R were never tested past 4,000 steps. Recovery = 1 - mean KL / mean KL(Z), all 17,000 test photos and the 342 parent-strong ones. Paired bootstrap, 2,000 resamples.

### Result (all four cuts; full reader output `results/oc_ladder/READ.md`)

Sanity: KL(A||A) = 0 in every job. Z at cut 11 equals oc_retrain (held .0223, test .0224, strong .0571). Every lr-0 entry equals the smaller build bit for bit (L, R4, R at every cut). Held KL is ordered W >= L >= R4 >= R at every cut, and W <= C at every cut.

Recovery (test 17,000 / parent-strong 342):

| Cut | KL(Z) test | C | W | L | R4 | R | WC (from C) | R4C (from C) |
|---|---|---|---|---|---|---|---|---|
| 11 | .022 | .731 / .773 | .752 / .820 | .767 / .850 | = L | = L | .785 / .863 | **.799 / .884** |
| 64 | .140 | .721 / .703 | .760 / .819 | .774 / .847 | = L | | .785 / .846 | **.797 / .872** |
| 192 | .613 | .742 / .721 | .816 / .869 | .830 / .890 | .832 / .892 | | .827 / .878 | **.843 / .897** |
| 384 | 2.706 | .714 / .667 | .867 / .877 | .897 / .914 | .904 / .922 | .907 / .924 | .876 / .885 | **.910 / .927** |

Gap split, test photos, paired bootstrap 95% CI:

| Cut | Writes: W - C | Reads: L - W | Block 1 + head: R4 - L | Stages 0-3: R - R4 | Total from Z: R4 - C |
|---|---|---|---|---|---|
| 11 | +.020 [.015, .025] | +.015 [.013, .017] | 0 (lr 0 chosen) | 0 (lr 0 chosen) | +.035 |
| 64 | +.039 [.035, .043] | +.014 [.012, .017] | 0 (lr 0 chosen) | | +.053 |
| 192 | +.074 [.070, .077] | +.014 [.013, .015] | +.002 [.002, .003] | | +.090 |
| 384 | +.153 [.149, .156] | +.030 [.029, .031] | +.007 [.006, .007] | +.003 [.002, .003] | +.190 |

Changed-job test (L; R4 nearly the same):

| Cut | Self median | Kept with self < .9 | Top take-over (unit -> deleted unit it now matches) | Changed job (self < .8 AND take >= .2) |
|---|---|---|---|---|
| 11 | .999 | 0 | .03 | 0 |
| 64 | .994 | 0 | .07 (n339 -> n476) | 0 |
| 192 | .984 | 0 | .15 (n430 -> n114) | 0 |
| 384 | .933 | 19 of 128 | .30 (n430 -> n501, self .87), .25 (n76 -> n374, self .90), .23 (n213 -> n501, self .91) | 0 |

In R at cut 384 (stages 0-3 free), the top take-over falls to .13: the earlier layers carry part of the job instead.

s3:69's skip column (the parent's own write) in R4: the largest change of all 256 stage-3 channels at cuts 192 and 384 (384: norm .043 vs median .024). At 11 and 64 R4 chose lr 0 for that group, so no reading there.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | cut 11: C within .02 of R4 | R4 - C = +.035 test, +.077 strong | **fail**. The tie seen before this plan was weak training (old R / R4) |
| P2 | C falls behind training faster as the cut grows | R4 - C: .035, .053, .090, .190 | **pass** |
| P3 | L - W grows and is the largest part at 384 | in the writes-first order, writes are the largest part at every cut (.153 of .190 at 384); reads .014-.030. The reads-first order was not run, and the split is not additive | **fail in this order** |
| P4 | no changed-job neuron at 11, at least one at 384 | none at any cut | **half** (first half pass, second fail by the letter) |
| P5 | first changed-job cut = first cut with L - W > .05 | neither happens | not testable |

Checks that failed or need care:
1. C's damping is at the grid's lower edge at cuts 11 and 64 (1e-6); interior at 192 (1e-4) and 384 (1e-3).
2. Trained lrs at the top of their grids: W write 3e-4 (192, 384), L read 1e-4 (64, 192, 384), R outer 3e-6 (384). The trained numbers may be underestimates, so every gap above may be larger.
3. C_held at cut 11 reads .778 strong against VECMATH 5's .760 (bar .005): **fail by the letter**. Likely cause: damping .01 here vs .1 there (different damping split). Not tested.
4. C has no constant term; W also trains bn2's shift. Part of W - C may be the constant (n297, n275 leaned on it in section 3). Not separated.
5. One network, one cut order (s3:69's readers), one seed. The parent-skip finding is partly built in: D was chosen as s3:69's readers.

Training from Z is itself under-trained, by the nesting rule: WC trains the same weights as W and wins at every cut (+.034 / +.025 / +.011 / +.009 test). So the from-C chain is the fairer reference:

| Cut | Writes: WC - C | Total: R4C - C | Total from Z: R4 - C |
|---|---|---|---|
| 11 | +.054 | +.068 | +.035 |
| 64 | +.064 | +.076 | +.053 |
| 192 | +.085 | +.101 | +.090 |
| 384 | +.162 | +.196 | +.190 |

(R4C - WC mixes reads with block 1 and head at 192 and 384.)

Against a fresh block 0 (T0 = B128aug_s0_b0: block 0 at 128 trained from random, A's block 1 fixed, NEURON-LOG M3; same photo split, same KL; 6,250 steps x 256 at lr 3e-3, about 3 times the photo views used here): test KL .237 = recovery .912 at cut 384. Here: R4C .910 (KL .244), R .907, L .897. So a fresh block 0 is a little better than every build trained from the cut, with a bigger budget. Not a large margin.

Reading, at true strength:
1. **The best repair at every cut is: calculate, then train from there (R4C).** It beats training from the cut network (R4) at every cut: +.032 at 11, +.023 at 64; at 192 and 384 the margins (+.010, +.005) come from one seed and a photo bootstrap, so they are not shown to be real. A fresh block 0 trained from random (T0) is about as good as R4C at 384 (.912 vs .910).
2. **The old claim "calculation ties training" was wrong.** It came from under-trained references (1,000 steps, one lr for all layers). With the fixed trainer, training beats the calculated build at every cut, and the gap grows: .068 to .196 (from C), .035 to .190 (from Z).
3. **Training the writes alone, with A's own gates and readers, gets most of the way.** At 384, W reaches .867 and WC .876 of R4C's .910; C reaches .714. Same gates, same readers. So the least-squares solve is far from the best write for fixed gates. Why (math, not yet measured): C rebuilds each deleted firing as well as possible, one number per cell; training sets the writes for the lowest KL at the output, through block 1's ReLUs. These are different targets once the rebuild is poor (at 384 only 128 firings are left to rebuild 384).
4. **The split into writes, reads and later layers is one order (writes first).** It is not additive, and the reads-first order was not run. So this does not show that gates matter less. What is order-free: fixed-gate trained writes reach .867-.876 of the .907-.910 that any trained build reaches at 384.
5. **No coarse neuron in the sense of the bar.** No kept neuron gives up its old job. At 384, kept neurons WIDEN: n430 keeps its own firing (self .87) and also fires where deleted n501 fired (take-over .30, correlation .58); n76 -> n374 (.25), n213 -> n501 (.23). The bar (self < .8) assumed a coarse neuron replaces its old job. The essay's coarse neuron (the parent that never split) keeps its job and covers its children's. That is a reading after the result, so it needs its own test before it counts.
6. **Training keeps the network close to A.** Every build starts from A's weights with small lrs. B (trained from random at 128) made a clear wheel neuron; this does not. Whether a coarse neuron needs a fresh start is open.

What this means for "as good as training, from calculation": with A's gates fixed, trained writes (W / WC) reach about 95% of the best trained recovery at 384. The next target for a calculation is W: writes chosen for the output KL with the gates fixed. Prior art for writes chosen by the output error, not the layer error: OBS (single weights); the KL solve in section 3 (data-limited there).

## 8. What training adds on top of the calculation (plan approved 2026-11-25; Shonu: "using vector analysis tools to look at these different networks' calculated vs trained splits")

Object: WC - C (training started from C, same gates, same readers), at every cut. Shonu's test for an optimum: train after the calculation and it does not help.

Weights of all 30 ladder builds were copied to the Mac on Shonu's request (`results/oc_ladder/n*/*.pt`, md5 checked against the cluster; numpy exports `w_<build>.npz` by `code/oc_vs_extract.py`). Scripts: `oc_vs_writes.py` (md5 d1cb3a70), `oc_vs_spatial.py`, `oc_vs_read.py` (0212df84), `oc_vs_harm.py` (8ecdd1f9), all Mac; `oc_vs_why.py` (a046aad0) cluster job 6993974 (A16, 15 min; smoke 6993963). Results `results/oc_vs/`.

Checks: C - A equals its fold formula (max rel. error 1e-6). Every build's KL in the new job equals the ladder's (max 3e-5). C + every part of WC - C + WC's shift equals WC (max 1e-6). A against itself: e = 0, q = 0, KL = 0. Smoke part shares equal the Mac numbers.

### 8.1 Where training's write change lies (weights only)

Each kept neuron's effective write (bn2 scale x conv2 column, 512 x 3 x 3 = 4608 numbers). A change is split into: inside the span of A's deleted writes (D), inside A's kept writes but outside D (K), outside every A write (rest). Random floors: D nD/4608, K nK/4608, rest .889.

| Cut | WC - C: D / K / rest | Trained beta vs C's beta (corr, norm ratio) | Summed over 3 x 3: share in deleted span (floor) | rest: share uniform over 3 x 3 (floor .111) |
|---|---|---|---|---|
| 11 | .02 / .22 / .76 | .996, 1.01 | .11 (.02) | .29 |
| 64 | .05 / .19 / .76 | .988, 1.03 | .22 (.13) | .27 |
| 192 | .11 / .13 / .76 | .945, 1.08 | .47 (.38) | .26 |
| 384 | .25 / .06 / .69 | .891, 1.09 | .81 (.75) | .32 |

1. Training from C keeps C's mix of deleted writes (corr .89-.996, same size).
2. What it adds lies in A's write span 2-3 times more often than chance, but the largest part (rest, .69-.76) is outside it. The rest part is 68-74% spatial (same channels, spread differently over the 3 x 3 positions) against a random floor of 89%: so it is ENRICHED in channel content, not mostly spatial by preference. Which component carries the gain is tested causally in 8.2b: the uniform (channel) component, not the spatial one.
3. W (trained from the cut, not from C) rebuilds a SMALLER version of C's mix: beta norm ratio .27 / .33 / .48 / .53, corr .79 / .75 / .61 / .39. It still beats C at every cut.

### 8.2 Which part carries the gain (causal: add one part to C, score KL on 17,000 test photos)

Bars written before the run: rest = drift if g(C+DK+shift) >= .90; rest carries the gain if g(C+rest+shift) >= .50. g = share of KL(C) - KL(WC).

| Cut | C+D | C+K | C+rest | C+shift | C+DK+shift (WC without rest) | C+rest+shift (WC without D, K) |
|---|---|---|---|---|---|---|
| 11 | .05 | .43 | .73 | .01 | .48 | .73 |
| 64 | .21 | .35 | .73 | .01 | .53 | .73 |
| 192 | .29 | .19 | .63 | .03 | .51 | .60 |
| 384 | .57 | .11 | .59 | .10 | .74 | .63 |

Verdict: **rest carries the gain at every cut (.60-.73); it is not optimizer drift (drift bar fails at every cut, .48-.74).** The parts overlap (shares add to more than 1). At 384 both carry: WC without rest keeps .74, WC without D and K keeps .63. The bn2 shift alone does almost nothing (.01-.10).

### 8.2b Channel or spatial? (follow-up job 6994054, `oc_vs_why.py spatial`, md5 33c0694a; first try 6994039 crashed after cut 11 at a check for a removed build, same numbers)

WC - C split into its 3 x 3-uniform part (the same 512-channel change at all 9 positions) and its spatial part (sums to 0 over the 9). Each added to C with WC's shift. Bar written before this run: "the gain is mostly spatial" if g(spatial) >= .5 and > g(uniform).

| Cut | g(uniform) | g(spatial) | rest only: g(uniform) / g(spatial) |
|---|---|---|---|
| 11 | .74 | .38 | .50 / .28 |
| 64 | .71 | .46 | .47 / .33 |
| 192 | .60 | .50 | .32 / .36 |
| 384 | .84 | .49 | .45 / .33 |

Verdict: **the bar fails at every cut. The gain is mostly the uniform (channel) part**, though both halves carry some (they overlap). The earlier "mostly spatial" wording (first draft of 8.1 and 8.3) was wrong and is withdrawn. The uniform part is the vector vecread reads (summed over 3 x 3).

### 8.3 Why: the stream error grows, the KL falls; the quadratic forecast explains the gain only at small cuts (step 5)

e = stream after block 0 (build minus A). q = 1/2 (J e)^T F (J e), the exact per-photo quadratic KL forecast through block 1 and the head. G = readers' Fisher pooled over cells and photos (512 x 512, held photos).

| Cut | |e|^2: WC / C | q / KL: C, WC | Pooled-G error (e^T G e): WC / C | Share of |e|^2 on G's top 20: C, WC |
|---|---|---|---|---|
| 11 | 1.21 | 1.00, 1.01 | 1.20 | .038, .036 |
| 64 | 1.28 | .98, 1.05 | 1.27 | .041, .038 |
| 192 | 1.41 | .94, 1.24 | 1.36 | .053, .046 |
| 384 | 1.31 | .61, 1.10 | 1.24 | .058, .049 |

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | WC has |e|^2 >= .9 x C's but lower KL | 1.21-1.41 x C's error, lower KL at every cut | **pass** |
| P2 | WC's error sits less along the readers' top Fisher directions | .036-.049 vs .038-.058: less, but by little. e^T G e / |e|^2 is the same for C and WC (1.9-2.1e-3) | **no information**: this pooled G measures only the size of the error |
| P3 | q within 30% of KL for C and WC at cuts 11, 64 | 1.00 / 1.01, .98 / 1.05 | **pass**. At 384 C's q is .61 of its KL (outside the quadratic range); WC is back inside (1.10) |

Does the quadratic forecast see training's GAIN (not only each build's total)? Added after the advisor review, from the saved per-photo arrays:

| Cut | (q_C - q_WC) / (KL_C - KL_WC) | Same for W | Per-photo correlation of the two gains |
|---|---|---|---|
| 11 | .98 | .96 | .92 |
| 64 | .75 | .76 | .84 |
| 192 | .32 | .39 | .58 |
| 384 | .23 | .35 | .25 |

Reading:
1. Training makes the stream error LARGER than C's (by 21-41%) and the KL smaller. So C's target (rebuild each deleted firing) is not the KL's target. This was "math, not yet measured" in 7, reading 3. Now measured.
2. At cut 11 the per-photo quadratic form of the stream error predicts training's whole gain (.98). At 64 it predicts three quarters. At 192 and 384 it predicts only a third and a quarter. So a calculation that minimizes the per-photo q, with the same gates, can in principle reach W's gain at cut 11 and most of it at 64. At 192 and 384 most of the gain is beyond any quadratic solve: it comes from the nonlinear part (C's own KL there is 1.06-1.6 x its q; likely ReLU flips in block 1, not separated).
3. The pooled Fisher used here carries no information about the gain: it scores both errors by their size only. This does not show that KFAC-style solves fail, and it cannot separate per-photo terms from cross-cell terms.

### 8.4 Vector reads of the changes (vecread, A's dictionary; floors: random mixes of A's writes top cosine 95th .30, leftover 5th .77)

| Cut | Lost write (sum of D's writes) | C - A, direction 1 | WC - C, net sum | cos(net WC - C, net C - A) |
|---|---|---|---|---|
| 11 | zigzagged / chequered / studded vs matted / flecked | the same (share .28) | cracked / marbled vs studded / polka-dotted (left .84) | -.44 |
| 64 | zigzagged / washer / polka-dotted vs flecked / forest | the same (share .13) | brick / warehouse (left .92, noise level) | -.24 |
| 192 | zigzagged / head / dog vs flecked / marbled | the same (share .16) | leg / neck (left .79) | +.20 |
| 384 | zigzagged / spiralled / dog vs flecked / marbled | the same (share .32) | flecked / marbled / pitted vs zigzagged (left .55) | -.79 |

1. C's main direction is the lost write's main direction at every cut (same top concepts, both poles). The lost content reads as one contrast: zigzag / dog / head against flecked / marbled.
2. The summed WC - C points against C's summed rebuild at 384 (cosine -.79), at 11 and 64 too (-.44, -.24), not at 192 (+.20). These are unweighted sums over kept neurons (no weighting by how much each fires), and the trained beta is 1.09 x C's in size (8.1). So this does NOT show that C over-writes. It shows only that training's added writes, summed, oppose C's main direction. At 11 the net WC - C reads as the same contrast reversed (cracked / marbled up, studded / polka-dotted down), near the noise floor (leftover .84).
3. Most single directions of WC - C read near the noise floor (leftover .55-.92). The gain-carrying part is the uniform one (8.2b), which this read does see, so the change is spread over many weak directions rather than a few concept directions. Not tested further.

### 8.5 Widening test (coarse = rough and widened; memory `feedback-coarse-widened`). Bars in the plan; the read metric changed after the first run

Reads compared as patterns (cov x r, whitened), the vecread rule. CHANGED after the first run: that run used raw read kernels, and the cut-384 rows then failed the read bar (.22 / .46 / .76). n339 at 64 passed with both metrics (.98). Percentile = rank of the named partner among all deleted neurons (at cut 11 the top is .91 = first of 11, so the .95 bar cannot pass there).

| Cut | Unit -> partner | Build | Keeps job (a) | Write change toward partner (pct) | Read change toward partner (pct) | Verdict |
|---|---|---|---|---|---|---|
| 11 | n5 -> n464 | L | 1.00 | .91 (first of 11) | .91 (first of 11) | top of 11 on both; bar not reachable |
| 64 | n339 -> n476 | L | 1.00 | .98 | .98 | **coarse (all 4 bars)** |
| 64 | n339 -> n476 | R4C | 1.02 | .98 | .98 | **coarse (all 4 bars)** |
| 192 | n430 -> n114 | L | 1.01 | .93 | .99 | fails: write |
| 384 | n430 -> n501 | L | .93 | .80 | .98 | fails: write |
| 384 | n213 -> n501 | L | .96 | .94 | .98 | fails: write, rough |
| 384 | n76 -> n374 | L | .99 | .88 | .91 | fails: write, read |

C and WC change no read, so they fail the read bar by construction. Their write change points at the partner (pct .98-1.00): that is C's beta.

Reading:
1. One unit passes every bar: n339 at cut 64 keeps its job (a = 1.00), its write and read both move toward deleted n476 (98th percentile), and its new write shares concepts with both (n339 seat / bus interior, n476 autobus / van / car). That is a coarse neuron in the essay's sense: rough and widened, same layer.
2. At 192 and 384, trained from the cut (L), the READ moves toward the partner (98th-99th percentile) but the WRITE does not (80th-94th). The kept neuron learns WHEN the partner fired; the partner's write content goes out through other neurons. This splits Shonu's coarse neuron into its two halves: the gate widens locally, the write is shared.
3. Limits: one network, one cut order, one seed. The units were picked as the top take-over (selection). The "rough" bar (>= 1 shared concept of 8) is weak.

Controls added after the advisor review (`code/oc_vs_widen_ctl.py`, `results/oc_vs/widen_ctl.md`):

| Cut | Unit -> partner | Rank among ALL kept units for that partner, L: write / read | R4C: write / read |
|---|---|---|---|
| 11 | n5 -> n464 | 1.00 / .99 | 1.00 / .99 |
| 64 | n339 -> n476 | 1.00 / 1.00 | 1.00 / 1.00 |
| 192 | n430 -> n114 | .85 / .99 | .97 / .90 |
| 384 | n430 -> n501 | .83 / .95 | .91 / .70 |
| 384 | n76 -> n374 | .84 / .88 | .97 / .82 |
| 384 | n213 -> n501 | .92 / .98 | .97 / .83 |

| Cut | Kept units with some partner in the top 5% of both lists, a >= .8 (L / R4C) | Shuffled partner labels, mean (max of 20) |
|---|---|---|
| 11 | 193 / 183 of 501 | 44 / 47 (58 / 60) |
| 64 | 353 / 346 of 448 | 106 / 106 (120 / 125) |
| 192 | 276 / 273 of 320 | 131 / 133 (144 / 144) |
| 384 | 125 / 121 of 128 | 83 / 86 (95 / 95) |

4. n339 at 64 (and n5 at 11) is the FIRST of all kept units for its partner, on write and read: specific, not a shared direction. The 384 units are not first (.70-.98): part of their move is a direction many units share.
5. Widening is widespread: at every cut far more kept units move both read and write toward some deleted neuron than shuffled labels give (64: 353 against 106). Caveat: the shuffle breaks the partner pairing but does not control A's own geometry (a unit whose read already resembles a deleted neuron may also have a similar write).

### 8.6 Harm to unrelated jobs (step 4; quiet = bottom 50% of photos by A's summed firing of D)

| Cut | Recovery on quiet photos: C / WC / L / R4C | Quiet photos worse than delete: C / WC / R4C | Classes (of 1000) worse than delete |
|---|---|---|---|
| 11 | .67 / .69 / .67 / .70 | .19 / .20 / .21 | 1-2 |
| 64 | .71 / .75 / .74 / .76 | .13 / .12 / .12 | 1-2 |
| 192 | .73 / .81 / .81 / .83 | .10 / .08 / .08 | 0 |
| 384 | .70 / .87 / .89 / .90 | .05 / .01 / .01 | 0-1 |

1. No build hurts the quiet photos on average: every recovery is positive, with tight CIs.
2. At cut 11 about 1 photo in 5 of the quiet half is worse than plain deletion, for C AND for every trained build. Training does not remove this leak. So the leak is not a flaw of the calculation alone.
3. At 192 and 384 the quiet half has nearly the delete KL of the busy tenth (D fires on almost every photo at large cuts). So this test informs only at cut 11, and weakly at 64.

### 8.7 What this means for "as good as training, from calculation"

- Same gates, same readers: training keeps C's mix of deleted writes and adds a change to the kept writes that makes the stream error LARGER and the KL smaller. C's target (rebuild each deleted firing) is not the KL's target.
- The added change that matters is mostly a channel change of the kept writes (8.2b), partly outside every A write direction (8.1). It is not optimizer drift (8.2).
- At cut 11 the per-photo quadratic KL form predicts all of training's gain (.98); at 64, three quarters. So at small cuts a calculation that minimizes the per-photo quadratic form (not the firing error) can in principle reach W. At 192 and 384 it predicts a quarter to a third: most of the gain there is nonlinear, beyond any quadratic solve.
- Ceiling and prior art first (memory `feedback-ceiling-first`): this target is the KL-weighted solve of section 3, which LOST at cut 11 because it was data-limited (2,000 held photos; the 16,000 train photos exist on the cluster but were outside that plan's scope). Published relatives to read in full before any build: OBS / Optimal Brain Compression (layer Hessian only = C's target), LLM Surgeon (KFAC of the loss Fisher), Kwon et al. 2022. The pooled Fisher tested here carried no information; this does not rule KFAC in or out.
- Not done from the plan: s3k concept cards of the read changes (only read-pattern cosines). Done after the review: G's top 5 directions read as no Broden concept (leftover .88-.98, top cosine <= .20, under the random-mix 95th .30); n430's own write reads zigzagged / chequered / interlaced with this tool (K5b keeps only spreads, not per-neuron readings, so this is a consistency check only).

## 9. Gates first: trained gate maps against the calculated build's (plan approved 2026-11-25; Shonu: "focus on how the gates are different across the trained and calculated stuffs")

Object: the gate map of a kept neuron j = the cells (photo, 7 x 7 position) where j fires. C and WC keep A's reads, so their gate maps ARE A's (checked: equal at every cut). The trained builds L and R4C change them. Mac only, numpy, 16 s for all 4 cuts: `code/oc_gate.py` (md5 30c985f3), reader `code/oc_gate_read.py` (94bad977), full output `results/oc_gate/READ.md`. Input: saved stage-3 maps of 2,000 Broden held-set-1 photos (`vecread/results/held_s3.npy`). A's block 0 on them reproduces `held_s4b0` (max rel. error 4.6e-4). New reader `vecread.gate_card`: Broden labels over-represented on a set of cells, against a shuffle within the same photos.

```
                A on     A off
trained on      kept     NEW-ON
trained off     LOST     -
```

| Cut | Build | new-on / A's on-cells (median) | lost / A's on-cells (median) | new-on cells: A's pre-activation percentile among j's off cells (median) | deleted firing, new-on vs near-miss cells that stayed off (median ratio; share of units > 1) |
|---|---|---|---|---|---|
| 11 | L / R4C | .012 / .011 | .018 / .017 | .994 / .995 | 1.38 (.75) / 1.38 (.77) |
| 64 | L / R4C | .030 / .022 | .068 / .076 | .983 / .984 | 1.20 (.85) / 1.22 (.88) |
| 192 | L / R4C | .036 / .024 | .115 / .141 | .973 / .976 | 1.12 (.77) / 1.16 (.82) |
| 384 | L / R4C | .505 / .280 | .016 / .050 | .906 / .932 | 1.07 (.76) / 1.08 (.73) |

Which knob: new-on cells with A's read and the trained threshold (bn1 shift) = .01-.03 of the build's new-on cells. With the trained read and A's threshold = .99-1.19. The threshold did not move. The read did.

Gate partner (the deleted unit whose A on-cells best explain j's new-on cells, by lift): equals the firing-based partner of section 7 for .16-.40 of units, against chance .003-.09. Raw-count partners at 192 and 384 are one deleted unit that fires nearly everywhere (lift ~.97), so lift is the usable measure.

Named (L): n5 -> n464 (11): n464 is the top partner by count and by lift (lift 10.7). n339 -> n476 (64): n476 is first by lift (3.18), 4th by count. The new-on cells read car / floor / street scenes, n476's cells truck / car / street. Shared concept: car. n76 -> n374 (384): the new-on cells read ceiling / sky / painted, the same top 3 as n374's cells. n430 at 192 and 384: the new-on cells read sky / ceiling / painted (shared with n114), not n501's drawer / pane.

| # | Prediction (written before the run) | Result | Verdict |
|---|---|---|---|
| P1 | new-on cells fall where deleted units fired, above chance, every cut (yes/no D-cell test) | 1.09 at 11. 1.00 at 64+, because some deleted unit fires on ~every cell there | pass at 11, **not testable** at 64+ as written. Graded and margin-matched measures (added after the run): above 1 at every cut |
| P2 | new-on share grows with the cut | .012, .030, .036, .505 | pass |
| P3 | lost share <= .1 at every cut | .115 (L) and .141 (R4C) at 192 | **fail** |
| P4 | n339's count partner is n476, and the cards share a concept | 4th by count (1st by lift). Shared: car | **fail by the letter** |
| P5 | the read moves the gate, not the threshold | read .99-1.19, threshold .01-.03; and the read's direction, not its length (check below) | pass |

Bug found and fixed during the read: `jobtest` stores `best_d` as an index into D, not a unit number. The first read showed 0 agreement everywhere. The partner agreement for R4C is scored against L's jobtest partners.

Checks added after the advisor review (bars fixed before each run; `READ.md` sections 2b, 2c):

| Check | Cut 11 | 64 | 192 | 384 | Verdict |
|---|---|---|---|---|---|
| Gain or tilt: trained read = alpha x A's read + leftover. alpha median (L) | 1.000 | 1.003 | .998 | .995 | the read's length does not change |
| new-on cells that the gain-only gate (alpha x A's read + trained shift) also opens (L / R4C) | .01 / .00 | .01 / .00 | .00 / .00 | .02 / .01 | **tilt** at every cut (bar <= .2) |
| Null: new-on vs near-miss stayed-off, mean firing per DELETED unit (L) | 1.38 | 1.20 | 1.12 | 1.07 | |
| same, mean firing per KEPT unit, j left out (L) | 1.02 | 1.02 | 1.00 | 1.08 | |
| same, input patch norm (L) | 1.06 | 1.07 | 1.08 | 1.07 | |
| deleted / kept, median [share of units > 1] (L) | 1.33 [.72] | 1.16 [.86] | 1.12 [.91] | 1.01 [.60] | **deleted-specific at 11, 64, 192. At 384: busy cells open** (R4C the same) |
| `gate_card` known answer: a label's own cells read back first | 32 of 45 = .71 | | | | **fail** (bar .8). Random 5% of cells give top lifts 1.25-1.39, not ~1 |

The gate cards (the named-unit concept readings above) are therefore NOT verified. Treat them as hints only. The n5 card (284 cells) is also low power.

Reading, at true strength:
1. **The gate change is a tilt of each kept neuron's read.** The threshold (bn1 shift) alone opens .01-.03 of the new cells. The read's length does not change (alpha ~1.00), and length alone opens .00-.02. What opens the cells is the read's DIRECTION.
2. **The cells that open are near misses.** On median, a new-on cell sits in the top 1-10% of j's off cells by A's own pre-activation. Part of this is built in: any small read change opens near-miss cells first.
3. **At cuts 11, 64 and 192, training picks WHICH near misses open: the ones where the deleted units fired.** Among cells equally close to firing in A, the opened ones carry 1.12-1.38x the deleted firing per unit, while kept units' firing is the same (1.00-1.02). 72-91% of units show it. At 384 this fails the null: the opened cells are simply busier cells (kept firing 1.08x, deleted 1.07x).
4. **Gates also close.** At 64 and 192 more cells close than open (lost .07-.14). Not read yet: which cells close, and why.
5. **At 384 the gates widen a lot** (L: +50% on-cells, median), and the widening is not aimed at the deleted cells (point 3).
6. **A gate calculation has a clear shape for cuts up to 192.** Inputs per cell: each kept unit's A pre-activation (its margin) and the deleted units' firing. Choice: tilt each kept read toward the near-miss cells where the deleted units fired. Knob: the read direction (not the threshold, not the length). Not built or tested. At 384 the shape is different and not understood.
7. Limits: Broden photos, not ImageNet. One network, one cut order, one seed. Partners and cards are for 5 picked pairs, and the card tool failed its test.

Next (not started): read the closed cells (point 4). Fix `gate_card` until it passes its known-answer test (likely label twins and rare labels). Then a first gate calculation from point 6, scored with the same tables against L.

## 10. Gates, part 2: read the tilt as vectors, and test one reason for it (plan approved 2026-11-25; Shonu: "doing some vecreads and comparing that with the original network")

Shonu's belief holds: C's gates are A's (section 9 checked C's kept reads and shifts equal A's, bit for bit, at every cut). So "A" here also means "C".

Object: the tilt of kept neuron j, `t_j = r_j - alpha_j a_j` (a = A's effective read, r = the trained read of L or R4C, 2304 numbers on `s3k`). One metric for all of it: the vecread read metric, inner product `u . cov . v` (cov of `s3k`). `alpha_j` and every cosine use it.

Rule under test (written before the run). Training tilts j's read toward the input on the cells where j already fires and C rebuilt the deleted units too little, and away where C rebuilt them too much. The deleted units count by how much j's write overlaps theirs.
Derived from: training's update of j's read at C, loss = half the squared block-0 stream error. At C kept firings equal A's, so the stream error `e = stream_C - stream_A` is exactly C's rebuild error of the deleted firing. Gradient on j's firing at a cell = C's write of j against e (a transposed 3 x 3 conv). The ReLU passes it only on j's on-cells. Predicted update of j's read: `q_j = - sum over j's on-cells of grad_j(cell) x patch(cell)`. Compared with the tilt after removing the part along `a_j`. Built from A and C only.
Known risk: section 8 found training makes stream error LARGER while KL falls at 192 and 384. If the rule fails there, the named gap is the KL weighting.

Every step runs at every cut (11, 64, 192, 384) for L and R4C. Photos: the same 2,000 held-set-1 photos as section 9 (a development reading, not a confirmation).

Controls (fixed before the run):
- Shuffle null: the gradient values `grad_j` permuted among j's own on-cells within each photo, 20 shuffles. Keeps "where j fires", destroys "where the deleted firing is missing".
- Neighbour control: j's on-cells, but the gradient values of the kept unit whose A read is most similar to `a_j` (read metric). Same reads nearby, a different write. Tests the "write overlap" part of the rule. (The plan said "q from a similar kept unit"; this is that control, written out.)

| # | Prediction (written before the run) | Bar |
|---|---|---|
| P0 | A's 512 kernels through `reads()` reproduce the vecread K5b spread | top z median 3.8, leftover median .80, each within .1 |
| P1 | the tilt is reproducible: same unit, L against R4C | median cosine above the mismatched-pair 99th, every cut |
| P2 | the tilt reads as a concept as much as a real read | median top z >= 3.4 (random 95th) |
| P3 | the rule points the right way | share of units with cosine(t, q) above their own shuffle 95th >= .5, cuts 11, 64, 192, R4C |
| P4 | the write overlap matters | median cosine(t, q) above the neighbour control, cuts 11, 64, 192, R4C |
| P5 | the rule fits R4C (started from C) better than L (started from delete) | median cosine R4C > L at each cut |
| P6 | the rule opens AND closes the right cells | predicted read `a + |t| unit(q)`: share of training's new-on cells it opens, and of training's lost cells it closes, >= 2x the shuffle null, cuts 11, 64, 192 |
| P7 | cut 384 | no pass predicted. Reported under the same rule |

### Results (2026-11-25, Mac, about 4 min per run; `code/oc_gate_vec.py` md5 26fa762f, reader `code/oc_gate_vec_read.py` 6b783fd7; all tables `results/oc_gate_vec/READ.md`, logs `run.log`, `run2.log`, `run3.log`)

Three runs. Run 2 added the mean-shift fix of P6. Run 3 fixed a bug in the neighbour control (it used only the cells where j AND the neighbour fire; now all of j's on-cells), and added three things after an advisor review: a random-direction control for P6, the deleted units' reads, and the "toward the deleted reads" test. All rule, shuffle and reproducibility numbers are the same in all three runs (same seed).

| # | Result | Verdict |
|---|---|---|
| P0 | top z median 3.817, leftover .799 | pass |
| known answer | a synthetic tilt at cosine .2 to the rule scores above its own shuffle 95th for .82-.86 of units | **fail** (bar .9). The synthetic is not a fair model of the real tilts: at 192 the real tilts (cos .115) clear their shuffle more often (.94) than the synthetic does |
| P1 | same unit L vs R4C: median cosine .83 / .83 / .86 / .84; mismatched 99th .32 / .27 / .34 / .41; 100% of units above it, every cut | pass. The tilt is real |
| P2 | tilt top z median: L 3.64 / 3.66 / 3.92 / 4.56, R4C 3.40 / 3.51 / 3.83 / 4.34 (A's reads 3.78-4.11) | pass (R4C at 11 exactly on the bar) |
| P3 | share of units with cos(t, q) above own shuffle 95th, R4C: .65 / .76 / .94 / .91 (L .68 / .80 / .92 / .86) | pass at 11, 64, 192 |
| P4 (run 3, fixed control) | cos(t, q) median R4C .125 / .136 / .115 / .294 against the neighbour control -.002 / .015 / .013 / .061 | pass |
| P5 | L is above R4C at every cut (.138 / .153 / .146 / .336 vs .125 / .136 / .115 / .294) | **fail** |
| P6 as built | the predicted read opened 8-34x more cells than training. Rule = shuffle | **fail, test flaw**: the read metric (centred covariance) does not see a shift of the mean input. The predicted tilt shifted the mean pre-activation 10-20x more than training's (.10-1.25 vs .005-.27 of j's spread) |
| P6 fixed (added after run 1, bar unchanged) | mean shift removed, length kept. Hit new (R4C), rule / shuffle / neighbour / random: .41/.39/.31/.29 (11), .47/.44/.33/.30 (64), .59/.54/.37/.30 (192), .35/.34/.25/.22 (384). Rule and shuffle open about the same number of cells (within 3%) | **fail** (bar 2x shuffle). Rule beats shuffle by 1.04-1.10x on opened cells, 1.1-1.3x on closed cells |
| P7 | 384: vectors pass (.91 above own shuffle). The fixed rule opens only .35-.57 as many cells as training. It picks closed cells best of all cuts (.46-.51 vs shuffle .35-.38) | reported |

Added in run 3: **does the tilt point at the deleted units' reads?** Per unit, the best cosine of the tilt with a read (its a_j part removed, read metric). Deleted reads against other kept reads, the same number of candidates m, 20 random draws.

| Cut | m | best cos with a deleted read, L / R4C | best cos with a kept read, L / R4C | share of units where deleted > kept, L / R4C |
|---|---|---|---|---|
| 11 | 11 | .249 / .215 | .140 / .137 | .74 / .73 |
| 64 | 64 | .273 / .265 | .203 / .199 | .85 / .84 |
| 192 | 192 | .307 / .318 | .271 / .273 | .71 / .79 |
| 384 | 127 | .286 / .296 | .242 / .248 | .80 / .84 |

No bar was set before this test. Limit: the kept candidates are not matched on similarity to `a_j` (only the `a_j` part is removed).

Named units against their deleted partner (vecread `s3k` top 5; data side = top-IoU labels on the held cells where the vector projects highest):

| Unit | A's read | tilt (L) | deleted partner's A read | closest deleted read to the tilt (L / R4C) | cos(t, rule) L / R4C (shuffle 95th) |
|---|---|---|---|---|---|
| n5, cut 11 | back pillow, cushion, foot, fabric | interlaced, paisley, eye, swirly. Cells: interlaced, zigzagged, paisley | n464: interlaced, paisley, food, eye, swirly. Cells: head, interlaced, cat | n464 (.64) / n464 (.67) | .46 / .43 (.39) |
| n339, cut 64 | arcades, gravestone, back, armchair. Cells: chair, person, car | wheel, car, van, motorbike. Cells: car .096, wheel, street | n476: van, arcades, car, plaza, street. Cells: car .102, street, bus | n476 (.49) / n476 (.55) | .65 / .65 (.43) |
| n430, cut 192 | striped, white, chequered, zigzagged | painting, shelves, waffled. Cells: sky, blue | n114: wall, shelves, frame. Cells: sky .065, white, blue | n114 (.47) / n405 (.39) | .24 / .28 (.23) |
| n430, cut 384 | (same) | sky, lighthouse, blue, highway. Cells: sky .136, blue | n501: tv monitor, keyboard, screen. Cells: sky .111, blue | n374 (.47) / n418 (.35) | .78 / .36 (.34) |
| n76, cut 384 | freckled, blotchy, marbled, stained | water tower, spiralled, television. Cells: sky .115, blue | n374: marbled, blotchy, stained, mountain. Cells: sky .129, blue | n284 (.53) / n284 (.32) | .67 / .37 (.36) |

Concept counts over all units (share of units with the concept in their top 5) are in `READ.md`. A's kept reads AND the deleted reads both read mostly as textures (potholed, stained, blotchy, crystalline; the vecread README's real-read hubs). The tilts list object parts (nose, fan, chandelier, wheel; cat, dog at 192-384). No baseline exists for tilt-type vectors (a read minus its own part), so this aggregate list is **not** read as a finding.

Reading, at true strength:
1. **The tilt is real.** Two separate trainings tilt each kept read the same way (.83-.86, every cut). It is small: 3% of the read's length at cut 11, 8% at 64, 12% at 192, 20-23% at 384.
2. **The rhyme: each kept read tilts toward the read of a deleted unit.** In 71-85% of units, at every cut, the tilt points at some deleted unit's read more than at any other kept unit's read. In the named cases, the tilt and the deleted partner read as the same concept: n5's tilt and n464 (interlaced, paisley, eye, swirly), n339's tilt and n476 (car, van). Compared with A (= C), training adds a piece of a deleted unit's read to the kept read.
3. **A partial reason: the stream-error gradient at C.** Built from A and C alone: on the cells where j already fires, move j's read toward the input where the deleted firing that j's write can carry is still missing. It beats the within-photo shuffle (P3). A same-read neighbour with a different write predicts nothing (P4). So the write decides which deleted read j tilts toward. For n339 the rule alone gives car, van, wheel, motorbike (cos .65 with the real tilt).
4. **But the rule is a small part of the tilt.** Median cosine .12-.15 at cuts 11-192, .29-.34 at 384. This is far above chance in 2,304 dimensions, but most of the tilt's direction is not explained. The rule also carries a part that is shared across units (toward table and ottoman, away from tree and forest, in 16-71% of units) that the tilts do not have.
5. **The rule does not pick the cells.** After the mean-shift fix, the rule opens training's new cells only 1.04-1.10x as often as the shuffle, which opens the same number of cells. Both beat a random direction (1.3-2x at the same order of open count). The cells come mostly from "a small tilt built from j's own on-cells". Where the missing firing sits adds little.
6. **P5 failed.** The rule, computed at C, fits L (trained from the cut) slightly better than R4C (trained from C). Not understood.
7. Named gap: the rule uses stream error, and section 8 found that training trades stream error for lower KL. The missing part of the tilt is most likely the KL weighting (which output directions matter), plus many training steps instead of one. Not tested.
8. Limits: development photos (section 9's 2,000), one seed, one cut order, the known-answer check failed its bar, and the "toward deleted reads" test had no bar before the run.

Next (one step, same scope, Mac): the same rule with the KL gradient at C in place of the stream error (A's block 1 and head on the saved stage-4 maps). Test: does the cosine with the tilt rise above .15, does the shared table / tree part go away, and does the rule then pick the cells?

## 11. Which kept neurons aim at a cut neuron, at which one, and is there a parent reason (plan approved 2026-11-25; Shonu: "what neurons are tilting and what neurons are not ... where to apply some sort of effect ... im suspecting there is some sort of parent/interp explanation")

Found before the run (read-only, saved files): by SIZE every kept read tilts by a similar amount (5th-95th percentile about 2x; gate change about 2.5x, top 10 neurons hold 4-5% of all change). By AIM there is a split: the cosine of the tilt with the closest deleted read runs from .04 to .50 at cut 11. So the object is the aim. L and R4C used the same seed (0) and data order, so their agreement is weaker evidence than two independent runs.

Object: aim(j, d) = cosine of kept neuron j's tilt with cut neuron d's A read, j's own read part removed (vecread read metric, as in section 10). Partner of j = the d with the best aim. Bars are judged on R4C (as in section 10). L is reported.

| # | Prediction (written before the run) | Bar |
|---|---|---|
| Q1 | j's aim at d rises when d is cut. Pairs: j kept at both cuts, d kept at c1 and cut at c2 ("entering"), against d kept at both ("staying") | median rise (entering) minus median rise (staying) above the 99th percentile of the same difference with the entering / staying labels permuted (1,000 times), for 11->64, 64->192, 192->384 |
| Q2 | parent: j's best aim rises with j's read weight on s3:69 (the read factor of the R1 score that ordered the cut) | Spearman above the shuffle 99th at cuts 11 and 64 |
| Q3 | the partner shares a parent channel with j: score = the largest per-channel overlap, min(share of j's read energy on channel c, share of d's) | top-1 hit rate >= 3x chance (1/|D|), every cut |
| Q4 | the shared parent adds beyond whole-read similarity | Q3's hit rate still >= 2x chance among candidates in the same whole-read-similarity fifth as the partner |
| Q5 | write size alone does not predict aim | Spearman of best aim with j's write norm within the shuffle 99th |

Also reported, no bar: best aim against C's `beta_C` mass on j and against co-firing with D (unit level); partner hit rates for whole-read similarity, `beta_C` and co-firing (pair level); A's reads of the top and bottom aim tenths and of D in vecread concepts; each neuron's self-correlation (`jobtest_L` self) next to its aim.

### Results (2026-11-25, Mac, 32 s; `code/oc_tilt_who.py` md5 361d564b for the main run, 1c8ffd97 after the checks were added, reader `code/oc_tilt_who_read.py` d7c91291; the top-partner check ran inline from the same functions, saved as `cross_matched.json`; tables `results/oc_tilt_who/READ.md`, log `run.log`)

Checks: the aims recomputed from the saved tilts equal section 10's (max error < 1e-6, asserted). Planted partners (a tilt at cosine .3 to the top shared-parent cut neuron) are found 100% at every cut.

| # | Result (R4C; L in READ.md, the same pattern) | Verdict |
|---|---|---|
| Q1 | all pairs: entering minus staying rise .007 / .009 / .003, null 99th .004 / .003 / .007 (11->64 / 64->192 / 192->384) | pass at 11->64 and 64->192, **fail** at 192->384 |
| Q1, top partner (added after the run) | the partner's aim before -> after it is cut: .05 -> .26 (11->64), .09 -> .30, .08 -> .29. The same pick among kept neurons: .06 -> .19, .11 -> .25, .10 -> .25. Extra rise .06-.075 | most of the jump is the pick itself. A real extra of about .04-.075 |
| Q1, named | n339's aim at n476: -.02 at cut 11 (n476 kept) -> .55 at cut 64 (n476 cut). L: -.06 -> .49 | the clearest case: the aim appears when the partner is cut |
| Q2 | best aim against j's read on s3:69: rho .27 (11), .11 (64), against null 99th .10 / .10 | pass at 11 and 64. Gone at 192 (.04 vs .13), at the null line at 384 |
| Q3 | partner = the cut neuron with the top shared parent channel: 1.4x / 2.9x / 6.6x / 3.0x chance (384 = 1 neuron) | **fail** (bar 3x every cut) |
| Q4 | the same within the partner's whole-read-similarity fifth: 1.1x / 1.8x / 4.2x / 2.4x | **fail** at 11 and 64 |
| Q5 | best aim against write norm: rho -.03 / .00 / .13 / .08, null 99th .10 / .11 / .13 / .20 | pass (192 R4C on the line) |

Reported without a bar:

| Cut | best aim against `beta_C` mass (rho; null 99th) | against co-firing with D | partner = top `beta_C` (x chance; neurons; partner's median percentile) | partner = top co-firing | partner = top whole-read similarity | self-correlation against aim (L) |
|---|---|---|---|---|---|---|
| 11 | .53 (.11) | .48 | 4.1x (188 of 501; .90) | 3.2x | 2.9x | -.50 |
| 64 | .42 (.10) | .23 | 16.6x (116 of 448; .94) | 11.0x | 8.6x | -.16 |
| 192 | .22 (.12) | .05 | 31.2x (52 of 320; .95) | 19.8x | 13.8x | -.07 |
| 384 | .27 (.20) | .07 | 21.0x (7 of 128; .94) | 18.0x | 6.0x | -.17 |

In L (trained from the cut, never saw C's writes) `beta_C` still picks the partner at 3.7x / 11.9x / 12.6x / 6.0x chance.

Vecread: A's reads of the top and bottom aim tenths both read as the real-read texture hubs (stained, blotchy, potholed, marbled). Broden concepts of a neuron's OWN read do not say who aims.

The 5 strongest aimers per cut (R4C; vecread top concepts): the tilt reads as the partner's read, while the neuron's own read is unrelated.

| Cut | j -> partner (aim) | j's own A read | j's tilt | partner's A read |
|---|---|---|---|---|
| 11 | n154 -> n464 (.81) | pole, bridge, water, concrete | nose, swirly, paisley, interlaced | interlaced, paisley, food, eye |
| 11 | n86 -> n23 (.72) | hair, smeared, stained, ear | porous, pitted, sprinkled, blotchy | pitted, porous, blotchy, stained |
| 64 | n328 -> n33 (.57) | dome, chimney, tower, bell | jacket, tent, escalator, curtain | tent, escalator, umbrella, jacket |
| 64 | n339 -> n476 (.55) | arcades, gravestone, back | car, wheel, van, motorbike | van, arcades, car, plaza |
| 64 | n185 -> n509 (.51) | stained, wood, marbled, alley | rock, stratified, brick, ruins | stratified, ruins, rock, potholed |
| 192 | n293 -> n236 (.57) | fence, stratified, potholed | street, grass, tile, park | ground, grass, water, tile |
| 384 | n372 -> n401 (.53) | woven, zigzagged, apartment building | coast, golf course, ocean, beach | coast, beach, sea, ocean |

Checks added after an advisor review (`oc_tilt_who.py --checks`, `results/oc_tilt_who/checks.json`):

| Check | Cut 11 | 64 | 192 | 384 |
|---|---|---|---|---|
| Hub-matched null: partner hit rate, `beta_C` (R4C), against score rows permuted across kept neurons (median; 99th) | .375 vs .092 (.128) | .259 vs .016 (.031) | .163 vs .006 (.019) | .055 vs .000 (.016) |
| the same, co-firing | .291 vs .094 (.122) | .172 vs .018 (.033) | .103 vs .006 (.019) | .047 vs .000 (.016) |
| the same, whole-read similarity | .265 vs .094 (.126) | .134 vs .018 (.031) | .072 vs .006 (.019) | .016 vs .000 (.016) |
| the same, shared parent channel | .124 vs .092 (.122) | .045 vs .016 (.029) | .034 vs .006 (.016) | .008 vs .000 (.016) |
| Most common partner (R4C): neuron, how many kept neurons aim at it | n23, 68 of 501 | n72, 25 of 448 | n260, 14 of 320 | n284, 10 of 128 |
| Overlap (Spearman over kept neurons): s3:69 read vs `beta_C` mass / vs co-firing | .12 / .15 | .04 / .03 | .03 / -.01 | .21 / .21 |
| `beta_C` mass vs co-firing | .61 | .49 | .31 | .35 |
| Top partner: share of kept neurons whose entering partner rose more than the same pick among kept neurons (R4C; L) | .75 (.73) | .70 (.67) | .72 (.59) | |

The hub-matched null is close to 1/|D| at every cut: popular partners exist but do not make the hit rates. The 1/|D| ratios in the table above stand.

Reading, at true strength:
1. **Who aims and at whom follows firing overlap, as C's calculation already sees it.** C's write coefficients `beta_C` (how much of cut neuron d's firing C rebuilds from kept neuron j's firing, calculated from A alone) pick j's partner at 4x-27x the hub-matched null, every cut above its 99th. The partner sits in the top 5-10% of `beta_C` in the median. Co-firing in A is second (it overlaps with `beta_C`, rho .31-.61). The same holds for L, which never saw C. So, in this network, **where the trained gate change goes can be read off before training: at the pairs C already chose for the writes.** This joins the writes and the gates into one choice ([[feedback-unified-calc]]).
2. **The parent, as tested here, is a second and separate signal.** At cuts 11 and 64, neurons that read s3:69 more aim harder (rho .27 / .11, Q2 pass). This is nearly independent of `beta_C` mass and co-firing (rho .03-.15). But the one parent score I tested for the PAIR (the largest energy share on one shared stage-3 channel) failed Q3 and Q4. That is a failure of this score, not a finding that the parent plays no part. Section 2 found the parent as a wider shared read direction (s3:69 only a fifth of it). A pair score built from that direction is not tested.
3. **The aim appears when the partner is cut.** n339's aim at n476 goes from -.02 to .55 when n476 enters the cut. For the top partner, 59-75% of kept neurons show a larger rise than the same pick among kept neurons. Over all pairs the effect is small (Q1: +.007 to +.009, fail at 192->384).
4. **The tilt imports the partner's concept.** For the strongest aimers, the tilt reads as the partner's own read (porous / pitted; tent / escalator; coast / beach; car / van), while the kept neuron's own read is about something else.
5. **Aiming goes with slightly less own firing.** Stronger aimers keep slightly less of their own firing pattern (rho -.50 at cut 11, -.07 to -.17 later). At cut 11 the whole range is only .996-.999.
6. Limits: one seed and one data order for L and R4C. Development photos. Hit counts at 384 are 1-7 neurons. No bar was set for the `beta_C` result before the run. Closed cells (section 9: .07-.14 of on-cells at 64 and 192) are still unread, so "the flipped cells are near the line" is shown for opened cells only.

Next (one step): use `beta_C` as the "where" and the partner's read as the "what": tilt each kept read toward its top-`beta_C` partner's read, with no training, and score it with section 10's gate-map tables against the trained tilt.

## 12. Same cut, three start points, one gate tool (plan approved 2026-11-25: "Isolate where the spread comes from"; Shonu: "we're not really isolating what we need to isolate")

Why: every build that showed the spread (VECMATH 9-11) started from A's weights (INDEX thread 2). C spreads because its gates are fixed (INDEX thread 1). This section holds the cut fixed (384, 128 neurons left) and changes only the start point.

| Build | Start | lr, steps |
|---|---|---|
| C (= A's gates of the 128 kept) | A | none |
| L | cut network (A's weights minus the cut) | AdamW ~1e-4, 4,000 |
| R4C | C | AdamW, 4,000 |
| B0, B1 (B128aug seeds 0, 1; block 0) | random. Blocks 0 and 1 both trained | AdamW 3e-3, wd .05, 6,250 |

T0 (random, A's block 1 fixed) is a named gap: its weights are not on the Mac and not in `/bigtemp/fkm5yp/n11in/weights.tar.gz`.

Object: the gate map (cells where a neuron fires), on the 2,000 saved Broden held photos. Photo use: reading only. Nothing is fitted.

Measures (`code/oc_absorb.py`): rec(d, j) = share of A neuron d's on-cells inside small neuron j's on-cells. Null: j's map taken from the photo 1,000 places away, same positions (keeps j's rate, breaks the match). ex = rec - null.
- d side: for each of the 384 cut A neurons, excess recall of the top 1, top 2 and top 5 small neurons (union).
- j side: for each small neuron, the number of A neurons (of 512) it covers (rec >= .5), minus the null count.

| # | Prediction (written before the run) | Bar |
|---|---|---|
| P3 | known answer first: B0's n122 covers A's wheel neurons (n79, n81, n37, n297, n496, n303, n458, n103) | n122 in B0's top 3 by excess recall for at least 4 of the 8 |
| P1 | from random (B0, B1), cut neurons are absorbed more than in L and R4C | median top-2 excess recall of B0 and B1 above both L and R4C |
| P2 | from A (L, R4C), the small network is mostly copies. From random it is coarse | median excess cover: L and R4C <= 1, B0 and B1 > 1 |

Caveats written now: B n122 is not selective (rate .478, N11-rates), so the null matters. B differs from L in start point AND learning rate AND a trained block 1. This separates "from A" from "from random", not one knob.

### Results (2026-11-25, Mac, 17 s; `code/oc_absorb.py` md5 c687e02a; `results/oc_absorb/absorb.json` md5 2056e912, `phi.npz`, `rec.npz`, logs `run.log`, `p3_photo.log`, `planted.log`, `tails.log`)

Checks: A's block 0 reproduces `held_s4b0` (4.6e-4). C's gates equal A's (asserted).

Tool change after the smoke run (200 photos), before the full run: B's neurons fire at .40-.42 of cells, A's at .17. The planned recall measure ("covers" = 50% of an A neuron's cells inside j's) then counts firing rate: each B neuron "covered" 106-140 A neurons. Added a rate-fair match: phi = correlation of two on/off maps (0 at chance for any rate). The recall numbers stay in the json. Every reading below uses phi.

Known answers:

| Check | Result | Verdict |
|---|---|---|
| P3 as written (cells): B0 n122 in the top 3 for >= 4 of A's 8 wheel neurons | ranks 5, 55, 5, 23, 81, 13, 22, 63. phi .03-.17 | **fail** |
| P3 the N11 way (per photo, peak response; added after the fail) | n122 vs the wheel-group mean .458 (N11: .46), rank 2 of 128. Best single A neuron n37 (.39), as in N11 | the tool reproduces N11 |
| ...against random groups (added) | a random 8-group of A neurons gets a B match of .60 at the 95th percentile (seed 1: .56) | **the wheel match is not above random groups on these photos**. Not a valid known answer (agrees with N11-ruleR-act, withdrawn) |
| Planted union (added): a fake gate = OR of 3 random A neurons, 50 trials | all 3 members in the top 3 in 50 of 50. Member phi median .44 / .51 / .63 | **pass**. A union of A gates shows up at phi >= .4 |

| Build | cut neuron's best match, phi: pct 10 / 50 / 90 / 99 | cut neurons matched >= .4 | small neuron's best A match, median | small neurons matching >= 2 A neurons at .3 | firing rate, median |
|---|---|---|---|---|---|
| C (= A's kept gates) | .14 / .19 / .27 / .35 | 0 of 384 | 1.00 | 11 of 128 | .166 |
| L (from the cut) | .16 / .21 / .29 / .36 | 0 | .75 (.15 of them >= .8) | 19 | .257 |
| R4C (from C) | .16 / .21 / .28 / .38 | 2 | .82 (.67 >= .8) | 20 | .213 |
| B0 (from random) | .15 / .20 / .28 / .35 | 0 | .24 | 4 | .405 |
| B1 (from random) | .14 / .20 / .27 / .34 | 0 | .25 | 4 | .424 |

Null: |phi| 99th with j's map from another photo: .08-.09.

| # | Verdict |
|---|---|
| P1 (from random absorbs more) | **fail**. Top-2 excess recall B0 .174, B1 .177 against L .189, R4C .197. By phi all five builds are equal (.20-.21) |
| P2 (from A = copies, from random = coarse) | **half**. L and R4C are copies of their own A neuron (.75, .82). B's neurons are NOT coarse unions of A's: their best match to any A neuron is .24-.25, far below the .44 a planted 3-union gives |

Reading, at true strength (cut 384, gates only, Broden photos):
1. **No build takes over a cut neuron's gate.** In every build, from every start point, the best match to a cut neuron's on-cells is about .20. A's own kept neurons already gave .19 (C). Training from A adds .02. Training from random adds nothing. So at this cut the cut jobs are not carried by any neuron firing where the cut neuron fired.
2. **Trained from A, each kept neuron stays itself** (best match to its own A neuron .75-.82). The tilts of VECMATH 9-11 are small changes on top of a copy.
3. **Trained from random, the neurons are a new set.** They match no A neuron (.24) and no union of A neurons (a union would give >= .44). The N11 wheel neuron does not survive a random-group null on these photos.
4. So "every neuron takes over the cut neurons' job" is not what the gates show. Where the job goes (writes, or a code that no single gate shows) is not read here.
5. Limits: one cut (384), Broden photos (2,000), B trained blocks 0 AND 1 (its block 0 may use a private code, N11-write-B). T0 (block 1 fixed at A's) would be the clean case. Its weights are missing.

### Checks after the advisor review (`code/oc_absorb_check.py` md5 ed0037e9, Mac 8 s, 1,000 photos; `results/oc_absorb/checks.json`, `checks.log`)

| Check | Result |
|---|---|
| Power on known take-overs: gate phi C -> L / R4C | cut 11 n5->n464 .112 -> .138 / .135. cut 64 n339->n476 .127 -> .166 / .168 (rank 21 -> 9 / 7). cut 192 n430->n114 .048 -> .081 / .059. cut 384 n430->n501 .084 -> .111 / .090, n76->n374 .316 -> .366 / .334, n213->n501 .262 -> .277 / .264. Same pairs by firing (jobtest_L): take .03-.30, corr .21-.59 |
| Planted partial widening: C gate of a kept neuron OR a random 10 / 25 / 50% of a cut neuron's cells (100 pairs per cut) | phi .02-.03 -> .08-.11 / .19-.21 / .36-.39 |
| Strength, not on/off: best match of each small neuron to any A neuron by ReLU-output correlation (median, 90th) | L .936, .959. R4C .956, .972. B0 .313, .402. B1 .324, .386. Top-10%-cells phi: L .82, R4C .84, B0 .22, B1 .22 |
| Scope: cut neuron's best gate match, median C -> L / R4C | cut 11 .205 -> .213 / .211. 64 .213 -> .225 / .227. 192 .211 -> .218 / .219. 384 .194 -> .212 / .209 |
| N11's own wheel statistic on these photos (top 200 of 2,000 = 10%, N11 used 1%) | n122's top 200 inside the 8 wheel neurons' union .76 (union = 49% of photos). Random 8-sets: median .63, 95th .695. Best single A neuron .285 |

Corrections to the reading above:
- Reading 1 holds at every cut, with a measured size. The known take-overs ARE visible in the gates, and they are small: +.02 to +.05 in phi, which the planted test puts below a 10% widening of the cut neuron's cells. The larger take-over numbers of VECMATH 7 / 8.5 (take up to .30) are in firing STRENGTH, mostly where the kept neuron already fired (VECMATH 6, math limit 4).
- Reading 3 corrected. "The N11 wheel neuron does not survive a random-group null" is WITHDRAWN. That null used a different statistic (group mean of peaks). With N11's own union statistic, n122 passes (.76 against the random 95th .695), weakly. What holds: per photo, B's n122 sits inside A's wheel family. Per cell, B's neurons match no A neuron and no union of A gates (gate phi .24, strength correlation .31-.32, planted union .44+). So from random, the coarse unit is coarse in WHICH PHOTOS it fires on, not a union of A's gates cell by cell. Possible tool reason not tested: B may fire at other positions of the same object.
- The phi measure and its .3 / .4 / .8 lines were chosen after the smoke run, and P3 failed as written. This section is a development reading, not a confirmation.

## 13. The essay's reading of a repair (2026-11-25, reading only, no run; plan "Why many small gate changes fit the essay")

Shonu: "how it's possible that all these neurons are changing very slightly at their gates, and yet somehow it's still interpretable".

What a cut plus a short retrain from A's weights shows (L, R4C):
- Every kept neuron stays itself: firing correlation with its own A neuron .94-.96 at cut 384 (section 12 checks).
- Every kept read tilts a little: 3% of its length at cut 11, 20-23% at 384 (section 10).
- Each tilt aims at one cut neuron whose concept is related (section 11): n339 (tilt: car, wheel, van) -> n476 (van, car), n5 -> n464 (interlaced, paisley), n86 -> n23 (pitted, porous), n372 (tilt: coast, beach, ocean) -> n401 (coast, beach, sea).
- No kept neuron takes over most of a cut neuron's cells (section 12).

What the essay says about this case (`../curves/map_agents/essay.md`, line numbers checked):

| Line | Quote | Prediction for a repair |
|---|---|---|
| 617 | "established routes have a head start in capturing residual error. They already provide ways to change the answer, so optimization can build on them." | The lost job is residual error. Kept neurons that already fire near it take it first |
| 769 | "the pull toward the niche that already exists for that thing, or for the nearest thing" | Each kept neuron takes the part nearest its own job (the partner table) |
| 777 | "What wins is whatever captures the residual error first ... what usually wins is the right area for the thing." | the same |
| 611 | an attack recruits jaguar neurons "at roughly a quarter of their usual strength" | A repair shows as graded strength in existing neurons |
| 771 | "a network trained on cars ... recognizes cars fine before its car neurons have formed. At that point it's just more distributed." | Early in learning a job is spread out |
| 771 | "Consolidating into something like a neuron seems to be less about steering the model and more about buildability" | One neuron forms when later layers must build on the job |
| 251-253 | layer 3b cut 480 -> 10, "retrained from scratch": "one 'curves & loops' neuron" | The coarse neuron comes from training small from the start |

Reading:
1. Each small change is interpretable: "neuron X (job P) now also fires a little on nearby thing Q that was lost".
2. The sum is spread, because many existing routes each sit near some part of the lost jobs.
3. No consolidation, because the essay's reason for it is missing: block 1 and the head stay A's or barely move, so nothing builds on the repaired jobs. The repair only steers the output.
4. The essay's coarse neuron is a different experiment (trained small from the start). So sections 9-12 read a REPAIR object, not the essay's coarse object. The "contradiction with the essay" came from reading a repair as if it were that experiment.

Predictions of this reading (not tested):
1. Every tilt's partner is concept-related, not only the strongest aimers' (section 11 named 7).
2. A network trained small from the start has broader neurons than a short repair of A at the same size.
3. A repair where later layers must build on the repaired layer consolidates more than a repair with later layers fixed.
