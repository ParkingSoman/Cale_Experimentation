# CALC: calculate a smaller network from the big one, with no training

Opened 2026-11-25. Plan: `~/.claude/plans/okay-right-now-pasted-content-shimmering-sutherland-of-1683f6ca-8334-4714-814f-4aa273a70b06.md`.

Known answer: the CLIP-KD pair (session `clip-kd-mac-setup`, files in `clipkd/weights/`).

| Model | Vision blocks | Width (stream / MLP) | Vision params | Zero-shot top-1 (README) |
|---|---|---|---|---|
| Teacher: OpenCLIP ViT-B/16, Laion-400M | 12 | 768 / 3072 | 86.2M | 67.1 |
| KD student: ViT-T/16, distilled from that teacher, from random | 12 | 192 / 768 | 5.6M | 42.6 |
| Baseline: ViT-T/16, same shape, no teacher | 12 | 192 / 768 | 5.6M | 30.6 |

The KD student's loss is not pure distillation. It is captions + contrastive KD + feature MSE on the 512 output (weight 2000), on CC3M+12M, 32 epochs, from random.

## 1. Prior art (Step 0)

Read by three reader agents on 2026-11-25. Each one read part of its paper word for word and the rest only as a page summary. The "Source" column says which.

| Work | What it does | No-training result | Source |
|---|---|---|---|
| SliceGPT (arXiv 2401.15024) | Turns LayerNorm into RMSNorm. Rotates the stream at each block into the PCA basis of that block's normed input (1,024 calibration sequences). Deletes the smallest components. A D x D map on each skip link. No training | Llama-2 7B zero-shot mean 69.0 dense -> 55.5 at 25% stream removed -> 51.5 at 30%. Perplexity 5.47 -> 17.17 at 50%. No vision results | method: word for word. Numbers: summary |
| Model folding (arXiv 2502.10216) | k-means on each channel's [read | write] weights. A cluster's read = the mean, its write = the sum. Variance repair with no data (Fold-AR) or synthetic images (Fold-DIR). No training | LLaMA-7B at 20% removed: perplexity 5.68 -> 13.33, mean accuracy 65.6 -> 50.2. ResNet-18 CIFAR-10 at 70%: "over 80%". No LLM result above 20% | method: word for word. Numbers: summary |
| Minitron (arXiv 2407.14679) | Ranks heads, MLP neurons and stream channels by activation size on 1,024 samples. Trims them. Then distills | Right after pruning the loss is large (distillation loss 5.35-6.33 before, .37-.38 after 1.8B tokens). One-shot 15B -> 4B (73% removed) is worse than going in two steps. Prune + distill beats training from scratch with 40x fewer tokens | method: word for word. Some numbers: summary |

Where published methods stop, and what we aim at:
1. ViT-B to ViT-T removes 75% of the stream, 75% of the MLP neurons and 75% of the heads at once. Published methods with no training report cuts of 20-50% at most. This work goes past that point.
2. Those methods pick the kept part by generic rules (PCA, clusters, large activations). Our route is different: read what the network itself built (the essay), and calculate the small network from that. Step 2 shows how the KD student relates to the teacher's stream, and so what the calculation must produce.
3. The target: a calculated ViT-T/16 that matches the KD student (42.6 top-1, its KL to the teacher) with no training.

## 2. Results

### 2.1 The target, scored the way a calculated network is scored (2026-11-25, numpy on saved files, no model run)

A calculated ViT-T built from the teacher has no text tower of its own. It writes into the teacher's 512 output space, so it is scored with the teacher's class list and scale (100). The same scoring applied to the saved student outputs. Photos: `clipkd/out/*_test.npz`, the test split (odd positions, first 5 per class = 5,000 photos). Photo use: scoring only. "First 1,000" = the first test photo of each class, the default set of `eval_fast.py`.

| Model | Class list | Photos | KL to teacher | Top-1 | Agrees with teacher | Median cos of 512 output with teacher's |
|---|---|---|---|---|---|---|
| KD student | its own (scale 56.6) | 5,000 | 1.637 | 41.1% | 47.5% | .778 |
| KD student | the teacher's (scale 100) | 5,000 | **1.719** | **40.9%** | **49.0%** | .778 |
| KD student | the teacher's (scale 100) | first 1,000 | 1.801 | 40.7% | 48.2% | .776 |
| Baseline | its own (scale 74.6) | 5,000 | 3.800 | 27.8% | 30.9% | .011 |
| Baseline | the teacher's (scale 100) | 5,000 | 11.347 | 0.1% | 0.1% | .011 |

Reading:
1. The KD student's output IS in the teacher's space (cos .78 per photo). The baseline's is in an unrelated space (cos .01). So the KD student is a fair target for a calculated network, and the baseline is a control only on its own class list.
2. The target for a calculated ViT-T/16 (teacher's class list): KL 1.72, top-1 40.9%, agreement 49.0% on the 5,000 test photos (1.80 / 40.7% / 48.2% on the first 1,000, the fast default).
3. Its loss (CLIP-KD `loss.py`, checked by the Clip KD session): feature MSE between normalized student and teacher embeddings, image and text, weight 2000, plus KL between student and teacher image-text similarity logits. The teacher file is the exact one in the KD script (sha256 55e67d44...).

### 2.2 Is the KD student's stream a linear view of the teacher's? (Step 2, `clipkd/read_stream.py`, 55 s on the Mac)

Photos for reading only: 2,000 reading photos (fit positions), 1,000 to fit the map, 1,000 to score it, all 197 tokens. R^2 = share of the student's stream variance (all 192 dims pooled) that a linear map from the teacher's 768-d stream predicts on the score half. "Shuffled" pairs each student photo with another photo's teacher stream. "PCA192" uses only the teacher's top 192 principal directions.

| Point | KD R^2 | Baseline R^2 | KD shuffled | Baseline shuffled | KD from teacher's top 192 PCs | Teacher variance in its top 192 PCs |
|---|---|---|---|---|---|---|
| entering block 1 | .977 | .969 | .123 | .137 | .916 | .959 |
| after block 1 | .924 | .919 | .137 | .160 | .849 | .846 |
| after 2 | .850 | .855 | .210 | .205 | .791 | .880 |
| after 3 | .832 | .802 | .248 | .236 | .782 | .808 |
| after 4 | .768 | .728 | .264 | .235 | .721 | .892 |
| after 5 | .677 | .647 | .178 | .203 | .626 | .892 |
| after 6 | .599 | .558 | .147 | .156 | .547 | .939 |
| after 7 | .490 | .468 | .056 | .096 | .434 | .923 |
| after 8 | .414 | .394 | .032 | .052 | .365 | .890 |
| after 9 | .371 | .324 | .021 | .030 | .331 | .842 |
| after 10 | .311 | .294 | .015 | .023 | .277 | .795 |
| after 11 | .289 | .249 | .017 | .019 | .256 | .778 |
| after 12 | .256 | .228 | .018 | .019 | .218 | .756 |

The per-token normalized stream (what each block reads) gives the same picture within .01.

| Prediction | Result |
|---|---|
| K1: KD more linear in the teacher than the baseline, in every block | passes in 11 of 13 points (not after block 2, .850 vs .855), but by only .00-.05 |

Reading:
1. Early blocks: both students' streams are close to a linear view of the teacher's (R^2 .98 -> .83 by block 3), the no-teacher baseline as much as the KD student. So this part is shared by any network trained on photos, not made by distillation.
2. WITHDRAWN as a reading of the network (section 2.4): the low deep-block R^2 (.26) was my tool. It pooled all 197 tokens, but the output reads only the class token (`pool_type = 'tok'`). On the class token the same R^2 is .63 after block 12. Tool gap: pooling tokens the output never reads.
3. The teacher's top 192 principal directions give most of what is linearly shared over all tokens (.218 of .256 after block 12). Not yet redone on the class token.
4. The shuffled control is .12-.26 in early blocks: part of the early match does not depend on the photo (position and class-token structure). It falls to about .02 in the deep blocks.
5. Microscope checks not yet done: (a) pooled R^2 is dominated by a few high-variance stream dims; (b) all 197 tokens are pooled, but the output reads only the class token after block 12; (c) the essay's unit is the neuron, not the stream. Step 3 (MLP neurons) is the next read.

### 2.3 Does each student MLP neuron match a teacher neuron or a small group? (Step 3, `clipkd/read_neurons.py`, 77 s on the Mac)

Photos for reading only: the same 2,000 photos as 2.2. Firing = MLP GELU output at the same block, every token. "Best single" = the teacher neuron with the highest correlation on the FIT half, measured on the SCORE half (median over the 768 student neurons). "Group of 8" = R^2 on SCORE from the 8 teacher neurons most correlated on FIT, least squares (class token + 8 fixed patch positions per photo). The no-teacher baseline is the control, read the same way.

| Block | KD best single (median) | Baseline best single | KD 90th pct | KD share >= .5 | Baseline share >= .5 | Teacher neurons used as best match (KD, of 3,072) | KD group of 8, R^2 | KD single, R^2 | Baseline group of 8, R^2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | .606 | .581 | .803 | .626 | .714 | 565 | | | |
| 2 | .514 | .488 | .664 | .557 | .454 | 611 | | | |
| 3 | .365 | .350 | .519 | .129 | .096 | 583 | .330 | .138 | .292 |
| 4 | .335 | .303 | .482 | .085 | .051 | 563 | | | |
| 5 | .296 | .231 | .430 | .040 | .016 | 559 | | | |
| 6 | .259 | .208 | .421 | .030 | .016 | 582 | .152 | .064 | .102 |
| 7 | .220 | .182 | .377 | .018 | .007 | 563 | | | |
| 8 | .191 | .157 | .353 | .018 | .007 | 522 | | | |
| 9 | .153 | .119 | .354 | .016 | .003 | 431 | .054 | .021 | .031 |
| 10 | .109 | .095 | .243 | .007 | .001 | 331 | | | |
| 11 | .100 | .088 | .181 | .003 | .000 | 183 | | | |
| 12 | .089 | .076 | .149 | .000 | .000 | 273 | .038 | .010 | .037 |

| Prediction | Result |
|---|---|
| K2 (as written: beat a random group) | not used: picking the best group beats random by construction. The baseline is the null instead |
| K2 against the baseline: KD neurons match the teacher clearly better than baseline neurons | weak: KD is above the baseline by .01-.06 in median best single, at every block but not by much |

Reading (superseded in part by section 2.4):
1. Blocks 1-2: student neurons have close teacher partners (median .5-.6), with or without the teacher. Shared by any network trained on photos. Stands.
2. WITHDRAWN as a reading of the network: the low deep-block match (.09 in block 12) pools patch tokens the output never reads. My reading failed here. Tool gap: token pooling, and no calibration (two good networks of the same shape were never compared). Section 2.4 redoes it on the class token with that calibration.
3. WITHDRAWN with 2.
4. Microscope checks not yet done, in order: (a) depth: a student block may match a different teacher depth (cross-depth table); (b) per-token correlation mixes the class token and patch tokens, which do different jobs; (c) the essay's same-neuron-across-sizes evidence is for models trained on the same data, while this student saw CC3M+12M and the teacher Laion-400M. The baseline (same data as KD, no teacher) matching almost as well says the data, not the teacher, sets most of the internal match.

### 2.4 Microscope check: the class token only, cross-depth, and a calibration control (`clipkd/read_cls.py` 67 s + `clipkd/analyze_cls.py` 22 s CPU)

Why: the image output reads only the class token after block 12 (`pool_type = 'tok'` in open_clip). Sections 2.2-2.3 pooled all 197 tokens. Per-dimension R^2 from the 2.2 sums gives the same picture as pooled R^2 (largest student dim holds at most 5% of variance), so one large dimension was not the problem.

Photos for reading only: 4,000 reading photos, fit positions 0-3 of each class (FIT = 0-1, SCORE = 2-3). Ridge strength picked on an inner split of FIT. Calibration control: the KD student read from the no-teacher baseline (same shape, same data, no shared training). It shows what "two good networks" look like with this tool.

| After block | KD <- teacher, stream R^2 | Baseline <- teacher | KD <- baseline (control) | KD neuron <- teacher (median best single) | Baseline neuron <- teacher | KD neuron <- baseline (control) |
|---|---|---|---|---|---|---|
| 1 | .978 | .966 | .952 | .775 | .774 | .751 |
| 2 | .974 | .980 | .961 | .599 | .681 | .607 |
| 3 | .972 | .968 | .964 | .620 | .648 | .632 |
| 4 | .950 | .916 | .947 | .616 | .585 | .629 |
| 5 | .907 | .893 | .904 | .568 | .537 | .605 |
| 6 | .847 | .862 | .854 | .506 | .498 | .557 |
| 7 | .820 | .807 | .831 | .456 | .473 | .506 |
| 8 | .759 | .705 | .771 | .460 | .386 | .456 |
| 9 | .697 | .647 | .707 | .319 | .294 | .341 |
| 10 | .629 | .561 | .599 | .237 | .187 | .237 |
| 11 | .605 | .522 | .616 | .216 | .151 | .265 |
| 12 | .628 | .530 | .616 | .300 | .194 | .311 |

Cross-depth (class token): the KD student's stream after block i is best predicted by the teacher's after block i at the ends, and by about one block EARLIER in the middle (KD 6 <- teacher 5, KD 7 <- 6, KD 8-9 <- 7). Neurons: KD blocks 4-8 match teacher blocks 3-7 best. The full 12 x 12 tables are in `clipkd/out/analyze_cls.json`.

Reading:
1. On what the output reads, the KD student's stream is close to a linear view of the teacher's: R^2 .97 through block 3, .85 at block 6, .63 at block 12.
2. Calibration: the KD student is as close to the teacher as it is to a same-shape network trained on its own data (.628 vs .616 at block 12, neurons .300 vs .311). So a median best-neuron match of .30 in block 12 is the level two good networks share, not a failed match.
3. Distillation's mark: in deep blocks the KD student is closer to the teacher than the baseline is (stream .63 vs .53, neurons .30 vs .19 at block 12).
4. Depth: the narrow student runs about one block behind the teacher in the middle and catches up by block 12.
5. Not yet done: the 10 named neuron pairs read the essay's way (plan Step 3). Step 3 is half done.

## 3. The tree of distinctions (`THEORY.md`, plan `pasted-content-id-0373-while-federated-coral`)

Prior art read 2026-11-25 for this section (Neuron Merging 2010.13160, Hamadouche & Hussain 2607.05457): rows in `THEORY.md` section 2.

### 3.1 How broadly neurons fire on the class token (description, no prediction; saved `out/cls_*.npz`, numpy)

Photos for reading only: the 4,000 fit-position photos of 2.4. Share = share of photos where the neuron's GELU output is above 0.

| Block | Teacher mean share | KD mean share | Baseline mean share | Teacher p90 | KD p90 | Baseline p90 |
|---|---|---|---|---|---|---|
| 1 | .017 | .110 | .055 | .000 | .613 | .066 |
| 4 | .054 | .122 | .054 | .139 | .597 | .124 |
| 7 | .063 | .141 | .043 | .227 | .516 | .163 |
| 10 | .022 | .043 | .034 | .050 | .111 | .077 |
| 12 | .039 | .062 | .029 | .091 | .151 | .063 |

Reading: the MEAN share is higher for KD than for the teacher in every block shown (1.6x to 6.5x), and the baseline sits near the teacher. This is a tail fact: the medians are about equal, and a few KD neurons fire on nearly every photo (block 1 p99 = 1.000). It is a description, not support for the gate part of the tree. Median share is near 0 for all three on the class token: most class-token neurons are quiet on most photos.

### 3.2 Does a student neuron match a teacher branch root? (`clipkd/read_cfc.py` 67 s Mac torch + `clipkd/tree_roots.py` 53 s numpy, 2026-11-25)

Photos for reading only: the same 4,000 fit-position photos (FIT 0-1 picks, SCORE 2-3 measures). Tree, roots, random roots, look-alike means: weights only. Co-firing groups: FIT-half pre-activations. Unit = class-token MLP pre-activation. Each student neuron picks its best unit among teacher blocks b and b-1 (3,072 units per block per set). Numbers = median over the 768 student neurons, SCORE half. Output `clipkd/out/tree_roots.json`.

Checks: recomputed teacher pre-activation through GELU equals the saved firing (max diff .0003). P0 planted root matched back 99.5% (bar 90%): PASS. Shuffled photos: -.004 to .005.

| Block | KD single | KD root | KD root (fix) | KD random root | KD co-firing root | KD look-alike mean | Base single | Base root | Base co-firing | G1 share of branches >= .75 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | .669 | .647 | .652 | .672 | .696 | .658 | .633 | .615 | .652 | .50 |
| 5 | .627 | .611 | .611 | .615 | .670 | .617 | .602 | .586 | .630 | .51 |
| 6 | .566 | .547 | .537 | .551 | .608 | .563 | .558 | .545 | .608 | .55 |
| 7 | .536 | .515 | .510 | .527 | .585 | .543 | .565 | .548 | .618 | .57 |
| 8 | .592 | .566 | .560 | .581 | .655 | .595 | .514 | .496 | .559 | .60 |
| 9 | .527 | .513 | .511 | .508 | .591 | .534 | .498 | .473 | .544 | .75 |
| 10 | .476 | .452 | .443 | .441 | .533 | .474 | .448 | .426 | .506 | .69 |
| 11 | .442 | .424 | .415 | .401 | .506 | .451 | .421 | .400 | .481 | .70 |
| 12 | .475 | .475 | .465 | .428 | .539 | .492 | .412 | .410 | .469 | .76 |

| ID | Prediction | Result |
|---|---|---|
| P0 | planted root found >= 90% | .995 PASS |
| C1 | root beats random AND co-firing by >= .02 in >= 6 of blocks 4-12 | FAIL: co-firing beats root in all 9 blocks (by .05-.09). Root beats random by >= .02 only in blocks 11-12 |
| C2 | root beats single by >= .02 in >= 6 blocks | FAIL: root is .00 to .03 BELOW single in every block |
| C3 | KD-baseline gap larger for roots than singles | FAIL: equal within .01 (single -.029 to .078, root -.033 to .070) |
| G1 | when a child is in its top 10%, root above its median in >= 75% of cases for >= half the branches | met in 9 of 9 blocks (.50-.76 of branches), but UNINFORMATIVE: no null, and the root is made from the children's reads |
| fix | root = the parent's write as the siblings read it | also below single (by .01-.03). Stopped here, as planned |

Reading:
1. This root rule lost. Its gaps (tool, not network): it reads adjacent MLP-to-MLP links only and skips attention and later blocks. It reads the class token only. It fixes k = 8, where the median child cosine is only .12-.27. The record (N11-weights, OC-fold-w) pointed the same way.
2. The registered rival won: co-firing-group roots beat single teacher neurons by .027-.064 (KD) and .019-.060 (baseline) in every block 4-12. The baseline never saw the teacher and gets the same gain, so this is NOT evidence that the student keeps the teacher's branches. It is the same outcome that withdrew N11-ruleR-act. Open lead only.
3. Not yet separated: co-firing roots lean toward high-variance directions, which may favour them by themselves. First next step: a control set of plain high-variance directions (same count, same scoring). A weight-only form of co-firing (reads compared through the span of upstream writes) comes after that control.
4. The look-alike mean (model folding) is equal to single neurons in early blocks and .01-.02 above in blocks 9-12.
5. Distillation's mark again: KD matches the teacher better than the baseline in blocks 4-5 and 8-12 (single: +.02 to +.08).

### 3.3 Vote families: siblings = neurons that take the same votes from the same parents (`clipkd/tree_roots.py`, 72 s numpy, 2026-11-25)

Plan: "find families by who reads the parent's output". Vote list of neuron j in block b: `v_j[p] = (read_j * ln2 gain) . write_p * s_p` for all 3,072 MLP parents p of block b-1. Family = j + its 7 nearest vote lists (cosine). Root = top shared read direction, as before. `s_p` = RMS firing of parent p (class token). Map A: `s_p` from position-0 photos (1,000). Map B: position-1 photos (1,000, disjoint). Map W: weights only (`s_p` = 1). Scoring as in 3.2. Old sets reproduce 3.2 exactly. P0 .995.

| Block | KD single | KD vote A | KD vote B | KD vote W | KD co-firing | Base single | Base vote A | M1 overlap A-B | M5 overlap W-A | members co-fire: vote A | vote W | 3.2 kids | random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | .669 | .655 | .656 | .662 | .696 | .633 | .623 | 1.00 | .86 | .190 | .190 | .171 | .125 |
| 5 | .627 | .616 | .616 | .613 | .670 | .602 | .587 | 1.00 | .86 | .173 | .176 | .142 | .103 |
| 6 | .566 | .552 | .551 | .548 | .608 | .558 | .548 | 1.00 | .86 | .164 | .162 | .123 | .097 |
| 7 | .536 | .523 | .522 | .530 | .585 | .565 | .553 | 1.00 | .14 | .137 | .152 | .115 | .101 |
| 8 | .592 | .559 | .561 | .590 | .655 | .514 | .486 | 1.00 | .14 | .121 | .145 | .113 | .106 |
| 9 | .527 | .508 | .503 | .527 | .591 | .498 | .468 | 1.00 | .14 | .116 | .122 | .100 | .094 |
| 10 | .476 | .463 | .464 | .468 | .533 | .448 | .429 | 1.00 | .29 | .109 | .114 | .089 | .082 |
| 11 | .442 | .449 | .448 | .451 | .506 | .421 | .420 | .86 | .43 | .140 | .132 | .084 | .072 |
| 12 | .475 | .493 | .492 | .494 | .539 | .412 | .425 | .86 | .43 | .161 | .151 | .105 | .068 |

| ID | Prediction | Result |
|---|---|---|
| M1 | maps A and B give the same families, overlap >= .8 | PASS: 1.00 in blocks 2-10, .86 in 11-12. Note: `s_p` is an RMS over 1,000 photos, a very stable number, so this pass is expected for this map |
| M2 | family members co-fire, median abs corr >= .3 | FAIL: .11-.19 in blocks 4-12. Above the 3.2 kids (.08-.17) and random groups (.07-.13) in every block |
| M3 | vote root beats single by >= .02 in >= 6 of blocks 4-12 | FAIL: vote A is .01-.03 below single in blocks 4-10, +.007 and +.018 above in blocks 11-12 |
| M4 | vote root within .02 of co-firing root in >= 6 blocks | FAIL: co-firing is .04-.10 higher in every block |
| M5 | weight-only map W gives the same families as map A, overlap >= .5 | mixed: .86 in blocks 2-6, .43 in 11-12, .14-.29 in blocks 7-10 (fail) |

Reading:
1. A second sample of photos from the same 1,000 classes gives the same families (A = B). A different kind of photo is untested. The weights alone give the same families only in blocks 2-6 (.86). In blocks 7-10 the photo numbers change the map (.14-.29).
2. The 3.2 diagnosis applies again: members co-fire only .11-.19, and the vote root's best match is one of its own 8 members in 89-99% of roots (read-only check, blocks 4, 8, 9, 12). The root is a blurred copy of one member. Vote roots are .01-.03 below single neurons in blocks 4-10, +.007 and +.018 in blocks 11-12. The baseline also gains at block 12 (+.013), so that gain is not distillation's.
3. Tool gap: `s_p` is an RMS, so an always-on parent counts as loud. In blocks 8-9 the top 10 parents hold 31-33% of all `s_p` squared (block 4: 1%). That is where maps W and A differ.

### 3.4 Control: is the co-firing win a family, or a lean toward high-variance directions? (read-only numpy, 2026-11-25, no file written)

Sets, 3,072 per teacher block, same scoring as 3.2: H1 = random directions shaped by the input spread (`g @ Sigma^(1/2)`, no neuron information). H2 = each teacher read tilted toward high variance (`Sigma @ r_j`). `Sigma` = covariance of the class-token MLP input on the FIT half.

| Block | KD single | KD co-firing root | H1 random, variance-shaped | H2 tilted single |
|---|---|---|---|---|
| 4 | .669 | .696 | .695 | .711 |
| 5 | .627 | .670 | .663 | .686 |
| 6 | .566 | .608 | .577 | .606 |
| 7 | .536 | .585 | .558 | .589 |
| 8 | .592 | .655 | .633 | .664 |
| 9 | .527 | .591 | .568 | .590 |
| 10 | .476 | .533 | .509 | .544 |
| 11 | .442 | .506 | .473 | .507 |
| 12 | .475 | .539 | .461 | .519 |

| ID | Prediction (written before the run) | Result |
|---|---|---|
| H1 | random variance-shaped directions well below single | FAIL: they BEAT single neurons in blocks 4-11 (+.02 to +.04) with no neuron information at all |
| H2 | tilted singles at least .02 below co-firing in >= 6 blocks | FAIL: within .02 of co-firing in all 9 blocks (-.020 to +.016) |

Reading:
1. The co-firing win was the lean toward high-variance directions, not a family. The lead from 3.2 is withdrawn.
2. The yardstick favours high-variance directions: a student neuron matches any direction that leans toward where the teacher's input varies most. This fits CALC 2.2: the student's 192-wide stream is close to a linear view of the teacher's top-variance directions.
3. So far no family rule (weight-children, parent's write, vote lists, co-firing) beats a single teacher neuron once this lean is controlled.
4. Next target: the stream part of the tree. Test whether a student neuron = a teacher read seen through the student's narrow stream (the teacher read projected on the teacher's top 192 stream directions), and then families inside that view, scored against the tilted singles (H2) as the yardstick.

### 3.5 Reconcile with the essay's coarse neuron: strongest crops and unions (`clipkd/read_maxfire.py` 138 s Mac + `clipkd/union_check.py` 8 min numpy + `clipkd/eye_sheets.py`, 2026-11-25)

Why (Shonu: "if it disagrees, then our build is wrong, not the essay"): 3.2-3.4 had three microscope errors against the essay's object.

| Our build (3.2-3.4) | The essay's coarse neuron | Effect |
|---|---|---|
| merge = shared read direction (an AND) | union of the family's firing (an OR): "one 'curves & loops' neuron instead of ten curve detectors" (line 251) | sibling reads are mostly private, so the shared direction collapsed onto one member (88-100%) |
| linear correlation of pre-activations over all photos | strongest firings ("five strongest natural crops", line 253) | ruled by the quiet bulk. Plain high-variance directions "won" (3.4) |
| class token only | strongest crop anywhere | most class-token neurons are quiet |

Photos for reading only: fit positions 0-7 of each class (8,000). FIT 0-3 picks members, SCORE 4-7 measures. Unit firing = max over 197 tokens. Strong set = top 1% (40 photos per half), same size for every unit. Union = per photo, max over members of firing / member's FIT 99th percentile. Greedy members from teacher blocks b and b-1.

| Block | U0 planted recovered | KD k=1 | KD k=2 | KD k=3 | KD k=4 | KD k3-k1 (median per neuron) | KD shuffled | Base k=1 | Base k=3 |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.00 | .050 | .050 | .050 | .025 | .000 | .011 | .050 | .050 |
| 6 | 1.00 | .075 | .075 | .075 | .075 | .000 | .010 | .050 | .050 |
| 8 | 1.00 | .125 | .125 | .125 | .125 | .000 | .009 | .125 | .100 |
| 10 | 1.00 | .125 | .125 | .100 | .100 | .000 | .012 | .075 | .075 |
| 12 | 1.00 | .100 | .100 | .100 | .100 | .000 | .008 | .050 | .050 |

(Blocks 5, 7, 9, 11 in `out/union_check.json`, same pattern. U0 .96-1.00 in blocks 4-12.)

| ID | Prediction | Result |
|---|---|---|
| U0 | planted union of 3 recovered >= 80% | PASS (.96-1.00) |
| U1 | union of 3 beats best single by >= .05 | FAIL: k3 - k1 = .000 in every block |
| U2 | rises to k=3 then flattens | FAIL: flat from k=1 |
| U3 | baseline same as KD | not testable (no gain in either) |

Overlap levels: the best teacher unit holds 2.5-12.5% of a student neuron's SCORE top photos (chance about 1%). KD is above the baseline in blocks 6-12.

**Eye reading (plan rule after a U1 fail). SUPERSEDED by 3.6: this table is a noun reading (the worked wrong example in `../NEURON-CHECKLIST.md`).** 10 KD neurons, 5 random from block 8 and 5 from block 12 (seed 7). Sheets in `clipkd/out/union_eye/`: row 1 = student top 6, row 2 = best single teacher neuron, rows 3-4 = union members 2-3, row 5 = union top 6. My reading, NOT blind, and the members were chosen on the student's own FIT photos.

| Student neuron | Student's job | Best single teacher neuron | Members 2-3 | Student broader than the single? |
|---|---|---|---|---|
| b8 kd478 | printed text and graphics on objects | labels and diagrams on products | dials and packaging; screens and devices | about the same |
| b8 kd524 | people's clothing (hoods, knits, scarves) | clothing on people | skirts and suits; people with instruments | about the same |
| b8 kd721 | hand-held devices: camera, gun, knife | cameras and boxy appliances | guns; radios and panels | yes: camera + gun |
| b8 kd444 | fine repeated lines: mesh, blinds, strands | things behind mesh or grids | screws, threads, hair; scales and grids | yes |
| b8 kd688 | animals and plants outdoors | forest-floor animals and plants | water and landscapes; dogs | yes |
| b12 kd4 | babies and toddlers | toddlers | babies in hats; babies | same |
| b12 kd218 | large buildings: palaces, towers, temples | churches and barns | mixed; domes and minarets | yes |
| b12 kd229 | plated food | plated dishes | pasta, pizza, fries; plated meals | about the same |
| b12 kd699 | maps, compass, dials, screens | clocks | screens and rulers; computers | yes: dial + screen |
| b12 kd669 | farm machinery and outdoor work | corn and harvest | logs and timber; tractors and barns | yes: corn + timber + tractor |

Reading:
1. By eye, every one of the 10 student neurons does the same KIND of job as its best teacher neuron. In 6 of 10 the student covers what 2-3 teacher neurons split between them (camera + gun, clocks + screens, corn + timber + tractors). That is the essay's coarse neuron. Not blind, so it is a lead for a blind test, not a result.
2. Two microscope errors explain why the numbers missed it:
   a. The union's top photos are the loudest member's top photos. In 6 of 10 sheets the union's top 6 come from one member (5 repeat the single, row 2. 1 repeats member 3). Dividing by each member's 99th percentile still lets one member's extreme values own the top 1%. So k = 2-4 could not change the score.
   b. Exact-photo overlap of the top 1% is too strict for a broad job. A "babies" job has far more than 40 qualifying photos, and two neurons that both do it pick different top 40s. The baby neuron pair (b12 kd4) has 26 of 40 FIT overlap, the highest, because that job is narrow.
3. Next fix (the essay's microscope, not a new object): a union whose strong set takes an EQUAL share of top photos from each member, and a job-level score (does the student's strong set fall inside the members' broader top 5%?), with the same-size rule kept. Then a blind agent reading of the sheets (student vs single vs union).

### 3.6 The checklist reading: poles, feeders and the tree as verbs (`clipkd/read_poles.py` 210 s Mac + `clipkd/tree_verbs.py` numpy, 2026-11-25)

Plan: `~/.claude/plans/id-like-you-to-fizzy-cherny-of-28afb932-72ad-4399-8251-822044650347.md`. Checklist: `../NEURON-CHECKLIST.md`.

Units: the 10 KD student neurons of 3.5, their 30 teacher members (best single, m2, m3), and 40 random teacher neurons per teacher block (the pool, for nulls). Photos for reading only: 8,000 fit-position photos. FIT 0-3 for crops and readers, SCORE 4-7 for judges and out-of-sample checks. Output `clipkd/out/poles.npz`, `clipkd/out/tree_verbs.json`.

Tool checks (all PASS):

| Check | Result |
|---|---|
| token t = patch row (t-1)//14, column (t-1)%14 (change one patch, only that token's embedding moves) | PASS, both models |
| max over tokens = saved `maxfire_*.npz` | rel diff .0003 teacher, .0004 KD |
| min pole value = cross pre-activation at that token | diff 0.0 |
| feeder contributions + constant = pre-activation (direct path, the token's own LayerNorm scale) | median rel error 4e-7, p99 2e-6 |
| class token is the max token | 1.7% of photos (named units), .9% (pool) |

**NOT A TREE TEST (correction, 2026-11-25, Shonu).** The "members" were picked by shared photos with a student neuron (3.5). They are co-firing units, not a parent and its children: the child rule (README "Finding children" R1: readers of the parent's write, in weights) was never used. T2-T6 below say nothing about parents and children, and their tree names ("siblings", "parent", "branch") are wrong for these units. Co-firing units fire on each other's photos by the way they were picked, so T2 could not find turning away. The actual tree test is CALC 3.7 (ResNet-18 parent s3:69 and its R1 children).

Number tests on co-firing units, rules fixed in `tree_verbs.py` before the full run:

| ID | Prediction | Result |
|---|---|---|
| T3 | fewer than 3 of 10 groups split (student < 0 at a member's top tokens, or a member > 0 at the student's negative tokens) | FAIL: 10 of 10 split. Almost all of it is the student being negative at the member's exact token (22 of 30 members) |
| T2 | at least half the same-block member pairs gate each other (i below the pool's 10th percentile on j's top tokens) | FAIL: 2 of 32 |
| T4 | at least 6 of 10 groups share top-10 feeders above the matched-pool 95th percentile | FAIL: 1 of 10 |
| T6 | description: rank of a block b-1 member among the block b member's MLP feeders | 14 pairs. Best ranks 403 and 986 (of 21,504 and 33,792 MLP feeders). Its contribution is .008 or less, against a top feeder's .03-.55. The lower member is never a top feeder of the upper one |

Microscope check after the result (SCORE photos, not used to pick members):

| Check | Result |
|---|---|
| student's max firing on each member's top-40 SCORE photos, as a percentile of the student's own range | .42-.99 (28 of 30 above .6). Pool neurons: median .42-.56 |
| distance between the student's peak token and the member's peak token, on the member's top photos | median 3.5-12 patches. Random token pairs: 6. Same token: 0-17%, except 2 members (75%, 57%) |
| (post hoc T2) member i's max firing on member j's top SCORE photos | 0 of 32 pairs gate. Every pair is above the pool median (.55-.98 against .41-.62) |
| (post hoc T4b) cosine of whole MLP-feeder vectors between members, against matched pool triples | above the pool's 95th percentile in 3 of 10 groups (g0 .084, g2 .268, g4 .072) |

Reading of the numbers:
1. The student and its members fire on the SAME PHOTOS (out of sample), but at DIFFERENT TOKENS of those photos. Token-level comparisons across two models (T2, T3) read one model at the other model's token. In deep ViT blocks that is the wrong microscope. The registered T2 and T3 fails stand as written. They do not show a split.
2. Members fire on each other's top photos. This follows from how they were picked (shared photos). It says nothing about siblings.
3. Members mostly have different feeders (T4 1 of 10, T4b 3 of 10). The b-1 member is never a top feeder of the b member (T6). Co-firing units need not share a parent, so this is not a tree result.
4. Tool gap (reader 1): a crop at the photo edge is filled with a mirror image (reflect padding). Readers were told nothing about it. Next crop tool: grey fill.

**Checklist reading (Step 2).** 8 blind readers (opus): 4 for positive poles, 4 for negative poles. Each read 10 anonymous units, one per group, FIT crops only. Cards: `clipkd/cards/reader*_{pos,neg}.md`, key in `clipkd/cards/KEY.json`. The table compresses their words. Token type is from `poles.npz` (share of top-40 crops on the class token).

<!-- verb-lint from here -->

| Group | Student (token) | Student verb: positive vs negative | Members' verbs (single; m2; m3) | Members vs student |
|---|---|---|---|---|
| g0 b12 kd218 | class token (.70) | tall narrow upright form rising against open sky, ornate many-windowed top vs close-packed small repeated units on patches | straight edges meeting at a peak, or parallel upright runs, on flat matte stone-toned surface; warm brown fine grain or recessed cells; fine regular small units over one broad smooth curved shape | surface facets of the same scene, at patch scale |
| g1 b12 kd229 | class token (1.00) | close heap of small rounded irregular glossy pieces in mixed warm colours filling a round dish-shaped area vs warm soft lumpy close surfaces on patches | smooth pale curved border line at the photo edge, close tabletop view; heap of small glossy warm pieces; rounded lumpy matte warm surfaces with crumbly grain | the same heap at patch scale, plus a border facet |
| g2 b12 kd4 | class token (1.00) | one small rounded soft-contoured form filling the frame, pastel, soft indoor light, low close view vs smooth pale pink-beige patches with small dark spots | same whole-photo layout (class token, 1.00); smooth pale pink-beige rounded surface with small dark spots; same | same job; m2 and m3 are its surface facet |
| g3 b12 kd669 | class token (.97) | rough natural ground in daylight with a sturdy angular apparatus standing on it vs fine crumbly grain on patches | rows of small rounded gold units and dry straw grain, gold field band (both tokens); rough brown stacked rounded units and cut grain; coarse dry parallel fibres or planks, rural daylight | earthy rough fibrous surfaces: facets at patch scale |
| g4 b12 kd699 | class token (1.00) | flat panel crowded with many small regular high-contrast marks vs the same small marks on patches | evenly spaced ticks along a curved border or straight bar; rows of small printed glyphs on a flat rectangle (class token); flat rectangular frame edges with fine rows of small marks | same pattern; members are sub-layouts (round scale, grid, frame) |
| g5 b8 kd444 | patch | thin strands crossing at angles into an open lattice vs smooth curved hard shells with nested round openings | thin straight bars crossing into a grid; faint thin semi-transparent strokes over a smooth surface; printed letter strokes, often at the photo border | single = same; m2, m3 = thin strokes that do not cross |
| g6 b8 kd478 | patch | thin dark printed lines and small glyphs on a flat field vs repeated straight parallel bars | printed outlines and lettering on flat colour; regular arrays of small rounded rectangles, slots and print lines; the corner of a flat rectangle framing small printed detail | same pattern in three layouts |
| g7 b8 kd524 | patch | soft draped folds where a soft material edge meets a smooth warm surface vs plain ground plane at the bottom of the photo | smooth warm surface meeting a sharp folded band, a V line; soft V fold at the bottom edge; smooth warm shaded surface ending at a rounded contour | same meeting line; facets |
| g8 b8 kd688 | patch | fine busy natural ground texture around a central subject vs upright dark forms standing on a flat grey plane | fine clutter of thin short strands in earth tones; near-uniform field with soft thin streaks in one direction; flat ground plane with fine even grain | same ground texture; m2 = a smoother facet |
| g9 b8 kd721 | patch | small crisp round hole or dot set into a smooth flat hard plate vs thin long straight dark lines on a pale ground | small round element set into a flat smooth plate; hard dark surface with small sharp repeated parts catching highlights; smooth hard moulded surface broken by a crisp seam or small round hole | same pattern |

Groups the 3.5 noun lists hid (the worked wrong example): "camera + gun" (g9) is ONE pattern, a small crisp round hole or part set into a hard plate. "Clocks + screens" (g4) is ONE pattern, a flat panel crowded with small regular marks. "Corn + timber + tractors" (g3) is ONE pattern, rough fibrous earthy surfaces in a rural daylight scene.

**Blind test (Step 3).** 6 judges (sonnet), SCORE crops only. Sheet = 24 top crops of m2 and m3 + 24 top crops of pool neurons of the same blocks. Scores: `clipkd/judge/SCORES.json`.

| ID | Prediction | Result |
|---|---|---|
| J1 | student's words pick m2/m3 above shuffled words in >= 7 of 10 groups | PASS: 9 of 10 (sheet A alone 7, sheet B alone 8). Hit rate .756 vs .625 |
| J2 | single teacher neuron's words do worse than the student's | FAIL: .740 vs .756. Lower in 5 of 10 groups, by 1-3 hits |

Sheet fault: in the 7 groups whose members share one block, sheet B had 16 distractors, not 24 (pool of 40, sheet A used 24). Those B sheets had 40 tiles. All three word conditions saw the same faulty sheet, so S vs T vs X stays a fair comparison there. Sheet A alone gives the same verdicts.

**T1 and T5.** T1 was conditional on J1 and J2 passing. J2 failed, so T1 is not scored. By the readings, the members' verbs are the student's pattern or a facet of it at patch scale, (no member has a different pattern A). This is a reading of co-firing units, not of children. T5 was not run: T4 failed, so there is no shared parent to read.

**Token type (read-only numpy on saved files).** The vision tower pools the class token only (`pool_type` = `tok`). So in block 12 an MLP firing on a patch token is read by nothing, in all three models. Block-12 patch firings are dead. In blocks 1-11 patches are live (later attention reads them).

| Share of neurons whose top-1% firings are mostly on the class token (`cls_*.npz` vs `maxfire_*.npz`, 4,000 photos) | Teacher | KD | Baseline |
|---|---|---|---|
| block 9 | .002 | .000 | .000 |
| block 10 | .001 | .005 | .000 |
| block 11 (live patches) | .005 | .211 | .046 |
| block 12 (patches dead) | .021 | .401 | .250 |

Block-12 students, compared on the class token (live) against teacher units. Overlap of top-40 photos (of 2,000, SCORE half; best teacher unit picked on the FIT half; chance about 1):

| Group | teacher block 12, class token | teacher block 11, class token | teacher block 11, max over live patches |
|---|---|---|---|
| g0 | 7 | 18 | 10 |
| g1 | 18 | 19 | 12 |
| g2 | 26 | 17 | 16 |
| g3 | 8 | 15 | 2 |
| g4 | 12 | 8 | 7 |

Gate numbers (checklist step 3, `out/tree_verbs.json` "gate"). Sum of a student's top-10 up-voting feeders, at its top crops and at its negative crops. Feeder names in the json use 0-based blocks.

| Student | at top crops | at negative crops | Top up-voters |
|---|---|---|---|
| block 8 (g5-g9) | 2.2 to 3.6 | -.87 to -1.46 | attention heads of blocks 6-8 in all five |
| block 12 (g0-g4) | 1.1 to 4.3 | -.45 to 1.7 | attention heads of blocks 8-12, plus 2 MLP neurons in g2 |

Reading:
1. One verb covers the union (J1 passes, out of sample). The best single teacher neuron's words cover it about as well (J2 fails). So there is no evidence that the student's job is wider than its best teacher neuron's job. m2 and m3 read as facets of the same pattern. The members (co-firing units, not children) fire on each other's photos and mostly have different feeders. This is the 3.5 result (k3 - k1 = .000), read the essay's way.
2. Judge sheet leak: targets are 12 crops from each of 2 neurons, distractors 1 crop from each of 24 neurons. A judge can find the two clusters without the words. That likely explains why shuffled words score .625, above .5. So J1's margin (.756 vs .625) is the words' real gain, and J2's small gap cannot separate S from T at this power.
3. Block 8: a student and its members do the same job on the same kind of token (11 of 15 members above the pool's 90th percentile at the members' own patches). The student's negative pole is where its up-voting heads switch off. Most of its feeders are attention heads.
4. Block 12: only the class token is live. The 3.5 union check (max over 197 tokens) mixed dead patch firings into block 12. For block 12, the class-token microscope of 3.2-3.4 was the right one. The students' negative-pole readings in g0-g4 describe dead patch tokens: they are a record, not a gate reading.
5. Block 12 on the class token: the student's photos match a teacher class-token unit (block 11 or 12) as well as or better than any live teacher patch unit, in all 5 groups. The teacher carries the same jobs on its own class token.
6. Block 11: KD does 21% of its jobs mainly on the class token, the baseline 5%, the teacher .5%. This is a description of distillation's mark, one block before the output. Its role is not yet tested.

Deviations from the plan: the `map-reader` agent was not usable (tied to CrakBox tools), so plain agents read. Each sheet and word condition had one judge, not two (6 judges, 10 sheets each). T5 was not run. J1 passes in g0 and g9 by less than 2 hits. Crops at the photo edge were filled with a mirror image.

### 3.7 The actual tree: parent s3:69, its R1 children, and the coarse neuron n101 (ResNet-18, pre-registered 2026-11-25 before any run)

Plan: `~/.claude/plans/id-like-you-to-fizzy-cherny-of-28afb932-72ad-4399-8251-822044650347.md` Step 2. Network A = ResNet-18. B = `B128aug_s1` (stage 4 trained at 1/4 width on A's stage 3). Shonu's frame: a neuron is a distinction, so children split what the parent lets through and turn siblings away.

| Object | Picked by |
|---|---|
| Parent | stage-3 channel s3:69 (Shonu's parent since 2026-11-21) |
| Children | the top 11 block-0 neurons of A's stage 4 by the child rule R1 (read weight on s3:69 x write norm, `onecoarse/results/oc_children_w.npz`): n305, n97, n23, n163, n385, n297, n464, n237, n108, n510, n275 |
| Coarse neuron | B's block-0 neuron n101 (matched to s3:69 in the onecoarse record) |
| Null neurons | 50 random A block-0 neurons with R1 score below the median (seed 3); for C5, the other 127 B block-0 neurons |

Photos for reading only: 4,000 test photos (4 per class by position, labels unused). Cells = the 7 x 7 grid of stage 4 block 0. A cell reads a 3 x 3 window of the 14 x 14 stage-3 map. Parent value at a cell = max of s3:69 (stage-3 output, after the ReLU) over that window. Parent-on = parent value in its top 20% over all (photo, cell). Child top cells = its 200 photos with the largest max pre-activation, at the max cell. Negative cells = its 200 photos with the lowest min pre-activation, at the min cell. Pre-activation = bn1(conv1) of block 0.

| ID | Tree part | Prediction | Pass rule (code) | Fails if |
|---|---|---|---|---|
| C1 | children split what the parent lets through | a child's top cells are parent-on | share of top cells that are parent-on > the null neurons' 95th percentile | under 8 of 11 children |
| C2 | children turn siblings away (weights) | child i votes down on child j's own conditions | j's conditions = j's top 3 read channels (bn-scaled, summed over 3 x 3, channel 69 left out). Vote = i's mean read on them. A pair turns away if the vote is below the 25th percentile of i's votes on 2,000 random channel triples | under 36 of 110 ordered pairs (binomial p < .05 against 25%) |
| C3 | children turn siblings away (firing) | on parent-on cells, child i is lower where child j fires | J = parent-on cells where j is in its top 10% (within parent-on). d = median of i on J minus median of i on all parent-on cells. A pair turns away if d is below the 10th percentile of i's d against the 50 null neurons in place of j | under 55 of 110 ordered pairs |
| C4 | a child's negative pole is a gate on the parent's group | a child's most negative cells are parent-on | share of negative cells that are parent-on > the null neurons' 95th percentile | under 8 of 11 children |
| C5 | the coarse neuron keeps the parent and drops the split | n101 fires on the union of the children's top cells, reads s3:69, and does not vote down on the children's conditions | coverage = share of the union where n101 is in its own top 10% of cells; n101's read on channel 69 ranks in its top 10 channels; n101's mean vote on child j's conditions is negative for at most 5 of 11 children | coverage under .5 or not above the other B neurons' 95th percentile; or channel 69 outside the top 10; or negative votes on 6 or more children |

Checks before any result: A's pre-activations from the torch model equal the ones from `w_A.npz` (bn rows = weight, bias, running mean, running var); the cell-to-window mapping is checked by changing one stage-3 window and watching which cell moves.

**3.7 results** (`onecoarse/code/oc_tree.py` 9 s Mac torch + `onecoarse/code/oc_tree_tests.py` 3 s numpy; `onecoarse/results/oc_tree.npz`, `oc_tree_tests.json`). Checks: A torch vs `w_A.npz` max diff 5e-7; changing stage-3 position (6, 8) moves only cell (3, 4): PASS.

| ID | Object (picked by) | Result |
|---|---|---|
| C1 | 11 R1 children vs 50 low-R1 null neurons | PASS 11 of 11. Share of top cells that are parent-on: .69-.97. Null 95th percentile .53 |
| C2 | 110 ordered R1 child pairs | FAIL 26 of 110 below the 25th percentile (bar 36; chance 27.5). Half the votes (55) are negative, as for random channels |
| C3 | 110 ordered R1 child pairs, 50 null neurons in place of j | FAIL 8 of 110 (bar 55). Where sibling j fires, child i is slightly HIGHER: median d +.055, d < 0 in 20% of pairs (null median .002) |
| C4 | 11 R1 children vs 50 null neurons | FAIL 0 of 11. Children's most negative cells are mostly where the parent is OFF (parent-on share .05-.48; null 95th percentile .82) |
| C5 | B's n101 vs the other 127 B neurons | PASS. Coverage of the children's union .634 (others: 95th percentile .46). s3:69 is n101's number-1 input channel. n101 votes down on 3 of 11 children's conditions |

Microscope checks after the result (post hoc, labelled; the registered verdicts stand):

| Check | Why | Result |
|---|---|---|
| C3 with the parent's value matched (10 bins inside parent-on) | where j fires the parent is strong, which pushes i up too | still FAIL: 9 of 110 pairs; median d +.039; d < 0 in 27% |
| C4 inside the parent's group: share of a child's 200 most negative parent-on cells that are a sibling's top cells, divided by the base rate | the parent is each child's strongest up-vote (R1), so its global minimum is "parent off" | children .89-1.47; null neurons median 1.40, 95th percentile 1.70. 0 of 11 children above the null |

**NULL MIS-SPECIFIED (found by review, same day).** "R1 score below the median" picks neurons that READ s3:69 NEGATIVELY: the score is signed and its median is -.06. So the null neurons vote against the parent, and their negative poles sit where the parent is on (82%). The null for C1, C3 and C4 (and the post-hoc C4) is broken. Against plain base rates the children lean TOWARD turning away: negative cells parent-on above the base rate (.20) in 7 of 11, lift inside the parent's group above 1.0 in 9 of 11. The registered verdicts stay on record as run, but they are not evidence about turning away.

Reading (the actual tree, one parent), until the checks below are done:
1. Nesting holds (C1: .69-.97 against a base rate near .20; this holds with any null).
2. C5 passes, but how n101 was picked is not yet checked: if it was picked by reading s3:69, the pass is partly built in.
3. Turning away between siblings: OPEN. Tool gaps: (a) the broken null; (b) no planted test showing C2-C4 can flag a child built to turn a sibling away; (c) C2's "condition" = top 3 summed positive read channels, not yet checked against the logged conditions; (d) the plan's blind readings of each child's negative pole inside the parent's group are not done.

**3.7 repair after review** (`onecoarse/code/oc_tree_check.py`, 10 s Mac torch + numpy; `onecoarse/results/oc_tree_check.json`). Post hoc, labelled. Neutral null = the 50 non-child A neurons with the smallest |R1 score| (all below .039). Planted test = 3 synthetic children: a real child's read with a strong down vote (minus its own largest channel read) on one sibling's 3 condition channels: (n305 vs n97), (n108 vs n163), (n275 vs n297).

| Test | Object (picked by) | Can it see turning away? (planted) | Real children |
|---|---|---|---|
| C1 nesting | 11 R1 children vs neutral null | | 11 of 11 above the null's 95th percentile (.68) |
| C2 weights, mean vote | 110 R1 child pairs | yes, 3 of 3 | 26 of 110 (chance 27.5) |
| C2 weights, min vote (one down vote counts) | same | yes, 3 of 3 | 27 of 110 (chance 27.5). Both cited pairs flagged (n23 vs n297 .009, n385 vs n163 .000) |
| C3 firing | 110 R1 child pairs vs neutral null | yes, 2 of 3 (n108 vs n163 d -.19, n275 vs n297 d -.10; n305 vs n97 missed) | 7 of 110. Median d +.055: siblings rise together |
| C4 global negative pole | children vs neutral null | NO: planted shares .31 / .39 / .66 against real .29 / .32 / .48 | the registered C4 test cannot see turning away. Its fail means nothing |
| C4 inside the parent's group | children vs neutral null | partly: planted lift 4.15 / 2.80 / 2.25 on the target sibling; the single-sibling null's 95th percentile is 3.87 | 0 of 11 above the null (children .89-1.47, null median 1.35) |
| C5 coverage | n101 vs B's next 10 readers of s3:69 (n101 was picked as B's top reader of s3:69, so "s3:69 first" is built in) | | n101 .63; the next 10: .29-.54 |

Reading after repair:
1. Nesting holds against a neutral null.
2. n101 covers the branch better than every other B neuron that reads the parent strongly.
3. Down votes on a sibling's conditions exist (n23 vs n297, n385 vs n163), but they are as common as down votes on random channels. In firing, siblings mostly rise together, by a test that catches 2 of 3 planted cases. So at this parent, turning away between R1 siblings is not targeted, by the tools that can see it.
4. Still open: the blind readings of each child's negative pole inside the parent's group (running); other parents; one layer later.

**3.7 blind readings** (`onecoarse/code/oc_tree_sheets.py`; sheets and cards in `onecoarse/results/tree_cards/`; 4 opus readers, anonymous codes, positive and negative poles read by different readers). Negative pole = the child's most negative cells INSIDE the parent's group (parent-on cells only).

<!-- verb-lint from here -->

| Unit (picked by) | Verb, positive pole | Negative pole inside the parent's group | Reads like the positive pole of |
|---|---|---|---|
| parent s3:69 | round form with a small centre and fine strands or rays running out from it | (after a ReLU; not read) | |
| coarse n101 (B's top reader of s3:69) | round form with fine grey-silver streaks or rays | broad smooth surface with slow shading, little fine detail | no child: it does not turn a child away |
| child n305 (R1) | compact upright figure, pointed tips over a thicker mass, smaller in the crop | front-on layout: two dark spots above one larger blob, symmetric | n97, n237 |
| child n385 (R1) | tapered wedge outline in profile, a small dark dot behind the tip | front-on mirror-symmetric two dots above a central blob, fibrous frame | n237, n97 |
| child n163 (R1) | small dark spot at the centre of a pale surround, lines meeting on it | large dark front-on mass broken by a pale stripe | n237 |
| child n97 (R1) | three dark spots in a triangle inside a fluffy round field, front-on | curved border or filled ellipse in saturated colour | n297 |
| child n275 (R1) | starburst of thin straight rays from one point | ring with a dark centre and a bright border, glossy dome | n510, n297 |
| child n237 (R1) | close front-on mirror-symmetric two dots above one, ring of fluffy texture | one long narrow curved band crossing the crop | n464 |
| child n297 (R1) | medium-large round hard border with radial or nested content inside | light rounded blob above a wide dark block | n23 (partly) |
| child n108 (R1) | dark round mass inside an arch-shaped opening in a lighter flat panel, side view | small round accent on a smooth panel; front-on fluffy layouts | n97, n237 (partly) |
| child n23 (R1) | pale low-texture patch with two or three small dark dots | dense small high-contrast repeated marks | none |
| child n464 (R1) | one long thick bending band, S-curve or coil | straight edges and right-angle corners on flat panels | none |
| child n510 (R1) | small glossy ring at the front of a tapered side-profile outline | bold flat graphic contrast, block letters, stripes | none |

The last column is MY matching of blind words, made from the words before the number check below. It is not blind.

Number check (post hoc): for each child, which sibling's top cells are most over-represented among its 200 most negative parent-on cells. The word-matched sibling is first for n305, n108, n275; second for n385; third for n163, n297; fourth for n237; sixth for n97.

Reading:
1. Nesting, read in words: the parent's "round form with a centre and strands running out" splits into hard borders and hubs (n297, n108, n275), front-on fluffy layouts (n97, n237, n163, n23), profile outlines with one dot (n385, n510), a whole small figure (n305), and a bending band (n464). n101 reads like the parent.
2. Turning away, read in words: 8 of 11 children's negative poles inside the parent's group read like a sibling's positive pole. The distinctions are clean: profile vs front-on, whole figure vs close layout, rays vs a ring, a fluffy layout vs a hard border. The numbers agree for 4 of 8 (first or second of 10).
3. Why C3 missed it: C3 compared MEDIANS. On typical cells siblings rise together (they share the parent). Turning away shows at a child's most negative cells, not on average.
4. Open, and the next test: a neutral null neuron's most negative parent-on cells also fall on some child's top cells about as strongly (null 95th percentile 3.87 vs children up to 3.7). The children tile the parent's group, so any neuron's negative cells there may resemble some child. Whether siblings turn each other away MORE than other neurons of the layer do needs the same blind reading on null neurons.

