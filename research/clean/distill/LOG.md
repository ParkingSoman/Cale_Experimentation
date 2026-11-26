# DISTILL: distill the whole of A into 95%, 50% and 5% of its neurons, then calculate them

Opened 2026-11-25. Plan: `~/.claude/plans/here-are-some-thoughts-crispy-crescent-of-83ec389c-f6e5-4bca-ab75-04cc88705d88.md`.

## 1. Why (Shonu, 2026-11-25)

| Old guiding light | New guiding light |
|---|---|
| Remove one capability (the readers of s3:69) and repair the rest | Distill A into a network at 95%, 50% and 5% of its neurons |
| Coarse neurons read from repairs of A | The best distilled network at each size is the target |
| Capability dial = the goal | Capability dial = a future feature |

Shonu: "no bullshit. we do distillation. we do the whole model. 95, 50, 5. we compare to the whole network through KL." And: "whatever the best is at those sizes, that's what i want. you can use the 35k photos."

The end goal is calculation: compute these networks from A with no training. A second session plans the calculation. This log holds the training targets.

First version of this plan (one block of stage 4 only) was REPLACED the same evening. Its queued job 6997473 was cancelled before it started. Its one product: T0 (block 0 at 128 from random, N11 job 5) was found on the cluster and pulled to `neuron/results/coarse/r18/distill/` (md5 `2b0df8aea24f9d2e4f8abdb1b9f78d92`).

## 2. Facts said once

1. Neurons, not weights: 95% of neurons = about 90% of weights, 50% = about 25%, 5% = about 0.25%.
2. The known best recipe at a fixed size: cut the big network down, then distill (Minitron, NVIDIA 2024). So every size gets two starts: R (from random) and P (cut from A). The lower held KL wins.
3. Distillation on a small photo set needs the same crop for teacher and student, mixup and a long schedule (Beyer et al. 2022). The trainer does all three.
4. 35,000 photos is a small set. The 100% R run shows how far a fresh network gets on these photos alone.

## 3. Design

| Size | Stem + stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| A (100%) | 64 | 128 | 256 | 512 |
| 95% | 61 | 122 | 243 | 486 |
| 50% | 32 | 64 | 128 | 256 |
| 5% | 3 | 6 | 13 | 26 |

| Start | How | LR grid |
|---|---|---|
| R | random (torchvision init) | 1e-3, 3e-3, 1e-2 |
| P | keep the top channels by Minitron activation score (inner: mean ReLU output per block; stream: mean abs stream at the stage's block outputs, one set per stage), copy A's weights | 1e-4, 3e-4, 1e-3 |

Photo split (reading photos only, 35 per class): train 28, held 2 (as before), test 5 (first 5 of the old test photos).

Recipe: KL(A ‖ student), temperature 1, same augmented photo for both, random-resized crop + flip + mixup on half the batches, AdamW wd .05, 5% warm-up then cosine, batch 256, 300 epochs (cut if a full-width run would take over 1.5 h), 300-step lr sweep, best held-KL step kept. Code: `distill/wide18.py`, `distill/train_whole.py`.

## 4. Runs

| Wave | Runs | Job | Status |
|---|---|---|---|
| smoke | checks + 5% P, 95% R, 60 steps | | |
| 1 | 95% R/P, 50% R/P, 5% R/P, 100% R, seed 0 | | |
| 2 | winning start per size seed 1; 5% and 50% winners at 2x length | | |

Outputs: `neuron/results/coarse/r18/distill_whole/` (cluster and Mac).

Changes made during the run (2026-11-25, all recorded in each run's `.json`):

| Change | Why |
|---|---|
| Train photos missing from the JPEG bundle (the 12 per class moved from the old test set) come from their 224 centre crop scaled to 256. `train_from_jpeg` in each json | the bundle holds only the old train photos (smoke: 518 of 904 found) |
| Main run capped at 90 min from the speed seen in the sweep (`--max_minutes`). `steps_wanted` and `steps` in each json | the plan's "cut to fit" rule, made automatic so smoke and wave 1 run in one job |
| LR grids extended one step down after the first 3 runs had started: R 3e-4..1e-2, P 3e-5..1e-3. Runs 100R, 95R, 95P used the old grids (R 1e-3..1e-2, P 1e-4..1e-3) | all three picked the LOWEST point of the old grid (an edge) |

Wave 1 job 6998166 (H100, serval07), started 21:33. Smoke passed: split 28 / 2 / 5 per class, A top-1 on the new test set 68.7%, full width alone 0.057 s/step.

## 5. Photo use

| Item | Photo use |
|---|---|
| Training | per-photo fit (28,000 train photos, augmented) |
| P keep scores | per-photo calibration (2,000 train photos) |
| Held KL (lr pick, best step) | held photos |
| Score table | test photos, scoring only |
| Wave 3 `_ph` runs | + 142,907 ImageNet-TRAIN photos (A saw them in its training), per-photo fit, labels unused (section 10) |
| Wave 3 C starts | per-photo calibration of the source student (2,000 train photos) |

## Calculation log

The calculation work (no training) is in `CALC.md`. Known answer: the CLIP-KD pair in `clipkd/`.

CLIP-KD files, scores and limits: `clipkd/README.md` (2026-11-25). Target for a calculated ViT-T, scored with the teacher's classifier on 5,000 test photos: kd student KL 1.719, top-1 40.9% (teacher 66.2%).

Tree of distinctions (2026-11-25): concept in `THEORY.md`, first check in `CALC.md` 3.1-3.2. C1-C3 FAILED: the first root rule (weight-children of one parent, k = 8) lost to single teacher neurons by .00-.03. The registered rival, co-firing groups, won by .027-.064 (KD) and .019-.060 (baseline). Open lead only. First next step: a high-variance-direction control.

Vote families (CALC 3.3): same map from a second photo sample (A = B 1.00); weights alone match in blocks 2-6 only. M2-M4 FAILED: vote roots .01-.03 below single neurons in blocks 4-10. Control (CALC 3.4): the co-firing win was a lean toward high-variance directions, withdrawn. Next: teacher reads seen through the student's narrow stream.

Reconcile (CALC 3.5): strongest-crop unions show no gain by numbers (U1 FAIL, k3-k1 = .000), but by eye (not blind) 10 of 10 student neurons share their best teacher neuron's job and 6 of 10 cover 2-3 teacher neurons' jobs. Microscope errors named: loudest member owns the union's top; exact-photo top-1% overlap too strict. Next: equal-share unions, job-level score, blind reading.

Checklist reading (CALC 3.6, 2026-11-25): `../NEURON-CHECKLIST.md` written. 10 student neurons + 30 teacher members read blind on both poles. Noun lists dissolve into one pattern each ("camera + gun" = a small crisp round hole in a hard plate). J1 PASS (student words cover the members, 9 of 10), J2 FAIL (the best single teacher neuron's words cover them as well). T2-T4 FAIL: members do not gate each other and rarely share feeders. Block 12: patch firings are dead (class-token pooling); on the class token the teacher carries the same jobs. Block 11: KD does 21% of jobs mainly on the class token (teacher .5%, baseline 5%). Tree update in `THEORY.md` section 7.

Correction (2026-11-25, Shonu): the CALC 3.6 "tree" rows (T2-T6) are WITHDRAWN as a tree test. Their units were picked by shared photos, not by the child rule R1. The real test, a parent and its R1 children, is CALC 3.7.

CALC 3.7 (2026-11-25), the actual tree on ResNet-18 parent s3:69, its 11 R1 children and B's n101: nesting PASS (C1 11/11), coarse neuron PASS (C5: n101 reads s3:69 first, covers .63 of the branch). Turning away: the first null was broken (fixed). Blind readings: inside the parent's group, 8 of 11 children's negative poles read like a sibling's positive pole (numbers agree 4 of 8); on typical cells siblings rise together. Open: is it special to siblings (null neurons need the same reading). THEORY section 8.

## 6. Wave 1 result (job 6998166, finished 2026-11-26 00:00; all 7 networks on the Mac, md5 checked)

| Neurons kept | Start | Weights | lr | Stop / steps | Held KL | Test KL | Test top-1 |
|---|---|---|---|---|---|---|---|
| 100% | R | 11,689,512 | 1e-3 (grid bottom, old grid) | 32790 / 32812 | 1.0717 | 1.0668 | 48.7% |
| 95% | P | 10,564,386 | 1e-4 (grid bottom, old grid) | 30604 / 32812 | 0.0998 | **0.0978** | **67.1%** |
| 95% | R | 10,564,386 | 1e-3 (grid bottom, old grid) | 32790 / 32812 | 1.0853 | 1.0775 | 47.8% |
| 50% | P | 3,055,880 | 1e-3 (grid TOP) | 32790 / 32812 | 0.9288 | **0.9320** | **50.8%** |
| 50% | R | 3,055,880 | 1e-3 (interior) | 31697 / 32812 | 1.6451 | 1.6068 | 38.0% |
| 5% | P | 56,372 | 1e-3 (grid TOP) | 32812 / 32812 | 3.7187 | 3.7386 | 10.1% |
| 5% | R | 56,372 | 1e-2 (grid TOP) | 32812 / 32812 | 3.6005 | **3.5744** | **12.1%** |

A on the same 5,000 test photos: top-1 68.7%.

Readings (numbers only):
1. 95%: cutting A and distilling (P) stays close to A (KL .098, top-1 -1.6 points). From random (R) is 11x worse in KL.
2. 50%: P wins (KL .93 vs 1.61).
3. 5%: R wins narrowly by held KL (3.60 vs 3.72). Both lrs at the TOP of their grids and both best at the last step, so the 5% winner is not settled.
4. Every R run, even at 100%, is far from A (KL about 1.07 at 100%): with 28,000 photos a fresh network does not reach A. This is the photo-budget ceiling (section 2, fact 4).
5. Every run except 95% P had its best step at or near the last step: under-trained by length.

Files: `neuron/results/coarse/r18/distill_whole/W<pct><start>_s0.{pt,json,log}`, `_testkl.npy` (per test photo), `_scores.pt` (P runs: keep scores and kept channel sets), `ckpt/W<pct><start>_s0_<step>.pt` (steps 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 32812). `W95R_s0` also packed its 5 smoke checkpoints (`ckpt/W95R_s0_smoke_*.pt`, ignore).

## 7. Wave 2 (job 7000578, A100 80 GB, started 2026-11-26 00:06)

Runs: 95P seed 1; 50P seed 0 at 2x (600 epochs); 5R seed 0 at 2x; 50P seed 1; 5R seed 1. LR grids one step past the wave-1 edges.

Sweeps settled the edge question: 50P picked 1e-3 with 3e-3 worse (2.44 vs 2.81 held). 5R picked 1e-2 with 3e-2 worse (5.43 vs 5.49). 95P picked 1e-4 with 3e-5 worse (.221 vs .287). So the wave-1 learning rates were the best of the extended grids.

Event 00:34: on the A100 (slower than wave 1's H100), the 100-min cap cut 95P seed 1 to 197 epochs (seed 0 had 300). That would mix seed with length. I stopped only that process (6 min into training) with `srun --overlap pkill`. The job went on and started 50P seed 1. 95P seed 1 reruns at the full 300 epochs as its own job after wave 2.

Wave 2 result (job 7000578 finished 02:37; all 4 finished networks on the Mac, md5 checked):

| Neurons kept | Start | Seed | Length | Held KL | Test KL | Test top-1 |
|---|---|---|---|---|---|---|
| 50% | P | 0 | 2x (600 ep) | 0.8622 | **0.8550** | **52.4%** |
| 50% | P | 1 | 1x | 0.9435 | 0.9386 | 50.8% |
| 5% | R | 0 | 2x (600 ep) | 3.5202 | **3.5212** | **12.5%** |
| 5% | R | 1 | 1x | 3.5719 | 3.5529 | 12.3% |

Readings:
1. Length: 2x beats 1x by more than the 1% tie rule at both sizes (50% P: held -7.2%; 5% R: held -2.2%). So the 2x runs are the best networks at 50% and 5%. Both still had their best step at or near the end: longer may help again.
2. Seeds (1x vs 1x): 50% P test KL gap .0066 (top-1 identical 50.8%). 5% R gap .0215.
3. 95% P seed 1, rerun at the full 300 epochs (job 7003485, A100, finished 03:36): held .0996, test KL .0982, top-1 67.3%. Seed gap at 95% P: .0004 in test KL (seed 0: .0978, 67.1%). Same lr picked (1e-4).

## 8. Best network made at each size (not converged at 50% and 5%)

| Neurons kept | Weights | Best run made | Held KL of the plain cut (no training) | Test KL to A | Test top-1 | File |
|---|---|---|---|---|---|---|
| 100% | 11.7 M | (A itself) | 0 | 0 | 68.7% | torchvision IMAGENET1K_V1 |
| 95% | 10.6 M | P, seed 0, 1x | 1.113 | 0.0978 | 67.1% | `distill_whole/W95P_s0.pt` |
| 50% | 3.1 M | P, seed 0, 2x | 8.413 | 0.8550 | 52.4% | `distill_whole/W50P_s0_x2.pt` |
| 5% | 56 K | R, seed 0, 2x | 7.543 | 3.5212 | 12.5% | `distill_whole/W5R_s0_x2.pt` |

100% from random (W100R_s0, KL 1.067) is the photo-budget reference, not a target.

Seed gaps of the winning start, 1x vs 1x, test KL: 95% .0004, 50% .0066, 5% .0215.

Open: at 5% the winning start (R, held 3.60 at 1x) beat P (3.72) by a small margin, and P's lr sat at the TOP of its grid (1e-3) with its best step at the end. P at 5% was not rerun (the plan reruns winners only). Asked Shonu 2026-11-26.

All 12 networks on the Mac: `neuron/results/coarse/r18/distill_whole/` (2.5 GB), each with 10 checkpoints. Cluster copies: `/bigtemp/fkm5yp/distill/*.tgz` (no backup there).

## 9. Notes for anyone using these networks (added 2026-11-26)

1. **Seeds of the cut-from-A runs share their start.** A P run copies A's weights for the kept channels. The seed changes only the photo order. So the 95% and 50% "seed gaps" (.0004, .0066) are photo-order gaps from one start. Only the 5% R pair (.0215) is a true two-start pair.
2. **Use only this log's test set.** The train set holds 12 of the 17 old test photos per class. Old tools (`r18.photo_sets`, `r18.Cache("test")`) would score these students on photos they trained on. The valid test set is `split()` in `distill/train_whole.py` (5 per class).
3. **No-training line (held KL of the plain cut, before any training = `curve[0].held_kl`):** 95% P 1.113, 50% P 8.413, 5% P 7.543. For scale, a random network starts at 6.1-6.8 (23.6 for one 5% init). At 50% and 5% the plain cut is worse than random. At 95% it is already close.
4. **"Best" means best of the runs made, not converged.** At 50% and 5% the best step is at or near the last step even at 2x length.
5. **Code edit outside `distill/`:** `neuron/coarse/train18c.py` got a `--suffix` flag (file names only) on 2026-11-25, for the replaced one-block plan.

## 10. Wave 3: improve the 50% and 5% networks (opened 2026-11-26)

Shonu, 2026-11-26: "keep improving the 50% and 5% networks so we can study it more. i think the 95% has already got there-ish." Plan: `~/.claude/plans/it-would-be-great-luminous-thimble-of-94441466-56c7-4857-97af-c515b5e8c9cc.md`.

### Step 0: what limits each size (saved networks, Mac, no training)

Clean centre crops of 2,000 train photos against the held photos. The held KL agrees with the logs (.9281 vs .9288, 3.6033 vs 3.6005).

| Network | Clean train KL | Held KL | Reading |
|---|---|---|---|
| W95P_s0 | 0.0437 | 0.1002 | close on both |
| W50P_s0 | 0.2678 | 0.9281 | memorizes its photos |
| W50P_s0_x2 | 0.2306 | 0.8625 | memorizes its photos. Photos limit it |
| W5R_s0 | 3.2169 | 3.6033 | cannot fit its train photos |
| W5R_s0_x2 | 3.1295 | 3.5248 | cannot fit its train photos. Its size limits it |
| W100R_s0 | 0.2197 | 1.0715 | same photo limit as 50% |

### Extra photos (Shonu approved 2026-11-26)

| Item | Value |
|---|---|
| Source | Hugging Face `evanarlian/imagenet_1k_resized_256`, revision 8de107e (same as our val photos), train split |
| Selection | every 9th photo per class in each shard (`distill/extra_photos.py`). The shards are sorted by class, so all 52 were read |
| Count | 142,907 photos, 82-146 per class (median 145) |
| Use | train set of the 50% `_ph` runs only. Labels unused (only the loader check reads them) |
| Where | cluster `/bigtemp/fkm5yp/distill/extra/extra_00..51.npz`. Mac md5 list `~/uva-gpu/outbox/extra_md5.txt` |

A (torchvision ResNet-18) trained on ImageNet-train. These photos are photos A saw. Held and test stay val photos that A did not see.

### Code (wave 3)

`train_whole.py` gained: `--extra`, `--steps`, EMA copy (scored only, raw weights decide wins), `--clip`, starts C / W / G, `--multi`, `--need_full`, and a loader check (A's top-1 on 2,000 extra photos). The wave 1-2 trainer is kept as `old/train_whole_wave12.py`. `--check` passes on the Mac (CHECK 4: a saved 50% student cut to its own width gives the same logits, max diff 0).

### Screen design (1x = 32,812 steps, seed 0)

| Tag | Change | Baseline |
|---|---|---|
| W50P_s0_ph | P start + 142,907 extra photos | W50P_s0 (.9288) |
| W50W_s0_ph | keep training W50P_s0_x2 on the extra photos, at the lr W50P_s0_ph picked | W50P_s0 and W50P_s0_ph |
| W50P_s0_mix | mixup on every batch | W50P_s0 |
| W50G_s0 | two-step cut: 75% for 12,000 steps, then 50% for 20,812 | W50P_s0 |
| W5{R,Rwd0,Rclip,C50,P}_s0_m | one process, same batches, mix .5. C50 = W50P_s0_x2 cut to 5% | W5R_s0_m |
| W5{R,Rwd0}_s0_m0 | one process, mix 0 | W5R_s0_m0 |

Win rule: best RAW held KL lower than its baseline by more than 2%. Control before the screen: student R's held-KL path in the old trainer (twice), the new single path and the new `--multi` path must agree within max(2 x the old run-to-run gap, 0.5%).

### Screen result (job 7006206, A100-SXM4 80 GB, 14:22-17:14; all 11 tgz on the Mac, md5 checked)

Control: student R's held-KL path agreed to 4 decimals in the old trainer (twice), the new single path and `--multi` (gap 0.0000%, step 60 held KL 5.7806 in all). Loader check: A's top-1 on 2,860 extra photos 76.3% (A on val test 68.6%; A trained on these photos).

| Tag | Change | Best raw held KL | vs baseline | Best EMA held KL | Test KL | Test top-1 |
|---|---|---|---|---|---|---|
| W50P_s0_ph | P + 142,907 extra photos | 0.7204 | **-22.4% WIN** (W50P_s0) | 0.7189 | 0.7008 | 56.0% |
| W50W_s0_ph | W50P_s0_x2 trained on, extra photos, lr 1e-3 | 0.6774 | **-27.1% WIN** (W50P_s0) | 0.6750 | 0.6431 | 56.7% |
| W50P_s0_mix | mixup on every batch | 0.9098 | -2.05% WIN, weak (bar -2%, photo-order gap 1.6%) | 0.9108 | 0.8997 | 51.7% |
| W50G_s0 | two-step cut 75% -> 50% | 0.9413 | +1.4% | 0.9415 | 0.9161 | 50.8% |
| W5R_s0_m | baseline, same batches | 3.5804 | | 3.5932 | 3.5984 | 11.7% |
| W5Rclip_s0_m | grad clip 1.0 | 3.5811 | +0.0% | 3.5895 | 3.5652 | 11.8% |
| W5C50_s0_m | W50P_s0_x2 cut to 5% | 3.5909 | +0.3% | 3.6079 | 3.5781 | 12.2% |
| W5Rwd0_s0_m | no weight decay | 3.6241 | +1.2% | 3.6210 | 3.6334 | 11.4% |
| W5P_s0_m | P from A, lr up to 1e-2 | 3.6753 | +2.7% | 3.6680 | 3.6572 | 11.4% |
| W5R_s0_m0 | no mixup | 3.6108 | +0.9% vs W5R_s0_m | 3.7960 | 3.5684 | 12.0% |
| W5Rwd0_s0_m0 | no mixup, no weight decay | 3.6267 | +0.4% vs W5R_s0_m0 | 3.6246 | 3.6007 | 12.1% |

Readings:
1. 50%: the extra photos are the big lever (-22% at 1x; test top-1 56.0% vs 52.4% for the old 2x best). Memorization was the limit, as Step 0 said.
2. Start again or keep going (Shonu's question): keeping the memorized network and adding photos did better (0.677 vs 0.720 raw). It is NOT a fair test: the warm network had 98,437 steps in total, the fresh one 32,812. The 4x fresh run (131,248 steps) is the fair comparison. Ash & Adams' warm-start harm does not show at these lengths.
3. 5%: no change wins. Less regularization, clipping and both cut starts do not help. Only length helped so far (wave 2: 2x = -2.2%).
4. 5% lrs: W5R_s0_m0 and W5Rwd0_s0_m picked 3e-2, the top of their grid. Neither won, so not rerun.
5. EMA: better than raw by at most .0073 (W5P_s0_m), and much worse once (W5R_s0_m0: 3.796 vs 3.611, lr 3e-2). It does not change any reading.

### Long runs (job 7006930, submitted 17:20)

| Tag | Recipe | Steps |
|---|---|---|
| W50P_s0_ph4 | P + extra photos + mixup on every batch (both 50% wins combined, by the rule) | 131,248 |
| W5R_s0_x4 | old recipe (R, mix .5, wd .05) | 131,248 |
