# FANOUT BOARD: read this first, every round

All paths are relative to `research/clean/`. Log with full results: `distill/FANOUT.md`. Scores: `distill/fanout/board.json`, printed by `curves/.venv/bin/python -I distill/fanout/table.py`.

## The task

From a big network, CALCULATE a smaller network (95%, 50%, 5% of each width; for CLIP also the ViT-T shape = 25%) with no training and no photos. Score = KL to the big network on test photos. Training reaches one point (the trained targets below). The calculation aims at the point training aims at. The end goal is a language model, so a rule must be GENERIC: written in parts (read, write, gate, stream), with a thin adapter per architecture.

## The core concept (`distill/THEORY.md` section 1). A tweak changes ONE part and names it.

"A smaller network is the parent's tree of distinctions, cut higher, in every layer at once."

| Part | The concept says | Open question |
|---|---|---|
| P0 Input | (added 2026-11-26) the calculation may run the network on a generic input from no data set | which generic input puts the network at its real operating point (F2: pink noise fails) |
| P1 Tree | children = readers of a parent's write, in weights (rule R1) | how to find parents for the whole network at once |
| P2 Read | a coarse neuron reads the root of its branch | which root rule |
| P3 Gate | its threshold opens on the UNION of the branch (an OR, it fires more broadly) | where to set the threshold |
| P4 Write | the branch's message, as the COARSE readers downstream read it | how to solve it with no photos |
| P5 Rank | merge the weakest distinction first: written strength x read strength; one ranking gives every size | which strengths |
| P6 Stream | the stream has no fixed basis, so the cut there is a projection | which projection |

## Rules (each one caused a withdrawal before)

| Rule |
|---|
| Network information only: weights, plus a generic input from no data set. No photo or text from any data set in a calculation. ResNet BatchNorm mean/var may NOT be used as data (language models have none). They are the known answer of the input check only |
| Photos are for scoring only. Every result states its photo use |
| Whole-network KL decides. A one-block check may help thinking, never a decision |
| One calculation for all sizes. One error: KL. No stacked methods. A fit needs "Derived from:" |
| Every test names "Object (picked by)". Children come from R1 only. Co-firing units are never called siblings or children |
| A prediction is written BEFORE the run, with a planted test (shows the tool can see the effect) and a named null |
| A disagreement with the essay (`curves/map_agents/essay.md`) = fix the microscope, not the object |
| ONE torch job on the Mac at a time. Only the runner runs torch. No GPU jobs |
| A gap is a next target, said with confidence. Numbers stay exact |

## Current board (KL to the big network, 1,000 test photos, one per class)

| Rung | CLIP 95% | CLIP 50% | CLIP ViT-T shape (25%) | CLIP 5% | ResNet 95% | ResNet 50% | ResNet 5% |
|---|---|---|---|---|---|---|---|
| magcut: weight-magnitude cut | 11.33 | 12.67 | 13.16 | 11.53 | 3.53 | 21.92 | 7.49 |
| noisecut: Minitron ranking on pink noise (THE FLOOR) | 1.139 | 9.031 | 10.404 | 11.231 | 1.760 | 9.254 | 15.245 |
| (check) Minitron on photos | | | | | 1.113 | 8.408 | 7.541 |
| trained target | | | 1.801 (KD student) | | .097 | .856 | 3.564 |
| trained target, wave 3 (LOG.md, 5,000 test photos; added 2026-11-26 night) | | | 1.719 (KD student, 5,000) | | .0978 (W95P_s0) | .5814 (W50P_s0_ph4) | 3.5039 (W5R_s0_x4) |

## Findings so far

F1. The loudest CLIP stream dims (163, 419, 171, 573, 97; dim 163 RMS 8-10 vs median .1-.3 from block 3 on) are the WEAKEST by weight (rank 0, 1, 35, 13, 19 from the bottom of 768). A weight-only cut removes them and the network falls to chance even at 95%. Keeping the stream dims with the largest RMS on noise: KL .263 at 95% stream-only. MLP-only 95% cut costs .205. One head per block costs .758.

F2. Pink noise FAILS the known-answer check on ResNet-18: rank correlation of per-channel variance with stored BatchNorm running_var, median .38 (photos .99). Noise variance is too small (median ratio .51, .12 in stage 4). So noise statistics are not yet the network's real operating point.

F3. Every floor rung is at chance at 50% on both networks (KL 9.0-9.3). The trained 50% ResNet is at .856.

## Code map

| File | What |
|---|---|
| `distill/fanout/board.py` | `--net clip|r18 --rule NAME --f 0.95,0.5,0.25,0.05 [--n 1000] [--split held]`. Builds with `rules/NAME.py`, scores, appends to `board.json`, saves per-photo KL in `fanout/out/` |
| `distill/fanout/rules/` | one module per rule: `vit(T, f) -> vitgen dict`, `r18(A, f) -> ResNet-18 module`. `noisecut.py` = the floor (copy its pattern) |
| `distill/fanout/vitgen.py` | CLIP vision tower as a dict with ANY width per block and its own forward. `cut(T, sd, heads, mlp)`, `norm` modes "ln" and "rms" (centred stream, divisor D), `forward(W, x, upto=i)` |
| `distill/fanout/generic.py` | `pink(n, seed)`: the generic input |
| `distill/wide18.py` | ResNet-18 at uniform width f; `cut_from_a(A, f, scores)`; `channel_scores` |
| `distill/fanout/check_bn.py` | the P0 known-answer check (noise stats vs stored BatchNorm) |
| `distill/THEORY.md`, `distill/CALC.md` | concept, prior art (section 1 of CALC: SliceGPT, model folding, Minitron), the tree checks 3.2-3.7 |
| `onecoarse/INDEX.md` | all older results, 7 threads (read part 1) |
| `curves/map_agents/essay.md` | the essay: the law and the inspiration |

## Rounds

(each round appends: proposals, the critic's verdicts, what was built, results, expected / happened / why)

### Round 1 (2026-11-26)

Full proposals and critic verdicts: `distill/fanout/round1/propose.json`.

| Part | Proposal | Critic |
|---|---|---|
| P0 Input | self-made input: start at pink noise, a few gradient steps that lower the network's own loss on targets it draws itself (language model: its own samples) | build (CLIP n = 256 or compare with a K = 0 row; the CLIP loss is a self-InfoNCE proxy, name it so; photo-mix calibration mandatory) |
| P1-P3 Tree/read/gate | OR-merge: a deleted unit joins its R1 branch's kept unit; merged gate fires as often as ANY member | revise (changes the write too; needs one-part ablation rows at whole-network KL; R1 weighted by noise RMS is an extension of R1, name it) |
| P4 Write | reader-matched write: kept writes solved against the KEPT readers, one closed form (unit fold P + stream fold A) | build, with a switch A = 0 (A conflicts with P6's read maps) |
| P5 Rank | written value x read gradient of the final output, per input, weakest removed first, re-measured each step | build (fix: mask ln_pre too; null = same formula with permuted pairing) |
| P6 Stream | one stream subspace kept by what readers AND the norm divisor read; free per-site read maps | revise (planted test that can fail: an unread high-variance direction must be dropped; separate basis from read maps) |

Critic's cross-cutting rules: report agree and top-1 next to every KL; a gain that lands near the flat level with no gain in agree is a calibration change, not a recovered job; never tune on check_bn; planted tests that pass by algebra are implementation checks, the nulls carry the evidence.

Step 0 of round 1 (me, runner): one part cut at a time (floor ranking), KL on 1,000 test photos.

| Cut only this set | CLIP 95% | CLIP 50% | CLIP 25% | CLIP 5% | ResNet 95% | ResNet 50% | ResNet 5% |
|---|---|---|---|---|---|---|---|
| stream | .280 | 11.104 | 11.431 | 8.903 | 1.076 | 18.051 | 6.134 |
| heads | .175 | 5.807 | 9.704 | 9.573 | | | |
| MLP (CLIP) / inner (ResNet) | .473 | 9.040 | 10.499 | 12.000 | .445 | 6.761 | 8.617 |

Flat-output level (KL of a uniform output, 1,000 test photos): CLIP 6.281, ResNet 5.737.

Reading: at 50%, ANY one part cut alone already puts the network at or past the flat level (best: CLIP heads 5.81, ResNet inner 6.76). So a fix to one part cannot move the 50% row alone. Every part needs a repair. Note: the noise ranking of CLIP MLP neurons is WORSE than the weight ranking at 95% (.473 vs .205, F1).

Build step (9 agents, 1.16M agent tokens, 66 min; `fanout/round1/build.json`). Reviewer's verdict, KL / agree on 1,000 test photos:

| Result | Numbers |
|---|---|
| REAL: P4 write fold, CLIP MLP-only 50% | 9.040 / .2% -> 6.154 / 4.1%; null 8.812 / 1.0% |
| REAL: P5 rank, ResNet whole 95% | 1.665 / 42.7% -> 1.570 / 46.4%; null 1.844 / 41.2%; seed spread .002 |
| REAL (half): ResNet mean fold of cut stream channels, stream-only 95% | 1.032 / 56.6% -> .897 / 61.2%; permuted-mean null .967 / 59.1% (about half) |
| REFUTED: the 50% stream damage is a scale or mean error | every CLIP divisor variant (k, D, rho, global rho) and every ResNet fold stays at chance (agree <= .7%) |
| FAIL: P5 on CLIP MLP and heads | MLP-only 95% .323 vs weight ranking .205; heads-only 50% 6.840 vs floor 5.807 |
| FAIL: P0 self-made input on ResNet | variance rank .542 (bar .55), 95% KL 1.665 -> 1.711; firing rates unmoved |
| NOT RUN: P0 on CLIP | 17.4 s per step from the patch-conv backward on MPS |

Lesson: at 50% and 5% almost every row has agree <= 1%. There KL moves with calibration only (the input alone moved ResNet 50% from 9.386 to 14.128). Such rows are not ranked. `table.py` now marks them "at chance". 85% is added as a readable decision size.

Shonu, after round 1: "we need to be going faster rather than just completely staying in theory land". Changes (me): `fanout/torchq.py` (a lock so many agents queue torch jobs one at a time, swap check, log `torchq.log`); `board.py --quick` (200 photos, every 5th class, ~10 s, rows to `scratch.jsonl`); `fanout/NOTES.md` (one line per experiment, @-addressed messages); `vitgen.embed` patch conv replaced by reshape + matmul (batch-8 forward + backward 4.0 s -> .22 s; identity check KL 1e-6).

### Round 2 (2026-11-26): fast loops

Seven agents (P0, P1-P3, P4, P5, P6, JOINT, live CRITIC), each in its own loop: one change, one quick test through the lock, one note, about 25 tests or 45 minutes, then promote the best rows to the board.

Round 2 result (7 agents, 152 quick tests, 1.18M agent tokens, 100 min; `fanout/round2/fastloop.json`, notes `fanout/NOTES.md`). Full rows, 1,000 test photos, KL / agree; every claim below has a same-form null and an off row, and the starred ones were read paired on the 800 photos outside the quick set by the critic.

| Part | Best rule (file) | Row | Off row | Null |
|---|---|---|---|---|
| CLIP whole 95% | MLP by weight, heads + stream by floor activity (`r2_p5_w0`) | .820 / 68.2% (held .907 / 65.3%) | floor 1.139 / 60.4% | MLP activity permuted ties: activity carries no MLP information on pink |
| CLIP MLP-only 85% * | weight order + P4 write fold (`r2_p4_wfold_mlp`) | .776 / 66.8% | weight order 1.240 / 59.2% | mean null 1.532 / 54.1% |
| CLIP stream-only 85% / 70% | one reader-weighted subspace G^1/2 C G^1/2 (`r2_p6_rG_rho`) | .955 / 62.7%; 3.206 / 25.6% | coordinates 1.394 / 55.7%; 6.845 / 6.9% | random subspace 6.205 / 8.1%; 10.688 / .4% |
| CLIP stream 85%, input | per-site PCA basis from the network's SELF-MADE input (P0) | 2.412 / 38.7% | pink 5.033 / 14.0% | shuffled-weights input 3.758 / 23.7% |
| ResNet whole 95% / 85% * | floor order + closed-loop stream fold + inner fold (`r2_joint_floor_sfon_on`) | 1.193 / 53.4%; 3.995 / 14.5% | floor 1.665 / 42.7%; 2.6% | stream-fold null 1.285 / 52.7%; 4.560 / 11.5% |
| ResNet stream-only 85% | per-reader read maps, open loop (`r2_p6_r18_map`) | 2.380 / 32.3% | mean fold 4.334 / 14.3% | same-form permuted 3.739 / 17.0% |

What round 2 settled:
1. The write fold works on both networks (CLIP MLP, ResNet stream and inner).
2. Rank CLIP MLP neurons by weight size, heads by activity. Pink activity carries no MLP information.
3. The stream needs the right FORM (a subspace or read map; photo coordinates stay at chance at 50%, a photo per-site subspace reaches 2.634 / 35.5%) AND the right INPUT (the self-made input beats its null by +15 agree at 85%; it closes about 35% of the pink-to-photo gap).
4. The tree's summed-read merge is the wrong form even with photo statistics (85%: 47.5% vs 64.0% with no merge). R1 partners are real (+8.0 agree over a shape-kept null). The union fires barely above a unit's own rate at 95/85%.
5. Dead ends: per-site PCA on pink (worse than one basis), head folds and head merges (OV merge breaks the network), self-made inputs on ResNet, P5's gradient-probe order (ResNet sign reversed on unseen photos).
6. Open: the P4 fold HURTS the whole CLIP cut (weight order .820 / 68.2% -> 1.074 / 61.7%); 50% is still at chance on both networks.
7. Selection: a knob picked on the 200 quick photos lost 31% of its gain on unseen photos. Pick knobs on the held split.

Process: the 1-slot lock gave a mean wait of 245 s; 3 slots then overloaded swap (10.8 GB in use); settled at 2 short + 1 long with a memory guard (memory free >= 35%, swap in use <= 8 GB). The lead reviewed the notes every 10 min (7 reviews).

### Round 3 (2026-11-26): fast loops, 6 agents

Agents INPUT, STREAM, WRITE, TREE, JOINT, CRITIC (86 quick tests, 1.0M agent tokens, 73 min; `fanout/round3/fastloop.json`). New this round: one prediction line before every run; the critic read every full claim paired on the 800 test photos outside the quick set (`fanout/r3_critic_holdout.py`, uses the new per-photo agree files).

Survivors (full 1,000 test photos, KL / agree; each with an off row, a same-form null, and a paired read on the 800 unseen photos):

| Part | Rule | Row | Off | Null |
|---|---|---|---|---|
| ResNet whole 85%: fold statistics from the network's SELF-MADE input | `r3_input_r18sf` (round-2 joint J, both folds on `r18_self`) | 2.295 / 32.6% (800 unseen: 33.5%, no shrink) | pink 4.001 / 14.7% | shuffled-weights input 3.892 / 15.1%; z +11.2 |
| CLIP whole 85% / 95%: "kslice" fold (kept units keep their own write; only the cut units' message is folded) | `r3_write_w_kslice` | 4.058 / 21.0%; .628 / 71.4% | 7.737 / 5.1%; .820 / 68.2% | roll pairing 4.785 / 17.4%; z +10.4 vs off |
| CLIP stream + MLP cut, heads full, 85% | `r3_write_sm_kslice` | 2.624 / 36.5% | 5.218 / 18.5% | 3.106 / 32.2% |
| CLIP whole 85%: rG stream + heads activity + MLP fold (one calculation) | `r3_joint_c_on` | 4.577 / 17.8% (800 unseen 15.9%; quick 25.5%) | stream -> coords 7.776 / 2.5%; fold off 6.076 / 11.8% | fold mean null harms (lower tier) |
| CLIP MLP-only 70% | `r2_p4_wfold_mlp` | 2.513 / 37.6% | 6.694 / 9.9% | 5.096 / 14.1% |
| CLIP stream-only 85% | rG with C from the self input | .742 / 67.6% | coords 1.284 / 56.2% | null input .780 / 64.9% (tie) |

What round 3 settled:
1. Why the fold hurt the whole CLIP cut: the closed-loop fold also re-solved the KEPT units' writes, and they absorbed the pink stream-cut damage. kslice fixes it (stream + MLP 85%: 9.5% -> 40.0% quick).
2. The network's self-made input moves SECOND-MOMENT statistics (fold fits, stream bases), not rankings. It took ResNet whole 85% from 14.7% to 32.6%. For the one-basis rG stream any optimised input ties the self input.
3. Where the CLIP stream cut dies: blocks 0-2 cost only .14 KL; blocks 3-11 carry the loss. Block 3 is where the loud stream dims switch on (step-0 F1). The loud-dims test is the first round-4 stream row.
4. 8 photos beat 256 self-made images for the stream basis at 85% (labelled check): the gap is the KIND of image. At 70% the number of images matters too.
5. WITHDRAWN: "rank CLIP MLP neurons by weight" (a random order ties at 95%). What stands: pink activity misleads the MLP ranking. Heads must be ranked by activity (by weight: -20 agree in the joint).
6. Dead ends: every fixed read tilt (fold partner, R1 partner, calculated descent, random) ties or loses to plain deletion + fold on pink. Raw fold partners are hubs (one unit is top partner of 18-22% of kept units). Per-site stream bases, a two-group basis, colour knobs, the fc fold.
7. Still at chance: 50% whole on both networks, ResNet 70%.
8. Selection: quick-picked knobs shrink about 30-38% on unseen photos.

Round 4 candidates (agents' lists): stack kslice into the rG joint (CLIP); self-input fold statistics into the ResNet joint, split per fold; the loud-dims stream test; score the CLIP "act" self input (made, unscored); find what an 8-photo basis has that the self input lacks.
