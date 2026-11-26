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
