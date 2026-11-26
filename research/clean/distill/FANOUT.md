# FANOUT: an agent team that calculates the small network from the big one

Opened 2026-11-26. Plan: `~/.claude/plans/pasted-content-id-c513-okay-i-synchronous-backus-of-0b446158-33d3-4082-a57e-d87104d7f73a.md`.
Board (agents read and write it): `fanout/BOARD.md`. Scores: `fanout/board.json` (print with `fanout/table.py`).

## 1. Why (Shonu, 2026-11-26)

Work went one session at a time. Shonu wants an agent fan-out that keeps the core concept (`THEORY.md` section 1, the tree of distinctions) and tests small changes to it. Task: from the big network, CALCULATE the network at 95%, 50% and 5%, from network information only. The end goal is a language model (already chosen), so every rule is generic across networks. Later the calculation gets skewed toward chosen capabilities.

## 2. Rules

| Allowed in a calculation | Not allowed |
|---|---|
| the network's weights (including the BN-folded inference map of ResNet-18) | any photo or text from a data set |
| a generic input that comes from no data set (Shonu confirmed 2026-11-26) | per-photo fits, any training step |
| | stored statistics used AS DATA (ResNet BatchNorm mean and variance). They are the known answer of step 0b only |

Photos are for scoring only. Main bar = CLIP ViT-B/16 to the ViT-T shape (closest to a language model). Second network = ResNet-18 (targets at 95/50/5).

## 3. Step 0 (2026-11-26, me): the scoreboard

Code: `fanout/board.py` (build a whole network with a rule, score KL to the big network, add a row), `fanout/vitgen.py` (CLIP vision tower as a dict with any width per block, own forward), `fanout/generic.py` (pink noise), `fanout/rules/` (one module per rule). Runs: Mac MPS, one torch job at a time. Speed: full-width CLIP on 1,000 photos 14 s, ResNet-18 3 s.

Checks (all PASS):

| Check | Result |
|---|---|
| identity rule, CLIP, f = 1 | KL .000001, top-1 67.1% (teacher 67.1%) |
| identity rule, ResNet-18, f = 1 | KL 0, top-1 69.7% |
| cut that keeps every index (CLIP) | KL .000 |
| logged plain cut (photo Minitron, held 2,000) | 1.113 / 8.408 / 7.541 vs log 1.113 / 8.413 / 7.543 |
| trained targets through the board (5,000 test) | .0977 / .855 / 3.527 vs log .0978 / .855 / 3.521 |
| CLIP-KD student through the board | 1.801 (1,000), 1.719 (5,000) = CALC 2.1 exactly |

Board, KL to the big network, 1,000 test photos (one per class). Photo use: scoring only.

| Rung | CLIP 95% | CLIP 50% | CLIP ViT-T shape (25%) | CLIP 5% | ResNet 95% | ResNet 50% | ResNet 5% |
|---|---|---|---|---|---|---|---|
| magcut: weight-magnitude cut (read size x write size) | 11.33 | 12.67 | 13.16 | 11.53 | 3.53 | 21.92 | 7.49 |
| noisecut: Minitron activation ranking on pink noise | 1.139 | 9.031 | 10.404 | 11.231 | 1.760 | 9.254 | 15.245 |
| (check) Minitron on photos, held 2,000 | | | | | 1.113 | 8.408 | 7.541 |
| our calculation | | | | | | | |
| trained target | | | 1.801 (KD student) | | .097 | .856 | 3.564 |

Uniform cut: every set (stream, heads, MLP neurons; ResNet stream and inner channels) keeps fraction f. Head dim stays 64.

### Findings of step 0

F1. The loudest CLIP stream dims are the weakest by weight. On noise input, dims 163, 419, 171, 573, 97 carry the largest stream values from block 3 on (dim 163: RMS 8-10 against a median .1-.3). By weight score (read size x write size) they rank 0, 1, 35, 13 and 19 from the BOTTOM of 768. A weight-only cut removes them first, and the network falls to chance even at 95% (stream-only 95% cut: KL 10.85). Keeping the stream dims with the largest RMS on noise gives KL .263, top-1 63.4% at the same 95%. Cutting only MLP neurons (95%) costs .205, only one head per block .758. Part changed: P6 stream. Photo use: none in the ranking (noise), scoring only.

F2. Plain pink noise FAILS the known-answer check (step 0b, `fanout/check_bn.py`, `fanout/check_bn.json`). Over ResNet-18's 20 BatchNorm layers, the rank correlation of per-channel input variance with the stored running_var has median .38 (min .14) on pink noise, .37 on white noise, and .99 (min .73) on 1,000 photos (reading only, the ceiling of the check). Means: .78 (pink), .99 (photos). Noise variance is too small: median ratio .51, only .12 at layer4.0.bn2. So no rule may treat pink-noise statistics as the network's real operating point yet. Named gap and target: a better generic input, ideally one the network makes itself (part P0).

F3. Every floor rung is at chance at 50% on both networks (KL 9.0-9.3). The trained 50% ResNet is at .856. The whole gap from 9 to .86 is what the calculation must close.

## 4. Rounds

### Round 1 (2026-11-26)

Propose workflow (5 theorists + 1 critic, 806k agent tokens, 18 min): summary table in `fanout/BOARD.md` round 1, full text `fanout/round1/propose.json`.

Step 0 of round 1 (critic's request, rules `fanout/rules/split_*.py`): one set cut at a time. At 50% any one set alone already reaches or passes the flat-output level (CLIP 6.281, ResNet 5.737): CLIP heads 5.807, MLP 9.040, stream 11.104; ResNet inner 6.761, stream 18.051. At 95%: CLIP stream .280, heads .175, MLP .473; ResNet stream 1.076, inner .445. The noise ranking of CLIP MLP neurons is worse than the weight ranking at 95% (.473 vs .205).

Shonu, mid-round: "they should communicate with each other too". Added a talk step: two turns of addressed messages between the theorists before the build.

Talk step (10 agents, 913k agent tokens, 9 min; `fanout/round1/talk.json`): 20 messages in turn 1, 20 replies in turn 2, then 5 revised proposals with interfaces. What the theorists agreed:

| Agreement | Detail |
|---|---|
| One input object | 256 images, seed 1, a cached tensor path every part takes as a parameter. Every row runs on pink first, then on P0's self-made input with no code change |
| One error | second-order KL to the teacher; one shared teacher statistics pass (firings, moments per site, one backward with an output probe) |
| Write ownership | P4 owns every write. P1-P3 dropped its own write and uses P4's fold on its merged firings |
| Stream fold | P4's stream fold A is OFF by default. P6 owns the stream. One head-to-head row later |
| Closed loop | statistics from the teacher once; reads, quantiles and writes on the student's own stream, front to back, each size from the teacher weights |
| Build order | harness; P6 stream diagnostics; P0 input; P5 rank; P4 write (A = 0); P6 full; P1-P3; joint front-to-back build per size; leave-one-out rows decide each part's share |

Harness change (me): `board.py --input PATH` passes a cached input to the rule as X; rows record the input. Shared input `fanout/cache/input_pink_n256_s1.pt` (256 pink images, batched seeds). Check: CLIP floor at 95% on it = 1.139 (unchanged). ResNet floor on it (new reference, the old floor used a different draw): 95/50/5% = 1.665 / 9.386 / 14.949.

Build step started: builders for P6 diagnostics, P0, P5, P4; one runner at a time; one reviewer.

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

Mid-round change (15:07, Shonu: "3 slots, swap guard", and "any fixes you make i want to apply to the current round"). Round 2's first 17 minutes, 1 slot: 15 torch jobs, mean wait 245 s, mean run 66 s, so one test per agent every 5-6 min. `torchq.py` now runs up to 3 short jobs at once plus one `--long` lane (generators, big sweeps). A job starts only if free swap >= 1 GB and memory free >= 30%. Tested with sleep jobs before the switch. Old version kept as `torchq_v1_backup.py`. A monitor watches free swap and failed jobs. Agents told in NOTES.md.

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
