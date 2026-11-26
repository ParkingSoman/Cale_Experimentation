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
