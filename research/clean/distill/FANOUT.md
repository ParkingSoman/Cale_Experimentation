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

(round summaries go here)
