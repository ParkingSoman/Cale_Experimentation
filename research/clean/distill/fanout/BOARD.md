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
