# LOOPS: the tree loop ledger

One entry per loop: essay passage, page diff, prediction (written before the run), look table, decision.
Page: `PAGE.md`. Code: `tree/build.py`, `tree/checks.py`, `tree/step0.py`, board rules `fanout/rules/tree_v1*.py`.
Plan: `~/.claude/plans/i-want-you-to-shimmying-horizon-of-b898640f-c4f4-4857-9b92-579315f83214.md`.

## Step 0: the tree's shape (weights + generic input, no photos)

Files: `tree/out/branches_{clip,r18}.json` (spread-based scores) and `*_rms.json` (the RMS version, a bug).

| Read | ResNet-18 (8 blocks) | CLIP (12 blocks, MLP) |
|---|---|---|
| distinct main parents per block | 23-153 | 65-306 |
| share of nodes with a sibling | .81-.94 | .96-.99 |
| ancestor chain depth (median) | 1 -> 4 (deeper blocks have longer chains) | 1-2 |
| roots reached by the last block | 8-12 | 29-72 |
| main parent clear (top / second score, median) | 1.14-1.37. 20-42% of nodes below 1.1 | 1.14-1.35. 20-40% below 1.1 |
| biggest family | 7-64 | 766-2,341: one block-0 head is the main parent of most MLP neurons in blocks 3-10, and of every head in blocks 4-10 |
| level the cut reaches | 95-50%: level 1. 25%: level 1-2. 5%: level 3-4 (blocks 4-7), roots (blocks 0-3) | 95-25%: level 1 everywhere. 5%: level 1 (blocks 4-11), roots (blocks 0-3) |

Readings:
1. ResNet-18 has a real tree. Chains grow with depth, and smaller sizes cut higher, as Shonu's cut-level rule says.
2. CLIP's tree is a star. Its chains are 2 steps long (MLP neuron -> block-0 head -> embed dim), so size changes only how much of level 1 merges. Possible cause: a head writes 64 directions and is scored as one parent, while an MLP neuron writes 1. Open, look item.
3. The "one parent" law is only partly true: 20-42% of nodes have a second parent within 10% of the first.

## Checks (before any score is read)

| Check | ResNet-18 | CLIP |
|---|---|---|
| identity f = 1 (200 photos) | KL -4e-9, agree 100% PASS | KL 9.4e-7 (float floor), agree 100% PASS |
| planted twins (3-16 per block, same read, write halved) | KL -4e-9 PASS | KL 9.4e-7 PASS |
| critic (fresh agent, page vs code) | no result-changing bug. 3 unstated choices written onto the page (wording only) | same |

Bugs found and fixed by the checks (page history in `PAGE.md`): RMS counted constant offsets as distinctions. Units that never fire got a threshold above every token. A GELU mean was clipped to +1e-8 (write x -1e6). The scale was measured after the gate shift. The threshold midpoint moved twins off their own zero. Each was a code bug. None changed the concept.

## Loop 1 (page v1): the first whole build

Essay passage (read in full): 223-241. "Smaller models may stop at coarser distinctions, with something closer to a general car neuron." The small model's last layer "is still differentiating, but more coarsely, and it ends early."

Page diff: none (first build). Stream full width, so the rows are NOT true 95/50/5% networks. Every row states its parameter count.

**Prediction (written before any score, 2026-11-26 late night):**

| Size | ResNet-18 agree (KL) | CLIP agree (KL) | Old best calculated (agree) |
|---|---|---|---|
| 95% | 55-70% (~1.0) | 65-80% | ResNet 42.7% floor; CLIP 71.4% |
| 85% | 30-45% | 30-50% | ResNet 32.6%; CLIP 21.0% |
| 50% | 5-15% (bar: > 10% on the 800 unseen) | 2-10% | chance on both |
| 25% / 5% | chance | chance | chance |

Leave-one-out at 85%: `rand` (random groups, same sizes) loses more than 5 agree points to the tree on both networks, if the tree matters. `write` (strongest member's write only) loses to the tree. `fire` (strongest member's own read and gate) is the closest rival; it may tie.

### Loop 1 result (2026-11-27, after four code fixes; all on the current code)

Code fixes before reading (page history in `PAGE.md`): read weight floor, write on the top 1%, all-firing union, write clipped to [0, 1]. The last one: a single nearly silent pair in CLIP block 1 (928+1524) had write weights 181 and 215 and alone sent CLIP to chance (found by bisection). Identity and planted checks pass on both networks after each fix.

| Row (agree) | ResNet 95 / 85 / 50% (1,000) | CLIP 95 / 85% (1,000) |
|---|---|---|
| tree v1 (union firing) | 73.6 / 34.7 / 0.3 | 71.1 / 14.9 |
| `fire` (strongest member's read and gate) | 74.7 / 41.6 / 0.2 | **77.4 / 41.6** |
| `rand` (same-form null) | - / 8.7 / - | - / 0.9 |
| old best calculated | 42.7 floor / 32.6 / chance | 71.4 / 21.0 |

Look: the union read IS realized (overlap: coarse rate = union rate; 85-99% of members' strong firings caught, ResNet). It still loses. At 50%, the built network fires 2x as broadly as the trained student (s4b0 .49 vs .24), and the stream drift makes nodes fire more broadly than designed (.31 designed).
800 unseen: CLIP fire vs union at 85%: 218 vs 6 photos, McNemar z +14.2. ResNet v2 vs union: z +3.0.
Prediction check: 95% and 85% beat the predicted ranges on CLIP; 50% at chance (predicted 2-15%: missed).

## Loop 2 (page v2): the coarse node fires as its strongest member

Essay 251: a crushed layer stays interpretable and coarser, "one curves & loops neuron", not superposition.
Page diff: line 7 = the branch's strongest member's own read and gate (v1 union is now row `union`).
Tests rejected on the way: `zero` (union read, no gate shift): far too narrow, write blow-up, KL 5e5. Ancestor-rate law (fire at the common ancestor's rate): predicted rates barely move (.29 vs .31), not built.

| Row (ResNet, 1,000) | 95% | 85% | 50% | 25% | 5% |
|---|---|---|---|---|---|
| v2 | 74.7 | **41.6** | 0.2 | 0.0 | 0.0 |
| `union` / `write` / `rand` | | 34.7 / 35.8 / 22.2 | | | |

Look at 50%: v2 fires too narrowly (s4b0 .14; trained .24; union .49). Each block alone at 50% costs a lot (agree .46-.74, KL .43-1.71); three coarse blocks reach 5% agree. Tree beats `rand` per block only in s2b0 and s3b1.
Decision: keep v2 as the best build so far. v2 = strongest member kept + branch write, so it is tree-chosen pruning, not yet a coarse node. The union is not refuted on CLIP (read realized only in part). 50% fails for a reason that every block shares. Next: loop 3 tests Shonu's cut rule as stated (one level for every layer), which page line 5 did not follow (it cut every block to the same fraction).

## Loops 3-5 (2026-11-27): three tree lines tested on ResNet, page v2 kept

Essay 223-241, read in full after these runs: siblings sit on each other's negative poles (taxi vs ambulance); the small model "ends early", its last neuron at the parent's distinction (cars, trains, carts vs boats, planes).

| Loop | Page line tested | Agree 95 / 85 / 50% (ResNet, 1,000) | Reading |
|---|---|---|---|
| v2 (reference) | | 74.7 / 41.6 / 0.2 | |
| 3 `L` | line 5: ONE cut level for every block (Shonu's rule), the size budget over the whole network | 75.3 / 41.0 / 0.3 | neutral. At 50% the level is still 1 everywhere. At 5% it reaches the roots (level 9) |
| 4 `P` | line 4: a node's parent = the coarse node that holds its main parent (tree read front to back on the coarse network) | 74.7 / 40.7 / 0.3 | neutral |
| 5 `W` | line 6: merge into the sibling whose WRITE, as the next readers read it, is closest (was: closest read) | 74.4 / **43.9** / 0.1 | +2.3 at 85%. 50% unchanged |

Reading: on v2, loops 3-5 only change WHICH node of each group is kept, so they test pruning choices, not the tree (habit H3). No prediction was written before these runs. All leave 50% at chance. One coarse block alone at 50% already costs KL .4-1.7. The loss lies in what a single coarse node can carry: one read and one write per branch. None of these lines changes that.
