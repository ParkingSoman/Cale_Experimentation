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
| critic (fresh agent, page vs code) | running | running |

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
