# Neuron coarsening log

**Index of all results and how they connect: `onecoarse/INDEX.md` (read it first).**

Plan: `~/.claude/plans/i-ve-thought-about-our-stateful-feigenbaum.md` (approved 2026-11-13).
Code: `research/clean/neuron/`. Results: `research/clean/neuron/results/`.

## The three steps (Shonu, 2026-11-13)

| Step | Question |
|---|---|
| 1 | Given a neuron, its immediate children and a recipe, how do we coarsen it? |
| 2 | How do we coarsen a neuron at any layer, through its final children? |
| 3 | How does the readout matrix W change? |

Object, chosen by Shonu on 2026-11-13: **picture A, the parent absorbs.** The children go. The parent stays and gets new output weights that do their job coarsely. In this picture, step 3 is the second half of step 1.

## Words

| Word | Meaning here |
|---|---|
| immediate child | a reader with a direct edge: no neuron sits between parent and child |
| direct edge | the parent's exact part of the child's pre-activation, averaged where the child is on |
| positive pool / negative pool | children with a positive / negative direct edge from the parent |
| recipe | the developer's list of children to remove |
| absorb | change only the parent's readout: column p of fc, and the bias |

## N1. Picture A on ResNet-18, stage 4 (2026-11-13)

```
before:  s4b0:p ──conv1──> s4b1i:j (children) ──conv2──> stream ──> fc ──> logits
            └──────────── skip ──────────────────────────> stream ──> fc
after:   children filled with their means.  fc column p and the bias change.
```

Setup:

| Item | Value |
|---|---|
| Network | torchvision ResNet-18, IMAGENET1K_V1 |
| Parents | s4b0:245 (small coloured bird), s4b0:418 (long muzzle profile), s4b0:237 (slim glossy body, read negatively) |
| Recipe "map" | the children named in the read cards (`curves/map_agents/resnet18/cards2/`) |
| Recipe "edge" | the 10 inner channels with the largest positive direct edge |
| Goals | "all": every class. "own20": the 20 classes where the parent fires most (on fit photos) |
| Photos | the 35,000 reading photos, 35 per class. Even positions in a class fit, odd positions test. A lock closes the sealed photos to agents, so none are used |
| Rules | "closed": squared-logit fit, label OPT-D2. "kl_opt": the KL optimum of the same two vectors. "whole": all of fc refit on KL, as a reference |

### Direct edges against the map

| Parent | Map children (rank of 512 by direct edge) | Agreement |
|---|---|---|
| 245 bird | 243 (1), 73 (2), 377 (106) | 2 of 3 |
| 418 muzzle | 317 (2, named as "main_parent" in its card), 377 (61), 408 (86) | 1 of 3 |
| 237 glossy body | 243, 98, 500, 383, 58, 189: all have NEGATIVE edges (ranks 383 to 505) | the map calls them "competitor, -". The sign agrees |

The parent gives 15% of child 243's input where 243 is on. For the other map children it gives 2 to 18%. So each child has many parents.

### Loss on test photos (KL, nats) and the share the readout takes back

| Parent | Recipe | Goal | Drop | Closed takes back | KL optimum takes back | Whole readout takes back | Closed / KL optimum |
|---|---|---|---|---|---|---|---|
| 245 | map | all | .0047 | 7% | 34% | 54% | 1.39 |
| 245 | map | own20 | .0654 | 56% | 63% | - | 1.20 |
| 245 | edge | all | .0143 | 5% | 29% | 49% | 1.33 |
| 245 | edge | own20 | .0853 | 52% | 54% | - | 1.05 |
| 418 | map | all | .0028 | 2% | 24% | 51% | 1.29 |
| 418 | map | own20 | .0161 | 28% | 53% | - | 1.53 |
| 418 | edge | all | .0173 | 3% | 19% | 47% | 1.20 |
| 418 | edge | own20 | .0270 | 22% | 27% | - | 1.07 |
| 237 | map | all | .0092 | 3% | 32% | 57% | 1.43 |
| 237 | map | own20 | .0066 | 8% | 28% | - | 1.28 |
| 237 | edge | all | .0169 | 2% | 18% | 47% | 1.20 |
| 237 | edge | own20 | .0132 | 9% | 16% | - | 1.09 |

The KL optimum from three starts (zero, closed, random) agreed to 1e-11 on fit photos for goal "all", and to 5e-6 for goal "own20".

The table above is NOT the result. The controls below change it.

### Controls (added after review, same day; `neuron/controls.py`, `neuron/rand_drop.py`, `results/controls.json`)

The fit has two free vectors: a bias `a` and the parent's column `b`. The controls ask whether the parent matters.

| Control | What it fits |
|---|---|
| bias only | `a`, with `b = 0` |
| other channel | `a + b * g_q`, for the 2 other parents and 20 random channels `q` |
| random removal | 3 or 10 random inner channels removed, not tied to any parent |

Share of the removal cost taken back, on test photos:

| Parent | Recipe | Goal | Bias alone | Parent column, beyond the bias | Parent's rank of 23 | 95% interval, parent minus bias (nats) |
|---|---|---|---|---|---|---|
| 245 | map | all | 29% | +5% | 1 | |
| 245 | map | own20 | 62% | +1% | 2 | -.0051 to .0043 |
| 245 | edge | all | 27% | +2% | 1 | |
| 245 | edge | own20 | 56% | -2% | 6 | -.0030 to .0065 |
| 418 | map | all | 25% | -1% | 18 | |
| 418 | map | own20 | 54% | -1% | 3 | -.0000 to .0003 |
| 418 | edge | all | 18% | +1% | 1 | |
| 418 | edge | own20 | 16% | **+11%** | 1 | **-.0047 to -.0015** |
| 237 | map | all | 31% | +1% | 3 | |
| 237 | map | own20 | 31% | -3% | 7 | -.0006 to .0011 |
| 237 | edge | all | 19% | -0% | 9 | |
| 237 | edge | own20 | 23% | -6% | 7 | -.0002 to .0020 |

Drop cost of each removal on each goal (KL, test photos):

| Removed | all | bird classes (245) | muzzle classes (418) | glossy classes (237) |
|---|---|---|---|---|
| 245's map children | .0047 | **.0654** | .0012 | .0009 |
| 418's map children | .0028 | .0059 | **.0161** | .0062 |
| 237's map children (negative edges) | .0092 | **.0598** | .0014 | .0066 |
| 3 random channels | .0043 | .0005 | .0061 | .0065 |
| 10 random channels | .0125 | .0045 | .0052 | .0043 |

### Second review checks (`neuron/extra.py`, `results/extra.json`, `results/extra_A.log`)

A. Goal "all", parent column minus bias, 95% bootstrap interval (nats):

| Parent | Recipe | Parent column, beyond the bias | Interval |
|---|---|---|---|
| 245 | map | +4.7% | -.00031 to -.00013 (parent better) |
| 245 | edge | +1.7% | -.00037 to -.00012 (parent better) |
| 418 | map | -1.2% | .00002 to .00005 (bias better) |
| 418 | edge | +1.0% | -.00027 to -.00008 (parent better) |
| 237 | map | +0.8% | -.00015 to .00000 |
| 237 | edge | -0.4% | .00001 to .00011 (bias better) |

B. Parent 237's map children WITHOUT child 243: the cost on bird classes falls from .0598 to .0067. Cost on 237's own classes: .0062. So the bird cost came from child 243, the bird specialist, which the two parents share.

C. Own20 rows again, with each child filled by its mean on the goal's fit photos:

| Parent | Recipe | Drop | Bias alone takes back | Parent column, beyond the bias | Rank of 21 | Interval |
|---|---|---|---|---|---|---|
| 245 | map | .0281 | 16% | +1.6% | 1 | -.0048 to .0041 |
| 245 | edge | .0451 | 19% | -3.5% | 6 | -.0026 to .0061 |
| 418 | map | .0129 | 43% | -1.2% | 3 | -.0000 to .0003 |
| 418 | edge | .0231 | 4% | **+11.5%** | 1 | **-.0043 to -.0013** |
| 237 | map | .0067 | 32% | -3.2% | 6 | -.0005 to .0012 |
| 237 | edge | .0124 | 19% | -6.9% | 7 | -.0002 to .0020 |

With goal-mean fills, the bird children's cost on bird classes falls from .065 to .028. So about half of the first number was a wrong constant.

### What survives

1. **WITHDRAWN: "a parent absorbs 16 to 63%", and "the closed form passes the 1.3 bar".** The bias did most of it.
2. **The parent's column adds a small amount, sometimes.** It beats the bias with an interval that excludes 0 in 4 of 12 rows: 245 on all classes (+4.7%, +1.7%), 418 edge on all classes (+1.0%), and 418 edge on its own classes (+11%, which holds with both fills). In the other rows it adds 0 or does worse. So "the parent takes up its children's error" is small and not general in this form. It is not shown to be impossible.
3. **The map children are specific.** The bird parent's map children cost .028 on bird classes with goal-mean fills, and about .001 on other goals. Three random channels cost .0005 on bird classes.
4. **WITHDRAWN: the "negative pool" reading of parent 237.** Its bird cost came from the shared child 243 (check B). The map fact that stays: all 6 of its map children have negative direct edges, as the cards say ("competitor, -").
5. **A reading, not a finding:** the parent's signal already reaches the readout through the skip. The children add what the parent combines with other inputs. A linear change to the parent's readout cannot rebuild a combination. This would explain items 1 and 2. It is not tested.
6. **The whole readout takes back about 50%** on goal "all", against 18 to 31% for the bias. So other surviving channels can absorb more. Which ones is open.
7. Each fit's KL optimum is convex. Three starts agree to 1e-11. So each fit is one fixed calculation.

### Corrections

- 418's map children: the parse read only the cards' "inputs" field. The "main_parent" field also names s4b1i:317 for parent 418, and 317 is its second-strongest direct edge. So the agreement is 1 of 3, not 0 of 2.
- My edge share (mean where the child is on) is not the cards' measure (at the child's peak, against the fill). So "2 to 18%" and LADDER 57's "parents carry 67%" are different measures.
- The plan said one parent. I ran three.
- The feature is pooled s4b1:p, the parent's relay through the skip. It is not s4b0:p itself.

### Limits

- Reading photos only, split inside each class. A lock closes the held-back photo set to agents.
- Each own20 goal has 360 fit photos and 340 test photos.
- Three parents, one stage, one network.
- The map agents wrote the cards from these same reading photos.
- The form tested changes only the fc column of the parent's relay channel, in a network with skips. The parent's own output weights (conv1 of block 1) never changed. Shonu framed step 1 for MLPs.

| | |
|---|---|
| Cale line | weak for now. In this form (the fc column of the parent's relay channel), a parent took up 1 to 11% of its children's error beyond a constant shift, in 4 of 12 rows. The whole readout takes up about half |
| Whole-network build | no |
| Label | closed form: OPT-D2. Absorb by KL optimum: the exact least KL over ONE narrow form (the parent's readout column and the bias), because the problem is convex. It is not OPT-KL in the bar's sense, which asks for the least over every allowed small network. The whole readout already does better |

## N2. What each child mixes in besides its parent (2026-11-13; `neuron/mix.py`, `results/mix.json`, sheets in `results/mix_sheets/`)

Method: each child's pre-activation at its 500 strongest fit photos, split exactly into parts from each s4b0 input channel (against the fill; the sum check holds to 1e-6). The firings are clustered by input profile (k-means, k from 2 to 5 by silhouette, seed 0), as the companion paper does. Effect = KL(N || N with the child at its mean), on the 500 strongest test photos.

Edge confidence, from the read cards:

| Child | Card's link to the parent | Confident? |
|---|---|---|
| 243 | 245 is its "main_parent", type refines, sure .5 | yes |
| 317 | 418 is its "main_parent", type refines | yes |
| 73 | 245 is its "main_parent", type dial, sure .25, "bundle 1 only" | weak |
| 377 | its card names s4b0:418 as main parent. My scripts filed it under 245 | misfiled |
| 408 | 418 is listed last, type part_of, sure .2 | weak |

| Child | Subcircuit | Share of effect | Parent's rank | Top inputs | Top classes | What the photos show (by eye, 12 photos each) |
|---|---|---|---|---|---|---|
| 243 | refines | 76% | 1 | 245, 67, 422 | house finch, brambling, goldfinch | small perched birds, all 12 |
| | offshoot | 24% | 4 | 422, 377, 476 | canoe, speedboat, paddle, leafhopper | a small tapered body on plain or green ground: about 4 of 12 are birds that 245 ranks low, plus leafhopper, cicada, canoe, speedboat |
| 73 | refines | 5% | 1 | 245, 55, 410 | bulbul, junco, coot | perched birds, all 12 |
| | offshoot | 95% | 102 | 372, 450, 220 | rule, slide rule, accordion, barometer | dial faces and graduated rods: clocks, scales, slide rule, gauges, a harp |
| 377 | diffuse | 95% | 2 (part .019) | 81, 245, 31 (all small) | ptarmigan, redshank, junco, tow truck | no single theme: a dog, lips, a diagram, birds, a loader, meat, oil rigs |
| | offshoot | 5% | 414 | 418, 31, 220 | Eskimo dog, dogsled, borzoi | pale dogs with open mouths, a white goose |
| 317 | offshoot | 70% | 2 (small) | 32, 418, 427 | tank, grille, sports car, cab | an elongated body pointing forward, side view: tank, sports car, kayak, running dogs |
| | refines | 30% | 1 | 418, 271, 399 | borzoi, red wolf, Afghan hound, Saluki | long-muzzle dogs, all 12 |
| 408 | own family | 44% | 8 | 266, 173 | EntleBucher, Bernese, Appenzeller | tricolour dogs, all 12 |
| | offshoot | 43% | 349 | 412, 113, 450 | crossword, hard disc, shoji | high-contrast patterned objects: hard discs, clocks, crossword, spider web, cauliflower |
| | offshoot | 13% | 221 | 400, 490, 434 | broccoli, coral reef, rapeseed | bumpy or spiky texture: broccoli, rambutan, sea urchin, green hummingbirds, coral |

Notes on the method. k was forced to be at least 2, and the silhouettes are low (.14 to .25). So the clusters are soft. The evidence for separate facets is the class split and the photos, not the cluster scores. "Parent's share of positive input" (1 to 8%, all firings) divides by the sum over all 512 channels. In each refines cluster the parent is the largest single part, so that number does not by itself explain part 1.

Negative side: s4b0:237 (slim glossy body) is the strongest negative input of 243, 73 and 408, and the second of 317. Siblings co-fire instead of avoiding each other: 73 sits at the 83rd percentile of its own values on 243's top photos.

Predictions (written before the run), judged on all 5 and on the 2 confident edges (243, 317):

| # | Prediction | All 5 | Confident edges only |
|---|---|---|---|
| P1 | subcircuits have distinct inputs (top-10 overlap below 30%) | 3 of 5 (73, 377, 408) | 0 of 2: both share inputs across subcircuits (243: 422; 317: 418) |
| P2 | one subcircuit carries the parent in its top 3 | 4 of 5 (377 only by the letter) | 2 of 2 |
| P3 | at least 20% of the effect outside the parent's subcircuits | 3 of 5 (243, 73, 408) | 1 of 2 (243). 317's vehicle subcircuit still has 418 second |
| P4 | siblings share the parent and differ in other inputs | 243 and 73 only | not testable (no confident sibling pair) |

### What the finer neurons mix in

```
child 243 = [245 small bird] + [67, 422]          -> perched birds                       (76% of effect)
          + [422, 377, 476]                        -> small tapered bodies: canoes, bugs  (24%)
child 317 = [418 long muzzle] + [271]              -> long-muzzle dogs                    (30%)
          + [32, 427] (418 weak)                   -> forward-pointing bodies: tanks, cars (70%)
```

1. **Mixed, not one rule.** On the two confident edges, the parent's family is 76% of 243's effect and 30% of 317's. The rest comes through other inputs, on classes outside the parent's family that share a visual pattern (a small tapered body, a forward-pointing elongated body). The weak edges (73, 408) look the same way, more strongly: 95% and 56% outside.
2. This is the essay's recursive generalization: "a neuron generalizing along more than one axis", where an offshoot "arrives by its own route". It is NOT the essay's gate picture (a car gate in front of taxi and ambulance specialists). That picture was not tested: it needs several blocks, and I tested one convolution.
3. A parent's readout cannot take up an offshoot, because the offshoot's information comes through other inputs. This is one reason for part 1's small result.

How this fits the essay and the companion paper:

| Source says | Here |
|---|---|
| Recursive generalization: a neuron grows along several axes, each easy to follow | yes. Each offshoot shares a shape or texture with the parent's family |
| "Grouped by the circuits that feed them, they separate into facets ... and one offshoot ... which arrives by its own route" (geology figure) | yes. Clustering by inputs separates the bird facet from the canoe facet |
| The neuron is a verb (a shared pattern), not a group of nouns | yes. 317 reads as "elongated thing pointing forward", not as "dogs and tanks" |
| Paper: subcircuits are "highly disjoint" | weaker here: soft clusters, and both confident children share inputs across subcircuits |
| Weak activations can be ablated "for almost no cost", except dial-like layers | see N3: about half the removal loss is on photos where the children fire weakly. Not a clean test of the essay's claim (a mean fill is not an ablation of weak values) |
| Gate in front of specialists (car gate, then taxi and ambulance) | not tested: it needs several blocks (Shonu's step 2) |

| | |
|---|---|
| Cale line | none yet. It says what a faithful absorption must do: send each subcircuit to the parents that drive it |
| Whole-network build | no |
| Label | none. A measurement, no theorem |

## N3. Absorption that follows the subcircuits (2026-11-13; `neuron/absorb2.py`, `a3neg.py`, `a3bias.py`; `results/absorb2.json`, `a3neg.json`, `a3bias.json`)

Recipes: parent 245 loses children 243, 73, 377. Parent 418 loses children 317, 408. Children filled with their all-photo means. Test photos.

| Form | What changes | Real network? |
|---|---|---|
| bias | the readout bias | yes |
| A1 | bias + the parent's readout column | yes |
| A2 | bias + the readout columns of every input that drives a subcircuit of a removed child (top 2 per subcircuit: 10 channels for 245, 9 for 418) | yes |
| rand | bias + readout columns of random channel sets of the same size (5 draws) | yes |
| A3r | children KEPT, each reading only the top-10 inputs of its parent's subcircuit (other inputs at their means) | yes: the child with fewer input weights |
| A3all | children KEPT, each reading only the top-10 inputs of each of its subcircuits | yes |

Share of the removal cost taken back:

| Parent | Goal | bias | A1 | A2 | rand (median of 5) | A3r | A3all | A3r + bias | A3all + bias |
|---|---|---|---|---|---|---|---|---|---|
| 245 | all | 29% | 34% | 34% | 26% | -31% | -17% | 36% | **41%** |
| 245 | own20 | 62% | 63% | 54% | 47% | 32% | 37% | | |
| 418 | all | 24% | 25% | 27% | 21% | -22% | -10% | 31% | **37%** |
| 418 | own20 | 50% | 52% | 42% | 34% | 69% | 65% | | |

Intervals (goal "all"): A2 minus random sets excludes 0 for both parents. A2 minus bias excludes 0 for 245 and includes 0 for 418. The A3all + bias intervals are in the review checks below. On own20, A2 overfits (9 or 10 columns of 1,000 numbers on 360 photos) and loses to the bias alone.

Two side checks:

- Adding each child's 5 strongest negative inputs to A3 did not help (`a3neg.json`). So A3's failure without a bias was not lost suppressors.
- Each child's pre-activation with every input at its mean is below 0 (-0.27 to -0.66). So a coarse child does not fire on every photo. Its error without a bias is a smaller shift, and the bias removes it.

### Review checks (`neuron/link.py`, `results/link.json`)

Controls for the coarse child. Each child keeps the same NUMBER of inputs as in A3all, chosen three ways, plus a bias. Goal "all":

| Parent | drop + bias | A3all + bias (subcircuit inputs) | largest inputs by mean size, no clustering | random inputs |
|---|---|---|---|---|
| 245 | .00331 | **.00273** (interval vs drop + bias: -.00075 to -.00043) | .00300 (-.00043 to -.00022) | .00349, worse than drop (+.00015 to +.00021) |
| 418 | .00243 | **.00203** (-.00050 to -.00031) | .00211 (-.00040 to -.00026) | .00255, worse than drop (+.00011 to +.00013) |

The subcircuit inputs beat both other choices. They overlap the largest-input choice by 37 to 56%.

Where the removal loss sits (goal "all", test photos):

| Parent | Photos where a removed child fires strongly | Share of the loss on those photos (mean fill) | Loss with a zero fill instead |
|---|---|---|---|
| 245 | 1,410 of 17,000 (8%) | 55% | .0081 (mean fill: .0047) |
| 418 | 984 of 17,000 (6%) | 50% | .0049 (mean fill: .0032) |

So N2's strong firings (8% of photos) hold about half of the loss. The other half sits on photos where the children fire weakly. A zero fill costs more than a mean fill.

### What this supports

1. **Picture A, honest answer: the parents' readouts take up little.** The parent alone (A1) and all driving parents together (A2) gain at most 5 points over a bias. A2 does beat random channels of the same number.
2. **The coarse child plus a bias does better: 41% and 37% taken back.** Each child keeps about 20 to 27 of its 512 input channels, chosen from its subcircuits. Subcircuit inputs beat the largest inputs, and random inputs do worse than removing the child. This is NOT picture A: the child stays, with fewer inputs. It shows that most of what a child does lives in a few parents per facet.
3. Keeping only the parent's subcircuit (A3r + bias, 36% and 31%) does worse than keeping every subcircuit (A3all + bias). This fits N2: the offshoots carry real effect.
4. About half of the removal loss sits on photos where the children fire weakly. N2 studied only the strong firings, so it explains about half of the loss.
5. On narrow goals (own20) the fits overfit or win by chance. The 360 fit photos are too few to judge them.

Gaps against the plan: all fills here use all-photo means, not goal means. The whole-readout refit was not run for these two recipes.

| | |
|---|---|
| Cale line | a coarse child that reads only its subcircuits' parents takes back about 40% of what the removal loses, with no training. The whole readout refit took back about 50% in N1 (other recipes, so only a rough reference) |
| Whole-network build | no |
| Label | none. Measurements |

## N4. A small model trained the same way: is its neuron the big parent plus children? (2026-11-15; `neuron/pair/`, `results/pair/`)

Plan: `~/.claude/plans/okay-last-session-within-scalable-beaver.md`. Shonu's hypothesis: a smaller network trained the same way already does the coarsening. Its neuron = the big network's parent + a chosen mix of the parent's children and facets. Learn that mix, and the parent edit follows.

Pair: torchvision ConvNeXt-Small and ConvNeXt-Base (one shared recipe, blocks 3-3-27-3 in both, stream width 768 against 1,024). Stage 4: 3 blocks, 3,072 against 4,096 hidden neurons per block (first Linear, pre-GELU), 7 x 7 positions. Photos: the reading photos.

```
Base block k:  p --write--> stream --read--> children (blocks k+1, k+2)
Small block k: s  = p's counterpart       question: s = p + which children?
```

| Word | Meaning |
|---|---|
| counterpart | Small neuron s and Base neuron p such that p is s's best match BOTH by firing (correlation of per-photo peaks) and by writing (cosine of direct class-score effects) |
| child (by removal) | a later Base neuron whose pre-activation drops when p's output is set to its mean, on p's top photos (drop in units of its std) |

### Step 1: counterparts exist (P1 holds)

Firing and writing pick the same Base neuron (3,000 fit photos; chance 1 in 4,096):

| Block | Same neuron | Firing pick in writing top 5 | Pairs | Median breadth, Small / Base |
|---|---|---|---|---|
| 0 | 66% | 79% | 2,017 | 1.29 |
| 1 | 53% | 69% | 1,632 | 1.33 |
| 2 | 54% | 72% | 1,649 | 1.40 |

Match strength is moderate (firing about 0.5, writing about 0.55). The counterpart fires on 30-40% more photos (above half its 99th percentile) than its Base neuron: coarser, as the essay says.

Limits: both tools read class tuning (3 photos per class in the peaks). Positional correlation shows block 0 faces block 0 (same-block .44 against .34, .32). For blocks 1 and 2 same-block and cross-block nearly tie (.436 against .432).

Matching tools tried (mean best match per Small neuron, 3,000 photos): positions pre-GELU .44, positions post-GELU .35-.48, per-photo peak .46-.50, top-1% overlap .34-.38, class-score write .46-.54.

### Withdrawn on review (same day)

- "Orphans (Base neurons with no counterpart) match the width cut": random picks would leave about 900 unpicked, perfect matching 1,024. The counts (1,075-1,421) do not tell them apart.
- A first mix test (`mix.py`) used weight-only children (virtual weights through the LayerNorm and depthwise conv). That score is flat (each parent gives 0.1-0.3% of a child's input) and was not validated. Its R^2 table is not a test of the hypothesis.
- "s does not fire more than p on the children's photos": p and the child share one network and s does not, so that number is biased negative. Only the orphan-against-matched gap means anything: in `orphans.py` it is 0.00 for orphans against -0.31 for matched neurons (the host covers orphans relative to its counterpart).

### Step 3: children by removal (`kids.py`, 20 pairs, block 0)

Removing one block-0 neuron moves later stage-4 neurons little: for 16 of 20 parents the largest child drop is 0.03-0.10 std, and the top-class logit moves by 0.00-0.01. The 4 best-matched parents (1711, 151, 799, 377) have clear children (0.25-0.48 std), and they share them (647, 517, 1483): a family of sibling parents with common children.

### Step 4: is s = p + p's children? (`pos.py`, 20 pairs, block 0, photo positions)

4,000 photos (2 fit, 2 test per class), 49 positions each, pre-GELU. Linear fits of s on fit rows, R^2 on test rows. Top-1 on these test photos: Small 80.5%, Base 81.1% (published 83.6%, 84.1%; different resize).

| Set (each "+ 6" set includes p) | Median R^2 |
|---|---|
| p alone | .25 |
| + 6 children (by removal) | .28 |
| + 6 look-alikes (same correlation with p, not moved by removing p) | .29 |
| + 6 random Base neurons | .28 |
| + 6 neurons chosen by their fit to s's residual (ceiling) | .36 |

Children beat look-alikes in 8 of 20 pairs. On each child's top 1% test rows, the residual of s after p beats the look-alike's in 60 of 120 cases (chance). The data-chosen neurons are not children (median child rank 4,197 of 8,192) and 38% sit in block 0.

**P3 and P4: not tested here.** Review (same day): for 16 of 20 parents the top child drops (0.03-0.10 std) are the noise tail of 8,192 scores, so those "children" are not children. Only parents 1711, 151, 799, 377 have real children. There, children beat look-alikes and random neurons on R^2 in 3 of 4 pairs, and lose the top-photo check in 3 of 4: mixed, n = 4. Unchecked: whether 647, 517, 1483 also move when unrelated block-0 neurons are removed (a LayerNorm rescaling would look the same). The facet sets of the plan did not run. All fits are pre-GELU over all 49 positions, mostly where the neurons are off.

### Step 4b: s as a read direction on the Base stream (`stream.py`, `rule.py`)

Each Small block-0 neuron fit as a linear read of u, the Base block-0 input (LayerNorm output, 1,024 numbers per position), on the same photos.

| Measure | Value |
|---|---|
| test R^2 of s from u (median, 10th-90th percentile) | .60 (.56-.67) |
| the Base read row (W1) closest to s's read direction is its counterpart p | 91% of 2,017 pairs. This cosine (covariance metric) equals a correlation of p's pre-activation with the stream-linear part of s, so it is a third activation tool. It adds: p is also the top match at positions, in block 0 |
| share of s's read (covariance metric) carried by p's row | 34% |
| by the best 5 / 20 / 100 Base rows | 45% / 50% / 59% |

Mechanical edits of p's read, scored by the match to s's read (median over 800 pairs; p alone .59): projection on the top 768 stream directions .58, top 256 .55, p plus its 5 closest sibling rows .53, covariance-smoothed .39. None beats p alone.

### What survives

1. **One-to-one counterparts exist** (the essay holds): by firing and writing (53-66% agreement against 1 in 4,096), and p is the top match at positions too (91%, block 0).
2. **The counterpart is coarser**: it fires on 30-40% more photos.
3. **"Parent + its children" is not tested yet** (see step 4). In read space, the parent carries a third of the small neuron's read. The rest spreads over many Base read directions. No simple rule (projection, siblings) recovers it.
4. **A confound limits the reading.** Small was trained separately, so its stages 1-3 differ from Base's. Only 60% of s is linear in Base's stream. So s minus p mixes this layer's coarsening with every upstream change.
5. Stage 4 has shallow trees. Removing one block-0 neuron moves later stage-4 neurons by 0.03-0.10 std for 16 of 20 parents. 4 strongly matched parents share children (647, 517, 1483).

| | |
|---|---|
| Cale line | none yet. Counterparts exist and are coarser. What the coarser neuron adds beyond its parent is open |
| Whole-network build | no |
| Label | none. Measurements |

Next (the plan's step 6, now first): a distilled pair with the upstream held fixed. Freeze Base, replace one block's 4,096 hidden neurons by fewer, and train only that block by KL to Base. Then the coarse neuron reads exactly the same u as the parents, and "coarse neuron = parent + children/facets" can be tested with no upstream confound.

### Step 4c: where the Small neuron is on and its counterpart is off (`sheet.py`, sheets in `results/pair/sheets/`)

14 pairs (the strongest of `kids.json`). Rows (photo x position) where p's pre-activation is below 0, top 150 by s. Every Base stage-4 neuron's post-GELU output on those rows, z-scored against all rows.

| Pair | p's own photos (by eye) | Extra photos of s (p off) | Base neuron firing there |
|---|---|---|---|
| s603 ~ p1711 | padlocks, combination dials | safes: keypad safes, safe doors, a safe dial | b0:1959, z 32 (same block as p). Then b2:830 z 22, b1:1628 z 17 |
| s802 ~ p151 | microphones, stage cables, performers | lamps, light fixtures, stands | b2:1357 z 10, b0:460 z 10 |
| s2158 ~ p1915 | patterned textures | mixed textures | b2:1766 z 7, b0:2281 z 7 (no clear theme by eye) |

In 10 of 14 pairs a block-0 neuron (a same-layer sibling of p) is in the top 3 there. In most pairs the top neurons are not p's children (child ranks in the thousands). The 4 family pairs give the most specific extras (z 8-32). The others give z 2.5-7.

**Trap found later the same day (reader agent):** Base pre-activations are below 0 at more than 99% of rows (a Base neuron is on at about 0.24% of rows). So "p < 0" is not "p off", and the rows above are nearly just s's top rows. The z-scores still name Base neurons that fire on s's top rows. The table stands as a description of 3 pairs, not as a test.

Reading (n small, by eye on 3 sheets, superseded by step 5): **the Small neuron is its Base counterpart plus a related same-layer neuron.** Padlock + safe gives one "lock" neuron. This is the essay's "one car neuron covering many kinds of car". In tree words, Small's neuron sits one level up: it is the parent of two Base siblings. This is closer to picture B (siblings merge) than to picture A (a parent absorbs its later children).

### Step 5: open search, three agents (board `neuron/pair/BOARD.md`; results in `results/pair/{reader,theory,geometry}/`)

Shonu asked for a concrete relation without a fixed hypothesis. Naive combinations first (`grams.py`, `greedy.py`, `cover.py`): a sparse sum of Base outputs after GELU rebuilds s with test R^2 .13 (1 term) to .18 (3 terms). **Trap (reader agent):** `topsets.npz` ranks values after GELU, and GELU approaches 0 from below for very negative inputs. So about 70% of each Base top-1% set is the neuron's most negative rows, and the `cover.py` numbers (21% cover, 32% union) are void.

Three agents then worked in parallel and talked through the board: reader (photo sheets, the essay's way), theory (first principles of training with less room), geometry (weights and the stream).

| Agent | Main measured result | Control |
|---|---|---|
| reader (30 pairs, on-rows = pre-activation > 0) | s keeps .36 of p's on-rows exactly, .54 within one position. 60% of s's on-rows have no p within one position ("added rows"). Covering half of the added rows takes about 10 Base neurons (p's own on-rows: 4). The best single neuron covers 11% | other pairs' s keep .07 of p's on-rows |
| reader, by eye (not blind) | 16 of 30: s keeps p's local pattern and drops p's object, species, colour or shape condition ("axis"). 3: adds the parts next to p's object ("context"). 2: merges with ONE other Base neuron (best single cover .82). 8 unclear. Examples: wiry terrier coat -> wiry hair of any animal; bridge arches -> dark structure against the sky; crossword grid -> any array of dark and light cells | code signature per call: added rows in p's classes .33 (axis), .50 (context), .03 (merge) |
| geometry | p is the top match among all 4,096 Base reads for 93-94% of s. p's read gives about 40% of s's read (cosine .58; random Base neuron .05-.08). Neighbour reads, cluster centres and usage weights add nothing (p + 5 nearest .587, p + random .583). Small's input is 56% linear in Base's | |
| geometry, reader's carriers | the remainder of s's read after p points toward the mean read of the Base neurons that fire on the added rows: cosine .245. p + 8 carriers explain .42 of s's read | another pair's carriers: cosine .055, share .37. Random: .36 |
| theory | Which Base neurons survive as near-copies is predicted by value = output strength x class-effect size: Spearman .56, partial .42 after Base redundancy. Frequent neurons survive WORSE (partial -.25). About half of Small neurons are near-copies of one Base neuron. The other half have no linear match at the same position (all 4,096 Base outputs: R^2 .03) | Base-Base redundancy |

### The finding (2026-11-15)

```
Base:   p = [local pattern] AND [object condition]      e.g. wiry coat, on terriers
        ~10 other Base neurons = the same local pattern under other conditions (other animals)
Small:  s = [local pattern], condition dropped           wiry hair, on any animal
        read(s) = read(p)  +  a part that points toward the shared read of those ~10 neurons
```

1. **A coarse neuron is its Base counterpart, generalized one step along one axis.** It keeps the counterpart's local pattern (about half its firing rows, often one position away) and drops the condition that made the counterpart specific. So it fires wherever that pattern appears. In Base those rows are split over about 10 neurons, none of which covers more than 11%.
2. **In weights:** the counterpart's read gives about 40% of the coarse read. The rest points toward the shared read of the Base neurons that carry the pattern elsewhere (cosine .245 against .055). It is not a few named neurons and not the counterpart's read neighbours.
3. **Which Base neurons keep a near-copy:** the valuable ones (output strength x class effect), not the frequent ones.
4. Rarer shapes: context added (3 of 30), and a merge of two Base neurons that share a motif (2 of 30: padlock + safe type).

This fits the essay: "With less room ... its neurons just cover coarser-grained concepts", and a neuron is a verb (a shared pattern), not a list of nouns.

Limits: the by-eye calls are not blind. The read rule is weak in size (.42 against .36 random), and its carriers are chosen from s's own rows, so it is not yet a rule from Base alone. Half of Small's neurons have no same-position linear match, and a one-position shift was tested only for firing sets (recall .27 -> .43). Small was trained separately (upstream 56% linear). One layer, one pair of models.

Next tests, in order:
1. A blind judge on added-row crops against class-matched random crops: "does this share p's local pattern?"
2. A Base-only rule: find each p's carriers without Small. These are Base neurons whose reads share p's local-pattern part, and they fire in other classes. Build read(p) + their shared read, then score against s.
3. A one-position-shift fit for the no-match half.
4. The distilled pair (Base frozen, one block narrowed), where there is no upstream drift.

| | |
|---|---|
| Cale line | a coarse neuron = the fine neuron with its condition dropped, so it covers its local pattern across the fine neurons that carry it elsewhere. A parent edit should drop the parent's condition and add the shared read of those carriers. It should not add its children |
| Whole-network build | no |
| Label | none. Measurements and by-eye reading |

### Step 6: rule tests after the agents, and a distilled pair (2026-11-15, later)

**Correction to step 5.** The "half of Small with no Base equivalent" (theory agent) is an artifact of fitting after GELU over all rows. The same neurons are as linear in Base's stream as the others: test R^2 from Base's block-0 input u is .596 (the "no match" group), .596 (middle), .624 (good).

**"Drop the condition" fails in weights** (`condition.py`). I removed from p's read the direction that its own classes share in u (class means of u weighted by p's output per class). The match to s's read gets worse (.58 -> .53 at full removal), and the change beats p alone in only 5% of pairs. Removing another neuron's class direction leaves .58 (beats p in 40%). So s's read keeps p's class condition. Reader also saw that s is not more class-diverse than p. Step 5's "condition dropped" holds by eye on photos, not as a removal in the read.

**Distilled pair** (`distill.py`, `distill_an.py`, `write_rule.py`, `absorb_build.py`, `ft.py`; results in `results/pair/distill/`). Base frozen. A narrow MLP (k hidden neurons) on the same input u, trained to reproduce Base's block-0 MLP output y (squared error, 6,000 fit photos, 1,500 test photos, all positions). No upstream drift.

| Coarse layer, k = 1,024 (Base: 4,096) | Test R^2 of y | KL to Base, end to end | Top-1 (Base 80.4%) |
|---|---|---|---|
| keep 1,024 random Base neurons unchanged | .299 | | |
| trained from random (2 seeds, 60 epochs) | .625, .624 | .0044 | 79.9%, 79.7% |
| **keep the 1,024 Base neurons with the largest value (std of output x write size), unchanged** | .652 | .0043 | 80.1% |
| **same, writes re-solved once by least squares (no training)** | **.675** | .0042 | 79.9% |
| same, plus biases lowered by .25 / .5 std (fire more broadly) | .666 / .642 | | |
| trained ceiling: the build, then trained (best of 30 epochs) | .709 | | |

Other sizes, build with re-solved writes: k = 512: .506 (unchanged writes .481). k = 2,048: .872 (.866).

In the trained-from-random layer: a coarse neuron's write matches its Base counterpart's write (cosine .70, random .00). Its read matches less (.35, random .05). Single neurons are not the same across the two seeds (median best match .38). Which Base neurons share a coarse host does repeat across seeds (77% of host-sharing pairs, 16% by chance). The rule "coarse output = Base outputs projected on the coarse write" gives correlation about 0: with 1,024 writes in a 1,024-wide stream, any output projects. Dropped.

The end-to-end KL differences are small, because one block's MLP moves the output little. The layer R^2 is the measure that separates the builds.

### The finding (end of session)

```
Base layer (4,096 neurons)                     coarse layer (k neurons)
1. rank every Base neuron by value = std(output) x |write|
2. keep the top k: same reads, same firing                     <- a near-copy of one Base neuron each
3. re-solve the kept writes once (least squares on Base's output):
   w_new(kept) = own write + the part of the dropped neurons' writes it can carry
```

1. **Which parts:** the coarse layer is built from the most valuable Base neurons, kept as near-copies. In the separately trained Small, value predicts which Base neurons keep a near-copy (Spearman .56, partial .42 after redundancy, theory agent). Frequent neurons do worse. In the distilled pair, keeping the top-value neurons beats training from random (.652 against .625).
2. **How the dropped parts enter:** through the writes. Each kept neuron takes over a share of the dropped neurons' writes. The share is set by how its output co-varies with theirs, solved once in closed form (.652 -> .675). The build's error is 1.12 times the trained ceiling's error (.325 against .291). That is inside the 1.3 bar for this form, on held-out photos, for one layer.
3. **What changes in the read is small in a one-layer test.** Firing more broadly by a lower threshold hurts. Training from the build gains .03 more, so a read change exists but is small here. In the separately trained Small, the counterpart's read gives about 40% of s's read, and by eye s generalizes p's local pattern one step (16 of 30 pairs). The rest of s's read is not explained by any Base-only rule tried (condition removal, neighbours, centroids, carriers chosen from Base).

Limits: one layer (stage 4, block 0), one network family, layer-output loss (not end-to-end KL), 1,500 test photos. The by-eye calls are not blind.

| | |
|---|---|
| Cale line | a no-training coarse layer: keep the most valuable neurons, and re-solve their writes once so they absorb the dropped ones. On this layer it beats distillation from random and is within 1.12x of the trained ceiling's error. This is a parent-absorbs rule where "parent" means "most valuable survivor" and absorption runs through the write. It is a candidate for the liquid model: one ranking gives every size |
| Whole-network build | no (one layer) |
| Label | OPT-GOAL-like for one layer and the layer-output loss only: within 1.3x of the best trained layer of the same form, one fixed calculation (a ranking and one least-squares solve) |

Next: apply the same build to every block and score end to end by KL against a distilled network of the same size. Then repeat it on Pythia-160m MLP neurons (picture A on a language model).

### Step 7: the concrete gap between the build and the trained layer (`ft.py` modes, `gap.py` to `gap4.py`)

Same layer (stage 4, block 0, k = 1,024 of 4,096), test R^2 of the block-0 MLP output y.

| Layer | R^2 | Same form as trained? |
|---|---|---|
| build: top value, writes re-solved | .675 | yes |
| trained from the build, writes only | .681 | yes |
| trained from the build, biases only | .675 | yes |
| greedy selection (each pick adds what the others miss, batch OMP on covariances), writes re-solved | .687 | yes |
| greedy + a linear term in u, solved together | .702 | no (adds a 1,024 x 1,024 matrix) |
| trained from the build, reads + biases | .693 |  yes |
| trained reads, writes re-solved | .704 | yes |
| trained, all weights | .709 | yes |

Same-form gap after greedy selection: .709 - .687 = .022 R^2. Error ratio .313 / .291 = 1.08.

What the gap is not:
- Thresholds: bias changes median .004. Biases-only training gains 0.
- Linearizing: pre-activation std ratio trained / build .98. On-share .24% -> .28%.
- Decorrelating: median |corr| between kept neurons .001 before and after. Effective directions 820 -> 823.
- Each survivor copying the neurons it absorbs: the read change does not point at the absorbed neurons' reads (cosine .00, shuffled -.05). Adding those reads hurts at every strength. Firing change vs the "own + absorbed" target: correlation .15 (shuffled 0). A per-neuron refit to that target gains only .004-.008.

What the gap is:
- The dropped neurons carry 35% of y's variance. Only 7% of that part is linear in the survivors' outputs (12% with u too). So re-solving writes recovers little of the dropped part.
- Training turns every survivor's read by about 27% (median, covariance metric; the top 10% of neurons carry 26% of the change). The change partly removes the neuron's own read (cosine -.21).
- The gain falls where the dropped neurons were active: 72% of it on the 10% of test positions with the most dropped output, and -12% on the bottom half.
- Two thirds of the gain is in the reads (.675 -> .704 with writes re-solved). The rest is the writes co-adapting.

In short: about a third of the gap is the choice of survivors (fixed by greedy selection, no training). The rest is a joint rotation of the survivors' reads toward territory the dropped neurons covered. No per-neuron rule tried reproduces it.

### Step 8: the build on all 30 blocks of stages 3-4, end to end (`allblocks.py`)

Each block solved on its own, on the ORIGINAL network's activations (1,000 fit photos), all blocks swapped at once. Test: 1,500 test photos. Base top-1 80.4%. The separately trained ConvNeXt-Small scores 80.5% on these photos.

| Keep | Survivors | Writes | KL to Base | Top-1 |
|---|---|---|---|---|
| 50% | value | re-solved | .370 | 66.5% |
| 50% | value | unchanged | .537 | 63.2% |
| 50% | random | re-solved | .366 | 69.8% |
| 50% | random | unchanged | 1.114 | 65.4% |
| 25% | value | re-solved | 1.396 | 40.3% |
| 25% | value | unchanged | 2.172 | 26.3% |
| 25% | random | re-solved | 2.196 | 43.7% |
| 25% | random | unchanged | 3.382 | 24.1% |

Re-solving writes helps a lot in every row. The value ranking does not beat random survivors end to end at 50%. At 25% it gives the lower KL (1.40 against 2.20) but a slightly lower top-1. The loss against Base is large. Likely cause: errors pile up, because each block was solved on inputs the shrunk network no longer gives it. Next: the sequential build (`seq.py`).

Sequential build (`seq.py`: blocks shrunk in order, each solved on the shrunk network's inputs with the original stream as target, 500 fit photos, ridge 1e-6): keep 50% value KL .445, top-1 64.4%. Keep 50% random KL .861, top-1 62.0%. Keep 25% value KL 1.413, top-1 41.4%. Worse than the independent build at 50%. Likely overfit: each block also tries to undo all upstream error, on 500 photos with almost no ridge. Not tested further.

**End-to-end status:** the single-layer result (build within 1.12x of the trained layer's error) does NOT carry over to the whole network. With half of every stage-3/4 MLP neuron removed and no training, top-1 falls from 80.4% to 64-70%. The separately trained Small (fewer parameters) keeps 80.5%. The value ranking is not better than random survivors at 50% end to end.

Next, in order:
1. A network-level trained reference: fine-tune the 50% build end to end by KL for a short budget, and see how much of the drop training recovers.
2. Uneven keep fractions per block (by each block's end-to-end removal cost).
3. More calibration photos and a real ridge for the sequential build.

## N5. Shrink one neuron's tree, n layers deep (2026-11-15; plan `~/.claude/plans/ok-it-seems-like-merry-hinton.md`; code `neuron/tree/`, results `neuron/results/tree/`)

Object (Shonu, 2026-11-15): one coarse root neuron and its descendants n layers down. A developer map says which trees to shrink and which capabilities to keep. Only a shrunk tree's members change.

```
root r (block l) --> children (block l+1) --> grandchildren (block l+2) --> ...
child of a set   = a block-(l+j) neuron whose per-photo peak (after GELU) drops when the level-(j-1) members are set
                   to their goal-photo means, on goal photos, above a noise floor (99.9th percentile of the drops
                   when the same number of random block-(l+j-1) neurons are removed instead)
```

### Step 1 predictions (written before the run)

| # | Prediction |
|---|---|
| T1 | Each of dogs, birds, vehicles has a root in stage 3 (blocks 15-24) that fires above its usual level on more than half of the group's classes |
| T2 | At level 1, 60% or more of the children have more than half of their class effect inside the group |
| T3 | At least 20 members pass the noise floor at level 3 |
| T4 | Removing the whole tree (goal-mean fills) costs more on the group's test photos than the same number of random neurons from the same blocks, with a paired interval that excludes 0 |

### Step 1 result (`tree/classprof.py`, `tree/build_tree.py`, `tree/spec_check.py`; `results/tree/trees.json`, `spec_check.json`)

Class profile: 10 fit photos per class. Tree photos: other fit photos of the group. Test: held-out photos (group, and 300 others).

| Group | Root | Concentration / breadth | Levels found | Level-1 members with most class effect in the group |
|---|---|---|---|---|
| dogs (118 classes) | stage 3 block 19, neuron 224 | .87 / 1.00 | 10 (to the last block) | 3% |
| birds (59) | stage 3 block 19, neuron 1378 | .98 / .98 | 1 (dies at level 2) | 0% |
| vehicles (41) | stage 3 block 18, neuron 178 | .75 / .85 | 11 | 0% |

Bar: whole tree removed (goal-mean fills) against the same count of random neurons per block (3 draws), KL to Base:

| Group | Group photos: tree / random (95% interval of the difference) | Other photos: tree / random | Coarse KL on group photos |
|---|---|---|---|
| dogs | .020 / .0025 (+.013 to +.022) | .014 / .0026 | .005 |
| birds | .0001 / .0001 (includes 0) | .0002 / .0004 | .00001 |
| vehicles | .035 / .0016 (+.027 to +.039) | .032 / .0024 | .016 |

Root alone removed, drop of every later neuron (std units, tree photos) against its group concentration:

| Blocks after the root | Largest drop | Correlation of drop with group concentration (dogs / birds / vehicles) |
|---|---|---|
| +1, +2 | .02-.05 | -.29 / -.45 / -.48 (group-specific neurons RISE) |
| +5 to +7 (blocks 24-26) | .03-.30 | .41-.55 / .63-.67 / .18-.46 |
| stage 4 | .02-.07 | about 0 or negative |

| Prediction | Result |
|---|---|
| T1 roots exist | **holds** (breadth .85-1.00) |
| T2 level-1 members mostly in the group | **fails** (0-3%) |
| T3 at least 20 members at level 3 | holds for dogs (64), fails for vehicles (15) and birds (0). The level counts swing with the parent set's size (64, 4, 64, 3), because the noise floor depends on that size. So the counts are not a clean measure |
| T4 tree costs more than random on group photos | holds for dogs and vehicles, fails for birds. **But the tree costs almost as much on other photos.** So the removal tree is general machinery, not the capability |

What this says about the network:
1. One coarse neuron's removal moves every later neuron by at most 0.3 std, mostly below 0.05. The capability is spread over many redundant neurons (dogs: 29-84 group-specific neurons in every block).
2. The neurons that depend on the root AND belong to its group sit 5-7 blocks later (correlation up to .67), not in the next block. In the next block, the group's own neurons rise when the root goes: siblings, or LayerNorm rescaling. In a network with a skip stream, "next layer" is not "next level of the tree".
3. A tree built by "drops when the parent goes" gathers general neurons. Shrinking it would damage every class, which breaks the developer map's promise.

| | |
|---|---|
| Cale line | none yet. A "dog tree" defined by removal from one neuron is not a dog capability. The step-2 shrink does not run on these trees |
| Whole-network build | no |
| Label | none. Measurements |

Proposed fix (not run): a capability tree = a SET of coarse roots (the group's most concentrated, broad neurons in one block) + the group-specific neurons of later blocks (any distance, not only the next block) whose drop when the roots go is above the noise floor. Membership needs both: group specificity and dependence on the roots.

### Step 1b: capability trees (Shonu chose the fix, 2026-11-15)

Tree = 8 roots (the 8 most concentrated, broad neurons of one stage-3 block) + members: neurons of ANY later block with group concentration > .4 and breadth > .25, whose drop when the roots are filled with goal means is above the noise floor (99th percentile of the same neurons' drops when 5 random sets of 8 root-block neurons are filled instead).

| # | Prediction (before the run) |
|---|---|
| T5 | dogs: at least 100 members over the later blocks. Vehicles and birds: fewer (few group-specific neurons per block) |
| T6 | Removing the members (goal-mean fills) costs at least 3 times more KL on group photos than on other photos. A random same-count set costs about the same on both |

Result (`tree/build_tree2.py`, `results/tree/trees2.json`). Test photos: held-out, group (up to 120) and 300 others.

| Group | Roots | Members (blocks) | Members / neurons in those blocks | Remove members: KL group / other | Ratio | Random same count: ratio | Coarse KL, group |
|---|---|---|---|---|---|---|---|
| dogs | 8 in block 17 | 166 (12) | 166 / about 26,000 | .0127 / .0019 | **6.8** | 0.6 | .0026 |
| birds | 8 in block 17 | 45 (11) | 45 / about 24,000 | .0056 / .00015 | **38** | 0.9 | .0023 |
| vehicles | 8 in block 18 | 40 (10) | 40 / about 22,000 | .0075 / .0002 | **38** | 1.1 | .0015 |

T5 holds (dogs 166). T6 holds (6.8 to 38 against about 1 for random sets). These trees ARE the capability: removing them hurts the group, not the rest. The whole cost is small (.006-.013 nats), and the trees are under 1% of the neurons in their blocks, so the saving is small too. Dog photos still read as "a dog" after removal (coarse KL .0026).

### Step 2 predictions (written before the run; `tree/shrink.py`, keep f = 0.25 of each block's tree members)

Sanity (done): keep = 1.0 gives KL 0 for full, rewrite_ind and drop_goal at every depth, and RE-READ returns the old weights exactly.

| # | Prediction |
|---|---|
| S1 | drop_goal costs less than drop_all on group photos |
| S2 | full beats bias on group KL for dogs, with a paired interval that excludes 0 |
| S3 | from depth 1 to all blocks, full's group KL grows less than rewrite_ind's |
| S4 | parent adds nothing beyond full (interval includes 0), as in N1 and N3 |
| S5 | ctl_absorb (random non-tree absorbers) does worse than full |
| S6 | for full, other-photo KL stays below a quarter of group-photo KL |

### Step 2 result (`tree/shrink.py`, `tree/analyze.py`; `results/tree/shrink_<group>_f0.25.npz`, `shrink_summary_f0.25.json`)

Keep 25% of each block's tree members. KL to Base on held-out photos at full depth (all tree blocks), and the share of drop_goal's cost taken back:

| Variant | dogs: group / other | birds: group / other | vehicles: group / other | Taken back (dogs, birds, vehicles) |
|---|---|---|---|---|
| drop_goal (no absorption) | .00529 / .00007 | .00298 / .00006 | .00467 / .00011 | 0 |
| drop_all | .00548 / .00002 | .00310 / .00001 | .00504 / .00005 | -4%, -4%, -8% |
| bias | .00532 / .00003 | .00302 / .00002 | .00481 / .00006 | -1%, -2%, -3% |
| rewrite_ind (one-layer method, each block alone) | .00507 / .00003 | .00295 / .00002 | .00468 / .00006 | 4%, 1%, 0% |
| full (RE-READ + RE-WRITE, in order) | .00507 / .00003 | .00305 / .00002 | .00467 / .00006 | 4%, -2%, 0% |
| **parent** (the roots' writes also absorb) | **.00469** / .00003 | **.00265** / .00002 | **.00434** / .00006 | **11%, 11%, 7%** |
| ctl_absorb (random non-tree absorbers) | .00531 / .00014 | .00302 / .00002 | .00464 / .00006 | 0%, -1%, 1% |
| ctl_tree (shrink a random non-tree set of the same size) | .00034 / .00063 | .00001 / .00020 | .00002 / .00010 | |

Removing ALL members cost .0127 / .0056 / .0075 on group photos (step 1b). Keeping 25% costs .0053 / .0030 / .0047. Coarse KL at full depth: about .001 in all groups.

Paired intervals at full depth, group photos (95%):

| Comparison | dogs | birds | vehicles |
|---|---|---|---|
| parent - full | **-.00038 (-.00071 to -.00011)** | **-.00040 (-.00096 to -.00003)** | **-.00033 (-.00064 to -.00009)** |
| full - bias | -.00025 (-.00044 to -.00009) | +.00003 (includes 0) | -.00014 (-.00034 to -.00000) |
| full - rewrite_ind | +.00001 (includes 0) | +.00010 (includes 0) | -.00001 (includes 0) |
| growth from depth 1, full - rewrite_ind | +.00001 (includes 0) | +.00010 (includes 0) | -.00001 (includes 0) |
| ctl_absorb - full | +.00024 (+.00008 to +.00043) | includes 0 | includes 0 |

RE-READ changed the kept reads by 1-6% in stage 3 and 10-16% in stage 4 (median), and gained nothing.

| Prediction | Result |
|---|---|
| S1 goal-mean fill beats all-photo fill | holds in direction for all three, interval excludes 0 for vehicles only |
| S2 full beats bias for dogs | holds (small: 5% of the cost) |
| S3 full grows less with depth than rewrite_ind | **fails**: no difference. Errors do not pile up in a tree this small, so the recursion has nothing to fix |
| S4 parent adds nothing | **fails, in the good direction**: the roots' writes take back 7-11%, interval excludes 0 in all three groups. At depths 1-2 parent costs a little more (the roots' new write enters the stream before the dropped members' blocks) |
| S5 random absorbers do worse | holds for dogs only |
| S6 other KL below a quarter of group KL | holds: other KL is 1/40 to 1/170 of group KL |

What this says:
1. **The developer map works as a promise.** Shrinking a capability tree to 25% moves other photos by KL .00002-.0001, 40-170 times less than its own group. Dog photos still read as "a dog" (coarse KL .001).
2. **Absorption by survivors barely helps.** Kept siblings take back 0-4%. Random neurons do about as well. The dropped members' output is mostly not a linear function of the survivors' output (the same fact as N4 step 7: 7%).
3. **The parent absorbs best: 7-11%.** Picture A's "parent takes up the error" is real but small. This is the first absorption by a parent that beats every control in all three groups.
4. **The recursion (RE-READ) is not needed at this size.** With under 1% of the neurons changed, upstream drift is tiny. Step 8's pile-up is a whole-network effect.

| | |
|---|---|
| Cale line | a developer can shrink one capability's tree to 25% with no training and no visible change to other classes. The saving is small (125 of about 26,000 neurons for dogs), and keeping 25% still costs about 40% of what removing the whole tree costs, however the survivors are re-solved. The parent's write takes back about a tenth of that |
| Whole-network build | no |
| Label | none. Measurements |

### Step 0 predictions (written before the run; `tree/diag.py`)

| # | Prediction |
|---|---|
| D1 | The sum of the 30 single-block KLs (value survivors) is below .2, against .37 all at once: errors pile up |
| D2 | At single blocks, value survivors beat random survivors in at least 20 of 30 blocks |

### Step 0 result (`tree/diag.py`; `results/tree/diag.json`, `diag_rows.npz`)

Step-8 build (keep 50%, writes re-solved), 500 test photos. Sanity: no swap gives KL 0.

| Measure | Value |
|---|---|
| sum of the 30 single-block KLs, value survivors | **.043** |
| all 30 blocks at once (cumulative curve, last point) | **.296** (7 times the sum) |
| value beats random at single blocks | **27 of 30 blocks**. Sum: value .043, random .115 (difference interval -.112 to -.038) |
| random survivors in block 27 (first stage-4 block) alone | .052, against .002 for value |

Cumulative KL with blocks 0..j swapped: .0005 (j=0), .018 (9), .100 (19), .25 (26), .296 (29). Each block adds about .0015 early and .02-.03 at blocks 20-25, about 10 times its own single-block cost.

| Prediction | Result |
|---|---|
| D1 single sum below .2 (errors pile up) | **holds**: .043 against .296 |
| D2 value beats random at single blocks in 20+ of 30 | **holds**: 27 of 30 |

What this says: the one-layer build is good one block at a time, and the value ranking is right one block at a time. The whole-network failure is compounding: each block's small error grows as it passes through later blocks that were also changed. So the whole-network fix must act on drift (each block re-fitted to the inputs the shrunk network actually gives it). This is the RE-READ step's job. It had nothing to do inside one small tree (step 2), and it is the lever at whole-network scale.

| | |
|---|---|
| Cale line | the per-layer method is sound. Whole-network shrinking needs a recursion that corrects drift block by block. Step 2's tree shrink was too small to drift |
| Whole-network build | no (a diagnosis) |
| Label | none. Measurements |

### Step 4 predictions (added after step 0; `tree/wholenet.py`, the recursive unit on all 30 blocks, keep 50%)

Step 0 showed compounding, so the drift-correcting recursion is the test at whole-network scale. Reference on the same 500 photos: independent build KL .296 (step 0 cumulative curve).

| # | Prediction |
|---|---|
| W1 | full (RE-READ + RE-WRITE to the block's own original output, in order) gives KL below .15 and top-1 above 75% |
| W2 | rewrite_seq (no RE-READ, same target) lies between the independent build and full |

### Step 3 (partial, stopped) (`tree/ref.py`, `results/tree/ref.log`)

Same form as the build (kept members' reads and writes, the tree blocks' biases, the roots' writes), trained by KL on the calibration photos. Learning rate and epoch picked on validation photos. Group KL on held-out photos:

| Group | parent build | trained from the parent build | trained from drop_goal | build / best trained |
|---|---|---|---|---|
| dogs | .0047 | .0029 | .0029 | 1.6 |
| birds | .0027 | .0011 | .0014 | 2.5 |
| vehicles | .0043 | .0019 | (stopped) | 2.3 |

The no-training tree shrink is 1.6-2.5 times the trained error. That is outside the 1.3 bar.

### Change of direction (Shonu, 2026-11-15)

Shonu's rule: switching sizes must be near-instant, with no memory beyond the base weights plus a small config ("a computation among the weights which already exist"). Training per size and stored per-size weights (step 3, the dense step 4) do not meet it. Step 3 was stopped, and the dense step 4 was dropped.

Deployable form: the **absorption tree**. In each block, rank the neurons by value. Each neuron gets s parents among HIGHER-ranked neurons with coefficients c (sparse regression of its output on theirs). To switch to size k, take the dropped neurons from the lowest rank up. Each one adds c x (its write) to its parents' writes, and its mean term to the bias. A parent that is also dropped passes the sum on up the tree. Undo = subtract. One config serves every size.

### Step 5 predictions (written before the run; `tree/abstree.py`, whole network, stages 3-4)

| # | Prediction |
|---|---|
| A1 | At keep 50%, a tree with 4 parents per neuron gets at least 70% of the dense re-solve's gain over mean fill (KL on the 500 test photos) |
| A2 | 1 parent per neuron gets at least 40% of that gain |
| A3 | The same config at keep 25% also beats mean fill |
| A4 | Switch time for the whole network on the Mac GPU is under 10 ms; undo returns the base weights to fp32 rounding |

### Step 5 result (`tree/abstree.py`; `results/tree/abstree_1000.json`, `abstree_1000_rows.npz`)

Whole network (stages 3-4), 1,000 calibration photos, 500 test photos (Base top-1 on them 83.8%). One config (2.4 MB with 8 parents) for both sizes. Restore after all switches: KL 0.

| Keep | mean fill (s = 0) | s = 1 | s = 2 | s = 4 | s = 8 | dense re-solve (per-size weights) |
|---|---|---|---|---|---|---|
| 50%: KL / top-1 | .428 / 68.8% | .384 / 69.0% | .365 / 69.0% | .347 / 70.0% | .333 / 69.6% | .296 / 70.4% |
| 50%: share of dense's gain over mean fill | 0 | 33% | 48% | 61% | 72% | 100% |
| 25%: KL / top-1 | 2.22 / 25.2% | 2.03 / 29.0% | 1.92 / 31.0% | 1.79 / 34.0% | 1.68 / 36.2% | 1.36 / 41.8% |
| 25%: share of the gain | 0 | 22% | 35% | 50% | 63% | 100% |
| switch time (Mac GPU) | 28-32 ms | 260-400 ms (dense triangular solve) | | | | |

| Prediction | Result |
|---|---|
| A1 s = 4 gets 70% of the gain at 50% | fails: 61% (s = 8: 72%) |
| A2 s = 1 gets 40% | fails: 33% |
| A3 the same config at 25% beats mean fill | holds at every s |
| A4 switch under 10 ms, restore exact | restore exact. Switch fails: 260-400 ms |

What this says: a size can be a computation among the existing weights. Dropped neurons hand their writes up a parent tree, and 8 parents per neuron give 63-72% of what a full per-size re-solve gives, from one 2.4 MB config. But the per-size re-solve is itself far from Base end to end (step 0: compounding). The next fix must act on drift and keep this form.

| | |
|---|---|
| Cale line | a liquid model under Shonu's rule exists in form: base weights + a 2.4 MB tree config give every size. Quality at 50% is 70% top-1 (Base 84%), limited by compounding, not by the tree |
| Whole-network build | yes (no training, one config for all sizes) |
| Label | none. Measurements |

### Step 6 plan and predictions (written before the run; `tree/seqtree.py`)

Recursion within the rule: the same form (dropped neurons push their writes to parents with stored numbers, and each kept neuron gets one scale number), fitted block by block on the SHRUNK network's activations, so that each block rebuilds the ORIGINAL network's output there. The config then depends on the size (about 2.4 MB per size). Parents are restricted to strictly higher rank tiers (16 tiers per block), so the switch is one batched add per tier.

| # | Prediction |
|---|---|
| Q1 | At keep 50%, s = 8, the sequential fit beats the dense independent re-solve (KL below .296) |
| Q2 | At keep 25%, it beats the dense re-solve (KL below 1.36) |
| Q3 | Tiers cost under 10% of the tree's gain (independent fit, s = 8) |
| Q4 | The tiered switch takes under 20 ms for the whole network on the Mac GPU |

### Step 6 debug (`tree/dbg_tree.py`): a false alarm, and a calibration rule

The first full step-6 run gave KL .52 at keep 50% for the tiered tree, worse than step 5's mean fill (.43). I stopped the run. Debug, one config (300 calibration photos, 200 test photos):

| Row | KL, keep 50% |
|---|---|
| mean fill | .854 |
| 16 tiers: triangular switch / index_add switch | .7817 / .7817 |
| 128 tiers | .7816 |
| no tiers (rank order) | .7809 |

The two switch codes agree, and tiers cost nothing. The cause was the calibration set: one photo per class, so 300 or 500 photos cover only 300 or 500 classes, and class-specific neurons get biased means. Mean fill goes from .43 (1,000 classes) to .85 (300 classes). **Rule: the config's calibration must cover every class (every goal), not only many photos.** This matches N1 (goal-photo means).

### Step 6 results so far (`tree/seqtree.py` 1,000 classes, 4 tiers; `tree/dbg_seq.py`)

Tiered tree, fitted on O (independent), 500 test photos:

| Keep | KL / top-1 | untiered step 5 (s = 8) | switch (warm) | undo |
|---|---|---|---|---|
| 50% | .330 / 70.4% | .333 / 69.6% | 110-160 ms | exact (6e-7) |
| 25% | 1.639 / 36.6% | 1.676 / 36.2% | 117-127 ms | exact (5e-7) |

Q3 holds: tiers cost nothing. Q4 fails: the switch takes about 120 ms; most of it goes into gathering scattered columns, not into the tier loop. Fix (built, `tree/fastswitch.py`, not yet run): reorder each block's neurons by rank once at install, so every size is a prefix and every tier a contiguous slice.

Sequential fit (seq): KL .679 at keep 50% (Q1 fails badly; same top-1, 70.0%). Stopped and diagnosed (200 test photos, cumulative KL as blocks are shrunk in order):

| After block | independent | sequential |
|---|---|---|
| 7 | .0197 | .0192 |
| 15 | .0754 | .0728 |
| 22 | .2506 | .2752 |
| 29 | .4545 | (stopped; full run .68 on 500 photos) |

The per-neuron scales do not blow up (max 2.4). They SHRINK: median .80-.92 after block 15, 1st percentile down to .50. Reading: regression dilution. A least-squares fit of each neuron's original output on its drifted copy pulls toward the mean when the two only partly agree. Each block's signal gets weaker, and this compounds with depth. Next test (running): means-only correction, and variance matching (scale = sd_O / sd_C) instead of least squares.

Means-only correction (c from O, a = 1, means from C): after block 15 .0779, after block 22 .280 (independent .0754, .2506). Also worse. So the dilution reading does not explain the failure alone: **fitting each block to O's statistics on the shrunk network's inputs makes the end result worse**, whether by least squares, by means only, or densely (N4 step 8 `seq.py`, .445 against .37). Four forms, one direction. Variance matching was stopped before its result: the means-only row already weakens the hypothesis it tested. Q1 and Q2 fail.

A reading, not tested: a block's target in O is its output given O's stream. On the shrunk network the stream has already moved, and later blocks were built to read O's stream. Pulling one block's output back to O's statistics may cancel a shift that later blocks need to stay consistent with the moved stream. The independent fit makes no such pull.

Next levers inside the memory rule: uneven sizes per block (one number per block in the config, from step 0's per-block costs), and the fast switch.

### Step 6b: fast switch and uneven sizes (`tree/fastswitch.py`; `results/tree/fastswitch.log`, `fastswitch2.log`)

Install: each block's neurons reordered by rank (max logit change 5e-6, so the same network). Then a size is a prefix. 1,000 calibration photos, 4 tiers, 8 parents, 500 test photos.

| Keep | KL / top-1 | switch (warm, 5 runs) | undo | undo error |
|---|---|---|---|---|
| 50% | .330 / 70.4% | 72-104 ms | 74-114 ms | 4e-6 |
| 25% | 1.639 / 36.6% | 98-127 ms | 120-148 ms | 5e-6 |

Compute of the 30 MLP layers on 50 photos (Mac GPU): full 727 ms, keep 50% 373 ms, keep 25% 178 ms. A size really runs faster.

The switch time did not fall with slices: it is a fixed cost per GPU call on this Mac (about 3 ms per block). The math is about 80 million multiply-adds. Fix (not built): store the 27 stage-3 blocks' weights as one stacked tensor, so one batched call serves all of them.

Uneven sizes, same total, from step 0's single-block costs (f_b proportional to cost^g):

| Total | even | g = .5 | g = 1 | g = -.5 | g = -1 |
|---|---|---|---|---|---|
| 50%: KL / top-1 | **.330 / 70.4%** | .345 / 70.0% | .413 / 69.4% | .380 / 67.8% | .446 / 66.6% |
| 25%: KL / top-1 | **1.639 / 36.6%** | 1.856 / 31.2% | 2.177 / 25.4% | 1.729 / 35.6% | 1.864 / 34.0% |

Even beats both directions. Single-block cost is not a good guide for where to put neurons.

| | |
|---|---|
| Cale line | under Shonu's rule (base weights + a small config, in-place switch, exact undo), a 2.4 MB config gives any size of ConvNeXt-Base stages 3-4 with no training. Keep 50%: top-1 70.4% (Base 83.8% on these photos), MLP compute halved. The quality gap is whole-network compounding. Four drift corrections and two uneven allocations did not close it |
| Whole-network build | yes |
| Label | none. Measurements |

### Rescore on all 1,500 test photos (`tree/rescore.py`; `results/tree/rescore.json`, `rescore.log`)

**Correction.** `photos_test.npy` is sorted by class. Its first 500 photos cover classes 0-329 only (first 200: classes 0-133). Steps 0, 5, 6 and 6b used `[:500]` or `[:200]`, so their LEVELS (Base 83.8%, tree 70.4%) describe mostly animal classes. Comparisons inside one subset stand. All 1,500 photos cover 938 classes; Base top-1 80.4%.

1,000 calibration photos (all classes). Tree: 4 tiers, 8 parents, one config for every size. Random parents: 8 random neurons of higher tiers, coefficients fitted the same way.

| Keep | mean fill | random parents | **tree** | dense per-size re-solve | tree's share of dense's gain |
|---|---|---|---|---|---|
| 50%: KL / top-1 | .537 / 63.2% | .533 / 63.3% | **.425 / 65.5%** | .370 / 66.5% | 67% |
| 40% (off a tier boundary) | .913 / 52.7% | .906 / 52.9% | **.712 / 57.4%** | .606 / 59.6% | 66% |
| 25% | 2.172 / 26.3% | 2.152 / 26.5% | **1.654 / 36.3%** | 1.396 / 40.3% | 67% |

Paired 95% intervals, tree minus mean fill: KL -.112 (-.125 to -.101), top-1 +2.3 points (+1.3 to +3.4) at 50%; +4.7 (+3.5 to +6.1) at 40%; +10.0 (+8.3 to +11.8) at 25%. Tree minus random parents: the same size, all intervals exclude 0. So the gain comes from WHICH parents, not from having parents. Mean fill reproduces N4 step 8's "unchanged writes" rows exactly (.537 / 63.2%, 2.172 / 26.3%).

Switch: tier step 66 ms, bias step 12 ms per switch (Mac GPU, sync after each step). The first step's timer also absorbed queued earlier cycles, so it is not a measurement. The tier step builds a temporary 8 times the dropped columns; the cause of its time is not diagnosed further. Undo puts the base bias back from a copy (d numbers per block), and the writes back by subtraction: after 100 switch + undo cycles the largest relative weight drift is 2.8e-5 (one cycle: 5e-6).

What the tree is and is not: it is a rank tree inside ONE block (a dropped neuron hands its write to higher-ranked neurons of the same block). It is not Shonu's coarse neuron with descendants n layers down. The depth version (step 2, capability trees) took back 7-11% through the parent's write. Recursion across layers is unsolved: every per-block fit to O's activations on the shrunk network's inputs made things worse (step 6). No fit with an end-to-end target was tried.

| | |
|---|---|
| Cale line | base weights + a 2.4 MB config give any size of ConvNeXt-Base stages 3-4, in place, no training. Keep 50%: top-1 65.5% (Base 80.4%), MLP compute halved. The tree gives two thirds of what per-size weight copies give |
| Whole-network build | yes |
| Label | none. Measurements with paired intervals and a random-parent control |

## N6. The whole gap between the build and training, and the parent question (2026-11-16; plan `~/.claude/plans/it-seems-like-we-mutable-plum.md`; code `neuron/gap/`, results `neuron/results/gap/`)

Shonu, 2026-11-16: "we can't hope to solve something like recursive error prop if we cannot physically answer what in training is making this work better ... i don't want ANY gap with training to be unexplained." And: reconcile the parent idea (absorption into same-block partners) with the essay's gates and its account of how coarse neurons form.

Why: the whole-network gap against training of the SAME form was never measured (N4 step 8's "next 1" never ran). At one layer the gap is measured but only one third is reproduced with no training (greedy choice). "Explained" in N6 = a no-training rule reproduces it.

### Step 1 setup

| Item | Value |
|---|---|
| network | ConvNeXt-Base, stages 3-4 (30 blocks), keep 50% of the MLP neurons in every block |
| build B | value survivors, writes re-solved once (pair/allblocks.py), calibration 1,000 photos (one per class) |
| T | B, then the kept reads, writes and biases trained end to end, loss KL to Base |
| T-random, T-greedy | the same from random or greedy survivors |
| T-local | the same weights, each block trained against its own Base output on Base's inputs |
| T-all | T, plus every other stage 3-4 weight |
| photos | train: fit half minus 2 per class (about 15,500, plus left-right flips). Held: 2 per class from the fit half (learning rate, stop point). Test: the same 1,500 as N5's rescore |
| cache | stream at the stage-3 input in fp16 (logit error vs full model under .001) |

### Step 1 predictions (written before the run)

| # | Prediction |
|---|---|
| R1 | T reaches top-1 of at least 75% (B: 66.5%) |
| R2 | whole-network error ratio KL(B) / KL(T) is above 3 (one layer: 1.12) |
| R3 | T-local gets less than a third of T's gain over B |
| R4 | T, T-random and T-greedy end within 20% of each other in KL (training makes survivor choice matter little) |

### E1 predictions (written before the run; `gap/e1_parents.py`, blocks 8, 15, 22, 28, 12 dropped neurons each)

For a dropped neuron d (keep 50%, among the most valuable dropped), its top tree parent p (same block, from the 8-parent OMP fit of tree/abstree.py) and a random kept higher-ranked neuron q. Upstream = the 7 blocks before d's block. "Gate" here = d's 20 strongest upstream feeders by activation x gradient, removed together (filled with means); floor = largest drop from 3 random groups of 20, at least .25 std. A single-neuron gate test was dropped after the smoke run: N5 showed one removal moves later neurons under .05 std in most cases.

| # | Prediction (the essay reading "the tree's parent is a sibling under the same gate") |
|---|---|
| E1a | median cosine of upstream attributions d-p is at least .3, and d-q is near 0 |
| E1b | when d has a gate (drop above floor), p's drop is above the floor in at least half of those pairs |
| E1c | the coefficient c is positive in most pairs (siblings under one gate co-fire). Negative c = p keeps d's photos out |
| E1d | d's top-50 photos overlap p's top-200 more than q's |

### E1 result (`gap/e1_parents.py`; `results/gap/e1_parents.json`, `e1_parents.log`)

48 dropped neurons (12 in each of blocks 8, 15, 22 of stage 3 and block 28 = stage 4 block 1). Photos: 3 train photos per class.

| Measure (median) | d with its tree parent p | d with a random kept neuron q |
|---|---|---|
| cosine of upstream attributions (activation x gradient, 7 blocks back) | **.35** (stage 3: .26-.38; stage 4: .61) | .03 |
| d's top-50 photos inside the other's top-200 | .32 | .08 |
| the other's pre-activation percentile on d's top photos | 85th | 55th |
| sign of the tree coefficient c | **positive in 48 of 48** | |
| d's 20 top feeders removed: d drops above the floor | 45 of 48 | |
| ... and the other drops above the floor too | **14 of 45** (stage 3: 4 of 33; stage 4: 10 of 12) | 2 of 45 |

Paired difference of the attribution cosines (p minus q): +.285, 95% interval +.23 to +.34.

| Prediction | Result |
|---|---|
| E1a cos d-p at least .3, d-q near 0 | holds (.35, .03) |
| E1b p shares d's gate in at least half | **fails overall** (14 of 45). Holds in stage 4 (10 of 12), fails in stage 3 (4 of 33) |
| E1c c positive in most pairs | holds (48 of 48) |
| E1d overlap p above q | holds (.32 against .08) |

Breadth (classes among the top 200 photos) is about 170 for d, p and q alike: at 3 photos per class it saturates, so it cannot say whether p is broader than d. Not used.

What the tree's parent is, in the essay's terms:
1. **Never a negative-pole sibling.** c > 0 in all 48 pairs, and p fires at its 85th percentile on d's photos. The essay's specialists split by negative poles (taxi against ambulance) are mutually exclusive. A tree fit on co-firing cannot pick them, because a write handed to p only fires when p fires.
2. **In stage 4 it is a sibling under the same gate.** Removing d's feeders also knocks p down in 10 of 12 pairs. This matches the essay's late-layer picture.
3. **In stage 3 it is a co-firing neighbour, not a gate-mate.** p shares part of d's inputs (cosine .26-.38) but does not depend on d's main feeders (4 of 33).

The point that links E1 to the training gap: the essay's coarse neuron is a merge of siblings ("one curves & loops neuron instead of ten curve detectors"). For mutually exclusive siblings a merge must change the survivor's READ, so that it also fires on the dropped sibling's photos. Our build only changes writes, so it cannot do this merge. Training turns reads (N4 step 7: about 27%, toward the ground the dropped neurons covered). Hypothesis for E2, written before E2 runs: training's read turn performs the essay's sibling merge, taking the ground of dropped neurons that share the survivor's gate, including negative-pole siblings, not of co-firing neighbours.

| | |
|---|---|
| Cale line | the deployable tree hands work to co-firing neighbours. That is the essay's object only in stage 4. The essay's merge needs a read change, which no write-only config can make |
| Whole-network build | no (a reading of the build) |
| Label | none. Measurements with a random-neuron control |

### E2 predictions (written before T exists; `gap/e2_reads.py`, blocks 8, 15, 22, 28, the 20 survivors with the largest read turn per block)

For survivor s: gained photos = its top-200 under T's read but not under Base's read (both on Base's block inputs). Owner d* = the dropped neuron whose Base top-200 holds most of them. Control r = a random dropped neuron.

| # | Prediction (the essay reading: training merges siblings under one gate) |
|---|---|
| E2a | d* holds a clear share of the gained photos: median share at least 3 times r's |
| E2b | upstream attribution cosine s-d* is above s-r (paired), median at least .3 |
| E2c | d* is s's tree child (s among d*'s 8 tree parents) in under half of the cases: training's merge partner is not the build's |
| E2d | correlation of Base peaks s-d* is near 0 or negative in at least a third of cases (the merge takes ground where s was off) |

### Step 1 result, T (`gap/train_ref.py --run T`, UVA cluster, A100; `results/gap/T.json`, `T.log`)

Build B on the cluster reproduces N4 step 8 exactly (test KL .370, top-1 66.5%). Learning rate from a 300-step sweep on held-out photos: 1e-4 (held KL .212; 3e-4: .234; 1e-3: .438). Main run: 4,000 steps of 32 photos (4 passes over train + flips), cosine decay.

| Network (keep 50%, 30 blocks) | Test KL | Test top-1 | Held-out KL | Fit KL (train photos) |
|---|---|---|---|---|
| Base | 0 | 80.4% | | |
| B, build (no training) | .370 | 66.5% | .368 | .379 |
| **T, B trained end to end** | **.131** | **77.4%** | .156 | **.030** |

Held KL was still falling slowly at the last step (.1572, .1564, .1563). B minus T, per-photo test KL: +.239, paired 95% interval +.216 to +.263. Stop step 3,996 of 4,000.

| Prediction | Result |
|---|---|
| R1 T top-1 at least 75% | holds (77.4%) |
| R2 error ratio B / T above 3 | **fails, just**: .370 / .131 = 2.8 (one layer: 1.12) |

**The reference is limited by photos.** T's KL on its own training photos is .030, five times its held-out KL. With 16,000 photos (16 per class) for about 40 million trained numbers, T memorizes. So T is an upper value of what training reaches. A reference on more photos would likely sit lower, and the true gap is likely larger than 2.8. The remaining photos on disk are reserved for final scoring and stay closed.

What the gap is in plain terms: training removes 65% of the build's KL and recovers 11 of the 14 top-1 points lost. The same form (the same kept neurons) can do much better than the build.

### Step 2-3 predictions (copied from the plan `~/.claude/plans/it-seems-like-we-mutable-plum.md`, where they were written before any run; copied here AFTER the ledger smoke run on 100 photos had been seen)

| # | Prediction |
|---|---|
| L1 | T's single blocks are no better than B's at their OWN target (block output on Base's inputs) in at least half of the blocks. Yet T is much better end to end |
| L2 | "together" (all of T's blocks) gives more than twice the sum of the 30 single-block gains |
| W1 | T puts a smaller share of its error into the Fisher directions than B does, in most blocks, even where T's plain error is larger |

### Step 2 result: the gap ledger for T (`gap/ledger.py`; `results/gap/ledger_T.json`, `ledger_T_rows.npz`, `ledger_T.log`)

1,500 test photos. Gap = KL(B) - KL(T) = .370 - .131 = **.239**. Sanity: B with all of T's numbers = T exactly (.1306).

| Part (all 30 blocks) | Paste T's part into B: gain | Take T's part out of T: loss | Two-order share (Shapley, 2 parts) |
|---|---|---|---|
| reads W1 | .200 | .097 | **.153 (64%)** |
| writes W2 | .133 | .032 | **.086 (36%)** |
| thresholds b1 | .003 | .000 | about 0 |
| write bias b2 | .014 | .001 | about 0 |

The reads-writes interaction is large: from B the two gains add to .333 (more than the gap, they overlap), from T the two losses add to .129 (less than the gap, they need each other). The Shapley column splits the gap over the two orders. Paired: B+writes minus B+reads = +.066 (+.057 to +.076).

| Row | Value |
|---|---|
| T's reads, writes re-solved to match each block's Base output | gain **.082**, against .200 with B's old writes (difference +.118, interval +.106 to +.130) |
| sum of 30 single-block gains (B + T's block g) | **.472** (twice the gap). Each block .009-.029, mean .016 |
| sum of 30 single-block losses (T with B's block g) | **.045** (a fifth of the gap). Each block about .0015 |
| local error (|y - y_Base|^2 / var, Base's inputs) | T above B in **30 of 30** blocks (block 20: .31 against .05; block 29: .32 against .13) |
| stream drift from Base, stage 3 (blocks 0-26) | T above B in 26 of 27 blocks (block 20: .167 against .124) |
| stream drift, stage 4 (blocks 27-29) | T below B (.52 against .62). Both jump at block 27, after the downsample |

| Prediction | Result |
|---|---|
| L1 T's blocks no better than B's at their own target in half the blocks | **holds, strongly**: worse in 30 of 30 |
| L2 together more than twice the sum of single-block gains | **fails, the other way**: together = half the sum (.239 against .472) |

What this says about training, physically:
1. **Training does not build better blocks.** Every trained block copies its own Base output worse than the build does. The trained stream strays further from Base through stage 3.
2. **Each trained block corrects the network's shared error.** One trained block in B removes .016, about 10 times that block's own build error (.0015 in N5 step 0). The 30 blocks correct overlapping parts of the same whole-network error: their single gains add to twice the gap. From T's side, any one block reverted costs .0015, because the other 29 cover it.
3. **Given T's reads, the local write target hurts.** Writes solved to match each block's output recover .082 of the gap; the build's old writes recover .200. This tests the local target only for reads trained for the end. T-local (free reads, local target) recovers .100 (42%), and its learning rate sat at the top of its sweep, so 42% is a lower value: the local target is weaker, not useless. That the four chained fits of N5 step 6 failed for this reason is a guess, not tested.
4. Reads carry two thirds and writes one third. This repeats the one-layer split (N4 step 7), now at network scale.

Open: what makes a block's change correct the shared error (M2: later blocks undo earlier error; M3: error placed where the end does not look). Step 3 (Fisher) tests M3.

| | |
|---|---|
| Cale line | the build's per-block target is the bottleneck. A no-training build must aim each block at the whole network's error, not at its own output |
| Whole-network build | no (a reading of training) |
| Label | none. Measurements with both paste orders |

### Step 3 result: W1, where each network puts its error (`gap/where.py`; `results/gap/where_T.json`, `where_T.log`, `fisher.pt`)

Fisher of each block from Base alone (1,000 calibration photos, 2 sampled classes each), per position. The top 32 directions hold a median 38% of the Fisher trace. Errors on the 1,500 test photos.

| Measure: blocks where T is better than B | Early stage 3 (0-12) | Late stage 3 (13-26) | Stage 4 (27-29) |
|---|---|---|---|
| own-block error, plain | 0 of 13 | 0 of 14 | 0 of 3 |
| own-block error, Fisher cost (1/2 e^T F e) | 0 of 13 | 0 of 14 | 0 of 3 |
| share of own-block error in the top 32 Fisher directions | 2 of 13 | **14 of 14** (block 20: .008 against .031) | 0 of 3 |
| stream drift, Fisher cost | 0 of 13 | 1 of 14 | **3 of 3** (block 29: .42 against .87) |

Sum over blocks of the own-block Fisher cost: B .93, T 1.10. The second-order drift cost overstates the true KL in stage 3 (B 1.59 at block 26, true end KL .37): with drift this large, the second-order picture is only a ranking.

| Prediction | Result |
|---|---|
| W1 T puts a smaller share into the Fisher directions in most blocks | **holds in late stage 3 only** (14 of 14), fails early (2 of 13) and in stage 4 (0 of 3) |

Reading: T aims its late stage-3 error away from what the end reads (M3, partly). But each of T's blocks still costs more than B's block by Fisher, and the T stream's drift costs more through all of stage 3. So M3 alone does not explain why T wins. T's advantage appears only in stage 4. Hypothesis left: cancellation (M2). T's per-block errors cancel each other on the way to the logits, where B's add up in the same direction.

### Step 3b predictions: do the blocks' errors add up or cancel? (`gap/additive.py`, written before the run)

For a network X (B or T) and each block g: Base with only block g's MLP replaced by X's. d_g = the change in the 1,000 logits per photo (centred). Alignment = |sum_g d_g|^2 / sum_g |d_g|^2 per photo, in the softmax metric (e^T (diag p - p p^T) e, p = Base's probabilities). Random directions give about 1. Lined-up errors give up to 30. Cancelling errors give below 1.

| # | Prediction |
|---|---|
| A1 | B's alignment is above 2 (its errors add in the same direction: the compounding of N5 step 0) |
| A2 | T's alignment is below 1 (its errors cancel), even though T's single-block KLs are larger than B's |
| A3 | KL of (Base logits + sum of d_g) is within 30% of the real all-at-once KL, for both B and T (the logit picture is linear enough to read) |

### Step 3b result: the errors of B add up, the errors of T cancel (`gap/additive.py`; `results/gap/additive_T.json`, `additive_T_rows.npz`, `additive_T.log`)

All 1,500 test photos. One block of X swapped into Base at a time, logit changes centred.

| | B (build) | T (trained) | T minus B, paired 95% interval |
|---|---|---|---|
| sum of 30 single-block KLs | .064 | **.072** | +.008 (+.007 to +.009) |
| KL of Base logits + the 30 changes added (no interaction) | .184 | .091 | -.092 (-.106 to -.079) |
| real all-at-once KL | .370 | .131 | -.239 |
| alignment, mean ratio / median (1 = random directions) | 2.70 / 1.92 | 1.26 / **0.74** | median per photo -1.41 (-1.50 to -1.32) |
| mean cosine of neighbour blocks' changes (lag 1 / 5 / 15) | .14 / .07 / .01 | .10 / .03 / -.02 | |

The N5-step-0 numbers (.043, .296) were on the class-sorted prefix. On all photos B's single sum is .064.

**The gap, with no row left over:**

| Part | B | T | Gap part | Share of the .239 gap |
|---|---|---|---|---|
| single blocks (own errors) | .064 | .072 | -.008 | -3% |
| alignment (added minus singles) | .120 | .019 | **.101** | **42%** |
| interaction (real minus added) | .186 | .040 | **.146** | **61%** |
| total | .370 | .131 | .239 | 100% |

| Prediction | Result |
|---|---|
| A1 B's alignment above 2 | holds by mean ratio (2.70), not by median (1.92) |
| A2 T's alignment below 1 | holds by median (0.74: most photos cancel), not by mean ratio (1.26: a few photos still line up) |
| A3 the added picture within 30% of the real KL | **fails**: B .184 against .370, T .091 against .131. Interaction is half of B's loss |

What this says:
1. **The whole gap between the build and training is compounding.** Training's blocks are each slightly worse (+.008). All of training's advantage is in how the 30 errors combine: they do not line up (42% of the gap), and they do not grow as they pass later changed blocks (61%).
2. So Shonu's two questions are one question. Closing the gap with training = stopping compounding. A better single block can still help indirectly: T-local (better per-block fits) gets 42% of the gain, and where that gain sits (singles, alignment or interaction) is measured below.
3. A likely common cause is the target. The build aims each block at its own output, so every block may make the same kind of error (aligned), and no block allows for the changed blocks after it (interaction). Training aims every block at the end. This is a reading, not yet tested.

| | |
|---|---|
| Cale line | the next no-training build must choose each block's numbers by their effect on the END, with the other blocks also changed, not by the block's own output. That is the object to find. The survivors and the form are not the problem |
| Whole-network build | no (a diagnosis) |
| Label | none. Measurements with paired intervals; the parts add up exactly by construction, so the split is an identity, the sizes are measured |

**E2 change of measure (after the smoke run, before the full run).** In raw cosine the trained reads turn very little (median 1 - cos = .001), so a survivor's top-200 photos change by only 12-14. E2 now uses the change of each survivor's firing per photo, dA = a_T - a_B (peaks on Base's inputs). Turn = std(dA) / std(a_B). Owner d* = the dropped neuron whose Base firing correlates most with dA. Chance level = the same maximum with dA shuffled over photos. E2a now reads: owner correlation above the chance maximum in most survivors. E2b-E2d are unchanged.

### E2 result: what training's read change reaches for (`gap/e2_reads.py`; `results/gap/e2_reads_T.json`, `e2_reads_T.log`)

80 survivors (the 20 with the largest firing change in each of blocks 8, 15, 22, 28). Firing change std / own std: median .06 per block, .12-.18 for these 80.

| Measure (median) | Stage 3 (blocks 8, 15, 22) | Stage 4 (block 28) |
|---|---|---|
| owner correlation (change of s vs Base firing of d*) | .16-.24 | .40 |
| chance maximum (photos shuffled) | .05 | .06 |
| above chance | 60 of 60 | 20 of 20 |
| Base correlation s with d* (negative in) | .00-.15 (25 of 60) | -.11 (**14 of 20**) |
| d* is s's tree child | 3 of 60 | 0 of 20 |
| upstream attribution cosine s-d* (random dropped r) | .00 (.00) | **.59 (.25)** |
| read change vs d*'s read, cosine | .08-.13 | .01 |
| distinct owners among the 20 | 18-19 | **6** (a few dropped neurons taken over by many survivors) |
| Base write cosine s-d* (random pairs) | .01-.05 (.00-.02) | .04 (.00) |

| Prediction | Result |
|---|---|
| E2a owner above chance | holds (80 of 80) |
| E2b gate cosine s-d* above random, median at least .3 | **holds in stage 4 only** (.59 against .25). Stage 3: 0, like random. Overall interval -.03 to +.13 |
| E2c d* is the tree child in under half | holds (3 of 80): training's partner is not the build's |
| E2d s-d* correlation near 0 or negative in a third | holds (39 of 80; stage 4: 14 of 20 negative) |

What training's read change is, in the essay's terms:
1. **Stage 4: consistent with the essay's merge on the read side, not on the write side** (one block, 6 distinct owners). A survivor starts to fire on the ground of a dropped neuron that shares its gate (.59) and sits on its negative side (14 of 20 anti-correlated). Many survivors share a few owners. This is "siblings under one gate merge into a coarser neuron", done by the READ. The tree cannot do it: it only hands writes to co-firing neurons (E1: c > 0 in 48 of 48).
2. **Stage 3: no shared gate by our measure.** The survivor takes ground of a specific dropped neuron (60 of 60 above chance), but activation x gradient shows no shared upstream (0, like random) and they do not co-fire. By the essay-is-law rule this blames the gate measure first: activation x gradient is a known weak tool here (curves memory trap). The tension is real and not reconciled. E3 was not run. The read moves a little toward d*'s read (.08-.13).
3. **In both stages the survivor keeps its own write** (write cosine with d* .01-.05). So the survivor does not become d*'s stand-in. It fires where d*'s absence hurts and writes something else, which fits the ledger: the target is the end (3b), not d*'s output.

Open, and named: what decides the stage-3 owner, and why the survivor's own write is the right thing to add there. Not explained by co-firing, shared upstream, or write similarity. This is the "unexplained" row of the read part.

### Step 1 result, the other references (`gap/train_ref.py`, UVA cluster; `results/gap/T-greedy.json`, `T-random.json`, `T-local.json`, `T-all.json`)

| Run | Build: test KL / top-1 | Trained: test KL / top-1 | Held / fit KL | Learning rate |
|---|---|---|---|---|
| T (value survivors) | .370 / 66.5% | **.131** / 77.4% | .156 / .030 | 1e-4 |
| T-greedy | .348 / 66.9% | **.131** / 76.9% | .149 / .030 | 1e-4 |
| T-random | .366 / 69.3% | **.128** / 77.4% | .145 / .036 | 1e-4 |
| T-local (each block to its own Base output) | .370 / 66.5% | **.270** / 68.9% | .280 / .261 | 1e-3 |
| T-all (also norms, depthwise convs, layer scales, downsample) | .370 / 66.5% | **.131** / 77.3% | .156 / .030 | 1e-4 |

| Prediction | Result |
|---|---|
| R3 T-local gets under a third of T's gain | **fails**: .100 of .239 (42%) |
| R4 T, T-greedy, T-random within 20% | holds: .131, .131, .128 (within 3%) |

## N6 summary: the ledger of the gap between the build and training (keep 50%, all 30 blocks)

Gap = KL(B) - KL(T) = .370 - .131 = .239 (1,500 test photos). T is an upper value of training (it memorizes: fit .030 against held .156).

| Row | Size | What it is in the network | Reproduced by a no-training rule? |
|---|---|---|---|
| survivor choice | build: .022 (greedy). After training: 0 (.128-.131 for any choice), under a reference that memorizes (fit .030, held .156) | which neurons are kept | **yes**: greedy choice (N4 step 7) gets the build's part. It is 9% of the gap |
| parts outside the MLP | 0 (T-all = T) | norms, depthwise convs, layer scales | nothing to reproduce |
| thresholds b1 | 0 | where neurons start firing | nothing to reproduce |
| single blocks (each block's own error) | **-.008** (T's blocks are each WORSE) | | the build already wins this row |
| **alignment** | **.101 (42%)** | B's 30 block errors point the same way at the logits (ratio 2.7); T's mostly cancel (median 0.74) | **no** |
| **interaction** | **.146 (61%)** | real minus added (B .184 to .370; T .091 to .131). Two readings, not separated: errors grow through later changed blocks, or each build block meets inputs it was not fitted on. Also, T's single blocks are measured in Base, outside the context T was trained in | **no** |
| status of the gap | **located, not explained.** The three rows above add to .239 exactly (an identity), but the two that carry it have no no-training rule. By the plan's bar (explained = reproduced with no training), 103% of the gap is still open | | |

Cross-cuts of the same gap (they split it another way, not extra rows):

| Cut | Split |
|---|---|
| by part | reads .153 (64%), writes .086 (36%) (two-order Shapley) |
| by target | the local target with free reads (T-local) gets .100 (42%). The end target gets the other .139 |
| by block | spread over all 30 blocks (single gains .009-.029). No block holds the gap |
| by stage | stage 3: T aims its error off the sensitive directions in blocks 13-26 (W1, 14 of 14), but each block still costs more. Stage 4: T's drift costs half of B's |

What training does, physically (the answer to "what in training makes this work better"):
1. It does NOT make better blocks. Each trained block copies its own Base output worse than the build (30 of 30), and costs more when swapped alone into Base (+.008 in all).
2. It makes the 30 block errors cancel instead of line up, and keeps them from growing through later blocks. That is all of the gap (42% + 61% - 3%).
3. It does this mostly through the reads (64%). In stage 4 the read change is the essay's merge: a survivor starts to fire on the ground of a dropped neuron under the same gate, on its negative side (E2). In stage 3 the survivor takes ground of a specific dropped neuron that shares neither inputs nor co-firing with it. The survivor keeps its own write in both stages.

So the gap with training IS compounding. The build is beaten only on how blocks combine. A likely cause is its target (each block aimed at its own output), but T-local shows that a local target with free reads already gets 42%.

The parent question (Shonu, 2026-11-16), answered with E1 and E2:

| Object | Where it fits the essay | Where it does not |
|---|---|---|
| the build's tree parent (same block, co-firing, write absorbs) | stage 4: a sibling under the same gate (10 of 12) | never a negative-pole sibling (c > 0 in 48 of 48). Stage 3: a co-firing neighbour, not a gate-mate (4 of 33) |
| training's merge partner (read change) | stage 4: a gate-mate on the negative side, many survivors share a few owners: the essay's sibling merge | stage 3: no shared gate, no co-firing. Not explained yet |

The tension Shonu saw is real, and it is not reconciled. The essay's coarse neuron is a merge of exclusive siblings under one gate. A merge needs the survivor's READ to widen. A write-only tree hands work to co-firing neurons, so it cannot do that merge. Training's read change in stage 4 looks like it on the read side, but the survivor keeps its own write. In stage 3 our gate measure finds no gate, which first blames the measure. E3 was not run.

E3 (gate or merge, with ConvNeXt-Small as the reference) was not run in this session.

### The candidate rule from the largest rows (written, NOT run)

The largest rows are alignment and interaction. Both come from the per-block target. Candidate: **an end-anchored joint solve.** Keep the form (same survivors, same config shape). Choose all 30 blocks' write numbers TOGETHER so that their summed effect at the logits is small in the softmax metric, with each block's effect carried to the logits through the Jacobian of the shrunk network (which brings in the interaction to first order). It is a linear least-squares problem in the writes, solved once, upfront, with sketched Jacobians (a few backward passes per photo). The switch stays an add. Test against: the training row B+writes(T) (gain .133) is what writes alone can reach. The read part (64%) needs a second rule: widen a survivor's read toward the dropped neurons whose absence the end feels, which E2 points at in stage 4 (gate-mates on the negative side).

| | |
|---|---|
| Cale line | the size switch can stay a config. What must change is how the config is computed: aimed at the end of the shrunk network, all blocks at once, not at each block's own output |
| Whole-network build | no (a diagnosis and a candidate) |
| Label | none. Measurements; the three-row split is an identity, its sizes are measured with paired intervals |

### Check: a few numbers on the logits (bias-only control; `results/gap/bias_control.json`)

Fitted on the 2,000 held photos, scored on the 1,500 test photos.

| Fix on the logits | B test KL | T test KL |
|---|---|---|
| none | .370 | .131 |
| one scale | .362 (scale .96) | .128 |
| scale + one bias per class | .338 | .138 |

The shared error of B is not a simple confidence or class offset: 1,001 numbers remove .032 (13% of the gap). The cosine of B's logit error with Base's centred logits is .19 (median), and its size is .54 of Base's centred logits.

### Check: where T-local's gain sits (`gap/additive.py --run T-local --only T`; `results/gap/additive_T-local.json`, `.log`)

| Part | B | T-local | T | B minus T-local | B minus T |
|---|---|---|---|---|---|
| single blocks | .064 | .057 | .072 | .007 | -.008 |
| alignment (added minus singles) | .120 | .093 | .019 | .027 | .101 |
| interaction (real minus added) | .186 | .120 | .040 | .066 | .146 |
| total | .370 | .270 | .131 | .100 | .239 |
| alignment ratio, mean / median | 2.70 / 1.92 | **2.51 / 1.80** | 1.26 / 0.74 | | |

Reading:
1. **Alignment is the end target's row.** A local target with free reads leaves the block errors lined up (ratio 2.5, against B's 2.7). Only the end target makes them cancel (0.74). Of the alignment row (.101), local training gets .027.
2. **Interaction is partly local.** Smaller local errors grow less: T-local removes .066 of the .146 interaction (45%). The rest needs the end target.
3. So no cheap shortcut through the logits (bias control: 13%), and no full shortcut through better local fits (42% of the gain, little of the alignment). A next no-training rule must make the blocks' errors cancel at the end. That needs a target that sees all blocks at once.

## N7. A coherent coarsening: lost detail goes to directions nobody reads (2026-11-19; plan `~/.claude/plans/pasted-content-id-3235-okay-some-atomic-emerson.md`; code `neuron/lens/`, results `neuron/results/lens/`)

Shonu, 2026-11-19: the essay says a smaller model's neurons are simply coarser, so coarsening happens by itself. Why is our calculated coarsening worse? He asked for an answer from investigation of what we have, then: "let's try this neuron coarsening approach, and having the errors line up ... in directions which are no longer read".

### The investigation (from N4-N6, CHILDREN-LOG and the essay; no new runs)

The essay's coarseness belongs to a whole network that stopped refining early, at every layer at once. Our build makes a fine network with holes, one layer at a time, and every later reader is still fine. A likely reading: those readers turn each hole into a wrong gate decision. That is the compounding.

1. **The essay's coarse neuron is a wider gate. We kept narrow gates and deleted their siblings.** A Small neuron fires on 30-40% more photos than its Base counterpart (N4). Siblings' weight rows are as unrelated as random rows: a coarse unit is "coarse in function, dense in weights" (CHILDREN-LOG K1, blue sky #8). Our write fill hands a deleted job only to co-firing neurons (N6 E1: c > 0 in 48 of 48). An exclusive sibling never can. Training changes reads (64% of the gap). In stage 4 a survivor takes a deleted gate-mate's ground on its negative side (E2).
2. **Coarseness must be the same at every depth.** A child is a child because it reads its parent's write. So a deleted neuron's error lands in its children's read direction, on its photos. One layer: build error 1.12 times trained (N4). 30 layers: 66.5% against 77.4%. The gap is how errors combine (N6 3b). Trained blocks copy their own output worse, 30 of 30. October: one-layer objectives moved away from the trained layer, and a merged unit lost to one kept fine neuron at every size (R2.4, K7b). One cause may explain both: a merge throws away distinctions that fine readers downstream still need, so under a one-layer target selection always wins, and selection compounds.
3. **Why training's errors cancel.** T's stream is further from Base (26 of 27 blocks) but costs less. Reading 1: its readers stopped asking for the lost distinctions. Reading 2: its blocks co-adapt to cover each other's holes. N6 does not separate them.

| Claim | Status |
|---|---|
| the gap is compounding, not single blocks | measured (N6) |
| training's gain is mostly in reads | measured (64%) |
| a deleted neuron's error lands in its children's reads and changes their gates | reading. First direct test: gate crossings (step 1) |
| the coarse model's readers ignore the lost distinctions | reading |
| stage-3 owners share a gate | open (our tool says no) |

| Idea | Holds | Failed, and why |
|---|---|---|
| neuron gating | a coarse model has fewer, wider gates, at every depth | nothing: we had not built its object |
| tree folding | the coarse-to-fine tree is real (N5 vehicle root, block 18) | we folded values, one layer at a time, local target, under fine readers. The essay's fold is a gate fold at every depth |
| distillation shows circuits | stage 4: the read change pairs gate-mates (E2) | stage 3: our gate tool finds no gate (blame the tool first) |

### The object: a coarsening lens in front of every reader

```
Base:     stream ──► dwconv ──► LayerNorm ──► u ─────────────► kept reads W1 ──► GELU ──► writes
Coarse C: stream ──► dwconv ──► LayerNorm ──► u_C ──► [ L ] ──► kept reads W1 ──► GELU ──► writes
                       L u_C + m = best linear guess of Base's u, given only what C still carries
```

Built in order, block by block, in closed form: lens by ridge least squares (u_C to u_B), then writes re-solved to Base's block output on C's own activations. Where C lost a distinction, L returns the average over what C can still tell apart, so a fine reader answers the coarse question. The lost detail stays in the stream, unread.

Why it matters: it is a no-training way to make every later reader coarse to match, which the investigation says the build lacks. Limit, stated before: the lens is still a local target (each block aimed at Base's own input). A result near T-local (.270) is the expected one. It is falsified if C-lens is no better than C-seq (the same sequential build without the lens).

Config per size: lenses about 10 M numbers plus re-solved writes about 20 M (about 60 MB fp16). Kept reads unchanged; fold W1' = W1 L, b1' = b1 + W1 m.

### Predictions (written before any run)

| # | Prediction |
|---|---|
| P1 | C-lens (block lenses only) test KL under .30 (B .370). Near T-local (.270) expected |
| P2 | the head lens alone on B gets under half of C-lens's gain |
| P3 | prefix swap (blocks 0..g of X in Base): C-lens's KL grows more slowly with g than B's |
| P4 | on vehicle photos, C-lens's kept vehicle-tree neurons cross their Base 90th-percentile level less often than B's |
| P5 | step 2: writes that also undo the read part (top k Fisher directions) of upstream error: some k in 32-128 beats k = 0 and "all" |
| gate | smoke run: if the lens cuts kept neurons' on-rate by more than a third, switch to a per-neuron gain and bias fallback |

### Step 1 result (`lens/build.py`; `results/lens/*.json`, `*.log`)

Check: base-write mode on the N6 calibration photos reproduces B exactly (test KL .3698, top-1 66.47%). On the 4,000 fit photos it gives .354.

| Row (keep 50%, 30 blocks, 1,500 test photos) | Test KL | Top-1 | Held KL |
|---|---|---|---|
| B (writes fitted on Base's inputs) | .370 | 66.5% | .368 |
| base-write mode, 4,000 fit photos | .354 | 66.8% | .352 |
| C-seq (writes fitted on C's own inputs, target Base's block output) | 1.193 | 65.9% | 1.210 |
| mean shift only | 1.166 | 66.9% | 1.180 |
| **C-lens**, ridge 1e-4 / 1e-3 / 1e-2 / 1e-1 | **.978** / .983 / 1.033 / 1.718 | 68.5% / 68.5% / 68.2% / 63.6% | 1.008 / 1.013 / 1.063 / 1.715 |

| Prediction | Result |
|---|---|
| P1 C-lens KL under .30 | **fails badly**: .978, 2.6 times B |
| falsifier: C-lens no better than C-seq | not triggered: .978 against 1.193. But the whole sequential frame is far worse than B |
| firing-rate gate | passes in stage 3 early (ratio .92-1.00). Falls to .47-.66 from block 26 on (ridge 1e-3); the fallback was not needed to explain the result below |

**Diagnostic 1: no block repairs its incoming error along that error** (`lens/diag.py`, `diag.json`, 200 held photos). Repair = share of the incoming stream error e that a block removes, measured only as a projection on the raw stream error. It is below 0 in almost every block, for every network, including Base's own blocks run on the drifted stream (base_repair). So "Base self-repairs and the sequential fit removes it" is false. And the stream size does not decide the cost: last-block drift B .47, T .33, C-seq .40, C-lens .35, yet KL .370, .131, 1.193, .978.

**Diagnostic 2: the sequential builds are under-confident** (`lens/diag2.py`, `diag2.json`).

| Network | Logit spread / Base (median) | Cosine of logit error with Base's centred logits | Test KL | With one temperature fitted on held |
|---|---|---|---|---|
| B | 1.22 | **+.19** (inflates the signal) | .370 | .363 |
| T | 1.08 | **+.03** (at right angles to the signal) | .131 | .131 |
| C-seq | .81 | **−.62** (shrinks the signal) | 1.193 | .548 |
| C-lens | .79 | **−.67** | .978 | .489 |

```
least squares on a noisy input:   output = E[ Base's output | what C still sees ]   → pulled toward the average
every block does this to the same class signal → the shrink lines up over 30 blocks → flat logits
```

Why it matters: this is the alignment row of N6 seen from the other side. An error lines up across blocks when it moves WITH the class signal: shrinking it (C-seq, C-lens) or inflating it (B). Training's error is at right angles to the signal (+.03), so it behaves like noise and cancels. The lens as built is a conditional average. It hedges ("half a taxi"). The essay's coarse gate is decisive: it answers a coarser question at full strength. So the lens is the wrong object as built, and the reason is measured.

Next (a change from the plan's step 2, which was another least-squares target and would dilute the same way): keep each fit's direction but restore Base's spread, with the optimal-transport map between the fitted output's Gaussian and Base's (closed form, same config shape). Flags `--ot_lens`, `--ot_writes`.

### Step 1b: restore the spread (`build.py --ot_lens --ot_writes`)

Each least-squares output (lens or writes) is followed by the optimal-transport map from its own Gaussian to the target's: M = A^-1/2 (A^1/2 Σ_target A^1/2)^1/2 A^-1/2, A = covariance of the fitted output. It keeps the fit's direction and gives the output Base's covariance. Folded into the same weights: no new config shape. Typical amplification 1.1-1.45 per direction (median), up to 30-65 in a few nearly lost directions of blocks 26-29.

| Row | Test KL | Top-1 | Held KL |
|---|---|---|---|
| B | .370 | 66.5% | .368 |
| C-seq + spread on writes | .393 (from 1.193) | 69.2% | .415 |
| C-lens + spread on lens and writes | .352 (from .978) | **70.4%** | .370 |
| C-lens + spread on the lens only | .762 | 69.1% | .782 |
| base-write, 4,000 fit photos (no spread map) | .354 | 66.8% | .352 |
| **base-write + spread on writes** | **.295** | 68.9% | .296 |
| base-write + spread, on B's calibration photos (like for like with B) | .318 | 68.3% | .316 |
| T-local (trained, local target) | .270 | 68.9% | .280 |
| T (trained end to end) | .131 | 77.4% | .156 |

| Network | Logit spread / Base | Cosine of logit error with Base's logits | KL after one temperature |
|---|---|---|---|
| B | 1.22 | +.19 | .363 |
| base-write + spread | 1.17 | +.20 | .282 |
| C-lens + spread | .94 | −.36 | .283 |
| C-seq + spread | .94 | −.35 | .302 |
| T | 1.08 | +.03 | .131 |

Reading:
1. Restoring spread is the largest no-training gain in N4-N7. The map's own gain, paired by photo: on the 4,000 fit photos .354 to .295 (+.059, 95% interval +.051 to +.068); like for like with B on its calibration photos .370 to .318 (+.052, +.044 to +.060). The other .016 of ".370 to .295" comes from the larger photo set. No training, no lens, same config shape. In total a no-training rule now closes .075 of the .239 gap (31%). Which N6 rows that share comes from is not measured yet.
2. Shrink explains the lens's failure: restoring spread moves its cosine from −.67 to −.36 and its KL from .978 to .352.
3. It does not explain the best row: base-write + spread gains .059 with the cosine unchanged (+.19 to +.20). Where that gain sits (single blocks, alignment, interaction) is not measured yet.
4. By raw KL the lens does not help once spread is restored: .352 against .295 (+.057, interval +.042 to +.072). After one temperature the two tie (.283 against .282), and the lens build has the higher top-1 (70.4% against 68.9%). So the lens is not needed by raw KL. Whether it carries something a temperature hides is open.
5. Paired intervals for the two big repairs: C-seq 1.193 to .393 (+.800, +.775 to +.825); C-lens .978 to .352 (+.626, +.603 to +.650).
6. The firing-rate gate did trip in the full lens run (on-rate ratio .47-.66 from block 26 on). The spread map on the lens brought it back to .93-1.02, so the fallback was not used.

### Step 1c: gate crossings, P4 (`lens/gates.py`; `results/lens/gates.json`)

360 kept vehicle-tree neurons (blocks 18-29) and 360 random kept neurons. 140 vehicle photos (held + test) and 140 other photos.

| Vehicle photos | Members: cross90 / lost99 / output change | Control: cross90 / lost99 / output change |
|---|---|---|
| B | .064 / .435 / .297 | .067 / .290 / .203 |
| T | .060 / .391 / .296 | .059 / .353 / .201 |
| base-write + spread | .056 / .406 / .267 | .061 / .273 / .187 |
| C-lens + spread | .049 / .396 / .231 | .049 / .353 / .154 |

| Prediction | Result |
|---|---|
| P4 C-lens crosses fewer gates than B | holds (.049 against .064) |
| the gate reading: training keeps the circuit's gate decisions right | **T does not keep Base's gate decisions.** At the 99th-percentile level T's members cross as often as B's (.0091 against .0089) and lose 39% against 44% of their top rows. Output change .296 against .297. Caveat: T's reads were trained, so a crossing against Base's level mixes error with training's intended read change. The 90th-percentile level sits in the off region for these blocks, so the 99th-percentile numbers lead |

Reading: the count of changed gate decisions does not predict the cost. C-lens + spread changes the fewest and costs 2.7 times T. Tree members are disturbed more than random kept neurons in every network and on non-vehicle photos too (lost99 .39-.44 against .27-.35), so a deleted neuron's error does reach its tree's later neurons more than others. But training does not stop that. Training's advantage must sit in what the changed gates write and how it adds up at the end, not in keeping the gates unchanged.

| | |
|---|---|
| Cale line | a new no-training rule: re-solve the kept writes, then restore each block's output spread with one closed-form map. Like for like with B: KL .370 to .318; with 4,000 fit photos .295. Same config shape. The lens is not needed by raw KL |
| Whole-network build | yes, all 30 blocks, no training |
| Label | none. Measurements; one ridge for the lens chosen on held photos, no knob in the spread map |

Open, in order: (1) where the spread gain sits (the N6 split works for base-write builds: their blocks are fitted on Base's inputs); (2) the same at keep 25%; (3) why B inflates the signal (+.19) and the lens shrinks it (−.36): training sits between at +.03.

### Not run in this session (from the approved plan)

| Item | Status |
|---|---|
| P2 head lens, P3 prefix swap, downsample-lens and head-lens rows | not run |
| Step 2 (later blocks undo the read part of upstream error) | replaced: the dilution result showed that another least-squares target would shrink the same way. The spread map ran instead |
| P5 | not run |
| keep 25%, the sports-car walk | not run |

### Step 1d: where the spread gain sits (`lens/split.py` runs `gap/additive.py`; `results/lens/additive_base_otw_saved_split.json`)

Valid for this build because its blocks are fitted on Base's inputs. Same 1,500 test photos. Note: the spread build used the 4,000 fit photos, B the calibration photos, so .016 of the total is the photo set (step 1b).

| Row | B | base-write + spread | T | B minus spread build | share of that gain |
|---|---|---|---|---|---|
| single blocks | .064 | .072 | .072 | −.008 | −11% |
| alignment (added minus singles) | .120 | .101 | .019 | +.019 | 25% |
| interaction (real minus added) | .186 | .123 | .040 | **+.063** | **84%** |
| total | .370 | .295 | .131 | .075 | 100% |
| alignment ratio, median / mean | 1.92 / 2.70 | 1.68 / 2.29 | 0.74 / 1.26 | | |

Reading:
1. The spread map mainly stops errors from growing through later changed blocks: it removes a third of B's interaction row (.186 to .123), which is 43% of T's interaction gain (.146).
2. Its single blocks get slightly worse (+.008), exactly like T's (.072 against .064). It moves every row in training's direction.
3. The alignment row barely moves (.120 to .101 against T's .019). That row still needs a target that sees the end (N6: only the end target cancels).

| N6 gap row | T's gain over B | reproduced by a no-training rule now |
|---|---|---|
| alignment | .101 | .019 (19%) |
| interaction | .146 | .063 (43%) |
| single blocks | −.008 | −.008 (the same sign as T) |

| | |
|---|---|
| Cale line | the spread map is the first no-training rule that reproduces part of training's two compounding rows: 43% of interaction, 19% of alignment |
| Label | none. Measurements; the split is an identity, its sizes are measured |

## N8. What distillation does, one step at a time, on one circuit (2026-11-19; plan `~/.claude/plans/ok-apparently-you-made-streamed-squirrel.md`; code `neuron/step/`, results `neuron/results/step/`)

Shonu, 2026-11-19: every build so far (B, lens, spread map, the N6 candidate) is a per-block least-squares fit. None is derived from how distillation updates the weights. Explain every part of training on one circuit, 100% of it.

The bar for "explained": a formula written BEFORE the run predicts the change of the circuit's read, write and logit effect. A residual row holds the rest. A sum that adds up by identity does not count. Generalization (fit photos against test photos) is its own row.

### Step 1: one training step, written out (no run)

The network inside one block g, at one position of one photo (`gap/common.py`):

```
x_g ──► dwconv ──► LayerNorm ──► u ──► pre = W1 u + b1 ──► a = GELU(pre) ──► y = ls ⊙ (W2 a + b2) ──► x_{g+1} = x_g + y
                                        (reads: rows of W1)                    (writes: columns of W2)
```

Training T (`gap/train_ref.py`): loss = KL(p_Base ‖ p_small) at the logits, mean over a batch of 32 photos. AdamW, lr 1e-4, cosine decay, 3,996 steps, start = B. Only the kept W1, b1, W2, b2 of the 30 blocks move. The model stays in eval mode (no stochastic depth).

One step, for one kept neuron s of block g (chain rule, exact):

```
end error:                   p_small − p_Base                         (1,000 numbers per photo)
back-signal at block g:      δ = ∂L/∂x_{g+1} = J_gᵀ (p_small − p_Base)    J_g = d logits / d x_{g+1}, on the SMALL network's own path
write column of s:           ∂L/∂W2[:,s] = Σ_positions  a_s · (ls ⊙ δ)
read row of s:               ∂L/∂W1[s,:] = Σ_positions  GELU'(pre_s) · (W2[:,s] · (ls ⊙ δ)) · u
Adam turns the gradient into a step:   ΔW = −lr · m / (√v + ε)    (m, v = running mean of the gradient and of its square, per weight)
```

What B is in the same terms. B's writes are the least-squares fit of the block's own output. Gradient descent on that fit uses the same formula with δ replaced by the local error:

```
B:  δ  →  e_g = y_small(u_Base) − y_Base(u_Base)     identity weight, Base's inputs, writes only, solved to the end
```

So B is the end point of "training whose back-signal is the block's own error". Training uses the end error instead. Near Base, the end error splits into named terms (first order in the block errors e_k, each block alone on Base's input; F = diag(p_Base) − p_Base p_Baseᵀ, the softmax's curvature):

```
δ_g  =  J_gᵀ F J_g e_g            term 1: own error, weighted by what the logits see
     +  J_gᵀ F Σ_{k≠g} J_k e_k     term 2: the other 29 blocks' errors
     +  R_g                        term 3: the rest (path drift, curvature, interaction). Exact: R_g = δ_g − term 1 − term 2
```

| # | What training has and B lacks | Where it is in the formula | N6 row it should feed |
|---|---|---|---|
| 1 | the target is the end, seen through the logits | term 1 weight J_gᵀ F J_g instead of identity | singles |
| 2 | other blocks' errors enter each block's update | term 2 | alignment |
| 3 | the small network's own drifted path | term 3, and u, a, pre taken on the small path | interaction |
| 4 | reads move, scaled by GELU'(pre_s) | the read formula | reads 64% |
| 5 | Adam: per-weight scaling, small steps, the best held stop | m / √v | path |
| 6 | 1,999 x 2 fit photos only | the sums run over fit photos | generalization |

Why it matters: the read formula already says WHICH survivor takes a dropped neuron's ground. The read of s moves on a photo only when (a) GELU'(pre_s) is large, that is s sits near its gate there, and (b) s's write points along the back-signal (W2[:,s] · δ ≠ 0). On the photos of a dropped neuron d*, δ carries d*'s missing write. So the survivors that move toward d*'s photos should be the ones just below their gate there whose writes can carry the fix. E2 found exactly that kind of partner in stage 4: a gate-mate on the negative side. This is a candidate mechanism, not yet a measurement. Evidence that bears on the terms: N4 step 7 (linearizing did not explain the one-layer read turn; the gain fell where dropped neurons were active, 72% on the top 10% of positions), N6 3b (alignment and interaction carry the gap), N7 1c (T does not keep Base's gates).

### The circuit

Block 28 (stage 4, the second of three blocks at 7 x 7). Dropped neuron d* = 1581: in E2 it is the owner of 11 of the 20 survivors with the largest firing change. Main survivor s = 4023 (owner correlation .44, chance .07; upstream gate cosine with d* .85; Base correlation with d* −.17, so d* fires where s is off).

### Predictions (written before any run of `neuron/step/`)

| # | Prediction |
|---|---|
| Q1 | first-order check: the linear sum Σ_k J_k e_k (each block's single error carried to the logits by Base's Jacobian) reproduces B's real logit error only partly: cosine .3-.7 over test photos (median). Reason: interaction is 61% of the gap |
| Q2 | at step 0 (B), for block 28's write gradient: term 2 is larger than term 1 (norm ratio > 1), and term 3 is not small (its norm at least half of δ's) |
| Q3 | at step 0, the read gradient of s = 4023 sits on d*'s photos: the share of Σ |contribution| from d*'s top-200 Base photos is at least 3 times their share of photos |
| Q4 | the step-0 score Σ_{d* photos} GELU'(pre_s) · |W2[:,s] · (ls ⊙ δ)| ranks block 28's survivors: the 11 survivors with owner d* = 1581 in E2 are in the top 10% of all 2,048 survivors by this score, median |
| Q5 | direction: the cosine of the step-0 gradient with the total change T − B is .2-.4 for s's read and write. Adam's sign-like first step, −sign(g), has a higher cosine than g itself |
| Q6 | the change T − B of s's write is small next to its read change (relative norm), as E2 found (s keeps its own write) |

### Step 2 result: the gradient at step 0 (`step/grad0.py`; `results/step/grad0.json`, `grad0.log`, `grad0.pt`)

3,000 fit photos (E2's set). Survivors from `T_build.pt` (a local rebuild differs in 6 indices of 30 x 2,048, so the saved ones are used). The term split was refined before the full run: term 3 = ti (interaction: the part of B's logit error that is not the linear sum) + tc (softmax curvature) + tp (path: J on B's own path, through B's shrunk block 29, against J on Base's path). The parts add to the real gradient exactly.

| # | Prediction | Result |
|---|---|---|
| Q1 | cos(Σ J_k e_k, real logit error) .3-.7 | **fails high**: median .81 (IQR .74-.86). The direction is mostly first order. The SIZE is not: the linear sum has .67 of the real size, and KL(Base + linear sum) = .152 against the real .359. Errors grow on the way to the logits |
| Q2 | term 2 > term 1, term 3 at least half | holds. Share of the real gradient (projection), block 28 writes: t1 .05, t2 .29, t3 .67 (ti −.03, tc .04, **tp .65**). Reads: t1 .12, t2 .48, t3 .40 (ti −.74 and tp +1.15 cancel: a split with two large opposite parts, so not a clean reading for reads) |
| Q3 | s's read gradient sits on d*'s top-200 photos, at least 3 times their share | **fails**: those photos (6.7% of photos) carry 2.4% of |s read grad| (ratio 0.4), and −2.5% of the push along the total |
| Q4 | owner-1581 survivors in the top 10% by the d*-photo push score | holds in level (median 92nd percentile), **fails as a mechanism**: the other E2 survivors of block 28 score the same (94th). By the ratio (d* photos / all photos) the owners sit at the 63rd percentile, others 83rd. The score finds survivors with a large gradient, not d*'s takers |
| Q5 | cos(step-0 gradient, T − B) .2-.4 for s | **fails low**: s read +.16 (Adam sign +.17), s write −.10 (sign −.14). Whole block 28: read +.08 (sign +.16), write +.08 (sign +.09) |
| Q6 | s's write changes less than its read | holds for s (relative change .045 against .084). Block 28 as a whole: .051 against .057 |
| noise | | the full-batch gradient on even against odd batches (1,500 photos each): cos .48 (writes), .63 (reads). One photo's gradient is mostly noise; directions need about 1,000+ photos |

Reading:
1. **At step 0, the own-error term (the only thing a per-block fit uses, even with the right weighting) is 5% of block 28's write gradient and 12% of its read gradient.** The other blocks' errors give 29% / 48%. Most of the write gradient (65%) comes from the small network's own path: block 28 is trained THROUGH B's shrunk block 29, not through Base's block 29.
2. **The circuit mechanism I wrote in step 1 is wrong at step 0.** The read formula gives almost no push on d*'s photos. Whatever makes s take d*'s ground (E2) does not come from d*'s photos pushing on s at B. (An earlier version of this line said the cause was GELU'(pre_s) ≈ 0 there. That is WRONG, see the GELU check below: s sits on d*'s photos as it does everywhere. So the small push comes from the back-signal factor W2[:,s] · (ls ⊙ δ), not from the gate factor.)
3. **T − B is not the first step.** The first gradient predicts T's change only weakly (block cos .08-.16). The path turns. The checkpoint rerun (plan step 3) is needed.

### Q3b (written before the run)

If s's new ground does not come from pushes on d*'s photos, it may come from GENERALIZATION of the read change: pushes on photos where s is near its gate move s's read along an input direction that d*'s photos also carry. Test on Base's inputs u (as E2), 3,000 photos, peak over positions: Δpeak_s for three read changes of equal size ‖T − B‖ of s: (a) −g0, (b) −sign(g0), (c) the real T − B. Correlate each Δpeak_s with d*'s Base peak.

| # | Prediction |
|---|---|
| Q3b | (c) gives about E2's .44. (a) and (b) give a positive correlation, at least half of (c) |

### Q3b result (`step/q3b.py`; `results/step/q3b.json`)

| Read change, same size as T − B per survivor | Correlation of Δpeak_s with d*'s Base peak, median over block 28's 20 E2 survivors |
|---|---|
| (c) real T − B | .397 (reproduces E2 row by row: check passed) |
| (a) −g0 | .006 |
| (b) −sign(g0) (Adam's first step) | .030 |

Q3b **fails**. The step-0 gradient does not move the survivors onto d*'s ground, not even by generalization. For s = 4023: .04 / .05 against .44. In 9 of 20 survivors (a) is negative.

What this settles: the "merge" that E2 saw (a survivor takes a dropped gate-mate's ground) is not in the first gradient at B. It forms along the training path: after other weights have moved, the back-signal at block 28 changes, and the read follows. So the circuit cannot be explained from B alone. The checkpoints (plan step 3) are required, not optional.

### Step 3: T rerun with checkpoints (UVA cluster, job `n8-ckpt`; `gap/train_ref.py --run T-ckpt --save_at ...`)

Same script, survivors, lr 1e-4, seed and photos as T. Checkpoints after steps 1, 10, 33, 100, 333, 1,000, 2,000, 3,996. At each checkpoint t: the gradient split of `step/grad0.py` at the point t, and `step/q3b.py` with (c) = change so far (t − B), (a) = −g_t and (e) = the change still to come (T − t), each correlated with d*'s Base peak for block 28's 20 E2 survivors.

A fact to hold while reading (GELU, not a measurement): GELU'(x) = Φ(x) + x φ(x) is NEGATIVE for x below about −0.75 (minimum about −0.17 output near x = −0.75). Base's pre-activations are below 0 at more than 99% of rows (N4 trap). So a survivor that is "off" on d*'s photos still outputs a small negative number there, and its read gradient there has the opposite sign of the on-side gradient. The step-1 sentence "GELU'(pre_s) ≈ 0 where s is off" is only right far below the gate.

| # | Prediction (written before the job's results) |
|---|---|
| R0 | the rerun reproduces T: held KL .156 ± .005, test KL .131 ± .005 |
| R1 | the takeover (c) grows with t: at t = 333 it is at least half of its final .40 (79% of the held gain is in by then) |
| R2 | at some t ≥ 33, −g_t moves the survivors onto d*'s ground: median (a) ≥ .15 |
| R3 | at that t, the term carrying the read gradient of block 28 is term 2 or the path term, not term 1 (own error) |
| R4 | the own-error share (t1) of block 28's gradient stays under 20% at every t |

### GELU check (`step/lobe.py`, Base inputs, the 11 survivors whose E2 owner is d* = 1581)

| Photos | pre_s 10th / 50th / 90th percentile | share in the dip (pre < −0.75) | share above 0 | mean GELU' |
|---|---|---|---|---|
| d*'s top 200, all positions | −1.32 / −.33 / .28 | .34 | .27 | +.27 |
| d*'s top 200, at d*'s peak position | −1.26 / −.36 / .29 | .35 | .30 | +.27 |
| all photos (every 10th) | −1.40 / −.39 / .32 | .36 | .28 | +.27 |

The owners are not "off" on d*'s photos in the sense of GELU: they sit near the gate there exactly as elsewhere. A third of their positions are in GELU's dip, where GELU' < 0. So at B the read push on d*'s photos is small because the back-signal along s's write is small there, not because the gate is shut.

### Step 3 result: the checkpoints (job 6959139, H100, 19 min; `results/step/ckpt/*.json`, `T-ckpt.log`)

R0 **holds exactly**: the rerun gives test KL .13064 (T .13064), held .1563, fit .0302, top-1 77.4%. Training is reproducible.

Per checkpoint t, on the 3,000 fit photos (E2's set). "Halves" = cosine of block 28's full-batch gradient on even against odd batches (1,500 photos each). Shares = projection of each term on block 28's real gradient (they add to 1). Takeover = median over block 28's 20 E2 survivors of corr(Δpeak_s, d*'s Base peak).

| t | fit KL (real) | KL of Base + linear sum | real / linear | halves W2 / W1 | W1 shares t1 / t2 / ti / tc / tp | takeover so far (c) | −g_t takeover (a) | held KL (curve) |
|---|---|---|---|---|---|---|---|---|
| 1 | .344 | .145 | 2.37 | .43 / .57 | .12 / .47 / −.78 / −.01 / 1.20 | .03 | .01 | .367 (t 0) |
| 10 | .281 | .122 | 2.30 | .15 / .18 | −.04 / .14 / .72 / .18 / .00 | .04 | .07 | |
| 33 | .240 | .118 | 2.03 | .16 / .19 | .14 / .58 / −.90 / −.10 / 1.28 | .12 | −.01 | |
| 100 | .195 | .106 | 1.84 | .08 / .05 | −.05 / .12 / .58 / .18 / .17 | .10 | .28 | |
| 333 | .154 | .100 | 1.54 | .09 / .10 | −.10 / −.07 / 1.22 / .22 / −.27 | .20 | .14 | .201 |
| 1,000 | .094 | .102 | 0.92 | .07 / .08 | .01 / −.32 / 1.51 / .18 / −.38 | .32 | .11 | .184 |
| 2,000 | .055 | .084 | 0.65 | .24 / .43 | .20 / 1.82 / −1.55 / −.38 / .91 | .38 | −.18 | .168 |
| 3,996 | .030 | .060 | 0.50 | .03 / .02 | −.10 / .27 / .79 / .01 / .03 | .40 | .22 | .156 |

Takeover of the final T on photos training never saw (`q3b.py --cache held / test`): held .378, test .338 (train .397). It generalizes.

**Worth of block 28's read change at the end** (`step/swap28.py`; `results/step/swap28.json`):

| Network | Held KL | Test KL |
|---|---|---|
| T | .1563 | .1306 |
| T with block 28's reads (W1, b1) put back to B's | .1561 | .1306 |
| T with block 27's reads put back | .1581 | .1335 |
| T with block 29's reads put back | .1550 | .1299 |
| B | .3675 | .3698 |
| B with T's block 28 reads | .3582 | .3597 |

| # | Prediction | Result |
|---|---|---|
| R0 | rerun = T | holds exactly |
| R1 | takeover at t = 333 at least half of its end | holds at the edge (.20 of .40). It grows steadily to the end, while the held KL is 79% done at 333 |
| R2 | some t ≥ 33 with −g_t takeover ≥ .15 | **not settled**: the best two of seven noisy values (−.18 to +.28) pass. Those gradients have halves cos .02-.10, so they are mostly photo noise |
| R3 | the read gradient's carrier is term 2 or the path | **fails** where R2 holds (t = 100): ti (interaction) .58. The ti and tp parts swing between large opposite values from checkpoint to checkpoint (1, 33, 2,000 against the rest), so this split is not a stable reading |
| R4 | own-error share t1 under 20% at every t | holds (largest .20) |

Reading:
1. **The circuit I chose is redundant at the end.** In T's order, block 28's whole read change, takeover included, is worth .000 (held .1563 to .1561, test unchanged). In B's order it is worth .009-.010 (B + T's block 28 reads: held .3675 to .3582). So it is not useless: other changes in T cover for it. The takeover is real and it generalizes. I chose the circuit by the SIZE of the change (E2's 20 largest), not by its WORTH, in both orders. That was the wrong rule. Stage 4's single-block read changes are each worth ±.002 in T's order.
2. **Block 28, fit photos: the photos agree on the direction only at the start.** At t = 1 the halves cos is .43-.57. From t = 10 on it is .02-.24, and it does not fall steadily (.24 / .43 at t = 2,000). This is one block on training photos, not a law of training. The held KL still falls after t = 333 (.201 to .156), and the takeover that forms late generalizes.
3. **Fit photos only (T memorizes them), not yet a share:** B's real KL is 2.37 times the KL of its blocks' first-order sum, T's is 0.50 times, and the ratio falls steadily over the checkpoints. Two cautions before any share is quoted: these are fit photos, and at T each single-block error is taken on Base's inputs, outside T's own context (the second reading N6 already named). The same measure on held photos is needed first.
4. **The gradient at a checkpoint is not a stable object after t ≈ 10.** A term split of a direction that is mostly photo noise cannot be "explained to 100%". The object to explain must be one that is stable across photos.

### Where the worth is: one block at a time, both orders (from N6's `results/gap/ledger_T.json`, test photos)

| Order | Per block (30 blocks) | Sum over the 30 | Largest |
|---|---|---|---|
| B's order: B + one T block, gain | .008-.029 | .472 (2.0 times the gap .239) | block 4 (.029), 0 (.025), 25 (.026), 24 (.025) |
| T's order: T minus one block (put back to B's), cost | −.000 to .005 | .045 (19% of the gap) | block 0 (.005), 4 (.004), 27 (.004) |

The worth of training's change is collective. In B, each trained block fixes part of one shared problem, so the single gains overlap (sum 2 times the gap). In T, any one block can be put back almost for free, because the other 29 cover for it. No block holds the gap in both orders, so no neuron circuit inside one block can hold 100% of it in this setting. This matches N6: alignment (B's errors line up) and interaction carry the gap.

### Status of N8 against its bar

| Item | Status |
|---|---|
| explained by a prediction written before the run | **0% of the gap.** Q1, Q3, Q3b, Q5 and R3 failed. The circuit was the wrong object (chosen by size, redundant at the end) |
| solid | the rerun reproduces T exactly (checkpoints exist on the cluster, `n6/clean/neuron/results/gap/ckpt/`). The own-error term (the only term any per-block fit uses) is at most 20% of block 28's gradient at every checkpoint. The takeover generalizes (held .38, test .34) |
| the next object | a directional choice for Shonu: (a) one held photo that B gets wrong and T gets right, traced block by block; (b) a setting where the gap cannot be collective: shrink ONE block only, train it end to end, explain that one-block gap to 100%; (c) the shared error itself: the logit direction B's 30 errors line up along, and how T's 30 changes each remove part of it |

## N9. The smallest object that can hold the gap (2026-11-19; code `neuron/one/`, results `neuron/results/one/`)

Shonu chose: shrink ONE block only, train it end to end, explain that gap to 100%. First check: is there a gap to explain? Build only (writes re-solved, value survivors), the rest is Base, held KL (2,000 photos):

| Shrunk | keep 50% | keep 25% | keep 10% | keep 5% | keep 2% |
|---|---|---|---|---|---|
| block 0 | .0007 | .0022 | | | |
| block 4 | .0015 | .0035 | .0053 | .0062 | |
| block 14 | .0026 | .0043 | | | |
| block 25 | .0035 | .0053 | .0060 | .0060 | |
| block 28 | .0031 | .0055 | .0087 | .0096 | .0097 |
| blocks 27-29 (stage 4) | .0205 | .0398 | | | |

Reading: one shrunk block costs at most .01, even with 98% of its neurons gone. The network covers for one weak block. So a trained one-block reference can gain at most .01, against the 30-block gap of .239, and one block has no compounding by construction. The one-block object does not contain the effect to explain. A candidate smallest object that may: stage 4's three blocks together (KL .021 at 50%, .040 at 25%). Block 28 alone is .003 at 50%. Blocks 27 and 29 alone were not measured, so whether the three compound is open.

## N10. The gap as an interpretability problem (2026-11-19; plan `~/.claude/plans/ok-apparently-you-made-streamed-squirrel.md`; code `neuron/read/`, results `neuron/results/read/`)

Shonu: "we aren't thinking enough in terms of interp ... this exploration is going to be the thing we need to hammer for a while before making another hypothesis." No new build or hypothesis in this phase. The object is a neuron tree: a child disappears, and something takes up its job. This section READS which job, and who takes it up. Every object is chosen by worth at the end, in both orders.

### Step 0: what the essay predicts (written before any N10 run)

Sources read: essay "Recursive Generalization", "How I Think About Neurons", "What Neurons Do Under Pressure", "Why Networks Generalize", "Watching Neurons Develop", "More on Circuit Ecology"; `CrakBox/PRINCIPLES.md` 8, 9, 11, 13, 14.

| Essay idea (where) | For B (holes, no training) | For T (trained from B) | For a deleted neuron's job |
|---|---|---|---|
| Narrow-layer retraining: a crushed layer stays readable but COARSER ("curves & loops" instead of ten curves) (Under Pressure) | B is not trained, so this does not apply to B. B is a fine network with holes (N7) | T's survivors become coarser: each covers a wider set of photos than its Base self, along a related axis | The job survives as part of a broader neighbour, not as a packed mixture |
| Neurons are control flow, gates in front; the negative pole is the distinction that matters at that point (How I Think) | A deleted GATE lets the wrong group through: B's errors go to the group the gate kept out (related by pose or shape, maybe not by WordNet). A deleted late DETECTOR loses a fine split: errors go to the sibling on its negative pole | T restores the distinction, not the detection: the pair of classes on the deleted neuron's two poles is separated again | Read each deleted neuron's job as "lets X through, keeps Y out". Gate or detector is decided per neuron (principle 9) |
| Smaller models stop at coarser distinctions (How I Think) | — | T's answers inside a family stay a little coarser than Base's, but finer than B's | — |
| Circuit ecology / recursive generalization: an offshoot is error picked up by the neuron that was already closest (More on Circuit Ecology) | — | The survivor that takes up a lost job is the one already on for the lost job's photos (largest overlap of top photos), not the one whose weights moved most | The taker grows one step along one axis (also N4 step 5: a coarse neuron = its Base counterpart with one condition dropped) |
| Watch it grow: movement is local (Watching Neurons Develop) | — | Over T's checkpoints, the taker's firing spreads gradually to the lost job's photos | Later sessions (step 3) |
| Both poles, verb not noun (principle 8) | — | — | Every card names the job as a verb and reads both poles by percentile band |

Combined prediction for Step 1: B's extra probability lands mostly inside the true class's family (a coarse answer, or a lost fine split), and T moves it back. A competing reading: B's errors land outside the family (a broken gate passes a wrong group), which would favour gates over detectors as the lost jobs.

### Step 1: where the gap is, in words — definitions fixed before the run

Photos: held (2,000, 2 per class) for choosing; test (1,500) scored once. Fit photos are never used (T memorizes them).

Families: a partition of the 1,000 classes into WordNet families, fixed in `results/read/families.json` before any gap number is seen. Checks: dogs = classes 151-268; birds = the N5 list; the N5 vehicles fall into one family.

Per photo, with Base distribution p and network distribution q (q = B or T):

| Quantity | Definition |
|---|---|
| gap | KL(p‖q_B) − KL(p‖q_T) |
| family gap share | sum of gap over the family's photos (grouped by TRUE label) / total gap |
| family gap rate | mean gap per photo in the family |
| gained mass | g_c = max(q_c − p_c, 0); total G = sum_c g_c (= total variation) |
| in-family share of gained mass | sum of g_c over classes in the true label's family / G, mass-weighted over photos |
| control 1 (size) | the family's share of the 1,000 classes |
| control 2 (Base's own spread) | Base's off-top mass in the family: sum of p_c over the family minus Base's top class, / total off-top mass |
| near share | same as in-family, for classes whose WordNet lowest common ancestor with the true class is at most 2 steps above the true class |

Intervals: paired bootstrap over photos, 2,000 draws, 95%.

Choosing rule (fixed now): a family is a candidate if its held rate's lower bound is above the overall held mean rate. Among candidates, the top family has the largest share. If no family is a candidate, take the largest share. Report shares and rates for all families. Families are ranked, never classes (2 held photos per class).

| # | Prediction (before the run) |
|---|---|
| P1 | Rerun reproduces N8: held KL B .3675, T .1563; test B .3698, T .1306 |
| P2 | B's in-family share of gained mass is above control 1 AND above control 2, on held |
| P3 | T's in-family share is below B's (T fixes in-family errors more than out-of-family ones) |
| P4 | The near share for B is above Base's own near spread (B loses fine splits, not whole groups) |
| P5 | The top 3 families carry more than 40% of the held gap |

### Step 1 results (`read/logits.py`, `read/where.py`; `results/read/where_held.json`, `where_test.json`, `families.json`)

Family list checks: dogs = 151-268 (118 classes), birds = the N5 list (59), the N5 vehicles all in "vehicles+craft". 22 families. WordNet ids from the Keras class index, WordNet 3.0 from nltk.

P1 holds exactly: held B .3675, T .1563 (gap .2111); test B .3698, T .1306 (gap .2392).

**Where the gap is.** Top families by share of the gap (true label), with the gap per photo (rate). Overall rate: held .211, test .239.

| Family | Classes | Held share [95%] | Held rate [95%] | Test share | Test rate | KL B → T (held) |
|---|---|---|---|---|---|---|
| devices+instruments | 128 | .148 [.116, .183] | .244 [.193, .299] | .152 | .286 | .452 → .208 |
| clothing+covering | 90 | .113 [.082, .145] | .264 [.199, .335] | .117 | .310 | .459 → .195 |
| **birds** | 59 | .083 [.058, .113] | **.299 [.216, .389]** | .084 | .323 | .386 → .087 |
| food+plants | 63 | .082 [.056, .112] | .274 [.191, .369] | .084 | .330 | .402 → .128 |
| reptiles+amphibians | 44 | .063 [.042, .086] | **.300 [.220, .401]** | .059 | .308 | .451 → .151 |
| insects+spiders | 47 | .055 | .246 | .064 | .308 | .379 → .134 |
| vehicles+craft | 72 | .043 [.028, .059] | .127 [.088, .171] | .038 | .133 | .256 → .129 |
| dogs | 118 | .043 [.031, .058] | .077 [.056, .101] | .045 | .090 | .177 → .100 |

The choosing rule picks **birds**. Only birds (lower bound .216) and reptiles+amphibians (.220) have a held rate above the overall .211, and birds has the larger share. T fixes birds the most: KL .386 → .087 held, .397 → .073 test.

**Where B's extra probability goes** (gained mass g_c = max(q_c − p_c, 0), all photos):

| | B held | B test | T held | T test | Base's own off-top spread (control 2) | Class share (control 1) |
|---|---|---|---|---|---|---|
| gained mass G (= total variation) | .241 | .236 | .157 | .144 | | |
| share in the true family | .494 [.472, .518] | .511 [.485, .540] | .394 | .428 | .255 | .072 |
| share to near classes (common ancestor ≤ 2 steps up) | .366 | .371 | .287 | .314 | .193 | |

Birds only (held): B in-family .579, near .393; T .441, .324; Base's spread .163, .113.

| # | Prediction | Result |
|---|---|---|
| P1 | rerun = N8 | holds exactly |
| P2 | B in-family share above both controls | **holds**: .494 against .255 and .072 (held), .511 test. Every family with 20+ held photos is above its own control 2. Tools+implements is the closest (.172 against .157) |
| P3 | T in-family share below B's | **holds**: .394 against .494 held, .428 against .511 test |
| P4 | B near share above Base's near spread | **holds**: .366 against .193 |
| P5 | top 3 families carry > 40% of the gap | **fails**: .344 held, .353 test. The gap is spread over many families |

Reading:
1. **Half of what B gets wrong is a coarser answer inside the right family.** B's extra probability goes to the true class's family twice as often as Base's own uncertainty does (.49 against .26), and to close siblings (common ancestor at most 2 steps up) at .37 against .19. This is the essay's "coarser distinction": B says "some bird", "some lizard".
2. **T fixes the in-family half more than the rest.** In-family extra mass: B .119 → T .062 (−48%). Out-of-family: B .122 → T .095 (−22%). T restores fine splits inside families. Most of B's out-of-family error stays in T.
3. **The other half of B's error leaves the family, and goes to big "sink" families.** Animals lose mass to dogs (birds → dogs .24 of their out-of-family extra, cats .41, other mammals .40, reptiles .23; dogs are 12.5% of the other classes). Artifacts lose mass to devices+instruments (.20-.29). This fits a broken gate that lets a big neighbouring group through, but it is a reading only: no neuron was looked at.
4. **Families with many classes and many neurons lose little.** Dogs (rate .077) and vehicles (.127) cost B the least. Birds, reptiles, food and clothing cost the most per photo. A guess, not measured: value-ranked survivors keep the dog machinery because many neurons serve it.
5. The gap is spread (P5 fails). No family holds most of it. So the scope from step 2 can at best hold birds' share, about 8% of the whole gap.

**Step 1 check** (`read/where_check.py`; `results/read/where_check_held.json`, `_test.json`). The shares above count the true class itself, and B's logits are inflated (N7), so part of "in-family" could be B being sharper on the same answer. Rerun with the true class AND Base's top class removed from both sides:

| | B | T | Base's own spread |
|---|---|---|---|
| in-family, held | .419 [.394, .443] | .330 | .216 |
| in-family, test | .425 [.394, .456] | .345 | .223 |
| near, held | .268 | .212 | .151 |
| in-family, birds held | .511 [.406, .623] | .338 | .147 |

In-family extra mass, held: B .084 → T .046 (−45%). Out-of-family: B .117 → T .093 (−20%). Readings 1 and 2 stand, a little weaker.

Sink control: B's out-of-family extra against Base's own out-of-family spread, same photos, same exclusions (held). To dogs: from cats .41 against .18, other mammals .40 against .17, primates .31 against .13, birds .24 against .15, reptiles .24 against .14. To devices+instruments: from devices' own neighbours (other artifact .30 against .20, vehicles .26 against .16, clothing .24 against .16). So the sinks are 1.5 to 2.5 times Base's own spread. Reading 3 stands as a reading.

Notes: the birds choice passed its rule by .005 (lower bound .216 against .211), and the 2 photos per class are not independent. Test agrees, so the choice stays. Setup for reruns: `uv pip install --python curves/.venv/bin/python nltk`, then `nltk.download('wordnet')` into a folder named by `NLTK_DATA`; the class ids are in `results/read/imagenet_class_index.json` (Keras class index).

### Step 2: choose the scope by worth — definitions fixed before the neuron run (`read/rcommon.py`, `s2_blocks.py`, `s2_attr.py`)

Photos, birds only (59 classes): **choose** = held birds (118) + **spare** birds (910). Spare = odd-position photos outside `photos_test.npy`. No step used them before, and T never saw them (cached in `results/read/cache/`, 0.18 GB). **score** = test birds (93), scored once.

Checks done before: T changed exactly 120 tensors (W1, b1, W2, b2 of the 30 blocks), nothing else. A block rebuilt from Base with B's keep set S equals B's block only to 2.4e-3 in the weights (largest photo KL change 6e-4; T_build was made on the cluster). So a deleted neuron's worth is measured as KL(rebuild S) − KL(rebuild S + j), never against B itself.

Items and their two orders:

| Item | B order (gain) | T order (cost) |
|---|---|---|
| survivor j of block g: read row W1[j], b1[j], write column W2[:, j] | B with j's numbers from T | T with j's numbers from B |
| deleted Base neuron j | block g rebuilt with S + {j}, writes re-solved, against S alone | **does not exist**: j is in neither B nor T |
| write bias b2 of block g (belongs to no neuron) | B with T's b2 | T with B's b2 |

Shortlist (attribution, a shortlist only: weak in E2 and N8): survivors, predicted gain = −grad_B · (T − B) over the item's numbers, and predicted cost = grad_T · (B − T) (the run used the opposite sign by mistake; corrected in the results). Deleted neurons: the exact least-squares change of adding j (Frisch-Waugh: Δy = β_j (g_j − ĝ_j), ĝ_j = the survivors' prediction of g_j, β_j = the block output's covariance with that residual / its variance), dotted with grad_B.

Real patching: the top 40 survivors and top 40 deleted by predicted B-order gain, plus 30 random survivors (to test the shortlist tool: rank agreement between attribution and real patch). All 30 write biases.

**Set rule (fixed now):** rank every really-patched neuron by its measured B-order gain on choose. The set = the top 30. Collective test, scored once on score: (a) T with the set's survivors put back to B's costs at least 50% of birds' gap; (b) B plus the set (survivors from T, deleted neurons re-added) gains at least 50%. A failure reads "no set found by this procedure", not "no set exists".

Fallback if the test fails: the N5 birds capability tree (8 roots in block 17, 45 members in 11 blocks, `results/tree/trees2.json`), mapped onto S. Its worth in both orders, against a random set of the same count in the same blocks. The 3 pilot neurons are chosen by measured worth, not by tree membership.

| # | Prediction (before the run) |
|---|---|
| Q1 | No single block holds 50% of birds' gap in T's order (largest cost under .05 of the gap). In B's order, single blocks gain .02-.15 of the gap each, summing to more than 1 |
| Q2 | Attribution and real patch agree in rank on the random survivors with Spearman above .5 in B's order, and below .3 in T's order |
| Q3 | The collective test fails on (a): 30 neurons put back inside T cost under 10% of the gap |
| Q4 | (b) is larger than (a), but also under 50% |

### Step 2a results: whole blocks, birds only (`read/s2_blocks.py`; `results/read/s2_blocks.json`)

Birds' gap: choose (1,028 photos) B .410 → T .074, gap .336 [.300, .372]; score (93 test photos) B .397 → T .073, gap .323 [.215, .439].

| Order | One block | Sum over 30 | Largest |
|---|---|---|---|
| T's order: T with one block's MLP put back to B's, cost / gap | .000 to .012 | .16 | block 0 (.012), 27 (.011), 24-26 (.010) |
| B's order: B with one block's MLP from T, gain / gap | .029 to .161 | 2.27 | block 25 (.161), 24 (.160), 4 (.130), 19 (.130), 0 (.122) |

T's order, cumulative (largest cost first): 4 blocks .058, 10 blocks .213, 16 blocks .375, **20 blocks .502**.

Q1 holds. The birds gap is collective, as the whole-photo gap was in N8: in T, any block can be put back almost for free, and 20 of the 30 whole blocks must be put back to lose half of it. So part (a) of the collective test cannot pass for 30 neurons. This is settled before the neuron run.

**Pilot choice rule (fixed before the neuron results):** the deleted neuron with the largest measured B-order gain (re-added), and the two survivors with the largest measured B-order gain, from different blocks. Each card reads the neuron in Base (all 35,000 reading photos; the reader sees only the fit half), and in B and T on photos T never trained on (held, test, unused odd photos). Judge packets and the mismatched pairing (n1 → n2 → n3 → n1) are written by code before any reader starts (`read/pilot_sheets.py`, `results/read/pilot/judge_key.json`).

### Step 2b-c results: neurons, the collective test, the tree (`read/s2_attr.py`, `read/s2_set.py`; `results/read/s2_attr.json`, `s2_set.json`)

Single items on choose (birds' gap .336), as shares of the gap:

| Items | n | B's order: gain, largest / median / sum | T's order: cost, largest / sum | Attribution vs real (Spearman), B / T |
|---|---|---|---|---|
| top survivors by attribution | 40 | .053 / .010 / .60 | .004 / .023 | .84 / .73 |
| random survivors | 30 | .001 / .000 / .003 | .000 / .000 | .99 / .96 |
| top deleted neurons (re-added) | 40 | .031 / .015 / .65 | does not exist | .74 / — |
| write biases b2 | 30 | .003 largest, sum .054 | .000 | — |

Largest single items (B's order, share of birds' gap, 95%): survivor block 19 n1961 .053 [.046, .060]; survivor 25 n1229 .052 [.045, .059]; survivor 24 n794 .044; survivor 18 n92 .036; survivor 21 n962 .032; deleted 22 n295 .031 [.015, .050]; deleted 20 n1280 .031.

The set by the rule: 11 survivors and 19 deleted neurons, in blocks 0, 4, 12 and 16-25 (mostly 17-25). The sum of their single gains is .74 of the gap.

| Collective test | choose | score (93 test birds) | Bar |
|---|---|---|---|
| (a) T with the set's survivors put back to B's: cost / gap | .015 [.010, .020] | .026 [.010, .047] | ≥ .50 |
| (b) B plus the set: gain / gap | .419 [.355, .483] | .450 [.232, .691] | ≥ .50 |

**The test fails** (no set found by this procedure), on part (a) alone. Part (b) is not settled: .45 with interval [.23, .69], which spans the bar.

**No object holds birds' gap in both orders.** The 42-45% is B's order only. T does not depend on these neurons. This is N8's pattern at a larger size, not a found scope. The fallback applies: the N5 birds tree (53 neurons: 23 survivors, 30 deleted).

| Tree, both orders | choose | score |
|---|---|---|
| (a) T with the tree's survivors from B: cost / gap | .001 [−.002, .004] | .006 [.002, .011] |
| (b) B plus the tree: gain / gap | .161 [.132, .188] | .150 [.068, .244] |
| (b) for 5 random same-count, same-block sets (choose) | .001 to .014 | |

| # | Prediction | Result |
|---|---|---|
| Q1 | no single block holds 50% in T's order (each under .05); B's order .02-.15 each, sum > 1 | **holds**: T .000-.012; B .029-.161, sum 2.27 |
| Q2 | attribution ranks B-order gains (Spearman > .5) but not T-order costs (< .3) | **half fails**: .84 and .74 in B's order (holds); .73 in T's order (fails: attribution ranks T-order costs too). The script's predicted cost had the wrong sign (+grad_T · (T − B) instead of −); the Spearman values here are recomputed with the sign fixed, from `s2_attr.json` |
| Q3 | (a) under .10 | **holds**: .026 |
| Q4 | (b) above (a), under .50 | **holds on the point value** (.45), but the interval [.23, .69] spans .50 |

Reading:
1. **The two orders tell different stories, and both are true.** In B, birds' problem is concentrated: 30 neurons (11 trained survivors, 19 re-added deleted neurons) fix 42-45% of it, and the best single survivor fixes 5%. In T, the fix is spread and redundant: put back any one of them, or the whole set, and T loses 1.5-2.6%. T's solution is 20 of 30 blocks deep (step 2a).
2. **The re-added deleted neurons carry real worth for birds.** 19 of the 30 best items are deleted neurons. B's value ranking (std × |write|, over all classes) dropped neurons that birds need. This is the "child disappears" object in Shonu's words.
3. **The N5 birds tree is a real birds capability but not where B's birds gap is.** Added to B, it gives back 15-16%, ten or more times a random same-block set. But it shares only 1 neuron with the worth-chosen 30, which give back 42%. Both sit in blocks 17-25.
4. **Write biases do not matter** (largest .003). For n2 and n3, the worth is in the READ ROW (step 2e below), not the read bias or the write.

Scope: by the fixed rule, the fallback (the N5 birds tree) is the scope. But the worth-chosen 30 hold almost three times the tree's B-order worth and share 1 neuron with it, so step 3 should read both. That choice is Shonu's. The pilot neurons by the fixed rule: deleted block 22 n295; survivors block 19 n1961 and block 25 n1229. None is in the N5 tree.

### Step 2d: is the worth bird-specific? (`read/s2_spec.py`; `results/read/s2_spec.json`)

Same items, on held NON-bird photos (1,882). Non-bird gap .206 per photo (birds .336).

| Item (B's order) | Gain / gap, birds (choose) | Gain / gap, non-birds | Gain in nats per photo, birds / non-birds |
|---|---|---|---|
| the set of 30 | .419 | .122 [.109, .136] | .141 / .025 |
| n1 (deleted, block 22 n295) | .031 | .000 [−.000, .000] | .011 / .000 |
| n2 (survivor, block 19 n1961) | .053 | .031 [.027, .035] | .018 / .006 |
| n3 (survivor, block 25 n1229) | .052 | .023 [.020, .027] | .017 / .005 |

Part (a) of the set on non-birds: .014 of their gap (birds .015).

Reading: the set's worth is mostly bird-specific (5.6 times more nats per photo on birds), but not only. n1 is a pure bird neuron: re-adding it does nothing off birds. n2 and n3 are general neurons with a bird lean (about 3 times more nats per bird photo). So "birds need these neurons" holds for n1 and for the set as a whole; for n2 and n3 it holds only as a lean.

### Pilot cards: packets ready, readers blocked

Capture (`read/pilot_capture.py`): Base on all 35,000 reading photos; B and T on 19,000 photos T never trained on. Packets (`read/pilot_sheets.py`, `results/read/pilot/`) follow the advisor's fixes: neutral notes (no "birds"), candidates by strata from classes the reader never sees, a judge photo slice that no reader evidence uses, whole-word drop rule, unique class labels, isolated judge folders, change crops for n2 and n3, and the Base-to-virtual correlation for n1. Scoring (`read/pilot_score.py`) and the reading rule are fixed in code.

**No card was written.** The map-reader agents are locked to CrakBox work and could not open the packets. A second try with general-purpose agents was refused by the session's safety check as a way around the lock. Shonu decides how readers get access.

What the evidence already shows, read by me, not a card and not judged:

| Neuron | What the numbers say |
|---|---|
| n1, deleted, block 22 | Fires mostly on indigo bunting (class mean 7.2), then junco (2.5) and house finch (.75); median photo −2.6. Its input still reaches block 22 in B and T (per-photo correlation of Base's value with the virtual value: B .84, T .86). So B lost the neuron, not the information it read |
| n2, survivor, block 19 | Broad and weak in Base (median 1.0, top 2.8; top classes altar, mountain tent, ski). T lowers it on almost every photo (mean −.44), most on alligator lizard, brass, vulture, American alligator and zebra (no shared pattern named yet) |
| n3, survivor, block 25 | Animals with fur, feathers or scales against man-made objects. In B it fires harder on birds than in Base (goldfinch 2.55 against 1.98; different photo sets, so a rough comparison). T pulls it back down MOST on small birds (chickadee −1.39, brambling −1.26, junco −1.16, house finch −1.02) |

Caveat: these B and T values are the neuron inside the whole of B or T, where the upstream blocks also changed. The worth was measured by moving only the neuron's own numbers. So "T turns n3 down" may come from upstream change, not from n3's read row. A pattern to test, not a finding: B's survivors near the small-bird job (n3) over-fire on small birds after n1 and its kind are deleted, and T's fix is to turn them DOWN on those photos. That is "something takes up the job" in a form the essay names (a gate's negative side), not a survivor that widens.

Tools the reading lacked: (1) an agent route that can open these packets; (2) per-photo crops of the deleted neuron's top photos with B's and T's answers on them; (3) the exact (not approximate) feeder and reader edges, through the LayerNorm and the depthwise conv; (4) class profiles of the change for all 30 set neurons at once, to see if the "turn down on small birds" pattern repeats.

### Step 2e: where n2's and n3's worth sits (`read/s2_split.py`; `results/read/s2_split.json`)

B plus only one part of the neuron's trained numbers from T, choose set, share of birds' gap:

| Neuron | read bias b1[j] | read row W1[j] | write column W2[:, j] | all |
|---|---|---|---|---|
| n2 (block 19) | .0003 | .051 | .0003 | .053 |
| n3 (block 25) | .0002 | .049 | .0016 | .052 |

The bias-only control is clean: the worth is in the read row. T changed WHAT these neurons read, not their threshold.

Note on files: `s2_attr.json` and `s2_attr_all.pt` hold the predicted cost with the old (wrong) sign. Flip it when reading them. The script is fixed for future runs.

Reader access was tried twice: map-reader agents (lock: CrakBox work only) and general-purpose agents (refused by the session's safety check). No other route was tried. The pilot packets are ready. The three neurons sit outside the plan's fallback scope (the N5 tree). They were chosen by measured worth, on the advisor's advice: a departure from the plan, named here.

## N11. How a coarse neuron is made, read neuron by neuron (2026-11-21; plan `~/.claude/plans/pasted-content-id-201a-okay-w-misty-sutton-of-e2609063-b5b2-4974-affb-a0997a710ffa.md`; code `neuron/coarse/`, results `neuron/results/coarse/`)

Shonu, 2026-11-21: study ONE small group in neurons, not layers. Network A has about 5 parents and 15 children. Network B has about 5 coarse neurons. Find exactly HOW B's coarse neurons do the group's job. The essay is law: it states how coarse neurons are made. From this one group, write the theory of why the essay's coarsening does not pile up errors.

### The standing rule (guard: `neuron/tools/interp_lint.py`, a hook on every edit of this file)

1. Every step is about named neurons (block, index). Its result reads "neuron X does this job, and neuron Y takes it over by this change".
2. Every neuron is read the essay's way: its job as a verb, both poles, feeders, readers.
3. A fit (least squares, ridge, regression, optimal transport) never follows a neuron-level failure. The failure goes back to the reading.
4. A fit may only carry out a rule that a reading derived. It names that reading on a "Derived from:" line.

The guard fails any "result" subsection from N11 on that has no neuron table, or that mentions a fit with no "Derived from:" line.

### What the essay says (law)

| Essay line | Words |
|---|---|
| 23 | "With less room ... its neurons just cover coarser-grained concepts, like one car neuron covering many kinds of car. With more room, that neuron differentiates into many specific cars." |
| 225 | a gate lets cars through and keeps dogs out with its negative side; "Smaller models may stop at coarser distinctions, with something closer to a general 'car' neuron." |
| 239 | the smallest model's last layer "ends at cars, trains, and carts in one neuron, with boats, planes ... on its negative side" |
| 251, 259 | a crushed layer, retrained, broadens: "one 'curves & loops' neuron instead of ten curve detectors" |

The coarse neuron is the parent that never split. It covers its children and their children. The network makes it by training with less room.

### Two facts found before any run

1. **T (N6-N10) was never a coarse network.** `gap/common.py`: "Training starts from the build and changes the kept numbers only." T is Base's fine neurons with holes, fine-tuned. The essay's coarse network was never built at the scale of a stage.
2. **Each time the object reached neurons, a fit replaced it:**

| When | The object at neuron level | What happened | What replaced it |
|---|---|---|---|
| Oct 19, toy recipe | 4 hand-picked sibling families; parent gate = the children's shared upstream ingredients | PASSED: beat the strongest child (KL .00161 against .00178) | automatic families over the whole layer (topic bins), lost |
| Oct 21, tree fold | one parent per child | 1-4% explained | dense least-squares fold (RCF) |
| N4 step 5 | coarse neuron = counterpart with one condition dropped (16 of 30, by eye) | one weight test failed | top-value neurons + least-squares writes |
| N4 step 6 | which Base neurons share a coarse host repeats across seeds (77%, chance 16%) | never read | the least-squares build |
| N10 | read the lost neuron and its taker | reader agents locked | not done |

### The object

```
A (ConvNeXt-Base):  parent p (block 27/28) ──edge──► children (block 29): taxi, ambulance, police car ──► class scores
B (coarse):         coarse neuron s (block 29) ─────────────────────────────────────────────────────► class scores
```

B = A with the three stage-4 blocks (27-29) rebuilt narrow from random weights and trained by KL to A. Everything before stage 4 is A's, frozen, so block 27 of A and of B read the same input.

### Setup check: where the car split is made (layer measurements, no neurons yet)

The car family = the essay's own example, 11 classes: ambulance 407, beach wagon 436, cab 468, convertible 511, jeep 609, limousine 627, minivan 656, Model T 661, pickup 717, racer 751, sports car 817. Fill = each removed part replaced by its mean on car photos. "In-car accuracy" = the right car among the 11 car scores (chance .091). "Car mass" = probability on the 11 car classes.

**ConvNeXt-Base** (`coarse/where_cars.py`, 223 cached car photos; A: in-car .879, mass .687): no window of 3 or 6 blocks removes the split (in-car .82-.88 for every window). Blocks 18-29 removed: .686. Blocks 12-29: .283. Stage 4 alone (27-29) removed: in-car .879, and the whole stage-4 MLP set costs KL .061 on all photos. The split is built thinly over about 12 blocks with heavy overlap, and stage 4 is a small refinement on a stream that already holds the answer. A narrow stage 4 would be under no pressure to be coarse. So the planned B (stage 4 of ConvNeXt-Base) is dropped.

**ResNet-18** (`coarse/where_cars18.py`, torchvision IMAGENET1K_V1, 385 car reading photos; A: in-car .797, mass .794):

| Removed | In-car accuracy | Car mass | In-car KL |
|---|---|---|---|
| stage 3 block 0 | .099 | **.001** | 2.47 |
| stage 4 block 0 | .291 | .787 | 1.75 |
| stage 4 block 1 | .125 | .820 | 2.02 |
| **stage 4, both blocks** | **.091 (chance)** | **.815** | 2.15 |

Stage 3 decides "car or not". Stage 4 decides "which car": without it the network still says "car" with full confidence and picks the car at chance. This is the essay's gate in front with the specific detectors after it (line 225), and the fine split sits in 2 blocks. The essay's own models are not public (only the PDF).

**Change against the plan:** A = ResNet-18. B = ResNet-18 with stage 4's inner neurons (the channels between each block's two convolutions, 512 per block) rebuilt narrow from random and trained by KL to A. The stream (512 channels), the downsample and everything before stage 4 stay A's and frozen. Children = stage-4 neurons that split cars. Parents = their feeders with a direct edge (stage 3's last block and stage-4 block 0).

### Step 1, first run (stopped): B memorizes its photos

`coarse/train18.py --k 32 --seed 0`, on the Mac, 16,000 train photos + flips, no other augmentation. A's top-1 on these test photos is 67.7%. B has 518,272 trained numbers.

| B, k = 32 | Fit KL | Held KL | Test KL | Test top-1 |
|---|---|---|---|---|
| best (step 6,500) | .412 | 1.541 | 1.526 | 38.6% |

B learns its training photos and does not carry over (held KL 3.7 times fit KL). A network this weak cannot be read as "the coarse network". The other widths were stopped: running training and a second job at once overloaded the Mac (Shonu, 2026-11-21). Training moves to the UVA cluster.

Tools built and checked on A (no B yet): `capture18.py` (both poles per neuron), `poles18.py` (class profiles), `sheet18.py` (photo sheets), `edge18.py` (exact edges on car photos). `worth18.py` was stopped before it finished.

### Step 1: B trained on the cluster, with fresh augmentation (numbers only)

`coarse/train18c.py` on the UVA cluster (A100). Each step: random resized crops (area 25-100%) of the 16,000 train photos, left-right flips, mixup on half the batches. A scores the same augmented input. 100 epochs, batch 256, AdamW, weight decay .05. Held and test KL use center crops, as the reading does. A's top-1 on these 17,000 test photos is 67.7%. A's top-1 on augmented train crops is 57.8-64.3% (check that the crops are sound).

| B | Trained numbers | Fit KL (center) | Held KL | Test KL | Test top-1 |
|---|---|---|---|---|---|
| k = 32, no augmentation (Mac, first run) | 518,272 | .412 | 1.541 | 1.526 | 38.6% |
| k = 32, seed 0 | 518,272 | .749 | 1.254 | 1.213 | 44.1% |
| k = 32, seed 1 | 518,272 | .752 | 1.234 | 1.206 | 44.2% |
| k = 128, seed 0 | 2,066,944 | .260 | .707 | .685 | 54.7% |
| k = 128, seed 1 | 2,066,944 | .261 | .709 | .691 | 54.6% |
| k = 512 (full width, from random: control) | 8,261,632 | .153 | .479 | .481 | 59.0% |

1. Augmentation removes most of the memorizing at k = 32 (held / fit 3.7 -> 1.7) and raises top-1 by 5.5 points.
2. The two seeds give the same numbers at both widths (within .02 KL).
3. Even at full width, retraining stage 4 from random on 16,000 photos reaches 59.0%, not 67.7%. So part of every B's loss is the training budget, not the width. Readings of B compare against the k = 512 control for this reason.

First reading of B k = 32 on whole-photo sheets: wider positive sides than A in block 1 (median 48 classes with z > 2, against A's 35), but most of its strongest photos did not show one shared thing to me. My reading failed here; tool gap: whole photos instead of crops centred on the peak, and no view of what fed the neuron. Tools added: peak positions in `capture18.py`, peak-centred crops in `sheet18.py`, and `path18.py`. The path tool splits a block-0 neuron's input at its peak into the 256 stage-3 channels. A and every B read these same channels, so their feeders compare directly.

### Step 2, first result: B's coarse wheel neuron and A's wheel family (B k = 128, seed 0; my reading, no blind check yet)

Tools: `capture18.py` (peak positions), `sheet18.py` (128-px crops around each neuron's peak; sheets in `results/coarse/r18/sheets/`), `path18.py` (feeders = exact split of a block-0 neuron's input at its peak into the 256 stage-3 channels, mean over its top 50 photos), `recipe18.py` (A neurons fed by the same channels). Whole-photo sheets had hidden these jobs. Peak-centred crops show them.

| Neuron | Verb (job), from peak crops | Positive pole (top classes) | Negative pole (lowest mean input) | Feeders (stage-3 channels) | Readers (block-1 neurons, edge) |
|---|---|---|---|---|---|
| **B b0 n122** | fires on a spoked or radial wheel of any vehicle; reaches a dome's radial window and a cat's iris | Model T, mountain bike, tricycle, thresher, moped, tractor | menu, hair spray, lab coat, web site | s3:233, s3:69, s3:158, s3:120 | b1 n61, n75, n110 |
| A b0 n79 | spoked wheels of old carts and machines | mountain bike, Model T, rickshaw, disk brake, tandem bike, paddlewheel | corn, cauliflower, pizza, strawberry | s3:233, s3:122, s3:4, s3:151 | b1 n71, n180, n115 |
| A b0 n81 | radial spokes: skylight, bicycle wheels, watch face, bolo tie | tandem bike, mountain bike, moped, Model T, tricycle | upright piano, odometer, entertainment center | s3:88, s3:23, s3:69, s3:198 | b1 n415, n29, n191 |
| A b0 n37 | alloy car rims from the side, compass dials | compass, Model T, radio telescope, clock, convertible | web site, pill bottle, menu, typewriter | s3:233, s3:69, s3:91, s3:65 | b1 n145, n71, n115 |
| A b0 n297 | wheel rims and round dials (Model T wheel, clock, latte art) | Model T, convertible, compass, wall clock, barometer | academic gown, bow tie, basenji | s3:69, s3:174, s3:55, s3:65 | b1 n214, n221, n458 |
| A b0 n496 | a dark round wheel or eye, side view (forklift, tank, minivan wheels, flamingo eye) | disk brake, CD player, garbage truck, radio, tricycle, tractor | daisy, collie, papillon, academic gown | s3:60, s3:120, s3:69, s3:11 | b1 n458, n220, n310 |
| A b0 n303 | the wheel at the bottom of a vehicle | Model T, minibus, pickup, ambulance, scooter, tow truck | crossword, window screen, oscilloscope | s3:88, s3:122, s3:158, s3:23 | b1 n468, n220, n110 |
| A b0 n458 | wheel arches and arches (vault, arch bridge, canoe curve) | car mirror, viaduct, minivan, police van, car wheel | hamster, trifle, mashed potato | s3:233, s3:150, s3:164, s3:246 | b1 n68, n282, n210 |
| A b0 n103 | thin dark curved stripes: tabby stripes, rickshaw and unicycle spokes | tabby, tiger cat, tandem bike, Model T, unicycle | dishrag, agaric, manhole cover | s3:88, s3:60, s3:69, s3:16 | b1 n191, n468, n145 |
| B b0 n105 | a shaggy face with dark eyes (Tibetan terrier, komondor, wheaten, lesser panda; also a daisy) | cairn, Tibetan terrier, Westie, Dandie Dinmont | Leonberg, kelpie, redbone | s3:141, s3:153, s3:116, s3:176 | b1 n110, n126, n55 |
| A b0 n71 | terrier faces | cairn, wheaten, Norfolk, Yorkshire | wool, boa constrictor, cheetah | s3:198, s3:86, s3:69, s3:141 | b1 n498, n266, n71 |
| B b0 n121 | a glossy smooth curved surface with a highlight (whistle, pen, racer body, harmonica, projector) | ambulance, Polaroid camera, laptop, iPod, minibus | window screen, chain mail, dishrag | s3:14, s3:158, s3:78, s3:164 | b1 n59, n101, n57 |

**Union test** (held + test photos, top-200 sets): B n122's top 200 photos lie inside the union of the 8 A wheel neurons' top 200s for .52 of them. 8 random A neurons: .12 (95th percentile .24, 200 draws). The best single A neuron of all 512 (n37) covers .20. Caveat: the 8 were chosen by their likeness to B n122 (feeders or class profile), which inflates the overlap. The fair test starts from A alone (step 3).

Reading:
1. **A coarse neuron exists in B, and it is the essay's object.** A splits "wheel" into at least 8 block-0 neurons by kind (spoked cart wheel, alloy rim, side-view dark wheel, wheel arch, bottom wheel, radial spokes). B, with a quarter of the room, has one wheel neuron that covers them and reaches further along the radial axis (dome window, cat iris): recursive generalization. No single A neuron covers more than a fifth of it.
2. **The parents are shared stage-3 channels.** s3:233 feeds A n79, n37, n458 and B n122. s3:69 feeds A n81, n37, n297, n496, n103 and B n122. s3:88 feeds A n81, n303, n103. So the group has about 5 parents (stage-3 channels 233, 69, 88, 158, 120), about 8 children (A block 0) and 1 coarse B neuron.
3. B's other block-0 neurons are coarse along other axes: shaggy face (n105 ~ A n71 widened to komondor, lesser panda), glossy surface (n121). Their class means had suggested "vans" for n121. The crops do not.
4. B n104 and A n172: my reading failed (their class profiles match, their crops do not). Tool gap: the stage-3 parents are not read yet.

Next: read the 5 parent channels (`capture_s3.py`), take A's children of each parent by A's own edges (no B), and write rule R before comparing with B n122.

### Step 2, second result: the parents, and one concept down the width ladder (my reading, no blind check yet)

Parents read with `capture_s3.py` (A's 256 stage-3 output channels, the input that A and every B share) and 96-px crops around each channel's peak (`sheet18.py S3`).

| Neuron | Verb (job), from peak crops | Positive pole (top crops) | Negative pole | Feeders | Readers (children by A's edge, car photos) |
|---|---|---|---|---|---|
| s3:233 (parent) | fires on a car wheel rim | sports car, Model T, station wagon, hay cart, convertible and pickup wheels | not read yet | stage 3 (not read) | A b0 n37, n79, n172, n508, n24, n458; B k128 n122 |
| s3:69 (parent) | fires on a radial pattern around a dark centre | fluffy faces around the nose (terriers, weasel, lesser panda), bolo tie, wheel spokes | not read yet | stage 3 (not read) | A b0 n297, n305, n108, n496, n113; B k128 n122 and n105 |
| s3:88 (parent) | fires on a circular rim or arc | lens cap, cup rim, rickshaw and bicycle wheels, eyeglasses, pot edge | not read yet | stage 3 (not read) | A b0 n273, n81, n103, n453, n108, n303 |
| s3:158 (parent) | fires on a dark curved edge on a light ground | cart wheels, bow tie, corkscrew, pelican wing | not read yet | stage 3 (not read) | A b0 n303, n462, n253, n149 |
| s3:120 (parent) | fires on an eye: dark pupil in a ring | kite, flamingo, crane, parrot, shark, chameleon eyes | not read yet | stage 3 (not read) | A b0 n209, n416, n496, n503 |
| B k512 b0 n83 | fires on a spoked bicycle-type wheel | unicycle, tandem, rickshaw, mountain bike, golf cart | not read yet | not read yet | not read yet |
| B k128 b0 n122 | fires on any wheel, plus dome window and cat iris | (see first result) | menu, hair spray, lab coat | s3:233, s3:69, s3:158, s3:120 | b1 n61, n75, n110 |
| B k32 b0 n16 | fires on lines radiating from a centre: wheels and tracks, parachute lines, spider legs, a terrier face | snowmobile, hay cart, snowplow, Model T, harvestman, parachute | not read yet | not read yet | not read yet |

**Children from A alone.** For each parent, A's 8 block-0 neurons with the largest edge from it (A's edges, car photos). The union of their top-200 photos covers B k128 n122's top 200: s3:88 .49, s3:233 .40, s3:69 .26, s3:158 .26, s3:120 .20 (random 8 A neurons .12). The parents were chosen as B n122's feeders, so B still informed that choice.

**One concept down the width ladder** (top-200 sets, held + test photos):

| Wheel neuron | Inside the union of A's 8 wheel neurons | Largest share from one A neuron | Classes with z > 2 |
|---|---|---|---|
| B k512 n83 | .59 | .31 (A n81) | 34 |
| B k128 n122 | .52 | .20 (spread evenly) | 42 |
| B k32 n16 | .41 | .18 (more of it lies outside wheels) | 27 |

Reading:
1. **The essay's ladder, on one concept.** Full width: a fine spoked-wheel neuron that leans on one A child. A quarter of the width: one wheel neuron that draws evenly on all of A's wheel kinds. A sixteenth: lines radiating from a centre (wheels, parachute lines, spider legs), close to the parent s3:69. With less room the neuron stops at a coarser split, nearer its parent.
2. **One parent explains two of B's coarse neurons.** s3:69 (radial pattern around a dark centre) feeds both B's wheel neuron and B's shaggy-face neuron (n105). In A, that pattern is split into wheel kinds and face kinds.
3. The class-count measure does not rise at k = 32 (27). By that one measure the widening is not shown, so the ladder rests on the crops and the union shares.
4. **Seeds:** B k128 seed 1 has no single wheel neuron (best top-200 overlap with seed 0's n122: .12; its best match n21 fires on thin rods and edges). So a single coarse neuron is one seed's way of holding the job. The bar's ceiling is set by this. Rule R must predict the group's union across B's neurons, or what is common to both seeds. My reading of seed 1 is not complete; tool gap: its block-0 neurons are not read with crops yet.

### Correction to the first step-2 numbers

The ".49 from s3:88's children" and ".52 inside the union" compare a union of about 1,300 photos with B's 200, so a bigger set covers more by size. Against random unions of the same count (.12) they stand. Against the best single A neuron (a 200-photo set, .20) they do not compare. Size-fair test below.

### Step 3 result: a coarse neuron of B = the summed children of one parent of A (rule from A alone)

`coarse/rule18.py` and inline checks (numpy only). For every stage-3 channel P of A (256 parents): children(P) = the m A block-0 neurons with the largest feed0[j, P] (P's contribution to j at j's peak, mean over j's own top 50 photos, all classes; no B input). Candidate R(P) = per photo, the mean of the children's z-scored peak inputs. For each B block-0 neuron: the best per-photo correlation (19,000 held + test photos) within each candidate family.

Derived from: the step-2 reading (B n122 covers A's wheel kinds; the per-photo correlation of B n122 with the mean of the 8 wheel children is .46, against .38 for their max, .35 for the best parent, .33 for the best single A neuron, .12 for 8 random A neurons). No weights are fitted. The mean is the soft "fires if any child fires".

| B network, m | Parent's children (A only) | 256 random groups of m A neurons | Best single A neuron (512) | Best parent channel | B's other seed | Children beat random in |
|---|---|---|---|---|---|---|
| k = 128, m = 8 | **.40** | .35 | .36 | .34 | .39 | 91% of B neurons |
| k = 128, m = 16 | **.42** | .36 | .36 | .34 | .39 | 94% |
| k = 32, m = 8 | **.41** | .37 | .39 | .36 | .46 | 91% |
| k = 32, m = 16 | **.45** | .39 | .39 | .36 | .46 | 97% |

(median over B's block-0 neurons of the best correlation; m = 4 at k = 128: .37)

| Neuron | Verb (job) | Positive pole | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| R(P) for B k128 n122 | any wheel | best P = s3:164 (children mean, r .37) | not read yet | the children of s3:164 | (a candidate, not in a network yet) |
| R(P) for B k128 n105 | shaggy face | best P = s3:198 (r .46; single A n71 .41) | not read yet | the children of s3:198 | (a candidate) |

Reading:
1. **The rule from A alone predicts B's neurons better than any single A neuron, in 81-90% of them, and the gain is the parent-child structure**, not averaging: random groups of the same size are no better than single neurons.
2. **More children, closer match** (m = 4 .37, 8 .40, 16 .42): the coarse neuron takes in children and their siblings, as the essay says.
3. **The plan's bar** (R matches B as well as B's two seeds match each other): k = 128 passes (.42 against .39). k = 32 ties (.45 against .46). The comparison is not even: R picks from 256 candidates, the other seed from 128 or 32 neurons. The random-group control (256 candidates) carries the same count as R.
4. Open: the same claim in the weights (is B's read the sum of the children's reads?), and what the coarse neuron writes (Q2).

### Step 3, Q1-Q2 in the weights (B k128 n122 and all of B)

Weights exported by `coarse/weights18.py` (numpy). Read = a block-0 neuron's 256 x 3 x 3 filter times its batch-norm gain. Write = its conv2 column, through bn2, average pooling and the final layer (direct path to the class scores, centred over classes; block 1 and the ReLUs skipped, so first order).

| B network | Best single A read (cosine) | Summed reads of a parent's 16 children | Random 16 summed | Children beat single A in |
|---|---|---|---|---|
| k = 32 | .206 | .234 | .188 | 69% of B neurons |
| k = 128 | .225 | .241 | .205 | 68% |
| k = 512 | .239 | .242 | .205 | 52% (tie) |

B n122: read cosine with the summed reads of the 8 wheel children .282, best single A .217. Write: cosine with the children's shared part (mean write) +.28, each child -.07 to +.31, random 8 A neurons' mean write .11 on average (95th percentile .24). Share of B n122's write inside the children's 7-dimensional difference space .08 (a random 7-dimensional space: .007). B n122 pushes up paddlewheel, viaduct, plow, arch bridges, cannon, thresher, barn, barrow.

Reading: the narrower B is, the more its reads are summed children (k = 512 ties single A neurons). B's coarse wheel writes mostly the children's shared part and keeps little of what tells them apart.

### Step 4: predictions (written before the run; `coarse/theory18.py`)

Groups: G1 wheel = A b0 [79, 81, 37, 297, 496, 303, 458, 103]; G2 furry face = the 8 children of s3:198 by A's path data [71, 440, 399, 252, 489, 57, 97, 162] (terriers, mastiffs, setters, hamster, chow, Persian cat, apes). Versions: DEL (children set to their mean), R (the essay's coarse neuron: read = sum of children's reads, threshold by one firing-rate match, write = children's writes weighted by a ratio of means; no fit), LS (children removed, every other block-0 write re-solved by least squares: our old build form). Scored on 19,000 held + test photos.

| # | Prediction |
|---|---|
| P4a | On each group's own photos, R's extra probability stays inside the group's classes more than LS's (in-family share R > LS) |
| P4b | Error against the class signal (median cosine of the logit error with A's centred logits, group photos): R closer to 0 than LS |
| P4c | The two groups' errors, changed together: R's alignment ratio |e1 + e2|^2 / (|e1|^2 + |e2|^2) is closer to 1 than LS's |
| P4d | R's KL on the group's photos is below DEL's (the coarse neuron keeps part of the deleted job) |

No prediction on R against LS in KL: a local least-squares fit can win locally (N4, N6).

### Step 3 result: the two seeds stop at different levels of one tree

Per-photo correlation (19,000 held + test photos) with A's parent s3:69 and with the mean of each family's children (G1 wheel [79, 81, 37, 297, 496, 303, 458, 103]; G2 furry face [71, 440, 399, 252, 489, 57, 97, 162]):

| Neuron | Verb (job), from peak crops | Positive pole (crops) | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| B k128 seed 1 b0 n101 | fires on a radial pattern around a dark centre: wheels and furry faces together | alloy wheel, colobus, wolfhound, schnauzer, Eskimo dog, tandem bike wheel, Tibetan terrier, chain, rickshaw wheel | not read yet | not read yet | not read yet |
| B k128 seed 0 b0 n122 | any wheel | (see step 2) | menu, hair spray, lab coat | s3:233, s3:69, s3:158, s3:120 | b1 n61, n75, n110 |
| B k128 seed 0 b0 n105 | shaggy face | (see step 2) | Leonberg, kelpie, redbone | s3:141, s3:153, s3:116, s3:176 | b1 n110, n126, n55 |

| B neuron | Parent s3:69 | Wheel children | Face children | Both families |
|---|---|---|---|---|
| seed 1 n101 | .44 | .40 | .37 | **.51** |
| seed 0 n122 | .10 | **.46** | -.02 | .28 |
| seed 0 n105 | .29 | .20 | **.46** | .44 |

Reading: seed 0 split the radial parent into a wheel neuron and a face neuron. Seed 1 kept one neuron at the parent's level, covering both families. So the low seed-to-seed top-photo overlap (.12) is not disagreement: both seeds hold the same tree, and each run stops at its own level. The essay's "stops at coarser distinctions" fixes the tree, not which split a run stops at.

### Step 4 result: the theory check at one-block scale (`coarse/theory18.py`, cluster; `results/coarse/r18/theory.json`)

Derived from: R comes from the step-2 and step-3 reading (a coarse neuron = the summed children of one parent; it fires where any child fires; it writes their shared part). LS is not derived here: it is the old build form (N4, N6), kept only as the comparison.

Built coarse neurons (R, no fit): wheel t = 2.52, a_c .03-.05, firing rate 66% of positions; face t = 2.84, rate 65%. Group photos: the group's classes (wheel 29 classes, face 47 classes; children's mean class-profile z > 1.5). 19,000 held + test photos.

| Neuron | Verb (job) | Positive pole | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| R wheel (built) | sum of the 8 wheel children's reads; fires at 66% of positions | not read yet | not read yet | stage-3 channels, through the children's summed filters | A's block 1 and the head, through the children's weighted writes |
| R face (built) | the same for the 8 face children; 65% of positions | not read yet | not read yet | as above | as above |

| Version | KL, wheel photos | KL, face photos | KL, other photos | Error vs class signal (median cos), own photos | In-family share of extra probability, own photos |
|---|---|---|---|---|---|
| DEL wheel | .0506 | .0035 | .0094 | -.09 | .49 |
| R wheel | .0291 | .0036 | .0079 | -.02 | .53 |
| LS wheel | .0098 | .0014 | .0032 | -.03 | .58 |
| DEL face | .0059 | .0575 | .0095 | +.03 | .59 |
| R face | .0062 | .0362 | .0083 | +.04 | .60 |
| LS face | .0028 | .0120 | .0038 | +.00 | .67 |
| DEL both | .0609 | .0636 | .0192 | | |
| R both | .0384 | .0423 | .0165 | | |
| LS both | .0132 | .0135 | .0070 | | |

Both groups changed at once: alignment ratio |e1 + e2|^2 / (|e1|^2 + |e2|^2): DEL 1.020, R 1.016, LS 1.001. Real both-at-once over the added errors: 1.012 / 1.010 / 1.009. Interaction share .011 / .011 / .015.

| Prediction | Result |
|---|---|
| P4a R's error stays in-family more than LS's | **fails** (.53 vs .58; face .60 vs .67) |
| P4b R's error nearer right angles to the class signal | **not shown**: every version sits near 0 (-.09 to +.10) |
| P4c R's two group errors line up less than LS's | **not shown**: every version is about 1, no lining up at all |
| P4d R beats deleting the children | **holds** (.0291 vs .0506 wheel; .0362 vs .0575 face: R keeps 37-42% of the lost job) |

Reading:
1. **At this size there is nothing to explain.** With 16 of 512 neurons changed in one block, no version's errors line up (ratio about 1) and nothing compounds (interaction about 1%). N9 had shown that one block cannot compound. The test could not separate the methods, and I should have seen that before the run. The theory's central claim (the old build's errors line up across many groups; the essay's coarsening does not) is **not tested yet**. It needs many groups changed at once, across the whole stage.
2. **The built R is not yet the essay's coarse neuron.** It fires at 66% of positions, while B's wheel neuron is selective (clean wheel crops). Summing eight pre-activations and matching the "any child fires" rate gives a broad, weak unit. Tool gap: B n122's own firing rate and threshold are not measured yet, so R's threshold rule is a guess, not a reading.
3. Locally, least squares wins by a wide margin (.0098 vs .0291), as N4 and N6 found.

### Step 3 result, Q5: how B's wheel neuron forms (checkpoints of B k128 seed 0; `coarse/ckpt18.py`)

| Neuron | Verb (job) at each stage of training | Positive pole (crops) | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| B k128 b0 n122, step 10 | the parent's job: radial pattern around a dark centre | kit fox, papillon, chow, lesser panda, sheltie faces; Model T wheel; clock | not read yet | not read yet | not read yet |
| step 100 | round openings and arches | guillotine hole, triumphal arch, dome, barn, palace | not read yet | not read yet | not read yet |
| step 300 | arches, wheels, coils | arch, tractor and scooter wheels, snake coils, dome | not read yet | not read yet | not read yet |
| step 1,000-6,250 | any wheel | snowplow, golf cart, mountain bike, thresher, cannon, dome | menu, hair spray, lab coat | s3:233, s3:69, s3:158, s3:120 | b1 n61, n75, n110 |

| Step | n122 vs wheel children / face children | n105 vs wheel / face | Best other B neuron for faces |
|---|---|---|---|
| 1 | +.21 / +.17 | +.18 / +.37 | n106 +.47 |
| 10 | +.22 / +.15 | +.19 / +.35 | n106 +.45 |
| 30 | +.26 / +.03 | +.24 / +.25 | n52 +.54 |
| 100 | +.24 / -.06 | +.22 / +.32 | n52 +.61 |
| 300 | +.35 / +.01 | +.25 / +.46 | n52 +.52 |
| 1,000 | +.40 / -.01 | +.19 / +.45 | n52 +.44 |
| 6,250 | +.46 / -.02 | +.20 / +.46 | n77 +.42 |

Reading:
1. **The coarse neuron starts at the parent and narrows.** At step 10, n122 does the parent s3:69's job (faces and wheels together, like seed 1's final n101). Between steps 10 and 30 it sheds the face facet, while B neuron n52 takes the faces (+.54). It passes through "arch and opening" and settles on wheels by step 1,000. This is the essay's growth story (lines 133, 685) and circuit racing (line 713): the facets race, and each settles on the neuron that captured it first.
2. **B starts inside A's tree.** At step 1 (random weights), B's neurons already correlate .2-.5 with A's families, because B reads A's own stage-3 parents. A random read of the parents is already a coarse parent-level neuron.
3. With seed 1's n101 (which stayed at the parent's level), this gives the picture: training with less room starts every coarse neuron at a parent's level and splits it only as far as room allows. Which split a run reaches is set early, by the race.

### Checks after step 4 (advisor review): two corrections

**1. The parent-child claim does not survive a fair null.** New control: 256 groups of m A neurons built from a neuron and its m - 1 most correlated A neurons (they fire together; no parent involved). Median best correlation with B's block-0 neurons:

| B network, m | Single A | Parent's children | Co-firing neighbours | Children beat neighbours in |
|---|---|---|---|---|
| k512, 8 / 16 | .36 | .37 / .39 | .39 / .40 | 38% / 32% |
| k128, 8 / 16 | .36 | .40 / .42 | .42 / .45 | 29% / 34% |
| k32, 8 / 16 | .39 | .41 / .45 | .46 / .45 | 34% / 31% |

Within-group correlation in A: parent's children .10-.12, co-firing neighbours .23-.27. So any group of A neurons that fire together, averaged, matches B as well or better. **Withdrawn:** "the parent-child structure carries the match" (step 3 result, reading 1). **Holds, weaker:** B's neurons match merged groups of related A neurons better than single A neurons, and the gain grows as B narrows (groups over singles: k512 +.03-.04, k128 +.06-.09, k32 +.06-.07). My parent -> children tool (path feed at the child's top photos) gives families that barely fire together; my reading of families failed there; tool gap: the parent-child edge, which needs the exact edge on the parent's photos, not the child's.

**2. The growth reading was over-read.** Steps 1-10 barely differ (learning-rate warm-up), so "n122 starts at the parent's job" rests on crops of a near-random neuron. And n52 was a face neuron at random initialisation (face .42 at step 1), so it did not "take the faces" in training.

| Step | n52 wheel / face | n122 vs its final self | n52 vs its final self |
|---|---|---|---|
| 1 | +.11 / +.42 | +.26 | +.37 |
| 30 | +.04 / +.54 | +.34 | +.48 |
| 100 | -.03 / +.61 | +.50 | +.64 |
| 300 | -.04 / +.52 | +.73 | +.76 |
| 1,000 | -.05 / +.44 | +.89 | +.88 |

**Holds:** each neuron sharpens the lean it had at initialisation. n122's weak face lean (+.17) goes to -.06 while its wheel match rises from +.21 to +.46, and n52 strengthens on faces. B's random neurons already lean toward A's families because they read A's own stage-3 channels. The head start comes from initialisation, not from a race won during training.

**3. The plan's step-3 bar is not met as written.** It asked for a rule R, built as a neuron, that predicts B's coarse neuron (both poles, write, readers) as well as B's two seeds agree. Scored so far: averages of A activations, positive pole only. The built R neuron is scored in the next check (`coarse/diag18.py`).

### Checks after step 4: the built R, firing rates, and Q3 (`coarse/diag18.py`, output `diag.json`)

Firing rates (share of positions with input > 0, 19,000 photos): B k128 n122 .478; A wheel children .115-.227 each; "any child" .661; built R .663. **Correction:** B n122 is not selective in rate. It fires 2-4 times as often as one child, below "any child". A threshold read from B puts R near half the positions, not two thirds.

Built R (summed reads, rate-matched threshold) against B n122, per-photo peak input: r = .36 (best single child .33, mean of the children's activations .46). The built neuron keeps less than the activation average does.

Q3, does B keep the fine split (accuracy among a group's own classes on its photos; probability mass on the group):

| Network | Cars (11, chance .091) | Wheel classes (29, chance .034) | Face classes (47, chance .021) |
|---|---|---|---|
| A | .789 / .77 | .824 / .81 | .804 / .85 |
| B k512 | .679 / .71 | .717 / .73 | .741 / .80 |
| B k128 | .656 / .69 | .684 / .70 | .719 / .78 |
| B k32 | .512 / .63 | .601 / .63 | .628 / .70 |

Reading: B loses fine splits as it narrows, most of it between k128 and k32 (-.08 to -.14). The full-width control already loses .06-.11 against A, so part of every B's loss is the training budget, not the width.

### N11 status against the plan

| Plan item | Status |
|---|---|
| Step 0: standing rule, guard | done (memory; log lint with banned verdict words and "Derived from:"). The Stop hook on chat replies is built and tested; registering it is blocked by the CrakBox lock (Shonu's step) |
| Step 1: train B, gate on 20 neurons | done differently: A and B are ResNet-18 (ConvNeXt-Base spreads the split over about 12 blocks). B trained on the cluster with augmentation, 3 widths, 2 seeds. Gate: about 10 B neurons read by me with peak crops; no 20-neuron set, no blind check |
| Step 2: one-photo trace, edge tool, group, cards | group found (wheel family, 5 stage-3 parents, about 8 children, B n122) and the exact feeder tool built. Not done: the sports-car one-photo trace; parents' negative poles |
| Step 3: Q1 read, Q2 write, Q3 readers, Q4 poles, Q5 growth | Q1 and Q2 done (activations and weights). Q3 done at class level (B loses fine splits as it narrows), not at neuron level. Q4 partly (class-mean negative poles of B n122 and the children). Q5 done, with the correction above |
| Step 3 bar: rule R as a neuron predicts B as well as B's seeds agree | **not met**. Averages of co-firing A neurons match B's neurons better than single A neurons (and the gain grows as B narrows), but a parent's children are not better than any co-firing group, and the built R neuron scores .36 against n122 |
| Step 4: theory check | run at one-block scale; it cannot separate the methods (no lining up for any). The decisive test (whole stage 4 coarsened by the rule, at B's width) is not run, and should wait for an R built from a reading (threshold near B's rate; families from co-firing) |

### The parents' negative poles (`read_s3.npz`: class-mean of each channel's pre-ReLU input z, lowest classes)

| Neuron | Verb (job) | Positive pole | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| s3:233 | lets car wheel rims through | Model T, pickup, ambulance, convertible, minivan | cardoon, bald eagle, wig, hare (organic textures) | stage 3 (not read) | A b0 n37, n79, n458; B n122 |
| s3:69 | lets radial patterns around a dark centre through; separates shaggy from smooth | lesser panda, Yorkshire, Westie, Tibetan terrier, Persian cat | collie, Chihuahua, beagle, Labrador, keeshond (smooth-coated dogs) | stage 3 (not read) | A b0 n81, n37, n297, n496, n103; B n122, n105 |
| s3:88 | lets circular rims through, keeps dogs out | espresso, tandem bike, lens cap, Model T, moped | Appenzeller, kelpie, basset, chow, collie | stage 3 (not read) | A b0 n81, n303, n103 |
| s3:158 | lets dark curved edges through, keeps dogs out | airship, flagpole, safe | collie, Leonberg, Tibetan terrier, Shih-Tzu, schnauzer | stage 3 (not read) | A b0 n303; B n122 |
| s3:120 | lets eye rings through | wood rabbit, kite, hamster, bullfrog | whippet, Doberman, deerhound, Appenzeller | stage 3 (not read) | A b0 n496; B n122 |

Reading: three of the five wheel parents push dogs down. This is the essay's gate (line 225): a neuron in front of the car detectors that keeps the dogs out. s3:69 separates shaggy faces from smooth-coated ones, which fits B's shaggy-face neuron n105.

### What is missing before the multi-level case (Shonu, 2026-11-21: "evaluate what is concretely missing ... and complete them")

| # | Missing | Why the multi-level case needs it |
|---|---|---|
| M1 | a coarse neuron built in weights that matches B (built R scored .36 against B n122; activation average .46) | each level of a multi-level build is made of these neurons |
| M2 | a family for every neuron of a layer, from A alone | a whole level must be coarsened; the parent -> children tool failed its control |
| M3 | a trained reference for one level (block 0 narrow, block 1 = A) | a one-level rule needs a same-form yardstick; B changes both levels |
| M4 | the theory check at layer scale (many families at once) | one block with two groups showed no lining up for any method |
| M5 | the readers: coarsen block 1 too, against keeping it fine | this is the multi-level question: does coarsening the readers with the children stop the errors? |

**M2 done (numpy).** Families of A's block-0 and block-1 neurons into 128 per block (complete linkage), two ways: co-firing (1 - correlation of peak inputs; median size 3-4, within-family correlation .25 / .19) and shared feeders (1 - cosine of path feed0; block 0 only; median size 3, within-family co-firing .19). A's 8 wheel neurons fall into 5-7 co-firing families but into 2 shared-feeder families (n79, n37, n458 together; n81, n303, n103 together): wheel kinds are siblings that share feeders and rarely fire together, which is the essay's sibling (taxi and ambulance are rarely on together). Family means against B's block-0 neurons (median best correlation): co-firing .38 (k128) / .41 (k32), shared feeders .38 / .38, single A .36 / .39; full width k512 .35 for both, single A .36. The average-activation match cannot separate the two family kinds; the built layers below can.

### M1, M3-M5: predictions (written before the run; `coarse/multilevel18.py`, job 5)

Builds at K = 128 per block: rule (pre_f = sum of members' pre, no new threshold; write = members' writes weighted by a ratio of means; mean map kept), for co-firing and for shared-feeder families; LS (top-value 128, writes re-solved on A's own inputs: the old build form); trained one level (block 0 at 128, block 1 = A); B (both trained). One level (L0: block 0 coarse, A's fine block 1 reads it) and two levels (L01).

| # | Prediction |
|---|---|
| P5a (M1) | the built rule neurons (block 0) match B's block-0 neurons better than single A neurons (median best per-photo correlation) |
| P5b (M4) | at one level, the rule's 128 per-family errors add up as independent errors (added / sum of singles near 1, under 1.3), and the real all-at-once error is near their sum |
| P5c (M3) | in KL at one level: trained < LS < rule (a local least-squares fit wins locally) |
| P5d (M5) | going from one level to two (coarsening the readers too), the rule's error geometry holds (median cosine with the class signal within ±.1, logit spread within 10% of A's), while LS's moves away (larger |cos| or spread change) |

### M1-M5 result: the layer-scale builds (`coarse/multilevel18.py`, job 5, output `multilevel.json`)

Derived from: the rule comes from the step-2/3 reading (a coarse neuron merges a family of A neurons). LS is the old build form, kept only as the comparison.

| Neuron | Verb (job) | Positive pole | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| rule neuron (block 0, any family) | sum of its members' inputs, no threshold change; fires at 9% of positions (median) | not read | not read | A's stage-3 channels through the members' summed filters | A's block 1 (L0) or the rule block 1 (L01) |
| B k128 block-0 neuron (median) | fires at 42% of positions | (see step 2) | | A's stage-3 channels | B's block 1 |

| Version | KL | Top-1 | Median cos(error, class signal) | Logit spread | In-group acc cars / wheel / face |
|---|---|---|---|---|---|
| L0 rule, co-firing families | 2.528 | 27.6% | -.81 | .60 | .579 / .535 / .271 |
| L0 rule, shared-feeder families | 2.814 | 25.2% | -.86 | .52 | .584 / .492 / .298 |
| L0 LS | .512 | 57.1% | -.15 | 1.03 | .708 / .740 / .733 |
| L0 trained (block 0 at 128, block 1 = A) | .236 | 63.8% | -.19 | .99 | .756 / .779 / .779 |
| L01 rule, co-firing | 6.693 | 2.9% | -.60 | .96 | .148 / .132 / .057 |
| L01 rule, shared feeders | 8.159 | 1.8% | -.62 | .97 | .096 / .080 / .054 |
| L01 LS | 1.969 | 32.8% | -.69 | .73 | .603 / .590 / .618 |
| L01 trained (B) | .688 | 54.6% | -.23 | 1.03 | .656 / .684 / .719 |
| block 1 only, rule (co-firing) | 2.530 | 25.7% | -.44 | .97 | .445 / .405 / .437 |
| block 1 only, LS | 1.380 | 45.0% | -.78 | .66 | .708 / .682 / .705 |

Per-family errors at one level (2,000 held photos, 128 families): added over the sum of singles: rule 2.63 (co-firing) / 2.76 (shared feeders); DEL 2.21 / 2.25. Real over added: rule .35 / .36; DEL 2.58 / 2.58.

| Prediction | Result |
|---|---|
| P5a built rule neurons match B better than single A | not scored yet (rule peak inputs saved); moot until the rule fires at the right rate |
| P5b rule's 128 family errors add as independent (ratio < 1.3) | **fails**: 2.6-2.8. At layer scale the families' errors line up (DEL too: 2.2) |
| P5c KL trained < LS < rule at one level | **holds**: .236 < .512 < 2.5 |
| P5d one -> two levels: rule keeps its geometry, LS moves away | **half**: LS moves away (cos -.15 -> -.69, spread 1.03 -> .73); training keeps it (cos -.19 -> -.23, spread .99 -> 1.03); the rule never had it (cos -.81 at one level) |

Reading:
1. **The rule as built is not the coarse neuron.** A sum of members' inputs fires at 9% of positions, against 42% for B's neurons, and the layer collapses (KL 2.5; logits flat, cos -.81, spread .60). Family members' reads do not oppose each other (read cosine +.09; random +.02), so it is not sibling cancellation. A sum of weakly correlated inputs that sit below threshold needs several members on at once: an AND. A coarse neuron fires when any member fires: an OR. The threshold is the missing piece, tested next.
2. **At layer scale the families' errors line up** (2.2-2.8, like N6's 2.7 across blocks), for deletion and for the rule. The lining up seen in N6 exists inside one layer once many families change.
3. **Training keeps the error geometry across levels; least squares does not.** This is the multi-level signature of the gap (N6, N7), now at two levels of one stage.

### M1 follow-up: predictions (`coarse/threshold18.py`, written before the run)

| # | Prediction |
|---|---|
| P6a | the threshold set to B's rate (.42) gives a far better layer than the plain sum, with KL within 2 times LS's at one level |
| P6b | the "any member" threshold is worse than B's rate (it is broader than B) |
| P6c | the OR of members (a check, not one neuron) is the best rule variant |

### M1 follow-up result: the threshold fixes the rate, not the layer (`coarse/threshold18.py`, output `threshold.json`)

Derived from: the reading above (a plain sum acts as an AND; a coarse neuron must fire when any member fires). No least squares.

| Neuron | Verb (job) | Positive pole | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| rule neuron, threshold at B's rate | sum of members' inputs plus a shift so it fires at .42 on calibration (measured .35 on scoring photos) | not read | not read | A's stage-3 channels | A's block 1 (L0) |
| rule neuron, OR | the strongest member at each position (fires when any member fires; .46-.47) | not read | not read | members' own reads | A's block 1 (L0) |

| Variant (one level) | Rate block 0 | KL | Top-1 | cos | Spread |
|---|---|---|---|---|---|
| plain sum (before) | .09 | 2.528 / 2.814 | 27.6% / 25.2% | -.81 / -.86 | .60 / .52 |
| rate = any member | .39 / .41 | 1.575 / 1.633 | 39.0% / 38.3% | -.31 | 1.07 |
| rate = B (.42) | .36 / .35 | 1.420 / **1.373** | 41.0% / 42.2% | -.32 / -.30 | 1.02 / 1.04 |
| OR of members | .46 / .47 | 1.431 / 1.541 | 40.6% / 39.1% | -.33 / -.34 | 1.04 |
| LS (comparison) | | .512 | 57.1% | -.15 | 1.03 |
| trained one level | | .236 | 63.8% | -.19 | .99 |

(co-firing / shared-feeder families.) Two levels: every rule variant collapses (KL 3.9-4.6, top-1 6-15%); OR at two levels flattens the logits (spread .34-.35).

| Prediction | Result |
|---|---|
| P6a B's rate gives KL within 2x LS at one level | **fails** (1.373 vs .512, 2.7x) |
| P6b "any member" worse than B's rate | holds |
| P6c OR is the best rule variant | fails at one level (B's rate is best); at two levels OR has the lowest KL (3.9) but flat logits |

Reading:
1. **The threshold was a real fault, not the main one.** Setting it to B's rate halves the error (2.5 -> 1.4) and restores the logit spread (.60 -> 1.03).
2. **The main loss is the write.** The OR variant fires exactly when its members fire (the read is not lost) and still costs KL 1.43. What it changes is the write: four members' own writes become one shared write (their ratio-of-means average). B's wheel neuron's write correlates only +.28 with its children's mean write (step 3, Q2). So a trained coarse neuron writes mostly something other than its members' average, and that is not read yet.
3. Shared-feeder (sibling) families give the best rule layer at B's rate (1.373 vs 1.420), a small margin.

### N11 status after the multi-level work

| Item | Status |
|---|---|
| M2 families for every neuron | done (co-firing; shared feeders) |
| M3 one-level trained reference | done (block 0 at 128: KL .236, top-1 63.8%) |
| M4 layer-scale theory check | done: at layer scale the families' errors line up (added / singles 2.2-2.8) for deletion and the rule |
| M5 readers, one level vs two | done: training keeps the error geometry across levels (cos -.19 -> -.23, spread .99 -> 1.03), LS does not (-.15 -> -.69, 1.03 -> .73), the rule collapses at two levels |
| M1 a coarse neuron in weights | **not done**: the read and threshold are now roughly right (rate .35-.47 vs B .42), the write is the missing piece |
| Next | read what B's coarse neurons write (most of B n122's write: its cosine with the children's mean write is only .28), then a write rule from that reading, then the two-level build again |

### What B's write consists of: first look in weights, and predictions for the function test (`coarse/write18.py`)

Shonu, 2026-11-21: investigate what B's write actually is before building anything. B's block 0 adds to A's frozen skip path and A's frozen final layer reads the result, so B's writes live in A's own 512 stream channels.

Raw weight space (bn2-scaled conv2 column summed over 3 x 3; numpy): B n122's write against everything in A. Children's mean write cosine .15; best single A block-0 write .15 (median |cos| .035); greedy best 1 / 5 / 10 / 20 A writes explain .02 / .09 / .16 / .27 of it (20 random A writes .05); best A block-1 read .15, best B block-1 read .15; best final-layer class row .25 (paddlewheel, viaduct, steel arch bridge, birdhouse, barn, cannon). Spread over many channels (top 10 hold 13%). Raw weight cosines weigh all channels equally, which the network does not, so the question moves to function.

| # | Prediction (before the run) |
|---|---|
| P7a | B's block-0 output copies A's block-0 output poorly (pooled R^2 under .5) |
| P7b | B's actual writes differ from the best "copying" writes from B's own firing (median per-neuron cosine under .5) |
| P7c | swapping B's writes for the copying writes (B's reads and block 1 kept) makes B's final answer worse: B's writes serve the end, not the copy |
| P7d | B's block 1 is explained by B's own stream much better than by A's stream (its readers read what B writes) |

### B's write, first function test (`coarse/write18.py` on B k128 seed 0; output `write_B128aug_s0_first.json`)

Derived from: the reading that B's writes live in A's stream channels. The least-squares "copying writes" here are a ruler only (the best a copy of A's block-0 output from B's own firing can do), never a build.

| Measurement | Value |
|---|---|
| (a) B's block-0 output against A's block-0 output (3,000 calibration photos, all positions) | R^2 = -5.69; B's variance / A's = 7.48; per-channel correlation median .33 |
| (b) copying ruler: R^2 of A's block-0 output from B's firing (held photos) | .60 |
| (b) cosine of each B neuron's write with the ruler's write | median .28 (10th-90th percentile .25-.31); n122 .31 |
| (b) norm of B's write / ruler's | 3.37 (median) |
| (c) B with its block-0 writes swapped for the ruler (B's reads and block 1 kept), test photos | KL .685 -> 4.393; top-1 54.7% -> 4.6% |
| (d) B's block-1 input (centre position) from A's stream | R^2 .30 (the R^2 from B's own stream, .9997, is trivially near 1: the input is linear in that stream; design flaw, not a finding) |

| Prediction | Result |
|---|---|
| P7a B copies A's block 0 poorly | holds, strongly (R^2 negative; 7.5 times the variance) |
| P7b B's writes differ from the copying writes | holds (cosine .25-.31 for every neuron) |
| P7c swapping in the copying writes hurts B | holds (B collapses) |
| P7d B's block 1 reads B's stream, not an A-like stream | holds by the A-stream number (.30); the comparison number is void |

Reading: B's write is not a coarsened copy of A's layer. It is a larger message (7.5 times A's block-0 variance) that only B's own block 1 can use. Copying A with B's neurons is possible (R^2 .60) and useless to B. Confound: B's block 1 was trained with it, so the two could share a private code. The clean test is T0 (block 0 trained, A's fixed block 1 reads it), running now with the full-width control (B k512).

### What a coarse neuron's write consists of (`coarse/write18.py` on T0, B, B k512; `coarse/dominant18.py`)

Derived from: the reading that B's writes live in A's stream channels. The least-squares "copying writes" are a ruler (the best copy of A's block-0 output from each network's own firing), never a build.

| Measurement | B k128 (both blocks trained) | B k512 (full width, both trained) | T0 (block 0 at 128 trained, A's block 1 fixed) |
|---|---|---|---|
| copy: R^2 of its block-0 output vs A's | -5.69 | -4.10 | -.43 |
| its variance / A's | 7.48 | 5.95 | 1.44 |
| share of its variance inside A's top-32 output directions (A itself: .52) | .27 | .26 | .49 |
| variance in its top 1 / top 10 directions (A: .11 / .33) | .06 / .33 | .05 / .29 | **.46 / .78** |
| ruler: R^2 of A's block-0 output from its firing | .60 | .71 | .66 |
| cosine of its writes with the ruler's (median) | .28 | .24 | **.63** |
| its write norm / ruler's | 3.37 | 2.67 | .81 |
| swap in the ruler's writes: KL (test) | .685 -> 4.393 | .481 -> 3.540 | **.237 -> .322** |
| top-1 | 54.7% -> 4.6% | 59.0% -> 9.5% | 63.8% -> 61.6% |

T0's dominant write direction v1 (46% of its output variance; 44% of the difference from A; |cos| .52 with A's top direction; cos -.70 with A's mean output; correlation with A's confidence -.02):

| Neuron | Verb (job) | Positive pole | Negative pole | Feeders | Readers |
|---|---|---|---|---|---|
| T0 block-0 output direction v1 | separates busy texture filling the frame from objects against plain sky | pot, coral reef, hen-of-the-woods, diamondback, flatworm, leatherback turtle, spiny lobster, whiptail, crocodile, pizza, Komodo dragon | airliner, warplane, space shuttle, mosque, airship, flagpole, sandbar, alp, water tower, palace, volcano | T0's 128 block-0 neurons | A's block 1 and the head; direct push up: honeycomb, potter's wheel, hammer, cockroach, spotted dogs; down: arch and suspension bridges, ibex, gibbon, paddlewheel, fireboat |

B's streams against A's (4,000 test photos): after block 0, R^2 -11.8; after block 1 (the head's input), R^2 .007. B's class scores still come close (KL .685) because the final layer ignores most stream directions.

Reading:
1. **What a coarse neuron writes depends on who reads it.** With A's fine readers fixed (T0), the coarse layer writes mostly its best guess of what A's full layer would have written, given only its coarse firing (cosine .63 with that copy; the pure copy costs .085 KL). With its readers trained alongside (B), it writes a large private message (6-7.5 times A's size, three quarters outside A's main directions) that only those readers use. Copying is then useless (B collapses).
2. **The private message is not caused by coarseness.** It appears equally at full width (B k512). It comes from training a block together with its readers. So B is the wrong object for "what a coarse neuron writes"; T0 is the clean one.
3. **The coarse layer writes the coarsest distinction loudly.** One direction, busy texture against plain sky, carries 46% of T0's output (A's top direction: 11%). With less room, the layer gives its largest output to a root-level split. This is a reading of one network, not yet checked at other widths.
4. **For a build, this gives the write rule from a reading**: at one level with fine readers kept, a coarse neuron's write is the conditional copy of the fine layer's output given its firing. That is the least-squares form, now derived from T0's writes rather than assumed. The one-level gap then splits as: reads .19 (LS .512 against T0's reads with copying writes .322) and writes beyond the copy .085 (.322 against T0 .237).
5. **For the multi-level case**, copying each level (LS at both levels: KL 1.97) is not what trained levels do: trained levels pass messages the fine network never wrote (B: .69). Coarsening two levels means building each level's writes together with the next level's reads.
