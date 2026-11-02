# Night report, 2026-10-20: essay concepts applied to coarsening

Plan: `~/.claude/plans/we-ve-been-plugging-away-snazzy-duckling.md`. Full log: `CHILDREN-LOG.md`, section
"GATE FAMILIES". Idea catalog: `ESSAY-IDEAS.md` (I1-I13). All numbers are dCE (added next-word loss) on SmolLM2-135M.

## The result in one line

A no-gradient method, RCF (route-capture fold), reaches the 20-minute trained parents on the keeper's sealed layers
and sealed text, for prose and code. It works one layer at a time and in a 10-layer block. It does not yet help
when every layer of the model is coarsened at 50% width.

## What RCF does

1. Keep the strongest K neurons of layer L whole (the trunks).
2. The removed neurons' output becomes "residual error".
3. Let the routes that already fire just before them capture it: layer L's attention heads and layer L-1's neurons.
4. One least-squares solve finds how much each route should write. The result is added to weights that already
   exist (layer L's attention output matrix and layer L-1's write matrix). One coarse model needs no extra numbers.

Axioms it comes from (the essay as law):

| Essay concept | How RCF uses it |
|---|---|
| Circuit racing: "established routes have a head start in capturing residual error". | The closest routes capture the removed children's work. |
| Umbrella to bee: new behavior is "an off-ramp between two routes the network already has". | RCF adds linear off-ramps from existing routes. No new gate is built, so the "one gate cannot do an OR" wall never applies. |
| The "gate in front" already exists upstream. | The coarse version of a child is written by its upstream route, not by a new unit next to it. |

## Evidence

### One layer at a time (K = 48 is 3% of the layer, K = 384 is 25%)

| Test | Result | File |
|---|---|---|
| G11, 6 fresh dev layers, K48, mixed calibration | Better than the cut on 6 of 6 layers, prose and code. Recovers 11-38% of the cut's damage on prose, 29-44% on code. | `results/clean-g10.json` (`_g11`) |
| G11, K384 | Better on 5 of 6 (prose) and 4 of 6 (code); the rest tie. | same |
| Keeper, sealed layers L9 and L21 | K48: better than the cut AND reaches the 20-min trained parents, prose and code, on both layers. K384: better on 3 of 4, even on 1. **Caveat:** the keeper's cut and trained parents use prose-only calibration, RCF uses prose + code. The PROSE buckets are a fair comparison. The CODE buckets mix method and calibration (see the correction run below). | ledger in `~/Cale-holdout` |
| Keeper correction run (RCF vs a cut with the same mixed calibration) | **Better in all 8 cells** (L9, L21 x K48, K384 x prose, code). The one earlier "even" cell (L9 K384 prose) is now better. Mixed calibration helps the cut on code and hurts it on prose, so the confound was real. Budget-exempt: 1 validation query left. | ledger |
| One layer at 50% width (K768) | L16 recovers 22%, L22 12%. | `clean-g10.json` (`_k768`) |

### Many layers at once

| Test | Result | File |
|---|---|---|
| Block of layers 11-20, K192 (12.5% width) | Gains add up: prose +0.731 to +0.589, code +0.593 to +0.448 (19% / 24% recovered). | `results/clean-g12.json` |
| Block 11-20, K768 (50%), more calibration text | Gain grows with calibration: 32+32 blocks tie, 128+128 blocks recover 8% (prose) and 6% (code). | same |
| Whole model, 25 layers, 50% width, 128+128 blocks | RCF everywhere: tie. RCF only on layers 10 and up: prose better by 0.010 (about 1.3%), code tie. **Exploratory:** "layers 10 and up" was chosen after seeing earlier runs. | same |
| Whole model, drift-correcting target (G12b) | At 32+32 blocks (starved): no gain. At 128+128 blocks: a trade-off, not a fix. Prose better (-0.040 vs the plain cut, 5.4%), code worse (+0.036). | same |

### Dead ends tonight (each is "this method, as built, failed", not "the idea is wrong")

| Idea | What happened |
|---|---|
| I1 shared-gate merge | SmolLM2's gates are not shared enough. Median tie error 0.62-0.68. |
| I2/I3 judgment gate | Lost on a copy test, which I now think is the wrong test for coarse units. |
| I6 variance collapse | Real, but harmless. Restoring it made things worse. |
| I4 grafting on the READ side (dense reads into the trunk's gate) | Leaks into the trunk's core. Ties or loses after fixes. RCF is the same graft done on the WRITE side, and that works (see below). |

### Bugs found

| Bug | Effect |
|---|---|
| Block position 0 (attention sink) was never in the fits but is in the evaluation. | Route writes with huge values there broke L12. Fixed with a "sink guard". It did NOT explain the old construction failures (checked). |
| Prose-only calibration | Folds did not carry over to code. Fixed with mixed prose + code calibration. |
| Too little calibration for big fits | Explains most of the 50%-width failures. |

## What I think it means

RCF is recursive generalization run in reverse, on the write side. Under that axiom a coarse neuron is a trunk whose
offshoots reach into its siblings' territory, each through its own route. Grafting the sibling's route into the
trunk's GATE (I4) leaked, because a dense read moves the trunk everywhere. Grafting on the WRITE side works: the
sibling's route already fires only in the sibling's territory, and RCF just teaches that route to write a coarse
summary of what the sibling wrote. The offshoot lives in the route, not in the trunk.

The coarse version of a removed neuron does not need a new unit. The network already has routes that fire just
before it. Those routes can write a coarse summary of what it wrote. That is the essay's "parent upstream" made
mechanical, and it is the first thing on this track that reaches trained quality with no gradient fit.

The open problem is scale across the whole model. Per layer, RCF repairs 20-40% of the damage. Across all layers at
50% width it repairs about 1%. Each fold is fitted to copy its own layer's output, so it cannot repair the way
errors from many layers combine. That is the same "downstream use" lesson as blue sky #10, one level up. A simple
drift-correcting target, rerun with enough calibration text, helps prose (5%) but hurts code: it moves the trade-off
instead of removing it. So next step 1 should use a real downstream window, not only a better local target.

## Suggested next steps (one question each)

1. **Downstream target for the whole model.** Fit each layer's fold to what later layers need, not to its own
   output. The cheapest version is the 3-layer lookahead window from R2.7 (minutes of fit, allowed). Question: does
   whole-model RCF then keep the per-layer gain?
2. **Uneven widths.** Folds help most in middle layers and hurt in early ones. Question: does a dial that thins
   middle layers more and early layers less beat an even cut at the same total size?
3. **Final keeper test** of rcf_v1 (single-layer) once the family is frozen. One validation query is left.
4. **Storage for a nested dial.** Rank-128 fold deltas keep about 75% of the gain at 16% of a layer per setting.
   Question: can neighbouring dial settings share one set of deltas, scaled?
5. **Bigger model.** Repeat G11 on SmolLM2-360M. The essay says larger models have clearer districts.

## Known gaps (not tested tonight)

- **No reasoning or recitation probe ran.** The essay says removing extreme gate responses breaks recitation, and
  removing double-negative activations lowers some reasoning scores. RCF edits attention and MLP write matrices, so
  it could hurt these without moving the average loss. Every number here is average next-word loss.
- **The sink guard is a deployment assumption.** It assumes block position 0 is always the same chat-template token
  in Cale, so the original output there is a constant. If that is false, the guard needs the original layer's output
  at position 0, which costs memory.
- **One small model only** (SmolLM2-135M).

## Essay ideas not run yet (for the next session to pick from)

| Idea | One-line test |
|---|---|
| I5 weak band = parent territory | Do siblings' shared weak activations sit where the judgment neuron fires? Use it as RCF's route selector. |
| I7 orphaned readers | How much of a cut's damage comes from later readers of removed neurons firing on missing input? Cut or mute them too. |
| I8 dial vs detector neurons | Split neurons by activation shape. Fold dials, keep detectors whole. |
| I9 grouping depends on zoom level | Context groups at the coarsest sizes, job families at mid sizes. |
| I10 fan-out order | Keep the neurons most read by later layers first (trunks), drop single-reader specialists first. |
| I12 foveated zoom | Keep trunks always, load a family's specialists only while its trunk fires. |
