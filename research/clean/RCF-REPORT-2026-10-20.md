# RCF report, 2026-10-20: scale, channels, and the width drop

Plan: `~/.claude/plans/okay-so-i-let-proud-taco.md`. Full log: `CHILDREN-LOG.md`, section "RCF SCALE + CHANNELS".
All numbers are dCE (added next-word loss) on SmolLM2-135M, dev text only. "Share" = share of the cut's damage
that a method wins back.

## Results in short

| Question | Answer | Evidence |
|---|---|---|
| Do "channels" (your extra-neuron idea) work? | No. A kept child is worth more than 1.5 channels. | B1, B1b: worse than equal-storage RCF in 12 of 12 cells |
| Does the fold disturb its own layer? | Yes. It grows with width. The fold from the layer before causes almost all of it. | A1, Q1 |
| Is there a free fix? | Partly. f1: a 15 s fit of the folds through the real layer, zero extra storage. It met the pre-registered bar (2/3 of the oracle's gain) on prose in 3 of 4 cells, but on code in only 2 of 4: FAIL by the letter. The Q4 control shows its gain is real (the cut with the same fit gains nothing). | Q2, Q4 |
| Is RCF weaker at wide widths because of a bug? | Partly. The disturbance was a bug and is mostly fixed. The rest is not settled. A 2-min trained layer keeps more of its strength at 50% width (pre-registered rule says "not natural"), but RCF + window beats that trained layer in all 8 cells, so it is not a true ceiling. At 50% width RCF matches or beats training. At 3% width it beats training by 20-34 points. | Q3, Q5-Q8 |
| Does a 3-layer window fit help? | Yes, a lot, at every width, also over the cut with the same fit. | A2 smoke, Q6 |

## The width drop: suspects and verdicts

| Suspect | Test | Verdict |
|---|---|---|
| Fold disturbs the kept neurons of its own layer | Q1 oracle placement, Q2/Q4 f1 fix | Real. Mostly fixed by f1 (prose fully, code partly). Code at L16, 50% width: 3% -> 20% (oracle 37%). |
| Too little calibration data | Q3 (128 vs 205 blocks per text) | No. +0.01 R^2 only. |
| Fit treats all positions alike | Q5 loss-weighted rows | Tested: no gain. (Weighting output DIRECTIONS was not tested: a fixed direction weighting cannot change the unregularized least-squares answer, and with the small ridge used the change is negligible.) |
| Fit copies its own layer instead of what later layers need | Q6 window fit | Helps at every width, but the drop with width remains. |
| Keep rule ignores the routes | Q7 keep what routes cannot predict | No. Picks 85-89% of the same neurons at 50%. |
| Natural: removed neurons at wide widths are weaker and less predictable | Q3 held-out R^2; Q8 trained reference | R^2 of the removed output falls ~20% from 3% to 50% width. Q8's trained layer drops less (to 79-89%), but RCF beats it everywhere, so Q8 is not a ceiling. Not settled. |

## Key numbers

Single layer, share of the cut's damage (prose / code):

| Layer, width | Closed-form fold | f1 (15 s, window 0) | Oracle (extra storage) |
|---|---|---|---|
| L16, 25% | 28 / 18 | 34-35 / 34-35 | 31 / 37 |
| L16, 50% | 19 / 3 | 23-24 / 20 | 25 / 37 |
| L22, 25% | 24 / 26 | 29 / 37 | 29 / 35 |
| L22, 50% | 14 / 34 | 17 / 37 | 20 / 46 |

With a 3-layer window (Q6), RCF vs the cut with the same window (prose / code): L16 51/53 (3%), 24/29 (50%); L22
42/49 (3%), 13/39 (50%).

Block of 10 layers (11-20) at 50% width, 30 s window fit per layer (A2 smoke): RCF-W beats CUT-W by .021 prose and
.015 code (both 95% below 0), about 14% of CUT-W's damage. Closed-form RCF on the same block gained .015 / .006.

## What this means for the design

1. The fold must be fitted through the real forward path, not by a one-step solve on recorded inputs. A short fit
   per layer (15-30 s) is enough and costs no storage.
2. The window fit should be part of the method. It helps RCF more than it helps the cut.
3. RCF is strongest at narrow widths (small laptops), where it beats a 2-min trained layer by a wide margin. At 50%
   width it matches or beats that trained layer. Whether more is possible at 50% needs a longer trained reference.

## Not done (on purpose)

- No whole-model run. Per the new rule, the design was fixed first with small tests. The next confirmation is the
  whole model with RCF + window fit vs the cut + window fit, ~15 s per layer, 2 arms: about 30 min on the Mac, or
  less on a rented GPU.
- No keeper query (1 left for route-capture).
- No true ceiling at 50% width: needs a long trained reference (GPU-scale).
- Untested variant: channels under the UPSTREAM route gates (layer L-1), where a "car" gate would sit.
- "Are RCF's routes really parents?" is still open.
- Every number is average next-word loss. No reasoning or recitation test has run.

## Addendum: the concrete account of the width effect (W1-W3)

| Measurement | Result |
|---|---|
| Does the leftover error cost more loss per unit at wide widths? | No. Same or less in 22 of 24 cells. Not the cause. |
| What drives the drop? | Only how much error RCF can remove at all (the local share): 39-44% at 3% width, 14-16% at 87.5%. |
| Why does the local share fall? | The removed neurons' output splits into: only the routes can predict it (falls .38 -> .08), both the kept neurons and the routes can (rises .08 -> .24), only the kept neurons can (small), neither can (.52-.61 at every width). The cut's refit already takes the "both" part. RCF can add only the "routes only" part. Formula routes_only / (routes_only + neither) matches the measured share within .04 in all 12 cells. |
| Is it a bug? | No. RCF reaches its linear limit at every width. |
| Where is the remaining gain? | The "neither" part. Layer L-2's neurons reach .05-.08 of it (zero-storage fold). Attention one layer back and the raw input add little. |
| Does RCF fall below training above 50%? | At 75% width RCF + window beats the 2-min trained layer in 3 of 4 cells. No clear trend, but the damage there is tiny and noisy. |

## Addendum 2: what made RCF stronger, and what did not (V3, S1, G, T1, G75, ES)

All numbers are the share of the cut's damage repaired, prose / code. Single layer (L16 or L22). Average next-word loss
only.

| Test | Question | Result | Verdict |
|---|---|---|---|
| V3 | Which weights carry the gain of the wide fit? | 3% width: layer L-1 gates. 50% width: kept gates and L-1 gates, about half each. | Points to gates |
| S1 / S1b | Can one set of L-1 gate changes serve two widths? | Keeps about .6 of the per-width gain at 3%, less at 50%. | Fail |
| G | RCF-G: also fit the kept gates and the L-1 gates | Beats RCF + window by 6.5-11.5 points. A cut start nearly catches up by 150 s, so the RCF start mainly makes the fit faster. | Pass |
| T1 / T1b | Fold layer L-2 routes too ("two-back") | Helps at 3% width (+4 to +8). At 50% width the extra fold disturbs layers L-1 and L. A 150 s fit does not fix L16. | Fail |
| G75 | RCF-G at 75% width | L22 prose collapses from 34 to -11 between 75 s and 150 s. Below the trained layer. | Unstable |
| ES | RCF-G with early stopping on held-out blocks | No collapse. 3 of 4 cells pass. Beats the 2-min trained layer at 75% width on both layers. | Pass |

Current best single-layer recipe: RCF closed form, then a short fit of the kept writes and gates, the L-1 writes and
gates, and the attention output, with early stopping. Zero extra storage.

Open design problem: the gate changes are per width. One file that serves all widths (the nested dial) cannot hold
per-width gate changes for free (S1). This needs a decision before the whole-model run.
