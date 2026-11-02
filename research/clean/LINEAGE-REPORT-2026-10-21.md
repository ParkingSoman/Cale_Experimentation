# Lineage report, 2026-10-21

Plan: `~/.claude/plans/okay-look-at-the-prancy-jellyfish.md`. Log: `LINEAGE-LOG.md`. Results: `results/lineage-*.json`.
Measured on Pythia-160M, dev layers 2, 5, 8, 19 training checkpoints, average next-word loss and an 8-passage
recitation probe. No sealed layer was read.

## Answer in short

Shonu's idea was: coarse neurons exist earlier in training, then split, so checkpoints hold the tree. In a normally
trained language model, the tree is not visible.

| Question | Answer | Evidence |
|---|---|---|
| Early twins that later split (A)? | No | Twin share <= .003 at every checkpoint, 3 of 3 layers (L1) |
| Job spread first, then gathered (B)? | Not resolved. Shape matches, levels are low, the rule could not resolve late neurons. | L1c |
| Final families present early as coarse units? | No (small gap, mostly present in a random network too) | L1d, co-firing families only |
| Growth order as the keep order? | Worse than energy order, better than random | L2b |
| When does a neuron's job settle? | Late (identity .38 at step 8000, .80 at 64000). May be the learning-rate schedule. | L1 |
| Is the merge wall a SwiGLU property? | No. Merges lose to the cut in GELU too (+.02 to +.06). | Side check |

## What it means

1. Post-hoc coarsening of existing models: RCF (keep real neurons, fold the rest into upstream routes) stays the
   best tool. Building new units loses in both MLP types we tried.
2. A growth tree is something training must be BUILT to leave. Two known ways: small initial weights (neurons
   condense into a few directions early, Zhou et al. 2022), or growth by splitting neurons (a split pair equals its
   parent exactly, Zhang et al. 2021), with a nested loss that keeps parents valid.
3. Saving checkpoints alone is not enough. Revised decision for Cale's own model: save checkpoints AND choose a
   training setup that leaves a tree, after a tiny test shows it works.

## Proposed next step (needs Shonu's yes: it is our own training)

A tiny model we train ourselves (vision or toy language model, 100k-2M weights, minutes per run on the Mac).
Two arms: normal initial weights vs small initial weights (or growth by splitting). Measure twin share during
training, then lineage merge vs cut at the end. Pass: lineage merge beats the cut in the built-tree arm, not in the
normal arm. This is also where vision helps: we can look at what each neuron does as it forms.

## Round 2 and 3 (same day, after Shonu: "maybe you didn't look in a certain way")

| Test | What it fixed | Result |
|---|---|---|
| T1 core identity | used strongest firings, not all tokens | Cores also settle late (.12 of final core at step 8000). FAIL |
| T2 territory split | a split needs no twins; looked for one neuron's territory later shared by two | No handover between slots. Neurons form in place. FAIL |
| T3 size tree | 70M neurons as parents of 160M neurons, same-size seed1 as null | 70M = seed1 (split share .05-.06 both). FAIL |
| T3b microscope check | correlation instead of top-token overlap | Even fewer matches (<1% at .775). Most neurons are model-specific at this size. |
| V1 vision, labels | small vs large nets on CIFAR-100 | Neurons not class-selective at all (= random start). Not informative. |
| V2 vision, product | cut L's MLP to 128 everywhere vs LN trained at that shape | Cut loses: fine -8.0, super -6.4 points. Weakest retrofit method (no RCF, no window fit), so a floor. Small model's errors are not more within-superclass (S .211 vs L .227). |
| V3 true families | | Not run: no class-level neurons to define families |

Suggestive (weak matches): depth structure transfers across models (best-match parents sit at the same relative depth in 70M,
seed1 and 160M).

Gating is not contradicted. Gating is a claim about DEPTH (a gate neuron upstream of its specialists) and about
separately trained models of very different size. Our SmolLM2 work already found the upstream gates, and RCF uses
them. What failed is the idea that the tree is visible in TRAINING TIME or as a neuron-to-neuron map between small
models.

Next step proposed (needs a yes, ~1-1.5 h on the Mac): a deeper, staged vision net trained long enough to have
class-level neurons, sizes S and L, plus a nested-trained (Matryoshka) L. Questions: do S's neurons look coarse
next to L's (V1 again), do true families merge better than cuts (V3), and does nested training make L's first 128
neurons coarse parents with the rest as their children (training-time liquid).
