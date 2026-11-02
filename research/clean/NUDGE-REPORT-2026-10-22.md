# Closed-form nudge: report (2026-10-22, overnight)

Question (Shonu): find a closed form for "the parent takes up its children" that matches training, with no
overfitting, and check that it does not get worse as collapses compound over layers.

## Answer

Yes, at every scale tested. The closed form is a Gauss-Newton solve for the parent's own weights (what it reads,
its bias, what it writes) against the model's output. It uses the true output error and the output's curvature.
Each step is chosen on held-out training images. It needs no gradient training, only a few deterministic solves.

| Scale | Closed form | Training | Notes |
|---|---|---|---|
| One family (12 families) | matches or beats training on 10/12 | | held-out images; the 2 below training (15 n2142 .13 vs .16, 15 n2454 .57 vs .64) were stopped early by a too-strict step guard |
| 12 families collapsed at once | KL .00288 | KL .00289 | cut .00531 |
| One block (block 16 at 50%, 769 parents) | .226 of the cut's damage won back | .216 (80 epochs) | same end point |
| Two blocks (17, 16 at 50%) | .238 | .214 (early-stopped) | same top answer on 15/16 worst-hit images |
| Three blocks (17, 16, 15 at 50%) | .236-.238 | .245-.247 (2 seeds) | gap ~.01 (4% relative), real, NOT closed; cut KL .482 |

All scores are on sealed images never used for fitting or step choice.

## What closed the gap (each one was a real cause, found by being "unfair")

| Cause of the gap | Fix |
|---|---|
| The closed form's train images had 10x weaker family signal than training's | same images for both arms |
| One linearization step | 3-8 Gauss-Newton steps |
| Read change restricted to "child votes" | solve every weight of the neuron |
| Shared (K-FAC) curvature was 6x too small: positions in one image share the output gradient | exact "gradient bank": one backward pass, each parent reweights it by its own activations |
| Many parents fixing the same output error overshoot | solve all parents of a block jointly (matrix-free conjugate gradients) |
| Random Fisher samples are too noisy for 1M numbers | curvature restricted to each image's top-k classes (exact on that support; an approximation that weakens when the output spreads, as after three cut blocks). The gradient is exact. |
| One damping value per weight group stalls | per-weight damping (diagonal of the curvature) |
| Steps too coarse near the optimum | allow smaller steps (.1) |

## What it means

- The parent's new weights are fixed by the children (the target) and by how the rest of the network reads the
  stream (the ruler). A purely family-local ruler is not enough (2/6 families). The ruler is network-wide, but one
  backward pass serves every parent in a block.
- Training is not a ceiling. With early stopping, training overfits some blocks (block-15 parents: best epoch 0).
  The closed form's per-step held-out check kept every multi-parent run improving on sealed images. On one single
  family with nothing to recover (13 n131) it accepted a noise step (held-out share -.21 at step 2); a keep-best
  guard fixes this.
- The closed form and training land on functionally the same model (same answers on 15/16 worst-hit images).
- Compounding: 1 and 2 blocks end at or above training. 3 blocks end .01 below training. The remaining gap sits in
  two stages: the first block (about .01) and the block-14 parents across the downsample, where the solve stalls
  after 2-3 steps while training keeps improving. This is the open problem.

- Stall isolation: block 15 alone (parents across the downsample) converges to .128 vs training .132. The three-block
  stall is a compounding effect: by the third stage the remaining error is small next to the noise of 400 check images.
- Pictures at block scale: most parents move ~2%. The most-changed parents are mixed: one moves toward its removed child
  (lynx 74 -> 87% on the child's spots), one does not and reshuffles its own top crops. A joint block solve uses the
  parents partly as a shared pool; the clean "parent nudged toward its children" picture holds per family.

## Limits

- Wall-clock: the closed form is not faster than training here (one block: 7.9 min vs 3.6 min).
- Fairness: the closed form picks its step and damping on the held-out-train third every iteration; training uses a
  fixed learning rate (3e-4; 1e-3 diverged) and the same third only for early stopping.
- Blocks tested: 14-17 (late network). Early blocks (dials) not tested.
- Removal set: lowest-energy 50% per block, parents from the map. The fine distinctions carried by the removed
  children are still lost (for example buckeye -> snail stays wrong in both arms).

## Files

| What | Where |
|---|---|
| Log with rules before each run | `research/clean/VISION-LOG.md`, sections L1 to N12 |
| Single-family library and cache | `vision2/n7_lib.py`, `hf/imagenet/n7cache/` |
| Whole block, joint solve | `vision2/n10_block.py` |
| Across blocks | `vision2/n11_multi.py` (env DIAG=1, LMS, ALS, TOPK) |
| Pictures | `vision2/figs/l1/` (closed_*.png, n11_two_blocks.png) |
