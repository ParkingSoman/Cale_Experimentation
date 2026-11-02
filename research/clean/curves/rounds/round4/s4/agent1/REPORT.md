# S4 agent1, resnet18, angle: start from N's wiring (channels)

## What I read
- The head has 512 last-layer channels. W has no low-rank core: the top singular value holds 3% of the variance, and ten hold 17%. Each class is read by many channels.
- A mass-bearing pair (c, i) is not carried by a few channels. Half of the positive logit needs a median of 24 channels (of 512).
- Mass-bearing pairs share reader channels: the overlap of the top-20 W channels is 8.3 for these pairs, and 0.8 for random pairs. So "classes that share readers" is a true signal, but it is a broad overlap, not a single verb.
- Top-down by channels: the first channel splits are animals/water (channels 280, 17, 255) versus objects. Below that: birds and sea animals (245, 182), dogs versus big land mammals (83), indoor wearables versus street fixtures (435, 182), vehicles/aircraft (427). I read these as sensible verbs. But they sit at the top, and almost no sbar mass lives there.
- Where the mass lives: 61% of off-diagonal mass is in entries above 0.01. Half the mass is in about 5 entries per row. 40% of the mass sits at nodes of the Ward tree with over 100 classes, so Ward on W puts many mass pairs far apart.
- Many of those far pairs are context pairs, not sibling kinds: desk -> desktop computer, Crock Pot -> hot pot, restaurant -> plate, scabbard -> holster, home theater -> loudspeaker. Their residuals are +1 to +2 in log units.

## How it became a tree
1. Top-down channel bisection (split on the first axis of W in each node, 2-means refine): 1.454 (W), 1.430 (positive part of W). It is worse than Ward 1.357. The reading is clean; the tree is too coarse where the mass is.
2. Ward on weighted or mixed versions of W and of class mean features H (activity weights, between-class covariance, W with H): 1.349 to 1.366. No real gain over Ward 1.357.
3. Wiring-predicted table: P = softmax over i of (H_c . W_i + b_i), with H_c the mean feature of class c. Symmetrise, then average-linkage clustering with distance 1 - P/max. This puts classes that write into each other's readers together. Results: 1.3266 (T=1), 1.322 (T=1.5), 1.3196 (final).
It does not use rows of the real table. It uses W, b and the class mean features of the photos only.

## Prediction and score
Prediction written before the last run (PREDICTION.txt): 1.32 on A, range 1.31-1.34. Result: **1.3196** (final tree, `tree_resnet18.json`, T=1.5, average linkage, raw symmetrised P). Ward on W is 1.357, shuffled 1.730.
Caution: I tried about 40 variants on A, so 1.3196 carries some selection on A. The true held-out number is B, and I did not look at it. Also, split A is the set the eval fits f on, so finer trees get a bonus there (collapsing Ward nodes of size <= 4 gives 1.379, <= 40 gives 1.504). Expect B to be worse.

## Limits
- Photo noise: two halves of the photos (10 vs 10) differ by about 2.2 rms per half in the mass-weighted log. The full 20-photo table has noise of order 1.5. So a rms near 1.0 on a tree claim looks out of reach on resnet18 for ANY tree; I measured only a crude estimate on A rows.
- A tree cannot express context pairs (desk <-> monitor) and a similar-kind relation at once; one f per node is symmetric in c and i.
- Same-leaf-set trees cannot carry the asymmetry (writer c vs reader i): desk -> monitor is large, monitor -> desk is not.
- Top splits by channels are readable verbs, but they do not move the score.

## A different claim form, if one fits
Not shown by my numbers; a note only. The wiring gives the table directly: log sbar_c[i] = alpha_c + beta_i + (H_c . W_i)/T' + tree term, with the name "channel bilinear plus tree". The error is the same rms and is 0 exactly when it holds. It names no build, size or zoom. I did not measure it as a fit, so I make no claim about its error. The rank-14 fit at about 1.33 on all rows is at the same level as my tree (1.32 on A), which suggests rank was not the limit; the noise floor is.

## Not done
resnet50 not reached. Files: `tree_resnet18.json`, `final.py`, `lib.py`, `fit.py`, `e1`-`e9.py`, `noise.py`.
