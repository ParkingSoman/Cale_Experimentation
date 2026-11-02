# Theory page: coarse neurons from fine ones (curves track)

> **2026-10-23: superseded by `PAPER.md`.** Under the weights-only rule, every row below that says DERIVED used
> least squares or gradients on images. Read it as FITTED. The .135 build is FITTED. Theorem 5 in PAPER.md proves the
> weights alone cannot fix the coarse network; the plain moments must come from somewhere (open decision).


Rewritten after every three cards. Method and rules: `~/.claude/plans/okay-i-want-you-optimized-perlis.md`.
Labels: DERIVED (from a principle, before the number), CHOSEN (picked on dev from a menu), READ-OFF (from a trained
solution). Only DERIVED counts toward the goal.

## Goal and the test for a method
A computation on the large network that gives the small network, with no training. Test: it gives ANY coarsening
choice in seconds, from fixed properties of the large network (weights, its activations and gradients, computed once).

## Current equation (k = 1, curve family in InceptionV1 3b)
1. READ (DERIVED, r CHOSEN from {0,1,4}): the coarse neuron is the best linear read of the family averaged over
   rotations about a point, H(q) = sum_theta relu(z_theta(q - r u_theta)), u_theta toward each member's centre of curvature.
2. WRITE (DERIVED, no knob): minimize the second-order output damage of the intact network, taken at the 4a readers'
   OUTPUTS, with the readers' ReLUs kept exact: for the set of readers that end up on, dh = on_new (v - u) +
   (on_new a - relu(a)) is linear in the write, so it is one linear solve (active set, one step from the plain
   Fisher write). G = the intact network's own output Fisher (labels sampled from its own answers).

## What the equation explains
| Observation | Explained by |
|---|---|
| the coarse neuron is a loop detector (feature vis, crops) | group average about a point (P1) |
| readers read h0 + h1 | P2 (low waves) |
| Bayes write shrinks the vote and loses | theorem: Bayes is optimal only if G is constant over images |
| training gates 60 straight-line / texture 4a neurons with the loop signal | G(x) couples curve readers to line readers on loop images: the Fisher write reproduces the sign on 42-49 of 63 |
| (RETRACTED 2026-10-23 audit) | the same negative sign is already in the original family columns (47 of 60) and in the Bayes write (45 of 60, corr .63-.67). It is not a Fisher effect. The Fisher write's KL gain stands |
| the trained read is not better than the derived read | with the Fisher write both give ~.20 (12000 images): at k=1 the gap is the WRITE |
| why the Fisher write still loses to training | it wakes readers that are off in the intact network (zero gradient there, so the second-order model charges nothing). Keeping them off brings it to .094, its prediction, below the trained build under the same treatment (.099) |
| the write gap | sealed: vmr .250 -> Fisher .176 -> active-set Fisher .135; trained .122 (derivation still loses by .014, se .003) |

## What it does not explain yet
- The last .014 (sealed) to training at k=1. The active set swings (step 2 is no better than step 1).
- Smoothing the ReLU slope (W4) failed: the pushes are large and systematic, not small noise.
- The Fisher write is still data-limited (1500 -> 12000 -> 35000 images: .241 -> .200 -> .177). Real damage is ~1.8x
  its second-order prediction, so higher-order terms matter for this write.
- Cost: one gradient pass over the analysis images per coarsening choice. Not yet seconds per choice.
- Per-image error of the second-order picture (corr .41-.53 only).
- The read gap (~.02 at k=1, ~.035 at k=3).
- k=1 and k>=2 use different recipes (k>=2 keeps members: that is pruning with a write, not a coarse neuron).
- Rulers: trained with lucent's leaky ReLU gradient (bug found 2026-10-22). A true-gradient k=3 rerun matches the old run to step 1000, so they probably stand.

## Honest score (sealed KL unless marked; lower is better)
| Build | KL | Label |
|---|---|---|
| curves k=1, derived read + vmr write | .250 | read DERIVED with r CHOSEN (9 variants); write vmr READ-OFF-confirmed |
| curves k=1, derived read + Fisher write (35000 images) | .176 | read r CHOSEN, write DERIVED |
| curves k=1, derived read + active-set Fisher write | **.135** | read r CHOSEN, write DERIVED |
| curves k=1, best known (trained) | .121 | ruler (leaky gradient: recheck) |
| curves k=3, kept members + Bayes | .139 | DERIVED (but pruning, not coarse neurons) |
| curves k=3, best known | .037 | ruler (leaky gradient: recheck) |
| circles k=2, kept members + Bayes | .062 | DERIVED (pruning) |
| circles k=2, best known | .031 | ruler (leaky gradient: recheck) |

## Open questions, ranked by how much gap they could explain
1. W3: the active-set Fisher write for k=3 curves, circles k=2, and both families at once (the combination test).
2. Why the active set swings; what the last .014 is.
4. More data without new images: mirror flips are a symmetry of the data (doubles the estimate).
5. Unify k=1 and k>=2 as real coarse neurons: units weighted by (1 + cos(theta - phi_i)), evenly spaced phi_i.
6. The read gap.

## Cost per coarsening choice (Shonu's second reason)
- Plain Fisher write: one gradient pass over 35000 analysis images (~20 min on this Mac).
- Active-set write: a second pass whose active set depends on the first solve (~40 min total). Open: is this a
  derivation or a two-step optimizer? Asked 2026-10-22.
- Not yet "seconds per choice": G is fixed, but the solve's features (g x c) depend on the choice.
