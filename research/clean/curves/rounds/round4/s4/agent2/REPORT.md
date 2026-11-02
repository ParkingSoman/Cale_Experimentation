# Report, agent2, ResNet-18 only (ResNet-50 not reached)

Tree: `tree_resnet18.json`. Score on split A with the tool: **1.306** (Ward on W: 1.357, shuffled: 1.730). I never ran split B.

## What I read
- Channels do not form groups. Over all photos, the mean |correlation| between two channels' h is 0.06. Only 0.16% of pairs pass 0.3. The pilot notes said the same. So "group by what fires together" cannot be done on channels. It works on classes.
- Classes fall into six big groups by which channels fire strongly on their photos. Cutting Ward on W into 6 gives:
  - wide scenes and big vehicles (135): channels 435 sky, 481 horizon, 56 water fire on 17-33% more of their photos than average;
  - dogs (126): 83, 280, 274;
  - birds (56): 245, 182, 255;
  - furry mammals (90): 255, 424, 311;
  - small crawlers, sea life, macro (197): 17, 257, 460;
  - indoor man-made objects (396): no channel stands out above 6%. Only weak, diffuse firing.
  The last group is the problem. It holds 40% of classes and no strong shared channel.
- Where ambiguous photos go (rows of A only). Each class has one to three close mates that hold most of its off-diagonal mass (diamondback to sidewinder 0.15; sunglasses to sunglass 0.21; wool to cardigan 0.17). The median class needs 5 classes for half its off-diagonal mass and 104 for 95%. The strong firings go to the mates. Weak firings spread over the whole coarse group.
- Where the tree fails (honest held-out check inside A, see below): the row's top mate is under-predicted (mean residual +1.15 at rank 1, +0.85 at ranks 2-3). The far tail (ranks 50+, 13% of weight, 39% of squared error) is over-predicted (mean residual -2.6, rms 3.1).

## How it became a tree
- Top levels: kept the W-space layout, since it matches the reading above (scenes, dogs, birds, mammals, small creatures, objects). I did not find a better top split by judgment. Re-weighting channels by firing strength (sd, mean, rms of h), cosine, or a learned low-rank metric on W gave no gain (all 1.36 on A; learned metric held-out 1.84-2.07, worse than W).
- Low levels: the mates. I used evidence from A rows only: for a pair (c, n), E = max(sbar_c[n], sbar_n[c]) over rows that are in A. Pair distance = d_W * (1 - 0.6 * min(1, E/0.1)). Then average linkage, binary. The idea: a pair that exchanges mass in A is pulled into a small clade. B-class pairs get pulled only through A-row columns, which is allowed.
- Flattening to wider fan-out hurt with average linkage. Not used.

## Honest check
The tool's A score is partly circular for this tree (the evidence comes from A rows). So I wrote `lib.py` with the tool's fit and a split of A into A1 (fit, evidence) and A2 (held-out rows, alpha refit). Means of two seeds, test on A2:

| Tree | Held-out rms on A2 |
|---|---|
| Shuffled Ward-W tree | 2.17 |
| Metric-learned (rank 64) | 1.84 |
| Ward on W | 1.82 |
| Average linkage on W | 1.85 |
| Mate-pulled average linkage (rho .6, sat .1) | 1.79 |

So the real gain over Ward is about 0.03, not the 0.05 seen on A. Expect B to move by a similar small amount.

## Predictions and scores
- Before scoring: A = about 1.20 (circular), honest gain 0.03-0.06. Result: A = 1.306, so the A number was wrong by 0.1 (less circular than I thought). Honest gain 0.03 is inside the prediction.
- Trees built with class-mean h of all classes (including B classes) scored 1.36-1.37 on A. They are diagnostic only, tainted for B, and not handed in.

## What a tree cannot express
- A row's one to three sharp mates. A tree gives them mass only if they sit in the row's own small clade, and mates are not symmetric or transitive (sunglass and sunglasses, but each also has own private neighbours).
- The weak-firing tail. It is c-specific, not shared by a node.
- So the form `alpha_c + beta_i + f(lca)` has a floor near 1.3 in-sample for trees and for rank 14. A split-half estimate puts photo sampling noise near 0.8 (rough), so the gap is mostly structure, not noise.

## Different claim form (suggestion, not tested)
Add a sparse mate term to each row:

log sbar_c[i] = alpha_c + beta_i + f(lca(c,i)) + m_c[i], with m_c nonzero for at most k mates per row (k about 3).

Numbers it names: alpha_c, beta_i, one f per node, and up to k (mate, value) pairs per row. Error = the same mass-weighted rms of the residual. The error is 0 exactly when the table equals the form. It names no build, size or zoom. Caveat: with k mates per row the form has 3000 free numbers, so it must be checked on held-out rows, and only the mate list from A rows can be used for B rows.

## Files
`lib.py` (held-out harness), `final.py` (builds the tree), `hyb.py`, `metric.py`, `wscale.py`, `var.py`, `resid.py`, `top.py`, `noise.py`, `chan.py`, `look.py`.
