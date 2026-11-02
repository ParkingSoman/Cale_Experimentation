# S4 agent3: question the tree form (ResNet-18 only)

## What I read
Residual of alpha+beta on A rows (500 rows). Effective number of mass-bearing entries: about 3900 (about 8 per row). The row off-diagonal mass: top 1 entry 24%, top 5 53%, top 10 66%. So the score is the head of each row.

What a row's head holds (12 random A rows, class names, cosine of head weights):
- A family part: collie for Shetland sheepdog (0.44), butternut and acorn squash for spaghetti squash, missile for projectile, palace and monastery for triumphal arch. A tree can say this.
- Side parts, each a different cause, mutually unrelated: lionfish -> window shade (0.31, cosW -0.07, stripes), puffer, spiny lobster, sea urchin (spines). Stethoscope -> lab coat, then Maltese dog, toy poodle, miniature poodle. Car mirror -> minivan, loupe, bubble, Border terrier. Dung beetle -> ground beetle, then soccer ball, leatherback turtle. Head cabbage -> artichoke, swing, caldron, croquet ball.
- Mean cosW among a row's top 8 is 0.02 to 0.38; for 6 of 12 rows it is under 0.2. So a class holds one family position AND several overlapping side memberships.
- Links are directed. Of row top-5 pairs inside A x A, only 33% are also top-5 in the reverse row. Residual R[c,i] vs R[i,c]: weighted corr 0.74. Some columns are sinks (window shade): they take mass from many rows, and their own row does not return it.
- After alpha+beta, the first component of the weighted residual holds 42% of the sum of squares. u and v both have one sign. Rank 1 (a row-wise multiple of one column profile) takes rms 1.74 to 1.33. It correlates 0.77 with the row diagonal mass: a row peakedness, like a row temperature.

## Tree
`tree_resnet18.json`: average linkage, cosine distance, on head weights W. Reason: cosine ignores weight-vector length (length is a column scale, and beta takes it). Tool score on A: 1.3447 (Ward 1.3571, shuffled 1.7304). All W-based linkages sit within 0.02 of Ward (ward on unit W 1.360, weighted-cos 1.356, average euclid 1.359, complete-cos 1.404, centroid 1.480). A tree built from the A-row column profiles (SVD of residual) scores 1.540, worse than W. Prediction I made before scoring: "within 0.02 of Ward". Result: -0.012. No larger gain. The tree could not express side memberships, directed sinks, or row peakedness.

## Different claim form that fits what I saw
Claim: log sbar_c[i] = alpha_c + beta_i + m_c^T B m_i, off the diagonal. Each class i has one short vector m_i (G numbers, "side memberships", read from its column). B is one shared G x G matrix, not symmetric (sinks). alpha_c is the row scale. No row-specific numbers except alpha_c. Error = weighted rms of the gap, 0 exactly when the claim holds. It names no build, size or zoom. It is not nested: it is a rank-G directed Gram form (the tree is the special case of nested hard memberships with an ultrametric kernel).

Measurement, A rows only. Rows of A split in two halves, A1 and A2 (random, seed 7; also 11 and 23). beta, m (columns, fitted on A1 rows), B (regressed from A1 row fits) are frozen. A2 rows: m_c for c in A2 is that class's column vector from A1 rows. Only alpha_c is refitted on A2. Weights sbar. Script: `sidegroups.py` (`score_tied`).

| Form (shared numbers fitted on A1, scored on A2, alpha_c refit) | seed 7 | seed 11 | seed 23 |
|---|---|---|---|
| tree only, average-cosine tree | 1.759 | 1.844 | 1.827 |
| tree only, Ward on W | 1.772 | 1.836 | 1.827 |
| flat (alpha+beta only) | 2.150 | 2.148 | 2.153 |
| tree + directed Gram G=24 | 1.659 | 1.694 | 1.673 |
| Ward tree + directed Gram G=24 | 1.642 | 1.701 | 1.662 |
| Gram G=24 alone, no tree | 1.645 | 1.656 | 1.644 |
| Gram alone, G=8 / 14 / 38 (seed 7, with tree) | 1.88 / 1.69 / 1.64 | | |

Here the tree has only 250 fit rows, so it scores worse than the tool's 1.345 (999 node numbers on 250 rows, and f fitted in-sample there).

## Things that did NOT work (so do not claim them)
- Rows with free link coefficients (G+1 numbers per row, fitted on the held-out row itself) gave 0.97 at G=24, and 0.83 at G=38. This is an overfit. With about 8 mass-bearing entries per row, 25 numbers interpolate the head. When I fitted the row numbers on a random half of the columns and scored on the other half, the result was 2.0 to 2.6 for every G, worse than tree-only. I discard it. My early predictions of 1.28 to 1.5 for this form were wrong for that reason.
- Row link strengths predicted from the row's own head weights W (PCA 32 or 128 dims): 1.77 to 2.3, worse than the column-membership version.
- Free embeddings trained with gradient on the symmetric form: held-out worse than no Gram term (1.88 vs 1.83).

## Plain conclusion
- The gain of the directed Gram form over the best tree is about 0.1 to 0.17 held out (1.66 vs 1.76 to 1.84), with seeds agreeing. The size is modest. Both are far from the 1.0 line.
- The tree adds nothing once the Gram form is there (1.64 to 1.70 with or without). The part that held-out rows share is a rank-24ish directed form with sinks, not nested families.
- I did not run ResNet-50, and I did not look at the B split.
- Gram form not tuned past G; the "tied" fit is a two-step regression, not a joint optimum, so 1.64 is probably not its floor.

Files: `tree_resnet18.json`, `sidegroups.py`, `look1.py` to `look5.py`, `trees.py`, `trees2.py`, `common.py`, `gram.py` (failed variant).
