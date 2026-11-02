# Referee report: Parts B.V.7 and B.Y.2 (2026-10-31, math-referee, round 1)

Scope: Part B.Y.2 (Definition DY3, Theorem TY4, Proposition TY5) and Part B.V.7 (Definition DV5, Lemma TV19,
Proposition TV19.1, Theorem TV19.2, Corollary TV19.3, Proposition TV19.4), as written before the fixes. Lean build
passes (9070 jobs), no `sorry`.

| Item | Verdict | Reason |
|---|---|---|
| B.V.7 intro "every result EQ" | GAP | TV19, TV19.1 are about one network; TV19.4 shows two least losses differ; TV19.4 and TY5 had no label |
| B.V.7 Scope "must share stream directions" | NOT MATH | a belief; make a conjecture or delete "must" |
| DV5 item 3 "fits in width" | GAP | the narrow network belongs to no class; "a narrow network of the same kind" never defined |
| DV5 norm constants | GAP | not said whether B' may change eps_l; c_l clashes (stream bias vs norm increment) |
| DV5 Comment (widths above d) | GAP (minor) | padding for d' > d |
| Lemma TV19 | OK | |
| Proposition TV19.1 | OK | |
| TV19.1 Comment on constant unread parts | NOT MATH | stated without proof; "does not" true only in some cases |
| Theorem TV19.2 part 1 | GAP (content) | by Lemma TV19 every member fits, so the two infima are over the same set: formally true, empty |
| TV19.2 part 2 | OK | |
| Corollary TV19.3 write side | OK | |
| TV19.3 read side | GAP | needs kept neurons to keep N's read vectors |
| TV19.3 Comment on transformers | NOT MATH | a transformer is not a DV1 network |
| Proposition TV19.4 | GAP (minor) | N not named; KL -> 0 to softmax convergence needs Pinsker; no label |
| DY3 copy channel | GAP (main) | undefined when positions change (every ResNet stage boundary) |
| DY3 build wiring | GAP | DH3 items 3, 4 not restated for skips |
| DY3 vs DY1 claim reading | GAP | pooled vs vector: choose one |
| TY4 part 1 | OK (given the wiring) | |
| TY4 part 2 "word for word" | GAP | the judgment shift fails across a stride copy: h_J = (2, 0), pooled 1, a stride copy carries 0 |
| TY4 part 3 | OK | |
| TY4 label OPT-KL | GAP | it is an equality of least losses (EQ); L_S has no closed form for k_S >= 1 |
| TY4 Comment "end in kept judgment channels" | GAP | J_w lies in the top layer of the subtree; delete |
| TY5 part 1 premise "J_w empty" | GAP (main) | with J_w empty TH1's premise 2 forces one area equal to P |
| TY5 notation | GAP (minor) | p_x is both a weight and a softmax; m_A > 0 |
| TY5 part 2 | OK (same premise issue) | |
| TY5 part 3 | OK | all of TH1's premises hold; numbers right |
| TY5 part 4 | GAP (minor) | attainment of the D2 least not shown; "class (b) build at G"; the integral remainder is in F11's proof |
| TY5 label and path | GAP | no label; R-unitmap / S-plainff missing from the path |
| TV19.2, TV19.3 paths | OK (over-listed) | R-normconst not used |
| Lean `written_step`, `quotient_read`, `kl_split` | OK | `kl_split` is the summed identity only; the minimum is by hand |

Content: structural 2 (TY4, TY5 parts 1-3); bookkeeping 5 (TV19, TV19.1, TV19.2, TV19.3, TY5 part 4); tightness
example 1 (TV19.4).

Ranked fixes proposed: (1) TY4: copies across position changes (average pools, or per-position judgment claims),
the skip wiring in DY3, one claim reading, label EQ; (2) TY5: replace "J_w empty" by "h_(J_w) constant on A_w",
labels, paths; (3) DV5 / TV19.2: an intrinsic narrow class (one isometry Q whose range holds E_N; kept writes Q^T w,
kept reads Q^T r, free new neurons) and a proof of the lift; kept reads in the class; norm constants; the intro;
the Scope and transformer sentences.

## Round 2 (2026-10-31)

Fixes of round 1 hold: TV19.2 is now a non-empty EQ over two different classes (lift and projection checked in both
norm forms and both logit kinds); DY3's average-pool copies keep pooled values along any chain of tilings and are DY3
channels; TY4 steps (i) to (v) hold with positions; TY5 part 1's premise is consistent with TH1's premise 2; TY5
parts 2 to 4 correct. Lean build passes (9070 jobs), no `sorry`.

| Item | Verdict | Reason |
|---|---|---|
| TV19.4 | GAP (main) | under DV6, Bc_1 is empty (dim E_N = 2 > 1), so "V(Bc_1) > 0" is an infimum over the empty set; it also does not test the sum k_S term |
| B.V.7 Scope | GAP (minor) | below dim E_N the class is empty; the open zone is dim E_N <= d' < dim E_N + sum k_S |
| TV19.2 lift citation | GAP (minor) | F13 part 3 is for square orthogonal Q; cite `norm_rot` (any linear isometry); name the free-kind bias |
| TV19.2 content | note | only says something when dim E_N + sum k_S < d; a first write spanning R^d gives E_N = R^d |
| DY3 branch error with positions | GAP (minor) | r_k is an array; read r_k^2 as a squared norm over positions |
| TY4 part 2, TY5 part 1 | GAP (minor) | h_(J_w) must read as pooled p_(J_w) |
| DECISIONS row R-jointgroup | GAP (minor) | stale "Corollary TV19.3" |
