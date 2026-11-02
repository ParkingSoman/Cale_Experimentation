# Referee report: Part B.Y (DY1, TY1, DY2, TY2, TY3), Definition DM10, `Coarse/Merge.lean` (2026-10-28, round 1)

Referee: math-referee agent (reads only the paper, DECISIONS.md and the Lean files). Summary as delivered.

Verdict: core proofs correct. TY1 sound; TY2 parts 1, 2, 4, 5 sound apart from an existence gap in part 1.
`lake build` passes, no `sorry`. Content: 0 structural results, 3 bookkeeping results (TY2 is a perturbation bound:
relu 1-Lipschitz plus Fact F11; TY1 a transfer lemma; TY3 a lookup and a dynamic program).

| # | Kind | Finding |
|---|---|---|
| 1 | GAP | TY3(3) computes a least cut into exactly k sets; DY2 asks for at most k_S, and the cost is not monotone under refinement (counterexample: two pairs, c = 3 on e0 and c = 1 at angle +-eps; the 2-cut costs ~3 eps, the whole set ~2 eps). Laminar children need not cover a node; virtual binarising nodes are not claims; S need not be a node. |
| 2 | FALSE SENTENCE | "A set of one channel always has a proportional claim" holds only if M lists the singleton claims. |
| 3 | HIDDEN PREMISE / OVERCLAIM | TY2 and TY3 need "M holds a merge partition of every group of every level": not in the tag or path. "Meets T5's graded form" holds only on such maps. E is a minimum over partitions; a list-only bound (sum over all claims) should be stated. DECISIONS section 2 lacked a row. Card 3-22 said "squared sum of the errors" (should be of their square roots). |
| 4 | GAP | eps_prop is algebraic: not shown computable in polynomial time in the bit model; storing a rounded value breaks "eps = 0 iff the claim holds". The zero test is exact in polynomial time. DY2's "least" compares sums of square roots. |
| 5 | WRONG REASON | The Comment's reason for no contradiction with TM4/TE5 is wrong: TM4 is about the split quantity and its instances have V_b = 0; TE5 is about V_rob; TY2 bounds D_P of one build. |
| 6a | GAP | B^m* existence not shown. |
| 6b | FALSE SENTENCE | DM10 Source: "c(S) classes are the sets DM10 claims with error 0" fails with unread members. |
| 6c | GAP | Copies for a layer with positions and an identity skip are an algebraic identity, not a re-writing of N; DN1 must carry proportional claims; tie-breaks. |
| 6d | MINOR | The closed form also uses P. |
| 7 | HIDDEN RESTRICTION | DY1 takes channels as units and position-shared boxes (node R-unitmap) but R-unitmap is not on the paths. |
| 8 | FALSE SENTENCE | DM10 "counts by how much the logit differences depend on it": c_u uses the pre-activation norm. |
| 9 | MINOR | DM10 forward-references DY1. |
| 10 | MINOR | Lean naming: the exact case is `merge_exact_sum`. |
| suggestion | | A pooled-value claim (Q, rep) with error sum_v |P_perp W_v| min_r |p_v - r p_rep|_P is exact strictly more often (it reads only what the logits read) and its bound needs only one triangle inequality. |
