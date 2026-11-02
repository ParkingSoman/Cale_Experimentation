# Referee report: Part B.V.3, round 2 (narrow scope), 2026-10-29

Referee: the `math-referee` agent (it may not write files; saved by the session from its report). Round 1 is
`REFEREE-BVO-2026-10-29.md`.

**Files read:** the worktree copies of `COARSE-THEORY.md` (B.V.3, DM11 to DM15, DV2, TV6, TV6.1, D.15),
`DECISIONS.md`, `REFEREE-BVO-2026-10-29.md`. **Lean:** `lake build` passes; no `sorry`. No item in scope has a Lean
lemma.

## Summary table

| Item | Check | Verdict | Reason |
|---|---|---|---|
| TV14: e_l recursion, Taylor step, beta_l, gamma_l, pointwise bound, L2 bound | proof | OK | every step checked |
| TV14 "second order" scaling | statement | GAP | formal homogeneity; no family of cuts realizes it |
| TV14 no-norm form only | path | GAP | no node covers this scope |
| TV14 | content | bookkeeping | a first-order Taylor expansion |
| DM15 part 1 ("M names V_S") | definition | GAP | which item of M names V_S? |
| DM15 groups are areas | definition | GAP (hidden restriction) | a merged area must have own part 0 |
| TV6.2 against T5 | statement | GAP | k_S + 1 not stated; graded promise bounds the projection cut, not the least build; R-map missing from the path |
| TV6.2 | content | bookkeeping | by construction: one certificate per level |
| DV3 | status | GAP | the route is a proof; the "open part" is about size, not truth |
| DM14 "members" | round-1 fix | OK | |
| TV12 sandwich V_N(G+) <= V_area(G) | round-1 fix | ERROR | false (finding 11); the round-1 suggestion was wrong |
| DV2 norm-constant sentence | round-1 fix | GAP | "at most one direction" rests on the false sandwich |
| TV9 premises and scope | round-1 fix | OK | |
| TV11 sharp form | round-1 fix | OK | one wording slip |
| TV12 folded-form note | round-1 fix | GAP (minor) | the sign case falls outside c_l >= 0 |

Count in scope: structural 0; bookkeeping 3 (TV14, TV6.2, DV3 once proved); definitions 1 (DM15).

## Findings (summary of the referee's text)

1. TV14 proof: fine (recursions, Taylor step including a' = a, l2 <= l1, the exact identity, Minkowski, F11).
2. TV14 scaling is a property of the formula with R', U', omega, S^N fixed. Leaks lie in an affine set whose distance
   from 0 is TV9 part 1's least leak; small leaks = TV9's near-cancel regime. "beta_l is a sum of leaks" -> "a weighted
   sum of earlier leak norms". Name the family in "tends to (1/4)||Lambda_N||^2".
3. TV14 (and TV10, TV13) are no-norm only: add a SCOPE node.
4. TV14 Lean: the scalar Taylor bound is provable in Mathlib; Rule 2 asks for it.
5. TV14 content: Lambda_N is the chain rule; soften "as in Shonu's picture" to "the expansion keeps signs".
6. DM15 part 1 undefined; fix: completeness through D_j (M holds (D_j, members, l) with dim S - dim(S cap D_j) <= k_S;
   V_S = S minus (S cap D_j)).
7. DM15: let a group be a sum of pairwise orthogonal areas (an area that is a sum of >= 2 groups has own part 0).
8. TV6.2 meets T5 in letter; the content moved into completeness. Suggested no-norm subadditivity lemma (one claim
   per member, along N's full area graph) makes a complete map zoom-free.
9. TV6.2: (a) k_S + 1 in N's units; (b) the graded clause bounds the projection cut, not the least build (as TY2);
   (c) exactness and switching fine; (d) nesting not needed; (e) orthogonality fine under S-laminar; (f) add R-map to
   the paths of TV6.1, TV6.2; (g) card 3-23 should say "side branch".
10. DV3 is provable: segment bound for nu' (or a TV8-style premise), the logits' norm slope, "second order in the
    leaks" only. Response-table mismatch.
11. ERROR: the sandwich V_N(G+) <= V_area(G) is false. B^+'s battery is written by constants; a cut at G+ with N's
    constants has first write Pi_+ e and projected writes. Counterexample: norm form, d = 3, no blocks, eps = 1,
    P = {1,2,3} x {-1,1}^2 uniform, e(x) = x, S = span(e_2, e_3), G: k_S = 0, G+: k_S = 1, U = (e_1^T; 0). V_area(G) = 0
    (drop S, c = 2, exact since |P_S x|^2 = 2). V_N(G+) > 0: keeping e_1 and any v in S with N's constant forces an
    exact affine fit of x1/(x1^2+3)^(1/2) by a x1/(x1^2+2)^(1/2) + c on {1, 2, 3}, which fails (slopes 1.070 vs 1.251);
    the loss is convex in (a, b, c) and v ranges over a compact set, so the infimum is positive. Fix: delete the first
    inequality; say "one constant direction".
12. Other round-1 fixes: DM14 fine; DV2 rests on 11; TV9 fine; TV11 slip ("part 2's condition always holds there");
    TV12 folded note: w' eps_0 = d eps_0 + c with c < 0, outside c_l >= 0.
13. Ledger: TV12 row "(makes R-normconst moot)"; R-cutclaim row "areas" -> "members"; R-normconst row carries the
    false sandwich; R-normconst DECIDED on a delegation: the caller must decide whether delegation counts (Rule 7).

## Session response (2026-10-29)

| Finding | Fix |
|---|---|
| 11, 12 (sandwich) | deleted from TV12's comment, DV2 and the R-normconst row; "one constant, never-read direction"; the referee's example cited in TV12's comment |
| 6, 7 (DM15) | completeness through D_j; groups are sums of pairwise orthogonal areas |
| 9 (TV6.2) | retitled "the stream branch in T5's form"; one constant direction stated; the graded clause bounds B_j's loss, not the least build; R-map on the paths; card 3-23 says "side branch" |
| 8 | the subadditivity route recorded in TV6.2's comment, not proved |
| 2, 5 (TV14) | degree-2 statement as a property of the formula, tied to TV9's least leak; comment softened (chain rule; the expansion keeps signs) |
| 3 | node S-nonorm (PENDING) on TV10, TV13, TV14; card 3-25 |
| 4 | open Lean item, stated in TV14's Lean line |
| 10 | DV3 restated with the three details as a route; "is it small on trained networks" named a Part C measurement |
| 12, 13 | TV11 slip, TV12 folded note, ledger rows fixed |
| 13 (delegation) | kept DECIDED under Shonu's words; flagged to Shonu in the session report |
