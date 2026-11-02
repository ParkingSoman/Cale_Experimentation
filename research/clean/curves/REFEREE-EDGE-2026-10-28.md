# Referee report: UNIT-BRANCH.md A.5, A.6, B.6, B.7, D.3 (2026-10-28, round 1)

Referee: math-referee agent (reads only the paper and the Lean files). Lean: `Coarse/Edge.lean` builds, no sorry; the
nine Lean names state what the paper lines say. Verdict of round 1: not acceptable as written (4 errors).

| # | Kind | Finding (condensed) | Fix |
|---|---|---|---|
| 1 | ERROR | TA6 used W_out[., v] = 0. By DU1 that unit has an empty out-boundary, and DU2 lets no unit read its box, so the helper's write is not allowed; an unread unit is already its own unit | TA6 restated for a softmax-unread unit with edges: W_out[., v] = alpha 1, alpha != 0. DU2's "graph tether" named in the TA6 comment, DA2, Remark B.8, card 3-17 |
| 2 | ERROR | TA6's DU6/DU7 clause fails for W_out[., v] = 0 (dS empty, so the box must be constant and send 0) | true after fix 1 (dS = {v}); proof now checks dS |
| 3 | ERROR | TA6's DU9 remark: if v has edges, DU9 lets the logits re-weight v in S's single build, and stationarity kills the derivative | DU8 kept (with a least single build); TA6 comment now says the conclusion fails under DU9, and why |
| 4 | ERROR | CU7's "in particular": orthogonal writes still interact at third order (-eps^3/3 for the TA7 instance) | CU7 rewritten as a graded statement, "zero to leading order"; TA7 part 3 says the remainder need not be 0 |
| 5 | GAP | CU7's first sentence is provable; its exact converse is not reachable by the route | first sentence is now Proposition TA8 part 1 with proof; TA8 part 2 (softmax-unread sets are independent) added; CU7 is the graded converse |
| 6 | GAP | TA3 part 3: vote vectors need not be attained by N; for t < b the routes of (t - b) e_1, (t - b) e_2 overlap when n >= 3 | vectors with K > abs(t - b), routes {1} and {2} for every t; "attained by N and P" named as a premise in part 3 and A.6 rows 2, 4 |
| 7 | GAP | Remark B.8's TU5 row read TU5 as an edge result | row says TU5 is stated in the activation reading and can be restated as an edge result (not written) |
| 8 | GAP | A.6 rows 8, 9 claimed more than TA4, TA7 prove | rows now "related"; row 8 gives the upper bound and leading order; row 9 says the essay's comparison is empirical |
| 9 | GAP | TA7 part 1's constant unnamed | operator-norm bound via the third cumulant: abs(R_x) <= (sqrt 2 / 4) abs(a) abs(b) (abs(a) + abs(b)), proof written |
| 10 | WORDING | tags of TA2, TA4, TA6, TA7 | per-part tags; TA7 needs disjoint sets; TA6 tag names v |
| 11 | WORDING | TA4 restates Fact F11 | F11 cited, bound with P_perp w_u |
| 12 | WORDING | three notions of "unread" | one notion: softmax-unread (DA1 item 5) |
| 13 | WORDING | DA1 "no new object" overclaims | "agree with D44, D41 on the canonical form, up to the logit-difference convention" |
| 14 | WORDING | TA5: affine, and u enters by its weights too | fixed |
| 15 | WORDING | splitting keeps only builds that replace both copies or neither; c > 0 not formalised | fixed in A.5 and TA2; Lean line says positive homogeneity is by hand |
| 16 | WORDING | Remark B.7: canonical form deletes zero-out-weight units; out-edge threshold fails the splitting test | both said |
| 17 | WORDING | A.6 "does not say" paragraph: add the gauge; label the grouping reading as ours; row 7 narrows a direction to a unit; "comment" is not "proof" | all four done |
| 18 | WORDING | Part 0 item 6 overclaimed | rewritten (removal interaction; TA6 for softmax-unread units) |
| 19 | WORDING | D.2 count stale; S-twoclass cited toys | count rewritten; TU6 only |
| 20 | WORDING | numbers and log quotes in Part B | moved to the Appendix |
| 21 | WORDING | TU7 needs the logits in T's out-boundary | sentence added to TU7's comment |

Checks that passed in round 1: TA1; TA2's D_P claim; TA3 parts 1, 2 for t > b; TA4 part 1 integral form and 1/4;
TA4 part 2 instance; TA6's f(0) = D*(S, 0), D*({v}, 1) = 0, I(S, 0; {v}, 0) = 0, the gradient of phi, the fold covering;
TA7 part 1 double integral; TA7 parts 2, 3 arithmetic.

## Round 2 (2026-10-28)

No errors, 2 gaps, 6 wording items. Verdict: acceptable after the two gap fixes. All passed checks: DA1 item 5, TA3,
TA6 (DU2 allows the write; covering; output-faithful; silence-preserving; values; DU8 clause), TA7 part 1 (each step of
the third-cumulant bound), TA8 parts 1 and 2.

| # | Kind | Finding | Fix |
|---|---|---|---|
| 1 | GAP | CU7's premise sat on a least box; the lower side of I uses the joint near-optimum; no small parameter | premise moved to N's columns (W_S^T H_x W_T = 0 where S and T co-fire); the last-layer statement is now proved as TA8 part 3 (third order in the logit changes, TU3's two-sided device); CU7 is now the deep version |
| 2 | GAP | "Under DU9 the conclusion fails" not proved | weakened: the helper relu(e z_v) gains nothing at a least DU9 build; whether I < 0 under DU9 is open |
| 3 | WORDING | TA6 tag; DU8 existence premise | tag lists the three classes; existence at budget 0 stated |
| 4 | WORDING | TA8: logit premise unneeded; all-alpha-0 case; part 2 needs only DA2 | fixed |
| 5 | WORDING | comment after TA8 overclaimed | "when T = {v}"; "can interact" |
| 6 | WORDING | TA8 Lean line | fixed |
| 7 | WORDING | DA2 "empty"; two-logit sentence | "imposes nothing"; last layer, and "changes the loss" |
| 8 | WORDING | TU8's term is "no cancellation" | fixed |

Note: TA8 part 3 (the third-order bound under orthogonal columns) was added after round 2, from finding 1's fix. It is not
refereed. TA8 is stated for frozen readers; under DU8 and DU9 it is open (comment after TA8).
