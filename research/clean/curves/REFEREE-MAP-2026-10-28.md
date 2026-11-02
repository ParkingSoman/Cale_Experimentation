# Referee report: Maps (T5), B.M, D.14, UNIT-BRANCH B.8, Map.lean (2026-10-28, round 1)

Referee: math-referee agent (opus). It had no write tool, so its report was saved here by the main session, unchanged in
content; the table is condensed to finding, verdict and fix. Lean: `lake build` passes (9064 jobs), no sorry.

| # | Item | Verdict | Finding | Fix (as done, see the end) |
|---|---|---|---|---|
| 1 | TM3 part 2, Part 0 | ERROR | a map may leave out claim types; the one-claim map {district(all layer-1 units, Omega = P)} has error 0 and makes the big group a union of districts, so T5 as worded asks for an exact fast rule on CLEAN-GROUP: P = NP. No new claim type can repair it. "The hardness must show up in another claim" does not follow | TM3 part 2 restated as a proved conflict with T5 (one-claim map); card 3-19 for Shonu |
| 2 | TM3 reading | GAP | the polynomial-time reading sits under R-mixclean, not on TM3's path | R-mixclean added to the paths of TM3, CM2 |
| 3 | TM3 part 1 | GAP | (i j) and the kappa-cycle generate S_kappa only when gcd(j - i, kappa) = 1 | generators (1 2) and (1 2 ... kappa) |
| 4 | TM3 part 3 | OK + WORDING | correct; a map is needed only on no-instances; the comment dropped "if a rule exists"; the reduction runs through Multicolored Clique | done |
| 5 | TM3 graded clause | GAP | nothing on the graded clause | remark: the one-claim map has all errors 0, so the graded clause adds nothing on these instances |
| 6 | CM1 | ERROR | false: P = {a, b, c}; s (region {a}), j1, t1 (on only at b), j2; z_s = (1, -1, -1 + delta); box relu((z_s + 1)/delta) makes the joint build exact for every delta while L_T > 0 is fixed and the errors go to 0 | CM1 marked false with this example; CM1' restated with a weight bound W |
| 7 | CM2 | GAP | cannot carry the exact clause while maps may omit it; (ii) holds on every instance; size 1 vs k_g; "closed form in N's weights" wrong (depends on P) | restated per group, in (N, P), tied to card 3-19 option (a) |
| 8 | DM3 | ERROR | Euclidean norm irrational: not polynomial-time computable | squared norm |
| 9 | DM5 | ERROR | Frobenius norms irrational; raw vs reduced form; orthogonality of rho_0 unchecked | squared defects; claim carries rational positive unit scales; orthogonality defect added; meaning stated without "exactly when D20" |
| 10 | DM2-DM4 | WORDING | "exactly when" needs p_x > 0; DM2's disjointness rules are syntactic | p_x > 0 assumed in DM1; rules named syntactic |
| 11 | DM3 source | WORDING | the gate sentence supports DM2, not constant in-votes | said |
| 12 | DM1-DM5 | OK | no claim names a build, size or zoom | - |
| 13 | "clean = a map with small errors" | ERROR | vacuous: every N has an error-0 one-district map | withdrawn; replaced by a pointer to card 3-19 (complete maps) |
| 14-16 | TM1 (a)-(d) | OK | v_S undefined when P minus the regions of S is empty | "any vector" |
| 17 | TM1 part 2 | GAP | K(S) under G' | sentence added |
| 18 | TM1 part 3 | OK | attainment not proved for n_y > 2 | already conditional |
| 19 | TM1 class | WORDING | D50 is for one hidden layer; "box" undefined there | Definition D50' added in B.M |
| 20 | TM1 tag/path | GAP | "covers" hidden; R-readweights not on path; "meets" undefined | tag phrase + node S-cover; R-readweights on path; "meets" defined |
| 21 | Map.lean citations | WORDING | split_least supports part 3 only; docstring overclaims | fixed |
| 22 | TA12 | OK | proof correct | - |
| 23 | TA12 premise | GAP | needs S read by the logits | added |
| 24 | TA12 title, B.8, B.9 | WORDING | condition is gamma != 0, not co-firing; scope of TM1; "the unit is the region" overclaims | retitled; scope restored; "regions meet U4 under TM1's premises" |
| 25 | DA3 | WORDING | Theis et al. use the empirical Fisher; "tied as strongly" unclear | hypothesis stated; G[u1, u2] = G[u1, u1] |
| 26-28 | ledger | GAP | section 2 rows; 1a and 3-18 pointers; UNIT-BRANCH not in the ledger | fixed; note added |

Verdict of round 1: TM1, TM3 parts 1 and 3, TA12 correct under their premises. Three real problems: (1) TM3 part 2
refutes T5 as worded; (2) CM1 false; (3) DM3, DM5 not computable, and the "clean" sentence vacuous.

## Round 2 (2026-10-28, opus; it had no write tool, condensed here)

Verdict: TM1 and TA12 correct; TM3 parts 1 and 2 correct under the polynomial-time reading; CM1's counterexample
correct. New problems and their fixes:

| # | Finding | Fix |
|---|---|---|
| 1 | TM3 part 3 (claim types) followed from part 2 in one line | restated for a polynomial-time class A of (map, zoom) pairs; the verifier also checks membership in A |
| 2 | the reduction of T43 starts from Clique, not Multicolored Clique; "maps a hard instance cannot have" too strong | fixed |
| 3 | "T5 as worded" overstates: T5 puts no time bound on the one-time computation | "under the polynomial-time reading" in TM3, Part 0, card 3-19, section 1a; Part 0 marks the choice as pending |
| 4, 5 | card 3-19 missed option (d) (card 3-5's options); T21 exact only for D2; "complete map" and the blend claim undefined; graded clause unchecked under (a) | card rewritten with (d), the D2 caveat, and the unchecked points stated |
| 7 | "scaling t1's column makes L_T as large as we like" false (L_T <= weighted entropy) | deleted |
| 8-10 | CM1' false under both readings of its weight bound (a kept unit j1 on at b with value delta does the job with weight 1/delta and no box); its quantifiers make it empty; no node for W | CM1' withdrawn; new Proposition TM4: no bound in the errors alone exists for the split |
| 11 | DM2, DM3 errors change under function-preserving rescaling | normalised per unit (share of activity outside Omega; squared moves over the unit's mean square) |
| 12 | DM5: products of generators, finiteness, trivial claim | rho_0 restricted to signed permutation matrices; identities extend to products (shown); trivial claim noted |
| 14 | empty region can meet two groups | "every non-empty region" |
| 16, 17 | R-readbias missing on paths; CM2 needs two logits, a time bound, a grouping | R-readbias on TM1/TM4; CM2 restated (two logits, polynomial time, grouping {S}), S-twoclass on its path |
| 18, 19 | ledger wording; S-cover needs eps_dist = 0 and R_0; "three limits" | fixed |
| 22-25 | DA3 "in expectation", Theis diagonal; TA12 cites D50'; B.9 adds shelter; D.14 heading | fixed |

Fixes of round 2 are not re-refereed.

## Round 3 (2026-10-28, opus; after Shonu's decision on card 3-19; condensed)

Verdict: TM5's proof correct for one logit order; TM1 part 4, TM4 parts 1-2, TM3 part 3's verifier correct. Problems:

| # | Finding | Fix |
|---|---|---|
| 1, 2 | ERROR: DM7 was one-sided and depended on class order. With N's logit rows swapped, the big group of T43 had blend error 0 (vertex points are strict maxima of z_u), its complete map existed, and the conflict with T5 came back (P = NP) | DM7 made two-sided (t_S outside [min, max] over the hull's points; two LPs per point); TM5 proved for both orders (second kind: e_u has entry sum 1, every t_S = 0 point at most 1 - eta/2); `model/tm5_blend.py` v2 checks both orders: 0 exceptions in 1778 group checks |
| 3, 4 | TM5's class order hidden; "any positive weights" for T59.1/T59.2 unstated | both said in the statement and proof |
| 5, 6 | TM5.1 part 2's map lacked shelter claims and disjoint nested districts; "the exact clause holds" too strong | districts S_0, S_1 \ S_0, ..., rest, each with shelter; "a closed-form rule meets the exact clause on these inputs" |
| 8, 9 | TM3's graded sentence stale; "T5" means the first form | rewritten (graded clause on T43's complete maps is open); "first form of T5" |
| 11, 12 | R = P waiver with empty-region groups; y_N on R_0 under part 4 | fixed |
| 13, 15 | TM4 part 1: two logits, delta in (0, 1); comment's blend condition always holds | fixed |
| 16, 17 | DM6: item 5 is an exclusion, not a need; it removes three or more logits; completeness is in effect an input restriction | said plainly in DM6; item 5 only for k_S < |S| |
| 20, 21 | DM3 uncentred; DM5 narrower than D20 | DM3 uses the variance; narrowing stated |
| 22, 23, 24 | H-map vacuous (one region); CM2 trivially true; CM3 ill-posed | H-map needs >= 2 label classes; CM2 non-triviality clause; CM3 needs R-deep's definition and a block-sparse premise |
| 26 | Part 0 overclaims | rewritten |
| 27 | GAP: the restriction of the graded promise is NOT forced by TM3 (f need only be 0 at 0), and was beyond Shonu's words | removed from the trunk text (hash re-stored); asked of Shonu |
| 28-30 | ledger rows stale; paths | fixed; TM4, TM5, TM5.1 rows in section 2 |
| 32 | "U4" undefined in the paper | replaced by a statement in the paper's own terms |

Round-3 fixes are not re-refereed.
