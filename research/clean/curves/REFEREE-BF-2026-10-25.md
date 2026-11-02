# Referee report: Part B.F (fold optimality), D.9, card 3-3 (2026-10-25)

Scope: `COARSE-THEORY.md` Lemma T31, Lemma T32a, Theorems T32, T33, T34, Corollary T35, Remark B.F.1; Part D.9
(Conjectures D25, D26, D27 and their status paragraphs); `DECISIONS.md` card 3-3 and the section 6 paths of these items.

What I ran:

| Check | Result |
|---|---|
| `model/t32_cover_gap.py` | ALL OK. 24 cells, 1176 pieces, all certified. Least bound 4/3 (D2 >= 1/24). Part 1 gives 3/80. |
| `model/t33_broad_gap.py` | ALL OK. 188 cells, 376 pieces, all certified. Least bound on D2 is 1/468. Part 1 gives 0. Run time 32 s, not 5 min. |
| `model/t34_signed_gap.py` | ALL OK. 318 cells, 636 pieces, all certified. Least bound on D2 is 7/60. Part 1 gives 25/216. |
| `lake build` in `lean/coarse` | Passes (style warnings only). |
| `grep sorry` in `Coarse/*.lean` | None. |

Summary. The three computer-assisted proofs are sound. Each script verifies by exact arithmetic every step the proof
relies on. The lemma is correct. The overstatements are in the text around the theorems:
- Corollary T35, Remark B.F.1 and card 3-3 claim more than T32 to T34 prove.
- The list of options in the fork leaves out two real premises.

## 1. Findings, ranked by severity

| # | Severity | Location | Problem | Fix |
|---|---|---|---|---|
| 1 | Major | T35 title and general sentence; card 3-3 ("distillation does not converge to folds of the families"); D27 status | The trunk's loss is KL (D_P). T32 and T34 are proved for D2 only. Only T33 covers KL. Covering folds are a subset of broad folds, so T33 refutes the KL half of D25 and of D26. Nothing refutes the KL half of D27 (signed folds of one family). So "distillation" (KL) is not shown to leave the family structure. Under KL, only "positive side only" is refuted. | In T35, say for each item which objective it covers. Retitle the corollary: "for D2, ...; for KL, only broad folds are refuted (T33)". In the D27 status, write "refuted for D2 by T34; the D_P half is open". |
| 2 | Major | T35 statement ("For perfectly clean networks ...: 1. The least build is not made of covering folds") and the general sentence ("the mix coefficients of a least unit are not confined ...") | These are universal statements. The theorems are existence results, one instance each. As written, the universal claim is false: Remark B.Z.1 gives a D20-clean network (T20: mirror, full-rank in d = 1, invariant mix) whose least width-1 unit for D2 is a covering fold. | Make the corollary existential: "There exist perfectly clean networks ... for which no least build is ...". Write "need not be confined", not "are not confined". |
| 3 | Major | T35 items 1 to 3; card 3-3 table ("What the best build does instead") | The descriptions are about "the least build". T32 and T34 only exhibit some build whose value is below the fold bound (3/80 < 1/24, and 25/216 < 7/60). Neither proves that this build is least. Only T33's build (value 0) is proved least. Item 1 is also wrong about the exhibited build. Its first unit has b = -8/5 = -(4/5 + 4/5), so it is exactly a covering fold (on the boundary of the cone). Only the second unit fires inside the square. | Proved claim for item 1: "every least build has a unit that is not, on P, a covering fold". The descriptions ("lowered threshold", "negative side", "mixes the families") hold for the exhibited builds; say so. In item 1, use the singular ("one unit fires where every member is silent"). |
| 4 | Major | Remark B.F.1 options; card 3-3 options | Two premises of T32 to T34 are not among the options. (e) The objective: D2 (node R-obj, RETRO) is a premise of T32 and T34. It is an escape, and it is the escape closest to the trunk's own words (KL). (f) Groups that cross families. The T34 unit is a covering fold (D22) of a three-unit group that meets both families. With A3 = (-1, 0, 1/2; -1), B2 = (-1, 1, -1/2; -1), B3 = (-1, -1, -1/2; -1): (-11/3, 11/9, -10/9; -44/9) = 13/18 A3 + 75/36 B2 + 31/36 B3 + 11/9 (0, 0, 0; -1), so m >= 0 and tau = 11/9 >= 0. D23 zooms allow such groups, so T34 refutes "one family per unit", not "folds of groups". T32 and T33 have one family, which is the whole layer. So they already rule out covering folds of every group. Option (f) rescues T34 only. | Add (e) and (f) to B.F.1 and to card 3-3. In T35 item 3 and on the card, replace "mixes the two families" with "is a covering fold of a group {A3, B2, B3} that meets both families". Correct D27 Route (3): when one-family cones fail, the result is not Lemma T31. Folds of cross-family groups with a covering threshold are still a real class between the two. |
| 5 | Medium | B.F.1 (b) and card 3-3 (b): "1/8 relative in T32", "from 0 to 1/468", "from 25/216 to 7/60" | These are written as exact losses. "1/8" comes from the three-unit instance at solver accuracy (D25 status: 1/36 against 1/32), not from T32. T32 proves a relative loss of at least (1/24)/(3/80) - 1 = 1/9. 1/468 and 7/60 are the smallest certified lower bounds, not proved least values. 25/216 is an upper bound on the least value over any builds. | Write: "T32: at least 1/9 relative (3/80 against 1/24); T33: 0 against at least 1/468; T34: at most 25/216 against at least 7/60". Also in the T33 proof, change "the least is 1/468" to "the least certified bound is 1/468". |
| 6 | Medium | T33 part 2 and proof ("the least such value is reached (Theorem T27)"; "For D_P the least value over broad folds is positive too") | T27 is proved for D22 folds (tau >= 0, a cone in u = (m, tau)). It is not proved for broad folds (beta free). The D_P sentence shows only that each broad build has D_P > 0. That does not show that the infimum is positive. The conclusion (no least D_P build is broad) needs only the pointwise fact, because the least D_P over all builds is 0. | Drop the T27 citation. The certificate's bound of 1/468 gives a positive infimum for D2 directly. For D_P, say "each broad build has D_P > 0, which suffices". If the positive infimum is wanted, add a Pinsker step: D2 >= 1/468 forces some point i with pi_i (g_i - t_i)^2 bounded below. With t_i bounded, the sigmoid gap at that point is bounded below, so KL is bounded below. |
| 7 | Medium | DECISIONS.md section 6 paths of D25, D26, D27 (and T31) | D25 says "Let N have one hidden layer". D22 folds are defined only for L = 1. S-onehidden is not on the conjectures' paths: a hidden restriction. T31 is also stated for one hidden layer, and its path is "(none)". | Add S-onehidden to the paths of D25, D26 and D27, and name "one hidden layer" in their tags. For T31: either add S-onehidden, or restate it for the first layer of any depth (the proof is unchanged). |
| 8 | Medium | Card 3-3 as a whole; node R-foldopt (kind ROOT) | Every result behind the card has R-obj (RETRO) and R-symN (RETRO), plus R-cleannet and R-fullrank (PENDING), on its path. The ledger rule says results under undecided nodes are side branches. The card presents them as forcing a ROOT decision. Section 4 also says only Shonu adds ROOT nodes, and R-foldopt is "By: open". | Say on the card that the fork is forced only for D2 and only under the pending definition D20. Make R-foldopt a STRUCT or SCOPE node until Shonu makes it a root. |
| 9 | Minor | Lemma T32a "Lean: `cert_ineq` (the inequality of part 1)" | `cert_ineq` proves only 2 sum pi v a - sum pi v^2 <= sum pi a^2 for pi >= 0 (completing the square). The content of part 1 is that the cone conditions give sum pi v g <= 0, and that step is not in Lean. So the Lean statement is weaker than the paper's statement. | Write "Lean: `cert_ineq` (the completing-the-square step of part 1; the duality step is by hand)". Or prove the full part 1 in Lean, with the R_j and mu_j as finite matrices. |
| 10 | Minor | Literature table; T32a Comment; proofs of T32 to T34 | Gordan's theorem and the conic Caratheodory theorem carry weight in every check. Neither is in the literature table (only Farkas, Corollary 7.1h, is). | Add both, each with its own hypothesis. Gordan: for a real matrix A, A theta > 0 has no solution if and only if y >= 0, y != 0, A^T y = 0 has one. Conic Caratheodory: a vector in the cone of finitely many vectors of R^m is in the cone of at most m linearly independent ones among them. Only the "if" direction of Gordan and the existence direction of Caratheodory are used. |
| 11 | Minor | Scripts, step `y = (A_^T W A_).pinv() ...` | The proof needs v to be exactly rational. The sympy pinv of a rational matrix is exact, and the printed minima are rational. But nothing enforces this. If a Float ever entered v, `sum(pv) != 0` and `Gs * sol == cm` would compare floats. | Add `assert all(isinstance(e, sp.Rational) for e in v)` after v is computed, in all three scripts. |
| 12 | Minor | T32 to T34 statements: "two logits, and gap equal to their sum" | The readout is not fixed. D20 needs W_out pi = sigma W_out. A readout with the same gap that breaks this (for example, adding c (u_1 - u_2) to both logits) is not perfectly clean. Reduced form also rescales the two families of T34 differently (norms 3/2 and sqrt(13)/2). This still works, because pi permutes within each family, but the text does not say it. | Write "logits (sum of the units, 0)", as T25 does. Spell out rho_0, pi and sigma = identity, and the reduced-form scales, as T25 does. |
| 13 | Minor | T33 Comment ("one ReLU leaning up ... never a concave rising one") | The comment treats the unit as a function of x_3 alone. On P the unit also depends on x_1 and x_2. This is a heuristic, not a consequence of the proof. | Label it "heuristic", or delete it. |
| 14 | Minor | D25 status paragraph | (i) It says "the least build pairs two such units along one member's direction" and then names T32 as "the exact statement". T32's build is diagonal, (-1, 1), and its first unit is exactly covering. (ii) "the optimal coarse units are broader than the union of the fine units" is not true of both units. (iii) Numerical values (1/36, 1/32, "KL shows a gap of the same size") are measurements in Part D. | Separate the numerical instance from T32. Fix the description of T32. Move the solver numbers to Part C or to the log, or keep them under a clearly marked "evidence, not used" line. |
| 15 | Minor | T31 | T31 is the "if" direction of T28 part 1 plus the spanning assumption. | Mark T31 as a corollary of T28 part 1, or cite T28. |
| 16 | Minor | DECISIONS.md: section 3 intro, section 8, node R-fullrank, card 3-3 (a) and (d) | (i) The section 3 intro still says "three decisions"; there are six cards. (ii) Section 8 shows R-fold as PENDING; the node table says DECIDED. B.F and R-foldopt are missing from the tree. (iii) R-fullrank is "carried by" D25 only. T32 to T35 and D26, D27 also carry it. (iv) Card (a): "T32-T34 must fail under it" is automatic if the instances lack the property. What is needed is a proof that the conjecture holds under it. (v) Card (d): "favour folds on average" changes the quantifier over mixes without saying so. | Update the intro, the tree and the carriers. In (a), write "then prove D25, D26 or D27 under it". Make (d) an explicit option: a stated class of mixes (for example, P with a density, or large invariant P), or cut it. |
| 17 | Minor | Paths of T32 (and T33, T34) | T25's path has R-fold because "part 3 is about fold builds (D22)". T32 part 2 is about D22 covering folds, but R-fold is not on T32's path. T33 and T34 use classes (broad and signed folds) that are defined only inside Conjectures D26 and D27, not in a Definition. | For consistency, add R-fold to T32's path. Promote broad folds and signed folds to a Definition next to D22, so that T33 and T34 refer to definitions, not to conjectures. |

## 2. Item by item

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| Lemma T31 | 2 Proof | OK | One line. The w_i span R^d, so w = W^T m. |
| Lemma T31 | 3 Tag | OK | Uses no hypothesis. |
| Lemma T31 | 6 Path | GAP | The statement assumes one hidden layer, and the path is "(none)" (finding 7). Duplicates T28 part 1 (finding 15). |
| Lemma T32a part 1 | 2 Proof | OK | sum pi (a - v)^2 >= 0, and sum pi v g = sum mu_j^T R_j theta_j <= 0 when sum pi v = 0. Correct, with gamma free. |
| Lemma T32a part 2 | 2 Proof | OK | The open cells are dense in int K. Finitely many closures cover cl(int K) = K, which needs a non-empty interior (stated). x~_i != 0 because of the last entry 1. The masked vector is correct on a closed cell. |
| Lemma T32a | 4 Lean | GAP | `cert_ineq` is only the completing-the-square step (finding 9). |
| Lemma T32a | 7 Literature | GAP | Gordan and Caratheodory are not in the table (finding 10). |
| Scripts (all three) | 3 Exactness | OK | Every float step only proposes. LP status picks a branch, and both branches are checked exactly. Rounded interior points are checked exactly with strict ">". Gordan: a sign-definite nullspace vector, exact, and the cone's own rows are included. Caratheodory: subsets of linearly independent rows of size at most the dimension, solved exactly. v is the exact residual on a face chosen by floats; sum pi v = 0 and the cone memberships are checked exactly. A wrong face can only cause a failure, never a false pass. The covering, broad and signed cones are each split into orthant (or sign) cones with non-empty interior, as part 2 needs. |
| Scripts | 3 Robustness | GAP (minor) | No assertion that v is rational (finding 11). |
| Theorem T32 | 2, 4 Proof | OK | Covering cone b <= -(abs(w_1) + abs(w_2)) is right: m_1 - m_3 = w_1, m_2 - m_4 = w_2, b = -sum m - tau, and min sum m = abs(w_1) + abs(w_2). T29 is not even needed: D22 gives the cone directly. Z_S is the square. Absorbing output weights and unordered pairs (1176 = 48 * 49 / 2) are fine. The upper bound 3/80 is exact. |
| Theorem T32 | 5 Clean | OK (text GAP) | C4 acts faithfully, W rho = pi W, pi b = b, invariant mix. The readout is not fixed (finding 12). |
| Theorem T32 | 6 Path | GAP (minor) | R-fold is missing by T25's convention (finding 17). |
| Theorem T33 | 2, 4 Proof | GAP (minor) | "Broad iff w_3 >= (abs(w_1) + abs(w_2)) / 2" is right. The certificates are exact. The T27 citation is wrong, and the D_P sentence says more than the argument shows (finding 6). The theorem's conclusion holds. |
| Theorem T33 | 5 Clean | OK | Orbit with 4 elements, faithful, equal biases, invariant weights (3, 1, 2). Targets 0, 1/2, 1 checked. |
| Theorem T33 Comment | 1 Stand-alone | NOT MATH | Heuristic profile argument (finding 13). |
| Theorem T34 | 2, 4 Proof | OK | cone(B) = {w_3 <= 0, max(abs(w_1), abs(w_2)) <= -2 w_3} is right. cone(A) is in -cone(B) and -cone(A) is in cone(B), so the union is {abs(w_3) >= max(abs(w_1), abs(w_2)) / 2}. The script's rows match. The build's direction is outside: 10/9 < 11/6. 636 pieces, least bound 7/60. |
| Theorem T34 | 5 Clean | OK (text GAP) | Two orbits under C4. Reduced-form scales differ per family, and this is harmless but not stated (finding 12). |
| Theorem T34 | Content | GAP | Its unit is a covering fold of the cross-family group {A3, B2, B3} (finding 4). |
| Corollary T35 | 2 Proof vs statement | GAP | Universal wording from existence results (finding 2). Descriptions of an unproved "least build" (finding 3). KL not covered for items 1 and 3 (finding 1). The general sentence is not earned as written. |
| Corollary T35 | 3 Tag | GAP | Tag "none", but the statement opens with D20, full rank and invariant mixes as premises. If it is made existential (finding 2), "none" is right. |
| Remark B.F.1 | Options = premises | GAP | Missing: the objective (D2) and cross-family groups. Losses are misstated (findings 4, 5). |
| Conjecture D25 | Statement, route | OK | A conjecture with a route. Refuted for D2 by T32, and for KL by T33 (covering folds are a subset of broad folds). The status should name T33 for the KL half. |
| D25 status | 1 Stand-alone | NOT MATH (part) | Solver numbers, and an inconsistent description of T32 (finding 14). |
| Conjecture D26 | Statement, status | OK | Refuted for D2 and KL by T33. The status is correct. |
| Conjecture D27 | Status | GAP | "Refuted exactly by T34" holds only for the D2 half (finding 1). Route (3) is a non sequitur (finding 4). |
| D25 to D27 | 6 Paths | GAP | S-onehidden is missing (finding 7). |
| Card 3-3 | Claims | GAP | Overstates (findings 1 to 4). It rests on RETRO and PENDING nodes but is posed as a ROOT fork (finding 8). Losses are wrong (finding 5). |
| DECISIONS sections 2, 3, 8 | Consistency | GAP (minor) | Stale intro, stale tree, carriers (finding 16). Section 2 rightly omits T32 to T35 (they are at distance 4 or more). |
| Lean build | 4 | OK | Builds; no `sorry`. |

## 3. Content count (check 5)

| Kind | Items |
|---|---|
| Structural (what the coarse network is, or is not) | 3: T32, T33, T34. All three are negative. Each refutes one candidate class on one instance. |
| Bookkeeping or method | 2: T31 (linear algebra, a restatement of T28 part 1) and T32a (a duality certificate for any convex quadratic over polyhedral cones). |
| Collection | 1: T35. |

No result says what the least build is. The positive content is in the exhibited builds. Each exhibited unit is a
signed fold with a raised threshold (tau >= 0) of a small group whose weight cone is pointed:

| Theorem | Unit | As a fold |
|---|---|---|
| T32, unit 1 | relu(-4/5 x_1 + 4/5 x_2 - 8/5) | relu((4/5)(z_2 + z_3)) |
| T32, unit 2 | relu(-3/10 x_1 + 3/10 x_2) | relu(-(3/10)(z_1 + z_4) - 3/5) |
| T33 | relu(-x_3 / 2) | relu(-(z_1 + z_3)/2 - 1) |
| T34 | relu(-11/3 x_1 + 11/9 x_2 - 10/9 x_3 - 44/9) | relu(13/18 z_A3 + 75/36 z_B2 + 31/36 z_B3 - 11/9) |

This is data for option (c), not a proved result. A conjecture built on it must exclude groups whose cone holds a
line. Without that, the class becomes empty of content: in T32 with the whole family, and in T34 with the whole layer,
it holds every unit.

## 4. The three most important fixes

1. **Make T35, B.F.1, card 3-3 and the D27 status say exactly what is proved.**
   - State T35 as an existence result ("there exist D20-clean networks ... for which no least build is ...").
   - Tag each item with its objective. Say that for KL only broad folds are refuted (T33, which also covers D25's KL half), and that D27's KL half is open.
   - Attach the descriptions to the exhibited builds, not to "the least build". Use the singular for T32.
   - Give the losses as bounds: at least 1/9 relative; at least 1/468; at most 25/216 against at least 7/60.
2. **Complete the fork.** Add two options to B.F.1 and card 3-3:
   - (e) The objective: D2 is node R-obj (RETRO), and KL is the trunk's loss.
   - (f) Groups that cross families: the T34 unit is a covering fold of {A3, B2, B3}, which a D23 zoom allows.

   Fix D27 Route (3). Treat the fork as forced only under D2 and the pending D20 (node kind, finding 8). Say that T32 and T33 already rule out covering folds of every group, so (f) changes only what T34 refutes.
3. **Close the proof and ledger gaps.**
   - Replace T33's T27 citation and D_P sentence with the pointwise argument (or a Pinsker bound).
   - Add S-onehidden to the paths of D25 to D27 (and T31), and R-fold to T32's path.
   - Add Gordan and conic Caratheodory to the literature table, with their hypotheses.
   - Relabel `cert_ineq` as the completing-the-square step.
   - Fix the readout as (sum, 0) in T32 to T34.
   - Add the rational-type assertion on v to the three scripts.
