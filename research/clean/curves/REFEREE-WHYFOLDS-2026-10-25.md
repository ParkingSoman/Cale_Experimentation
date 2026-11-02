# Referee report: "Why folds" (Part B.Z), 2026-10-25

Scope: Definition D22 (with tau), Remark B.Z.0, Definition D24 (G1, G2, G3), Theorems T26 to T29, Corollary T30,
Remark B.Z.3 in `COARSE-THEORY.md`; card 3-2 and the paths of T26 to T30 in `DECISIONS.md`.

What I ran:

| Run | Result |
|---|---|
| `lake build` in `lean/coarse` | passes (9055 jobs; only long-line warnings) |
| grep for `sorry`, `admit`, `axiom` in `Coarse/*.lean` | none |
| `model/t26_weights.py` (copy in scratchpad) | N_2's gap = T; 26/7 (any), 26/7 (N_1 folds), 84/11 (N_2 folds) |
| my own grid check (240001 kinks per side) | 3.7142857 at a falling kink -10/7; 7.6363636 at a rising kink 20/11. Agrees |
| `model/t27_foldcost.py` | 0 violations (training never below the fold rule) |
| `model/t28_degenerate.py`, `model/t29_farkas.py` | all OK (but see F14 on what they test) |

## Summary table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| D22 (fold, tau >= 0) | definition | OK | The cone form is right: (w, b) = sum m_i (w_i, b_i) + tau (0, -1). Finitely generated, so closed |
| D22, sentence "T29 shows these are exactly the units that fire only where some member fires" | proof link | GAP | True only when Z_S is not empty (T29's own premise), and only among single units relu(w^T x + b) on all of R^d |
| D22, "the one-hidden-layer, ReLU form of D10's non-negative units" | definition | GAP | A non-negative mix of member outputs, sum m_i relu(z_i), is not a fold (Proposition T1'). The claim needs a proof or must go |
| Remark B.Z.0 | proof | OK | Both counterexamples are right. A fold of the silent unit alone is 0 on P |
| Remark B.Z.2, "cannot use a unit of N that is silent on P" | scope | GAP | False for a group that mixes the silent unit with active ones (F12) |
| D24 G1 | definition | GAP | Contradicts B.T's reading of "does not train". It is also too weak to force folds (F1, F2) |
| D24 G2 | definition | GAP | Holds for EVERY unit when W has rank d. Defined for units, used for rules (F2, F5) |
| D24 G3 | definition | GAP | A fair reading, but a modelling choice. Its answer depends on the domain (R^d, not the image cube) and on the unit form (F4) |
| T26 part 1 | proof | OK | One line. Bookkeeping |
| T26 part 2 | proof | OK | N_2 = T on the points. Cones {b <= -abs(w)} and {w >= 0, b <= 3w} are right. 26/7 and 84/11 confirmed. Groups not named (F11) |
| T26, G1 by one example | quantifier | OK | G1 is existential, so one pair is enough. Different least values give disjoint least sets; say so in one line |
| T27 part 1, count | proof | OK | (2 G R)^k with R <= 2 sum_{i<=s} C(n_P - 1, i) is a valid upper bound. The orthant only removes regions |
| T27 part 1, "independent of d", "exponent of order s k" | statement | GAP | The rank of the zeta_j is at most min(s, d) + 1. G can be 2^n. T22's hardness carries over to folds with s in place of d (F3) |
| T27 part 2 vs part 1 | fairness | GAP | Compares a lower bound for all teachers (under N-anynet, a DROPPED node) with an upper bound for folds (F3) |
| T28 part 1 | proof | OK | The kernel-vector argument is right. Bookkeeping |
| T28 part 2 | proof | OK | c = min makes every target >= 0. A linear solve. Bookkeeping |
| T28 Comment ("fewer than d + 2 images") | statement | GAP | Drops affine independence. T20 part 4 is a counterexample (F6) |
| T28 part 3 | proof | GAP | Holds for one fixed group per unit. Choosing groups adds k log G. W counts the gap, not two logits. The hypotheses of the cited bound are not stated (F7) |
| T29 | proof | OK | Farkas is applied correctly: A = W_S, right side -b_S, delta = -b, so tau = sum m_i b_i - b >= 0. The empty case is right |
| T29 Comment (G3 implies G2) | proof | OK | Right when Z_S is not empty |
| T30 part 1 | statement | GAP | "training in both senses" contradicts B.T. "Exponent of order d" uses T22, whose path has N-anynet (F1, F3) |
| T30 part 2 | statement | GAP | Per group with Z_S not empty, among single ReLU units on R^d: right. G2 used at unit level without saying so |
| T30 part 3 | statement | GAP | "Instead of d" is a change of parameter, not an escape (F3) |
| T30 final sentence | logic | GAP | Only G3 does the work. G1 and G2 do not force folds. Z_S not empty is dropped (F2) |
| Remark B.Z.3 | honesty | GAP | Honest that folds can lose. Silent on F2 to F4. "The mix's natural law" is not defined |
| Remark B.Z.3, "The essay's crush is evidence ..." | stand-alone | NOT MATH | A measurement from the essay, not a fact, hypothesis or proof. Allowed only as motivation |
| Card 3-2, "instant switching is easy" | stand-alone | NOT MATH | An engineering claim with no proof (F13) |
| Card 3-2 | claims | GAP | "The goals force folds", "samples of order k s", "at most d + 1 images" and the ETH comparison all overclaim. An option is dropped (F9) |
| Paths T26 to T30 | ledger | GAP | S-twoclass, S-onehidden, N-anynet and R-obj are missing where proofs use them (F8) |
| Lean `fold_covers` | faithfulness | OK | A pointwise form of "a fold covers its group". Stated honestly as the easy direction |
| `lake build`, `sorry` | build | OK | Passes; no `sorry` |
| Literature: Schrijver, affine Farkas | hypothesis | OK | Non-empty polyhedron stated and used. Read from secondary sources, said openly |
| Literature: Bartlett et al. 2019 | hypothesis | GAP | Hypothesis column says "none". The bound needs a fixed architecture with W parameters and piecewise-linear activations |
| Literature framing (SAE, dictionary, superposition) | framing | OK | None enters |

Content count (check 5). Structural (says what the coarse network IS): 2. They are T29 (covering = fold, by Farkas)
and T26 part 2 (the fold rule sees the weights, one exact example). Bookkeeping, re-parametrisation or citation: 6.
They are T26 part 1 (the loss reads labels), T27 (T21 in the coordinates zeta), T28 part 1 (row space), T28 part 2
(interpolation through d + 1 points), T28 part 3 (a cited bound) and T30 (a collection). The subsection's weight
rests on T29 alone.

## Findings, ranked by severity

### F1 (high). The paper calls distillation "training". Its own reading of "does not train" says the opposite

| | |
|---|---|
| Location | D24 G1, the parenthesis "which is training"; T30 part 1, "training in both senses"; card 3-2, row 1 |
| Problem | The B.T reading table says: "does not train" = an algorithm with a proved guarantee; T21's exact global optimisation "counts as 'not training'". T30 now calls the same exact rule training. The fold rule of T27 is also exact fitting of a small network to N's labels on P. Only the class differs. So "training" does not separate the two rules. The "second sense" (cost grows with the input size) is not defined anywhere, and it does not separate them either (F3) |
| Fix | Delete "which is training" and "training in both senses". State G1 as what it is: the output is not a function of N's answers on P. Call it a necessary condition. Say that both rules are exact minimisation of D_P over a class, and that the classes differ. If "training" is to be a condition, define it once, and check B.T against the definition |

### F2 (high). The three goals do not jointly force folds. Only G3 does

| | |
|---|---|
| Location | T30 final sentence; card 3-2 ("So (T30): the goals force folds"; the "What it forces" column) |
| Problem | (a) G1 is existential. Any rule whose class depends on N's weights passes it: subset selection of N's units, linear-span folds, or any function of the weights. G1 rules out the any-unit rule. It does not pick folds. (b) G2 uses the fibres of z = W x + b. When W has rank d, every fibre is one point, so EVERY unit passes. A first layer with n >= d units in general position has rank d. Example: in Remark B.Z.0's network the rows (1, 0), (-1, 0), (0, 1) span R^2. So relu(x_2), the T25 build, passes G2. G2 therefore does not remove T25's failure. (c) G3, with Z_S not empty, is equivalent to D22 (T29). So the whole case is G3, and G3 is a formal reading of "fold up what exists" |
| Fix | Rewrite T30's conclusion as follows. "G3 (Z_S not empty, single ReLU units, domain R^d) holds exactly for D22's folds. G3 implies G2. G2 is empty when rank W = d. The any-unit rule fails G1 (T26 part 1). The width-1 fold rule satisfies G1 (T26 part 2)." In card 3-2, say plainly that the decision rests on G3. Say that G3 is one reading of the goal, and that other readings give other answers (F4) |

### F3 (high). The complexity comparison (T27 against T22) is not fair, and T22's hardness carries over to folds

| | |
|---|---|
| Location | T27 statement ("independent of the input size d"); T30 parts 1 and 3; card 3-2, row 1 |
| Problem | (a) T22 is a lower bound in the worst case, for every teacher. Its path has N-anynet: the teachers memorise labels, and Shonu's R-least decision chose not to require "every network". T27 part 1 is an upper bound for folds. (b) The hardness carries over. Take the group S = {relu(e_l^T x), relu(-e_l^T x) : l = 1..d}, so s = 2d and Z_S = {0} is not empty. D22's folds of S are all units with b <= 0. Build the teacher from Lemma T22a for the labels minus a linear part. Add the 2d units, with output weights that give back the linear part (relu(t) - relu(-t) = t). The labels are unchanged and no unit is dead. T22's clique build B* has b = 1 - 2/eta <= 0, because eta <= 2, so B* is a fold. T22's other direction ("L(B) < Gamma + 1/16 gives a clique") holds for every build, so it holds for folds. So the least fold build of width 1 has no algorithm with running time rho(s) L^{o(s)} (L the bit length of the input) unless ETH fails. (c) The zeta_j lie in an affine image of R^d, so the arrangement has rank at most min(s, d) + 1. The exponent is of order min(s, d) k, not s k. (d) The factor G^k is not small: if any group may be chosen, G = 2^n |
| Fix | Add a proposition: "T22 holds for fold builds with s in place of d" (the proof above). Restate T27 with min(s, d) and G^k shown. Delete "instead of d" from T30 and from the card. Add N-anynet to the paths that use T22 (F8) |

### F4 (medium). "Folds are the largest class with G3" holds only under three premises that are not stated

| | |
|---|---|
| Location | D22 (the T29 sentence and the D10 sentence); D24 G3; T29; T30 part 2 and final sentence; card row 3 |
| Problem | (i) Unit form. T29 classifies units of the form relu(w^T x + b). The unit sum m_i relu(z_i) with m >= 0 also covers S, and it is not one ReLU unit (Proposition T1'). So "largest class" is "largest class of single ReLU units". The D10 sentence in D22 is also not proved. (ii) Z_S not empty. D22's sentence and T30's final sentence drop it. When Z_S is empty, every unit covers S. By Farkas, (0, 1) is then in the cone of the (w_i, b_i), so D22's cone is cone(w_i) x R. That is strictly smaller than all units unless the w_i positively span R^d. (iii) Domain. G3 asks for silence on all of R^d. On a bounded domain K, such as the cube [0, 1]^d of B.T's reading, the class grows. Example: d = 2, S = {relu(x_1 - 1)}, K = [-3, 3]^2. The unit relu(x_1 + 0.1 x_2 - 1.5) is 0 wherever x_1 <= 1 in K, but (1, 0.1) is not in the span of (1, 0) |
| Fix | Put all three into D24 G3 and T29: unit form, domain R^d, Z_S not empty. Require Z_S not empty in D22 and D23 (groups must have a common silent set), or carry it into T30's conclusion. Delete the D10 sentence, or prove it for a stated reading of D10. Add one sentence: G3 on a bounded domain gives a larger class (a Farkas statement with box multipliers) |

### F5 (medium). G2 is a condition on units, but T30 applies it to rules

| | |
|---|---|
| Location | D24 G2 ("three conditions on a rule"); T30 part 2 |
| Problem | D24 never says what G2 means for a rule. A build-level G2 ("y_B is constant on the fibres of z") is strictly weaker than the unit-level one. Take r in the row space of W and v in its kernel. The width-4 gap relu((r+v)^T x) - relu(-(r+v)^T x) - relu(v^T x) + relu(-v^T x) equals r^T x. The build passes; its units fail |
| Fix | Define it: "a rule satisfies G2 if every unit of every returned build reads only through N", and call this a choice. Or prove that the two levels agree for builds with no pair of opposite units (along a fibre line, kinks can cancel only between units with proportional (w, b)) |

### F6 (medium). The T28 comment and card 3-2 drop affine independence, and the statement is then false

| | |
|---|---|
| Location | T28 Comment ("For a developer's sample of fewer than d + 2 images ... a perfect answer for every network"); card 3-2 row 2 ("a sample of at most d + 1 images gets a perfect copy of EVERY network") |
| Problem | T20 part 4 is a counterexample: seven collinear points in R^d with d >= 6, and the least width-1 value is positive |
| Fix | Add "affinely independent (true for points in general position)" in both places |

### F7 (medium). T28 part 3 is stated for one fixed group per unit; the card turns it into a sample claim

| | |
|---|---|
| Location | T28 part 3; literature row for Bartlett et al.; card row 2 ("Folds need samples of order k s, not d") |
| Problem | (a) The rule picks a group for each unit: a union of up to G^k classes. Pseudo-dimension then grows by O(k log G). (b) W = k(s + 2) + 1 counts the parameters of the gap network, not of two logits. (c) The cited bound needs a fixed architecture with W parameters and piecewise-linear activations; the literature row says "none". (d) A pseudo-dimension upper bound gives a sample size only through a uniform-convergence theorem for a bounded loss. None is cited, and KL with unbounded logits is not bounded. (e) "Not d" fails when s >= d: the folds of {±e_l} include every unit with b <= 0, whose pseudo-dimension is at least d |
| Fix | State "for a fixed assignment of groups; with G choices per unit add k log G". Say "the gap". Fill the hypothesis column. Delete "need samples of order k s" from the card, or prove a sample bound with a stated bounded loss |

### F8 (medium). Paths and tags miss premises that the proofs use

| | |
|---|---|
| Location | `DECISIONS.md` section 6, rows T26 to T30; tags in Part B.Z |
| Problem | T28 part 2 says "two logits" in its statement, and part 3 counts a gap; the path lacks S-twoclass. T28, T29 and T30 sit in an L = 1 section; T27 lists S-onehidden but they do not. T30 collects T22 (N-ny2, N-onehidden, N-anynet), T26 part 2 (R-obj, N-ny2) and T27 (S-twoclass, S-onehidden), but its path is only R-fold. T27's tag omits "a finite mix" (T19 and T21 have it). "Z_S not empty" is a premise in T29 and T30 with no TECH entry. T26 is tagged "(under: none)" while part 2 is D2 only (the path has R-obj; this follows the T25 precedent) |
| Fix | T28 path: add S-twoclass and S-onehidden. T29 path: S-onehidden (or say that the statement holds for any affine z_i, which it does). T30 path: the union of its sources, N-anynet included. Add "a finite mix" to T27's tag. Add a TECH alias for "Z_S not empty" |

### F9 (medium). Card 3-2 narrows the decision Shonu is asked to make

| | |
|---|---|
| Location | card 3-2, options table; compare Remark B.Z.2 and node R-fold in section 4 |
| Problem | B.Z.2 and the node R-fold list three ways out: folds; a property of the mix; any units, accepting accidental features. The card offers two. The mix-property option is gone without a reason |
| Fix | Restore it, with B.Z.2's Jensen caution, or state why it is dropped |

### F10 (low). G1 must fix the architecture

| | |
|---|---|
| Location | D24 G1 |
| Problem | G1 quantifies over all N, N'. If N and N' may differ in depth, the trunk's any-unit rule (builds "of the same kind as N") returns different classes, so it passes G1 trivially. Then T26 part 1 does not make it a distillation rule |
| Fix | "two networks N, N' with the same d, n_y and depth" |

### F11 (low). T26 part 2 does not name its groups

| | |
|---|---|
| Location | T26 part 2 |
| Problem | "fold builds of width 1": folds of which groups? The cones used are those of the whole layer. That equals the union over all groups, since each subgroup's cone is inside it. Also, G1 needs different returned sets; the proof shows different least values |
| Fix | Say "folds of any group (equivalently, of the whole layer)". Add: D2 of a fixed build depends only on its logits on P, so different least values give disjoint least sets |

### F12 (low). Remark B.Z.2 overstates what B.Z.0 shows

| | |
|---|---|
| Location | Remark B.Z.2, "it cannot use a unit of N that is silent on P either" |
| Problem | B.Z.0 treats the group made of the silent unit alone. In the group {z_1, z_2, z_3}, a fold can have m_3 > 0, and its values on P then change with x_2. Example: 100 z_1 + z_3 at x_1 = 2 gives 1, and without z_3 it gives 0. On T25's mix this does not help: the reachable data vectors on {2, 3} keep the ratio >= 2. But the general sentence is false |
| Fix | "cannot use a silent unit in a group of silent units" |

### F13 (low). Remark B.Z.3 and the "What comes next" paragraph

| | |
|---|---|
| Location | Remark B.Z.3; card 3-2, last paragraph |
| Problem | B.Z.3 is honest that folds can lose. It says nothing about F2, F3 and F4. "The mix's natural law" is not defined. The essay's crush is a measurement: NOT MATH, allowed only as motivation. "Instant switching is easy (store the best fold build for each width)" is an engineering claim. D23 itself says that switching costs order n, and a table grows with the number of widths and mixes |
| Fix | Add to B.Z.3: "the case rests on G3; G2 is empty at full rank; the fold rule is as hard in s as the any-unit rule is in d". Define the law, or drop the sentence. Replace "easy" with what is proved: a table of least fold builds, one per width, exists for each mix |

### F14 (low). The checks do not test the hard cases

| | |
|---|---|
| Location | `model/t29_farkas.py`, `model/t27_foldcost.py`, `model/t28_degenerate.py` |
| Problem | t29 tests random w in R^4 against a group of 3. Such a w is almost never in the span, so in-span units with a negative m or tau < 0 (the cases B.Z.0 is about) are never tested. t27 is one-sided: training never beats the rule. t28 part 1 checks only W v = 0 |
| Fix | In t29, add in-span units with one negative m_i, and units with tau < 0. In t27, add a check of the pieces against a dense grid of u at s = 1 |

## The three most important fixes

1. **Rewrite T30, its final sentence and card 3-2 so they say what is proved.** The case rests on G3 alone. State G3's
   three premises: single ReLU units, domain R^d, Z_S not empty. G1 is only a necessary condition. G2 is empty when
   rank W = d. Delete "training in both senses": it contradicts the B.T reading. (F1, F2, F4)
2. **Make the complexity comparison honest.** Prove that T22 carries over to folds with s in place of d (group {±e_l},
   b <= 0). Restate T27 with exponent min(s, d) k and the factor G^k. Note that T22 rests on N-anynet. Then delete
   "instead of d" from T30 part 3 and from the card. (F3)
3. **Repair the ledger and the card's false sentences.** Add S-twoclass, S-onehidden, N-anynet and R-obj to the paths
   of T28 to T30. Add affine independence to the T28 comment and the card. Delete "samples of order k s". Restore the
   mix-property option in card 3-2. (F6, F7, F8, F9)
