# Referee report: Part B.C and the Part 0 paragraph on T** (2026-10-25)

Scope: `COARSE-THEORY.md` Part 0 (trunk paragraph) and Part B.C (D18, D19, Lemma T24a, Theorem T24, Corollary T24.1,
Remark B.C.1, D20, D21). `DECISIONS.md` sections 1, 2, 3 (card 3-1), 4 (nodes R-score, R-cleannet) and 6 (paths).
Check script `model/t24_score.py` was run: ALL OK. `lake build` passes. No `sorry` in `Coarse/*.lean`.

## 1. Table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| Part 0, T** paragraph | stand-alone | GAP | It writes eps(N) with no group. D21 defines eps_Gamma(N). With Gamma trivial, eps = 0 for almost every network. |
| Part 0, lines after T** | stand-alone | GAP | It still measures distance from T* ("smaller than T*", "distance 0"). T** has no mix P, so T20, T22 and T23 are not at distance 0 from T**. |
| Part 0, "T24 shows the score must be chosen" | overclaim | GAP | T24 refutes one reading (score on the training law). It does not force a choice among exactly three options. |
| D18 goal, goal-weighted law, goal score | definition | GAP | Well defined (w_g > 0, mu_g a law, S_{mu,1} = D_mu). But the goal only reweights inputs. The divergence still compares all classes. This choice has no node. Notation clash: "mu_1" is D18's goal-weighted law for g = 1 and also T24's uniform law. |
| D19 well-trained / optimal | definition | GAP | Correct, and "optimal implies well-trained" holds. But "of its kind" leaves the width open. With Lemma T24a it is vacuous for self-labelled laws. Its use as the reading of the trunk's "trained well" has no node. |
| Lemma T24a | proof | OK | Gibbs identity, valid in [0, inf]. The converse (N optimal on Q implies Q's label law = q, mu-a.e.) is true and short but not stated. Remark (iii) needs it. |
| Lemma T24a | content | bookkeeping | True of any map to probability vectors. Says nothing about what a coarse network is. |
| Theorem T24 part 1, uniqueness on {0,1,2,3} | proof | OK | Second differences give t in (0, 1]. w < 0 is excluded at x = 2, 3. Then gamma = 0, t = 1, beta = 1. Checked by hand and by the script. |
| Theorem T24 part 1, use of T20 part 2 | proof | OK | mu_1 is T20's P. A mu_2-best build has unit = w relu(x - 1), w > 0. T20 part 2 bounds every alpha, gamma for that unit. D_P half rests on T20's interval check (not Lean). |
| Theorem T24 part 1, existence of best builds | proof | GAP | "No build is best for both" does not need existence. Part 3 is non-vacuous only if each law has a best build. T19 (mu_1) and the explicit minimisers (mu_2, mu_3, mu_4) give it, but T24 does not say so. |
| Theorem T24 part 2, piecewise argument | proof | GAP | The method is right. The rational functions and piece minima are not in the paper. The proof is "see the script". Two hand computations (below) would make the key steps readable. |
| Theorem T24 part 2, mu_3 minimiser 14/11 | proof | OK | Verified by hand: on the piece (1, 2), V >= min_g [(g-2)^2 + (g-1)^2 + 9 g^2] = 46/11 at g = 3/11, with equality only at t = 14/11. Other pieces: script only. |
| Theorem T24 part 2, mu_4 set [2, 3) | proof | GAP | The set is right. The proof says "critical points and ends" but does not say why t = 3 is excluded. At t = 3 the unit is zero on the support, the build is constant, and V = 90/11, not 6. The value 6 is a limit there, not reached. |
| Theorem T24 part 2, mirror units | proof | OK | relu(t - x) is the mirror of relu(x + t). Both laws and T are mirror-symmetric, so V^-(t) = V^+(-t). |
| Theorem T24 part 2, disjointness | proof | OK | mu_3-best data vectors have support {2, 3} or {-3, -2}. mu_4-best data vectors have support {3} or {-3}. No positive multiple matches. |
| Theorem T24 part 2, objective | tag/path | GAP | Holds only for D2. D2 is node R-obj. R-obj is not on T24's path. Remark B.C.1 and card 3-1 quote part 2 without "for D2" in the card. |
| Theorem T24 part 3 | proof | OK | One output for two laws, and no build is best for both. The qualifier "for every law on which N is optimal" is empty by T24a: it means "for every law". |
| Theorem T24 | content | structural (negative, small) | The content is "the best build depends on the input law, and optimality does not restrict the input law". Expected once T24a is stated. |
| Theorem T24 | Lean | GAP | `t20_d2_gap_a` states the D2 bound for the unit relu(x - 1). The proof also needs `t20_d2_competitor` (not named). The part-1 uniqueness and the part-2 piece minima are not in Lean. Both reduce to one-variable quadratics (nlinarith size). |
| Corollary T24.1 | proof | OK | Point masses are laws, w_g(x) > 0 keeps them point masses, and an average is at most the sup. Sup, not max, is correct (unbounded input space). |
| Corollary T24.1 | content | bookkeeping | True of any score that is an average of a pointwise loss. |
| Corollary T24.1 | consequence not stated | GAP | The right side has no g. So option (iii) does not depend on the goal. The trunk asks for "best for G". |
| Remark B.C.1 (ii) | overclaim | GAP | By T24a, any property implied by optimality on Q holds for every input marginal. A property that fixes the best build must restrict mu itself. The remark does not say this. Cites `PAPER.md` 35.2 (marked "evidence only": acceptable as a pointer only). |
| Remark B.C.1 (iii) | overclaim | GAP | "Defined from N alone" is right, and it is the problem: G drops out (T24.1). "Driven by inputs that are not images" is a belief, not proved. |
| Remark B.C.1 (i) | overclaim | NOT MATH | "No canonical rule is known yet" is a literature claim with no source. Data-free distillation builds input laws from N alone (for example Nayak et al. 2019, zero-shot KD; Yin et al. 2020, DeepInversion). None of them is in the literature table. |
| Remark B.C.1, "items 4 to 6 of the plan" | stand-alone | GAP | Refers to a plan file outside the paper. |
| D20, condition 1 | definition | GAP | "An action" implies a homomorphism, but it is not written. Without it, duplicate units allow intertwiners that are not homomorphisms, and the coupling across layers (rho_l = pi_l) means a homomorphic choice is not automatic. |
| D20, "with Gamma trivial every network is perfectly clean" | definition | GAP | False as written: condition 3 (every unit read) can fail. Correct form: every network whose hidden units are all read. |
| D20, invariance | definition | GAP | Not invariant under the ReLU rescaling symmetry. Example: relu(2x) + 2 relu(-x) = 2 abs(x) is mirror-invariant as a function, but W rho_0 = (-2, 1) and pi W = (-1, 2). Duplicate units with split read weights fail too. Cleanliness depends on the parametrisation, not the function. |
| D20, strength | definition | GAP | Units that ignore one input coordinate are invariant under the mirror of that coordinate (pi = identity). So any network becomes perfectly clean for a nontrivial Gamma after one ignored coordinate is added. D20 alone does not exclude the hard instances of T22. |
| D20, mirror example | check | OK | W1 = (1, -1)^T, b = (-1, -1), W_out = [[1, 1], [0, 0]], c = 0, rho_0 = -1, pi = swap, sigma = I. All four equations hold. Both columns of W_out are non-zero. |
| D21, "eps = 0 exactly when perfectly clean" | proof | OK (with fixes) | Correct if the minimum runs over homomorphisms (a finite set, so it is reached) and 0/0 is defined (W_l = 0 or W_out = 0). |
| D21, continuity | definition | GAP | The value +infinity for an unread unit makes eps jump. A read weight of 1e-12 gives a finite eps. A bound "that depends only on eps and is 0 at eps = 0" cannot see this. |
| D20, D21 placement | stand-alone | GAP | Node R-cleannet says "Definition in Part A". They sit in Part B.C. |
| T24 path (N-ny2, N-onehidden) | path | GAP | Missing: N-finite (all four laws are finite; T20's path has it), R-obj (part 2, and the D2 half of part 1), TECH "k = 1", d = 1 (T20 part 4 lifts to every d; T24 does not say so). The "perfectly clean" sentence uses D20, which is under PENDING R-cleannet. |
| T24a path (none) | path | OK | Uses only Gibbs. D19 adds no premise to the proof. |
| T24.1 path (none) | path | GAP (minor) | Uses D18's form of the goal (input reweighting with w_g > 0). That form is an unmarked choice. |
| DECISIONS section 2 | claim | GAP | Still headed for T*. No row for T24, T24a or T24.1, the only results at T**. Distance-0 claims are relative to T*. |
| DECISIONS section 3 header | claim | GAP | Says "R-nest and R-root, PENDING". R-nest and R-least are DECIDED in section 4. |
| DECISIONS section 3 preamble vs 3-1 | claim | GAP | "Options are exactly the premises of those theorems." In 3-1, (i) and (iii) both drop one premise (score on the training law). The premises N-ny2, N-onehidden, width 1 and D2 are not discussed as exits (3a does this for T20). |
| DECISIONS R-score row vs card 3-1 | claim | GAP | Row says "working reading (i)". Card recommends (ii). |
| Card 3-1 text | claim | GAP | "Take a perfectly clean network ... the best possible predictor on any list labelled by itself": true of every network (T24a). Cleanliness does no work in that sentence. "Even two lists that are themselves mirror-symmetric" needs "for D2". |

Count for check 5. Results: 3. Structural: 1 (T24, a small counterexample). Bookkeeping: 2 (T24a, T24.1). Definitions: 4. No result in Part B.C says what the coarse network is. That is expected for a section that only sets up the score, but it is a finding: the section has no positive result.

Literature (check 7): no SAE, dictionary or superposition framing enters. D20 is weight-sharing equivariance with permutation representations. It is not cited. Sources to add to the table: Wood and Shawe-Taylor 1996 (representation theory and invariant networks), Ravanbakhsh, Schneider and Poczos 2017 ("Equivariance Through Parameter-Sharing"). For the rescaling fix: Phuong and Lampert 2020 ("Functional vs. parametric equivalence of ReLU networks"). For option (i): the data-free distillation papers named above, with their own hypotheses (none of them has a proved guarantee).

## 2. Findings, ranked by severity

### F1 (high). Option (iii) has no goal

Location: Corollary T24.1; Remark B.C.1 (iii); DECISIONS card 3-1 row (iii).

Problem: sup over x of KL(q(x) || softmax(y_B(x))) does not contain g. Point masses are admissible for every x, and w_g(x) > 0 keeps them point masses, so the goal weighting cancels. The worst-case score is defined from N alone, not from N and G. The trunk asks for "the best for G". The card calls (iii) "fully defined from N alone, today" and "ready", without saying that the goal is lost.

Fix: state it as part 2 of T24.1: "the right side does not depend on g". In the card, say (iii) is incompatible with "best for G" under D18. If (iii) is kept, the goal must enter the pointwise loss, not the input weighting. One candidate: merge the classes with g_c = 0 into one class before the KL. Make that a new definition with a node.

### F2 (high). "Well-trained" (D19) is vacuous for self-labelled laws, and its role in T** has no node

Location: D19, Lemma T24a, Theorem T24 statement 1 and 3, Remark B.C.1 (ii), card 3-1 first paragraph, R-least row.

Problem: T24a shows every network is optimal on Q_mu for every mu. So "N is optimal on Q_{mu_1} and Q_{mu_2}" says nothing about N. The qualifier "for every law on which N is optimal" in part 3 and T24.1 means "for every law". The converse is also true and not stated: if N is optimal on Q (finite cross-entropy), then Q's label law equals q, mu-a.e. Together these say: optimality fixes the labels and leaves the input marginal free. So any "stronger joint property" in option (ii) must restrict the marginal mu. A property phrased with labels drawn from N is still empty. Also, Shonu's R-least choice ("trained well and clean") is read here as D19 plus D20 with no node, and under that reading it does not escape T23 (T23's teacher is clean for the trivial group when all its units are read, and it is optimal on its own labels).

Fix: (a) Add the converse to T24a: "N is optimal on Q if and only if Q = Q_mu for its marginal mu". (b) Restate T24 part 3 as "for every input law" and say in one line that this is what "optimal" allows. (c) In Remark B.C.1 (ii), say that the property must restrict mu. (d) Give the reading "well-trained = D19" a node, or say D19 is not the reading of R-least.

### F3 (high). Hidden premises on T24's path

Location: DECISIONS section 6, row Theorem T24.

Problem: the path lists N-ny2 and N-onehidden only. The proof also uses: laws on finitely many points (N-finite, as in T20's row); the objective D2 for part 2 and for half of part 1 (R-obj, a CONV node, so it changes the distance); width 1 (TECH "k = 1"); input dimension 1. The sentence "Theorem T24 is a statement about a perfectly clean network" uses D20 (PENDING node R-cleannet).

Fix: split the row. Part 1: N-ny2, N-onehidden, N-finite, TECH (k = 1). Part 2: add R-obj. Add a part 4 to T24 that lifts to every d, as T20 part 4 does, or add a d = 1 entry. Mark the cleanliness sentence as "under R-cleannet (PENDING)".

### F4 (medium). T24 part 2 is not readable without the script, and the t = 3 end is not explained

Location: T24 proof (2).

Problem: the paper gives no formula. The reader must trust `model/t24_score.py`. The end t = 3 is a limit in the script, not a value. At t = 3 the unit is zero on the support, the build is the constant build, and V = 90/11 for mu_4.

Fix: print the two key computations in the proof:
- mu_3, kink in (1, 2): the unit is zero at x <= 1, and alpha and t can match T at x = 2 and x = 3 exactly. So V >= min_g [(g-2)^2 + (g-1)^2 + 9 g^2] = 506/121 = 46/11, at g = 3/11, with equality only when (3 - t)/(2 - t) = (2 - g)/(1 - g) = 19/8, that is t = 14/11.
- mu_4, unit supported on {3}: V = min_g [3(g-2)^2 + 2(g-1)^2 + 3 g^2] = 6 at g = 1.

Then add one sentence: "at t = 3 and beyond the unit is zero on the support, the build is constant, and its value is 90/11 > 6". Add a table of the minimum of V on each of the eight pieces (two numbers per law). Put the two quadratic bounds and the mu_3 value of the e_3 unit (14/3 > 46/11) in Lean, in the style of `t20_d2_gap_a`.

### F5 (medium). D20 is parametrisation-dependent, its "trivial Gamma" sentence is wrong, and it does not restrict the function class

Location: D20, D21, Part 0 ("eps(N)").

Problem:
1. "With Gamma trivial every network is perfectly clean" is false: condition 3 can fail.
2. The homomorphism is implicit in the word "action". D21 minimises over "permutation actions"; this set must be homomorphisms Gamma -> S_{n_l}, or eps = 0 may hold without a family structure.
3. Not invariant under the positive rescaling of a ReLU unit, or under splitting a unit into duplicates. Example: relu(2x) + 2 relu(-x) is mirror-invariant but not perfectly clean.
4. With pi_l = identity allowed, any network that ignores one input coordinate is perfectly clean for the mirror of that coordinate. So D20 does not exclude the hard instances of T22. An exact closed form "whenever eps = 0" cannot come from D20 alone.
5. The trunk writes eps(N). D21 gives eps_Gamma(N). With Gamma trivial, eps is 0 for every network whose units are all read.
6. The +infinity for an unread unit makes eps discontinuous.

Fix: (a) Write "a group homomorphism pi_l: Gamma -> permutation matrices" and the same for sigma. (b) Correct the trivial-Gamma sentence. (c) Define D20 on a reduced form: merge duplicate units, and normalise each unit's incoming weights to norm 1, moving the scale to the next layer. Then cite Phuong and Lampert 2020 for when the reduced form is unique. (d) Make Gamma an explicit input of T** (or of node R-cleannet), and require that Gamma acts without fixed units, or bound the number of orbits per layer. State in the card that, without such a condition, D20 contains T22's instances (one line of proof: the ignored coordinate). (e) Replace the +infinity by a continuous read term, for example min over units of |column| / |W_out|, or drop condition 3 from eps and keep it as a separate condition. (f) Move D20 and D21 to Part A, as the node row says.

### F6 (medium). Stale claims at the trunk

Location: Part 0, the paragraph after T**; DECISIONS sections 2 and 3.

Problem: Part 0 says B.C works on T** and then says "every result ... is about a smaller question than T*" and lists distance-0 results relative to T*. Section 2 has no T24 row. Section 3's header says R-nest is PENDING, while section 4 says DECIDED. The R-score row says "working reading (i)", while card 3-1 recommends (ii). Section 3 says the options are "exactly the premises" of the theorems. For 3-1 that is not so: (i) and (iii) are two ways to drop one premise, and the other premises of T24 (two logits, one hidden layer, width 1, finite laws, D2 for part 2) are not discussed as exits.

Fix: retitle section 2 "What is known at T* (the old trunk)" and add a section for T** with T24, T24a, T24.1. In Part 0, say which distances are measured from T*. Make the R-score row and card 3-1 agree. In 3-1, add the sentence that 3a has for T20: the other premises are inside T**, so they are not exits.

### F7 (low). D18's goal is an unmarked choice

Location: D18; T24.1 path.

Problem: T** says "which outputs matter". D18 makes the goal reweight inputs. The KL still compares every class. That is one reading among several (merge unwanted classes; weight the KL terms by class). It changes T24.1's consequence (F1). It has no node.

Fix: give the reading a node under R-score, or state it as part of R-score's options. Rename T24's laws (nu_1 to nu_4) so that mu_g keeps its D18 meaning.

### F8 (low). Remark B.C.1 (i): unsourced literature claim

Location: Remark B.C.1 (i); card 3-1 row (i).

Problem: "No canonical rule is known yet" is a claim about the literature with no source. Data-free distillation builds input laws from N alone (Nayak et al. 2019, from the last layer's weights; Yin et al. 2020, DeepInversion, from batch-norm statistics). They have no proved guarantee, but they are candidates.

Fix: weaken to "no rule with a proved property is stated in this paper". Add those papers to the literature table, with their own hypotheses (batch-norm statistics as a description of the training law; class-similarity from the readout).

### F9 (low). Small items

| Location | Problem | Fix |
|---|---|---|
| T24 statement 2 | "46/11 (times the weight normaliser)" is loose. D2 = residual / (4 W), W = 13 for mu_3 and 11 for mu_4. | Write the factor. |
| T24 "Lean:" line | Omits `t20_d2_competitor`, which part 1 uses. | Name it. |
| T24 proof (1) | "the only best build ... (as a function)": only the gap function is unique, not the build. | Say "every best build has gap relu(x - 1)". |
| T24 proof | Existence of best builds is not stated. | Add: mu_1 by T19, mu_2 by value 0, mu_3 and mu_4 by the explicit minimisers. |
| Lemma T24a "Lean:" | Rule 2 asks for a finite version in Lean. | Gibbs for finitely many classes is short in Lean. |
| Remark B.C.1 | "items 4 to 6 of the plan" and `PAPER.md` 35.2 point outside the paper. | Name the trunk clauses instead. Keep 35.2 as a pointer only. |
| D19 | "Networks of its kind" leaves the width open. | Say "same depth and widths as N". |

## 3. The three most important fixes

1. **State what T24a and T24.1 really give, and change D18 or option (iii).** Add the converse of T24a (N is optimal on Q exactly when Q = Q_mu). Restate T24 part 3 and T24.1 as "for every input law". Add to T24.1 that the worst case does not depend on g. Then either move the goal into the pointwise loss (a new definition with a node), or remove (iii) as a way to reach "best for G".
2. **Fix T24's path and split it by part.** Add N-finite, TECH k = 1 and d = 1 (or a lift to every d) to both parts. Add R-obj to part 2. Mark the "perfectly clean" sentence as under R-cleannet (PENDING). Put the two hand computations of F4, and the t = 3 sentence, in the proof of part 2.
3. **Repair D20 and D21 before anything is proved under them.** Explicit homomorphisms. Correct the trivial-Gamma sentence. A reduced form (no duplicates, normalised units) so that cleanliness depends on the function. A stated condition on Gamma (no fixed units, or few orbits), since without it D20 contains T22's hard instances. Gamma as an explicit input of the trunk's eps.
