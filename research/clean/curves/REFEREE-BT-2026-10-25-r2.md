# Referee report, round 2: Part B.T (T23, restated T21, T20, T22 part 4, readings, D17, DECISIONS section 3)

Date: 2026-10-25. First report: `REFEREE-BT-2026-10-25.md`. Fixes listed in the Appendix entry "Referee report 11".
Files read: `COARSE-THEORY.md` (Part 0, A.4 rows, B.T, D.8, Appendix entry), `DECISIONS.md` (sections 1 to 6),
`lean/coarse/Coarse/Trunk.lean`, `model/t23_noattain.py`, `model/t20_nesting.py` (and its stored JSON).

## Verdict on Theorem T23

**T23 is correct.** Every step was re-derived by hand.

| Step | Check | Result |
|---|---|---|
| Reduction by G(y) = (y_1 - y_3, y_2 - y_3) | G is linear and onto R^2, so every 2 x 2 matrix A and every c occur | OK |
| D_P = 0 iff G matches | softmax(y) = softmax(y') iff y - y' is in span(1) iff G(y) = G(y'); every pi_i = 1/6 > 0 | OK |
| D2_P = 0 iff G matches | P_perp delta = 0 iff delta is in span(1) | OK |
| Limiting sequence | W s_i = (i - 3)(-1, 1) for i <= 5; W s_6 = (2m, m); outputs (c1_i, relu(x)/m) and (0, 1) | OK |
| Not reached: span step | 1, c1, c2 are independent (c2 = e_6; c1 is not constant on the first five points). So span(1, u_1, u_2) = U | OK (independence not written out) |
| Second differences | D c1 = (0, -1, 0) at x = -1, 0, 1; D 1 = D c2 = 0 | OK |
| beta < 0 forces t = 0 | supp h_t meets {-1, 0, 1} in {0} only: \|t\| < 1, \|t + 1\| >= 1, \|t - 1\| >= 1, so t = 0 | OK |
| beta = 0 | x = 0 gives alpha = 0; x = -2 (relu(x)) gives 2 beta = 0; x = 2 (relu(-x)) gives -4 beta = 0 | OK |
| Case a = 0 | not written. The unit is constant on the line, D u = 0, so beta = 0 at once | GAP (one phrase) |
| Teacher | Lemma T22a, vector form: five shared kinks, one broken line per logit, third logit 0. One hidden layer, ReLU, affine head. `t23_noattain.py` checks it exactly | OK |

`model/t23_noattain.py` ran here and printed "ALL OK: True". The objectives fall like 1/m^2.

Whether T23 refutes the trunk "as worded" is a separate question. The answer is: yes, but only under three readings
that Shonu has not reviewed (finding 1).

## Build and checks

| Check | Result |
|---|---|
| `lake build` in `lean/coarse` | passes (9055 jobs) |
| `sorry`, `admit`, `axiom` in `Coarse/*.lean` | none |
| `model/t23_noattain.py` | ran, all OK |
| `model/t20_nesting.py` | did NOT run here. The default Python has no sympy. Python 3.11 has a scipy/numpy version clash. The stored `t20_nesting.json` agrees with every printed fraction. A hand evaluation of D_P(B_c') gives about 0.04107, which agrees with the printed upper bound 410684/10^7 |

## Table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| Theorem T23 | proof | OK | Complete. The case a = 0 is one missing phrase |
| Theorem T23 | statement | GAP (minor) | Stated only for n_y = 3 and d = 2. Both extend at no cost (zero logits; affine embedding). Not stated |
| Theorem T23 | tag, path | GAP | "(under: none)" is right for an existence statement. Its transfer to T\* rests on three readings with no node (finding 1) |
| Theorem T23 | Lean | OK (weak) | `relu_second_diff` is the identity used. The finite core of the not-reached step is decidable and could be in Lean |
| Reading table, sentence on counterexamples | logic | GAP (major) | "No reading narrows the question" is the wrong test. A reading that WIDENS T\* can carry a counterexample outside the trunk |
| Reading "a trained network" | logic | GAP | "So it is the result of training on its own labels" does not follow from "it is a global minimiser" |
| Reading "does not train" | stand-alone | OK (note) | Honest, but the phrase now adds nothing to "an algorithm with a proved guarantee". Say so |
| Reading "of the same kind" | logic | GAP | Same depth narrows the build class. Width 3 already reaches T23's target. The NEG results need this reading |
| Theorem T21 part 1 | proof | OK | Pieces, region restriction, bound on R: correct |
| Theorem T21 part 2 | proof | OK | Incremental listing with margin-1 LPs is correct; at most 2R tests per step. Polynomial-time LP needs a citation (KTK covers it as a special case; say so) |
| Theorem T21 part 3 | proof | GAP (minor) | "The solution set meets the optimal face, so a rational minimiser exists" needs: the set of minimisers is a rational polyhedron |
| Theorem T21 part 3 | statement | OK (wording) | "Grows like n^{d k}" is an upper bound. Write "at most of order" |
| Theorem T21 part 4 | statement | OK | A non-claim, correctly worded |
| Theorem T20 part 1 | statement | OK | Quantifiers now right: every value-0 build has these units; each pair has a value-0 build |
| Theorem T20 part 4 | proof | OK | Data vectors depend on (w, b) only through w_1 and b. The teacher and the competitor embed too |
| Theorem T20 part 2, KL | proof | OK (minor gaps) | Fenchel bound correct. The paper does not say that s* lies in (0, 1)^7 (the script asserts it). Library and precision not named. s_0 not printed |
| `t20_nesting.py` | check | GAP (minor) | It re-derives the competitor by BFGS and asserts exact equality to the printed fractions. That depends on the machine. Evaluate the printed fractions directly |
| Theorem T22 part 4 | statement | GAP | The factor 1 + 1/(16 N_V) uses a quantity of the reduction, not of the input. The problem is not well defined on all inputs |
| Theorem T22 part 4 | proof | OK | Gamma < N_V gives L(B) < Gamma + 1/16. Also right for D2 (Gamma = N_V - kappa) |
| Theorem T22, closing sentence | statement | GAP (minor) | "Allows a stated eps" must say: eps is part of the input. For a fixed eps nothing is proved |
| Remark B.T.1 | proof | OK | Orthant example, its realisation by three points, and the role of 1 are right |
| Remark B.T.2 | stand-alone | NOT MATH (minor) | "The numerical minimiser has its kink near 8/5 / 3/2" is a measurement in Part B |
| Remark B.T.3 | overclaim | GAP (minor) | "Their teacher has as many units as points" is still there (width n - 1). Report 11 marked it fixed |
| Part 0, line 42 | claim | GAP | "Part B.T adds results whose paths hold only settings of counterexamples (distance 0)" is false: T19 and T21 have distance 3 |
| Part 0, "proved conflicts inside the trunk's own words" | overclaim | GAP | True only under the three readings of finding 1 |
| Part 0, T21 summary | claim | OK (minor) | Needs "with rational data" |
| Conjecture D17 | tag, path | GAP | Path "(none)", but the statement is for two logits and a finite mix. S-twoclass and S-finite belong on it |
| D.8 lead sentence | claim | GAP | "What is left open is two logits with more than one hidden layer" omits n_y >= 3 with depth >= 2, which T23 does not touch |
| D17 route | route | OK | Steps 1 and 3 are correct. Step 2's "images of polyhedral pieces under bilinear maps" is loose (the relu pattern constraint is not polyhedral in the parameters), acceptable for a route |
| Lit: Lim, Michalek and Qi | hypothesis stated | GAP (minor) | Hypotheses omit "two outputs, one hidden layer of width 2, ReLU with biases, inputs in R^2". "d-th secant locus" clashes with d. "Their proof is a sketch" is a judgement a reader cannot check |
| Lit: Kozlov, Tarasov, Khachiyan | hypothesis stated | OK | Rational data, bit model. Note that it also covers the LPs of T21 part 2 |
| SAE or superposition framing | framing | OK | None in B.T or D.8 |
| DECISIONS 2, T20 row | claim | GAP (minor) | "The least build of width 1" (singular). Uniqueness is not proved |
| DECISIONS 2, T23 row | claim | OK (with finding 1) | Correct summary, under the readings |
| DECISIONS 3-0, L1 to L4 | premises | GAP | "Every mix" is a premise of T23 and has no option. L4 cites T19 without its finite-mix premise |
| DECISIONS 3-0, "T20 and T22 survive L1 for small eps" | claim | GAP (major) | Not proved for T20. True for T22 only when eps is an input and may be as small as 1/(32 Z) |
| DECISIONS 3-0, "weights grow without bound" | claim | GAP (minor) | True (a bounded minimising sequence has a convergent subsequence that attains 0), but the paper does not state it |
| DECISIONS 3a, 3b | premises | GAP | Both tables speak of "the least build" as if R-least were settled. Under L1 they must be restated |
| DECISIONS 3b, option C | claim | GAP | "So any polynomially small relative error" is too strong. Only factors at most 1 + 1/(16 N_V) are ruled out |
| DECISIONS 3b, option D | claim | OK (minor) | Add "finite rational mix" |
| DECISIONS 3a, "one input number" | claim | GAP (minor) | Obsolete after T20 part 4. The N3 sentence asked for by report 11 (finding 8) is not there |
| Ledger: N-finite | paths | GAP (minor) | T14 and T15 carry N-finite. T20, T22 and T23 use a finite P and do not. Make it consistent |

## Content count (check 5)

| Kind | Items | Count |
|---|---|---|
| Structural (what the least build is, or what cannot be) | T19, T20, T21, T22, T23 | 5 |
| Bookkeeping (true of any loss or any ReLU) | (l1)-(l4), Lemma T22a | 2 |

T23 is the example of Lim, Michalek and Qi with a self-contained proof. Its new content is the transfer to KL and to
the trunk's build class.

## Findings, ranked by severity

### 1. The negative results need three readings that widen or fix T\*. None is a node. (major)

Location: B.T reading table, sentence "For the counterexamples and hardness results (T20, T22, T23) no reading narrows
the question". Part 0 ("proved conflicts inside the trunk's own words"). `DECISIONS.md` section 2 (T20, T22, T23
rows), section 3a ("are not ways out"), section 6 (paths of T20, T22, T23).

Problem. A counterexample refutes T\* only if its instance lies inside T\*. "No reading narrows the question" is the
wrong test. The question for a counterexample is whether a reading is broader than T\*, or fixes a build class that
T\* does not fix. Three readings do this.

| Reading | What it does | Why the NEG results need it |
|---|---|---|
| "a trained network" = any network | widens the teachers | T20, T22, T23 use hand-built teachers (T22a). If "trained" means "trained on images by a named method", none of them applies |
| "an image mix" = any finite list of points in R^d | widens the mixes | T23 uses six points in R^2 with negative coordinates. T22 uses points on circles |
| "of the same kind" = same number of hidden layers, width k in each | fixes the build class | A wider class can escape. On T23's six points a width-3 network reaches the target: c1 = relu(-x) - 2 relu(x) + 2 relu(y), c2 = relu(y). A depth-2 build of width 2 is not ruled out either. The NEG rule of `DECISIONS.md` section 4 says this itself |

`DECISIONS.md` section 1 says "of the same kind" and "trained" wait for Shonu's review. So T20, T22 and T23 are
conflicts with T\* as read in B.T, not "inside the trunk's own words".

The justification for the first reading is also a non sequitur. "Every network N is a global minimiser of D_P toward
its own outputs, so it is the result of training on its own labels." Being a global minimiser does not make it the
output of a training run.

Fix.
- Replace the sentence by: "For the counterexamples and hardness results (T20, T22, T23) a reading must not be
  broader than T\*. Three readings are used: every network counts as trained; every finite weighted list of points
  counts as a mix; builds have the depth of N and width k in each layer. The results refute T\* under these readings."
- Replace the justification by: "Every network N is a global minimiser of the squared logit loss on the data set
  {(x_i, y_N(x_i))}, which it fits exactly. So no property shared by all global minimisers of a training loss
  excludes N."
- Add one node (kind READ or NEG, PENDING) for "trained = any network", and put it on the paths of T20, T22, T23.
  Shonu decides it. Options A, L4 and N3 are the ways to reject it.
- Prove one lemma that removes most of the second reading: invariance under an injective affine map of the input.
  If phi(x) = M x + m with M injective (R^d0 into R^d), the units on phi(P) are exactly the units on P (w restricted to
  the image), so every value is unchanged. Then T23 holds in every d >= 2 and with all points in [0, 1]^d. T20 part 4
  is a special case.
- In Part 0 and `DECISIONS.md` section 2, write "under the readings of B.T".

### 2. "T20 and T22 survive option L1 for small eps" is not proved (major)

Location: `DECISIONS.md` section 3-0, paragraph after the options table. This sentence carries the recommendation
(L1) and the later sections 3a and 3b.

Problem.
- T20. The "strict margin" is a margin at width 1: V(u) for the four units exceeds the width-1 least value. Under L1
  the width-2 side changes too. An eps-best width-2 build need not have its units among the four. Its units only need
  to be near them. Take u_t = relu(x - t) with t slightly above 1. Then the least value of width-2 builds that contain
  u_t tends to 0 as t tends to 1, because u_t tends to relu(x - 1). So a statement about eps-best builds needs a
  separation argument that the paper does not give.
- T22. It is stated with an eps that is part of the input and may be 1/(32 Z). Z grows with the instance. For a fixed
  tolerance (say 0.01), T22 proves nothing. "Every real rule works to a tolerance anyway" (the recommendation) points
  to a fixed tolerance.

Fix. State and prove the T20 claim as a lemma, or write "expected, not proved". A proof route: (a) V(u) is continuous
in the kink t near t = 1 and t = -1 (both signs of the unit), so for eps below half the width-1 margin, the units of eps-best width-1 builds
stay at distance at least delta from the four. (b) f(u) = least D_P over width-2 builds that contain u is positive at
every such u (T20 part 1 with T19). (c) f is bounded below on that compact set of units (needs lower
semicontinuity of f, which needs a closedness argument as in T19). Then eps0 = min(margin/2, min f) works. For T22,
write: "T22 survives L1 only when eps is an input, given in binary. A fixed eps is open."

Add one line to 3a and 3b: both tables assume that least builds exist. Under L1 their options must be restated for
eps-best builds.

### 3. T22 part 4: the factor depends on the reduction, not on the input (moderate, easy)

Location: T22 part 4. `DECISIONS.md` section 3b, option C. Section 2, T22 row.

Problem.
- "D_P(B) <= (1 + 1/(16 N_V)) inf D_P, where N_V is the number of vertices in the reduction". A problem must be
  defined on every input (N, P). N_V is not a function of the input in general.
- Option C: "rules out a relative error of 1 + 1/(16 N_V), so any polynomially small relative error". Ruling out the
  factor 1 + 1/(16 N_V) rules out every tighter factor. It does not rule out looser ones such as 1 + S^{-1/2} or
  1 + 1/log S. Also "rules out" lacks "unless P = NP".

Fix.
- State part 4 with the factor 1 + 1/(16 n), n the number of points of P. In the reduction n >= N_V, so
  (1 + 1/(16 n)) Gamma <= Gamma + Gamma/(16 N_V) < Gamma + 1/16. The proof is unchanged.
- For option C, either write "a relative error of 1/(16 n) or smaller is ruled out unless P = NP", or prove the
  stronger form by padding. Add hidden units with zero output weight to the teacher. This increases S by any
  polynomial and changes no value. Then the factor 1 + S^{-c} is ruled out for every fixed c > 0.

### 4. Part 0: a false sentence about distances (moderate, easy)

Location: Part 0, lines 41 to 42.

Problem. "Part B.T adds results whose paths hold only settings of counterexamples (distance 0)." T19 and T21 have the
path S-twoclass, S-onehidden, S-finite: three SCOPE nodes, distance 3. Lemma T22a has an empty path.

Fix. "Part B.T adds three negative results at distance 0 (T20, T22, T23), Lemma T22a with an empty path, and two
positive results at distance 3 (T19, T21)."

### 5. Conjecture D17 and D.8: a hidden scope and a missing open case (moderate)

Location: D.8 lead sentence and the D17 tag. `DECISIONS.md` section 6, row "Conjecture D17".

Problem.
- D17 is stated for two logits and finite mixes. Its path is "(none)". Report 11 asked for S-finite on it; the fix put
  S-finite on T19 and T21 only.
- "What is left open is the case of two logits with more than one hidden layer." Also open: n_y >= 3 with two or
  more hidden layers. T23 is for one hidden layer, and a depth-2 build of width 2 may reach its target.
- Conversely, T23 extends at no cost to every n_y >= 3: give the teacher extra logits equal to 0, and the same proof
  applies (c1 and c2 must still lie in span(1, u_1, u_2)). This is not stated.

Fix. Put S-twoclass and S-finite on D17's path. Add a corollary to T23: "the same holds for every n_y >= 3 and every
d >= 2" (zero logits; the affine-embedding lemma of finding 1). Rewrite the D.8 lead: "T23 settles one hidden layer
for every n_y >= 3. Open: two logits at depth >= 2 (D17), and n_y >= 3 at depth >= 2."

### 6. DECISIONS 3-0: the option list misses one premise (moderate)

Location: `DECISIONS.md` section 3-0.

Problem. The section says the options "are exactly the premises of those theorems". T23 has these premises: exact
attainment (L1), unbounded weights (L2), width exactly k in the same class (L3), every network (L4), and every mix.
"Every mix" has no option. A law with a density is not covered by T23, and no result about it is proved. L4 cites T19
without its finite-mix premise.

Fix. Add L5: give up "every mix" (for example, laws with a density, or a named property of mixes). Say that nothing is
proved there. In L4, write "two logits, one hidden layer and a finite mix (T19)".

### 7. T21 part 3: the rational-minimiser step (minor)

Location: T21 proof, part 3.

Problem. "Their solution set meets the optimal face, so a rational minimiser exists." An affine space with rational
equations can meet a set without meeting it in a rational point. The step needs the set of minimisers to be a
rational polyhedron.

Fix. Use the KKT polyhedron, not the optimal face. Let (x*, lambda*) be a KKT pair with active set J. The set
K = {(x, lambda) : Q x + q = A_J^T lambda, A_J x = b_J, A_i x <= b_i for i not in J, lambda >= 0} is defined by rational
data and is non-empty. Every x in the projection of K is a minimiser, because KKT is sufficient for a convex program.
A non-empty rational polyhedron has a rational point. Also cite LP in polynomial time for part 2 (KTK covers it).

### 8. T23: small omissions (minor)

Location: T23 statement and proof. `DECISIONS.md` section 3-0, plain words.

Problem and fix.
- The case a = 0 (constant unit on the line): add "if a = 0 then D u = 0 and beta = 0".
- The independence of 1, c1, c2: add one line (c2 = e_6; c1 is not constant on the first five points).
- `DECISIONS.md` says that the weights grow without bound. Add to T23: "every minimising sequence has unbounded
  weights", with the proof: a bounded one has a convergent subsequence, D_P is continuous in the weights, so the limit
  attains 0. This line is also the proof that option L2 changes the answer.
- Lean. The finite core is: for all reals a, b0, alpha, beta, if max(a x + b0, 0) = alpha + beta c1(x) for
  x = -2, ..., 2, then beta = 0. This is a finite real statement. A case split on the five `max` terms with `linarith`
  should prove it. The paper's rule 2 asks for the finite version in Lean.
- Plain words in 3-0: "better and better, with no limit" can be read as "with no limiting network". Write "as close as
  we like".

### 9. T20 KL certificate: the paper still cannot be re-checked alone (minor)

Location: T20 proof, part 2. `model/t20_nesting.py`.

Problem.
- The paper does not say that s* = s_0 + A z lies in (0, 1)^7. The Fenchel inequality and H need it. The script
  asserts it.
- The interval library and its precision (mpmath `iv`, 60 digits) are not named. Report 11 asked for this.
- s_0 is not printed. Any s_0 gives a valid bound, but the printed bound 416449/10^7 has a margin of about 1.4e-8 over
  the computed enclosure and depends on s_0.
- "That z is irrational (q is)" is not proved and not needed. Write "z is not known to be rational".
- The script recomputes the competitor with BFGS and asserts that the rounded result equals 969140/968833 and
  219763/439837. This depends on the optimiser and the machine. The certificate should evaluate the printed fractions
  directly.

Fix. Add the sentence "s* lies in (0, 1)^7 (checked)". Name the library and precision. Print s_0 for the two units
(or store it in the JSON and cite that file as the certificate data). Change the script to take the printed alpha and
gamma as inputs.

### 10. Wording and stand-alone items (minor)

| Location | Problem | Fix |
|---|---|---|
| T22, closing sentence | "allows a stated eps": for a fixed eps nothing is proved | "with eps part of the input, given in binary" |
| Remark B.T.2, last sentence | "the numerical minimiser has its kink near 8/5 / 3/2" is a measurement | Delete, or move to the check table in the Appendix |
| Remark B.T.3 | "as many units as points" is still there | "a teacher that fits arbitrary labels on its points (width n - 1)" |
| Reading "does not train" | after this reading the phrase adds nothing to "an algorithm with a proved guarantee" | say this in the row |
| T21 part 3, Part 0 | "grows like n^{d k}" | "at most of order 2^k n^{d k}" |
| `DECISIONS.md` section 2, T20 row | "the least build of width 1" (singular) | "no least build of width 1 shares a unit with a least build of width 2" |
| `DECISIONS.md` section 3a | "one input number" is obsolete (T20 part 4); the N3 sentence of report 11 is missing | delete "one input number"; add "a property under N3 must exclude T20's teacher; T20 is proved only for two logits and one hidden layer" |
| `DECISIONS.md` section 3b, option D | no mention of the finite rational mix | add it |
| Ledger | N-finite is on T14 and T15, not on T20, T22, T23, which also use a finite P | either add it to all three, or drop it from T14 and T15 with the B.T reading as the reason |
| Lit row, Lim, Michalek and Qi | hypotheses incomplete; "d-th secant locus" clashes with the input dimension d; "their proof is a sketch" cannot be checked by a reader | list "two outputs, one hidden layer of width 2, ReLU with biases, inputs in R^2"; write "k-th"; replace the judgement by "T23 gives a self-contained proof" |

## The three most important fixes, ranked

1. **State the readings that the negative results need (finding 1).** Three readings carry T20, T22 and T23 to the
   trunk: every network counts as trained; every finite point list counts as a mix; builds have the depth of N and
   width k. Rewrite the reading-table sentence, Part 0 and `DECISIONS.md` section 2 to say "under these readings". Add
   a PENDING node for "trained = any network" on the three paths. Prove the affine-embedding lemma. It puts T23's
   points in [0, 1]^d for every d >= 2.
2. **Prove or withdraw "T20 and T22 survive option L1" (finding 2).** For T20, prove a separation lemma: for small
   eps, the units of eps-best width-1 builds stay away from the four units, and the width-2 value with such a unit is
   bounded below. For T22, say that it survives only with eps as an input. Mark 3a and 3b as conditional on 3-0.
3. **Make T22 part 4 a well-posed problem and fix option C (finding 3).** Use the factor 1 + 1/(16 n), n the number
   of points of P. Then either weaken option C to "a relative error of 1/(16 n) or smaller", or add the padding
   argument for 1 + S^{-c}. Also fix the false distance sentence in Part 0 (finding 4) and put S-twoclass and
   S-finite on D17 (finding 5). Both are one-line fixes.
