# Referee report: Part B.T (the trunk itself), Conjecture D17, new literature rows

Date: 2026-10-25. Scope: `COARSE-THEORY.md` Part 0 (new paragraph), A.4 (four new rows), B.T (all items), D.8.
Lean file: `lean/coarse/Coarse/Trunk.lean`. Ledger: `DECISIONS.md` sections 2, 3, 4, 6.

## Build

| Check | Result |
|---|---|
| `lake build` in `lean/coarse` | passes (9055 jobs). Only style warnings (long lines) in `Trunk.lean` |
| `sorry`, `admit`, `axiom` in `Coarse/*.lean` | none |
| `Coarse.lean` imports `Coarse.Trunk` | yes |

## Table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| Reading table of B.T | stand-alone, paths | GAP | "None of them narrows the question" is false for positive results (finite P). The words "trained" and "does not train" get no reading, yet T20, T22 and T21 depend on them |
| Definition D15 | stand-alone | OK | Clear. "Equal on P up to scale" is an equivalence relation on non-zero data vectors, which part 3 of T20 needs |
| Definition D15 vs T19 | stand-alone | GAP (minor) | D15 defines "build of width k" only for a teacher with one hidden layer, but T19 takes "any network" N. Harmless (N enters only through Delta_N(x_i)), but define builds of width k for any teacher |
| Definition D16 | stand-alone | OK | Strong implies weak is immediate. "The trunk asks at least for weak nesting" is a reading, stated as one |
| (l1)-(l4) | proof | OK | Checked: the formula for l, the derivative, convexity, the limits, and log cosh at t = 0 |
| Theorem T19 | proof | OK | Sign step, boundary of Theta_A, finite union of finitely generated cones, coercivity: all correct |
| Theorem T19 | tag and path | GAP | The proof uses a finite P. The trunk's mix need not be finite. No node for this on the path |
| Remark B.T.1 | proof | OK | The orthant example is correct. It is even realised by relu data vectors (see finding 9) |
| Theorem T20 part 1 | proof | OK (statement GAP) | The second-difference argument is complete. The "if" half of the statement is false as written: output weights matter |
| Theorem T20 part 2, D2 | proof, Lean | OK | 85/26, 493/110 and 16/5 are checked. Lean proves the lower bounds for every alpha and gamma |
| Theorem T20 part 2, KL | proof | GAP | The duality inequality is correct. The paper does not print the certified numbers. It also says "rational s", but the script's s is irrational |
| Theorem T20 part 3 | proof | OK | Follows from parts 1 and 2. T19 gives existence, so the conflict is not vacuous |
| Theorem T20 | path | OK | A counterexample for one admissible (N, P) refutes a "for every N, P" claim. N-dim1 can be removed (finding 8) |
| Remark B.T.2 | content | GAP (minor) | It drops the strict-gap premise of T6. "Because it must fit both sides" is heuristic |
| Lemma T22a | proof, Lean | OK | The bound on tau and the broken line are correct. Lean `broken_line_interp` matches. The sorting step and the tau step are by hand, and the paper says so |
| Theorem T21 | proof | GAP | Exactness is shown only for D2_P, and only with rational data (not stated). The cost "polynomial in n" does not follow from "list the active sets" |
| Theorem T21 | tag and path | GAP | Finite P and rational N are premises. Neither is on the path |
| Theorem T22 | proof | OK | Points, distinctness, constants, midpoint step, both directions and the ETH step are all correct (details below) |
| Theorem T22 | statement | GAP (minor) | The last sentence (a trunk rule solves TRUNK-1) is vacuous after T20, and "eps = 0" meets a representation problem |
| Theorem T22 | path | OK | Restricting the inputs of a problem preserves hardness. The build class for these teachers equals the trunk's build class under the "same kind" reading |
| Remark B.T.3 | overclaim | GAP | It repeats T21's "exact" for D_P. "As many units as points" is loose (width n - 1) |
| Part 0, new paragraph | overclaim | GAP | "An exact rule that does not train finds it" (see T21). Line 41 ("only Proposition T1' has an empty path") is now false |
| Conjecture D17 | route | GAP | For n_y >= 3, KL and D2 are coercive only modulo per-point constants. Closedness of the set of Y is not enough. A possibly decisive reference is missing |
| Lit: Froese, Hertrich, Niedermeier 2022 | hypothesis stated | OK | The hypothesis (l_p loss, output sign, bias) is stated. T22 redoes the bounds itself |
| Lit: Cygan et al. 2015 | hypothesis stated | OK (minor) | Quoted second-hand, and the row says so |
| Lit: Arora, Basu, Mianjy, Mukherjee 2018 | hypothesis stated | GAP (minor) | The row omits their model of computation (exact convex programs over the reals). That model is the gap in T21 |
| Lit: Petersen, Raslan, Voigtlaender 2021 | hypothesis stated | OK | Read from the abstract only, and the row says so. Used as context only |
| SAE or superposition framing in B.T | framing | OK | None |
| DECISIONS section 2, T22 row | claim | GAP (minor) | "Needs time S^{Omega(d)}" is stronger than the theorem, which rules out rho(d) S^{o(d)} |
| DECISIONS section 3b, options C and D | claim | GAP | C: the additive eps hides a relative statement that is more useful (finding 6). D: "exact rule of T21" fails for KL, and "deeper forms" do not exist yet |

## Content count (check 5)

| Kind | Items | Count |
|---|---|---|
| Structural (what the least build is, or what cannot be) | T19 (existence), T20 (no nesting), T21 (the problem is a finite union of convex programs), T22 (hardness) | 4 |
| Bookkeeping (true of any loss or any ReLU) | (l1)-(l4), Lemma T22a (standard interpolation by a broken line) | 2 |

This section is not bookkeeping. Novelty is uneven. T19 and the region reduction of T21 are close to folklore: the
algorithm of Arora et al. 2018 already uses them. T22 adapts Froese et al. 2022 with care. T20 is the new result.

## Findings, ranked by severity

### 1. The reading table hides restrictions (major)

Location: B.T, "Reading of the trunk's words", in particular the sentence "None of them narrows the question". Also
the paths of T19, T21 and D17 in `DECISIONS.md` section 6.

Problem.
- A finite P narrows the question for every positive result. T19, T21 and D17 are statements about finite mixes. The
  trunk's mix may be a law with a density, for example natural images. For such laws, Petersen, Raslan and Voigtlaender
  (the paper's own literature row) say that sets of network functions are not closed. So T19 may fail there. This is
  a restriction with no node. T14 and T15 carry `N-finite`. T19, T21 and D17 do not carry it, or an `S-finite` node.
- "A trained network" has no reading. T20 and T22 transfer to the trunk only if "trained" excludes nothing. Their
  teachers are built by hand. This is exactly the escape that `DECISIONS.md` option A offers. So the reading must be
  explicit.
- "Does not train" has no reading. T21 calls its rule one that "does not train". But T21's rule is exact global
  minimisation of the training objective over all activation patterns. That is training by any ordinary meaning.
- The bit model ("fast" = polynomial in bit length) requires rational N and P. That is harmless for hardness. For T21
  it is an unstated premise.

Fix.
- Replace "None of them narrows the question" by: "For negative results (T20, T22) none of them narrows the question.
  For positive results (T19, T21, D17) the finite mix is a restriction."
- Add a node (`S-finite`, SCOPE). Put it on the paths of T19, T21 and D17.
- Add a row for "trained network". One choice that keeps T20 and T22 at the trunk is "any network". It has a one-line
  justification: every network is a global minimiser of the squared logit-regression loss on its own outputs over P.
  So no property of optima can exclude it. State this as a remark with that proof.
- Either define "does not train" (for example, "an algorithm whose output is a fixed function of (N, P), with no
  random initialisation"), or delete the phrase from T21, Part 0 and Remark B.T.3.

### 2. T21: "exact" holds only for D2_P, and the cost proof does not give the stated cost (major)

Location: Theorem T21, its proof, Part 0 line 49, Remark B.T.3, `DECISIONS.md` section 3b option D.

Problem.
- (a) For D_P (KL), each piece is a convex program with a transcendental objective. Its minimiser is irrational in
  general. No finite algorithm in the bit model returns it exactly. The theorem gives exactness ("a rational
  minimiser exists") only for D2_P. Yet the sentence "So an exact rule that does not train exists" is read in Part 0
  and in B.T.3 as a statement about D_P. The helper script itself says that its pieces are "accurate to the solver
  tolerance, not exact".
- (b) For D2_P, the proof finds the minimiser "by listing the finitely many active sets". There are up to 2^{n k}
  active sets. So the proof gives a cost exponential in n. That contradicts "polynomial in n for fixed d and k".
- (c) The proof also needs to list the regions of the arrangement in time polynomial in n for fixed d. It gives only
  the count of regions (Cover, Zaslavsky), not a way to list them.
- (d) Rational data (rational points x_i, rational weights pi_i, rational teacher gaps Delta_N(x_i)) is used and not
  stated.

Fix.
- State T21 in two parts.
  - D2_P with rational data: exact, in time poly(n, bit length) times (2R)^k. Cite a polynomial-time algorithm for
    convex quadratic programs with rational data (Kozlov, Tarasov and Khachiyan 1979). Cite a polynomial-time
    enumeration of the cells of an arrangement for fixed dimension (Edelsbrunner, O'Rourke and Seidel 1986). Add both
    to the literature table.
  - D_P: within any eps > 0, in time polynomial in log(1/eps). This needs an a-priori bound on the minimiser. It
    comes from coercivity with explicit constants: by (l3) and (l2) the sublevel set F(v) <= F(0) is a box of
    computable size. Then use an ellipsoid or interior-point method.
- Alternatively, say once that T21 is in the real-number model, as Arora et al. are. Then drop "exact rule"
  everywhere the bit model is meant.

### 3. T20 part 2 (KL): the computer-assisted step cannot be checked from the paper and is misdescribed (major)

Location: T20 proof part (2), the paragraph "The check `model/t20_nesting.py` gives rational s ...".

Problem.
- The duality bound is correct. The bound is log(1 + e^f) >= s f + H(s): the convex conjugate, with equality at
  s = sigma(f). With the two linear constraints it gives V(u) >= (1/7) sum (H(s_i) - H(q_i)).
- But the constraints contain q_i = sigma(T(x_i)), which are irrational. The script does not use a rational s. It
  uses s* = s0 + A z*. Here s0 is rational and z* solves the normal equations in interval arithmetic, so s* is known
  only as an enclosure. The certificate is still valid: the constraints hold exactly at s*, and H is evaluated on
  intervals. But "rational s" is false.
- The paper prints none of these numbers: the competitor's (alpha, gamma), the four certified lower bounds, the upper
  bound, or the margins. So the strict inequality that the KL half of T20 rests on is asserted, not shown. A reader
  of the paper alone cannot check it.

Fix.
- Describe s correctly, as "s = s0 + A z*, with s0 rational and z* enclosed by interval arithmetic".
- Print in the proof the competitor's (alpha, gamma) and the four intervals for V(u) and for D_P(B_c'), so the
  inequality can be read and re-checked. Constants of a proof are mathematics, not Part C measurements.
- Name the interval library and its rounding guarantee (mpmath `iv`).

### 4. Conjecture D17: the route proves the wrong closedness (moderate)

Location: D.8, route step (1). Also Remark B.T.1.

Problem.
- For n_y >= 3, D_P and D2_P depend on Y only modulo matrices g 1^T (one free constant per point). They are coercive
  only on that quotient.
- A minimiser exists if the image of the set of Y in the quotient (the row-centred logit matrices) is closed.
  Closedness of the set of Y does not imply this. The projection of a closed set that is not convex need not be
  closed.
- For n_y = 2, T19 avoids the problem, because the gap vector already is the quotient.
- The margin argument: the limit h 1[L > 0] + relu(h) 1[L = 0] equals relu(h + M L) - M relu(L). This is correct, but
  the bound on M depends on max |h(x_i)| as well as on the smallest non-zero |L(x_i)|.
- Statement: "every network N, every finite mix" carries the finite-mix restriction of finding 1.
- Missing literature. The referee recalls Lim, Michalek and Qi, "Best k-layer neural network approximations"
  (Constructive Approximation, 2022, arXiv 1907.01507). It studies exactly whether the finite-sample least-squares
  infimum is reached for networks with one hidden layer and vector outputs, and states results for ReLU. This is from
  memory and not checked here. It must be read before D17 is called open. It may settle the D2 case of D17 for some
  n_y, possibly negatively. Its relation to T19 (scalar output) must also be checked.

Fix.
- Restate route step (1) as the closedness of {P_perp-row-centred Y}. Equivalently, the closedness of the set of
  logit-difference matrices (Y e_c - Y e_1), c = 2..n_y.
- Add the max |h| dependence to the bound on M.
- Add the finite-mix node.
- Read Lim, Michalek and Qi. Add a row to the table with its hypothesis, then restate D17 (or retract it, in the D2
  case) accordingly.

### 5. T20 part 1: the statement's quantifier (moderate, easy)

Location: T20 statement, item 1.

Problem. "A build of width 2 has value 0 if and only if its two hidden units are equal on P up to scale to ...". The
"if" direction is false. A build with those units and wrong output weights has a positive value. The proof shows the
correct statement.

Fix. Write: "Every build of width 2 with value 0 has its units equal on P up to scale to {relu(x - 1), relu(-x - 1)}
or to {relu(x + 1), relu(1 - x)}. For each of these pairs some build of value 0 has those units (output weights 1,
gamma = 0 or gamma = -2)."

### 6. T22: the last sentence, and a stronger form for option C (moderate)

Location: T22, the sentence after item 3. `DECISIONS.md` section 2 (T22 row) and section 3b (option C).

Problem.
- "A rule that meets the trunk for every network and every mix also solves TRUNK-1 (take k = 1 and eps = 0)." By T20
  no such rule exists at all, whatever its speed, so the sentence is vacuous.
- An exact KL minimiser is irrational in general, so "eps = 0" with a rational output is not available.
- The meaningful claim concerns the sub-requirement "least at k = 1".
- Section 2's "needs time S^{Omega(d)}" is stronger than "no rho(d) S^{o(d)} algorithm".
- Option C says T22 "rules out every error below 1/(32 Z)". This hides the scale. On the hard instances inf D_P is
  Gamma/Z itself. The proof gives a relative gap: Gamma/Z if a clique exists, at least (Gamma + 1/16)/Z if not.

Fix.
- Replace the sentence by: "A rule that, for every network and mix, returns a rational build of width 1 within eps of
  the least, for eps given in binary, solves TRUNK-1."
- In section 2, write "has no rho(d) S^{o(d)} algorithm under ETH".
- Add a corollary with a one-line proof from the same constants. Assume N_V > kappa, so that Gamma > 0. Then
  returning a build within the factor c = 1 + 1/(16 N_V) of inf D_P is NP-hard. Proof: in the yes-case the returned
  build has L(B) <= c Gamma = Gamma + Gamma/(16 N_V) < Gamma + 1/16, strictly, because Gamma < N_V. The extraction
  of C needs that strict inequality. (A factor of exactly 1 + 1/(16 Gamma) does not work. It allows
  L(B) = Gamma + 1/16. Use any factor strictly below it, for example 1 + 1/(32 Gamma).) Since N_V < S, the factor is
  1 + 1/poly(S). Option C then reads as follows. A relative error of 1/poly is ruled out. A constant relative error
  is open.

### 7. Part 0, B.T.3, B.T.2: overclaims and one false sentence (moderate)

Location and problem.
- Part 0, line 41: "Among the results of Parts B and D only Proposition T1' has an empty path". This is now false.
  Lemma T22a and Conjecture D17 have the path "(none)" in section 6, and T20 and T22 are at distance 0.
- Part 0, line 49, and Remark B.T.3: "an exact rule that does not train finds it". See findings 1 and 2.
- Remark B.T.3: "their teacher has as many units as points". The width is n - 1. More to the point, the teacher
  realises arbitrary labels. That is the feature a named property must exclude, not the width as such.
- Remark B.T.2: "They nest because the optimum at width k + 1 adds one eigenvector". This holds at widths with a
  strict eigenvalue gap (T6, D.1). "Puts its single kink elsewhere, because it must fit both sides" is an explanation,
  not a proved fact. T20 proves only that the unit is not one of the four.

Fix. Correct line 41. Add "for D2_P" (or "within eps") to the T21 mentions. Add the strict-gap premise to B.T.2 and
mark the "because" as heuristic.

### 8. Paths of the negative results: one node can go, two cannot (minor, but it answers question 6)

Location: `DECISIONS.md` nodes `N-dim1`, `N-ny2`, `N-onehidden`, section 3a.

Assessment. The carry-over logic is right. A counterexample for one admissible (N, P) refutes a claim "for every N
and P". Hardness on a sub-family of inputs is hardness of the general problem. Two conditions are needed:
- (i) the trunk's build class for these teachers equals the D15 builds of the same depth. This holds under the
  reading "same kind = same number of hidden layers".
- (ii) "trained" excludes nothing (finding 1).

What the NEG nodes record is different: what a later property (option A, N3) must exclude.
- `N-dim1` is removable by a proved lemma. Put the seven points on a line in R^D (other coordinates 0, or a fixed
  offset). Every unit's data vector is then relu(w_1 x_1 + b'), so the set of data vectors and every value in T20 is
  unchanged. T20 holds in every input dimension.
- `N-ny2` and `N-onehidden` are not removable this way. With n_y >= 3, or with deeper builds, the build set is larger
  and the per-point loss changes. So T20 for 1000 classes or for a deep teacher is not proved.
- Section 3a says these premises "are not ways out". That is right for the trunk as worded, and wrong for any
  restricted trunk that option N3 might adopt.

Fix. Add the line-embedding lemma and delete `N-dim1` from T20's path. In 3a, add one sentence. A property under N3
must exclude T20's teacher, and T20 is proved only for two logits and one hidden layer.

### 9. Remark B.T.1: the example is stronger than stated (minor)

Location: Remark B.T.1.

Problem. The orthant example is not only "a general finite union of closed cones". Take n = 3 affinely independent
points in R^2. Then X~ is invertible, and the relu data vectors are exactly the non-negative orthant. So the
non-closed limit is realised by relu networks without an output bias. The sentence "Sets of this kind need not be
closed for a general finite union of closed cones" is also hard to parse.

Fix. State the realisation. Then say that with the bias vector 1, span(u_1, u_2, 1) is generically all of R^3, so
this example disappears. That is why D17 is open and not false.

### 10. Lean lines (minor)

| Item | Note |
|---|---|
| `relu_second_diff` | Matches the h_t identity |
| `t20_d2_gap_a`, `t20_d2_gap_b` | Prove the lower bounds for every alpha and gamma. This is the needed direction. Attainment is not needed and is not claimed in Lean |
| `t20_d2_competitor` | Proves 16/5 < 85/26 < 493/110. The paper's Lean line attributes this to `t20_d2_gap_a` and `t20_d2_gap_b`; name `t20_d2_competitor` there |
| `relu_affine_monotone`, `mono_mid_ge_min` | Match the midpoint step |
| `broken_line_interp` | Assumes `StrictMono s` on all of ℕ. That is a stronger hypothesis than the lemma's finite sorted list, and harmless (extend the list). The tau step and the sorting are by hand, and the paper says so |

### 11. Literature rows (minor)

- The Arora et al. row should state their model of computation (exact solution of convex programs). It is the source
  of the gap in finding 2.
- The Cygan et al. row is second-hand. Cite also the primary source of the ETH bound for (Multicolored) Clique, for
  example Chen et al. 2006, or read Cygan et al. Chapter 14.
- No row carries an SAE or superposition framing.

## Answers to the specific questions

| Question | Answer |
|---|---|
| 1. T19 | Correct. Boundary: at X~_i theta = 0 both sides vanish, so Theta_A with non-strict inequalities is right. Sign step: abs(alpha) theta stays in Theta_A, and alpha = 0 gives theta = 0, which lies in every cone. Images of polyhedral cones are finitely generated, hence closed. Coercivity: each term is >= 0 and tends to infinity, the sum is finite, and every pi_i > 0. The general l' case is fine: continuous and infinite at both ends implies bounded below |
| 2. T20 part 1 | Correct. supp h_t in I has at most two adjacent integers. -1 and 1 force separate supports. 0 and -2 are excluded as argued, so t_1 = -1 and beta_1 = 1, and symmetrically for t_2. The four sign cases give gamma = 0, gamma = -2, or no solution. Only the statement's "if" is wrong (finding 5) |
| 2. T20 part 2 | The Fenchel step is correct. The computer-assisted step is valid as a certificate (constraints exact at s*, interval H), but misdescribed ("rational s") and unprinted (finding 3) |
| 2. T20 part 3 | Follows from parts 1 and 2. Transitivity of "equal up to scale" on non-zero vectors is needed and holds. T19 makes the conflict non-vacuous |
| 3. Lemma T22a | Correct. At most C(n, 2)(d - 1) roots, fewer than the (d - 1) n^2 + 1 candidates. The broken-line slopes telescope to mu_j |
| 4. T21 regions | Correct. Only full-dimensional regions are needed: their closures cover R^{d+1}, since no row (x_i, 1) is zero, and Theta_A is that closure. The bound 2 sum_{i <= d} C(n - 1, i) is right for a central arrangement in R^{d+1}. The gaps are elsewhere (finding 2) |
| 5. T22 | Correct in every step checked. Distinctness: norm 1 / norm in (0, 1) with one block / two blocks, and a chord is fixed by a non-central midpoint. Constants: l_0 >= 1/8 (e^2 >= 7, e <= 3). Also l_0 <= log 2 and delta (N_V - kappa + 1) = 1/16. Next, log cosh(delta / 2) >= delta^2 / 9 >= delta^2 / 12, and M delta^2 / 12 >= N_V + 1 > Gamma + 1/16. Finally eps = 1/(32 Z) gives L <= Gamma + 1/32. Free sign, scale and bias: handled by the monotone phi. Weights in place of copies: fine, since every pi_i > 0. Teacher: exact labels by T22a. ETH: S <= |V|^c with c independent of kappa, the reduction's own cost |V|^{O(1)} is |V|^{o(kappa)}, so rho(2 kappa) S^{o(2 kappa)} gives an f(kappa) |V|^{o(kappa)} algorithm. The paper should mention the reduction's own cost. Part 3 (D2): constants check (l(0; 2) = 1, slope <= 1 on [0, delta], f^2 / 4 >= delta^2 / 12) |
| 6. Tags and paths | T20 and T22 are existential or hardness statements whose instance is the setting. "(under: none)" is correct for them. The carry-over is valid under conditions (i) and (ii) of finding 8. Hidden premises are on the positive side: finite P and rational data for T19, T21 and D17 (finding 1) |
| 7. B.T.3 and Part 0 | Overclaim on T21's exactness for KL. Line 41 is false. "Does not train" is undefined (findings 1, 2, 7) |

## The three most important fixes, ranked

1. **Make the readings honest (finding 1).** Add rows for "trained" and "does not train". Add the one-line remark that
   every network is a global minimiser of logit regression on its own outputs. Add a finite-mix node on the paths of
   T19, T21 and D17. Delete "None of them narrows the question".
2. **Restate T21 (finding 2).** Make it exact for D2_P with rational data. Cite polynomial-time convex QP and
   arrangement enumeration. Give an eps-version for D_P with an explicit a-priori box. Then correct Part 0, Remark
   B.T.3 and option D of `DECISIONS.md` section 3b.
3. **Put the T20 KL certificate into the paper (finding 3).** Print the competitor's weights and the four certified
   intervals against the upper bound. Describe s as the interval solution it is. Also fix the "if" of T20 part 1
   (finding 5) and remove `N-dim1` by the line-embedding lemma (finding 8).
