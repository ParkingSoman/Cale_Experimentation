# Referee report: COARSE-THEORY.md, Parts A and B (2026-10-24)

Scope: Parts A and B, the Lean files in `lean/coarse/Coarse/`, and the three scripts in `model/`. Parts C and D are
not refereed. `NEURON-MATH.md`, `PAPER.md` and `CURVES-LOG.md` were not read. Where an argument needs them, that is
a finding.

Lean: `lake build` exits 0 (9051 jobs). The visible tail of the log shows only style (line-length) warnings. `grep`
on the sources finds no `sorry`, `admit`, `axiom` or `native_decide` in `Coarse/*.lean`.

## 1. Item table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| A.0 Setting | stand-alone | GAP | One input field A(x) for layer L. In an Inception block each branch has its own input (the curve units read 3b's 5x5-reduce output). Say which A. |
| F1 | proof | OK | Definitions. LRN (below 3b) is not listed, but nothing above 3b needs it. |
| F2 | proof / Lean | OK, with a GAP in the gloss | `relu_add_iff` states F2 exactly. The gloss "a sum of ReLU units is a single ReLU unit only where..." does not follow: F2 is only about relu(z1 + z2), not about relu(z') for some other z'. That is T1'. |
| F3 | proof | OK, scope GAP | True for shifts in sigma Z^2, away from borders. The closing sentence "translation is a symmetry of the architecture" is used later (B.1) for non-lattice shifts, which F3 does not give. |
| F4 | proof | GAP (false as stated) | Counterexample: the output set is a thin circular cone about (1,1,1) inside the positive orthant. A rotation about (1,1,1) by any angle preserves it and is not a permutation. The step "Q maps outputs to outputs, so Q has non-negative entries" needs a premise such as "each unit fires alone on some input" (each e_i is in the closed cone of outputs). The proof is also outsourced to `NEURON-MATH.md`. |
| F4 Lean `nonneg_orth_col_single` | Lean | GAP | Assumes entrywise non-negative Q. That is exactly the missing step. The Lean result is the core only. |
| F5 | proof | OK | Standard. KL direction matches D7. |
| F6 / `reynolds_le` | proof / Lean | OK | Lean is more general (no linearity of the action). F6 is not used anywhere in Part B. |
| F7 | source | OK | Source given. Not used in Part B. |
| F8 | proof | OK, minor | One line (gradient of the ReLU unit). Write it in place of the pointer to `PAPER.md` Thm. 2. Not used in Part B. |
| F9 | proof | GAP / not stand-alone | Proof is only `PAPER.md` Thm. 24 and 34.3. "Same mean gradient" is |theta| linear constraints, not one. Two such laws with different second moments exist only if the second-moment integrand is not in the span of 1 and the gradient components on the support of P. That is a genericity condition, not a fact for every network. |
| F10 | source | GAP (self-contradiction) | "the fourth reads a second 3 x 3 max-pool of it" contradicts "for each branch, the readers' input at pooled position p is W_b times the pooled 3b outputs at p". The fourth branch has its own map M_b. Source (Szegedy et al.) is not in A.4. |
| D1 | definition | OK | |
| D2 | definition | GAP | On the square image domain of A.0, a rotation by an angle not a multiple of 90 degrees does not map X to X (corners leave, blanks enter). So D2 is not a group action on X. The comment covers interpolation, not the domain. X must be redefined (images on the plane, or on a disc) and the network extended to it. |
| D3 | definition | GAP | The position index is dropped. Units exist at every position p. D3 says "rotation about the unit's position" but f_0 has no position. State f_{i,p} = R_{theta_i, p} . f_{0,p} and the translation covariance of f_{0,p}. The weights m_i and the measure mu_F are never used. |
| D4 | definition | GAP | Same dropped p (max over i only, or also over p?). D4 allows any g_i. T1 needs g_i = gamma^i (a complete, equally spaced orbit). Part C counts "spread from equal spacing" inside eps_clean, but D4 does not contain it. |
| D5 | definition | OK | "circuit" is defined and never used. |
| D6 | definition | GAP | The WHAT t_a = sum lambda h_i is at layer L, before the pool. B.0 and B.1 use pooled signals t(x, p). Max-pool of a sum is not the sum of max-pools, so "t = Lambda h" with h pooled is not a D6 WHAT unless the members share arg-max patterns. |
| D7 | definition | OK, minor | "a minimiser" needs "if one exists". D7's e_x is the change of reader OUTPUTS; B.0's e_x is the change of reader INPUTS (with D_x). Make one notation. |
| H-clean | hypothesis | NOT MATH (one sentence) | "A stability statement then bounds the effect of eps > 0" is written as a result. It is Conjecture T4 part 2. |
| H-sym | hypothesis | GAP (unsatisfiable on X) | No law on 224 x 224 images is invariant under SE(2), or under Z_N rotation for N not dividing 4. Needs the redefined X of D2. The gravity comment is an empirical claim with no source. |
| H-contour | hypothesis | GAP | kappa_0 ~ nu "with a density". T3 and T3.1 use kappa_0 Gaussian, which H-contour does not give. "Smooth contours" is wrong for Brownian curvature (curves are C^2, not smooth). "Rendered as edges on an independent background" is not defined. |
| H-read | hypothesis | GAP | (a) R_g must act on positions as well as reader channels (G_x couples positions). (b) For the frozen-bench metric D_x J^T Pi J D_x with 0/1 gates, equivariance forces R_g to be a permutation of readers, under the same fires-alone premise as F4 (each reader fires alone on some input; otherwise a rotation inside a fixed gate block is still equivariant). Say "permutation". The instance script already uses one. |
| H-stat | hypothesis | OK | |
| A.4 table | literature | GAP (minor) | No SAE, dictionary or superposition framing enters any definition or proof. Missing rows for borrowed results that ARE used: Eckart and Young, Gray 2006 (circulants), Davis and Kahan, Perron and Frobenius, Fejer kernel, Szegedy et al. 2015 (F10). The K-FAC row says "tools only", but the constant Kronecker metric used by T0 part 3 and T1 is K-FAC's own assumption (independence of activity and back-propagated curvature). That assumption is not stated. |
| B.0 Setting | stand-alone | GAP | "Pointwise readers (Fact F10 shape)" inherits the F10 contradiction. Use a branch-specific M_b. |
| Lemma B0.1 | proof, tag | OK | Uses only its stated premises. Local, as it says. |
| T0 part 1 | proof | GAP (index slip) | Block (a, b) of S from (B t_p)^T G_pp' (B t_p') is t_a(p) t_b(p') G_pp', so S = E sum t(x,p) t(x,p')^T (x) G_x(p,p'). The paper has t(x,p') t(x,p)^T. They agree only when every block G_x(p,p') is symmetric (true for M (x) K, false for the frozen bench in general). R is correct. The constant c is not defined. Finite second moments are not assumed. |
| T0 part 1 Lean `quad_moment_suff` | Lean | GAP (weaker) | Finite-support laws, a generic scalar read-out, an abstract moment. The reduction "write e^T G e as a sum of squares of read-outs of (vec B, 1)" is not in Lean, and S is not stated, so Lean cannot catch the index slip. |
| T0 part 2 | proof | OK | |
| T0 part 3 | proof | GAP (minor) | The formula inverts Tbar = E sum K t t^T; invertibility is not assumed. |
| T0 part 3 Lean `write_in_span` | Lean | GAP (weaker) | The normal equations are a hypothesis, not derived. The moments are unweighted sample moments, not the K-weighted Hbar, Tbar. Reduction (factor K = L L^T into samples) not stated. |
| T0 part 4 | proof | GAP | (a) "G_x = D_x (x) I" makes the gate position-independent; real gates depend on p. (b) Part 3 needs a CONSTANT metric. A per-input gate is a sample weight, so the proof is weighted least squares per reader, not "apply (3)". (c) "the write leaves the joint span" is a "need not lie in" claim. It needs an example, not a proof of the formula. |
| T0 part 4 Lean `gated_write_in_span` | Lean | GAP (states the other direction) | One gate Dg shared by all r readers. The conclusion is B = W * (one matrix), that is, IN the span. The paper claims per-reader matrices and leaving the span. Applying the lemma with r = 1 per reader gives the formula, but not the "leaves" claim. |
| Corollary T0.1 | proof, tag | GAP (quantifier) | "Different (S, R)" does not give a different write: (2S, 2R) has the same minimiser S^-1 R. Needs S_1^-1 R_1 != S_2^-1 R_2. Rests on F9, which is outsourced and needs a genericity condition. Tag H-stat is right. |
| B.1 Setting | stand-alone | GAP (hidden hypotheses) | Quietly assumes: (1) positions on a continuum; (2) translation covariance at non-lattice shifts (F3 gives only sigma Z^2) to move "rotation about the unit's position" (D3) to "rotation about a fixed point"; (3) g_i = gamma^i, a complete equally spaced orbit; (4) no pool between L and the readers, or a pool that commutes with gamma (the 3 x 3 stride-2 square max-pool commutes only with 90-degree turns); (5) a constant Kronecker metric G = M (x) K for all inputs (not the frozen bench of D6, whose metric is per input); (6) K rotation-invariant, with support bounded enough that C is finite (a stationary law on the whole plane makes the double integral diverge otherwise); (7) X redefined so that H-sym can hold. The "So" deriving h_{i+1}(gamma . x, gamma p) = h_i(x, p) is an extra assumption, not a derivation. |
| T1 tag | tags | GAP | Tagged (H-clean, H-sym, H-read). The proof also consumes items (1) to (7) above. None is a named hypothesis. The objective it optimises (constant metric) is not D_P or the frozen-bench Q_P, and no bound links them. |
| T1 part 1 | proof | OK given the setting | Correct under the setting. The proof uses H-clean, H-sym, H-read, K-invariance and R^T M R = M. |
| T1 part 2 | proof | GAP | (a) "the" best k-space needs lambda_k > lambda_(k+1); part 5 shows the tie happens. (b) "Pi_V is the part carried by V" must be the projection onto C^{1/2} V, not V. (c) Needs M positive definite for M^{1/2} and a free write; inherited from T0 part 3, say it. |
| T1 part 3 | proof / Lean | OK | Complete given the setting. `perm_fixes_simple_eigvec`, `const_of_rotate_invariant`, `k1_read_coeff` state the steps used. (Docstring of `perm_fixes_simple_eigvec` says "T1 part 2"; it is part 3.) "Neither depends on P, K or M" is true of the formulas; the premise does depend on them (B1.2 says so). |
| T1 part 4 | proof | GAP (premise too weak) | "strictly decreasing in abs(n) for abs(n) <= d + 1" does not exclude a mode n > d + 1 with a larger eigenvalue than mode d. Example: N = 10, d = 1, lambda_0 > lambda_1 > lambda_2 < lambda_5 > lambda_1. B1.2 itself warns about n = N/2. "separated from the rest" does not follow. Needs lambda_d > max over abs(n) > d of lambda_n. Also needs d + 1 <= N/2 for the premise to make sense. |
| T1 part 4, Fejer step | proof | GAP (minor) | "classical; checked for d = 1, 2" is not a proof. One line closes it: the coefficient matrix of the shifted kernels in the e^{in theta} basis is a (2d+1)-point DFT matrix times diag(1 - abs(n)/(d+1)), both invertible. Sampling at N >= 2d + 1 points is injective on degree-d trigonometric polynomials (at most 2d zeros); say so. |
| T1 part 4 Lean `bump_nonneg` | Lean | OK | States 1 + cos >= 0, as used. |
| T1 part 5 | proof | GAP | Inherits the part 4 premise gap. "h0" is not defined (the constant mode). |
| Remark B1.1 | stand-alone | NOT MATH (last two sentences) | "one vector of N numbers per orbit of readers comes back" is an unproved claim. |
| Remark B1.2 | proof | OK | The sufficient condition (positive entries of C^{1/2} Phi C^{1/2}) is correct by Perron and Frobenius. |
| Remark B1.3 | stand-alone | NOT MATH | "the log found that bumps fire falsely off target" is a measurement in Part B. "the write absorbs any change of basis" needs one line (B Lambda = (B A^-1)(A Lambda)), not `PAPER.md` Thm. 11. |
| Remark B1.4 | proof | GAP / not stand-alone | Claims "parts 1, 2, 4, 5 hold with Sigma" and points to `PAPER.md` Thm. 10b. With K over positions the absorbed mean is K-weighted; mu is not defined. Sigma need not be positive definite. |
| Proposition T1' | proof, tag | OK | Correct for open U in field space. Tag none is right. |
| T1' Lean line | Lean | GAP | `relu_add_iff` is not the "one-dimensional core" of T1'. T1' is about any z', `relu_add_iff` only about z' = z_1 + z_2. |
| T1' Consequence | stand-alone | NOT MATH | "natural inputs cross several of them" is empirical. Also T1' is about open sets of fields; real inputs lie on a thin subset, where relu(z') can match t without contradiction. |
| B.2 Setting | definition | OK | The linearised laws are definitions. |
| Lemma T3.0 | proof | OK | Correct, both directions, mod 2 pi. The script checks only the forward direction on a counter-clockwise circle. |
| Theorem T3 | proof | OK | All Wiener integrals, the limit and the monotone rise are right. Tag none is right. The Ito isometry is classical. |
| Corollary T3.1 | proof, tag | GAP (not yet a theorem) | (a) "ideal tangent-element detector" is not defined. (b) The proof uses kappa_0 Gaussian, which H-contour does not give; for general nu, E[phi given y] is not linear. The s -> 0 limit beta -> 2 may hold for any nu with finite variance; that needs a proof. (c) It uses the linearised law for the actual curvature process; the error of the linearisation as s -> 0 is asserted, not shown. (d) Conditioning on a curve passing through the origin at angle 0 is a measure-zero event (Palm distribution). (e) With Gaussian tuning of width w_y in offset and w_phi in angle, the co-activity is the Gaussian with covariance Sigma + diag(w_y^2, w_phi^2) (plus the reference detector's width), and the peak slope becomes s Cov(phi, y) / (Var y + w_y^2 + ...). Finite tuning attenuates the slope from 2 toward 3/2, so "decidable" needs w. (f) Tag: the 3/2 clause is about the direction process, not H-contour. (g) This co-activity is not the C of B.1. |
| Conjecture T2 | route | OK as conjecture | Route step (ii) "rank one" is doubtful: other curves fire subsets of a tangent ring. The Caution cites the log (NOT MATH, minor). |
| Conjecture T4 | route | GAP / not stand-alone | Part 1 uses "the event-frequency vector of `NEURON-MATH.md` section 33", which is not defined here. Part 2's eps_clean is a feature distance; the route needs a bound from eps_clean to the change of (S, R). |
| `exact_no_gain` (Exact.lean) | Lean | OK, unused | Not cited by the paper. It is `subst`. |
| `model/t1_identities.py` | script | OK, wording GAP | Checks what it says. "Exactly zero" for the circulant part is a 50-digit numeric threshold (1e-40), not symbolic. Fejer independence is checked for the continuous functions, not for the N sampled vectors. |
| `model/t3_pairing.py` | script | OK | Checks the integrals and slopes. Lemma T3.0: forward direction on one orientation only. |
| `model/t1_instance.py` | script | GAP | One position, no K, no pool. It does not touch the hidden geometric assumptions of B.1. The "H-sym broken" control does not break H-sym: the input law (i.i.d. entries) is shift-invariant; only the sample is not. The "H-read broken" control reports that the write coefficient stays 1/N, but for the summed read, which is no longer optimal there; that check is empty. No instance tests the part 4 premise gap. |
| Lean build | Lean | OK | `lake build` passes. No `sorry`. |

## 2. Stand-alone (Part C and the Appendix deleted)

Items that are a measurement, an engineering claim or an unproved belief written as a result:

| Item | What is not mathematics |
|---|---|
| H-clean, last sentence | promises a stability statement that is only Conjecture T4 part 2 |
| H-sym comment | gravity claim, no source |
| F4 proof, F9 proof, T0.1 proof, B1.4 | proof lives only in `NEURON-MATH.md` or `PAPER.md` |
| Remark B1.1, last two sentences | unproved claim |
| Remark B1.3 | "the log found that bumps fire falsely off target" |
| T1' Consequence | "natural inputs cross several of them" |
| T2 Caution | log result |
| T4 part 1 | object defined only in `NEURON-MATH.md` section 33 |
| "What B.0 says", last sentence | "removes the data dependence of the k = 1 write altogether": the premise of T1 part 3 still depends on the data |

## 3. Hypothesis tags

| Item | Tag | Verdict |
|---|---|---|
| B0.1 | none | Right |
| T0 | none | Right (parts 3 and 4 carry their premises in the statement) |
| T0.1 | H-stat | Right, plus F9's unstated genericity condition |
| T1 | H-clean, H-sym, H-read | Wrong: also uses a constant Kronecker metric, K rotation-invariant, continuum positions with non-lattice translation covariance, equal spacing g_i = gamma^i, and no (or rotation-equivariant) pooling |
| T1' | none | Right |
| T3.0, T3 | none | Right |
| T3.1 | H-contour | Wrong: uses Gaussian kappa_0, linearisation, an undefined detector model; one clause is about the direction process |

## 4. Content count

| Kind | Items | Count |
|---|---|---|
| Structural (says what the coarse network is) | T1 (parts 3 to 5 carry the content; parts 1 and 2 are the circulant and Eckart-Young computation) | 1, conditional on a surrogate metric |
| Candidate structural, not yet a theorem | T3.1 | 1 |
| Data-model computation (about the image law, not the network) | T3 | 1 |
| Bookkeeping (true of any quadratic, any ReLU, any plane geometry) | B0.1, T0, T0.1, T1', T3.0 | 5 |
| Facts | F1 to F9 are generic (any ReLU network). F10 is an architecture fact. F6, F7, F8 are not used in Part B | 10 |

Finding: the proved part of the paper is mostly bookkeeping. The one structural theorem, T1, holds for a surrogate
objective (constant Kronecker metric) whose link to D_P is not stated, inside a setting with several unnamed
hypotheses. The structural content is therefore exactly as strong as those hidden hypotheses.

## 5. Literature

No definition and no borrowed result uses an SAE, dictionary-learning or superposition framing. T1 merges members
into orbit modes; it does not split them into sparse directions. Compatible.

Borrowed results whose hypotheses are not stated: K-FAC (the independence of activity and curvature, which is what a
constant Kronecker metric means), Eckart and Young (uniqueness needs a gap at k), Davis and Kahan (needs a gap),
Fejer, Gray 2006, Perron and Frobenius, Szegedy et al. 2015. Add rows to the A.4 table.

## 6. The three most important fixes, ranked

1. **Split T1 and name its real hypothesis.** By T0 part 1, the whole objective, for any read and write, depends on
   the data only through the fourth-moment tensor T = E_P sum_{p,p'} h(x,p) h(x,p')^T (x) G_x(p,p'). Name
   **H-kron**: T = C (x) M. This is K-FAC's independence assumption, and it is weaker than a constant metric. Then:
   - **T1a (under: H-circ, H-kron).** H-circ: C and Phi are invariant under the cyclic shift of the members. Parts 2
     to 5 follow by the present proof, with no geometry at all.
   - **Lemma T1b (sufficient condition for H-circ).** Under a new Definition D8 (symmetric model): X = images on the
     plane (or a disc), positions on a continuum, translation covariance at all shifts, g_i = gamma^i, no pool or a
     gamma-equivariant pool with the read taken before the pool, K rotation-invariant with bounded support. Plus
     H-clean, H-sym, H-read with R_g a permutation of readers. Then C and Phi are circulant.
   Part C then measures the distance of C and Phi from circulant matrices, and the distance of T from the nearest
   Kronecker product. Both are defined on the network as it is, so no rotation of a square image is needed.
2. **Fix the premises of T1 parts 2, 4 and 5.** Part 2: "the best k-space" only when lambda_k > lambda_(k+1);
   otherwise "a best k-space", with the set of optima described. Write Pi_V as the projection onto C^{1/2} V.
   Part 4: replace "strictly decreasing for abs(n) <= d + 1" with "lambda_d > max over abs(n) > d of lambda_n". Give
   the one-line DFT proof of Fejer independence and the sampling-injectivity line.
3. **Make T3.1 a theorem or a weaker statement.** Define a detector (delta tuning, or Gaussian tuning of widths
   w_y and w_phi). Either add "kappa_0 Gaussian" to H-contour or prove the s -> 0 limit for any nu with finite
   variance. Prove (or state as a lemma with a bound) that the linearised law approximates the curvature process as
   s -> 0. State the attenuated slope s Cov / (Var y + w_y^2 + ...). Drop "decidable" unless w is known. Retag: the
   3/2 clause is under the direction process.

Next tier, each one is short:

| Fix | Item |
|---|---|
| Add the premise "each unit fires alone on some input" (or "the closed cone of outputs is the orthant"); the thin-cone rotation is a counterexample now | F4 |
| Write S = E sum t(x,p) t(x,p')^T (x) G_x(p,p'); define c; assume finite second moments and Tbar invertible | T0 parts 1 and 3 |
| State "S_1^-1 R_1 != S_2^-1 R_2"; give a finite two-law construction with its genericity condition, so F9 and T0.1 stand here | F9, T0.1 |
| Give the fourth branch its own map M_b; reconcile D6 (pre-pool WHAT) with B.0 and B.1 (pooled signals) | F10, D6, B.0 |
| Make the gate per position; prove part 4 as per-reader weighted least squares; change "leaves the joint span" to "need not lie in", with an example | T0 part 4 |
| Restate the Lean lines that claim more than Lean states: `nonneg_orth_col_single`, `quad_moment_suff`, `write_in_span`, `gated_write_in_span`, and the T1' line | Lean |
