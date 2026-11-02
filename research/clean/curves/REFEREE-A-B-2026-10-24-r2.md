# Referee report, second pass: COARSE-THEORY.md revision 2, Parts A and B (2026-10-24)

Scope: Parts A and B of revision 2, the Lean files in `lean/coarse/Coarse/`, the scripts in `model/` (read, not
run), and the first report `REFEREE-A-B-2026-10-24.md`. Part C (definitions only) and Part D (empty) are not refereed.
`NEURON-MATH.md`, `PAPER.md` and `CURVES-LOG.md` were not read.

Lean: `lake build` passes (9051 jobs). Only style warnings (line length, one flexible `simp`). `grep` finds no
`sorry`, `admit`, `axiom` or `native_decide` in `Coarse/` or `Coarse.lean`.

## 1. First-report findings: fixed or only reworded

| First-report item | Status | Reason |
|---|---|---|
| T1 split; H-kron named | Fixed | T1a (under: H-kron) and Lemma T1b exist. T1b's proof is correct (section 2). |
| H-sym unsatisfiable on X; D2 not an action | Fixed, with a new problem | D8 makes H-sym satisfiable. D8 adds a rotation clause that is false for InceptionV1 and not used (finding 3). |
| "So h_{i+1}(gamma x, gamma p) = h_i(x, p)" assumed | Fixed | Derived in T1b from H-clean and shift equivariance. |
| T1 part 2 uniqueness, P_V | Fixed | Strict gap stated; P_V projects onto C^{1/2} V. |
| T1 part 4 premise | Fixed | min over abs(n) <= d > max over abs(n) > d. |
| Fejer step | Fixed | DFT times diagonal; sampling injectivity. Both correct. |
| T1 part 5 "h0" undefined | Not addressed | "span(h0, v)" still uses an undefined h0. |
| F4 false as stated | Fixed | Premise "fires alone" added. Proof and Lean are correct. |
| T0 part 1 index slip, c, Tbar | Fixed | S and R checked term by term (section 2). Finite second moments still unstated. |
| T0.1 quantifier | Fixed by weakening | Now conditional on its own conclusion's content. It is a tautology. |
| F9 outsourced | Fixed | Proof in place and correct. Its link to S and R is only the conditional in T0.1. |
| F10 contradiction | Fixed | Branch maps M_1, M_4. Architecture numbers 192 + 96 + 16 + 64 = 368 are right. |
| T0 part 4 | Fixed | Per-reader weighted least squares; example is correct (heuristic choice of a, b; finding 9). |
| H-read on positions, permutation | Fixed | |
| Log citations in Part B | Fixed | Moved to the Appendix. |
| Literature rows | Fixed | Rows added. K-FAC row equates H-kron with independence; that is not exact (finding 8). |
| B1.3 basis change | Fixed | One line given. New sentence "contiguous arcs ... approximately equal space" is a belief. |
| B1.4 K-weighting, mu, Sigma definite | Reworded, not fixed | Same formula Sigma = C - mu mu^T; still wrong (finding 4). The Appendix says "proof written". |
| T3.1 detector undefined, Gaussian nu, tag | Partly fixed | D9 added and tag corrected. The new proof has a conditional/unconditional error (finding 1). |
| T3.1 (g): the D9 co-activity is not the C of B.1 | Not addressed | T2 route (ii) still uses "co-activity" with no link to C. |
| T4 part 1 undefined object | Fixed | Now points to Remark B1.1. |
| T4 part 2 link eps_clean -> (S, R) | Not addressed | Route still bounds the change of S^-1 R by eps with no bound from eps_clean to delta S, delta R. |
| A.0 one input field A(x) | Not addressed | Members of an Inception block can sit in different branches with different inputs. |
| D3, D4 position index; m_i, mu_F unused | Not addressed | D3 and D4 still have no p. T1b supplies p inside its proof. m_i and mu_F are still unused. |
| D5 "circuit" unused | Not addressed | Minor. |
| H-contour "smooth", "rendered as edges" | Not addressed | Curvature-process curves are C^2, not smooth. "Rendered as edges" is undefined. |
| Lean lines claiming more than Lean states | Fixed | The paper now says the Lean lines assume the normal equations. `exact_no_gain` is mislabelled (section 4). |
| Scripts | Fixed in wording | The plane-model instance for T1b is still missing (stated). |

## 2. The new mathematics

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| D8 plane model | definition | GAP | "Every layer below L commutes with every element of SE(2) acting on the plane" forces every channel of A(x) to be invariant under rotation about its own position (an oriented conv kernel K commutes only if K = gamma K). False for InceptionV1 from conv1 on. T1b never uses it: it uses shift equivariance, the pooling clause, H-sym and H-read. |
| D8, finiteness | definition | GAP (minor) | Under H-sym (a stationary law on the plane) T is finite only if the metric localises in absolute position. A translation-covariant metric gives T infinite. H-kron assumes T finite, but Fubini in T1b needs absolute integrability. Say which metric. |
| D9 ideal detectors | definition | GAP | (a) It conditions on arc length s, which a detector at position v does not see. The right ideal object is the occupation density (integral over s). In the linearised law x(s) = s, so the two agree there; say so. (b) "converges only in the sense of distributions" is asserted, not shown, and the ideal object is not shown to be any limit of the width-w responses. (c) Two finite-width models (kernel k, Gaussian errors) are given and not related. (d) One w scales both a length and an angle. |
| Lemma T1b | proof | OK | Shift step h_i(x, p) = f_0((tau_p gamma^i)^-1 x) correct; gamma tau_{-p} gamma^-1 = tau_{-gamma p} correct; (Pi (x) R) T (Pi (x) R)^T = T correct; Kronecker factors up to a scalar a, trace gives a = 1, correct; R^T M R = M follows from R M R^T = M because R is orthogonal. Equalities hold P-a.s. (exact clean is an L2(P) statement); say so. |
| Lemma T1b | tag | OK | Uses H-clean, H-sym, H-read, H-kron and D8 (named in the statement). H-kron is used only in the last step. |
| T1a part 1 | proof | OK | Y, Z reduction, projection onto C^{1/2} V, c = 1/2 abs(Y)^2 all check. The characterisation of all optima is Ky Fan, not Eckart and Young; cite it. |
| T1a part 2 | proof | GAP (minor) | "the best read spaces are spans of Fourier modes" fails under a cross-frequency tie (lambda_2 = lambda_3): a line inside the joint eigenspace, such as cos 2 theta + cos 3 theta, is optimal and is not a span of modes. |
| T1a part 3 | proof / Lean | OK | Premise reduces to lambda_0 > lambda_n for all n != 0. `perm_fixes_simple_eigvec`, `const_of_rotate_invariant`, `k1_read_coeff` match. |
| T1a part 4 | proof | OK | Premise, uniqueness, DFT step and sampling injectivity are correct. |
| T1a part 5 | statement | GAP (minor) | h0 undefined. Needs N >= 3 for a two-dimensional frequency-1 plane. |
| T1a | tag | GAP (convention) | Honest only because "circulant", "linear pooling" and "same-position read" sit in the statement. By the paper's own rule these are hypotheses. Linear pooling is false for 3b (max-pool) and has no Part C row. |
| B.0 "several branches add" | stand-alone | GAP | The frozen-bench metric D J^T Pi J D couples readers of different branches, so the objective does not split by branch. Also the read Lambda is the pre-pool WHAT, shared by branches with different pools (M_1, M_4), so T1a must be a sum over branches with different C_b. Mode spans stay optimal only if the order premise holds for every branch. Neither is stated. |
| Remark B1.1 | proof | GAP | Lambda_{Rj} = Pi Lambda_j needs a read whose row space is Pi-invariant (summed read, or a span of modes). Under the gated metric nothing shows the best k = 1 read is the summed read. |
| Remark B1.4 | proof | GAP (not fixed) | Under G = M (x) K the optimal bias is the K-weighted mean, and the centred matrix is Sigma = C - m m^T / (1^T K 1), m = E sum_{p, p'} K(p, p') h(x, p'). The paper's C - mu mu^T drops the weight and the normaliser (in the plane, the double integral of K). Sigma is only semi-definite; T1a needs it definite. B1.4 treats a constant metric, T1a is under H-kron. It carries a proof and no tag. |
| T0 part 4 example | proof | OK, minor | B = (a, b, 1), span needs B_3 = a + b, correct. "a + b != 1" is argued by words. A two-input law (x1: h = (1, 0), only reader 1 open; x2: h = (0, 1), only reader 2 open) gives a = b = 1. The reason "Lambda_j t-weights reproduce t" is garbled: 1^T Lambda_j = 1 because 1^T h = t. |
| F4 | proof / Lean | OK | `fires_alone_nonneg` plus `nonneg_orth_col_single` state the two steps. The "consequence" sentence (a rotation symmetry must act on the plane) does not follow; it is a comment. The premise has no Part C row. |
| F7 | proof | GAP (minor) | Needs a loss that depends on the weights only through the network function. Weight decay breaks balance, and F8 and H-stat assume weight decay. ReLU is not differentiable at 0 (Du, Hu and Lee use the Clarke sense). F7 is still unused. |
| F9 | proof | OK | Score has mean zero; Fisher added. "Infinite mass" for P_train makes F8's sum diverge; drop it. |
| T3 part 1 | proof | OK | Kernel (s - w) w / s and variance sigma^2 s^3 / 30 are correct. No moment of nu is needed at all. |
| T3 part 1 gloss | statement | GAP | "the partner angle is 2 alpha ... for every curvature law nu" is meaningful only relative to the spread of phi. If nu = delta_0, phi is itself of order s^{3/2}, Var(err) / Var(phi) = 1/10 for all s, and the best linear slope is 5/2, not 2. Needs Var nu = tau^2 > 0; then the ratio is O(rho s). |
| T3 parts 2, 3 | proof | OK | Slope formula, monotone rise, direction-process residual correct. |
| Corollary T3.1, curvature half | proof | GAP (first gap) | "Theorem T3 parts 1 and 3 give the conditional law of phi given y" is false for part 1. Part 1 is an unconditional moment. Cov(phi - 2y/s, y) = sigma^2 s^4 / 40 != 0, so the conditional law is not centred at 2y/s. For Gaussian nu, E[phi - 2y/s given y] = (rho / 10) y / (1 + rho s / 5); the conditional variance is sigma^2 s^3 / 30 minus a positive term. For non-Gaussian nu the conditional law is not computed. |
| Corollary T3.1, direction half | proof | OK | Gaussian and centred: conditional mean (3/2) y / s, variance sigma^2 s / 4. |
| Corollary T3.2 | statement | GAP | (a) Not one test: each law is scored with its own predictor (2 and 3/2). Use one statistic: Var(phi' - 2 y'/s) = sigma^2 s^3 / 30 + ... (curvature) against sigma^2 s / 3 + ... (direction; kernel (2w - s)/s). (b) D9 adds errors to each element; the proof adds them only at arc length s. Reference errors add more constant and s^-2 terms; the exponent claim survives, the stated constants are wrong. (c) "decides without the widths" needs exact variances at four or more distances and sigma > 0. |
| T3.2 check in `model/t3_pairing.py` | script | GAP (minor) | The width terms are added by hand on both sides, so the check re-checks only T3 part 1 and the direction kernel. |
| T3 comment | stand-alone | NOT MATH | "its error is of higher order in the turning angle" is an unproved order claim. |

## 3. Hypothesis tags

| Item | Tag | Verdict |
|---|---|---|
| B0.1 | none | Right. |
| T0 | none | Right. Finite second moments unstated. |
| T0.1 | H-stat | The proof uses nothing. H-stat enters only the premise's meaning. The result is a tautology. |
| T1a | H-kron | Honest only through premises in the statement. Linear pooling, circulant C and Phi, same-position read are hypotheses in all but name. |
| T1b | H-clean, H-sym, H-read, H-kron | Right. D8's rotation clause is carried but unused. |
| B1.1, B1.4 | none (remarks) | Both carry proofs. B1.1 uses H-clean, H-sym, H-read (stated in text). B1.4 uses a constant metric, H-sym and K rotation-invariant, not stated as a tag. |
| T1' | none | Right. |
| T3.0, T3 | none | Right (each law is a premise of its part). |
| T3.1, T3.2 | none | Right as tags. They are about D9 objects, not the network's units. "Network units are D9 detectors" is an unnamed hypothesis whenever Part C applies them. |

## 4. Lean

| Name | Paper claim | Verdict |
|---|---|---|
| `relu_add_iff` | F2 | Same statement. |
| `fires_alone_nonneg`, `nonneg_orth_col_single` | F4 | Same two steps. Composition not stated as one theorem; trivial. |
| `reynolds_le` | F6 | Stronger (no linearity). |
| `quad_moment_suff`, `write_in_span`, `gated_write_in_span` | T0 parts 1, 3, 4 | Weaker, and the paper now says so. `write_in_span` docstring says "for any metric"; it needs an invertible constant M. |
| `perm_fixes_simple_eigvec`, `const_of_rotate_invariant`, `k1_read_coeff` | T1a part 3 | Same steps. |
| `bump_nonneg` | T1a part 4, d = 1 | Same. Fejer non-negativity for general d is not in Lean; not claimed. |
| `exact_no_gain` | cited by T0.1 as "the converse direction" | Mislabelled. It is a consequence of T0 part 1 (same statistic, same optimum), proved by `subst`. |
| `Basic.lean`, `Exact.lean` docstrings | | Still cite `NEURON-MATH.md` section numbers as the source of record. |
| Build | | Passes. No `sorry`. |

## 5. Stand-alone (Part C and the Appendix deleted)

| Item | What is not mathematics |
|---|---|
| T3 comment | unproved order of the linearisation error |
| D9 | "converges only in the sense of distributions" asserted |
| Remark B1.3 | "contiguous arcs ... approximately equal space" |
| H-kron comment | "this is the independence assumption behind K-FAC": independence of h and G gives a sum over (p, p') of Kronecker products, not one, unless E G_x(p, p') = M k(p, p') |
| T2 route (ii) | "rank one" co-activity under the circle law, with no definition linking co-activity to C |
| T3.1 comment against T3.2 | T3.1 says deciding needs the widths; T3.2 says it does not. Say slope needs widths, exponent does not |

## 6. Content count

| Kind | Items | Count |
|---|---|---|
| Structural (says what the coarse network is) | T1a (parts 3 to 5), Lemma T1b | 2, both conditional on H-kron; T1a also on circulant data and linear pooling |
| Structural remark, not yet a result | B1.1 (needs the read condition), B1.4 (formula wrong) | 2 |
| Data-model (about the image law and ideal detectors, not the network) | T3, T3.1, T3.2 | 3 |
| Bookkeeping (any quadratic, any ReLU, plane geometry) | B0.1, T0, T0.1, T1', T3.0 | 5 |
| Facts | F1 to F10; F6, F7 unused in Part B; F8 used only through F9 and H-stat | 10 |

Finding: 2 of 10 proved results are structural, up from 1. Both rest on H-kron, which has no derivation and no
sufficient condition other than a constant metric. The paper is still mostly bookkeeping plus a data model.

## 7. Literature

No definition or borrowed result uses an SAE, dictionary-learning or superposition framing. T1a merges members into
harmonic spaces; the Fejer basis is non-negative and is one basis among many (B1.3). Compatible.

Borrowed hypotheses not stated: Ky Fan (the characterisation of all optima in T1a part 1); Du, Hu and Lee (loss depends
on the function only, no weight decay); K-FAC (H-kron is stronger than independence; see section 5).

## 8. The three most important fixes, ranked

1. **Repair Corollary T3.1 and T3.2.** For the curvature half of T3.1, either state the y-averaged moment
   (integral of (phi - 2y/s)^2 against the co-activity = sigma^2 s^3 / 30), or restrict to Gaussian nu and state the
   exact conditional mean 2y/s + (rho / 10) y / (1 + rho s / 5) and the conditional variance. Restate T3.2 with one
   statistic, phi' - 2y'/s (sigma^2 s^3 / 30 against sigma^2 s / 3), with errors on both elements as D9 says.
   Add tau > 0 to the gloss of T3 part 1.
2. **Weaken D8 to shift equivariance below L.** Drop the rotation clause (unused, and false for any network with an
   oriented filter below L). State that the metric in T1b localises in absolute position, so that T is finite under
   H-sym. Promote "linear pooling" and "circulant C and Phi" to named hypotheses with Part C rows.
3. **Fix B1.4 and B1.1, and the branch claim in B.0.** B1.4: Sigma = C - m m^T / (1^T K 1), assume it definite, give a
   tag. B1.1: add "for a read whose row space is Pi-invariant". B.0: replace "branches add" by one stacked reader
   vector with cross-branch blocks, and state that T1a with several branches needs the order premise for every C_b.

Next tier, each one is short:

| Fix | Item |
|---|---|
| Replace "spans of Fourier modes" by "spans of top eigenvectors; mode spans when the lambda at place k is not shared across frequencies"; define h0; N >= 3 | T1a parts 2, 5 |
| Define co-activity as the occupation density; note it equals the fixed-s density when x(s) = s | D9 |
| Give the two-input example for a + b != 1; fix the "reproduce t" reason | T0 part 4 |
| Restrict F7 to losses that depend only on the network function | F7 |
| State H-kron as exact factorisation, not as K-FAC independence | H-kron, A.4 |
| Add the position index to D3 and D4; delete m_i, mu_F, "circuit" or use them | D3, D4, D5 |
| Relabel `exact_no_gain`; fix the `write_in_span` docstring | Lean |
| Add the bound from eps_clean to (delta S, delta R) to the T4 route | T4 part 2 |
