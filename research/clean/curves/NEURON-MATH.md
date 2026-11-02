# The coarse neuron theorem: a working paper (curves in InceptionV1 3b)

Started 2026-10-24. Plan: `~/.claude/plans/okay-so-the-last-harmonic-stallman.md`. Log cards: `CURVES-LOG.md`,
section "THEOREM (2026-10-24)". Earlier paper: `PAPER.md` (its proved results are used here by number, "P-Thm 1" etc.).

This paper is built like a proof. It states the theorem it wants first. Every statement has an ID, a status and the
statements it depends on (the ledger, section 3). A check that refutes a statement sends the work back to the theory:
the statement is marked REFUTED, everything that depends on it is marked SUSPENDED, and the reason is written down
before any new run.

| Label on numbers | Meaning |
|---|---|
| DERIVED | weights (and file constants) only, by a rule written before the number |
| FITTED | any average or solve over an image set |
| TRAINED | gradient descent on the output loss. One imperfect example of a coarse network, never the bar |

---

## 1. Why the large network should be enough (read first; re-read when stuck)

| # | Reason | Source | What it implies |
|---|---|---|---|
| W1 | A network is a measure over neurons; width N = N samples of it. Trained wide two-layer networks converge to one limit measure (Mei, Montanari, Nguyen 2018; Chizat, Bach 2018). Deep results are partial (Nguyen, Pham). For InceptionV1 at width 480 this is a FRAME, not a theorem. | literature (hedged) | Large and small networks are two discretizations of one object. Coarsening is quadrature with fewer nodes. A trained small network is one noisy quadrature. |
| W2 | The weights hold a metric on neuron space. | P-Cor 2 (proved): kernel inner products = error-weighted co-activation | A quadrature needs a distance between neurons; we have one with no images. |
| W3 | The impossibility was proved only for exact stationarity. For SGD training the per-neuron imprint excluded every write-moving alternative built. | P-Thm 35.1, P-Stmt 35.2, X4-X9 | The weights may pin the write for this network. |
| W4 | The alternatives that break uniqueness are huge: +.017 KL needed 10x extra self-labelled mass, +.053 needed 100x. | X6 | Wide on paper, narrow in practice. |
| W5 | The next layer's weights predict the real-image Fisher's structure (.81-.87; noise .32-.45). | P-section 26.2 | The natural-image metric is in the weights. |
| W6 | A rotation family has one natural measure: uniform over angle (Haar). | group theory | The generalist mix of a symmetric family is fixed by the symmetry. |
| W7 | Features are learned from the data (sparse coding of natural images gives Gabors: Olshausen, Field 1996). | literature | The weights encode the input statistics by construction. P-Thm 5 hides only the frequency of inputs the network already gets right. |
| W8 | The trend: the weights-made coarse neuron moved toward the real atto neuron (6/6); network-made writes beat a real-data write (.0661 vs .0681). | MAP-STATE; P-section 32 | Each session closed part of the gap without data. |
| W9 | Training is provably imperfect here: dead units (P-Thm 13), start-inherited basis (I5), broken rotation symmetry (S3), random leftovers in unused directions (C3), k = 10 random cannot find the copy. | past checks | A trained network is a sample, not the truth. |

**The reframe.** The weights fix the function for every input. P-Thm 5 and P-Thm 34.3 judge a coarse network by KL
averaged over a data distribution, so they need that distribution by definition: that is a choice of norm, not a fact
about information. With a norm defined by the network itself (W2, W5, W6), no outside data is needed by construction.
Scope, stated honestly: P-Cor 2 gives this norm on the LINEAR span of the members' pre-activations. A new coarse unit is
a new ReLU, and its norm needs one more object: how the members' gates overlap with the new unit's gate (the gated
plain moment of C5 and P-Stmt 13.2). So the question collapses to one named object (L6) and one transfer lemma (L8).

**Rule for failures.** A failed check means a wrong definition or a wrong premise, not "impossible" (the essay: blame
the microscope, not the network).

---

## 2. Part 0: the hidden first principles of the earlier work

| # | Hidden principle | Forced or chosen | What it caused |
|---|---|---|---|
| H1 | "Best coarse network" = argmin of KL over a data distribution | chosen | P-Thm 5, 24, 34.3 need data by definition |
| H2 | The trained network is the bar | chosen | Every build was judged by its distance to training |
| H3 | Second-order KL (Fisher) as the equation of the write | chosen | The write needs per-image metric statistics (P-S3: the 60% "shape" part) |
| H4 | Stationarity (P-Thm 2) as THE link from weights to data | chosen | "Weights = data moments" leads straight to "missing moments" |
| H5 | A neuron = a vector in R^800 | chosen | It hides the orbit: orientation, curvature centre |
| H6 | Readers stay fine and frozen | chosen (test bench) | Coarse units must fake detail for fine readers: the write problem |
| H7 | A coarse unit = one ReLU of the same wiring | partly forced (product) | A sum of ReLUs is not a ReLU (P-Lemma 3) |
| H8 | One family at a time, one k at a time | chosen | Nesting across k was never scored |

This paper keeps H6 and H7 as the test bench (so numbers stay comparable) and drops H1-H5 and H8 as definitions.

---

## 3. The target theorems and the ledger

**Setting.** Family of N fine neurons: read kernels K_i (32 x 5 x 5 on the 5x5 bottleneck), biases b_i, write columns
W_.i (368 reader inputs over the four 4a input convs). Contribution to the readers: u(x) = sum_i W_.i h_i(x), with
h_i = pool(relu(z_i)). As a measure: u(x) = integral W(phi) h(x; phi) d mu_N(phi), phi = the neuron's parameter
(for curves: orientation on the circle).

**Main Theorem (target, CONJECTURE).** If the family is clean (its members lie on one group orbit up to a measured
error eps_orbit), a map C_k exists that reads the WEIGHTS ONLY and gives k ReLU units with writes such that:
1. Optimality: in the network's own norm ||.||_theta (L2 + L6), C_k is the best k-unit approximation of u among
   one-ReLU-per-unit builds.
2. Training is an imperfect estimate: for any trained k-unit build T, ||u - u_C||_theta <= ||u - u_T||_theta; on
   natural images KL(C) <= KL(T) + Delta, with Delta bounded by the distance between ||.||_theta and the natural norm (L8).
3. Nesting: C_1 is the coarse level of C_3 (multiresolution: fine = coarse + detail).

**Theorem B (target, CONJECTURE).** Width-k networks trained on the same data approximate the same limit measure as
the width-N network; C_k(theta_N) is at least as close to it as a trained width-k network, up to terms shrinking in N.

**Ledger.** Status: PROVED / PROVED-LEAN / CHECKED / CONJECTURE / OPEN / REFUTED / SUSPENDED.
| ID | Statement (short) | Depends on | Status |
|---|---|---|---|
| L1 | Orbit: the 10 kernels are one template under rotation | - | REFUTED (Prop. 6.1: impossible at a ReLU layer) |
| L1' | Sampled field: equal norms, angle coordinate recoverable from weights, eps_orbit large | - | CHECKED (R1) |
| L2 | Network's norm: P-Cor 2 metric, Haar-averaged over the orbit | L1 | SUSPENDED (L1 refuted) |
| L2' | Network's measure on the family = empirical measure mu_N of recovered angles x importance | L1', P-Cor 1 | definition (carries the network's emphasis) |
| L3 | One ReLU carries a sum of members exactly where they agree in sign | - | PROVED (P-Lemma 3, P-Prop 14.3) |
| L4 | Co-firing and the co-firing shift come from kernel overlap | P-Cor 2, L3 | CHECKED in rank (.76); shift direction (C1) |
| L5 | Equal-angle arcs | L1, L2 | SUSPENDED |
| L5' | Best k-partition = quantization of mu_N into contiguous cells of near-equal mass | L1', L2', L3 | CONJECTURE (M's arcs 3 / 3-4 / 3 consistent) |
| L6 | GATE STATISTICS: members' co-firing (read) and readers' on/off states (write) | L2', L4, Prop. 7.5 | OPEN (the single open object) |
| L7' | True coarse write lies in the joint span of member columns | L1', L3 | REFUTED for KL (R2); PROVED for reader-input matching (Prop. 7.2) |
| P7.2 | Reader-input matching: optimal write in the family span for any read | - | PROVED |
| P7.3 | Write = sum over all neurons the unit resembles | P-Thm 2, H1 | REFUTED (R2b); reason Prop. 7.4 (double counting) |
| P7.5 | Readers' gates make the write reader-specific: row-wise span, out of the joint span | - | PROVED |
| L8 | Transfer: the network's norm is close to the natural-image norm | W5, L2' | OPEN |
| L9 | Group averaging of the write cannot hurt a convex invariant objective | - | PROVED-LEAN (section 5); no exact group to apply it (Prop. 6.1) |
| P9.1 | Free reader biases are redundant with a dial unit | - | PROVED + CHECKED (B2) |
| P11.1 | B3: readers re-route by one universal Lambda; Lambda = S_F,rest S_rest,rest^-1 | P7.2 | PROVED; closed form beats a CAPPED 2000-step ruler (-.0091) but loses to a converged one (+.0082, X-B3-long); best as a starting point (.0446) |
| S | The layer activity second moments from the network alone | P11.1 | OPEN (network-made: .086 vs real .065) |
| 12.4 | One Gaussian for the pre-activations gives the post-ReLU covariance | S | REFUTED (G1: family rows .714 < .80; means exact) |
| 12.4' | A Gaussian scale mixture (one shared contrast) fixes it | 12.4 | REFUTED (G2: worse, .632; the shared contrast concentrates) |
| 12.4'' | Sparse mixture over a dictionary; the open part = co-occurrence law of the dictionary amounts | 12.4' | CONJECTURE (written, not run) |
| 12.3, 12.5, 12.6 | Recursion to the pixels, bottleneck ReLUs, readers' gates | 12.4 | SUSPENDED (each uses the 12.4 ReLU step) |

---

## 4. L1: is the curve family one orbit? (statement, then check R1)

**Definition 4.1 (exact orbit).** The family is an exact orbit of the rotation group SO(2) if there are a template K_0,
an orthogonal representation R(alpha) on the 32 bottleneck channels, the spatial rotation P(alpha) of the window, and
angles theta_i with K_i = (R(theta_i) x P(theta_i)) K_0; and an orthogonal representation S(alpha) on reader space with
W_.i = S(theta_i) W_.0.

**Lemma 4.2 (channel-free consequences).** Under Definition 4.1:
1. ||K_i|| and ||W_.i|| are the same for every member.
2. <K_i, K_j> = c(theta_j - theta_i) and <W_.i, W_.j> = d(theta_j - theta_i) for even functions c, d.
3. For any pair, min over orthogonal Q of ||Q P(theta_j - theta_i) K_i - K_j|| = 0, where K is read as a 32 x 25 matrix
   (Q acts on channels). This residual does not depend on the unknown R.
**Proof.** (1)-(2): with g_i = R(theta_i) x P(theta_i) orthogonal, <g_i K_0, g_j K_0> = <K_0, g_i^T g_j K_0> =
<K_0, g(theta_j - theta_i) K_0> =: c(theta_j - theta_i). Swapping i and j gives c(-x) = c(x) because the inner product is
symmetric. The same for S. (3): take Q = R(theta_j) R(theta_i)^T; P commutes with the channel action. QED.
(P(alpha) is exactly orthogonal only for multiples of 90 degrees on the 5x5 grid; other angles use interpolation, an
approximation we report.)

**Check R1 (weights only). Prediction written before running:**
| Part | Measure | Prediction | Baseline |
|---|---|---|---|
| (a) | CV of ||K_i|| and of ||W_.i|| | <= .10 each | - |
| (b) | R^2 of the best even fit c(x) = a0 + a1 cos x + a2 cos 2x over the 45 pairs, for K and for W | >= .5, above the 95th percentile of 1000 angle shuffles | shuffled angles |
| (c) | angle recovery: alpha* = argmin over alpha (5-deg grid) of the Procrustes residual of Lemma 4.2(3) | median abs(alpha* - (theta_j - theta_i)) <= 30 deg; >= 70% of pairs within 45 deg | uniform random angle: median 90 deg |
| (d) | exact grid pairs: (50, 320) at -90 deg, (70, 250) and (170, 350) at 180 deg | rotated residual < unrotated, and < the 5th percentile of the same residual to the other 86 5x5-branch neurons | unrotated; other neurons |
If (b) and (c) hold, L1 is CHECKED and eps_orbit is the measured residual. If they fail, L1 is REFUTED and L2, L5, L7
(as written) are SUSPENDED.

---

## 5. L9: group averaging cannot hurt (statement and proof)

**Setting.** A finite group G (for example the rotations by multiples of 90 degrees, or a cyclic subgroup) acts on
builds. Fix a read that is G-invariant (g . read = read for all g, as a loop detector should be). The objective
J(B) (any loss of the write B) is G-invariant: J(g . B) = J(B).

**Theorem 5.1 (Reynolds averaging).** If J is convex in B and G-invariant, then for every write B the group average
B-bar = (1/|G|) sum_g g . B satisfies J(B-bar) <= J(B). Equality holds when B is already invariant.
**Proof.** Convexity (Jensen): J((1/|G|) sum_g g . B) <= (1/|G|) sum_g J(g . B). Invariance: each term equals J(B). QED.
**Scope.** (1) The output KL is convex in the write only to second order (P-Thm 7 quadratic). (2) Natural images are
not exactly rotation-invariant (gravity: horizontal and vertical edges are more common), so J on natural images is
invariant only approximately. (3) Nothing here is claimed for the READ, where the objective is not convex.
**Consequence for training (an imperfect example).** If the trained write is not G-invariant while the read is, the
non-invariant part of the trained write is, under (1)-(2), pure loss in any invariant norm. Check R2 measures it.
Status: PROVED (Lean port pending).

---

## 6. Result of R1, and what it does to the theory

**Check R1 (weights only, `r1_orbit.py`).**
| Part | Prediction | Result | Verdict |
|---|---|---|---|
| (a) norm CV | <= .10 | kernels .035, columns .025 | HOLDS |
| (b) even circulant fit, R^2 | >= .5, above shuffle 95th pct | kernels .278, columns .427; shuffle 95th pct .149 / .147 (rank .998 / .999) | real structure, but FAILED the threshold |
| (c) angle recovery | median <= 30 deg, >= 70% within 45 | median 22 deg, 82% within 45 (with the one global sign convention, the other sign gives 110 deg / 26%) | HOLDS |
| (d) exact grid pairs | rotated < unrotated, and < 5th pct of other neurons | rotated < unrotated 3/3; beats the other-neuron 5th pct 1/3 (343 -> 349 only) | PARTLY FAILS |
(d) shows the per-pair Procrustes test is weak: with 32 free channel directions and 25 positions, an orthogonal map can
match many unrelated neurons. The test (c) is still meaningful because it compares angles of one pair, not neurons.

**Why the exact orbit fails: a proof.**
**Proposition 6.1 (a ReLU layer admits only permutation symmetries).** Let X in R^n_+ be the outputs of a ReLU layer, and
suppose image rotation by alpha acts on them linearly by an orthogonal Q(alpha): X(rotated image) = Q(alpha) X(image),
for a set of images whose X span R^n. Then every Q(alpha) is a permutation matrix. Hence a continuous rotation group
SO(2) can act only trivially: no exact continuous orbit exists at a ReLU layer.
**Proof.** Q maps the non-negative orthant into itself (outputs stay outputs), so each column Q e_i is >= 0. The columns
are orthonormal; two non-negative vectors with inner product 0 have disjoint supports. n non-zero vectors with disjoint
supports in R^n each have exactly one non-zero entry, which is 1 by unit norm: Q is a permutation. The map alpha ->
Q(alpha) is a continuous homomorphism from the connected group SO(2) into the finite permutation group, so it is
constant: Q = I. QED.
**Meaning.** The 32 bottleneck channels cannot rotate into each other continuously. A rotation family at a ReLU layer is
at best a SAMPLED field: each member is a sample at angle theta_i, with other parameters (radius, curvature, shape) free
to vary between samples. That is exactly what R1 sees: the angle is a clean coordinate (c), the norms are equal (a), and
the inner products depend on the angle gap only in part (b).

**Revised L1 (L1').** The family is a sampled field over the circle: (i) equal norms (CHECKED: CV .035 / .025);
(ii) an angle coordinate recoverable from the weights alone (CHECKED: median error 22 deg); (iii) inner products depend on
the angle gap with R^2 .28 (kernels) and .43 (columns): eps_orbit is large. Status: CHECKED for (i)-(ii).

**Ledger changes (refutation protocol).**
| ID | Old | New |
|---|---|---|
| L1 | exact orbit | REFUTED (Prop. 6.1 proves it cannot hold). Replaced by L1' (CHECKED) |
| L2 | Haar measure over the orbit | SUSPENDED. Revised L2': the network's measure on the family is the EMPIRICAL measure mu_N of the recovered angles, weighted by importance (P-Cor 1: weight norm^2; here equal). It needs no symmetry, and it carries the network's emphasis: more members where the network spends more neurons (around 170-180 and 320-10 deg). |
| L5 | equal-ANGLE arcs | SUSPENDED. Revised L5': the best k-partition is a quantization of mu_N into k contiguous cells; with equal importance, cells of near-equal member count. (M's trained arcs: 3 / 3-4 / 3 members: consistent; to be stated as a theorem.) |
| L9 | Theorem 5.1 (Reynolds) | still PROVED, but its use in R2 needs a finite group acting on the writes; by Prop. 6.1 none is exact. R2 is redefined below. |

**Answer to "how does the network say what to emphasise".** In this frame the emphasis is the measure mu_N itself: how
many neurons the network spends per stretch of angle, times their importance (weight norm^2, P-Cor 1). Both are read
from the weights. The coarse network should preserve this measure, not a uniform one.

## 7. L7': the true coarse write lies in the span of the fine writes (statement, then check R2)

**Statement 7.1 (L7').** In the sampled-field frame, a coarse unit stands for a smoothed piece of the field:
c_m ~ sum_i kappa_mi h_i (kappa_m >= 0, supported on cell m). The readers then receive
sum_i W_.i h_i ~ sum_m B_m c_m only if B_m = sum_i lambda_mi W_.i, ONE set of coefficients for all 368 reader inputs
(the coefficients say which members the unit stands for; they cannot differ between readers). So the true coarse write
lies in the 10-dimensional span of the member columns, jointly over the four 4a convs.
Depends on: L1', L3. Status: CONJECTURE.
**Consequence (imperfect example).** Split a trained write into its projection on that span plus a remainder. L7' says
the remainder is training error: removing it should not hurt.

**Check R2 (written before running).** For the trained k = 1 twins (`L-k1-m379-lr3e-2`, `L-k1-r-lr3e-2`) and M
(`L-k3-m-lr3e-2`): keep the trained read; replace each write column by (i) its least-squares projection on the joint
span of the 10 stacked member columns (368 x 10), the L7' write; (ii) per-conv projection (40 coefficients per unit),
a diagnostic. The column bias stays zero as in training. Paired dev KL on all 5000 dev images. Also report the
fraction of each trained column's energy outside the joint span.
PREDICTION: the joint-span write costs at most 2 paired se more than the trained write for all three builds (L7' holds;
the remainder is training error). If it costs more than 2 se, L7' is REFUTED: the trained write uses reader
directions outside the members' span, and the reason must be found in the theory (likely the read: the trained coarse
unit is not a smoothed piece of the field, so the readers must correct it).

**Check R2 (`r2_span.py`, dev-5000, paired).**
| Build | trained write | joint-span write | paired | per-conv span | paired | energy outside joint span |
|---|---|---|---|---|---|---|
| k = 1, start 379 | .1244 | .2916 | +.167 (se .011) | .2199 | +.096 (.008) | 51% |
| k = 1, random start | .1256 | .3041 | +.179 (.011) | .2262 | +.101 (.008) | 52% |
| k = 3, M | .0361 | .0580 | +.022 (.002) | .0496 | +.014 (.002) | 12-14% |
PREDICTION (cost <= 2 se): FAILED for all three. L7' is REFUTED for the KL objective.

**Why: a proof that the span statement is right for one objective and wrong for ours.**
**Proposition 7.2 (span theorem for reader-input matching).** For ANY read c (any coarse units) and any constant
positive-definite metric M on reader inputs, the write minimising E||u - B c||_M^2 is B* = E[u c^T] E[c c^T]^-1 =
sum_i W_.i E[h_i c^T] E[c c^T]^-1. It lies in the span of the member columns, and does not depend on M.
**Proof.** Normal equations: E[(u - B c) c^T] = 0 (M cancels because it multiplies on the left and is invertible). Substitute
u = sum_i W_.i h_i. QED.
So a write outside the span can only help through something Prop. 7.2 leaves out. In our objective there are exactly two
such things: the per-image metric (P-Lemma 26.3) and the readers' own ReLUs. A reader that is off ignores any error in its
input, as long as it stays off. So the KL objective lets a write push signal into readers the family never fed, provided
the push keeps them on the right side of their gates, or moves them where the output needs it.
**Meaning.** Training's out-of-span half is not noise. It is the coarse network using the readers' nonlinearity. Under H6
(fine frozen readers with ReLUs), the true coarse network uses it too. L7' is replaced by the question: what does the
out-of-span part compute?

**Statement 7.3 (the write follows what the unit computes; CONJECTURE).** Readers read features, not neuron indices. If a
coarse unit's pooled activity decomposes over ALL 480 3b neurons, c~ = a_0 + sum_n a_n h~_n + noise, then its write is
B = sum_n a_n W_.n (the columns of every neuron it resembles, not only the family's).
**Derivation (first order).** P-Thm 2(A) for every 3b neuron n: lam W_.n = -E[sum_p delta(p) h~_n(p)]. The same identity
for the coarse unit in a crushed network that is also stationary, with H1 (delta' = delta): lam B = -E[sum delta c~]
= sum_n a_n (-E[sum delta h~_n]) + (noise term) = lam sum_n a_n W_.n, if the noise is uncorrelated with delta. QED under H1.
**Why it explains R2.** The trained k = 1 unit is a loop detector. Loops also drive non-family 3b neurons (the circle
family of Phase 5, and others). So its a_n are large on non-family neurons, and its write lies along their columns: out
of the family span. At k = 3 the units are broad curve detectors, close to the family, so less lies outside (12-14%).
**Check R2b (premise check with images, FITTED coefficients; written before running).** For the same three trained reads:
regress each unit's pooled activity on the 480 pooled 3b activities plus a constant (2000 train images, least squares),
and set B = sum_n a_n W_.n. Baselines: the trained write; the in-span least-squares write of Prop. 7.2.
PREDICTION: B from 7.3 is within .02 of the trained write at k = 1 (it recovers most of the +.17), and beats the in-span
write by more than 2 se at k = 1. If it does not, 7.3 is REFUTED and H1 (delta' = delta) is the first premise to question.
If it holds, the write reduces to the coefficients a_n: how a unit decomposes over existing neurons. That is a gate object
of the same kind as L6, and becomes the single open object.

**Check R2b (`r2b_follow.py`, FITTED coefficients, dev-5000).**
| Build | trained | 7.3 write | in-span lsq write (Prop. 7.2) | 7.3 - in-span |
|---|---|---|---|---|
| k = 1, start 379 | .1244 | .4364 | .3468 | +.090 (se .014) |
| k = 1, random | .1256 | .4973 | .3466 | +.151 (.014) |
| k = 3, M | .0361 | .1092 | .0885 | +.021 (.005) |
Share of a^2 on the family: k = 1 .33 / .16; k = 3 .73-.84. PREDICTION: FAILED. Statement 7.3 is REFUTED.

**Proposition 7.4 (why 7.3 fails: double counting).** In the crushed network the 470 non-family neurons keep their own
columns. A write component along a non-family column W_.n sends that neuron's signal a second time. Under reader-input
matching the optimal write is in the family span (Prop. 7.2), so every non-family component of B is pure loss for that
objective. In the stationarity derivation, the false premise is H1 (delta' = delta): the crushed network's reader error
contains exactly the duplicated signal. QED.

**The object: what training's out-of-span half does (weights only, `scratchpad o1_outspan.py`).**
| Build | entries negative | sum of out-of-span part | sum of in-span part | best single-column match |
|---|---|---|---|---|
| k = 1, start 379 | 66% | -5.47 | +1.15 | abs(cos) <= .19 |
| k = 1, random | 70% | -6.98 | +1.10 | <= .20 |
| k = 3, M (3 units) | 60-65% | -1.4 to -2.5 | +0.1 to +1.3 | <= .24 |
The out-of-span part is a broad NEGATIVE push over most readers, not a copy of any neuron. Same shape in both k = 1 twins
(started differently), so it is a property of the problem, not of one run.

**Proposition 7.5 (why gates move the write out of the span).** Let reader j have output relu(s_j + u_j), with s_j the
input from the other neurons. Count the error only where reader j is on in the intact network (gate g_j). The best write
for reader j is B_j = E[g_j u_j c] / E[g_j c^2] = sum_i W_ji lambda_ij with lambda_ij = E[g_j h_i c] / E[g_j c^2]. The
coefficients lambda_ij depend on the READER through its gate, so the write leaves the joint span (one lambda for all
readers). Each reader's coarse weight is still a mix of ITS OWN fine weights (row-wise span). Readers that are off in the
intact network but would be woken by the coarse unit add a penalty that pushes their B_j negative.
**Proof.** For fixed j the gated least squares has normal equation E[g_j (u_j - B_j c) c] = 0. Substitute u_j. QED.
This is the write-side twin of the read's problem: both sides need GATE statistics (who is on where).

**Where the theory stands after three checks.**
| Fact | Status |
|---|---|
| A rotation family at a ReLU layer is a sampled field, never an exact orbit | PROVED (Prop. 6.1) |
| The angle coordinate and equal importance are in the weights | CHECKED (R1) |
| For reader-input matching the write is in the family span, for ANY read | PROVED (Prop. 7.2) |
| The readers' gates move the true write out of the span; that part is real (+.17 at k = 1, +.022 at k = 3) | PROVED (Prop. 7.5) + CHECKED (R2) |
| Training's out-of-span half is not noise | CHECKED (R2): the "imperfect example" claim does NOT hold for the write's out-of-span part |
| Everything still open reduces to gate statistics: members' co-firing (read) and readers' on/off states (write) | the single open object (L6, widened) |

## 8. The trained neuron as an imperfect example: the twins (statement, then check R2c)

**Setting.** Two k = 1 trainings of the same architecture from different starts: twin A (start = member 379) and twin B
(random start). Dev-5000: .1244 and .1256. Their disagreement is training error by definition: the target is the same.

**Gauge.** relu(a z) = a relu(z) for a > 0, so (K, b, B) and (aK, ab, B/a) are the same network. Fix a by balance
(P-Thm 1): ||K||^2 + b^2 = ||B||^2.

**Statement 8.1 (averaging two imperfect examples).** Write each twin as theta_T = theta* + e_T (truth + training error).
If the loss L is convex on the segment from theta_A to theta_B (gauge-fixed), then L((theta_A + theta_B)/2) <=
(L(theta_A) + L(theta_B))/2, strictly less when the errors point in different directions and L has positive curvature
along e_A - e_B. **Proof.** Jensen on the segment. QED.
So if the twins are two noisy samples of one truer neuron in one valley, their average is better than both on average.
If the average is worse, the twins sit in different valleys (the k = 1 optimum is not unique), which is a fact about the
target itself (Main Theorem part 1 would then need a choice of valley).

**Check R2c (written before running).** (a) Weights only: gauge-fixed cos(K_A, K_B), the biases, cos(B_A, B_B).
(b) Dev-5000 paired: averaged read + averaged write; averaged read + each twin's write; the two twins.
PREDICTION: (a) kernel cos between .3 and .7 (P-C3: kernels trained without weight decay keep start leftovers in
directions the input never excites; the functional correlation of twins is .79, P-C4); write cos >= .8.
(b) The averaged build scores <= the better twin (.1244), and at least 2 paired se below the mean of the twins.
If (b) holds, the common part is the truer object and training error is real at k = 1 (Shonu's claim, on the read).
If the average is worse by more than 2 se, 8.1's premise (one convex valley) is REFUTED.

**Check R2c, first run (`r2c_twins.py`, balance gauge).** (a) kernel cos .513 (predicted .3-.7: HOLDS); write cos .948
(predicted >= .8: HOLDS); biases +10.3 / -25.6; gauge-fixed kernel norms .2 / .1. (b) avg read + avg write .1401: WORSE
than twin A by +.0157 (se .0027). avg read + A's write .1813; + B's write .3887. PREDICTION (b): FAILED.
**Reason (a false premise, not yet a verdict on 8.1).** The balance gauge rests on P-Thm 1, which needs weight decay. The
twins were trained with plain Adam and no decay (`family.py` line 125), so balance is not a property of these networks.
Worse, the bias (in pre-activation units, ~10-25) dominates ||K||^2 + b^2, so the gauge scaled the two kernels by
factors 2x apart. The "average" then mixed two differently-scaled functions.
**Corrected gauge (Definition 8.2).** Without decay, the only meaningful gauge is the one that makes the two examples
closest: a* = argmin_a ||K_A - a K_B||^2 + ||B_A - B_B / a||^2 (weight entries only; the bias rides along as a b_B).
Statement 8.1 needs exactly this: errors measured after removing the symmetry.
PREDICTION (unchanged from above, for the corrected gauge): the averaged build is <= twin A and >= 2 se below the twin
mean. If it still fails, 8.1's one-valley premise is REFUTED.

**Check R2c, corrected gauge.** a* = .794. Kernel cos .513, write cos .948, biases +74 / -387 (in pre units), kernel norms
1.4 / 1.2. Dev-5000: avg read + avg write .1253; twins .1244 / .1256. Paired vs the twin mean +.0003 (se .0007); vs twin A
+.0009 (se .0012). PREDICTION (b): the one-valley premise HOLDS (the average is not worse), but the predicted strict gain
(>= 2 se below the twin mean) FAILED: the average is equal.
**Statement 8.3 (what the twin disagreement is).** By 8.1, equality on the segment means zero curvature along e_A - e_B.
So the two k = 1 examples differ only in FLAT directions of the loss: kernel cos .51 and biases 74 vs -387, yet the same
function to within .001 KL. Those directions are (i) input directions the bottleneck never excites (P-C3) and (ii) the
trade between the bias and the kernel's component along the mean input (X >= 0, so K . E[X] acts as a second bias).
**Meaning for "trained = imperfect example".** At k = 1, on this score (KL on natural images, frozen fine readers), the
trained neuron's error lies where the loss cannot see it. Training reaches the valley floor. This agrees with P-Thm 12
(two starts agree to .001, so the optimizer gap is ~0 at k = 1). Room to beat training at k = 1 can only come from the
finite-image gap, or from a different score (H6, H8). At k = 3 training IS imperfect in a visible way (dead units from a
random start, P-Thm 13; a start-inherited partition, I5), so k = 3 is where a derived build can win on this score.
Ledger: 8.1 PROVED; its one-valley premise CHECKED at k = 1; 8.3 CHECKED.

---

# Part II: readers adapt (Shonu's decision, 2026-10-24)

**Decision.** H6 is relaxed. The readers' weights on the coarse units (B) AND the readers' biases (368 numbers, the
4a input-conv biases) are part of what the math derives. Everything else stays frozen. This is closer to a truly
smaller network: a reader that loses its fine inputs may re-centre its threshold. Old numbers stay as history; the new
bench needs its own trained ruler.

**Bench B2.** Coarse read (K', b'), write B, reader-bias change Delta-b (368). Score: KL(intact || crushed), dev-5000.

**Ruler for B2 (TRAINED, an imperfect example).** Warm starts from the best known B1 networks (twin A at k = 1, M at
k = 3), Delta-b = 0 at the start, all of (K', b', B, Delta-b) trained: 1000 steps, lr .01 cosine, early stop on dev 1000
(`b2_train.py`).
**Prediction (written before training).** (i) k = 1 improves from .124 to .100-.115 (the constant-fill effect, P-section 7:
reader-side constants alone took removal .64 to .39). (ii) The out-of-span share of the trained write stays >= 40% at
k = 1: by Prop. 7.5 the out-of-span part is c-dependent gating, which a constant cannot replace. If the share drops below
20%, the out-of-span half was mean (DC) compensation, and Prop. 7.5 is not the main mechanism.

**Formal proofs (Lean 4 + Mathlib, `lean/coarse/Coarse/Basic.lean`, compiles with no `sorry`).**
| Lean theorem | Paper statement | Status |
|---|---|---|
| `relu_add_iff` | L3: relu(a + b) = relu(a) + relu(b) iff a b >= 0 | PROVED-LEAN |
| `reynolds_le` | L9 / Theorem 5.1: group average cannot raise a convex invariant objective | PROVED-LEAN |
| `nonneg_orth_disjoint`, `nonneg_orth_col_single` | Prop. 6.1 core: a non-negative orthogonal matrix has exactly one entry 1 per column (a permutation) | PROVED-LEAN (the topological step, "a continuous map from SO(2) to a finite group is constant", is on paper only) |
| `write_in_span` | Prop. 7.2: the least-squares write is W Lambda, for any invertible metric | PROVED-LEAN |

**B2 ruler (`b2_train.py`, `b2_score.py`, dev-5000).**
| | B1 (biases frozen) | B2 (reader biases free) | out-of-span share of the write | reader-bias change |
|---|---|---|---|---|
| k = 1 | .1244 | .1217 (se .0062) | 51% (B1: 51%) | mean -.001, sd .019 (biases are .26 +- .09) |
| k = 3 | .0361 | .0350 (se .0017) | 12-14% (B1: 12-14%) | mean +.0005, sd .014 |
Joint-span projection cost under B2: k = 1 +.178 (se .011); k = 3 +.021 (se .002).
PREDICTION (i) (k = 1 to .100-.115): FAILED, the gain is .003. (ii) (out-of-span share >= 40% at k = 1): HOLDS.

**Proposition 9.1 (why free reader biases add almost nothing: the dial is a bias).** If a coarse unit is on at every
position, c = v . X + b', then its write adds B b' to every reader: a change of the coarse bias b' by e is exactly a change
of the reader biases by B e. So for an always-on unit, Delta-b and (b', B) are degenerate along the direction B, and the
free reader biases only add the 367 directions orthogonal to B. **Proof.** On positions where c > 0, B c = B (v . X) + B b'.
QED. The trained k = 1 unit is on at ~70% of positions (a dial), so this holds approximately. The constant fill that
took removal .64 to .39 (P-section 7) is already supplied by the dial; that is part of what the dial IS.
**Meaning.** Relaxing H6 by reader biases is nearly a no-op. The out-of-span gating (Prop. 7.5) is c-dependent and stays.
A real relaxation of H6 would let the readers' weights on OTHER 3b channels adapt (the readers become coarse readers).
That changes the ruler again (open choice for Shonu).

## 9. Reading the k = 3 example: the in-span write is the kernel overlap (observation, then statement)

**Observation (M, weights only, `scratchpad o2_k3span.py`).** Write each unit's write in member coordinates
(lambda_m = least-squares coefficients on the 10 member columns) and its read in member coordinates (coefficients on the 10
member kernels, raw weight metric; in-span energy .68-.77).
| unit | read centre | cos(lambda, read coef) | cos(lambda, plain kernel overlap) | cos(lambda, max-shift overlap, |s| <= 2) |
|---|---|---|---|---|
| 0 | 10 deg (330) | .83 | .91 | .94 |
| 1 | 120 deg (388) | .92 | .89 | .93 |
| 2 | 250 deg (385) | .84 | .85 | .91 |
The read is concentrated on one member; the write spreads over its angular neighbours (unit 0: 320, 350, 10, 50, 70 deg
with .22 / .44 / .57 / .34 / .17). In quadrature language: a node READS near its own angle and its write covers its
CELL of the field.

**Statement 9.2 (the in-span write is fixed by the weights up to one scale per unit).** By Prop. 7.2 the in-span write is
lambda_mi = (regression of member i's activity on unit m), i.e. co-activation. By P-Cor 2 the kernel overlap
<K_i, T_s K'_m> is the error-weighted co-activation; by P-Conj. B the error weighting keeps the co-firing PATTERN. So
lambda_m is proportional to the max-shift kernel overlap of each member with the unit's read: a weights-only quantity.
Status: CHECKED on M (cos .91-.94); the scale per unit is open (it is a gate statistic: E[c^2]).
**What it does not cover.** The in-span part alone costs +.021 at k = 3 (the joint-span projection). The out-of-span
gating (Prop. 7.5) is the rest of the write.

## 10. State at the end of this stretch (2026-10-24)

| Result | Status |
|---|---|
| Prop. 6.1: no exact continuous orbit at a ReLU layer; families are sampled fields | PROVED-LEAN (algebraic core) |
| The network's emphasis = empirical measure of its neurons on the angle coordinate x importance | definition; angles CHECKED from weights (median error 22 deg) |
| Prop. 7.2: reader-input matching puts the write in the family span for any read | PROVED-LEAN |
| Prop. 7.5: readers' gates move the true write out of the span; this part is real (+.17 at k = 1, +.02 at k = 3) | PROVED + CHECKED |
| Statement 8.3: the k = 1 twins differ only in flat directions; training reaches the valley floor at k = 1 | CHECKED |
| Prop. 9.1: free reader biases are redundant with a dial unit's bias | PROVED + CHECKED (B2 = B1 within .003) |
| Statement 9.2: the in-span write direction = kernel overlap with the unit's read | CHECKED on M (cos .91-.94) |
| The single open object | GATE STATISTICS: unit scales E[c^2], the readers' gating, members' co-firing for the read |

---

# Part III: readers fully adapt (Shonu's decision, 2026-10-24)

**Bench B3.** Coarse units (K', b'), their write B, AND every reader's weights on the 470 remaining 3b channels plus the
reader biases (the four 4a input convs: 368 x 471 + 368 numbers) are free. Everything below 3b and above the 4a input convs
stays frozen. This is a smaller network whose readers have become coarse readers. Score: KL(intact || crushed), dev-5000.

**Rulers (TRAINED, imperfect examples; `b3_train.py`).** Warm starts from the B2 rulers with reader changes at 0; 2000
steps, lr .003 cosine, early stop on dev 1000. Plus a k = 0 ruler: no coarse unit, readers adapt alone (how much the
readers can rebuild from the other 470 neurons).
**Prediction (written before training).**
| Ruler | Prediction | Reason |
|---|---|---|
| k = 0 | .25 - .45 | between removal .66 and the constant fill .39 plus re-routing through correlated neurons (circles, lines) |
| k = 1 | <= .100 (B2: .1217) | readers can stop expecting orientation detail they cannot get |
| k = 3 | <= .030 (B2: .0350) | same, smaller effect (less detail lost) |
| out-of-span share of the coarse write, k = 1 | drops below 40% (B2: 51%) | Prop. 7.5's gating can move into the readers' own weights on correlated neurons |

## 11. How should the readers re-route? (derived before the B3 rulers finish)

**Proposition 11.1 (universal re-expression under reader-input matching).** Let z~ = (h_rest, c, 1) collect the pooled
activities of the 470 remaining neurons, the k coarse units and a constant. The lost signal is u_F = W_F h_F (W_F: the
368 x 10 family columns). The reader changes (Delta-W, B, Delta-b) that minimise E||u_F - Delta-W h_rest - B c - Delta-b||_M^2,
for any constant invertible metric M, are
  [Delta-W, B, Delta-b] = W_F Lambda,   Lambda = E[h_F z~^T] E[z~ z~^T]^-1   (10 x (470 + k + 1)).
So every reader re-routes with the SAME matrix Lambda, scaled by its own old weights on the curve neurons. Lambda says how
each curve neuron is re-expressed by the rest of the layer plus the coarse units.
**Proof.** Normal equations as in Prop. 7.2 (`write_in_span` with H = h_F and C = z~); M cancels. QED.
**Consequences (testable on the trained B3 rulers, weights only).**
1. The columns of Delta-W lie in the 10-dimensional span of the family columns W_F (rank <= 10, a fixed subspace).
2. B lies in the same span (Prop. 7.2 again).
3. Deviations from the span are reader-specific and come only from the readers' gates (Prop. 7.5).
**Check R3b (when the rulers finish; written now).** Fraction of the energy of the trained Delta-W (and of B) inside
span(W_F), jointly. PREDICTION: >= 60% for Delta-W at k = 0 and k = 1 (random 10-dim subspace of the 368-dim reader space:
2.7%). If it holds, the B3 problem reduces to Lambda, the re-expression of each curve neuron by the rest of the layer:
co-activation statistics of the layer (the same gate-statistics object, now across all of 3b).

**B3 ruler k = 0 (readers adapt alone, no coarse unit).** dev-5000 .0738 (se .0038); early-stopped on dev-1000 (.0887).
PREDICTION (.25 - .45): FAILED in the favourable direction. With readers free, removing the whole curve family costs only
.074, less than the trained k = 1 on bench B1 (.124). The other 470 neurons carry most of what the readers took from the
curve family; the readers only had to re-route. (Caveat: dev-1000 was used for early stopping and is inside dev-5000.)

**Check R3b/R3c at k = 0 (`r3b_span.py`, `r3c_func.py`).** Raw energy of the trained Delta-W inside span(W_F): 33%
(random 10-dim subspace 3.4%); top-10 singular values hold 62%; |Delta-W| is 6% of |W_rest|. PREDICTION (>= 60% of raw
energy): FAILED. But raw energy is the wrong measure for weights trained without decay (Statement 8.3, P-C3: Adam leaves
leftovers in flat directions). Functional test, dev-5000:
| k = 0 build | KL |
|---|---|
| readers frozen (removal) | .664 |
| Delta-W projected on span(W_F) (the universal re-expression of Prop. 11.1) | .191 (se .010) |
| trained Delta-W | .074 (se .004) |
The universal part recovers (.664 - .191) / (.664 - .074) = 80% of the trained gain. The remaining 20% (.117) is
reader-specific, exactly the kind of term Prop. 7.5 attributes to the readers' gates.
**Statement 11.2.** Under B3, Prop. 11.1 carries 80% of the re-routing gain at k = 0. The open objects are (a) Lambda,
the re-expression of each curve neuron by the rest of the layer (layer co-activation), and (b) the reader-specific gate
correction. Status: CHECKED at k = 0.

**B3 ruler k = 1.** dev-5000 .0678 (se .0033), warm start from B2 k = 1. The coarse unit adds only .006 over k = 0 (.0738)
once readers adapt. PREDICTION (<= .100): HOLDS. (k = 3 pending.)

**Check R3d (`r3d_lambda.py`): the closed form of Prop. 11.1, no training, no coarse unit.** Lambda from the pooled 3b
second-moment matrix S (481 x 481 with a constant; 2000 real train images, FITTED), Delta-W = W_F Lambda_rest,
Delta-b = W_F Lambda_const. dev-5000 .0647 (se .0038). Trained B3 k = 0 ruler: .0738; trained B3 k = 1: .0678.
Each curve neuron's second moment explained linearly by the other 470 neurons: 57-76%.
**Statement 11.3 (a formula beats its training example; provisional until the paired check R3e).** On bench B3 at k = 0
the closed-form optimum of the stated objective (reader-input matching, Prop. 11.1) beats 2000 Adam steps on the true
KL objective by ~.009, with no gradient step. In the gap split (P-Thm 12) this means the optimizer gap O(T) of the
trained ruler exceeds the formula's approximation gap A: the trained network is an imperfect example of the solution
the theorem names. The only data-dependent object in the formula is S, the layer's activity second moments.

**B3 ruler k = 3.** dev-5000 .0297 (se .0014), warm start from B2 k = 3. PREDICTION (<= .030): HOLDS.

**Check R3e (`r3e_paired.py`, dev-5000, paired per image).**
| Build (bench B3, no gradient step unless TRAINED) | KL | paired vs trained k = 0 | paired vs trained k = 1 | label |
|---|---|---|---|---|
| Prop. 11.1 formula, S from 2000 real train images | .0647 (se .0038) | **-.0091 (se .0031)** | -.0031 (.0033) | FITTED |
| Prop. 11.1 formula, S from the 256 network-made class inversions (+ flips) | .0860 (.0048) | +.0122 (.0037) | +.0182 (.0038) | network-made (weights + the fixed E1 generation rule; label open) |
| trained k = 0 | .0738 (.0038) | 0 | | TRAINED |
| trained k = 1 | .0678 (.0033) | | 0 | TRAINED |
**Statement 11.3 (CHECKED).** On bench B3 the closed-form formula with real moments beats the trained k = 0 example by
.0091 (2.9 paired se) with no gradient step, and ties the trained k = 1 example while using no coarse unit at all. The
trained networks are imperfect examples of the solution Prop. 11.1 names. (The rulers were early-stopped on dev-1000, a
subset of dev-5000, which favours the rulers, not the formula.)
**The network-only version.** With S from the network's own inputs the formula reaches .0860, with no real image and no
training: far below the B1 rulers (.124 at k = 1) and .012 from the trained k = 0 example. The remaining gap is the
difference between natural-image moments and network-made moments: S itself (P-section 25: class inversions are modes,
not typical inputs).

**Where the theorem stands now (Part III).**
| Object | Status |
|---|---|
| Re-routing structure: Delta-W = W_F Lambda, one Lambda for all readers | PROVED (Prop. 11.1); carries 80% of the trained gain (R3c) and, in closed form, beats training (R3e) |
| Lambda = S_F,rest S_rest,rest^-1: needs only the layer's activity second moments S | PROVED |
| S from the network alone | OPEN: network-made moments give .086 vs real .065 |
| Reader-specific gate correction (Prop. 7.5) | OPEN: the other 20% of the trained re-routing |
| The coarse unit under B3 | worth .006 at k = 1, .044 at k = 3 (k = 0 .0738 vs k = 3 .0297) |

**Caution on the rulers (advisor audit).** Every B3 ruler stopped at or near its 2000-step cap while still improving
(best steps 1750 / 1750 / 2000, lr .003, 165k new parameters from zero). So Statement 11.3 currently says "beats a
2000-step run", not "beats the best known". Two runs decide it (P-Thm 12):
- **Exactness test X-B3.** Train B3 k = 0 started AT the formula (2000 steps, lr .003). Any fall below .0647 is a lower
  bound on the formula's approximation gap A. PREDICTION: it falls by more than 2 paired se (the formula ignores the
  readers' gates, Prop. 7.5, worth up to the 20% of R3c).
- **Stronger from-zero rulers.** 6000 steps, lr .01 and .03, early stop on dev-1000 (eval every 250, patience 6).
  PREDICTION: at least one reaches <= .065, i.e. the formula's win over a converged from-zero ruler shrinks below 2 se.
Statement 11.3 is restated after these runs, whatever they show.

**Check R3f (`r3f_lamdiff.py`): where the network-made moments differ.** Lambda rows (470 coefficients per curve neuron):
cos(real, network-made) .75 - .84; norm ratio .95 - 1.27. The constant terms differ strongly (e.g. 3.41 vs 1.81, 2.86 vs
-.20). 3b activity levels E[h^2]: Spearman .49, median ratio 1.58, within x2 for 57%. Normalised co-activation over all
480 x 480 pairs: corr .72. So the directions transfer partly, the scales mostly, and the constants (the mean levels:
"how often", P-Thm 5's blind object) least.
**Statement 11.4 (written before the split run).** The network-made gap (.086 vs .065) splits into the constant term
(mean levels) and the 470 re-routing coefficients (co-activation pattern). PREDICTION: the constant carries most of it:
real coefficients + network-made constant lands near .086; network-made coefficients + real constant lands near .065
(within .008). If instead the coefficients carry most, the pattern (mode vs typical, P-section 25) is the obstacle.

**Check R3g (`r3g_split.py`, dev-5000).**
| coefficients from | constant from | KL |
|---|---|---|
| real | real | .0647 |
| real | network-made | .0648 |
| network-made | real | .0857 |
| network-made | network-made | .0860 |
PREDICTION (the constant carries most of the gap): FAILED, in the opposite direction. The mean levels do not matter at all
(.0001); the whole network-made gap sits in the 470 re-routing coefficients, i.e. in the normalised co-activation PATTERN
of the 3b layer (real vs network-made corr .72).
**Statement 11.5 (the obstacle moved).** On bench B3 the object P-Thm 5 proves invisible to the weights ("how often":
mean levels, the constant) is irrelevant to the re-routing. What matters is WHICH neurons fire together on natural inputs
(the normalised second moments). The weights are known to carry that pattern in rank (P-Conj. B: kernel overlap vs
co-firing, Spearman .76, within the 5x5 branch). So the open object is now precise and is not the blind one:
  derive the normalised co-activation pattern of the 3b layer (all four branches) from the weights.
Why the means drop out is not yet proved. A candidate: the reader biases enter the KL only through readers near their
thresholds, and Prop. 9.1 / B2 already showed the KL is nearly flat in reader biases (+.003). To be stated properly.

**Literature check (field method, advisor audit).** Prop. 11.1 is the reconstruction step of channel pruning: He, Zhang,
Sun 2017 ("Channel pruning for accelerating very deep neural networks") and ThiNet (Luo, Wu, Lin 2017) refit the next
layer's weights by least squares on the remaining channels' activations, exactly Lambda = S_F,rest S_rest,rest^-1 through
W_F. That literature reports that the reconstruction alone beats short fine-tuning and is usually followed by
fine-tuning, which is why the exactness test X-B3 matters. Weights-only relatives: Neuron Merging (Kim, Park, Kim 2020)
builds the compensation from weight similarity alone (using the ReLU positive-scaling property); data-free parameter
pruning (Srinivas, Babu 2015) merges similar neurons. Statement 9.2 (overlap predicts the in-span coefficients, cos .91-.94)
is in the same family. New here: the result that the means drop out and only the normalised pattern matters (11.5), the
gated reason for the remaining 20% (Prop. 7.5), and the framing of the trained rulers as imperfect examples.
**Label correction.** The class-inversion builds are "network-made", not DERIVED: whether a fixed generation rule counts
as derived is an open decision for Shonu (test: doubling the generator's steps must change nothing by more than 2 se;
not yet run). Table R3e's label is corrected to network-made.

**Correction to Statement 11.5 (instrument flaw found before reporting).** In the R3g split the "constant" is the
regression INTERCEPT, not the mean levels: the mean of the lost signal enters mostly through Lambda_rest x E[h_rest]. So
R3g separated "intercept" from "everything else", and "the means drop out" does not follow. Statement 11.5 is SUSPENDED.
**Correct split (Statement 11.5', prediction written before running).** Write Lambda_rest = Cov_F,rest Cov_rest,rest^-1
(centred: the PATTERN) and const = mu_F - Lambda_rest mu_rest (the MEANS). Build the four hybrids (covariance from real or
network-made, means from real or network-made) from the saved moment matrices. PREDICTION: the covariance carries most of
the gap (real cov + made means within .008 of .065), because R3f shows the scales and levels differ by factors ~1.2-1.6
while the pattern correlation is only .72.

**Check R3h (`r3h_split2.py`, dev-5000).**
| covariance from | means from | KL |
|---|---|---|
| real | real | .0652 (se .0038) |
| real | network-made | .0663 (.0039) |
| network-made | real | .0865 (.0049) |
| network-made | network-made | .0903 (.0050) |
Mean levels, network-made vs real: median ratio 1.40, Spearman .33 (they differ a lot). PREDICTION (real cov + made means
within .008 of .065): HOLDS (+.0011). **Statement 11.5' (CHECKED).** On bench B3 the means of the layer (the "how often"
object of P-Thm 5) carry .001 - .004 of the network-made gap; the centred covariance of the 3b layer carries ~.021. The
open object of the whole k = 0 problem is the covariance pattern of the 3b layer on natural inputs, not its frequencies.
(Ledger: 11.5 SUSPENDED and replaced by 11.5', CHECKED.)

## 12. First weights-only route to the 3b covariance (statement, then check R3i)

**Definition 12.1 (effective kernels).** Map every 3b neuron to the 3a output space (256 channels x 5 x 5 window) by
composing through its branch, with the bottleneck ReLUs replaced by the identity: 1x1 branch = its weights at the centre;
3x3 branch = 3x3 kernel times the 3x3 bottleneck; 5x5 branch = 5x5 kernel times the 5x5 bottleneck; pool branch = its
weights spread evenly over the 3x3 window (max-pool replaced by an average). G = E E^T (480 x 480), plain metric.
**Statement 12.2 (CONJECTURE).** G, read as a covariance, gives Lambda close enough to keep most of the formula's gain.
This ignores three things on purpose: the 3a input correlations, the bottleneck ReLUs, and pre- vs post-ReLU.
**Check R3i (written before running).** corr of the normalised off-diagonal of G with the real normalised covariance; and
the formula built from G (scales from G's diagonal, means from the network-made set, which carry only .001 - .004).
PREDICTION: pattern corr .3 - .5; KL between .10 and .30 (removal .664, real-moment formula .065). A result below .10
would mean the plain kernel geometry already carries most of the covariance pattern.

**Check R3i (`r3i_effkern.py`).** Pattern corr (normalised off-diagonal, all 480 x 480) of G with the real covariance:
.467; family rows (10 x 470): .556. Formula from G (weights only; means from the network-made set, which carry only
.001 - .004): dev-5000 .1765 (se .0079). PREDICTION (corr .3 - .5; KL .10 - .30): HOLDS.
**Meaning.** With no image and no training, the plain kernel geometry recovers (.664 - .177) / (.664 - .065) = 81% of the
real-moment formula's gain. The three ignored pieces (3a input correlations, bottleneck ReLUs, pre- vs post-ReLU) are
each a named, derivable correction: the next statements. Ladder on bench B3, k = 0:
| covariance source | KL |
|---|---|
| none (removal, readers frozen) | .664 |
| kernel geometry, weights only (12.1) | .177 |
| network-made inputs | .090 |
| real images (FITTED) | .065 |
| trained ruler, 2000 steps | .074 |

**Check X-B3 (exactness test, `b3_train.py formula`).** Training started AT the formula: dev-5000 .0446 (se .0022), from
.0647; best dev-1000 step 1625 of 2000. PREDICTION (falls by more than 2 se): HOLDS. So A(formula) >= .020 (P-Thm 12), and
the from-zero 2000-step ruler had an optimizer gap O >= .029 (.0738 vs .0446 for the same architecture).
**Statement 11.3, restated.** (1) The closed form beats a 2000-step from-zero training run by .0091 (2.9 se). (2) It is
.020 above the best known network of the same structure, which is training started at the formula. (3) Starting at the
formula gives training a far better solution than starting at zero (.0446 vs .0738). The formula's missing .020 is what
Prop. 11.1 leaves out by design: the readers' gates (Prop. 7.5) and the per-image metric.

**Check: stronger from-zero rulers (`b3_train.py k0`, 6000 steps, eval every 250, patience 6).**
| Ruler | lr | best step | train KL at end | dev-5000 KL (se) |
|---|---|---|---|---|
| B3-k0 (old) | .003 | 2000 of 2000 (cap) | n/a | .0738 |
| B3-k0-lr01 | .01 | 5500 of 6000 | .033 | .0900 (.0039) |
| B3-k0-lr03 | .03 | 6000 of 6000 | .122 | .3420 (.0125) |
PREDICTION (at least one reaches <= .065): REFUTED. Both are worse than the old ruler. The lr .01 run reaches a train KL
of .033 but a dev KL of .098: it over-fits the 165k free weights. The lr .03 run is unstable for 3000 steps. So a larger
step size does not make a stronger ruler. The open case is the SMALL step size run longer: the old lr .003 ruler was still
improving at its cap. Run X-B3-long: 6000 steps at lr .003, eval every 250, patience 6. PREDICTION (carried over): it
reaches within 2 paired se of the formula (.0647 +- .006), so the formula's win over a converged from-zero ruler is not
yet proved. Statement 11.3 part (1) stays PROVISIONAL until this run.

**Next statements (written, not run).** Each removes one approximation of Definition 12.1, in order of expected size.
| ID | Correction | Mathematical form | Needs |
|---|---|---|---|
| 12.3 | 3a input correlations | pre-activation covariance = E^T Sigma_3a E, with Sigma_3a from the same construction one layer down (recursion to the pixels) | a pixel covariance at the bottom: white (weights only) or 1/f^2 (a law: needs Shonu's OK) |
| 12.4 | pre- vs post-ReLU | for jointly Gaussian pre-activations (CLT over 800+ inputs), E[relu(a) relu(b)] has a closed form in (mu_a, mu_b, sigma_a, sigma_b, rho) (the arc-cosine kernel with means) | the pre-activation means: the biases plus E . mu_input |
| 12.5 | bottleneck ReLUs | the same closed form applied at the bottleneck, inside the recursion | as 12.4 |
| 12.6 | the readers' gates (the formula's missing .020, X-B3) | gated least squares per reader (Prop. 7.5) with gate probabilities from the same Gaussian model | as 12.4 |
Together these are the mean-field signal-propagation recursion (Poole et al. 2016; Cho, Saul 2009 for the arc-cosine kernel),
here used for a TRAINED network to supply the one object Prop. 11.1 needs. Risk, named in advance: the Gaussian step is
exactly what failed for the READ in P-C5 (sparse non-negative inputs); here it is applied to PRE-activations of 800+
inputs, where the CLT is more plausible. The first check of 12.4 is direct: Gaussian-model post-ReLU covariance from the
TRUE pre-activation moments vs the true post-ReLU covariance.

**Check G1 (`g1_gauss.py`): first check of Statement 12.4, premise only (FITTED moments, no score).** Inputs: the TRUE
mean and covariance of the 480 3b pre-activations (same 2000 train images as R3d, all 28 x 28 positions, no pooling).
Gaussian model: draw 400k vectors from N(mu, C), apply ReLU, take the covariance. Truth: the covariance of the true
post-ReLU activity at the same positions. Baseline: the pre-activation covariance itself (no ReLU step). Diagnostic only:
the true post-ReLU covariance before and after the 3 x 3 max-pool (the formula uses the pooled one).
Measures: (a) per-channel variance ratio, model / truth; (b) pattern correlation of the normalised off-diagonal entries,
all pairs and the 10 x 470 family rows; (c) cos of the Lambda rows (Prop. 11.1) built from each covariance.
PREDICTION: (a) within x1.5 for >= 80% of channels; the model UNDER-estimates channels with a strongly negative mean,
because a mixture over images has heavier tails than one Gaussian. (b) all pairs >= .85, family rows >= .80, and both
above the no-ReLU baseline. (c) Lambda-row cos >= .80. If (b) fails, the Gaussian step is the wrong object for
post-ReLU structure and 12.4 is REFUTED before any build.

**Result G1.**
| Measure | Prediction | Gaussian model | No-ReLU baseline | Verdict |
|---|---|---|---|---|
| (a) variance within x1.5 | >= 80% | 75% (median ratio .78; within x2 93%) | n/a | FAILS narrowly |
| (a) direction: under-estimates negative-mean channels | yes | ratio .72 (mean/sd in [-1.5,-.5), n 294) vs .84 (near 0) | n/a | HOLDS |
| post-ReLU means | not predicted | median ratio .99 | n/a | exact |
| (b) pattern, all pairs | >= .85 | .877 | .855 | HOLDS |
| (b) pattern, family rows | >= .80 | .714 | .682 | FAILS |
| (c) Lambda-row cos | >= .80 | .63 - .85 (median .775) | .23 - .43 | FAILS narrowly |
Diagnostic (the max-pool step): true unpooled vs true pooled post-ReLU, family rows .948, Lambda cos .84 - .91. The
Gaussian model against the pooled truth (what the formula uses): family rows .638, Lambda cos .64 - .80. For comparison,
the network-made moments of R3f gave Lambda cos .75 - .84.
**Statement 12.4 as written: REFUTED** on its own criterion (b, family rows). What it does get: the post-ReLU means
exactly, and most of the Lambda direction that the ReLU adds (.33 -> .77 against the no-ReLU baseline).
**Mathematical reason.** One Gaussian has kurtosis 3. The real pre-activations are a MIXTURE over positions and images:
some positions hold an edge, most are flat. A scale mixture has heavier tails. For a channel with a negative mean, the
post-ReLU variance comes from the tail, so one Gaussian under-estimates it, as (a) shows. The family rows are the most
tail-driven entries: the curve neurons fire rarely (negative mean) and co-fire only on curved edges.
**Statement 12.4' (restated, Gaussian scale mixture).** Pre-activations at a position are p = mu + sqrt(v) g, with g ~
N(0, Cn) and one non-negative scalar v (the local contrast) independent of g (Wainwright, Simoncelli 2000 for natural
images). Then E[relu(p_a) relu(p_b)] is the v-average of the 12.4 closed form. Premise check G2: v = |p - mu|^2 / E|p -
mu|^2 per position, Cn = E[(p - mu)(p - mu)^T / v], sample v from its empirical law. FITTED, premise only.
PREDICTION: variance within x1.5 >= 85% and median ratio in [.9, 1.1]; family rows >= .80; Lambda cos median >= .80 (all
against the unpooled truth). If it holds, the open object shrinks from a 480 x 480 matrix to one 1-D law of v plus Cn.

**Result G2.**
| Measure | Prediction | G1 (one Gaussian) | G2 (scale mixture) | Verdict |
|---|---|---|---|---|
| variance within x1.5 | >= 85% | 75% | 79% (median ratio .83) | FAILS |
| family rows | >= .80 | .714 | .632 | FAILS (worse) |
| Lambda cos median | >= .80 | .775 | .738 | FAILS (worse) |
| v law | not predicted | n/a | E[v^2]/E[v]^2 = 1.12, 99th pct 1.94 | narrow |
**Statement 12.4': REFUTED.** **Mathematical reason.** v is the squared norm over 480 channels. By the law of large
numbers it concentrates (E[v^2]/E[v]^2 = 1.12), so it cannot see an event in a few channels. The heavy tails are not one
shared contrast. They sit in specific low-dimensional directions: at most positions only a few features are present (a
curve, an edge), and each one lifts a few channels. That is a SPARSE mixture, not a scale mixture.
**Statement 12.4'' (restated, sparse mixture; written, not run).** p = mu + sum_f s_f d_f + g, with a dictionary of
directions d_f, sparse non-negative amounts s_f (most zero at a given position), and a Gaussian rest g. The post-ReLU
covariance is then the s-average of the 12.4 closed form. The family rows depend mainly on which d_f lift a curve neuron
AND another neuron together, and on how often. Where the network can supply this: the d_f are candidates for the 3a
features read through the effective kernels (R3i), which are weights. The frequencies of s_f are the open part. This is
the same object the past sessions called "which inputs the network gets" (Theorem 34.3), now located precisely: not a
480 x 480 matrix, but a co-occurrence law over a dictionary.

**Ledger action after G1 and G2.** 12.3, 12.5 and 12.6 each use the 12.4 ReLU step. They are SUSPENDED. So the
mean-field signal-propagation route, as written, is blocked at its ReLU step. It stays blocked until the ReLU step has a
working form. 12.4'' is not run with fitted frequencies: R3d already shows that fitted moments work, so such a run proves
nothing new. It is run only when a weights-only candidate for the frequency law exists.
**Calibration note (the verdicts stand).** Criterion (b) was not tied to a score. Criterion (c) nearly held (.775 vs
.80), and network-made moments with Lambda cos .75 - .84 scored .086. So the score cost of the one-Gaussian step is
probably in the network-made range, not catastrophic. This is a guess from a neighbour, not a measurement. One build from
the G1 pooled covariance would measure it.

**Check X-B3-long (`b3_train.py k0`, 6000 steps, lr .003; paired by `r3j_paired_long.py`, dev-5000).**
| Build | Label | dev-5000 KL (se) | minus formula (paired se) |
|---|---|---|---|
| Prop. 11.1 formula, real moments | FITTED | .0647 (.0038) | 0 |
| B3-k0, 2000 steps (capped) | TRAINED | .0738 (.0038) | +.0091 (.0031) |
| B3-k0-lr003-long, best step 5500 of 6000 | TRAINED | .0564 (.0027) | **-.0082 (.0028)** |
| B3-k0-formulastart | TRAINED from the formula | .0446 (.0022) | -.0201 (.0028) |
Train KL at step 6000: .022, dev-1000 .0695 (the run over-fits but still improves on dev). PREDICTION (within 2 se of
the formula): REFUTED, in the other direction. A converged from-zero run beats the formula by 2.9 paired se.
**Statement 11.3 part (1): REFUTED.** The formula beat only a training run stopped too early. **Mathematical reason.**
Prop. 11.1 solves reader-INPUT matching (least squares on the 4a input activities). The score is KL after the readers'
ReLUs and the rest of the network. The difference is exactly what Prop. 7.5 names: the readers' gates, plus the per-image
metric. X-B3 had already bounded this gap from below (A >= .020). The capped ruler hid it.
**Statement 11.3'' (what stands).** (1) The closed form is .0082 worse than converged from-zero training. (2) As a
starting point it is worth much more: training started at the formula reaches .0446 by step 1625, against .0564 at step
5500 from zero (paired -.0118). So the formula carries information that training from zero does not find in 6000 steps.
(3) The unit's worth under B3 was measured against capped rulers (k = 0 .0738, k = 3 .0297, both 2000 steps). Against
the best k = 0 network (.0446) the k = 3 unit is worth about .015, not .044. The k = 3 ruler is also capped, so this
number may move either way.

---

# Part IV: exactness (Shonu's decision, 2026-10-24, plan `okay-for-this-session-twinkly-gosling.md`)

**Rule.** A build that uses real-image statistics is never a result for the generalist, not even an intermediate one
(Shonu: real-image statistics replicate training, "a spiral"). So the Prop. 11.1 build from 2000 real images (.0647,
Statement 11.3) is a FITTED microscope, not a formula. The generalist's weights-only start is R3i (.177).
**Target.** A formula so good that training started at it gains nothing. Training started at a build X is the
exactness test; its gain is X's distance from the optimum that training can reach.

## 13. Definitions and the target theorem

**Setting.** Bench B3 (Part III). Parameters theta' = (coarse units, their write, Delta-W, Delta-b). An image mix D (a
probability distribution over images, with mirror flips). KL_D(theta') = E_D KL(intact || crushed).

**Definition 13.1 (sufficient statistic).** A map m from mixes to a finite-dimensional space is SUFFICIENT for an
objective L_D if L_D = L_D' as functions of theta' whenever m(D) = m(D'). A formula is a map F from statistics to builds.
F is EXACT for L if F(m(D)) is a global minimiser of L_D for every D.

**Definition 13.2 (the four builds).**
| Object | Meaning | Label |
|---|---|---|
| m_theta | the network's own statistic, a function of the weights only | DERIVED |
| G = F(m_theta) | the generalist | DERIVED |
| D_n(X) | F(m-hat) with m-hat from n images of D; no gradient step | MIX-DERIVED (developer) |
| T(X) | training on D started at build X | TRAINED |

**Theorem E (target, CONJECTURE).** There is a sufficient statistic m of FIXED size (not growing with the number of
images) and an exact formula F such that, for every mix D, training on D started at F(m(D)) cannot lower KL_D.
**Corollary E.1 (generalist; follows from E).** Call a mix D "uninformative" when m(D) = m_theta. Training on an
uninformative mix cannot improve G. On any other mix, training's gain over G is the value of the information
m(D) - m_theta, and the developer build D_n(G) recovers it from n images.
**Remark 13.3 (when E has content).** m(D) = D itself is always sufficient, so E is empty without the size condition.
E has content when m is small: fixed size, and a few hundred images estimate it to within the score's noise
(statement E-small, check R1).
**Remark 13.4 (scope).** For the generalist, E.1 can only be proved in an objective defined from the weights. On real
images the error term is Delta (L8). E.1 is stated in that form.

**Ledger (Part IV).**
| ID | Statement | Depends on | Status |
|---|---|---|---|
| E | Theorem E | E-LS, 13.6, E-small | CONJECTURE |
| E.1 | generalist corollary | E, theta-stat | CONJECTURE |
| E-LS | E for reader-input matching (level 0 below), m = S | Prop. 11.1 | PROVED (Prop. 13.5); Lean pending |
| E-KL0 | E for the true KL with m = S | E-LS | REFUTED (X-B3: training from F(S) gains .020). Reason (Prop. 7.5): the readers' gates; to verify (R2) |
| 13.6 | level 1 (gated) carries the largest share of the .020 | Prop. 13.5, Prop. 7.5 | CONJECTURE |
| E-small | m is small: n of a few hundred images estimates it within .003 | E-LS | OPEN (R1) |
| theta-stat | m_theta (weights) = m of the network's training mix | W7, R3i | OPEN (R4) |
| k3-mix | at k = 3 the coarse unit's read re-mix is part of m | 13.6 | OPEN (R5) |

## 14. The ladder of objectives: what m must contain

**Setup.** At k = 0 the family's 10 channels are removed and reader j (one of the 368 output channels of the four 4a
input convs, at one of the 14 x 14 positions) changes its pre-activation by -e_j, with
  e_j(x) = w_j . h_F(x) - DeltaW_j . h_R(x) - Delta-b_j,   z = (h_F, h_R, 1) in R^481,
where w_j = reader j's old weights on the family and h is the pooled 3b activity that reader j sees (for the pool
branch reader: the 3 x 3 max-pooled activity p, not x; Prop. 11.1's code used x for all four, an approximation corrected
here). Reader j's output changes by delta_j = relu(a_j) - relu(a_j - e_j), a_j its intact pre-activation. To second order
in delta, KL(intact || crushed) = 1/2 delta^T F_x delta, F_x = the Fisher of the logits pulled back to the reader outputs.

**The ladder.** Each level replaces part of the true objective by a simpler one.
| Level | delta replaced by | F_x replaced by | Objective | Statistic m |
|---|---|---|---|---|
| 0 | e (no gates) | constant | sum_j E[e_j^2] (Prop. 11.1) | S = E[z z^T] (481 x 481, one matrix) |
| 1 | g_j e_j, g_j = 1[a_j > 0] (first order in e) | constant diagonal | sum_j c_j E[g_j e_j^2] | S^(j) = E[g_j z z^T], one per reader channel |
| 2 | g_j e_j | per-image diagonal f_j(x) | sum_j E[f_j g_j e_j^2] | E[f_j g_j z z^T] per reader channel |
| 3 | g_j e_j | full F_x | couples readers and positions | fourth-order object |
| exact | delta | exact KL | KL_D | D itself |

**Proposition 13.5 (each of levels 0-2 is exact for its own objective).** At levels 0, 1 and 2 the objective splits
into one quadratic per reader channel, the statistic in the table is sufficient, and the minimiser is
  DeltaW_j, Delta-b_j = w_j Lambda_j,   Lambda_j = S^(j)_{F,R} (S^(j)_{R,R})^-1   (R = the 470 others + constant),
with S^(j) the level's weighted second moment (level 0: the same S for every j). The per-reader constant c_j cancels.
**Proof.** Reader j's parameters (DeltaW_j, Delta-b_j) appear only in reader j's term, so the objective is a sum of
independent problems. Each term is E[omega_j (w_j h_F - beta . z_R)^2] with a weight omega_j >= 0 (1, g_j or f_j g_j);
expanding the square gives w^T S^(j) w-type terms only, so it depends on D only through S^(j) (sufficiency). The
normal equations are w_j S^(j)_{F,R} = beta S^(j)_{R,R}, which gives the formula when S^(j)_{R,R} is invertible. A
positive c_j multiplies the whole term and does not move its minimiser. QED. (Lean: `quad_moment_suff`,
`gated_write_in_span`, section 14.1.)

**Consequence.** E-LS (Theorem E at level 0) is proved: training on the level-0 objective cannot improve F(S). The
X-B3 gain (.020) is therefore exactly the KL's departure from level 0. The ladder splits it:
  gap(F(m_0)) = [level 1 - level 0] + [level 2 - level 1] + [level 3 - level 2] + [exact - level 3].
**Statement 13.6 (CONJECTURE, verifies the reason given for E-KL0).** The gates (level 1) carry the largest share: the
level-1 build F(m_1) closes at least half of the .020, i.e. KL(F(m_1)) <= .055 at n = 2000. Reason: Prop. 7.5 showed
the readers' gates move the true write out of the span by +.17 at k = 1 (bench B1); at level 0 the formula also fits
reader inputs where the reader is off, which cannot matter for the output.

**Check R2 (`x2_gated.py`, dev-5000, MIX-DERIVED, 2000 train images).**
| Build | KL (se) | paired vs level 0 |
|---|---|---|
| level 0 (R3d) | .0647 (.0038) | 0 |
| level 0, pool reader fed its true input p | .0633 (.0038) | -.0014 (.0006) |
| level 1 (gated moments), n = 2000 | .0603 (.0035) | -.0043 (.0020) |
| level 1, n = 256 | .0628 (.0036) | -.0019 (.0020) |
| converged from-zero training (TRAINED) | .0564 | |
| training started at level 0 (TRAINED, best known) | .0446 | |
**Statement 13.6: REFUTED.** The gates close .004 of the .020 (about 20%), not half. Dependents: k3-mix SUSPENDED (it
assumed the gated formula is the right write for k = 3). The reason given for E-KL0 (Prop. 7.5, gates) is true but
small on bench B3.

**Loop step 7: candidate reasons, as statements.** Level 1 differs from the true KL in exactly three ways (section 14):
| ID | Reason | What it claims |
|---|---|---|
| R-a | per-image metric (level 2) | the output cares about each reader by a factor that varies strongly per image and position (sure images count ~0 in KL, P-I2/I3); level 1 weights all equally |
| R-b | coupling (level 3) | errors in different readers and positions add or cancel through later layers; a per-reader objective cannot see this |
| R-c | beyond second order | the errors are large (removal costs .664), so gate flips and the curvature of the KL matter |
**Loop step 8: one measurement that separates them (R2b).** For each build B, evaluate on 500 held-out train images:
  L2(B) = 1/2 E sum_u f_u e_u^2,   L3(B) = 1/2 E_x E_{y ~ p(.|x)} (sum_u grad_u log p_y . e_u)^2,
with u = (reader channel, position), e_u = the build's change in reader pre-activation, f_u = E_y (grad_u log p_y)^2 the
diagonal of the intact network's Fisher pulled back to the readers (the gate is inside the gradient). L3 is the exact
second-order Taylor term of the KL (Fisher identity; the labels y are drawn from the network itself, no true labels).
Also L3' with the exact post-ReLU change delta (gate flips at the readers, linear above).
Each reason predicts a different pattern on the pair (level-1 build, build trained from level 0):
| If | L2 ranks trained < level 1 | L3 ranks trained < level 1 | L3 tracks the KL ratio |
|---|---|---|---|
| R-a | yes | yes | yes |
| R-b | no | yes | yes |
| R-c | no or yes | no, or yes with a ratio far from the KL's | no (L3' closer) |

**Check R2b (`x2b_reason.py`, `x2b_exact.py`; 500 dev images outside dev-1000; L3 exact as 1/2 Var_{c~p}(J_c e),
J e by central difference, eps 1e-2 vs 1e-3 agree within 2-4%).**
| Build | KL | L2 | L3 | L3' | flip term L3' - L3 |
|---|---|---|---|---|---|
| removal | .658 | .039 | .736 | .741 | .005 |
| level 0 | .077 | .0092 | .080 | .087 | .007 |
| level 1 | .073 | .0079 | .060 | .078 | **.019** |
| trained from zero | .058 | .020 | .062 | .068 | .006 |
| trained from level 0 | .050 | .0123 | .055 | .060 | .006 |
| Paired difference | KL | L2 | L3 | L3' |
|---|---|---|---|---|
| trained from level 0 minus level 0 | -.0269 (.011) | +.0032 (.0004) | -.0256 (.013) | -.0270 (.012) |
| level 1 minus level 0 | -.0042 (.006) | -.0013 (.0002) | -.0206 (.006) | -.0087 (.005) |
**Statement 14.7 (verified reasons, loop step 8).**
1. (R-b, coupling.) Training's gain over level 0 is reproduced by the full second-order term (L3 -.0256 against KL
   -.0269) and has the WRONG sign in its diagonal part (L2 +.0032, 8 se). The trained build makes larger errors at
   each reader, arranged so that they cancel through later layers. R-a (per-image diagonal metric) is refuted.
2. (R-c, flips.) The level-1 build gains .021 to second order with linear gates, but only .004 in KL. Its flip term is
   .019 against .005-.007 for every other build. Level 1 puts error at positions where the reader is off; where that
   error is negative the reader turns on, and level 1 cannot see it.
Status: CHECKED (both). E-KL0's reason is restated: not the gates, but the coupling of errors through later layers.

**Proposition 14.8 (which coupling can move a write).** Let the objective be sum over readers j, j' of
M_R[j, j'] (U_j - B_j Z)^T M_P (U_j' - B_j' Z), with every reader's re-routing B_j free, the SAME regressors Z (the 3b
activity at the samples/positions) for all readers, M_R invertible (reader metric) and M_P a positive definite metric
over samples and positions. Then the minimiser is B_j = U_j M_P Z^T (Z M_P Z^T)^-1 for every j: it does not depend on
M_R at all. **Proof.** The normal equations are sum_j' M_R[j, j'] (U_j' - B_j' Z) M_P Z^T = 0 for every j; M_R is
invertible, so (U_j - B_j Z) M_P Z^T = 0 for each j. This is `gated_write_in_span` with Dg := M_P (that lemma never
uses that Dg is diagonal). QED (PROVED-LEAN through `gated_write_in_span`).
**Consequence.** A coupling between DIFFERENT READERS can move the write only through its per-image or per-position
variation; a coupling between POSITIONS (later layers integrate over space: 3x3 and 5x5 convs, max-pools) moves it
directly. So the cheapest place for the coupling of 14.7.1 is the spatial metric M_P.

**Check R2c (`x2c_blocks.py`): blocks of L3, exact Jacobian rows for the top-10 classes (mass ~1.0; matches the FD L3).**
| Block of L3 | level 0 | trained from level 0 | difference |
|---|---|---|---|
| diagonal | .0092 | .0124 | +.0032 |
| same reader channel, all positions | .0296 | .0502 | +.0206 |
| same position, all readers | .0135 | .0153 | +.0018 |
| cross terms: different readers AND different positions | .0458 | .0014 | **-.0444** |
| full | .0797 | .0545 | -.0252 |
PREDICTION (the spatial block carries >= 60%) REFUTED. Training's gain is entirely in the cross terms, which almost
vanish for the trained build.
**Mathematical reason (loop step 7).** At level 0 every reader re-routes with the same Lambda (Prop. 11.1), so its error
is e_j(pos) = w_j . r(pos), with ONE 10-dimensional residual r = h_F - Lambda z_R shared by all readers. All 368 readers'
errors are fixed linear images of the same 10 signals, exactly the signals the later layers were trained to read from
the family. So the errors add up coherently downstream, through every reader and every position at once: a large cross
term. Level 0 minimises E|r|^2 with the identity weight on the 10 x 196 residual entries. The KL weights r by the
downstream sensitivity to the FAMILY's signal: g_c(x, pos) = W_F^T J_c(x, pos) in R^10 (J_c = gradient of logit c with
respect to the reader outputs). In the span the true second-order objective is
  L3_span(Lambda) = 1/2 E_x Var_{c ~ p} ( t_c(x) - <Lambda, A_c(x)> ),
  t_c = sum_pos g_c(pos) . h_F(pos),   A_c = sum_pos g_c(pos) z_R(pos)^T  (10 x 471).
**Proposition 14.9 (in-span coupled formula; closed form).** L3_span is a quadratic in Lambda (4710 numbers). Its
sufficient statistic is m_3s = (E_x Cov_c(vec A_c), E_x Cov_c(vec A_c, t_c)), fixed size (4710 x 4710 + 4710), and its
minimiser is the solution of one linear system: E Cov_c(vec A_c) vec Lambda = E Cov_c(vec A_c, t_c). No iteration, no
knob except the same tiny ridge as before. Proof: Var is a quadratic form; normal equations; `quad_moment_suff` with the
per-class weights p_c (centred) gives sufficiency. QED.
**Which sensitivity (from R-c).** With J taken at the readers' PRE-activations the gate is inside J, off readers get zero
weight, and the formula repeats level 1's flip failure. With J taken at the readers' OUTPUTS (gate-free) an off
reader's error is weighted as if it passed, as level 0 does (level 0's flip term .007). The formula uses the output J;
the pre-activation J is run once as the control that R-c predicts will fail.
**Statement 14.10 (CONJECTURE; prediction written before the run).** F(m_3s), n = 2000 (same images), output J:
(a) dev-5000 KL <= .052 (closes >= 60% of the .0633 -> .0446 gap from level 0p); (b) beats converged from-zero training
(.0564) by > 2 paired se; (c) its cross term drops below .015 and its flip term stays <= .010; (d) the gated-J control is
worse than (a) by > 2 se; (e) n = 256 within .004 of n = 2000.

**Check R3' (`x3_coupled.py`, dev-5000).** F(m_3s), n = 2000: output J .2486, pre-activation J .2433; n = 260: 93.8.
Statement 14.10 REFUTED (all parts that were scored).
**Loop step 8 (`x3b_diag.py`).** No bug: the statistic predicts the L3 difference on the 2000 images exactly (-.0754 both
ways). On those images the coupled build has L3 = .0000 but KL .139; on dev L3 .212 and KL .209. M has 309 eigenvalues
above 1e-3 of the largest (of 4710).
**Statement 14.11 (verified reason).** (1) Over-fitting: one image constrains L3_span by at most K - 1 = 9 numbers (its
logit variance under the network's own uncertainty; sure images give ~0, P-I2), against 196 x 10 for level 0. At n = 2000
the solve interpolates its images. (2) Second-order breakdown: at the interpolating solution the true KL (.139) is far
from its second-order term (0); the step is outside the region where the Taylor model holds (|Lambda_rest| x12.6).
So E-small FAILS for m_3s, and m_3s is the right object only where the step stays small.
**Repair (Statement 14.12).** By the standing rule (the formula is defined on the POPULATION moments of the mix;
estimation is a separate question), estimate m_3s on all 35000 train images. This separates (1) from (2).

**Check R3'' (`x3c_population.py`, dev-5000, MIX-DERIVED on all 35000 train images; one linear solve, no gradient step).**
| Build | KL (se) | paired vs converged from-zero training (.0564) | paired vs training from level 0 (.0446) |
|---|---|---|---|
| level 0, pool fix (n = 2000) | .0633 (.0038) | | |
| coupled, output J, n = 8000 | .1019 (.0049) | +.0455 (.0045) | |
| coupled, output J, n = 35000 | **.0552 (.0027)** | **-.0012 (.0026)** | +.0106 (.0021) |
| coupled, pre-activation J, n = 35000 | .0561 (.0027) | -.0004 (.0025) | +.0115 (.0021) |
**Statement 14.12 (CHECKED, partly).** With the statistic estimated on the whole train split, the in-span coupled
formula ties a converged 5500-step training run from zero, with one linear solve. So the from-zero training's whole gain
over level 0 is explained by Prop. 14.9's objective: coupling of the family's lost signal through later layers, weighted
by the network's own uncertainty. 14.10(d) (the gated-J control is worse by > 2 se) is REFUTED: pre-activation J was slightly better at n = 2000
(.2433 vs .2486) and tied at n = 35000 (.0561 vs .0552). R-c does not decide between the two sensitivities here.
**What it does NOT give.** (1) E-small FAILS at level 3: 8000 images are worse than level 0, and the curve is still
steep at 35000. The coupling statistic is data-hungry because each image constrains it by <= 9 numbers (14.11).
(2) It is not the generalist: it uses 35000 real images and the intact network's gradients on them. (3) It is .0106
above the best known network (training started at level 0). Candidates for that rest: the statistic is not yet at its
population value; the reader-specific (out-of-span) part, Prop. 7.5; the second-order breakdown, 14.11(2).

## 15. Ledger, rewritten after the R1-R3'' cycles (2026-10-24)

| ID | Statement | Depends on | Status |
|---|---|---|---|
| E | Theorem E (fixed small m, exact F) | E-LS, 14.9, E-small | CONJECTURE; at level 0 PROVED for its own objective; for the KL the obstacle is now named (coupling, data-hungry) |
| E.1 | generalist corollary | E, theta-stat | CONJECTURE |
| E-LS | E for reader-input matching, m = S | Prop. 11.1 | PROVED (Prop. 13.5; Lean `quad_moment_suff`, `gated_write_in_span`) |
| E-small, level 0 | S estimated by 64 images within .003 | E-LS | CHECKED (R1) |
| E-KL0 | E for the KL with m = S | E-LS | REFUTED (gap .020) |
| 13.6 | the gates carry most of the gap | Prop. 7.5 | REFUTED (R2: .004 of .020) |
| 14.7 | verified reasons: training's gain = coupling (R-b); level 1 fails by flips (R-c) | R2b | CHECKED |
| 14.8 | a reader-reader coupling cannot move the write unless it varies per image | - | PROVED-LEAN |
| R2c | the coupling sits in cross terms (different readers AND positions), not in blocks | 14.7 | CHECKED |
| 14.9 | in-span coupled formula: one linear solve, statistic m_3s (4710 x 4710) | 14.7 | PROVED (closed form) |
| 14.10 | F(m_3s) at n = 2000 closes >= 60% | 14.9 | REFUTED (over-fits: .249) |
| 14.11 | verified reason: <= 9 constraints per image (interpolation) + second-order breakdown | 14.10 | CHECKED (x3b_diag) |
| 14.12 | population-scale F(m_3s) ties from-zero training | 14.9, 14.11 | CHECKED (.0552 vs .0564) |
| E-small, level 3 | m_3s estimated by a few hundred images | 14.9 | REFUTED (8000 images worse than level 0) |
| theta-stat | m_theta from weights = m of the training mix | W7, W5 | OPEN; now the coupling metric is the object to get from the weights |
| k3-mix | k = 3 coarse unit read re-mix | 13.6 | SUSPENDED (13.6 refuted) |

**Where the target theorem stands (k = 0, bench B3).**
| Build | KL | Labels |
|---|---|---|
| removal | .664 | |
| weights only (R3i) | .177 | DERIVED (+ network-made means) |
| level 0, 64 images | .067 | MIX-DERIVED |
| coupled, 35000 images, one solve | .0552 | MIX-DERIVED |
| from-zero training, converged | .0564 | TRAINED (one imperfect example) |
| training started at level 0 | .0446 | TRAINED (best known) |
The exactness gap of the best formula is .0106. The object that would close most of it is now precise: the coupling
metric of Prop. 14.9 (how the later layers combine errors in the family's 10 signals across positions, weighted by the
network's uncertainty). Images estimate it badly (<= 9 numbers per image). The weights carry this kind of metric in part
(W5: the next layer's weights predict the real-image Fisher's off-diagonal structure, .81-.87). That is the next
statement to state, for both scenarios.

**Check X-coupled (the exactness test on the best formula).** Training started at the 35000-image coupled build (X-B3
recipe: 2000 steps, lr .003, Adam, early stop on dev-1000) ends at .0542: -.0010 (paired se .0015) from its start, and
+.0096 (se .0016) above the best known network. The training curve leaves the start at once (dev-1000 .057 -> .12 by
step 500) and comes back (.058 at step 1625). PREDICTION (gain > 2 se, end .043-.047): FAILED.
**Statement 15.1 (what the exactness test measures; CHECKED here).** "Training started at X gains nothing" is not
"X is optimal". It also holds when X sits in a narrow region that the optimizer's first steps leave: Adam moves every
coordinate by about lr, and the coupled build carries large coefficients (|Lambda_rest| x12.6 against level 0) in
directions the output barely feels (14.11), so the first steps break it. Training from level 0 (a broad, coercive point)
reached .0446; training from the coupled build did not. **Consequence for Theorem E.** The exactness test must be read
together with the best known network: a formula is exact only if training from it gains nothing AND no network of the
same structure is known to be better. On that combined test the coupled formula is .0096 short. Theorem E stays
CONJECTURE. Ledger: add 15.1 CHECKED.

---

# Part V: route A, the coupling metric from the network (Shonu's decision, 2026-10-24)

## 16. Is the coupling a property of the network or of each image?

**Rewrite of the in-span objective.** With r(x, p) = h_F(x, p) - Lambda z_R(x, p) in R^10 (the residual of the family's
re-expression at position p) and g_c(x, p) = W_F^T J_c(x, p),
  L3_span(Lambda) = 1/2 E_x sum_{p, q} r(x, p)^T K_x(p, q) r(x, q),   K_x(p, q) = Cov_{c ~ p(.|x)}(g_c(p), g_c(q)),
a 1960 x 1960 metric on the family's 10 signals at 196 positions, one per image.
**Statement 16.1 (factorisation; CONJECTURE).** Replacing K_x by its average Kbar = E_x K_x keeps most of the coupling
gain:
  L3bar(Lambda) = 1/2 E_x sum_{p, q} r(x, p)^T Kbar(p, q) r(x, q).
**Why it matters.** If 16.1 holds, the object splits into two pieces: Kbar, which depends on the downstream network
only (how later layers combine the family's signals across positions, averaged), and the activity cross-moments
E[z(p) z(q)^T], which every image constrains at all 196 x 196 position pairs (no 9-per-image limit). Route A then
needs Kbar from the weights. If 16.1 fails, the coupling lives in per-image variation (which images the network is
unsure about) and route A must derive a per-image object instead.
**Proposition 16.2 (closed form of the factorised objective).** Let H[(f, a), (g, b)] = E_x sum_{p, q} Kbar_{fg}(p, q)
z_a(x, p) z_b(x, q) over the 481 activities (480 channels + constant). The minimiser of L3bar is the solution of
  sum_g H[(f, R), (g, R)] Lambda_g = sum_g H[(f, R), (g, F_g)]   for every f   (one 4710-unknown system).
Proof: L3bar is a quadratic in Lambda with this Hessian and gradient (expand r); `quad_moment_suff` gives sufficiency
for the statistic (Kbar, H). QED.
**Corollary 16.3 (from Prop. 14.8).** If Kbar = K_F (x) K_P (a family-channel metric times a spatial metric), K_F drops
out and only the spatial metric K_P moves the write. Measured below: how close Kbar is to such a product.
**Check R5' (prediction before the run).** Kbar from 2000 train images (FITTED microscope; images only CHECK 16.1),
activity moments from the same 2000 and from the first 256.
| Quantity | Prediction |
|---|---|
| build, n = 2000 | <= .058 (keeps >= 60% of the 35000-image gain, .0633 -> .0552) |
| build, n = 256 | within .004 of n = 2000 (no over-fit: every image constrains every position pair) |
| Kbar nearest Kronecker product K_F (x) K_P | explains >= 70% of Kbar's energy |
If the build is above .0633 (worse than level 0p), 16.1 is REFUTED and the per-image variation carries the coupling.

**Statement 16.4 (what the coupling should look like in images, if the network is clean; CONJECTURE).** Shonu: the
cancellation must be an interpretable object. The essay's curve circuit gives one: later layers build circles and loops
by summing curve detectors whose arcs lie on ONE circle (co-circular curves at different positions, each pointing at
the common centre). This is the "association field" of contour integration (Field, Hayes, Hess 1993). So:
  Kbar couples curve member f at position p with member g at position q exactly when f and g are co-circular: there is
  a circle of the family's preferred radius (~60 px) through p and q with f's arc at p and g's arc at q, both
  pointing at its centre.
Then the cross term of level 0 has a plain meaning: level 0 loses the curve signal around a ring with errors that all
line up with one circle, and the later circle and loop readers add them up. Training breaks that alignment.
**Predictions (written before any picture).**
| # | Prediction | Null it beats |
|---|---|---|
| a | in the offset-averaged Kbar, for member f the strongest partner g at offset d has the orientation of the common-centre rule (circle of radius 60 px through p, centre in f's direction), circular correlation >= .5 over the coupled offsets | "same orientation" (parallel / collinear contours): lower correlation |
| b | the coupling energy of member f sits on a ring through p bending toward f's centre direction, not on a symmetric blob | isotropic blob |
| c | Kbar is NOT close to a Kronecker product (< 50% energy), because the coupled orientation depends on the offset. (This contradicts the R5' table line "Kronecker >= 70%"; the measurement decides.) | |
| d | the dev images with the largest level-0 cross term show round things (wheels, dials, eyes, loops, rims); their per-position contribution maps light up along the ring | textured or straight-edge scenes |

**Check R5' parts 1-3 (`x5_kbar.py`, `x5b_offsets.py`, `x5c_maps.py`; Kbar from 2000 train images, FITTED microscope).**
| Prediction | Result | Verdict |
|---|---|---|
| (a) partner orientation follows the common-centre rule | sharp offsets (rules differ > 60 deg): cos .696 vs .211 for "same orientation"; 9 of 10 members | HOLDS |
| (b) coupling energy on a ring bending to the member's centre | the energy is the SAME horizontal band (~±48 px wide, ±16 px tall) for all 10 members (picture `cache/x5_maps.png`) | REFUTED |
| (c) not Kronecker (< 50%) / R5' line (>= 70%) | .669 | both FAIL |
**Statement 16.5 (what the picture shows).** Kbar has two parts. (1) A spatial envelope shared by all members
(the Kronecker part, 67%): a horizontal band. (2) On top of it, WHICH member is coupled to which at each offset follows
co-circularity: the essay's curves-to-circles composition, i.e. contour integration (association field). Part (2) is the
interpretable, clean object Shonu asked for. Part (1) needs a reason: why horizontal, and why the same for all members?
**Candidate reason (loop step 7).** The envelope is the image statistics, not the network: natural photos are wider
than tall in their content (horizon, bodies, vehicles), so the network's uncertain logits depend on horizontally
adjacent positions together. **Check (step 8):** recompute Kbar on the same images rotated by 90 deg. If the reason is
right, the band turns vertical. If it stays horizontal, the envelope comes from the network (its weights).
Step 8 result: on 500 images rotated by 90 deg the horizontal/vertical ratio of the envelope falls from 1.41 to 1.15 but
stays horizontal in the network's frame. "Image statistics alone" is REFUTED. The envelope is part photos, part network.

**Check R5' part 4 (`x5e_images.py`, picture `cache/x5_images.png`): which images carry the cross term.** The top 5%
of dev images carry 95.5% of level 0's cross term. Of the top 12: five clearly round objects (melon halves, washbasin,
teddy bear, combination-lock dial, bathtub), two curved shapes (bow tie, snake on sand), five others. Prediction (>= 7)
BORDERLINE. On the round objects, level 0's per-position error contribution runs ALONG the curved contour (rind, rim, dial
ring); the network trained from level 0 removes it there (dial 2.47 -> .15; melon 11.1 -> 2.9; basin 5.1 -> 2.2).
**Statement 16.6 (the image-level meaning; CHECKED in parts (a) and (d)).** The coupling is contour integration. The
later layers read the curve family as whole curved contours: rims, dials, rinds, lobes. Level 0 re-expresses each
missing curve neuron locally from its neighbours. On a round object, all these small local errors point the same way
along the contour, so the readers of round things add them up. This is rare (5% of images carry 95% of it) and often
decisive for the class (level-0 KL 8.0 on the bow tie). Training learns to spread the errors so they do not line up
along contours. A clean network predicts exactly this: the composition curves -> circles is a known, interpretable
circuit (the essay; Olah et al. 2020), and the sharp test (a) shows Kbar follows its co-circular geometry.

**Check R5' part 5 (`x5f_factor.py`, dev-5000; Kbar FITTED from 2000 images, activity moments from n images).**
| Build | KL (se) | paired vs converged from-zero (.0564) | paired vs best known (.0446) |
|---|---|---|---|
| factorised, n = 2000 | **.0491 (.0029)** | **-.0074 (.0023)** | +.0045 (.0020) |
| factorised, n = 256 | .0534 (.0030) | -.0030 (.0024) | +.0088 (.0021) |
| per-image coupled, n = 35000 | .0552 (.0027) | -.0012 (.0026) | +.0106 (.0021) |
**Statement 16.1: CHECKED, stronger than predicted** (<= .058 HOLDS; n = 256 within .004 FAILS by .0003). Averaging the
metric over images keeps the coupling and removes the per-image over-fit. |DeltaW| is 1.33x level 0's (per-image
build: 2.80x): a broad point. So the object of route A is Kbar: a fixed 3920 x 3920 metric on the family's signals at
196 positions, made of a shared spatial envelope (67%) and a co-circular orientation rule.

**Next statement (route A proper, written before any run).** Statement 16.7: the co-circular part of Kbar is fixed by the
first downstream spatial kernels. Weights only: for each 4a unit u behind a 3x3 or 5x5 conv, its effective kernel on the
family, k_u(f, d) = sum over its bottleneck inputs of (spatial kernel at offset d) x (bottleneck weight on the family
channel's column W_F). Then M(f@p, g@q) = sum_u sum_c k_u(f, c - p) k_u(g, c - q) (u at every centre c). PREDICTION:
the sharp co-circularity test on M gives cos >= .5 (vs same-orientation lower), and M's offset pattern correlates with
Kbar's >= .5. If it holds, the next build replaces Kbar by M (plus the envelope) with no image at all.

**Check 16.7 (`x5g_weights.py`, weights only).** M (the Gram of 4a's 3x3 / 5x5 effective kernels on the family) against
Kbar: offset-pattern corr .718 (.699 without d = 0). Sharp co-circularity test: M .780 vs same-orientation .117, 10 of 10
members (Kbar: .802 vs .210). **Statement 16.7: CHECKED.** The co-circular part of the coupling is written in the
weights of the next layer: the contour-integration circuit, read directly from the weights, with no image.

**Check 16.8 (`x5h_wbuild.py`).** Factorised formula with the WEIGHTS metric (Gram of all four 4a branches' effective
kernels on the family; no image in the metric; Lambda does not depend on the metric's scale, so no knob). Activity
moments from n images (developer form).
| Build | metric | KL (se) | paired vs level 0p |
|---|---|---|---|
| level 0p | none (identity) | .0633 | 0 |
| weights metric, n = 2000 | 4a weights | **.0570 (.0033)** | **-.0063 (.0009)** |
| weights metric, n = 256 | 4a weights | .0594 (.0033) | -.0040 (.0010) |
| Kbar-factorised, n = 2000 | images (FITTED) | .0491 (.0029) | -.0142 (.0013) |
**Statement 16.8: CHECKED.** One layer of weights gives 44% of the image metric's gain. What it lacks, by
construction: (1) the shared horizontal envelope, which Kbar shows is long-range (±3 cells and more) and therefore made
by deeper layers; (2) the weighting by how much the OUTPUT cares (Kbar carries the network's uncertainty; M weights every
4a unit equally). Both are the same object one or more layers further up: the next statement composes the metric through
4b (and up), the way R3i composed the read through the bottleneck.

**Check X-factor (exactness test on the factorised build).** Training started at it (X-B3 recipe) ends at **.0430**
(se .0021): -.0061 (se .0019) below its start, and -.0016 (se .0006) below the previous best known network (.0446).
The best known k = 0 network now comes from a formula start. Predictions: gain > 2 se HOLDS; ends <= .0446 HOLDS;
"no kick-out" FAILS (dev-1000 .056 -> .103 at step 250).
**Correction to Statement 15.1 (REFUTED as stated).** The old run from level 0 (`logs/b3_fstart.log`) shows the same
kick (.076 -> .113 at step 250). The kick is the optimizer's fresh Adam steps at lr .003, on every start, not a sign of a
sharp formula. So the X-B3 recipe is not a local test: it leaves the start and re-descends, and where it ends depends on
where it started (per-image coupled start .0542, level 0 start .0446, factorised start .0430). What survives of 15.1:
"training from X gains nothing" does not prove X optimal (the coupled start showed it). **Statement 15.2 (instrument).**
A local exactness test needs steps that stay inside the region of the start: a small step size with a warm-up, or the
gradient norm at the start. To be fixed before the next exactness claim.

## 17. State of route A (2026-10-24)
| Build (k = 0, bench B3) | metric source | KL | label |
|---|---|---|---|
| level 0p | none | .0633 | MIX-DERIVED (2000 images) |
| weights metric (16.8) | 4a weights | .0570 | metric DERIVED; activity MIX-DERIVED |
| Kbar-factorised (16.1) | 2000 images | .0491 | FITTED metric (microscope) |
| converged from-zero training | | .0564 | TRAINED |
| trained from the factorised build | | **.0430** | TRAINED (best known) |
The open object is the part of Kbar that one layer of weights misses (56% of its gain): the long-range shared envelope
and the output weighting. Next statement: compose the weights metric through 4b and up.

**Check X-factor-local (Statement 15.2's instrument: lr 3e-4, 500 steps).** No kick-out. Training started at the
factorised build: .0491 -> **.0404** (-.0087, se .0013). The same recipe started at level 0: .0647 -> .0452. Paired, the
formula start ends .0048 lower (se .0006). **Statement 17.1 (CHECKED).** (1) The factorised formula has a real local
defect of about .009: Theorem E does not hold for it yet. (2) It is a far better start than level 0 under an identical
recipe, so it carries information that training from level 0 does not recover in 500 small steps. (3) The earlier
lr .003 runs were hurt by the kick; the best known k = 0 network is now .0404 (TRAINED from the formula). All earlier
"training from X" numbers in Parts III-IV used the kicking recipe and are rulers of that recipe only.

---

# Part VI: circuits (Shonu's decision, 2026-10-24, plan `okay-last-session-landed-velvet-wren.md`)

**Why a new object.** Part V showed that the formula's error is not per reader: level 0 fixes each reader on its own,
training arranges errors so they cancel through later layers (14.7, R2c), and the image meaning is contour integration
(16.6). The essay names this level ("Edge Featurizers"): edge motifs of ordinary neurons "predicted where ablation would
matter much better than comparable groupings of activations", and "a representation matters only through what reads
it". So the unit here is an EDGE between the network's own families. Nothing is split into new features.

**Two gaps (k = 0, bench B3, dev-5000).**
| Gap | From | To | Size |
|---|---|---|---|
| A | weights-metric build (16.8) .0570 | Kbar build (16.1) .0491 | .0079 (Shonu's "56%"); corrected in section 21 to .0061 from .0552 plus a .0018 pool-input fix |
| B | Kbar build .0491 | best known (local training from it, 17.1) .0404 | .0087 |

## 18. Definitions and the target theorem

**Setting.** Layers after 3b: readers R (the four 4a input convs, outputs post-ReLU r in R^{368 x 196}), then the 4a
output H in R^{508 x 196}: H = (r_1x1, relu(C3 * r_3b), relu(C5 * r_5b), r_pool), C3, C5 the 3x3 / 5x5 spatial convs.
Family signals s = (h_F, p_F) in R^{2 x 10 x 196} (x-input and max-pooled input of the pool reader).

**Definition 18.1 (edge).** An edge from family A (layer l) to family B (layer l + 1) is the pair (W_BA, D_B): W_BA the
linear map from A's signals at all positions to B's pre-activations (all linear parts between them composed; Voss et
al. 2021 "expanded weights"), D_B(x) the diagonal gate of B on image x.
**Definition 18.2 (circuit).** A circuit from the family to layer L is a chain of edges; its operator on image x is
A_L(x) = D_L W_L ... D_1 W_1 (the Jacobian of layer L's outputs with respect to the family's signals, ReLUs fixed at x).
**Definition 18.3 (circuit metric at layer L).** Sigma_L(x) = Cov_{c ~ p(.|x)}(J^L_c(x)), J^L_c the gradient of logit c
with respect to layer L's outputs; Sigma_L = E_x Sigma_L(x). At the logits Sigma_out(x) = Cov_c(e_c) = diag(p) - p p^T.
**Definition 18.4 (metric transport).** K^L := E_x A_L(x)^T Sigma_L(x) A_L(x): the metric on the family's signals seen
through the circuit up to layer L.

**Proposition 18.5 (transport identity).** For every L between the readers and the logits, K^L = Kbar.
Proof. Chain rule: J^fam_c(x) = A_L(x)^T J^L_c(x) for every c (ReLUs fixed at x, a.e. exact). Cov_c is quadratic, so
Cov_c(A^T J_c) = A^T Cov_c(J_c) A. Average over x. QED. (Exact; the check I1 verifies the code and the care points:
J at reader OUTPUTS, no gate between the 1x1 / pool readers and H, the pool reader's max-pooled input.)
**Consequence.** Every approximation of Kbar is a named replacement inside 18.5: the gates D by their firing rates, the
per-image metric Sigma_L(x) by its average, the joint average by a product of averages (independence across layers),
the metric at L by its diagonal. Each replacement is a ledger statement with a check.

**Postulates of a clean circuit (each testable).**
| ID | Postulate | Check |
|---|---|---|
| P1 | equivariance: an edge between two orientation families depends on relative angle and the offset in the target's frame (Olah et al. 2020); few angle harmonics | harmonic content of W and of Kbar's angle blocks |
| P2 | an edge opens when its feature is present: co-opening is structured | I3: which downstream units carry the cross term on which images |
| P3 | circuit independence: E_x A^T Sigma(x) A ~ E[A]^T E[Sigma] E[A] layer by layer (gates independent of the metric above) | A3. K-FAC / mean-field assume it; 16.6 (5% of images carry 95%) argues against it |

**Theorem C (target, CONJECTURE).**
1. (identity) Prop. 18.5.
2. (truncation) A circuit metric from the weights plus one gate statistic per edge, carried a small fixed number of
   layers, gives a build within 2 paired se of the Kbar build (closes Gap A).
3. (exactness) A build from the full circuit objective, plus the parts Gap B's checks name, is not improved by local
   training (closes Gap B).
**Named open object.** The gate statistic per edge (firing rates and their co-variation with the metric above) is the
same object as L6 and 12.4''. Fallback order if the weights cannot give it: weights, network-made inputs, a natural-image
law only with Shonu's OK.

**Ledger (Part VI).**
| ID | Statement | Depends on | Status |
|---|---|---|---|
| C | Theorem C | 18.5, A1-A3, B1-B3 | CONJECTURE |
| 18.5 | transport identity | chain rule | PROVED; check I1 |
| I2 | Kbar's top eigenmodes are contours (arcs / rings) and carry level 0's cross term | 16.6 | OPEN |
| I3 | a few clean downstream units carry the cross term | P2 | OPEN |
| A1 | Gap A sits in the per-4a-unit importance (diagonal of Sigma_4a) | 18.5 | OPEN |
| A2 | Gap A sits in the coupling between 4a units (off-diagonal of Sigma_4a) | 18.5 | OPEN |
| A3 | P3 at 4a: firing-rate gates cost < 2 se | 18.5 | OPEN |
| B1 | split of training's local move: shared re-routing / reader-specific / biases | Prop. 11.1, 7.5 | OPEN |
| B2 | the .0087 appears at second order | 14.7 instrument | OPEN |
| B3 | the readers training moves are clean circle / loop readers | B1 | OPEN |

## 19. First checks of Part VI

**Check I1 (`c1_split4a.py`).** Prop. 18.5 holds in the code: transport vs direct, relative error 3.8e-7 on 20 images
(float32); the 2000-image transport reproduces the cached Kbar to 2.0e-7. CHECKED.

**Check I2 (`c2_modes.py`, `cache/c2_modes.png`).** PREDICTION (top modes are arcs / rings; top 3 carry > 50% of level
0p's cost) REFUTED. Top 3 carry .166, top 20 .456, 108 modes for 80%. The top modes are low spatial harmonics of a
CENTRED window (blob, top / bottom, left / right, centre-surround), each mixing all ten members. Kbar's per-position
energy is a centred blob (corners 5-20x weaker); the 4a weights metric M is flat.
**Statement 19.1 (reason, CHECKED by the energy map).** Kbar = (where the output looks: a centred spatial window, plus
its low harmonics) x (the co-circular orientation rule of 16.7). The contour geometry is real but rides on the window, so
eigenmodes sort by the window's harmonics, not by contours. Prediction it makes and the old statement did not: the
per-position energy map is centred (seen) and M lacks it (seen). Whether the window carries Gap A: A1-A3.

**Check B1 (`c5_gapb.py`).** Training's local move S -> T: in-span part (one shared Delta-Lambda, least squares over all
368 readers) has 8% of |Delta-W|^2 and moves KL by -.0002 (se .0002). The reader-specific remainder has 92% and moves
.0491 -> .0407 (-.0084, se .0013; T: -.0087). Bias change: 0. PREDICTION (remainder >= half) HOLDS, strongly.
**Check B2 (same script, 500 x2b images).** S -> T: KL -.0111, L3 -.0116, L3' -.0117 (se ~.006). S -> S+rem: KL -.0110,
L3 -.0130. PREDICTION (L3 shows >= 60%) HOLDS: the whole of Gap B is second order. Flip term unchanged.
**Check B4 (`c6_branch.py`).** One Delta-Lambda per branch explains .278 of the move (random 4-way split .227 +- .015);
S + per-branch part .0462 (a third of the gain). PREDICTION (>= 70%, <= .0425) REFUTED. Exception: inside the 5x5
bottleneck branch (16 readers, 33% of the move) one Lambda explains 60%. The remainder is moderately low rank (top 1
singular direction .31, top 10 .66, 25 for 80%).

**Statement 19.2 (what Gap B is; CHECKED by B1 + B2).** Gap B is the in-span restriction of the factorised formula.
Prop. 11.1, 14.9 and 16.2 all set every reader's re-routing to w_j Lambda with ONE shared Lambda. The second-order
objective itself (no flips, no higher order) prefers reader-specific re-routings, and training finds them.
**Proposition 19.3 (when the restriction is free; converse of Prop. 14.8).** For L(B) = 1/2 E sum e^T K e with
e_j(p) = w_j . h_F(p) - B_j . z(p), K a metric over (reader, position) and the SAME regressors z for all readers: if K =
M_R (x) M_P (Kronecker, M_R invertible) the minimiser is B_j = w_j Lambda (Prop. 14.8). If not, the gradient of L at the
best in-span point has a component outside the span, unless K's non-Kronecker part is orthogonal to the residual.
Proof of the first part: Prop. 14.8. Second part: at the in-span optimum the normal equations hold only after
projection onto the span (W_F^T applied); the full gradient is sum_{j', p, q} K[(j,p),(j',q)] w_j' E[z(p) r(q)^T], which
equals w_j G for one G for every j only when K[(j,p),(j',q)] = M_R[j,j'] M_P[p,q]. QED.
**Circuit reading.** Readers are different EDGES into the rest of the network: the 1x1 and pool readers are 4a outputs
themselves; the bottleneck readers feed a 3x3 or a 5x5 conv. So each reader's error is coupled to others with its own
spatial pattern: K is not Kronecker. The circuit metric must be kept at READER level, not projected onto the family.
B4 shows that "same branch" is not the right grouping (except for the 5x5 branch).

**Statement B5 (CONJECTURE; next check).** Solve the SAME averaged second-order objective with a reader-level metric
(FITTED from images: offset-averaged reader-reader metric K_jj'(d) for |d| <= 6 cells, times a separable spatial window
w(p) from its per-position energy) over all per-reader re-routings (one linear system, solved by conjugate gradients;
368 x 471 unknowns; Hessian sum_d K(d) (x) Y(d), Y(d) = offset activity moments). Predictions: (a) its in-span
restriction lands within .003 of the Kbar build (.0491): calibrates the offset + window approximation; (b) the
per-reader minimiser reaches <= .0430 (closes >= 70% of Gap B), with no gradient step.

## 20. Gap A: the circuit continues past 4a

**Check A1-A3 (`c1_split4a.py`, FITTED microscopes, dev-5000).** K = E A^T Sigma A at 4a, with:
| Var | gates (3x3 / 5x5 branch) | Sigma at 4a outputs | KL | vs Kbar build (paired se) |
|---|---|---|---|---|
| a | none | I | .0552 | +.0062 (.0006) |
| b | firing rate | I | .0574 | +.0083 |
| c | none | diag of E Sigma (per-unit, per-position importance) | .0557 | +.0066 |
| e | firing rate | diag of E Sigma | .0576 | +.0085 |
| f | exact per image | diag of Sigma(x) | .0564 | +.0074 |
| g | none | full Sigma(x) | **.0488** | **-.0003 (.0003)** |
| d | firing rate | full Sigma(x) | .0493 | +.0002 (.0002) |
**Statements.** A1 (per-unit importance carries Gap A) REFUTED: no diagonal metric beats M. A2 (the coupling between
4a units carries it) CHECKED: the full metric with no gates at all ties Kbar. A3 (P3 at the family -> 4a edge) CHECKED:
gates at their rates, or no gates, cost nothing. So the whole of Gap A is the OFF-DIAGONAL part of the 4a-level circuit
metric: how errors in different 4a units (and positions) combine further up. The edge family -> 4a is fully described
by its weights (no gate statistic needed); what is missing lies above.
**Corollary 20.1 (where the window lives).** Variant c contains the centred window of 19.1 (per-position importance)
and gains nothing. So the window alone is not the missing part; what matters is the coupling ACROSS units and positions,
which the window only modulates.
**Statement A4 (CONJECTURE; next check).** Because gates did not matter at the first edge, try the gate-free transport
all the way: K^(L) = A_L^T A_L with A_L the Jacobian of the LINEARISED network (ReLU -> identity, max-pool -> average,
biases off) from the family's signals to layer L, Sigma_L = I (logits: I - 11^T/n). Weights only. Predictions in the log.

**Check A4 (`c8_ladder.py`; DERIVED metric = linearised weights from the family to layer L, no gates, no image).**
| Depth L | KL | vs Kbar build (paired se) | effective rank of K | corner / centre energy |
|---|---|---|---|---|
| 4a (= M with exact pool input) | .0552 | +.0062 (.0006) | 1264 | ~.8 (borders) |
| 4b | .0512 | +.0021 (.0004) | 732 | .45 |
| 4d | **.0489** | **-.0001 (.0005)** | 210 | .24 |
| 5b | .0516 | +.0025 (.0008) | 24 | .06 |
| logits (I - 11^T/n) | .0579 | +.0088 (.0014) | 8 | .08 |
| Kbar (images, FITTED) | .0491 | 0 | 324 | ~.06 |
**Statement 20.2 (CHECKED, with a CHOSEN depth).** The weights of 4a-4d alone, composed without gates, give a coupling
metric whose build ties the image metric: Gap A closes with no image in the metric. The two pieces that 16.8 lacked
both come from composition: the long range (each layer widens the coupling) and the centred window (fewer paths reach
the border; the window sharpens with depth from the weights alone). The window is therefore mostly made by the network,
not by photo statistics (consistent with the 90-degree check of section 16).
**Statement 20.3 (why deeper is worse; reason, to verify).** Without gates the deep products collapse onto a few
directions (effective rank 1264 -> 8). In the real network each image opens a different gate pattern, and averaging
over patterns adds an independent "gate noise" term at every layer: E[D M D] = (r r^T) o M + diag(r (1 - r)) o diag(M)
for independent Bernoulli gates with rates r. That term keeps the rank up. At 4a it is small (A3); it accumulates with
depth. Prediction only this reason makes: the image-gated metric E_x A_x^T A_x at 5b (true gates per image, Sigma = I)
has effective rank far above the gate-free 24 (near Kbar's hundreds).
**Consequence for Theorem C.2.** With the Bernoulli transport the depth stops being a knob: carry the metric to the
logits. The rates r are the named open object (L6 / 12.4''). Until then, 4d is a CHOSEN depth.
**Check 20.3 (`c9_gaterank.py`, 8 images).** True-gated E_x A_x^T A_x at 5b (Sigma = I): effective rank 1566-1931 per
image, 3223 for the 8-image average, against 24 gate-free. The reason is VERIFIED. Deep composition needs the gate
statistics; shallow composition (to 4d) does not. This is where the named open object (L6 / 12.4'') enters the circuit
theory, and now its role is precise: it sets how far the metric can be carried before it collapses.

**Check (band, cached tensors).** Horizontal / vertical neighbour energy: Kbar 1.40, Kbar on rotated images 1.21,
weights-only metric 4a 1.01, 4b 1.54, 4d 1.28. The band of 16.5 is written in the weights of 4b and up. It is a
separate statement from the window (20.2): the band is the SHAPE of the coupling between neighbours, the window is
WHERE on the image the coupling is strong. Both come from the weights.

## 21. Ledger of Part VI, rewritten (after 11 checks)

**Gap bookkeeping, corrected.** The one-layer weights metric with the pool reader fed its exact (max-pooled) input is
.0552, not 16.8's .0570 (which spread the pool reader over 3x3 of x). So: pool-input fix .0570 -> .0552 (.0018);
Gap A = .0552 -> .0491 (.0061); Gap B = .0491 -> .0404 (.0087).
| ID | Statement | Status |
|---|---|---|
| 18.5 | transport identity | PROVED; CHECKED (2e-7) |
| I2 | top Kbar modes are contours | REFUTED; reason 19.1 (centred window x orientation rule) CHECKED |
| I3 | a few clean downstream units carry the coupling | OPEN (not run) |
| A1 | Gap A = per-unit importance | REFUTED (diagonal metrics .055-.058) |
| A2 | Gap A = coupling between 4a units | CHECKED (full Sigma, no gates: .0488) |
| A3 | P3 at the family -> 4a edge: gates cost nothing | CHECKED |
| A4 / 20.2 | gate-free weights composed to 4d close Gap A | CHECKED (.0489 vs .0491), depth CHOSEN |
| 20.3 | deeper is worse because gate-free products collapse; gates keep rank | reason VERIFIED (rank 3223 vs 24) |
| band | the horizontal band is in the weights (4b+) | CHECKED |
| B1 | Gap B = reader-specific part of training's move | CHECKED (.0084 of .0087) |
| B2 | Gap B is second order | CHECKED (L3 -.0116 vs KL -.0111) |
| B4 | per-branch re-routing explains it | REFUTED (.278 vs random .227); 5x5 branch exception |
| 19.2 / 19.3 | Gap B = the in-span restriction; non-Kronecker reader metric moves the write | CHECKED (19.2); PROVED (19.3) |
| B5 (a) | stationary + window reader-level objective, in span, reproduces Kbar build | CHECKED (.0507, +.0016) |
| B5 (b) | its per-reader minimiser closes >= 70% of Gap B | running |
| X-w4d | local exactness test on the DERIVED-metric build | running |
**Labels, to keep every headline honest.** All builds here: metric DERIVED or FITTED as stated; activity moments
MIX-DERIVED (2000 images). The generalist still has no network-made activity moments (weights-only R3i .177). So no
statement in Part VI is "a weights-only build ties training".

## 22. Gap B is the image-level link between "circuit on" and "error there"

**Check B5 (`c7_reader.py`).** Stationary + window reader-level metric (FITTED, |d| <= 6), solved by CG: in span .0507
(within .0016 of the Kbar build: the approximation is admissible); per reader .0543. PREDICTION (b) (<= .0430) REFUTED.
**Loop step 7-8 (`c10_objective.py`, `c11_exactavg.py`).** Three reasons, one measurement each:
| Reason | Measurement | Verdict |
|---|---|---|
| R1 estimation noise in K | independent K estimate (cos .867 with the first) still ranks the per-reader solution best | REFUTED |
| R2 stationary + window model | exact averaged objective by samples (no model) also penalises training's move: S .0381, S+rem .0523 | REFUTED |
| R3 averaging the metric over images | per-image objective on the same images credits training: S .1059, S+rem .0931 | CHECKED |
**Statement 22.1 (verified reason for Gap B).** The factorised objective replaces E_x[K_x (x) z_x z_x^T] by
E_x[K_x] (x) E_x[z_x z_x^T]. Inside the family's span the difference is small (16.1: factorising even helped, by removing
over-fit). Outside the span it is decisive: training's reader-specific move is penalised by every averaged metric
(+34% to +37%) and credited by the per-image one (-12%). The missing object is the covariance across images between the
metric (how much the output cares, per reader and position) and the activity (where the family's signal is lost).
**Circuit reading.** On an image with a round object, the contour circuit is ON: its readers' gates are open, the output
is sensitive along the contour, and the curve family's signal is strong exactly there. On other images all three are off
together. So the per-image link is the circuit's own on/off state. This is the SAME object as 20.3 (deep composition
needs the per-image gate patterns) and as L6 / 12.4'': the joint statistics of which circuits are open on an image.
**Statement C.3 (candidate, CONJECTURE; not yet built).** Mixture over circuit states: E_x[K_x (x) z_x z_x^T] ~
sum_s pi_s K_s (x) S_s, where s runs over a few circuit states (which circuits are open), K_s = the metric transported
with state s's gates (weights + gate pattern), S_s = activity moments given s, pi_s = how often s occurs. Clean-network
postulate P2 says the states are few and named (e.g. "round object present"). Two predictions to test first: (a) with
s = the top 5% of images by level-0 cross term vs the rest (2 states, FITTED split), the 2-state objective credits
training's move as the per-image one does; (b) its per-reader minimiser beats the in-span build. If (a) holds, the open
object is reduced to a few gate patterns and their rates: the network-made fallback (class inversions, section 25) can
give both.

**Check I3 (`c14_units.py`, `cache/c14_units.png`).** The level-0 error, carried by the weights-only map to 4d, spreads
over many units (top 12 of 528 carry .107; prediction > .30 FAILS). Of the top 12, 6-7 are round-shape units (eyes,
knobs, a golf ball, rings, round dials and seals, spots) and 3 are straight-line / box units; their negative poles are
mostly dog faces and cars. So the coupling runs through clean units of the curves -> round-things circuit, with many
units sharing it (consistent with I2: no single contour mode).
**Check C.3 (a) and C.3' (`c12_states.py`, `c13_window.py`).** Neither a 2-state split of images (top 5% / 10%) nor a
per-image spatial window (even the oracle) makes the averaged objective credit training's move (ratios 1.31-1.42 vs the
per-image .88). **Statement 22.2 (reason, partly verified).** The per-image link of Gap B is the per-image DIRECTION of the
metric (rank <= 9: the gradients of the classes competing on that image, i.e. which downstream circuit paths are open),
not which images and not where in the image. C.3 is REFUTED in its 2-state form; the open object stays the per-image
gate pattern (L6), now located exactly.

## 23. State at the end of the session (2026-10-24)

**Cold test (sealed split, 10000 images, one look; `c15_cold.py`).**
| Build (k = 0, bench B3) | metric | sealed KL | label |
|---|---|---|---|
| level 0p | none | .0585 | MIX-DERIVED |
| one-layer weights metric (16.8) | 4a weights | .0528 | metric DERIVED |
| 4a, exact pool input | 4a weights | .0509 | metric DERIVED |
| 4b | weights 4a-4b, no gates | .0474 | metric DERIVED |
| **4d** | **weights 4a-4d, no gates** | **.0456** | metric DERIVED, depth CHOSEN |
| 5b | weights 4a-5b, no gates | .0496 | metric DERIVED |
| Kbar build | 2000 images | .0455 | FITTED microscope |
| local training from 4d | | .0390 | TRAINED |
| local training from Kbar build (best known) | | .0385 | TRAINED |
4d minus Kbar: +.0000 (se .0003). Activity moments: MIX-DERIVED (2000 images) for every build in this table.

**What a circuit is (the answer of this session).** A circuit of clean neurons is a chain of EDGES (weight kernel +
gate). Its effect on the coarse-network problem is one object: the metric carried down the chain from the output to the
family (Prop. 18.5, exact). The session located every part of the old error in that object:
| Part of the error | Size (dev) | Where it lives | From the weights? |
|---|---|---|---|
| pool-input fix | .0018 | the pool reader sees max-pooled input | yes |
| Gap A | .0061 | coupling between 4a units, carried up to 4d (contour -> round-things units; long-range band and centred window included) | YES: composed weights, no gates (.0489 dev / .0456 sealed = image metric) |
| Gap B | .0087 | per-reader re-routing, second order, driven by the per-image direction of the metric (which circuit paths are open on that image) | no: needs the per-image gate pattern (L6) |
**The one open object, now precise.** Per-image gate patterns: they (1) keep deep composition from collapsing (20.3:
rank 3223 vs 24), so they would remove the CHOSEN depth; and (2) carry Gap B (22.1-22.2). Next: the Bernoulli /
sampled-gate transport (20.3), first with true gates (microscope), then image rates, then network-made rates (section 25,
the accepted fallback). And network-made activity moments for the generalist (R3i .177 is still the weights-only start).

**Check 22.2 (`c16_matched.py`).** Metric from 2 other train images of the SAME label as the activity image: S+rem / S
= 1.472 (random partners 1.275; the image's own metric .879). PREDICTION (< 1) REFUTED. **Statement 22.2 REFUTED**:
"which classes compete" does not carry Gap B. Ruled out so far: which images (2 states), where in the image (window,
oracle included), which classes (same label). Only the image's own metric credits training. Status of Gap B's compact
form: OPEN. (Note: c7_reader's L_img .0951 < S .1059 in c11 is in-sample, its moments were built from those images;
it is not evidence. S+rem and T are covered by B2 on dev images.)

## 24. Ledger of Part VI, final for this session
| ID | Statement | Status |
|---|---|---|
| C.1 / 18.5 | transport identity | PROVED; CHECKED (2e-7) |
| C.2 | weights + gate statistics carried a fixed depth close Gap A | CHECKED with no gates at depth 4d (.0489 dev, .0456 sealed = Kbar); depth CHOSEN from {4a, 4b, 4d, 5b, logits} (4c, 4e not built); the cold test does not remove the CHOSEN label |
| C.3 | exactness at the circuit level | OPEN: local training still gains Gap B (.0079 from the 4d build) |
| C.3 (2-state mixture) | two circuit states carry Gap B | REFUTED |
| I2 | contour eigenmodes | REFUTED; 19.1 CHECKED |
| I3 | clean units carry the coupling | round-shape units: HOLDS (6-7 of 12); concentration: REFUTED (spread) |
| A1 / A2 / A3 | importance / coupling / gates at 4a | REFUTED / CHECKED / CHECKED |
| 20.2 | composition gives range, band and window | CHECKED (band 1.54 at 4b, 1.28 at 4d; window .24 at 4d) |
| 20.3 | deep collapse without gates; gates keep rank | reason VERIFIED (3223 vs 24) |
| B1 / B2 | Gap B reader-specific / second order | CHECKED / CHECKED |
| B4 | per-branch re-routing | REFUTED (5x5 exception) |
| 19.2 / 19.3 | in-span restriction / non-Kronecker metric moves the write | CHECKED / PROVED |
| B5 (a) / (b) | stationary reader metric in span / per reader | CHECKED / REFUTED |
| 22.1 | averaging the metric over images removes Gap B (R1, R2 refuted) | CHECKED |
| 22.2 | the per-image link = which classes compete | REFUTED |
| C.3' | per-image spatial window | REFUTED (oracle too) |
| X-w4d | local exactness from the DERIVED-metric build | gain .0079 = Gap B; ends tied with best known |
| COLD | sealed split | HOLDS (4d = Kbar; order 4a > 4b > 4d < 5b) |
Every build: activity moments MIX-DERIVED (2000 images). No statement here is "a weights-only build ties training".

## Appendix to Part VI: literature, borrowed or rejected
| Source | Borrowed | Rejected or caution |
|---|---|---|
| Essay, "Edge Featurizers", "A City vs a Knot" | edges between ordinary neurons as the unit; "a representation matters only through what reads it" | |
| Olah et al. 2020 (equivariance); Cammarata et al. 2021 (curve circuits) | edges between rotation families depend on relative angle | curve-circuits body text not read in full |
| Weiler and Cesa 2019; Cohen and Welling 2017 | family-to-family edges as functions of relative angle with few harmonics | the family is not an exact orbit (Prop. 6.1); P1 not tested this session |
| Voss et al. 2021 (expanded weights) | the composed gate-free metric of A4 is this idea, carried through layers | signed products cancel; deep expanded weights collapse (20.3 explains why) |
| K-FAC (Martens and Grosse 2015); NISP (Yu et al. 2018); mean-field backward pass | the transport recursion C4 | they assume gates independent of the metric above; true at 4a (A3), false deeper (20.3) and for Gap B (22.1) |
| Anthropic attribution graphs 2025; Arora et al. 2026 (neuron-basis circuits) | edge = activity x weight x gradient | transcoder nodes not used; neuron basis only |
| Williams and Jacobs 1997; Citti-Sarti (contour completion) | composition of local steps as a resolvent | LAW-DERIVED, not used; predicts oriented tails, while the measured band is a weights property of 4b+ |
| Compositional neuron groups (arXiv 2506.10920) | | dictionary-style; excluded |

---

# Part VII: Gap B and the depth, from first principles (Shonu, 2026-10-24: "figure out Gap B; make depth 4d calculated")

## 25. The coarsening is a weight perturbation of one layer: the Fisher frame

**Setting.** Reader layer weights theta = (W_j)_{j <= 368}, W_j in R^481 acting on y(x, p) = (3b activity at p, 1) (pool
readers on the max-pooled activity). The k = 0 build changes reader j's weights by delta_j = (-w_j on the family
columns, +B_j on the rest), i.e. e_j(x, p) = -delta_j . y(x, p).
**Proposition 25.1 (exact second-order cost = Fisher form).** L3(delta) = 1/2 delta^T F delta, with F = E_x Cov_{c ~ p(.|x)}
(grad_theta logit_c(x)) the Fisher information of the reader layer at the intact network (labels drawn from the network
itself), and grad_{W_j} logit_c(x) = sum_p g_c(x)(j, p) y(x, p) =: Gamma_c(x)[j, :], g_c = gradient at reader j's output.
Proof: L3 = 1/2 E_x Var_c(sum_{j,p} g_c(j,p) e_j(p)) (section 14) and sum_{j,p} g_c(j,p) e_j(p) = -<Gamma_c(x), delta>. QED.
**Corollary 25.2 (the coarsening problem is Optimal Brain Surgeon on one layer).** Removing the family = forcing the
family block of theta to zero; the best compensation is the OBS solution
  delta_R* = argmin over the rest block = F_RR^{-1} F_RF w   (Hassibi and Stork 1993, block form),
with F restricted to each reader's own rows. B2 showed the second-order term carries all of Gap B, so the OBS solution
with the TRUE F is what local training approaches.
**Corollary 25.3 (the three builds are three approximations of F).**
| Build | F replaced by | Known as |
|---|---|---|
| level 0 | I (x) E[y y^T] (output metric = identity per reader) | layer-wise OBS: ThiNet, OBC / SparseGPT |
| factorised (Kbar), in span | E_x[g g^T] (x) E_x[y y^T], spatially resolved | K-FAC / KFC (Martens and Grosse 2015; Grosse and Martens 2016) |
| training (local) | the true F, reached by preconditioned descent from the start | full-Fisher OBS, early-stopped |
So Gap A = how well the G-factor (E g g^T) is estimated; Gap B = the error of the Kronecker factorisation itself:
F - E[g g^T] (x) E[y y^T] = the part of the fourth moment E[(g g^T) (x) (y y^T)] that is not a product of second moments.
This matches 22.1: averaging the metric over images = taking the Kronecker product of averages.
**Statement 25.4 (CONJECTURE: where the non-Kronecker part lives).** In the Kronecker eigenbasis (u_m of the G-factor,
v_n of the A-factor), the true F is close to diagonal, and its diagonal departs from the product of eigenvalues:
s_mn = E_x sum_c (u_m^T Gamma_c(x) v_n)^2 != lambda_m sigma_n (EKFAC, George et al. 2018). The ratio
rho_mn = s_mn / (lambda_m sigma_n) is the coupling between "how the output reads reader mode m" and "how strongly
activity mode n is present", per image. Circuit reading: rho_mn > 1 where a circuit's output path (m) and its own input
pattern (n) are active on the SAME images (round objects open the round-object path and carry the curve signal).
**Why this is the right next object.** (1) s_mn is a first moment of a per-image quantity: every image informs all
177k entries (unlike the per-image coupled formula of 14.9, <= 9 numbers per image). (2) Its OBS solve is cheap: F v =
U_G (s o (U_G^T V U_A)) U_A^T. (3) It is the smallest correction that sees the image-level link 22.1 found.
**Checks (predictions in the log before each run).**
| ID | Check | Refutes 25.4 if |
|---|---|---|
| F1 | 25.1 numerically: 1/2 delta^T F delta (sampled) = L_img of c11 for S, S+rem, T | mismatch > 1% |
| F2 | EKFAC objective ranks training's move: L_E(S+rem) / L_E(S) | ratio >= 1 (like every averaged metric) |
| F3 | OBS with the EKFAC Fisher (CG, no knob): dev KL | closes < 50% of Gap B |
| F4 | one Newton step from S with the factorised Hessian and the TRUE gradient (first moment) | lands far from T |

## 26. The depth: from a chosen cut to a computed sum over depths

**Setting.** Kbar = E_x A_x^T Sigma_x A_x with A_x = W_L D_{L-1,x} ... D_{1,x} W_1 (gated Jacobian from the family to the
logits). The gate-free build to depth L replaces every D by 1 below L and the whole network above L by Sigma_L = I.
**Proposition 26.1 (independent-gate transport is a weighted sum over depths).** If the gates are independent Bernoulli
variables with rates r (independent of each other and of Sigma), then with Abar_l = the rate-scaled mean map to layer l
and M_l = the transported metric at layer l,
  E[D M D] = (r r^T) o M + diag(r (1 - r) o diag M),
  K_fam = Abar_L^T M_L Abar_L + sum_l Abar_l^T diag(r_l (1 - r_l) o diag M_l) Abar_l.
Proof: E[D_u D_v] = r_u r_v for u != v and r_u for u = v; unroll the recursion from the top. QED.
**Reading.** Each layer l adds a term "compose without gates up to l, then inject gate noise at l" (a Gram at depth
l), weighted by r_l (1 - r_l) times how much the output cares about layer l (diag M_l). The mean path decays by about
r^2 per layer. So the gate-free cut at depth L is a crude version of a sum over all depths, and the weights of that
sum are CALCULATED from the rates and the weights. The best single depth (4d) should be where this weighting puts its
mass.
**Statement 26.2 (CONJECTURE).** The Bernoulli transport to the logits (Sigma_out = network's class covariance), with
rates r: (a) with true per-image gates instead of Bernoulli, ties Kbar (exact up to the output metric); (b) with
Bernoulli gates at the image rates, ties the 4d build; (c) its depth weights peak at 4c-4d. Then the depth is no longer
chosen, and the only input is the rate map r (first moments: the named open object, now one number per unit).
**Computation.** Sample, do not unroll: draw masks D ~ Bernoulli(r) and output vectors xi with Cov = Sigma_out,
back-propagate through the masked linear network to the family; K_fam = E[g g^T]. Expectation exact; Monte Carlo
error ~ 1 / sqrt(samples).

**Corrections to 25-26 (before any run).**
1. Jacobian: Prop. 25.1 uses g at the reader OUTPUTS (gate-free at the readers), as Kbar, c11 and all of Part VI. The
   true reader-weight Fisher puts the reader gate inside g (pre-activation); B2 showed the two agree within the flip term.
2. 26.2 (a) as written is Prop. 18.5. Split: (a2) true gates + image-averaged Sigma_out (tests the gate-output
   correlation); (a3) true gates + uniform I - 11^T/n (tests output weighting).
3. Pools under random gates: a max-pool is a random one-hot selector over its window (each position chosen with its own
   rate); like a ReLU gate it adds a diagonal term. The average pool of c8 is its mean. Bernoulli transport uses the
   one-hot selector with uniform choice; stated once, kept at every depth.
4. Independent gates drop exactly the gate-activity correlation that 22.1 located. A Bernoulli transport that ties the 4d
   build is a success for the DEPTH question and says nothing about Gap B.
**Checks, revised.** F1 dropped (it equals c11's L_img). F0: gradient norm of L_img at T vs S on the free coordinates
(T near-stationary -> Cor. 25.2 verified directly). F2: three objectives on the same builds: plain KFC, EKFAC, L_img;
s_mn from 256 vs 2000 images. F3 only if F2 credits training's move; then read rho_mn (which reader modes co-vary with
which activity modes). F4 = Newton step with a per-image gradient: a MICROSCOPE (a Gauss-Newton step), never a formula.
**Generalist route, named now.** Everything in Part VII is FITTED / MIX-DERIVED (s_mn, gradients and rates from images).
Second-step candidates: rates r from network-made P(fires) (section 25 of PAPER: class inversions, levels within 2x for
~80% of neurons); rho_mn from network-made circuit-on inputs (feature visualizations and class inversions are, by
construction, inputs where a circuit's path and its input pattern co-occur).

**Check F0 / F2 (`c17_fisher.py`).** Ratios L(S+rem) / L(S): per-image F .908, plain KFC 1.073, EKFAC 1.021 (s from 256
images 1.019; s is well estimated, Spearman .982). **25.4 REFUTED**: the non-Kronecker part is not a rescaling of the
diagonal in the Kronecker eigenbasis; it is OFF-diagonal there. F0: the gradient of the per-image objective barely
shrinks from S to T (.893) and training's move is almost orthogonal to it (cos .07). **Statement 25.5 (reason for F0,
partly verified by the cos).** T is an early-stopped, Adam-preconditioned point, not the OBS optimum of the true F. So
Gap B (.0087) is a LOWER bound on the second-order gain still available beyond the factorised build: the true OBS point
may lie further down. F4 (an exact Krylov solve of the true objective) measures how much further.
**Statement 25.6 (next object: the Fisher restricted to a few circuit modes; CONJECTURE).** Write the reader-side
gradient in the eigenbasis of the G-factor: g_c(x)(j, p) = sum_k alpha_k(x, c, p) u_k(j). Then exactly
  F = sum_{k, k'} (u_k u_k'^T) (x) A^(kk'),   A^(kk') = E_x sum_c beta_k(x,c) beta_k'(x,c)^T,   beta_k = u_k^T Gamma_c,
i.e. a SUM of Kronecker products: reader mode k (a direction the output reads, a "circuit") times the activity
statistics seen THROUGH that circuit (activity weighted by where and when circuit k is engaged). Plain KFC sets
A^(kk') = lambda_k delta_kk' A; EKFAC keeps it diagonal in A's eigenbasis. 25.6 says: keep the top K circuit modes
exactly (all 962 inputs, all cross terms k, k'), Kronecker for the rest. Prediction (check F2'): the ratio L(S+rem)/L(S)
drops below 1 already at K <= 20 of 368 (Gap B lives in a few circuit modes). Control: the same with the top K ACTIVITY
modes kept exactly instead (reader side Kronecker).

**Check F2' (`c18_modes.py`).** No top-K subspace (K <= 80 of 368, reader or activity side) credits training's move
(ratios 1.08-1.28); the full per-image F does (.908). Training's move lives mostly in reader modes 20-240 of the
G-factor (about 75% of its energy); the 5x5-bottleneck readers sit there (77% in modes 80-240). **Statement 25.7 (reason,
CHECKED by the energy table).** Gap B is not a low-dimensional circuit object in either factor. It is the per-image
coupling between the readers the output reads most (top modes) and the mid-spectrum readers (5x5 bottleneck). The
size of what is available is the Newton decrement at S, 1/2 g_S^T F^-1 g_S; F4 measures it directly.

**Check F4 (`c19_pcg.py`, microscope).** Exact Krylov solve of the true second-order objective on n = 2000 images, from S:
the first step lowers dev KL by .0021 (24% of Gap B); further steps fit the 2000 images (their objective .0999 -> .0192)
and dev KL rises back. The gradient at S is only 1.7x its own sampling noise (true signal about .7 of the noise).
**Proposition 25.8 (how many images Gap B needs).** Let g = grad L(S) (population), g_x its per-image terms, P a fixed
preconditioner, tau^2 = tr(P^-1 Cov(g_x)). The n-image estimate obeys E |g_n|^2_P = |g|^2_P + tau^2 / n, so a step along
it carries more signal than noise only when n > n* = tau^2 / |g|^2_P. Proof: the cross term has mean zero; the variance
of a mean of n i.i.d. terms is Cov / n. QED. Measured: n* ~ 2000 / .7 ~ 2900 images.
**Statement 25.9 (what Gap B is; CHECKED as far as the measurements go).** Gap B is the Newton decrement at the
circuit-metric (Kronecker) optimum of the reader layer's OBS problem. Its direction is a fourth-moment statistic of the
image distribution (the per-image coupling of the top and mid reader modes with the activity, 25.7), and each image
fixes only about 1/n* ~ 1/2900 of it. It is not class, image group, spatial window, an eigenbasis rescaling or a top-mode
subspace (all refuted). In Theorem E's terms (section 13), Gap B is the value of DISTRIBUTION information: m(D) must
contain this fourth moment, and E-small fails for it (a few hundred images are far below n*). Training gets it because
500 steps x 64 images = 32000 image-gradients >> n*.
**Consequence for the two scenarios.** Developer: Gap B is reachable only with a large sample (n >> 3000) or by training.
Generalist: a weights-only formula can reach it only if the weights hold this fourth moment (P-Thm 31's per-sample
gradient-energy imprint is the one channel known to carry fourth-moment information). Not tested this session.

**Check 26.2 (`c20_depth.py`; all variants carried to the logits, no depth cut).** Code check: the true-gate linear copy
reproduces Kbar to 5.5e-6 (first version 6% off: max-pool ties at 0 pass gradient because J is taken at the reader
outputs; fixed with the pool's own indices).
| Variant | gates | output metric | dev KL | vs Kbar (se) | effective rank |
|---|---|---|---|---|---|
| a1 | true, per image | per image (= Kbar) | .0491 | 0 | 324 |
| a2 | true | image average | .0491 | +.0000 (.0001) | 565 |
| a3 | true | uniform I - 11^T/n | **.0490** | -.0001 (.0001) | 568 |
| b | independent Bernoulli at image rates | image average | .0520 | +.0029 (.0004) | 918 |
| b_uni | same | uniform | .0519 | +.0028 (.0004) | 916 |
| 4d (gate-free, CHOSEN depth) | none | identity at 4d | .0489 | -.0001 | 210 |
**Statement 26.3 (CHECKED).** Once the gates are the true ones, the output metric can be the uniform class covariance:
the output side carries no image information (a3). So Kbar = E_x A_x^T (I - 11^T/n) A_x: the whole of Gap A's object is
"which gate patterns occur", nothing else.
**26.2 (b) REFUTED, reason (verified by the rank).** Independent gates at the right rates over-whiten: rank 918 against
the true 568. The Bernoulli noise term of Prop. 26.1 is too large because real gates are CORRELATED: on one image many
units open together (clean units fire on the same object). So the depth question reduces to one object: the correlation
structure of the gates. Two kinds are possible, within a layer (co-firing) and across layers (whole paths opening
together = a circuit being on). Check c22 separates them.
**Why the gate-free cut at 4d worked (statement 26.4, CONJECTURE).** Gate-free composition has no gate noise at all
(rank collapses with depth); independent gates have too much (rank too high). True correlated gates sit in between.
The gate-free cut at depth L and the true transport agree when the noise the correlated gates inject above L is
roughly isotropic, so that "identity at L" stands in for it. A calculated depth is then the L at which the gate-free
metric's spectrum matches the true one. Prediction (checkable now on cached metrics): the effective rank of the
gate-free K^(L) crosses the true a3 rank (568) between 4b (732) and 4d (210), and the cosine with a3 peaks at 4b-4d.

**Check c22 (gate shuffles; uniform output metric).**
| Gate patterns | dev KL | vs true (se) | rank |
|---|---|---|---|
| true, one image for every layer (a3) | .0490 | 0 | 568 |
| 4a-4e from one image, 5a-5b from another | .0491 | +.0001 (.0000) | 543 |
| every layer from a different image | .0499 | +.0009 (.0001) | 643 |
| independent Bernoulli at the true rates | .0520 | +.0030 | 916 |
**Statement 26.5 (CHECKED).** Of what the gate correlations carry, 70% is co-firing inside a layer (which units open
together on one image) and 30% is coherence of the gate pattern ACROSS layers inside the 4a-4e stage; coherence across
the 4e -> 5a boundary carries nothing. In circuit terms: the circuit fed by the curve family is one coherent object from
4a through 4e, and what happens above it is independent of which way it went. This is a measured edge of the circuit,
and it explains why a cut in the 4c-4e range is the right single depth for the gate-free approximation; it does not yet
compute the exact cut (4d), because the gate-free metric is not the true-gated one within the stage.
**What would remove the depth entirely.** 26.3 + 26.5 say Kbar = E over gate patterns of A^T (I - 11^T/n) A, with the
pattern distribution coherent inside 4a-4e. If the network's OWN inputs (class inversions, fixed rule) produce gate
patterns with the right statistics, the metric is carried to the logits with no cut and no image: check c23.

**Check c23 (`c23_netmade.py`; uniform output metric, carried to the logits, NO depth cut).**
| Gate patterns from | dev KL | vs Kbar (se) | vs 4d (se) | rank |
|---|---|---|---|---|
| 256 real train images | .0492 | +.0002 (.0002) | +.0003 (.0004) | 527 |
| **256 class inversions (network-made, fixed rule, no photo)** | **.0491** | **+.0000 (.0002)** | +.0001 (.0004) | 512 |
**Theorem-level statement 26.6 (CHECKED on dev; cold test pending).** The coupling metric of Gap A is
  K = E_{x ~ network-made inputs} A_x^T (I - 11^T/n) A_x,
the expected Gram of the gated Jacobian from the family's signals to the logits, with the gate patterns of the inputs
the network makes for itself and the uniform class covariance at the output. No depth cut, no output weighting, no
photo. The CHOSEN depth (4d) is replaced by a calculation: carry the metric through the whole network with the network's
own gate patterns. Why the old blockers do not apply: the network is certain on its own inputs (P-Thm 24), but 26.3
shows the output metric can be uniform, so certainty never enters; class inversions are modes, not typical inputs
(section 25 of PAPER), but 26.5 shows what matters is the CORRELATION structure of gates (co-firing and 4a-4e path
coherence), and class inversions evidently carry it.
**Why the gate-free 4d cut worked, in this light.** Inside the coherent 4a-4e stage the gate patterns are strongly
correlated (26.5), so a gate-free product is a fair stand-in there; above 4e gates decorrelate from the family's circuit
and a gate-free product collapses (20.3). The best single cut therefore falls inside 4c-4e. 26.6 makes the cut
unnecessary.

**Checks c21 / c24 / c26 / c27 (Gap B solves).** One KFC-preconditioned step with exact line search: gradient from 2000 /
8000 / 35000 images -> -.0021 / -.0030 / -.0031 (direction cos with the population .80 / .94 / 1). Streamed Krylov (8000
images): -.0033 at the stop (it 2), -.0042 at it 4. Network-made gradient (256 class inversions): -.0002. Sealed: Krylov
it 2 -.0028, one 35000 step -.0025 vs the Kbar build; T is still .0043 ahead.
**Statement 25.10 (why the exact solver stalls; two parts, to verify).**
(a) Conditioning. PCG with preconditioner P converges at a rate set by the spread of the eigenvalues of P^-1 F on the
Krylov space. P = KFC misjudges F exactly in the out-of-span directions (that misjudgement IS Gap B, 25.4-25.7), so
P^-1 F is badly spread there and PCG needs many steps. Measured sign of it: held-out L fell 12% for 7% of dev KL, while
T's L fell 8% for 18% of dev KL: PCG first spends its steps where P^-1 F is large, which are not the KL's directions.
(b) Truncation. Every Gap B instrument since c11 used the top-10 renormalised class metric. B2 (full softmax) credited
S -> S+rem by 20%, the top-10 L_img by 9%, the uniform metric by 14% (c25). The truncated objective weakens the signal.
Repair for (b): the full output covariance diag(p) - p p^T, sampled with xi = (I - p 1^T) diag(sqrt p) eps. (a) is not
repaired by data; it bounds what a few Krylov steps can reach.
**Correction to 25.8.** The measured requirement is for the DIRECTION of the gradient: ~8000 images give cos .94 with the
population (the n* ~ 2900 estimate used the noise of the squared norm and is too low for direction).
**Status.** "The exact solver reached ~50% of Gap B" is a measurement at the stop; the rest is unmeasured, not shown to
be unreachable.

**Forward pointers (stale lines above).** Section 20-24: "depth 4d CHOSEN" is superseded by 26.6 (no cut: network-made
gate patterns carried to the logits). Section 25's check table: F1 dropped (equals L_img); F3 not run (F2 refuted).

## 27. Ledger of Part VII (2026-10-24)
| ID | Statement | Status |
|---|---|---|
| 25.1 | second-order cost = Fisher form of the reader layer | PROVED |
| 25.2 / 25.3 | coarsening = one-layer OBS; level 0 = layer-wise OBS, factorised = KFC, training ~ full-Fisher OBS | PROVED (identification); "training reaches the OBS optimum" REFUTED (F0: Adam, early stop) |
| 25.4 | Gap B = EKFAC diagonal correction | REFUTED (ratio 1.02) |
| 25.5 | T is an early-stopped Adam point, so Gap B is a lower bound | partly CHECKED (cos .07 with the gradient) |
| 25.6 | Gap B lives in a few circuit (reader) modes | REFUTED (K <= 80 of 368) |
| 25.7 | Gap B = per-image coupling of top and mid reader modes (5x5 bottleneck) | CHECKED (energy table) |
| 25.8 | data requirement n* = tau^2 / |g|^2 | PROVED; direction needs ~8000 images (measured) |
| 25.9 | Gap B = distribution information (a 4th moment of the image mix) | CHECKED: network-made inputs carry none of it; real images reach ~50% by exact solves |
| 25.10 | the solver stalls: (a) Kronecker conditioning, (b) top-10 truncation | stated; (b) under test (KRYLOV-FULL) |
| 26.1 | independent-gate transport = weighted sum over depths | PROVED |
| 26.2 (b) | Bernoulli gates at true rates match the 4d build | REFUTED (.0520) |
| 26.3 | with true gates the output metric can be uniform | CHECKED (.0490) |
| 26.4 | a spectrum-matching rule picks the depth | REFUTED in spirit (cosine peaks at 4b; Bernoulli has the top cosine but builds worse) |
| 26.5 | gate correlations: 70% within-layer co-firing, 30% path coherence in 4a-4e, 0 across 4e -> 5a | CHECKED |
| 26.6 | network-made gate patterns + uniform output metric, carried to the logits = Kbar | CHECKED (dev .0491) and COLD (sealed .0455 = Kbar) |
**Sealed state (c27).** Kbar build .0455; 4d (gate-free, CHOSEN depth) .0456; netmade-256 metric (no cut, DERIVED)
.0455; Krylov it 2 (real images, 8000) .0427; one 35000-image step .0430; best known T .0385.
Guard: activity moments are MIX-DERIVED (2000 real images) in every build; the Gap B steps also use real images.
No weights-only build ties training.

**Final checks of this stretch.** (1) Local exactness from the network-made metric build: .0491 -> .0403 (gain .0088 =
Gap B), tied with the best known network (-.0001, se .0002). (2) 32 class inversions give the same metric build as 256
(.0489 vs .0491): the derived metric is cheap. (3) Krylov with the full output covariance: no better than with the top-10
metric (best .0456-.0462). **25.10(b) REFUTED; 25.10(a) stands**: the exact second-order solvers stall at ~40-50% of Gap
B because the Kronecker preconditioner is poor exactly in Gap B's directions; training's Adam (a per-coordinate
preconditioner over 177k coordinates) gets through them. **Next statement (not run): the right preconditioner for Gap B
is coordinate-wise, the diagonal of F in the reader x input basis, combined with the circuit metric (Kronecker) on the
span.** It would tell whether the second half of Gap B is reachable by an exact solve or needs training's many steps.
**State of the two questions (2026-10-24).**
| Question | Answer | Label |
|---|---|---|
| Depth 4d | removed: carry the metric to the logits through the network's own gate patterns (32-256 class inversions), uniform output metric; ties the image metric (dev .0491, sealed .0455); local training from it reaches .0403 = best known | metric DERIVED (weights + e1 rule; step-doubling OPEN) |
| Gap B | the Kronecker error of the reader layer's OBS problem: a fourth moment of the real image mix (same-image gate-activity link); network-made inputs carry none of it; ~8000 real images fix its direction; exact solves reach ~40% (sealed -.0028), stalled by conditioning | characterised; ~40% closed with real images (MIX-DERIVED); rest open |
Guard: activity moments are MIX-DERIVED (2000 real images) in every build.

---

# Part VIII: the large network is enough (Shonu, 2026-10-24 evening, plan `last-session-made-quite-lively-flute.md`)

**Frame.** Coarse-graining whose target is the distillation optimum. The small network is the large network seen at
a coarser resolution, built from its own parts (families, edges, circuit metrics). "Best" is defined by distillation:
match the intact network's function. Training is one approximate, stochastic solver of that optimum. We look for
its closed form. Level 0 is the first term, the circuit metric (Part VI) the second, the same-image term the next.

**Name.** "Gap B" is renamed **the same-image term**: on one image, the circuit is on AND the build errs there.

**Target (Statement S, Sufficiency).** The generalist build needs three things, each a function of the weights:
(1) the answers (the intact network labels any input), (2) the metric (DERIVED, 26.6), (3) the questions: a mix q of
network-made inputs whose optimum is the optimum of the real mix P. Theorems P-5, P-24, P-34.3 stay true: no
weights-only rule is optimal for EVERY mix that fits the weights. That is a worst-case statement. S asks for one good
mix made by the network, and 28.3-28.4 say exactly what "good" requires.

## 28. The distillation optimum as a moment equation

**Setting.** v in R^{368 x 962} = the reader-layer weight change (masked), split v = (w, v_R): the family block w is
fixed (removal), v_R (the free coordinates, |R| = 368 x 471) is chosen. For an input x and an output direction xi
(uniform metric: xi ~ N(0, Pi), Pi = I - 11^T/n, enough by 26.3 and c25), Gamma_xi(x) = sum_p g_xi(x, p) y(x, p)^T is the
gradient of xi . logits with respect to the reader-layer weights. The build's output error on x in direction xi is
o_xi(x; v) = <Gamma_xi(x), v>. For a mix q:
  L_q(v) = 1/2 E_q E_xi o_xi(x; v)^2 = 1/2 v^T F_q v,   F_q = E_q E_xi vec Gamma vec Gamma^T.

**Proposition 28.1 (the optimum is a moment equation; PROVED).** v_R minimises L_q iff
  g_q(v) := E_q E_xi [ o_xi(x; v) Gamma_xi,R(x) ] = 0;
if F_q,RR is positive definite the root is unique: v_R*(q) = -F_q,RR^-1 F_q,RF w.
Proof: L_q is a convex quadratic in v_R with gradient g_q. QED.
Reading: training on q (SGD, one sampled o Gamma_R per step) is a Robbins-Monro solver of this equation. The exact-KL
version replaces o by the exact per-input output change; the first-order condition keeps the moment form.

**Proposition 28.2 (what makes a good question; PROVED).** (a) An input with o_xi(x; v) = 0 for all xi adds nothing to
g_q(v). (b) With q's own curvature F_q,RR (the proof needs exactly this matrix; a preconditioner is only a tool):
  g_q^T F_q,RR^-1 g_q <= E_q E_xi o^2 = 2 L_q(v).
Proof of (b): g^T F^-1 g = sup_u (2 u.g - u^T F u) = sup_u E[2 o (u.Gamma_R) - (u.Gamma_R)^2] <= E[o^2], since
2ab - b^2 <= a^2 pointwise. QED.
Reading: the Newton decrement of q at the build S (the gain q can teach) is at most S's own loss on q. A mix of inputs
on which S is nearly right cannot teach the move, whatever its direction.

**Theorem 28.3 (transfer bound: the mix matters only through the leftover; PROVED).** Let P be the real mix and q any
mix with F_q,RR positive definite, and kappa = lambda_max(F_q,RR^-1/2 F_P,RR F_q,RR^-1/2) (how well q covers the
directions P uses). Then
  L_P(v*(q)) - L_P(v*(P)) <= kappa L_q(v*(P)).
Proof. g_q is affine in v_R with slope F_q,RR and vanishes at v*(q), so Delta := v_R*(q) - v_R*(P) = -F_q,RR^-1 g_q(v*(P)).
L_P is quadratic with minimum at v*(P), so the excess is 1/2 Delta^T F_P,RR Delta <= 1/2 kappa Delta^T F_q,RR Delta =
1/2 kappa g^T F_q,RR^-1 g <= kappa L_q(v*(P)) by 28.2(b). QED.
Corollaries. (i) Exact copy: if the best build is exact on q's inputs (L_q(v*(P)) = 0), q gives the real optimum,
whatever its weights. (ii) The two levers of a question mix are COVERAGE (kappa: q must see every direction P's Fisher
uses) and LEFTOVER (P's optimum must be nearly exact on q's inputs). Neither needs q = P. This is the precise form of
"the mix only decides how to spread the leftover errors".

**Proposition 28.4 (the root condition; the sufficient statistic; PROVED).** v*(q) = v*(P) iff g_q(v*(P)) = 0 (given
F_q,RR positive definite). So the root depends on q only through one vector in R^|R|, not through F_q. It is invariant
under any rescaling of q's weights and of F_q. Near S, the first Newton step with a shared preconditioner M,
-M^-1 g_q(S), points along P's step iff g_q(S) is parallel to g_P(S): q must reproduce the DIRECTION of one vector,
and its length is free (it only sets the step, which the line search on P-free curvature fixes).
**The vector in circuit terms (identity, PROVED from the definitions).** With E_xi xi xi^T = Pi and J_x(p) the gated
Jacobian of the logits with respect to the reader outputs at position p:
  g_q(S) = E_q sum_{p, p'} K_x(p', p) e_x(p) y_R(x, p')^T,   K_x(p', p) = J_x(p')^T Pi J_x(p),   e_x(p) = v_S y(x, p),
i.e. the per-image coupling metric (Part V-VI) times the reader-input error times the activity, on the SAME image.
The factorised build S zeroes the version with E[K_x] in place of K_x (in span). So, in span, g_q(S) is the covariance
over q of two per-image objects: K_x (where the circuit carries signal to the output: the gate pattern) and e_x y_R^T
(where the family's signal is lost, with the other channels' activity). The same-image term = this covariance.
26.6 showed network-made inputs reproduce the MEAN of K_x. 28.4 says the questions must also reproduce its
COVARIANCE with the error, up to one positive scale.

**Statement 28.5 (the reason c26 failed; TO VERIFY, check K0).** The 256 class pictures are easy questions: S's loss
on them, L_q(S), is small, so by 28.2(b) the gain they can teach is small (c26: step length 1.2e-11 vs 9.8e-11 for 256
photos). Alternative reason: their error is not small but its covariance with K_x points elsewhere (direction). K0
separates the two: error energy, Newton decrement / error energy (how coherent the pushes are), and direction.

**Ledger edit (28.7).** 25.9 is restated: "the 256 class pictures do not carry the same-image term (equal count with
photos)". It was one question-maker failing, not a statement about the network. The open object is now: a network-made
mix q whose g_q(S) is parallel to g_P(S) (28.4), with coverage (28.3).

**Check K0 (`c40_errenergy.py`).** S's error energy per input: class pictures 7.08e4, photos 1.20e4 (5.9x MORE on class
pictures). Direction (cos of M^-1 g with the 8000-photo direction): 256 photos .938, 256 class pictures .313.
**28.5 REFUTED** (they are not easy questions). The direction is the reason. Side result: under the uniform metric, 256
photos already fix the direction (.938); 25.8's "~8000 images" was for the top-10 class metric. So 256 GOOD questions are
enough, by measurement.

**Proposition 28.6 (two parts of the push; PROVED).** Pair the gates of one input with the activity of an independent
input from the same mix. Then exactly
  g_q(S) = g_q^fact(S) + g_q^cov(S),
  g_q^fact = sum_{p,p'} E_q[K(p', p)] E_q[e(p) y_R(p')^T]   (gates and activity from independent inputs),
  g_q^cov  = sum_{p,p'} Cov_q( K(p', p), e(p) y_R(p')^T )   (the same-image term proper).
Proof: E[ab] = E[a]E[b] + Cov(a, b) entrywise, applied to the identity of 28.4. QED.
**Statement 28.7 (reason for K0; TO VERIFY).** S is the factorised optimum for the PHOTO activity moments (its
re-routing Lambda was fit to photos: the MIX-DERIVED guard). On class pictures the family's co-activity with the other
channels differs (5.9x error energy), so g^fact is large: it pushes S toward re-fitting level 0 / Kbar to the class
pictures' own activity. That is a mismatch of first-order mix statistics, not the same-image term, and it drowns the
covariance part. The same-image term itself, g^cov, may well be present in network-made inputs.
Prediction only this reason makes (K0b, shuffle estimator for g^fact): (i) photos: |g^fact|_M / |g|_M < .3 (S is their
factorised optimum); (ii) class pictures: |g^fact|_M / |g|_M > .7; (iii) cos(M^-1 g^cov_class, photo-8000 direction) > .6
(raw: .313). If (iii) holds, the network's own inputs carry the same-image term, and Statement S needs only the
covariance part from them.

**Check K0b (`c41_split.py`, random derangements; cyclic shifts leaked class adjacency, both sets are class-sorted).**
| set | |g^fact|/|g| | cos fact | cos cov (vs 8000 photos) |
|---|---|---|---|
| 256 photos | .146 | .070 | .939 |
| 256 class pictures | .370 | -.315 | .492 |
28.7 PARTLY VERIFIED: (i) holds (S is the photos' factorised optimum, their push is 85% same-image term); the class
pictures' factorised part is real and points AGAINST the photo move; removing it lifts the alignment .31 -> .49. (ii) and
(iii) refuted: the class pictures' same-image covariance itself matches the photos' only at cos .54.
**Statement 28.8 (reason for the remaining mismatch; TO VERIFY).** On photos the same-image term is contour integration
(16.6): the error at one position couples, through K_x(p', p), to activity at OTHER positions along a round contour
(co-circular partners). Class pictures are optimised on plain pixels with no natural-image prior: they hold fragments
and textures, not long coherent contours. So their covariance has the same-position part (p = p') but not the
cross-position part (p != p'). Prediction (K0c): photos' cross-position part carries > 50% of |g^cov|; class pictures'
cross part is misaligned (cos < .4) while their same-position part is aligned (cos > .6).

**Check K0c (`c42_contour.py`).** Share of |g^cov| in the cross-position part: photos .973, class pictures .984.
cos(class, photo): same-position .761, cross-position .531. **28.8 REFUTED**: the class pictures are not missing the
contour coupling; they have it, at about half the right direction.

**What the three checks establish (K0, K0b, K0c).**
1. The class pictures are hard questions, not easy ones (5.9x the photos' error energy).
2. Their push has two parts. The factorised part (37%) points AGAINST the photo move. It comes from S being fit on
   photo activity moments: on a different mix, S is not the factorised optimum (28.7 (i), verified).
3. Their same-image term is real, contour-like (98% cross-position, like photos' 97%) and half-aligned with the photos'
   (cos .49-.54).
4. 256 photos fix the direction to .94. So the target is concrete: 256 network-made questions whose same-image term
   points where the photos' does.
The network's inputs carry the same KIND of term. What differs is its direction. 25.9 ("network-made inputs carry none")
is wrong as stated: class pictures carry about half of its direction.

**Proposition 28.9 (the questions live at 3b; PROVED).** In InceptionV1 the readers' input is pool(3b) and everything
above it (4a ... logits) is a fixed function of the 3b activity map A_x (480 x 28 x 28). So y_x, e_x(S), the gates above
the readers and K_x are all functions of A_x, and g_q(S), g_q^fact, g_q^cov depend on q ONLY through the law of A under q
(its push-forward to 3b). Proof: composition. QED.
Consequences. (1) "Natural" means, for this problem, one object: the law of 3b activity configurations (which features
are on, where, together). Pixels are one way to make them. (2) This is the same object as L6 (gate statistics),
12.4'' (co-occurrence law of a dictionary) and 22.1 (circuit state). The open problem of Parts I-VIII is ONE object.
(3) The network's own description of that law sits in the weights that READ 3b: each 4a unit's kernel is a typical
3b configuration it was trained to detect (W7: features are learned from the data). A network-made question can be
built at 3b directly, from the readers' own preferred configurations, without pixels.

**Check K1' (`c43_covstep.py`): the first network-made same-image step.** Direction -M^-1 g^cov from 256 class
pictures (cos .49 with photos), length from the class pictures' own curvature: dev KL -.0008 vs S (se .0003); with the
photo line search (FITTED) -.0009. The two lengths agree within 10%. Gain = cos^2 x the photo step's, as the geometry of
28.4 predicts. So the remaining distance is ONE number: the direction alignment (.49 now, .94 for 256 photos).

**Statement 28.10 (reason for the half alignment; TO VERIFY).** Law of total covariance over the class y:
  g^cov_P = E_y[ Cov_P(K, Z | y) ] + Cov_y( E[K | y], E[Z | y] )   (Z = e y_R^T, summed over p, p' as in 28.6),
a WITHIN-class part (pose, position, clutter, whether a round object is present) and a BETWEEN-class part. One class
picture per class can only carry the between-class part. 22.2 already found that same-label metric images do not credit
training's move, a sign that the within-class part dominates. Prediction (K0d, same-class shuffle): within-class part
> 60% of |g^cov_P|_M, and the class pictures' cov aligns better with the photos' BETWEEN part than with the within part.
Consequence if verified: the questions must vary WITHIN a class the way the network's circuit varies, many per class,
not one mode per class.

**Check K0d (`c44_within.py`, 128 classes x 2 photos).** |within|/|cov| .957, |between|/|cov| .468, cos(within, between)
-.04; cos with the 8000-photo direction: within .846, between .089. Class pictures' cov: cos .305 with the photos'
within part, .096 with the between part. **28.10: part 1 CHECKED, part 2 REFUTED.**
**Statement 28.11 (CHECKED by K0d).** The photos' same-image term is class-independent: its between-class part is
orthogonal to the target direction. It is a covariance of CONTENT (is a round contour there, where, is the circuit on),
which varies inside every class. With 28.9: the term is fixed by how 3b content varies across inputs, and labels play
no role. The class pictures carry part of it through their content, not their class.
**Derived question-maker (K2, rule fixed before the run).** Replace "one picture per class" by "one picture per reader
of the family": for each of the 368 reader channels j, maximise reader j's pre-activation at ONE position (the centre of
its 14 x 14 map). Why each choice: (a) readers, because the term lives where 3b content meets the readers (28.4, 28.9);
(b) all 368, no selection; (c) one position, because the photos' term is driven by single objects (16.6: 5% of photos,
one round thing) and is 97% cross-position coupling around it (K0c); the rest of the picture stays free. Everything else
= the e1 rule (plain pixels, lucent standard transforms, Adam .05, 256 steps, feature-vis ReLU). Label: DERIVED
(weights + fixed rule); step-doubling (K4) decides if the rule is CHOSEN.

**Check K2 (`c45_readerq.py`, `c46_readerstep.py`).** Reader questions (368, DERIVED rule above): same-image direction
cos .609 with the 8000-photo direction (class pictures .49); network-made step (direction AND length from the questions):
dev -.0012 vs S (se .0002) = 46% of the 256-photo step (class pictures 31%). Their factorised part is large (84% of the
push; each picture is one patch in flat gray, so its activity moments are far from photos'), and Prop. 28.6 removes it.
Gain ~ cos^2 again. Step-doubling (K4) running.

## 29. Ledger of Part VIII (2026-10-24 evening)
| ID | Statement | Status |
|---|---|---|
| S | Sufficiency: answers, metric, questions all from the weights | TARGET; answers + metric done; questions at 46% of the photo-256 step |
| 28.1 | the distillation optimum is a moment equation; training = Robbins-Monro solver | PROVED |
| 28.2 | gain a mix can teach <= the build's own loss on it | PROVED |
| 28.3 | transfer bound: excess on P <= kappa (coverage) x leftover of P's optimum on q | PROVED |
| 28.4 | root condition: q must reproduce the DIRECTION of one vector, g_P(S) | PROVED; 256 photos give it to .94 (K0) |
| 25.9 (old) | "network-made inputs carry none of the same-image term" | REFUTED (class pictures carry cos .49 of it once 28.6 is applied; reader questions .61) |
| 28.5 | class pictures failed because they are easy | REFUTED (5.9x more error energy) |
| 28.6 | push = factorised part + same-image covariance | PROVED |
| 28.7 | class pictures' push is spoiled by the factorised part | PARTLY CHECKED (anti-aligned -.32; removing it .31 -> .49) |
| 28.8 | class pictures lack the cross-position (contour) part | REFUTED (98% cross-position) |
| 28.9 | the term depends on q only through the law of 3b activity maps | PROVED |
| 28.10/28.11 | the photos' term is within-class, class-independent | CHECKED (between-class part orthogonal, .09) |
| K2 rule | one question per family reader, one position | DERIVED; cov .61, dev -.0012 (46%); step-doubling OPEN |
**Where the remaining half is (to derive next).** By 28.9 + 28.11 the target is the variation of 3b CONTENT across
inputs. Reader questions put one object in an empty field; photos put objects in clutter (other content at the other
positions, which the cross-position coupling of K0c reads). Next statement to write before any run: the cross-position
term needs the CONTEXT around the object, and which context the network expects is again a 3b law the readers' weights
describe.

**Check K2-net (`c47_netprec.py`).** The step's preconditioner was the c17 KFC from 2000 photos. With both KFC factors
built from the 368 reader questions instead: dev -.0014 vs S (se .0003), at least as good as with the photo factors
(-.0012). Count check at 256 questions: -.0010 (photo M) / -.0012 (network-made M), still above the class pictures.
So the same-image step is network-made end to end: questions, gradient, metric and step length. The start point S is
still the Kbar build on 2000 photos' activity moments (MIX-DERIVED); that guard is unchanged.

**Check K5 (`c48_cold.py`, sealed, one look).** vs Kbar build S (.0455): 256-photo step -.0019 (se .0004); class
pictures -.0008; reader questions, network-made end to end, -.0011 (se .0002) = 58% of the 256-photo step; best known T
.0385. Dev gains hold on sealed within 2 se.
**State of Statement S (2026-10-24 evening).** Answers: from the network. Metric: from the network (26.6). Questions:
from the network, carrying 54-58% of what 256 photos carry for the same-image step, by a rule with no classes and no
photos. The start S still uses photo activity moments. 25.9 is replaced: the network's own questions carry the
same-image term, in part, and the part is measured.

## 30. Closing the question gap: the questions must turn on the whole circuit

**Two gaps, named.** Q (questions): the network-made same-image direction vs the photos' (cos .61 vs .94; sealed step
-.0011 vs -.0019). C (solver): even photos with exact solves reach only ~40% of S -> T (25.10). This section closes Q.

**Lemma 30.1 (irrelevant questions are harmless; PROVED).** If a question x has o_xi(x; S) = 0 for all xi, adding it to q
leaves g_q(S) unchanged up to the factor n/(n+1) (28.2a), so the DIRECTION is unchanged (28.4). Questions that never
touch the family need no filter: a rule may make one question per unit of a whole layer without selecting units.
(For the factorised part, an added question also enters the shuffle pairs; 28.6 removes that part exactly in
expectation, so the same-image part keeps the property.)

**Statement 30.2 (CONJECTURE: why reader questions reach only .61).** K_x is the gated Jacobian from the readers to the
output. 26.5 measured that the family's circuit is ONE coherent object from 4a through 4e (gate patterns coherent
inside 4a-4e, independent across 4e -> 5a). The same-image term is the covariance of K_x with the family's error
(28.6). A reader question turns on the circuit's FIRST layer only, in an empty gray field: its gates in 4b-4e are
those of "a patch in nothing", so its K_x varies with the patch in the wrong way. Photos vary the whole circuit's state:
object plus context, as 4b-4e see it. Derived rule (K6): one question per unit at the TOP of the coherent stage
(mixed4e, 832 channels), unit at the centre of its map, e1 settings otherwise. A 4e unit's receptive field covers the
object and its context, and its optimum is the network's own picture of "this circuit on". No selection (Lemma 30.1).
Prediction: same-image cos > .75 and a network-made step better than the reader questions'. Fair-count first run: a
fixed 256-subset of the 832 units (rng 49).

## 31. The solver gap C: the sampled metric makes the system underdetermined

**Proposition 31.1 (counting; PROVED).** Let the per-image output metric used in the objective have rank r_x (top-10
class metric: r_x <= 10; c28's sampled full covariance: r_x <= 10 samples). Then rank F_n <= n r_x. The free block has
|R| = 368 x 471 = 173,328 unknowns. For n r_x < |R| (n < 17,300 images at r_x = 10) F_n,RR is singular: the solve fixes
v_R only on an n r_x-dimensional subspace; the rest is set by the preconditioner and by noise. Proof: F_n is a sum of
n r_x rank-one terms. QED.
**Reading of the old stall (25.10).** c24 used 8000 images x top-10 = 80,000 < 173,328: the system was underdetermined
by more than half. The held-out objective rose after 2-4 Krylov steps (over-fit) and the KFC preconditioner decided the
undetermined half: that is 25.10(a) and the "8000 images for direction" note, seen from one fact.
**Proposition 31.2 (exact output metric by one JVP and one VJP; PROVED).** For the uniform metric Pi,
  F v = E_x sum_{p'} [ J_x(p')^T Pi u_x ] y_R(x, p')^T,   u_x = sum_p J_x(p) (v y(x, p)),
i.e. a forward-mode product (u_x = the logit change caused by v) and one backward pass of Pi u_x. It is exact in xi
(no sampling), costs about one backward per image (cheaper than 10 sampled xi), and its per-image rank is up to 999.
Proof: E_xi[xi xi^T] = Pi in the sampled form. QED.
**Statement 31.3 (CONJECTURE).** With the exact uniform metric, n = 2000 images give n r_x ~ 2,000,000 >> |R|: the system
is determined, the held-out objective keeps falling, and CG from S goes far past the one-step gain. If its optimum also
improves KL past the stall (-.0033 dev), gap C is a metric-rank artifact, not a property of training. Uniform metric
justification: 26.3 (Gap A) and c25 (it credits training's move: ratio .863, more than the class metric's .908).

**Check C1 (`c50_exactcg.py photos 2000`, MICROSCOPE: photos).** PCG from S with the exact uniform metric (self-check vs
50 sampled xi: cos .905, norm .949). Held-out objective falls at every iteration (576 -> 464 by it 8); dev .0491 ->
.0463 -> .0458 -> .0446 -> .0436 -> .0431 -> .0430 -> .0427 -> .0425 (it 8, 75% of S -> T). 31.3 CHECKED so far: gap C was
the rank-10 sampled metric (31.1), not a limit of closed-form solves. Training is not needed to get past the stall.

**Proposition 31.4 (one object; PROVED from 28.9 and 31.2).** The exact-metric objective F (all of it: the factorised
part that built S, and the same-image part) is a functional of the law of 3b activity maps under the mix, composed
with the fixed network above 3b. So for the generalist, everything that is still MIX-DERIVED (the activity moments of
S, the same-image term) is one object: Law(A_3b) under the training mix. A network-made question set whose 3b law
matches it closes the start, the metric and the same-image term together, and the exact solve (31.2) then needs no
photo at all.

**Fast check C1-path (`c52_path.py`, cached iterates).** The gate-free exact-uniform path heads toward training's move
(cos .58-.60 in the cost metric for its 1-4; projection on T - S reaches 1.41 at the best iterate) but its plain size is
8.6x -> 51x -> 73x training's whole move: it travels far along near-flat directions.
**Proposition 31.5 (the gate-free objective is not the Taylor expansion of KL; PROVED).** The build changes reader j's
PRE-activation by -e_j; its output changes by relu(a_j - e_j) - relu(a_j) = -D_j e_j + (flip terms), D_j = 1[a_j > 0].
So the second-order Taylor expansion of KL in v is 1/2 (D e)^T J^T Pi_x J (D e) with Pi_x = diag(p_x) - p_x p_x^T, not
the gate-free 1/2 e^T J^T Pi J e used since Part VI (correction 1 there). Where a reader is closed (D = 0) the gate-free
model credits changes that the network ignores. A solver that minimises it is free to move along those positions;
large moves there later OPEN gates (flip terms) and hurt. Proof: chain rule through the ReLU. QED.
Prediction (C2): the gated, exact-softmax solve has no late rise and stays small in plain norm.

**Checks C1b and C2 (partial).** Along the gate-free path, the true KL (held-out train photos) is lowest at it 8 and then
rises; the KL's own gate-free second-order objective (class metric) stays flat there. The gated, exact-softmax solve
got WORSE from its first step (dev .0493, .0504; S .0491) while its own held-out objective fell. 31.5's prediction is
REFUTED in its fixed-gate form; its mechanism (gates) is CONFIRMED by C1b.
**Proposition 31.6 (frozen-gate models have exactly flat directions; PROVED).** Fix gates D (gate-free: D = 1; gated:
D = intact gates). The model objective is 1/2 E (D e)^T J^T Pi J (D e). Any change of v that alters e only at
(reader, position, image) triples with D = 0 has zero model cost (gated) or a cost that does not reflect the network
(gate-free). Preconditioned CG on a consistent singular system converges on range(H) but its null-space component is
P^-1 r, set by the preconditioner and never corrected (standard; Kaasschieter 1988). The gated model has MORE flat
directions than the gate-free one, which is why the gated solve went wrong at once.
**Definition 31.7 (the exact-gate objective).** Keep the output to second order, the readers exact:
  L_ex(v) = 1/2 E_x (J_x Delta_x(v))^T Pi_x (J_x Delta_x(v)),   Delta_x(v) = relu(a_x - e_x(v)) - relu(a_x),
with a_x the intact reader pre-activations. It has no flat directions of the 31.6 kind: a closed reader is free only
until -e crosses its margin a < 0, then it is charged. It equals the gated model while no gate flips.
**Statement 31.8 (TO CHECK, small and fast).** Along the cached gate-free path, L_ex tracks the true KL (same argmin,
same late rise), while the gate-free model does not. If it holds, L_ex is the objective to solve, and its minimiser
needs only a Gauss-Newton loop that re-reads the gates at the current point.

**Check C3 (`c53_lex.py`, 300 photos).** Along the cached gate-free path: L_ex 311 -> 263.7 (lowest at it 8, where the
true KL is lowest on 1000 photos) -> 288 at it 15 (+8% from it 11); the gate-free model keeps falling. **31.8 CHECKED.**
**Definition 31.9 (the solve of L_ex).** Gradient: grad L_ex(v) = - E_x sum_p [D_v(x,p) o (J^T Pi J Delta_x(v))(p)]
y(x,p)^T, with D_v = the readers' gates UNDER the current build (re-read at every step). Direction: preconditioned
Polak-Ribiere+ conjugate gradient (no knob; restarts when beta < 0). Length: the gated quadratic's minimiser along the
direction, halved until L_ex falls (so every accepted step lowers the exact-gate objective). Each step is a linear-algebra
step on a fixed statistic of the mix; the fixed point is the root of the exact-gate moment equation (28.1 with Delta).

**Checks C4, C5.** C4 (exact-gate solve, 500 photos): held-out L_ex lowest at it 4-5, dev-1000 -.0077 vs S (T - S = -.0091
on the same images), but the move is still 45x training's. C5: 99.5-99.9% of the c50 path's plain move is in the BIAS
columns, and the photo step's bias part alone gives -.0025 of the full -.0028 (the weight part alone -.0028).
**Proposition 31.10 (the drift is the uncentered parametrisation; PROVED for KFC, check C6 for the effect).** With
y = (h, 1) and channel means mu, A = E[y y^T] = C A' C^T, C = I + mu e_const^T, A' = blockdiag(Cov(h), 1). The bias
direction and the "weights on the means" direction W mu change the output identically on the mean image, so A has a
direction of size ~1 next to channel scales ~7.6e4, and A^-1 magnifies it ~7.6e4x. KFC therefore proposes huge bias moves
cancelled on average by weight moves; their effect is large exactly on images away from the mean, where they flip
gates. Centered KFC (Grosse and Martens 2016) preconditions in y' = (h - mu, 1): step = G^-1 (R C) A'^-1 C^-1. The means
are in A's constant column already (no new data). The Fisher itself is parametrisation-invariant; only the
preconditioner's geometry changes, so the solve's fixed point is unchanged and only its path is cleaned.

**Check C6 (centered KFC, 500 photos).** Dev-1000 -.0052 at it 5 (uncentered C4: -.0077); held-out L_ex falls steadily;
plain move 1500-3000x training's. **31.10's predicted effect REFUTED.** Correction to C1-path/C4: the plain norm of a
move depends on the parametrisation (a centred bias carries W mu), so "50x training's move" measured nothing physical.
The physical damage measure is the number of reader gates a move flips on held-out images (31.6-31.7), or L_ex itself.

## 31'. Ledger for gap C (consolidated; no more solver variants)
| ID | Statement | Status |
|---|---|---|
| 31.1 | the old stall = rank-10 sampled metric, system underdetermined below 17,300 images | PROVED; confirmed by C1 |
| 31.2 | exact metric by forward + backward product | PROVED; self-check cos .905 vs 50 sampled xi |
| 31.3 | with the exact metric the closed-form solve goes past the stall | CHECKED: C1 it 8 (derived stop) dev -.0066 = 75% of S -> T (2000 photos, FITTED microscope) |
| 31.5-31.7 | the gate-free model is not the Taylor expansion of KL; L_ex (exact reader gates) is | PROVED; C3: L_ex tracks the late KL rise, the gate-free model does not |
| 31.6 evidence | "drift along flat directions" via plain norm | WITHDRAWN as evidence (C6: plain norm is parametrisation-dependent); 31.6 stays as a proposition |
| 31.9 | solving L_ex with re-read gates | C4/C6: works, stops by held-out L_ex; at fixed n it over-fits (500 photos: it 4-6; 2000: it 8) |
| 31.10 | centring fixes the path | REFUTED (C6) |
**Reading.** What is left of gap C on photos is a DATA question (the solve over-fits at fixed n), not an objective
question. Data is exactly what network-made questions supply for free: once the questions are right (gap Q), the
solve can use as many as it needs.

## 32. Gap Q is 368 numbers
**Fact (C5 + CPU split).** The KFC step is 99.9% bias (A's constant entry 1 vs channel scale ~7.6e4) and the bias part
alone carries ~90% of the one-step gain. So for the one-step correction, the questions must reproduce ONE vector
c in R^368: c_j = constant column of g^cov at reader j = E[ sum_p (J^T Pi J e)(j, p) ] (same-image part), the
covariance, per reader, between "the output reads reader j here" and "the family's signal is lost here".
**Proposition 32.1 (bias-only closed form; PROVED).** Restricted to bias moves b in R^368, the second-order objective
is 1/2 b^T K_0 b + b^T c, with K_0 = E[(sum_p g)(sum_p g)^T] (the constant-field curvature, 368 x 368; NOT the KFC G,
which pairs same positions only). So b* = -K_0^-1 c: one 368 x 368 solve, no iteration.
**Measured (CPU, cached).** cos(c_q, c_photo8000): 256 photos .90, photo within-class .82, reader questions .58, class
pictures .40; wrong-sign readers 16% / 24% / 31% / 41%. On photos 20 readers hold half of |c|^2.
**Statement 32.2 (derivation of c from the network; CONJECTURE).** Write c as a sum over circuit events: c =
sum_e pi_e phi(e), where phi(e) = the same-image contribution of event e (a reader's circuit on with the family's
signal present) and pi_e = how much of it the training mix holds. Reader question j is the network's own picture of
event j, so phi(x_j) estimates the SHAPE of event j's contribution. The amount pi_j is a frequency, which no stationary
statistic of the weights holds (P-Thm 5), but the SGD imprint does: P-Thm 31 gives, for every ReLU neuron, D_j =
||u_j||^2 - ||v_j||^2 proportional to reader j's per-sample gradient energy under the training mix (checked .62 at 3b,
N6). Formula (no free parameter; the common scale is irrelevant by 28.4):
  c_formula = sum_j (D_j / E_j) phi(x_j),   E_j = reader j's own gradient energy on question j.
Prediction: cos(c_formula, c_photo) > .70 (uniform weighting: .58). Baselines named now: uniform weights (.58); weights
= |u_j|^2 alone; weights = D_j with no E_j normalisation. Margin: beat uniform by > .05. Earlier imprint uses failed
as a TARGET for reweighting (N8, infeasible) and for pixels (N9/N10, adversarial); here D is used only as event
frequency, which is the quantity Theorem 31 measures.

**Checks of 32.2 (CPU + one 32-s pass).** Imprint weighting: cos -.055 (uniform .504). **32.2 REFUTED**; reason measured:
a reader question's contribution is spread over all readers (its own reader holds 0.4% of it), so "question j = event
j" is false. Amplitude normalisations do not help (.40-.52).
**Proposition 32.3 (the question gap splits into shapes and frequencies; MEASURED, diagnostic).** Non-negative weights
over the 368 reader questions, fitted on half the readers' coordinates, reproduce the photo vector on the other half at
.78 (uniform weights .54; 256 photos .91). So of the one-step question gap, about two thirds is the FREQUENCY of the
network's own events (how much each question should count) and one third is SHAPE (events the questions do not make).
The fitted frequencies are spread (80% nonzero) and match no weights-only property tried (bias, norms, imprint, energies).
**What this means for Statement S.** The answers, the metric, and the event shapes come from the network. What is left
is a frequency vector over the network's own events: how often the training mix turns each one on. That is exactly
the object P-Theorem 5 says stationary weights cannot hold, and the one channel known to hold frequencies (the SGD
imprint) does not match it here. It is also exactly the "image mix" Shonu declared a product input on 2026-10-23 (the
developer re-mix; the generalist mix). The question for the generalist becomes sharp: is there a weights-only rule for
a 368-dimensional frequency vector, or is that vector the one declared input?
**Check (c58).** A photo-made c from n photos matches the 8000-photo c at .55 (32), .60 (64), .84 (128). The reader
questions at uniform weight (.54-.58) are therefore worth about 32-64 photos for the same-image term. Fitting their
frequencies to a few photos does worse than the photos themselves: the frequency vector is not cheaper to learn than
the term itself.

## 33. State at the end of the night (2026-10-24)
| Gap | Where it stands | Label |
|---|---|---|
| answers, metric | from the network (26.6) | DERIVED |
| C (solver) | closed-form exact-metric solve, derived stop: dev .0425 = 75% of S -> T on 2000 photos; the rest is data-limited (over-fit at fixed n); L_ex (exact reader gates) is the right objective | FITTED microscope; objective PROVED |
| Q (questions) | network-made one-step: dev -.0014, sealed -.0011 (58% of a 256-photo step); bias column carries ~90%; reader questions = 32-64 photos' worth; remainder = frequencies (~2/3) + shapes (~1/3) | DERIVED questions; frequencies OPEN |
| start S | still photo activity moments | MIX-DERIVED |
Open object, one sentence: how often the training mix turns on each of the network's own events. P-Thm 5 says
stationary weights cannot hold it; the SGD imprint does not match it (32.2); a few photos do not pin it faster than
the term itself. Decision for Shonu: declare it the mix input, or keep hunting a weights-only rule for it.
