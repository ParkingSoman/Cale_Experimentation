# Coarse neurons from weights: a working paper (curves in InceptionV1 3b)

Started 2026-10-23. Plan: `~/.claude/plans/first-thing-i-want-enchanted-sutherland.md`. Log cards: `CURVES-LOG.md`
section "WEIGHTS-ONLY (2026-10-23)". This file is rewritten after every result. It holds definitions, statements,
proofs, and the checks that ground them.

**Rule.** The final map Phi reads the WEIGHTS of the large network only. Images may only CHECK an assumption.
| Label | Meaning |
|---|---|
| DERIVED | computed from the weights only, by a rule written before the number |
| FITTED | least squares or statistics on images (what earlier sessions called "derived") |
| TRAINED | gradient descent on the output loss (a ruler or a microscope only) |

## 1. Setting

| Symbol | Meaning |
|---|---|
| X_q | the 5x5 window of the 3b 5x5-branch bottleneck at position q (32 x 5 x 5 = 800 numbers, all >= 0) |
| K_i, b_i | kernel and bias of curve neuron i (i = 1..10) |
| z_i(q) = K_i . X_q + b_i, f_i = relu(z_i) | pre- and post-activation |
| h_i = pool(f_i) | the 3x3 stride-2 max-pool that 4a reads |
| W_ji | weight from 3b channel i to 4a reader input j (368 readers over four 1x1 convs) |
| delta_j(p) = dl/du_j(p) | the loss's error signal at reader j, position p |
| e_i(q) = dl/df_i(q) | error signal at neuron i's output; g_i = 1[z_i > 0] e_i at its input |
| lam | the weight-decay constant of training |

Problem: a map Phi(weights) = (K', b', B, beta) for k coarse neurons that replace the family, with low
KL(intact || crushed) on natural images.

## 2. Theorem 1 (balance). PROVED, and it HOLDS on the weights

**Statement.** Let theta* be a stationary point of E[l] + (lam/2)||theta||^2. For every ReLU neuron whose output
feeds only linear maps and max-pools: ||K_i||^2 + b_i^2 = sum_j W_ji^2 (input norm^2 = output norm^2).

**Proof.** Replace (K_i, b_i) by (a K_i, a b_i) and column W_.i by W_.i / a, a > 0. relu(a z) = a relu(z) and
max-pool commutes with a positive scale, so the network computes the same function for every a. So E[l] does not
change along this curve. The regulariser changes by (lam/2)(a^2 (||K_i||^2 + b_i^2) + a^-2 ||W_.i||^2). At a
stationary point its derivative at a = 1 must vanish: lam (||K_i||^2 + b_i^2 - ||W_.i||^2) = 0. QED.

**Check (weights only, no images; `scratchpad balance.py`).**
| Layer | neurons | in^2 / out^2, median | middle half |
|---|---|---|---|
| 3b (out = all four 4a input convs) | 480 | 1.02 | 0.95 - 1.09 |
| 3a (out = all four 3b input convs) | 256 | 0.996 | 0.94 - 1.05 |
| curve neurons | 10 | 1.0 - 1.17 | |
| 3b neurons with bias^2 counted | 480 | 1.09 | |
| bottlenecks of 3a, 3b, 4a, bias^2 counted | 16-128 each | 0.76 - 0.91 | biases explain only ~0.05 of it |

Meaning: lucid InceptionV1 sits at a weight-decay balance point to about 5-9%. A plain gradient flow from the
usual random start would conserve in^2 - out^2 at its start value (~ -0.6 for a 3b neuron, rough estimate), not 0.
So the balance is evidence for weight decay at a stationary point. It is not a proof of it. Open hole: the
bottleneck units sit 10-25% below balance. So the premise of Theorem 2 is approximate (training stopped before
full balance, or the bottlenecks had a different decay). It is not exact.

## 3. Theorem 2 (the weights are data moments). PROVED under stationarity

**Statement.** At a stationary point (expectations over the TRAINING distribution, sums over positions):
| | Identity |
|---|---|
| (A) write | lam W_ji = - E[ sum_p delta_j(p) h_i(p) ] |
| (B) read | lam K_i = - E[ sum_q g_i(q) X_q ],  lam b_i = - E[ sum_q g_i(q) ] if biases are decayed |

**Proof.** The gradient of E[l] with respect to W_ji and K_i is the right-hand side with the opposite sign. Set the
gradient of the regularised loss to zero. QED. (Theorem 1 follows: contract (A) with W_.i and (B) with (K_i, b_i).
Both give -E[sum e_i f_i].)

**Corollaries (exact under the same premise).**
1. Importance. lam ||W_.i||^2 = -E[sum_q e_i f_i] = the fall in loss per unit of log-scale of neuron i. Weight norm^2
   IS the neuron's first-order importance.
2. The weight metric is a data metric. For any shift s: lam <K_i, T_s K_j> = -E[sum_q g_i(q) (z_j(q + s) - b_j)].
   The plain inner product of two kernels, one shifted, is the error-weighted co-activation of neuron i with the
   shifted neuron j. So the raw weight metric needs no whitening and no images. It already means something.
3. The weights only see data THROUGH GATES. Every identity carries 1[z_i > 0] of some existing neuron, or its
   activity h_i. Nothing in (A) or (B) sees a neuron's error signal where that neuron is off.

**Direct check: NOT possible on held-out images (card C2).** On 8000 validation images (train split, with mirror
flips, true labels), the mean loss gradient is stable (split-half cos .66 - .95) but almost at right angles to the
weights (cos(-g, theta) between -.08 and -.00 on nine layers). The prediction (cos >= .5) FAILED. Reading: the
identity is about the images the network was TRAINED on. On unseen images a large stable gradient is normal (the
generalisation gap), and our label map is rough (921 unique labels). We do not have the training images. So Theorem 2
rests on its proof plus the balance law (Theorem 1), which is its consequence and does hold.

## 4. The zeroth-order coarse neuron

**Lemma 3 (ReLU additivity).** relu(z_1 + ... + z_n) = relu(z_1) + ... + relu(z_n) exactly when no two of the z
have opposite signs. So one ReLU can carry the sum of a family only on inputs where the members agree in sign.

**Proposition 4 (zeroth order).** Suppose the crushed network is also a stationary point and (H1) its reader error
equals the intact one, delta' = delta, and (H2) the coarse gate equals each moved member's gate,
1[c(q) > 0] = 1[z_i(q + s_i) > 0]. Then the coarse neuron is
K' = sum_i T_{s_i} K_i, b' = sum_i b_i, B = sum_i W_.i (up to the scale gauge of Theorem 1).
**Proof.** Put c = sum_i z_i(. + s_i) and the write B = sum_i W_.i into (A) and (B) for the crushed net, use H1 and
H2 term by term, and use (A), (B) for each member. QED.

**Which shifts? (Corollary 2 + Lemma 3.)** The shifts should make the moved members agree as much as possible in the
error-weighted sense. By Corollary 2 that is: maximise ||sum_i T_{s_i} K_i||^2 over integer shifts. Weights only.
**Check C1 (weights only, no images; `c1_interference.py`).** Prediction: a ring, each member moved toward its own
centre of curvature, radius ~1 position. Result: energy 1.59x the unshifted sum. Direction toward the centre of
curvature on average (cos of mean angle error .997), but scattered (resultant .67, two members off by 80-95 deg).
Radius 1.5 positions (~12 px). The earlier session CHOSE r = 1 on dev images. Partly holds.

**But Proposition 4 is only part of the trained neuron (check C4, 100 images).** After removing means, the trained
k=1 neuron's pre-activation correlates with the C1 moved sum at .42 (r = 1 shifts: .43; unmoved sum: .15 - .22).
Two trained twins agree at .79. So the moved sum is a real part of the coarse neuron (moving doubles the correlation)
but it is about (.42/.79)^2 ~ 28% of the part that training reproduces. Prediction (>= .8): FAILED.
The moved sum's baseline is far below zero (mean -5,900, sd 2,400). It fires at 0.7% with bias 0 (H2 holds only on
loops). The trained neuron sits above zero (mean +350, sd 620): a dial, on ~70%.

**Correction to the plan (check C3).** The plan said "the trained kernel has 1.5% of its energy in the curve span, so
any linear mix of fine kernels is false". That was the raw weight metric on a kernel trained WITHOUT weight decay,
where parts the input never excites keep random start values. In the input's own metric (E[X X^T], 100 images) the
random-start trained kernel has 61% in the curve span and 93% in the span of all 96 5x5-branch kernels. Retracted.

## 5. Theorem 5 (blindness): the weights alone cannot fix the coarse network. PROVED

**Statement.** Let theta* be a local minimum of R_D(theta) = sum over the training set of l(theta; x, y)
+ (lam/2)||theta||^2, with l the cross-entropy. Take ANY distribution Q of extra input images, and label each one by
sampling y from the network's own answer p_theta*(y | x). Add them to the training set. Then theta* is still a
local minimum, with the same weights. But every plain average over inputs (family means, gated input covariances,
co-activations) changes, and so does the best coarse network for KL(intact || crushed) and for the training loss.
So no map Phi(weights) can give the best coarsening for every training set that is consistent with the weights.

**Proof.** For one extra image x, the expected gradient over its sampled label is
E_{y ~ p(.|x)}[grad log p(y | x)] = 0 (the score has mean zero). So the extra images add nothing to the gradient,
and theta* stays stationary. Their expected Hessian is the Fisher matrix of x, which is positive semi-definite, so
theta* stays a local minimum. The weights do not change. The plain moments are averages over the new input set, so
they change with Q. The KL objective of a coarse network is an average over inputs, so its minimiser changes with Q
for a generic Q. QED.

**In one line:** the weights are blind to every image the network already explains exactly (in distribution).
They store how the training data DIFFERS from the network (error-weighted moments, Theorem 2). They do not store
how often the network sees what it already gets right. The KL objective averages over exactly those images.

**The gate form (why the read fails, check C5).** For one neuron, Theorem 2 gives E[1[z_i > 0] e_i X] (error-weighted,
on its gate). The least-squares read that works needs E[X X^T 1[z_i(. + s_i) > 0]] K_i (plain, on the moved gate):
E[X relu(z)] = E[X X^T 1[z > 0]] K + b E[X 1[z > 0]]. For Gaussian inputs this is P(on) Sigma K (Bussgang), and the
read would be the plain moved sum. The inputs here are non-negative and sparse, so it is not. C5 measures the
cost of using the moved sum instead: +.116 dev KL (se .010) under the same write. That gap IS the gated plain moment
that no identity stores.

**What fixes the plain moments in principle.** Every layer's input distribution is the push-forward of the pixel
distribution through the weights below it. So weights + the pixel distribution determine every plain moment
exactly. Natural images have nearly universal pixel statistics (scale invariance, a 1/f^2 power spectrum,
occlusion: the dead-leaves model). This is the one place where something other than the weights must enter.

## 6. Status of the information audit (plan section 4)

| # | Quantity | Status |
|---|---|---|
| Q1 | metric for kernel inner products | DERIVED: the raw weight metric is the error-weighted data metric (Corollary 2) |
| Q2 | post-ReLU co-activation (Bayes betas) | open. Theorem 2 gives error-weighted, not plain, moments |
| Q3 | shifts (centres of curvature) | DERIVED in direction (C1), radius 1.5 vs CHOSEN 1 |
| Q4 | coarse kernel | zeroth order DERIVED (Prop. 4); it is ~28% of the trained neuron's reproducible signal |
| Q5 | bias | zeroth order = sum of member biases = consensus (0.7% on). The dial level is open (Theorem 5) |
| Q6-Q7 | write direction and gain | zeroth order B = sum of member columns; gain is a gauge at zeroth order, fixed only at second order |
| Q8 | reader coupling G | open. Neural Feature Ansatz test not run yet |
| new | means E[h_i] (the constant fill) | NOT in the dictionary. Even the .391 "constant fill" baseline is FITTED |

## 7. Score (dev unless marked; lower is better)

| Build | KL | Label |
|---|---|---|
| family removed | .59 - .64 | DERIVED (trivially) |
| constant fill | .391 | FITTED (needs the family means) |
| D1 (moved sum, consensus bias, Bayes write) | .366 | read DERIVED, write FITTED |
| C5: C1 moved sum, never clips, Fisher write (1500 img) | .357 | read DERIVED, write FITTED |
| C5: FITTED read, same write | .241 | FITTED |
| C5: C1 moved sum, consensus, same write | .845 | the instrument over-fits: a neuron on 0.6% of positions gives the 1500-image solve almost no informative rows. Not a verdict on the consensus read |
| fitted read + active-set Fisher write (sealed) | .135 | FITTED |
| trained k=1 (sealed) | .122 | TRAINED (ruler) |

---

# Part II: the declared mix (2026-10-23, plan `~/.claude/plans/i-d-like-you-to-squishy-yao.md`)

Theorem 5 says no map Phi(weights) is right for every training set. It says nothing against Phi(weights, nu), where
nu is a DECLARED input that says which images matter and what the app reads from the output. Shonu's decision: nu is
a product lever. Cale ships the generalist nu (the large model's own training mix). A developer may declare another.

Labels from here on:
| Label | Meaning |
|---|---|
| DERIVED | weights only |
| MIX-DERIVED | weights + averages over the declared nu, by a fixed, pre-declared sequence of closed-form solves |
| FITTED | anything chosen on dev images, or a solve iterated until it looks good |
| TRAINED | gradient descent on the output loss |

## 8. Definitions

**Def. 6 (capability, mix).** A capability is a = (Q_a, S_a, pi_a): a distribution Q_a of images, a set S_a of output
classes the app reads, and a weight pi_a > 0. Its damage on image x is D_a(x) = KL(p_S(x) || p'_S(x)), where p_S is the
intact network's answer restricted to S_a and renormalised, and p'_S the same for the crushed network. A declared mix
is a finite list of capabilities. Its objective is L(theta') = sum_a pi_a E_{x ~ Q_a} D_a(x).
- The generalist mix is one capability: Q = the training distribution, S = all classes. Its stand-in here is the
  train split (35 images per class). The trained k=1 ruler already minimises exactly this objective, so it IS the
  generalist bar ("the same mix, trained for the smaller model").
- "Taxi vs ambulance" is Q = taxi and ambulance photos, S = {cab, ambulance}.

**Lemma 8.1 (second order).** Let du(x) be the change of the reader inputs (all 368 readers, all 14 x 14 positions)
that the crush causes on image x. Then D_a(x) = 1/2 du^T F_a(x) du + O(|du|^3), with
F_a(x) = J_S^T (diag p_S - p_S p_S^T) J_S and J_S = d(logits on S_a) / du at the intact network.
**Proof.** KL(p || softmax(l + e)) restricted to S has value 0 and gradient 0 at e = 0, and Hessian diag p_S - p_S p_S^T.
So it is 1/2 e^T (diag p_S - p_S p_S^T) e + O(|e|^3). Put e = J_S du + O(|du|^2). The O(|du|^2) part of e meets only
the O(|e|) part of the expansion, which is zero, so it enters at O(|du|^3). QED.

**Corollary 8.1b (one mix holds every capability).** The second-order objective is
L2 = 1/2 integral du(x)^T F du(x) dnu(x, F), where nu = sum_a pi_a (Q_a pushed to pairs (x, F_a(x))).
So a list of capabilities is ONE measure nu on pairs (image, output metric). Each image carries its own metric: the
outputs its capability reads. This is the precise answer to "one Q cannot serve taxi-vs-ambulance AND nose-vs-mouth":
a plain image mix cannot, but a mix of (image, metric) pairs can, and it is still one object.

## 9. What the mix must supply

**Setting.** u = A z: what the family gives the readers (A = the family's 4a columns, z = pooled family outputs).
v = B c~ + beta: what k coarse neurons give them, c = relu(K' * X + b'), c~ = pool(c). du = A z - B c~ - beta.

**Theorem 7 (sufficiency).** For a FIXED read (K', b'), the write (B, beta) that minimises L2 solves N w = m, where
w stacks (B, beta), Phi(x) is the linear map w -> B c~(x) + beta, and
N = integral Phi^T F Phi dnu, m = integral Phi^T F A z dnu.
The least-squares read of a target y (for example the moved sum H) with position weights omega solves
G (K', b') = r, G = integral sum_q omega X~_q X~_q^T dnu, r = integral sum_q omega X~_q y_q dnu.
So the mix enters the build ONLY through (G, r) and (N, m). These are the moments M(nu).
**Proof.** Both objectives are quadratic in their unknowns. Set the gradient to zero. QED.

Meaning of each moment in this network:
| Moment | Size here (k = 1) | Meaning |
|---|---|---|
| G | 3873 x 3873 | how the 32 bottleneck channels co-fire in an 11 x 11 window, on this mix |
| r | 3873 | which window patterns come with curves on this mix (the moved sum H) |
| N | 736 x 736 | how much each reader's input, moved by the coarse neuron, changes the outputs this mix reads |
| m | 736 | the same, for what the 10 curve neurons gave those readers |

**Lemma 8.2 (mixes add).** Every moment is an integral against nu, so it is linear in nu:
M(sum_a pi_a nu_a) = sum_a pi_a M(nu_a), for a FIXED read. **Proof.** Linearity of the integral. QED.
- Consequence: store M for a library of basis pieces once (Compile). Any declared mix is a weight vector over the
  library. Its moments are a weighted sum, and the solves take seconds (Assemble). No image is touched again.
- Phi is not linear in M (it solves linear systems). Only M is.

**The read-write coupling.** N and m depend on the read through c~(x). A new mix gives a new read. So exact write
moments for the new read need a new pass over images. Lemma 8.3 removes that pass.

**Lemma 8.3 (compile the write once).** Fix a reference read c_0 (the generalist read). For any read c and any write
w, L2(c, w) = L2(c, w*(c)) + (w - w*(c))^T N_c (w - w*(c)) exactly, because L2 is quadratic in w. So the write
w*(c_0) computed from moments compiled with c_0 loses (w*(c_0) - w*(c))^T N_c (w*(c_0) - w*(c)) against the best
write for c. That is second order in how far the mix moves the read. **Proof.** Expand a quadratic around its minimum. QED.
- B0 measured how far the mix moves the read in 3b: pre-activation corr .98 between animals and the rest.
- So Compile can store (N, m) under the generalist read, per piece, and stay linear. Whether the second-order loss
  is small enough is a check, not a fact.

## 10. The effective mix per family, and the tilt

**Proposition 9 (effective mix per family).** Take k = 1 and a write whose column at each pooled position is a
fixed direction b^ (zeroth order: b^ = the sum of the member columns, Prop. 4, weights only). Ignore the coupling of
different positions in F (the diagonal-block approximation, to check). Then L2 depends on the coarse neuron only through
L2 = 1/2 integral sum_p omega(x, p) (c*(x, p) - c~(x, p))^2 dnu + const, with
omega(x, p) = b^T F_pp(x) b^ and c*(x, p) = b^T F_pp(x) (A z(x, p) - beta) / omega(x, p).
**Proof.** Put v = b^ c~ + beta in 1/2 du^T F du at each position, and complete the square in the scalar c~. QED.
- omega is the family's effective mix: each image and place, weighted by how much this coarse neuron's write moves
  the outputs that image's capability reads. A capability the family does not affect has omega near 0 and leaves the
  coarse neuron unchanged. So "nose vs mouth" weighs the families that tell noses from mouths, and no developer names
  a neuron.
- c* is the target the neuron should produce: the family's effect on the readers, projected on the write direction in
  the capability's output metric. The moved sum H is a weights-only stand-in for c*. Prop. 9 says what H stands in for.
- Open: the diagonal-block step. Pooled positions share later readers, so F couples them.

**Proposition 8 (tilt), restated after thinking it through.** The plan said: tilt Q toward images where child i fires,
and the coarse neuron becomes child i. With the target H fixed, that is false: an exponential tilt by child i's
activity also raises its co-firing neighbours (corr .10 - .24), and a point-mass limit just fits H on one image.
The correct statement needs the share of the family's activity that child i holds, r_i(x) = a_i(x) / sum_j a_j(x):
- If Q_t is Q tilted by exp(t r_i), and Q has images on which only child i is active, then as t grows Q_t
  concentrates there, H = child i's moved output there, and the target becomes child i.
- Whether the BUILD then returns child i depends on the read rule. Exact case E2 (new requirement on Phi, like
  "k = 10 returns the network"): on a mix where only member i is active, Phi must return member i.
- The linear least-squares read fails E2 in principle: it fits a linear function to relu(z_i), then clips it. On the
  set where relu(z_i) > 0 the target IS linear in X, so least squares gated to that set returns (K_i, b_i) exactly.
  So member consistency forces a gated read. To check (E2 below), then derive the gate for the full family.

## 11. What the coarse neuron should compute: the children's weights (Theorem 10)

Split the coarse neuron into two questions. WHAT should it compute (which combination of its children, and what it
writes)? HOW does one ReLU read that from the bottleneck? This section answers WHAT exactly. B0 says the mix barely
moves the HOW (read pre-activation corr .98 between animals and the rest), so the mix lever lives mostly here.

**Setting.** z(x, p) in R^n, n = 10: the children's pooled outputs at reader position p (all >= 0). The readers get
u = A z. A coarse neuron that computes a combination of its children gives c = relu(lam^T z) and the readers get b c.
F-bar: a fixed metric on reader inputs (hypothesis L2: the output metric does not depend on z; F-bar = identity is the
weights-only case, F-bar = the mix's average output Fisher is the mix case). The mix enters through
C(nu) = E_nu[z z^T] (10 x 10, the children's co-activity on the mix; uncentred, so it is entrywise >= 0).

**Lemma 10.1 (no clipping).** If lam >= 0 entrywise, relu(lam^T z) = lam^T z, because z >= 0.
**Proof.** A sum of non-negative numbers is non-negative. QED.
(This is Lemma 3 again: one ReLU carries a combination of its children exactly when they all agree in sign. On
OUTPUTS they always do. The difficulty of Lemma 3 lives entirely in the HOW.)

**Theorem 10 (k = 1).** Under L2, with lam >= 0, the best coarse neuron for E_nu (Az - b lam^T z)^T F-bar (Az - b lam^T z) is
- lam = the top eigenvector of P = A^T F-bar A C(nu), and
- b = A C lam / (lam^T C lam) (the children's columns, averaged with the weights C lam).
The loss is tr(A^T F-bar A C) - mu_1, where mu_1 is the top eigenvalue of P.
If A^T F-bar A and C are entrywise positive, P is entrywise positive, so by Perron-Frobenius lam > 0. Then the
constraint lam >= 0 is automatic and Lemma 10.1 makes the ReLU exact.
**Proof.** For fixed lam the loss is quadratic in b, minimised at b = A C lam / (lam^T C lam). Substituting leaves
tr(A^T F A C) - (lam^T C A^T F A C lam) / (lam^T C lam), a generalised Rayleigh quotient. Its maximiser solves
C A^T F A C lam = mu C lam, i.e. P lam = mu lam for invertible C. The top root wins. Perron-Frobenius for a positive
matrix gives a positive top eigenvector. QED.

**What this says about Shonu's lever.** The children's weights lam are fixed by TWO factors:
| Factor | Source | Meaning |
|---|---|---|
| A^T F-bar A (10 x 10) | the weights (and, through F-bar, which outputs the mix reads) | how much the readers need each child, and how much two children's messages overlap |
| C(nu) (10 x 10) | the declared mix | how often and how strongly each child fires, and with whom |
- If children never fire together (C diagonal, entries d_i), lam is the top eigenvector of (A^T F A) diag(d): a child
  that carries more energy on the mix gets more weight. "Represent common children more, rare ones less" is this.
- The whole mix enters through 55 numbers (C is symmetric 10 x 10), plus F-bar if the app reads only some outputs.
  Both are linear in nu (Lemma 8.2), so Assemble is an eigenvector of a 10 x 10 matrix: microseconds.
- The old "Bayes write" is the special case lam = 1 (equal weights, the moved sum H). Theorem 10 says equal weights
  are optimal only when 1 happens to be the top eigenvector of P.

**Theorem 11 (k coarse neurons).** Without the sign constraint, the best rank-k map is the top-k SVD of
Y = F-bar^(1/2) A C^(1/2) (Eckart-Young): write B = F-bar^(-1/2) U_k S_k, read Lam = C^(-1/2) V_k, loss = sum of
the remaining squared singular values. The loss depends on Lam only through the SPAN of its columns (B absorbs any
change of basis). So: **if the top-k span contains a basis of non-negative vectors, k ReLU coarse neurons reach the
unconstrained optimum exactly** (choose that basis, then Lemma 10.1).
**Proof.** Eckart-Young; then M = B Lam^T is unchanged by Lam -> Lam R, B -> B R^-T for invertible R. QED.

**Corollary 11.1 (rotation families: evenly spaced bumps).** If the family is a rotation orbit (A^T F A and C
circulant in the members' angles theta_i), the singular vectors are harmonics: h0 (constant), then the pairs
(cos theta, sin theta), (cos 2 theta, sin 2 theta), ... The span of h0 and h1 has the non-negative basis
1 + cos(theta - phi_j), phi_j = 0, 120, 240 deg (three bumps); h0 + one h1 component has the basis 1 +- cos(theta - phi).
For degree d, the 2d + 1 shifted Fejer kernels are a non-negative basis. So for a rotation family the best k = 1, 2, 3
(and k = 2d + 1) coarse neurons are exact ReLU units: k = 1 is h0, the "curves and loops" neuron; k = 3 is three
bumps 120 deg apart. **Prediction from the theory, not seen before: the best k = 3 are BUMPS (each child weighted by
1 + cos(theta_i - phi_j)), not three kept members.** The 2026-10-22 k = 3 build kept members (pruning) and lost to
training by .017.
**Proof.** A circulant matrix is diagonalised by the Fourier basis. The bump identities are trigonometry:
1 + cos(t - phi) >= 0, and three of them at 120 deg span {1, cos t, sin t}. Fejer: K_d(t) >= 0 and its 2d + 1 shifts
span the trigonometric polynomials of degree d. QED.

**What is still assumed.** L2 (one metric for all images: the Fisher work showed the per-image metric matters), and
that the shift by the moving (< 2 positions at 28 x 28) is below one pooled cell, so C of the unmoved pooled
children is the right C. The HOW (one ReLU reading lam^T z from the bottleneck) is the read problem of section 9.

**Check T1-T2 (log, MIX).** T1 grounded the structure (the h1 pair: 85-88%; the mix moves lam 4.9x beyond noise).
T2 FAILED: the trained twins agree with each other on lam at .83 but with Theorem 10 at only .46-.53, and one neuron in
Theorem 10's model carries only 44% of the family's message, while the trained neuron removes ~80% of the damage.
The false assumption: Theorem 10 has no per-reader constant beta. So its one neuron is spent on the family's MEAN,
which a constant does for free.

**Theorem 10b (with the per-reader constant).** Add beta in R^368 (constant over positions and images). Under L2,
beta = A mu - b lam^T mu (mu = E_nu z), and Theorem 10 holds with C replaced by the covariance
Sigma(nu) = C - mu mu^T: lam = top eigenvector of Phi Sigma, b = A Sigma lam / (lam^T Sigma lam), where
Phi = A^T F-bar A is the child-level metric. The loss splits:
| Part | Second-order share of the family's message |
|---|---|
| carried by beta | mu^T Phi mu / tr(Phi C) |
| carried by the neuron | mu_1(Phi Sigma) / tr(Phi C) |
| lost | the rest |
**Proof.** For fixed (b, lam) the loss is quadratic in beta. Its minimiser is the mean residual. Substituting
leaves Theorem 10's problem on centred z. QED.
- Lemma 10.1 still makes the ReLU exact if lam >= 0, now on lam^T z (the centring sits in beta, not in the ReLU).

**Check T3 FAILED.** With beta, one neuron carries .26 (beta) + .27 (neuron) = .53 of the message (trained ~.80 of
the damage). The centred optimum puts weights -.09 to -.11 on the three opposite-facing members, so it reports which
way a curve faces. The trained neuron keeps all weights >= 0 and reports how much curve there is. Twins agree at .83;
the theory agrees with them at .50-.57.

**What the three checks say about the assumptions.**
| Assumption | Status |
|---|---|
| L1: the coarse neuron computes a combination of its children | half true: R^2 .46-.49 for the trained neuron |
| L2: one output metric for all images | very likely false: 8% of images carry the damage, 48 of the top 50 are man-made, and the per-image Fisher write beat the constant-metric write (.176 vs .250) |
| b in span(A): it writes only where the children wrote | false for training: cos .39-.57 on the 1x1 and 5x5 convs (the "gate" on line readers) |
The theorems are correct for their model. The model is not the network's problem yet. The next conjecture must drop L2
first, because the damage itself says the metric changes from image to image.

**Check T4 (instrument fixed: moved children, lam >= 0).** The constrained optimum of Theorem 10b (exact, by
enumerating supports) is lam = (.06, .19, .04, .28, .15, .28, 0, 0, 0, .01) in s1 order. It correlates with the trained
neuron's children weights at .82 and .81; the two trained twins agree with each other at .78. HOLDS as stated.
RETRACTED (T4b): "predicts which children the neuron favours as well as training reproduces itself" overclaims. With
10 points the 95% interval of r = .82 is .39 - .96. Simple rules score .55 - .73 on the same target, and plain uniform
weights have cosine .93 / .85 with the trained weights (the theory: .85 / .89). The trained neuron is close to an even
mix of its children. So at k = 1 the theory's child weights are not yet distinguishable from simple rules by this
instrument. Only a functional test (paired KL on dev) can separate them. Still open: the
trained neuron is only R^2 .52 a combination of its moved children, and moving trades the HOW against the WHAT (the
readers read the children unmoved).

**Audit note (2026-10-23 afternoon).** The 2026-10-22 claim that the Fisher write "reproduces the training gate on
straight-line readers, which Bayes cannot see" is RETRACTED: the negative sign is already in the original family
columns (47 of 60 readers) and in the Bayes write (45 of 60). The Fisher write's paired KL gain stands. From here on,
every prediction names its simple-rule baselines and a margin before the number.

---

# Part III: beating the best trained network (2026-10-23 afternoon, plan `~/.claude/plans/okay-i-d-like-you-enumerated-wreath.md`)

Headline bar (Shonu's choice): the BEST KNOWN trained network of the same size. For curves k = 3 that is the network
trained from kept members 330/388/385 ("M"): sealed .037, dev-5000 .0361.

## 12. What "beat training" can mean

**Theorem 12 (gap split).** Fix the architecture (k coarse units), the mix nu and the loss L. Let L* be the infimum of
L over all networks of this architecture. For any build Phi and any trained network T:
L(Phi) - L(T) = A(Phi) - O(T) - S(T), with A(Phi) = L(Phi) - L* >= 0 (the build's approximation gap),
O(T) = L(T_inf) - L* >= 0 (the optimizer's gap: T_inf = what the same training reaches with unlimited images) and
S(T) = L(T) - L(T_inf) (the finite-image gap).
**Proof.** Add and subtract L* and L(T_inf). A and O are >= 0 by the definition of L*. QED.
- So a build beats training only where O + S > A. At k = 1 two different starts agree to .001 (sealed .1205 / .1215),
  so O is about 0 there and nothing can beat training by more than S.
- **Exactness test.** If A(Phi) = 0, then Phi is a global minimiser, so the gradient of L at Phi is 0 and training
  started at Phi cannot lower L. So any fall of L under training started at Phi is a lower bound on A(Phi).
  (The 2026-10-22 derived k = 3: .139 -> .037 under training, so A >= .10.)

**Theorem 13 (dead units stay dead).** If a coarse unit j has c_j(x, q) = relu(K'_j . X_q + b'_j) = 0 for every training
input and position, then the gradient of the training loss is 0 in K'_j, b'_j AND in its write column B_.j. So
plain gradient descent never revives it. (Adam keeps a decaying momentum, so it moves a dead unit only a bounded
amount after the death; in practice it stays dead, see the check.)
**Proof.** dL/dK'_j = E[sum_q e_j(q) 1[pre > 0] X_q] and dL/db'_j the same without X: every term carries the gate,
which is 0. dL/dB_.j = E[sum_p delta(p) c~_j(p)], and c~_j = pool(c_j) = 0. QED.
- **Check (I1).** The random-start k = 3 network (sealed .1217) has two units on at 0.0% of positions and one at 98%.
  It is a k = 1 network. This is the whole reason it "stalled at the k = 1 level" (k = 1: .1205). The old number .122 is
  NOT a k = 3 training reference. Any future random-start reference must keep all units alive first.

## 13. What training made at k = 3, and what the WHAT theorem does not decide

**Observation (I2, I3; saved model M).** Each of M's three units is a non-negative mix of ONE contiguous arc of members,
about 0 elsewhere: unit 0 330/340/349 (+379 small), unit 1 324/388/342 (+343), unit 2 385/406/379 (+343). The arcs
are close to the nearest-start-member cells of the start 10/120/250 deg. On the pooled children: R^2 .75-.81 UNMOVED
(moved .38-.61). So at k = 3 there is no centre-of-curvature move: the units are broad curve detectors, not loops.

**Statement 13.1 (span, not basis).** Theorem 11 fixes the best k-unit WHAT only up to the SPAN of the units' children
weights (any invertible change of basis is absorbed by the write). Corollary 11.1 chose 1 + cos bumps only because
they are a non-negative basis. M's units are NOT in the top-3 harmonic span: an arc indicator has strong higher
harmonics. So the WHAT theorem under-determines the build, and something outside it chose M's basis. Two candidates,
not yet separated:
1. the HOW: Lemma 3 says one ReLU carries a sum of members exactly where they agree in sign. Adjacent members co-fire
   (+.10 to +.24); members 90 deg or more apart do not. So narrow contiguous arcs are the bases one ReLU can compute.
2. inheritance: M started from three members, and its arcs sit on their nearest-member cells.
Check I5 separates them.

**Check I4 (read from the large network only).** Replace M's read by its least-squares projection on the members'
pre-activations (kernel sum_i w_i K_i, bias sum_i w_i b_i + b0), keep M's write: dev-5000 .0807 vs .0361, paired
+.0446 (se .0036). The fit used all positions, including deep negatives the ReLU never passes, so the instrument is
re-run on the on-set before this becomes a statement (I4b).

**Check I4b (instrument fixed: fit on the unit's on-set only).** .0697 vs M .0361, paired +.0337 (se .0030).
**Statement 13.2.** At k = 3 a read made only of the members' kernels (sum_i w_i K_i, any bias) loses ~.034 KL against
the trained read under the same write. The trained read uses input directions outside the members' span. This is the
k = 3 form of C5: for non-negative, sparse inputs the best ReLU read of a sum of member ReLUs is not in their span
(no Bussgang), so it needs the mix's gated input moments, which the weights do not hold (Theorem 5).

**Check U5 (WHAT shares, second order, no images).** Eckart-Young ceiling .563; 1 + cos bumps .532; M .512; best
contiguous arc partition .508; kept members .30-.34 (beta carries .260 in all). So the WHAT separates every broad
basis from kept members, and almost not among broad bases. The remaining choice is the HOW's.

## 14. The HOW: one ReLU on the bottleneck for a declared target

**Setting.** Unit j has a target t_j(x, q) = sum_i Lam_ji relu(z_i(x, q)) at every 28 x 28 position (unmoved: I2-I3
show the k = 3 units read their children unmoved). The unit is c_j = relu(v_j . X_q + b_j), X_q the 5 x 5 window.

**Lemma 14.1 (gated least squares).** For a fixed on-set S, the best (v, b) for sum over q in S of (v . X_q + b - t_j)^2
solves G_S (v, b) = r_S, with G_S = E_nu[sum_{q in S} X~_q X~_q^T], r_S = E_nu[sum_{q in S} X~_q t_j(q)], X~ = (X, 1).
**Proof.** On S the ReLU is the identity, so the problem is linear least squares. QED.
- Declared on-set: S_j = {q : t_j(q) > 0} (the unit should be on exactly where its target is). No knob.

**Proposition 14.2 (member consistency, exact case E2).** If Lam_j = e_i (the target is one member), then S_j = {z_i > 0}
and t_j = K_i . X + b_i on S_j, so Lemma 14.1 returns (K_i, b_i) exactly (when G_S is invertible).
**Proof.** The target is affine on S_j with residual 0, and the least-squares minimiser is unique. QED.

**Proposition 14.3 (when the HOW is exact).** The residual of Lemma 14.1 is 0 if and only if t_j is affine in X on S_j.
For t_j = sum_{i in arc} Lam_i relu(z_i), it is affine on S_j when every member of the arc keeps one sign on S_j
(Lemma 3). So the residual measures how far the arc's members are from firing together. Narrow arcs of co-firing
neighbours make it small. Wide bases (bumps, which put weight on members 90 deg apart that never co-fire) make it large.
This is the quantitative form of Statement 13.1, candidate 1.

**What this step does not charge (named before any run).** Positions with t_j = 0 where v . X + b > 0: the unit
fires falsely. Lemma 14.1 ignores them. It is the read-side twin of the false waking that cost the plain Fisher write
2x its prediction (W2e). If the build fails, check this first.

**Checks H, I5 (the HOW, and where the partition comes from).**
| Read (same lsq write on all) | realized share | false-on | dev-5000 KL | paired vs M's trained read | label |
|---|---|---|---|---|---|
| M's trained read | .423 | | .0784 | 0 | TRAINED |
| arc partitions #1-#4 of U5, Lemma 14.1 | .429-.438 | .25-.30 | .0748-.0837 | -.0035 to +.0053 | MIX-DERIVED |
| 1 + cos bumps, Lemma 14.1 | .410 | .70 | .0859 | +.0075 (.0023) | MIX-DERIVED |
| kept members (Lemma 14.1 returns them exactly) | .317 | 0 | .1325 | +.0542 (.0046) | DERIVED |
- Prop. 14.2 holds in code (kernel cos 1.0000). Prop. 14.3 holds as predicted: bumps have the best WHAT but the worst
  realized share among broad bases, because they fire falsely on 70% of their off-target positions.
- I5: training started from members 50/170/290 grows arcs around THOSE members. The partition is inherited from the
  start, and U5 shows many partitions within .01 of the best. So which partition is a free choice (gauge-like), and the
  derivation need not spend itself on it.
**Statement 14.4.** At k = 3 the read is closed-form: target from the WHAT (arc mix of unmoved children), on-set =
where the target is positive, one gated least-squares solve on the generalist mix. Under a common write it is as good
as the trained read (best -.0035, se .0022). No knob. Label MIX-DERIVED.

## 15. The write at k = 3 (open)
**Check W3.** On M's own read, the active-set Fisher write (4000 train images; THE pre-registered build) gives .0682
vs the trained write .0361 (paired +.0321, se .0025). It beats the lsq simple rule by .015 (4 se). At k = 1 the same
write was .013 short of training; at k = 3 it is .032 short. 4000 images for 1472 unknowns is 2.7 images per unknown
(k = 1 used 16-48). Diagnosis below decides estimation versus theory.

---

# Part IV: the generalist with no images (2026-10-23 evening, plan `~/.claude/plans/i-want-you-to-majestic-milner.md`)

**Rule (Shonu).** The GENERALIST build uses no image data: not in the read, the write or the mix. Images only CHECK a
statement. (Developer capabilities may use images later; Part II stays for them.) The generalist bar is unchanged:
"the same training data, a smaller network" = the trained rulers (k = 1 sealed .122; k = 3 bar B dev .0361).
| Label | Meaning |
|---|---|
| DERIVED | weights + file constants only |
| LAW-DERIVED | weights + a stated law of natural images, no image set (only with Shonu's OK) |
| FITTED | any average or solve over an image set. Every Part II-III build is FITTED here (THE FORMULA: .0507) |
| TRAINED | ruler or microscope |

**What the file holds (checked).** 58 weight and 58 bias tensors. Architecture constants (kernel sizes, strides,
pools, LRN: size 9, alpha 1e-4, beta 0.5, k 1). One plain training average: the input offset 117, the ImageNet mean
pixel. No BatchNorm statistics. So InceptionV1 is the hardest case for "the weights hold the data".

**Theorem 5 still stands, and covers the read too.** It is about every plain average over inputs, so it applies to the
read's gated window statistics as much as to the write's moments (C5, I4b measured both). What it does not forbid: a
rule that picks one canonical input distribution from the weights, or a law that turns error-weighted averages
(Theorem 2) into plain ones. Part IV looks for both. Fallback order: the weights (conjectures A-D), then inputs the
network makes for itself (E), then a law of natural images (only with Shonu's OK).

## 16. The strict build: Proposition 4 alone (D1)

**Build.** Read = moved sum of member kernels (C1 shifts), bias = sum of member biases, write = sum of member columns.
No number is free: Theorem 1's balance fixes only the gauge (read x a, write / a), not the product. (The plan said
balance would set the gain. That was wrong and is withdrawn.)

**Check D1 (dev-5000).** Removed .664. Moved .754 (on 0.57% of positions). Paired +.090 (se .024): WORSE than removing
the family. Predicted .50-.62: FAILED.

**Statement 16.1 (why: cross-talk of the column sum).** Where the unit fires, readers receive (sum_i W_.i) c~ with c~
of the order of one member's output. When only member j is truly active, the intact network sends W_.j h_j. The error
is (sum_{i != j} W_.i) c~. Its energy relative to the signal is ||sum_{i != j} W_.i||^2 / ||W_.j||^2, a weights-only
number: 14.7 - 19.9 for the ten members (column cosines mean .11, range -.43 to .65). Proposition 4 is exact only under
its premise H2 (all members on together). The family's members mostly fire apart, so H2 is false almost everywhere.
**Proof.** Substitute the build into u' = B c~ and subtract u = sum_i W_.i h_i at a position where h_i = 0 for i != j. QED.

**Corollary 16.2 (the write must be a weighted mean).** Theorem 10 with uncorrelated children (C = diag(d)) and equal
children weights gives b = sum_i d_i W_.i / sum_i d_i: the activity-weighted MEAN of the columns. The column SUM is
right only when every member fires with every other. So the strict build lacks exactly the activity shares d_i, the
"share of data each child has seen". Conjecture A asks whether the weights carry them.

**Side fact.** The moved unit is balanced like a trained neuron (in^2/out^2 = 1.094; members 1.0-1.17). The unmoved
sum is not (.753). The C1 shifts therefore also make the coarse unit consistent with Theorem 1.

## 17. What the weights say about the data's shares (conjectures A and B)

**Conjecture A (weight norm = activity share): FALSE in general.** Corollary 1 makes ||W_.i||^2 the error-weighted
importance. Across the 480 neurons of 3b it does not rank the plain downstream energy E[f^2] ||W_.i||^2 (Spearman .29;
predicted >= .7), and E[f^2] is not constant either (CV .68). So the norms do not convert error-weighted activity into
plain activity. **Exception that matters here:** inside the curve orbit both are near uniform (norm shares .092-.110,
true energy shares .062-.128). For a rotation family, equal shares is what the weights say and roughly what is true.

**Conjecture B (kernel overlap = co-firing): HOLDS in rank.** The max kernel cosine over shifts up to 2 positions ranks
the 45 member pairs like their plain pooled correlation (Spearman .76; angle-gap rule .48). So the weights carry WHO
fires with whom, not HOW OFTEN.

**Statement 17.1 (what transfers).** Error weighting (Theorem 2) preserves the pattern of co-firing (a property of
the kernels' geometry) but not activity magnitudes (a property of how often inputs occur). This is Theorem 5 seen in
the numbers: frequency is exactly what the weights are blind to.

**The class shares are visible.** The last layer's biases imply a near-uniform class prior (softmax of the biases:
entropy 6.901 of the maximum 6.916; max/min ratio 3.3), as ImageNet's balanced classes would give. Row norms are even
too (CV .14). So at the coarsest level, the weights SUGGEST even class shares.
This supports a self-made input set with classes spread evenly (section 18). It is an observation from the
biases, not a proved identity: the weight rows also shape the class prior.

## 18. Inputs the network makes for itself (check E)

**Rule (fixed before any number).** One input per class, 256 classes evenly spaced over the logits (255 are real
ImageNet classes). Maximize that class's logit on a plain pixel image: no 1/f law, no ImageNet colour matrix.
Lucent's standard transforms, Adam lr .05, 256 steps. Nothing was tuned on images. It is still a fixed rule with chosen
numbers, not a theorem. Whether a published, fixed generation rule counts as DERIVED is Shonu's decision.

**Result: the READ side transfers.** On 256 self-made inputs vs real train images:
| Quantity | self-made vs real |
|---|---|
| children co-firing pattern (45 pairs) | Pearson .951 |
| member on-rates | ratio .89 - 1.35 |
| Lemma 14.1 read (k = 3, arcs #2), kernel cosine | .875 - .916 |
| read + lsq write, both from the inputs (dev-5000) | self .0696, real at equal count .0740 (paired -.0044, se .0017) |

**Retraction (my plan's pushback).** The plan said the read was where I expected the most trouble without data, and
pushed back on Shonu's view that the read could be done without images. Checks E1-E3 show he was right: the read
transfers fully through the network's own inputs. The write failed instead.

**Result: the output metric does NOT transfer.** The Fisher write on self-made inputs is .524 vs .174 for real inputs
(equal count 256; both over-fit at this size). The network is certain on its own inputs (mean 1 - sum p^2 = .0049 vs
.125 on real images).

**Statement 18.1 (what self-made inputs can and cannot supply).** Class inversions give the network's picture of
WHAT IS IN its training images: low-level structure, who fires with whom, how often. That is all the read needs (B0
already said the read barely depends on which natural images). They cannot give WHERE THE NETWORK IS UNSURE, because
the rule drives each input to certainty. The output metric (diag p - p p^T) vanishes there. Theorem 5 predicts this:
the inputs the network explains perfectly are exactly the inputs its weights are blind to.
**Proof sketch.** At p = e_y, diag p - p p^T = 0, so F(x) = J^T 0 J = 0. Near certainty it scales with 1 - sum p^2. QED.

**What would fix it (not run; for Shonu).** Theorem 2 says the weights store error-weighted moments, i.e. where the
network is unsure. So the matching self-made inputs are boundary inputs: for class a and its nearest class b by the
last layer's weights, maximize log p_a + log p_b (target 50/50). This keeps the rule weights-only and targets exactly
what the weights encode. Open: the share of boundary inputs in the mix is itself a plain frequency (Theorem 5).

**Score (dev-5000; k = 3).**
| Build | KL | Label |
|---|---|---|
| D1 strict (Prop. 4) | .754 (removal .664) | DERIVED |
| self-made read + self-made lsq write | **.0696** | DERIVED if the generation rule counts (section 18) |
| real read + real lsq, same 256 images | .0740 | FITTED |
| THE FORMULA (real arcs read + Fisher step1, 35000 images) | .0507 | FITTED |
| bar B (best trained k = 3) | .0361 | TRAINED |

---

# Part V: inputs from the network itself (P1), then the coarse network from them (P2) (2026-10-23 night)

Shonu's framing: P1 = a formula that makes inputs from ANY trained network, with no image set. P2 = the coarse
network built from those inputs. A density defined from the network alone plus a standard sampler counts as a formula,
if doubling the sampler's steps changes no reported number by more than 2 paired se (otherwise the setting is CHOSEN).

## 19. What the network fixes, and what it does not

**Theorem 19 (decomposition).** Any labeled input distribution factors as p(x, y) = p(y | x) p(x).
1. The network gives a conditional p_theta(y | x) = softmax(logits(x)). For P2 this is the TARGET: the coarse network
   must reproduce the network's answers, so whether p_theta equals the data's conditional is irrelevant to P2.
2. The weights do not fix p(x) (Theorem 5: self-labeled extra inputs change p(x) and leave every weight in place).
3. So the network gives everything P2 needs except one object: the distribution of inputs p(x).
**Proof.** 1 is the definition of the classifier and of the KL objective. 2 is Theorem 5. 3 follows. QED.

**Check I1 (calibration, train split).** p_theta is strongly over-confident on unseen images: top-1 .656, mean
confidence .912, ECE .256 (predicted .02-.08: FAILED). So p_theta is not the data's conditional here. By part 1, this
does not affect P2.

**Lemma 19.1 (maximizing vs sampling).** Let q be any input distribution with class shares Z_y = E_q[p_theta(y | x)].
(a) If y ~ Z and x | y has density p_theta(y | x) q(x) / Z_y, then x ~ q and y | x ~ p_theta(y | x) exactly. So the
generated set carries the network's own uncertainty AS IT IS UNDER q.
(b) If instead x = argmax_x log p_theta(y | x) (+ any bounded prior term), then as the logit gap grows the input goes to
certainty: 1 - sum p^2 -> 0. This holds for every q.
**Proof.** (a) The joint is p_theta(y | x) q(x): sum over y gives q(x), divide by it to get the conditional. (b) The
maximiser of a logit that is unbounded above has p_y -> 1. With 256 steps of maximizing, check E measured mean
1 - sum p^2 = .0049 against .125 on real images. QED.
**Consequence.** Maximizing is ruled out for any write that uses the output metric (the E6 failure is (b)). Sampling is
necessary, and the prior q decides how unsure the inputs are. q is the one open object of P1.
(Correction: the plan claimed sampling reproduces the real uncertainty "for any q". That is false and is withdrawn.)

**Proposition 19.2 (the Fisher write is blind to sure inputs).** The write moments of Theorem 7 are
N = integral Phi^T F Phi dp and m = integral Phi^T F A z dp, with F(x) = J^T (diag p - p p^T) J. Where the network is
sure (p near a vertex), diag p - p p^T -> 0, so those inputs add nothing to N or m. Hence the write depends on p(x)
only through its unsure part, and adding any amount of sure inputs leaves the write unchanged (up to a common scale,
which the solve ignores).
**Proof.** ||diag p - p p^T|| <= 1 - max_y p_y, so the integrand vanishes as p goes to a vertex. QED.
**Checks I2, I3 (step1 write on M's read, dev-5000).** 4000 unsure inputs (u = 1 - sum p^2 > .1): .0698. 4000 random:
.171. 4000 sure: .230. All 10498 unsure inputs: .0454, equal to all 35000 inputs (.0443; paired +.0010, se .0012).
**Consequence for P1.** The write needs unsure inputs only. Their share in the mix does not matter. The read needs
the natural low-level structure that class inversions already give (section 18). So P1 has a precise target: a formula
for UNSURE inputs with natural structure.

**Check I4.** (a) The logit level LSE(x) = log sum exp(logits) orders real (55) > pixel-shuffled (26) > noise (18), so it
carries density information, as energy-based out-of-distribution work reports. But class inversions reach 491: the
level is unbounded above, so a Boltzmann prior exp(LSE) would be ruled by extreme, sure inputs (Prop. 19.2 says the
write cannot use those). (b) The conv1 filters spread their energy evenly over the real pixel-patch eigen-directions
(slope -.04): the first layer does not store the pixel spectrum. Both predictions about a ready-made prior inside the
network fail as stated.

## 20. Where the information about p(x) actually sits (derivation from I1-I4)

Three facts now line up.
1. Theorem 2: lam theta = - E[grad_theta l(x, y)]. The weights ARE an average of per-input gradients. A per-input
   gradient is near zero on inputs the network gets right with certainty (p_y -> 1 gives p - e_y -> 0). So the weights
   are an average over the inputs where the network was unsure or wrong.
2. Proposition 19.2: the Fisher write uses only unsure inputs.
3. Theorem 5: the weights are blind to inputs the network explains exactly. The sure, correct inputs are such inputs.

So the part of p(x) that the write needs and the part that the weights record overlap: both are the unsure inputs.
The overlap is not exact. Self-labeled UNSURE inputs (Theorem 5's construction, labels drawn from p_theta) add zero mean
gradient but non-zero Fisher. So the weights record the ERROR field (a mean), while the write needs the uncertainty
(a variance). This gap must be measured (I5), not assumed away.

**The derived generator (conjecture G).** Choose inputs (and labels) whose gradients reproduce the weights:
find a set S = {(x_s, y_s, c_s)} with sum_s c_s grad_theta l(theta; x_s, y_s) = - lam theta, and among all such sets
the one of maximal entropy (or minimal size). This is Theorem 2 read backwards. It needs only the weights (and the
decay constant lam, a training setting). The density form: the maximum-entropy distribution under the moment
constraint E[grad l] = - lam theta is the exponential family p(x, y) proportional to q0(x, y) exp(mu . grad_theta l(x, y)),
with one multiplier mu per weight. Its energy is a directional derivative of the loss in weight space: two forward
passes per input.

**Literature (checked 2026-10-23: arXiv 2206.07758, 2307.01827, 2602.08723).** Haim et al. 2022 (NeurIPS), "Reconstructing Training Data from Trained Neural
Networks": the stationarity (KKT) condition of the max-margin problem lets one rebuild actual training images from the
weights of small networks. Buzaglo et al. 2023, "Deconstructing Data Reconstruction": extends this to many classes,
weight decay and general losses. The images they recover are the high-gradient (margin) inputs: in our language, the
unsure inputs that Proposition 19.2 says the write needs. Gradient-matching dataset condensation (Zhao et al. 2021)
matches the gradients of real data. Conjecture G matches the stationarity condition instead, with no data.
Buzaglo et al. also report that weight decay makes reconstruction EASIER (InceptionV1 was trained with weight decay;
Theorem 1 balance holds). A 2026 note (arXiv 2602.08723) shows the stationarity equations can have many solutions:
Theorem 5's non-uniqueness, seen from the reconstruction side. So conjecture G needs a selection rule (max entropy).

## 21. Why "match the weights" collapses (the plateau theorem)

**Setting.** For a labeled input set nu (inputs x, labels y), the per-input gradient is grad_theta l = J(x)^T (p(x) - e_y),
with J = d logits / d theta. Conjecture G asks for nu with E_nu[J^T (p - e_y)] = - lam theta. Write the residual
R(nu) = || lam theta + E_nu[J^T (p - e_y)] || / || lam theta ||.

**Lemma 21.1 (only label disagreement counts).** Let q(. | x) = E_nu[e_y | x] be the generated label distribution at x,
and d(x) = q(. | x) - p(x) its disagreement with the network. Then E_nu[grad l] = - E_x[J(x)^T d(x)].
**Proof.** Average J^T (p - e_y) over y given x. QED.

**Theorem 21 (plateau).** Every nu whose labels agree with the network (d = 0: self-labeled sets, or inputs on which the
network is certain of the generated label) has R(nu) = 1 exactly. This set is huge (any input distribution, labeled by
the network). (Withdrawn: an earlier draft said the plateau is "flat at the scale of lam". That is unproved; the
gradient of R^2 off the set can be large. What traps descent is that R = 1 exactly on a huge set that attracts every
path from R ~ 10^4, and leaving it needs all inputs' contributions to line up along theta at once.)
**Proof.** d = 0 gives E[grad l] = 0 and R = ||lam theta|| / ||lam theta|| = 1. QED.
- This is Theorem 5 as a landscape. The I5 runs found exactly the plateau: c -> 0 (no inputs), then sure inputs
  (d -> 0), then the warm start pushed into certainty. From a random start the residual is ~10^4, and the fastest
  way down is to shrink d, which lands on the plateau. The optimizer did what the theorem says it will.
- So matching the mean gradient cannot work by descent from an arbitrary start. The formula must say directly which
  disagreement field d the data had. That is what section 22 derives.

**Remark 21.2 (the cause, and the formulation it forces).** With cross-entropy,
grad_theta l(x, y) = - sum_{k != y} p_k(x) (grad f_y - grad f_k)(x). The coefficients p_k(x) are TIED to the input: they
go to 0 as the input becomes sure. That tie is the escape route of every G2 run. The reconstruction literature uses
the multiclass KKT form with FREE coefficients instead:
  lam theta = sum_s sum_{k != y_s} c_{s,k} (grad f_{y_s} - grad f_k)(x_s),   c >= 0.
A pairwise margin gradient does not vanish on sure inputs, so there is no plateau. At a true cross-entropy +
weight-decay stationary point c_{s,k} = p_k(x_s) / N. So "c agrees with p" becomes a CHECK after solving, not a
constraint the optimizer can collapse. This is the next card (not run).

**Selection principle (a decision, stated as math).** The stationarity equations have many solutions (Theorem 5;
arXiv 2602.08723). Two opposite rules pick one:
| Rule | Picks | Known behaviour |
|---|---|---|
| maximum entropy (section 22) | the least-structured input set consistent with the weights | leans to the prior; with a flat prior, noise (Cor. 22.2) |
| minimal support (Haim, Buzaglo: ~2N candidates) | the smallest input set consistent with the weights | recovers actual training images on small networks |
Which one is "the network's own data" is Shonu's call.

## 22. The first-order maximum-entropy generator

**Lemma 22.1 (Euler identity for ReLU networks with biases).** For a network of L layers (ReLU, max-pool, linear),
logits are unchanged in form under W_l -> a W_l, b_l -> a^l b_l, and scale as a^L. So with the Euler direction
v = (W_l, l b_l)_l: v . J(x)^T = L f(x)^T (f = logits). Contracting stationarity with v:
**E_train[ f_y(x) - E_{p(x)}[f(x)] ] = lam sum_l (||W_l||^2 + l ||b_l||^2) / L =: kappa.**
**Proof.** Differentiate f(a-scaled) = a^L f at a = 1. Then v . E[J^T (p - e_y)] = L E[p . f - f_y] = - lam v . theta. QED.
- Meaning: the training labels' logit beats the network's own average logit by kappa on average. kappa needs only the
  weights and lam. On a self-labeled set the left side is 0 (Theorem 21). kappa > 0 is the ONE scalar that measures how
  far the training data sat from the plateau.

**Theorem 22 (maximum entropy under the Euler constraint).** Among all labeled sets with E[f_y - E_p f] = kappa, the one
of maximal entropy relative to a prior q0(x) x uniform(y) is
  p*(x, y) proportional to q0(x) exp(beta (f_y(x) - E_{p(x)} f(x))),
with beta set by the constraint. Its parts:
| Part | Formula | Meaning |
|---|---|---|
| labels given x | softmax(beta f(x)) | the network's own answer at inverse temperature beta (sharper if beta > 1) |
| inputs | q0(x) exp(LSE(beta f(x)) - beta E_p f(x)) | a bounded tilt of the prior |
| beta = 1 | q0(x) exp(H(p(x))) | since LSE(f) - E_p f = H(p): the tilt favours UNSURE inputs, by e^(entropy) |
**Proof.** Maximise H(nu) - mu (E_nu[g] - kappa) with g = f_y - E_p f: the solution is q0 exp(mu g) (Gibbs). Summing
over y gives the input part. At beta = 1, log sum_y e^{f_y} - sum_y p_y f_y = - sum_y p_y log p_y. QED.
- Relation to G1 (flat Langevin, y uniform, x proportional to p(y|x)): at beta = 1, p* = G1 x exp(H(p(x))). The derived
  tilt pushes G1's samples toward inputs where the network is unsure. That is what Proposition 19.2 says the write needs.
- Unlike exp(LSE) (check I4: unbounded), the tilt is bounded by log(number of classes) at beta = 1.
- Still open: q0. Theorem 22 is first order: it keeps only the projection of stationarity on v, not all |theta|
  constraints.

**Checks (written before running), on the toy where the train set is known.**
- X1 (exact case, Lemma 22.1): on the toy train set, E[f_y - E_p f] equals kappa within 10% (the toy residual is .24,
  so not exact).
- X2 (Theorem 22, flat q0, beta from the constraint): samples of p* give a Fisher write closer to truth than G1 (.160),
  by more than 2 paired se. And their mean 1 - sum p^2 is at least truth's (.116).

**Check X1 (toy, exact case): HOLDS.** On the toy train set, E[f_y - E_p f] = .0331 (se .0018) and kappa from the
weights = .0331: ratio 1.000. (Instrument: v . J^T 1 = 12.5334 vs L sum f = 12.5335.) The full stationarity residual
of the toy is .24, but its projection on the Euler direction is exact, as LBFGS converges fastest along the scale.
On fresh true samples the same excess is -.267: the network is over-confident off its training set (as I1 found for
InceptionV1). So the weights pin one exact scalar about the training data: how far its labels sat above the
network's own answers.

**Corollary 22.2 (the first-order tilt cannot create structure).** At beta = 1 the input part of p* is
q0(x) e^{H(p(x))}, and 1 <= e^H <= K (K classes). So p* and q0 differ in density by at most a factor K everywhere.
Near beta = 1 the bound is close to K. With a flat q0 the samples are still essentially flat noise.
**Proof.** 0 <= H <= log K. QED.
- So X2 with a flat prior is predicted to FAIL by the theory, and is not run. The Euler scalar decides WHICH inputs
  among those q0 offers (unsure ones, by up to K-fold). It cannot decide WHAT natural inputs look like.
- Where pixel-level structure can come from: the remaining |theta| - 1 constraints. The full maximum-entropy family is
  p(x, y) proportional to q0(x) exp(- mu . grad_theta l(x, y)), with one multiplier per weight. The constraints of the
  FIRST layer act on grad_{W_1} l = sum_q g_1(q) X_q^T: gated errors times pixel patches. Only through them do the
  weights shape what pixels look like. This is the next mathematical object: the dual problem for mu, layer by layer,
  starting from conv1.

**Status of P1 after the toy.**
| Statement | Status |
|---|---|
| Theorem 19 (network fixes p(y given x), not p(x)) | proved |
| Lemma 19.1 (maximizing destroys uncertainty) | proved; G0 Fisher write = no repair on the toy |
| Prop. 19.2 (the write ignores sure inputs) | proved; I2/I3 on InceptionV1 |
| Theorem 21 (plateau: matching the mean gradient collapses) | proved; explains all four G2 failures |
| Lemma 22.1 (Euler scalar kappa from weights) | proved; X1 exact on the toy (ratio 1.000) |
| Theorem 22 (first-order max-ent generator) | proved; bounded tilt (Cor. 22.2), so q0 still decides structure |
| Full max-ent family (all multipliers) | open |
| KKT form with free coefficients (Remark 21.2) | the plateau-free next card, not run |

**Blockers before Lemma 22.1 touches InceptionV1.** (1) The two LRN layers (k = 1) are not positively homogeneous, so
the Euler direction needs a correction there. (2) lam is a training setting, not in the file (the Caffe GoogLeNet
solver is reported to use 0.0002: to verify, not assume).

## 23. The network rebuilds its own unsure inputs (Shonu's conjecture as a theorem)

Shonu (2026-10-23): rebuilding real training images is acceptable, because it uses only the network's own
properties. His conjecture: the network knows what it is unsure about, so a general method can create those inputs.

**Theorem 23.** Let theta* be a stationary point of (1/N) sum_i l(theta; x_i, y_i) + (lam/2) ||theta||^2, l the
cross-entropy, ANY architecture. Then
  lam theta* = (1/N) sum_i sum_{k != y_i} p_k(x_i) (grad f_{y_i} - grad f_k)(x_i).
1. Each training input enters the weights through pairwise terms whose coefficients p_k(x_i) / N are its own
   uncertainty toward each wrong class. A sure input (p_{y_i} -> 1) enters with coefficients -> 0. The weights are a
   sum over the unsure training inputs, each weighted by how unsure it is. ("The network knows what it is unsure about.")
2. The write's moments (Theorem 7) weight each input by diag p - p p^T, which also vanishes on sure inputs
   (Prop. 19.2). So what the weights record and what the write needs live on the same inputs.
3. Replace p_k(x_i) / N by free coefficients c_{s,k} >= 0. Then lam and N are absorbed into c: the reconstruction
   problem "find (x_s, y_s, c_s) with sum c_{s,k} (grad f_{y_s} - grad f_k)(x_s) = lam theta" needs the weights only,
   up to the overall scale of c. (This removes the decay-constant blocker for InceptionV1.)
4. At a true stationary point, c_{s,k} / c_{s,k'} = p_k(x_s) / p_k'(x_s): a CHECK after solving.
**Proof.** For cross-entropy, grad l = sum_k p_k grad f_k - grad f_y = sum_{k != y} p_k (grad f_k - grad f_y), since
sum_k p_k = 1. Set the gradient of the regularised objective to 0. Parts 1-4 read off the identity. QED.

**Corollary 23.1 (no plateau).** Let R(x, c) = || lam theta - sum c_{s,k} g_{s,k} || / || lam theta ||, with
g_{s,k} = (grad f_{y_s} - grad f_k)(x_s). At c = 0, R = 1. If some pair has < theta, g_{s,k} > > 0, then increasing that
c_{s,k} from 0 lowers R strictly. For a positively homogeneous network, < v, g_{s,k} > = L (f_{y_s} - f_k)(x_s) (Lemma
22.1), which is > 0 whenever y_s is the network's top class at x_s. So c = 0 is not a resting point as long as the
candidates' labels are their top class (exact in the Euler metric v; approximate in theta for networks with biases).
And g_{s,k} does not vanish on sure inputs: a margin gradient grows with the logit gap. So the escape route of
Theorem 21 (make inputs sure, so their terms vanish) is closed.
**Proof.** R^2(t e_{s,k}) = 1 - 2t < lam theta, g > / ||lam theta||^2 + O(t^2). The rest is Lemma 22.1. QED.

**Checks T1-T5 (toy, m = 1000, 3000 steps).** T1 (instrument) .2411 = the floor. T2 FAILED: residual 1.05 (min .94).
T3 FAILED: no training image rebuilt. T4 FAILED: c not proportional to p (-.06). T5: simple write .080 (truth .034), the
best of any generator; Fisher write .189 (truth .016), worse than G1. Co-firing with the truth .82, the best so far.

**Decomposition (no new run).**
| Inputs, coefficients | residual |
|---|---|
| true set, true c = p_k / (N lam) | .241 |
| true set, best c (non-negative least squares) | .047 |
| true set, only the top 10 / 100 / 300 contributing points | 3.26 / 7.53 / 3.75 |
| rebuilt inputs, the optimizer's c | 1.106 |
| rebuilt inputs, best c for the same inputs | .302 |

**Correction (Theorem 23, reading of part 1): open, not retracted.** The partial-sum test above (top 300 points with
their TRUE c: 3.75) breaks the cancellation by construction, so it does not test the support. The honest test for a
generator with free c is the best c restricted to a subset (below). Until it runs, the support question is open.
(An earlier draft here retracted part 1 on the partial-sum evidence. That draft is withdrawn.)

**Check S (support with free c, restricted to subsets of the true training set).**
| Subset | best-c residual | random subset of the same size |
|---|---|---|
| all 1000 | .047 | |
| the 27 most unsure (u > .1) | .856 | .943 (sd .013) |
| top 30 / 100 / 300 by uncertainty-weighted contribution | .917 / .623 / .169 | .951 / .774 / .535 |
| the 973 sure ones only | .249 | |
Pre-registered (beat random by > .1 at 27 and 30): FAILED narrowly (.087, .035). At 100-300 the effect is strong.
**Statement 23.5 (Shonu's conjecture, graded form).** The weights are carried mostly by the LESS-SURE part of the data:
the least-sure 30% reproduce theta 3x better than a random 30%, and removing the 27 most unsure inputs raises the
residual 5x. They are not carried by a tiny set of very unsure inputs. So "the network knows what it is unsure
about" holds as a weighting over the data, with weight rising smoothly as confidence falls.

**Statement 23.2 (the coefficient step is convex, and it was the failure).** For fixed inputs, min over c >= 0 of
|| theta - G^T c || is a non-negative least-squares problem: convex, with an exact solution c*(x). The joint first-order
run left c far from c* (1.106 vs .302). By Danskin's theorem the reduced objective R*(x) = min_c R(x, c) has gradient
grad_x R(x, c*(x)), so the right algorithm solves c exactly and moves only x. (Variable projection; a consequence of
convexity, not a tuning choice.)

**Statement 23.3 (corrected): the equation DOES prefer the training set.** Compared on equal terms (best c for each):
true training set .047, rebuilt inputs .302. The reduced objective R*(x) = min_c R(x, c) prefers the training set by 6x.
The .241 (true set, true c) measures how far training stopped from stationarity. It is not an identifiability floor,
because the reconstruction minimises over c. So non-uniqueness was NOT shown. What was shown: the optimizer never got
near the true inputs' optimum in x either. (An earlier draft compared .302 with .241 and claimed non-identifiability.
Withdrawn: it compared two different c regimes.)

**T4 retired.** The best c is not unique even at the true inputs (.047 with free c vs .241 with c = p_k / (N lam)): 3000
unknowns against fewer effective constraints. "c proportional to p" is one solution among many, never singled out by
the solve. So T4's pass condition cannot be met even by a perfect reconstruction. Theorem 23 part 4 is an identity at
the true c, not a check on a solved c.

**Note on the per-layer numbers.** The relative residuals of the bias layers (c2.bias 1.06, fc.bias 1.24 on the true
set) are relative to tiny norms. Those layers carry almost none of the total squared residual.

**Statement 23.4 (what P2 got anyway).** Inputs that satisfy the weights' equation only to ~.3 already have the most
natural internal statistics of any generator (co-firing .82) and the best simple write (.080). This matches the
discussion after Corollary 22.2: the constraints of the lower layers are where the weights shape pixel structure. The
Fisher write is still poor (.189): the unsure part of these inputs is not the network's real unsure part.

**What follows.** Variable projection (Statement 23.2): exact c, gradient steps on x only. Outcomes, fixed before it
runs: (a) R* <= .10 AND training images rebuilt: the reconstruction literature replicated here; (b) R* <= .10 without
training images: many non-training solutions exist at that level, and then a tighter-converged toy is the next object;
(c) R* stays near .3: the x-landscape is the obstacle, not c.

**Check V (variable projection, exact c every 300 steps).** R* fell from .726 to .258 and stalled there (true inputs:
.047). No training image rebuilt. Co-firing with the truth .86, the best so far. But the P2 writes got WORSE than G3's:
simple .110 (G3 .080), Fisher .468 (G3 .189). Outcome (c) as pre-written for THIS protocol (m = N, 3000 steps). It is not shown that training images are
unrecoverable here: the equation prefers the true inputs 6x (.047 vs .26), and the published recipe (m = 2N
candidates, its initialisation, ~10^4+ steps) was not run.

## 24. Why fitting the weights' equation better made the write worse

**Theorem 24 (the weights fix first moments; the write needs second moments).** Stationarity is ONE vector equation,
lam theta = - E[grad l]: a first moment of the per-input gradients. The Fisher write needs
F-bar = E[J^T (diag p - p p^T) J] over the data (Theorem 7): a second moment. For any input distribution nu and any
self-labeled extra set Q (labels drawn from p_theta), E_{nu + Q}[grad l] = E_nu[grad l] but F-bar(nu + Q) =
F-bar(nu) + F-bar(Q), and F-bar(Q) is positive semi-definite and non-zero wherever Q has unsure inputs. So the weights
do not determine the write's metric, and no reconstruction rule that only satisfies stationarity can pin it.
**Proof.** Theorem 5's construction: the score has mean zero under self-labels, and its outer product has mean equal to
the Fisher, which is positive on unsure inputs. QED.
- This is the "mean versus variance" gap noted in section 20, now proved and measured. The reconstruction literature
  recovers real training images in practice because its networks are small, over-fit and very close to stationary.
  In principle the second moments stay free.

**Statement 24.1 (why lowering R* hurt; a HYPOTHESIS consistent with checks V and W, not proved).** Non-negative least squares returns SPARSE c (at most as many non-zero
coefficients as independent constraints; here 1500 of 3000, with total mass 3.0 against 7.6 for the true set). So the
solve concentrates the weights' first moment on fewer, more extreme inputs. A concentrated set can match a first
moment exactly while its second moment (what the write uses) is far from the data's. Fitting the first moment better
(.30 -> .26) moved the second moment further away. So R* is the wrong measure of usefulness for P2.
**Check W (refit, no generation).** Two causes, both real. (1) Junk: V gave 155 candidates c = 0; dropping them
improves V's Fisher write from .468 to .300 (the prediction "worse" FAILED). (2) Concentration: weighting by c makes the
Fisher write worse in both runs (G3 .189 -> .239; V .300 -> .358), and V (top 10% hold 29% of the c mass; G3 18%) stays
worse than G3 after the junk is gone (.300 vs .189).

**Consequence for P1 (the decision Shonu already made, now with a reason).** Since stationarity leaves the second
moments free, the SELECTION RULE decides them. The sparse rule (non-negative least squares; minimal support) is the
one that concentrates. The maximum-entropy rule picks, among all input sets that satisfy the weights' equation, the
most spread one. That is the rule that does not invent concentration the weights do not contain. Shonu allowed either
rule. Theorem 24 says they differ exactly where P2 needs them, and gives a reason to prefer maximum entropy for P2.
- The maximum-entropy form is section 22's family with ALL multipliers, p(x, y) proportional to
  q0(x) exp(- mu . grad_theta l(x, y)), with mu set by the stationarity constraint. Its x-part needs a sampler (Langevin),
  not a least-squares solve.

**Open (practical size of Theorem 24's freedom).** In principle the write's metric is free. In practice it may barely
move across different NATURAL unsure sets of equal size (B0 for the write). I2 tested the share of unsure inputs, not
which unsure inputs. One check would settle it: Fisher writes from disjoint real unsure subsets of equal size.

**Ledger (toy, P2 proxy, KL on 5000 fresh true samples).**
| Inputs | simple write | Fisher write | co-firing |
|---|---|---|---|
| truth | .034 | .016 | 1 |
| G0 inversion | 2.10 | 10.2 | .66 |
| G1 Langevin, flat | 1.08 | .160 | .60 |
| G3 KKT, joint optimizer | .080 | .189 | .82 |
| V KKT, exact c | .110 | .468 | .86 |

## 25. Where "how often" lives in the network (Shonu's question)

**Not in the size of the weights.** Training moves a weight only through errors; an input the network gets right stops
pushing. So a weight's size records "how often x how wrong" (Theorem 2). Check A: rank match with real firing
frequency .29 (3b).

**In what the network computes.** (1) Class shares: at stationarity lam b_k = freq_k - E[p_k]; with undecayed biases
E[p_k] = freq_k exactly. The last layer implies near-uniform shares (section 17). (2) Neuron frequency by the class
decomposition (exact): P(neuron fires) = sum_y P(y) P(fires | y). The network gives P(y); it gives P(fires | y) only
through inputs of class y.

**Checks F1, F2 (class inversions as the class-y inputs; 256 vs 256 real; real images only check).**
| Layer | rank match of on-rates | reliability ceiling | share of ceiling | median ratio self / real | within x2 |
|---|---|---|---|---|---|
| 3a | .594 | .983 | .60 | 1.02 | 91% |
| 3b | .454 | .973 | .47 | 1.20 | 82% |
| 4a | .527 | .950 | .55 | 1.12 | 83% |
| 4d | .526 | .899 | .59 | 1.24 | 82% |
| 5b | .325 | .696 | .47 | 1.52 | 78% |
F2 (480 x 480 co-firing of 3b): Pearson .780 (predicted >= .6: HOLDS). F1 (>= .7 in 3b, rising with depth): FAILED.
Beats the weight-size rule (.454 vs .29 in 3b).

**Statement 25.1 (modes are not typical).** A class inversion is the mode of class y's input distribution (it maximizes
the class score). In high dimensions the mode lies far from the typical set, so the frequency of a neuron at the mode
differs from its frequency over typical class-y inputs. The class decomposition needs typical inputs. This is why levels
come out right within x2 for ~80% of neurons, but the ORDER of neurons is only half right, at every depth. The deepest
layer is worst (ratio 1.52) because the maximization acts there directly (logit level 9x, check I4).
**Proof sketch.** For a d-dimensional distribution with density concentrated on a shell, the mode has density e^{O(d)}
above typical points and sits at distance O(sqrt d) from them; averages over the shell are not values at the mode. QED.

**What this says about Shonu's question.** "How often" IS in the network, as the pair (class shares, p(x | y)), and
p(x | y) is proportional to p_theta(y | x) q(x). The class shares and p_theta(y | x) are fully in the network. The one
missing object is again q(x), now with a sharper meaning: the network fixes WHICH class an input belongs to and how
sure it is, but not which inputs are TYPICAL. Sections 19 (sampling, not maximizing), 24 (spread, not concentration)
and 25 (typical, not mode) are the same gap seen three ways. One object closes all three: a canonical q(x).

## 26. The static assumption, questioned (Shonu's method: assume the network IS enough)

Every gap claim so far (Theorem 5, Theorem 24, check A, the read's need for gated plain moments, modes vs typical
inputs) rests on one model: the weights are SOME stationary point of the loss, and stationarity is all they say about
the data. Theorems 5 and 24 are proved for that model. But the weights are not an arbitrary stationary point. They are
the one gradient descent REACHED from a random start, on this data. With weight decay, the final weights are a
time-weighted sum of every gradient along the path: inputs the network explains now still pushed it early, and those
pushes built the features. Theorem 5's construction adds self-labeled inputs at the END; a network trained on that set
from the start would land elsewhere. So the scope of Theorems 5 and 24 is "any stationary point". Whether the point
training reached carries more is a separate, open question, and it is where this section looks.

**Conjecture 26.1 (Neural Feature Ansatz; Radhakrishnan et al., Science 2024).** For a trained layer whose weights W
read an input h, W^T W is proportional to a power of the AGOP, E_x[ sum_positions J^T J ], J = d logits / d h.
- The AGOP is a PLAIN average over the training data, and a SECOND moment. It comes from the dynamics, so it lies
  outside Theorems 5 and 24.
- The write's metric is the same object with the output metric inside: F-bar = E[ J^T (diag p - p p^T) J ].
- If the conjecture holds where the readers read the curve family (4a reading 3b), the weights carry the write's
  metric. Theorem 24's freedom would then be closed for trained networks by the dynamics.

**Checks N1, N3 (500 train images as the check; split-half reliability .86-.99).** Cosine of each layer's input-conv
weight Gram W^T W with the square root of the data AGOP: 3a->3b .939, 3b->4a .918, 4a->4b .923, 4d->4e .901; with the
square root of the output Fisher .899-.939; with the plain activity moment .45-.65; with the identity .51-.62. On the
off-diagonal entries alone: AGOP^.5 .90-.94, Fisher^.5 .90-.94, plain .20-.35. Conjecture 26.1 HOLDS at every depth
tested, with power 1/2 (the form reported for deep networks).

**Controls (written before running).** (1) With no data at all (G_u = I), cos(W^T W, (W^T W)^.5) = .93-.95, as high as N1.
So N1's cosine is explained by the sandwich structure (AGOP at the input = W^T G_u W) and is NOT evidence of data
content. The prediction (< .85) FAILED. (2) Real > shuffled > noise (3b->4a: .918 / .800 / .790): HOLDS.
**The discriminating comparison.** Target: the off-diagonal entries of the REAL-image Fisher at each layer input.
| site | from the weights alone (Gram) | network's Fisher on noise | on shuffled pixels |
|---|---|---|---|
| 3a->3b | .867 (Gram^2 .891) | .742 | .823 |
| 3b->4a | .874 | .380 | .445 |
| 4a->4b | .809 | .359 | .399 |
| 4d->4e | .839 | .324 | .384 |

**Statement 26.2 (corrected).** From 3b on, the next layer's weights predict the structure of the real-data Fisher
metric (.81-.87, off-diagonal) about twice as well as the network's own Fisher on unnatural inputs (.32-.45). So the
weights carry a part of the natural-data metric that the forward map alone does not reveal. This is the first gap claim
where the network was tested against a proved limit (Theorem 24) and held something the limit says is free. The first
draft claimed more (N1's .92 as proof of data content); that claim is withdrawn by control 1.
**Lemma 26.3 (proved): a constant metric cannot help the unconstrained write.** If every reader's column is free and
every reader sees the same regressors (the coarse outputs and a constant), weighted least squares with ANY fixed
positive-definite metric returns the ordinary least-squares write (identical regressors: GLS = OLS). So the true average
Fisher, or the weights' version of it, gives exactly the lsq write (.0836 on M's read). The Fisher write's gain
(.0836 -> .0443) comes entirely from the metric's variation from image to image, with the readers' gates.
**Where the weights' metric does help.** Wherever the problem is constrained: the WHAT (Theorems 10-11 are rank-k
problems in Phi = A^T F-bar A), a write restricted in rank or sign, and the choice of shifts. There it replaces the
mix's F-bar with no images.
- Caveat (structure, not a flaw): the AGOP at a layer's input equals W^T G_z W, with G_z the AGOP at its output. So the
  match also says G_z is close to a power-consistent form. The claim used here is only the practical one: the Fisher
  metric at 3b's output is readable from 4a's weights.
- What it does not give yet: the per-image Fisher. The best writes so far used per-image metrics with the readers'
  ReLU gates exact (sections 9, 15). A constant metric loses the gates (the "woken readers" failure, W2e).

## 27. The last gap, stated exactly

What the build needs, and where each piece comes from (sections 10-26):
| Piece | Source with no outside data |
|---|---|
| WHAT the coarse neuron computes (children weights, Theorems 10-11) | weights: co-firing rank (B, .76); metric Phi from the next layer's weights (26.2) |
| HOW one ReLU reads it (Lemma 14.1) | the network's own class inputs (E1-E3: equal to real images) |
| the write | ? |

**Proposition 27.1 (what the write needs, exactly).** Write the per-image output metric at the readers as
F(x) = J(x)^T (diag p(x) - p(x) p(x)^T) J(x). J and p are computed by the network for ANY input x, so F is a function the
network supplies. The Fisher write solves N w = m with N = E_x[ Phi(x)^T F(x) Phi(x) ], m = E_x[ Phi(x)^T F(x) A z(x) ].
Split F = F-bar + (F - F-bar). The F-bar part gives the lsq write (Lemma 26.3). So the write's whole gain over lsq
lives in the cross moment
  E_x[ c(x) c(x)^T (x) (F(x) - F-bar) ]:
how the output's sensitivity varies TOGETHER with the coarse unit's activity, over inputs. Every factor inside the
expectation is computed by the network. Only the distribution of x is not.
**Proof.** Substitute the split into N and m; the F-bar terms are the constant-metric problem, which Lemma 26.3 solves
by lsq. QED.

**Corollary 27.2 (K-FAC does not close it).** Any factorised metric N = E[c c^T] (x) F-bar (input moment times output
metric, as in K-FAC) drops the cross moment, so it returns the lsq write too. So no product of an activity statistic and
a sensitivity statistic, however accurate each one is, can give the Fisher write.

So the open object is narrow: the joint distribution of (activity, sensitivity) on unsure natural inputs. Two outcomes
are possible:
1. The write barely depends on WHICH natural unsure inputs carry this cross moment (as the read barely depended on
   the mix, B0 .98). Then any natural-looking unsure inputs the network makes will do, and the gap closes.
2. It depends strongly on which inputs. Then the specific training distribution matters, and Theorem 5 is the precise
   reason outside information is needed.
Check S1 decides between them.

**Check S1 (which unsure inputs; step1 write on M's read, 4000 unsure train images each).** Animal classes only .0791,
non-animal only .0837, two random unsure sets .0674 / .0649. Animal vs non-animal: -.0045 (se .0041). A write fit on
half the world transfers to all classes at a cost of .012-.016 against a broad mix. **Outcome 1 holds:** the write needs
natural unsure inputs, and barely depends on which. This is the write's version of B0.

**Check S2 (cost).** Exact top-10 Fisher on random unsure real images: 1000 -> .0824, 2000 -> .0582. So about 2000
inputs carry the write.

**The network-made unsure inputs (rule fixed once, section 28 tests it).** Class a spread over the 1000 classes;
partner b = the class whose last-layer weight row is closest to a's (weights only); minimise the cross-entropy to
1/2 e_a + 1/2 e_b (bounded; optimum p_a = p_b = 1/2); plain pixels; the same transforms and step size as check E;
128 steps; mirror images added (the network was trained to be mirror-invariant). On 64 inputs: 1 - sum p^2 = .381
(real unsure pool .39), logit level 23 (real 55, inversions 473), member co-firing vs real .932.
- Why this rule follows from the theory: Lemma 19.1 rules out maximising; Proposition 19.2 says the write needs unsure
  inputs; S1 says which unsure inputs matters little; check E says this pixel rule keeps natural low-level structure.

**Statement 27.3 (why the toy cannot test generators).** A network's answer constrains its input by at most
log2(number of classes) bits per input through the label alone. The 4-class toy gives 2 bits: class inversions on the
toy lose natural internal statistics (co-firing .66) and boundary inputs too (.57; Fisher write .550 vs truth .016).
InceptionV1's 1000 classes and depth pin its inputs far more (inversions co-firing .94). So generator quality is a
property of how much the NETWORK knows about its inputs, and the toy knows little. (The toy stays valid for the
stationarity questions of sections 21-24, which do not depend on this.)

## 28. Coverage: what a network-made input set must do for the write

**Checks E8, E10 (boundary inputs alone).** The Fisher write from 2048 boundary inputs exploded (dev KL 2.41 on M's read;
step0 from 512: 10.08 vs .152 from 512 real unsure). Not the constant beta (H3: beta from real images still 5.05), not
contrast (C29: real unsure at half contrast gives the same KL), not mainly the size of the columns (H4).

**Theorem 28.1 (the coverage criterion).** Let w* be the write the real data's normal equations give (N_real w* = m_real),
and w_q the one from a generated set (N_q w_q = m_q). Its excess second-order loss on real data is
(w_q - w*)^T N_real (w_q - w*). Write w_q - w* in the generalised eigenbasis of the pair (N_real, N_q): directions v with
N_real v = rho N_q v. Along a direction with ratio rho, any error in m_q is amplified by N_q^-1 and then weighed by
N_real, so its cost grows like rho. A generated set works only if the largest ratios stay near the finite-sample level
of real data itself.
**Proof.** Linear algebra: in the generalised basis, N_q = I and N_real = diag(rho); the cost of a component e is rho e^2,
and e = (N_q^-1 (m_q - N_q w*))_v. QED.
**Check H5.** Boundary inputs: largest ratio 1.4e7, 10th 3.3e4, 100th 412; 91% of the excess energy in the top 100
directions. Real vs independent real (512 each): 301 / 192 / 49.

**Proposition 28.2 (reader gates).** The write's metric sits at the readers' pre-ReLU inputs, so it carries each reader's
gate: F(x) has zero rows and columns for readers that are off at every position of x. If reader r is off on every
generated input, every parameter of reader r (its column for each coarse unit and its constant) has N_q = 0, and the
ratio rho is unbounded in those directions.
**Proof.** d logits / d u_r = 0 where the reader's ReLU is off, so the r-th rows of J vanish. QED.
**Check H6.** 23-45 of 368 readers never fire on the generated sets (they fire 35-37% on real images); the worst
directions put 54% of their weight on them.

**Statement 28.3 (the network supplies its own coverage).** For each silent reader, the network's own feature
visualisation (same plain-pixel rule) switches it on: 52 of 52. Adding these 104 inputs to the boundary set cut the
largest ratio from 1.4e7 to 1.2e3 (real-vs-real: 301) and the step0 KL from 10.08 to .169 (real 512: .152). The
remaining worst directions are 99% per-reader constants, and by the 100th direction coverage equals real-vs-real.
So no write direction is structurally missing from the network-made set.

**What is left (check E14).** With exact step1, M's read + the network-made write gives .140; the fully network-made
formula .166; both lose to lsq (.097 / .077). The coverage gap (~4x at the worst direction) means each network-made input
is worth ~1/4 of a real unsure image (2256 inputs ~ 512 real), and the Fisher write loses to lsq below ~1000
real-equivalent inputs. This is effective sample size, a count, not missing information. Check E15 tests it with
the network's own invariances (shifts, mirrors).

**Check E15 (more samples from the network's invariances: 4 shifts x mirrors, 9024 inputs).** M read + step1 .098
(was .140), ties lsq on the set (.103). **Check BV (bias or variance, in the real metric).** Gap .429 = network-made
variance .125 + real variance .068 + bias .236. Bias dominates, so more inputs of the same kind cannot close it.

**Statement 28.4 (where the bias sat: the metric-weighted constant).** The worst remaining directions were the
per-reader constants (H7: 99%). Keeping the network-made columns B and replacing the solve's constant:
| constant from | dev KL |
|---|---|
| the Fisher solve, E_q[F (Az - Bc)] / E_q[F] | .128 |
| class inversions, plain mean | .113 |
| the network-made set, plain mean | .093 |
| real images (diagnostic) | .094 |
The plain mean over the network-made set is as good as real images. Reason: the metric-weighted mean puts its weight on
the set's most sensitive positions, which are its least natural parts (reader pictures, boundary regions). The plain
mean is a first moment of activity, the kind of statistic that transfers across input sets (as the read's did, E1,
B0). So the columns (the cross moment, Prop. 27.1) come from the Fisher metric, and the constant from the plain mean.
This split is the constant-metric part of Prop. 27.1 done with the robust statistic.

**Check E17 (columns from the Fisher metric, constant from the plain mean; network-made inputs only).** On M's read:
step1 .0807 vs lsq on the same inputs .1070, paired -.0263 (se .0042). The network-made Fisher write now beats the
simple rule by the same margin that real data gives (real unsure 2000: .058 vs lsq .084). So the write's cross moment
(Prop. 27.1) IS supplied by inputs the network makes, once (i) unsure inputs are made by sampling-like targets, not
maximising (Lemma 19.1), (ii) every reader is covered by the network's own picture of it (Prop. 28.2, Statement
28.3), and (iii) the constant is taken from the plain mean (Statement 28.4). With the self read fit on class inversions
only, the same write gives .0984: the read must be fit on the inputs the write uses (check E18).

## 29. What the network supplies, and the one object it has not yet supplied (no new runs)

**Checks E18-E20, read as statements (they were run as patches; recorded as such).** E18: refitting the self read on the
network-made set changes nothing (.0984). E19: adding the [Fisher - lsq] correction measured on the boundary set to a
good baseline hurts (.0935). E20: the network's metric at a temperature on class inversions depends strongly on T
(M read .152 / .099 / .111 at T0/2, T0, 2T0) and never beats lsq: T is a knob, so this route is retired. One new fact:
the identity-metric write fit on the 256 class inversions gives M's read .0661, better than the same write on 1000 real
images (.0836).

**Proposition 29.1 (the write's three ingredients).** By Theorem 7 and Prop. 27.1 the optimal write depends on the input
distribution only through
| # | Object | What it is |
|---|---|---|
| (i) | E[phi phi^T], E[phi z^T] | plain co-activity of the coarse units, the family and a constant |
| (ii) | F-bar = E[F(x)] | the average output sensitivity at the readers |
| (iii) | E[(phi phi^T) (x) (F(x) - F-bar)] and its m-analogue | how sensitivity varies TOGETHER with activity across inputs |
(i) and (ii) with a constant metric give the identity-metric write (Lemma 26.3); (iii) is the entire Fisher gain.
**Proof.** Expand N and m with F = F-bar + (F - F-bar). QED.

**What the network supplies (measured).**
| Object | Source with no outside data | Evidence |
|---|---|---|
| (i) | the network's class inversions | co-firing .95 (E1); identity-metric write .0661 on M's read, better than 1000 real images (.0836) |
| (ii) | the next layer's weights (dynamics, Statement 26.2) | real-metric structure .81-.87 vs .32-.45 from unnatural inputs |
| the WHAT (Theorems 10-11) | weights + (i) + (ii) | network-made Phi vs real .99, Sigma .90 (E9) |
| the HOW (Lemma 14.1) | class inversions | equal to the real-image read under a common write (E3: +.0074) |
| (iii) | network-made unsure inputs (boundary + the network's picture of every reader) | coverage within 4x of real-vs-real (Theorem 28.1, H7) and a Fisher gain over lsq of the real size on the same inputs (E17: -.026), but BIASED: bias .236 > variance .125 in the real metric (BV) |

**Statement 29.2 (why (iii) is the hard object).** (iii) is a second moment of the JOINT law of (activity, sensitivity).
Stationarity fixes first moments only (Theorem 24). The dynamics leave (ii), the average sensitivity, in the weights
(Statement 26.2); no identity found so far leaves the covariance (iii). Each generator's (iii) is set by its rule:
maximising makes F = 0 (Lemma 19.1); boundary targets make F rank one in class space with weak activity (E10-E12);
a temperature makes it T-dependent (E20). S1 showed (iii) is robust across NATURAL unsure subsets (animal vs non-animal,
.0045), so (iii) is a property of where natural images are unsure, not of which classes they show.

**The accounting (k = 3, dev-5000).**
| Build | KL | Uses outside data? |
|---|---|---|
| fully network-made: self read + identity-metric write on class inversions | .0771 | no |
| M's (trained) read + identity-metric write on class inversions | .0661 | the read was trained |
| identity-metric write on 1000 real images, M's read | .0836 | yes |
| Fisher write on 35000 real images, M's read | .0443 | yes |
| THE FORMULA (real arcs read + Fisher write, 35000 real) | .0507 | yes |
| bar B (best trained) | .0361 | yes |
So the network alone already beats a write built from 1000 real images. The remaining gap to the best data-based build
(~.026) splits into the read (self vs M under the same write: .011) and object (iii) (.066 -> .044 on M's read: .022).

**Where the question now stands, precisely.** The need for outside information has been dismantled for every ingredient
except one: the covariance (iii) between the coarse units' activity and the output's sensitivity over natural unsure
inputs. It is not proved that the network lacks (iii); it is proved that stationarity cannot give it (Theorem 24), and
measured that the generators tried so far give a biased version. The next mathematical object is therefore an
identity, from the training dynamics like the ansatz, that ties (iii) to the weights; or a toy construction of two
training sets that leave the weights (and their ansatz statistics) the same but change (iii). Either one settles it.

## 30. Object (iii) as curvature, and a weights-side form of it

**Lemma 30.1 ((iii) is the loss curvature in the family's own write).** Perturb the large network's family columns
A -> A + dA. To second order the loss changes by 1/2 E[ z^T dA^T F dA z ] (F = J^T Omega J at the readers, Gauss-Newton).
So E[(z z^T) (x) F], the metric-weighted co-activity whose non-separable part is (iii), is the Hessian of the loss with
respect to the large network's own parameters A (Optimal Brain Surgeon's object). The coarse write asks for the best
B Lambda close to A in exactly this curvature (Theorem 10 with the separable Phi (x) C replaced by this Hessian).
**Proof.** du = dA z; Lemma 8.1. QED.

**Proposition 30.2 (the ansatz one layer earlier gives a gated co-sensitivity from the kernels).** Let K (800 x 96) hold
the 96 kernels of 3b's 5x5 branch, which read the bottleneck window X, and g = d f / d z (z: their pre-activations).
The AGOP at X is E[ sum_q J_X^T J_X ] = K M K^T with
  M_ij = E[ sum_q 1(z_i > 0) 1(z_j > 0) g_i g_j ]   (96 x 96),
the output's sensitivity to channels i and j measured ONLY where both fire. If the ansatz K K^T = c (K M K^T)^(1/2) holds
and K has full column rank, then M = c^-2 K^T K.
**Proof.** K = U S V^T. Then U S^2 U^T = c (U S V^T M V S U^T)^(1/2); squaring in the U-basis, S^4 = c^2 S V^T M V S, so
V^T M V = c^-2 S^2 and M = c^-2 V S^2 V^T = c^-2 K^T K. QED.
- Meaning: the family's own kernel Gram K_i . K_j would equal (up to one constant) how sensitive the output is to two
  curve neurons at the places where both fire: an activity-sensitivity moment of the kind (iii) is, from weights only.
- It also explains check B (kernel overlap ranks co-firing at .76): M carries the gates.

**Check N4 (written before running; real images only check).** On 500 train images, measure M (96 x 96, gradients of
the logits by random projections and of the Fisher form) and compare with K^T K. PREDICTION: off-diagonal cosine
cos(M, K^T K) > .7, above the ungated sensitivity E[g g^T] (cos < .5 with K^T K) and above plain co-activation
E[relu(z) relu(z)^T]. Also on the 10 x 10 curve block.

**Check N4: FAILED.** Off-diagonal cos(M, K^T K) = .32 over 96 channels (.54 on the curve block); plain co-activation
.47 / .45. (M equals its ungated form exactly: a pre-activation gradient is already zero where the channel is off.)
**Statement 30.3 (what the ansatz pins, exactly).** The ansatz holds only as a match of SANDWICHED matrices,
W^T W vs W^T G W (control 1 of section 26). Proposition 30.2 needed it exactly enough to invert the sandwich, and it is
not. So the weights pin the input-side metric (Statement 26.2) but not the inner, gated metric G (here M) of which
(iii) is made. Neither stationarity (Theorem 24) nor the ansatz gives (iii).

**Conjecture 30.4 (the weights' naturalness score).** The one statistic the weights pin that separates natural from
unnatural inputs is the input-side match S_l = off-diagonal cos(W_l^T W_l, AGOP_l) at deep layers (real .81-.87,
noise .32-.45). If the network-made sets are biased because they sit off the natural manifold in deep features, they
score low on S_l. Check N5 (written before running): S at 3b->4a, 4a->4b, 4d->4e for real unsure, class inversions,
boundary inputs, reader pictures (128 each). PREDICTION: real >= .8 at all three; boundary and inversions < .6 at
4a->4b and 4d->4e. If so, S_l is the network's own target for its inputs (the second-moment analogue of DeepInversion,
with weights in place of BatchNorm statistics).

**Check N5: FAILED as a target.** S_l (3b->4a / 4a->4b / 4d->4e): real unsure .880 / .803 / .868; class inversions .838 /
.824 / .819; boundary .709 / .717 / .796; reader pictures .832 / .752 / .827. The network-made sets pass it (inversions
as well as real), so S_l rejects only gross unnaturalness (noise .32-.45). It cannot remove the bias in (iii).

## 31. The imprint of discrete training steps: second moments in the weights

**Theorem 31 (SGD balance shift).** Take a ReLU neuron with input weights u (with bias) and output weights v (its
columns in the next layer). The loss is invariant under (u, v) -> (a u, v / a), so for EVERY sample
<u, grad_u l> = <v, grad_v l>. Train by SGD with step eta, weight decay lam, minibatch gradients g_u, g_v. Then
D = ||u||^2 - ||v||^2 obeys D' = (1 - 2 eta lam) D + eta^2 ( ||g_u + lam u||^2 - ||g_v + lam v||^2 ),
because the first-order term -2 eta (<u, g_u> - <v, g_v>) vanishes by the symmetry. At equilibrium (E D' = D), with
E g = -lam theta (Theorem 2):
  **||u||^2 - ||v||^2 = (eta / (2 lam B)) ( tr Cov(per-sample g_u) - tr Cov(per-sample g_v) ).**
**Proof.** Expand the squared norms after one step; take expectations; at stationarity E||g + lam theta||^2 is the
minibatch gradient variance, (1/B) times the per-sample one. QED.
- Gradient flow (eta -> 0) gives exact balance (Theorem 1). The measured imbalance (bottlenecks 10-25% below balance,
  the open hole of section 2) is then the imprint of the step size, and it stores per-sample gradient SECOND MOMENTS.
- The output-side energy is E|| sum_p delta(p) h_i(p) ||^2: activity times sensitivity, the kind of object (iii) is.
  The input-side energy is E|| sum_q g_i(q) X_q ||^2. Stationarity (first moments) cannot see either (Theorem 24).
- One scalar per neuron, with one unknown common factor eta / (2 lam B). A rank test across neurons is free of it.

**Check N6 (written before running).** Across the 480 neurons of 3b: D_i from the weights vs the measured per-sample
gradient energy difference E||g_u||^2 - E||g_v||^2 (200 train images, true labels: data as a CHECK only).
PREDICTION: Spearman > .5. Baselines: |D_i| vs the plain activity E[h_i^2] (an energy proxy without sensitivity).

**Check N6: HOLDS.** Across 3b's 480 neurons: Spearman(D_i, E||g_u||^2 - E||g_v||^2) = .621 (normalised .613); input side
alone .417, output side alone .175, plain activity -.265. In^2 / out^2 median 1.092 (82% of neurons above balance).
**Statement 31.1 (the network does carry second moments of its data).** Theorem 24 assumed an exact stationary point
under gradient flow. Real training takes finite steps, and every rescaling symmetry then stores, in the weights, a
difference of per-sample gradient energies, which are second moments over the training data (activity x sensitivity on
the output side). One scalar per hidden ReLU neuron, thousands over the network, from the weights alone.

**Corollary 31.2 (targets for network-made inputs).** For a candidate input set q with weights w_s, define the
predicted imbalance D_i(q) = E_q||g_u,i||^2 - E_q||g_v,i||^2 (per-sample gradients under the network's own labels, i.e.
Fisher energies). Theorem 31 says the training data satisfies D_i(train) proportional to D_i(weights) for every neuron.
So the weights give thousands of linear constraints on the input distribution's second moments: a set that violates
them has the wrong second moments, which is what the bias in (iii) is.
**Check N7 (written before running).** Spearman(D_i(weights), D_i(q)) for real unsure (Fisher energies), class
inversions, boundary, reader pictures, and the combined network-made set (128 each). PREDICTION: real ~ .6; network-made
sets < .4. If so, reweighting the network-made pool to meet these constraints is the derived next step.

**Check N7: HOLDS.** Spearman(D(weights), D(q)) over 480 neurons: real unsure .704, real .665; class inversions .213,
boundary .264, reader pictures .239, boundary + readers .310. The weights' second-moment constraints (Theorem 31) tell
real inputs from network-made ones, which the input-side score S_l could not (N5).

**Construction 31.3 (the derived generator: maximum entropy under the SGD-imprint constraints).** From a pool of
network-made inputs s with per-neuron Fisher-energy vectors d_s (Corollary 31.2), take weights
  w_s proportional to exp( mu . P d_s ),   P = projection orthogonal to D(weights),
with mu chosen so that sum_s w_s P d_s = 0 (the weighted set's energy vector is parallel to D(weights); the scale is
free). This is section 22's maximum-entropy family with Theorem 31's constraints in place of stationarity: the least
concentrated reweighting of the network's own inputs that carries the second moments the weights record.
**Check N8 (held-out neurons; written before running).** Fit mu on the even-indexed neurons of 3b; measure the rank match
on the odd-indexed ones. PREDICTION: held-out Spearman rises from the unweighted pool's value (~.3) to > .5, with an
effective sample size above 20% of the pool.

**Check N8: FAILED; reason: infeasible.** The max-entropy tilt left the held-out match unchanged (.322 vs .316). The best
non-negative combination of the 1384 pool vectors d_s reaches cosine .424 with D(weights) (real unsure mean vector: .533;
inversions alone .177). A reweighting can only reach the cone of the pool's own energy vectors, and D(weights) lies
outside it. **Statement 31.5:** the network-made inputs so far lack the second-moment patterns the weights record; new
inputs are needed, not new weights on old ones.

**Construction 31.4 (the SGD-imprint generator).** Make inputs by descending on the weights' own target: move the pixels
of a set so that its per-neuron Fisher-energy vector sum_s d(x_s) lines up with D(weights) (cosine), starting from the
class inversions (natural low-level structure, check E). It uses only the weights. It needs second derivatives through
the network (max-pools replaced by an identical, twice-differentiable form).
**Check N9 (smallest test; written before running).** 16 inputs, 40 steps, objective on the even-indexed neurons of 3b,
scored on the odd ones. PREDICTION: held-out Spearman rises from ~.2 (inversions) to > .5; curve-member co-firing vs
real stays > .85.

**Check N9 (16 inputs, 40 steps).** Held-out neurons .165 -> .337 (fitted half .194 -> .533); member co-firing .912 ->
.909; pixel change sd 3.4. The prediction (> .5) FAILED, but the held-out match doubled with low-level statistics kept.
With 16 inputs and 240 constraints the fit is far under-determined, so only shared structure carries to held-out neurons.

**Definition 31.6 (a direct measure of (iii)).** Give each child its own scale on its write column, u = sum_i s_i A_i z_i
(s = 1 is the network). The Fisher information of the 10 scales,
  Psi_ij = E_x[ sum_r p_r (d log p_r / d s_i)(d log p_r / d s_j) ],  d f / d s_i = sum_p g(p)^T A_i z_i(p),
is the curvature of Lemma 30.1 restricted to child-level rescalings: activity times sensitivity, jointly, with no
separable approximation. It is what the write's Fisher gain is made of, at the child level.
**Check N10 (written before running).** Start from 48 boundary inputs (unsure); run Construction 31.4 with all 480 3b
neurons as the target (30 steps); compare Psi (normalised: off-diagonal pattern and diagonal shape) with real unsure
images (128). PREDICTION: the imprint inputs match real Psi better than their boundary starts by > .2 on the pattern
correlation.

**Check N10: FAILED; N9 retracted.** Re-running N9 gave a held-out match of .174 (first run .337): the gain does not
replicate. Moving 48 boundary inputs toward the imprint target made the child-scale Fisher Psi LESS like real (pattern
.498 -> .395; scale .135 -> .054). Reason: 480 scalar gradient energies are second-order quantities of a deep network;
tiny pixel changes (sd 3.5) meet them adversarially without moving the inputs toward natural images. A small set of
scalar statistics cannot pin an input distribution against an optimizer.

## 32. The information channels in trained weights, and what each carries

Every way the data can be imprinted in the weights found in this work falls into one of three channels.
| Channel | Mechanism | What it carries | Evidence |
|---|---|---|---|
| 1. stationarity | E[grad l] = - lam theta (Theorem 2) | first moments of per-sample gradients, for every parameter | Theorems 2, 23; X1 exact on the toy |
| 2. symmetry imprints | each continuous symmetry of the loss turns the SGD step's second-order term into a drift along its orbit (Theorem 31) | ONE second-moment scalar per symmetry generator: per ReLU neuron, input-side minus output-side gradient energy | N6: .62; N7: real .70 vs network-made .21-.31 |
| 3. dynamical alignment | the weight Gram aligns with the input-side gradient average (the ansatz) | sandwiched second moments W^T G W, not the inner metric G | 26.2: .81-.87 discriminating; N4: inner metric not recoverable |

**Proposition 32.1 (what the write's object (iii) needs, against the channels).** At the child level, (iii) is Psi (the
Fisher of child-level rescalings, Definition 31.6): Psi_ii (one child's activity times output sensitivity) and Psi_ij,
i != j (two different children's activities covarying with sensitivity).
1. Psi_ii is tied to channel 2: the output-side gradient energy of member i, E|| sum_p delta(p) h_i(p) ||^2, contains the
   projection E[(A_i^T sum_p delta(p) h_i(p))^2] that Psi_ii is (up to the Fisher form). It is entangled with the
   input-side energy, one scalar per neuron.
2. Psi_ij for i != j: no continuous symmetry of a ReLU network mixes two neurons (the ReLU breaks rotations among them),
   so channel 2 has no generator for pairs. Channel 1 carries first moments only (Theorem 24). Channel 3 carries
   sandwiched averages, not activity-weighted ones (N4, Statement 30.3). So none of the three channels carries the
   pairwise activity-sensitivity covariation.
**Proof of 2.** A one-parameter symmetry group acting on (u_i, u_j, v_i, v_j) that leaves relu-network functions fixed
must commute with the elementwise ReLU; the only such linear maps are positive diagonal scalings and permutations, so
the connected symmetry group is the product of per-neuron scalings. Each generator gives one scalar (Theorem 31).
Channels 1 and 3 are the identities above. QED (for the channels identified here).

**Where the question stands (precise).** Outside information is no longer needed for: what the coarse neuron computes,
how it reads its inputs, the coarse units' co-activity, the average output metric, write coverage, and the write's
constant. It is still needed, with the tools found so far, for ONE object: the pairwise covariation (over natural
unsure inputs) between different children's activity and the output's sensitivity. The diagonal of that object is in
the weights (channel 2, entangled); the off-diagonal is in none of the three channels. This is a precise statement, not
a proof of impossibility: a fourth channel could exist.
**The decisive next check (derived, not run).** Does the write need the off-diagonal part? Build the real-data Fisher
write with Psi's off-diagonal removed (the child-level metric replaced by its diagonal) and compare with the full write.
If the diagonal-only write keeps most of the Fisher gain, the remaining dependence reduces to channel 2's scalars, and
the work becomes disentangling them (an inflow-outflow system of gradient energies over the network graph). If it loses
the gain, the pairwise covariation is the irreducible outside information, and Proposition 32.1 is the precise reason.

**Correction to section 32's last paragraph.** "Remove Psi's off-diagonal" is not well defined for the reader-space write.
**Proposition 32.2 (the scalar-sensitivity write).** Replace F(x, p) by s(x, p) F-bar, with s(x, p) the position's total
output sensitivity (trace of its metric). By Lemma 26.3 applied with weights, the write becomes least squares weighted
by s, so the only joint information it uses is how activity covaries with a SCALAR sensitivity, a per-unit quantity of
the kind channel 2 stores, with no pairwise shape structure.
**Check S3 (written before running; real unsure images as a check).** On 2000 real unsure images, M's read: identity lsq,
s-weighted lsq, Fisher step0. PREDICTION: if s-weighted lsq recovers > 70% of the gap from lsq to step0, the write's
need reduces to per-unit (channel-2-type) quantities; if < 30%, the per-image shape is needed.

**Check S3.** Real unsure 2000, M's read: identity lsq .0798, s-weighted lsq .0754, Fisher step0 .0681. The scalar
weighting recovers 38% of the gap (prediction thresholds 30 / 70: neither). So about two fifths of the Fisher gain is
per-unit importance (the kind channel 2 stores), and three fifths is the per-image shape of the metric (which readers
matter on which image): the pairwise object of Proposition 32.1. (For scale: M's read + identity lsq on 256 network-made
class inversions, .0661, already beats this real-data step0 write.)

## 33. Summary of the dismantling (state at the end of this session)

| Ingredient | Outside data needed? | How the network supplies it |
|---|---|---|
| what the coarse neuron computes (children weights) | no | weights + network-made co-activity + weights' metric (E9: .99 / .90) |
| how one ReLU reads it | no | Lemma 14.1 on the network's class inversions (E3) |
| co-activity of coarse units and family | no | class inversions (E1 .95; their lsq write beats 1000 real images) |
| average output metric | no | next layer's weights (26.2, channel 3) |
| write coverage of every reader | no | the network's own picture of each silent reader (28.3) |
| the write's constant | no | plain mean over network-made inputs (28.4) |
| per-unit importance part of the Fisher gain (~40%) | in the weights in principle | channel 2 (Theorem 31), entangled, not yet extracted |
| per-image metric SHAPE part of the Fisher gain (~60%) | yes, with the channels found | pairwise activity-sensitivity covariation; no channel carries it (Prop. 32.1) |
Best fully network-made k = 3 build: .0771 (self read) / .0661 with M's read; best real-data build .0443-.0507; bar B .0361.

## 34. Why no weights-only method can recover the write's remaining part (the impossibility argument)

**Setting.** A training procedure T maps a data distribution p (over labeled inputs) to weights theta = T(p). A
coarsening method is ANY map Phi from weights to a coarse network. Its quality is judged on p: the KL-optimal write for
p is w*(p) = argmin_w L_p(w), with L_p(w) = (w - w*(p))^T N_p (w - w*(p)) + const to second order (Theorem 7).

**Definition 34.1 (consistent distributions).** p' is consistent with theta if theta is a valid outcome of training on p'
in every channel through which data reaches the weights (section 32):
  (C1) stationarity: E_p'[grad l(theta)] = - lam theta;
  (C2) symmetry imprints: for every ReLU neuron n, the gradient-energy imbalance E_p'||g_u,n||^2 - E_p'||g_v,n||^2 equals
       that of p (Theorem 31);
  (C4) the SGD implicit-regularisation shift: E_p'[ grad_theta ||grad l||^2 ] equals that of p (Smith et al. 2021: SGD
       follows L + (eta / 4B) E||grad l - grad L||^2; this term's gradient enters the stationarity of finite-step SGD;
       channel 2 is its projection on the symmetry directions).
(The alignment of section 26, channel 3, is a relation between weights and data, not an equation that moves weights;
with theta fixed it adds no constraint.)

**Lemma 34.2 (self-labeled mixtures keep C1).** For any input distribution q, label each input by sampling y from
p_theta(y | x). Then E[grad l(theta)] over q is 0 (the score has mean zero), so the training measure p' = p + eps q satisfies C1 exactly.

**Theorem 34.3 (non-identifiability of the optimal coarse network).** Suppose there are two self-labeled input measures
q1, q2 with equal C2 and C4 integrals but different write moments (N_q1, m_q1) != (N_q2, m_q2) in the sense that the
optimal writes of p1 = p + eps q1 and p2 = p + eps q2 differ. Then theta is consistent with both p1 and p2 (C1, C2, C4),
and for EVERY weights-only map Phi,
  max_{i = 1, 2} [ L_pi(Phi(theta)) - L_pi(w*(pi)) ]  >=  1/2 (w1* - w2*)^T (N_p1^-1 + N_p2^-1)^-1 (w1* - w2*)  >  0.
**Proof.** Phi(theta) is one write w for both. For quadratics, min_w [ (w - a)^T N1 (w - a) + (w - b)^T N2 (w - b) ]
= (a - b)^T (N1^-1 + N2^-1)^-1 (a - b) (complete the square). The larger of the two terms is at least half the sum. QED.

**Why the premise is generic (existence).** C2 and C4 are finitely many linear functionals of q (one per ReLU neuron,
plus one per parameter for C4). The self-labeled measures q form an infinite-dimensional convex cone. The write
moments involve c(x), the activity of the NEW coarse units, a function the large network does not compute, multiplied by
the per-image metric: they are not linear combinations of the C2/C4 functionals. So the joint kernel of the finitely many
constraints contains directions that change the write. The finite-dimensional check of this premise is check X3.
**What the theorem does and does not say.** It says the KL-optimal coarse network is not a function of the weights:
for every weights-only method there is a training distribution, consistent with the weights in every known channel, on
which it is suboptimal by the bound above. It does not say the TRUE training distribution's write is far from what a
method finds; that depends on how unnatural the consistent alternatives are. The weights cannot tell natural data from
self-labeled mass that leaves every channel unchanged, so a weights-only method cannot prefer the natural one.

**Check X3 (written before running).** Demonstrate the premise for channels C1 and C2 (the 480 neurons of 3b) on
InceptionV1: from the network-made pool (1384 inputs, their self-labeled energy vectors d_s from N8), find q >= 0,
sum q = 1, with sum_s q_s d_s = 0 (imprint-neutral); C1 holds by Lemma 34.2. Then compare the optimal step0 write of
p1 = real unsure (2000) with p2 = p1 + q (equal mass), and evaluate the bound and the real-dev KL of w*(p2).
PREDICTION: an imprint-neutral q exists (residual < 1% of a typical |d_s|), and the bound is large: w*(p2) costs > .02
dev KL more than w*(p1) on real images.

**Checks X3-X9 (InceptionV1, curve family k = 3, M's read; real unsure 4000 as p1, its optimal write .0532 dev KL).**
| Alternative p2 (consistent with theta in...) | write change on real images |
|---|---|
| p1 + q, q = 12 sure inputs (C1, C2 trivially: zero gradients) | 0 (sure inputs are invisible to weights AND write) |
| p1 + q, q imprint-neutral unsure mixture (C1, C2 "equal"), eps up to 1000 | <= .0002 |
| p1 + q, same with a write-mass floor | infeasible (X8) |
| p1 + q, C2 as the theorem states it (direction of D; scale eta/2 lam B unknown) | no network-made mixture satisfies it at all (X9) |
| p1 + eps q, q = the whole network-made boundary + reader set (C1 only), eps = 1 / 10 / 100 | +.0000 / +.017 / +.053 (X6) |
| p1 + q, q = adversarial single input (C1 only), eps 10 | +.0009 (X5) |

**Theorem 35.1 (the impossibility, where it holds).** If training leaves only channel C1 (exact stationary points:
gradient flow with weight decay run to convergence), then the KL-optimal coarse network is not a function of the
weights. Proof: Theorem 34.3 with q any self-labeled measure; X6 exhibits one on InceptionV1 for which the optimal
writes of p1 and p1 + 100 q differ by .053 dev KL, so every weights-only method loses at least the bound of 34.3 on one
of them. The ambiguity is the MASS of "calibrated ambiguous" data (inputs whose true labels are drawn with exactly the
network's probabilities): it has zero expected gradient, so stationarity cannot see it, yet it carries output-metric
mass, which the write uses.

**Statement 35.2 (why the impossibility does not go through for SGD-trained networks, as far as tested).** Finite-step
training adds channel C2 (Theorem 31). Its constraint is on the DIRECTION of the per-neuron imbalance vector. Natural
inputs already point along it (N7: .70), so natural additions are consistent, but they are "more of the same" and leave
the write unchanged (X6 at eps = 1; X5). Every write-moving addition we could build points elsewhere (network-made sets
.21-.31; no mixture reaches the direction: X9), and every imprint-neutral mixture carries no write mass (X4, X7, X8).
So within everything tested, the imprint excludes exactly the alternatives that would move the write. The mechanism: a
write-moving input is one where the coarse units are active and the output is sensitive, which is large output-side
gradient energy for the family neurons; adding it shifts their imbalance components unless it is balanced the way
the training data was. This is evidence, not a proof, that channel C2 pins what the write needs.

**Answer to "fundamental or not".** (1) For networks trained to exact stationarity, the remaining dependence is
fundamental: proved (35.1), with a measured gap. (2) For networks trained by SGD (InceptionV1), the proof breaks at
channel C2, and the evidence says the imprint carries the missing information. Getting around the dependence is then a
precise network-only problem: build inputs whose per-neuron imbalance vectors are parallel to the weights' D at EVERY
layer. One layer (480 constraints) was gamed by tiny adversarial pixel changes (N10). The next step is all layers at
once (several thousand constraints), starting from natural-structure inputs.
