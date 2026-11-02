import Mathlib

/-!
# COARSE-THEORY.md Part B.H: one recursion over a tree of areas

ReLU is written `max t 0`.

* `relu_affine_mono` : `max (mu * g + nu) 0` is monotone in `g` when `0 ≤ mu`.
* `no_zigzag` : Proposition TH2 part 2. A gap `a * relu (mu * g + nu) + gamma` never takes the values 0, 1, 0, 1 at
  `g = 1, 2, 3, 4`.
* `relu_pos_eq` : Proposition TH3. Two equal positive ReLU values come from equal pre-activations.
* `capture_le` : Proposition TH3. `4 c ^ 2 + 3 s ^ 2 ≤ 4` when `c ^ 2 + s ^ 2 ≤ 1`.

Part B.P (summaries passed up the tree):
* `energy_split` : Theorem TP1. A weighted residual energy splits when the two parts are orthogonal.
* `trace_bound` : Theorem TP1 part 1. The inequality step of the lower bound.
* `topk_trunc` : Theorem TP2 part 3. A compressed summary costs at most half the trace it drops.
* `th3_summary` : Theorem TP2 part 4. The numbers 3/2, 2 and 3 of Proposition TH3's instance.
-/

/-- A ReLU of an affine function with a non-negative slope is monotone. -/
theorem relu_affine_mono (mu nu g g' : ℝ) (hmu : 0 ≤ mu) (hg : g ≤ g') :
    max (mu * g + nu) 0 ≤ max (mu * g' + nu) 0 := by
  have : mu * g + nu ≤ mu * g' + nu := by nlinarith
  exact max_le_max this le_rfl

/-- Proposition TH2 part 2: one unit on one input `g` cannot zigzag. With `r_i = relu (mu * i + nu)`, the
differences `r_2 - r_1` and `r_3 - r_2` share a weak sign (monotone), but the zigzag needs
`a (r_2 - r_1) = 1` and `a (r_3 - r_2) = -1`. -/
theorem no_zigzag (a mu nu gamma : ℝ) :
    ¬ (a * max (mu * 1 + nu) 0 + gamma = 0 ∧ a * max (mu * 2 + nu) 0 + gamma = 1 ∧
       a * max (mu * 3 + nu) 0 + gamma = 0 ∧ a * max (mu * 4 + nu) 0 + gamma = 1) := by
  rintro ⟨h1, h2, h3, _⟩
  set r1 := max (mu * 1 + nu) 0
  set r2 := max (mu * 2 + nu) 0
  set r3 := max (mu * 3 + nu) 0
  have e1 : a * (r2 - r1) = 1 := by linarith
  have e2 : a * (r3 - r2) = -1 := by linarith
  have hprod : 0 ≤ (r2 - r1) * (r3 - r2) := by
    rcases le_total 0 mu with hmu | hmu
    · have := relu_affine_mono mu nu 1 2 hmu (by norm_num)
      have := relu_affine_mono mu nu 2 3 hmu (by norm_num)
      exact mul_nonneg (by linarith) (by linarith)
    · have m1 := relu_affine_mono (-mu) nu (-2) (-1) (by linarith) (by norm_num)
      have m2 := relu_affine_mono (-mu) nu (-3) (-2) (by linarith) (by norm_num)
      have q1 : -mu * (-1) + nu = mu * 1 + nu := by ring
      have q2 : -mu * (-2) + nu = mu * 2 + nu := by ring
      have q3 : -mu * (-3) + nu = mu * 3 + nu := by ring
      rw [q1, q2] at m1
      rw [q2, q3] at m2
      exact mul_nonneg_of_nonpos_of_nonpos (by linarith) (by linarith)
  have : a * (r2 - r1) * (a * (r3 - r2)) = -1 := by rw [e1, e2]; ring
  nlinarith [sq_nonneg a, mul_nonneg (sq_nonneg a) hprod]

/-- Proposition TH3: if two ReLU values are equal and positive, the pre-activations are equal. -/
theorem relu_pos_eq (q q' : ℝ) (h : max q 0 = max q' 0) (hpos : 0 < max q 0) : q = q' := by
  have hq : 0 < q := by
    by_contra hc; push Not at hc; rw [max_eq_right hc] at hpos; exact lt_irrefl 0 hpos
  have hq' : 0 < q' := by
    by_contra hc; push Not at hc; rw [max_eq_right hc, max_eq_left hq.le] at h; linarith
  rwa [max_eq_left hq.le, max_eq_left hq'.le] at h

/-- Proposition TH3: the step-by-step unit captures at most `4 cos ^ 2 + 3 sin ^ 2 ≤ 4`. -/
theorem capture_le (c s : ℝ) (h : c ^ 2 + s ^ 2 ≤ 1) : 4 * c ^ 2 + 3 * s ^ 2 ≤ 4 := by
  nlinarith [sq_nonneg s]

/-- Theorem TP1: if `r` is orthogonal to `u` in the weighted inner product, the energy of `r + u` is the sum of the
energies (one coordinate; the vector case is a sum of these). -/
theorem energy_split {ι : Type*} (s : Finset ι) (p r u : ι → ℝ) (h : ∑ i ∈ s, p i * r i * u i = 0) :
    ∑ i ∈ s, p i * (r i + u i) ^ 2 = ∑ i ∈ s, p i * r i ^ 2 + ∑ i ∈ s, p i * u i ^ 2 := by
  have e : ∀ i ∈ s, p i * (r i + u i) ^ 2 = p i * r i ^ 2 + p i * u i ^ 2 + 2 * (p i * r i * u i) := by
    intro i _; ring
  rw [Finset.sum_congr rfl e, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, h]
  ring

/-- Theorem TP1 part 1: with `c ≤ top` (Ky Fan) and `0 ≤ a ≤ eps` (the captured residual is at most its trace), the
terms `(1/2)(E - c - a)` are at least `(1/2)(E - top) - (1/2) eps`. -/
theorem trace_bound (E c a top eps : ℝ) (hc : c ≤ top) (ha : a ≤ eps) :
    (1 / 2) * (E - top) - (1 / 2) * eps ≤ (1 / 2) * (E - c - a) := by
  linarith

/-- Theorem TP2 part 3: with `top ≤ top' + tau` (subadditivity, trace bound) and `top' ≤ x` (the build captures at
least the compressed top), the build's terms `(1/2)(E - x)` are at most `(1/2)(E - top) + tau / 2`. -/
theorem topk_trunc (E x top top' tau : ℝ) (h1 : top ≤ top' + tau) (h2 : top' ≤ x) :
    (1 / 2) * (E - x) ≤ (1 / 2) * (E - top) + tau / 2 := by
  linarith

/-- Theorem TP2 part 4: in Proposition TH3's instance, `E = 7 + 3`; the summary rule gives `(1/2)(7 - 4) + (1/2)(3 - 3)`
at G_1 and `(1/2)(10 - 6)` at G_2; the compressed summary gives `(1/2)(10 - 4)`, within `2 + 3/2`. -/
theorem th3_summary :
    (1 / 2 : ℝ) * (7 - 4) + (1 / 2) * (3 - 3) = 3 / 2 ∧ (1 / 2 : ℝ) * (10 - 6) = 2 ∧
    (1 / 2 : ℝ) * (10 - 4) = 3 ∧ (3 : ℝ) ≤ 2 + 3 / 2 := by
  norm_num
