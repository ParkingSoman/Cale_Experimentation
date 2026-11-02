import Mathlib

/-!
# COARSE-THEORY.md Part B.R: routes and uses (Definitions D43 to D45, Lemma T46a, Proposition T47)

* `route_mean_norm_le` : `‖∑ w_g • m_g‖ ≤ ∑ w_g ‖m_g‖` for weights `w_g ≥ 0`. It gives `π ≥ 0`, and, applied to the
  parts of one route, that splitting a route can only raise `π` (refinement).
* `packed_pi` : the packing index of the planted two-job unit, `1 - ((p + q)^2 / 2) / (p^2 + q^2)`, equals
  `(p - q)^2 / (2 (p^2 + q^2))`.
* `packed_pi_pos` : it is positive when `p ≠ q`.
-/

open Finset

/-- Lemma T46a parts 1 and 3: the norm of a weighted sum of route means is at most the weighted sum of their norms. -/
theorem route_mean_norm_le {ι E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E] (s : Finset ι)
    (w : ι → ℝ) (m : ι → E) (hw : ∀ g ∈ s, 0 ≤ w g) :
    ‖∑ g ∈ s, w g • m g‖ ≤ ∑ g ∈ s, w g * ‖m g‖ := by
  refine (norm_sum_le _ _).trans (le_of_eq (sum_congr rfl fun g hg => ?_))
  rw [norm_smul, Real.norm_of_nonneg (hw g hg)]

/-- Proposition T47 part 1: the closed form of the planted two-job unit's packing index. -/
theorem packed_pi (p q : ℝ) (h : p ^ 2 + q ^ 2 ≠ 0) :
    1 - ((p + q) ^ 2 / 2) / (p ^ 2 + q ^ 2) = (p - q) ^ 2 / (2 * (p ^ 2 + q ^ 2)) := by
  field_simp
  ring

/-- Proposition T47 part 1: the index is positive when the two routes' probabilities differ. -/
theorem packed_pi_pos (p q : ℝ) (hpq : p ≠ q) : 0 < (p - q) ^ 2 / (2 * (p ^ 2 + q ^ 2)) := by
  have h1 : 0 < (p - q) ^ 2 := by positivity
  have h2 : 0 < p ^ 2 + q ^ 2 := by
    rcases ne_or_eq p 0 with hp | hp
    · positivity
    · have : q ≠ 0 := by rintro rfl; exact hpq hp
      positivity
  positivity

/-- Proposition T47 part 1 (revision 2, class reads = the softmax Jacobian): with route uses proportional to
`(p, q, -(p + q))` and `(q, p, -(p + q))`, the final identity of the packing index. -/
theorem packed_pi_jac (p q : ℝ) (h : p ^ 2 + q ^ 2 + (p + q) ^ 2 ≠ 0) :
    1 - (3 * (p + q) ^ 2 / 2) / (p ^ 2 + q ^ 2 + (p + q) ^ 2)
      = (p - q) ^ 2 / (2 * (p ^ 2 + q ^ 2 + (p + q) ^ 2)) := by
  field_simp
  ring

/-- Proposition T47 part 1: its sign. -/
theorem packed_pi_jac_pos (p q : ℝ) (hpq : p ≠ q) :
    0 < (p - q) ^ 2 / (2 * (p ^ 2 + q ^ 2 + (p + q) ^ 2)) := by
  have h1 : 0 < (p - q) ^ 2 := by
    have : p - q ≠ 0 := sub_ne_zero.mpr hpq
    positivity
  have h2 : 0 < p ^ 2 + q ^ 2 + (p + q) ^ 2 := by
    have : 0 < (p - q) ^ 2 := h1
    nlinarith [sq_nonneg (p + q), sq_nonneg p, sq_nonneg q]
  positivity
