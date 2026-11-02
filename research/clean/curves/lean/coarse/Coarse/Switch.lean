import Mathlib

/-!
# COARSE-THEORY.md Part B.7: formulas for the gates' reads (finite steps)

* `fiber_split` : Lemma T17a, finite form. On one value of the kept features (a fiber of a finite law), the weighted
  mean is the best constant, with the exact split of the weighted squared error.
* `gate_expand` : Theorem T17, the expansion of an affine gate times a value when the input is
  split into a seen part and an unseen part.
* `resolvent_shift` : Theorem T17.2 part 3, the termwise identity `μ D - B = A` behind the stationarity polynomial.
* `antisymm_trace_sq` : Proposition T17.3 part 2, the trace of the square of an antisymmetric matrix is minus the sum
  of its squared entries, so a zero trace forces the matrix to vanish.
* `antisymm_zero_of_trace` : Proposition T17.3 part 2, the consequence: an antisymmetric matrix whose square has
  zero trace is zero.
-/

open Finset BigOperators Matrix

/-- Lemma T17a (one fiber). For weights `p` with nonzero total `P` and the weighted mean `m = (∑ p y) / P`, the
weighted squared error to any constant `c` splits into the error to the mean plus `P (m - c)²`. -/
theorem fiber_split {n : ℕ} (p y : Fin n → ℝ) (c : ℝ) (hP : ∑ i, p i ≠ 0) :
    ∑ i, p i * (y i - c) ^ 2 =
      ∑ i, p i * (y i - (∑ j, p j * y j) / ∑ j, p j) ^ 2
        + (∑ j, p j) * ((∑ j, p j * y j) / ∑ j, p j - c) ^ 2 := by
  set P := ∑ j, p j with hPdef
  set S := ∑ j, p j * y j with hSdef
  have key : ∀ i, p i * (y i - c) ^ 2 =
      p i * (y i - S / P) ^ 2 + p i * (S / P - c) ^ 2 + 2 * (S / P - c) * (p i * y i - p i * (S / P)) := by
    intro i; ring
  have hcross : ∑ i, (p i * y i - p i * (S / P)) = 0 := by
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul]
    field_simp
    ring
  rw [Finset.sum_congr rfl (fun i _ => key i), Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, hcross, ← Finset.sum_mul]
  ring

/-- Theorem T17 (the expansion). With the input split as seen part `v₀` plus unseen part `e`, and the
gate argument split as `t₀ + u`, an affine gate times the value expands into four terms; given the seen part, only the
first and the last survive in expectation. -/
theorem gate_expand (α β t₀ u v₀ e : ℝ) :
    (α * (t₀ + u) + β) * (v₀ + e) = (α * t₀ + β) * v₀ + (α * t₀ + β) * e + α * u * v₀ + α * u * e := by
  ring

/-- Theorem T17.2 part 3 (one term). For `μ ≠ λ`, `μ r / (μ - λ)² - λ r / (μ - λ)² = r / (μ - λ)`. Summed over the
eigenvalues this is `μ D - B = A`. -/
theorem resolvent_shift (μ l r : ℝ) (h : μ ≠ l) :
    μ * (r / (μ - l) ^ 2) - l * (r / (μ - l) ^ 2) = r / (μ - l) := by
  have h' : μ - l ≠ 0 := sub_ne_zero.mpr h
  field_simp

/-- Proposition T17.3 part 2. For an antisymmetric matrix `Y`, `tr (Y Y) = - ∑ Y i j ²`. -/
theorem antisymm_trace_sq {n : ℕ} (Y : Matrix (Fin n) (Fin n) ℝ) (h : Yᵀ = -Y) :
    Matrix.trace (Y * Y) = -∑ i, ∑ j, Y i j ^ 2 := by
  have hji : ∀ i j, Y j i = -Y i j := by
    intro i j
    have := congrFun (congrFun h i) j
    simpa [Matrix.transpose_apply] using this
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply]
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [hji i j]
  ring

/-- Proposition T17.3 part 2 (the consequence). An antisymmetric matrix whose square has zero trace is zero. -/
theorem antisymm_zero_of_trace {n : ℕ} (Y : Matrix (Fin n) (Fin n) ℝ) (h : Yᵀ = -Y)
    (h0 : Matrix.trace (Y * Y) = 0) : Y = 0 := by
  rw [antisymm_trace_sq Y h] at h0
  have hsum : ∑ i, ∑ j, Y i j ^ 2 = 0 := by linarith
  have hnn : ∀ i, 0 ≤ ∑ j, Y i j ^ 2 := fun i => Finset.sum_nonneg (fun j _ => sq_nonneg _)
  ext i j
  have hi : ∑ j, Y i j ^ 2 = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => hnn i)).mp hsum i (Finset.mem_univ i)
  have hij : Y i j ^ 2 = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg (Y i j))).mp hi j (Finset.mem_univ j)
  simpa using hij
