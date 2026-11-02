import Mathlib
import Coarse.Basic

/-!
# Part IV (NEURON-MATH.md section 14): exactness lemmas

* `quad_moment_suff` : Prop. 13.5, sufficiency. A weighted squared error depends on the mix only through the
  weighted second-moment matrix. Two mixes with the same matrix give the same objective for every parameter.
* `gated_write_in_span` : Prop. 13.5, the minimiser. With a diagonal non-negative weight (the reader's gate), the
  normal equations give `B = W * (H * Dg * Cᵀ * (C * Dg * Cᵀ)⁻¹)`: the re-routing is the old family weights times
  one per-reader matrix built from the GATED moments.
* `exact_no_gain` : Corollary E.1, the logic. If a build minimises the objective of a mix, and a second mix has the
  same sufficient statistic, no parameter does better on the second mix.
-/

open Finset BigOperators Matrix

/-- Weighted second moment of a finite mix: `S = ∑ₓ p x • z x z xᵀ` (the weight `p x` may include a gate). -/
def moment {X d : ℕ} (p : Fin X → ℝ) (z : Fin X → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  fun a b => ∑ x, p x * z x a * z x b

/-- The weighted squared error of a linear read-out `v` equals the quadratic form of the moment. -/
theorem quad_eq_moment {X d : ℕ} (p : Fin X → ℝ) (z : Fin X → Fin d → ℝ) (v : Fin d → ℝ) :
    ∑ x, p x * (∑ a, v a * z x a) ^ 2 = ∑ a, ∑ b, v a * moment p z a b * v b := by
  unfold moment
  simp only [sq, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun b _ => ?_)
  refine Finset.sum_congr rfl (fun x _ => ?_)
  ring

/-- Prop. 13.5, sufficiency: two mixes (any supports, any weights) with the same weighted moment give the same
objective for every read-out `v`. -/
theorem quad_moment_suff {X Y d : ℕ} (p : Fin X → ℝ) (z : Fin X → Fin d → ℝ) (q : Fin Y → ℝ)
    (y : Fin Y → Fin d → ℝ) (h : moment p z = moment q y) (v : Fin d → ℝ) :
    ∑ x, p x * (∑ a, v a * z x a) ^ 2 = ∑ x, q x * (∑ a, v a * y x a) ^ 2 := by
  rw [quad_eq_moment, quad_eq_moment, h]

/-- Prop. 13.5, the gated minimiser. `Dg` is the diagonal gate weight over samples. If the write `B` solves the gated
normal equations for an invertible metric `M`, it equals the family weights `W` times one matrix from gated moments. -/
theorem gated_write_in_span {r m s k : ℕ} (W : Matrix (Fin r) (Fin m) ℝ) (H : Matrix (Fin m) (Fin s) ℝ)
    (C : Matrix (Fin k) (Fin s) ℝ) (Dg : Matrix (Fin s) (Fin s) ℝ) (B : Matrix (Fin r) (Fin k) ℝ)
    (M : Matrix (Fin r) (Fin r) ℝ) (hM : IsUnit M.det) (hC : IsUnit (C * Dg * Cᵀ).det)
    (hnormal : M * (W * H - B * C) * Dg * Cᵀ = 0) :
    B = W * (H * Dg * Cᵀ * (C * Dg * Cᵀ)⁻¹) := by
  have h1 : (W * H - B * C) * Dg * Cᵀ = 0 := by
    have h := congrArg (fun X => M⁻¹ * X) hnormal
    simp only [Matrix.mul_zero] at h
    simpa [Matrix.mul_assoc, ← Matrix.mul_assoc M⁻¹, Matrix.nonsing_inv_mul _ hM] using h
  have h2 : W * H * Dg * Cᵀ = B * (C * Dg * Cᵀ) := by
    rw [Matrix.sub_mul, Matrix.sub_mul, sub_eq_zero] at h1; simpa [Matrix.mul_assoc] using h1
  calc B = B * (C * Dg * Cᵀ) * (C * Dg * Cᵀ)⁻¹ := by
          rw [Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hC, Matrix.mul_one]
    _ = W * H * Dg * Cᵀ * (C * Dg * Cᵀ)⁻¹ := by rw [h2]
    _ = W * (H * Dg * Cᵀ * (C * Dg * Cᵀ)⁻¹) := by simp [Matrix.mul_assoc]

/-- Corollary E.1, the logic: if `θ₀` minimises `L m₁` and `m₂ = m₁`, then no `θ` does better on `L m₂`. Training
on an uninformative mix (same sufficient statistic) has zero gain over the exact formula. -/
theorem exact_no_gain {Θ Mstat : Type*} (L : Mstat → Θ → ℝ) (m₁ m₂ : Mstat) (θ₀ : Θ)
    (hmin : ∀ θ, L m₁ θ₀ ≤ L m₁ θ) (hsame : m₂ = m₁) : ∀ θ, L m₂ θ₀ ≤ L m₂ θ := by
  subst hsame; exact hmin
