import Mathlib

/-!
# Coarse neuron theorem: the algebraic lemmas (NEURON-MATH.md)

* `relu_add_iff`  : L3 / P-Lemma 3, one ReLU carries a sum exactly when the terms agree in sign.
* `reynolds_le`   : L9 / Theorem 5.1, group averaging cannot raise a convex invariant objective.
* `nonneg_orth_col_single` : Prop. 6.1 (algebraic core), a non-negative orthogonal matrix has exactly one
  non-zero entry in each column, equal to 1 (so it is a permutation matrix).
* `write_in_span` : Prop. 7.2, the least-squares write lies in the span of the member columns, for any metric.
-/

open Finset BigOperators Matrix

/-- ReLU as `max x 0`. -/
noncomputable def relu (x : ℝ) : ℝ := max x 0

/-- L3: `relu (a + b) = relu a + relu b` exactly when `a` and `b` do not have opposite signs. -/
theorem relu_add_iff (a b : ℝ) :
    relu (a + b) = relu a + relu b ↔ 0 ≤ a * b := by
  unfold relu
  constructor
  · intro h
    by_contra hneg
    push Not at hneg
    rcases lt_or_gt_of_ne (show a ≠ 0 by rintro rfl; simp at hneg) with ha | ha
    · have hb : 0 < b := by nlinarith
      rcases le_or_gt (a + b) 0 with hab | hab
      · rw [max_eq_right hab, max_eq_right ha.le, max_eq_left hb.le] at h; linarith
      · rw [max_eq_left hab.le, max_eq_right ha.le, max_eq_left hb.le] at h; linarith
    · have hb : b < 0 := by nlinarith
      rcases le_or_gt (a + b) 0 with hab | hab
      · rw [max_eq_right hab, max_eq_left ha.le, max_eq_right hb.le] at h; linarith
      · rw [max_eq_left hab.le, max_eq_left ha.le, max_eq_right hb.le] at h; linarith
  · intro h
    rcases le_total 0 a with ha | ha <;> rcases le_total 0 b with hb | hb
    · rw [max_eq_left ha, max_eq_left hb, max_eq_left (by linarith)]
    · have : b = 0 ∨ a = 0 := by
        rcases eq_or_lt_of_le ha with h0 | h0
        · exact Or.inr h0.symm
        · left; nlinarith
      rcases this with rfl | rfl <;> simp [max_eq_left ha, max_eq_right hb]
    · have : a = 0 ∨ b = 0 := by
        rcases eq_or_lt_of_le hb with h0 | h0
        · exact Or.inr h0.symm
        · left; nlinarith
      rcases this with rfl | rfl <;> simp [max_eq_right ha, max_eq_left hb]
    · rw [max_eq_right ha, max_eq_right hb, max_eq_right (by linarith)]; ring

/-- L9 (Theorem 5.1): a convex objective that is invariant under a finite group action cannot increase when a
point is replaced by its group average (the Reynolds operator). -/
theorem reynolds_le {G E : Type*} [Fintype G] [Nonempty G] [AddCommGroup E] [Module ℝ E]
    (act : G → E → E) (J : E → ℝ) (hJ : ConvexOn ℝ Set.univ J) (hinv : ∀ g x, J (act g x) = J x) (x : E) :
    J ((Fintype.card G : ℝ)⁻¹ • ∑ g, act g x) ≤ J x := by
  have hc : (0 : ℝ) < Fintype.card G := by exact_mod_cast Fintype.card_pos
  have key := hJ.map_sum_le (t := Finset.univ) (w := fun _ => (Fintype.card G : ℝ)⁻¹) (p := fun g => act g x)
    (fun _ _ => by positivity) (by simp [Finset.card_univ]) (fun _ _ => Set.mem_univ _)
  simp only [hinv, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, smul_eq_mul] at key
  rw [Finset.smul_sum]
  calc J (∑ g, (Fintype.card G : ℝ)⁻¹ • act g x) ≤ (Fintype.card G : ℝ) * ((Fintype.card G : ℝ)⁻¹ * J x) := key
    _ = J x := by field_simp

/-- Prop. 6.1, step 1: two non-negative vectors with inner product zero have disjoint supports. -/
theorem nonneg_orth_disjoint {n : ℕ} (u v : Fin n → ℝ) (hu : ∀ i, 0 ≤ u i) (hv : ∀ i, 0 ≤ v i)
    (h : ∑ i, u i * v i = 0) (i : Fin n) : u i * v i = 0 :=
  (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => mul_nonneg (hu j) (hv j))).1 h i (Finset.mem_univ i)

/-- Prop. 6.1, algebraic core: if `Q` has non-negative entries and orthonormal rows and columns, every column has
exactly one non-zero entry, and that entry is 1. Hence `Q` is a permutation matrix. -/
theorem nonneg_orth_col_single {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) (hQ : ∀ i j, 0 ≤ Q i j)
    (hcol : Qᵀ * Q = 1) (hrow : Q * Qᵀ = 1) (j : Fin n) :
    ∃ i, Q i j = 1 ∧ ∀ i', i' ≠ i → Q i' j = 0 := by
  -- column j has norm 1
  have hnorm : ∑ i, Q i j * Q i j = 1 := by
    have := congrFun (congrFun hcol j) j
    simpa [Matrix.mul_apply, Matrix.one_apply] using this
  -- some entry of column j is non-zero
  obtain ⟨i, hi⟩ : ∃ i, Q i j ≠ 0 := by
    by_contra hall; push Not at hall; simp [hall] at hnorm
  -- rows are orthogonal and non-negative, so column j has at most one non-zero entry
  have honly : ∀ i', i' ≠ i → Q i' j = 0 := by
    intro i' hne
    have horth : ∑ k, Q i k * Q i' k = 0 := by
      have := congrFun (congrFun hrow i) i'
      simpa [Matrix.mul_apply, Matrix.one_apply, Ne.symm hne] using this
    have := nonneg_orth_disjoint (fun k => Q i k) (fun k => Q i' k) (fun k => hQ i k) (fun k => hQ i' k) horth j
    rcases mul_eq_zero.1 this with h0 | h0
    · exact absurd h0 hi
    · exact h0
  refine ⟨i, ?_, honly⟩
  -- the norm reduces to the single entry, which is non-negative with square 1
  have hsum : ∑ k, Q k j * Q k j = Q i j * Q i j := by
    rw [Finset.sum_eq_single i (fun k _ hk => by rw [honly k hk]; ring) (by simp)]
  rw [hsum] at hnorm
  have h0 := hQ i j
  nlinarith [sq_nonneg (Q i j - 1), sq_nonneg (Q i j + 1)]

/-- Prop. 7.2: if `U = W * H` (reader inputs are the member columns times member activities) and the write `B`
satisfies the normal equations `M * (U - B * C) * Cᵀ = 0` for an invertible metric `M` and invertible `C * Cᵀ`,
then `B` lies in the column span of `W`: `B = W * Λ` for some `Λ`, and the metric `M` does not matter. -/
theorem write_in_span {r m s k : ℕ} (W : Matrix (Fin r) (Fin m) ℝ) (H : Matrix (Fin m) (Fin s) ℝ)
    (C : Matrix (Fin k) (Fin s) ℝ) (B : Matrix (Fin r) (Fin k) ℝ) (M : Matrix (Fin r) (Fin r) ℝ)
    (hM : IsUnit M.det) (hC : IsUnit (C * Cᵀ).det)
    (hnormal : M * (W * H - B * C) * Cᵀ = 0) :
    B = W * (H * Cᵀ * (C * Cᵀ)⁻¹) := by
  have h1 : (W * H - B * C) * Cᵀ = 0 := by
    have h := congrArg (fun X => M⁻¹ * X) hnormal
    simp only [Matrix.mul_zero] at h
    rwa [Matrix.mul_assoc, ← Matrix.mul_assoc M⁻¹, Matrix.nonsing_inv_mul _ hM, Matrix.one_mul] at h
  have h2 : W * H * Cᵀ = B * (C * Cᵀ) := by
    rw [Matrix.sub_mul, sub_eq_zero] at h1; simpa [Matrix.mul_assoc] using h1
  calc B = B * (C * Cᵀ) * (C * Cᵀ)⁻¹ := by rw [Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hC, Matrix.mul_one]
    _ = W * H * Cᵀ * (C * Cᵀ)⁻¹ := by rw [h2]
    _ = W * (H * Cᵀ * (C * Cᵀ)⁻¹) := by simp [Matrix.mul_assoc]
