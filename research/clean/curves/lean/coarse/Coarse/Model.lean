import Mathlib
import Coarse.Basic

/-!
# COARSE-THEORY.md Part B.1: the symmetric model

* `perm_fixes_simple_eigvec` : a symmetry of a matrix fixes its simple eigenvector when the eigenvector has a
  non-zero entry sum (T1a part 3: the k = 1 read is constant on orbits).
* `const_of_rotate_invariant` : a vector on `Fin n` that is invariant under the cyclic shift is constant.
* `k1_read_coeff` : if the co-activity matrix has constant row sums, the k = 1 write coefficient for the summed
  read is `1 / N` for every member (T1a part 3: the write is the mean of the family's columns, data-free).
* `fires_alone_nonneg` : Fact F4 premise, a symmetry of a ReLU layer whose units each fire alone is non-negative.
* `bump_nonneg` : the bump `1 + cos (t - φ)` is non-negative (T1a part 4).
-/

open Finset BigOperators Matrix

/-- Fact F4 with its premise. Let `S` be a set of non-negative vectors (the outputs of a ReLU layer) that contains a
positive multiple of every basis vector (each unit fires alone on some input). If an orthogonal `Q` maps `S` into
itself, every entry of `Q` is non-negative, so (by `nonneg_orth_col_single`) `Q` is a permutation matrix. -/
theorem fires_alone_nonneg {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) (S : Set (Fin n → ℝ))
    (hS : ∀ v ∈ S, ∀ i, 0 ≤ v i) (hmaps : ∀ v ∈ S, Q *ᵥ v ∈ S)
    (halone : ∀ j, ∃ c : ℝ, 0 < c ∧ (fun i => if i = j then c else 0) ∈ S) :
    ∀ i j, 0 ≤ Q i j := by
  intro i j
  obtain ⟨c, hc, hmem⟩ := halone j
  have h := hS _ (hmaps _ hmem) i
  have : (Q *ᵥ (fun i => if i = j then c else 0)) i = Q i j * c := by
    simp [mulVec, dotProduct]
  rw [this] at h
  exact nonneg_of_mul_nonneg_left h hc

/-- T1a part 3. Let `σ` be a symmetry of `A` (`A (σ i) (σ j) = A i j`). If `v` is an eigenvector for `μ`, every
eigenvector for `μ` is a multiple of `v` (simple eigenvalue), and the entries of `v` have a non-zero sum, then `v`
is invariant under `σ`. -/
theorem perm_fixes_simple_eigvec {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (σ : Equiv.Perm (Fin n))
    (hA : ∀ i j, A (σ i) (σ j) = A i j) (μ : ℝ) (v : Fin n → ℝ) (hv : A *ᵥ v = μ • v)
    (hsimple : ∀ w : Fin n → ℝ, A *ᵥ w = μ • w → ∃ c : ℝ, w = c • v) (hsum : ∑ i, v i ≠ 0) :
    ∀ i, v (σ i) = v i := by
  set w : Fin n → ℝ := fun i => v (σ i) with hw
  have hAw : A *ᵥ w = μ • w := by
    funext i
    have h1 : (A *ᵥ w) i = ∑ j, A (σ i) (σ j) * v (σ j) := by
      simp only [mulVec, dotProduct, hw, hA]
    have h2 : ∑ j, A (σ i) (σ j) * v (σ j) = ∑ k, A (σ i) k * v k :=
      Equiv.sum_comp σ (fun k => A (σ i) k * v k)
    have h3 : (A *ᵥ v) (σ i) = μ * v (σ i) := by rw [hv]; simp
    rw [h1, h2]
    simpa [mulVec, dotProduct, hw] using h3
  obtain ⟨c, hc⟩ := hsimple w hAw
  have hsw : ∑ i, w i = ∑ i, v i := Equiv.sum_comp σ v
  have hc1 : c = 1 := by
    have : ∑ i, w i = c * ∑ i, v i := by
      rw [hc]; simp [Finset.mul_sum]
    rw [hsw] at this
    have h' : (c - 1) * ∑ i, v i = 0 := by linarith
    rcases mul_eq_zero.1 h' with h0 | h0
    · linarith
    · exact absurd h0 hsum
  intro i
  have := congrFun hc i
  simp [hw, hc1] at this
  exact this

/-- A vector on `Fin (n + 1)` that is invariant under the cyclic shift `finRotate` is constant. -/
theorem const_of_rotate_invariant {n : ℕ} (v : Fin (n + 1) → ℝ)
    (hv : ∀ i, v (finRotate (n + 1) i) = v i) : ∀ i, v i = v 0 := by
  have step : ∀ m : ℕ, ∀ h : m < n + 1, v ⟨m, h⟩ = v 0 := by
    intro m
    induction m with
    | zero => intro h; rfl
    | succ m ih =>
      intro h
      have hm : m < n + 1 := Nat.lt_of_succ_lt h
      have hlast : (⟨m, hm⟩ : Fin (n + 1)) ≠ Fin.last n := by
        intro he; have := congrArg Fin.val he; simp at this; omega
      have hrot : finRotate (n + 1) ⟨m, hm⟩ = ⟨m + 1, h⟩ := by
        rw [finRotate_of_lt (by simpa [Fin.lt_def] using
          (lt_of_le_of_ne (Fin.le_last _) hlast))]
      rw [← hrot, hv]; exact ih hm
  intro i; exact step i.val i.isLt

/-- T1a part 3. If the co-activity matrix `C` has constant row sums `c ≠ 0` (true for a circulant matrix), the
write coefficient of the summed read, `(C 1) / (1ᵀ C 1)`, equals `1 / N` for every member. -/
theorem k1_read_coeff {N : ℕ} (hN : 0 < N) (C : Matrix (Fin N) (Fin N) ℝ) (c : ℝ) (hc : c ≠ 0)
    (hrow : C *ᵥ (fun _ => (1 : ℝ)) = fun _ => c) :
    (∑ i, (C *ᵥ (fun _ => (1 : ℝ))) i)⁻¹ • (C *ᵥ (fun _ => (1 : ℝ))) = fun _ => (N : ℝ)⁻¹ := by
  rw [hrow]
  funext i
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Pi.smul_apply, smul_eq_mul]
  have hN' : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  field_simp

/-- T1a part 4. A bump is non-negative, so a read with bump coefficients is a non-negative combination. -/
theorem bump_nonneg (t φ : ℝ) : 0 ≤ 1 + Real.cos (t - φ) := by
  linarith [Real.neg_one_le_cos (t - φ)]
