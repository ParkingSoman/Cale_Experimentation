import Mathlib

/-!
# COARSE-THEORY.md Part B.4: the fold of a gated chain (finite steps)

* `softmax_var_le` : Fact F11, the variance lemma: for a probability vector `q`,
  `∑ q v² - (∑ q v)² ≤ (1/2) ∑ v²`.
* `gated_sum` : Proposition T7, a combination of members that share one gate is one gated unit.
* `gate_commute` : Lemma T8, a scalar gate commutes with an edge matrix.
* `chain_rank_le` : Lemma T8, a map that passes through `k` units has rank at most `k`.
* `nested_proj` : Theorem T6, for orthogonal projections with nested ranges, `P Q = P`.
* `merge_compose` : Theorem T6, merges between nested levels compose.
* `proj_commute_chain` : Theorem T10, a projection that commutes with every edge commutes with the whole chain.
-/

open Finset BigOperators Matrix

/-- Fact F11 (variance lemma). The variance of `v` under a probability vector `q` is at most half of `|v|²`. -/
theorem softmax_var_le {n : ℕ} (q v : Fin n → ℝ) (hq : ∀ i, 0 ≤ q i) (hsum : ∑ i, q i = 1) :
    ∑ i, q i * v i ^ 2 - (∑ i, q i * v i) ^ 2 ≤ (1 / 2) * ∑ i, v i ^ 2 := by
  -- the double-sum identity
  have hexp : ∀ i j, q i * q j * (v i - v j) ^ 2 =
      q i * v i ^ 2 * q j + q j * v j ^ 2 * q i - 2 * (q i * v i) * (q j * v j) := by
    intro i j; ring
  have hid : ∑ i, ∑ j, q i * q j * (v i - v j) ^ 2 =
      2 * (∑ i, q i * v i ^ 2) - 2 * (∑ i, q i * v i) ^ 2 := by
    simp_rw [hexp, Finset.sum_sub_distrib, Finset.sum_add_distrib]
    have a1 : ∑ i, ∑ j, q i * v i ^ 2 * q j = ∑ i, q i * v i ^ 2 := by
      simp_rw [← Finset.mul_sum, hsum, mul_one]
    have a2 : ∑ i, ∑ j, q j * v j ^ 2 * q i = ∑ i, q i * v i ^ 2 := by
      rw [Finset.sum_comm]; simp_rw [← Finset.mul_sum, hsum, mul_one]
    have a3 : ∑ i, ∑ j, 2 * (q i * v i) * (q j * v j) = 2 * (∑ i, q i * v i) ^ 2 := by
      have hi : ∀ i, ∑ j, 2 * (q i * v i) * (q j * v j) = 2 * (q i * v i) * ∑ j, q j * v j :=
        fun i => by rw [Finset.mul_sum]
      simp_rw [hi, ← Finset.sum_mul, ← Finset.mul_sum]; ring
    rw [a1, a2, a3]; ring
  -- termwise bound, with the diagonal handled exactly
  have hterm : ∀ i j, q i * q j * (v i - v j) ^ 2 ≤
      2 * q i * q j * (v i ^ 2 + v j ^ 2) - (if i = j then 4 * q i ^ 2 * v i ^ 2 else 0) := by
    intro i j
    split_ifs with h
    · subst h; nlinarith [sq_nonneg (v i), hq i]
    · have hqq : 0 ≤ q i * q j := mul_nonneg (hq i) (hq j)
      nlinarith [mul_nonneg hqq (sq_nonneg (v i + v j))]
  have hbound : ∑ i, ∑ j, q i * q j * (v i - v j) ^ 2 ≤
      ∑ i, ∑ j, (2 * q i * q j * (v i ^ 2 + v j ^ 2) - (if i = j then 4 * q i ^ 2 * v i ^ 2 else 0)) :=
    Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => hterm i j))
  have hrhs : ∑ i, ∑ j, (2 * q i * q j * (v i ^ 2 + v j ^ 2) - (if i = j then 4 * q i ^ 2 * v i ^ 2 else 0)) =
      ∑ i, (4 * q i * v i ^ 2 - 4 * q i ^ 2 * v i ^ 2) := by
    have e : ∀ i j, 2 * q i * q j * (v i ^ 2 + v j ^ 2) =
        2 * (q i * v i ^ 2) * q j + 2 * (q j * v j ^ 2) * q i := by
      intro i j; ring
    simp_rw [Finset.sum_sub_distrib, e, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_univ,
      ite_true]
    have b1 : ∑ i, ∑ j, 2 * (q i * v i ^ 2) * q j = 2 * ∑ i, q i * v i ^ 2 := by
      simp_rw [← Finset.mul_sum, hsum, mul_one]; rw [Finset.mul_sum]
    have b2 : ∑ i, ∑ j, 2 * (q j * v j ^ 2) * q i = 2 * ∑ i, q i * v i ^ 2 := by
      rw [Finset.sum_comm]; simp_rw [← Finset.mul_sum, hsum, mul_one]; rw [Finset.mul_sum]
    have b4 : ∑ x, 4 * q x * v x ^ 2 = 4 * ∑ i, q i * v i ^ 2 := by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i _ => by ring)
    rw [b1, b2, b4]; ring
  have hfin : ∑ i, (4 * q i * v i ^ 2 - 4 * q i ^ 2 * v i ^ 2) ≤ ∑ i, v i ^ 2 :=
    Finset.sum_le_sum (fun i _ => by nlinarith [mul_nonneg (sq_nonneg (v i)) (sq_nonneg (1 - 2 * q i))])
  linarith

/-- Proposition T7. Members that share one gate `g`: any combination of their outputs is `g` times the same
combination of their pre-activations, so it is one gated unit with the combined kernel. -/
theorem gated_sum {N : ℕ} (g : ℝ) (lam z : Fin N → ℝ) :
    ∑ i, lam i * (g * z i) = g * ∑ i, lam i * z i := by
  rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i _ => by ring)

/-- Lemma T8. A scalar gate commutes with an edge matrix. -/
theorem gate_commute {m n : ℕ} (w : Matrix (Fin m) (Fin n) ℝ) (g : ℝ) (v : Fin n → ℝ) :
    w *ᵥ (g • v) = g • (w *ᵥ v) :=
  Matrix.mulVec_smul w g v

/-- Lemma T8. A linear map that passes through `k` units has rank at most `k`. -/
theorem chain_rank_le {m n k : ℕ} (A : Matrix (Fin m) (Fin k) ℝ) (B : Matrix (Fin k) (Fin n) ℝ) :
    (A * B).rank ≤ k := by
  have h1 := Matrix.rank_mul_le_right A B
  have h2 := Matrix.rank_le_card_height B
  simp only [Fintype.card_fin] at h2
  omega

/-- Theorem T6. If `P` and `Q` are symmetric and the range of `P` lies in the range of `Q` (`Q P = P`), then
`P Q = P`: folding to the finer level and then to the coarser level is the same as folding to the coarser level. -/
theorem nested_proj {n : ℕ} (P Q : Matrix (Fin n) (Fin n) ℝ) (hP : Pᵀ = P) (hQ : Qᵀ = Q) (hQP : Q * P = P) :
    P * Q = P := by
  have h := congrArg Matrix.transpose hQP
  rw [Matrix.transpose_mul, hP, hQ] at h
  exact h

/-- Theorem T6. If the columns of `U₁` lie in the span of the orthonormal columns of `U₂` (`U₂ U₂ᵀ U₁ = U₁`), the
merge from level 3 to level 1 equals the merge from 3 to 2 followed by the merge from 2 to 1. -/
theorem merge_compose {n k₁ k₂ k₃ : ℕ} (U₁ : Matrix (Fin n) (Fin k₁) ℝ) (U₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (U₃ : Matrix (Fin n) (Fin k₃) ℝ) (h : U₂ * U₂ᵀ * U₁ = U₁) :
    (U₁ᵀ * U₂) * (U₂ᵀ * U₃) = U₁ᵀ * U₃ := by
  have ht := congrArg Matrix.transpose h
  rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose] at ht
  calc (U₁ᵀ * U₂) * (U₂ᵀ * U₃) = (U₁ᵀ * (U₂ * U₂ᵀ)) * U₃ := by simp only [Matrix.mul_assoc]
    _ = U₁ᵀ * U₃ := by rw [ht]

/-- Theorem T10. A projection that commutes with every edge of a chain commutes with their product, so truncating
the first layer is the same as truncating every layer. -/
theorem proj_commute_chain {n : ℕ} (P : Matrix (Fin n) (Fin n) ℝ) (ws : List (Matrix (Fin n) (Fin n) ℝ))
    (h : ∀ w ∈ ws, P * w = w * P) : P * ws.prod = ws.prod * P := by
  induction ws with
  | nil => simp
  | cons w ws ih =>
    have hw : P * w = w * P := h w (by simp)
    have hrest : P * ws.prod = ws.prod * P := ih (fun u hu => h u (by simp [hu]))
    rw [List.prod_cons, ← Matrix.mul_assoc, hw, Matrix.mul_assoc, hrest, Matrix.mul_assoc]

/-- Corollary T10.3 (lower variance bound). If every weight of a probability vector `q` is at least `c ≥ 0`, the
variance of `v` under `q` is at least `c` times the plain sum of squares about the plain mean. -/
theorem softmax_var_ge {n : ℕ} (hn : 0 < n) (q v : Fin n → ℝ) (c : ℝ) (hc : 0 ≤ c)
    (hq : ∀ i, c ≤ q i) (hsum : ∑ i, q i = 1) :
    c * (∑ i, v i ^ 2 - (∑ i, v i) ^ 2 / n) ≤ ∑ i, q i * v i ^ 2 - (∑ i, q i * v i) ^ 2 := by
  set m := ∑ i, q i * v i with hm
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  -- the q-variance as a weighted sum of squares about m
  have hvar : ∑ i, q i * v i ^ 2 - m ^ 2 = ∑ i, q i * (v i - m) ^ 2 := by
    have e : ∀ i, q i * (v i - m) ^ 2 = q i * v i ^ 2 - 2 * m * (q i * v i) + m ^ 2 * q i := by
      intro i; ring
    simp_rw [e, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum, ← hm]; ring
  -- termwise lower bound by c
  have hterm : c * ∑ i, (v i - m) ^ 2 ≤ ∑ i, q i * (v i - m) ^ 2 := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_right (hq i) (sq_nonneg _))
  -- the plain mean minimises the plain sum of squares
  have hmin : ∑ i, v i ^ 2 - (∑ i, v i) ^ 2 / n ≤ ∑ i, (v i - m) ^ 2 := by
    have e : ∑ i, (v i - m) ^ 2 = ∑ i, v i ^ 2 - 2 * m * ∑ i, v i + n * m ^ 2 := by
      have e' : ∀ i, (v i - m) ^ 2 = v i ^ 2 - 2 * m * v i + m ^ 2 := by intro i; ring
      simp_rw [e', Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const,
        Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [e]
    have key : 0 ≤ (n * m - ∑ i, v i) ^ 2 / n := div_nonneg (sq_nonneg _) hn'.le
    have : (n * m - ∑ i, v i) ^ 2 / n = n * m ^ 2 - 2 * m * ∑ i, v i + (∑ i, v i) ^ 2 / n := by
      field_simp; ring
    linarith
  rw [hvar]
  nlinarith [mul_le_mul_of_nonneg_left hmin hc]

/-- Theorem T9 part 4 (the gap step, in an eigenbasis). Let `S` hold the `k` indices of the top eigenvalues
(`μ j ≥ a` on `S`, `μ j ≤ b` off `S`, `b ≤ a`), and let `q j ∈ [0, 1]` with `∑ q = k` (the diagonal of a rank-`k`
projection in that basis). Then `tr(M P) - tr(M Q) ≥ (a - b) * (k - tr(P Q))`, written in coordinates. -/
theorem gap_trace {n k : ℕ} (μ q : Fin n → ℝ) (S : Finset (Fin n)) (hS : S.card = k) (a b : ℝ)
    (hin : ∀ j ∈ S, a ≤ μ j) (hout : ∀ j ∉ S, μ j ≤ b)
    (hq0 : ∀ j, 0 ≤ q j) (hq1 : ∀ j, q j ≤ 1) (hsum : ∑ j, q j = k) :
    (a - b) * ∑ j ∈ Sᶜ, q j ≤ ∑ j ∈ S, μ j * (1 - q j) - ∑ j ∈ Sᶜ, μ j * q j := by
  have hsplit : ∑ j ∈ S, q j + ∑ j ∈ Sᶜ, q j = k := by rw [Finset.sum_add_sum_compl, hsum]
  have h1 : a * ∑ j ∈ S, (1 - q j) ≤ ∑ j ∈ S, μ j * (1 - q j) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun j hj => mul_le_mul_of_nonneg_right (hin j hj) (by linarith [hq1 j]))
  have h2 : ∑ j ∈ Sᶜ, μ j * q j ≤ b * ∑ j ∈ Sᶜ, q j := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun j hj => mul_le_mul_of_nonneg_right
      (hout j (Finset.mem_compl.mp hj)) (hq0 j))
  have h3 : ∑ j ∈ S, (1 - q j) = ∑ j ∈ Sᶜ, q j := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, hS, nsmul_eq_mul, mul_one]; linarith
  rw [h3] at h1
  nlinarith
