import Mathlib

/-!
# COARSE-THEORY.md Part B.5: self-contained folds and a lower bound for every network (finite steps)

* `faithful_read` : Lemma T11a, a build whose effective read kernels equal the network's reads the same value
  from the same gated input.
* `kyfan_coord` : Theorems T11 and T12 and Proposition T17.3 part 4, Ky Fan's maximum principle in an eigenbasis: weights `q j ∈ [0, 1]` with
  total at most `k` cannot collect more than the `k` largest non-negative eigenvalues.
* `proj_add_orth` : Theorem T11, the sum of two orthogonal projections that annihilate each other is a projection
  (forced part plus free part).
* `circulant_colsum` : Corollary T11.2, every column of a circulant matrix has the same sum, so the transpose of a
  circulant edge sends the all-ones vector to a multiple of itself.
* `prefix_span_mono` : Theorem T11 part 4, the units of a smaller width are a prefix of one stored ordered list, so
  their span lies in the span of every longer prefix.
* `visible_step` : Theorem T13, if the reads `B'` of the layer above, composed with every spatial edge `w q`, agree
  with the reads `w' q B` of the layer below on a subspace `R` (the reachable space) that holds every input `v q`,
  then the folded layer computes `B'` of the full layer.
-/

open Finset BigOperators Matrix

/-- Lemma T11a (one step). If the build's effective read kernel `K' W'ᵀ S'` equals the network's `K Wᵀ S`, then on
every gated input `G a` the build's read of its own layer equals the network's read of its layer. -/
theorem faithful_read {d n n' k k' r : ℕ} (K : Matrix (Fin d) (Fin n) ℝ) (W : Matrix (Fin n') (Fin n) ℝ)
    (S : Matrix (Fin n') (Fin r) ℝ) (K' : Matrix (Fin d) (Fin k) ℝ) (W' : Matrix (Fin k') (Fin k) ℝ)
    (S' : Matrix (Fin k') (Fin r) ℝ) (h : K' * W'ᵀ * S' = K * Wᵀ * S) (G : ℝ) (a : Fin d → ℝ) :
    S'ᵀ *ᵥ (G • ((W' * K'ᵀ) *ᵥ a)) = Sᵀ *ᵥ (G • ((W * Kᵀ) *ᵥ a)) := by
  have e1 : S'ᵀ * (W' * K'ᵀ) = (K' * W'ᵀ * S')ᵀ := by
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose]
  have e2 : Sᵀ * (W * Kᵀ) = (K * Wᵀ * S)ᵀ := by
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose]
  rw [Matrix.mulVec_smul, Matrix.mulVec_smul, Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, e1, e2, h]

/-- Theorems T11 and T12 (Ky Fan in an eigenbasis). Let `S` hold `k` indices whose eigenvalues are at least a
threshold `a ≥ 0`, with every other eigenvalue at most `a`. If `q j ∈ [0, 1]` and `∑ q ≤ k` (the diagonal of a
projection of rank at most `k` in that basis), then `∑ μ q ≤ ∑_{j ∈ S} μ j`. -/
theorem kyfan_coord {n k : ℕ} (μ q : Fin n → ℝ) (S : Finset (Fin n)) (hS : S.card = k) (a : ℝ) (ha : 0 ≤ a)
    (hin : ∀ j ∈ S, a ≤ μ j) (hout : ∀ j ∉ S, μ j ≤ a)
    (hq0 : ∀ j, 0 ≤ q j) (hq1 : ∀ j, q j ≤ 1) (hsum : ∑ j, q j ≤ k) :
    ∑ j, μ j * q j ≤ ∑ j ∈ S, μ j := by
  have hsplit : ∑ j ∈ S, q j + ∑ j ∈ Sᶜ, q j ≤ k := by rw [Finset.sum_add_sum_compl]; exact hsum
  have hmu : ∑ j, μ j * q j = ∑ j ∈ S, μ j * q j + ∑ j ∈ Sᶜ, μ j * q j := (Finset.sum_add_sum_compl S _).symm
  have h1 : a * ∑ j ∈ S, (1 - q j) ≤ ∑ j ∈ S, μ j * (1 - q j) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun j hj => mul_le_mul_of_nonneg_right (hin j hj) (by linarith [hq1 j]))
  have h2 : ∑ j ∈ Sᶜ, μ j * q j ≤ a * ∑ j ∈ Sᶜ, q j := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun j hj => mul_le_mul_of_nonneg_right
      (hout j (Finset.mem_compl.mp hj)) (hq0 j))
  have h3 : ∑ j ∈ S, (1 - q j) = k - ∑ j ∈ S, q j := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, hS, nsmul_eq_mul, mul_one]
  have h4 : ∑ j ∈ S, μ j * (1 - q j) = ∑ j ∈ S, μ j - ∑ j ∈ S, μ j * q j := by
    rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl (fun j _ => by ring)
  have h5 : a * ∑ j ∈ Sᶜ, q j ≤ a * (k - ∑ j ∈ S, q j) := mul_le_mul_of_nonneg_left (by linarith) ha
  rw [h3] at h1
  rw [hmu]
  linarith

/-- Theorem T11 (forced part plus free part). Two projections that annihilate each other add to a projection. -/
theorem proj_add_orth {n : ℕ} (P Q : Matrix (Fin n) (Fin n) ℝ) (hP : P * P = P) (hQ : Q * Q = Q)
    (hPQ : P * Q = 0) (hQP : Q * P = 0) : (P + Q) * (P + Q) = P + Q := by
  rw [Matrix.add_mul, Matrix.mul_add, Matrix.mul_add, hP, hQ, hPQ, hQP, add_zero, zero_add]

/-- Corollary T11.2. In a circulant matrix `w i j = c (i - j)`, every column `j` has the same sum. -/
theorem circulant_colsum {N : ℕ} [NeZero N] (c : ZMod N → ℝ) (j : ZMod N) :
    ∑ i, c (i - j) = ∑ i, c i :=
  Equiv.sum_comp (Equiv.subRight j) c

/-- Theorem T11 part 4 (instant switching). The units of width `k` are the first `k` entries of one stored ordered
list `b`. For `k ≤ k'` their span lies in the span of the first `k'` entries. -/
theorem prefix_span_mono {V : Type*} [AddCommGroup V] [Module ℝ V] {n : ℕ} (b : Fin n → V) {k k' : ℕ}
    (h : k ≤ k') :
    Submodule.span ℝ (b '' {j | (j : ℕ) < k}) ≤ Submodule.span ℝ (b '' {j | (j : ℕ) < k'}) :=
  Submodule.span_mono (Set.image_mono (fun j (hj : (j : ℕ) < k) => lt_of_lt_of_le hj h))

/-- Theorem T13 (one layer of the induction). A spatial edge sums `w q` applied to the layer below at the offsets
`q ∈ s`, then a scalar gate `g`. If `B' (w q x) = w' q (B x)` for every offset and every `x` in a subspace `R`, and
every input `v q` lies in `R`, the folded edge `w'` applied to the folded layer `B v` gives `B'` applied to the full
layer above. -/
theorem visible_step {ι : Type*} {n n' a a' : ℕ} (s : Finset ι) (R : Submodule ℝ (Fin n → ℝ))
    (B : Matrix (Fin a) (Fin n) ℝ) (B' : Matrix (Fin a') (Fin n') ℝ) (w : ι → Matrix (Fin n') (Fin n) ℝ)
    (w' : ι → Matrix (Fin a') (Fin a) ℝ) (h : ∀ q ∈ s, ∀ x ∈ R, B' *ᵥ (w q *ᵥ x) = w' q *ᵥ (B *ᵥ x))
    (g : ℝ) (v : ι → Fin n → ℝ) (hv : ∀ q ∈ s, v q ∈ R) :
    B' *ᵥ (g • ∑ q ∈ s, w q *ᵥ v q) = g • ∑ q ∈ s, w' q *ᵥ (B *ᵥ v q) := by
  rw [Matrix.mulVec_smul, Matrix.mulVec_sum]
  congr 1
  exact Finset.sum_congr rfl (fun q hq => h q hq (v q) (hv q hq))

/-- Theorem T14 (the rank-1 step). For a unit vector `x` (`∑ x² = 1`) and any scale `λ`, the squared error of
`λ x` against `t` is at least `∑ t² - (∑ t x)²`, with equality at `λ = ∑ t x`. So the best rank-1 fit along a unit
direction leaves `|t|² - ⟨t, x⟩²`, and the best rank-1 approximation of a tensor leaves `|T|² - ‖T‖σ²`. -/
theorem rank1_fit {n : ℕ} (t x : Fin n → ℝ) (hx : ∑ i, x i ^ 2 = 1) (lam : ℝ) :
    ∑ i, t i ^ 2 - (∑ i, t i * x i) ^ 2 ≤ ∑ i, (t i - lam * x i) ^ 2 := by
  have e : ∑ i, (t i - lam * x i) ^ 2 =
      ∑ i, t i ^ 2 - 2 * lam * ∑ i, t i * x i + lam ^ 2 * ∑ i, x i ^ 2 := by
    have : ∀ i, (t i - lam * x i) ^ 2 = t i ^ 2 - 2 * lam * (t i * x i) + lam ^ 2 * x i ^ 2 := by
      intro i; ring
    simp_rw [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [e, hx]
  nlinarith [sq_nonneg (lam - ∑ i, t i * x i)]

/-- Theorem T14 (the rank-1 step, equality case). At `λ = ∑ t x` the bound of `rank1_fit` is attained. -/
theorem rank1_fit_eq {n : ℕ} (t x : Fin n → ℝ) (hx : ∑ i, x i ^ 2 = 1) :
    ∑ i, (t i - (∑ j, t j * x j) * x i) ^ 2 = ∑ i, t i ^ 2 - (∑ i, t i * x i) ^ 2 := by
  set s := ∑ j, t j * x j with hs
  have : ∀ i, (t i - s * x i) ^ 2 = t i ^ 2 - 2 * s * (t i * x i) + s ^ 2 * x i ^ 2 := by intro i; ring
  simp_rw [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← hs, hx]
  ring

/-- Theorem T14 (the head split). If `Hᵀ H = 2 I` and `Hᵀ e = 0`, then `|H x + e|² = 2 |x|² + |e|²`. Applied column by
column to `H X_q + ω_q η⊥ κᵀ`, it gives the value of a width-(1, 1) build. -/
theorem orth_head_split {m n : ℕ} (H : Matrix (Fin m) (Fin n) ℝ) (hH : Hᵀ * H = (2 : ℝ) • (1 : Matrix (Fin n) (Fin n) ℝ))
    (x : Fin n → ℝ) (e : Fin m → ℝ) (he : Hᵀ *ᵥ e = 0) :
    (H *ᵥ x + e) ⬝ᵥ (H *ᵥ x + e) = 2 * (x ⬝ᵥ x) + e ⬝ᵥ e := by
  have h1 : (H *ᵥ x) ⬝ᵥ (H *ᵥ x) = 2 * (x ⬝ᵥ x) := by
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec, hH, Matrix.smul_mulVec,
      Matrix.one_mulVec, smul_dotProduct, smul_eq_mul]
  have h2 : (H *ᵥ x) ⬝ᵥ e = 0 := by
    rw [dotProduct_comm, Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, he, zero_dotProduct]
  have h3 : e ⬝ᵥ (H *ᵥ x) = 0 := by rw [dotProduct_comm]; exact h2
  rw [add_dotProduct, dotProduct_add, dotProduct_add, h1, h2, h3]
  ring

/-- Theorem T16 (effective reads). A gate that reads its layer through `S'`, where the layer is `G` times the
build's map `W' K'ᵀ` of the input `a`, reads `G` times the input through the single effective kernel `K' W'ᵀ S'`. -/
theorem effective_read {d k k' r : ℕ} (K' : Matrix (Fin d) (Fin k) ℝ) (W' : Matrix (Fin k') (Fin k) ℝ)
    (S' : Matrix (Fin k') (Fin r) ℝ) (G : ℝ) (a : Fin d → ℝ) :
    S'ᵀ *ᵥ (G • ((W' * K'ᵀ) *ᵥ a)) = G • ((K' * W'ᵀ * S')ᵀ *ᵥ a) := by
  have e : S'ᵀ * (W' * K'ᵀ) = (K' * W'ᵀ * S')ᵀ := by
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose]
  rw [Matrix.mulVec_smul, Matrix.mulVec_mulVec, e]
