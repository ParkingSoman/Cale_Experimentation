import Mathlib

/-!
# COARSE-THEORY.md Part B.T: the trunk itself (finite steps)

* `relu_second_diff` : Theorem T20 part 1, the second difference of a relu kink is a hat function.
* `t20_d2_gap_a`, `t20_d2_gap_b`, `t20_d2_competitor` : Theorem T20 part 2 for D2, the exact least-squares bounds.
* `relu_affine_monotone`, `mono_mid_ge_min` : Theorem T22, the midpoint step (a width-1 gap is monotone along a
  segment, so its value at the midpoint is at least the smaller end value).
* `broken_line_interp` : Lemma T22a, the broken line through given points is a sum of relu units.
-/

open Finset BigOperators

/-- Theorem T20 part 1: `relu(m - 1 - t) - 2 relu(m - t) + relu(m + 1 - t) = max(0, 1 - |m - t|)`. -/
theorem relu_second_diff (m t : ℝ) :
    max (m - 1 - t) 0 - 2 * max (m - t) 0 + max (m + 1 - t) 0 = max 0 (1 - |m - t|) := by
  simp only [max_def, abs_eq_max_neg]
  split_ifs <;> linarith

/-- Theorem T20 part 2, unit `relu(x - 1)`: on the points `-3, ..., 3` its data vector is `(0,0,0,0,0,1,2)` and the
teacher's gap is `(2,1,0,0,0,1,2)`. Every build `alpha u + gamma` has residual sum of squares at least `85/26`. -/
theorem t20_d2_gap_a (α γ : ℝ) :
    85 / 26 ≤ (γ - 2) ^ 2 + (γ - 1) ^ 2 + γ ^ 2 + γ ^ 2 + γ ^ 2 + (α + γ - 1) ^ 2 + (2 * α + γ - 2) ^ 2 := by
  nlinarith [sq_nonneg (α - 17 / 26 + 3 / 5 * (γ - 15 / 26)), sq_nonneg (γ - 15 / 26)]

/-- Theorem T20 part 2, unit `relu(x + 1)`: data vector `(0,0,0,1,2,3,4)`; residual at least `493/110`. -/
theorem t20_d2_gap_b (α γ : ℝ) :
    493 / 110 ≤ (γ - 2) ^ 2 + (γ - 1) ^ 2 + γ ^ 2 + (α + γ) ^ 2 + (2 * α + γ) ^ 2 + (3 * α + γ - 1) ^ 2
      + (4 * α + γ - 2) ^ 2 := by
  nlinarith [sq_nonneg (α - 17 / 110 + 1 / 3 * (γ - 7 / 11)), sq_nonneg (γ - 7 / 11)]

/-- Theorem T20 part 2: the build with gap `relu(x - 8/5) + 3/5` has residual `16/5`, below both bounds. -/
theorem t20_d2_competitor :
    ((3 : ℝ) / 5 - 2) ^ 2 + (3 / 5 - 1) ^ 2 + (3 / 5) ^ 2 + (3 / 5) ^ 2 + (3 / 5) ^ 2
      + (2 / 5 + 3 / 5 - 1) ^ 2 + (7 / 5 + 3 / 5 - 2) ^ 2 = 16 / 5
    ∧ (16 : ℝ) / 5 < 85 / 26 ∧ (85 : ℝ) / 26 < 493 / 110 := by
  refine ⟨by norm_num, by norm_num, by norm_num⟩

/-- Theorem T22: the gap of a width-1 build, `t ↦ alpha relu(t) + gamma`, is monotone or antitone. -/
theorem relu_affine_monotone (α γ : ℝ) :
    Monotone (fun t : ℝ => α * max t 0 + γ) ∨ Antitone (fun t : ℝ => α * max t 0 + γ) := by
  rcases le_total 0 α with h | h
  · left
    intro a b hab
    have : max a 0 ≤ max b 0 := max_le_max hab le_rfl
    simp only
    nlinarith
  · right
    intro a b hab
    have : max a 0 ≤ max b 0 := max_le_max hab le_rfl
    simp only
    nlinarith

/-- Theorem T22, midpoint step: a monotone or antitone `phi` takes at the midpoint a value at least the smaller of its
values at the ends. With `a = w^T x_u + b` and `b' = w^T x_v + b`, the midpoint of the arguments is the argument at
the midpoint `m_uv`, because the argument is affine. -/
theorem mono_mid_ge_min (φ : ℝ → ℝ) (hφ : Monotone φ ∨ Antitone φ) (a b : ℝ) :
    min (φ a) (φ b) ≤ φ ((a + b) / 2) := by
  rcases le_total a b with hab | hab
  · rcases hφ with h | h
    · exact (min_le_left _ _).trans (h (by linarith))
    · exact (min_le_right _ _).trans (h (by linarith))
  · rcases hφ with h | h
    · exact (min_le_right _ _).trans (h (by linarith))
    · exact (min_le_left _ _).trans (h (by linarith))

/-- Lemma T22a: slope of the broken line on the j-th piece. -/
noncomputable def blSlope (s t : ℕ → ℝ) (j : ℕ) : ℝ := (t (j + 1) - t j) / (s (j + 1) - s j)

/-- Lemma T22a: the slope of the piece before the j-th (zero before the first point). -/
noncomputable def blPrev (s t : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | j + 1 => blSlope s t j

/-- Lemma T22a: the output weight of hidden unit j, `mu_j - mu_{j-1}`. -/
noncomputable def blCoef (s t : ℕ → ℝ) (j : ℕ) : ℝ := blSlope s t j - blPrev s t j

lemma blCoef_sum (s t : ℕ → ℝ) (i : ℕ) : ∑ j ∈ range (i + 1), blCoef s t j = blSlope s t i := by
  induction i with
  | zero => simp [blCoef, blPrev]
  | succ i ih => rw [Finset.sum_range_succ, ih]; simp [blCoef, blPrev]

lemma bl_lin (s t : ℕ → ℝ) (hs : StrictMono s) (i : ℕ) :
    ∑ j ∈ range i, blCoef s t j * (s i - s j) = t i - t 0 := by
  induction i with
  | zero => simp
  | succ i ih =>
    have hsplit : ∀ j, blCoef s t j * (s (i + 1) - s j)
        = blCoef s t j * (s (i + 1) - s i) + blCoef s t j * (s i - s j) := by intro j; ring
    rw [Finset.sum_congr rfl (fun j _ => hsplit j), Finset.sum_add_distrib, ← Finset.sum_mul, blCoef_sum,
      Finset.sum_range_succ, ih]
    have hne : s (i + 1) - s i ≠ 0 := by
      have := hs (Nat.lt_succ_self i); linarith
    simp only [blSlope, sub_self, mul_zero, add_zero]
    field_simp
    ring

/-- Lemma T22a: with points `s 0 < s 1 < ... < s n` and labels `t i`, the sum of relu units
`t 0 + sum_{j < n} (mu_j - mu_{j-1}) relu(x - s j)` takes the value `t i` at `s i`, for every `i <= n`. -/
theorem broken_line_interp (s t : ℕ → ℝ) (hs : StrictMono s) (n i : ℕ) (hi : i ≤ n) :
    t 0 + ∑ j ∈ range n, blCoef s t j * max (s i - s j) 0 = t i := by
  have h1 : ∑ j ∈ range n, blCoef s t j * max (s i - s j) 0
      = ∑ j ∈ range i, blCoef s t j * (s i - s j) := by
    rw [← Finset.sum_range_add_sum_Ico _ hi]
    have hz : ∑ j ∈ Finset.Ico i n, blCoef s t j * max (s i - s j) 0 = 0 := by
      apply Finset.sum_eq_zero
      intro j hj
      rw [Finset.mem_Ico] at hj
      have : s i ≤ s j := hs.monotone hj.1
      rw [max_eq_right (by linarith)]
      ring
    rw [hz, add_zero]
    apply Finset.sum_congr rfl
    intro j hj
    rw [Finset.mem_range] at hj
    rw [max_eq_left (by have := hs hj; linarith)]
  rw [h1, bl_lin s t hs i]
  ring

/-- Theorem T23, finite core: if a relu unit `relu(a x + b)` equals `alpha + beta c1` at `x = -2, ..., 2`, with
`c1 = (2, 1, 0, -2, -4)` there, then `beta = 0`. -/
theorem t23_core (a b α β : ℝ) (h1 : max (a * (-2) + b) 0 = α + 2 * β) (h2 : max (a * (-1) + b) 0 = α + β)
    (h3 : max b 0 = α) (h4 : max (a * 1 + b) 0 = α - 2 * β) (h5 : max (a * 2 + b) 0 = α - 4 * β) : β = 0 := by
  simp only [max_def] at h1 h2 h3 h4 h5
  split_ifs at h1 h2 h3 h4 h5 <;> linarith

/-- Theorem T25 part 1: the equivariance identities for the mirror `diag(-1, 1)`. First layer `W_1` (rows `(1, 0)`
and `(-1, 0)`), the unit swap `pi`, the biases `(-1, -1)` and the readout (rows `(1, 1)` and `(0, 0)`). -/
theorem t25_mirror :
    (!![1, 0; -1, 0] : Matrix (Fin 2) (Fin 2) ℤ) * !![-1, 0; 0, 1] = !![0, 1; 1, 0] * !![1, 0; -1, 0] ∧
    Matrix.mulVec (!![0, 1; 1, 0] : Matrix (Fin 2) (Fin 2) ℤ) ![-1, -1] = ![-1, -1] ∧
    (!![1, 1; 0, 0] : Matrix (Fin 2) (Fin 2) ℤ) * !![0, 1; 1, 0] = !![1, 1; 0, 0] := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  · ext i; fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]

/-- Theorem T29, the direction "a fold covers its group": if every member of the group is silent (`z i ≤ 0`), a
non-negative mix of their pre-activations with a raised threshold is silent too. -/
theorem fold_covers {s : ℕ} (z m : Fin s → ℝ) (τ : ℝ) (hm : ∀ i, 0 ≤ m i) (hτ : 0 ≤ τ)
    (hz : ∀ i, z i ≤ 0) : max (∑ i, m i * z i - τ) 0 = 0 := by
  have h : ∑ i, m i * z i ≤ 0 :=
    Finset.sum_nonpos (fun i _ => by nlinarith [hm i, hz i])
  exact max_eq_right (by linarith)

/-- Lemma T32a, the inequality behind the dual certificates: for non-negative weights `π`,
`2 Σ π v a - Σ π v² ≤ Σ π a²` (expand `Σ π (a - v)² ≥ 0`). -/
theorem cert_ineq {n : ℕ} (π a v : Fin n → ℝ) (hπ : ∀ i, 0 ≤ π i) :
    2 * ∑ i, π i * v i * a i - ∑ i, π i * v i ^ 2 ≤ ∑ i, π i * a i ^ 2 := by
  have h : ∀ i, 2 * (π i * v i * a i) - π i * v i ^ 2 ≤ π i * a i ^ 2 := fun i => by
    nlinarith [mul_nonneg (hπ i) (sq_nonneg (a i - v i))]
  have hs := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => h i)
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum] at hs
  linarith
