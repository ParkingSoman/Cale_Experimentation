/-
Section B.Q of COARSE-THEORY.md: the packing theorem (Theorem TQ1).
The one-point step of its proof: moving the logits of a softmax that equals s by d costs at most
sum_i s_i (exp d_i - 1 - d_i) in KL(s || .), because log z <= z - 1.
Here KL(s || softmax(log s + d)) = log (sum_i s_i exp d_i) - sum_i s_i d_i.
-/
import Mathlib

open Finset

theorem pack_step {I : Type*} [Fintype I] (s d : I → ℝ) (hs : ∀ i, 0 ≤ s i) (h1 : ∑ i, s i = 1) :
    Real.log (∑ i, s i * Real.exp (d i)) - ∑ i, s i * d i
      ≤ ∑ i, s i * (Real.exp (d i) - 1 - d i) := by
  have hpos : 0 < ∑ i, s i * Real.exp (d i) := by
    obtain ⟨i, hi⟩ : ∃ i, 0 < s i := by
      by_contra h
      push Not at h
      have : ∑ i, s i ≤ 0 := Finset.sum_nonpos (fun i _ => h i)
      linarith
    exact Finset.sum_pos' (fun j _ => mul_nonneg (hs j) (Real.exp_pos _).le)
      ⟨i, Finset.mem_univ _, mul_pos hi (Real.exp_pos _)⟩
  have hlog := Real.log_le_sub_one_of_pos hpos
  have hsplit : ∑ i, s i * (Real.exp (d i) - 1 - d i)
      = ∑ i, s i * Real.exp (d i) - ∑ i, s i - ∑ i, s i * d i := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    congr 1
    ext i
    ring
  rw [hsplit, h1]
  linarith

/-- The terms of classes where d = 0 vanish: only the classes outside the active set count. -/
theorem pack_term_zero (s d : ℝ) (hd : d = 0) : s * (Real.exp d - 1 - d) = 0 := by
  subst hd; simp

/-- phi(t) = exp t - 1 - t is >= 0 (so every term of the bound is >= 0). -/
theorem phi_nonneg (t : ℝ) : 0 ≤ Real.exp t - 1 - t := by
  have := Real.add_one_le_exp t
  linarith

/-- For t <= 0, phi(t) <= -t: the closed form eps_act <= 2 Delta_max mu of Definition DQ1. -/
theorem phi_le_neg (t : ℝ) (ht : t ≤ 0) : Real.exp t - 1 - t ≤ -t := by
  have : Real.exp t ≤ 1 := by
    calc Real.exp t ≤ Real.exp 0 := Real.exp_le_exp.mpr ht
      _ = 1 := Real.exp_zero
  linarith

/-- phi(t) = exp t - 1 - t vanishes only at t = 0 (Proposition TQ2 part 1). -/
theorem phi_eq_zero {t : ℝ} (h : Real.exp t - 1 - t = 0) : t = 0 := by
  by_contra ht
  have := Real.add_one_lt_exp ht
  linarith

/-- The step of Theorem TQ3 part 2, with a reference distribution q different from s:
log (sum_i q_i exp d_i) - sum_i s_i d_i <= sum_i (q_i exp d_i - q_i - s_i d_i), because log z <= z - 1. -/
theorem pack_step2 {I : Type*} [Fintype I] (s q d : I → ℝ) (hq0 : ∀ i, 0 ≤ q i) (hq : ∑ i, q i = 1) :
    Real.log (∑ i, q i * Real.exp (d i)) - ∑ i, s i * d i
      ≤ ∑ i, (q i * Real.exp (d i) - q i - s i * d i) := by
  have hpos : 0 < ∑ i, q i * Real.exp (d i) := by
    obtain ⟨i, hi⟩ : ∃ i, 0 < q i := by
      by_contra h
      push Not at h
      have : ∑ i, q i ≤ 0 := Finset.sum_nonpos (fun i _ => h i)
      linarith
    exact Finset.sum_pos' (fun j _ => mul_nonneg (hq0 j) (Real.exp_pos _).le)
      ⟨i, Finset.mem_univ _, mul_pos hi (Real.exp_pos _)⟩
  have hlog := Real.log_le_sub_one_of_pos hpos
  have hsplit : ∑ i, (q i * Real.exp (d i) - q i - s i * d i)
      = ∑ i, q i * Real.exp (d i) - ∑ i, q i - ∑ i, s i * d i := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
  rw [hsplit, hq]
  linarith

/-- The lower step of Theorem TQ5: for w > -1, log (1 + w) >= w / (1 + w), so
log (1 + w) - w >= - w^2 / (1 + w). -/
theorem log_one_add_ge {w : ℝ} (hw : -1 < w) : w / (1 + w) ≤ Real.log (1 + w) := by
  have hpos : 0 < 1 + w := by linarith
  have h := Real.one_sub_inv_le_log_of_pos hpos
  have : w / (1 + w) = 1 - (1 + w)⁻¹ := by field_simp; ring
  rw [this]; exact h

/-- The counting step of Lemma TQ6: (sum of k numbers)^2 <= k * (sum of their squares). -/
theorem sq_sum_le_card_fin (k : ℕ) (f : Fin k → ℝ) :
    (∑ i, f i) ^ 2 ≤ (k : ℝ) * ∑ i, f i ^ 2 := by
  have := sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset (Fin k))) (f := f)
  simpa using this

/-!
Lemma TQ12 of COARSE-THEORY.md (teacher-student identity). `A` and `A'` stand for log-sum-exp of the logits `l`, `l'`.
With `s` the softmax of `l`: cross entropy at label `y` is `A - l y`, and KL(s || softmax l') is `A' - A - ∑ s (l' - l)`.
-/
theorem ts_point {I : Type*} [Fintype I] [DecidableEq I] (l l' s : I → ℝ) (A A' : ℝ) (y : I) :
    (A' - l' y) - (A - l y)
      = (A' - A - ∑ i, s i * (l' i - l i))
        - ∑ i, ((if i = y then 1 else 0) - s i) * (l' i - l i) := by
  have h1 : ∑ i, ((if i = y then (1 : ℝ) else 0) - s i) * (l' i - l i)
      = (l' y - l y) - ∑ i, s i * (l' i - l i) := by
    have : ∀ i, ((if i = y then (1 : ℝ) else 0) - s i) * (l' i - l i)
        = (if i = y then (l' i - l i) else 0) - s i * (l' i - l i) := by
      intro i
      split_ifs <;> ring
    simp only [this, Finset.sum_sub_distrib, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [h1]
  ring

theorem ts_identity {J K : Type*} [Fintype J] [Fintype K] (W W' G M : J → ℝ) (b b' g : K → ℝ)
    (lam CE CE' D : ℝ)
    (hCE : CE' - CE = D - ∑ j, M j * (W' j - W j) - ∑ k, g k * (b' k - b k))
    (hM : ∀ j, M j = lam * W j + G j) :
    (CE' + lam / 2 * ∑ j, W' j ^ 2) - (CE + lam / 2 * ∑ j, W j ^ 2)
      = D + lam / 2 * ∑ j, (W' j - W j) ^ 2
        - ∑ j, G j * (W' j - W j) - ∑ k, g k * (b' k - b k) := by
  have h1 : ∑ j, M j * (W' j - W j)
      = lam * (∑ j, W j * W' j - ∑ j, W j ^ 2) + ∑ j, G j * (W' j - W j) := by
    rw [← Finset.sum_sub_distrib, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [hM j]
    ring
  have h2 : ∑ j, (W' j - W j) ^ 2
      = ∑ j, W' j ^ 2 - 2 * ∑ j, W j * W' j + ∑ j, W j ^ 2 := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    ring
  rw [h2]
  rw [h1] at hCE
  linear_combination hCE

/-- Lemma TQ12 part 5: the sandwich, from part 2 at the two leasts and R(W_R) <= R(W_D). -/
theorem ts_sandwich (RR RD R0 DR DD lam a b x y : ℝ)
    (hR : RR - R0 = DR + lam / 2 * a - x) (hD : RD - R0 = DD + lam / 2 * b - y) (hle : RR ≤ RD) :
    DR - DD ≤ lam / 2 * (b - a) + (x - y) := by
  nlinarith

/-- Theorem TQ17, the row identity. Row c of a projector Pi with Pi 1 = 0: diagonal d, partner entry p, tails x i (K - 2 of them).
Facts: the row sums to 0, and its squares sum to d. Then the squared distance of the row from the pole projector's row
(entries beta, beta, and -1/K, with beta = 1/2 - 1/K) equals beta - p. -/
theorem pole_row_identity {I : Type*} [Fintype I] (K d p : ℝ) (x : I → ℝ) (hK : K ≠ 0)
    (hn : (Fintype.card I : ℝ) = K - 2)
    (h1 : d + p + ∑ i, x i = 0) (h2 : d ^ 2 + p ^ 2 + ∑ i, x i ^ 2 = d) :
    (d - (1 / 2 - 1 / K)) ^ 2 + (p - (1 / 2 - 1 / K)) ^ 2 + ∑ i, (x i + 1 / K) ^ 2
      = (1 / 2 - 1 / K) - p := by
  have h3 : ∑ i, (x i + 1 / K) ^ 2
      = ∑ i, x i ^ 2 + 2 * (1 / K) * ∑ i, x i + (Fintype.card I : ℝ) * (1 / K) ^ 2 := by
    have : ∀ i, (x i + 1 / K) ^ 2 = x i ^ 2 + 2 * (1 / K) * x i + (1 / K) ^ 2 := fun i => by ring
    simp only [this, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      ← Finset.mul_sum]
  have hB : ∑ i, x i = -d - p := by linarith
  have hA : ∑ i, x i ^ 2 = d - d ^ 2 - p ^ 2 := by linarith
  rw [h3, hn, hB, hA]
  field_simp
  ring

/-- Theorem TQ17, the row inequality: the row's net gain is g(T), once its squares are replaced through the identity. -/
theorem pole_row_net (mt gam T pen sq : ℝ) (hgam : 0 ≤ gam) (hsq : sq = T - T ^ 2) (hpen : pen ≤ gam * sq) (R : ℝ) :
    (mt - gam) * T + gam * T ^ 2 - R ≤ mt * T - (pen + R) := by
  subst hsq
  nlinarith

/-- Lemma TQ23.a, Step 4 of COARSE-THEORY.md: two points. -/
theorem tq23_two_points (p q : ℝ) : (p - q) ^ 2 / 2 ≤ p ^ 2 + q ^ 2 := by
  nlinarith [sq_nonneg (p + q)]

/-- Lemma TQ23.a, Step 3: the split that lets the shared class pin the gauge. -/
theorem tq23_gauge (f y θ : ℝ) (h0 : 0 < θ) :
    (1 - θ) * f ^ 2 - (1 / θ - 1) * y ^ 2 ≤ (f - y) ^ 2 := by
  have hθ : θ ≠ 0 := h0.ne'
  have key : (f - y) ^ 2 - ((1 - θ) * f ^ 2 - (1 / θ - 1) * y ^ 2) = (θ * f - y) ^ 2 / θ := by
    field_simp
    ring
  have : 0 ≤ (θ * f - y) ^ 2 / θ := div_nonneg (sq_nonneg _) h0.le
  linarith

/-- Lemma TQ27, the arithmetic step: the minority weight squared is at most the product of the two weights. -/
theorem tq27_clean (a b r : ℝ) (hr : 0 ≤ r) (ha : r ≤ a) (hb : r ≤ b) : r ^ 2 ≤ a * b := by
  nlinarith [mul_le_mul ha hb hr (le_trans hr ha)]

/-- Proposition TQ30 part 1, the algebraic step: two opposite entries of a rank-one matrix bound the product of its diagonal values. -/
theorem tq30_two_two (uw vj uj vw : ℝ) :
    2 * |(uw * vw) * (uj * vj)| ≤ (uw * vj) ^ 2 + (uj * vw) ^ 2 := by
  have h : (uw * vw) * (uj * vj) = (uw * vj) * (uj * vw) := by ring
  rw [h]
  have := two_mul_le_add_sq (|uw * vj|) (|uj * vw|)
  have h1 : |uw * vj| ^ 2 = (uw * vj) ^ 2 := sq_abs _
  have h2 : |uj * vw| ^ 2 = (uj * vw) ^ 2 := sq_abs _
  rw [abs_mul (uw * vj) (uj * vw)]
  nlinarith
