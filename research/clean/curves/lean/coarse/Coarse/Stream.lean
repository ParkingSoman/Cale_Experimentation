/-
Section B.V of COARSE-THEORY.md: the stream (Fact F13, Theorem TV1).
The steps: the norm n_eps(h) = h / sqrt(|h|^2 + eps) commutes with any linear isometry; a part of the stream that a
reader does not read, orthogonal to the rest and with energy c, leaves the normed read unchanged when the norm's
constant grows by c; and the temperature of Theorem TV1 part 3 is within |A - A'| / A' of 1.
-/
import Mathlib

namespace Coarse.Stream

/-- Fact F13 part 3 (and Lemma TV0's read step): a linear isometry Q commutes with the norm,
Q (h / sqrt(|h|^2 + eps)) = Q h / sqrt(|Q h|^2 + eps). -/
theorem norm_rot {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    (Q : E →ₗᵢ[ℝ] F) (h : E) (ε : ℝ) :
    Q ((Real.sqrt (‖h‖ ^ 2 + ε))⁻¹ • h) = (Real.sqrt (‖Q h‖ ^ 2 + ε))⁻¹ • Q h := by
  rw [LinearIsometry.norm_map, LinearIsometry.map_smul]

/-- Theorem TV1 part 2, the read step: with h = hk + hd, hd orthogonal to hk, unread by R (R hd = 0) and of energy
c = |hd|^2, the read of the cut stream hk with constant eps + c equals the read of h with constant eps. -/
theorem read_unchanged {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [AddCommGroup F] [Module ℝ F] (R : E →ₗ[ℝ] F) (hk hd : E) (ε c : ℝ)
    (horth : inner ℝ hk hd = 0) (hR : R hd = 0) (hc : ‖hd‖ ^ 2 = c) :
    R ((Real.sqrt (‖hk‖ ^ 2 + (ε + c)))⁻¹ • hk) =
      R ((Real.sqrt (‖hk + hd‖ ^ 2 + ε))⁻¹ • (hk + hd)) := by
  have hp : ‖hk + hd‖ ^ 2 = ‖hk‖ ^ 2 + ‖hd‖ ^ 2 := by
    have := norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero (𝕜 := ℝ) hk hd horth
    nlinarith [this]
  have hs : ‖hk‖ ^ 2 + (ε + c) = ‖hk + hd‖ ^ 2 + ε := by rw [hp, hc]; ring
  rw [hs, map_smul, map_smul, map_add, hR, add_zero]

/-- Theorem TV1 part 3: the temperature t = sqrt(A / A') satisfies |t - 1| <= |A - A'| / A'. -/
theorem temp_le (A A' : ℝ) (hA : 0 ≤ A) (hA' : 0 < A') :
    |Real.sqrt (A / A') - 1| ≤ |A - A'| / A' := by
  set t := Real.sqrt (A / A') with ht
  have t0 : 0 ≤ t := Real.sqrt_nonneg _
  have tsq : t ^ 2 = A / A' := Real.sq_sqrt (div_nonneg hA hA'.le)
  have h1 : |t - 1| ≤ |t - 1| * (t + 1) := by
    have := mul_nonneg (abs_nonneg (t - 1)) t0
    nlinarith
  have h2 : |t - 1| * (t + 1) = |t ^ 2 - 1| := by
    rw [show t ^ 2 - 1 = (t - 1) * (t + 1) by ring, abs_mul,
      abs_of_nonneg (by linarith : (0 : ℝ) ≤ t + 1)]
  have h3 : |t ^ 2 - 1| = |A - A'| / A' := by
    rw [tsq, show A / A' - 1 = (A - A') / A' by field_simp, abs_div, abs_of_pos hA']
  linarith [h1, h2, h3]

/-- Theorem TV6, step (a): nu(z) = sqrt(|z|^2 + eps) is 1-Lipschitz in |z|:
|sqrt(p^2 + eps) - sqrt(q^2 + eps)| <= |p - q| for p, q >= 0 (then use | |a| - |b| | <= |a - b|). -/
theorem nu_lip (p q ε : ℝ) (hp : 0 ≤ p) (hq : 0 ≤ q) (hε : 0 ≤ ε) :
    |Real.sqrt (p ^ 2 + ε) - Real.sqrt (q ^ 2 + ε)| ≤ |p - q| := by
  set A := Real.sqrt (p ^ 2 + ε)
  set B := Real.sqrt (q ^ 2 + ε)
  have hA : 0 ≤ A := Real.sqrt_nonneg _
  have hB : 0 ≤ B := Real.sqrt_nonneg _
  have hA2 : A ^ 2 = p ^ 2 + ε := Real.sq_sqrt (by positivity)
  have hB2 : B ^ 2 = q ^ 2 + ε := Real.sq_sqrt (by positivity)
  have hAB : p * q + ε ≤ A * B := by
    have h1 : A * B = Real.sqrt ((p ^ 2 + ε) * (q ^ 2 + ε)) := by
      rw [Real.sqrt_mul (by positivity)]
    have h2 : p * q + ε = Real.sqrt ((p * q + ε) ^ 2) := by
      rw [Real.sqrt_sq (by positivity)]
    rw [h1, h2]
    apply Real.sqrt_le_sqrt
    nlinarith [sq_nonneg (p - q), hε]
  have hsq : (A - B) ^ 2 ≤ (p - q) ^ 2 := by nlinarith [hA2, hB2, hAB]
  exact sq_le_sq.mp hsq

/-- Theorem TV6, step (a): x |1/B - 1/A| <= t / A when 0 <= x <= B, A, B > 0 and |A - B| <= t
(x = |a'|, B = nu(a'), A = nu(a), t = |a' - a|). -/
theorem scale_step (x A B t : ℝ) (hx : 0 ≤ x) (hA : 0 < A) (hB : 0 < B) (hxB : x ≤ B)
    (ht : |A - B| ≤ t) : x * |1 / B - 1 / A| ≤ t / A := by
  have e : 1 / B - 1 / A = (A - B) / (A * B) := by field_simp
  have key : x * |A - B| ≤ B * t := mul_le_mul hxB ht (abs_nonneg _) hB.le
  rw [e, abs_div, abs_of_pos (mul_pos hA hB), ← mul_div_assoc, div_le_div_iff₀ (mul_pos hA hB) hA]
  nlinarith [key, hA]

/-! Part B.V.3: overlap and cancellation. -/

/-- Fact F14 part 1 and Lemma TV10: a read of the moved stream, (R A⁻¹)(A h) = R h, for a linear equivalence A. -/
theorem gl_read {E F : Type*} [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]
    (A : E ≃ₗ[ℝ] E) (R : E →ₗ[ℝ] F) (h : E) :
    (R.comp A.symm.toLinearMap) (A h) = R h := by
  simp

/-- Fact F14 part 2: with the norm, the read of the moved stream is N's read scaled by one number,
(R A⁻¹)((|A h|^2 + eps)^(-1/2) • A h) = (|A h|^2 + eps)^(-1/2) • R h. -/
theorem norm_scale_read {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [AddCommGroup F] [Module ℝ F] (A : E ≃ₗ[ℝ] E) (R : E →ₗ[ℝ] F) (h : E) (ε : ℝ) :
    (R.comp A.symm.toLinearMap) ((Real.sqrt (‖A h‖ ^ 2 + ε))⁻¹ • A h) =
      (Real.sqrt (‖A h‖ ^ 2 + ε))⁻¹ • R h := by
  simp

/-- Lemma TV11 part 2: if z lies in the dropped part (r' z = 0) and in a kept area (r' z = r z), then r z = 0. -/
theorem crowd_no_read {E : Type*} [AddCommGroup E] [Module ℝ E] (r r' : E →ₗ[ℝ] ℝ) (z : E)
    (hD : r' z = 0) (hE : r' z = r z) : r z = 0 := by
  rw [← hE, hD]

/-- Lemma TV12: a direction holding c^(1/2) adds c to the squared size that the norm reads. -/
theorem battery_const (p c ε : ℝ) (hc : 0 ≤ c) : p + Real.sqrt c ^ 2 + ε = p + (ε + c) := by
  rw [Real.sq_sqrt hc]; ring

/-- Theorem TV9 part 1: the least-squares fit is a nearest point. The orthogonal projection of the target t onto the
span K of the fitted functions is no farther from t than any element of K. -/
theorem ols_le {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (K : Submodule ℝ E)
    [K.HasOrthogonalProjection] (t : E) (k : K) :
    ‖t - K.starProjection t‖ ≤ ‖t - (k : E)‖ := by
  rw [Submodule.starProjection_minimal]
  exact ciInf_le ⟨0, Set.forall_mem_range.mpr fun _ => norm_nonneg _⟩ k

/-- Lemma TV15.1 part 3 (ReLU, a > 0, chosen slope 1): the remainder relu(a + u) - relu(a) - u is at most
max(t - a, 0) in size for |u| <= t. -/
theorem relu_rem_pos (a u t : ℝ) (ha : 0 < a) (hu : |u| ≤ t) :
    |max (a + u) 0 - max a 0 - u| ≤ max (t - a) 0 := by
  rw [max_eq_left ha.le]
  have h1 := neg_abs_le u
  rcases le_total 0 (a + u) with h | h
  · rw [max_eq_left h]; simp [le_max_right]
  · rw [max_eq_right h]
    rw [show (0 : ℝ) - a - u = -(a + u) by ring, abs_neg, abs_of_nonpos h]
    exact le_max_of_le_left (by linarith)

/-- Lemma TV15.1 part 3 (ReLU, a < 0, chosen slope 0): the remainder relu(a + u) - relu(a) is at most
max(t - |a|, 0) in size for |u| <= t. -/
theorem relu_rem_neg (a u t : ℝ) (ha : a < 0) (hu : |u| ≤ t) :
    |max (a + u) 0 - max a 0 - 0 * u| ≤ max (t - |a|) 0 := by
  rw [max_eq_right ha.le, abs_of_neg ha]
  have h1 := le_abs_self u
  rcases le_total 0 (a + u) with h | h
  · rw [max_eq_left h]
    rw [show a + u - 0 - 0 * u = a + u by ring, abs_of_nonneg h]
    exact le_max_of_le_left (by linarith)
  · rw [max_eq_right h]; simp [le_max_right]

/-- Lemma TV15.2 part 2: the identity behind the norm's second-derivative constant. With c = p . v (|v| = 1) and
q = |p|^2, nu^4 |D^2 n[v, v]|^2 = 4 c^2 + (3 c^2 - 1)^2 q - 4 c^2 (3 c^2 - 1) = 4/3 - (4 - 3 q)(3 c^2 - 1)^2 / 3. -/
theorem norm_hess_identity (c q : ℝ) :
    4 * c ^ 2 + (3 * c ^ 2 - 1) ^ 2 * q - 4 * c ^ 2 * (3 * c ^ 2 - 1)
      = 4 / 3 - (4 - 3 * q) * (3 * c ^ 2 - 1) ^ 2 / 3 := by
  ring

/-- Lemma TV15.2 part 2: for q <= 4/3 (in the lemma q = |z|^2 / (|z|^2 + eps) <= 1) the quantity is at most 4/3, so
|D^2 n[v, v]| <= (2 / sqrt 3) |v|^2 / nu^2. -/
theorem norm_hess_le (c q : ℝ) (hq : q ≤ 4 / 3) :
    4 * c ^ 2 + (3 * c ^ 2 - 1) ^ 2 * q - 4 * c ^ 2 * (3 * c ^ 2 - 1) ≤ 4 / 3 := by
  rw [norm_hess_identity]
  have : 0 ≤ (4 - 3 * q) * (3 * c ^ 2 - 1) ^ 2 := mul_nonneg (by linarith) (sq_nonneg _)
  linarith

/-- Lemma TV15.2 part 2, the vector step: for |v| = 1, c = <p, v> and q = |p|^2,
|-2 c v + (3 c^2 - 1) p|^2 = 4 c^2 + (3 c^2 - 1)^2 q - 4 c^2 (3 c^2 - 1). -/
theorem norm_hess_vec {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (p v : E) (hv : ‖v‖ = 1) :
    ‖(-2 * inner ℝ p v) • v + (3 * (inner ℝ p v) ^ 2 - 1) • p‖ ^ 2
      = 4 * (inner ℝ p v) ^ 2 + (3 * (inner ℝ p v) ^ 2 - 1) ^ 2 * ‖p‖ ^ 2
        - 4 * (inner ℝ p v) ^ 2 * (3 * (inner ℝ p v) ^ 2 - 1) := by
  rw [@norm_add_sq_real, norm_smul, norm_smul, inner_smul_left, inner_smul_right, hv, real_inner_comm v p]
  simp only [Real.norm_eq_abs, mul_pow, sq_abs, conj_trivial]
  ring

/-- Theorem TV18 step (a): for matrices with entries >= 0, entrywise A <= A' and 0 <= v <= v' give
A v <= A' v' entry by entry. -/
theorem mulvec_mono {n : Type*} [Fintype n] (A A' : Matrix n n ℝ) (v v' : n → ℝ)
    (hA : ∀ i j, 0 ≤ A i j) (hAA : ∀ i j, A i j ≤ A' i j) (hv : ∀ j, 0 ≤ v j) (hvv : ∀ j, v j ≤ v' j) :
    ∀ i, A.mulVec v i ≤ A'.mulVec v' i := by
  intro i
  simp only [Matrix.mulVec, dotProduct]
  apply Finset.sum_le_sum
  intro j _
  exact mul_le_mul (hAA i j) (hvv j) (hv j) ((hA i j).trans (hAA i j))

/-- Lemma TV19, the step: if a linear map Pi fixes h and h' (both in the kept subspace), then
h + Pi (h' - h) = h'. -/
theorem written_step {E : Type*} [AddCommGroup E] [Module ℝ E] (Pi : E →ₗ[ℝ] E) (h h' : E)
    (hh : Pi h = h) (hh' : Pi h' = h') : h + Pi (h' - h) = h' := by
  rw [map_sub, hh, hh']
  abel

/-- Proposition TV19.1 part 1, the read step: if a read R kills h - w (the dropped part lies in the read kernel),
then R w = R h. -/
theorem quotient_read {E F : Type*} [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]
    (R : E →ₗ[ℝ] F) (h w : E) (hz : R (h - w) = 0) : R w = R h := by
  rw [map_sub, sub_eq_zero] at hz
  exact hz.symm

end Coarse.Stream
