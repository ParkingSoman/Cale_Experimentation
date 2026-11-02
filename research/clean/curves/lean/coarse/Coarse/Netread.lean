import Mathlib

/-!
# COARSE-THEORY.md Part B.L: the broken cases read as networks

ReLU is written `max t 0`.

* `psd_parts_zero` : Theorem TL1. If a sum of non-negative terms is 0, every term is 0 (a vector in the kernel of a
  sum of positive semidefinite summaries is in the kernel of each one).
* `weighted_sq_zero` : Theorem TL1. A weighted sum of squares with positive weights is 0 only when every square is
  0 (a direction in the kernel of a node's summary is read as 0 at every point of its area).
* `relu_push_down` : Theorem TL2. A unit that reads units which are on can be moved one layer down: the ReLUs in
  between act as the identity.
* `lift_compensate` : Theorem TL2. Two functions that agree on an area and are constant off it differ by a constant
  times the indicator of the complement.
-/

/-- Theorem TL1: non-negative terms with sum 0 are all 0. -/
theorem psd_parts_zero {ι : Type*} (s : Finset ι) (f : ι → ℝ) (h0 : ∀ i ∈ s, 0 ≤ f i)
    (h : ∑ i ∈ s, f i = 0) : ∀ i ∈ s, f i = 0 :=
  (Finset.sum_eq_zero_iff_of_nonneg h0).mp h

/-- Theorem TL1: with positive weights, a weighted sum of squares is 0 only when every value is 0. -/
theorem weighted_sq_zero {ι : Type*} (s : Finset ι) (p r : ι → ℝ) (hp : ∀ i ∈ s, 0 < p i)
    (h : ∑ i ∈ s, p i * r i ^ 2 = 0) : ∀ i ∈ s, r i = 0 := by
  intro i hi
  have h0 : ∀ j ∈ s, 0 ≤ p j * r j ^ 2 := fun j hj => mul_nonneg (hp j hj).le (sq_nonneg _)
  have := (Finset.sum_eq_zero_iff_of_nonneg h0).mp h i hi
  have hpi : p i ≠ 0 := (hp i hi).ne'
  have : r i ^ 2 = 0 := by
    rcases mul_eq_zero.mp this with h1 | h1
    · exact absurd h1 hpi
    · exact h1
  exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this

/-- Theorem TL2: when every unit read is on (pre-activation `z i ≥ 0`), the ReLUs between act as the identity, so
the reading unit equals a ReLU of an affine function of the pre-activations one layer down. -/
theorem relu_push_down {ι : Type*} (s : Finset ι) (m z : ι → ℝ) (beta : ℝ) (hz : ∀ i ∈ s, 0 ≤ z i) :
    max (∑ i ∈ s, m i * max (z i) 0 + beta) 0 = max (∑ i ∈ s, m i * z i + beta) 0 := by
  have : ∑ i ∈ s, m i * max (z i) 0 = ∑ i ∈ s, m i * z i := by
    refine Finset.sum_congr rfl ?_
    intro i hi
    rw [max_eq_left (hz i hi)]
  rw [this]

/-- Theorem TL2: if `f = g` on an area and `f = c0`, `g = c1` off it, and `ind` is 1 on the area and 0 off it, then
`f = g + (c0 - c1) (1 - ind)` everywhere. -/
theorem lift_compensate {X : Type*} (A : Set X) (f g ind : X → ℝ) (c0 c1 : ℝ)
    (hin : ∀ x ∈ A, f x = g x) (hf : ∀ x ∉ A, f x = c0) (hg : ∀ x ∉ A, g x = c1)
    (hi1 : ∀ x ∈ A, ind x = 1) (hi0 : ∀ x ∉ A, ind x = 0) :
    ∀ x, f x = g x + (c0 - c1) * (1 - ind x) := by
  intro x
  by_cases hx : x ∈ A
  · rw [hin x hx, hi1 x hx]; ring
  · rw [hf x hx, hg x hx, hi0 x hx]; ring

/-! Part B.L.2 (deep layers).

* `gate_indicator` : Theorem TL3. A gate whose pre-activation is `kappa` on its area and `zeta` off it gives the
  area's indicator as an affine function of that pre-activation.
* `relu_sheltered` : Theorem TL3. A unit that is on inside its area and has the constant pre-activation `zeta <= 0`
  off it equals its pre-activation minus `zeta` times the indicator of the complement.
* `within_between` : Proposition TL4. A weighted sum of squares about any point splits into the part about the
  group mean and the group mass times the squared distance of the mean.
-/

/-- Theorem TL3: the gate's pre-activation gives the indicator of its area. -/
theorem gate_indicator {X : Type*} (A : Set X) (pre ind : X → ℝ) (kappa zeta : ℝ) (hne : kappa ≠ zeta)
    (hin : ∀ x ∈ A, pre x = kappa) (hout : ∀ x ∉ A, pre x = zeta)
    (hi1 : ∀ x ∈ A, ind x = 1) (hi0 : ∀ x ∉ A, ind x = 0) :
    ∀ x, ind x = (pre x - zeta) / (kappa - zeta) := by
  intro x
  have hd : kappa - zeta ≠ 0 := sub_ne_zero.mpr hne
  by_cases hx : x ∈ A
  · rw [hi1 x hx, hin x hx, div_self hd]
  · rw [hi0 x hx, hout x hx, sub_self, zero_div]

/-- Theorem TL3: a sheltered unit that is on inside its area is affine in its pre-activation and the indicator. -/
theorem relu_sheltered {X : Type*} (A : Set X) (z ind : X → ℝ) (zeta : ℝ) (hz : zeta ≤ 0)
    (hin : ∀ x ∈ A, 0 ≤ z x) (hout : ∀ x ∉ A, z x = zeta)
    (hi1 : ∀ x ∈ A, ind x = 1) (hi0 : ∀ x ∉ A, ind x = 0) :
    ∀ x, max (z x) 0 = z x - zeta * (1 - ind x) := by
  intro x
  by_cases hx : x ∈ A
  · rw [max_eq_left (hin x hx), hi1 x hx]; ring
  · rw [hout x hx, max_eq_right hz, hi0 x hx]; ring

/-- Proposition TL4: within plus between, for one group. -/
theorem within_between {ι : Type*} (s : Finset ι) (p a : ι → ℝ) (c : ℝ) (hP : ∑ i ∈ s, p i ≠ 0) :
    ∑ i ∈ s, p i * (a i - c) ^ 2 =
      ∑ i ∈ s, p i * (a i - (∑ j ∈ s, p j * a j) / (∑ j ∈ s, p j)) ^ 2 +
        (∑ i ∈ s, p i) * ((∑ j ∈ s, p j * a j) / (∑ j ∈ s, p j) - c) ^ 2 := by
  set P := ∑ j ∈ s, p j with hPdef
  set m := (∑ j ∈ s, p j * a j) / P with hm
  have hsplit : ∀ i ∈ s, p i * (a i - c) ^ 2 =
      p i * (a i - m) ^ 2 + 2 * (m - c) * (p i * a i - p i * m) + p i * (m - c) ^ 2 := by
    intro i _; ring
  rw [Finset.sum_congr rfl hsplit, Finset.sum_add_distrib, Finset.sum_add_distrib]
  have hmP : m * P = ∑ j ∈ s, p j * a j := by
    rw [hm]; field_simp
  have hzero : ∑ i ∈ s, 2 * (m - c) * (p i * a i - p i * m) = 0 := by
    rw [← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.sum_mul, mul_comm (∑ i ∈ s, p i) m, hmP, sub_self,
      mul_zero]
  rw [hzero, add_zero, ← Finset.sum_mul]

/-- Theorem TL6: the error of the affine indicator of a gate, split into the gate's own defect and the error of its
pre-activation. -/
theorem gate_err_identity (kappa zeta ind z zh : ℝ) (hne : kappa - zeta ≠ 0) :
    (kappa - zeta) * (ind - (zh - zeta) / (kappa - zeta)) = ((kappa - zeta) * ind - z + zeta) + (z - zh) := by
  field_simp
  ring

/-- Theorem TL6: the error of a unit's affine stand-in, split into its own defect, the error of its pre-activation and
the error of its area's indicator. -/
theorem unit_err_identity (h z zh zeta ind indh : ℝ) :
    h - (zh - zeta * (1 - indh)) = (h - z + zeta * (1 - ind)) + (z - zh) + zeta * (ind - indh) := by
  ring
