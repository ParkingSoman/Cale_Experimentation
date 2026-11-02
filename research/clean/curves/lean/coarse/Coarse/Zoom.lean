import Mathlib

/-!
# COARSE-THEORY.md Part B.W: the developer's zoom (Theorem T58)

A group of units whose pre-activations are all `≥ 0` or all `≤ 0` at an input (an AND group, Definition D51) can be
replaced by units that read a non-negative mix of the members' pre-activations, with no change at that input.
ReLU is written `max t 0`.

* `and_collapse` : for `v ≥ 0`, `∑ v i * relu (z i) = relu (∑ v i * z i)` when the `z i` share one weak sign.
* `and_collapse_split` : a row with both signs splits into its positive and negative parts, two units.
-/

open Finset

/-- Theorem T58 part 1 (one row): a non-negative mix of the members' outputs is one ReLU of the same mix of their
pre-activations, at every input where the members are on together or off together. -/
theorem and_collapse {ι : Type*} (s : Finset ι) (v z : ι → ℝ) (hv : ∀ i ∈ s, 0 ≤ v i)
    (hz : (∀ i ∈ s, 0 ≤ z i) ∨ (∀ i ∈ s, z i ≤ 0)) :
    ∑ i ∈ s, v i * max (z i) 0 = max (∑ i ∈ s, v i * z i) 0 := by
  rcases hz with hpos | hneg
  · have h1 : ∑ i ∈ s, v i * max (z i) 0 = ∑ i ∈ s, v i * z i :=
      sum_congr rfl (fun i hi => by rw [max_eq_left (hpos i hi)])
    have h2 : 0 ≤ ∑ i ∈ s, v i * z i := sum_nonneg (fun i hi => mul_nonneg (hv i hi) (hpos i hi))
    rw [h1, max_eq_left h2]
  · have h1 : ∑ i ∈ s, v i * max (z i) 0 = 0 :=
      sum_eq_zero (fun i hi => by rw [max_eq_right (hneg i hi), mul_zero])
    have h2 : ∑ i ∈ s, v i * z i ≤ 0 :=
      sum_nonpos (fun i hi => mul_nonpos_of_nonneg_of_nonpos (hv i hi) (hneg i hi))
    rw [h1, max_eq_right h2]

/-- Theorem T58 part 2 (two logits): any row `a` is the difference of two non-negative rows, so an AND group read by
one row collapses to two units, `relu (∑ a⁺ z) - relu (∑ a⁻ z)`. -/
theorem and_collapse_split {ι : Type*} (s : Finset ι) (a z : ι → ℝ)
    (hz : (∀ i ∈ s, 0 ≤ z i) ∨ (∀ i ∈ s, z i ≤ 0)) :
    ∑ i ∈ s, a i * max (z i) 0 =
      max (∑ i ∈ s, max (a i) 0 * z i) 0 - max (∑ i ∈ s, max (-a i) 0 * z i) 0 := by
  rw [← and_collapse s (fun i => max (a i) 0) z (fun i _ => le_max_right _ _) hz,
    ← and_collapse s (fun i => max (-a i) 0) z (fun i _ => le_max_right _ _) hz, ← sum_sub_distrib]
  refine sum_congr rfl (fun i _ => ?_)
  have : a i = max (a i) 0 - max (-a i) 0 := by
    rcases le_total 0 (a i) with h | h
    · rw [max_eq_left h, max_eq_right (by linarith)]; ring
    · rw [max_eq_right h, max_eq_left (by linarith)]; ring
  calc a i * max (z i) 0 = (max (a i) 0 - max (-a i) 0) * max (z i) 0 := by rw [← this]
    _ = max (a i) 0 * max (z i) 0 - max (-a i) 0 * max (z i) 0 := by ring

/-- Lemma T62a (a unit that switches): for `p, q ≥ 0`, the ReLU of the midpoint value `(p - q) / 2`
minus the average of the end values `relu p = p` and `relu (-q) = 0` is `-min p q / 2`. -/
theorem dip_switch (p q : ℝ) (hp : 0 ≤ p) (hq : 0 ≤ q) :
    max ((p - q) / 2) 0 - (max p 0 + max (-q) 0) / 2 = -(min p q) / 2 := by
  rw [max_eq_left hp, max_eq_right (by linarith : -q ≤ 0)]
  rcases le_total q p with h | h
  · rw [max_eq_left (by linarith : (0 : ℝ) ≤ (p - q) / 2), min_eq_right h]; ring
  · rw [max_eq_right (by linarith : (p - q) / 2 ≤ 0), min_eq_left h]; ring

/-- Lemma T62a (a unit that does not switch): if both end values share a weak sign, the ReLU of the
midpoint value is the average of the end values. -/
theorem dip_same_sign (u v : ℝ) (h : (0 ≤ u ∧ 0 ≤ v) ∨ (u ≤ 0 ∧ v ≤ 0)) :
    max ((u + v) / 2) 0 = (max u 0 + max v 0) / 2 := by
  rcases h with ⟨hu, hv⟩ | ⟨hu, hv⟩
  · rw [max_eq_left hu, max_eq_left hv, max_eq_left (by linarith)]
  · rw [max_eq_right hu, max_eq_right hv, max_eq_right (by linarith)]; ring

/-- Theorem T63 (one kink): a kink at `s` in `[0, 1]` contributes
`relu (lam - s) - lam * relu (1 - s) = -min (s * (1 - lam)) (lam * (1 - s))` to the `lam`-dip. -/
theorem kink_green (lam s : ℝ) (hl0 : 0 ≤ lam) (hl1 : lam ≤ 1) (hs0 : 0 ≤ s) (hs1 : s ≤ 1) :
    max (lam - s) 0 - lam * max (1 - s) 0 = -(min (s * (1 - lam)) (lam * (1 - s))) := by
  rw [max_eq_left (by linarith : (0 : ℝ) ≤ 1 - s)]
  rcases le_total s lam with h | h
  · rw [max_eq_left (by linarith : (0 : ℝ) ≤ lam - s),
      min_eq_left (by nlinarith : s * (1 - lam) ≤ lam * (1 - s))]
    ring
  · rw [max_eq_right (by linarith : lam - s ≤ 0),
      min_eq_right (by nlinarith : lam * (1 - s) ≤ s * (1 - lam))]
    ring
