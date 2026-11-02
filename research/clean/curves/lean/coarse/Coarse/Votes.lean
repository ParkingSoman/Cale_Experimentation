import Mathlib

/-!
# COARSE-THEORY.md Part A.5: the votes of a reader (Definition D41, Lemma T45a)

A reader adds the votes `a i = W_{r,u} h_u(x)` of the units it reads, before its own nonlinearity. The cancelled
part of the votes is `∑ |a i| - |∑ a i|`.

* `vote_cancel_bounds` : the cancelled part lies between `0` and the total vote `∑ |a i|`.
* `vote_cancel_zero_iff` : nothing is cancelled exactly when all the votes have one sign.
-/

open Finset

/-- Lemma T45a part 1: `0 ≤ ∑ |a i| - |∑ a i| ≤ ∑ |a i|`. -/
theorem vote_cancel_bounds {ι : Type*} (s : Finset ι) (a : ι → ℝ) :
    0 ≤ (∑ i ∈ s, |a i|) - |∑ i ∈ s, a i| ∧ (∑ i ∈ s, |a i|) - |∑ i ∈ s, a i| ≤ ∑ i ∈ s, |a i| := by
  refine ⟨by linarith [abs_sum_le_sum_abs a s], by linarith [abs_nonneg (∑ i ∈ s, a i)]⟩

/-- Lemma T45a part 2: `|∑ a i| = ∑ |a i|` exactly when every vote is `≥ 0` or every vote is `≤ 0`. -/
theorem vote_cancel_zero_iff {ι : Type*} (s : Finset ι) (a : ι → ℝ) :
    |∑ i ∈ s, a i| = ∑ i ∈ s, |a i| ↔ (∀ i ∈ s, 0 ≤ a i) ∨ (∀ i ∈ s, a i ≤ 0) := by
  constructor
  · intro h
    rcases le_or_gt 0 (∑ i ∈ s, a i) with hpos | hneg
    · left
      have hsum : ∑ i ∈ s, (|a i| - a i) = 0 := by
        rw [sum_sub_distrib, ← h, abs_of_nonneg hpos, sub_self]
      have hz := (sum_eq_zero_iff_of_nonneg (fun i _ => sub_nonneg.mpr (le_abs_self (a i)))).mp hsum
      intro i hi
      have := hz i hi
      linarith [abs_nonneg (a i)]
    · right
      have hsum : ∑ i ∈ s, (|a i| + a i) = 0 := by
        rw [sum_add_distrib, ← h, abs_of_neg hneg]; ring
      have hz := (sum_eq_zero_iff_of_nonneg (fun i _ => by linarith [neg_abs_le (a i)])).mp hsum
      intro i hi
      have := hz i hi
      linarith [abs_nonneg (a i)]
  · rintro (h | h)
    · rw [abs_of_nonneg (sum_nonneg h)]
      exact sum_congr rfl (fun i hi => (abs_of_nonneg (h i hi)).symm)
    · rw [abs_of_nonpos (sum_nonpos h), ← sum_neg_distrib]
      exact sum_congr rfl (fun i hi => (abs_of_nonpos (h i hi)).symm)
