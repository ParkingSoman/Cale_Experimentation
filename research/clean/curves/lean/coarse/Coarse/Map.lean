/-
Trunk T5, Part B.M of COARSE-THEORY.md: the finite steps of Theorem TM1
(maps, regions, judgment units).
-/
import Mathlib

open Finset BigOperators

namespace Coarse.Map

/-- TM1, the judgment step. Let the units `k ∈ K` be 0 at every input outside the region `R`, and let the weights
`a` make their sum equal to 1 at every input of `R`. Adding `w * a k` to the readers' weight on each unit `k` moves
the output by `w` on `R` and leaves it alone outside `R` (one logit coordinate; a vector `w` is coordinatewise). -/
theorem judgment_shift {X ι : Type*} (K : Finset ι) (R : Set X) [DecidablePred (· ∈ R)]
    (h : ι → X → ℝ) (a : ι → ℝ) (y : X → ℝ) (w : ℝ)
    (hoff : ∀ x, x ∉ R → ∀ k ∈ K, h k x = 0)
    (hon : ∀ x, x ∈ R → ∑ k ∈ K, a k * h k x = 1) (x : X) :
    y x + ∑ k ∈ K, (w * a k) * h k x = y x + (if x ∈ R then w else 0) := by
  by_cases hx : x ∈ R
  · have : ∑ k ∈ K, (w * a k) * h k x = w * ∑ k ∈ K, a k * h k x := by
      rw [mul_sum]; exact sum_congr rfl (fun k _ => by ring)
    rw [this, hon x hx, mul_one]; simp [hx]
  · have : ∑ k ∈ K, (w * a k) * h k x = 0 :=
      sum_eq_zero (fun k hk => by rw [hoff x hx k hk, mul_zero])
    rw [this]; simp [hx]

/-- TM1 part 3, the splitting step. If the loss of a build is a sum over regions of a term that
depends only on that region's own parameters, then a choice of least parameters for each region
gives a loss no larger than that of any other build. -/
theorem split_least {ι : Type*} (s : Finset ι) (A : ι → Type*) (F : ∀ i, A i → ℝ)
    (a0 : ∀ i, A i)
    (hmin : ∀ i ∈ s, ∀ t : A i, F i (a0 i) ≤ F i t) (θ : ∀ i, A i) :
    ∑ i ∈ s, F i (a0 i) ≤ ∑ i ∈ s, F i (θ i) :=
  sum_le_sum (fun i hi => hmin i hi (θ i))

/-- TM1, the lower-bound step. If every region's term is at least that region's infimum `m i`, the loss of every
build is at least the sum of the `m i`. -/
theorem split_lower {ι : Type*} (s : Finset ι) (G m : ι → ℝ) (hG : ∀ i ∈ s, m i ≤ G i) :
    ∑ i ∈ s, m i ≤ ∑ i ∈ s, G i :=
  sum_le_sum hG

/-- TM1, the cancelling step for the constants: the outputs of groups outside their own regions
add one constant vector `v g` everywhere; a global bias `c - ∑ v` removes all of them at once
(one logit coordinate). -/
theorem constants_cancel {γ : Type*} (G : Finset γ) (v : γ → ℝ) (c : ℝ) :
    (c - ∑ g ∈ G, v g) + ∑ g ∈ G, v g = c := by ring

end Coarse.Map
