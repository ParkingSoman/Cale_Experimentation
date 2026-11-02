import Mathlib

/-!
# UNIT-BRANCH.md (branch of COARSE-THEORY.md): the unit of coarsening

* `silent_add` (Theorems TU2, TU5, summation step): if at every point one of two single losses is zero and the joint
  loss equals the other one, the joint total is the sum of the single totals.
* `cross_zero_add` (Theorem TU4 part 1): a function of two independent arguments that is a sum of a function of each
  has infimum equal to the sum of the infima (stated for least values attained at given points).
* `cover_compose` (Proposition TU9): covering composes.
-/

open Finset

/-- Theorems TU2 and TU5 (summation step). -/
theorem silent_add {α : Type*} (s : Finset α) (p f g h : α → ℝ)
    (hx : ∀ x ∈ s, (g x = 0 ∧ h x = f x) ∨ (f x = 0 ∧ h x = g x)) :
    ∑ x ∈ s, p x * h x = ∑ x ∈ s, p x * f x + ∑ x ∈ s, p x * g x := by
  rw [← sum_add_distrib]
  refine sum_congr rfl (fun x hxs => ?_)
  rcases hx x hxs with ⟨hg, hh⟩ | ⟨hf, hh⟩
  · rw [hh, hg]; ring
  · rw [hh, hf]; ring

/-- Theorem TU4 part 1: if the joint loss is F a + G b for every pair, and a0, b0 are least points of F and G,
then (a0, b0) is a least pair and the least joint value is F a0 + G b0. -/
theorem cross_zero_add {A B : Type*} (F : A → ℝ) (G : B → ℝ) (J : A → B → ℝ)
    (hJ : ∀ a b, J a b = F a + G b) (a0 : A) (b0 : B)
    (ha : ∀ a, F a0 ≤ F a) (hb : ∀ b, G b0 ≤ G b) :
    (∀ a b, J a0 b0 ≤ J a b) ∧ J a0 b0 = F a0 + G b0 := by
  refine ⟨fun a b => ?_, hJ a0 b0⟩
  rw [hJ, hJ]
  exact add_le_add (ha a) (hb b)

/-- Proposition TU9: covering composes. `offS` = every member of S is off; `offC` = every unit of the first box is
off; `offC'` = every unit of the second box is off. -/
theorem cover_compose (offS offC offC' : Prop) (h1 : offS → offC) (h2 : offC → offC') : offS → offC' :=
  fun h => h2 (h1 h)
