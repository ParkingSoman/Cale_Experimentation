/-
Section B.N of COARSE-THEORY.md: what the error term counts when a network grows (Propositions TN1, TN2, Theorem TN3).
The scalar steps: n copies with shares lam_i that add to 1 carry what one unit carried; a merged weight has a swing at
most the shared-out swings; entries that each lie between their two ends add to a sum between the two sums; a weighted
mean is at most its largest entry.
-/
import Mathlib

open Finset BigOperators

namespace Coarse.Scale

/-- Proposition TN1 (lift): n copies with shares `lam i` that add to 1, each carrying `lam i * a`, carry `a` in all.
This is the step for every term of E(B^0) and every mass under a copy. -/
theorem copy_mass {n : ℕ} (lam : Fin n → ℝ) (hsum : ∑ i, lam i = 1) (a : ℝ) :
    ∑ i, lam i * a = a := by
  rw [← Finset.sum_mul, hsum, one_mul]

/-- Proposition TN1 (merge): if every copy's logit weights lie in `[lo i, hi i]` and the shares are `>= 0`, the merged
weights `sum_i lam i * w i k` lie in `[sum lam lo, sum lam hi]`. So the merged swing is at most the share-weighted sum of
the copies' swings. -/
theorem merge_osc {n m : ℕ} (lam : Fin n → ℝ) (hlam : ∀ i, 0 ≤ lam i) (w : Fin n → Fin m → ℝ)
    (lo hi : Fin n → ℝ) (hw : ∀ i k, lo i ≤ w i k ∧ w i k ≤ hi i) (k : Fin m) :
    ∑ i, lam i * lo i ≤ ∑ i, lam i * w i k ∧ ∑ i, lam i * w i k ≤ ∑ i, lam i * hi i := by
  constructor
  · exact Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hw i k).1 (hlam i))
  · exact Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hw i k).2 (hlam i))

/-- Corollary TN1.1 (merge): the merged weights' spread between any two logits is at most the share-weighted sum of
the copies' spreads, so osc(sum lam_i w_i) <= sum lam_i osc(w_i). -/
theorem merge_osc_le {n m : ℕ} (lam : Fin n → ℝ) (hlam : ∀ i, 0 ≤ lam i) (w : Fin n → Fin m → ℝ)
    (lo hi : Fin n → ℝ) (hw : ∀ i k, lo i ≤ w i k ∧ w i k ≤ hi i) (k l : Fin m) :
    ∑ i, lam i * w i k - ∑ i, lam i * w i l ≤ ∑ i, lam i * (hi i - lo i) := by
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_le_sum (fun i _ => ?_)
  have h : lam i * w i k - lam i * w i l = lam i * (w i k - w i l) := by ring
  rw [h]
  exact mul_le_mul_of_nonneg_left (by linarith [(hw i k).2, (hw i l).1]) (hlam i)

/-- Propositions TN1 and TN2 (tables): entries that each lie between their clean value `a i` and N's value `b i`, all
in the same order, add to a sum between the two sums. -/
theorem between_sum {n : ℕ} (a x b : Fin n → ℝ) (h : ∀ i, a i ≤ x i ∧ x i ≤ b i) :
    ∑ i, a i ≤ ∑ i, x i ∧ ∑ i, x i ≤ ∑ i, b i := by
  constructor
  · exact Finset.sum_le_sum (fun i _ => (h i).1)
  · exact Finset.sum_le_sum (fun i _ => (h i).2)

/-- Theorem TN3: a sum of masses times relative deviations is the total mass times a weighted mean, and that mean is at
most the largest relative deviation. -/
theorem mass_mean {n : ℕ} (μ r : Fin n → ℝ) (hμ : ∀ i, 0 ≤ μ i) (R : ℝ) (hr : ∀ i, r i ≤ R) :
    ∑ i, μ i * r i ≤ (∑ i, μ i) * R := by
  rw [Finset.sum_mul]
  exact Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hr i) (hμ i))

/-- Theorem TN3 part 3: the oscillation of a sum is at most the sum of the oscillations, for any finite number of
terms (each term `v i` has its entries in `[lo i, hi i]`). Used for Lambda_M <= rho(N) phi(N). -/
theorem osc_sum {n m : ℕ} (v : Fin n → Fin m → ℝ) (lo hi : Fin n → ℝ) (hv : ∀ i k, lo i ≤ v i k ∧ v i k ≤ hi i)
    (k l : Fin m) :
    ∑ i, v i k - ∑ i, v i l ≤ ∑ i, (hi i - lo i) := by
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_le_sum (fun i _ => by linarith [(hv i k).2, (hv i l).1])

end Coarse.Scale
