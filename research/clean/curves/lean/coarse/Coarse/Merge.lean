/-
Section B.Y of COARSE-THEORY.md: the merge rule (Definition DY2, Theorem TY2).
The scalar steps: relu is 1-Lipschitz; a positive multiple inside relu comes out, so the members of a proportional
family write through one channel with the sum of their scaled weights.
-/
import Mathlib

open Finset BigOperators

namespace Coarse.Merge

/-- Theorem TY2 (1): relu is 1-Lipschitz, |relu a - relu b| <= |a - b|. -/
theorem relu_lip (a b : ℝ) : |max a 0 - max b 0| ≤ |a - b| :=
  abs_max_sub_max_le_abs a b 0

/-- Theorem TY2 (1): a read member writes through its normalised pre-activation, w relu(s t) = (s w) relu(t), s > 0. -/
theorem merge_exact (w s t : ℝ) (hs : 0 < s) : w * max (s * t) 0 = (s * w) * max t 0 := by
  have h : max (s * t) 0 = s * max t 0 := by
    rw [mul_max_of_nonneg _ _ hs.le, mul_zero]
  rw [h]; ring

/-- Theorem TY2 (2): members whose pre-activations are positive multiples s_u t of one function t write together
through relu(t) with the summed weight sum_u s_u w_u. -/
theorem merge_exact_sum {n : ℕ} (w s : Fin n → ℝ) (hs : ∀ i, 0 < s i) (t : ℝ) :
    ∑ i, w i * max (s i * t) 0 = (∑ i, s i * w i) * max t 0 := by
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl (fun i _ => merge_exact (w i) (s i) t (hs i))


/-- Proposition TY5 part 1, the identity: with lb the log of the mean softmax, the summed KL to any q splits into the
summed KL to the mean plus the mass times KL(mean || q). Written with lp = log p, lq = log q, lb = log pbar and the
weighted column sums S_i = sum_x w_x p_(x,i) = m pbar_i, so it is a finite-sum identity. -/
theorem kl_split {X I : Type*} [Fintype X] [Fintype I] (w : X → ℝ) (p lp : X → I → ℝ) (lq lb : I → ℝ) :
    (∑ x, w x * ∑ i, p x i * (lp x i - lq i)) =
      (∑ x, w x * ∑ i, p x i * (lp x i - lb i)) + ∑ i, (∑ x, w x * p x i) * (lb i - lq i) := by
  have h : (∑ i, (∑ x, w x * p x i) * (lb i - lq i)) = ∑ x, w x * ∑ i, p x i * (lb i - lq i) := by
    simp only [Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun x _ => Finset.sum_congr rfl (fun i _ => by ring))
  rw [h, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [← mul_add, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl (fun i _ => by ring)

/-- Lemma TI0 part 2 (Part B.I), the chain rule of KL over a partition: with `part` sending each point to its part
and lb c the log of part c's mean softmax, the summed KL to any q is the summed KL of each point to its own part's
mean plus, for each part, its column sums times (lb c - lq). With lb c = log of part c's mean, the last term is
sum over c of m_c KL(sbar_c || q). -/
theorem kl_chain {X C I : Type*} [Fintype X] [Fintype C] [DecidableEq C] [Fintype I] (part : X → C)
    (w : X → ℝ) (p lp : X → I → ℝ) (lq : I → ℝ) (lb : C → I → ℝ) :
    (∑ x, w x * ∑ i, p x i * (lp x i - lq i)) =
      (∑ x, w x * ∑ i, p x i * (lp x i - lb (part x) i)) +
        ∑ c, ∑ i, (∑ x ∈ Finset.univ.filter (fun x => part x = c), w x * p x i) * (lb c i - lq i) := by
  have h : (∑ c, ∑ i, (∑ x ∈ Finset.univ.filter (fun x => part x = c), w x * p x i) * (lb c i - lq i)) =
      ∑ x, w x * ∑ i, p x i * (lb (part x) i - lq i) := by
    calc _ = ∑ c, ∑ x ∈ Finset.univ.filter (fun x => part x = c), w x * ∑ i, p x i * (lb c i - lq i) := by
          refine Finset.sum_congr rfl (fun c _ => ?_)
          simp only [Finset.sum_mul, Finset.mul_sum]
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl (fun x _ => Finset.sum_congr rfl (fun i _ => by ring))
      _ = ∑ c, ∑ x ∈ Finset.univ.filter (fun x => part x = c), w x * ∑ i, p x i * (lb (part x) i - lq i) := by
          refine Finset.sum_congr rfl (fun c _ => Finset.sum_congr rfl (fun x hx => ?_))
          rw [(Finset.mem_filter.mp hx).2]
      _ = _ := Finset.sum_fiberwise _ _ _
  rw [h, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [← mul_add, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl (fun i _ => by ring)

end Coarse.Merge
