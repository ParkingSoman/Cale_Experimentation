import Mathlib
import Coarse.Basic

/-!
# UNIT-BRANCH.md B.6 and B.7: activations and edges

* `loss_eq_of_edges_eq` (Lemma TA1, summation step): a sum of per-point losses that read only the edges is the same
  for two builds whose edges agree at every point.
* `rescale_edge`, `split_edge` (Lemma TA2 part 2): rescaling a unit (and its out-weight inversely) or splitting it
  into two copies changes its activation reading but not the flow on its out-edges.
* `removal_cost_D2`, `same_act_diff_cost` (Lemma TA2 part 1): two units with the same activation at every point but
  different gap weights have different removal costs.
* `votes_same_sum`, `relu_neg_eq` (Lemma TA3): two different vote vectors give the same sum; every negative
  pre-activation gives the same output.
* `rank_flip` (Lemma TA4 part 2): the unit with the larger activation can have the smaller removal cost.
* `relu_gate` (Lemma TA5, one step): relu z = (gate of z) * z.
* `exists_help` (Proposition TA6, D2 form): if a unit's activation is correlated with a residual on the mix, a new
  write of that activation lowers the squared loss. The unit's own out-weight does not appear.
* `two_logit_hessian`, `ortho_instance` (Proposition TA7): with two logits the KL curvature pairs two writes through
  their gap weights only; with three logits two non-zero writes can be orthogonal in the curvature.
-/

open Finset

/-- Lemma TA1 (summation step). -/
theorem loss_eq_of_edges_eq {X E : Type*} (s : Finset X) (p : X → ℝ) (f : X → E → ℝ) (e₁ e₂ : X → E)
    (h : ∀ x ∈ s, e₁ x = e₂ x) :
    ∑ x ∈ s, p x * f x (e₁ x) = ∑ x ∈ s, p x * f x (e₂ x) :=
  sum_congr rfl (fun x hx => by rw [h x hx])

/-- Lemma TA2 part 2 (rescaling): activation `c * h`, out-weight `w / c`, same flow `h * w`. -/
theorem rescale_edge (c h w : ℝ) (hc : c ≠ 0) : (c * h) * (w / c) = h * w := by
  field_simp

/-- Lemma TA2 part 2 (splitting): two copies with out-weights `a * w` and `(1 - a) * w` send the flow of one. -/
theorem split_edge (a w h : ℝ) : a * w * h + (1 - a) * w * h = w * h := by ring

/-- Lemma TA2 part 1: the D2 removal cost of a last-layer unit with gap weight `d` is `d ^ 2` times its
activation energy on the mix. -/
theorem removal_cost_D2 {X : Type*} (s : Finset X) (p h : X → ℝ) (d : ℝ) :
    ∑ x ∈ s, p x * (d * h x) ^ 2 = d ^ 2 * ∑ x ∈ s, p x * h x ^ 2 := by
  rw [mul_sum]
  exact sum_congr rfl (fun x _ => by ring)

/-- Lemma TA2 part 1: same activation `h`, gap weights `d₁ ≠ ± d₂`, positive energy: different removal costs. -/
theorem same_act_diff_cost {X : Type*} (s : Finset X) (p h : X → ℝ) (d₁ d₂ : ℝ)
    (hd : d₁ ^ 2 ≠ d₂ ^ 2) (hE : 0 < ∑ x ∈ s, p x * h x ^ 2) :
    ∑ x ∈ s, p x * (d₁ * h x) ^ 2 ≠ ∑ x ∈ s, p x * (d₂ * h x) ^ 2 := by
  rw [removal_cost_D2, removal_cost_D2]
  intro heq
  exact hd (mul_right_cancel₀ (ne_of_gt hE) heq)

/-- Lemma TA3 part 1: two vote vectors with disjoint supports and the same sum. -/
theorem votes_same_sum (t : ℝ) (ht : t ≠ 0) :
    ∃ c₁ c₂ : Fin 2 → ℝ, c₁ ≠ c₂ ∧ c₁ 1 = 0 ∧ c₂ 0 = 0 ∧ ∑ i, c₁ i = ∑ i, c₂ i := by
  refine ⟨![t, 0], ![0, t], fun h => ht ?_, rfl, rfl, by simp [Fin.sum_univ_two]⟩
  simpa using congrFun h 0

/-- Lemma TA3 part 2: relu sends every non-positive pre-activation to the same output. -/
theorem relu_neg_eq (a b : ℝ) (ha : a ≤ 0) (hb : b ≤ 0) : relu a = relu b := by
  unfold relu
  rw [max_eq_right ha, max_eq_right hb]

/-- Lemma TA4 part 2 (instance): unit u has activation 2 and gap weight 1/10, unit v has activation 1 and gap
weight 1. u is stronger, v costs more to remove. -/
theorem rank_flip : (1 : ℝ) < 2 ∧ (2 * (1 / 10 : ℝ)) ^ 2 < (1 * 1 : ℝ) ^ 2 := by
  constructor <;> norm_num

/-- Lemma TA5 (one step): a ReLU output is its gate times its pre-activation. -/
theorem relu_gate (z : ℝ) : relu z = (if 0 < z then 1 else 0) * z := by
  unfold relu
  split_ifs with h
  · rw [max_eq_left h.le, one_mul]
  · rw [max_eq_right (not_lt.mp h), zero_mul]

/-- Proposition TA6 (D2 form): if `A = ∑ p r h ≠ 0` (the activation `h` is correlated with the residual `r` on the
mix, weights `p > 0`), some new write `e` of `h` lowers `∑ p r²`. -/
theorem exists_help {X : Type*} (s : Finset X) (p r h : X → ℝ) (hp : ∀ x ∈ s, 0 < p x)
    (hA : ∑ x ∈ s, p x * r x * h x ≠ 0) :
    ∃ e : ℝ, ∑ x ∈ s, p x * (r x - e * h x) ^ 2 < ∑ x ∈ s, p x * r x ^ 2 := by
  set A := ∑ x ∈ s, p x * r x * h x with hAdef
  set B := ∑ x ∈ s, p x * h x ^ 2 with hBdef
  have hB : 0 < B := by
    have hnn : ∀ x ∈ s, 0 ≤ p x * h x ^ 2 := fun x hx => mul_nonneg (hp x hx).le (sq_nonneg _)
    rcases (sum_nonneg hnn).lt_or_eq with hlt | heq
    · exact hlt
    · exfalso
      have hz := (sum_eq_zero_iff_of_nonneg hnn).mp heq.symm
      apply hA
      refine sum_eq_zero (fun x hx => ?_)
      have h0 : h x = 0 := by
        have := hz x hx
        rcases mul_eq_zero.mp this with h1 | h1
        · exact absurd h1 (ne_of_gt (hp x hx))
        · exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h1
      rw [h0, mul_zero]
  have key : ∀ e : ℝ, ∑ x ∈ s, p x * (r x - e * h x) ^ 2
      = ∑ x ∈ s, p x * r x ^ 2 - 2 * e * A + e ^ 2 * B := by
    intro e
    rw [hAdef, hBdef, mul_sum, mul_sum, ← sum_sub_distrib, ← sum_add_distrib]
    exact sum_congr rfl (fun x _ => by ring)
  refine ⟨A / B, ?_⟩
  rw [key]
  have hA2 : 0 < A ^ 2 / B := div_pos (by positivity) hB
  have : -2 * (A / B) * A + (A / B) ^ 2 * B = -(A ^ 2 / B) := by
    field_simp
    ring
  linarith

/-- Proposition TA7 (two logits): with `s₁ + s₂ = 1`, the KL curvature `diag s - s sᵀ` pairs the writes `w` and `w'`
as `s₁ s₂ (w₁ - w₂)(w₁' - w₂')`: only the gap weights enter. -/
theorem two_logit_hessian (s₁ s₂ w₁ w₂ v₁ v₂ : ℝ) (hs : s₁ + s₂ = 1) :
    s₁ * w₁ * v₁ + s₂ * w₂ * v₂ - (s₁ * w₁ + s₂ * w₂) * (s₁ * v₁ + s₂ * v₂)
      = s₁ * s₂ * (w₁ - w₂) * (v₁ - v₂) := by
  have : s₂ = 1 - s₁ := by linarith
  subst this
  ring

/-- Proposition TA7 (three logits, instance): at uniform softmax the curvature is `(1/3)(I - J/3)`; the writes
`(1, -1, 0)` and `(1, 1, -2)` are both non-zero and orthogonal in it. -/
theorem ortho_instance :
    (1 / 3 : ℝ) * (1 * 1 + (-1) * 1 + 0 * (-2)) - (1 / 9) * ((1 + (-1) + 0) * (1 + 1 + (-2))) = 0 := by
  norm_num
