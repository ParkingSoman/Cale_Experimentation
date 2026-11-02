/-
Part B.E of COARSE-THEORY.md (errors that stay small): the finite steps of Lemma TE0, Proposition TE1 part 3 and
Theorem TE4.
-/
import Mathlib

open Finset BigOperators

namespace Coarse.Robust

/-- Lemma TE0, the log-sum-exp step (upper side). If every entry of `d` is at most `M`, adding `d` to the logits
raises log-sum-exp by at most `M`. -/
theorem lse_shift_le {n : ℕ} (hn : 0 < n) (y d : Fin n → ℝ) (M : ℝ) (hM : ∀ i, d i ≤ M) :
    Real.log (∑ i, Real.exp (y i + d i)) ≤ Real.log (∑ i, Real.exp (y i)) + M := by
  have hpos : 0 < ∑ i, Real.exp (y i) :=
    sum_pos (fun i _ => Real.exp_pos _) ⟨⟨0, hn⟩, mem_univ _⟩
  have hpos' : 0 < ∑ i, Real.exp (y i + d i) :=
    sum_pos (fun i _ => Real.exp_pos _) ⟨⟨0, hn⟩, mem_univ _⟩
  have hle : ∑ i, Real.exp (y i + d i) ≤ Real.exp M * ∑ i, Real.exp (y i) := by
    rw [mul_sum]
    refine sum_le_sum (fun i _ => ?_)
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [hM i])
  calc Real.log (∑ i, Real.exp (y i + d i)) ≤ Real.log (Real.exp M * ∑ i, Real.exp (y i)) :=
        Real.log_le_log hpos' hle
    _ = Real.log (∑ i, Real.exp (y i)) + M := by
        rw [Real.log_mul (Real.exp_pos M).ne' hpos.ne', Real.log_exp]; ring

/-- Lemma TE0, the log-sum-exp step (lower side). If every entry of `d` is at least `m`, adding `d` raises
log-sum-exp by at least `m`. -/
theorem lse_shift_ge {n : ℕ} (hn : 0 < n) (y d : Fin n → ℝ) (m : ℝ) (hm : ∀ i, m ≤ d i) :
    Real.log (∑ i, Real.exp (y i)) + m ≤ Real.log (∑ i, Real.exp (y i + d i)) := by
  have hpos : 0 < ∑ i, Real.exp (y i) :=
    sum_pos (fun i _ => Real.exp_pos _) ⟨⟨0, hn⟩, mem_univ _⟩
  have hle : Real.exp m * ∑ i, Real.exp (y i) ≤ ∑ i, Real.exp (y i + d i) := by
    rw [mul_sum]
    refine sum_le_sum (fun i _ => ?_)
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [hm i])
  calc Real.log (∑ i, Real.exp (y i)) + m = Real.log (Real.exp m * ∑ i, Real.exp (y i)) := by
        rw [Real.log_mul (Real.exp_pos m).ne' hpos.ne', Real.log_exp]; ring
    _ ≤ Real.log (∑ i, Real.exp (y i + d i)) :=
        Real.log_le_log (mul_pos (Real.exp_pos m) hpos) hle

/-- The cross-entropy part of KL(q || softmax y): `f q y = - Σ q_i y_i + log Σ exp y_i`. KL differs from it by a
term that depends on `q` only. -/
noncomputable def ce {n : ℕ} (q y : Fin n → ℝ) : ℝ :=
  -(∑ i, q i * y i) + Real.log (∑ i, Real.exp (y i))

/-- Lemma TE0 (KL moves by at most the oscillation of the logit change). Let `q` be a probability vector and let every
entry of `d` lie in `[m, M]`. Then `|f q (y + d) - f q y| ≤ M - m`. -/
theorem ce_osc {n : ℕ} (hn : 0 < n) (q y d : Fin n → ℝ) (hq : ∀ i, 0 ≤ q i) (hq1 : ∑ i, q i = 1)
    (m M : ℝ) (hm : ∀ i, m ≤ d i) (hM : ∀ i, d i ≤ M) :
    |ce q (fun i => y i + d i) - ce q y| ≤ M - m := by
  have hqd_le : ∑ i, q i * d i ≤ M := by
    calc ∑ i, q i * d i ≤ ∑ i, q i * M := sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hM i) (hq i))
      _ = M := by rw [← sum_mul, hq1, one_mul]
  have hqd_ge : m ≤ ∑ i, q i * d i := by
    calc m = ∑ i, q i * m := by rw [← sum_mul, hq1, one_mul]
      _ ≤ ∑ i, q i * d i := sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hm i) (hq i))
  have hsplit : ∑ i, q i * (y i + d i) = ∑ i, q i * y i + ∑ i, q i * d i := by
    rw [← sum_add_distrib]; exact sum_congr rfl (fun i _ => by ring)
  have hu := lse_shift_le hn y d M hM
  have hl := lse_shift_ge hn y d m hm
  unfold ce
  rw [hsplit, abs_le]
  constructor <;> linarith

/-- Proposition TE1 part 3, the counting step. If a build's loss on the network with the leak (`Dδ`) is within
`osc * δ * g` of its loss on the network without it (`D0`, which is at least the least loss `V0` there), and its loss
with the leak is at most `Vδ + η`, then `osc * (δ * g) ≥ V0 - Vδ - η`: the swing it gives the leaking unit grows like
`1 / δ`. -/
theorem swing_needed (Dδ D0 V0 Vδ η osc δ g : ℝ) (hD0 : V0 ≤ D0) (hclose : D0 - Dδ ≤ osc * δ * g)
    (hgood : Dδ ≤ Vδ + η) : V0 - Vδ - η ≤ osc * (δ * g) := by
  nlinarith

/-- Theorem TE4, the sandwich. The robust optimum is at least the least loss on the clean table (`lo`), because the
clean table is one of the networks the map describes, and at most the worst-case loss of the clean assembly `B0`
(`up`), which in turn is at most `lo` plus the error term `E`. -/
theorem sandwich (lo rob up E : ℝ) (hlo : lo ≤ rob) (hup : rob ≤ up) (hE : up ≤ lo + E) :
    |rob - lo| ≤ E := by
  rw [abs_le]; constructor <;> linarith

/-- Theorem TE4, the oscillation of a sum is at most the sum of the oscillations (the step that splits the error term
by its sources). Stated for two terms with given bounds. -/
theorem osc_add {n : ℕ} (a b : Fin n → ℝ) (ma Ma mb Mb : ℝ) (ha : ∀ i, ma ≤ a i ∧ a i ≤ Ma)
    (hb : ∀ i, mb ≤ b i ∧ b i ≤ Mb) (i : Fin n) :
    ma + mb ≤ a i + b i ∧ a i + b i ≤ Ma + Mb := by
  constructor <;> linarith [(ha i).1, (ha i).2, (hb i).1, (hb i).2]

/-- Theorem TE4, a leak entry: a unit with logit weights `w` (all in `[lo, hi]`) whose value changes by `t` changes the
logits by `t * w`, whose entries lie in an interval of length `|t| * (hi - lo)`. -/
theorem scaled_osc {n : ℕ} (w : Fin n → ℝ) (lo hi t : ℝ) (hw : ∀ i, lo ≤ w i ∧ w i ≤ hi) (i j : Fin n) :
    t * w i - t * w j ≤ |t| * (hi - lo) := by
  have h1 : t * w i - t * w j = t * (w i - w j) := by ring
  rw [h1]
  calc t * (w i - w j) ≤ |t * (w i - w j)| := le_abs_self _
    _ = |t| * |w i - w j| := abs_mul _ _
    _ ≤ |t| * (hi - lo) := by
        apply mul_le_mul_of_nonneg_left _ (abs_nonneg t)
        rw [abs_le]; constructor <;> linarith [(hw i).1, (hw i).2, (hw j).1, (hw j).2]

end Coarse.Robust
