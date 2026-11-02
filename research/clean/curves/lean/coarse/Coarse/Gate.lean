import Mathlib

/-!
# COARSE-THEORY.md Part B.G: a gate in front of a family (finite steps)

* `t40_gate_floor` : Theorem T40 part 1. On the twelve gate points of instance I1, every affine function of
  `(h_1, h_2, h_3, h_4)` has residual sum of squares at least `2` against N's gap. (The gate is `1` on all twelve
  points, so its coefficient is absorbed into the constant.)
* `t40_bstar` : Theorem T40 part 2. The build `B*` has residual sum of squares `70/49` on the thirteen points, and
  `(1/13) (70/49) / 4 = 5/182 < 7/182 = (1/13) * 2 / 4`.
-/

/-- Theorem T40 part 1. Points of the gate region and N's gap there:
the four axis points `2 e_i` (gap 1), the four corners `2 e_i + 2 e_{i+1}` (gap 2), the four points `e_i + e_{i+1}`
(gap 0), indices mod 4. The corners are dropped from the left side (their squares are non-negative). -/
theorem t40_gate_floor (c0 c1 c2 c3 c4 : ℝ) :
    2 ≤ (c0 + 2 * c1 - 1) ^ 2 + (c0 + 2 * c2 - 1) ^ 2 + (c0 + 2 * c3 - 1) ^ 2 + (c0 + 2 * c4 - 1) ^ 2
      + (c0 + c1 + c2) ^ 2 + (c0 + c2 + c3) ^ 2 + (c0 + c3 + c4) ^ 2 + (c0 + c4 + c1) ^ 2
      + (c0 + 2 * c1 + 2 * c2 - 2) ^ 2 + (c0 + 2 * c2 + 2 * c3 - 2) ^ 2
      + (c0 + 2 * c3 + 2 * c4 - 2) ^ 2 + (c0 + 2 * c4 + 2 * c1 - 2) ^ 2 := by
  nlinarith [sq_nonneg (c0 + 2 * c1 - 1 + 1 / 2), sq_nonneg (c0 + 2 * c2 - 1 + 1 / 2),
    sq_nonneg (c0 + 2 * c3 - 1 + 1 / 2), sq_nonneg (c0 + 2 * c4 - 1 + 1 / 2),
    sq_nonneg (c0 + c1 + c2 - 1 / 2), sq_nonneg (c0 + c2 + c3 - 1 / 2),
    sq_nonneg (c0 + c3 + c4 - 1 / 2), sq_nonneg (c0 + c4 + c1 - 1 / 2),
    sq_nonneg (c0 + 2 * c1 + 2 * c2 - 2), sq_nonneg (c0 + 2 * c2 + 2 * c3 - 2),
    sq_nonneg (c0 + 2 * c3 + 2 * c4 - 2), sq_nonneg (c0 + 2 * c4 + 2 * c1 - 2)]

/-- Theorem T40 part 2. The unit's pre-activations at the five kinds of points, the build's values, and the
residual sum of squares. -/
theorem t40_bstar :
    max ((19 / 14 : ℚ) * 2 + 1 / 2 * 0 - 1 - 1) 0 + 2 / 7 = 1 ∧
    max ((19 / 14 : ℚ) * 0 + 1 / 2 * 2 - 1 - 1) 0 + 2 / 7 = 2 / 7 ∧
    max ((19 / 14 : ℚ) * 2 + 1 / 2 * 2 - 1 - 1) 0 + 2 / 7 = 2 ∧
    max ((19 / 14 : ℚ) * 1 + 1 / 2 * 1 - 1 - 1) 0 + 2 / 7 = 2 / 7 ∧
    max ((19 / 14 : ℚ) * 0 + 1 / 2 * 0 - 0 - 1) 0 + 2 / 7 = 2 / 7 ∧
    (2 * (2 / 7 - 1) ^ 2 + 5 * (2 / 7 : ℚ) ^ 2 = 70 / 49) ∧
    ((1 / 13 : ℚ) * (70 / 49) / 4 = 5 / 182) ∧
    ((5 / 182 : ℚ) < (1 / 13) * 2 / 4) := by
  norm_num [max_def]
