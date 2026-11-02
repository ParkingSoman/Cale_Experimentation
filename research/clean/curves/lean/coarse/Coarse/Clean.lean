import Mathlib

/-!
# COARSE-THEORY.md Part B.K: the clean detector network (Lemma T43a, Theorem T43)

* `t43_r_choice` : the threshold `r = 1 - η/8` meets `1/2 ≤ r < 1` and `1 - η/2 ≤ r²` for `0 < η ≤ 2`.
* `t43_r_ge` : from `r² ≥ 1 - η/2` and `0 ≤ r ≤ 1` it follows that `r ≥ 1 - η/2`.
* `t43_vertex_sibling_off` : at a vertex point, a sibling detector is off (`c - r < 0`).
* `t43_midpoint_off` : at a same-block midpoint, the two detectors of the pair are off (`(1 + c)/2 - r ≤ 0`).
* `t43_third_off` : at a same-block midpoint, a third detector of the block is off.
* `t43_same_block_apart` : two detectors of one block are never on together on the domain (Cauchy-Schwarz in the
  plane, written with coordinates).
* `t43_cross_block_apart` : two detectors of different blocks are never on together on the domain.
* `mono_or_anti_comp` : a composition of two maps, each monotone or antitone, is monotone or antitone (the midpoint
  step of Theorem T22 for builds of sizes `(1, ..., 1)`, with `mono_mid_ge_min` of `Coarse.Trunk`).
-/

/-- Lemma T43a: the choice `r = 1 - η/8`. -/
theorem t43_r_choice (η : ℝ) (h0 : 0 < η) (h2 : η ≤ 2) :
    1 / 2 ≤ 1 - η / 8 ∧ 1 - η / 8 < 1 ∧ 1 - η / 2 ≤ (1 - η / 8) ^ 2 := by
  refine ⟨by linarith, by linarith, ?_⟩
  nlinarith [sq_nonneg η]

/-- Lemma T43a: `r ≥ r²` on `[0, 1]`, so `r² ≥ 1 - η/2` gives `r ≥ 1 - η/2`. -/
theorem t43_r_ge (r η : ℝ) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hsq : 1 - η / 2 ≤ r ^ 2) : 1 - η / 2 ≤ r := by
  nlinarith

/-- Lemma T43a part 6: at a vertex point, a sibling detector reads `c = ⟨x~_i', x~_i⟩ ≤ 1 - η` and is off. -/
theorem t43_vertex_sibling_off (c r η : ℝ) (h0 : 0 < η) (hc : c ≤ 1 - η) (hr : 1 - η / 2 ≤ r) : c - r < 0 := by
  linarith

/-- Lemma T43a part 6: at the midpoint of two vertex points of one block, each of the pair's detectors reads
`(1 + c)/2` and is off. -/
theorem t43_midpoint_off (c r η : ℝ) (hc : c ≤ 1 - η) (hr : 1 - η / 2 ≤ r) : (1 + c) / 2 - r ≤ 0 := by
  linarith

/-- Lemma T43a part 6: at the midpoint of two vertex points of one block, a third detector of the block reads
`(c₁ + c₂)/2` with both `cᵢ ≤ 1 - η` and is off. -/
theorem t43_third_off (c₁ c₂ r η : ℝ) (h0 : 0 < η) (h1 : c₁ ≤ 1 - η) (h2 : c₂ ≤ 1 - η) (hr : 1 - η / 2 ≤ r) :
    (c₁ + c₂) / 2 - r < 0 := by
  linarith

/-- Lemma T43a part 2: two detectors of one block, with unit directions `a`, `b` (`⟨a, b⟩ ≤ 1 - η`), are never on
together at a block value `y` with `|y| ≤ 1`, when `r ≥ 0` and `r² ≥ 1 - η/2`. -/
theorem t43_same_block_apart (a₁ a₂ b₁ b₂ y₁ y₂ r η : ℝ)
    (ha : a₁ ^ 2 + a₂ ^ 2 = 1) (hb : b₁ ^ 2 + b₂ ^ 2 = 1) (hab : a₁ * b₁ + a₂ * b₂ ≤ 1 - η)
    (hy : y₁ ^ 2 + y₂ ^ 2 ≤ 1) (hr0 : 0 ≤ r) (hr2 : 1 - η / 2 ≤ r ^ 2)
    (hp : r < a₁ * y₁ + a₂ * y₂) (hq : r < b₁ * y₁ + b₂ * y₂) : False := by
  -- s = ⟨a + b, y⟩ > 2r ≥ 0, so s² > 4r² ≥ 4 - 2η; but s² ≤ |a + b|² |y|² ≤ 2 + 2⟨a, b⟩ ≤ 4 - 2η.
  set s := (a₁ + b₁) * y₁ + (a₂ + b₂) * y₂ with hs
  have hs2 : 2 * r < s := by rw [hs]; linarith
  have hlag : s ^ 2 ≤ ((a₁ + b₁) ^ 2 + (a₂ + b₂) ^ 2) * (y₁ ^ 2 + y₂ ^ 2) := by
    rw [hs]; nlinarith [sq_nonneg ((a₁ + b₁) * y₂ - (a₂ + b₂) * y₁)]
  have hn : (a₁ + b₁) ^ 2 + (a₂ + b₂) ^ 2 = 2 + 2 * (a₁ * b₁ + a₂ * b₂) := by nlinarith
  have hn0 : 0 ≤ (a₁ + b₁) ^ 2 + (a₂ + b₂) ^ 2 := by positivity
  have hup : s ^ 2 ≤ 4 - 2 * η := by
    have : ((a₁ + b₁) ^ 2 + (a₂ + b₂) ^ 2) * (y₁ ^ 2 + y₂ ^ 2) ≤ (a₁ + b₁) ^ 2 + (a₂ + b₂) ^ 2 := by
      nlinarith
    linarith
  have hlow : 4 * r ^ 2 < s ^ 2 := by nlinarith
  linarith

/-- Lemma T43a part 2: two detectors of different blocks are never on together on the domain
`|x_(1)| + ... + |x_(κ)| ≤ 1` when `r ≥ 1/2` (each reads at most its own block's norm). -/
theorem t43_cross_block_apart (p q n₁ n₂ r : ℝ) (hp : r < p) (hq : r < q) (hpn : p ≤ n₁) (hqn : q ≤ n₂)
    (hX : n₁ + n₂ ≤ 1) (hr : 1 / 2 ≤ r) : False := by
  linarith

/-- Theorem T43: a composition of two maps of the real line, each monotone or antitone, is monotone or antitone.
By induction over the layers, the gap of a build of sizes `(1, ..., 1)` is such a map of one affine argument. -/
theorem mono_or_anti_comp (f g : ℝ → ℝ) (hf : Monotone f ∨ Antitone f) (hg : Monotone g ∨ Antitone g) :
    Monotone (f ∘ g) ∨ Antitone (f ∘ g) := by
  rcases hf with hf | hf <;> rcases hg with hg | hg
  · exact Or.inl (hf.comp hg)
  · exact Or.inr (hf.comp_antitone hg)
  · exact Or.inr (hf.comp_monotone hg)
  · exact Or.inl (hf.comp hg)
