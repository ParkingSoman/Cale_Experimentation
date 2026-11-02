import Mathlib

/-!
# COARSE-THEORY.md Part B.S: the shape of the optimum

* `transfer_bound` : Theorem T42. In any metric space, with `O` any set (the output set of the given widths), `yN`
  and `yN'` two points (the two networks' centred outputs on the mix) and `b` a point of `O` within `η` of the
  distance from `yN'` to `O` (a near-least build for `N'`): the distance from `b` to `yN` is at most the distance
  from `yN` to `O`, plus `2 dist yN yN'`, plus `η`.
* `infDist_lipschitz` : Theorem T42 part 1, the optimum value moves at most as far as the network.
-/

open Metric

/-- Theorem T42 part 1: `|infDist yN O - infDist yN' O| ≤ dist yN yN'`. -/
theorem infDist_lipschitz {α : Type*} [MetricSpace α] (O : Set α) (yN yN' : α) :
    |infDist yN O - infDist yN' O| ≤ dist yN yN' := by
  rw [abs_le]
  constructor
  · have h := infDist_le_infDist_add_dist (s := O) (x := yN') (y := yN)
    rw [dist_comm] at h
    linarith
  · have h := infDist_le_infDist_add_dist (s := O) (x := yN) (y := yN')
    linarith

/-- Theorem T42 part 2: a near-least point for `yN'` is near-least for `yN`, up to twice the distance. -/
theorem transfer_bound {α : Type*} [MetricSpace α] (O : Set α) (yN yN' b : α) (η : ℝ)
    (hb : dist b yN' ≤ infDist yN' O + η) :
    dist b yN ≤ infDist yN O + 2 * dist yN yN' + η := by
  have h1 : dist b yN ≤ dist b yN' + dist yN' yN := dist_triangle b yN' yN
  have h2 := infDist_le_infDist_add_dist (s := O) (x := yN') (y := yN)
  rw [dist_comm yN' yN] at h1 h2
  linarith

/-- Theorem T41 part 2: over a finite non-empty index set of pieces (patterns), the infimum of the objective over the
union of the pieces equals the infimum over one piece (an optimal pattern). `S σ` is the set of objective values of
the builds with pattern `σ`, in the extended reals (an empty piece has infimum `⊤`). -/
theorem t41_min_over_pieces {ι : Type*} [Finite ι] [Nonempty ι] (S : ι → Set EReal) :
    ∃ i, sInf (⋃ j, S j) = sInf (S i) := by
  obtain ⟨i, hi⟩ := Finite.exists_min (fun j => sInf (S j))
  refine ⟨i, ?_⟩
  rw [sInf_iUnion]
  exact le_antisymm (iInf_le _ i) (le_iInf hi)

/-- Theorem T41 part 4: a build with value `v` in piece `i` has `v ≥ inf over its piece ≥ inf over all pieces`. -/
theorem t41_error {ι : Type*} (S : ι → Set EReal) (i : ι) (v : EReal) (hv : v ∈ S i) :
    sInf (⋃ j, S j) ≤ sInf (S i) ∧ sInf (S i) ≤ v :=
  ⟨sInf_le_sInf (Set.subset_iUnion S i), sInf_le hv⟩
