# Referee report: the decision ledger (2026-10-25, late)

Referee: math-referee agent (opus), narrow task: hypothesis tags and paths (DECISIONS.md section 6), and the claims of
DECISIONS.md sections 2-3 and paper Part 0. No lake build.

## Findings (summary) and what was done

| Finding | Fix |
|---|---|
| 26 paths missed a premise; 5 named a node the item does not use | 11 new nodes (R-pwl, R-diagmetric, R-faithful, R-range, R-linseen, R-field, R-readergate, S-linear, R-gaussnu, N-linear, N-onereadpos). R-bench added to every family-branch path. R-gate on T14, R-onecol on T15. R-local removed from B0.1, R-pointwise from T7, R-obj and R-eq from T10.1. D1 uses R-field in place of R-clean. "linear-seen" moved from TECH to R-linseen |
| (a) "T14's networks and builds are not ReLU networks" is backwards: on a finite law a linear unit is a ReLU unit with a large bias. What blocks the transfer is that ReLU builds form a larger feasible set (different optimal value, no reduction). The NEG rule and Card 8 had the same error | Section 2, the NEG kind and Card 8 rewritten: hardness of "the best build" carries over only for the same build set, or through a stated reduction |
| (b) T12 is a floor on D2, under an affine-head premise; not shown reachable by a build of N's kind; a floor for KL only through T10.3 | Section 2 row and Part 0 restated |
| (c) T1' covers ReLU units reading one common field only; not units that read new lower layers (the T* case) | Section 2 row and paths note restated |
| (d) "an exact fast rule for every network is likely impossible" was used as if established; F9 path had no R-stat; "P must be an input" is Shonu's choice, not a consequence of F9; CONV "false for every real network" is too strong; D7 is defined for family builds only | Belief marked as a conjecture in section 2, section 3 and Part 0; F9 row path R-stat; reason 1 of section 3 restated; Part 0 says D_P is D7's formula over whole-network builds |

Not changed: the referee notes that H-sym in Conjecture D1 is redundant. The tag still names it, so the path keeps it,
with a note. Fixing the tag is a paper edit left for the next session.
