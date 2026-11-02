# Referee report: Part B.V.3, overlap and cancellation (round 1 of B.V.3), 2026-10-29

Referee: the `math-referee` agent (it may not write files; this copy was saved by the session from its report, word for
word in substance, with the table and the findings kept).

**Files read:** the worktree copies of `COARSE-THEORY.md`, `lean/coarse/Coarse/Stream.lean`, `DECISIONS.md`.

**Lean:** `lake build` passes (only style warnings). No `sorry` in `Coarse/*.lean`.

## Short answers to the seven questions

| Q | Answer |
|---|---|
| 1 | The proofs are correct as written. The faults are in scope, classes and wording (findings 4, 7 to 12, 14, 16 to 18). |
| 2 | Yes. Every z_l(x) lies in D perp, so the objective sees only Pi r'. Part 4 is correct, also for the logits with a re-fit bias. On a finite P with \|P\| <= d' + 1, part 4's premise holds generically: interpolation (finding 4). |
| 3 | Yes. J = Pi_K on D perp is a bijection onto K, Pi_K = Pi_K Pi, the induction closes both ways. The function class is fixed by D. Claim errors and bounds are not (finding 8). |
| 4 | B^+ is a network of DV1 with N's constants. It is not a DV2 cut: the constant first write and bias writes are new writes. Cost "at most one direction", no room when every k_S = 0 (findings 10, 11). |
| 5 | The DM14 rewrite removes the conflict in the letter only. Which claims the map holds encodes the cut and its size. T5 absorbs that only through a completeness condition for area maps, which is not defined (finding 12). |
| 6 | Yes, TV13 is exact, including a' = a. Lipschitz sigma is needed only for \|s\| <= Lip (finding 15). |
| 7 | DV3 is not well posed as stated (norm mismatch; c depends on B). The route is in effect a proof: prove it as a Proposition, move "c small on trained networks" to Part C (finding 16). |

## Table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| Fact F14 parts 1, 2 | proof | OK | inductions correct |
| Fact F14 Comment | statement | GAP | "only whether the sum is direct" holds for the function class, not for claim errors (orthogonal P_A); finer GL invariants for non-direct families of 3 or more |
| TV9 parts 1, 2 | proof | OK | OLS; min over D perp = min over all r'; rows in D perp |
| TV9 part 3 | tag | GAP | uses TV6 (Lipschitz sigma), unstated; eps'_l may be below eps_l |
| TV9 part 4 | scope | GAP | interpolation when \|P\| <= d' + 1 |
| TV9 part 4 | class | GAP | V_area = 0 needs B* in DV2 (eps'_l >= eps_l) |
| TV9 Lean | match | weaker | `ols_le` is the nearest-point property only; honestly labelled |
| TV9.1 | proof | OK | wording: the read weight is 1, the large weight is the logit's s |
| TV9.2 | strength | weaker than possible | holds for every network whose computation is a function of Pi e |
| TV10 | proof | OK | Comment overstates: function class only; Lean `gl_read` is for an automorphism |
| TV11 | proof | OK | sharp form: r' exists iff r vanishes on D cap E; Comment wrong for pairwise-trivial non-direct families (three lines in a plane); vacuous under laminar DM11 |
| TV12 | proof | OK | class: B^+ is DV1 but not a DV2 cut (new constant writes); "at most one" direction; sandwich missing; folded-form caveat (RMSNorm constant scales with width) |
| TV13 | proof | OK | exact; \|s\| <= Lip needs Lipschitz sigma; "quantum algorithm" is a metaphor |
| DV3 | well posed | GAP | L2 against pointwise remainder; c depends on B's reads; "small on trained networks" is a measurement; route is a proof; sigma' Lipschitz needs a node; Lambda_N names a build unless reads are canonical |
| DM14 last paragraph | statement | GAP | D includes own parts (not areas): "members of the decomposition"; the cut moved into which claims the map holds; completeness for area maps undefined |
| TV4 part 1 sentence | logic | GAP | "holds V_S as an area holds its readout claim" is a non sequitur; the TV3 route is not in M's errors |
| TV6.1 Comment | statement | GAP | needs D to be the dropped part of a cut at G; own-part issue |
| DM11 note | statement | GAP | "areas whose sum is direct" ill-defined for nested families |
| DECISIONS | consistency | GAP | stale "under R-cutclaim" text in card 3-23, the TV6.1 row; card 3-25 option (a) overstates |

Count: structural 1 (TV9 part 4, "kept content predicts N's reads", with TV9.1 and TV7 as the separating pair).
Bookkeeping 7. Conjecture 1 (DV3, provable).

## Numbered findings (summary of the referee's text)

1. F14 parts 1, 2: fine; Lean matches.
2. F14 Comment: true of the function, not of the paper's errors (eps_br, eps_w, eps_ro, eps_en, eps_cl, TV6's
   decomposition use orthogonal projections; only O(d)-invariant). Non-direct families have finer GL invariants.
   Extend the basis to R^d.
3. TV9 parts 1, 2: fine. Replace the garbled sentence "Any r' in D perp gives the same value as its own form" with:
   r' . z_l = (Pi r') . z_l.
4. TV9 part 4: correct (no Lipschitz needed). Scope: if the z_l(x) are affinely independent (generic when
   \|P\| <= d - dim D + 1), every function on P is affine in z_l: interpolation. In the norm form the premise depends on
   eps'_l.
5. TV9 parts 3, 4: class gap. State eps'_l >= eps_l and Lipschitz sigma_l.
6. TV9.1: fine. "Large weights" wording.
7. TV9.2: fine; state for every network whose first write is a function of Pi e(x).
8. TV10: fine. Comment: "the class of functions is fixed by D; its certificates are not". Lean: "with A replaced by J".
9. TV11: sharp form; Comment's condition is D cap (sum of kept areas) = 0, not pairwise; vacuous under laminar DM11.
10. TV12: correct; B^+ is not a DV2 cut. Amend DV2 or keep R-normconst as a node about DV2. Lower bounds TV7, TV9.2
    survive (the battery is constant).
11. TV12 cost: "at most one"; no room when every k_S = 0; sandwich V_N(G+) <= V_area(G) <= V_N(G); TV3 part 3's
    "one direction" is two in N's-constants units; folded form: RMSNorm of width w folds to w eps_0.
12. DM14 / R-cutclaim: own parts; the cut moved into which claims the map holds. Define completeness for area maps;
    until then R-cutclaim is parked, not moot. Stale text in DECISIONS.md.
13. TV6.1 Comment: D must be the dropped part of a cut at G; budget k + 1 in N's-constants units.
14. TV4 part 1: write "holds V_S as an area and the readout claim (S, V_S)"; keep the map route as the trunk route.
15. TV13: fine; add Lipschitz for \|s\| <= Lip; it is true of any coordinatewise nonlinearity; drop the metaphor.
16. DV3: (a) pointwise remainder needs pointwise statement; (b) c = c(N, \|R'\|, \|U'\|); (c) trained-network smallness
    is a measurement; (d) prove it: exact telescoping plus Taylor; (e) canonical reads for a claim error; (f) sigma'
    Lipschitz needs a node.
17. DM11 note: the widening that matches: any two areas nested or meeting only in 0, and for each area (and the top)
    the sum of its children is direct. Card 3-25's cons: every claim error and TV2/TV6 must be redefined with oblique
    projections.
18. Paths: TV9 (Lipschitz, eps'), TV12 (widened DV2), TV13 (Lipschitz), DV3 (sigma' Lipschitz). Card 3-23 "moot" for
    R-normconst and R-cutclaim: moved, not removed.
19. Literature: fine.
20. Lean: builds; `ols_le` weaker than the item; `gl_read` automorphism.

## Three most important fixes (referee's ranking)

1. Define completeness for area maps; DM14 "members of the decomposition"; fix stale ledger text.
2. Make TV12's build class honest (amend DV2 or keep R-normconst), "at most one", the sandwich, eps'_l >= eps_l in TV9,
   the folded-form note.
3. Restate DV3 pointwise and prove it as a Proposition; canonical reads; sigma' Lipschitz on a node; trained-network
   smallness to Part C.

## Session response (2026-10-29)

| Finding | Fix |
|---|---|
| 1 (completeness) | Definition DM15 (an area map complete for a zoom); DM14 says "members of the decomposition"; Corollary TV6.2 gives T5's two promises for complete area maps; R-cutclaim stays DROPPED now that completeness is defined |
| 2 (TV12 class) | DV2 amended: a cut may hold one constant direction outside the groups (equivalently re-set constants); node R-normconst kept as a decision about DV2 (DECIDED by Shonu's delegation); "at most one"; the sandwich; folded-form note; TV9 asks eps'_l >= eps_l |
| 3 (DV3) | Proposition TV14 (proved, no-norm form, pointwise remainder from N and B's reads); node S-smoothact (PENDING) for sigma' Lipschitz; Conjecture DV3 = TV14 in the norm form, with a route |
| 2, 8, 9, 17 | F14, TV10, TV11 comments and the DM11 note restated (function class vs certificates; sharp TV11; children's sums direct) |
| 4 to 7, 14, 15 | TV9 scope sentence and premises, TV9.1 wording, TV9.2 strong form, TV4 sentence, TV13 premise and metaphor |
| 18 | paths and the S-purestream text updated |
