# Referee report 25: Part B.W, Part D.13, H-group and H-recur (2026-10-27)

Referee: math-referee agent (opus). Scope: Definitions D50, D51; Theorems T54, T55, T58; Proposition T57, T59;
Corollary T59.1; Remarks B.W.1 to B.W.3; Conjectures D52, D53; Hypotheses H-group, H-recur; their Part C rows.
Lean: `lake build` passes; `Coarse/Zoom.lean` has only line-length warnings; no `sorry`.

Verdict: the B.W proofs are sound within their stated premises. H-group, D52/H-recur and D53 were vacuous or false as
written.

| # | Kind | Finding | Fix made |
|---|---|---|---|
| 1 | ERROR | H-group vacuous: one group = whole layer gives V_b = V_a under full rank; the T34/T35 claims are false under that quantifier | restated over the grouping into N's families with sizes k_F >= 0; comment: trivial when a family spans the input (T54.3), content in low-rank families, fails on T25; T34 dropped; Part C row and R-groupopt row restated |
| 2 | ERROR | D53 refuted by T20 (least unit fires on {2, 3}, not the union); R-symN hidden | D53 restated: a least OR step takes one member's region (symmetry broken), fit of T38.1 there; "a Gamma-invariant mix" in the tag, R-symN on the path |
| 3 | ERROR | D52 / H-recur true with m = 0; "few" undefined | G_0 = no groups; every step changes one group with exactly two children and target size 1 |
| 4 | GAP | T59's exact merge (a > 0, gamma = 0) narrower than class (b) | two notions defined (exact, strict exact); part 2 strengthened to any a, gamma (monotone phi); T59.1 strengthened to "no (b) build at G has D_P = 0" |
| 5 | GAP | good pairs merging exactly was numeric only | new Corollary T59.2 with the referee's unit c'(z_u + z_v) + 2 - c'(1 - 2r), c' >= 4/eta, full case list |
| 6 | GAP | D50: groups may dissolve; kept units joining groups widens T4 | containment condition added; the widening stated as a reading |
| 7 | GAP | "the least one"; reading pre-activations is a reading | "a least one", non-uniqueness noted; (b) stated as a reading of T4, with the reason (outputs would add a hidden layer) |
| 8 | GAP | T57 needs rational outputs and a relative version | both added to CLEAN-GROUP; polynomial map (m, beta) <-> (w, b) stated |
| 9 | GAP | T58 as an (r) step only when children are N's units | stated; otherwise T58 with p in place of z_S |
| 10 | GAP | T58 path lacks S-twoclass | added |
| 11 | GAP | T59's "one row alpha" premise | stated (two logits with S on top, or one reader); S-twoclass on the path |
| 12 | GAP | B.W.2: needs >= 2 children per step; D2 only | both added; D_P caveat stated |
| 13 | GAP | B.W.3 true only for l = L | restricted; layers above kept as N's for T58 |
| 14 | MINOR | B.W.1's evidence for H-recur is a tautology | removed from B.W.1, H-recur comment and card 3-11; "unlike an AND group read with a one-signed row" |
| 15 | NOT MATH | numeric check inside Part D | moved to the H-recur row of Part C; D52 stated for D_P and D2_P |
| 16 | MINOR | AND and OR overlap; T59 comment needs alpha >= 0 | both stated |
| 17 | MINOR | T31 cited for one group; T21 part 2 listing; pi_i > 0; T58 part 2 clause weaker | "as in T31"; T21 part 2 cited; positive weights in T54; "a basis of non-negative vectors" |
| 18 | GAP | card 3-11 overclaim and wrong citations; card 3-10 clique row unproved; section 2 lacks T4 results | card 3-11 rewritten (T57, T55, Remark B.W.2 as "not proved"); clique row marked "sketch only"; section 2 rows for T57, T58, T59 |

Content count (referee): structural results T57, T59.1 (now also T59.2) are facts about the detector network;
T54, T55, T58, T59 are bookkeeping true of any ReLU layer. No result in B.W says yet what the coarse network of a clean
N is.

## Round 2 (same day)

Correct and complete: D50 items 1, 2 and comment; T59 parts 1 to 3 and the two notions of exact merge; T59.1;
T59.2 (every case checked against D40 and Lemma T43a; with T59.1 it covers every pair of N's units); T58 part 2;
Remarks B.W.1, B.W.3.

| # | Kind | Finding | Fix made |
|---|---|---|---|
| 1 | ERROR | D53 refuted on the detector network (kappa = 2, m = 2, group {d_(1,1), d_(2,1)}, P = two vertex points and the origin: an exact merge fires on both members) | D53 restricted to the whole layer, D2, and mixes that hold a silent blend for every two members; the counterexample recorded in the route |
| 2 | ERROR | H-recur and D52 "every target" false: two-child merges reach only groups of target size 1 | quantified over targets whose groups have at least two units and target size 1; other sizes open |
| 3 | GAP | eps_recur measured the top level only; objective unnamed | eps_recur = max over targets of min over zooms of max over levels; the toy row says "top level only, D2", level 1 automatic, level 2 not compared |
| 4 | GAP | H-group, H-recur: depth, logits, clean definition, attainment unstated | one hidden layer, two logits, D20 with families from D38, finite mix (T19) |
| 5 | GAP | T59.2 comment "free merges = clique" unproved | "suggests, but does not prove" |
| 6 | GAP | T38.1 is for width-1 builds | D53 restricted to the whole layer |
| 7 | GAP | s_0 undefined in B.W.2 | defined (children per step, target size 1) |
| 8 | MINOR | T58 comment overclaims a least step | "keeps B_j's function on P; costs no more than D_P(B_j); not shown least" |
| 9 | MINOR | W_S clash between T58 and D50 | T58 uses Omega_S |
| 10 | MINOR | T57: finite rational mix; D2 version | both stated |
| 11 | MINOR | "B_0 = N" is a choice | "take B_0 = N" |
| 12 | MINOR | eps_group depends on k | a list over k |
| 13 | MINOR | T59.2 kept weights | "kept with N's weight c" |
| 14 | MINOR | "real case" vs D38; card 3-10 cited T34 | comment agrees with D38 (relaxed families needed); card 3-10 cites T25 |

Round-2 fixes are not re-refereed.

## Round 3 (same day): general T59.1 to T59.3, Lemma T60, Corollary T60.1

Correct: T59.1 (any set with a bad pair), T59.2 (sigma <= 1 with equality exactly at members' vertex points; the case
split covers every point of P), T59.3, T60 part 1, T60.1 (disjoint replacements add; kept children are in their row
space; kept units can be copied). Remark B.W.3 agrees.

| # | Kind | Finding | Fix made |
|---|---|---|---|
| 1 | GAP | "H-recur holds on the detector networks" is proved only for T43's mixes (fails on other mixes on X: the midpoint of two members in different blocks has sigma = 1) | "with the mixes of Theorem T43" in the H-recur comment, T60.1's comment and the ledger row; the counterexample stated |
| 2 | GAP | T60 part 2's hypothesis ambiguous | restated for pre-activations on P; note: spanning needs independent rows and enough points |
| 3 | GAP | "forces a loss" and "cannot be undone" did not follow | new part 3 (a direction dropped on a child never returns at a later level), proved by induction; "forces a loss" removed, with a note that another build may reach the same D_P |
| 4 | MINOR | T60 part 1 last sentence unclear | restated (changed groups have the property; unchanged groups and kept units are B_j's) |
| 5 | MINOR | the copy rule needs one fixed c' | c' fixed for the whole zoom |
| 6 | MINOR | H-recur is existential | existence of zooms of the form stated |
| 7 | MINOR | T59.1 needs T43a part 5 | cited |
| 8 | MINOR | tags (under: none) vs the PENDING node R-cleannet | already on the paths of T59.1 to T60.1 (as for T43); no change |

Round-3 fixes are not re-refereed.

## Round 4 (same day): B.W.2 (D54, D55, Lemma T62a, Corollary T62b, Theorem T62)

No errors. Checked correct: the affine relations and the four-switch contradiction, the parity step with weak signs,
one switching unit per segment, the presence case analysis, both symmetries (block swap; per-block reflection rho,
m = 2), the use of (l1), and the agreement with Lean (dip_switch, dip_same_sign; lake build passes, no sorry).

| # | Kind | Finding | Fix made |
|---|---|---|---|
| 1 | GAP | the comment's "one neuron cannot fire on both members (T59)" was unproved in the joint build | new T62 part 4 (each unit fires at exactly one of its members: the referee's L_k argument plus a case for "neither"), so every unit is split |
| 2 | GAP | "T61's units are split dials" unproved | one line added (firing weight 3/4; split by part 4) |
| 3 | MINOR | "highest toward A" is wrong (unit 1 is 20 at B, 16 at A) | corrected in the paper, card 3-13 and LADDER: on at A, off at C, highest at B |
| 4 | MINOR | positive weights; T43a part 5; Lean scope; D55 citation and two logits; D54 non-empty P_j; min notation; title "zero-loss"; R-group on T62a/T62b paths | all fixed |

Round-4 fixes are not re-refereed.

## Round 5 (same day): Part B.X (D58 to D61, T63 to T66, Remark B.X.1, Telgarsky row)

Correct: T63 (kinks are zeros; representation; bracket = -K; Lean kink_green), T64 (jump; t*-identity), T65 (only the
three values on P are used), T65.1, T66 (count and induction), the comment's class (a) tau = 1 and class (b) tau_b = 2.

| # | Kind | Finding | Fix made |
|---|---|---|---|
| 1 | ERROR | D61's "exactly when" false (T61 instance: tau = 1, no exact width-1 build); least vs exact mixed | restated: for free families (H-freefam) the bound tau_b is reached, by presence units; R-freefam, R-circuit on its path |
| 2 | ERROR | T65 comment cited T43 for "no width-1 build exact" | cites T64 part 3 (values 2, 0, 2 at A, mid AB, B) |
| 3 | GAP | T66: "N's kind" cannot be both one and k layers; no S, no readers | stated as a plain depth separation (under: none, path N-onehidden); coarse-circuit reading in the comment |
| 4 | GAP | D60 not well-posed for layered networks | layered reading only, relay pairs counted as new units |
| 5 | GAP | D58 redefines D5's circuit | stated as a set of units whose paths are D5's chains |
| 6 | GAP | tau_b undefined | defined via span(z_S, 1) and the lines no kept unit switches across |
| 7 | GAP | tau called "a property of the circuit"; kinks in the profile unused | "depends only on the gap on P and P"; D59 renamed dip profile, kinks noted as unused by T65 |
| 8 | NOT MATH / GAP | Remark B.X.1 mixed analogies and consequences; "only two"; last sentence a thesis | rewritten: consequences (T59.1 via T64 part 3; T62 via T62b and its sign argument) vs analogies (AND, free); the thesis dropped, with the counterexample |
| 9 | MINOR | T64 needs a != 0; T63 zero wording (every kink is a strict sign change); intro overclaim on T59; T65a comment | all fixed |
| 10 | GAP | Telgarsky row: hypotheses not stated; 2015 is the finite version | row restated |
| 11 | NOT MATH | card 3-14 overclaimed | rewritten; recommendation changed to (a) for now |

Content count (referee): 0 structural results; T63, T64, T65a, T65, T65.1 are elementary; T66 a known separation proved
directly. Not re-refereed.
