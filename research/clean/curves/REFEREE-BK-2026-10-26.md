# Referee report 20: Part A.5, Part B.K and the Part 0 lens paragraph (2026-10-26)

Referee: math-referee agent (reads only the paper and the Lean files). Files read: COARSE-THEORY.md (Part 0, A.3,
A.4, A.5, B.T T19-T23, D20, D21, T32-T34, B.K), lean/coarse/Coarse/Clean.lean, DECISIONS.md (sections 1, 2, card 3-5,
nodes, paths).

## Bottom line
- The proofs are sound: every inequality of Lemma T43a and every step of the reduction in Theorem T43 re-checked;
  Lean builds, no `sorry`.
- The problems are in the definitions (D39) and in the claims of what T43 shows.

## Verified with no finding
- r = 1 - eta/8 in [3/4, 1); r^2 >= 1 - eta/2; r >= 1 - eta/2 > 1 - eta.
- Same-block apartness by Cauchy-Schwarz; cross-block apartness on X; values at vertex points and both kinds of
  midpoints; the points lie in X; D20 for S_kappa (orthogonal block permutations, common norm in reduced form,
  faithful for kappa >= 2); Q4 fails at layer 2.
- T43: colored blow-up (no loops makes (j, i), (j', i) bad); S_kappa-invariance of the mix; same objective as T22;
  sizes (1, ..., 1) give monotone-or-antitone gaps, so the midpoint step holds (`mono_mid_ge_min` takes "Monotone or
  Antitone"); B* lifts to depth L; ETH step; relative step.

## Findings
1. (gap, serious) D39 was not a property of N alone: Gamma and the family structure were free; the trivial group with
   singleton families makes Q1 hold for every N and Q2 empty. Fix: Gamma canonical in (N, X), families = its orbits.
2. (gap) Q4 had no content: a constant unit relu(0 x + 1) is a "gate" containing every region; T43 extends to
   Q4-satisfying networks at depth with constant gates. Fix: gate region a proper subset of X, or extend T43.
3. (gap, serious) Exact Q2 excludes broadly tuned families (a member on at a sibling's peak input), including the
   paper's own detector model (D9, H-gram) and the curve units. The essay sentences cited are about one neuron's two
   poles. Fix: graded Q2, or a peak form.
4. (gap) X carries T43a part 2: on a product domain (box) detectors of different blocks co-fire; D40's X is not a
   product. Say so.
5. (wording, repeated) "all of layer 1 as one family" and "a family in each block" contradict Q1 (families = orbits =
   same direction across blocks). Fix: state Q2, Q5 for the orbit families.
6. (gap) T*** never says "fast": it allows a one-time zoom with no cost bound. T43 shows the zoom cannot run in
   polynomial time (unless P = NP), and a closed form cannot be evaluated in polynomial time. Rephrase.
7. (gap) Overclaims: "no symmetry condition" (only S_kappa shown); Q4 fails at L >= 2; Part 0 "clean by those
   principles" only for L = 1; "depends on the mix, not on the network" should be "carried by the mix alone".
8. (gap) Comment (iii) and card 3-5: what is proved is about the set where the gap exceeds delta for builds with
   L(B) < Gamma + 1/16, not about the unit's firing set and not about least builds (existence at L >= 2 not shown).
9. (gap) Option (d) is live: eta is of order m^-4, c = 16/eta of order m^4; bounded sharpness, a box domain, or a
   non-empty Q4 each exclude D40 without excluding "detectors read by one positive reader". Also "N is exact on T43's
   mixes" is empty (D_P = 0 for the teacher on every mix).
10. (gap) Q3 depends on the reference logit for n_y >= 3. Fix: all pairs c != c'.
11. (wording) C5 overclaims T43. Fix: "in T43 the network is the same for every graph, so memorising is a property of
    (N, P)".
12. (wording) unbounded-X rule for the entries; comment (c) "different normals" should be "non-antiparallel".
13. (wording) C3 "restates" should be "motivated by".
14. (not math as sourced) Remark A.5.1 cited a script; add explicit co-firing points for T33 ((1/2, 1/2, 2)) and T34
    ((1, 1, 2)).
15. (wording) Remark A.5.2: "alternate" not literal (a midpoint's projection lies strictly between its endpoints');
    convexity must include the flat left tail (mu_0 = 0); name the co-firing point.
16. (small gap, path) T43's S_kappa-invariant mix premise not on its path (T32-T34 carry R-symN); T43 missing from
    DECISIONS.md section 2; the chosen X and Gamma are hidden choices under R-cleannet.

## Three most important fixes
1. Make clean a function of (N, X) (findings 1, 5).
2. Give Q4 content and relax exact Q2 (findings 2, 3).
3. State what T43 refutes under T*** and cut the overclaims (findings 6-9).
