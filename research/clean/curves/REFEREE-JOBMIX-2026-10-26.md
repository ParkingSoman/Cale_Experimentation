# Referee report: job-weighted mixes and read tests (2026-10-26, night)

Scope: Definition D48, Lemma T51, Corollary T51.1, Remark B.K.2, Definition D49, Lemma T52 in `COARSE-THEORY.md`.
Context read, not refereed again: D40, D43 to D45, T22 (proof), T43a, T43, T46d, T49, T50.
Also read: `model/t51_jobmix_listen.py` and its JSON (all checks pass), `DECISIONS.md` cards 3-8, 3-9 and the path rows.
Lean: no new item has a Lean name. `grep sorry` in `lean/coarse/Coarse/*.lean` finds nothing. `lake build` not run
(no new Lean).

## Verdicts

| Item | Verdict |
|---|---|
| D48 | Well defined. The gloss "(option (i) ...: the data N was built for)" is not a mathematical condition, and T51 does not satisfy it. |
| T51 | Proof correct. The facts it uses follow from the STATED parts 2, 5, 6 of T43a. Path is missing R-explaw. |
| T51.1 | Proof correct for the problem as stated. The title overclaims: it holds only when the base law is part of the input. |
| B.K.2 | Midpoint claim correct. The sentence on "units whose targets are single midpoints" has a false step and an unneeded hedge. |
| D49 | delta is well defined. gamma can be infinite under D43's base laws (first moment only). The conclusion sentence overclaims for delta. |
| T52 | Parts 1 and 2 correct. Part 3 is missing a premise (xi < 0 is false when the other inputs are never on). Title and comment overclaim. |

## Findings

### Lemma T52

1. **Error (T52 part 3).** The statement says xi(u) < 0. Lemma T46d gives xi < 0 only "whenever both have positive
   probability", that is, P(h_u > 0) > 0 AND P(R > 0) > 0 under nu_l. Part 3 does not state the second premise. If the
   other inputs are never on (R = 0 nu_l-a.s.), then xi(u) = Cov(1[h_u = 0], 1) = 0, not < 0. Fix: add "and the other
   inputs are on with positive probability under nu" to part 3. delta(u) = 1 and gamma(u) = 0 are correctly proved
   (T_u = W_ru because there is one reader; h_u > 0 mu_u-a.s.).
2. **OK (T52 part 1).** If W_ru > 0, then z_r >= R_r because h_u >= 0, so the term is in [0, W_ru]. If W_ru < 0, then
   z_r <= R_r, so the term is in [0, |W_ru|]. The sum is in [0, T_u]. Correct.
3. **OK (T52 part 2).** gamma = 0 follows from R'_r = 0 mu_u-a.s. alone; b_r <= 0 is used only for delta. With
   R_r = b_r <= 0 the second indicator is 0; if W_ru < 0, then z_r < 0 because h_u > 0 mu_u-a.s. Correct.
   Wording: the parenthetical "(the reader reads u alone, or its other inputs are off wherever u is on)" gives two
   sufficient conditions. It is not equivalent to R'_r = 0 (other inputs can be on and cancel). Write "for example".
4. **Overclaim (T52 title, D49 last sentence).** "Neither ... credits alternatives by its sign" is false for delta.
   Part 3 gives delta = 1, the largest value: delta gives full credit to alternatives. What is proved is different:
   delta >= 0 for every unit (part 1), so its sign never gives a negative verdict; gamma = 0 for alternatives (part 3).
   Proposed title: "the deciding read is never negative, and the co-active read of a reader of alternatives is 0".
5. **Overclaim (T52 part 1, last clause).** "It can separate networks by its size only." delta = 0 is possible (no
   reader is ever switched by u; the paper's own check J3 with b_r = +1 gives delta = 0). So the sign separates
   "u never decides a reader" from "u sometimes decides". The real claim is that random networks have delta > 0.
   That is not proved. State it as a conjecture, or drop the clause.
6. **Overclaim (T52 comment).** "Evidence that alternatives are built for one reader can only come from how they
   share the inputs." T52 rules out two tests, not every test. The weights are evidence too: two alternatives read
   with positive weight by one reader is a property of W, not of co-occurrence under nu. Write "for tests built from
   the votes at one input, ..." and drop "only".
7. **Wording.** The names delta and gamma clash with delta (threshold) and gamma (output bias) in the proof of T22,
   and with gamma (a permutation) in T43a and T43. B.K.2 uses T22's notation nearby. Rename (for example delta_dec,
   gamma_co).
8. **Lean (T52, T51).** "Lean: none." Part 1 is pure sign cases, and T51 is a finite identity. Rule 2 of the track
   asks for Lean here. Not an error in the mathematics.

### Definition D49

9. **Gap (well-definedness of gamma).** gamma(u) = E_{mu_u}[sum_r sign(W_ru) R'_r] = E_nu[h_u sum_r sign(W_ru) R'_r]
   / E_nu[h_u]. R'_r is a sum of terms W_rk h_k. So gamma needs E_nu[h_u h_k] finite. D43 asks only for a finite first
   moment. Two ReLU units of linear growth under a law with finite mean and infinite variance give gamma = +infinity
   or undefined. Fix: ask for a finite second moment in D49, or define gamma only when the expectation exists.
10. **OK with wording (T_u > 0).** For a live unit of a hidden layer l < L, T_u = sum_r |W_ru|. The canonical form of
    D43 deletes every unit whose outgoing weights are all 0, so T_u > 0. D49 does not say this. Add "computed on the
    canonical form, so T_u > 0", or copy D45's convention "0 when T_u = 0".
11. **Wording (scale of gamma).** delta lies in [0, 1] (T52 part 1). gamma is not divided by T_u and is not bounded.
    The two tests are not on the same scale. State it.
12. **Path note.** D49 and T52 restrict to dense layers. D45 item 2' allows 1 x 1 convolutions too. The narrower class
    is not recorded at R-listen. Minor.

### Definition D48

13. **Gap (the gloss on the base law).** D48 says nu is "a finite list of inputs with weights (option (i) of node
    R-explaw: the data N was built for)". The formal premise is only "a finite list". "The data N was built for" is
    not a condition that a proof can check. T51 takes nu = P, and N^L depends only on kappa, m and L, so N^L was not
    built for P in any sense that is a function of N. Fix: separate the formal premise (nu is any finite list) from
    the reading (nu should be N's own data law), and say which one each result uses.
14. **Wording (D48 comment).** "Targets' weights match nu's" is vague. The exact condition, when at most one unit of
    layer l is on at each atom of nu, is: each h_j is constant on the atoms of nu where it is on. Then
    nu = sum_j nu(h_j > 0) mu_j + nu(silent) mu_0. Say this.
15. **OK.** "On all of the silent part, layer l gives one activation pattern" is true, and more: every layer above l is
    then constant, so N's output is constant on the silent part. The comment could say this.

### Lemma T51

16. **OK (the two activation facts).** At the vertex point (j, i) only d_(j,i) is on (T43a part 2, parenthetical). At
    every midpoint of two distinct vertex points every layer-1 unit is 0: T43a part 5 gives logit difference = c times
    the sum of the layer-1 units, part 6 gives 0 there, and the units are >= 0. (In the proof: a pair unit at a
    same-block midpoint is <= 1 - eta/2 - r = -3 eta/8 < 0.) So the citation "the proof of T43a part 6" can cite the
    stated parts 2, 5 and 6 instead. Under nu = P each target is the point mass at its vertex point, all kappa m units
    are live (weight 1/Z > 0 each), and the silent part is exactly the set of bad-pair midpoints. The midpoints are
    distinct from each other and from the vertex points (proof of T22), so each carries weight M. The identity holds;
    the exact check J1 confirms it in four instances.
17. **Gap (minor, a_0).** mu_0 needs n_bad >= 1. This holds because m >= 2 (D40): each color has a same-color pair.
    Say so.
18. **Path (hidden restriction).** T51 fixes a base law (nu = P). That is a choice at node R-explaw. T49 lists
    R-explaw on its path; T51's path (R-cleannet, R-jobmix) does not. D48 itself names option (i) of R-explaw. Add
    R-explaw to the paths of T51 and T51.1.

### Corollary T51.1

19. **OK (the reduction).** P is invariant under S_kappa (proof of T43). Z = kappa m + M n_bad, and M is polynomial in
    N_V = kappa m, so a has kappa m + 1 rationals of polynomial bit length. P_a = P, and N's logits on the points do
    not change, so every build has the same objective. The points of P_a are the points of P, so the relative version
    (n = number of points) transfers. d = 2 kappa and S stays polynomial in m, so part 2 transfers. Parts 1 to 3 hold.
20. **Overclaim (scope).** In JOB-TRUNK the base law nu is part of the input. So the instance chooses every single
    input through nu, which is what D48 meant to take away from the user. The corollary shows: "D48 with a base law
    supplied by the instance does not remove the hardness." It says nothing about D48 with a base law fixed by N
    (option (i) or (ii) of R-explaw as written). In that reading the T43 reduction carries no graph: N^L depends only
    on (kappa, m, L), so nu(N^L) cannot depend on the graph, and a depends on the graph only through n_bad (T51's own
    last sentence). Fix: put "when the base law is part of the input" in the title and statement, and add one
    sentence: if nu is a function of N, the reduction of T43 gives nothing, and hardness is open there.
21. **Wording.** JOB-TRUNK is "CLEAN-TRUNK with the mix given as a pair", so P_a must also be S_kappa-invariant. The
    statement puts invariance on nu only. Say that P_a (or a) must be invariant too; the hard instance satisfies it
    (all a_(j,i) = 1/Z). Also say how the entries of a are matched to the units (for example by the index (j, i)).

### Remark B.K.2

22. **OK (the midpoint step).** For sizes (1, ..., 1), T43's proof shows the gap is phi(w^T x + b) with phi monotone or
    antitone. So the gap is monotone along every segment, and its value at a midpoint is at least the smaller of the
    two end values. N's logit difference is 0 at the midpoint of ANY two distinct vertex points (T43a part 6 is not
    limited to bad pairs), so N does not depend on which midpoints nu holds. Correct.
    Wording: "two vertex points it serves" is not defined. Write "two vertex points where its gap is above delta".
    "Width 1" means sizes (1, ..., 1) at L >= 2.
23. **Overclaim ("two places").** "Two places for such a pattern are open" reads as a complete list. No argument shows
    it is complete (for example, a base law in which the midpoints are partly in units' targets, with small
    activations, is neither). Write "two places are ..., among others" or prove the dichotomy.
24. **Error ("its centred read must vanish there").** The premise "may not change N's logits at its midpoint" is a
    choice of reduction, not a consequence: a different network with other labels at midpoints is a different
    instance. And the step does not follow: another unit that is on at the same midpoint can cancel the contribution,
    so the centred read of each unit need not be 0. Also the centred read W~_cu is one number per class, not a function
    of the input, so "vanish there" is not defined; S_u at the midpoint is 0 automatically (q_1 = 1/2), which says
    nothing about the unit.
25. **Gap (the hedge is decidable).** "Whether such units can be read at all (lambda > 0 of D45) is not shown." For a
    two-class network and u in the last hidden layer, S_u = (W_out[1,u] - W_out[2,u]) (q_1 - 1/2) and
    T_u = |W_out[1,u] - W_out[2,u]|. If mu_u is the point mass at x0, then
    lambda(u) = s_u (q_1(x0) - E_nu[q_1]), with s_u the sign of W_out[1,u] - W_out[2,u].
    With no cancelling unit, the read must be 0, so T_u = 0 and D45 sets lambda(u) = 0: u is not read. With
    cancelling units on at x0 whose targets are also the point mass at x0, they have both signs s, so at least one has
    lambda <= 0 (all of them, if E_nu[q_1] = q_1(x0)). For T43's P, E_nu[q_1] > 1/2 = q_1(x0). This is a three-line
    lemma; state it in place of the hedge.

### Tags, paths, content

26. **Tags.** Every new result is "(under: none)". No proof uses a named hypothesis. OK.
27. **Paths.** T51 and T51.1 miss R-explaw (finding 18). T52's path (R-cleannet, R-listen) matches its premises
    (D45 item 2', T46d). The dense-only restriction of D49 is not recorded (finding 12).
28. **Content.** Structural results among the new items: 0. Bookkeeping: 3 (T51 is an identity on a finite list; T51.1
    composes two reductions; T52 is sign cases true of any ReLU reader). Two definitions and one remark. The new
    material records two negative facts (D48 alone does not stop the T43 reduction; two read tests do not separate
    alternatives). It says nothing new about what a clean network is.
29. **Literature.** No borrowed result and no sparse-direction framing in the new items. OK.

## The three most important fixes, ranked

1. **T52 part 3:** add the premise "the other inputs are on with positive probability under nu". Without it, xi = 0
   and the claim xi < 0 is false (finding 1).
2. **T51.1 and D48:** state that the base law is part of the input, and say the corollary is silent when nu is fixed
   by N (option (i) or (ii) of R-explaw). Separate D48's formal premise from its gloss. Add R-explaw to the paths of
   T51 and T51.1 (findings 13, 18, 20).
3. **B.K.2:** replace the "second kind" sentences with the lemma of finding 25:
   lambda(u) = s_u (q_1(x0) - E_nu[q_1]) for a last-layer unit of a two-class network whose target is a point mass.
   Drop "its centred read must vanish" and "two places" as a complete list (findings 23 to 25). Also retitle T52 so it
   does not say delta fails to credit alternatives (finding 4).
