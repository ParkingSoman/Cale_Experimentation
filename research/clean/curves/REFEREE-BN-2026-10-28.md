# Referee report: B.N (size), the DE2 item 4 change, DN1, DN2, H-mass, Scale.lean (2026-10-28, round 1)

Referee: `math-referee` agent (reads only the paper, the ledger and the Lean files). `lake build` passed, no `sorry`.

## Findings (as reported)

1. ERROR. TN3 part 3: D_M <= M phi is false when a dead-moved unit has osc(W_N[., u]) > 0 (one judgment unit, 0 on P:
   M = phi = 0, D_M = p(R) osc(W) / |a_k| > 0). The proof changed phi's definition halfway. Fix: add D_dead(N), or
   assume no dead-moved unit.
2. ERROR. DN1, last sentence: "M' is complete when M is" fails through DM6 item 5, which counts |S|: with
   |S| <= k_S < |S| + n - 1 the copy needs a new blend claim, whose error can be positive. DM6 item 5 itself counts units.
3. GAP. TN1 part 4 (E): needs (N', M', G') to meet TE4's premises (unstated); tag and path lack S-cover, S-regiongroup.
   Fix: split into its own statement tagged as TE4, with the premise carry-over.
4. GAP. TN4: uses a hypothesis with no node (H-scale's quiet form is about one map rule and Lambda; TN4 needs maps
   meeting TE4's premises, Lambda°, and phi_K, sigma of the assemblies). Fixes: use Lambda in place of Lambda°; restate
   H-scale for such maps; replace "phi_K, sigma do not grow" by "the largest r_j does not grow" (on N alone); name the
   E_dead = 0 premise. Also: the A.7 paragraph says TN3 uses the quiet form (TN4 does); DECISIONS row R-mass says no
   theorem uses it.
5. GAP. TN2 part 4, "at most n - 1": each term <= 1 gives only n - eps_u; missing step: eps_u is an A-weighted mean of
   the u_i terms.
6. (Question answered.) TN2's same-sign claim holds under the premises; "has no flat claim" and the share-sum premise
   in M' are redundant; the zero-share-sum case breaks it.
7. (Question answered.) TN1's lift and merge are right for V_rob, V_b, L_S, L^0_S, including boxes that read z; make
   explicit that the judgment shift is right because M' gives every u_i the weight a_u.
8. TN1 part 5 is right but leaves out: the flat-error list gets n entries; a new blend claim can be needed; family
   claims are dropped.
9. ERROR (MINOR). DN2's rescaling list: A_u r_u = d_u scales by c.
10. GAP. Lean lines: all five lemmas are one-line scalar steps; `merge_osc` gives entry bounds, not the osc step;
    `copy_mass` is not TN2's step; `mass_mean` is not TN4's step; `between_sum` covers one order.
11. GAP. TN4 comment item 3 (ResNet): layer4 output is relu(bn3(conv3) + skip), outside D50'; only phi, O, D_M,
    Lambda_M are defined there; "pooled value is a split" is an analogy needing three premises.
12. NOT MATH. "A result proved and checked on a small network is a result for a large network" mixes a universal proof
    with a numerical check (also in card 3-21); TM3 depends on bit length.
13. MINOR. TN3 part 1 is an equality; conventions for zero masses; TN2's "optimum can get better by TE1" is about V_b
    only; H-mass part (2) needs "there exist e-near assemblies".
14. Content: 0 structural results, 4 bookkeeping. The real finding is negative and about the definitions: eps_dist,
    eps_shel and DM6 item 5 count units.
15. Literature: no sparse-direction framing.

Ranked fixes: (1) redefine eps_dist and eps_shel as the largest member's term (copy- and F1-invariant, 0 exactly when
the claim holds, polynomial), treat flat claims the same way, make DM6 item 5 copy-invariant (count classes of
positively proportional members); (2) restate and re-tag TN4; (3) fix the two false statements and the two missing steps.

Verdict (referee): the DE2 change keeps TE3, TE4 and TE5. B.N's proofs are correct except for two false statements (TN3
part 3, DN1's completeness sentence) and the mis-tagged TN4. All four results are bookkeeping; the real content is
that the district and shelter errors and DM6 item 5 count units and should be replaced.
