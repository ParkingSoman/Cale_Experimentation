# Referee report: UNIT-BRANCH.md and Coarse/Unit.lean (2026-10-27)

Lean: `lake build Coarse.Unit` passes, with no `sorry`.
Checked and correct: TU2; TU3's Hessian bound 1/2 (sharp) and the integral form of the second difference; TU5 at depth; TU6's numbers; TU7's derivative; TU9; TU10 once z_u is injective; the |x| instance (I = -5/4).

## Findings

1. **Gap. TU1(1) and DU2: layer placement.** "A choice of layers exists" is false.
   - Counterexample: S = {u1 in layer 1, w in layer 3}, with no path u1 -> w, so S is convex. Let j in layer 2 feed w, and let r in layer 2 read u1. A box unit that reads j and is read by r has no layer.
   - The same failure occurs for a one-layer S whenever box units chain. In the last hidden layer, chaining adds hidden layers, which D50's comment forbids.
   - The build is acyclic (a cycle would give a path S -> r -> ... -> j -> S), but it is not layered in the sense of A.0.
   - **Fix:** say "feedforward ReLU network" in U1 and TU1, and prove acyclicity with that argument. Alternatively, give each box unit a layer, and let it read only lower layers and be read only by higher layers. Delete "the layer range of S's own members".

2. **Gap. TU1(2).** "Its value depends on S's members" is a statement about the graph only. The real failure is a cycle: box -> r -> ... -> v -> box. **Fix:** state that some build has a directed cycle.

3. **Gap. TU3, the consequence.** The upper bound uses two eps-near single builds, so the slack is 2 eps. **Fix:** write "+ 2 eps". Say that |.| is the Euclidean norm.

4. **Gap. TU4(3), the left-inverse step.** Multiplying F C_S Sigma = 0 on the left by F^(1/2) gives F^(3/2) C_S Sigma = 0. **Fix:** from Sigma^T C_S^T F C_S Sigma = 0 you get |F^(1/2) C_S Sigma|^2 = 0. Then apply the left inverse.

5. **Overclaim. TU4(3), its comment, and Part 0 item 2.** Full column rank of F^(1/2) C_S forces |S| <= rank F <= n_y. So the "degenerate case" is the generic case for every larger set. What is proved is one-way: Sigma_ST = 0 implies U4, and U4 implies F C_S Sigma_ST = 0 = F C_T Sigma_TS. **Fix:** state these two results, not "exactly when". Replace "no two members' writes are dependent" with "the writes are linearly independent".

6. **Wording. TU4 setting.** A constant F is not "the form of D_P near N": the Hessian diag(s_x) - s_x s_x^T changes with x. The free logit constant is not part of DU2. **Fix:** list both as model choices in the tag. Move the deep-linear claim ("a block of co-varying activity across layers") to Part D as a conjecture.

7. **Wording. TU6.**
   - Part 1 cites "T61's table" for the signs of B and D. That table lists the build units' pre-activations. **Fix:** compute from D40 (r = 39/40). At A: z_B = 4/5 - r, z_D = -r. At C: z_B = -r, z_D = 4/5 - r. At midAC: z_B = z_D = 2/5 - r. All are < 0.
   - Part 4: the supports are {vertex A, vertex C} and {vertex B, vertex D}.
   - Define D2* and the D2 form of I.
   - Part 4 uses T29: state its premises (Z_S contains 0, and the domain is R^4).
   - The joint build at budgets (1, 2) is legitimate: relu(1) is class (b) with m = 0. Say that the interaction shown needs unequal budgets and a constant, non-covering unit.

8. **Gap. DU3, last sentence.** For S at layer >= 2, the in-boundary values are >= 0. On attainable values, T29 does not apply. **Fix:** quantify over all of R^(in-boundary), and add T29's premise that Z_S is non-empty.

9. **Gap. TU10's hypothesis.** "Takes different values at different points" can be read as "not constant". The spanning step needs z_u injective on P. **Fix:** write "separates the points of P".

10. **Gap. The converse premises (TU7, TU8(2), Part 0 item 3).** "Cancellation" is undefined. Co-firing alone does not give gamma_v != 0: you also need g_x != 0 at a co-firing point, that is, removing S changes the softmax there. **Fix:** define "no cancellation" as: gamma_v != 0 whenever h_v(x) g_x != 0 for some x. State the converse as closure under co-firing at the points where S matters. In Part 0, drop "as soon as they do fire together".

11. **Gap. TU8, "finest unit system".**
    - The components are taken over the whole layer. A family that covers only some units can be finer: if u and v are joined only through a unit outside the family, TU5 already gives U4 for {u} and {v}.
    - **Fix:** require the family to partition the layer.
    - Dead units are singleton components, so under H-district the components are the districts minus their dead members.

12. **Overclaim. H-district and "the components are districts".** The essay's gate is "in front of" the district (upstream). H-district puts it inside the last layer as a co-firing member, and the identification of components with districts relies on that. **Fix:** rename the hypothesis "a co-firing hub per district". Alternatively, state that an upstream gate does not by itself join last-layer members.

13. **Gap and overclaim. TU11(1).**
    - Its tag omits "free boxes of class (b)" and "a least box exists".
    - "True for generic weights" fails at depth >= 2: inputs whose layer L-1 is all zero give equal z_u, and this happens on an open set of weights.
    - Both orders are used, so every set must be exact. "Except at most one" is loose.
    - D* is non-increasing in k, so "exact at every budget" means "removable at budget 0".
    - "The whole layer is the only unit system" and "every level of a zoom must be optimised as a whole" do not follow: any single set is a unit system, and TU10 says nothing about chains of levels.
    - **Fix:** state only "a family of two or more sets meets U4 only if each set is removable at no cost". Restrict "generic" to one hidden layer.

14. **Gap. TU9 and Remarks B.1, B.2.**
    - TU9's union part needs S ∪ T to be convex.
    - B.1's lower units join a district only if S stays convex and unlinked. That is a condition on wiring, not on firing.
    - B.2 ("negative pre-activations can only keep a unit off") is proved only for folds (m >= 0, T29). Restrict it to that case.

15. **Overclaim. Remark B.3.** TU5 shows that "covering on P" suffices (with disjoint supports). TU10 shows that all of class (b) fails. Neither shows that independence "asks for" covering on P. **Fix:** write "suffices".

16. **Error. D.1, row T59.2.** T59.2's presence unit has offset c'(|J| r - 1) + 2, which is > 0 when |J| >= 2. So tau < 0, and the unit is NOT covering in the sense of DU3. T60.1's comment shows it firing while its members are off. **Fix:** say "covering on P".

17. **Overclaim. D.1, other rows.**
    - T32 to T35: there is one family and no neighbour. The free gain is a better fit inside the group (T32's centre-square unit), not TU10's help. Delete "price of independence".
    - T29: "asked for by independence" repeats finding 15.
    - T43: every member is its own component, so a multi-member group is a choice of unit system, not "inside a unit".
    - T61/T62: write "consistent with", not "explained by".

18. **Overclaim. D.2.**
    - R-circuit "answered" contradicts line 3 (PENDING). The depth converse is Conjecture DU1. "Whole network (free)" is at most the last layer. TU1 admits convex wiring circuits.
    - R-deep "goes away" is unproved, and "U4 fails anyway" is proved only for the last layer.
    - S-twoclass: TU4(3) needs |S| <= n_y.
    - **Fix:** change these verdicts to "partly informed", and cite the premises.

19. **Wording. Lean lines.**
    - `cross_zero_add` covers attained minima, one direction only. TU4(1) claims infimum additivity and "exactly".
    - `cover_compose` is propositional composition. It omits the shared in-boundary value and the union part.
    - Say both. Also list "convex, unlinked" in the TU5 comment.

## Three most important fixes
1. Findings 1 and 2: restate U1 and TU1 for feedforward builds with explicit layers. Every "at any depth" claim rests on this.
2. Findings 5, 10 and 13: state the converses (TU4(3), TU8(2), TU11(1)) with their real premises. Cut Part 0 and D.2 down to match.
3. Findings 12, 15, 16 and 17: weaken these claims to what is proved: the gate as an upstream unit, covering as forced, and T59.2 or T32 as instances of help.
