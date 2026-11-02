# Referee report: Part B.L (round 1, 2026-10-29)

Agent: math-referee. Read the paper and the Lean files only. Items: Remark B.L.1, Theorem TL1, Definition DL1,
Theorem TL2, Corollary TL2.1; Lean `Coarse/Netread.lean`. `lake build` passes; no `sorry`.

## Answers to the five questions

1. TL1 part 1: the new unit is an allowed (r) step (read linear in the rows of T(S'), reads only the children's
   pre-activations and the z of W'). TP1 part 2's computation needs only: units on everywhere, read on T_w equal to
   q^T B_w ztil plus F_w terms, shelter claims, V = Q orthonormal in the image of P_perp; the TP1 type gives all four;
   Pi_w is the same at both levels (a sibling merge keeps J_w). The induction is sound only as an existence statement
   over a chain of eigenvector choices (finding 2).
2. TL1 part 2(b): the eigenvector argument is right (a PSD sum with orthogonal images is a direct sum); gaps in
   findings 3, 4, 9.
3. TL2: kinds (A) to (D) cover every unit under premises 1 and 2 (checked layer by layer). The constant off the area is
   handled correctly (shelter on T(S) from G_j, on D from G_(j+1), district claims, judgment indicator, copies of J'_u).
   Two citations wrong (finding 8).
4. Claim levels: every claim is supplied where it is used; one redundant clause (finding 6).
5. False statements: findings 1, 3, 4; quantifier error: finding 2; premise off the path: finding 11.

## Findings

1. [structural, false statement] Remark B.L.1 item 1, "this is the only way a merge step can lose": TL1 is sufficient
   only, from a TP1-type B_j. (K) also fails with no reuse when the merge grows (one child, k' > k_1, rank > k_1).
   Overview and card also drop the TP1-type premise ("kept" should be "box").
2. [structural, quantifier] TL1 part 1 last sentence and part 3: (K) depends on the directions chosen at earlier
   steps; with ties, the choice changes (K) at the next step. Part 3 "from summaries alone" false without the chosen
   directions. B_0 must be of TP1 type; "there is an (r) sequence". Fix: existence over choices, or a gap premise.
3. [structural, ill-posed] (K) and TP1 type for k' or k_c > n_y - 1: k' orthonormal vectors in an (n_y - 1)-space do not
   exist; 2(a) false for k' >= n_y. Fix: TP1's zero padding.
4. [structural, proof gap] TL1 part 2(b) re-chooses B_j's directions. The case k' <= k_c holds by the invariant-subspace
   argument for any B_j. ker Sigma' also needs ker Sigma_(W').
5. [bookkeeping] TP1 type asks an exact scale; allow c_t > 0, V's columns q_t / c_t.
6. [bookkeeping, redundant] TL2 premise 3's judgment clause is given by TH1 premise 2 at G_(j+1) (K_u = J'_u forced).
7. [bookkeeping] TL2's Comment, Remark item 2 and card 3-27 overstate ("at equal size", "never worse"); premise 4 is
   k_(S') >= n_j(S'); "finer class contained, on P, in coarser class".
8. [bookkeeping, citation] TL2 proof cites DH3 item 4 for kept units of S'; the reason is the branch claims of S' at
   G_(j+1). Copy units of the replaced boxes should be listed.
9. [bookkeeping] TP1 part 2's computation does not use eps_lin = 0; part 1 holds at any eps_lin with
   D2_P(B_(j+1)) = Lin(G_(j+1)), within (1/2) eps_lin of least: a graded statement for free.
10. [bookkeeping] TL2.1: "after layer 3" should be "after layer 2"; premise 2 of step 2 not checked; V_b(G_2) <= V_b(G_1)
    is new (in fact equal); Remark item 2 and card omit a_2.
11. [bookkeeping, path] TP1 type narrows the (r) sequence: state as existence or name it. TL1 part 4 uses TH3's
    three-logit instance: add N-ny3 to TL1's path.
12. [bookkeeping, Lean] The Lean lemmas are scalar forms; `psd_parts_zero` unused in the written proof. Label them.
13. [content] TL2 is a rewriting identity (relu(z) = z for z >= 0; an indicator fixes a function constant off an
    area); it shows the box class grows with depth, a property of DH3's box definition, not of N. Count: 1 structural
    result (TL1 part 1) against 5 bookkeeping items.

Ranked fixes: (1) TL1 quantifiers (findings 2, 4); (2) remove the false converse (finding 1); (3) make (K) well posed
for k' >= n_y (finding 3).
