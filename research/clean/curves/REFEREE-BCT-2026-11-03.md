# Referee report: B.CT, the channel graph (2026-11-03)

One read by the `math-referee` agent, of the first version of the section. The referee ran nothing and checked examples by hand. I then fixed the section, and the fixed version has my check only (`model/tc_tree.py`, 0 fails).

| # | Referee's finding | What I did |
|---|---|---|
| 1 | TC2 is false at V_c = 0: z_c = u - v, u = v in {0, 2}, K_1 = {u}. Then V_c eps_par = 0, delta_1 = 0, and c moves. The check missed it because it computed e, never eps | TC2 now uses the sum of e(c, K_l). The corner is in the check |
| 2 | "Open" and the TM4 citation in TC2's comment are wrong. Parent claims do not escape for N_K: q >= 10, w in {0, 2}, c = q + d(w - 1), c' = q, z_e = (c - c')/d + 5, y = (h_e, 0), K without w. All errors tend to 0, D_P(N_K) stays fixed. Only bounded weights or clean instances stay open. V_c also carries N's norms. "Graded form" overclaims, since TC2 stops at delta_L | New Proposition TC3, with q taken in {10, 12} so V_c stays at least 1. Name of TC2 changed. The comment now says these things |
| 3 | Unstated premise: the activation acts on each channel alone. "Any function" in DC1 was too wide. TC2 also assumes 1-Lipschitz | Both are now stated |
| 4 | TC1 is close to a tautology. Part 1 unrolls the premise, part 2 is linearity, part 3 is borrowed. Part 4 needs different claims. The content: one-layer tests at N's values certify the whole cut. Stronger forms: (a) an iff with value-level claims, (b) with kept units frozen and logits free, a convex fit on K_L's values, (c) on plain dense layers, e = A_cD^T Cov(h_D) A_cD. No result uses the tree | A comment in the paper says this plainly. Part 4 fixed. (c) is new Lemma TC4. (a) and (b) are not taken up |
| 5 | DC2 meets DM1 by the letter. In spirit Q is the build's kept set. (c, whole layer) has error 0 always, so DC3's edges can be empty. A signed mean change is 0 on P for affine reads. "A map made complete" had no definition. The copy test holds only for affine reads | DC2 says the spirit point. DC3 needs p to be needed in Q. Strength is now root mean square. "Closed on a cut" is defined, and DM6 is not changed. The copy test is scoped |
| 6 | Residual layers are a scope matter: inner channels cannot be cut, so "smaller" is unproved. Absorbing a fill into a bias needs an affine read. The script tests plain layers only | A scope comment is in the paper. Whether Step 6's build is smaller depends on removing inner channels too |
| 7 | ONE-BUILD is right. "What a cut costs below the top is exactly the errors" is too strong at eps > 0 | Sentence replaced |
| 8 | The example z = q + |u - v| works. It is a residual block with q on the skip, not one plain layer. z = q + u - v makes the same point there | Left as written |

Open after this report: the strength measured in `score_acts.py` is a signed mean fall on the child's top photos, which is the agents' `acts` test and not DC3's strength. Part C says which is which.
