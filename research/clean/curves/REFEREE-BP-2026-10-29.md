# Referee report: Part B.P (Passing summaries up the tree), 2026-10-29

Scope: Definitions DP1, DP2; Theorem TP1; Theorem TP2; Proposition TP3; Remark (KL); the Comment after Proposition
TH3 (last two sentences); the "Status (2026-10-29)" paragraph of D.16. B.H (DH1 to DH3, TH0 to TH3) was read for
definitions only and is not re-refereed.

Files read: `COARSE-THEORY.md` (worktree `summary`), `lean/coarse/Coarse/Recur.lean`, `DECISIONS.md` sections 1, 2,
3-25, 5 and 6. Note: `DECISIONS.md` changed while I read it (the path row of TP1 moved from line 1550 to line 1588
between two reads). The rows I checked are those of the second read; their text was the same in both.

Lean: `lake build` passes (9069 jobs, only line-length warnings). `grep sorry` over `Coarse/*.lean`: none.
`Coarse.lean` imports `Coarse.Recur`.

## Summary table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| DP1 | stand-alone, T5 rules | OK, with GAP | Meets the four T5 rules (from N, P, claim; polynomial; 0 iff it holds; no build, size or zoom). Gap: J is tied to M's judgment claim or empty, while DH3 lets J_w be any set when the area is P (finding 9) |
| DP2 | stand-alone | OK | Sigma_w = B_w C_zz B_w^T is rational and PSD with image in im P_perp; E_w - tr Sigma_w = eps_lin uses the normal equations (one line missing, finding 10) |
| TP1 part 1 | proof | OK | Projection, "Pi_w g in Wim", split K_w = Sigma_w + R_w and Ky Fan are all correct. Uses none of TH1's premises |
| TP1 part 2 | proof | OK | Always-on box, shelter constancy of other tops, phi_w in F_w, free terms absorb Y - Ytil - phi_w: all correct |
| TP1 part 3 | quantifiers | GAP (minor) | Hypothesis gives linear-read claims for one group S; the sum over all S needs them for every group |
| TP1 | tag | GAP | Tag says "every claim it uses has eps = 0", but the linear-read claim may have any error; D2_P (R-obj, on the path) has no tag phrase |
| TP1 | content | structural (light) | Names the coarse optimum at eps_lin = 0, but the claim makes the problem linear; it is reduced-rank regression with a shared output space |
| TP2 part 1 | proof | OK, bookkeeping | Sums over disjoint node sets |
| TP2 part 2 | proof | GAP | "the build's weights from those and the B_w" is false: beta_t, U_w, t_w, v_S need more per-node data (finding 1) |
| TP2 part 3 | proof | OK, bookkeeping | Correct. One missing line: Sigma'_w <= Sigma_w puts the image of Sigma'_w in im P_perp |
| TP2 part 4 | proof | GAP (overclaim) | In the instance the values agree (both 3, and 3 is reached), but the paper proves only ">=", and the general reading "TH3's (r) step is a compressed summary" is not proved (finding 3) |
| TP2 part 5 | numbers | OK, with a remark | 1/8, 1/40, 1/10, 1/20 and 0 are right. The lower bound Lin - eps/2 is 0 at G_1 and G_2 (finding 4) |
| Two-logit note | proof | GAP | Lin_S = (1/2) eps_lin(S) whenever k_S >= 1, so the note is a tautology; it omits the W' term and needs k >= 1 for each child; it does not prove Route (2) (finding 5) |
| TP3 parts 1, 2 | cost claims | GAP | Counts are in exact real arithmetic with an exact SVD (unstated model); Lin_S and q_t are free of N, the build is not (findings 1, 7) |
| TP3 part 3 | proof | OK, bookkeeping | Nesting in k and "base + one term per unit" follow from TP1 part 2 and TH1's assembly |
| TP3 part 4 and Comment | trunk | GAP | A switch costs the size of the changed boxes and J_w weights, which grows with N's width; "meets that wording" holds only under a pointer-swap reading (finding 6) |
| Remark (KL) | proof | OK | D_P <= (1/2) Lin(G) follows from F11. No bound on the KL gap to the least |
| TH3 Comment edit | claim | GAP (minor) | "TP2 proves it" is true for Lin and q_t, for D2, and for sibling merges only |
| D.16 Status | claim | GAP | Route (1) as stated (premise: a least box is on throughout) is not what TP1 proves; Route (2) is not proved by the note (findings 2, 5) |
| Lean names | match | OK (shells) | `energy_split`, `trace_bound`, `topk_trunc`, `th3_summary` are scalar facts; the Lean lines say so. No matrix step is in Lean |
| Literature | framing | GAP (minor) | No sparse-direction framing. TP1 is reduced-rank regression (Izenman, in the table, hypothesis "nonsingular covariance"); not cited (finding 11) |

Content count for B.P: structural 1 (TP1, light). Bookkeeping 3 (TP2 parts 1 to 3, TP3, Remark KL; TP2 counted once).
Instance numbers 2 (TP2 parts 4 and 5).

## Findings

**1. The build of a level is not computed from the summaries and the B_w alone. (false statement)**
TP2 part 2 says "the build's weights [are computed] from those and the B_w". TP3 part 4 says the boxes of every level
come from "a one-time computation ... (parts 1 and 2)". The build of TP1 part 2 needs more than DP2 stores:
- beta_t must make m_t^T z_T(x) + beta_t > 0 at every x in P. m_t depends on q_t, and q_t depends on the level. So
  beta_t needs the range of the values B_w z_(T_w)(x) over P. That is data on P, not in s_w or B_w.
- U_w and t_w come from Y - Ytil - phi_w. This needs the F_w-coordinates of Y and of z_(T_w) - ztil on w's area.
- phi_w and the constant output v_S need the shelter constants zeta_(w') of the other tops.
- The box weights m_t = (B_w^T q_t) have |T(S)| entries, and U_w has |J_w| columns. Writing them costs time that grows
  with N's width.
So the B.P intro item 5 ("the cost of a level does not depend on N's width, depth or on P") and the TP3 Comment ("even
computing a new level from summaries ... does not touch N") are false for the build. They are true for Lin_S and the
directions q_t. Card 3-25 ("a merge costs at most about n_y^3, whatever the size of N") has the same overclaim.
Fix. Add to DP2 a per-node **build record**: B_w; the F_w-coordinates A_w of Y and G_w of B_w z_(T_w) on R_w; the
vector b_w = B_w zeta_w (the value of B_w z_(T_w) outside R_w, by the shelter claim); and the scalar
rho_w = max over x in P of |B_w z_(T_w)(x)|. Then beta_t = 1 + sum over w in W_S of rho_w works for every unit
vector q_t. Restate TP2 part 2 and TP3 part 2 in two parts. (a) Lin_S and q_t: from summaries, O(n_y^3) per merge,
free of N and P. (b) The build: from (a) and the build records, in O(k_S sum over w of n_y (|T_w| + |J_w| + 1))
operations, linear in the width of the group's top and judgment units, and free of P and of depth.

**2. D.16 Status: Route (1) as written is not what is proved; it can be proved. (gap)**
Route (1) assumes a least box of each group is on at every point of its areas, and says the least loss is then the
eigen-residual. TP1 proves L2_S = Lin_S only at eps_lin = 0, which is a different premise. The status sentence
"Route (1) is proved ... with the linear-read claim in place of the premise" changes the statement.
Fix (a proof, short). Corollary: if a box of S that attains L2_S has every unit on at every point of S's areas, then
L2_S = Lin_S for any eps_lin. Proof: on w's area such a box writes P_perp V M_w z_(T_w) + const, so
Pi_w g = P_perp V M_w ztil has coordinates in the span of ztil, which is orthogonal to r_w. In TP1 part 1 the term
tr(Q^T R_W Q) is then 0, so the bound is L2_S >= Lin_S, and TP1 part 2 reaches it. This also says what the eps/2
slack is: it is the most that a switching box can take from the residual r_w. With this corollary the status
sentence can say: Route (1) is proved as stated (Corollary), and under the linear-read claim (TP1).

**3. TP2 part 4: the identification is only an inequality in the text, and only an instance. (gap)**
The proof shows every (r) step has D2_P(B_2) >= (1/2)(10 - top_1(Sigma'_1 + Sigma_2)) = 3. This is TH3's bound
restated. That the (r) step's best value equals the compressed rule's value needs attainment. It holds: the unit
relu(p_1 + c) (p_1 = B_1's pre-activation for S_1, linear in s_a on Omega_1) with v = e_a is an allowed (r) step and
gives 3/2 + 3/2 = 3. The compressed build of part 3 with Q' = ehat_a is this unit. Write that line.
The heading "Proposition TH3's (r) step is a compressed summary", the intro item 4 "Proposition TH3's (r) step is such
a rule", the Comment "child 1's least unit is a compressed summary", and card 3-25 "exactly such a cut-down table"
read as general statements. In general they are false or undefined. An (r) step captures on child c the second
moment of E[Ytil | p_c], the projection onto all functions of the child's pre-activation. That matrix lies below
K_w = Sigma_w + R_w, not below Sigma_w, so it is not a Sigma'_w of part 3 when eps_lin > 0. And "the part of Sigma_1
that child 1's least unit keeps" has no definition outside the instance.
Fix. State part 4 as "In the instance of TH3, min over (r) steps of D2_P(B_2) = 3, which is the value of the rule of
part 3 with Sigma'_1 = 4 ehat_a ehat_a^T". Change the intro and card wording to "in TH3's instance". If a general
statement is wanted, define Sigma^(r)_c = sum p_x E[Ytil | p_c] E[Ytil | p_c]^T and prove the analogue of part 3
with K_w in place of Sigma_w.

**4. TP1's lower bound is the eigen-tail of the linear part; it is often 0. (gap in presentation; bookkeeping)**
Lin_S - (1/2) eps_lin(S) = (1/2)(tr Sigma_W - top_k(Sigma_W)). So the sandwich is
(1/2) tail_k(Sigma_W) <= L2_S <= (1/2) tail_k(Sigma_W) + (1/2) eps_lin(S).
When k_S >= rank Sigma_W (for example two logits and k_S >= 1) the lower bound is 0. In TP2 part 5 at G_1 and G_2 it
is exactly 0, so TP2 does not see the positive loss that TH2 part 2 proves (for KL). "The summary pins the least
loss" holds only when eps_lin is small next to that tail.
Fix. State the sandwich in this form in TP1 part 3 and say it is vacuous when k_S >= rank Sigma_W.

**5. The two-logit note does not prove Route (2). (gap)**
With two logits, every Sigma_w lies on one line, so for k_S >= 1, top_1(Sigma_W) = tr Sigma_W and
Lin_S = (1/2) eps_lin(S). The note's identity "Lin_(S') = sum of the children's Lin" is then additivity of eps_lin.
It also (a) needs k >= 1 for every child (a child with k = 0 has Lin = (1/2) E), and (b) omits the term
(1/2) eps_lin(W') of the nodes W' that the merge takes from kept units. Route (2) is about least losses L2. The note
gives L2_(S') = sum of L2 only at eps_lin = 0, where every term is 0. So "Route (2) is the two-logit note" in D.16
overclaims. The note also uses two logits (node S-twoclass), which is on no path.
Fix. Make it a Corollary tagged "two logits": Lin_S = (1/2) eps_lin(S) for k_S >= 1, so
L2_(S') <= (1/2)(sum over children of eps_lin + eps_lin(W')). With finding 2's corollary, Route (2) under Route (1)'s
premise follows (L2 = Lin, and Lin adds when every k >= 1). Change the D.16 sentence to match.

**6. Switching against the trunk's wording. (gap)**
TP3 part 4 reads a build off stored values. Reading the changed groups' boxes and their J_w weights costs time equal
to their size. That size grows with |T(S)| and |J_w|, so with N's width. "Part 4 meets that wording" holds only
when a switch is a pointer swap to a stored network.
Fix. State the cost: "a switch copies O(sum over changed groups S of k_S (|T(S)| + n_y) + n_y sum |J_w|) stored
numbers and does no arithmetic on N, P or the summaries". Then say which reading of "instant" this meets.

**7. Model of computation in TP3. (hidden restriction, TECH)**
The counts "O(n_y r_S min(n_y, r_S)) operations" for an SVD, and F_w = B_w C_zz^(1/2), are real-arithmetic counts
with an exact eigen-solver. Eigenvalues and eigenvectors of a rational matrix are algebraic, and in general not
rational (a square root already at n_y = 3); for n_y - 1 >= 5 they are in general not expressible by radicals. So Lin_S, q_t and F_w are not rational, while
DP1 and D.16 talk of rational data in polynomial time. With approximate q_t the box is not exact at eps_lin = 0.
Fix. Name the model (real RAM with an eigen-oracle, or precision delta with an added error term in TP1 part 2).
Store Sigma_w itself (rational, n_y x n_y) or a rational LDL^T factor instead of C_zz^(1/2). Also: TP3 part 1's
factor has |T_w| columns until a thin SVD cuts it to rank; say so. TP3 part 2's "q groups" must count the nodes W'.

**8. Tags against paths. (hidden restriction)**
(a) TP1, TP2 and TP3 are for D2_P only. Their paths in DECISIONS.md section 6 list R-obj, but the tags have no
phrase for it. Add a tag phrase ("the objective D2_P") that maps to R-obj.
(b) The tags of TP1 to TP3 say "every claim it uses has eps = 0". The linear-read claims may have any error. Write
"every claim other than the linear-read claims has eps = 0", and add the phrase to the alias table.
(c) The graded bound (1/2) eps_lin(S) is graded in one claim type only. District, shelter, judgment and branch
claims must be exactly 0. This is weaker than T5's "a bound that depends only on the list of the claims' eps values".
Card 3-25 calls it "the trunk's graded form"; it should say "graded in eps_lin, with the other claims at 0, for D2".
(d) DM6 (complete maps) is not extended. So TP1's "exact closed form at eps = 0" is not shown to be T5's exactness
clause. Say whether a map with linear-read claims at eps 0 is complete for the zoom, or say that TP1 is outside that
clause.
(e) Part 1 of TP1 uses none of TH1's premises, only the definitions of L2_S and DP1. Tag it separately; it is a
fact about every network with a tree map.

**9. DP1: which J. (gap, minor)**
DP1 takes J from M's judgment claim, or J empty when R = P. DH3 allows any J_w (with sub(w) minus J_w non-empty) when
the area is P and no judgment claim is held. Then the claim "(area of w, J_w)" of TP1 is not a DP1 claim.
Fix. Let DP1's claim name any set J of units of U(w). The claim still names only an area and a set of units, so it
still meets T5's rules.

**10. Missing lines. (gap, minor)**
(a) "r_w is orthogonal to every coordinate of ztil" needs the normal equations with the Moore-Penrose inverse:
C_yz C_zz^+ C_zz = C_yz, because a vector v with C_zz v = 0 has v^T ztil = 0 at every point with p_x > 0, so
C_yz v = 0. Write it once in the Comment of DP2. It is also where "0 exactly when the claim holds" needs p_x > 0.
(b) TP1 part 2: q_t = 0 for t > n_y - 1 makes Q not orthonormal; say the zero columns add nothing.
(c) TP2 part 3: say why the eigenvectors of Sigma'_W can be taken in im P_perp (0 <= Sigma'_w <= Sigma_w gives
ker Sigma_w inside ker Sigma'_w).
(d) TP1 Comment: "its linear-read error is positive" (T57) is asserted, not computed. Prove it or drop it.

**11. Literature. (gap, minor)**
TP1 at eps_lin = 0 is reduced-rank regression with a shared output space and per-area free regressors (Izenman 1975,
in the table, with the hypothesis "nonsingular covariance"). TP1 treats the singular case through the pseudo-inverse.
Cite it in TP1 and note that its hypothesis is not needed here. Subadditivity of top_k (TP2 part 3) follows from Ky
Fan's maximum principle; say "follows from". No definition or borrowed result splits a family into sparse directions:
the summary is a dense second-moment matrix.

**12. "Up the tree" is across, not up. (structural)**
TP2 part 1 adds the matrices of sibling nodes under one parent. A node's own summary is never made from its
children's summaries: at a lift it is computed again from N (TP2 part 5). So B.P carries the zoom recursion (sibling
merges) and not the depth recursion. B.H's aim ("zoom and depth as one step") and TH4's "each node reading only its
children's summaries" are met only for sibling merges. The TH3 Comment edit ("Theorem TP2 proves it") should say
"for sibling merges, for Lin and the box directions, and for D2".
Fix (a statement to try, or a weaker claim). Either prove a lift rule (the parent's Sigma from its children's records,
which needs the parent's top to read the children's units linearly, a new claim), or say plainly in the intro that
depth steps recompute summaries from N in the one-time pass.

## Checks the caller asked for, in short

1. TP1 part 1: the reduction to Pi_w, "Pi_w g lies in Wim", the split K_w = Sigma_w + R_w, and Ky Fan are correct.
   One line is missing (finding 10a).
2. TP1 part 2: correct. Other tops are constant on w's area by their shelter claims (DM3: constant outside the
   region; areas disjoint). phi_w = VC - QQ^T B_w ztil has coordinates in F_w and values in im P_perp. The free terms
   absorb Y - Ytil - phi_w because F_w = span{1, h_j : j in J_w}.
3. TP2 part 3: correct. Part 4: an inequality in the text; equality holds in the instance but needs one more line;
   the general reading is not proved (finding 3).
4. TP2 part 5: 1/20, 1/10, 0 are right; the lower bound is 0 there (finding 4). The two-logit note: finding 5.
5. TP3: nesting and "base + one term per unit" are right. The cost claims hold for Lin_S and q_t only, in a real
   model (findings 1, 7). Part 4 meets the trunk only under a pointer-swap reading (finding 6).
6. DP1 meets T5's four rules. The graded bound depends only on eps_lin values, but only with the other claims at 0
   and only for D2 (finding 8c).
7. Hidden restrictions: R-obj not in the tags; "eps = 0" tag text wrong for linear-read claims; computation model;
   two logits in the note (findings 5, 7, 8). False statements: finding 1 (intro item 5, TP2 part 2, TP3 Comment,
   card 3-25).

## The three most important fixes, ranked

1. Finding 1. Add the per-node build record to DP2, and split the cost claims: Lin_S and q_t free of N and P; the
   build linear in the width of the group's top and judgment units. Correct the intro item 5, TP2 part 2, TP3 part 4,
   the TP3 Comment and card 3-25.
2. Findings 8 and 6. Align with the trunk: an R-obj tag phrase; "other claims at eps 0" in the tag; say that the
   graded bound is in eps_lin only and for D2 only; say whether DM6 covers linear-read claims; state the switch
   cost exactly.
3. Findings 2, 3 and 5. Prove the Route (1) corollary (a box on throughout its areas gives L2_S = Lin_S for any
   eps_lin). Restate TP2 part 4 as an instance equality with the attaining unit. Make the two-logit note a tagged
   corollary with the W' term and k >= 1. Then rewrite the D.16 status to match.
