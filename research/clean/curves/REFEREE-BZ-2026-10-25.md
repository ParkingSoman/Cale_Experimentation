# Referee report: Part B.Z (D22, D23, Remark B.Z.1, Theorem T25, Remark B.Z.2), Part 0, card 3-2 (2026-10-25)

Files read: COARSE-THEORY.md (Part 0, B.T, B.Z, D20/D21), DECISIONS.md, lean/coarse/Coarse/*.lean. `lake build`: passes, no `sorry` (linter warnings only). The scripts were not run.

## Findings, ranked

### 1. (GAP, serious) Option (a) does not remove over-fitting; T25 depends on a rank defect of N
Location: D22; Remark B.Z.2 ("A fold build cannot do this"); card 3-2 ("this kind of over-fitting is impossible").
D22 allows the group S to be all units, with arbitrary real m_i and an arbitrary offset beta. So the fold units are exactly the units relu(w^T x + b) with w in rowspace(W_1) and b any real. The paper never states this one-line lemma. "Fold of N's own units" is therefore only "reads the input subspace that N's first layer spans".

Counterexample to the claim "a fold build cannot read what N ignores". Add to N a third unit relu(x_2 - 100) with output weight (delta, 0), delta != 0.
- The unit is 0 on P, because x_2 <= 3 there. So N' = N on P.
- The unit is fixed by the mirror, and W_out pi = W_out still holds. So N' is perfectly clean for Gamma, in reduced form, with faithful action. P is still symmetric.
- relu(1*z_3 + 100) = relu(x_2) is a fold. So a fold build of width 1 reaches D_P = 0, and the least fold build is again the "over-fitted" one.

T25 therefore shows only that a rank-deficient N plus a mix that correlates an ignored coordinate with the answer separates the two classes. Whenever rank W_1 = d, fold builds equal all builds and the restriction removes nothing. The same holds for (a) in the card.
Fix: (i) state the lemma "fold units = {w in rowspace(W_1)}" in B.Z. (ii) Add this N' as a Proposition, or at least a remark, since it is a two-line proof. (iii) Rewrite Remark B.Z.2 and the card: (a) blocks reading directions outside rowspace(W_1) only. Delete "impossible by construction". (iv) A real fix needs a mix-dependent notion, for example folds of units that are live on P, or a penalty tied to the mix. Mark that as the open decision.

### 2. (GAP) D23: zoom is not shown to be more than D22, the offset system is not shown solvable, and the switching claim is overstated
(a) A cut build is a fold build in which each unit is also a fold of the group below its node. That is a combinatorial constraint: w_v must lie in span{w_i : i in S_v}. If the rows of W_1 are independent (n <= d), any unit with weight outside the span of a proper subset of rows forces S_v = all leaves. Then only the root can serve a non-trivial width, and "exact for P at widths k" fails in general. B.Z intro asks "are the best builds zooms?". T25 treats fold builds (D22) only. Nothing relates least fold builds to cuts of one tree. Remark B.Z.1 works only because n = 2 and d = 1, where the group constraint is vacuous (the paper says so).
(b) Offsets. The build offset is sum_{v in C} c_v. Exactness at width k needs sum_{v in C_k} c_v = gamma_k, the optimal offset. This is a linear system with n equations in 2n-1 unknowns. Solvability needs the incidence vectors of the chosen cuts to be linearly independent. For non-nested cuts that can fail. Not argued. Fix: put the offset per cut (one scalar per width), or prove independence.
(c) "Costs a number of operations proportional to that difference, independent of the size of N": the difference |C Δ C'| can be about n (all leaves vs the root), and each node carries a weight vector in R^d. The cost is O((|C|+|C'|)(d+n_y)), which is the cost of reading the output, so it is bounded by the output size, not by N's width. A stored table of least builds has the same cost (see item 5). The tree gives a storage saving only. State the cost correctly and drop the implication that the tree is what makes switching instant.
(d) D22/D23 are for one hidden layer. T*** has L layers and sizes (k_1..k_L). The multi-layer fold is not defined, yet card 3-2 plans around depth.

### 3. (GAP, medium) Theorem T25 proof: premises and a cited step
- Part 1 is correct: W_1 rho_0 = pi W_1, pi b = b, W_out pi = W_out; reduced form rescales both units by sqrt2 together; faithful; both units read; P invariant because T is even.
- Part 2 is correct. The "(l1)" citation is only needed to say D >= 0.
- Part 3: the reduction to the d = 1 instance is correct (fold gap depends on x_1 only, and the seven x_1 values have weight 1/7 each). Positivity cites "T20 part 1 shows value 0 needs two kinks". T20 part 1 is a statement about width 2. The width-1 case follows from the same second-difference argument: D(Delta_B - T) = beta*h_t - (e_-1 + e_1), and h_t has support in two adjacent integers while -1 and 1 are not adjacent. This is one line, but it is not what T20 states. Write it in. It is also needed for D2_P and for D_P (via (l1)).
- Path (DECISIONS section 6 row): lists R-cleannet, N-ny2, N-onehidden. The proof also uses: R-fold (the statement is about D22; PENDING), R-obj (parts 2 and 3 state D2_P, as in the T24 row), and T19 (path S-twoclass, S-onehidden, which coincide here with N-ny2, N-onehidden but are different nodes; plus TECH "a finite mix"). Add R-fold and R-obj to the path. Note that the D20 hypotheses are on the path but not in the "(under: none)" tag. That is allowed by the convention in Part 0; say so in the row.
- "Even for a symmetric mix": the symmetry plays no role in the proof (the proof needs only rank deficiency and x_2 = T(x_1) on P). It is decoration, which suggests more than is shown. Also, P lives on a curve in R^2 with x_2 a deterministic function of x_1. Say that it is an artificial mix (as the paper does for T20/T22).
- Lean: "none, by hand", with the reason that the identities are "finite matrix products". That reason argues for Lean, not against it. The equivariance identities are decidable on concrete matrices, and `relu_second_diff` already exists. CLAUDE.md rule 2 asks for the finite version in Lean with the reason for any gap. This is a policy gap. There is no Lean statement that could be weaker than the paper's (no Lean name is given).

### 4. (GAP) Remark B.Z.2 and card 3-2
- "Over-fitting in its purest form": over-fitting needs a law other than the one being fitted. The objective of T*** is D_P on P only. On P the feature x_2 is exactly right, not accidental. Say: "the least build uses a feature that is correct on P and that N ignores" and note that off P nothing is claimed. The word "over-fitting" is Shonu's concern, not a result.
- Option (b), independence of the ignored part from the read part, is offered as a way out, and the card lists "prove that the best build is then a fold" as next work. That may be false. For a build g(x_1, x_2), E(g - T(x_1))^2 = E(g_bar - T)^2 + Var_{x_2}(g). Averaging over x_2 can turn one unit into a two-kink function of x_1, at the price of the variance term. Nothing shows the price always wins. Label (b) a conjecture that could fail, not a proved route. For a finite mix, "independent" means the mix is a product set; say that.
- Option (c) is simply the current trunk. Fine.
- Card claim "Every 'fold' of the teacher's own units does worse": true for this N only (see item 1).
- Card claim "with ONE hidden layer, instant switching is easy by storing the best build for each width": correct only as a statement about switching, for L = 1, once the builds exist. It silently assumes (i) a least build exists for each width (T23: not for three logits with any new units; for fold builds with two logits T19's proof goes through verbatim, since the restriction is a polyhedral subspace, and that is worth stating as a corollary); (ii) it can be computed: T22 says NP-hard for every network and mix, and the trunk's "exact closed form for clean N" has no theorem yet; (iii) storage is about n^2 (d + n_y) numbers, not n^2. It also makes D23's tree unnecessary for T***, so the card should say what the tree buys (storage, and a path to depth).

### 5. (GAP, minor) Remark B.Z.1
Correct in substance: z_1 = x-1 and z_2 = -x-1 give (m_1-m_2)x - (m_1+m_2) + beta, which covers every affine function of x. The leaves with a_v = (1,0), c_v = 0 give N. Two leaves under one root satisfy "at least two children". Issues: T20 part 2 does not state that a least width-1 build exists with a given form; existence is T19, and it needs to be cited. The remark fixes one least build per objective (KL and D2 may have different ones; the zoom must name the objective). It also says "T20 breaks the weaker idea ... it does not break the zoom": this is vacuous at d = 1 (the paper admits this); it should not be listed as evidence for the zoom.

### 6. (GAP) Part 0 and DECISIONS: stale sentences
- Part 0: "the nodes above T** (R-least, R-nest, R-root, now decided, and R-score, R-cleannet, R-goal, pending)". False now: R-score is DECIDED (the mix came back), R-goal is DROPPED, R-fold (PENDING) is missing. Also "Every result of Parts B and D is about a smaller or simpler question than T*" conflicts with "Part B.T works at T* itself" and ignores T24/T25, which are at T** and T***. Part 0 does not mention T25 or the tension between "built from any new units" in the trunk text and the zoom (R-fold). The revision header (lines 3-13) stops at revision 7 and omits B.T, B.C, B.Z.
- DECISIONS section 2 is titled "what is known at the trunk (T*)" and has no T25 row. Section 3 header says "R-score: PENDING" and card 3-1 still recommends (ii) under T**, although the node table says DECIDED. Section 8's diagram marks R-least, R-nest, R-root as PENDING; the node table says DECIDED. Section 3 card 3-2 is placed before 3-1 and called "the first decision under T***".

### 7. Literature (OK)
B.Z borrows nothing. "Fold" is an interpretability-to-math object defined here; no SAE/dictionary framing. T19 is cited with its hypotheses (two logits, one hidden layer, finite mix). No finding.

## Table

| Item | Check | Verdict | One-line reason |
|---|---|---|---|
| D22 | defined, one hidden layer | GAP | Equals "w in rowspace(W_1), any bias". Lemma missing. One layer only |
| D23 tree and cut | well-defined | GAP | Group-span constraint not related to D22. Offset system sum c_v not shown solvable |
| D23 switching cost | correct as stated | GAP | Difference up to n. Cost is output size. A stored table costs the same |
| Remark B.Z.1 | T20 is a zoom | OK (vacuous) | Spans affine functions as claimed. Needs T19 cited. Tree constraint empty at d = 1 |
| T25 part 1 | D20, symmetric P | OK | Identities, reduced form, faithfulness, P invariance check out |
| T25 part 2 | relu(x_2) = T on P | OK | Correct |
| T25 part 3 | fold builds, positivity | GAP | Reduction correct. Positivity cites T20 part 1 for a width-1 statement it does not make. Write the 1-line argument |
| T25 tag and path | hypotheses | GAP | Path misses R-fold and R-obj; T19 nodes differ from N-ny2, N-onehidden; "symmetric" unused |
| T25 meaning | "a fold cannot over-fit" | GAP | False: N' with a dead unit reading x_2 is clean and its fold builds reach 0 |
| T25 Lean | named statement | GAP | "none, by hand" with a reason that supports Lean. No `T25` in Lean. Build passes, no `sorry` |
| Remark B.Z.2 | options and wording | GAP | "Over-fitting" not defined for a mix-only objective. (b) unproved and possibly false. (a) overclaimed |
| Card 3-2 switching claim | store best build per width | GAP | True for L = 1 only for switching. Existence and computation assumed. Storage count wrong |
| Part 0 trunk paragraph | stale sentences | GAP | R-score, R-goal, R-fold status wrong. Revision header out of date |
| DECISIONS sections 2, 3, 8 | claims about trunk | GAP | Stale status (PENDING vs DECIDED), no T25 row |
| Literature | framing, hypotheses | OK | No borrowed result in B.Z |
| `lake build` | pass, no sorry | OK | Passes with linter warnings |

Content count in B.Z: structural results 1 (T25, and it is a counterexample about rank, see item 1); definitions 2; remarks 2 (one vacuous, one interpretation). Bookkeeping identities: 0 (the D20 checks in T25 part 1 are finite). The section is thin: it has one structural result, and the result's reading is overclaimed.

## Three most important fixes, ranked

1. Prove the lemma "fold units are exactly the units with weight in rowspace(W_1)" and add the N' example (a dead unit reading x_2). Rewrite T25's meaning, Remark B.Z.2 and card 3-2: option (a) blocks only directions outside the span of the first layer, so it does not remove the over-fitting that T25 shows. Present T25 as "rank-deficient N", and say that a mix-dependent restriction is the real open decision.
2. Repair D23: state the group-span constraint and say T25 covers fold builds, not cuts; make the offsets per cut or prove independence; state the switching cost as O(output size) and say a stored table of least builds does the same; define the multi-layer fold or say the section is for L = 1.
3. Fix the supporting text: write the width-1 positivity argument (second difference with non-adjacent support) inside T25; add R-fold and R-obj to the T25 path; move option (b) to "conjecture, may fail" and replace "over-fitting" by "uses a feature correct on P"; update Part 0 (R-score DECIDED, R-goal DROPPED, R-fold PENDING) and DECISIONS sections 2, 3, 8; give T25 a Lean statement for the finite steps.
