# Referee report: Part B.V (the stream), 2026-10-29

Scope: Facts F12, F13; Definitions DM11, DM12, DM13; Part B.V (DV1, DV2, TV0, TV1, TV1.1, TV2, TV3, TV3.1, TV4);
Conjecture DV1 (D.15); the five new rows of the A.4 literature table; `lean/coarse/Coarse/Stream.lean`;
card 3-23 and the B.V rows of section 6 of `DECISIONS.md`. Line numbers refer to `COARSE-THEORY.md` as of this report.

## Summary

| Question | Verdict |
|---|---|
| Lean | `lake build` passes (9068 jobs; only line-length warnings in `Stream.lean`). No `sorry`, `admit` or `axiom` in `Coarse/*.lean`. `norm_rot`, `read_unchanged` and `temp_le` state exactly the one-step lemmas the paper names. Every induction is by hand, as the paper says. |
| 1. Proofs | No false step in TV0, TV1 (forward parts), TV1.1, TV2, TV3, TV4. TV1 part 2's converse is correct, but it is weaker than the text around it claims (findings 3 and 4). Conjecture DV1 has a wrong coefficient (finding 9). |
| 2. Hidden premises | Three: the build re-sets the norm constant (no node); TV2's energy premise is a joint claim about the build's dropped part; DV1's folding sentence needs a final norm before the logits (findings 5 to 7). |
| 3. F13 part 2 | Correct as stated. The premise "every reader reads through a LayerNorm" is needed and is enough. |
| 4. DM11 to DM13 against T5 | They are 0 exactly when the claim holds. They fail "polynomial time" in the bit model in the same way as DM10, and say nothing about it. DM1 does not list subspaces as claim objects. The readout and energy claims together name the cut, so "no claim names a build" holds in letter only (finding 10). |
| 5. Comment (T5's graded form) | Overstated, in the same way as B.Y. It is graded only in the readout errors. Block-read and energy errors must be exactly 0. Under T5 as written, areas are not districts, so neither promise of T5 applies yet (finding 2). |
| 6. Content | 7 results. 1 structural (TV1, by part 2 with its converse; its parts 1 and 3 are bookkeeping). 6 bookkeeping or borrowed (TV0, TV1.1, TV2, TV3, TV3.1, TV4). The most important finding is structural: for a generic network with a wide block, no nonzero area meets the premise of TV2 to TV4 (finding 1). |

## Findings

### Errors and gaps

**1. (gap, scope; S-readoutarea, TV2 to TV4, lines 4615-4617, 4698-4781.)** eps_br(S) = 0 for a nonzero S holds iff
S lies in K = the intersection over l of ker R_l. That is, iff the stacked read matrix [R_1; ...; R_L] has a nonzero
kernel. Suppose one block has q_l >= d and R_l has full column rank. This is the generic case for a transformer MLP
(q_l = 4d). Then K = {0}, and TV2, TV3, TV3.1 and TV4 apply to no nonzero area. The set of DV1 networks with K != {0}
(when some q_l >= d, or more generally when the sum of the q_l is >= d) is a closed null set of weights. This is the same
kind of premise that Theorem T44 flagged for exact symmetry.
So the analogy "the stream's analogue of the last hidden layer" (line 4616 and node S-readoutarea) misleads. Every
network has a last hidden layer. A generic network has no nonzero readout area. And when eps_br > 0, however small, B.V
gives no bound (Conjecture DV1 is open). So S-readoutarea is an exactness premise on a null set, not a graded one.
Fix (a proof, not an experiment): add a lemma. "The areas with eps_br = 0 are exactly the subspaces of K. K = {0} for
generic weights when the sum of the q_l is >= d." Then restate the positive content of B.V as one statement: "Keep
K-perp whole. Replace K by at most C - 1 directions (TV3 with S = K). The cut is exact when the dropped part of K has
constant energy at every norm." Say in the B.V scope and in card 3-23 that this is the whole of what B.V proves at
error 0.

**2. (gap, overstatement; Comment (T5's graded form), lines 4724-4727.)** The comment says the bound "is a function of
the claims' errors alone and is 0 when they are 0, for any number of logits, as in Theorem TY2". This overstates in
three ways.
(a) The bound is graded only in the readout errors. The block-read errors and the energy errors must be exactly 0.
T5's graded clause asks for a bound whenever some claim has eps > 0. A map with eps_br(S) = 1e-12, or with a tiny energy
error at a block norm, gets no bound from TV2. Card 3-23 itself records that the bound fails badly when a block reads a
little of a cut area.
(b) T5's text (DECISIONS section 1) says that districts are "disjoint sets of N's units, each inside one hidden layer".
Areas are not. Node R-streamregion changes that wording and is PENDING. DM6 (completeness) is not extended to areas. So
under T5 as written, TV2 meets neither the exactness promise nor the graded promise. TY2 had a "Comment (completeness)"
that said so for its case (card 3-22). TV2 has no such comment.
(c) The phrase "energy claims with error 0 ... for every group" is wrong. The energy claim is about D, the sum of the
dropped parts of all groups. It is one joint claim, not one claim per group (see finding 6).
Fix: "Under R-streamregion (PENDING), on maps whose block-read and energy claims have error exactly 0, the bound is a
function of the readout errors alone. Block-read and energy errors are not graded. Their graded form is part of
Conjecture DV1. DM6 is not yet defined for areas, so T5's exactness promise does not cover B^a."

**3. (gap, overstatement; TV1 Comment, line 4687, and B.V intro, item 2, lines 4606-4608.)** The intro says the part
"can be dropped exactly when its energy is the same at every point of P before each norm. This is the only way the
dropped part acts." The comment says "otherwise it must be kept". The converse in TV1 part 2 proves less. It proves:
"if the block reads of N_D equal N's on P, then e_l = c_l at the points where R_l h_(l-1) != 0". It does not prove
"D_P(N_D) = 0 implies constant energy". That statement is false.
Counterexample: L = 1, D = span(u) unread by R_1 and U, W_1 = 0, P_D c_1 = u. Let P = {x, x'} with P_D e(x) = 0
and P_D e(x') = -2u, and R_1 e != 0 at both points. Then e_1 takes the values 0 and 4 (not constant), while
e_2 = |P_D e + u|^2 = 1 at both points. Take c_1 = 2 and c_2 = 1. The reads of block 1 change at both points, but W_1 = 0,
so nothing reaches the stream, and the logits are equal. So D_P(N_D) = 0 while the energy before block 1 varies. (A
write that U does not read is not enough for such an example: the final norm reads every direction through |h|.)
Fix: state the converse for D_P = 0 only under a premise that makes the reads observable at the logits. Otherwise
replace "exactly when" by "when, and (for the reads) only when". Replace "must be kept" by "cannot be dropped by a
projection cut without changing some read".

**4. (gap, definition; TV1 part 2, lines 4656-4658.)** "The reads of every block" is not defined. The converse is true
if a read is R_l n_(eps_l)(h_(l-1)), before sigma_l and before b_l. It is false if a read is the inside value g_l,
because sigma_l need not be injective. Fix: define the read of block l as R_l n_(eps_l)(h_(l-1)) in DV1.

**5. (hidden premise; DV2, line 4633; TV1, line 4652; TV2, line 4702.)** The build chooses its own norm constants,
eps'_l = eps_l + c_l. This is load-bearing. Without it, TV1 part 2 holds only when the dropped energy is 0 on P (that is
TV1.1). This is a reading of "a network of the same kind as N" of the same type as R-readbias (a reader re-sets a
constant). No node carries it. The paths of TV1 and TV2 in section 6 do not show it. Fix: name it in the text of node
R-streamregion, or add a node and a card. Mark which results use it: TV1 parts 2 and 3, TV2, TV3.1 and TV4 do; TV1.1
and TV1 part 1 do not.

**6. (hidden premise; TV2, lines 4700-4708; TV4 part 2, lines 4768-4769.)** The energy claim (D, l) is about
D = the sum over S of (S minus V_S). This object depends on the cut in every group at once. Part 3 of TV2 and part 2 of
TV4 ("the term of an area depends only on S and V_S"; "a group that appears at two levels with the same V_S has the
same term") are true of the terms. They are not true of the premise. The premise changes whenever any group of the
level changes its cut. Fix (a stronger premise with local form): use per-area energy claims (S minus V_S, l). On
orthogonal parts the energies add, so per-area error 0 implies the joint claim. Then the premise is local too, and
TV2 part 3 and TV4 part 2 can say so.

**7. (hidden premise; DV1, lines 4625-4626.)** "An MLP-only network of Fact F12 kind 3 with one position, RMSNorm or
LayerNorm, is a network of this definition after folding." DV1 needs a norm before the logits. F12 kind 3 says only
that each block reads through a norm. It says nothing about the logits. F13 part 2 needs every reader, including the
logits, to read through a LayerNorm. Also, DV1's no-norm form is all-or-nothing. A network with block norms and no final
norm is in neither form. Fix: add "and the logits read the stream through a final norm (as in ViT's `encoder.ln` and
GPT's `ln_f`)", or let each reader have its own choice of norm.

**8. (gap; TV3 with TV3.1, lines 4729-4760; scope line 4616.)** TV3 chooses V*_k without looking at the energy of its
complement. TV2's bound holds for that cut only if the energy claims for D(V*) have error 0. So to use TV3.1, the map
must contain an energy claim about the complement of the rule's own choice. The scope line "TV3's closed form minimises
a bound" is true only among cuts whose energy premise holds, and V*_k need not be one of them. Fix: state in TV3.1 and
in the scope line that the energy premise is about the complement of V*_k. Or, in the norm form, minimise the readout
error over the V for which the per-area energy claim (S minus V, l) holds (finding 6). That set is not a closed form in
general, so say so.

**9. (error; Conjecture DV1, lines 6826-6829.)** The conjectured bound is (1/4)(sum over S of Gamma_S eps(S)^(1/2))^2,
with "Gamma_S the gain" and "zero gain for areas no block reads gives Theorem TV2". With gain 0 the term is 0, not
TV2's term, which has coefficient 1. The route (ii) itself says the gains multiply as (1 + gain). So the coefficient
must be Gamma_S = the product over later blocks of (1 + g_l), which is 1 at zero gain. Also, "eps(S) is a claim error"
does not name a claim type, and "the gain" has no definition. As written, it is not yet a statement that can be proved
or refuted. Fix: define g_l (for example Lip(sigma_l) |W_l| |R_l| times the Lipschitz constant of n_(eps_l) on the
relevant set). Name the claim (the readout claim of DM12 at the depth of S, or a new claim type). Correct the
coefficient.

**10. (gap; DM11 to DM13 against T5 and DM1, lines 1119-1126, 1257, 1267, 1273.)**
(a) DM1 lists a claim's objects as lists of units, points of P, rationals and permutations. Subspaces are not on the
list. Fix: "a subspace is given by a list of rational spanning vectors". Then P_A = B (B^T B)^(-1) B^T is rational and
polynomial-time.
(b) "All errors are computed from N and P in polynomial time" (line 1257) and "computed in polynomial time (normal
equations)" (line 1267) are false in DM1's bit model for the write, energy and readout claims. Each one needs N's forward
pass: sigma_l (GELU uses erf) and n_eps (square roots) at every block. Even h_L is not rational. Only eps_br is rational
in N's weights. DM10 recorded the same problem and tied it to node R-propcomplete. DM11 to DM13 say nothing. Fix: state
the reading (to b bits in time polynomial in size and b, in a model where sigma and the square root can be computed to b
bits). Tie it to a node. Note that "is the error 0?" is then not decided exactly. For sigma = relu in the no-norm form,
the errors are rational and the claim as written holds.
(c) The readout claim (S, V) and the energy claim (D, l) with D = the sum of the S minus V_S together fix the
projection cut and its constants exactly. "No claim names a build or a size" holds in letter (the objects are
subspaces). In substance the map names the cut, and the dimension of V is the target size. A map that serves a zoom
must hold one V and one D per level. Say this plainly, as DY2 does for merge partitions. When TV4 part 1 takes V_S "by TV3 in closed form", the
readout claim (S, V*_S) is made by the rule, not held by M. Then TV2's bound is in the error of a claim outside the map
(the readout twin of finding 8).
(d) Good, and worth stating: every error of DM11 to DM13 does not change under the rotation of F13 part 3, if the areas
rotate with it (for example |R_l Q^T P_(QA)|_F = |R_l P_A|_F). Earlier claim types state their invariance under Fact F1.
These should state theirs under F13 part 3. The energy error has the units of |h|^4 and is not normalised. That is
allowed, but it differs from DM3 and DM8.

**11. (gap, wording; card 3-23 in DECISIONS.md, lines 154-164.)** "For areas at the readout, the stream gives what the
neuron work could not: ... a bound in the claims' errors alone, for any number of logits." TY2 already gives such a
bound for any number of logits (DECISIONS row of TY2, line 120). "A closed form for the best cut" is the best cut for
the bound, not for the loss. "TV3 ... at most C - 1 ever needed" drops the energy premise. The card also does not say
finding 1 (generic networks have no nonzero readout area). Fix the card before Shonu decides it, because its
recommendations rest on these claims.

### Wording

**12. (TV1 Comment, lines 4684-4686, and A.4 essay row, line 677.)** "Part 2 is the essay's observation in a theorem."
The essay's unread dimension holds most of the mass and the network changes it "to scale all the other dimensions".
Its energy varies on P. That is the case where part 2's premise fails. The essay's case is the converse of part 2 and
Corollary TV1.1 (keep the line), not part 2. Fix both places.

**13. (TV3.1 title and last sentence, lines 4753-4756.)** The title "needs at most C - 1 directions" and the sentence
"with two logits, every such area can be cut to one direction exactly" drop the energy premise that the second
sentence states. Also, TV3.1 is vacuous when C - 1 >= dim S. For ImageNet (C = 1000) and d = 768, it gives nothing.
Fix: put the energy premise in the title, or write "(in the no-norm form, or under the energy claims)".

**14. (TV3 proof, line 4740.)** "contains every rank-k map with columns in 1 perp" should be "rank at most k". The set
of maps P_perp L P_V Q_S over the V with dim V <= k and all L is exactly {A : rank A <= k, columns of A in 1-perp}. It is
not the set of all rank <= k maps. The next sentence closes this correctly: P_perp A has rank <= k, and
|P_perp (M - A) Sigma^(1/2)|_F <= |(M - A) Sigma^(1/2)|_F because M = P_perp M. State the equality of sets in that form.
The singular case is correct. The v_i with sigma_i > 0 lie in the range of Sigma^(1/2), so A* Sigma^(1/2) = B*. The
vectors (Sigma^(1/2))^+ v_i are independent, because (Sigma^(1/2))^+ is injective on that range. The nesting claim is
correct for a fixed choice of the v_i.

**15. (TV1 part 3, line 4664.)** "m = min ... > 0" reads as part of a definition. It is a consequence. If c_(L+1) > 0,
then A' > 0. If c_(L+1) = 0, then e_(L+1) = 0 on P and A' = A > 0, because N is defined on P. Say so in one line.

**16. (F12, lines 226-240.)** (a) "Three kinds" reads as a classification. It is not exhaustive. A post-norm transformer
(h_l = LN(h_(l-1) + f_l(h_(l-1)))) has a norm on the stream itself and is none of the three. Write "three examples".
(b) "In kind 3 nothing [acts on single coordinates]" is false before folding. LayerNorm's gain gamma and bias beta act
per coordinate. It is true after F13 part 1 folds them into the readers. Say "after folding".
(c) F12 is a reading of cited code, not a theorem. That is acceptable for a Fact with a source.

**17. (F13 Comment, lines 263-264.)** "Only subspaces and angles between read and write vectors are features of the
function." The invariants of the action of O(d) are the inner products among all read and write vectors, the first write
and the biases, including their norms. Subspaces themselves rotate. Say "only quantities that do not change under a
rotation of the stream (inner products of read and write vectors)". Note also that this is a statement about the
function. Trained weights can still pick a basis.

**18. (Literature rows, lines 673-677.)**
(a) SliceGPT. Its column "the hypotheses it makes" gives a property of its method. SliceGPT's own premise for
computational invariance is a pre-LN network in which every reader reads through a norm, after LayerNorm is turned into
RMSNorm by absorbing the mean subtraction into the writes. That is F13 parts 1 and 2. Put that premise in the column.
(b) Izenman 1975. The classical reduced-rank regression assumes a nonsingular covariance. TV3 proves the singular case
itself, so nothing breaks. But "none" in the hypotheses column is wrong.
(c) Framing. No new row splits clean families into sparse directions. SliceGPT uses principal components, and Elhage
et al. 2021 give the no-privileged-basis view. Both are compatible. But the TV4 comment (line 4779: "keep the 'dog'
and 'cat' directions of an 'animal' area") reads concepts as directions. The A.4 row for the linear representation
hypothesis (line 668) says "Use with care: not used". Either mark the TV4 example as an illustration that no proof uses,
or update that row.

**19. (Tags; TV3, line 4729; section 6 row of TV3, line 1382.)** The proof of TV3 is linear algebra about eps_ro for any
subspace S. It uses neither "areas read only by the logits" (S-readoutarea) nor R-readweights. The tag and the path
over-state its premises. This makes it look narrower than it is. It is not a validity problem. TV0 and TV1 likewise
need a projection cut of any subspace D, not an area cut. The tag "area cuts" is harmless, but DV2 defines the
projection cut only for D of the form "sum of S minus V_S". State in DV2 that N_D is defined for any subspace D.

## Proof checks that pass (for the record)

| Item | Check | Result |
|---|---|---|
| F13 part 1 | gamma (.) h / (|h|^2/d + eps_0)^(1/2) = d^(1/2) diag(gamma) n_(d eps_0)(h) | correct |
| F13 part 2 | LN(h) = RMSNorm(C h) + beta with biased variance \|C h\|^2 / d (as in torch); C C = C, so every LN sees the same value after C is applied to the writes | correct; the premise "every reader through an LN" is needed (a raw reader would see C h, not h) and is enough |
| F13 part 3 | n_eps(Q h) = Q n_eps(h); induction | correct; `norm_rot` states it for any linear isometry |
| TV0 | B^T B = I, B B^T = Pi, B^T Pi = B^T, \|B^T ht\| = \|ht\| on D-perp | correct; independent of B |
| TV1 part 1 | R_l Pi = R_l, U Pi = U; ht_l = Pi h_l | correct |
| TV1 part 2, forward | denominator \|Pi h\|^2 + eps_l + c_l = \|h\|^2 + eps_l when e_l = c_l; this includes eps_l = 0, where the cut needs eps'_l = c_l > 0 unless e_l = 0 on P | correct on P |
| TV1 part 2, converse | R_l h (a - b) = 0 gives a = b where R_l h != 0; at the logits, equal softmax gives P_perp U h_L (a - b) = 0 | correct as stated (but see findings 3 and 4 on what it covers) |
| TV1 part 3 | \|t - 1\| = \|t^2 - 1\| / (t + 1) <= \|A - A'\| / A', A - A' = e - c; Fact F11 in the direction KL(softmax(y_N) \|\| softmax(y_N + delta)); \|n(h)\| <= 1 | correct; `temp_le` states the scalar step |
| TV1.1 | P_D' h = P_D' P_D h = 0 on P | correct |
| TV2 | The block induction of TV1 part 2 uses only R_l P_D = 0 (true, since D lies in the sum of the S and R_l P_S = 0) and e_l = c_l for l <= L. It never uses U, so it holds when the logits read D. Then P_R Pi = P_R, P_(V_S) Pi = P_(V_S), U = U P_R + sum U P_S, Minkowski, Fact F11 | correct |
| TV3 | reduction to min over rank <= k of \|(M - A) Sigma^(1/2)\|_F^2; Eckart-Young lower bound; attainment with singular Sigma; row space of A*; nesting; rank M <= C - 1 | correct (wording in finding 14) |
| TV4 part 3 | P_(V_i) P_V = P_(V_i); the residual is the sum of the children's residuals; Minkowski; TV3 part 1 | correct |
| Lean | `lake build` in `lean/coarse` | passes, no `sorry` |

## Structural or bookkeeping

| Result | Rating | Reason |
|---|---|---|
| Lemma TV0 | bookkeeping | change of basis for a projection |
| Theorem TV1 | structural (part 2 with its converse); bookkeeping (parts 1 and 3) | Part 2 says what an unread subspace does in a normed network: it acts only through the denominator of each norm. Part 3 is a perturbation bound. |
| Corollary TV1.1 | bookkeeping | TV1 part 2 with c = 0 |
| Theorem TV2 | bookkeeping | Minkowski plus Fact F11. It holds for any linear readout after an exact block induction. |
| Proposition TV3 | borrowed | reduced-rank regression (Eckart-Young) on the readout. Part 3 says that the logits read at most C - 1 centred directions. That is true of any linear readout. |
| Corollary TV3.1 | bookkeeping | TV3 part 3 with TV2 part 2 |
| Proposition TV4 | bookkeeping | a lookup, and Minkowski over children |

Count: 1 structural, 6 bookkeeping or borrowed (7 results). Finding 1 shows that the structural content of the
positive part (TV2 to TV4) is in the premise, not in the theorems: the readout areas are the subspaces of K, and
K = {0} for generic weights.

## The three most important fixes, ranked

1. **State the readout kernel K, and what B.V covers at error 0** (finding 1). Add a lemma: the areas with eps_br = 0
   are the subspaces of K = the intersection of the ker R_l, and K = {0} for generic weights when the sum of the q_l is
   >= d. Restate TV2 to TV4 as one result about K. Then correct the scope line, node S-readoutarea and card 3-23 ("the
   stream's analogue of the last hidden layer" is not true).
2. **Rewrite the Comment (T5's graded form) and the card** (findings 2 and 11). Say that the bound is graded in the
   readout errors only. Say that block-read and energy errors are exactness premises, and that their graded form is
   Conjecture DV1. Say that under T5 as written (districts are sets of units; DM6 is not defined for areas) neither of
   T5's promises applies until R-streamregion is decided. Remove "what the neuron work could not".
3. **Make the premises of TV1 and TV2 honest and local** (findings 3, 5, 6, 8). Weaken "exactly when" and "must be
   kept" to what the converse proves. Name the build's re-set of the norm constant as a node or put it in the text of
   R-streamregion. Replace the joint energy claim on D by per-area energy claims (S minus V_S, l). State that TV3.1 needs
   the energy claim on the complement of the rule's own V*_k. Also correct the coefficient of Conjecture DV1 (finding 9)
   to the product of (1 + g_l).
