---
name: project-curves-track
description: "NOW (2026-10-29: stream areas B.V, TV1 norm, TV2 readout split, card 3-23); before (2026-10-28: ResNet-18 map+zoom, B.Y merge rule TY2, card 3-22); earlier NOW (2026-10-25 late: TRUNK RESET, DECISIONS.md ledger, root decision R-root pending; before: 2026-10-25 eve SWITCH FORMULA: B.7 T17 switch read = teacher gate arg regressed onto kept units; open = which units to keep (D8); earlier 2026-10-25 NOODLE: B.5 self-gated + spatial folds, LADDER.md; earlier 2026-10-24 EXACTNESS: formula that training cannot improve; generalist weights-only, developer re-mix; plan okay-for-this-session-twinkly-gosling.md): curves track. Essay's 3b crush in lucid InceptionV1, family crush (10 curve neurons -> k), derive the coarse neuron; log research/clean/curves/CURVES-LOG.md"
metadata:
  node_type: memory
  type: project
  originSessionId: d639c272-cfbf-458d-b68f-6deb03de9fc5
  modified: 2026-10-29T21:37:49.423Z
---

## NOW (2026-11-01, main, ROADMAP.md session 1 continued; plan ~/.claude/plans/i-think-we-are-agile-raccoon.md; LADDER 42)
Shonu: close row 3 first; roadmap must END in the full Cale experience on the WHOLE of ResNet-50, no detraction
(written into ROADMAP.md: goal quote, standing rule, sessions 3/4, hard gate). Also: card items were math details ->
card 3-32 DECIDED by delegation (R-active, S-squeezecomplete); rule saved in [[feedback-trunk-decisions]]. Item 5:
exact = value with a PROVED matching lower bound (not unknown count). FINDING: DH3 boxes write straight to logits, so
no "below the top" squeeze (B5 dropped). PROVED: Definition DQ5 leak problem (leak points: stacked rank <= k, own block
exact up to 1_K shift, value = odds-weighted phi of spills; convex at fixed span); Theorem TQ5 LIMIT-1: complete squeeze,
N_m lowers outside logits by m: L^on <= e^-m Lambda every m, e^m L^on -> Lambda (exact split identity (*)); Lemma TQ6
rank-k unit-diagonal off-diag >= n^2/k - n, tight frames. NEG: TQ4 pairing fails for 3+ areas (model/tq_ngauge.py).
Conj DQ4: Lambda NP-hard in #areas (Peeters min-rank; gap: quantitative separation). ResNet-18 (pc_active_free.py,
G=100 k=38): free dirs gain 2.3% (pinned), alignments unstructured, KL/Lambda = 0.34 -> FAR from confident limit:
TQ5 regime is wrong for trained net. Referee REFEREE-BQ3 (1 error, 8 gaps, fixed; TQ5 = always-on, along N_m, no rate). ESSAY HINT (Shonu: read the essay + what clean structure gives free; and 'mathematics are the only foundations'): 'coarser, never fractured', 'PCA-like shared representations' -> keep what areas SHARE. Proposition TQ7 (SECOND-ORDER, NOT refereed): areas' Fisher metrics within 1 +- eta of one metric -> closed-form shared span within (1+eta)/(1-eta); model/tq_shared.py 0 fails; premise FAILS on ResNet-18 (pc_eta.py eta = 1-1e-8). Orientation (pc_shared.py --fullread): shared span loses .027/.104 (k 38/14) vs canonical packed span same reads .156/.200, aligned packed own-block reads .097/.265. NEXT: Question DQ6 = claim 'own-block movement built from shared patterns' + matching lower bound. Row 3 NOT closed. 2026-11-02 (LADDER 43, plan v2 'row 3 by the shared span'): pc_span_search.py: shared span is the object (joint search -11%, packed/random starts 2.5x worse; 61%/54% of loss on near-tie quarter; TI4 area-mean model prefers another span -> wrong model). Lemma TQ8 (per-point KL = (1/2)Var_s(d) x [g(-omega), g(omega)], under none, tq_point.py 0 fails); planned 2-sided TQ8 FALSE (cheap downward pushes), revised. Lemma TQ9 certificate (grid+exact dual+cell step; needs on-inside premise) valid but impractical (tq_cert.py: nothing pruned at n_y=4). Referee REFEREE-BQ4 fixed (TQ7 not first-order tight). Trained small net still not certified. NEXT: second-order certificate; DQ6 claim. Heartbeat rule: 15-min Monitor on long runs. 2026-11-02 later (LADDER 44, plan v3 'small dropped part', 6-item definition of done D1-D6): TQ10 PROVED+refereed (REFEREE-BQ5): family s^e = kept + e*dropped, least = e^2 Phi + O(e^3), Phi convex weighted LS (per-point Fisher), pinning via Pinsker + sin-Theta, Phi_free for switching; constants ~ powers of 1/s_min (LIMIT-2). tq_small.py 25 inst 0 fails. ResNet-18: Phi(s0 metric) .120 vs best .025 (5x off); Phi(own metric) .022 (value OK) but its build .104 (bad tilt: quadratic over-penalises cheap moves on unlikely classes); untilted quadratic 47x true KL. DQ7 rank claim (names map nodes; legal after fix), Cor TQ10.1 ONE-BUILD. Certificate TQ9 + dual cell step + head form: referee found impossible rank condition (fixed), rerun still -inf at cap -> D4 FAIL. Triage: 37/40 pending nodes decided by delegation; card 3-33 has R-mixclean, R-circuit, R-wellposed. Row 3 NOT done. NEXT: DQ8 = pinning lemma with head-class constants, then convex Psi (exact KL, linearised tilt) on ResNet-18.

## NOW (2026-10-31 night, main, ROADMAP.md session 1): ROW 3 = PACKING (plan ~/.claude/plans/okay-last-session-i-jiggly-aho.md; paper B.Q, D.18; card 3-32 PENDING; card 3-31 ANSWERED (c), S-keepdrop DROPPED)
Shonu: keep-or-drop "is the same as pruning"; fix row 3 (squeezed groups); 4-session ROADMAP.md to instant ResNet zoom;
standing rules [[feedback-no-unchosen-tradeoffs]]. Plan's lemma L-sep (tree knapsack: no unit serves two separated
nodes) is FALSE (model/ts_sep.py): with per-area offsets one direction serves areas with disjoint confident class
sets (m=6: shared .0005 vs pure .103; common offset too). Form = PACKING. DQ1 active-class claim (area, K), error =
outside mass weighted by signed own move + envelope rho_K = |Pi_K Y|_2, one shift per class (v1 with |Y|_2 envelope
was ~60 nats wide on ResNet). TQ1: sum_w L_w(k) <= L_S <= packed <= sum eps_act (own-class rank <= k, disjoint K;
max r_w units vs TI2's rank Sigma) = ONE-BUILD (referee: lower side 0; eps_act is a SUMMARY not a claim). TQ2: eps_act=0 iff
Y=0 (property of DQ1); every row-3 least > 0 (Grassmannian compactness); sharing beats knapsack (network instance).
WITHDRAWN: "row 3 is never exact" (my definition's fault; TI1 exact with L>0). REFEREE-BQ-2026-10-31.md saved. DQ1 conj: first-
order tight (ts_tree E: ratio -> 3.41). Lean Coarse/Pack.lean pack_step. Checks ts_sep, ts_tree 0 fails.
REALISM (pc_active.py, ResNet-18 proxy areas from top-1 class clusters): outside mass .065-.205 (premise holds);
eps_act/I 1.6-3.6 (bound NOT useful yet); canonical packed choice keeps only 12-51% of information; discriminator
pc_active_opt.py: FITTED head of packed form keeps 72% (300 areas, k=14), 90% (100, k=38), 40% (k=2) -> construction is
the weak part (canonical alignment), network allows cheap squeezing. NEXT: aligned packing + its bound.
CONTINUED (Shonu: "you haven't completed the work ... tried different approaches"): TQ3 PROVED (clean-limit sandwich:
sum (1-mu/p) own-block leasts <= L_S <= sum own-block + sum E_w, E_w<=2D mu; ts_tq3.py 0 fails, ratio -> 1.0006 at a
least 0.083>0). Scales alone close 10-17% (pc_active_scale.py); full k x k gauge per area with own SVD directions
(pc_active_scale.py --gauge) closes ALL the gap in complete case (G=100 k=38: .099 vs fitted .104; G=300 k=14: .265
vs .276) -> alignment = the one missing piece = Conjecture DQ2 (balancing, Moore 1981). Gating approach: does not
apply at top layer. Card 3-32 item 5 asks if closed-form dirs + fixed-size alignment + convex fit = "exact".
REFEREE-BQ2: DQ2 "balancing" WRONG -> Lemma TQ4 PROVED (von Neumann lower trace + AM-GM): min tr(TPT^T)+c tr(T^-1QT^-T)
= 2 sqrt(c) sum sqrt(p_down q_up), REVERSE pairing (ts_tq4.py 0 fails); TQ2.3 instance fixed (inputs sigma,c+,c-:
shelter); TQ3 Remark -> Conjecture DQ3; pack_step2 in Pack.lean. NEXT: n-area alignment; reverse pairing on ResNet proxy. Referee done, all findings fixed, not re-refereed.

## NOW (2026-10-31 late, worktree .claude/worktrees/kl, ANCHOR-MERGED into main; referee REFEREE-BI: proofs correct, summary sentence 'only when uniform' was false -> 'if', TI2.1 overclaim qualified: graded in eps_lin only, DM6 not defined for DP1/DL1 claims): THE THEORY ON KL (plan ~/.claude/plans/okay-you-ve-done-a-ticklish-garden.md; paper B.I, D.17; LADDER 40; card 3-30 ANSWERED (a), card 3-31 PENDING)
Shonu: "refit everything, but specifically to find the optimal KL ... for everything, including deep networks";
he expected it to be close. R-obj DROPPED (his words in DECISIONS). Told him first: not a refit for squeezed groups.
Proved (OPT-KL): DI1 information table (m, sbar, I_w = sum p KL(s_x || sbar)); TI0 chain rule (Lean kl_chain in
Merge.lean); TI1 k=0 group least = sum I_w any depth, merge adds, lift adds + m_c KL(sbar_c || sbar_u) (only district
claims + flat J); TI2 k >= rank Sigma: least <= eps_lin/4, TP1 box within it; TI2.1 keep-or-drop levels (NEW node
S-keepdrop PENDING, card 3-31) exact even where V > 0. TI3: least build matches N's mean softmax per area; always-on
least = min over output spans of convex soft-label logistic fits, attained (EQ). TI4: scale a: least = a^2 x D2 problem
in each area's Fisher metric H_w + O(a^3); one mean softmax -> (1/2) tail_k(H^1/2 Sigma H^1/2); uniform -> D2's.
TI5 (NEG): D2 tables (+ means) don't fix KL least (k=0 pairs exact by primes / concavity; k=1, 3 classes: a^2/12 vs
a^2/16, D2 line 3/32). Conjectures DI1 (squeezed case NP-hard, route Gillis-Glineur, gap: Fisher matrices only),
DI2 (least KL unit may SWITCH inside its area: optimiser found 1.2%/4.5% below always-on at a=1/2, 1 in uniform
instance; uncertified) -> TL5's "on over whole area" may be a D2 artifact. Tl_deep net: TP1 1-unit box KL 2.63 vs
drop-all optimum 0.58. Checks model/ti_info.py, ti_open.py 0 fails. Squeezed case (0<k<rank) OPEN = main gap.

## Before (2026-10-31, worktree .claude/worktrees/joint, merged into main): ONE BUDGET + OPTIMALITY AUDIT (plan ~/.claude/plans/pasted-content-id-8b09-okay-correct-witty-tower.md; paper B.V.7, B.Y.2; LADDER 39; cards 3-29, 3-30)
Shonu: coarsen a capability, not neuron vs stream; then THE BAR: "find the optimum ... well executed distillation is
trying to get to", never "done" without confronting optimality ([[feedback-error-meaning]]; CLAUDE.md "The bar";
turn reminder rule 0). Labels OPT-KL / EQ / OPT-D2 / ONE-BUILD.
Proved: DV6 + TV19.2 (EQ: narrow unit builds, one isometry Q with range >= E_N, same least loss as wide once
d' >= dim E_N + sum k_S; lift and projection); TV19 (fits in written span), TV19.1 (no norm: modulo read kernel,
Kalman); TV19.4 (relaxed class; a full first write forces the full stream). DY3 (skips, channel units, average-pool
copies across 2x2 tilings, wiring) + TY4 (EQ: TH1's split for ResNet stages; capability = subtree group of inner +
stream channels; no trunk change). TY5: k=0 group with constant judgment -> KL least = log mean softmax per area
(OPT-KL closed form); D2 choice loses exactly m KL(sbar || softmax(ybar)); clean example loses 0.0064; general only
factor 1/(2 mu). => TP1/TL5 are OPT-D2 (R-obj RETRO) and do NOT meet the bar. Two referee rounds
(REFEREE-BV7-BY2-2026-10-31.md): round 1 3 main gaps (TV19.2 empty, stride copies, TY5 premise), round 2 TV19.4 vacuous;
all fixed. Checks model/tv_written.py, model/ty_stage.py 0 fails. Lean written_step, quotient_read (Stream.lean),
kl_split (Merge.lean). Nodes S-skipff, R-jointgroup PENDING (card 3-29); card 3-30 R-obj (rec: KL as target, D2 only
labelled stepping stone). Open: KL least of one clean group with k >= 1 (no closed form); graded TY4; stream below
budget. Next session option: ResNet-18 layer4 stage cache (~4 GB), measure TH1 claims' errors and gap to a
well-run distillation. Merge note: anchor script refused DECISIONS (adjacent hunks); main == base verified, copied.

## Before (2026-10-29 night, worktree .claude/worktrees/smallnet, ANCHOR-merged into main by scratchpad merge_main.py): DEEP LAYERS (paper B.L.2; LADDER 38; card 3-27 updated)
Shonu: "continue until you can find the theorem which finds the optimal small network for deep layer cases".
TL3: district + shelter (all districts below) + on-inside (every node) + judgment (every split node) at eps 0 =>
logits affine in the subtree's top pre-acts (gate pre = kappa on area, zeta off => indicator affine). TL3.1 eps_lin 0.
TL3.2 SCOPE (referee): every split below the top = flat 0/1 direction of the top layer; generic points with
|R_w| > 2|T_w| => no splits (linear stack). TL4: Sigma_u = sum children + Gamma (between table; needs layer-L tables
AND means, masses, c_N). Corollary TL5: TP1 closed form least at any depth, tables by add / add+between. TL6 (replaced a
FALSE conjecture: normalised errors fail): eps_lin <= |P W_out|^2 eps_L^2, absolute defects, weight norms, 1/gate
margin; gap to least only where TP1 applies; loose 1e4-1e9 on model/tl_deep.py part E. Two referee rounds
(REFEREE-BL2). Open: tight graded bound; gates reading varying features; Part C measurement.

## Before (2026-10-29 evening, worktree .claude/worktrees/normproof, MERGED into main by ANCHOR merge): STREAM ROUND 3: NORMS, EVERY ACTIVATION, ANGLES (plan ~/.claude/plans/create-another-worktree-for-dapper-flamingo.md; paper B.V.4, B.V.5, D.15; LADDER 36; card 3-25 ANSWERED)
Shonu on card 3-25: S-laminar is a development decision (mine); no smooth-activation restriction ("a nice general
method"); prove the norm conjecture. New standing rule in [[feedback-trunk-decisions]].
Proved: DV4 remainder modulus psi (any Lipschitz sigma); TV15.1 (smooth (Lip'/2)t^2; ReLU (1-alpha)max(t-|a|,0));
TV15.2 norm slope (I-nn^T)/nu, |D^2 n| <= (2/sqrt3)/nu^2 sharp (identity 4/3 - (4-3q)(3c^2-1)^2/3); TV13.1 exact
identity with secant matrices; TV15 = DV3 PROVED, both forms, every Lipschitz activation; TV15.3 orders (ReLU no norm:
exact when windows inside N's margin, stated at t=1); TV16 angles (any cut dropping D = orthogonal cut in other
coordinates; TV6 parts 1-2 for any direct decomposition with Q_A stream side, Q_F^T read side); TV17 (norm: oblique
energy signed; per-area energies do not add, two 3-dim examples) -> energy claim on whole dropped part. Nodes:
S-laminar DECIDED by delegation (kept only on TV2, TV3.1, TV4), S-smoothact and S-nonorm DROPPED (moot), NEW R-cutclass
DECIDED by delegation (D any complement; any C with kernel D). Checks model/tv_norm.py 0 fails ~8,100 checks; Lean
relu_rem_pos/neg, norm_hess_identity/le/vec. Referee REFEREE-BV45: TV15 correct; 3 errors + 2 gaps (angles, one
corollary sentence) fixed, not re-refereed. Synthetic: with norms TV15/D_P -> ~4 as leaks shrink (TV6 stays ~1e7);
worse than TV6 at full leaks. MERGE INCIDENT: my whole-file copy at 16:44 overwrote smallnet's 16:43 merge; diff3 -m
DROPPED lines (do not use it on these files); fixed by scratchpad anchor_merge.py (difflib hunks placed by unchanged
neighbours). Lesson: cmp right before copy is not enough when two sessions merge within a minute; always anchor-merge.
Open: TV2/TV3.1/TV4 with whole-D energy claims at an angle; remainder on trained nets; attention.
ROUND 4 (same worktree, same day, "continue until a decision"): B.V.6 member claims DM16 (one per member+block,
zoom-free), TV18 (any sum of members D: loss <= (sum sqrt member claims)^2/4; norm form orthogonal members, no-norm
any angle), TV18.1 (T5 form every zoom: certified members cover dim S - k_S = member completeness; knapsack DP;
5 parts), TV18.2 (norm at angle: cross energies in gamma^B, cut keeps eps+max(c,0), no premise), TV2.1 (TV2 at
angle, oblique cut, oblique energy claim on whole D), TV15 part 2 (beta^t = min(beta, |delta^N|+gamma)). Trained
synthetic nets (model/tv_trained.py): Lambda_N alone within 2.7-6.5x of D_P with norms; remainder is the weak link;
part 2 fixes small cuts (8.6e13 -> 47). Referee REFEREE-BV6 (TV18/TV18.2 correct; 1 error, 3 gaps fixed). R-normconst
widened by me (c_l > -eps_l, for TV2.1/TV17). Checks tv_member.py and tv_norm.py 0 fails. Merged to main by
anchor_merge (main unchanged). LADDER 37. Card 3-28 PENDING: leave pure streams for attention now (rec yes).

## Before (2026-10-29 latest, worktree .claude/worktrees/smallnet, in main; my whole-file copy raced the stream session's copy one minute later, and that session re-applied B.L by anchor merge. LESSON: never copy whole files into main, even after a cmp check; always anchor-merge): BROKEN CASES READ AS NETWORKS (plan ~/.claude/plans/pasted-content-id-3bfd-okay-we-glittery-quiche.md; paper B.L; LADDER 35; card 3-27)
Shonu: an optimal small interpretable network exists, so find the right FORM by reading the network, not by explaining
hardness. TH3 = reuse (u_b, u_c both write e_b; child 1 keeps e_a). Zigzag = lift reads past bottleneck g.
TL1 (sufficient only): from TP1-type boxes, a merge (r) step reaches Lin when (K): each kept merge direction lies in
K_c + ker Sigma_c; graded (within eps_lin/2); holds if children lossless or orthogonal images with k' <= k_c; fails in
TH3. DL1 on-inside claim (node R-oninside PENDING). TL2: one-layer lift through on-inside units re-expresses every finer
unit one layer down; k >= n_j => loss does not rise (KL and D2); TL2.1 zigzag. Open: lift with smaller size (box must
do the children's switching; T57 there). Referee REFEREE-BL: 13 findings all fixed, not re-refereed. Check
model/tl_netread.py 0 fails (generic random nets: (K) only 2/40). Lean Coarse/Netread.lean.

## Before (2026-10-29, worktree .claude/worktrees/summary, MERGED into main by file copy; main was unchanged since base. The stream worktree must anchor-merge, not copy whole files, or B.P is lost): SUMMARIES PASSED UP THE TREE (plan ~/.claude/plans/so-we-ve-shown-that-kind-wave.md; paper B.P, D.16 status; LADDER 33; card 3-26)
Shonu: after TH3 ("naive recursion not optimal"), explain the summary idea, prove the same with summaries, keep it
cheap and keep instant switching. Explained in chat (TH3 picture: child 1 table e_a 4, e_b 3; add tables; top-1 = e_b
6; loss 2 vs naive 3). Paper B.P: DP1 linear-read claim (node R-linread PENDING; a claim on N: centred logits linear in
top pre-activations after free terms; eps = residual energy), DP2 summary (E_w, Sigma_w n_y x n_y) + build record
(B_w, A_w, G_w, b_w, rho_w). TP1: (1/2)tail_k <= L2_S <= Lin_S = (1/2)tail + (1/2)eps; always-on box reaches Lin;
TP1.1 Route (1) of TH4 as stated; TP2 sibling merges add Sigma, every level pinned from summaries, compressed summary
loses <= tau/2, TH3 instance (r) = compressed (3 vs 2); lifts recompute from N (depth NOT carried); TP2.1 two logits;
TP3 cost: value + directions O(n_y^3) no access to N, build linear in top width, switch selects stored weights.
Referee REFEREE-BP-2026-10-29.md: 12 findings, all fixed, not re-refereed (main one: build is not from summaries
alone). Checks model/tp_summary.py 149/0; Lean energy_split, trace_bound, topk_trunc, th3_summary in Recur.lean.
Card 3-26 (stream worktree took 3-25 for S-laminar). Open: Route (3) switching boxes, lift rule, KL.

## NOW (2026-10-29, stream track round 2, worktree .claude/worktrees/stream): OVERLAP AND CANCELLATION (plan ~/.claude/plans/you-did-some-great-radiant-matsumoto.md; paper F14, B.V.3, D.15 DV3; cards 3-23 ANSWERED, 3-25)
Two parallel tracks now (Shonu): ResNet/tree (worktree recur, B.H) and residual streams in general (this one).
Card 3-23 answers: S-purestream DECIDED ("Yes, study pure streams right now"); R-streamregion DECIDED + trunk T5 amended
(areas sentence; hash re-stored; plan approval = approval); S-readoutarea, R-normconst, R-cutclaim DROPPED as moot
(his words: "they don't really seem like decisions ... work out with the map itself"): S-readoutarea = block-read claim
at 0 (alias -> R-map); R-normconst via Lemma TV12 (eps re-set = one extra constant unread "battery" direction);
R-cutclaim via DM14 rewrite (D = sum of map areas -> claim names map objects; TV3 = area-finding, outside trunk). New
node S-laminar PENDING (card 3-25): he allows areas at an angle; DM11 still laminar.
Math: F14 (no norm: GL(d) invariance, angle meaningless; norm: angle acts via one scalar |Ah|); TV10 (no-norm cut fixed
by D alone; oblique = orthogonal); TV11 (overlap vs crowding: keep E exactly while ignoring D iff D cap E = 0); TV9 (his
quantum idea, build cancels: least-leak re-fit read = OLS of N's read on n'(Pi h); exact when reads affine in kept
content); TV9.1 copy example; TV9.2 lower bound (cut is a function of Pi e: KL to fiber mean; = TV7's c(s)); TV13
(no norm: logit change = exact signed sum of carried leaks with secant slopes; TV6 = it with sizes); DV3 conjecture
(N's own slopes, second-order remainder). Lean Stream.lean gl_read, norm_scale_read, crowd_no_read, battery_const,
ols_le. Check model/tv_cancel.py 0 fails/40. Measured (random dense, no norm): TV6/D_P 1.7e4 = softmax step 6 x signs
across blocks 2.4 x signs inside 31 x slopes-at-Lip+area sizes 33; B* (TV9) loss 2.5x below projection cut (39/40) but
its TV6 bound is not smaller (T grows with r'). So ~half the log-gap is lost cancellation.
After 2 referee rounds (REFEREE-BVO, REFEREE-BVO2; NOTE REFEREE-BV3 is the OLD B.V.2 report, do not overwrite):
TV14 PROVED (no norm, sigma' Lipschitz: signed sum with N's own slopes + pointwise quadratic remainder; beats TV6 only
when leaks small: family N_t t=.01 ratio 46-274 vs TV6 ~1e4; at full random leaks worse); DM15 completeness for area
maps (via D_j; groups = sums of orthogonal areas); Corollary TV6.2 = stream branch in T5 form (by construction, side
branch under S-laminar). Round-1 sandwich V_N(G+)<=V_area(G) was FALSE (referee's own suggestion), deleted: re-set
eps = one constant never-read direction, not a stream direction. R-normconst DECIDED by delegation (flagged to Shonu;
referee noted Rule 7 question); S-readoutarea, R-cutclaim DROPPED. New PENDING: S-laminar, S-smoothact, S-nonorm
(card 3-25). DV3 = TV14 with norms (route = proof outline). Next: prove DV3; per-member claims (map complete for every
zoom, referee finding 8); Lean Taylor step. Merged into main 2026-10-29 (LADDER 34). Worktree `stream` kept.

## Before (2026-10-29, parallel session): ONE RECURSION OVER A TREE OF AREAS (plan ~/.claude/plans/there-s-currently-another-session-virtual-puffin.md; paper B.H, D.16; LADDER 32; card 3-24)
Shonu: clean = gating, earlier coarse / later finer; think recursively and locally; prove recursion optimal; "zoom
recursion and depth recursion are the same thing" ([[feedback-one-recursion]]). Worked in worktree
.claude/worktrees/recur. The paper is not tracked by version control: copied in, merged back by an anchor script,
while the stream session edited B.V at the same time.
Found: DH1 tree map (districts in every layer, areas nested), DH2 branch claim (no unit outside a subtree reads it),
DH3 subtree groups + boxes (read top pre-activations; copy units uncounted); steps = creation / sibling merge / lift.
TH0 locality; TH1 split at ANY grouping, any depths (bookkeeping, like TM1); TH2 part 1: along sibling merges V_b is
monotone, and an exact top gives an exact (r) sequence (generalizes T60.1); part 2: lifts do not nest (chain
a -> g -> f zigzag: coarser exact, finer lossy); TH3: lossy sibling merge, 3 logits, D2: least 2, every (r) >= 3
(child drops direction e_b); TH4 conjecture in oracle form (summaries up the tree; one-node problem hard by T57).
Nodes R-tree, R-branch, S-plainff PENDING (card 3-24 replaces 3-11, 3-12). Referee REFEREE-BH: no false step; 3 false
statements, TH4 ill-posed, hidden plain-feed-forward restriction; all fixed, not re-refereed. Checks model/th_t61,
th_tree, th_lift, th_break (0 failures); Lean Coarse/Recur.lean. Next: ResNet with the tree (skips: S-plainff or the
stream of B.V), then a finer map.

## NOW (2026-10-29): THE STREAM AS AREAS (plan ~/.claude/plans/pasted-content-id-e1b3-okay-last-steady-waffle.md; paper F12, F13, DM11-13, B.V, D.15; LADDER 31; card 3-23)
Shonu: "let's try this subspace idea ... regions of the model rather than orderings" (after an explainer of the residual
stream; he liked "the stream tree is the shadow of the neuron tree"). Kinds of stream (F12, from code): ResNet relu on
stream, ConvNeXt depthwise filter per channel, ViT/GPT pure. F13: norm folding + rotation invariance (SliceGPT's).
DM11 areas = laminar subspaces; claims block-read, write; DM12 readout claim; DM13 energy claim. TV1: unread part acts
only through the norm; drop exact iff energy constant on P (essay's loud dim); TV1.1 keep the line. TV2: areas only
logits read: D_P <= (sum sqrt eps_ro)^2/4 (eps-only bound, any C). TV3: best readout claim = Eckart-Young on
P_perp U Q_S Sigma^(1/2) (reduced-rank regression), <= C-1 dirs, nested; minimises bound not loss. TV4 lookup.
REFEREE (REFEREE-BV-2026-10-29.md): no false step; 1 structural (TV1 part 2 + converse), 6 bookkeeping. KEY: Lemma TV5 (added): areas no block reads = subspaces of K = cap ker R_l; K = {0} for generic weights when sum q_l >= d (transformer MLP 4d) -> TV2-TV4 exact only on a null set; real next step = graded Conjecture DV1 (read leak beta, gains prod(1+g_l)). Fixed: per-area energy claims, node R-normconst (build re-sets norm eps), converse is about reads not D_P, essay's loud dim = varying energy = must keep. THEN Shonu: "you are not done yet since you only handled a very trivial zero case" -> B.V.2 graded case: TV6 (any cut: D_P <= 1/4(||omega|| + sum_l u^T M_l ||lambda_l||)^2, leak = read change from N alone, M = prod(I+T), no-norm T follows area graph, norm couples all areas), TV7 (v1 was an F1 rescaling [round 2]; v2 two-block had incomplete map [round 3]; FINAL: normed one block writes delta from D, logits read with ~1/delta; every claim of every area <= delta^2 incl. eps_br^w normalised by E|h_l|^2 (F1 + stream-scaling invariant); loss fixed 0.218; node N-amplify NEG), DM14 carried leak claim (from N,P,D; node R-cutclaim: claim names the cut) + TV6.1 (T5 graded form in letter, one fixed cut). TV8 (computed premise; coupling a+2eta|r|; no tighter on dense random nets). Check model/tv_graded.py 0 fails/40 dense nets K=0; bound loose: median 1e4 (no norm) to 1e7 (norm); operator-norm form no tighter and loses routing. Open: tightness (cancellation), clean nets, attention. Nodes S-purestream, R-streamregion (trunk wording),
S-readoutarea, R-normconst PENDING (card 3-23). Check model/tv_regions.py 0 fails/40 (first run's failures were a QR-of-dependent-columns bug in
the script); read-leak premise needed (bound fails 10/10). Lean Coarse/Stream.lean. Shonu's puzzle kept for later: the
essay's 4x stream crush lost 72->60% while 3b crushed 48x stayed readable (no accuracy given there); his guess:
training went badly; test = a no-training cut vs their retrained narrow stream.

## Before (2026-10-28): RESNET-18 MAP + ZOOM (plan ~/.claude/plans/okay-i-really-enjoyed-humble-allen.md; paper DM10, B.Y; LADDER 30; card 3-22)
Shonu: "create a good map. I think our maps kind of suck" of the torchvision ResNet, then "the full compressible, expandable
experience". Cache cache/r18 (z of last block fp16 + logits, 35k train + 10k sealed, 2.4 GB, DISK.md). Scripts in
curves/zoom/. FOUND (Part C H-map ResNet-18): no channel-level map of A.7 types at ANY layer: channel on for ~860/1000
classes, nn correlation 0.24 (no near-copies, c(S)=|S|), PCA 202/512 dims for 90%; strongest 0.1% entries = concept
(~61 classes) but logits use broad rest (KL 4.9). Paper: DY1 (block with skip; box = conv2 row + 1x1 skip projection,
torchvision BasicBlock class ties widths so "same layer types" only), TY1 (TM1 carries word for word), DM10 proportional
claim (output-weighted, copy/rescale invariant), DY2 merge rule (closed-form readers V_Q = sum s_u W_u), TY2 bound
D <= E^2/4 from F11 = meets T5 graded FORM ONLY on maps holding merge partitions (referee REFEREE-BY: 0 structural,
3 bookkeeping; perturbation bound; TY2/TY3 side branch under PENDING R-unitmap + R-propcomplete; exact-k DP fixed to
at-most-k; referee suggests pooled-output proportional claim next); TY3 switching = lookup + tree DP. Lean Merge.lean;
model/ty_merge.py 0 failures. Zoom (single group, k 512->1): merge*+refit ~= prune ~= random down to ~96; merge beats
both 48..16; trained boxes better everywhere (k=64: .749 vs 1.022); bound 1e4-1e6x loose. Pages: cards
https://claude.ai/artifact/D4nGkYa9e6Xgz9jLk9r8Lt, slider https://claude.ai/artifact/Y22GZTSNSVKJ26U85GusJx.
Card 3-22: R-unitmap (a keep / b InceptionV1 first [rec] / c strong-core+weak-spread claim), R-propcomplete (count merge
maps as complete [rec yes]); card 3-12 R-deep still open (rec B). Chrome extension was not connected: pages not
browser-checked (JS syntax ok).

## Before (2026-10-28): SIZE (plan ~/.claude/plans/okay-it-seems-like-quirky-mitten.md; paper B.N; LADDER 29; card 3-21)
Shonu: larger nets are easier to read (essay); formalize "larger + cleaner -> better" so small-net work carries to large.
Found: copy test (DN1: unit -> n copies sharing its in-weights) changes nothing real (TN1) but eps_dist, eps_shel grow by
(n-1) x u's term -> 'errors accrue' is our sums' fault. DE2 item 4 FIXED: judgment shares omega_k ~ |a_k| A^R_k (equal
split failed the copy test by up to 0.30). TN2 narrow splits of a kept unit never raise V_rob. TN3: E <= masses x weighted
means + D°_M + E_dead, nothing counts units; Lambda <= rho phi. TN4 under H-scale (quiet form) + H-mass (node R-mass PENDING).
Checks model/tn_copy.py 8/8; Lean Coarse/Scale.lean. Part C pc_mass.py: O flat ~22 (H-mass pt 1 holds), Lambda .455->.390,
fair phi .392->.401 (fair form fails): leaks cancel more at logits (9.6->13.3). Torch is ONLY in /opt/homebrew/bin/python3.11.
REFEREE-BN-2026-10-28.md: 3 errors 6 gaps, all fixed (not re-refereed): DM2/DM3 errors now LARGEST member term (not sum),
DM8 flat claim names a set (largest term), DM6 item 5 counts classes c(S) of positively proportional members; TN1.1 split out
(E for copies); TN3 pt 3 + D_dead; TN4 on H-scale MAP form (Lambda_M, max r_j, max sigma_u); ResNet last block = skip ->
outside D50'. Referee: 4 bookkeeping results, 0 structural; content = our definitions counted units.

## Before (2026-10-28): ROBUST OPTIMUM (plan ~/.claude/plans/okay-i-think-you-sparkling-sutton.md; paper B.E; LADDER 28; card 3-20)
Shonu: TM4's blow-up "seems to me like a mathematical error in actually finding the optimal subnetworks ... treat this as
the error it likely is" (not a design decision). Found: TE1 (free reader weights -> least loss depends only on the SPAN
of kept units' value vectors; a delta-leak is worth a full one; swing ~1/delta). Fix DE2 robust optimum: worst loss over
U(N, M) = tables between N's values and the clean table (leaks 0, shelter mean, flat mean, judgment projected to sum 1);
node R-wellposed PENDING (trunk wording change waits for Shonu). TE3 (= V_b at error 0), TE4 sandwich: clean split <=
V_rob <= clean split + E, E = sum error x swing (linear district, sqrt others); TE4.1 TM4 instances fixed. Lean
Coarse/Robust.lean (ce_osc etc.). Checks model/te_robust.py (A-D), te_robust_e.py (25 rows; FIRST version failed:
judgment entries had no clean value -> L^0 read the wiggle with 1/eps; fixed in DE2 item 4). Open: E uses swings of the
clean least build (T5 asks errors only); TE0 loose 50-130x on real logits.
Part C (torchvision ResNet-18..152, resnets.py): H-weak HOLDS (keep strongest quarter: KL .036 -> .011, top-1 loss <= .8,
falls ~3x with size); H-scale MIXED (dog/car pooled majority map: output-unit Lambda .455 -> .390, ~7 se; KL of removing
leaks .34 -> .14; activity shares flat ~.40; map far from clean, no specific top writer). Nodes R-weak, R-scale PENDING.
Preprocessing check: RGB 73.4 vs published 76.1 (BGR 57.8). Referee REFEREE-BE-2026-10-28.md.
REFEREE (2 errors, fixed): (1) my "graded clause met if B^0 swings = N's" was FALSE -> Proposition TE5 (referee instance:
N reads a 1/W difference with weight W, leaks W^-1/2, sends sqrt(W); all A.7 errors -> 0 but V_rob stays .0121 above the
clean split; B^0 loss 0.86/9.9/99.9; te_robust.py F) -> new claim type DM9 QUIET claim (what N's readers receive through a
district's leaks; = H-scale's Lambda numerator). (2) judgment clean value made rescaling-equivariant: h + (1-s)/(|K'| a_k);
DM4 one judgment per region, a != 0; DM8 now variance-normalised. TE4 upper bound = inf over e-near clean assemblies.
T5 graded clause still NOT met: terms (a),(b) carry the clean assembly's swings (zoom-dependent). Round-1 fixes not re-refereed.
Next: bound terms (a),(b); quiet errors on a finer real map (positions, middle layers); deeper layers (CM3).

## Before (2026-10-28, later): TRUNK T5 = mapped network (plan ~/.claude/plans/ok-we-made-nice-wiggly-lampson.md)
Shonu: assume the network is already mapped; every map term is math; finding the map is outside the trunk (see
[[feedback-mapped-network]]). T5 confirmed by plan approval, hash stored, node R-map DECIDED. COARSE-THEORY.md A.7
(DM1 claim/map, DM2 district+region, DM3 shelter, DM4 judgment, DM5 family; H-map, node R-mapreal PENDING), B.M:
TM1 PROVED (last layer, class (b) free logits: district+shelter+judgment at eps 0 => grouping splits exactly by
regions; nodes S-lastlayer, S-regiongroup PENDING, card 3-18), TM3 PROVED (T43 instances: district, shelter, symmetry
claims all eps 0 => exact rule from them gives P = NP; any fast-checkable claim set: else NP = coNP), D.14 CM1 (graded
split; blocked by unbounded box sensitivity), CM2 ("no silent blend" claim via T59 part 2). UNIT-BRANCH.md B.8: DA3
edge tie G (gain Fisher, row L11), TA12 PROVED (DU9: a softmax-unread unit ties by co-activity, I > 0), Remark B.9:
under decided reader rules the unit = region (activity), edge tie only for frozen readers. Lean Coarse/Map.lean.
Checks: units/tm1_split.py v3 box-by-box exact 1e-16 (v1/v2 optimiser too weak, logged); ta12_sub.py I = +0.046.
Part C: pc_map.py, dog/other map at mixed5b: district error 0.39, 7/1024 channels < 0.1 -> bad map at last layer;
next = map in 4c-4e or finer regions. LADDER 26. Card 3-17 update (tether no longer decides the unit).
REFEREE round 1 (REFEREE-MAP-2026-10-28.md): TM1, TA12, TM3 parts 1/3 correct. TM3 part 2 = PROVED CONFLICT with T5 as
worded (one-district map, region = all images, error 0 -> exact fast rule on T43 -> P = NP; new claim types can't help,
maps may omit them) -> card 3-19 (a complete maps [rec] / b drop fast / c keep). CM1 FALSE (shelter error delta, box
weight 1/delta) -> CM1' with weight bound W. DM3/DM5 squared norms, DM5 rational scales + orthogonality. "clean = map
with small errors" vacuous (withdrawn). New node S-cover. ROUND 2: TM3 part 3 restated for a poly-time CLASS A of
maps; conflict holds only under the polynomial-time reading (T5 has no time bound); card 3-19 now (a) complete maps
[rec] / (b) drop poly reading / (c) keep / (d) restrict inputs (card 3-5). CM1' also false -> withdrawn; new
Proposition TM4 (no bound on the split in errors alone). DM2/DM3 normalised per unit (rescaling-invariant); DM5 signed
permutations. Round-2 fixes not re-refereed.
COMPLETE MAPS (Shonu DECIDED card 3-19 (a): "the exactness promise applies only to complete maps"; trunk text updated,
hash stored; I extended it to the graded promise too, flagged). New: DM6 complete map (relation map x zoom: last layer,
district+shelter+judgment(not for R = P)+cover-or-flat+blend claim, two logits), DM7 blend claim (T59 pt 2 form),
DM8 flat claim; TM1 part 4 (flat uncovered units OK); TM4 part 2 (tiny district error amplified by a kept unit, weight
1/delta); TM5 PROVED (D40 + T43 mixes: blend error 0 <=> no bad pair <=> exact size-1 build, T59.2 formula) and
Cor TM5.1 (big group has no complete error-0 map -> TM3 conflict gone; single-group zooms exact); CM3 deep (route
TU16). Checks: model/tm5_blend.py 889 sets 0 exceptions; units/tm4_dist.py (1st run failed: N read j1; fixed);
tm1_split v4 flat OK (a unit varying on one region breaks the l_T formula, not independence). Part C pc_map2.py dog/car:
5b median channel share 0.27, 4e 0.30, few specific channels -> shared kept units varying on several regions = the open
case (S-cover). Next: claim type for shared kept units; CM2 general; graded clause open (not forced fork: TM4 is about
the split, not every rule).
REFEREE ROUND 3 fixed: DM7 now TWO-SIDED (one-sided failed with logit rows swapped -> conflict back); TM5 proved both
orders (tm5_blend v2, 1778 checks); TM5.1 map needs shelter claims + disjoint nested districts; graded-promise limit
REMOVED from trunk (not forced; beyond Shonu's words) -> open question for Shonu; DM3 variance-normalised; H-map needs
>= 2 label classes; CM2 non-triviality; CM3 needs R-deep. Trunk hash re-stored.
Shonu DOUBTED the InceptionV1 map result (rightly). pc_map3.py: 4c:447 car detector share 0.010; top dog/car writers
at 5b mostly <= 0.06 strong-firing. My "most neurons shared" was a misreading (medians over irrelevant channels);
withdrawn in Part C, LADDER 27. Real gap: map errors should weigh a unit by how much the output depends on it.
LESSON: before concluding from a layer-wide statistic, check it on a known unit (e.g. Zoom In's 4c:447) and on the
units that matter for the question.

Older history (before the T5 map work): [[project-curves-history]].
