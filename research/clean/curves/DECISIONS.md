# Decisions: the fixed goal and every step away from it

Started 2026-10-25 (evening). Plan: `~/.claude/plans/we-ve-been-doing-pretty-squishy-hanrahan.md`.

This file holds three things.

1. The **trunk**: the one goal every session serves. It does not change unless Shonu changes it.
2. The **restriction tree**: every place where the paper stops talking about the trunk and talks about something
   smaller or simpler. Each one is a **node** with an ID.
3. The **path** of every result: the nodes it sits under, so we can see how far it is from the trunk.

Rules (enforced by `tools/paper_lint.py` and `tools/ledger.py`, see section 7):

| Rule | Meaning |
|---|---|
| A restriction is a decision | No new restriction enters the paper without a node here. A new node starts PENDING. Only Shonu moves it to DECIDED. |
| Results under an undecided node are side branches | They stay in the paper as mathematics. They do not enter the chain of D.1 as steps toward the trunk. |
| Never widen the scope without the optimum inside it | Shonu's rule (2026-10-25). It applies inside a decided branch. |
| Technical premises are listed too | A premise like "this matrix is positive definite" is not a decision, but it must match a TECH entry, so nothing hides in a tag. |

---

## 1. The trunk

**Status: CONFIRMED by Shonu (2026-10-28), by approving the plan
`~/.claude/plans/ok-we-made-nice-wiggly-lampson.md`, which states this exact text and says that approval confirms it.
Amended 2026-10-29 (areas, node R-streamregion): Shonu answered card 3-23 ("I'm going to be fine with some like
subspace crossing") and approved the plan `~/.claude/plans/you-did-some-great-radiant-matsumoto.md`, which states the
added sentence and says that approval approves it.**
It replaces T4 (2026-10-27), which replaced T\*\*\*, T\*\* and the draft T\*. The old texts are kept below the table as
a record.

<!-- trunk -->
**Trunk T5.** Input: a network N with L hidden layers; base inputs P (a finite list of inputs with weights: the
inputs N was trained on, or a developer's); a map M of N on P; and a zoom chosen by the developer: groupings G_0, G_1,
..., G_m, from fine to coarse. A map is a finite list of claims about N and P. Each claim has one of the types named in
the paper. Each claim has an error eps >= 0 that is computed from N, P and the claim alone, in time polynomial in their
size, and eps = 0 exactly when the claim holds. No claim refers to a build, a target size or the zoom. The map names a
set of districts: disjoint sets of N's units, each inside one hidden layer, or, in a pure-stream network, areas:
subspaces of the stream that run through every block. An area group is replaced by k_g stream directions inside the
area. A grouping is a set of disjoint groups.
Each group is a union of the map's districts, with a target size k_g. Units in no group are kept. Each group of
G_(j+1) is a union of groups of G_j. Output, by a rule that reads N, P, M and the zoom and does not train: for each
level j, a network B_j of the same kind as N in which each group of G_j is replaced by k_g new units that read only
that group's units, with the least distillation loss D_P(B_j) = E_P KL(softmax(y_N(x)) || softmax(y_B(x))) among all
such networks. The rule may do a one-time computation up front for the zoom. After it, switching to any level is
instant: it costs time that does not grow with the size of N. The exactness promise applies only when M is complete
for the zoom (a condition between map and zoom, checkable in polynomial time, defined in the paper): when M is complete
for the zoom and every claim of M has eps = 0, the rule is an exact closed form. When some claim of M has eps > 0, the
rule comes with a proved bound that depends only on the list of the claims' eps values, and the bound is 0 when all of
them are 0. How M is found (by people, agents or a program) is outside the trunk.
<!-- /trunk -->

Shonu's choices that fix the trunk:

| Part | Shonu's words | Date |
|---|---|---|
| Goal | "Whole network, any size" (chosen over "one family, one coarse unit" and "both, family first") | 2026-10-25 |
| Building blocks | "Any new units" (chosen over "subset of old units" and "decide later") | 2026-10-25 |
| Clean, well-trained networks (R-least) | "I actually don't want to introduce error limits yet ... I want to assume networks are trained well and that they are clean ... this kind of goes into L3 and L4" | 2026-10-25 |
| A goal, not one best network (R-nest) | "in some cases you want to classify kangaroos, in other cases, you want to classify cars ... which ones you want to keep coarse, which ones you don't" | 2026-10-25 |
| No input data (R-root) | "I want to define a property of networks ... I don't want to require input diagnostics ... you can derive all the information you need to create a small network which satisfies a particular goal from solely the larger network" | 2026-10-25 |
| Clean is a spectrum | "if a network is perfectly clean, there are NO error bounds ... if it's this X or Y amount of difference from this metric ... this is what the result will be ... if a network is perfectly clean, I want a closed solution" | 2026-10-25 |
| One size per layer | "rather than having K, you would have like some sort of list like K1, K2, all the way to KN" | 2026-10-25 |
| Instant switching | "I do want instant switching as a capability that's supported by the mathematics ... I'm fine if, like, in order to define a particular goal, you do some calculations up front" | 2026-10-25 |
| A mix is an input again (REVERSES "No input data") | "I think I imposed an unnecessary restriction ... the generalist mix and a developer-defined mix are ultimately just mixes in general. And so you can just define that generally and do your theorems" | 2026-10-25 |
| The zoom | "this is kind of like a developer defining a zoom ... based on this sample ... you can actually degrade the neurons and fold up the neurons in a particular way ... choosing which neurons to fold in certain ways" | 2026-10-25 |
| Grouping as input (T4) | "given the fact we know which neurons we want to be coarse and which ones we want to be fine ... how can we quickly calculate from the larger network to get to this exact smaller network?"; on the draft: "yes i do agree with the trunk you've proposed" | 2026-10-27 |
| The developer chooses the degrade order (the zoom tree in T4) | "for now, a developer can kind of choose like, not just the grouping, but how the grouping degrades over time" | 2026-10-27 |
| New units read only their group (option (b) of card 3-10) | delegated: "whatever decision you choose, that'll be fine". I chose (b): under (a) the grouping would not shape what any new unit computes (card 3-10) | 2026-10-27 |
| A mapped network (T5) | "we could assume within the trunk that we already know how neurons are grouped ... assume that we're given a network which is already mapped"; "we can't just, you know, say, hey, this is mapped and not have a mathematical theory attached to that all the way to the fundamentals"; options chosen: "Yes, draft T5" and "Graded eps"; then approved the plan stating T5's text | 2026-10-28 |
| Complete maps (card 3-19, option (a)) | "i would say yes, the exactness promise applies only to complete maps". The trunk restricts the exactness promise only. (A first edit also restricted the graded promise, as my reading; referee round 3 showed TM3 does not force that, so it was taken back out. Whether the graded promise should also need complete maps is asked of Shonu) | 2026-10-28 |
| "Well-trained" | folded into the cleanliness distance eps(N) (a reading stated in the approved plan): with a developer's mix, N's training law is not P, and "optimal on its own labels" adds no condition (Lemma T24a) | 2026-10-25 |
| The image mix is an input (SUPERSEDED by "No input data", then restored by "A mix is an input again") | "the image mix Q is a declared input (generalist = training mix; developer = list of capabilities)" | 2026-10-23 |

Trunk T4 (2026-10-27; replaced 2026-10-28 by T5), kept as a record: "Input: a network N with L hidden layers; base inputs P (a finite list of inputs with weights: the inputs N was trained on, or a developer's); and a zoom chosen by the developer: groupings G_0, G_1, ..., G_m, from fine to coarse. A grouping is a set of disjoint groups. Each group is a set of N's units in one hidden layer, with a target size k_g. Units in no group are kept. Each group of G_(j+1) is a union of groups of G_j. Output, by a rule that reads N, P and the zoom and does not train: for each level j, a network B_j of the same kind as N in which each group of G_j is replaced by k_g new units that read only that group's units, with the least distillation loss D_P(B_j) = E_P KL(softmax(y_N(x)) || softmax(y_B(x))) among all such networks. The rule may do a one-time computation up front for the zoom. After it, switching to any level is instant: it costs time that does not grow with the size of N. When N is perfectly clean (eps(N) = 0) the rule is an exact closed form. When eps(N) > 0, the rule comes with a proved bound that depends only on eps(N), and the bound is 0 at eps = 0."

Trunk T\*\*\* (2026-10-25, late night; replaced 2026-10-27 by T4), kept as a record: "Input: a network N with L hidden layers, a mix P (a finite list of inputs with weights: the generalist mix or a developer's mix), and a list of sizes (k_1, ..., k_L), one per hidden layer, with k_l at most N's width at layer l. Output, by a rule that reads N and P and does not train: a network B of the same kind as N with k_l units in hidden layer l, built from any new units, with the least distillation loss D_P(B) = E_P KL(softmax(y_N(x)) || softmax(y_B(x))) among all such networks. For a fixed mix, the rule may do a one-time computation up front (the zoom). After it, switching to any other size list is instant: it costs time that does not grow with the size of N. When N is perfectly clean (eps(N) = 0) the rule is an exact closed form. When N is at cleanliness distance eps(N) > 0, the rule comes with a proved bound that depends only on eps(N), and the bound is 0 at eps = 0."

Trunk T\*\* (2026-10-25, late night; replaced the same night), kept as a record: "Input: a network N with L hidden layers, a goal G, and a list of sizes (k_1, ..., k_L), one per hidden layer, with k_l at most N's width at layer l. The goal says which of N's outputs matter and how much (for example: kangaroos, or cars, or a generalist weighting). Output, by a rule that reads only N and G (no input data, no training): a network B of the same kind as N with k_l units in hidden layer l, built from any new units, that is the best for G among all such networks. For a fixed goal, the rule may do a one-time computation up front. After it, switching to any other size list is instant: it costs time that does not grow with the size of N (for example because each smaller build is a part of the larger build, read off without new computation). "Best" is measured by a score that is defined from N and G alone. When N is perfectly clean (eps(N) = 0) the rule is an exact closed form. When N is at cleanliness distance eps(N) > 0, the rule comes with a proved bound that depends only on eps(N), and the bound is 0 at eps = 0."

Old draft trunk T\* (2026-10-25 evening), kept as a record: "Input: a trained network N, an image mix P, and a size k
(k units in each layer). Output, by a rule that reads N and P and does not train: a network B_k of the same kind as N,
with k units in each layer, built from any new units, such that the distillation loss D_P(B_k) is the least over all
such networks of size k. This holds for every k down to 1, and the networks B_k for different k nest, so that
switching size is instant." Part B.T of the paper was written against it.

## 1a. What T5 changes for the open nodes (2026-10-28)

Under T5 the map is an input, so a question of the form "is every clean network like this?" becomes "which claim type
says this, and what is its eps?". Verdicts below are my reading. Statuses do not change until Shonu says so.

| Node | Verdict under T5 | Why |
|---|---|---|
| R-group | changed by T5 | groups are now unions of the map's districts (T5 text) |
| R-district | becomes a claim type | H-district is the district claim (plus a judgment or hub claim), with an eps |
| R-indep | becomes a claim type | under TM1's premises regions can be coarsened independently, exactly; no graded form is known for the least loss (CM1 and CM1' are false, Proposition TM4); for the robust optimum a graded form exists under node R-wellposed (Theorem TE4, side branch), with the swings of the clean assembly in it (Proposition TE5) |
| R-cleannet | changed | the principles become claim types; "clean = some map has small errors" is empty (one district, every image), so clean must ask for maps fine enough for the zoom (card 3-19) |
| R-groupopt | becomes a claim type | H-group is a family claim inside a district |
| R-circuit | decided, groups (card 3-33, 2026-11-01) | T5 keeps groups in one layer; a circuit can enter as a claim about how districts connect; a group that is not a circuit's cross-section is first suspected as a wrong analysis |
| R-mixclean | answered in part by TM3 and TM5 | the first form of T5 conflicted with T43 under the polynomial-time reading (TM3); with complete maps (card 3-19, decided) and the two-sided blend claim, T43's big groups have no complete error-0 map in either logit order (TM5.1); other hard families are not checked |
| R-edgetether, R-outfaith | unchanged | they choose the class of boxes, not the map |
| R-readbias, R-readweights | unchanged (DECIDED) | readers' re-fits are part of the build class |
| R-kinds, R-freefam, R-recur, R-deep | unchanged | inside-district questions; each may become a claim type later |
| R-jobmix, R-listen, R-explaw, R-fullrank, R-dialfit | unchanged | about the mix, read tests or one measurement |

## 2. What is known at the trunk itself

These are the results closest to the trunk. Each row says what it does and does not give. Part B.T of the paper
(2026-10-25, late) works at the trunk directly, under the readings stated at its start. The proved conflicts below
hold for T\* under those readings, in particular "trained network" = any network of the kind (node N-anynet).

| Item | Path | What it says for T\* |
|---|---|---|
| Theorem T25 | R-cleannet, R-fold, NEG | Under T\*\*\*: for a perfectly clean network and a mirror-symmetric mix, the least build over any new units reads an input the network does not use (it equals the answer on the mix). Fold builds cannot. This forces R-fold (section 3-2). |
| Theorem T24 | NEG, R-cleannet (and R-obj for its D2 parts) | Under the new trunk T\*\*: a perfectly clean network (two mirror-image units) is optimal on its own labels for every input law, yet different laws, even mirror-symmetric ones, need different best builds. So "best on the training data" cannot be read off the network alone. This forces R-score (section 3-1). |
| Theorem T23 | NEG only (distance 0) | A proved conflict. For one teacher with three logits (six points in the plane), builds of width 2 come as close as we like to a perfect copy, but none reaches it, and their weights grow without bound on the way. So "the least" does not exist for every network. (The example is from Lim, Michalek and Qi.) |
| Theorem T20 | NEG only (distance 0) | A proved conflict. For one teacher of width 2 (seven points on a line), the least builds of width 1 and the least builds of width 2 share no unit. So "least at every k" and "the B_k nest" cannot both hold for every network. |
| Theorem T22 | NEG only (distance 0) | A proved conflict. Finding a least build of width 1 (even within a small stated eps) for every network and every mix is NP-hard, and under the Exponential Time Hypothesis has no algorithm with running time rho(d) S^{o(d)}. Even a relative error of 1 + 1/poly is ruled out. So "least", "fast" and "for every network and mix" cannot all hold, unless P = NP. |
| Theorem TM3 | R-map, R-group, R-mixclean, N-ny2, N-onehidden, R-cleannet, R-symN | A proved conflict with the first form of T5 (before card 3-19) under the polynomial-time reading of "exact closed form" (T5's text puts no time bound on the one-time computation). The map "one district = all layer-1 neurons, region = every image" has error 0, so on T43's instances T5 asks for a fast exact rule: P = NP. For a polynomial-time class A of maps on which a rule meets the clause, some no-instance has no small error-0 map in A (else NP = coNP). This forces card 3-19. |
| Corollary TM5.1 | R-map, R-group, R-mixclean, S-onehidden, S-twoclass, N-ny2, R-cleannet | The only positive result at the current trunk's exactness promise: on T43's instances, groups without a bad pair get a complete error-0 map and an exact formula (T59.2); the big group has no complete error-0 map, so TM3's conflict does not arise there |
| Theorem TM5 | R-map, R-group, R-mixclean, S-onehidden, S-twoclass, N-ny2, R-cleannet | On T43's instances, in either logit order: two-sided blend error 0 <=> no bad pair <=> exact size-1 build |
| Proposition TM4 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup, N-ny2, N-onehidden | The graded promise cannot be met through TM1's split: a tiny shelter or district error is amplified by a weight 1/delta. Not a conflict with T5 (a rule may solve those instances directly) |
| Theorem TM1 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup | Under T5: with district, shelter and judgment claims at error 0, the least loss of a grouping in the last hidden layer is the sum of its groups' least losses on their own regions; switching one group costs only that group. |
| Theorem TY2 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, R-unitmap, R-propcomplete | Under T5, on maps that hold a merge partition of every group (proportional claims, DM10): the merge rule has closed-form weights and D_P <= (sum of sqrt of all proportional errors)^2 / 4, 0 when they are 0, for any number of logits. A perturbation bound; it says nothing about the least build when the errors are large. On ResNet-18 the bound is 10^4 to 10^6 times the loss |
| Theorem T43 | N-ny2, R-cleannet, R-symN | Under T\*\*\* read with a polynomial-time zoom: for the detector networks (clean under the principles of paper A.5, all of them at one hidden layer) and mixes invariant under their block permutations, the zoom cannot run in polynomial time unless P = NP. The hard instance is carried by the mix. This forces R-mixclean (card 3-5). |
| Proposition T57 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | Under T4: one big group handed over in one step (all detectors -> 1 unit) is CLEAN-TRUNK of T43, so the developer's grouping does not remove the hardness when a group is big. |
| Theorems T58, T59 | R-group, S-twoclass | Under T4 (distance 0 apart from the two-logit clause): an AND group collapses exactly from the weights; an exact one-unit merge is one linear program, blocked by a silent blend in the mix. |
| Theorem T41 | (none) (distance 0) | Bookkeeping: the optimum is the least of finitely many pieces (activation patterns); an optimal pattern always exists, even when no least build does; a rule's error is at least the price of its piece. |
| Lemma T22a | (none) | Any labels on any finite list of points are the logits of some ReLU network with one hidden layer. This is what lets a hardness result about training become one about the trunk. |
| Theorem T19 | S-twoclass, S-onehidden, S-finite | For two logits, one hidden layer and a finite mix, a least build exists. Three logits: not always (T23). More layers: Conjecture D17. |
| Theorem T21 | S-twoclass, S-onehidden, S-finite | Same setting: the least splits into finitely many convex problems. For D2 an exact rule exists, with a cost that grows like n^{d k}. For KL the pieces are convex but transcendental. |
| Fact F9 | R-stat | At a stationary point of training, the weights do not fix the data's second moments. That P is an input of the rule is Shonu's choice (section 1), not a consequence of F9. |
| Proposition T1' | none | A positive sum of ReLU units that read one common field is not one ReLU unit of that field on an open set where two hinges are crossed. |
| Theorem T12 | R-obj | A floor on the squared logit error D2 for every map whose logits are affine in k numbers. |
| Theorem T14 | R-obj, R-sgate, R-gate, N-linear, N-onereadpos, N-finite | A narrower hardness result (linear chains with spatial windows, objective D2). T22 is the hardness result for T\* itself. |

The earlier text of this section said that "no exact fast rule exists for every network" was a belief. Theorem T22
now proves it, for the trunk's own builds and the trunk's own score, under P != NP.

## 3. The decisions the trunk's own words force (R-least, R-nest, R-root, R-score, R-fold: DECIDED 2026-10-25; R-foldopt: DECIDED 2026-10-26; R-mixclean: PENDING)

Before 2026-10-25 (late) this section offered three options (structure, approximation, search). They were my guesses,
not forced by any theorem, and they are withdrawn. The decisions below are forced by proved theorems. Their
options are exactly the premises of those theorems: to escape a theorem, one of its premises must go. R-least comes
first, because the other two speak of "the least build".

### 3-38. Does line I2 count under the Unique Games Conjecture? (2026-11-02; COARSE-THEORY.md B.Q.12; LADDER 52; nothing blocks on this)

Directional because: this changes the end condition of card 3-37. Line I2 says "unless P = NP".

```
proved with no conjecture:       a fast rule within a factor below about 1.36 gives P = NP (Proposition TQ30, k = 1)
proved under the conjecture:     a fast rule within any fixed factor gives P = NP (Proposition TQ31, k = 1)
at k = 1 with no conjecture:     every factor asks for NP-hardness of least-uncut at every factor, a known open problem
```

| Option | What it means | Gains | Loses |
|---|---|---|---|
| (a) Keep I2 as written | Only an unconditional proof counts | The test stays strict | I2 may stay open for reasons that have nothing to do with networks |
| (b) Count I2 under the Unique Games Conjecture | The conjecture is widely used, but it is open and it could be false | I2 is met now | "Impossible" would rest on an open conjecture, and you asked for proof |

My recommendation: (a). You asked for a proof, and I5 is the larger gap anyway, so the choice does not change the state today.

### 3-37. How row 3 ends: it passes on real networks, or it is proved impossible (2026-11-02; ANSWERED by Shonu; plan i-need-you-to-tingly-orbit.md)

Directional because: this sets what "closed" means for row 3, and so what the next sessions wait for.

Shonu (2026-11-02): "this does not have any utility unless it passes on real networks. i can understand if we prove its mathematically impossible to do so but we haven't proved it either way, and so we need to work until that happens." He approved the plan with the tests below, and did not change the factor 2 in P4.

This answers card 3-35 question 1: option (a), row 3 stays open. The roadmap's written end ("an exact theorem, checked, refereed") does not count by itself.

| Test | PASS: every line must hold |
|---|---|
| P1 | A claim type exists. Its error is computed in polynomial time. It names no build, size or zoom |
| P2 | At error 0, a closed form gives the least KL. It is proved, refereed and checked against a certified global optimum on small nets, hand-built and trained |
| P3 | At error above 0, a proved bound depends only on the errors, is 0 at 0, and is first-order tight (ratio measured) |
| P4 | On ResNet-18 and ResNet-50, at k = 14 and k = 38, on the trunk's real box (k units reading z), with a map from agents: the rule's build is at most 2 times its proved floor |

| Test | IMPOSSIBLE: every line must hold |
|---|---|
| I1 | A theorem on a family of networks, with every claim at error 0 |
| I2 | Unless P = NP, no fast rule is within any fixed factor of the least |
| I3 | The family is trained-like. A real training run reaches each small instance, and the hard tables are global minima of their own training objective |
| I4 | The family matches what ResNet tables measure: mean tail about .31, log range up to about e^39, KL not quadratic on most of the loss |
| I5 | The hard tables pass the essay's clean principles, read by agents and not only by my definitions |

Said plainly: one fixed network has one fixed least, so no hardness theorem can say that ResNet-50 itself cannot be solved. IMPOSSIBLE means that a family that looks like ResNet has no rule. Shonu may still object that real tables are not in that family. An IMPOSSIBLE result conflicts with the ResNet-50 end condition, so it returns to him on a new card with the theorem.

### 3-36. The general form of row 3 is proved hard. What should the exact promise at a squeeze mean? (2026-11-02; COARSE-THEORY.md B.Q.11; LADDER 50; nothing blocks on this)

Directional because: this sets what "exact" promises a developer at 0 < k < rank, after a theorem limits it. No theorem forces one answer, and option (c) moves off the least KL.

```
proved:        on networks whose every claim holds, the squeeze least is NP-hard (Theorem TQ23), at the accuracy about 1 / (8 m n)
not excluded:  a fast rule within a factor of 2 or 10
the trunk:     its "exact closed form" has no time limit, and R-mixclean (e) allows any cost. So the trunk is not contradicted
on real nets:  no claim found. Product tail misses badly. One Ward tree reduces the error from 1.81 to 1.43, far from 0
```

Update (2026-11-02, Propositions TQ30 and TQ31): option (c) is now limited by a theorem. On networks with every claim at 0, a fast rule within a factor below about 1.36 gives P = NP, and within any fixed factor gives P = NP under the Unique Games Conjecture. The second holds at any movement and any tail mass. So (c) needs a claim too. These tables are graphs, not shown to be trained-like.

Which promise should a squeeze carry?

| Option | What it means | Gains | Loses |
|---|---|---|---|
| (a) | Exact only under a claim that leaves the hard networks out, a structural claim on N (Corollary TQ23.2 says some such claim is needed for a fast rule) | Keeps "exact" and keeps it fast. It fits your "which claim would make this exact" | I do not have this claim. The two I measured miss real networks. It may not exist in usable form |
| (b) | Exact by an exponential one-time computation, which the trunk already allows | No new claim. No change to the trunk | At k = 14 on 1000 classes the search has 13790 free angles, against your "not a terrible amount of GPUs". And by Corollary TQ23.3 its answer need not carry a short proof at this accuracy |
| (c) | A proved bound within a fixed factor of the least, on every network | Hardness does not exclude it. It may reach real tables | It is a move off the least KL, and it is a tradeoff you have not chosen. No such bound exists yet: the best floor on a real table is a factor of about 850 |

My recommendation: (a), and keep working on the claim with agents reading the network. I would try (c) only if you say so. You do not need to answer now: you said intermediate results need no call from you, and work continues under (a) unless you say otherwise.

### 3-35. Row 3 has its first exact theorems, and real networks miss their premises (2026-11-01; COARSE-THEORY.md B.Q.9; LADDER 46)

Directional because: this sets what "exact" promises a developer, and node S-polescope is a narrowing left after the general attempt failed.

```
what is now proved:
   N's table is exactly pole-shaped, total written rank C/2     -> the rule writes the least build, in closed form
   any table, any k: a build passes a test of two numbers       -> that build is a proved global least
what real networks do (ResNet-18, ResNet-50, a trained CIFAR net):
   test ratio, should be <= 1:                                     1e3 to 1e160
   pole form against a free fit, k = C/2:                          over 100 times higher
```

| Question | What you decide |
|---|---|
| 1 | Is row 3 closed with these theorems, or does it stay open until a claim real networks nearly meet exists? |
| 2 | Node S-polescope: for now, the exact promise at C logits covers only levels whose total written rank (kept rank plus the sum of k_g) is C/2, and whose kept units lie in the span of the pair functions. Also, this branch drops the shelter, judgment and flat items of completeness: a pole claim alone is enough, so its districts say nothing. Accept this as a temporary narrowing? |

| Option | What it means | Gains | Loses |
|---|---|---|---|
| (a) Row 3 stays open. The next work is a dual point that is not linear in the projector, since the linear shape misses real networks, with or without the tail (Part C, DQ15) | Your rule stands: "a premise that trained networks miss badly is the wrong premise" | No detraction from the end condition | Session 2 waits, or merges with this, since its wall is the same one. I have no candidate yet |
| (b) Call row 3 closed as an exact theorem on its class, as the roadmap's row 1 words say, and carry the miss into session 2 by name | The roadmap moves | Keeps the sessions in order | The row says closed while no exact statement holds on any real network. You may read this as a detraction |
| (c) Promise exactness only where the rule's own test passes, and report a fit with no optimality claim elsewhere | The rule certifies itself on each zoom (Corollary TQ18.1) | Honest, cheap to ship | Today the test passes on no real network, so the promise is empty in practice. I do not recommend it as an end state |

Not an option: a cap on weights (TQ13's limit). You removed it on card 3-33.

My recommendation: (a), and accept S-polescope as temporary. Reason: the exact theorems are real mathematics, but your end condition asks for the least on the whole of ResNet-50. The obstacle now has a precise form: a dual point built linearly from the projector must stay a set of probabilities, and on real rows what is left undone is far larger than the probabilities it must not overdraw. I expected to lump the tail away (Conjecture DQ15). I measured that before proving it, and it misses too: on a typical row, by about 500 times at best on a table, and by 30000 times or more on the read. So I do not have a claim that real networks nearly meet, and I do not have a candidate.

Said plainly on the trunk's words: a pole claim's counts |J_g| are the sizes. So "No claim refers to a target size" is met in its letter only. If you read this as against the trunk, the claim goes, and with it T5's exact promise for a squeeze at more than two logits. Theorem TQ17, as mathematics about a table, and Theorem TQ18, inside the one-time computation, stay either way.

Corrected the same day (LADDER 47): this card first said the exact class goes against the essay. It does not. Theorem TQ17's family is the essay's toy's world, where nothing is related to anything. Real fits hold relatives on one pole, at every k, in every object I measured. You do not need to answer this card: you said it is an intermediate result, and it stays open.

### 3-34. The trunk's box sits about 7 times above the floor of any k-number code at the top (2026-11-01; nothing you must answer)

Directional because: option (b) below would change the trunk's words. You said: "if there's a distinction between a group and a circuit, we might be analyzing the network a little wrong". The plan promised to bring you this number.

```
the trunk's box:   512 pre-activations z -> k units relu(m . z + beta) -> 1000 logits     14 units: 1.23 nats    38 units: 0.34 nats
free codes:        any k numbers at each photo                        -> 1000 logits     14 units: 0.18 nats    38 units: 0.047 nats
```

What the gap means: a last-layer group is one linear read of z, with a gate. The free codes appear to encode which class (0.184 against a table of class rows at 0.171), and they let the layers below be any size. So the floor is the cost of writing class identity in k numbers, an oracle, not a circuit of N. The gap is the cost of computing it from z with one read. How much of it N's own circuits can recover, in fewer units, is unmeasured.

| Option | What it means | Gains | Loses |
|---|---|---|---|
| (c) a circuit is a stack of groups, one per layer | Inside the trunk as written. A level replaces a group in several layers, each with its own k_g, and each box reads the box below through N's weights (Definition DH3 item 3). A hypothesis, and it fits your reading of a group as a circuit's cross-section: whether N's own stack can compute what the top needs in fewer units is the open question | no trunk change; nonlinear codes at the top, every unit counted | the theorem for a stack is new work, and the last block is near a pass-through (the agents: about 0.88 of a last-layer channel's input is its own skip), so the stack must reach into stage 3 |
| (a) stay at one last-layer group | Report the least of the box as row 3's answer | simplest | leaves the gap, about 7 times |
| (b) coarse circuits with new units in any layer | Definition D60; changes the trunk | most freedom | new units with no N row to read through; much further from a closed form |

My plan unless you say otherwise: test (c). It needs no change to the trunk, and it could recover anywhere between none and all of the gap.

### 3-33. The three decisions that are really yours (triage 2026-11-02; ANSWERED 2026-11-01 by Shonu, in his own words below; none blocks row 3)

**Answered.** Directional because: each answer sets what the product promises.

| Node | Shonu's words | Recorded as | What follows |
|---|---|---|---|
| R-mixclean | "I'm fine with the one-time computation up front being non-polynomial as long as it is exact and I'd also like it to not take a terrible amount of GPUs or anything because then that defeats the purpose" | (e), DECIDED | Hardness in the number of areas is no longer a wall. Said plainly: the time limit is gone, but no exact method exists yet (Lemma TQ9 did not finish on a 4-class net) |
| R-circuit | "Let's do groups for now ... if there's a distinction between a group and a circuit, we might be analyzing the network a little wrong, and that might lead you to something better" | (a), DECIDED, with his check as a standing rule | A group should be the cross-section of a circuit. If they differ, first suspect the analysis. Finding on 2026-11-01: my proxy areas (class clusters) are groups of nouns, not circuits |
| R-wellposed | "ultimately we should just be aiming for whatever the optimization or the optimal distillation function is going for, which I believe is least KL ... we already have something that's well formulated. And so I think that should be what we're aiming for, not something else" | NO, node DROPPED | The trunk stays as written. B.E and B.N stay as mathematics, off the trunk. The blow-up of Proposition TM4 stays a wall for session 2, to be met with better claims (first route: errors that measure where a unit is on, Proposition TE1). If a theorem shows a true conflict, it comes back as a card with that theorem |

My error on item 3: the card asked about something the trunk already settles. Mapping (his words, same day): "you need to
have subagents figure that stuff out rather than trying to develop a one size fits all sort of algorithm". This is not a
node: the trunk leaves map finding outside the math. Rule in `CLAUDE.md`.

The original card:

On 2026-11-02 I triaged all 40 pending nodes by your rule (technical = mine, decided by delegation with your quote
of 2026-11-01; directional = yours). 35 were technical and are now DECIDED by delegation; R-recur and R-deep were
merged into R-tree and are DROPPED. These three change what the product promises, so they stay with you:

| # | Node | The question in plain words | Options | My recommendation |
|---|---|---|---|---|
| 1 | R-mixclean | Must the one-time computation for a zoom finish in polynomial time? | (e) no time limit, the trunk as written; (a)-(d) narrower promises (see the node row) | (e): keep the trunk as written; complete maps (card 3-19) already keep the exact promise away from the hard cases |
| 2 | R-circuit | What does a developer coarsen: groups of units inside one layer, or whole circuits across layers? | (a) groups in one layer (T4); (b) circuits across layers, replaced by a coarse circuit; (c) like (b) but new units only in the layers the circuit uses | (a) for now: the tree of areas (R-tree) already lets a group span a subtree; (b) is a later widening |
| 3 | R-wellposed | Change the trunk's word "least loss" to "least worst-case loss over tables between N's values and the map's clean values"? | yes / no | yes: at error 0 nothing changes (TE3), and it fixes the blow-up you called "a mathematical error" |

### 3-32. Squeezing (row 3): what the test found, what is proved, and your calls (COARSE-THEORY.md B.Q; ANSWERED 2026-11-01 by delegation: Shonu, "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make")

**What happened (2026-10-31).** The plan's deciding lemma (L-sep: "no unit serves two separated sub-capabilities")
is FALSE. `model/ts_sep.py`, one unit, a = 1 (the outputs of the clean network of TQ2 part 3, every claim at error 0 after the referee's fix of its inputs):

| Confidence of each area in its own classes | Mass outside its own classes | Best unit for one area | One unit for both areas |
|---|---|---|---|
| none | 0.53 | 0.0546 | 0.0505 |
| medium | 0.055 | 0.0986 | 0.0088 |
| high | 0.003 | 0.1027 | 0.0005 |

Read as a network: each area's own offset pushes the other area's classes down. Then one output direction means "which
breed" on dog photos and "which make" on car photos. So the form of row 3 is PACKING: a squeeze needs about as many
units as its largest sub-capability, not the sum.

**What is proved, with honest labels.**

| Result | Label | Meets your bar? |
|---|---|---|
| TQ1 part 1: the group's least >= the sum of each area's own least | lower bound | a piece of the answer |
| TQ1 part 2: a packed build within the sum of the active-class summaries | ONE-BUILD | NO: its lower side is 0, and the summary is 0 only when nothing is lost |
| TQ2 part 2: every squeezed group loses something (least > 0) | property | a fact, not an impossibility |
| TQ2 part 3: one unit for two areas beats every unit for one area | NEG | kills the tree knapsack |

**Withdrawn (referee).** I first wrote "row 3 is never exact, a fact of softmax, not a choice". That was wrong. My
proof showed only that MY error formula is never 0. Row 1 is exact with a loss above 0, so a positive loss does not
rule out an exact form. By your rule, that "impossibility" was my definition's fault.

**Measured on ResNet-18 (Part C, proxy areas).** The areas favour their own classes (outside mass 0.07 to 0.2). My
packed build is weak there. A fitted head of the same packed form keeps 72% to 90% of the information. So the network
allows cheap squeezing. My construction is what falls short.

**Your calls.**

| # | Item | My recommendation |
|---|---|---|
| 1 | Node R-active: maps name a class set K for each node of a squeezed group; the active-class summary eps_act (Definition DQ1) is computed from N. It is a summary like the information table, not a claim | accept |
| 2 | Node S-squeezecomplete: TQ1's packed build asks each area to move its own classes in at most k directions. The trunk puts completeness only on the exact promise. Here it sits on a bound, so it needs your yes | accept for now |
| 3 | "Useful" for a bound: first-order tight (bound / true gap stays bounded as the errors go to 0), measured on trained networks | accept |
| 4 | Open question: does some claim type give row 3 an EXACT closed form? Nothing here shows it is impossible. Proved since (Theorem TQ3): as the areas become confident, the least of a squeeze over always-on boxes tends to the sum of each area's own-block least, from both sides, even where that least is > 0 | work on it next |
| 5 | A real question for you (not told): what remains at finite confidence is one finite problem, how the shared directions line up. At a fixed output span the fit inside each area is an exact convex program (as TI3 already reads the trunk); an area squeezed below its own rank is again a non-convex choice of span. Does "directions in closed form (or from one small fixed-size problem) plus an exact convex fit inside each area" count as the trunk's "exact closed form"? Your answer decides whether row 3 can ever be marked exact in the trunk's words | your call; I recommend yes, with the fixed-size problem stated in the paper |

(Updated 2026-11-01, Theorem TQ17, Corollary TQ17.1, Theorem TQ18: the first exact cases exist, at loss above 0; see card 3-35 and LADDER 46.) What stayed open at the time of this card: the least of row 3 itself (no OPT-KL result yet); first-order tightness (Conjecture DQ1, for
comparable confidence only; the referee sketches a failure for uneven confidence, with a fix by per-area scales).

**Answered (2026-11-01), by delegation.** Shonu's words: "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make".

| Item | What I decided | Why |
|---|---|---|
| 1 R-active | accept (DECIDED by delegation) | K_w is the analyst's choice; eps_act is a summary of N |
| 2 S-squeezecomplete | accept for now (DECIDED by delegation) | the left-out case is the one-area squeeze, a different object, still open (DI1, DQ3) |
| 3 "useful" = first-order tight | already Shonu's standing rule (ROADMAP.md) | not a new decision |
| 4 exact form for row 3 | the work itself; still OPEN | partial only: Theorem TQ5 (2026-11-01) is LIMIT-1, not OPT-KL or EQ: along N_m the always-on least is e^-m times the leak least of Definition DQ5, no rate (REFEREE-BQ3 finding B7) |
| 5 meaning of "exact" | recorded at plan approval, refined by me: exact = a value with a PROVED matching lower bound (closed form, or a convex program proved tight under a named claim), plus exact convex fits; the number of unknowns is not the test | plan `i-think-we-are-agile-raccoon.md` |

### 3-31. S-keepdrop: what a coarse level may do when the target is the least KL (COARSE-THEORY.md B.I; ANSWERED 2026-10-31: (c), node DROPPED, Shonu: "you can only keep them or drop them. That's the same as pruning ... we need to completely fix this third row and figure out an optimal thing"; the work is ROADMAP.md session 1, Part B.Q, card 3-32)

**What I found (2026-10-31), plainly.** You asked for the least KL everywhere, deep networks included. Part B.I has
three answers, one per group size:

| Group size k (units the group keeps) | Least KL | Label |
|---|---|---|
| k = 0 (the capability is removed) | closed form: on each area, the log of N's mean probabilities. The table passes up the tree: merge = add, lift = add + one KL term per child (Theorem TI1) | OPT-KL |
| k >= rank of the group's table (the capability is kept whole) | at most a quarter of the linear-read error; 0 when clean (Theorem TI2) | OPT-KL |
| 0 < k < rank (the capability is squeezed) | at the time of this card, no closed form (updated 2026-11-01: exact cases exist, card 3-35). The squared-loss eigenvector answer need not be the KL answer (Proposition TI5). To second order it is the eigenvector answer in the Fisher metric of the area's mean output (Theorem TI4). A computed example says the best unit may even switch off inside its area (Conjecture DI2). It may be NP-hard (Conjecture DI1) | open |

```
example (Proposition TI5 part 2): three classes, one unit, every claim at error 0
  mean output uniform        -> KL least ~ a^2/12, every direction ties (squared loss and KL agree)
  mean output (1/2,1/4,1/4)  -> KL least ~ a^2/16, along "class 1 against the rest"
                                the squared-loss build can pick (0,1,-1) and lose 3/2 x
```

Why it matters: a level of a zoom is least for KL, in closed form, when every capability on it is either removed or
kept whole. That is node S-keepdrop. It is a narrowing: a level may not squeeze a capability into fewer units than
its rank.

| Word | Meaning |
|---|---|
| rank of the group's table | how many independent output directions the capability moves (at most classes - 1) |
| Fisher metric | the way KL weighs an output direction: moving probability between likely classes counts more |

**Question (S-keepdrop): may a coarse level only remove or keep whole capabilities, for now?**

| Option | Gain | Cost |
|---|---|---|
| (a) exact global optimisation over output directions (and switch points) as the rule | the true least, any k | not a closed form; the search grows with the number of classes; maybe no fast rule exists (DI1) |
| (b) keep-or-drop levels (node S-keepdrop) [rec, now] | every level least for KL in closed form today (Corollary TI2.1), with every claim except the linear-read claims at error 0 and a gap of at most a quarter of the linear-read errors; the ResNet example can run on it | a narrowing: no squeezing |
| (c) find a claim type that makes squeezing a closed form [rec, next research] | the trunk as written | research; TI4 says where to look (areas with one mean output) |
| (d) the second-order rule of TI4 as a labelled stepping stone | usable for squeezing | an approximation: off your bar |

My recommendation: (b) now, so the ResNet work has exact KL levels, and (c) as the next research step. Not (a) for
1000 classes. (d) only with its label.

### 3-30. R-obj: the least distillation loss is KL; the closed forms are for the squared loss (COARSE-THEORY.md TY5; ANSWERED 2026-10-31: (a), node DROPPED, Shonu: "specifically to find the optimal KL ... for everything, including deep networks"; the work is Part B.I and card 3-31)

**What I found (2026-10-31), plainly.** Our deep-layer closed forms (Theorem TP1, Corollary TL5: "the optimal small
network at every depth") are least for the squared error of the centred logits (D2), not for the distillation loss
(KL). I chose D2 on 2026-10-25 (node R-obj, RETRO). You never decided it. Proposition TY5 shows what that costs:

```
a group replaced by nothing (k = 0), on one area:
  KL-least logits  = log( mean of N's softmax )      <- the true optimum, a closed form
  D2-least logits  = mean of N's logits               <- what our closed forms do
  extra loss       = KL( mean softmax || softmax(mean logits) )      exact; 0 only if the two agree

example, a network with EVERY claim at error 0:  two points, softmax (0.9, 0.1) and (0.5, 0.5)
  mean softmax (0.7, 0.3)   vs   softmax of mean logits (0.75, 0.25)   -> extra loss 0.0064 > 0
any size k: the D2 build is within a factor 1 / (2 mu) of the KL least, mu = smallest probability on the way
  (mu <= 1/1000 on ImageNet): a confidence term, not a cleanliness term
```

Why it matters: your bar is the least KL loss, with errors that depend only on how clean N is. The D2 closed forms
miss it even when N is perfectly clean. So they do not meet the bar.

| Word | Meaning |
|---|---|
| KL | the distillation loss: how far the small network's probabilities are from N's |
| D2 | half the squared distance between the two networks' logits, after removing each one's mean |
| mu | the smallest probability the softmax gives any class on the straight line from N's logits to the build's |

**Question (R-obj): which objective may the closed forms use?**

| Option | Gain | Cost |
|---|---|---|
| (a) KL only [rec, as the target] | the bar exactly: the true optimum | for a group with k >= 1 units the KL least is a reduced-rank logistic regression with soft targets; no closed form is known, so new claim types or new math are needed. Until then the deep results are open problems, not answers |
| (b) keep D2 as a stepping stone, always labelled OPT-D2, with TY5's gap stated next to it | the current closed forms stay usable, honestly labelled | not the optimum; the gap does not shrink with cleanliness |
| (c) keep D2 as if it were the optimum | none worth having | breaks your bar |

My recommendation: (a) as the target, with (b) only as a labelled stepping stone. The next math step under (a): the
KL least of one clean group (TH1's L_S under linear-read claims). It is an explicit small problem in N's own
numbers. The open question is whether it has a closed form, or a claim type that gives one.

### 3-29. S-skipff and R-jointgroup: one group for a capability (COARSE-THEORY.md B.V.7, B.Y.2; PENDING)

Your idea (2026-10-31): a developer coarsens a capability, so neurons and stream should be one cut, with no extra
loss from the stream. What the math says:

```
pure stream (transformer-like, Theorem TV19.2):
  coarse neurons write at most sum k_S directions  ->  cut the stream to what is written: NO extra loss (exact, any loss)
  needed width = dim E_N + sum k_S,   E_N = first write (embedding) + biases + kept neurons' writes
  a full embedding or many kept writers  ->  E_N fills the stream  ->  nothing free (Proposition TV19.4)

ResNet (Theorem TY4):
  a ReLU acts on every stream channel  ->  a stream channel IS a unit
  capability = one subtree group: inner channels + stream channels of several blocks  ->  the least KL loss splits
  into one problem per group, exactly, when the map's claims hold (a reduction: each problem is still open for KL)
```

Why it matters: your "one error budget" is a theorem in both kinds of network. In ResNet it needs no change of the
trunk's wording. In a pure stream it needs one, but only for a stream narrower than the written width.

**Question 1 (S-skipff): may Part B.Y.2 study feed-forward networks with skips and channel units (a ResNet after its
stem)?** It is wider than plain feed-forward (S-plainff) and covers torchvision ResNets. It leaves out attention,
norms on the stream and max pools inside the network.

| Option | Gain | Cost |
|---|---|---|
| (a) yes [rec] | the split of the least (EQ: the optimum reduces to one small problem per capability) reaches ResNets, so next session's run tests theorems about the optimum | a scope you did not name in those words |
| (b) no | nothing new | the ResNet run would test nothing proved |

**Question 2 (R-jointgroup): in a pure stream, may one group hold units and stream directions together?**

| Option | Gain | Cost |
|---|---|---|
| (a) yes, later [rec] | a stream narrower than the write budget, where capabilities must share stream directions; that least is a new optimum | a change of the trunk's wording; nothing is proved for it yet |
| (b) no | the trunk stays as it is | the stream is free only down to dim E_N + sum k_S |

### 3-28. S-purestream: leave pure streams for attention now? (COARSE-THEORY.md B.V.4 to B.V.6; PENDING)

**Where the stream track stands (2026-10-29).** In pure-stream networks (one position, MLP blocks, norms) the math is
now general: any Lipschitz activation (ReLU included), both forms (with and without norms), areas at any angle, and
one zoom-free map that serves every zoom (Theorem TV18, Corollary TV18.1). The open items left inside pure streams
are mathematical, and I can work on them without you: a tight remainder for large cuts (on trained networks the
signed first-order sum is within 3 to 7 times the loss, but the remainder bound is not), and signed claims that
name one member.

**The question.** You decided S-purestream with "Yes, study pure streams right now". The networks Cale ships are
transformers: their blocks also mix positions (attention). Reading the stream math on them needs attention.

| Option | Gain | Loss |
|---|---|---|
| (a) next round: attention, one head first, then many [rec] | the math reaches the networks Cale ships; B.V's cuts, leaks and member claims are the natural objects to carry over (an attention head reads the stream through a norm and writes back, like an MLP block, but its read mixes positions) | a new class of networks; the first results will again be exact cases |
| (b) stay in pure streams one more round | tighter remainder for large cuts; signed member claims | the gap to real transformers stays |

My recommendation: (a). It widens the class, but it reverses your "right now", so it is your call.

### 3-27. R-oninside: the on-inside claim, so that a lift moves units one layer down (COARSE-THEORY.md B.L; PENDING)

Your words (2026-10-29): the small network "is literally existing in the network ... an interpretable object which
we understand how it was formed", so the failures should be explained "based not just on the mathematics but on the
network itself, which can point us to a different solution".

What the network shows in the two broken cases:

```
TH3 (a merge):   area 1: u_a -> e_a,  u_b -> e_b        area 2: u_c -> e_b
                 child 1 alone: e_a is louder (4 > 3), so it keeps e_a
                 both areas:    e_b is louder (3 + 3 = 6), because two neurons REUSE it
                 -> the child cannot see the reuse; only the readers (the logit weights) show it
```

Why it matters: when the merge is no bigger than any child, naive step-by-step can lose only when siblings share
output directions (reuse) and a child dropped one (Theorem TL1, a sufficient condition).

```
TH2 part 2 (a lift):  x_1 -> a_1 -> g -> f_1, f_2, f_3 -> gap 0, 1, 0, 1      x_2 -> a_2 (nobody reads a_2)
                      g, a_1, a_2 are ON (pre-activation >= 0) at every input, so they act as the identity
                      -> every finer neuron is a neuron one layer down (Theorem TL2)
                      -> a lift with room for every finer neuron can only LOWER the loss; the box at G_3 reads past g
```

Why it matters: a lift by itself is not a coarsening. Only a smaller size is. "Lifts do not nest" is normal.

| Word | Meaning in the network |
|---|---|
| on-inside claim | inside a node's area, the node's neurons never switch off, and its gate (judgment) neurons are flat |
| its error | how much of each neuron's pre-activation lies below 0 inside the area, and how much the gate varies there |
| condition (K) | every output direction the merge keeps is, inside each child's areas, one that child already kept |

| Result | Plain meaning | Status |
|---|---|---|
| Theorem TL1 | a merge built from the children's box neurons (of the kind TP1 builds) is least at error 0 when (K) holds, and within half the linear-read errors of least otherwise. (K) is read from the summaries, the sizes and the children's chosen directions (from the summaries alone when eigenvalues do not tie at a cut). It holds when every child lost nothing, or when the children write orthogonal output directions and the merge is no bigger than any child. It is a sufficient condition only | proved; check `model/tl_netread.py` part A |
| TL1 part 4 | TH3 is a case where (K) fails | proved |
| Theorem TL2 | a one-layer lift through neurons that are on inside their area (gate flat) rewrites every finer neuron one layer down; when the lifted box has room for all of them (k >= n_j) the least loss does not rise, for KL and D2 | proved; check part B |
| Corollary TL2.1 | TH2 part 2's instance meets TL2's premises, so its order (exact at G_3, lossy at G_1, G_2) is the normal one | proved |
| Theorem TL3 + Corollaries TL3.1, TL3.2 | deep layers: when inside every area the neurons stay on, every gate is flat on its area and flat on its negative side, the whole subtree is a LINEAR function of its top neurons; this needs the top layer to carry, as flat directions, the area code of every finer gate (on generic points it leaves no split below the top) | proved; check `model/tl_deep.py` part A |
| Proposition TL4 | a lift adds the children's tables plus one between table: the votes of the gates it swallows (law of total variance) | proved; check part B |
| Corollary TL5 | the optimal small network at any depth, for such subtrees (no switching below the top except along flat 0/1 directions of the top layer): per group, k neurons on over the whole area, reading the top neurons along B^T q, writing q (top eigenvectors of the table); loss = 1/2 x (energy - top k eigenvalues); tables by add (merge) and add plus between (lift) | proved; check parts C and D (a local optimiser never beat it) |
| Theorem TL6 | with errors: each node's linear-read error is bounded by absolute defects carried up by N's weight norms and multiplied by 1 / (gate margin) at each gate; this bounds the gap to the least only when the errors sit in on-inside claims and in shelter claims below the groups' tops; the bound holds but is loose (10^4 to 10^9 x on a perturbed synthetic net) | proved; check part E |
| open | a tight bound with errors; gates that read features varying inside the area (real networks) | open |

**Question (R-oninside): accept the on-inside claim as a claim type?**

| Option | Gain | Cost |
|---|---|---|
| (a) yes [rec] | the depth step of the one recursion becomes "move each neuron one layer down", exact where the layer passed is linear in its area. It is a claim on N, not a limit on the box class | it says nothing where neurons switch inside their own area |
| (b) no | nothing new to accept | TL2 stays a side result |

My recommendation: (a). The on-inside claim is now the main premise of the deep-layer result (TL3 to TL5), not
only of TL2. The next step: a tighter graded bound, and measuring the claims' errors on a real network (Part C).

### 3-26. R-linread: the linear-read claim, so that summaries can be passed up the tree (COARSE-THEORY.md B.P, D.16; PENDING)

Your words (2026-10-29): "I'd like to try to prove a very similar thing, but now passing up the summary", with the
summary cheap to make and to use, and instant switching kept in the running.

The problem (Proposition TH3): child 1 keeps the one direction that is best for itself and drops the direction its
parent needs. The fix: each node passes up a table, not its best neuron.

| Word | Meaning in the network |
|---|---|
| summary of a node | one number (how much the outputs move inside the node's area) and one table with a row and a column per logit (how much a box could copy in each output direction) |
| linear-read claim | inside the node's area, after the free terms (a constant and the kept judgment neurons), the centred logits are a linear function of the pre-activations of the node's top neurons |
| its error eps_lin | the part of the outputs' movement in the area that no linear function of those pre-activations copies |

```
child 1 table: e_a 4, e_b 3        child 2 table: e_b 3
merge = add the tables:  e_a 4, e_b 6
keep the top k = 1 direction (e_b)  ->  loss 1/2 x (10 - 6) = 2   (the true least; the old step-by-step gave 3)
```

What the math says (Part B.P; referee REFEREE-BP-2026-10-29.md, all findings fixed, not re-refereed):

| Result | Plain meaning | Status |
|---|---|---|
| Theorem TP1 | for one group: least loss is between (1/2) x (the eigenvalues below the top k) and Lin = that plus eps/2. A box of always-on neurons reaches Lin. At eps = 0 it is a closed form, and no box of any shape does better. When k is at least the rank of the summed table, the lower side is 0 and only "least <= Lin" is left | proved; check `model/tp_summary.py` |
| Corollary TP1.1 | among boxes whose neurons stay on inside their areas, Lin is least for ANY eps (Route (1) of TH4, as it was stated) | proved |
| Theorem TP2 | a sibling merge adds the tables; every level is pinned by the tables alone, in any order; passing up a cut-down table loses at most half of what it cuts. In TH3's instance the old step-by-step is exactly such a cut-down table (3 against 2) | proved |
| Corollary TP2.1 | two logits: a merge of size 1 loses nothing beyond its children and its new nodes, when least boxes stay on (Route (2) of TH4) | proved |
| Proposition TP3 | cost: one pass of N over the inputs plus one small least-squares solve per node. The least loss and box directions of a level: about n_y^3 operations (n_y = number of logits), with no access to N. Writing the box weights: linear in the width of the group's top and judgment neurons. Box neurons nest in k. A switch selects stored weights (as in TY3) | proved |

What it does not do: a lift (a step to an earlier layer) recomputes the higher node's table from N in the one-time
pass. So the recursion is proved for sibling merges, not for depth.

**Question 1 (R-linread): accept the linear-read claim as a claim type?**

| Option | Gain | Cost |
|---|---|---|
| (a) yes [rec] | exact closed form at eps = 0, and a gap of at most eps/2 otherwise, for the objective D2. The gap is graded in the linear-read errors only, with every other claim at error 0. It is a claim on N, not a limit on the box class, so the least is still taken over every box | it is useful only where eps_lin is small: nodes whose own neurons do the last bending of the outputs. A node with gates below it inside its area usually has a large eps_lin, and there the hard one-node problem of T57 stays |
| (b) no | nothing new to accept | the summary result stays a statement about always-on boxes only (Corollary TP1.1) |

**Question 2 (R-linread): does a map with linear-read claims at error 0 count as complete (DM6)?** If yes, TP1's
closed form becomes T5's exactness clause for D2. Rec: yes, for D2 only, once Question 1 is (a).

My recommendation: (a), and yes. Open after this card: boxes that switch inside their own area (Route (3) of
Conjecture TH4), a summary rule for lifts, and KL in place of D2.

### 3-25. S-laminar, S-smoothact and S-nonorm: may the areas of a stream map sit at an angle? (COARSE-THEORY.md DM11, F14, B.V.3, TV14; ANSWERED 2026-10-29)

**What you said (card 3-23).** "I'm going to be fine with some like subspace crossing because the ultimate thing is:
are we able to kind of deal with these sorts of concepts on their own."

**What the math says (B.V.3, proved; checked in `model/tv_cancel.py`, 0 failures).**

```
two areas S1, S2 at an angle (their sum is direct: they meet only in 0)

without norms (F14 part 1, TV10)   an angle is not a property of the network: a change of stream coordinates
                                   makes them orthogonal and changes nothing; a cut is fixed by what it drops
with norms (F14 part 2)            an angle acts through ONE number per norm: the stream's size
crowded (TV11 part 2)              if the areas share a vector, no cut can drop one and keep the other whole
```

Why it matters: your reading is right. Overlap is fine as long as the areas meet only in 0. Crowding (more areas than
directions) is the one case where concepts cannot be handled one at a time.

**The question.** Definition DM11 still asks areas to be nested or at right angles. Widening it changes TV2 (an energy
sum per area), TV6 (a decomposition into areas) and DV2 (the grouping).

| Option | Gain | Loss |
|---|---|---|
| (a) widen next session: any two areas nested or meeting only in 0, children's sums direct [rec] | matches your words; without norms the class of functions cuts compute does not see the angle (F14, TV10) | every claim error and the bounds of TV2 and TV6 must be redefined with oblique projections; the choice of complement becomes part of a claim |
| (b) keep right angles | the theorems stay as they are | a map may not name two concepts that lean on each other |

**A second, technical question (S-smoothact).** Proposition TV14 (the signed carrying from N's own slopes) needs
activations with a Lipschitz derivative (tanh, GELU, SiLU), so it leaves out ReLU. Theorem TV6 and Lemma TV13 cover
ReLU. Rec: accept (the pure-stream models we target use GELU or SiLU).

**A third, technical question (S-nonorm).** Lemmas TV10, TV13 and Proposition TV14 are proved without norms. Real
transformers have norms, so this is the main limit on TV14. Conjecture DV3 is the norm form, with a route. Rec: accept
for now, and prove DV3 next.

**Answered (2026-10-29).** Shonu's words:

| Node | His words | Result |
|---|---|---|
| S-laminar | "S-Laminar is certainly a development decision, so you don't need to ask me about how we kind of do that. That's completely fine for you to make those decisions on your own." | DECIDED by delegation: (a), widen (B.V.5); new node R-cutclass for the wider class of cuts |
| S-smoothact | "I just personally don't see how we should have results for the residual stream, which is mainly about you know directions and things like that, which is which does better or worse depending on what the neurons themselves are doing ... I just want a nice general method over these residual streams, and I think that you know we don't need to place those sorts of restrictions on anything" | DROPPED as moot: Theorem TV15 holds for every Lipschitz activation |
| S-nonorm | "I'd like you to prove that norm's conjecture, otherwise, you know, we're not going to be able to apply this in the future" | DROPPED as moot: Conjecture DV3 proved (Theorem TV15, both forms) |

His standing rule (same message): "a lot of these decisions, they're like, you know, you could see what it actually
does or what it doesn't do in terms of mathematics before saying, like, before asking me ... when you give me a
decision like that, it's not really like something that's very productive for me to respond to". So: a technical
scope question gets its math first, and I decide it. Only a restriction that narrows the class to get a proof
through still comes to him as a card.

### 3-24. R-tree, R-branch and S-plainff: one recursion over a tree of areas (COARSE-THEORY.md B.H, D.16; PENDING; replaces cards 3-11 and 3-12)

Your words (2026-10-29): a clean network works by gating; earlier decisions are coarse, later ones finer; think
recursively and locally. And: "i see zoom recursion and depth recursion as the same thing ... we should treat this as
one unified idea we want to prove is optimal". So cards 3-11 (R-recur, step by step) and 3-12 (R-deep, deeper layers)
are now one question, this card. Your words decide only that they are ONE question. The readings below are open.

The object: a **tree of areas**. An area is a set of inputs of the mix (for example "car photos").

| Word | Meaning in the network |
|---|---|
| node v = (layer l, area R) | the neurons of layer l that may be on only inside R (district claims, error 0) |
| children of v | nodes of layer l + 1 whose area lies inside R (finer decisions behind v's gate) |
| subtree group | some children of one node u, with everything below them: the units a zoom step replaces |
| box | the k new neurons that replace a subtree group. They read the group's top neurons' pre-activations (as in class (b)) and are carried to the logits by copy neurons relu(h) = h |

```
layer 2:            [ vehicle gate ]                     <- node u, kept
                     /           \
layer 3:      [ car gate ]    [ boat ]                   <- top of the group (children of u)
               /      \
layer 4:   [ taxi ] [ ambulance ]                        <- inside the group

group = car subtree + boat subtree  ->  k new neurons that read the layer-3 pre-activations
```

Two kinds of zoom step:

| Step | What changes | Example |
|---|---|---|
| sibling merge | two groups under the same node u become one group; the top layer stays | car subtree + boat subtree |
| lift | a group joins its parent's node; the top moves up one layer | car + boat subtrees + the vehicle gate |

What the math says (Part B.H):

| Result | Plain meaning | Status |
|---|---|---|
| Lemma TH0 | inside a node's area, everything later in the network is a function of that node's neurons | proved |
| Theorem TH1 | at ANY grouping of subtree groups, at any mix of depths, the least loss is a sum over the groups, each on its own areas. Least builds combine; switching one group changes only its box | proved |
| Proposition TH2 part 1 | sibling merges nest: along them the least loss can only grow (with sizes that do not grow), and if the coarsest level is exact, every level is exact and is built step by step from the level below | proved |
| Proposition TH2 part 2 | lifts do NOT nest: a coarser level can be exact while a finer level is not. What fails is the order of the levels | proved (exact instance) |
| Proposition TH3 | a sibling merge with 3 logits that must lose something: step by step is strictly worse than the least build | proved (exact instance, objective D2) |
| Conjecture TH4 | each node passes a small summary up the tree (not its neurons), and a parent's least loss is a fast function of its children's summaries (the hard one-node problem is taken as given) | open |

So: your "zoom and depth are one recursion" is TRUE for the split (TH1: one statement covers every depth). It is
true for nesting only when a step keeps the top layer. A lift changes what the box can read, and that can help or
hurt.

**Question 1 (R-tree): what may a box read?**

| Option | A box reads | Gain | Cost |
|---|---|---|---|
| (a) [rec] | the pre-activations of its group's top neurons | TH1: every group is local, any depth. Same words as T5 and class (b) | lifts do not nest (TH2 part 2) |
| (b) | (a) plus the outputs of every kept gate above it (the decision chain) | lifts nest (a finer box can copy a coarser one) | a box may then vary on its siblings' areas, so groups help each other and TH1's split is lost (TA12's mechanism) |

**Question 2 (R-tree): copy neurons.** A box in layer l < L reaches the logits only through copy neurons
relu(h) = h in layers l + 1 to L. They are exact (h >= 0) and carry no new information. Rec: they do not count
toward k_g (in a ResNet the skip does this for free).

**Question 4 (S-plainff): plain feed-forward networks.** Part B.H is written for networks whose layers are dense and
read only the layer before (no skips, no positions), as the referee pointed out. ResNet has skips, so the next step
(ResNet with the tree) needs either the stream form of card 3-23 or a reading of skips. Rec: accept the scope for now.

**Question 3 (R-branch): the branch claim.** TH1 needs: no neuron outside a subtree reads the subtree on the mix
("the dogs go on to their own area, which doesn't have to worry about cars"). Without it, removing the taxi neurons
can switch on a dog neuron that they kept off. Rec: accept it as a claim type with its error (Definition DH2).

Your choice:

| Question | Options | My recommendation |
|---|---|---|
| What may a box read? | (a) top pre-activations / (b) plus the decision chain | (a) |
| Do copy neurons count toward k_g? | no / yes | no |
| Accept the branch claim? | yes / no | yes |
| Accept plain feed-forward as Part B.H's scope for now? | yes / no | yes |

### 3-23. S-purestream, R-streamregion, S-readoutarea, R-normconst and R-cutclaim: a map of the stream made of areas (COARSE-THEORY.md F12, F13, DM11 to DM13, B.V, D.15; ANSWERED 2026-10-29)

**Answered (2026-10-29).** Shonu: "Yes, study pure streams right now" (S-purestream DECIDED); "I'm going to be fine
with some like subspace crossing" (R-streamregion DECIDED; the trunk sentence was amended; overlap is card 3-25); for
the other three: "they don't really seem like decisions, they kind of seem like things that you could work out with
the map itself". Worked out in B.V.3: S-readoutarea is the map's exactness condition (dropped); R-normconst is one
more stream direction (Lemma TV12), which widens DV2's class, so it stays a decision and I chose (a) under his
delegation; R-cutclaim goes away once D is a sum of the map's members and completeness for area maps is defined
(DM14, DM15; dropped). His quantum picture became Theorem TV9 (the build cancels a leak), Lemma TV13 and Proposition
TV14 (the network's own signed cancelling, from N's slopes), and Corollary TV6.2 puts B.V in T5's form (a side branch while S-laminar was pending; decided 2026-10-29, card 3-25).

**What you asked.** "Let's try this subspace idea ... we're thinking of regions of the model rather than orderings."

**What the math says (B.V, proved; checked on a planted network, `model/tv_regions.py`, 0 failures; referee
REFEREE-BV-2026-10-29.md, all findings fixed).**

```
pure stream (ViT, GPT-style):   only rotation-free quantities matter (inner products of reads and writes), not coordinates (F13)
an area:                        a subspace of the stream; areas are nested or at right angles (DM11)
a cut:                          keep fewer directions inside an area; same network with narrower read/write matrices

TV1  a part that NO reader reads still acts on every read, through the norm (the size every norm divides by)
     -> a cut leaves every read unchanged exactly when its energy is the same on every input
     -> the essay's loud direction has varying energy, so a cut must keep it (one direction if it lies on a line)
TV2  areas that no block reads: loss <= (sum over areas of sqrt(readout error))^2 / 4, exact at 0
TV3  inside such an area, the best k directions for that bound are a closed form (weighted SVD); C - 1 always suffice
TV5  BUT: the areas no block reads are the common kernel K of all block read matrices,
     and K = {0} for generic weights (a transformer MLP alone has 4d read rows)
TV6  the graded case (pure-stream networks with Lipschitz sigma): loss <= 1/4 (direct readout loss + each block's
     leak carried to the logits)^2; errors grow by (I + T) per block; without a norm the carrying follows the area graph
TV7  no bound from claims made one block at a time can exist: the block writes a value of size delta taken from the
     cut part, the logits read it with weight about 1/delta; every claim about every area has error <= delta^2, the
     loss stays fixed
TV6.1 with the new "carried leak" claim (DM14), a bound in claim errors alone, for one fixed cut (T5's graded form for complete area maps: Corollary TV6.2, added in round 2)
TV8  a computed check lets the norm's coupling use each neuron's largest read (tighter only for neurons that read little)
```

Why it matters: the exact results hold only on a null set of weights, so the graded theorem TV6 is the one that covers
real networks. Its bound is valid but loose on random dense networks (a median 10^4 times the loss without norms). The check shows why it is
needed: when a block reads a little of a cut area, the bound fails badly. TY2 already gave an error-only bound for any
number of logits; what B.V adds is the role of the norm (TV1) and a closed form for the best cut (TV3), both at the
readout.

**Question 1 (S-purestream): may this branch study pure-stream networks only (MLP blocks, one position, a final norm)?**

| Option | Gain | Loss |
|---|---|---|
| (a) yes, for now [rec] | exact theorems; the kind of stream that transformers and ViT have | leaves out ResNet (ReLU on the stream), ConvNeXt (per-channel filter), attention |
| (b) no, every kind at once | one theory for all | the areas of kinds 1 and 2 are not subspaces; the math splits by kind anyway (Fact F12) |

**Question 2 (R-streamregion): may a district of the map be an area of the stream (a subspace that crosses every block)?**
T5 says a district is "a set of N's units inside one hidden layer". Option (a) [rec]: yes, for pure-stream networks, and
a group may be an area cut to at most k_S directions inside it. Option (b): no, keep T5's wording; then B.V stays a side
branch. This changes T5's wording, so only you decide it.

**Question 3 (S-readoutarea): keep the exact results at areas that no block reads?** By TV5 this premise holds only on a
null set of weights. Option (a) [rec]: keep them as the exact end of Theorem TV6 (proved since). Option (b): drop B.V's positive part and go straight to the graded case.

**Question 5 (R-cutclaim): may a claim name the part a cut drops?** T5 says no claim refers to a build or a target
size. A carried leak claim (DM14) names D, so with its group it names the cut and its size; TV7 shows that some claim
of this kind (one that carries a leak through later blocks) is needed for a graded bound. Option (a) [rec]: yes, as
a reading of T5 for stream maps. Option (b): no; then B.V has a graded bound only with N's gains (TV6), not in claim
errors alone.

**Question 4 (R-normconst): may a build re-set each norm's small constant eps?** TV1, TV2 and TV4 use it: the dropped
energy folds into the constant. It is the same kind of reading as R-readbias ("yea they can move their biases").
Option (a) [rec]: yes. Option (b): no; then a part with nonzero energy can never be dropped exactly.

### 3-22. R-unitmap and R-propcomplete: what a map of ResNet-18 can be made of, and what counts as complete (COARSE-THEORY.md A.7 DM10, B.Y, Part C; PENDING)

**What you asked.** "Create a good map" of the torchvision ResNet, then the full compress-and-expand zoom on it.

**What the network says (Part C, H-map row for ResNet-18; measured, 35000 photos).**

```
last layer of ResNet-18: 512 channels x 7 x 7, the outputs read the mean over the 49 positions

a typical channel ............ on for ~860 of the 1000 classes (never off outside a small set of classes)
its closest other channel .... correlation 0.24 (no channel is a copy of another, so nothing merges exactly)
its strongest 0.1% of entries  ~61 classes, a real concept (see the cards page)
the outputs .................. rely on the broad rest: keeping only the strong 0.1% gives KL 4.9
every other layer checked .... the same picture
```

Why it matters: the paper's claim types assume a unit is off outside its region (districts) or a near-copy of its family
(the new proportional claim). ResNet-18's channels are neither. So "our maps suck" was not only how we built them: at
the level of single channels and label classes, no good map of these types exists for this network.

**What the math adds (B.Y, proved).** A rule that needs no regions and no two-logit limit: merge channels that the map
claims are near-copies, write the new readers down from N's weights (Theorem TY2). Its loss is at most a quarter of the
squared sum of the square roots of the claims' errors, and 0 when they are 0. On maps that hold such near-copy claims
for every group, this is the first rule in the paper with the bound that T5's graded promise asks for. On ResNet-18 the bound is far too large to use, because the channels are not near-copies.

**Question 1 (R-unitmap): what should a map of this network be made of?**

| Option | Meaning | Gain | Loss |
|---|---|---|---|
| (a) keep units = channels | ResNet-18 has no small-error map; the zoom on it is measured, not promised | nothing changes in the theory | no exactness and no useful bound on ResNet-18 |
| (b) map InceptionV1 first | run the same map-and-zoom program on InceptionV1 (4c to 5b), where single channels are known to be specific (4c:447) | tests the theory where its premises may hold | delays the torchvision ladder |
| (c) a graded "strong core plus weak spread" claim type | a unit = a strong, specific part plus a broad weak part (the essay: "weak activations radiate outward"); needs a definition, the copy test and its own node before any run | fits what the cards show | new theory; unknown whether a rule can use it |

My recommendation: (b), with (a) kept as the ResNet-18 result. The claim types were written from InceptionV1's specific
channels, so that is where we learn whether the method works when its premises hold. (c) is the long-term fix if (b)
works and ResNets stay broad.

**Question 2 (R-propcomplete): should a map with near-copy claims for every group count as complete?** DM6 now asks for
districts, shelters, judgments and (below c(S)) a two-logit blend claim. Theorem TY2 gives an exact closed form and a
bound from proportional claims alone, for any number of logits. Counting such maps as complete widens the exactness
promise. My recommendation: yes, as an added item ("or every group has a merge partition in the map").

**Also open (card 3-12, R-deep).** The layers below the last one still wait for your call on card 3-12. My
recommendation there is (B): the next block re-fits its weights on the new channels (least squares, a closed form).

### 3-21. R-mass: does growth inflate the write? (COARSE-THEORY.md B.N; PENDING)

**What you asked.** A larger network has more neurons, so its small errors could add up. You asked for math that
says a larger, cleaner network gives a better result anyway, so that work on a small network carries to a large one.

**What the math says (Propositions TN1, TN2, Theorem TN3; all proved, under R-wellposed like the rest of B.E).**

```
copy one neuron into n neurons that share its output      -> the network does not change
  function, least loss, robust optimum, clean split, E    -> all the same            (TN1)
  district, shelter error (as first defined: sums)        -> grew by (n - 1) x that neuron's term
  -> CHANGED: each error is now the largest neuron's term; flat claims and the blend-claim count changed the same way
narrow neurons that add up to one broad kept neuron       -> robust optimum never worse  (TN2)
E (how far the coarse network can be from the clean split) -> masses x weighted means + what N sends through errors
                                                             no term counts neurons    (TN3)
```

So "errors add up with size" was true of OUR error definitions, not of the network. They are fixed. The bound
weighs each neuron by its share of what it sends to the outputs, and the shares add to 1. The referee rates these as
bookkeeping results: their value is that they found the counting fault and make the size dependence explicit.

**What stays a hypothesis.** The bound falls with size only if the numbers in it fall (Corollary TN4):
1. H-scale (R-scale): the errors fall with size. Three forms: what the leaks send to the outputs (quiet form), where
   neurons fire weighted by what they send (fair form), and the largest single-neuron deviation for a full map (map
   form, the one Corollary TN4 uses; not measured yet).
2. H-mass (R-mass, this card): (1) the output spread does not grow with size; (2) the clean small network does not
   need large weights of both signs that cancel.

**What the ResNet ladder shows (Part C, one coarse dog/car map).**

| | ResNet-18 | ResNet-152 |
|---|---|---|
| output spread O (H-mass part 1) | 22.3 | 22.0 (flat) |
| what the leaks send, as a share of O (quiet form) | 0.455 | 0.390 (falls, about 7 se) |
| fair deviation (fair form) | 0.392 | 0.401 (does not fall) |

The larger nets are cleaner in what the outputs receive, because their leaks cancel more. They are not more specific
in where channels fire, on this coarse map. H-mass part (2) waits until the clean small networks are built.

| | |
|---|---|
| Meaning | Along a family of sizes, the last layer's total output does not grow, and the clean small network's weights stay in proportion to it. |
| Bought | Corollary TN4: under H-scale (map form) and H-mass, the bound on E does not grow with size and falls when the errors fall. The theorems hold at every size; the numbers in them must still be measured for each network (a check at one size does not transfer). New layer kinds (residual adds, normalisation, attention) are new work: the torchvision ResNet's last block (relu of a sum with a skip) is already one. |
| Lost | Families where a larger net needs large cancelling weights. There the bound can grow even when every error falls. |
| Not claimed | That trained larger nets ARE cleaner (Part C decides per family). That they are easier to map (finding the map is outside T5). That the real gap falls (only its upper bound does). |
| Recommendation | Accept R-mass as a measured hypothesis, like H-weak: no theorem on the main line depends on it, and part (1) already holds on the ResNet ladder. |

### 3-20. R-wellposed: the least loss cannot see "small" (COARSE-THEORY.md B.E; PENDING: the corrected trunk wording waits for your confirmation)

**What went wrong.** You said TM4's blow-up looks like an error in the math, not a decision. It is. The step that
fails is the trunk's object, "the least loss among all such networks".

```
kept neuron j1:   values (1, delta, 0) on photos a, b, c          N's own weight on j1 = 0
small network:    reads j1 with weight about 1/delta              -> fixes photo b exactly
delta = 0.1, 0.01, 0.001:  the gain is the SAME every time        (Proposition TE1 part 2)
price:            j1's swing (how far it can push the outputs) grows like 1/delta   (TE1 part 3)
```

With free reader weights, the least loss sees only WHERE a neuron is on, never HOW MUCH (TE1 part 1: it depends only
on the span of the value vectors). The district, shelter, judgment and flat errors measure how much. So no such error
can bound it.

**The correction (Definition DE2).** Judge a small network by its worst loss over every TABLE of last-layer values
between N's own values and the clean values the map describes, with N's answers as the teacher. (Tables, not networks:
a table need not come from any weights.)

```
U = { N's values,  the clean values (leaks set to 0, sheltered inputs set to one value, flat units set to their mean,
      judgment sums set to exactly 1), and every mix of the two, entry by entry }
robust optimum = least over small networks of (worst loss over U)
```

In words: the small network must work whether or not each leak is there. This is the essay's ablation ("you can
ablate almost all the weak activations of the network for almost no cost") used as the definition.

All results below are side branches until you confirm R-wellposed (the confirmed T5 still says "least loss").

| Status | Result |
|---|---|
| proved | TE3: with every error 0, U is N's table alone, so the robust optimum is the old least loss. The exact clause does not change. |
| proved | TE4: clean split sum <= robust optimum <= clean split sum + E. E = sum over moved entries of (size of the move) x (swing of the neuron that reads it): the clean small network's swings for kept neurons and boxes, N's own swings at photos of regions that no group touches. Linear in district errors, square root in the others. |
| proved | TE4.1: on both TM4 instances the robust optimum equals the clean value for every delta. The blow-up is gone. |
| proved (referee's instance) | TE5: your graded clause is still NOT met by this rule (TM1's formula on the clean table; other rules are not ruled out). A complete map whose errors all go to 0, while the robust optimum stays a fixed amount above the clean split. Reason: N itself reads a tiny difference with a huge weight (W), and the same weight sends sqrt(W) through a tiny leak. N is "loud" through its errors. |
| new claim type | DM9 quiet claim: what N's own readers receive through a district's leaks (output units). TE5's instance has a large quiet error, so it is not a small-error case after all. With quiet claims, N's part of E is bounded by claim errors. |
| open | the clean small network's swings (E's first two terms). They depend on the zoom, so they cannot be claim errors. Whether they can be bounded by claim errors is open. |

**What changes in the trunk text, if you agree.** One phrase: "with the least distillation loss ... among all such
networks" becomes "with the least worst-case distillation loss over the tables of last-layer values between N's own
and the map's clean values, with N's answers as the teacher". The bound sentence would also need a change, because
the bound depends on swings too, not only on the errors. That waits until the open line is settled.

Why it matters: small errors now give small changes when N's own readers are quiet about the errors. The essay's
weak-activation test measures exactly that (term (c) of TE4, Part C H-weak).

### 3-19. T5 meets T43: which maps must the exact clause cover? (COARSE-THEORY.md Theorems TM3, TM4; DECIDED 2026-10-28: (a), Shonu: "i would say yes, the exactness promise applies only to complete maps")

**What we found.** A map may say very little. This map is always true:

```
map M_1:  one district = every neuron of the layer, region = every image   -> error 0 (nothing to violate)
T5 says:  every claim has error 0  ->  the rule must give the exact best coarse network
read "closed form" as "computable in polynomial time" (as for T43):
on T43's clique detectors (one big group -> 1 neuron) that is NP-hard  ->  needs P = NP (Theorem TM3 part 2)
```

New claim types cannot fix it alone, because a map may leave them out. TM3 part 3: if the clause is asked only on a
fast-checkable class A of maps and a rule meets it there, then some hard instance has no small error-0 map in A (else
NP = coNP). The same root makes "clean = has a map with small errors" empty. Separately, the graded clause cannot be met
through TM1's split with its three claim types: a tiny shelter error can be amplified by a new neuron with a huge weight
(Proposition TM4).

**Your choice.**

| Option | Meaning |
|---|---|
| (a) complete maps [my recommendation] | the exact and graded clauses apply only to maps that are complete for the zoom: a polynomial-time relation between map and zoom (not a claim) that asks, for every group, for claims of a required list of types. Now defined: Definition DM6, with TM1's district, shelter and judgment claims and the two-sided blend claim DM7. Clean = a complete map with small errors exists. On T43's instances the big group's blend claim has positive error, so the exact conflict disappears. Not checked: the graded clause there (the bound would have to be at least T22's hardness gap at those errors), and TM4 says the bound needs a further claim |
| (b) drop the polynomial-time reading | "exact closed form" may take longer (Remark B.K.1 (e)). For D2 with two logits and one hidden layer, Theorem T21 gives an exact rule with cost about n^{dk}; for KL its pieces are transcendental, so there is no exact closed form even then |
| (c) keep T5 and the reading | T5 can then be met only if P = NP (proved) |
| (d) restrict the inputs | exclude T43's networks, mixes or big groups (the options of card 3-5, R-mixclean) |

Why it matters: under (a), the map carries the information that makes the problem easy, and the theorems say which
information that is. That is your original idea, with "a map that says enough" made precise.

### 3-18. S-lastlayer, S-cover and S-regiongroup: where the splitting theorem TM1 stops (COARSE-THEORY.md B.M; PENDING)

These are not choices of meaning. They mark where the proof of TM1 stops. Each is a "partial result" node, like
S-onehidden.

```
map:      region "car photos"  = {taxi neurons, ambulance neurons, car neuron (judgment)}
          region "dog photos"  = {ear neurons, snout neurons, dog neuron (judgment)}
zoom:     group 1 = taxi + ambulance neurons -> 1 new neuron      (car neuron kept)
          group 2 = ear + snout neurons      -> 1 new neuron      (dog neuron kept)
TM1:      best loss of the level = best loss of group 1 on car photos + best loss of group 2 on dog photos
```

| Node | What TM1 needs | What is open |
|---|---|---|
| S-lastlayer | the groups lie in the last hidden layer, so the outputs add the groups' changes | deeper layers: the changes meet at later neurons (Conjecture CU2 of UNIT-BRANCH.md) |
| S-cover | every neuron of the last hidden layer is in some district of the map | TM1 part 4 replaces this by a flat claim for each neuron outside the districts (constant on every region); neurons that vary on a region and are in no district remain open |
| S-regiongroup | each region meets at most one group, and its judgment neuron stays kept | two groups in one region fire together, so they interact; how much is open (no bound is proved) |

Why it matters: inside these three limits a map gives exact independence, under the reader rules you decided (readers
re-fit biases and weights).

### 3-17. R-edgetether: may a coarse group send signals along directions its neurons never sent? (UNIT-BRANCH.md DA2, TA6, CU7; PENDING)

**What we found.** You asked why the essay distrusts activations. Read as math (UNIT-BRANCH.md A.6, B.6), an activation
alone drops two things: which inputs voted for it (TA3), and who reads it (TA1, TA2, TA4). Then B.7 found the same gap
in our own definitions:

```
last hidden layer:   neuron u ---(+1 to "cat", 0 to "dog")---> outputs     neuron v ---(+1 to "cat", +1 to "dog")---> outputs
                     v adds the same to every output, so no probability changes: in effect nobody reads v
                     u and v are ON on the same images
coarsen u away:      the output loses u's signal
our box rules:       v's box may keep relu(z_v) and give it a NEW direction of weights -> it patches u's loss
result:              v, which changes no probability, is tied to u's group (TA6)
```

The box gave v a use that the network never gave it. (A neuron with all weights 0 is already safe: our rules let a box
write only to neurons the old group was wired to. In a fully connected layer that is every neuron, so the wiring rule
does not stop anything else.) The essay says "a representation matters only through what reads
it". So the fix is on the write side: a coarse group may only send along the directions its own neurons already send.
The read side was already fixed by DU6 and DU7 (card 3-15).

**Your choice.**

| Option | Meaning |
|---|---|
| (a) tether writes [my recommendation] | a coarse group's new neurons write only inside the span of what its old neurons wrote. A group that nothing reads can only be removed. Proved: help between two groups then needs overlap inside the helper's own output directions (TA8); a group that changes no probability never interacts; if two groups' output directions are orthogonal where they are ON together, only a tiny (third-order) interaction is left (TA8 part 3). Deeper layers: Conjecture CU7 |
| (b) keep free writes | then the unit stays the co-firing components (TU8), including neurons that nothing reads |

Why it matters: under (a) the unit is decided by connections plus activity, as the essay's edge motifs suggest. Under
(b) it is decided by activity alone.

**Update (2026-10-28, Trunk T5).** With the reader rules you decided (readers re-fit weights), the tether does not keep
an unread neuron out of a group. The readers themselves may start to read it (UNIT-BRANCH.md Proposition TA12; exact
toy: interaction +0.046). Under T5, a map's regions meet the independence demand under TM1's premises (COARSE-THEORY.md
Theorem TM1), and this card only changes which coarse neurons a group may build. My recommendation is now: leave R-edgetether
PENDING until a theorem needs it.

### 3-16. R-readbias: may a reader re-set its threshold when its inputs are coarsened? (UNIT-BRANCH.md DU8; DECIDED 2026-10-27: (a), Shonu: "yea they can move their biases")

**What we found.** You asked what the strange "help" between coarsened groups really is. In every toy we built, almost
all of it was one thing:

```
4 neurons (2 "lower", 2 "upper") --coarsen--> 1 new neuron u >= 0, written along one direction
reader (logit V1 minus logit V2) receives  u x (write)  ->  one sign wherever u is ON, 0 where it is OFF
-> "upper versus lower" needs the reader's threshold to move; our builds froze it at N's value
-> another group's new neuron is ON with a fixed value outside its own group and acts as the missing threshold
```

| Toy | Help at budgets (1,1), thresholds frozen | Help, readers re-fit their bias |
|---|---|---|
| original circle toy | -0.18 | +0.006 |
| sheltered districts | -0.239 | +0.0001 |
| gate in front | -0.203 | +0.0001 |
| shared pose units | -0.052 | -0.003 |

Theorem TU19 says why (one ReLU unit is one-sided). Remark B.6 lists the sources of help found so far: a frozen
threshold (its global part is gone with re-fit readers); silence used as an opposite ("T is off, so answer like S",
which survives re-fit readers, TU21's example, and is worth at most one more unit, TU18); a rebuilt unit (the other half
of a dial, TU20; impossible for sheltered groups, TU17); and a shared reader bias (either sign).

**Your choice.**

| Option | Meaning |
|---|---|
| (a) readers re-fit their bias [my recommendation, together with "silence stays silence" (DU7) of card 3-15] | when a group is coarsened, each neuron that reads it may move its threshold. Trunk T4 does not forbid it ("a network of the same kind ... among all such networks"). In all four toys it removed the help with free coarse neurons. In general it does not remove "silence used as an opposite"; DU7 does |
| (b) readers keep N's biases | then the frozen-threshold help stays, and card 3-15's principles (DU6, DU7) are needed to block it |

### 3-15. R-indep: may the developer coarsen different parts of the network independently? (UNIT-BRANCH.md; PENDING; rewritten 2026-10-27, night)

**What changed.** The first version of this card asked you to choose between "free" and "covering" coarse units.
You answered (2026-10-27, night) that a clean, well-trained network should already be mostly independent, and that
independence should be graded like every part of clean. The math agrees with the first point in a precise way:

| Result | What it says |
|---|---|
| Remark B.4 | the old "no independence" result (TU10) holds for every network, clean or not; so it is about the free box, which reads how far below zero an OFF neuron is, a value N's readers never receive |
| DU6, TU12, TU13 | if a box may depend on the input only through what its group sent, it can still send one constant where the group is silent; that constant can help another group a lot (lab: I = -0.107 on the circle toy at budgets (1, 2)) |
| DU7, TU15 | if also "silence stays silence" (the essay: negative poles are gates, not opposites), the box is covering on the mix, and groups that are never ON together are exactly independent |
| TU14 | with weak outputs read as silence, the interaction is at most a sum of strong co-firing terms and dropped weak-write terms: a graded number |

So "covering on the mix" is no longer a choice I put to you. It follows from two readings of the essay (node
R-outfaith, PENDING, because it is still a restriction of Trunk T4's class (b)).

**The question that is left.** Not "free or covering", but: how large is the graded term (strong co-firing plus
dropped weak writes) for clean networks, and can it be computed from N alone, as an entry of eps(N)? That is a math
and measurement question. Nothing here asks you to choose. If you disagree with either reader principle (DU6 or DU7),
say so, and R-outfaith goes back to open.

### 3-14. R-circuit: should the trunk coarsen circuits instead of groups of one layer? (Part B.X; PENDING)

What we found (Part B.X; the referee checked it, and its count is that these are elementary results plus a known
depth separation, not yet a theory of the least coarse circuit). Restrict the network's answer to the straight line
between two photos: it becomes a piecewise-linear function of the position on the line, bending where some neuron, in
any layer, strictly changes sign. The dip at a point of the line is minus a sum of the bends (Theorem T63; the
one-bend identity is in Lean, the rest by hand; checked exactly on random deep networks).

```
N:               photo --> [many neurons, several layers] --> answer     dips on the mix's lines (from its answers on the mix)
coarse circuit:  photo --> [fewer neurons, any layers]    --> answer     must bend upward across every line where N dips
```

| Result | What it says |
|---|---|
| T64 | a one-layer neuron bends at most once on a line and is monotone there; this gives T62a, T62b and the two-point case of T59 |
| T65 | an exact one-layer build needs at least tau(D) neurons: the fewest cuts that cross every dip line of the mix. tau depends only on N's answers on the mix. It is a lower bound only: on T61's instance tau = 1, yet no single neuron is exact |
| T65.1 | T20's old impossibility is this count |
| T66 | a plain network with k layers of 2 neurons is exact on a zigzag where one layer needs 2^(k-1) - 1 neurons (Telgarsky's known result) |
| D61 (conjecture) | for families that are free on the mix (H-freefam), the bound is reached, by presence detectors |

Why it matters: the dip lines are computed once from the mix, not neuron by neuron, and a coarse circuit that may span
layers can, for some jobs, be far smaller (T66). What is NOT shown: that clean networks have such jobs, what a least
coarse circuit is, or that this view decides the kind of coarse neuron (T61's instance is a counterexample to the last).

Update (2026-10-27, unit branch, `UNIT-BRANCH.md`). The math was asked which unit it picks:

| Finding | Theorem |
|---|---|
| A set of units across layers can be replaced as one box exactly when it is closed under the paths between its members (convex) | TU1 |
| With coarse units that stay silent outside their district on the mix, sets that never fire together are independent at any depth (if convex and unlinked); in the last hidden layer the finest units are the co-firing components | TU5, TU8 |
| With free coarse units, no two sets that cost something are independent (last hidden layer) | TU10, TU11 |

So the math points to a unit set by activity on the mix (who fires together), with wiring entering only through
convexity and "unlinked". It is not a wiring circuit and not a one-layer group as such. The depth version of the
converse is open (Conjecture DU1). Card 3-15 holds the decision this forces.

Your choices:

| Question | Options | My recommendation |
|---|---|---|
| What does the developer's zoom coarsen? | (a) keep T4: groups of units in one layer; use dip lines as an analysis tool; (b) change T4: each group is a circuit (units in any layers) with a budget of new units, replaced by a least coarse circuit (Definition D60, layered, relays counted); (c) like (b), but new units only in the layers the circuit already uses | (a) for now. The circuit view is a sound language and a lower bound, but the evidence that coarse circuits across layers help is only the general T66. Revisit (b) once a clean network with a deep job is found (Conjecture D61's route) |
| Draft wording for (b), if you choose it anyway | "Each group is a circuit of N (a set of N's units in any hidden layers) with a budget of new units ... a network B_j in which each circuit of G_j is replaced by a coarse circuit with at most its budget of new units, with the least D_P" | confirm or change in your words |

### 3-13. R-kinds: how to coarsen a conflicting group (Theorem T62, Remark B.W.4, Hypothesis H-freefam; PENDING)

What we found. A group is **free** if the mix holds no "silent blend" between its members' photos, and **conflicting**
if it does (Definition D55). Every silent blend must be paid for by some neuron switching ON/OFF across it (Corollary
T62b, the dip identity). That one rule explains all the kinds of coarse neuron we have seen:

| Group | Best coarse neuron | Proof |
|---|---|---|
| AND (members ON together) | linear algebra on the votes, exact | T58 |
| free | a presence detector ON where its members are ON, exact | T59.2 |
| conflicting, alone | a presence detector of a clique of its members; it drops the others and pays | T59.1, Remark B.W.4 |
| conflicting groups together | split units: each ON at exactly one of its own members and OFF at the other ("A, not C"); in T61 they are dials, ON on most photos; exact on T61's instance | T61, T62 |

```
split dial for {A, C}:   ON at A ... ON at B, D and three blends ... OFF at C    <- "taxi and ambulance on each other's negative poles"
```

The essay's reading (Hypothesis H-freefam, PENDING). Some families are a continuum: the blend of two curves is
another curve, so some member fires on it. Such a family is free, and it coarsens into a detector (the essay's
"curves & loops" neuron). Other families are discrete alternatives (taxi, ambulance): their blend is "nothing". Such a
family is conflicting, and its best coarse neuron is a contrast with the alternatives at opposite poles. The detector
networks of T43 are built from discrete alternatives on purpose.

Why it matters: coarsening still works under T4, but "coarsen" does not mean "make a coarse detector" for every
group. And step by step fails exactly where conflicting groups must cooperate (T61): the split dials have to be fitted
together.

Your choices:

| Question | Options | My recommendation |
|---|---|---|
| How should the rule coarsen a group? | (a) keep coarsening, with a kind of coarse neuron for each kind of group (the table above); (b) as (a), and the rule itself checks each of the developer's groups against the mix (free or conflicting, by Proposition T59's linear program) and picks the kind; (c) coarsening is the wrong frame for conflicting groups | (b). The check is one linear program per group; free groups get detectors, step by step; conflicting groups that meet at one level get their split dials fitted jointly |
| Test H-freefam on InceptionV1 next? | yes (measure silent blends inside the curve family, Part C) / not yet | yes, after card 3-12: it decides which kind of coarse neuron the real curve family needs |

### 3-12. R-deep: what "reads only its group" means in deeper layers (Remark B.W.3; PENDING)

**Merged into card 3-24 (2026-10-29).** Shonu: "i see zoom recursion and depth recursion as the same thing". The
question now lives in card 3-24 (node R-tree). This card stays as the record.

The trunk T4 says each new neuron reads only its group's units. In layer 1 this is clear: the group's members read the
photo, so the new neuron reads the members' sums (class (b)). In layer 2 and up it is not clear, because the neurons
below may themselves have been replaced.

```
layer 1 (N):     c0 c36 c72 ...            wheel window light
layer 1 (B):     [ coarse curve ]          [ coarse car part ]        <- already merged
layer 2 (N):     circle neurons that read c0, c36, ...
layer 2 (B):     [ coarse circle ] reads ... what?  N's circle neurons read c0, c36, ..., which no longer exist
```

Why it matters: Cale's real networks have many layers, and the developer will group neurons in several of them. Every
result of Part B.W so far is for one hidden layer, or for the top layer with the layers below kept as N's.

Three readings:

| Option | What a new neuron in layer l reads | At layer 1 | Cost |
|---|---|---|---|
| (A) through N's wiring | its group's members' sums as N computes them, with each merged neuron below standing in for its group through a "decode" (one set of weights per merged group: how much of each old member the merged neuron stands for). B's weight from a merged neuron v to a layer-l neuron is then N's weights times v's decode | exactly class (b) | keeps "built from its group" at every depth; the decodes are new unknowns shared by all readers |
| (B) free read of what replaced its inputs | any weights on the B-neurons below that replaced the neurons its members read | not class (b): it is any unit restricted to the members' inputs | most freedom; the developer's grouping shapes less |
| (C) one layer at a time | the zoom changes one layer per level, top layer first; when layer l changes, layers below are N's and layers above refit | class (b) | simplest; all of B.W carries over; but a level can never have two coarsened layers that depend on each other's choice |

My recommendation: (A) as the meaning, because it is the only reading that equals class (b) in layer 1 and keeps
"each coarse neuron is built from its group" at every depth. Use (C) as the order of the proofs: first zooms that
change one layer at a time, then general zooms under (A).

Your choice:

| Question | Options | My recommendation |
|---|---|---|
| What does "reads only its group" mean in layer 2 and up? | (A) through N's wiring with decodes / (B) free read / (C) one layer at a time only | (A), proved in the order of (C) |

### 3-11. R-recur: is step-by-step coarsening optimal? (Hypothesis H-recur; PENDING)

**Merged into card 3-24 (2026-10-29).** Shonu: "i see zoom recursion and depth recursion as the same thing". The
question now lives in card 3-24 (node R-tree). This card stays as the record.

Your hypothesis (2026-10-27), from the essay: a smaller model's coarse neurons are coarser versions of a small
model's coarse neurons. So the zoom can be built one step at a time.

```
level 0 (N):   c0 c36 c72 c108 c144 c180 c216 c252 c288 c324     (10 curve neurons)
level 1:       [c0 c36 c72] [c108 c144 c180] [c216 c252 c288 c324]  -> 3 new neurons
level 2:       [ the three level-1 neurons ]                        -> 1 new neuron
```

Three ways to build the level-2 neuron (Definition D50):

| Way | It reads | Least loss | Cost |
|---|---|---|---|
| (a) any | any neuron below | lowest | NP-hard in general (T22, T43) |
| (b) the group | the 10 original curve neurons' pre-activations (any weights, any offset) | same or higher than (a) | NP-hard for one big group (Proposition T57) |
| (r) recursive | only the 3 level-1 neurons | same or higher than (b) | polynomial for each step that merges few neurons, for D2 (Theorem T55); no bound yet across many steps |

Why it matters: if (r) reaches the (b) optimum for the developer's tree, the rule is fast and exact. T4 already
says the developer chooses the tree. So the question is: for a clean N, is there a tree that makes (r) optimal, and
what does it look like?

Known so far:

| Fact | Where |
|---|---|
| on the detector networks, a pair of detectors merges with no loss exactly when the mix does not hold their blend | Corollaries T59.1, T59.2 |
| on the smallest detector network, every merge order has some step-by-step path to the optimum; the orders that start with a free merge reach it on every tie choice | Part C, H-recur row (numeric) |
| a fast rule that picks the tree and always wins would give P = NP, if the sizes of the numbers stayed small along the tree (not proved) | Remark B.W.2 |

No decision is needed now. H-recur is a hypothesis to test (Conjecture D52).

### 3-10. R-group: the developer chooses which neurons go coarse (R-group DECIDED 2026-10-27 as Trunk T4; R-groupopt PENDING)

Your proposal (2026-10-27). The hard part of T22 and T43 is the choice of which neurons to merge. Let the developer
make that choice, later by meaning and in large groups. The rule then only has to compute the best small network for
that choice, quickly and exactly.

```
the developer gives:   group 1 = {curve 0deg, ..., curve 324deg}  -> 1 new neuron
                       group 2 = {wheel, window, headlight}        -> 1 new neuron
                       every other neuron                          -> kept as it is
the rule computes:     the weights of the new neurons and of everything after them (no training)
```

Why it matters: this changes the input of the trunk (sizes per layer become a grouping), so only you can accept it.

What is already proved about a given grouping:

| Case | Result | Where |
|---|---|---|
| groups given, each with at most s members | exact method, time polynomial in the number of photos, exponent about s x k | Theorem T27 |
| one big group given (2 x input size) | still NP-hard | Proposition T27.1 (its network fails Q3, so it is not clean) |
| a big group in a clean network | OPEN | - |
| T43's clean hard network, the developer names exactly the detectors one new neuron covers (a clique) | easy, sketch only: T22's clique unit is a (b) unit of that group and the kept units fit the rest; not yet a stated result | - |
| T43's network, a group of two detectors | merges with no loss exactly when their blend is not in the mix | Corollaries T59.1, T59.2 |
| a group whose members are on together (AND) | collapses exactly from the weights, rank + 1 new neurons (2 for two logits) | Theorem T58 |
| T43's clean hard network, one big group (all detectors) -> 1 new neuron | the rule must still choose which detectors that neuron covers: the clique puzzle. Under (a) this is T43 itself (NP-hard). Under (b): target T57 (correction, 2026-10-27: the earlier row said "easy" for any given groups) | Theorem T43 |

Draft trunk wording (yours to confirm or change):

> Input: a network N, a grouping (each group is a set of N's neurons in one layer, with a target size k_g; neurons in
> no group are kept as they are), and base photos P (the photos N was trained on, or the developer's). Output, by a
> rule that reads N, the grouping and P and does not train: a network B in which each group is replaced by k_g new
> neurons, with the least D_P(B) among all such networks. Changing one group's size is instant. Exact closed form
> when N is perfectly clean; a bound in eps(N) otherwise.

The hypothesis inside it (node R-groupopt, H-group): the best small network over ANY new neurons is a grouped one.
Then the grouped answer is the distillation optimum. H-group is trivially true when one family's weights span the
input (Theorem T54 part 3). Theorem T25 fails it, although T25 passes strict Q2: its one family's weights do not span
the input, and the best neuron reads an input that N ignores. So H-group has content only for low-rank families
(card 3-10 of the referee round, 2026-10-27; T34 and T35 do not test it, because each of their families spans R^3).

Your choices:

| Question | Options | My recommendation |
|---|---|---|
| Accept the draft trunk wording? | yes / change it | DECIDED: "yes i do agree with the trunk you've proposed", with the zoom tree added ("how the grouping degrades over time") |
| What does "a new neuron replaces a group" mean? | (a) it feeds the readers in place of the group and may read anything below; (b) it also reads only the group's members (a fold) | DECIDED (b), delegated ("whatever decision you choose"). Reason: under (a) each new neuron may compute anything, so the grouping only sets a count and does not shape what the neuron computes. Your test "(a) = (b) for clean N" is node R-groupopt |

### 3-9. R-listen: the listening part of clean (Definition D49, Lemma T52; PENDING)

In plain words. Picture the network as a building of workers. Each worker lights up when it sees its thing. Workers
upstairs listen to workers downstairs. The listening part of clean asks: when a worker lights up, do its listeners
really respond to it? A network with random wiring must fail.

There are two kinds of sensible listener:

| Kind | Picture | What its inputs do |
|---|---|---|
| Team | a "car" listener hears wheel, window and headlight | its inputs light up together |
| Alternatives | a "some curve" listener hears the 30-degree curve or the 60-degree curve | its inputs take turns, never together |

The strict test of card 3-7 (agreement) rewards teams and punishes alternatives (Lemma T46d). I tried two fixes on
paper (D49). Both fail on alternatives, for one shared reason (Lemma T52): when one alternative is on, the others are
silent. They say nothing at that moment, so no test of the form "do the others agree right now?" can give them credit.

| Fix | What it asks | Why it fails |
|---|---|---|
| "Does this worker decide the listener?" | how often the worker's own vote turns the listener on | it is never below 0, also for random wiring; only its size could separate |
| "Do the inputs that are on agree?" | the votes of the other workers that are on at the same moment | exactly 0 for alternatives and for a listener with one input; random wiring gives a coin flip |

The next idea: score both kinds. Teams light up together more often than chance. Alternatives light up together less
often than chance (ideally never). Random wiring gives neither. Costs: the score needs a random-wiring baseline (a
size test, not a yes/no sign), and real curve detectors overlap a little, so "never together" must be graded.

Your choices:

| Question | Options | My recommendation |
|---|---|---|
| Should clean listening allow both teams and alternatives? | yes / only teams (the strict test) | Yes |
| Accept a graded test (compared with random wiring) instead of a yes/no sign? | yes / keep looking for a sign test | Yes, for now. The next step is a definition of "together more or less often than chance" and the toy table |

### 3-8. R-jobmix: write the exam in the network's own words (Definition D48, Lemma T51, Corollary T51.1; PENDING)

In plain words. The mix is an exam for the small network: a list of photos, with points for each photo. Your idea
(2026-10-26 night): strange exams may be hard, but sensible ones should not be.

The proposal (D48): you do not hand over a list of photos. You say how much each worker's job counts, plus how much
"nothing here" counts. The photos inside each job come from the photos the network was built on. Mash-ups that the
network cannot tell apart can no longer be picked one by one.

What I found (Lemma T51): the hard exam of T43 is already an exam of this kind, when its own photos are the base
photos. Each prototype lights up exactly one worker. Every mash-up lights up none, so all mash-ups sit in "nothing
here". You only say how much "nothing here" counts. Which mash-ups are in the base photos was fixed by the photos, and
that is where the puzzle is.

So the puzzle moved: from the person who writes the exam to the base photos, and there to the photos where the big
network says "nothing". The big network never had to notice which mash-ups were in its photos, because its answer on
all of them is "nothing". The small network has to notice, because it cannot draw such fine lines.

What is proved, and what is not (the referee sharpened this):
- If whoever poses the problem may also choose the base photos, the puzzle stays hard (Corollary T51.1).
- If the base photos are fixed by the network (the photos it was really built on), the old proof says nothing. In
  that proof the job weights depend only on the number of bad pairs, so the puzzle cannot be passed through them.
  Whether such exams can still be hard is OPEN. This is the hopeful case for your idea.
- A puzzle could also hide inside the network, as extra workers whose job is one exact mash-up. Those workers must
  leave the big network's answer unchanged. Lemma T53: then either nobody listens to them, or (if they cancel each
  other) one of them is heard backwards. Either way the network is not clean. (Cancelling by other kinds of
  worker is not covered.)

How this relates to the goal you dropped on 2026-10-25 (T\*\*, card 3-1): that goal had no photos at all, and T24
killed it (the same clean network, built on two different photo sets, has two different best small copies). D48 keeps
real photos (option (i) of R-explaw). It changes only your control: "how much of each job", not "which single photos".
The generalist exam then becomes a reweighting of the training photos, not the training photos themselves.

Your choices:

| Question | Options | My recommendation |
|---|---|---|
| Write the exam in the network's words (D48), instead of any list of photos? | yes / no (keep any list) | Yes, but it is not enough alone |
| What should we ask of the "nothing here" photos? | (a) nothing: some photo sets stay hard; (b) they carry no hidden pattern (for example, spread like a fixed reference law); (c) a score: how much hidden pattern they carry | (b) as the target statement, (c) as its score. Both need a definition first, and (b) needs the trunk to allow more than finite lists |

### 3-7. R-cleannet, revision 3: clean = interpretable, from exploration (Definitions D43 to D45; PENDING)

In plain words: I explored InceptionV1 the way the essay does (log: `research/clean/explore/EXPLORE-LOG.md`) and
turned the moves that mattered into a definition. A unit is clean when it is:

| Part | Plain meaning | Where it came from |
|---|---|---|
| READ | Where the unit fires, its readers turn on (the ones it votes for) or stay off (the ones it votes against) | the essay's "built to be read"; a random network fails this and passes everything else |
| ONE JOB | Split the unit's inputs by which unit below casts the largest vote (its "route"). Every route feeds the same readers | the essay's "look upstream at how it's built and downstream at what it does, and both line up" |

Why this answers your point about 4e:55: no class name enters. "Cats and cars" is a statement about our labels. The
question D45 asks is whether the cat route and the car route feed different readers. For 4e:55, one layer up, they
mostly do not.

Proved: the indices use no labels, no scales and no way of writing the readout (T46); three planted circuits
behave as they should (a two-job unit scores above 0; the V case and the ambulance score 0, and all three are read;
the old D42 got the ambulance wrong) (T47); in a simple one-input class of networks, the perfectly clean ones include
a whole region of weights (T48), while exact symmetry holds only on a set of size zero (T44); T43's detector networks
are perfectly clean under D45 (T49), so the T43 hardness stands (it lives in the mix). Limit: at the last hidden
layer, the use includes which class wins, so D45 cannot tell packing from reuse there (comment (h)).

Your choices:

| Question | Options | My recommendation |
|---|---|---|
| Accept "read + one job" (D45) as the definition of clean? | yes / change a part | Yes |
| Which inputs define a unit's target (node R-explaw)? The moves D45 rests on need a law on inputs | (i) the data the network was built for; (ii) inputs the network generates itself (feature visualization with jitter and rotation) | (i) for now: (ii) agreed with (i) on 4e:55's poles, but routes and reads were only measured with (i) |
| The READ part: which form? Most units of a random network also have a positive read index (0.77 of them, about 30 times smaller); Lemma T46b proves why (a reader always responds a little to the unit's own vote) | (1) weak form: lambda > 0. T48 and T49 hold; the T43 detector networks are clean in it, so T43's hardness stands. (2) agreement form: the excess read xi > 0 (the reader's OTHER votes agree with the unit). It passed H-agree on fresh units (trained 34 of 36 positive; random 11 of 32; sizes 7.7 times apart). The T43 networks fail it (T50). Costs: it also fails a coarse reader that ORs siblings which are never on together (Lemma T46d), which is the family-and-coarse-unit picture this track began with; and the referee sketches T43-style networks with nested helpers that may pass it and stay hard (Conjecture D47) | Use (2) as the read test for real networks (it is the one that separates trained from untrained), and keep (1) for the theory until D47 is settled. Do not expect (2) alone to remove the hardness |
| Use measured one layer up, or further? | one layer (now) / to the output | One layer: the output version recorded which class won, not the unit's job |

### 3-6. R-cleannet, revision 2: a definition under which real networks can be clean (Definitions D41, D42; Theorems T44, T45; PENDING)

In plain words: you said our clean networks are rare, while the essay says trained networks are clean. You were
right, and the cause was our definition, not the networks.

| Why the old definition made clean rare | What the new one does |
|---|---|
| It built families from an exact symmetry. Theorem T44 proves that exact symmetry holds only on a set of measure zero. This is the one place that made clean rare | Symmetry is not a principle any more. The essay never mentions symmetry. Symmetry stays as an optional hypothesis |
| It checked every possible input, including noise images (this made the scores strict, not rare) | Each unit is judged on its own "target": the inputs where it fires, each counted by how strongly it fires. The essay: "think of a neuron as a target", and "a blind maximum tests the neuron's center, not its breadth" |
| It took the worst unit (also strict, not rare) | Each entry is a share over units, weighted by how much the readers use each unit. The essay counts "roughly 90% of neuron poles" |

The three principles (each a number from 0 to 1, computed from the weights and the input space only):

| Principle | Plain meaning | Essay sentence |
|---|---|---|
| V, votes agree | Where a unit fires, each of its readers hears votes of one sign. A reader may vote against a unit only where that unit is off. This also catches "arbitrary packing" | "A neuron is built from votes"; "the V neuron's negative side is an O" |
| G, a gate in front | Some unit below that this unit reads positively is on across this unit's whole target | "often its job is to gate a region of the network" |
| B, built upon | Some reader above that reads this unit positively is on across this unit's whole target | "built to be read, and built upon, by its downstream readers" |

What is proved: under the new definition, the perfectly clean networks contain an open set (Theorem T45), so they
are not rare in the way exact symmetry is. Inside that set (the "margin-clean" networks) a small change of the weights
keeps a clean network clean. Outside it, a small change can break cleanliness. What is not proved: that trained networks score low. That
is Hypothesis H-typical, to be measured on InceptionV1 against a random network and a network that packs unrelated
jobs together.

What did not change: the detector networks of T43 are still perfectly clean at one hidden layer. So your hardness
conjecture is still refuted there (card 3-5 stands).

Your choices:

| Question | Options | My recommendation |
|---|---|---|
| PARKED 2026-10-26 evening (Shonu: "the first principle is actually exploration"; this waits for the exploration log). First: which form of V? The current V marks the essay's own ambulance circuit as unclean, because on a taxi the ambulance unit hears "car body" (+) and "taxi sign" (-) together. That is the distinction the essay calls clean, not a cancellation | (1) keep V as it is; (2) judge votes on the reader's own target and let B (extended to the last layer) catch arbitrary packing | (2). Check E9: the ambulance circuit gets B = 0 and V 0.056 (was 0.149); a packing circuit gets B 0.215; the T22 and T23 teachers still fail; T20, T32 still pass. V does not reach 0 for the ambulance under (2), so V stays graded there |
| Then: accept D42 (with your choice of V) as the definition of clean (node R-cleannet)? | yes / change a principle / add a principle | Yes. Before the H-typical run, the definition must also be extended to convolutional layers with pooling (InceptionV1) |
| Drop "siblings never fire together" (old Q2)? | drop (siblings may overlap, as curve detectors do) / keep | Drop. What it aimed at is now inside V |
| At the last layer, should every pair of classes count as a reader? | every pair (now: two units that fire together must rank every pair of classes the same way, which is strict with 1000 classes) / only the pairs that matter at the input | Open. I lean to "only the pairs that matter", but it needs a definition first |
| Which inputs count as "the space of images"? | the plain pixel box / the box smoothed by small shifts and rotations (the essay says feature visualization needs "transformation robustness") | Measure both in the first H-typical run |

### 3-5. R-mixclean: a clean network does not make the hard part easy for every mix (Theorem T43, PENDING)

In plain words: you guessed that the NP-hardness (T22) would go away for a good clean network. The hard networks of
T22 were "broken lines" whose output goes up and down along one direction. The new principles of clean (paper A.5)
rule them out. But T43 builds a different hard case, with networks that pass the principles: all of them at one
hidden layer, and all except the gate principle (Q4) at more layers.

| Piece | What it is |
|---|---|
| The network | Several separate input blocks. In each block, narrow direction detectors. No two detectors are ever on together. One reader adds them all up with positive weights. With more layers, one coarse unit sits above them (the "hanging-ears" unit). Swapping the blocks leaves the network unchanged. |
| The mix | Each detector's best input (a "prototype"), plus the blend of each "bad" pair of prototypes, with a large weight. Swapping the blocks leaves the mix unchanged too. |
| Why it is hard | One coarse unit must fire on prototypes from different blocks. But a single unit that fires on two prototypes also fires on their blend. So it must pick prototypes with no bad blend between them. That is the clique problem. |

What this shows. The network does not depend on the hard instance at all; only the mix does. So for these networks
the hard part is carried by the mix. A definition of clean may use only the network (your rule C1). So no definition
that accepts these networks can remove this hardness for every mix.

What it does NOT show (the referee made me cut three overclaims):

| Claim I cut | What is actually true |
|---|---|
| "This contradicts the trunk" | The trunk never says "fast". It allows any one-time computation per mix (the zoom). T43 says: unless P = NP, that zoom cannot run in polynomial time for these networks. It conflicts with the trunk only if you want the zoom to be polynomial-time. |
| "The best unit fires on a union of the network's detectors" | Proved only for good-enough builds, and only about where their output is large: there it sits exactly on a union of the network's own detectors (a clique). Nothing is proved about the single best build. |
| "Option (d) only rules out the essay's picture" | The detector networks are very sharp. Their weights grow like m^4 for m detectors per block. A principle that limits sharpness, or a box-shaped input domain, rules them out without ruling out "detectors read by one reader". Whether the hardness survives that is open. |

The options:

| Option | Meaning | What it costs or needs |
|---|---|---|
| (e) keep the trunk as written | No time limit on the one-time zoom | Then T43 is no conflict. But the "closed form" may then take exponential time to compute for some mixes. |
| (a) a property of the mix | Allow only mixes with some property beyond symmetry | You said P should be "something the network can use at some capacity". That has to become a definition first. |
| (b) a weaker meaning of "polynomial" | Polynomial when one coarse unit merges only a few blocks or families | Not proved. In T43 the number of blocks is half the input dimension, so it cannot tell the two apart. |
| (c) an error | Give up "exact" | T22 part 4 carries over: even an error factor of 1 + 1/(16 n) stays hard. Looser errors are open. |
| (d) a principle these networks fail | For example, bounded sharpness, or a box domain | Open whether the hardness survives. This is the most interesting mathematical route. |

My recommendation: decide (e) first, because it tells us whether a time limit on the zoom is part of your goal. If
yes, try (d) with bounded sharpness first, since it stays a property of the network alone (your rule C1).

Also open from this session (paper A.5), for your choice:

| Question | Options |
|---|---|
| Q2: may siblings in one family overlap? | Strict: never on together (rules out T32 to T34, but also broadly tuned real families like curve detectors). Broad: each one is the strongest at its own peak (keeps T32). |
| What is a family when the network has no exact symmetry (the ear units)? | Open. Routes: the units one coarse reader summarises, or maximal sets of units that are never on together and share a reader. |
| Which input domain belongs to a network's kind? | The detector networks are clean on a ball-like domain, not on a box. |

### 3-4. R-gatefold: a gate in front of each family (Lemma T39, Conjecture D34; PARKED 2026-10-26, nodes DROPPED)

**Parked (2026-10-26).** Shonu: "keep gates as like a side thing and maybe come back to it later". The gate work
was tied to one way of solving the problem. The object now is the general optimum (B.S). Part B.G stays in the paper
as a record. The three nodes below are DROPPED, so its results are a side branch.

Shonu (2026-10-26), asked "Is this your meaning of fold: the network's own gate decides where the coarse unit is on,
and an easy formula decides what it computes there?", answered: "yes that is it. i also want to ensure that you are
still doing this on these sorts of toy examples because that's how we move forward in the theorem. we haven't yet
expanded the theorem to realistic networks. everything must be derived from mathematical foundations".

In plain words: the tiny networks of T32 to T34 have no gate, so they could not test this meaning. The new nodes add
one.

| Node | What it assumes | Kind |
|---|---|---|
| R-gatefold | a unit in layer 1 (the gate) such that the network's output is constant wherever the gate is silent: all of the family's distinctions live inside the gate's region. With a symmetry group, the group fixes the gate. | STRUCT (a claim about clean networks) |
| R-flatgate | the gate has one value on its whole region | CONV (makes a proof step work) |
| S-keepfirst | two hidden layers, and the build keeps layer 1 | SCOPE (the smallest network that can have a gate in front) |

| Result | Status | What it says |
|---|---|---|
| Lemma T39 | proved | If the coarse unit's region is the gate's region, no such unit beats one least-squares fit (or one logistic fit for KL) on that region, and the fit is reached when one sign condition holds (T39a). No search, no training. |
| Lemma T39a | proved | When that best unit can be built (one sign condition) |
| Corollary T39.1 | proved; instance I3 checked by evaluation | If no member bends inside the gate, the gate is flat on its region, and the members' sum minus N's value off the gate has one sign inside it, then a unit built from N's own weights copies N exactly |
| Remark B.G.1 | checked by evaluation | Even then, other exact builds exist that never read the gate |
| Theorem T40 | proved (Lean for the arithmetic) | One perfectly clean gated network where no gate-region unit is best (7/182 against 5/182) |
| Conjecture D34 | refuted (T40) | "With one gate, the best one-unit build takes the gate's region" |

Shonu (2026-10-26) on the error that matters: "i mean error in what the optimal small network would be given the
large network". T39.1 is a case where the optimal small network costs nothing, so it does not test that error. On
T40's toy the gate rule's error against the optimal small network is at least 1/91.

What is proved is one sufficient condition (T39.1) and one counterexample (T40). Which property of the network inside
the gate decides the case is open. "Every family sits behind a gate and is straight inside it" is one candidate for
option (a) on card 3-3. No node for it yet. These three nodes are now DROPPED (parked, see above).

### 3-3. R-foldopt: for symmetric clean networks, distillation does not converge to folds of the families (Corollary T35; DECIDED 2026-10-26: option (c))

**Decided (2026-10-26).** Shonu: "Folds and other interpretability stuff ... They're really just my hypotheses for the
ultimate thing I'm really interested in is ... what is actually the optimal subnetwork given a large network" and "I
don't ever want us to like say, hey, we have to do this method just because". So option (c): the optimum is the
object. Folds, gates and families are hypotheses about it, never required methods. Option (a) was framed as "a
stronger clean". Shonu reframed it: "not making a stronger version of clean, but just a good version of what a clean
network is". That is now the work under node R-cleannet (paper Part A.5). The text below is the card as it was.

In plain words: we tested your conjecture, "for a clean network, the best small network is a fold of its families",
with exact proofs on tiny perfectly clean networks (families of rotated units, rotation-symmetric samples). The proofs
are exhaustive and exact (Lemma T32a), and two of them hold for the trunk's own score, KL.

| Theorem | Network | Width | Score | What beats the folds |
|---|---|---|---|---|
| T32 | one family of four, in the plane | 2 | squared error | a build with a unit that fires where every fine unit is silent (covering folds lose by at least 1/9) |
| T33 | one family of four, in 3-D | 1 | squared error and KL | a unit that points against the family (its negative side); it copies the network exactly |
| T34 | two families of four, in 3-D | 1 | squared error and KL | a unit that mixes the two families |

What fails is "each coarse unit is a fold of one family", and "coarse units fire only where fine ones fire". Folds of
mixed groups with free thresholds survive, but they say nothing: when the layer's weights point in all directions they
include every unit (Corollary T35). So, with "clean" meaning symmetric families (D20), distillation does not converge
to folds of the families. This is a fork. The ways around it are the theorems' premises:

| Option | Meaning | What it costs or needs |
|---|---|---|
| (a) a stronger "clean" | find the property of real clean networks that these tiny networks lack | The essay's candidate: the coarse unit of each family is itself present in the network ("pale, wavy and head-side ear neurons all feed one hanging-ears neuron"). None of the three networks has that. It must be defined from the weights, then T32-T34 must fail under it. |
| (b) folds as a restriction | build from folds anyway and accept a loss | Losses at eps = 0: T32 at least 1/9 relative; T33 from 0 to at least 1/468; T34 from at most 25/216 to at least 7/60. You rejected bounds at eps = 0, so this changes your bar. |
| (c) keep any units | drop the fold conjecture; characterise what distillation converges to directly | A problem in its own right (best ReLU-ridge approximation of a sum of ReLU ridges); the three builds above are the first data. |

Not ways out: the score (T33 and T34 hold for KL), and folds of mixed groups (they restrict nothing). Larger samples,
depth and more classes are open, but a counterexample in a special case refutes the general claim.

The fork's premises rest on nodes you have not decided: R-cleannet (D20 as "clean"), R-fullrank, R-symN. Option (a) is
a proposal to replace R-cleannet.

My recommendation: (a), because it is your thesis route and it is testable: define "the coarse unit is present" as a
property of the weights, then check whether T32-T34's failures disappear under it. Keep (c)'s data as the control.

**Update (2026-10-25, later): what option (c) found.** You asked what the optimum actually is. Paper section B.O.

| Result | What it says |
|---|---|
| Lemma T36 | if the network and the mix are symmetric, every turned copy of a best build is also best |
| D29, D30 (rings, n = 3 to 6) | numerically contradicted, not proved false: the best single unit was found off every mirror axis; firing where the family is silent gave no gain |
| Theorem T38 | a necessary condition at the optimum: each coarse unit's straight-line piece is the best straight-line fit, on its own half-space, of what the other units leave unexplained (a backfitting fixed point) |
| Corollary T38.1 | with one unit: on its half-space the build is the best straight-line fit of the big network; off it the build is the big network's average there |
| Corollary T38.2 | inside a deep network, with plain error at the next layer: the fit is of the family's output as seen through the coarse unit's own outgoing weights |
| Corollary T38.3 | when later layers look at neighbouring positions, the condition weighs errors at those positions too |
| H-dialfit (InceptionV1 3b) | the trained coarse unit is NOT near the position-local fit of T38.2 (correlation 41/100, bar 8/10; slope 1.79, predicted 1) |

Reading (one unit, Definition D31): a coarse unit is a "dial" (fires on at least half the mix: a straight-line summary
there) or a "detector" (fires on less: a summary on a small part, the average elsewhere). T38 does not say which region
a unit takes. That part is the hard one (Theorem T22). With two or more units the clean reading holds only when their
regions are nested or apart. This does not decide the fork. It replaces option (c)'s "a problem in its own right" with
a known necessary structure plus one open part: the choice of regions.

**Update (2026-10-26).** Shonu confirmed his meaning of fold: the gate decides the region, a formula decides the
value ("yes that is it", full quote in card 3-4). This is close to option (a). R-foldopt itself stays PENDING: the
answer was about the meaning of fold, not a choice among (a), (b) and (c). The work under it is card 3-4.

### 3-2. R-fold: any new units, or folds of the network's own units? (Theorems T25 to T30; DECIDED 2026-10-25: any new units kept, and "for clean networks the optimum is a fold" became Conjecture D25)

In plain words: T25 showed that if any new units are allowed, the best small network can read a feature of the sample
that the big network never uses. You asked for the case for folds to come from the project's goals. Each goal became
a condition (Definition D24); a referee then checked what each one really forces. The honest result (Corollary T30):

| Your goal | As a condition | What it forces |
|---|---|---|
| "The training has already been done" | G1: the answer must depend on the network's weights, not only on its answers on the sample | Rules out the unrestricted rule: with any new units the answer depends on the network only through its answers on the sample (T26), so it is distillation. But G1 does not single out folds: any class built from the network's weights passes it. |
| "Download once, fold many times" | G2: a unit adds no distinction the network does not make | Restricts nothing in the usual case, where the first layer reads every input direction (T28). Weak. |
| "Fold up something that already exists" | G3: a coarse unit fires only where one of its fine units fires | Forces folds exactly (T29, Farkas), for single ReLU units, groups with a common silent input, and inputs over all of R^d. This is the condition the case rests on. |

Two more facts. With any new units, a developer's sample of at most d + 1 images in general position gets a perfect
copy of EVERY network from its answers alone (T28 part 2). The exact fold rule costs an exponent of order
(group size) x k, small for a zoom over small families (T27); for large groups it is as hard as the unrestricted
problem (Proposition T27.1).

So the decision is really: is G3 what you mean by "fold up what exists"? If yes, folds follow.

| Option | Meaning |
|---|---|
| (a) folds only (the zoom, G3) | The trunk's "built from any new units" becomes "built from folds of N's units". Recommended, if G3 matches your meaning. |
| (b) any new units, plus a property of the mix | Keeps the trunk's words, with a condition on every mix that removes accidental features. Not shown to make the best build a fold, and it may fail (Remark B.Z.2). |
| (c) any new units | The rule is distillation (T26); on small samples its answer carries nothing from the network (T28). |

Not shown: that folds lose nothing for clean networks. That is the open question after this decision. Also open: depth
(whether the best folds of layer 1 depend on the size chosen for layer 2), which is where switching sizes gets hard.

### 3-1. R-score: what "best" means with no input data (Theorem T24, PENDING; the first decision under T\*\*)

In plain words: the new trunk gives the rule no data. But "the best small network" still has to be best at
something. Theorem T24 shows the obvious answer does not work. Take a perfectly clean network (two mirror-image units).
It is the best possible predictor on any list of inputs labelled by itself. Yet for two such lists, even two lists
that are themselves mirror-symmetric, the best 1-unit network is different. So a rule that sees only the network
cannot be best for "the data it was trained on".

| Option | Score | What we would prove next | What it means |
|---|---|---|---|
| (i) intrinsic | a score built from N itself by a fixed rule (an input law made from N, for example from its own feature visualizations, or a comparison on N's own units) | first a candidate rule, then the closed form for clean networks under it | Matches "derive everything from the larger network". Data-free distillation already builds input laws from a network alone, but with free choices (priors, step counts); a rule with no free choices is not known to me. It has to be stated first. |
| (ii) training law, with a stronger property | KL on the training data, plus a property that restricts the training INPUTS (by Lemma T24a, "trained to the optimum" puts no condition on the inputs at all) | that the property makes the network's weights fix the best build. A lead: SGD leaves per-neuron imprints in the weights (old log, evidence only) | Closest to the essay's evidence (its crush retrained on the training data). Needs "well-trained" to mean "trained by SGD on natural images" in a precise form. |
| (iii) worst case | the worst single input anywhere (Corollary T24.1: the worst case over all consistent data collapses to this) | the closed form for clean networks under the worst-case score | Fully defined from N alone, today. But it is driven by inputs that are not images, so "best" means "best on the worst noise". And with goals that reweight inputs (D18) it ignores the goal; keeping the goal needs a pointwise goal loss (node R-goal). |

The options are the ways around T24's two premises: that the score is KL on the training data, and that the only
condition on that data is "the network is optimal on it".

My recommendation: (ii) with SGD as the property, because it is the only option whose "best" matches what the essay
measured, and the old log has a lead for it. (i) is closest to your words, but it needs a canonical construction that
does not exist yet. (iii) is ready today but measures the wrong thing.

### 3-0. R-least: what "the least" means when it does not exist (Theorem T23)

In plain words: for one small network with three classes, small networks of width 2 can copy it better and better,
with no limit, but no network of width 2 copies it exactly. Their weights grow without bound on the way. So "the best
network of width 2" does not exist for this network.

| Option | Give up | What we would prove next | What it means for Cale |
|---|---|---|---|
| L1 | "the least" | builds within a stated eps of the infimum (an eps-best build) | The natural repair. Every size has an eps-best build. The nesting and speed questions are then asked for eps-best builds. |
| L2 | "any weights" | the least over builds with weights of size at most R (this exists, by compactness) | A size cap on the weights. Shipping networks have finite weights anyway, but R becomes a new input. |
| L3 | "same kind as N" | the least over the closure (limits of networks, which may not be networks of width k) | The limit can need more units than k, so the size promise breaks. |
| L4 | "every network" | a named property under which the least exists (two logits, one hidden layer and a finite mix is one such case: T19) | Needs a property real networks have. |
| L5 | "every mix" | a named property of image mixes under which the least exists | T23's six points are artificial. |

Not proved: whether T20 survives option L1. Under L1, eps-best builds of width 2 can have units close to the four
units of T20 without being equal to them, so T20's margin (a width-1 margin) does not settle it; a separation lemma is
needed. T22 survives L1 when eps is part of the input and may be as small as 1/(32 Z); for a fixed tolerance nothing
is proved. Sections 3a and 3b speak of least builds, so they are conditional on the answer to 3-0.

### 3a. R-nest: least at every size, or nested (Theorem T20)

In plain words: for one tiny network, the best 2-unit network and the best 1-unit network share no unit. So no single
stored network gives the best network at every size by keeping its first units.

| Option | Give up | What we would prove next | What it means for Cale |
|---|---|---|---|
| N1 | nesting | the least build at each size, separately | Cale keeps one network per size, or builds the size it needs. Switching size means loading another network. |
| N2 | "least at every size" | the best nested list, judged as a whole by a stated rule (for example a weighted sum over sizes, or the best largest network with its prefixes as they fall) | One stored network, instant switch. A smaller size is good but not the best possible. |
| N3 | "every network" | a named property of networks under which least builds nest. The shared-gate class of Theorems T6 and T9 is one such class | Needs a property that real networks have, and a proof. |

T20's other premises (one hidden layer, two logits, KL or D2) are inside the trunk, so they are not ways out: a
counterexample in a special case refutes the general claim. They matter only for a restricted trunk under N3 (nodes
N-ny2 and N-onehidden).

### 3b. R-root: least, fast, for every network and every mix (Theorems T21 and T22)

In plain words: finding the best 1-unit network for a given network is as hard as finding a clique in a graph, a
classic hard problem. So no quick calculation does it for every network and every list of images, unless P = NP. A slow
exact calculation exists (T21). Its cost grows like n^{d k}, which is far too slow when d is the size of an image.

| Option | Give up | What we would prove next | What it means for Cale |
|---|---|---|---|
| A | "every network" | an exact fast rule for networks with a named property | The property must exclude T22's teachers. Those memorise arbitrary labels on artificial points, with one unit per point. |
| B | "every mix" | an exact fast rule for mixes with a named property | T22's mixes are artificial point clouds, not images. |
| C | "least" | a fast rule with a proved bound on how far it is from the least | T22 part 4 rules out a relative error of 1 + 1/(16 n), n the number of points. Looser errors (a constant, or 1 + 1/sqrt(S)) are open. |
| D | "fast" | an exact rule run once per model before release | T21 gives one only for two logits, one hidden layer and D2 (cost n^{d k}); for KL it gives finitely many convex problems. Deeper builds have no exact rule yet. Only workable if the network's structure makes the effective d small. |

A and B are close: both name a property that real networks or real images have and that T22's artificial instances
lack. That property has to be stated before anything can be proved under it.

My recommendation (Shonu decides): for 3-0, L1 (eps-best), since every real rule works to a tolerance anyway. For 3a, N2, since it keeps one download and an instant switch, which is what Cale
needs. For 3b, A and B together, as one property of "networks trained on images". The first step would be to look at
which feature of T22's hard instances a trained network never has, and state that as a definition.

## 4. The nodes

Kinds:

| Kind | Meaning | Who may add it |
|---|---|---|
| ROOT | The root decision | Shonu |
| CONV | A convenience: makes a proof work, is false for real networks | proposed by me, decided by Shonu |
| STRUCT | A claim about trained networks or about images. Can be near true. Measurable in Part C | proposed by me, decided by Shonu |
| SCOPE | A smaller question than T\* (for example one family, not the whole network) | proposed by me, decided by Shonu |
| NEG | The setting of a counterexample or a hardness proof. A hardness result about "the best build" carries over to another class only when the set of allowed builds is the same, or through a stated reduction. Containment of the networks alone is not enough | I may add it, with that check written down |

Statuses: DECIDED (Shonu, with his words and a date), PENDING (waits for Shonu), RETRO (added before 2026-10-25
evening, waits for Shonu's review), DROPPED (Shonu removed the branch; its results stay as a record).

<!-- nodes -->
| ID | Kind | Axis | Real network -> restriction | Carried by | By | Added | Status |
|---|---|---|---|---|---|---|---|
| R-root | ROOT | goal | exact and fast and every network -> give up one (section 3) | this file | Shonu: "I want to define a property of networks ... I don't want to require input diagnostics" (option A, and no input data) (2026-10-25) | 2026-10-25 | DECIDED |
| R-family | SCOPE | scope | whole network -> one family replaced, the rest of the network kept | D6, Setting of B.0, B.0 to B.3 | me (2026-10-22, the family crush) | 2026-10-24 | RETRO |
| R-local | CONV | objective | KL -> its second-order form at each input, no gate flips | Lemma B0.1, Setting of B.0 | me | 2026-10-24 | RETRO |
| R-fixedread | SCOPE | scope | best read and write -> best write for a fixed read | Theorem T0 | me | 2026-10-24 | RETRO |
| R-bench | SCOPE | scope | all weights free -> only some reader weights free (frozen, biases, re-route) | D6 | Shonu: "readers adapt", then "readers fully adapt" (2026-10-24; recorded by me from memory notes, not yet confirmed in chat) | 2026-10-24 | DECIDED |
| R-obj | CONV | objective | KL -> squared error of centred logits (D2) | D11, Parts B.4 to B.7, TP1, TL5; Proposition TY5 prices it (card 3-30); Part B.I replaces it by KL where it can | me (B.4); removed by Shonu on card 3-30, option (a): "I'd like you to just refit everything, but specifically to find the optimal KL. And I really think that this should optimal KL specifically for everything, including deep networks" (2026-10-31). Its results stay as a record, labelled OPT-D2 | 2026-10-25 | DROPPED |
| R-gate | CONV | nonlinearity | one ReLU gate per unit -> one gate per family and position, from a separate gate path that every build keeps at full size | H-gate, D11 | Shonu chose H-gate (plan of 2026-10-24, FOLD; recorded by me from the plan, not yet confirmed in chat) | 2026-10-25 | DECIDED |
| R-selfgate | CONV | nonlinearity | one ReLU gate per unit -> one gate per layer and position, computed from fixed reads of the value path | H-selfgate, D12 | me, in a plan Shonu approved as a whole (NOODLE) | 2026-10-25 | RETRO |
| R-pointwise | CONV | edges | spatial windows at every layer -> 1 x 1 edges, a fixed feature map below the first family, a weighted sum over positions at the top | D11, D12 | me | 2026-10-25 | RETRO |
| R-sgate | CONV | edges | ReLU layers with windows -> windows allowed, but gates shared per position (from a gate path, or self-gated at one position) | H-sgate, D13 | me | 2026-10-25 | RETRO |
| R-extra | SCOPE | size | k units in each layer -> k at the top, more below (r extra units, or the visible widths) | T11.1, T13 | me | 2026-10-25 | RETRO |
| R-gauss | CONV | input law | natural images -> Gaussian input field, independent positions | H-gauss | me, in a plan Shonu approved as a whole (SWITCH) | 2026-10-25 | RETRO |
| R-centred | CONV | input law | input with a bias entry -> centred input (m_a = 0) | T17.2, T17.3 | me | 2026-10-25 | RETRO |
| R-depth1 | CONV | depth | many layers -> one layer (m = 1) | T15, T17 to T17.3, D9 | me | 2026-10-25 | RETRO |
| R-onecol | CONV | nonlinearity | a gate may read many sums -> each gate reads one sum | T17 to T17.3, D9 | me | 2026-10-25 | RETRO |
| R-affine | CONV | nonlinearity | step gate (as in ReLU) -> affine gate phi(t) = alpha t + beta | T17, T17.2, T17.3, D9 | me | 2026-10-25 | RETRO |
| R-lip | CONV | nonlinearity | step gate -> smooth gate with bounded slope (a price is paid for the curve) | T17.1 | me | 2026-10-25 | RETRO |
| R-step | CONV | nonlinearity | one ReLU gate per unit -> one shared step gate | T15, T17.1 part 5, D7 | me | 2026-10-25 | RETRO |
| R-pool | CONV | pooling | max-pool -> linear pooling | H-pool | me | 2026-10-24 | RETRO |
| R-constmetric | CONV | objective | per-input metric -> one constant metric | Remark B1.4, T0 part 3 | me | 2026-10-24 | RETRO |
| R-kron | CONV | data | the data object of T0 -> one exact Kronecker product | H-kron | me | 2026-10-24 | RETRO |
| R-plane | CONV | input law | pixel images -> fields on the whole plane (every rotation acts) | D8 | me | 2026-10-24 | RETRO |
| R-clean | STRUCT | structure | any family -> an exact orbit of one template, equally spaced | H-clean | the essay's thesis; named by me | 2026-10-24 | RETRO |
| R-read | STRUCT | structure | any readers -> readers that treat the family symmetrically | H-read | the essay's thesis; named by me | 2026-10-24 | RETRO |
| R-sym | STRUCT | input law | any image law -> invariant under rotations and shifts | H-sym | me | 2026-10-24 | RETRO |
| R-circ | STRUCT | structure | any co-activity and reader matrices -> circulant ones (a symmetry stated directly) | Theorem T1a | me | 2026-10-24 | RETRO |
| R-joint | STRUCT | structure | one family -> several families with one shared symmetry | H-joint | me | 2026-10-24 | RETRO |
| R-gram | STRUCT | structure | co-activity from images -> co-activity equal to the kernel Gram matrix | H-gram | me | 2026-10-24 | RETRO |
| R-eq | STRUCT | structure | any chain -> a chain that commutes with a cyclic symmetry | H-eq | me | 2026-10-25 | RETRO |
| R-symN | STRUCT | input law | any image law -> invariant under the finite symmetry of H-eq | H-symN | me | 2026-10-25 | RETRO |
| R-sumread | STRUCT | structure | any readout -> a readout that sees only the member sum | T10.1 | me | 2026-10-25 | RETRO |
| R-invread | STRUCT | structure | any gate read -> one rotation-invariant read per gate | T11.2 | me | 2026-10-25 | RETRO |
| R-contour | STRUCT | input law | any image law -> random smooth curves (curvature process) | H-contour | me (from the literature) | 2026-10-24 | RETRO |
| R-detector | STRUCT | structure | any curve unit -> an ideal detector of one tangent element | H-detector | me | 2026-10-24 | RETRO |
| R-shot | STRUCT | input law | any image law -> curves that add, as a Poisson process | H-shot | me | 2026-10-24 | RETRO |
| R-stat | STRUCT | training | any weights -> weights at a stationary point of the training loss | H-stat | me; DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out"; I chose accept: TQ12 part 2 writes its error out with no premise, parts 3 and 4 state the premise; realism checked on a net whose whole P we hold | 2026-10-24 | DECIDED |
| S-datamath | SCOPE | scope | a statement about networks -> a statement about image laws only | Part B.2 | me | 2026-10-24 | RETRO |
| N-finite | NEG | input law | any law -> a law on finitely many points | T14, T15, D7, D9 | me | 2026-10-25 | RETRO |
| N-onepos | NEG | edges | many positions -> one position | T15, D9 | me | 2026-10-25 | RETRO |
| N-ny2 | NEG | output | many classes -> two logits | T15, D9 | me | 2026-10-25 | RETRO |
| R-pwl | STRUCT | architecture | any layers above the family -> piecewise-linear layers (ReLU, max-pool, linear) | Lemma B0.1 | me | 2026-10-25 | RETRO |
| R-diagmetric | CONV | objective | the full metric of T0 -> a diagonal, position-local metric weighted by each reader's gate | T0 part 4, Lemma T1c | me | 2026-10-24 | RETRO |
| R-faithful | SCOPE | scope | best over all builds -> best over builds that keep the network's switch reads | T11, T11.2 | me | 2026-10-25 | RETRO |
| R-range | SCOPE | scope | every build -> builds (and teacher) whose logits have range at most R | T10.3 | me | 2026-10-25 | RETRO |
| R-linseen | CONV | input law | every kept space -> kept spaces whose unseen part has a linear conditional mean | T17, T17.1 | me | 2026-10-25 | RETRO |
| R-field | STRUCT | structure | any family -> a field on the circle sampled at the network's own angles (unequal spacing allowed) | Conjecture D1 | me | 2026-10-24 | RETRO |
| R-readergate | STRUCT | structure | any reader gate -> a gate that is a function of the contour at the reader's position | Conjecture D2 | me | 2026-10-24 | RETRO |
| S-linear | SCOPE | scope | curves of any length -> the linearised short-arc regime | Setting of B.2 | me | 2026-10-24 | RETRO |
| R-gaussnu | STRUCT | input law | any law of the starting curvature -> a Gaussian law | T3 part 2, T3.1 part 2 | me | 2026-10-24 | RETRO |
| N-linear | NEG | nonlinearity | gated chain -> constant gates (a linear chain) | T14 | me | 2026-10-25 | RETRO |
| N-onereadpos | NEG | edges | readout over all positions -> readout at one position | T14 | me | 2026-10-25 | RETRO |
| R-nest | ROOT | goal | least at every k and nested -> give up one (Theorem T20 proves both cannot hold for every network) | this file, section 3 | Shonu: "defining a goal for nesting"; best builds are defined per goal; instant switching kept as a cost ("I do want instant switching as a capability that's supported by the mathematics") (2026-10-25) | 2026-10-25 | DECIDED |
| S-twoclass | SCOPE | output | many logits -> two logits. A positive result proved only there; not proposed as a restriction (the general case is Conjecture D17) | T19, T21 | me (partial result); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as the stated scope of the results that use it | 2026-10-25 | DECIDED |
| R-mapreal | STRUCT | structure | any network -> a network with a map of small errors whose regions are extensions of human concepts (Hypothesis H-map). Under T5 a map is an input, so this node is about whether real networks have good maps, not about the rule | COARSE-THEORY.md A.7, Part C | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a named hypothesis, measured in Part C | 2026-10-28 | DECIDED |
| S-cover | SCOPE | map | any map (Trunk T5) -> a map whose districts of the last hidden layer contain every unit of that layer; with district sets equal or disjoint and eps_dist = 0, this puts the supports of all units of the layer inside one partition of P (the regions and R_0) | Theorem TM1, card 3-18 | me (partial result); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as proof scope | 2026-10-28 | DECIDED |
| S-lastlayer | SCOPE | depth | groups in any hidden layer (Trunk T5) -> groups in the last hidden layer, where the logits are linear in the units and a build's change at a point is a sum of the groups' changes. A positive result proved only there; deeper layers need the meeting-unit analysis (Conjecture CU2 of UNIT-BRANCH.md) | Theorem TM1, card 3-18 | me (partial result); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as proof scope | 2026-10-28 | DECIDED |
| S-regiongroup | SCOPE | zoom | any zoom whose groups are unions of districts (Trunk T5) -> zooms in which every region meets at most one group and the judgment units of that region are kept | Theorem TM1, card 3-18 | me (partial result); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as proof scope | 2026-10-28 | DECIDED |
| S-onehidden | SCOPE | depth | any number of hidden layers -> one hidden layer. A positive result proved only there; not proposed as a restriction | T19, T21 | me (partial result); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as proof scope | 2026-10-25 | DECIDED |
| N-onehidden | NEG | depth | any network -> a teacher with one hidden layer, builds with one hidden layer. Carry-over check: these builds are exactly the trunk's builds for such a teacher, so the result is about T\* itself | T20, T22 | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as the setting of a counterexample | 2026-10-25 | DECIDED |
| R-least | ROOT | goal | "the least" -> a reading that exists (Theorem T23 proves the least need not exist) | this file, section 3 | Shonu: "I actually don't want to introduce error limits yet ... assume networks are trained well and that they are clean" (L4, with L3 acceptable) (2026-10-25) | 2026-10-25 | DECIDED |
| S-finite | SCOPE | input law | any law -> a finite mix. A reading of "an image mix" (the rule reads P), but it narrows the positive results: a law with a density is not covered | T19, T21 | moot: Shonu removed the image mix from the input (R-root, 2026-10-25) | 2026-10-25 | DROPPED |
| N-anynet | NEG | training | trained network -> any network of the kind. The counterexamples' teachers memorise chosen labels; they need not be the output of any training run on images. This is the option "every network" in each decision of section 3 | T20, T22, T23 | Shonu chose clean, well-trained networks (R-least, 2026-10-25); the results stay as the record of why | 2026-10-25 | DROPPED |
| N-ny3 | NEG | output | many classes -> three logits. Carry-over check: the builds are exactly the trunk's builds for that teacher | T23 | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as the setting of a counterexample | 2026-10-25 | DECIDED |
| R-score | ROOT | goal | KL on an input law -> a score defined from N and G alone. Packages: (i) intrinsic, (ii) on the training law with a property that restricts the input law, (iii) worst case over every law consistent with N. See card 3-1 | T\*\*, Part B.C | Shonu: "I think I imposed an unnecessary restriction ... the generalist mix and a developer-defined mix are ultimately just mixes in general" (a mix is an input; score = KL on the mix) (2026-10-25) | 2026-10-25 | DECIDED |
| R-goal | SCOPE | goal | a goal of any form -> a goal that reweights inputs by N's class probabilities (Definition D18), not a goal inside the pointwise loss. Under the worst-case score (iii) this form loses the goal (Corollary T24.1) | T\*\*, Part B.C | moot: Shonu made the mix an input (R-score, 2026-10-25); a class reweighting is one kind of mix | 2026-10-25 | DROPPED |
| R-cleannet | STRUCT | structure | any network -> a perfectly clean network (Definition in Part A; built from the essay's "built to be read, and built upon, by its downstream readers"), with the cleanliness distance eps(N) measuring how far a real network is | T\*\*, Part B.C | Shonu chose clean networks (R-least). 2026-10-26, Shonu: "not making a stronger version of clean, but just a good version of what a clean network is"; clean is a property of N alone, with no mix ("done on the things a network can do"); eps is a list, one entry per principle ("multiple metrics"); clean is "an object which satisfies a bunch of principles". D20 is one candidate principle; the definition is open (Part A.5). 2026-10-26, later, Shonu: clean must describe real networks (the essay's title is "neural networks are already clean"); clean from the network alone, not the training inputs ("you should be able to determine if a network is clean by interpreting the neurons and circuits themselves ... rather than needing extra data"); take the math from the essay. Proposal: Definitions D41 and D42 (votes, gate, built upon, judged on whole targets), card 3-6. 2026-10-26 evening, Shonu: "when we say clean, we really mean interpretable"; "the first principle is actually exploration". After the exploration: Definitions D43 to D45 (paper A.6): a unit is READ and has ONE JOB; card 3-7; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept D45 as the definition of clean | 2026-10-25 | DECIDED |
| R-mixclean | ROOT | goal | "exact closed form for every mix, for clean networks, with a polynomial-time zoom" -> one of: keep the trunk as written (no time bound on the zoom); a property of the mix beyond symmetry; fast for a bounded number of merged blocks or families; a stated error; a principle of clean that the detector networks fail. Forced by Theorem T43 | this file, card 3-5, card 3-33 | Shonu chose (e), keep the trunk as written, no time limit, on card 3-33: "I'm fine with the one-time computation up front being non-polynomial as long as it is exact and I'd also like it to not take a terrible amount of GPUs" (2026-11-01) | 2026-10-26 | DECIDED |
| R-jobmix | ROOT | goal | the mix is any finite list of inputs with weights -> the mix is built from a base law nu by weights on the network's jobs (Definition D48): the user chooses how much each unit's target and the silent part count, not single inputs | this file, card 3-8 | open. Shonu (2026-10-26 night): "if the large network fails on this test itself then we should likely think of some other way to incorporate mixes entirely"; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: yes: the mix built from the network's own jobs (D48), not alone sufficient | 2026-10-26 | DECIDED |
| R-listen | STRUCT | structure | the read part of clean -> one of: weak (lambda > 0), agreement (xi > 0), or a test that credits both teams and alternatives (Definition D49 candidates, Lemma T52; D49 is for dense layers only) | paper A.6, card 3-9 | open. Shonu (2026-10-26 night): "if this fails on clean networks then we need to think of a different version of testing listening"; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: yes to both forms of the read test and to a graded test | 2026-10-26 | DECIDED |
| R-group | ROOT | goal | the sizes (k_1, ..., k_L) per layer and any new units -> a zoom chosen by the developer (nested groupings G_0, ..., G_m; each group = a set of N's units in one layer with a target size k_g; units in no group are kept), base inputs P, and new units that read only their own group's units (option (b) of card 3-10). This is Trunk T4 | this file, section 1, card 3-10 | Shonu: "yes i do agree with the trunk you've proposed"; "a developer can kind of choose like, not just the grouping, but how the grouping degrades over time"; on (a) or (b): "whatever decision you choose, that'll be fine" (I chose (b); under (a) the grouping would not shape what a new unit computes) (2026-10-27) | 2026-10-27 | DECIDED |
| R-map | ROOT | goal | the rule reads N, P and the zoom -> the rule also reads a map M of N on P (typed claims, each with an eps computable from N and P in polynomial time; no claim names a build, size or zoom); groups are unions of the map's districts; exactness and bounds are in the list of claim eps values, and (2026-10-28, card 3-19 (a)) apply only to maps complete for the zoom. This is Trunk T5 | this file, section 1 and section 1a | Shonu: "assume that we're given a network which is already mapped"; "we can't just ... say, hey, this is mapped and not have a mathematical theory attached to that all the way to the fundamentals"; chose "Yes, draft T5" and "Graded eps"; approved the plan with T5's text; then "i would say yes, the exactness promise applies only to complete maps" (2026-10-28) | 2026-10-28 | DECIDED |
| R-groupopt | STRUCT | structure | the least build over any new units at width k (class (a) of Definition D50) -> a (b) build at the grouping into N's families, for some sizes k_F >= 0 summing to k (Hypothesis H-group). Trivially true when a family's rows span the input (T54 part 3); fails on T25; the content is in low-rank families (the real case) | card 3-10 | open. Shonu (2026-10-27): "if the network is clean, then it should theoretically be both, because if the neurons are circling the same concepts, then it shouldn't really matter. And so I think that would be a good thing to test"; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a named hypothesis to test | 2026-10-27 | DECIDED |
| R-recur | STRUCT | structure | the least build of each level of the zoom (class (b)) -> the build made step by step, each new unit reading only its children's new units (class (r) of Definition D50) (Hypothesis H-recur): for a clean N, some zoom tree makes (r) reach the least (b) value at every level | card 3-11 | open. Shonu (2026-10-27): "is this recursive idea the optimal way of doing things or not? It would also be pretty cool"; DROPPED as moot (triage 2026-11-02, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"): merged into R-tree (card 3-24) | 2026-10-27 | DROPPED |
| R-deep | ROOT | goal | "new units that read only that group's units" in hidden layers l >= 2, where the units below may be new -> one of: (A) through N's wiring with one decode per merged group below; (B) any weights on the B-units that replaced the members' inputs; (C) zooms that change one layer per level | card 3-12, Remark B.W.3 | open; DROPPED as moot (triage 2026-11-02, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"): merged into R-tree (card 3-24) | 2026-10-27 | DROPPED |
| R-tree | STRUCT | structure | groups in one layer, read through class (b) -> groups made of subtrees of a tree map (Definition DH1: districts in every layer, areas nested from layer to layer), each replaced by a box that reads the pre-activations of the group's top units and is carried to the logits by copy units that do not count toward k_g (Definition DH3); merges R-recur and R-deep into one question | COARSE-THEORY.md B.H, card 3-24 | me, from Shonu: "i see zoom recursion and depth recursion as the same thing ... we should treat this as one unified idea we want to prove is optimal" (2026-10-29). His words make it ONE node; the readings are open; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: (a): a box reads its group's top pre-activations; copy units do not count toward k_g | 2026-10-29 | DECIDED |
| R-branch | STRUCT | structure | removing a group may change kept units that read it -> maps that hold a branch claim (Definition DH2, error 0) for every subtree a group is made of: no unit outside the subtree reads it on P | COARSE-THEORY.md DH2, card 3-24 | me (forced by the proof of TH1: a kept unit kept off by inhibition from a removed unit turns on); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: yes: the branch claim is a claim type | 2026-10-29 | DECIDED |
| S-plainff | SCOPE | depth | a network N with L hidden layers -> a plain feed-forward network: each hidden layer dense and reading only the layer before, no positions, no skips. Part B.H's tree, branch claims and copy units are written for it; a ResNet needs the stream form (Part B.V) or a reading of skips | COARSE-THEORY.md B.H setting, card 3-24 | me (referee REFEREE-BH-2026-10-29.md finding 15); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as proof scope (B.H); S-skipff carries it to ResNets | 2026-10-29 | DECIDED |
| S-skipff | SCOPE | depth | a network N with L hidden layers -> a feed-forward network with skips and channel units (Definition DY3): ReLU channels with values at positions, each pre-activation affine in the channels of any earlier layers, logits linear in the mean-pooled last layer, position grids that tile each other in equal tiles. Wider than S-plainff (skips and positions allowed); covers a torchvision ResNet after its stem, with batch norms folded. Leaves out norms on the stream, attention, max pools inside the network | COARSE-THEORY.md DY3, B.Y.2, card 3-29 | me (2026-10-31): Shonu asked for ResNet ("I want to look at the ResNet stuff"); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: (a) yes: B.Y.2 covers ResNet skips and channel units | 2026-10-31 | DECIDED |
| R-linread | STRUCT | map | claim types of Part A.7 and B.H -> also the linear-read claim (Definition DP1): inside a node's area, after the free terms (a constant and the kept judgment units), N's centred logits are linear in the top pre-activations; error = the energy no linear function captures. A claim on N alone (no build, size or zoom). Used by TP1 to TP3, which need it for exactness and pay (1/2) x its error otherwise; Corollary TL3.1 proves it at error 0 from district, shelter, on-inside and judgment claims, and Theorem TL6 bounds it | COARSE-THEORY.md DP1, B.P, card 3-26 | me, from Shonu: "try to prove a very similar thing, but now passing up the summary" (2026-10-29); the claim type is my choice; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: (a) yes: the linear-read claim is a claim type | 2026-10-29 | DECIDED |
| R-oninside | STRUCT | map | claim types of Part A.7, B.H and B.P -> also the on-inside claim (Definition DL1): inside a node's area its units do not switch (pre-activation >= 0) and its kept judgment units are flat; error = the share of each unit's energy below 0, and each judgment unit's variation inside the area. A claim on N alone (no build, size or zoom). Used by Theorem TL2 (a lift through such units re-expresses every finer unit one layer down) and by Theorems TL3, TL6, Corollaries TL3.1, TL3.2, TL5 and Proposition TL4 (with K = the judgment sets, at every node below the group's top: a clean subtree is affine in its top) | COARSE-THEORY.md DL1, B.L, card 3-27 | me, from Shonu: understand the failures "based not just on the mathematics but on the network itself" (2026-10-29); the claim type is my choice; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: (a) yes: the on-inside claim is a claim type | 2026-10-29 | DECIDED |
| R-freefam | STRUCT | structure | any families -> families that are free on N's own mix: the blend of two members' firing inputs makes the gap no lower than their average (Hypothesis H-freefam). Continuous families (curves at all angles) are the candidate; discrete alternatives (the detector networks) fail it | card 3-13 | open (proposed by me, 2026-10-27, from Theorem T62 and Remark B.W.4); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a named hypothesis to test | 2026-10-27 | DECIDED |
| R-kinds | ROOT | goal | one kind of coarse unit for every group -> a kind per group, chosen by whether the group is AND, free or conflicting on the mix (options (a), (b), (c) of card 3-13) | card 3-13 | open; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: the card's recommendation (one kind per group, refined later) | 2026-10-27 | DECIDED |
| R-circuit | ROOT | goal | groups of units of one hidden layer replaced by new units of that layer (T4) -> circuits (units in any layers) replaced by coarse circuits that may place new units in any layer (Definition D60); options (a), (b), (c) of card 3-14 | card 3-14, Part B.X, card 3-33 | Shonu chose (a), groups, on card 3-33: "Let's do groups for now ... if there's a distinction between a group and a circuit, we might be analyzing the network a little wrong" (2026-11-01). Earlier (2026-10-27): "how can we mathematically define a circuit? And then, how can we unify all of these four concepts into a circuit methodology ... maybe like a coarse circuit" | 2026-10-27 | DECIDED |
| R-explaw | STRUCT | structure | the base law nu on inputs that defines a unit's target (Definition D43): (i) the data law of N's kind, or (ii) a law built from N alone (for example feature visualization with robustness transforms) | paper A.6 | Shonu (2026-10-26): decide after the exploration; the exploration log marks which moves need real images (O9 and O10, on which D45 rests, need a law on inputs; O5 found the same poles without data); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: (i) the data law, for now | 2026-10-26 | DECIDED |
| R-fold | ROOT | building blocks | any new units -> folds of N's own units? Theorem T25: the least build over any new units can read an input feature N ignores | T\*\*\*, Part B.Z | Shonu: "distillation is currently like probably the most general best way ... what is it trying to optimize towards? ... the idea is folding is going to be that thing ... It's not the most mathematically optimal way given X, Y, and Z constraints" (any new units kept; fold-optimality for clean networks is a conjecture about the optimum, not a restriction) (2026-10-25) | 2026-10-25 | DECIDED |
| R-fullrank | STRUCT | structure | any first layer -> first-layer weights that span the input space (then every unit is a loose fold of N's units, Lemma T31; T25's failure needs a rank-deficient first layer) | Conjectures D25-D27, Theorems T32-T34, Corollary T35 | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as a technical premise of the results that use it | 2026-10-25 | DECIDED |
| R-foldopt | ROOT | goal | "for clean networks (D20), distillation converges to folds of the network's families" -> one of: a stronger notion of clean; folds as a restriction with a loss; keep any units and characterise the optimum. Forced by Corollary T35 | this file, section 3-3 | Shonu: "They're really just my hypotheses ... I don't ever want us to like say, hey, we have to do this method just because" (option (c): the optimum itself is the object; folds, gates and other interpretability ideas are hypotheses about it, never required methods; option (a) became the search for the right definition of clean, node R-cleannet) (2026-10-26) | 2026-10-25 | DECIDED |
| R-dialfit | STRUCT | structure | any trained coarse unit -> a unit whose pre-activation, on its own region of the crush's training mix, is the least-squares affine fit (slope 1) of the family's output seen through the unit's own write (Corollary T38.2; first version: the plain summed output) (Hypothesis H-dialfit; Part C measures it) | Hypothesis H-dialfit | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a named hypothesis, measured in Part C | 2026-10-25 | DECIDED |
| R-gatefold | STRUCT | structure | a clean network -> a clean network with a gate in front of each family: a layer-1 unit outside whose region N's gap is constant (Hypothesis H-front); with a symmetry group, the group fixes the gate (tag phrase "Gamma fixes the gate") | H-front, Lemma T39, Conjecture D34 | me, from Shonu's "yes that is it" (2026-10-26) to the reading "the network's own gate decides where the coarse unit is on, and an easy formula decides what it computes there"; the node itself waits for his decision; Shonu: "keep gates as like a side thing and maybe come back to it later" (parked; B.G stays as a record) (2026-10-26) | 2026-10-26 | DROPPED |
| R-flatgate | CONV | nonlinearity | a ReLU gate -> a gate with one value on its whole region (Hypothesis H-flatgate) | H-flatgate, Lemma T39a, Corollary T39.1, Conjecture D34 | me; Shonu: "keep gates as like a side thing and maybe come back to it later" (parked; B.G stays as a record) (2026-10-26) | 2026-10-26 | DROPPED |
| S-keepfirst | SCOPE | scope | any depth and any size list -> two hidden layers, the build keeps N's layer 1 and replaces layer 2 | Setting of B.G, Lemma T39, Conjecture D34 | me; Shonu: "keep gates as like a side thing and maybe come back to it later" (parked; B.G stays as a record) (2026-10-26) | 2026-10-26 | DROPPED |
| R-indep | ROOT | goal | each zoom level optimised as a whole -> the developer may coarsen different sets independently (Demand U4 of UNIT-BRANCH.md). Rewritten 2026-10-27 (night): not a yes/no choice; with output-faithful boxes (R-outfaith) independence is graded (Theorems TU13, TU14), and the open question is how large the interaction is for clean networks | card 3-15, UNIT-BRANCH.md | open; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a graded measurement (card 3-15: nothing to choose) | 2026-10-27 | DECIDED |
| R-outfaith | STRUCT | units | any new units of the group's inputs (class (b)) -> boxes whose output depends on the image only through what the group's members sent to their readers, optionally with weak outputs read as silence (Definition DU6 of UNIT-BRANCH.md), and, in its stronger form, that send nothing where the members sent nothing (Definition DU7; source: the essay's "become another reader" and its weak-activation ablation) | card 3-15, UNIT-BRANCH.md B.5 | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as a premise of the results that use it (a reading of the essay) | 2026-10-27 | DECIDED |
| R-readbias | STRUCT | units | readers of a replaced set keep N's biases (Definition DU2) -> they may re-fit their biases (Definition DU8 of UNIT-BRANCH.md); a reading of Trunk T4's 'a network of the same kind ... among all such networks', which does not say that readers' biases are N's | card 3-16, UNIT-BRANCH.md TU19, TU20, Remark B.6 | Shonu: "yea they can move their biases" (2026-10-27) | 2026-10-27 | DECIDED |
| R-readweights | STRUCT | units | readers re-fit only their biases (DU8) -> they also re-fit their weights on kept units (Definition DU9 of UNIT-BRANCH.md); a reading of Trunk T4 like R-readbias | card 3-16, UNIT-BRANCH.md TU22 | Shonu: "yes" (2026-10-27), to "may readers also re-fit their weights on the neurons we keep?" | 2026-10-27 | DECIDED |
| R-edgetether | STRUCT | units | a box may give its units any write to the out-boundary (Definition DU2, Trunk T4's class (b)) -> a box writes only inside the span of its members' out-weight columns (Definition DA2 of UNIT-BRANCH.md; source: the essay's "a representation matters only through what reads it" and "become another reader") | card 3-17, UNIT-BRANCH.md TA6, TA8, CU7 | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept, unused: no theorem needs it yet | 2026-10-28 | DECIDED |
| R-district | STRUCT | structure | any network -> a network whose last-layer units split into districts, each with a gate that is on wherever a member is on, and no two districts on together on the mix (Hypothesis H-district of UNIT-BRANCH.md) | UNIT-BRANCH.md A.3, Corollary TU8 | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: yes: districts are map objects | 2026-10-27 | DECIDED |
| S-linunits | SCOPE | units | ReLU units -> linear units with one hidden layer and a constant metric. A partial result only, not proposed as a restriction | UNIT-BRANCH.md Theorem TU4 | me (partial result); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as proof scope | 2026-10-27 | DECIDED |
| R-weak | STRUCT | structure | any trained network -> a network whose own readers send little through weak activations of its last hidden layer (Hypothesis H-weak; source: the essay's weak-activation ablation). Measured in Part C; no theorem uses it | COARSE-THEORY.md A.7, Part C | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a named hypothesis, measured in Part C | 2026-10-28 | DECIDED |
| R-scale | STRUCT | structure | one network -> a family of sizes in which a fixed map's errors fall with size (Hypothesis H-scale, three forms: quiet, fair and map, Definition DN2; source: Shonu's reading of the essay, "larger networks are actually way cleaner"). Measured in Part C (quiet and fair forms); Corollary TN4 uses the map form | COARSE-THEORY.md A.7, Part C | me; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a named hypothesis, measured in Part C | 2026-10-28 | DECIDED |
| R-mass | STRUCT | structure | a family of sizes -> one in which the last hidden layer's output spread O(N) does not grow and the clean assemblies' kept and box masses stay within a fixed multiple of O(N) (Hypothesis H-mass, Definition DN2). Corollary TN4 uses it; part (1) is measured in Part C | COARSE-THEORY.md A.7, B.N, Part C, card 3-21 | me, from Shonu: "since the larger networks are hopefully more well-trained, if we choose the right larger networks, then we're going to see some really nice results" (2026-10-28); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: kept as a named hypothesis, measured in Part C | 2026-10-28 | DECIDED |
| R-wellposed | ROOT | goal | the least D_P among all such networks (any weights) -> the least worst-case D_P over the networks that the map describes within its errors (the robust optimum, Definition DE2 of COARSE-THEORY.md); with every error 0 the two are the same, so the exact clause does not change. Direction forced by Proposition TE1: with free reader weights the least loss sees only where a unit is on, not how much, so no error that measures size can bound it (TM4) | card 3-20, COARSE-THEORY.md B.E | me, from Shonu: "this seems to me like a mathematical error in actually finding the optimal subnetworks given a larger network, and we should treat this as the error it likely is" (2026-10-28); removed by Shonu on card 3-33: "we should just be aiming for whatever the optimization or the optimal distillation function is going for, which I believe is least KL ... not something else" (2026-11-01). Its results stay as mathematics, off the trunk | 2026-10-28 | DROPPED |
| R-unitmap | STRUCT | structure | maps whose units are N's channels and whose regions are sets of label classes -> (a) keep that, (b) map InceptionV1 first, or (c) a graded claim type for a strong core plus a weak spread. Forced by data, not by a theorem: ResNet-18 has no channel-level map of the A.7 types (Part C) | COARSE-THEORY.md A.7, Part C (H-map, ResNet-18), card 3-22 | me, from Shonu: "create a good map. I think our maps kind of suck right now"; DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: (a) keep the ResNet-18 result; a real map of the whole network is ROADMAP session 3 | 2026-10-28 | DECIDED |
| R-jointgroup | STRUCT | structure | in a pure-stream network a group is a union of districts (units) or of areas (stream subspaces) -> also a joint group: units of several blocks together with stream directions, replaced by k_g units and k'_g stream directions. Needed only for a stream narrower than the write budget dim E_N + sum k_S of Theorem TV19.2; at or above it the stream is free | COARSE-THEORY.md B.V.7, card 3-29 | me, from Shonu: "theyre thinking about coarsening a capability or coarsening knowledge" (2026-10-31); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: (a) yes: a group may hold units and stream directions | 2026-10-31 | DECIDED |
| S-keepdrop | SCOPE | zoom | a zoom chosen by the developer, any target size k_g for each group -> each changed group of a level is dropped (k_S = 0) or kept with at least as many units as the rank of its D2 table Sigma_(W_S) (Definition DP2). Then every level is least for KL in closed form (Corollary TI2.1). A level cannot blend capabilities into fewer units than they need | COARSE-THEORY.md B.I, card 3-31 | me (2026-10-31), from Shonu: "specifically to find the optimal KL" (card 3-30); this narrowing is my proposal. DROPPED 2026-10-31, Shonu: "you can only keep them or drop them. That's the same as pruning"; TI2.1 stays as a record | 2026-10-31 | DROPPED |
| R-active | STRUCT | map | maps whose claims are of the types of A.7, DH1, DL1, DP1 -> also a class set K_w named for each node of a squeezed group, and the active-class summary eps_act(A_w, K_w) of Definition DQ1 (a summary of N like DI1, not a claim: REFEREE-BQ finding 3). Used by TQ1 parts 2 to 4 (a bound in summaries, label ONE-BUILD) and Hypothesis H-active | COARSE-THEORY.md B.Q, card 3-32 | me (2026-10-31), from Shonu: "fix this third row", and the test model/ts_sep.py; DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose accept: the K_w are the analyst's choice (TQ3 note) and eps_act is a summary of N, not a claim, so the node narrows no network | 2026-11-01 | DECIDED |
| S-squeezecomplete | SCOPE | zoom | complete maps (DM6) -> for a squeezed group (0 < k_S < rank), complete also means: every node of the group has own-class rank (Definition DQ1) <= k_S, and the nodes carry pairwise disjoint active-class claims. A squeeze below an area's own rank stays the open case (DI1) | COARSE-THEORY.md B.Q, card 3-32 | me (2026-10-31); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose accept for now: the case it leaves out (an area squeezed below its own rank) is a different object, the one-area squeeze, still worked on (DI1, DQ3); used by TQ1 parts 2 to 4, TQ5, DQ4 | 2026-11-01 | DECIDED |
| R-constclaim | STRUCT | map | maps whose claims are of the types named so far -> also constancy claims of Definition DQ10 (a partition of P; error: the information N holds inside the regions) and KL-quiet claims of Definition DQ12. It adds a claim type, so it narrows no network; it names no build, size or zoom | COARSE-THEORY.md B.Q.9, Theorem TQ14 | me (2026-11-01); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out"; I chose accept: a new claim type only widens what a map can say | 2026-11-01 | DECIDED |
| R-poleclaim | STRUCT | map | maps whose claims are of the types named so far, and complete maps (DM6) that squeeze only at two logits -> also pole claims of Definition DQ13 (a partition, one class per region, a pairing, a split of the pairs among the groups and the kept units, real numbers; error from N's outputs, pre-activations and values), and DM6 item 5 reads: for k_S < c(S), a blend claim at two logits, or a pole claim whose sets are the groups. It adds a claim type and a second way to be complete, and that second way drops DM6 items 2, 3 and 4 (shelter, judgment and flat claims), since Corollary TQ17.1 uses none of them. Its real numbers depart from DM1 | COARSE-THEORY.md B.Q.9, Theorem TQ17, Corollary TQ17.1 | me (2026-11-01); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out"; I chose accept: the old item 5 made T5 promise nothing at C logits. Its narrow scope is node S-polescope, which is yours | 2026-11-01 | DECIDED |
| R-certclaim | STRUCT | map | maps whose claims are of the types named so far -> also certificate claims naming a triple of some rank | COARSE-THEORY.md B.Q.9, Definition DQ14 | me (2026-11-01); removed the same day after REFEREE-BQ10: a triple is a build, and one claim per rank is one claim per target size, against the trunk's words. The test of Theorem TQ18 stays inside the one-time computation | 2026-11-01 | DROPPED |
| S-polescope | SCOPE | zoom | every squeezed size of any zoom on a map made complete for it (ROADMAP.md) -> for now, at C logits, only levels whose total written rank is C/2 (kept rank plus the sum of k_g), with many groups and kept units allowed (DM6 item 5, Corollary TQ17.1). This is the narrowing left after the general attempt: no other exact class in the map's terms is proved | card 3-35 | me (2026-11-01), for Shonu's decision on card 3-35 | 2026-11-01 | PENDING |
| R-readcurv | STRUCT | map | maps whose claims are of the types named so far -> also read curvature claims of Definition DQ23 (a partition; error: the spectral distance of the read's Fisher information from a sum of Kronecker products, one per region). It adds a claim type, so it narrows no network; it names no build, size or zoom | COARSE-THEORY.md B.Q.12 | me (2026-11-02); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; a new claim type only widens what a map can say | 2026-11-02 | DECIDED |
| R-weightgap | STRUCT | map | maps whose claims are of the types named so far -> also weight gap claims of Definition DQ22 (a partition and one common set of near classes; error: sigma_2 of the doubly scaled table outside the near classes, plus eps_const). It adds a claim type, so it narrows no network; it names no build, size or zoom | COARSE-THEORY.md B.Q.12 | me (2026-11-02); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; a new claim type only widens what a map can say | 2026-11-02 | DECIDED |
| R-prodtail | STRUCT | map | maps whose claims are of the types named so far -> also product tail claims of Definition DQ16 (a partition and near classes per region; error: the weighted log-scale distance from the product shape outside the near classes, plus eps_const). It adds a claim type, so it narrows no network; it names no build, size or zoom | COARSE-THEORY.md B.Q.11 | me (2026-11-02); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; a new claim type only widens what a map can say | 2026-11-02 | DECIDED |
| R-treeweight | STRUCT | map | maps whose claims are of the types named so far -> also tree weight claims of Definition DQ19 (a partition with one class per region and a tree on the classes; error: weighted log-scale distance from that shape, plus eps_const; the monotone premise and the absence of scales in Conjecture DQ17 are premises of that conjecture only). It adds a claim type, so it narrows no network; it names no build, size or zoom | COARSE-THEORY.md B.Q.11 | me (2026-11-02); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; a new claim type only widens what a map can say | 2026-11-02 | DECIDED |
| R-rankclaim | STRUCT | map | maps whose claims are of the types of A.7, DH1, DL1, DP1 -> also rank claims (W, r) of Definition DQ7 (a set W of map nodes and r; error: movement off the rank-r shared span in N's own Fisher metric); no completeness use (REFEREE-BQ5 E4); used by Corollary TQ10.1 (ONE-BUILD) | COARSE-THEORY.md B.Q (DQ7, TQ10) | me (2026-11-02); DECIDED by delegation, Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; a claim type with an error computed from N alone, 0 exactly at rank <= r; it narrows no network | 2026-11-02 | DECIDED |
| R-propcomplete | SCOPE | map | maps -> maps that hold a merge partition (proportional claims, DM10) of every group of every level, which TY2 and TY3 need; and complete maps as in DM6 -> DM6 plus "or every group of every level has a merge partition of proportional claims (DM10)"; widens the exactness promise to any number of logits through Theorem TY2 | COARSE-THEORY.md DM6, DM10, B.Y, card 3-22 | me (partial result; referee REFEREE-BY-2026-10-28.md finding 3); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: yes: maps with proportional claims count as complete | 2026-10-28 | DECIDED |
| S-purestream | SCOPE | network | any network N -> pure-stream networks (Definition DV1: one position, MLP blocks; each block reads the stream only through a norm (or, in the no-norm form, directly) and linear maps and writes by a linear map; Fact F12 kind 3; Lipschitz sigma_l for TV6 to TV8, TV9 part 3, the bound of TV13, TV6.2, TV15, TV15.3, DM16, TV18, TV18.1, TV18.2; eps_l > 0 in the norm form for DM14, TV6.1 and TV6.2, while TV6 and TV8 ask eps'_l > 0 of the build). Leaves out ResNet (a ReLU on the stream), ConvNeXt (a per-channel filter on the stream), attention and positions | COARSE-THEORY.md B.V, card 3-23 | Shonu: "Yes, study pure streams right now" (2026-10-29, card 3-23, option (a)) | 2026-10-29 | DECIDED |
| R-streamregion | ROOT | map | districts are disjoint sets of N's units inside one hidden layer (T5) -> a district may also be an area of the stream: a subspace that runs through every block; a group may be an area, replaced by k_g directions inside it (Definition DV2). T5's wording amended 2026-10-29 | COARSE-THEORY.md DM11, B.V, card 3-23 | Shonu: "I'm going to be fine with some like subspace crossing because the ultimate thing is: are we able to kind of deal with these sorts of concepts on their own" (2026-10-29, card 3-23); the amended trunk sentence approved with the plan you-did-some-great-radiant-matsumoto.md | 2026-10-29 | DECIDED |
| S-readoutarea | SCOPE | depth | areas anywhere in the stream -> areas that no block reads (only the logits read them). Moot: this premise is the block-read claim at error 0, the map's exactness condition (alias moved to R-map), and TV2 is the zero-leak case of TV6 | COARSE-THEORY.md TV2, TV3, TV4, card 3-23 | moot, by the math (B.V.3); Shonu on this node: "they don't really seem like decisions, they kind of seem like things that you could work out with the map itself" | 2026-10-29 | DROPPED |
| R-normconst | STRUCT | units | a build keeps N's norm constants eps_l and writes only projections of N's writes (DV2) -> a build may re-set each norm's constant to eps_l + c_l, c_l >= 0; by Lemma TV12 the same as N's constants and one more stream direction written only with constants and never read (one constant direction, not one more direction of N's stream; V_area(G) <= V_N(G)). Widened 2026-10-29 (B.V.6): c_l > -eps_l (any positive constant), so that the oblique cut of Corollary TV2.1 and Proposition TV17 part 1 can match a negative mean energy change (Proposition TV18.2 no longer needs it); the class grows, so every upper bound on V_area survives; Lemma TV12 describes the case c_l >= 0 | COARSE-THEORY.md DV2, TV12, card 3-23 | Shonu, delegating the card 3-23 technical nodes: "I'd encourage you to explore the math for what's good for our ultimate goal ... I don't really need to decide anything for this" (2026-10-29); I chose (a), with Lemma TV12 as the reason (referee REFEREE-BVO-2026-10-29.md finding 10: it widens DV2, so it is a decision, not moot); the widening chosen by me under Shonu's delegation of development decisions ("That's completely fine for you to make those decisions on your own", 2026-10-29) | 2026-10-29 | DECIDED |
| N-amplify | NEG | depth | any network -> a normed pure-stream network with one block of one neuron, sigma the identity, two logits and d = 3, whose logits read with weight about 1 / delta a value of size delta that the block writes. Carry-over check: the builds of TV7 are all re-fit area cuts of Definition DV2 for this network, so the result is about the class of B.V itself | COARSE-THEORY.md TV7 | me (referees: REFEREE-BV2 found the first version a rescaling; REFEREE-BV3 found the two-block version's map incomplete); DECIDED by delegation (triage 2026-11-02), Shonu (2026-11-01): "some of the decisions you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to make"; I chose: accept as the setting of a counterexample | 2026-10-29 | DECIDED |
| R-cutclaim | ROOT | map | T5: no claim refers to a build or a target size -> a claim may name a subspace D that a cut drops (DM14). Moot: in a map, D is a sum of members of the map's decomposition, so a carried leak claim names map objects only, as a district claim does (DM14 rewritten); TV3 becomes a way to find areas, outside the trunk | COARSE-THEORY.md DM14, TV4, TV6.1, card 3-23 | moot, by the rewrite of DM14; Shonu on this node: "they don't really seem like decisions, they kind of seem like things that you could work out with the map itself" | 2026-10-29 | DROPPED |
| S-laminar | SCOPE | map | areas at any angle whose sum is direct (Shonu: "fine with some like subspace crossing") -> a laminar family: any two areas nested or orthogonal (Definition DM11). Widened: DM11, DV2 and DM15 allow angles (B.V.5). The restriction stays only as a premise of Theorem TV2, Corollary TV3.1 and Proposition TV4 (one energy claim per dropped area) and of the norm form of Theorem TV18 and Corollary TV18.1 parts 1-4 (orthogonal members; Proposition TV18.2 and TV18.1 part 5 cover angles); Proposition TV17 part 2 shows these do not add at an angle with norms. Theorem TV6 parts 1 and 2 and everything built on them hold at an angle (Lemma TV16 part 5) | COARSE-THEORY.md DM11, F14, TV10, TV11, TV16, TV17, B.V.5, card 3-25 | Shonu, delegating: "S-Laminar is certainly a development decision, so you don't need to ask me about how we kind of do that. That's completely fine for you to make those decisions on your own" (2026-10-29); I chose (a), widen, with Lemma TV16 and Proposition TV17 of B.V.5 as the reason | 2026-10-29 | DECIDED |
| R-cutclass | STRUCT | units | a cut drops D = the sum over S of the orthogonal complement of V_S in S and keeps D perp by the projection B^T Pi (DV2) -> D may be any complement of V_S in S, and the cut may keep the rest by any matrix C with kernel D (the orthogonal cut in other coordinates, Lemma TV16 part 1; for example the oblique cut). Without norms the second widening changes no class of functions (TV16 part 2); with norms it does (one scalar per norm, TV16 part 3). No earlier result breaks: upper bounds on V_area survive a larger class, and the lower bounds of TV7 and TV9.2 go through Pi e, of which C e is a function | COARSE-THEORY.md DV2, TV16, TV17 | Shonu, delegating development decisions: "That's completely fine for you to make those decisions on your own" (2026-10-29); I chose to widen (referee REFEREE-BV45-2026-10-29.md findings 4 and 6a: a widening of the build class is a decision, as for R-normconst; option (a) of finding 4 removes a hidden narrowing at an angle) | 2026-10-29 | DECIDED |
| S-smoothact | SCOPE | network | Lipschitz sigma_l (S-purestream) -> sigma_l with a Lipschitz derivative (tanh, GELU, SiLU; not ReLU). Used by Proposition TV14 and Conjecture DV3; Theorem TV6 and Lemma TV13 cover ReLU. Moot: Theorem TV15 holds for every Lipschitz activation (the remainder modulus of Definition DV4 in place of Lip'); only Proposition TV14, now a special case, keeps the premise. A case premise inside one part, when the theorem or other parts cover the other cases (TV15.3 parts 2 and 3, TV16 part 2), is not a restriction; Lemma TV10 has only the no-norm case, so it keeps S-nonorm | COARSE-THEORY.md TV14, TV15, D.15, card 3-25 | moot, by the math (B.V.4); Shonu on this node: "I just want a nice general method over these residual streams, and I think that you know we don't need to place those sorts of restrictions on anything" (2026-10-29) | 2026-10-29 | DROPPED |
| S-nonorm | SCOPE | network | both forms of Definition DV1 -> the no-norm form (no norm before any read). Used by Lemmas TV10, TV13 and Proposition TV14; the target networks (transformers, ViT) have norms, so this is TV14's main limit; Conjecture DV3 is the norm form. Moot: Lemma TV13.1 and Theorem TV15 hold in both forms (DV3 proved); Lemma TV10 keeps the tag because the no-norm form is its subject (with norms Fact F14 part 2 says what changes), and TV13, TV14 are now special cases. A case premise inside one part, when the theorem or other parts cover the other cases (TV15.3 parts 2 and 3, TV16 part 2), is not a restriction; Lemma TV10 has only the no-norm case, so it keeps S-nonorm | COARSE-THEORY.md TV10, TV13, TV13.1, TV14, TV15, card 3-25 | moot, by the math (B.V.4); Shonu on this node: "I'd like you to prove that norm's conjecture, otherwise, you know, we're not going to be able to apply this in the future" (2026-10-29) | 2026-10-29 | DROPPED |
<!-- /nodes -->

## 5. Where each phrase of a tag goes

Every phrase inside an `(under: ...)` tag must be one of these, or a joining word. The linter removes each matched
phrase and fails on any text that is left. This is how a hidden restriction becomes visible: it has no entry here,
so the lint fails until it gets a node.

<!-- aliases -->
| Phrase in a tag | Node |
|---|---|
| H-selfgate with step gates phi_l(t) = 1[t > 0] | R-selfgate, R-step |
| H-selfgate with phi(t) = alpha t | R-selfgate, R-affine |
| H-sgate with a gate path | R-sgate, R-gate |
| H-gauss with m_a = 0 and Sigma positive definite | R-gauss, R-centred, TECH |
| H-gate | R-gate |
| H-selfgate | R-selfgate |
| H-sgate | R-sgate |
| H-gauss | R-gauss |
| H-dialfit | R-dialfit |
| H-group | R-groupopt |
| H-recur | R-recur |
| H-freefam | R-freefam |
| H-clean | R-clean |
| H-read | R-read |
| H-sym | R-sym |
| H-symN | R-symN |
| H-kron | R-kron |
| H-pool | R-pool |
| H-joint | R-joint |
| H-gram | R-gram |
| H-eq | R-eq |
| H-contour | R-contour |
| H-detector | R-detector |
| H-shot | R-shot |
| H-stat | R-stat |
| depth m = 1 | R-depth1 |
| depth 1 | R-depth1 |
| one read column | R-onecol |
| phi(t) = alpha t + beta with alpha != 0 | R-affine, TECH |
| phi(t) = alpha t + beta | R-affine |
| phi absolutely continuous with \|phi'\| <= L almost everywhere | R-lip |
| phi(t) = 1[t > 0] and sigma > 0 | R-step, TECH |
| a constant metric G = M (x) K | R-constmetric |
| with circulant C and Phi as a stated premise | R-circ |
| every edge w_l has equal column sums, and H = alpha' 1^T | R-sumread |
| H-eq with R_y = I implies the last two | TECH |
| the conditions of H-eq for a self-gated chain: N_l = N, the map gamma, and parts 1, 3 and 4 | R-eq |
| every read matrix S_l a single column s_l 1 (an invariant read) | R-invread |
| rho invariant | R-eq |
| a sampled-field form of H-clean and H-read: C_ij = c(theta_j - theta_i) and Phi_ij = phi(theta_j - theta_i) for continuous even positive-definite functions c, phi | R-field, R-read |
| the chains built satisfy H-sgate | R-sgate |
| Chat positive definite | TECH |
| depth m >= 2 | TECH |
| k = 1 | TECH |
| 1 <= k < d | TECH |
| a linear-seen kernel K' | R-linseen |
| finite second moments of y and of u_R for every R, as defined in part 1 | TECH |
| finite second moments of y | TECH |
| the premises of Theorem T16 part 5 | TECH |
| any law | TECH |
| hardness relative to Hillar and Lim 2013: the theorem that deciding the spectral norm is NP-hard and the theorem that best rank-1 approximation is NP-hard, see the literature table for their numbers | TECH |
| two logits | S-twoclass |
| one hidden layer | S-onehidden |
| a finite mix | TECH |
| a perfectly clean network for Gamma | R-cleannet |
| a full-rank first layer | R-fullrank |
| a Gamma-invariant mix | R-symN |
| H-front | R-gatefold |
| H-flatgate | R-flatgate |
| first layer kept | S-keepfirst |
| Gamma fixes the gate | R-gatefold |
| H-typical | R-cleannet |
| H-route | R-cleannet, R-explaw |
| H-agree | R-cleannet, R-explaw |
| groups in the last hidden layer | S-lastlayer |
| tree weight claims | R-treeweight |
| class (b) builds | R-group |
| each region meets at most one group and its judgment units are kept | S-regiongroup |
| every claim it uses has eps = 0 | R-map |
| H-map | R-mapreal |
| the map covers the last hidden layer | S-cover |
| channel units with position-shared boxes | R-unitmap |
| M holds a merge partition of every group of every level | R-propcomplete |
| the robust optimum | R-wellposed |
| H-weak | R-weak |
| H-scale | R-scale |
| H-mass | R-mass |
| pure-stream networks | S-purestream |
| area cuts | R-streamregion |
| areas read only by the logits | R-map |
| a gain counterexample | N-amplify, N-ny2 |
| carried leak claims | R-streamregion |
| member claims | R-streamregion |
| sigma with a Lipschitz derivative | S-smoothact |
| the no-norm form | S-nonorm |
| a tree map | R-tree |
| subtree groups | R-tree |
| branch claims | R-branch |
| judgment claims in hidden layers | R-branch |
| plain feed-forward networks | S-plainff |
| networks with skips and channel units | S-skipff |
| linear-read claims | R-linread |
| on-inside claims | R-oninside |
| keep-or-drop zooms | S-keepdrop |
| every claim other than the linear-read claims has eps = 0 | R-map |
| the objective D2_P | R-obj |
| active-class summaries | R-active |
| H-active | R-active |
| squeeze-complete groups | S-squeezecomplete |
| pole claims | R-poleclaim |
| the Unique Games Conjecture | TECH |
| part 3 under the Unique Games Conjecture | TECH |
| part 3's last line under the Unique Games Conjecture | TECH |
<!-- /aliases -->

## 6. The path of every result

Path = every node the result sits under, including nodes that its tag does not name (for example the class of its
Part). Distance = the number of CONV, STRUCT and SCOPE nodes on the path. `tools/tree_report.py` checks that every
node named by a tag is on the path, and prints the table with distances.

<!-- paths -->
| Item | Path | Note |
|---|---|---|
| Lemma B0.1 | R-family, R-bench, R-pwl | proves the local form for KL itself, so R-local is not a premise here |
| Theorem T0 | R-family, R-bench, R-local, R-fixedread, R-constmetric, R-diagmetric | parts 3 and 4 use the two metric nodes |
| Corollary T0.1 | R-family, R-bench, R-local, R-fixedread, R-stat |  |
| Theorem T1a | R-family, R-bench, R-local, R-kron, R-pool, R-circ | the circulant premise is a symmetry premise (Lemma T1b derives it from H-clean, H-sym, H-read) |
| Lemma T1b | R-family, R-bench, R-local, R-clean, R-sym, R-read, R-kron, R-plane | plane model D8 is implicit |
| Remark B1.1 | R-family, R-bench, R-local, R-clean, R-sym, R-read, R-plane, R-kron, R-pool, R-circ | it works under T1a |
| Lemma T1c | R-family, R-bench, R-local, R-clean, R-sym, R-read, R-plane, R-fixedread, R-diagmetric |  |
| Remark B1.4 | R-family, R-bench, R-local, R-pool, R-constmetric, R-clean, R-sym, R-read, R-plane | biases bench |
| Proposition T1' | (none) | the statement only: ReLU units that read one common field. Its use for T* goes through the HOW step of D6 |
| Lemma T3.0 | S-datamath | plane geometry |
| Theorem T3 | S-datamath, S-linear, R-gaussnu | about contour laws, not networks; part 2 uses a Gaussian nu |
| Corollary T3.1 | S-datamath, S-linear, R-gaussnu, R-detector |  |
| Corollary T3.2 | S-datamath, S-linear, R-detector |  |
| Conjecture T2 | R-family, R-bench, R-local, R-clean, R-sym, R-read, R-contour, R-plane, R-kron, R-pool, R-detector, R-shot | route items use T1a part 3 and the detector model |
| Theorem T5 | R-family, R-bench, R-local, R-clean, R-sym, R-read, R-kron, R-pool, R-joint, R-plane |  |
| Conjecture T4 | R-family, R-bench, R-local, R-clean, R-sym, R-read, R-contour, R-plane, R-fixedread, R-diagmetric, R-kron, R-pool, R-detector, R-shot |  |
| Lemma T8 | R-gate, R-pointwise | |
| Theorem T9 | R-obj, R-gate, R-pointwise | |
| Proposition T7 | R-obj, R-gate | the identity uses no edges; its second sentence uses D2 |
| Theorem T6 | R-obj, R-gate, R-pointwise | |
| Theorem T10 | R-obj, R-gate, R-pointwise, R-eq, R-symN | |
| Corollary T10.1 | R-gate, R-pointwise, R-sumread | exact on every input, so no objective is used; H-eq is one way to get the premises, not a premise |
| Corollary T10.2 | R-gate, R-pointwise | a bound from R-obj back to KL |
| Corollary T10.3 | R-gate, R-pointwise, R-range | a bound from R-obj back to KL, for range-bounded builds |
| Remark B4.1 | R-obj, R-gate, R-pointwise, R-eq, R-symN | |
| Theorem T12 | R-obj | the floor, for every network with a linear head |
| Lemma T11a | R-selfgate, R-pointwise | |
| Theorem T11 | R-obj, R-selfgate, R-pointwise, R-faithful |  |
| Corollary T11.1 | R-obj, R-selfgate, R-pointwise, R-extra | |
| Corollary T11.2 | R-obj, R-selfgate, R-pointwise, R-eq, R-symN, R-invread, R-faithful, R-extra | part 3 is read-faithful; part 4 uses T11.1 |
| Theorem T13 | R-obj, R-sgate, R-extra, R-pointwise, R-gate | visible widths can exceed k; R-pointwise only in parts 4 and 5, R-gate only in the gate-path clause of part 5 |
| Theorem T14 | R-obj, R-sgate, R-gate, N-linear, N-onereadpos, N-finite | negative result for one decision problem (depth 2, width (1, 1)); see section 2 |
| Theorem T15 | R-obj, R-selfgate, R-pointwise, R-step, R-depth1, R-onecol, N-finite, N-onepos, N-ny2 | negative result (counterexample) |
| Theorem T16 | R-obj, R-selfgate, R-pointwise | |
| Corollary T16.1 | R-obj, R-selfgate, R-pointwise | |
| Lemma T17a | R-obj, R-selfgate, R-pointwise | tag "none": the class of D12 is implicit |
| Theorem T17 | R-obj, R-selfgate, R-pointwise, R-gauss, R-depth1, R-onecol, R-affine, R-linseen |  |
| Proposition T17.1 | R-obj, R-selfgate, R-pointwise, R-gauss, R-depth1, R-onecol, R-lip, R-step, R-linseen |  |
| Theorem T17.2 | R-obj, R-selfgate, R-pointwise, R-gauss, R-centred, R-depth1, R-onecol, R-affine | |
| Proposition T17.3 | R-obj, R-selfgate, R-pointwise, R-gauss, R-centred, R-depth1, R-onecol, R-affine | |
| Proposition T18 | R-obj, R-selfgate, R-pointwise | |
| Conjecture D1 | R-family, R-bench, R-local, R-field, R-read, R-sym, R-kron, R-pool, R-plane | R-sym is named by the tag but is redundant (the kernel form is assumed directly) |
| Conjecture D2 | R-family, R-bench, R-local, R-clean, R-sym, R-read, R-contour, R-detector, R-plane, R-fixedread, R-diagmetric, R-shot, R-readergate |  |
| Conjecture D3b | R-obj, R-sgate, R-gate, R-eq, R-symN | |
| Conjecture D7 | R-obj, R-selfgate, R-pointwise, R-step, N-finite | negative (hardness) |
| Conjecture D9 | R-obj, R-selfgate, R-pointwise, R-depth1, R-onecol, R-affine, N-finite, N-onepos, N-ny2 | negative (hardness) |
| Conjecture D8 | R-obj, R-selfgate, R-pointwise, R-gauss, R-centred, R-depth1, R-onecol, R-affine | the last open item (rev 7) |
| Theorem T19 | S-twoclass, S-onehidden | positive result; three logits fail (T23); deeper builds: Conjecture D17 |
| Theorem T23 | N-ny3, N-onehidden, N-anynet | a counterexample at the trunk: for this N and P "the least" does not exist |
| Theorem T20 | N-ny2, N-onehidden, N-anynet | a counterexample at the trunk: T\* as worded ("least at every k" and "nest") has no solution for this N and P |
| Lemma T22a | (none) | |
| Theorem T21 | S-twoclass, S-onehidden | positive result (finitely many convex pieces; an exact rule for D2, slow in d k) |
| Theorem T22 | N-ny2, N-onehidden, N-anynet | hardness at the trunk: the same builds as T\* for these N |
| Conjecture D17 | S-twoclass | depth >= 2 |
| Lemma T24a | (none) | Gibbs's inequality: a network is optimal on its own labels, for every input law |
| Theorem T24 | N-ny2, N-onehidden, R-cleannet, R-obj | a counterexample at T\*\*: the training-law score is not fixed by the network, even for a perfectly clean network (D20). R-obj only for the D2 halves (part 2 and half of part 1). Width k = 1; d = 1 lifts to every d (points on a line) |
| Corollary T24.1 | (none) | the worst case over consistent laws is the worst single input |
| Theorem T25 | R-cleannet, R-fold, R-obj, N-ny2, N-onehidden | a counterexample at T\*\*\*: for a perfectly clean network and a symmetric mix, the least build over any new units is not a fold (it reads an input N ignores). d = 2, width 1. R-fold: part 3 is about fold builds (Definition D22); R-obj: the D2 half |
| Theorem T26 | R-fold, R-obj, N-ny2, N-onehidden | part 1: no node (any build class not depending on N); part 2: folds (R-fold), D2 (R-obj), a two-logit one-hidden-layer example |
| Theorem T27 | S-twoclass, S-onehidden, R-fold | the fold version of T21, exponent min(s, d) k |
| Theorem T28 | R-fold, S-twoclass, S-onehidden | part 2 uses two logits; part 3 fold builds |
| Theorem T29 | R-fold, S-onehidden | a characterization of D22's folds (Farkas), for single ReLU units on all of R^d with Z_S not empty |
| Corollary T30 | R-fold, R-obj, S-twoclass, S-onehidden, N-ny2, N-onehidden, N-anynet | collects T22, T25-T29 |
| Proposition T27.1 | R-fold, N-ny2, N-onehidden, N-anynet | T22 carried over to folds of a large group (s = 2d); its hard teachers are memorisers (N-anynet, dropped under the clean-network decision) |
| Lemma T31 | S-onehidden | one line of linear algebra (one hidden layer) |
| Conjecture D25 | R-cleannet, R-fullrank, R-symN, S-onehidden | covering folds; refuted by T32 (D2) and T33 (D2 and KL) |
| Conjecture D26 | R-cleannet, R-fullrank, R-symN, S-onehidden | broad folds; refuted by T33 |
| Theorem T32 | R-cleannet, R-fullrank, R-symN, R-fold, R-obj, N-ny2, N-onehidden | exact refutation of D25 (D2, width 2) |
| Theorem T33 | R-cleannet, R-fullrank, R-symN, R-fold, R-obj, N-ny2, N-onehidden | exact refutation of D26 (width 1; D2 bound 1/468, and KL) |
| Conjecture D27 | R-cleannet, R-fullrank, R-symN, S-onehidden | signed folds of one family; refuted by T34 |
| Theorem T34 | R-cleannet, R-fullrank, R-symN, R-fold, R-obj, N-ny2, N-onehidden | exact refutation of D27 (width 1, two families) |
| Lemma T32a | (none) | the dual-certificate method of T32-T34 |
| Corollary T35 | R-cleannet, R-fullrank, R-symN, R-obj, N-ny2, N-onehidden | the family structure fails (T32-T34) |
| Lemma T36 | R-symN | symmetry of the objective (premises: an invariant mix and N equivariant on it; D20 implies the second) |
| Lemma T37 | (none) | finite symmetric criticality |
| Conjecture D29 | R-cleannet, R-symN, S-onehidden | least width-1 unit on a mirror axis (dihedral rings) |
| Conjecture D30 | R-cleannet, R-symN, S-onehidden | least width-1 unit fires where the family is silent |
| Theorem T38 | S-twoclass, S-onehidden | first-order structure of a least build |
| Corollary T38.1 | S-twoclass, S-onehidden | width 1: one region, a regression on it, the average elsewhere |
| Corollary T38.2 | S-twoclass, S-onehidden | a unit in a deep network fits the family's output through its own write (identity metric at the readers, no pooling, a bias shift; a scalar target and one ReLU layer by construction) |
| Corollary T38.3 | (none) | first-order condition when the readers' metric couples positions (a normal equation with correlated errors) |
| Lemma T39 | S-twoclass, S-keepfirst, R-gatefold | the least gate fold is a fit on the gate's region (the easy formula, given the region) |
| Lemma T39a | S-twoclass, S-keepfirst, R-gatefold, R-flatgate | when the fit on the gate's region is realisable by one unit |
| Conjecture D34 | S-twoclass, S-keepfirst, R-gatefold, R-flatgate, R-cleannet, R-symN | one gate, width 1: the least build is a gate fold (exact test in `model/tg_gate_instance.py`) |
| Corollary T39.1 | S-twoclass, S-keepfirst, R-gatefold, R-flatgate | exact gate fold from N's weights when no member bends inside the gate |
| Theorem T40 | S-twoclass, S-keepfirst, R-gatefold, R-flatgate, R-cleannet, R-symN | a statement about one fixed network (I1); part 0 proves that I1 meets D34's premises, so the path is D34's. Exact refutation of D34 for D2 |
| Theorem T41 | (none) | the optimum is a minimum over finitely many pieces (any depth, any number of logits, a finite mix) |
| Corollary T41.1 | S-twoclass, S-onehidden | the sign-fixed parts of a piece have closed polyhedral cones as closures (T19), each piece problem is 2^k convex problems (T21); a projection when no constraint is active |
| Corollary T41.2 | S-twoclass, S-keepfirst | T41.1 on the points z = h(x) when the build keeps layer 1 |
| Theorem T42 | R-obj | the optimum is 1-Lipschitz in the network's outputs on the mix (D2 distance); a least build for a nearby network is near-least |
| Lemma T43a | R-cleannet | the detector network of D40 is perfectly clean (D20) and satisfies Q2, Q3 (and Q5 at depth) of D39; its values at vertex points and midpoints |
| Theorem T43 | N-ny2, R-cleannet, R-symN | hardness at the trunk for clean networks: T22 holds for the detector networks of D40 (any depth, sizes all 1) and mixes invariant under their symmetry; NOT under N-anynet |
| Theorem T44 | R-cleannet | exact symmetry that moves a unit (Q1 of D39; D20 for a non-trivial group) holds only on a null set of weights, for a polytope domain. The path is R-cleannet because Aut_X(N) is defined in the proposed D38 |
| Lemma T45a | R-cleannet | the cancelled share of a reader's votes (D41) lies in [0, 1] and is 0 exactly when the votes have one sign |
| Theorem T45 | R-cleannet | networks perfectly clean under D42 contain a non-empty open set of weights (margin-clean networks) |
| Lemma T46a | R-cleannet | bounds and refinement of the packing index, bounds of the read index (D45) |
| Theorem T46 | R-cleannet | the packing and read indices do not change under permutations, positive rescalings or a relabelling of the classes |
| Proposition T47 | R-cleannet | three planted circuits, each with an input where the featured unit is off: a two-job unit has packing index above 0; the V case and the ambulance have 0; all three are read; D42's V is above 0 for the ambulance |
| Theorem T48 | R-cleannet | for dense networks on X = [1, 2] (one input coordinate) with the uniform base law, the networks perfectly clean under D45 contain a non-empty open set of weights |
| Proposition T49 | R-cleannet, R-explaw | the detector networks of D40 at one hidden layer are perfectly clean under D45 with the uniform base law on their domain |
| Lemma T46b | R-cleannet | the sign of the read index is nearly automatic when readers' other inputs ignore the unit |
| Lemma T46c | R-cleannet | the excess read is 0 when the readers' other inputs ignore the unit |
| Proposition T50 | R-cleannet | the detector networks of D40 at one hidden layer fail the agreement form of D45 (excess read < 0) |
| Lemma T51 | R-cleannet, R-explaw, R-jobmix | T43's hard mixes are job-weighted mixes (D48) of themselves as base law |
| Corollary T51.1 | N-ny2, R-cleannet, R-symN, R-explaw, R-jobmix | T43's hardness holds when the mix is a job-weighted mix on a base law that the input supplies; silent when the base law is fixed by N |
| Lemma T53 | N-ny2, R-cleannet | a last-layer unit whose target is one point has lambda = s_u (q_1(x0) - E q_1); units that cancel at one point cannot all be read |
| Lemma T52 | R-cleannet, R-listen | the deciding read is never negative; the co-active read is 0 for a reader of alternatives (D49) |
| Conjecture D47 | N-ny2, R-cleannet, R-symN, R-explaw | TRUNK-1 (sizes all 1, two logits, mixes invariant under N's symmetries) stays NP-hard for networks perfectly clean in the agreement form under the uniform base law on X; option (d) of R-mixclean |
| Lemma T46d | R-cleannet | the excess read of a bias-free OR reader is the co-occurrence covariance; units never on together with the rest have xi < 0 |
| Theorem T54 | S-twoclass, S-onehidden, R-group | classes (r) inside (b) inside (a); least builds exist; (b) = (a) when each group's weights span the input |
| Theorem T55 | S-twoclass, S-onehidden, R-group | one step of the zoom: a product over changed groups of (2 R_t)^(k_t) convex pieces, exponent min(s_t, d) k_t; no bound across many steps |
| Proposition T57 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | a big group handed over in one step on the detector networks is CLEAN-TRUNK (T43), so NP-hard |
| Theorem T58 | R-group, S-twoclass | an AND group collapses exactly to k = rank + 1 units (2 for two logits), from the weights only |
| Proposition T59 | R-group, S-twoclass | exact one-unit merge = one linear program; a blend of firing points where the group is silent blocks it; AND groups always pass |
| Corollary T59.1 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | a group with a bad pair (T43 instance): no zero-loss one-unit (b) replacement |
| Corollary T59.2 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | a group with no bad pair merges exactly, unit c' (sum of z) + const |
| Corollary T59.3 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | one-unit zero-loss groups = colourful cliques |
| Lemma T60 | S-onehidden, R-group | (r) copies a direct (b) unit iff its parts lie in the children's row spaces |
| Corollary T60.1 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | H-recur holds on the detector networks with the mixes of T43 for targets with no bad pair, along every zoom |
| Proposition T61 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | two lossy groups exact together: anti-detector units; best builds of a level need not extend the level before |
| Lemma T62a | S-onehidden | dip identity: the gap at a midpoint minus the average of the ends is a sum over switching units |
| Corollary T62b | S-onehidden | every conflicting segment is paid by a switching unit with a positive output weight |
| Theorem T62 | R-group, N-ny2, N-onehidden, R-cleannet, R-symN | family grouping of T61's instance, zero-loss builds: positive weights, two switches per unit, no presence unit, every unit split |
| Conjecture D57 | S-twoclass, S-onehidden, R-group, R-cleannet | kinds of coarse unit on the detector networks: presence for free groups, clique presence for one conflicting group, split units for conflicting groups together |
| Theorem T63 | (none) | kink formula: the dip at any point of a line is minus a sum of slope jumps times a tent weight, any depth |
| Proposition T64 | S-onehidden | one unit of one hidden layer: monotone along a line, at most one kink; gives T62a, T62b, T59 part 2 |
| Theorem T65 | S-twoclass, S-onehidden | an exact one-layer build has at least tau(D) units: a covering number of the mix's strict dip lines |
| Corollary T65.1 | S-onehidden, N-onehidden | T20's width-1 impossibility is a covering bound (two disjoint dip lines) |
| Lemma T65a | (none) | parity: an affine function switches across an even number of sides of a closed polygon |
| Proposition T66 | N-onehidden | plain-network depth separation on a grid: k layers of two units vs at least 2^(k-1) - 1 units in one layer |
| Conjecture D61 | S-twoclass, S-onehidden, R-cleannet, R-freefam, R-circuit | for free families the covering bound tau_b is reached, by presence units |
| Conjecture D52 | S-twoclass, S-onehidden, R-group, R-recur, R-cleannet | Hypothesis H-recur with merges of two units |
| Conjecture D53 | S-twoclass, S-onehidden, R-group, R-cleannet, R-symN | whole layer, D2, silent blends in the mix: a least OR step fires only on one member's points, fit of T38.1 there |
| Theorem TM1 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup | Under T5: a map whose district, shelter and judgment claims hold splits a grouping into its regions; groups in different regions do not interact, and switching one group costs only that group |
| Theorem TM3 | R-map, R-group, R-mixclean, N-ny2, N-onehidden, R-cleannet, R-symN | Under T5 read with polynomial time: the one-district map has error 0, so T5 asks for a fast exact rule on T43's instances (P = NP); for a polynomial-time class A of maps on which a rule meets the clause, some no-instance has no small error-0 map in A (else NP = coNP) |
| Proposition TM4 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup, N-ny2, N-onehidden | the split has no bound in the claims' errors alone: a tiny shelter error (new unit) or a tiny district error (kept unit) is amplified by a weight 1/delta; CM1 and CM1' withdrawn |
| Conjecture CM2 | R-map, R-group, R-mixclean, S-onehidden, S-twoclass | a "no silent blend" claim type that carries T5's exact clause inside a region; special case = T59.1, T59.2 |
| Theorem TM5 | R-map, R-group, R-mixclean, S-onehidden, S-twoclass, N-ny2, R-cleannet | on T43's point set (any positive weights, either logit order): two-sided blend error 0 <=> no bad pair <=> an exact size-1 build (T59.2's formula) |
| Corollary TM5.1 | R-map, R-group, R-mixclean, S-onehidden, S-twoclass, N-ny2, R-cleannet | with complete maps the big group of T43 has no complete error-0 map (no conflict on these inputs); zooms with one group per level and no bad pair get the exact formula |
| Conjecture CM3 | R-map, R-group, R-readbias, R-readweights | the split for groups in deeper layers whose descendants never meet below the logits (route TU16) |
| Lemma TE0 | (none) | KL moves by at most the oscillation of the logit change |
| Proposition TE1 | R-map, R-group, R-readweights, R-readbias, S-lastlayer | with free reader weights the least loss depends on a kept unit only through the span of its values on P; a leak of size delta is worth as much as a full-size one, and reaching that gain needs a swing that grows like 1/delta |
| Proposition TE3 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer | the robust optimum equals the least loss when every error is 0, is never below it, and asks a build to work with and without each leak |
| Theorem TE4 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup | graded split: the robust optimum lies between the clean split sum and that sum plus E (each moved entry times the swing of the unit that carries it: the clean assembly's swings in terms (a), (b), N's own in term (c), which quiet claims bound); does NOT meet T5's graded clause (TE5) |
| Corollary TE4.1 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup, N-ny2, N-onehidden | on TM4's two instances the robust optimum equals the clean value for every delta: the blow-up is gone |
| Proposition TE5 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer, N-ny2, N-onehidden | the graded clause is not met by TE4: a complete map whose district, shelter and judgment errors tend to 0 while the robust optimum stays a fixed amount above the clean split and TM1's assembly loses without bound; N sends a large amount through a tiny leak (large quiet error, DM9) |
| Proposition TN1 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer | a copy of a unit (the most a network can grow without changing) changes no quantity T5 asks for (function, V_b, V_rob, clean split, least E) and, with the largest-term errors of 2026-10-28, no error of the map; the first, summed forms grew by (n - 1) times u's term |
| Corollary TN1.1 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup | copies keep TE4's premises and the least E over e-near clean assemblies; the lift keeps E, the merge does not raise it |
| Proposition TN2 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer | narrower units that add up to one kept unit never raise V_b or V_rob and leave the fair deviation and the deviation sent unchanged |
| Theorem TN3 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup | TE4's E is at most two masses times two weighted means plus the deviation N sends; all are unchanged by copies and rescaling; no term counts units |
| Corollary TN4 | R-wellposed, R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup, R-scale, R-mass | along a family of sizes, under H-scale (map form) and H-mass, the bound on E does not grow and falls when the per-unit deviations fall |
| Lemma TY1 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, S-cover, S-regiongroup, R-unitmap | TM1 holds word for word for a last block with positions and a skip (Definition DY1): the proof never uses how the group's pre-activations are computed |
| Theorem TY2 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, R-unitmap, R-propcomplete | the merge rule (Definition DY2): closed-form weights; loss at most E^2/4 with E the sum of the square roots of the proportional errors (DM10) it uses; exact when they are 0; any number of logits; no districts or regions needed |
| Proposition TY3 | R-map, R-group, R-readweights, R-readbias, S-lastlayer, R-unitmap, R-propcomplete | switching between zoom levels is a lookup after a one-time computation; the bound is a sum over groups; least-cost cuts of a laminar claim tree for every k by one dynamic program |
| Lemma TV0 | S-purestream, R-streamregion | a cut of the stream is the network with every write projected onto the kept subspace; only the subspace matters, not its basis |
| Theorem TV1 | S-purestream, R-streamregion, R-normconst | a part of the stream that no reader reads can be dropped exactly without a norm; with norms, exactly when its energy is the same at every point of P before each norm (then it folds into the norm's constant); a temperature bound otherwise |
| Corollary TV1.1 | S-purestream, R-streamregion | if the unread content lies on one line (the essay's unread direction), keeping that line costs one direction and the cut is exact |
| Theorem TV2 | S-purestream, R-streamregion, S-laminar, R-map, R-readweights, R-readbias, R-normconst, R-cutclass | for areas no block reads: the loss of the area cut is at most a quarter of the squared sum of the square roots of the readout errors (DM12) it uses, area by area; exact when they are 0; a bound in the claims' errors alone |
| Proposition TV3 | S-purestream, R-streamregion | the least readout error of an area with at most k directions is the tail of the singular values of M Sigma^(1/2) (reduced-rank regression, Eckart-Young); the minimisers nest in k; it is the least bound, not the least loss |
| Corollary TV2.1 | S-purestream, R-streamregion, R-map, R-readweights, R-readbias, R-normconst, R-cutclass | TV2 at an angle: oblique cut, one oblique energy claim on the whole dropped part; loss <= (sum of sqrt readout errors)^2 / 4, exact at 0; TV2's per-area part 3 does not carry over |
| Corollary TV3.1 | S-purestream, R-streamregion, S-laminar, R-map, R-readweights, R-readbias, R-normconst, R-cutclass | an area that no block reads never needs more than C - 1 directions (one for two logits): the cut to them is exact when the energy claims hold |
| Proposition TV4 | S-purestream, R-streamregion, S-laminar, R-map, R-readweights, R-readbias, R-normconst, R-cutclass | switching across a zoom of readout areas is a lookup after a one-time computation; the bound is a sum over areas; a cut built from an area's children costs at most the sum of the children's terms |
| Lemma TV5 | S-purestream, R-streamregion, R-map, R-readweights, R-readbias, R-normconst | the areas that no block reads are exactly the subspaces of the readout kernel K; K = {0} for generic weights once the blocks have at least d neurons in all; so TV2 to TV4 at error 0 apply only on a null set of weights |
| Theorem TV6 | S-purestream, R-streamregion, R-readweights, R-readbias, R-normconst | any cut of the stream: loss <= (direct readout loss + each block's leak carried to the logits)^2 / 4; errors grow by (I + T) per block (the identity path); without a norm the carrying follows the area graph; with norms every area couples. No exactness premise; gains of N inside |
| Proposition TV7 | S-purestream, R-streamregion, R-readweights, R-readbias, N-amplify, N-ny2, R-normconst, R-cutclass | no bound in the errors of claims made one block at a time (both forms of the block-read error, for every area) tends to 0 with them: the block writes a value of size delta from the cut part, the logits read it with weight about 1/delta; the least loss stays c(s) |
| Corollary TV6.1 | S-purestream, R-streamregion, R-normconst, R-map, R-cutclass | with carried leak claims (DM14; D a sum of members of the map's decomposition), the projection cut's loss is at most (sum of their square roots)^2 / 4: a bound in claim errors alone, 0 at 0; the errors carry N's gains |
| Proposition TV8 | S-purestream, R-streamregion, R-readweights, R-readbias, R-normconst | when the computed stream error stays below eta rho, the norm's coupling in TV6 uses each neuron's largest read on P plus 2 eta times its read norm |
| Theorem TV9 | S-purestream, R-streamregion, R-readweights, R-readbias, R-normconst, R-cutclass | the build cancels a leak: each re-fit read's least leak is a closed form (least-squares fit of N's read on the cut's kept content); exact when every read of N is affine in the kept content on P |
| Proposition TV9.1 | S-purestream, R-streamregion, R-readweights, R-readbias | a copy the cut can read instead: the projection cut loses log 2 - H, the TV9 cut is exact |
| Proposition TV9.2 | S-purestream, R-streamregion | no cut that drops D recovers what the first write loses: D_P >= E KL(q or the mean of q over points with the same kept first write) |
| Lemma TV10 | S-purestream, R-streamregion, R-readweights, R-readbias, S-nonorm | without norms a cut is fixed by what it drops: oblique cuts (along any complement) compute the same functions |
| Lemma TV11 | S-purestream, R-streamregion | overlap and crowding: a re-fit read can ignore D and read E exactly iff r vanishes on D cap E; for every r iff D meets E only in 0; vacuous under laminar areas (S-laminar) |
| Lemma TV12 | S-purestream, R-streamregion, R-cutclass, R-normconst | a re-set norm constant is one more stream direction holding a constant, never read (one constant, never-read direction) |
| Lemma TV13 | S-purestream, R-streamregion, R-readweights, R-readbias, S-nonorm | without norms the logit change of any cut is exactly a signed sum of carried leaks; TV6 is this with signs and slopes replaced by sizes |
| Proposition TV14 | S-purestream, R-streamregion, S-smoothact, R-readweights, R-readbias, S-nonorm | no norm: the signed sum with N's own slopes, from N, P, D and the cut's reads, gives the logit change up to a pointwise remainder of second order in the leaks; the bound tends to Fact F11's as the leaks shrink |
| Corollary TV6.2 | S-purestream, R-streamregion, R-normconst, R-map, R-cutclass | the stream branch in T5's form for area maps complete for a zoom (DM15): projection cut per level, bound on its loss in the carried leak errors alone, exact closed form at error 0, switching by lookup; by construction |
| Lemma TV13.1 | S-purestream, R-streamregion, R-normconst, R-readweights, R-readbias | both forms: the logit change of any cut is exactly a signed sum of carried leaks; the norm adds its secant matrix as a second slope |
| Lemma TV15.1 | S-purestream | the remainder modulus of an activation: (Lip'/2) t^2 when smooth; o(t) at a differentiable point; 0 inside the distance to the nearest kink (ReLU: (1 - alpha) max(t - abs(a), 0)) |
| Lemma TV15.2 | S-purestream | the norm's slope (I - n n^T)/nu and its remainder: second derivative at most (2/sqrt 3)/nu^2 (sharp); premise-free bounds on a segment |
| Theorem TV15 | S-purestream, R-streamregion, R-normconst, R-readweights, R-readbias | both forms, every Lipschitz activation: the signed sum with N's own activation and norm slopes gives the logit change up to a pointwise remainder computed from N, P, D and the cut's reads; proves Conjecture DV3; TV14 is a special case |
| Corollary TV15.3 | S-purestream, R-streamregion, R-normconst, R-readweights, R-readbias | order of TV15's remainder: second order for smooth activations; for ReLU-type activations the activation part is 0 inside N's margin on P (no norm: Lambda_N is then the exact change); o(t) at differentiable points |
| Lemma TV16 | S-purestream, R-streamregion, R-readweights, R-readbias, R-cutclass | areas at an angle: every cut that drops D is the orthogonal cut in other coordinates; no norm: same functions and V_area; norm: one scalar per norm; TV13-TV15 hold for the oblique cut; TV6 parts 1-2 hold for any direct decomposition with oblique projections |
| Proposition TV17 | S-purestream, R-streamregion, R-normconst, R-cutclass, R-readweights, R-readbias | norm form at an angle: the oblique cut's energy is signed (exact when it is constant and eps + c > 0); per-area energies do not add (a 3-dim example with every per-area claim at 0 and every cut lossy); so energy claims are made on the whole dropped part |
| Theorem TV18 | S-purestream, R-streamregion, S-laminar, R-normconst, R-cutclass, R-map | a map complete for every zoom at once: for every sum of members D, the cut that drops D has loss <= (sum of sqrt of the member claims of the members in D)^2 / 4; claims name one member each and no zoom; norm form with orthogonal members, no-norm form at any angle |
| Corollary TV18.1 | S-purestream, R-streamregion, S-laminar, R-normconst, R-cutclass, R-map | the stream branch in T5's form for every zoom of groups made of members, from one zoom-free map: per group a knapsack picks the members to drop; bound in the claims' errors alone; exact closed form at 0; switching by lookup |
| Proposition TV18.2 | S-purestream, R-streamregion, R-normconst, R-cutclass, R-map | TV18 in the norm form with members at an angle: cross terms with every other member go into each member's claim; zoom-free gains with s(r); the cut keeps eps + max(c(D), 0) and each member carries the negative mean cross energies it touches; no premise |
| Lemma TV19 | S-purestream, R-streamregion | a fact about one network (used by TV19.2): every pure-stream network fits, with the same neurons and the same function on P, in the width of its written span (the span of its stream on P); no claim, any activation, either norm form |
| Proposition TV19.1 | S-purestream, R-streamregion, R-normconst | a fact about one network: no norm, a network fits in the width of its written span modulo its read kernel (minimal realization); with a norm, the same in the exact case where the dropped part's energy is constant on P |
| Theorem TV19.2 | S-purestream, R-streamregion | EQ: the narrow unit builds of a zoom (Definition DV6, defined from N alone: one isometry whose range holds E_N) have the same least loss as the wide ones, for any objective, once the width is at least dim E_N + sum k_S; each optimum is the other cut (Lemma TV19) or lifted |
| Proposition TV19.4 | S-purestream, R-streamregion | an example where two least losses differ (no label): a full first write forces the full stream: the dim E term cannot be dropped |
| Lemma TH0 | R-map, R-group, R-tree, R-branch, S-plainff | inside a node's area everything later is a function of the node's units; with branch claims, removing a group changes no kept unit |
| Theorem TH1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff | the split down the tree: at any grouping of subtrees, at any mix of depths, the least loss (D_P and D2_P) is a sum of one local problem per group; with parents in layer L - 1 it is a case of TM1 |
| Theorem TY4 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, R-unitmap, S-skipff | EQ: the least KL loss at a tree grouping equals the sum of one local least per group (Theorem TH1) for feed-forward networks with skips and channel units (a ResNet after its stem; copies carry channels across strides by average pools); joint groups of inner and stream channels are subtree groups |
| Proposition TY5 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, R-unitmap, S-skipff, R-obj | part 1 OPT-KL closed form: a group replaced by nothing, with judgment values constant on each area, is replaced by the log of the mean softmax per area; parts 2 to 4 price D2: exactly one KL per area there, a network with every claim at 0 where the D2-least build is not KL-least, and in general only a factor 1 / (2 mu) (confidence, not cleanliness) |
| Lemma TI0 | (none) | OPT-KL tool: the chain rule of KL over a partition: the sum of KL(s_x \|\| q) over a set is its information plus mass x KL(mean \|\| q); information of a union = sum of the parts' information + the between terms |
| Theorem TI1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside | OPT-KL closed form: a group replaced by nothing (kept judgment units flat on each area) has KL least = sum of its nodes' information I_w, at any depth, attained by log of the mean softmax per area; sibling merges add; a lift adds the children's information plus mass x KL(child mean \|\| parent mean); a zoom of such groups is least at every level from one table pass |
| Theorem TI2 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-linread | OPT-KL: a group with at least rank Sigma units has KL least <= eps_lin / 4, and TP1's box is within eps_lin / 4 of it (0 at error 0); for any size the TP1 box has KL <= Lin_S / 2 |
| Corollary TI2.1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, S-keepdrop | OPT-KL: at a grouping of dropped and fully kept groups the least KL is the dropped groups' information plus at most eps_lin / 4 per kept group; the assembled build is within that of the least, exact at error 0 even where the least is > 0 |
| Theorem TI3 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff | EQ (part 2) and a property of every KL least: on every area the least build's mean softmax equals N's (and its features' moments match); among boxes on throughout the areas the least is a least over k-dimensional output spans of convex soft-label logistic regressions, and it is attained |
| Theorem TI4 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread | OPT-KL to second order only: with the variation inside the areas scaled by a, the KL least is a^2 times a reduced-rank problem in each area's Fisher metric H_w plus O(a^3); with one mean softmax it is (1/2) tail_k(H^(1/2) Sigma H^(1/2)); D2's directions are KL's to this order only when that mean softmax is uniform; switching boxes gain nothing to this order |
| Proposition TI5 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread | NEG: the D2 summaries (E, Sigma), even with the area means, do not fix the KL least: two pairs of clean instances with equal tables and different KL least (k = 0, two logits; k = 1, three logits, where a D2-least build loses a factor 3/2 at second order) |
| Conjecture DI1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread | open case: for clean groups with 0 < k < rank the KL least is NP-hard to compute within eps (route: TI4's per-area Fisher metrics and weighted low-rank approximation) |
| Theorem TQ1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-active, S-squeezecomplete | part 1: the group least >= the sum of the nodes' own leasts; parts 2 to 4: ONE-BUILD, a packed box with max own-class rank units has terms <= sum eps_act (summaries of N, not claims' errors); a level of drop, keep and packed groups has one bound |
| Proposition TQ2 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-active, S-squeezecomplete | part 1: eps_act = 0 only for a node without variation (a property of the summary DQ1, under none); part 2: every row-3 least (k_S < rank Sigma_(W_S)) is > 0; part 3: a clean network where one unit serving two nodes beats every unit serving one (the tree knapsack is not the form). Withdrawn 2026-10-31 (REFEREE-BQ): "row 3 is never exact" |
| Conjecture DQ1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-active, S-squeezecomplete | first-order tightness of TQ1's bound for nodes of comparable confidence (route: chain rule over classes, balance of the two prices); uneven confidence may break it (referee sketch) |
| Theorem TQ3 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-active | LIMIT: sum of (1 - mu_w/p_w) own-block leasts <= least of a squeezed group <= sum of own-block leasts + sum E_w (E_w <= 2 D_w mu_w); both sides meet as the areas become confident; relative gap -> 0 where the own-block leasts are > 0 (always-on boxes) |
| Conjecture DQ2 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-active, S-squeezecomplete | the cost of sharing between two packed areas, to second order: sqrt(mu_1 mu_2) sum sqrt(p_i q_i) with reverse pairing (Lemma TQ4); the first form (balancing) was wrong |
| Conjecture DQ3 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-active | the recursion of TQ3 down the tree (the demoted Remark) |
| Lemma TQ4 | (none) | min over T of tr(T P T^T) + c tr(T^-1 Q T^-T) = 2 sqrt(c) sum sqrt(p_down q_up): reverse pairing |
| Theorem TQ5 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-active, S-squeezecomplete | LIMIT-1 (always-on boxes, along N_m, no rate): confident limit of a complete squeeze: L^on_S(m) <= e^-m Lambda_S for every m, and e^m L^on_S(m) -> Lambda_S (the leak least of Definition DQ5, a least over spans of convex problems); the build from a minimiser is optimal to first order (always-on boxes) |
| Lemma TQ6 | (none) | a rank-k n x n matrix with unit diagonal has off-diagonal energy >= n^2/k - n; equality iff (n/k) x a projection with constant diagonal (equal-norm tight frame); exists for every n >= k |
| Conjecture DQ4 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-active, S-squeezecomplete | the leak least of DQ5 is NP-hard in the number of areas (own rank 1, k = 3); route via min-rank (Peeters 1996); equal weights closed form (TQ6) |
| Corollary TQ10.1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-rankclaim | ONE-BUILD: shared-span build <= (1/2) g(omega) eps_rank |
| Proposition TQ7 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread | SECOND-ORDER: one metric for the group within 1 +- eta: closed-form shared span within (1 + eta)/(1 - eta) of the least, Phi pinned between (1 -+ eta) tail/2 |
| Lemma TQ8 | (none) | relative curvature sandwich of KL at N's own point: factor in [g(-omega), g(omega)], omega the spread of the change |
| Proposition TQ11 | R-group, R-readweights, R-readbias, S-lastlayer | L_free(k) <= V_b(k) <= L_on(k); every network with k new top units that keeps T's values is at least L_free(k), and reaches it when lower layers are free; a proxy of a cells and k directions is at least L_free^0(k + a - 1) and can lie below L_free(k) |
| Proposition TQ13 | (none) | certificate: for rank <= k and \|W\|_F <= R, D >= mean entropy gap - R \|Gamma_q\|_(k); over every weight through R*(U) |
| Theorem TQ14 | R-group, R-readweights, R-readbias, S-lastlayer, R-constclaim | where N is constant on regions, the least of a squeeze is the least of a table of K rows; with z constant too, a group and a circuit agree; graded in the claim's errors against free codes |
| Conjecture DQ11 | (none) | symmetric table with K = 2k: the pole build (TQ16) is the least; proved as Theorem TQ17 part 4 |
| Theorem TQ17 | (none) | the closed form of the table least at K = C = 2k, above 0, with a matching lower bound; graded for the table in part 6 |
| Theorem TQ18 | (none) | a global certificate of a triple at any k: stationary, g <= rho sigma / 2, g <= rho; free codes or an always-on read, kept units allowed; graded in parts 3 and 5; row-wise in part 6 |
| Lemma TQ19 | (none) | lumping the tail lowers the KL, linearly in the logits |
| Corollary TQ18.1 | R-group, R-readweights, R-readbias, S-lastlayer | a read that passes the free-code test is V_b(k), with T's units kept |
| Lemma TQ20 | (none) | the exact global test: D(l') - D(l) = sum of p_x [KL(q_x \|\| q'_x) - <s_x - q_x, l'_x - l_x>] |
| Proposition TQ21 | (none) | tables with a family tree: relatives on one pole, at several cuts, under TQ18's premise, which forces fine levels below 1/K |
| Proposition TQ22 | (none) | minors: L_Z(k) is at least the sum of the leasts of family sub-tables, each with k units |
| Theorem TZ1 | (none) | the patchwork certificate: one local projector per patch (round name D3) |
| Theorem TZ2 | (none) | the reflection certificate (round name D4) |
| Theorem TZ3 | (none) | moment certificate for an always-on read (round name E6) |
| Theorem TZ4 | (none) | packing dual (round name C10) |
| Corollary TZ5 | (none) | level tail, any k (round name C11) |
| Lemma TZ6 | (none) | the gap is convex in the table (round name I1) |
| Theorem TZ7 | (none) | graded in the largest log-ratio, first order (round name I2) |
| Theorem TZ8 | (none) | graded, second order, at a table with margin (round name I3) |
| Theorem TZ9 | (none) | sharp moment certificate (round name J1) |
| Theorem TZ10 | (none) | what z cannot compute: an exact split (round name J3.1) |
| Theorem TZ11 | (none) | exact read cost at a binding rank (round name J3.3) |
| Proposition TZ12 | R-group, R-readweights, R-readbias, S-lastlayer | hinge form of a gate; the read is generically not V_b (round name J4.1) |
| Theorem TZ13 | R-group, R-readweights, R-readbias, S-lastlayer | a gate can remove all of the read cost (round name J4.2) |
| Theorem TZ14 | (none) | the wall theorem (round name H1) |
| Theorem TZ15 | (none) | the limit of the wall program (round name H2) |
| Lemma TZ16 | (none) | structure of the wall program (round name H3) |
| Theorem TZ17 | (none) | the wall program at k = 1 (round name H4) |
| Theorem TZ18 | (none) | the level constant (round name H5) |
| Lemma TZ19 | (none) | the two-table identity (round name F2) |
| Lemma TZ20 | (none) | overflow: Sylvester with numbers |
| Theorem TZ21 | (none) | kept families, one shared budget |
| Lemma TZ22 | (none) | exact information dual |
| Theorem TZ23 | (none) | information floor |
| Proposition TZ24 | (none) | ceiling of information floors |
| Theorem TZ25 | (none) | a certified coupling for every head |
| Lemma TZ26 | (none) | the fixed-span dual and the row cap |
| Lemma TZ27 | (none) | the worst entry of a projected vector |
| Theorem TZ28 | (none) | the weighted rectangle dual |
| Proposition TZ29 | (none) | the ceiling on one rectangle |
| Proposition TZ30 | (none) | the block ceiling |
| Lemma TZ31 | (none) | the box as a floor at a span plus a distillation of codes |
| Lemma TZ32 | (none) | four-term split over regions |
| Theorem TZ33 | (none) | the signed cut program |
| Lemma TZ34 | (none) | a perfect separator |
| Corollary TQ23.1 | R-map, R-tree, R-branch, R-oninside, R-linread, S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | no polynomial closed form at k_S = 1 on maps with every claim at 0, unless P = NP |
| Corollary TQ23.2 | S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | an exact claim type with a polynomial-time closed form must leave some of these networks out, unless NP = coNP |
| Corollary TQ23.3 | S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | no universal short certificate at factor 1 + 1/(8 m n), unless NP = coNP; no constant factor is excluded |
| Proposition TQ24 | (none) | with no scales, the low-weight graph of a monotone tree pattern, square or rectangular, has its largest biclique by dynamic programming, so the reduction of TQ23 does not embed there |
| Proposition TQ25 | (none) | a threshold on a function of two unary sizes cannot be NP-hard unless P = NP |
| Proposition TQ26 | S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | a source with one bad entry lifts to a ratio below 1: a limit of method |
| Lemma TQ27 | (none) | a set with few conflicts holds a large independent set; every f-coloring of a conflict system leaves pair mass about 1 / (R^2 f) |
| Proposition TQ28 | (none) | at rank r, a build fits all but 1 / b(r) of a line digraph's diagonal, on any graph |
| Proposition TQ30 | R-map, R-tree, R-branch, R-oninside, R-linread, S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | V_b(1) equals a vertex cover number within 1 + O(delta), with every claim at 0: a fixed factor below about 1.36 is NP-hard |
| Proposition TQ31 | R-map, R-tree, R-branch, R-oninside, R-linread, S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | with signed M, every fixed factor is hard at k = 1 under the Unique Games Conjecture, at any movement and any tail mass |
| Proposition TQ32 | (none) | on a finite list, an information floor over all decoders is 0: a limit of method |
| Lemma TQ29 | (none) | KL is a variance along the geometric path between s and q, at any amplitude; a tool, no lower bound over competitors |
| Conjecture DQ21 | S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | every fixed factor is NP-hard with no conjecture; the known routes and their limits |
| Conjecture DQ17 | R-treeweight | at one unit, a least order of the wall program respects a monotone tree, with no scales |
| Lemma TQ23.c | (none) | padding: rank-r fitting on M plus r - 1 heavy blocks holds the biclique value |
| Lemma TQ23.a | S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | the squeeze of N(M, W) with areas kept is rank-r weighted fitting, within 4 alpha + 6 eta |
| Lemma TQ23.b | S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | the same with no kept units, at k = m - 1 + r, for B large |
| Theorem TQ23 | R-map, R-tree, R-branch, R-oninside, R-linread, S-plainff, R-group, R-readweights, R-readbias, S-lastlayer | the squeeze least is NP-hard on networks with every claim at error 0; proves DI1 |
| Conjecture DQ15 | (none) | with confident families, a squeeze's cost beyond its minors is a leak term |
| Corollary TQ17.1 | R-group, R-readweights, R-readbias, S-lastlayer, R-constclaim, R-poleclaim, S-polescope | T5's exact promise for a squeeze at C logits, at total written rank C/2 |
| Proposition TQ16 | (none) | the pole build: stationary, value in closed form, above 0 |
| Lemma TQ15 | (none) | the least over spans is semiconcave along rotations: a certificate from the corners of a cell, with an error quadratic in the cell size |
| Lemma TQ12 | R-stat | teacher-student identity for heads on N's own h: excess of N's training objective = D_P + (lambda/2)\|W' - W\|^2, with the stationarity errors written out in part 2; part 1 holds for any logits |
| Lemma TQ9 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside | certificate for the always-on least with one coarse unit: grid of cells, dual values, Lipschitz step with code bounds |
| Theorem TQ10 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread | LIMIT-2: small dropped part: least = e^2 Phi + O(e^3), Phi closed form (weighted least squares, per-point Fisher), span pinned to the shared span; constants implicit |
| Conjecture DQ8 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread | pinning of near-least builds with constants from N; then Psi sandwiches the always-on least |
| Conjecture DI2 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread | for KL a least box may switch inside its area (L_S < L_S^on) at eps_lin = 0, unlike D2 (TP1 part 1); a computed candidate in model/ti_open.py part D; route: interval arithmetic and dual bounds per line |
| Proposition TH2 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff | along sibling merges (creations included, sizes not growing) V_b is monotone and an exact top level gives an exact (r) sequence; lifts do not nest (exact instance: coarser level exact, finer levels lossy) |
| Proposition TH3 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, N-ny3 | at a lossy sibling merge with three logits, D2_P: least 2, every (r) sequence at least 3; a child drops a direction its parent needs |
| Conjecture TH4 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff | oracle form: a parent's least loss (D2_P) is a polynomial-time function of its children's rational summaries, the one-node problem given by an oracle; closed form (Eckart-Young) when least boxes are on throughout their areas |
| Theorem TP1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-linread, R-obj | one group, D2_P: (1/2) tail_k <= least <= Lin_S = (1/2) tail_k + (1/2) eps_lin, from the sum of the nodes' summary matrices; a box of always-on units reaches Lin_S; exact closed form at error 0 |
| Theorem TP2 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-linread, R-obj | summaries pass up the tree: a sibling merge adds the matrices; every level is computed from summaries alone, in any order, within (1/2) eps_lin of least; a compressed summary loses at most half the trace it drops; in TH3's instance the (r) step is such a compression; lifts recompute summaries from N |
| Proposition TP3 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-linread, R-obj | cost: one pass of N over P and one least-squares solve per node; a level's value and box directions cost O(|W_S| n_y^2 + n_y^3) with no access to N; its weights cost linear in the width of the group's top and judgment units; box units nest in the size; a switch selects stored weights |
| Corollary TP1.1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-linread, R-obj | among boxes on at every point of their areas the least is Lin_S for any linear-read error: Route (1) of TH4 as stated |
| Corollary TP2.1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-linread, R-obj, S-twoclass | two logits, sizes >= 1: Lin_S = half the linear-read error; under Route (1)'s premise a sibling merge loses nothing beyond its children and its new nodes (Route (2) of TH4) |
| Theorem TL1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-linread, R-obj, N-ny3 | from a build whose boxes are of TP1 type, a merge step made from the children's box units reaches Lin of the next level when condition (K) holds (every kept direction of the merge is, inside each child's areas, one its box writes); within half the linear-read errors of least, least at error 0; (K) holds when every child lost nothing, or siblings write orthogonal directions and k' <= k_c; sufficient only; fails in TH3's instance |
| Theorem TL2 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside | a one-layer lift through units that are on in their area (judgment units flat) re-expresses every finer unit one layer down; when k_(S') >= n_j(S') the least loss does not rise, for D_P and D2_P |
| Corollary TL2.1 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside | TH2 part 2's instance meets TL2's premises: V_b(G_3) <= V_b(G_2) <= V_b(G_1); the second inequality is new (in fact an equality) |
| Theorem TL3 | R-map, R-tree, S-plainff, R-oninside, R-branch | district, shelter, on-inside and judgment (any hidden layer) claims at eps 0 in a subtree: on its area the logits and every area indicator below are affine in its top pre-activations |
| Corollary TL3.1 | R-map, R-tree, S-plainff, R-oninside, R-branch, R-linread | in TL3's setting the linear-read error of the subtree's node is 0 |
| Corollary TL3.2 | R-map, R-tree, S-plainff, R-oninside, R-branch | TL3's premises force every split below the top to be a flat 0/1 direction of the top layer; with points in general position and more than 2|T_w| of them, no split below the top |
| Proposition TL4 | R-map, R-tree, S-plainff, R-oninside, R-branch, R-linread | at a lift the parent's table is the sum of its children's tables plus one between table of the children's masses and mean outputs (law of total variance) |
| Corollary TL5 | R-map, R-group, R-readweights, R-readbias, R-tree, R-branch, S-plainff, R-oninside, R-linread, R-obj | for a tree grouping whose nodes meet TL3: least D2 = sum of (1/2)(E - top_k Sigma), attained by always-on boxes; tables by add (merge) and add plus between (lift) from the layer-L tables, the masses and c_N; nested; switching selects stored weights; scope: subtrees with no switching below the top except along flat 0/1 directions of the top layer (linear subtrees at points in general position, TL3.2) |
| Theorem TL6 | R-map, R-tree, S-plainff, R-linread | with errors: eps_lin(w) <= |P_perp W_out|^2 eps_L^2, eps built layer by layer from absolute defects, weight norms, shelter values and inverse gate margins; uses no claim; bounds the gap to the least only where TP1 applies |
<!-- /paths -->

## 7. The guard

| Guard | What it checks | File |
|---|---|---|
| Ledger lint | Every tag phrase matches section 5. Every tagged item has a row in section 6. Every node a tag names is on the item's path. Every node on a path exists in section 4. Every hypothesis of Part A.3 has an alias. DECIDED rows name Shonu and a date. RETRO rows were added on or before 2026-10-25. | `tools/ledger.py`, called by `tools/paper_lint.py` |
| Trunk lock | The text between the trunk markers has the hash in `tools/trunk.sha256`. I change the hash only after Shonu says so. | `tools/ledger.py` |
| Turn reminder | Prints the trunk and the open decisions on every prompt | `tools/turn_reminder.sh` |
| Referee | Asks whether any proof uses a premise that is not on its path (the linter sees only tags) | `.claude/agents/math-referee.md` |

---

## 8. The tree in one picture

Distance = number of steps away from the trunk. The arrow marks where revision 7 of the paper stopped.

```
T*  whole network, any size, KL, ReLU, real images
 |
 +-- B.T at the trunk: T19, T21 (positive, two logits, one hidden layer, finite mix); T23, T20, T22 (proved conflicts)
  +-- R-least  the least need not exist (T23): clean networks         [DECIDED]
 +-- R-nest   nesting (T20): goal-indexed; switching kept           [DECIDED]
 +-- R-root   least, fast, every network (T22): a property of nets  [DECIDED]
 +-- R-score  best without data? (T24): a mix is an input (T***)    [DECIDED]
 +-- R-fold   any new units kept; folds = conjecture (T25)           [DECIDED]
 +-- R-foldopt  the family structure fails for D20-clean (T32-T35)   [DECIDED: (c), the optimum is the object]
 +-- R-mixclean  clean networks keep T22's hardness via the mix (T43)  [PENDING]
 +-- R-group  the developer gives the zoom; new units read their group  [DECIDED: Trunk T4]
 +-- R-groupopt, R-recur  (a) = (b)? step by step optimal? (3-10, 3-11) [PENDING]
 +-- R-deep  "reads only its group" in layers 2 and up (3-12)        [PENDING]
 +-- R-kinds, R-freefam  a kind of coarse unit per kind of group (3-13) [PENDING]
 +-- R-circuit  coarsen circuits across layers, not groups (3-14)     [PENDING]
 +-- R-indep  coarsen parts independently: graded, via output-faithful boxes (3-15) [PENDING]
 |     +-- R-outfaith  a box sees only what the group sent to its readers (3-15) [PENDING]
 |     +-- R-readbias  readers of a coarsened group re-fit their bias (3-16) [DECIDED]
 |     +-- R-edgetether  a box writes only where its members already write (3-17) [PENDING]
      |
      +-- Proposition T1'   (distance 0: sums of ReLU units are not one unit)
      |
      +-- R-obj   KL -> squared logit error
      |    +-- Theorem T12   floor for every network               (distance 1)
      |    +-- R-gate   one shared switch, kept at full size        [DECIDED]
      |    |    +-- T9, T6, T7, T8                                  (distance 2-3)
      |    |    +-- R-eq, R-symN   symmetry: T10, T10.1             (distance 5)
      |    +-- R-selfgate   one shared switch from its own sums
      |    |    +-- T16, T18 (distance 3), T11 (distance 4)
      |    |    +-- R-extra   extra units: T11.1                    (distance 4)
      |    |    +-- R-step    counterexample T15, hardness D7
      |    |    +-- R-gauss + R-depth1 + R-onecol + R-affine
      |    |         T17, T17.2, T17.3, D8                          (distance 8-9)  <-- rev 7 stopped here
      |    +-- R-sgate   spatial windows: T13 (5); hardness T14 (3)
      |
      +-- R-family   one family, rest of the network kept
      |    +-- T0, T1a-c, T5, T2, T4, D1, D2                        (distance 3-14)
      |
      +-- S-datamath   statements about images only: T3            (distance 1-3)
```

## 9. Cards (plain words, one per axis)

Each card: what the restriction means for a real network, what it bought, what it loses, and my recommendation.
Shonu decides. The order is from the root down, because a choice at the root can make a lower card moot.

### Card 0. The root (R-least, R-nest and R-root): see section 3

Rewritten 2026-10-25 (late) from Theorems T20 and T22. The old options A/B/C are withdrawn.

### Card 1. The score (R-obj, R-local, R-constmetric)

| | |
|---|---|
| Meaning | A real small network is judged by KL: how far its class probabilities are from the big network's. The paper judges it by the squared difference of the raw class scores (logits), after removing their mean. |
| Bought | Exact formulas. With squared error, "best network of size k" becomes "keep the top k directions" (a principal-component problem). |
| Lost | KL cares most about the classes the network is unsure between. Squared error treats all classes the same. A network that is best in squared error need not be best in KL. |
| Known bridge | KL is at most a quarter of the squared error (Fact F11), so a small squared error gives a small KL. The reverse comparison has a factor that grows with the number of classes (Corollary T10.3). |
| Recommendation | Keep R-obj on the main line as a step, with "link 0" (from squared error back to KL) written down as a debt. R-local and R-constmetric belong to the family branch (card 6) and go with it. |

### Card 2. The switches (R-gate, R-selfgate, R-step, R-affine, R-lip, R-onecol)

| | |
|---|---|
| Meaning | In a real network every neuron has its own on/off switch (ReLU). The paper replaces that by ONE shared switch per layer and position. R-gate: the switch is computed by a separate path that the small network copies at full size. R-selfgate: the switch reads a fixed sum of its own layer, so the small network can compute it. R-affine: the "switch" is a straight line, not on/off. |
| Bought | With the switches fixed, the rest of the network is linear, so the best small network is a principal-component formula (T9, T11). |
| Lost | Per-neuron switches. This is the biggest gap between the theory and every real network. Proposition T1' shows why it matters: a mix of ReLU neurons is not one ReLU neuron. |
| Recommendation | R-gate stays DECIDED as a stepping stone. R-selfgate: keep, since it removes the full-size copy and so moves toward the trunk. R-affine, R-onecol, R-lip: side branch. An affine "switch" is not a switch, and these nodes do not move toward ReLU. The trunk-ward step on this axis is per-neuron switches with a proved bound (option B of the root). |

### Card 3. Edges, depth and size (R-pointwise, R-sgate, R-depth1, R-extra)

| | |
|---|---|
| Meaning | Real layers look at a small window of nearby positions and stack many layers. R-pointwise: each layer looks at one position only. R-sgate: windows allowed. R-depth1: one layer only. R-extra: "size k" means k units at the top layer, with more allowed below. |
| Bought | Pointwise: the fold is one matrix problem. Windows: exact at the "visible widths" (T13), which can stay large. |
| Lost | Depth 1 loses everything about how layers compose. R-extra changes what "size k" means. |
| Recommendation | R-sgate: keep (closer to the trunk than R-pointwise). R-depth1: side branch. R-extra is really a question about the trunk's words: does "size k" mean k units in every layer (as the trunk says now), or k at the top? Shonu decides. |

### Card 4. The input (R-gauss, R-centred, R-plane, R-sym, R-symN)

| | |
|---|---|
| Meaning | R-gauss: the input to the chain is random Gaussian noise with no link between nearby positions. R-plane, R-sym: images on an endless plane whose statistics do not change under rotation. |
| Bought | Gaussian: the part of the input that a small network does not keep is independent of what it keeps, so the best switch read has a formula (T17). Symmetry: harmonic (Fourier) forms. |
| Lost | Gaussian noise is never an image, and neighbouring positions in an image are strongly linked. Rotation symmetry is close to true for natural images on average, but never exact on a pixel grid. |
| Recommendation | R-gauss and R-centred: side branch. R-sym and R-symN are claims about images that can be near true. Keep them as candidates under option A. |

### Card 5. Short cuts in the data table (R-kron, R-pool, R-gram)

| | |
|---|---|
| Meaning | R-kron: a big table of co-activity splits exactly into two small tables. R-pool: max-pool replaced by an average. R-gram: co-activity equals the overlap of the weights. |
| Bought | Small closed forms in the family branch. |
| Lost | Max-pool is in InceptionV1. The other two are false in general. |
| Recommendation | Side branch, with the family branch (card 6). R-pool comes back on the main line later, because max-pool is part of the trunk's networks. |

### Card 6. One family instead of the whole network (R-family, R-fixedread, R-bench, S-datamath)

| | |
|---|---|
| Meaning | B.0 to B.3 replace one family (example: the 10 curve neurons) and keep the rest of the network. This was the goal before 2026-10-25. |
| Bought | The symmetric results: the orbit sum, the bumps, the contour pairing (T1a, T1b, T1c, T5, T3). |
| Lost | Under the trunk Shonu chose, a family is a part, not the whole network. |
| Recommendation | Mark R-family as a side branch that may become a rung. Its symmetry results are the best material we have for option A (a "clean" whole network is plausibly a network made of clean families). |

### Card 7. Structure claims (R-clean, R-read, R-circ, R-eq, R-joint, R-sumread, R-invread, R-contour, R-detector, R-shot, R-stat)

| | |
|---|---|
| Meaning | Claims that trained networks or images have a shape: families are rotated copies of one template (clean), readers treat them evenly, curves in images follow a smooth random law, the weights sit at a stationary point of training. |
| Bought | Everything with content in the family branch, and T10 in the chain branch. |
| Lost | Nothing is lost if the claim is near true. The open question is how near, which Part C measures. |
| Recommendation | Under option A these are the candidates for "the" property. The first job under A is one definition of "clean" for a whole network. Today H-eq (a whole chain that commutes with a symmetry) is the closest one. |

### Card 8. Settings of negative results (N-finite, N-onepos, N-ny2)

| | |
|---|---|
| Meaning | Counterexamples and hardness proofs use small, special inputs (a few points, one position, two classes). |
| Note | A hardness result about "the best build" carries over only when the allowed builds are the same, or through a stated reduction. T14's networks can be written as ReLU networks, but ReLU builds are a larger set, so their best value is a different number. The loss is also D2, not KL. So T14 and T15 say nothing proved about ReLU networks with KL. |
| Recommendation | No decision needed. Keep the containment question visible (section 2). |

### If Shonu accepts the recommendations

| Line | Results |
|---|---|
| Main line (steps toward the trunk) | T1', T12, then R-gate (T9, T6), then R-selfgate (T11, T11.1, T16), then R-sgate (T13) |
| Debts on the main line | per-neuron switches (ReLU), max-pool, KL (link 0), and the structure property of option A |
| Side branches (kept as mathematics) | the family branch (B.0 to B.3), the Gaussian switch formulas (B.7: T17 to T17.3, D8, D9), the counterexample settings |

## 10. Referee

Report and fixes: `REFEREE-DECISIONS-2026-10-25.md` (2026-10-25). All findings fixed except one tag edit in the paper
(H-sym in Conjecture D1 is redundant), which is listed there.
