# The noodle ladder: plain summary

Plain-words companion to `COARSE-THEORY.md`. The paper is the formal source. This file may quote toy numbers. The paper's
Parts A, B and D may not.

Last update: 2026-10-25 evening (revision 7 of the paper, Part B.7: formulas for the switches).

---

## 1. What "noodlable" means

A network is **noodlable** when you can:

1. Pick any size, down to one unit.
2. Get the best network of that size from a formula, with no training.
3. Switch between sizes instantly.

The formula needs one table of numbers measured on inputs (a covariance). It does not need the weights alone. We
proved that the weights alone cannot give that table in general (Fact F9).

## 2. What used to stay full size, and what is true now

| Part that did not shrink | Status now | Paper item | Cost |
|---|---|---|---|
| The switch path (gates) | **Folds.** The small network computes its own switches. | Theorem T11, Corollary T11.1 | The small network must keep the sums that the switches read ("forced reads", r of them). |
| The layers below the first family | **Folds** down to the raw input. | Theorem T13 | Each layer needs its "visible width". With spatial windows this can stay large. |
| Nothing at all, for any network | A **floor**: no network of any kind, with k units before a linear head, can beat the formula's target. | Theorem T12 | None. It needs no assumption about the network. |

## 3. The three building blocks

### Block 1. The floor (Theorem T12, no assumptions)

Take the big network's outputs. Find their k strongest directions (principal components). No small network whose last
step is "k numbers, then a linear head" can do better than keeping exactly those k directions. What comes below those
k numbers does not matter: ReLU, max-pool, any depth.

Two limits. The score here is the squared error of the outputs, not the true distillation loss (KL). And the head must
be linear on k numbers.

So for every size k we know the best possible score before we build anything.

### Block 2. Switches that read their own layer (Theorem T11, Corollary T11.1)

New class of network: each switch reads a fixed sum of the units below it (Definition D12). Example: "on when the
sum of all members is positive".

| Small-network shape | Result |
|---|---|
| Same width k at every layer, keeping the switch sums | Closed-form best network. Keep the switch sums, then add the top directions of what they leave unexplained. |
| r extra units below the top layer, k at the top | Reaches the floor exactly. This is the best network of any kind with k units at the top. |
| With rotation symmetry, and each switch reading one rotation-invariant sum | Only one switch sum is forced: the orbit sum (r = 1). So **one unit per layer, switches included,** is a real network. Symmetry alone is not enough: a switch that reads the size of a wave pattern needs r = 2. |

### Block 3. Spatial edges (Theorem T13)

Real lower layers mix nearby positions. Work down from the top:

1. The top needs k directions.
2. Those directions, seen through the spatial edges, need some directions of the layer below.
3. Repeat down to the input.

4. At each layer, keep only the directions that the layer actually uses on real inputs (the "reachable" part).

The result at each layer is the **visible width**. Folding to it is exact and reaches the floor for the top width.

The referee caught an error here. My first version skipped step 4, and its claim "at most k + r units per layer" was
false. With step 4 the claim is proved, for 1 x 1 edges. A second referee round checked the new proof and found it
correct.

| Edges | Visible width per layer | Smallest possible? |
|---|---|---|
| 1 x 1, separate switch path | exactly k | yes (proved) |
| 1 x 1, self-switching | between k and k + r | not proved; at uniform width, switch-changing builds can do better (T15) |
| spatial window | can grow toward the input | not proved; below it, deciding the best score is NP-hard in general (T14) |

## 3b. Where the exact formula stops (Part B.6)

Rule for this work (Shonu, 2026-10-25): never widen the scope without the optimal solution inside it. The two
conjectures D6 and D3b sat just outside the proved scope. Both are now settled, and both show where the formula ends.

| Direction | Inside the boundary (formula, every size a prefix) | One step outside |
|---|---|---|
| Spatial windows (was D3b) | Every layer at its visible width reaches the floor (T13). | Narrowing to width 1 at two layers joined by a window: deciding the best score is as hard as a known NP-hard tensor problem (T14). So, unless P = NP, no fast exact method works for every network. Networks with rotation symmetry are not covered by this and stay open (D3b). |
| Switches that change (was D6) | Keep the switch sums, plus r extra units below the top: reaches the floor (T11.1). | At the same width everywhere, changing the switches can beat the formula, and the best result can still miss the floor (T15, proved exactly on 4 points: floor 0, best 1/32, formula 1/16). Whether that best can be found fast is open (Conjecture D7). |

What this means for the rule. Inside the boundary we always have the optimum. Outside it, "the optimum" exists but
no general fast formula can find it (T14), or none is known (D7). So the scope we can widen while keeping exact
optimality is set by the size measure: count size at the top layer, and let the lower layers take their visible
widths (plus r units for the switches).

## 3c. Why the formula is beaten, and what the true optimum is (Theorem T16)

**The one fact.** A network's switches depend only on its **effective reads**: the directions in input space that the
switches look at. Fix those reads, and every switch is fixed on every input. Then the rest of the network is
straight-line math, and its best weights are a closed formula.

**So the true optimum is:** search over the reads, and use the formula for everything else. The search space is small:
the reads only (here 2 directions of 5 numbers each), not all the weights.

**The T11 formula is one point of that search:** it uses the teacher's own reads. It is beaten whenever other reads
score better.

**Why other reads can score better.** The loss splits into two parts:

| Part | Meaning | At the teacher's reads |
|---|---|---|
| Switch error | what wrong switches cost | zero |
| Width error | what the k units cannot carry | large: r of the k units must hold the switch inputs, even when those carry no output |

Other reads accept some switch error to cut the width error.

**What the checks show.**

| Check | Result |
|---|---|
| Exact instance (`model/t16_instance.py`), width 2 | formula 12737; best reads 1190; best fully trained network 1212; floor 349 |
| Exact instance, width 3 | formula 138.9; best reads 36.9; best fully trained network 41.5; floor 30.5 |
| Toy, width 1 (`toy2/c5_why.py`) | the trained network ignores the teacher's switches (correlation about 0). Its one unit carries the strongest output mode (50% of its in-span energy on mode 4, 0.9% on mode 0). 94% of its kernel reads input directions that the teacher's units do not use. |

So the formula did not fail because training found something mysterious. It failed because it insists on exact
switches, and at a fixed width that wastes units. Searching the reads finds the real optimum, and here it is at least as
good as full training.

**A side result (Corollary T16.1).** At the same width everywhere, extra depth adds nothing to what the network can
carry. An optimal network needs only one set of k input directions, switches that read inside them, and a head. Depth
only changes which switches are possible.

**Checked by a referee.** The referee found T16 mostly correct. It also caught a gap: one formula needed an invertible
covariance, and a read that switches everything off breaks that. It caught two overstatements too. All three are fixed in
the paper.

**What is still not solved.** The search over reads is not convex. On these small cases a multi-start search finds the
best value, and training never beat it. Whether it can always be done fast is Conjecture D7. For on/off switches and a
fixed input dimension, an exact finite method exists: enumerate the switch patterns.

## 3d. The best switch by formula, not by copy (Part B.7)

**The one fact.** A small network sees the input only through the units it keeps. So its switches can only use what
those units see. The best it can ever do is the big network's output **averaged over what it cannot see** (Lemma
T17a). This is a second floor, next to T12. It holds for any data.

**The formula for the switch (Theorem T17).** Assume the input to the chain is Gaussian (new named hypothesis
H-gauss), the switches are straight-line (affine), and the chain has one layer. Then:

| Part of the small network | Formula |
|---|---|
| Switch read | The big network's switch argument, **predicted from the kept units** by least squares. Not a copy. |
| Head | The big head, seen through the same prediction |
| Bias | Absorbs the average effect of what the kept units cannot see |

This build reaches the floor exactly, for every choice of kept units. So the inner search of T16 has a formula. The
copy of T11 is the special case in which the kept units happen to see the whole switch argument.

**Curved switches (Proposition T17.1).** The same build works, with an explicit price. The price depends on one number:
how much of the switch argument the kept units cannot see (sigma). The price goes to zero as sigma goes to zero.

**Which units to keep, at width 1 (Theorem T17.2).** The best single unit lies on one curve:

| End of the curve | What the unit does |
|---|---|
| One end | Copies the big network's switch read (what T11 does) |
| Other end | Carries the strongest output direction (what T9 does) |
| The best | Usually strictly between. One number mu sets where. mu is a root of one polynomial, so there are only a few candidates. |

**Larger widths (Proposition T17.3).** The best kept units satisfy a "self-consistent eigenproblem": they are
eigenvectors of a matrix that depends on them. Sharper (part 4, added after the referee): the best kept space is one
"read-carrying" direction p plus k - 1 strong output directions away from it. So width k is a search over one vector p,
with an eigenvalue problem inside. Whether a fast method always finds the best p is Conjecture D8.

**Any data, any switches (Proposition T18, the subset fold).** Keep some switch reads exactly. Let the others read
their best guess inside the kept units. Try every subset. Never worse than the copy (T11), never better than the floor
(T12).

**What the checks show.**

| Check | Result |
|---|---|
| Exact Gaussian instance (`model/t17_instance.py`), formula switch | residual orthogonal to the kept units to 1e-17. No random or optimised switch read beats it. |
| Width 1 | best root of the polynomial = brute-force best (7693). Copy: 6242. Strongest output direction: 5051. |
| Width 2 | the best space commutes with its matrix (2e-8). The fixed-point iteration found the best value from all 10 starts. |
| Softplus switch | price below the bound at every angle, and 0 when the kept units see the switch argument |
| Symbolic algebra (`model/t17_sympy.py`) | all identities give 0 |
| T16 instance, subset fold (`model/t18_instance.py`), width 2 | copy 12737, **subset fold 9231**, best search 1190, floor 349 |
| T16 instance, width 3 | copy 138.9, subset fold 138.9 (the copy wins), best search 36.9, floor 30.5 |

**Honest reading.** The switch formula is exact where its hypothesis holds (Gaussian input, affine switches, one layer).
On the T16 instance (curved switches, two layers) the subset fold closes only part of the gap. The reason is now clear
from T17.2: the switch read is easy once the kept units are fixed. The hard part is choosing **which units to keep**. The
subset fold only tries the T11 family of kept units, and T17.2 shows that the best kept units lie off that family.

**Where hardness can still sit.** With affine switches and Gaussian input, width 1 is not hard (finitely many
candidates). So any hardness for affine switches needs data outside that theorem's premises (non-Gaussian data, a bias entry, width 2 or
more, or two layers), and only once the root-finding is written out as an algorithm. Conjecture D9 (new) is one such
case: data made of finitely many points, plain linear switches.

**Checked by a referee, twice.** Round 1 (`REFEREE-B7-2026-10-25.md`) found every proof correct and many overclaims in
the comments. The biggest: I had called one end of the curve "the T9 choice", and that was wrong. Round 2
(`REFEREE-B7-2026-10-25-r2.md`) checked the fixes and the new one-vector result. It found one real error, in the route
of D9: the reduction went the wrong way (it would give co-NP-hardness, not NP-hardness). All findings are fixed in the
paper.

## 4. Instant switching

All sizes nest. So we store one network, best directions first, and size k uses the first rows and columns at each
layer. Switching size changes the prefix lengths and one bias vector. Nothing is copied.

| Where it holds | Paper item |
|---|---|
| Gate path kept | Theorem T6 |
| Self-gated, uniform width | Theorem T11 part 4 |
| Spatial edges | Theorem T13 part 3 |

Checked in the exact instances: running the stored largest network with only its first units gives the small
network's value to rounding error.

**Cost of the switch formula (B.7).** The formula switch read depends on which units are kept at each width. So the switch
reads become one vector per width, like the bias, not a prefix. At width 2 or more the kept units themselves may also
change with the width. So the B.7 builds are not claimed to be prefixes of one stored network.

**Setup cost and parallel work.** Switching size is instant (a prefix). The one-time setup is not instant: it is one
eigenvalue problem on a covariance table. With rotation symmetry that problem splits into floor(N/2) + 1 separate
scalar problems, one per wave (Theorem T10 part 2), and they can run in parallel. Nothing more about parallel cost is
claimed. Literal constant time for the setup is impossible: the answer depends on every entry of the table, so the
computer must at least read them all.

## 5. Honest costs (what the toys showed)

| Finding | Where |
|---|---|
| If the orbit sum is a weak direction, forcing it costs a lot at uniform width. In the toy, mode 0 was the weakest of 8 modes. The width-1 self-contained network scored 641600 against the floor's 318661. | toy Y12 |
| One extra unit below the top removes that whole cost. The build hit the floor to 1e-15. | toy Y12b |
| At uniform width, networks that are free to change their switch reads beat the switch-keeping formula by 11% to 25%. They never beat the floor. So the formula is the best switch-keeping network, not the best network overall at that width. | toy Y13, Conjecture D6 (now stated as: for some chain, the best score lies strictly between the floor and the formula) |
| With a 3-position window and random weights, the bottom layer needed its full width even for one unit at the top. The fold is exact but does not shrink the lower layers there. | `model/t13_instance.py` |

## 6. The training wheels, updated

| Wheel | Status |
|---|---|
| Separate gate path kept at full size | **Removed** (H-selfgate, T11). |
| Only 1 x 1 edges, nothing below the first family | **Removed** at the visible widths (H-sgate, T13). Smaller widths: NP-hard to decide in general (T14); symmetric networks open (D3b). |
| Each unit is its own switch (ReLU) | **Still on.** Written as a self-gated layer, a ReLU layer forces every unit's read, so nothing folds in that class. The floor (T12) still applies. |
| Max-pool | Still on. |
| Squared logit error instead of KL | Still on (bounds only: Corollaries T10.2, T10.3). |
| Data table (covariance) needed | Stays. Fact F9 says it must. |
| Symmetry | Extra, not needed. It only gives the harmonic form and r = 1. |

## 7. Next rungs

0. **Which units to keep (after B.7).** The switch read now has a formula. The open part is the choice of kept units:
   prove Conjecture D8 (width 2 and more, Gaussian input), and carry the resolvent curve of T17.2 to curved switches and
   two layers, then test it on the T16 instance against the best search.
1. **ReLU.** Bound how far a ReLU network is from a self-gated one, then use T11 plus a stability bound. Inside the
   boundary of section 3b only.
2. **Conjecture D7.** Prove that the best switch-changing network is hard to find (the reduction from single-ReLU
   training needs a teacher that can output any labels).
3. **Structure that moves the boundary.** T14 is about every network. Symmetric or "clean" networks may stay solvable
   below the visible widths. This is the open part of D3b.

---

## 8. Back to the trunk (2026-10-25, late evening)

Shonu fixed the goal (the **trunk**): the best whole network of every size, built from any new units, scored by KL
on the image mix, with instant switching. Every rung above is a step away from it. `DECISIONS.md` now names each step
(a **node**), says who chose it, and gives every result its distance from the trunk. A picture of the tree is in
`DECISIONS.md` section 8.

| Distance | Results |
|---|---|
| 0 | Proposition T1' |
| 1 | Theorem T12 (the floor) |
| 3-5 | the fold T9, T6; the self-gated fold T16, T11, T11.1; spatial windows T13; hardness T14 |
| 8-9 | the switch formulas of B.7 (T17, T17.1, T17.2, T17.3, D8). This is where revision 7 stopped |
| 6-14 | the family branch (B.0 to B.3), for example T0, T1a, T5, T2, T4 |

So the last rung (section 7 item 0, "which units to keep") was eight steps from the goal. A referee
(2026-10-25) found 26 results whose paths missed a premise. Those are fixed, and the distances above include them. Shonu did not decide most of those
steps one by one. They came inside plans approved as a whole. The next decision is the root: an exact fast rule for every network is likely out of reach, so
"exact", "fast" or "every network" must give (`DECISIONS.md` section 3). Until Shonu decides, no new rung starts.

## 9. Working at the trunk itself (2026-10-25, late night)

Shonu: the root options of section 8 were guesses. Push the general goal until a theorem forces a choice.
Part B.T of the paper does that. It asks three questions of the goal's own words, with no assumption about the network.

| Question | Answer | Result |
|---|---|---|
| Does a best small network exist? | Not always. With three classes, small networks can copy a 5-unit network better and better without end, but none of width 2 copies it exactly (example of Lim, Michalek and Qi). With two classes and one hidden layer: yes | Theorem T23 (proved conflict), Theorem T19, Conjecture D17 |
| Can the best networks of every size sit inside one stored network? | No, not for every network. A teacher with 2 units on 7 points already breaks it | Theorem T20 (proved conflict) |
| Can a quick calculation find the best network for every network? | No, unless P = NP. Finding the best 1-unit network is as hard as finding a clique in a graph | Theorem T22 (proved conflict) |
| Can a slow exact calculation find it, with no training? | For two classes, one hidden layer and the squared-error score: yes, cost grows like n^{d k}. For KL the problem splits into finitely many convex problems | Theorem T21 |

The example of T20, in one picture. The big network's output is a bathtub on the points -3 to 3: high at both ends,
flat in the middle. Two units fit it exactly, with kinks at -1 and 1. The best single unit must cover both sides at
once, so it puts its kink somewhere else. No unit of the best 2-unit network is the best 1-unit network.

The three choices now on the table (`DECISIONS.md` section 3) come from these theorems. To get past a theorem, one of
its premises must go.

| Decision | Options |
|---|---|
| R-least (from T23) | L1 accept "within eps of the best"; L2 cap the size of the weights; L3 allow limits of networks; L4 name a property under which the best exists |
| R-nest (from T20) | N1 drop nesting; N2 keep nesting and accept "good, not best" at smaller sizes; N3 name a property of networks under which the best builds nest |
| R-root (from T22) | A name a property of networks; B name a property of image mixes; C accept a proved error bound; D accept a slow exact rule run once per model |

The hard cases of T22 are artificial: a teacher that memorises random labels on made-up points, one unit per point.
Options A and B both say "real networks trained on real images are not like that". That would need a definition first.

## 10. The new trunk, and the first fork under it (2026-10-25, late night)

Shonu answered the three forks of section 9 and confirmed a new trunk T\*\* (`DECISIONS.md` section 1):

| Part | New trunk |
|---|---|
| Input | the network N, a goal G (which classes matter, how much), sizes k_1, ..., k_L (one per layer) |
| No data | the rule reads only N and G |
| Output | the best network for G at those sizes |
| Switching | one computation per goal up front, then instant size changes |
| Exactness | exact closed form when N is perfectly clean (eps(N) = 0); a bound in eps(N) otherwise |

First result under it (Part B.C, Theorem T24): "best on the data the network was trained on" cannot be computed from
the network alone. Example: the bathtub network of T20 is perfectly clean (its two units are mirror images). It is the
best possible predictor on any input list that it labels itself. But for two such lists, even two mirror-symmetric
ones, the best 1-unit network differs (kink at 14/11 for one, a unit that fires only at the end point for the other).

So the score must be chosen (node R-score):

| Option | "Best" means |
|---|---|
| (i) intrinsic | best under a score built from N by a fixed rule (not yet defined) |
| (ii) training data, stronger property | best on the training data, with a property (for example "trained by SGD") that makes the weights carry what is needed |
| (iii) worst case | best on the worst single input (what the worst case over all consistent data reduces to) |

Also proposed for review: a definition of a perfectly clean network (families = moved copies of one template under a
symmetry group, read the same way; Definition D20) and a cleanliness distance computed from the weights (D21).

## 11. The mix is back, and the zoom (2026-10-25, late night)

Shonu reversed the no-data rule (a mix is an input again: the generalist mix and a developer's mix are both just
mixes) and described the product as a **zoom**: from a sample, fold up some neurons and keep others fine. Trunk
T\*\*\* records this (`DECISIONS.md` section 1).

| Product idea | Math object (Part B.Z) |
|---|---|
| a developer's sample | the mix P |
| fold up neurons | a fold: a new unit that reads only a group of the network's own pre-activations (D22) |
| which neurons fold | a zoom: a tree over the units, with a stored fold at each node (D23) |
| a hardware change | moving the cut through the tree; the cost does not grow with the network |

Two results. T20's bathtub is a zoom (its best 1-unit build is a fold of its two units, mixed 5 to 1), so it does not
break the zoom. But Theorem T25: if the trunk allows any new units, the best build can read an input feature that the
network ignores, when that feature happens to carry the answer in the sample. That is over-fitting to the sample.
Fold builds cannot do it. So "any new units" against "folds only" is the decision now on the table (node R-fold).

A referee showed the fold must be strict to work: a non-negative mix of the network's own pre-activations, with no new
offset. Otherwise a fold of a unit that is silent on the sample can still read the accidental feature. The strict
fold matches the essay's coarse unit (a sum of detectors), and T20's best 1-unit build is such a fold (13 to 3 for
squared error).

## 12. Why folds, from the project's goals (2026-10-25, late night)

Shonu asked for the case for "folds only" to come from the project's goals. Each goal became a condition; a referee
then checked what each one really forces (Corollary T30).

| Goal | Condition | What it forces |
|---|---|---|
| "The training has already been done" | the answer must depend on the network's weights, not only its answers (G1) | Rules out "any new units": that answer depends only on the network's answers on the sample, so it is distillation (T26). Does not single out folds. |
| "Download once, fold many times" | a unit adds no distinction the network does not make (G2) | Weak: no restriction when the first layer reads every input direction (T28). |
| "Fold up what exists" | a coarse unit fires only where one of its fine units fires (G3) | Exactly the folds (T29, Farkas). The case rests on this one. |

Also: with any new units, a small sample (at most d + 1 images in general position) gets a perfect copy of every
network from its answers alone. The exact fold rule is cheap for small groups (families) and as hard as the general
problem for large groups. Not shown: that folds lose nothing for clean networks.

## 13. What distillation converges to, for clean networks (2026-10-25, late night)

Shonu reframed the project: distillation trained to the end converges to the best small network on the mix (the
trunk, with any new units). The conjecture was: for clean networks that network is a fold of the network's families.
Shonu then asked for proofs, not computations, until a theorem forces a choice. Result: three exact theorems, each an
exhaustive exact computer-assisted proof (Lemma T32a), on tiny perfectly clean networks with symmetric samples.

| Theorem | Network | Width | The best build is not... | ...because it |
|---|---|---|---|---|
| T32 | one family of four, plane | 2 | covering folds (squared error 3/80 against 1/24) | fires where every fine unit is silent |
| T33 | one family of four, 3-D | 1 | broad folds (0 against 1/468) | reads the family's negative side |
| T34 | two families of four, 3-D | 1 | signed folds of one family (25/216 against 7/60) | mixes the two families |

T33 and T34 also hold for KL, the trunk's own score (T34's KL bound is certified with interval arithmetic). A referee
then showed that T34's unit is a covering fold of a group that crosses the two families; folds of mixed groups with free
thresholds survive but say nothing, since they include every unit when the layer's weights point in all directions.

Corollary T35 (an existence statement): least builds need not be folds of one family, nor covering folds of any group.
So with "clean" meaning symmetric families, distillation does not converge to folds of the families. That is the fork
(`DECISIONS.md` 3-3, R-foldopt): (a) a stronger "clean" (candidate: each family's coarse unit is already in the
network, the essay's "hanging-ears neuron"); (b) folds as a restriction, with the losses above; (c) drop the fold
conjecture and characterise the optimum directly.

## 14. What the optimum is, for clean networks (2026-10-25, late night)

Shonu asked what the optimum actually is, in math and in plain meaning. Paper section B.O.

| Step | Result |
|---|---|
| Symmetry | Lemma T36: turned copies of a best build are best. D29 (best unit on a mirror axis) and D30 (best unit fires where the family is silent) have no numerical support on ring examples (not proved false). T20 proves exactly that the optimum can break the symmetry. |
| Structure | Theorem T38 (a necessary condition): each coarse unit's straight-line piece is the best fit, on its own half-space, of what the other units leave unexplained. With one unit (T38.1): on its half-space the build is the best straight-line fit of the big network, elsewhere its average. Checked exactly on the builds of T20, T32, T33, T34. |
| Deep networks | Corollary T38.2: the target is the family's output seen through the unit's own outgoing weights. Corollary T38.3: if later layers look at neighbouring positions, the condition weighs the errors there too. |
| Real check | H-dialfit on the trained 3b coarse unit: on at 83/100 of positions (a dial), but its pre-activation correlates only 41/100 with the position-local fit. Not at T38.2's fixed point. The likely reason is T38.3 (the readers of 3b reach neighbours), which matches the log's "members moved to nearby positions". |

Meaning (one unit): a coarse unit is a dial (fires on at least half the mix: a straight-line summary there) or a
detector (fires on less: a summary on a small part, the average elsewhere). A referee (report 17) caught that this
reading is false for two or more units unless their regions are nested or apart. The open part is which regions the units take. That is the NP-hard part (T22).

## 15. A gate in front of the family (2026-10-26)

Shonu confirmed his meaning of fold: the network's own gate decides where the coarse unit is on, and an easy formula
decides what it computes there. The tiny networks of section 13 had no gate, so this was untested. Paper section B.G
adds the smallest network that can have one: two hidden layers, a gate unit in layer 1, a family of four in layer 2
that is silent wherever the gate is silent. Toy networks only (Shonu's rule).

| Result | What it says |
|---|---|
| Lemma T39 | If the coarse unit's region is the gate's region, the best such unit is one least-squares fit (one logistic fit for KL) on that region. No search, no training. |
| Corollary T39.1 | If no member bends inside the gate (each one is a straight line there), that unit copies the big network exactly. Its weights are the big network's own: the sum of the members that are on inside the gate, switched by the gate. |
| Theorem T40 | If the members bend inside the gate, the gate's region is not the best region. Exact: every gate-region unit costs at least 7/182; a unit that bends inside the gate costs 5/182. |
| Conjecture D34 | "With a gate in front, the best unit takes the gate's region." False (T40 exactly; the search found the same on all 8 cases tried). |

Meaning: what is proved is one good case and one bad case. Good: members that are straight lines inside the gate
(the essay's "gate = region, value = distinction inside it") give an exact build from the weights, though not the only
one (a referee found an exact build that ignores the gate). Bad: on one toy where the members bend inside the gate,
the best single unit bends where the network bends, copies one opposite pair of members, puts a constant on the rest,
and breaks the symmetry. Which property decides the case in general is open.
Correction (Shonu, same day): the error that matters is the gap between the rule's small network and the best small
network of the same size, not the gap to the big network. Under that reading T39.1 is the easy case: the best small
network there pays no cost at all, so it does not test the rule. On the T40 toy the gate rule misses the best small
network by 1/91 (7/182 against 5/182). No result yet gives a rule that hits the best small network when that network
pays a cost.

Next statements: several gates (width = number of gates); three or more logits (one family may need several coarse
units inside one gate); nested gates (a gate in front of gates, so a smaller size means a coarser gate).

## 16. The shape of the optimum (2026-10-26)

Shonu asked for one mathematical object that says what the optimal small network is, so that later results become
special cases of it. Paper section B.S.

The object, in four steps:

1. Take every small network of the given sizes. Run it on the mix. Write down its outputs (one row per input).
2. All these output tables together form one shape: the output set (Definition D35).
3. The optimal small network is the point of that shape nearest to the big network's outputs (Definition D36).
4. The shape is a union of finitely many pieces. A piece is fixed by which units are on at which inputs (Definition
   D37, Theorem T41). For one hidden layer, each piece is a simple convex region, and finding its best point is one
   easy calculation (Corollary T41.1).

What this means for "error" (the rule's network against the best small network): a rule's error comes only from
picking the wrong piece (T41 part 4). Once the piece is right, the rest is one convex calculation. So the whole
question is: which piece?

One general bound (Theorem T42, proved, also in Lean): if the big network's outputs move by some distance, the
distance from them to the best small network moves by at most that distance. So the best small network for a nearby
clean network is close to optimal. In the paper's error (a difference of losses) the bound is
2 eps delta + 2 eps^2: zero when eps is zero, but it also grows with delta, how far the best small network is from
the big one. So it is not yet the trunk's "bound that depends only on eps". It also needs the clean network to have
the same sizes as N: a referee sketch suggests that with any size allowed, eps would be zero for almost every network,
and the bound would say nothing. This eps is a proposal for Shonu (node R-cleannet), not a decision. Checked on 80
random toy cases: the real excess was at most two thirds of what the bound allows.

Earlier results in this language (T9 is an analogue in the gated-chain class, not a special case):

| Result | Reading |
|---|---|
| T9 (gated chains, an analogue) | the gates are fixed; the output set is all matrices of rank at most k; the best point is the principal-direction truncation |
| T19, T21 | one hidden layer: the pieces are convex, so a best network exists and an exact rule exists |
| T22 | picking the piece is NP-hard for general networks |
| T23 | with three outputs the shape has holes in its edge, so a best network may not exist; a best piece still does |
| T20 | the best pieces at sizes 1 and 2 DO nest, but the best networks inside them do not (the kink moves from 8/5 to 1) |
| T38 | the condition at a best network (which sits in a best piece) |
| T39 | the gate fixes the piece; the best point is one fit |
| T40 | the gate's piece is not the best piece |

What it shows on the T40 toy: the best one-unit network is two points in output space (one network and its quarter
turn), reached from 56 pieces. So the optimum is cleaner in output space than in weight space.

What comes next, as special cases: for a perfectly clean network, a formula that names the best piece from the
weights. That is the open part of the trunk.

## 17. What a clean network is, and what it cannot fix (2026-10-26, evening)

Shonu's direction: the optimum itself is the object. Folds, gates and families are hypotheses about it, never
methods we must use. Gates are parked. The goal of the session: "a good version of what a clean network is". Clean
must use the network alone (no mix), may need several numbers, and is an object with several principles.

**The principles (paper A.5, Definition D39).** Each is motivated by a sentence of the essay. A family is now fixed by
the network itself: the units that the network's own symmetries move into each other.

| Principle | Plain meaning | Essay |
|---|---|---|
| Q1 moved copies | the network has a family: one unit in several turned or moved copies | rotated copies of curve units |
| Q2 siblings are alternatives | strict: two members of one family are never on together; broad: each member is the strongest at its own peak | "an ambulance neuron might have a taxi on its negative pole" |
| Q3 votes agree | a reader never votes for and against two units that are on together | "a neuron is built from votes" |
| Q4 a gate in front | a unit sits inside the region of a unit before it (a real gate, not one that is on everywhere) | "a neuron in front of them acts as a gate" |
| Q5 the coarse unit is present | one unit after a family is on wherever a member is on | "ear neurons all feed one hanging-ears neuron" |

What they do to the old toy networks (each failure has an explicit point where two units are on together):

| Network | Verdict |
|---|---|
| T20 | clean |
| the broken lines behind T22 and T23 | not clean: the reader votes for and against units that are on together (Q3) |
| T32, T33, T34 | not clean under strict Q2 (two turned members are on together); T32 is clean under broad Q2 |

So strict Q2 drops the tiny networks you suspected "we don't really care about". But strict Q2 may be too strong: real
curve detectors overlap with their neighbours. Strict or broad is your call.

**What clean cannot fix (Theorem T43).** A family of networks passes the principles (all of them at one hidden layer;
all but Q4 at more layers) and still keeps T22's hardness. In each of several input blocks sit narrow direction
detectors, never on together, all read by one positive reader. The mix holds each detector's best input plus blends
of some pairs. A single coarse unit that fires on two best inputs also fires on their blend, so the best coarse unit
must pick detectors with no bad blend between them. That is the clique problem. The network does not depend on the
graph at all. Only the mix does.

What this means, exactly. The trunk allows any one-time computation per mix (the zoom). T43 says that, unless P = NP,
this zoom cannot run in polynomial time for these clean networks. It is a conflict only if you want the zoom to be
polynomial-time.

Two honest limits. The detector networks are very sharp (their weights grow like m^4 for m detectors per block). And
they are clean only on a ball-shaped input domain, not on a box. A principle that limits sharpness might rule them
out. Whether the hardness survives that is open, and it is the most interesting next question.

**The decision this forces (card 3-5, R-mixclean).** First: (e) is a time limit on the zoom part of the goal? If yes:
(a) a property of the mix, (b) a weaker meaning of "polynomial", (c) give up "exact", or (d) a principle these
networks fail, such as bounded sharpness. Recommendation: answer (e) first; if yes, try (d) with bounded sharpness.

**Still open in the definition.** Strict or broad Q2. What a family is when there is no exact symmetry (the ear
units). Which input domain belongs to a network's kind. The cleanliness numbers for Q1, Q4 and Q5.

## 18. Where the hard puzzle hides, and what "listening" should mean (2026-10-26, night)

The picture used in this section: the network is a building of workers. Each worker lights up when it sees its thing.
Workers upstairs listen to workers downstairs. The mix is an exam for the small network: photos, with points each.

**The exam in the network's own words (Definition D48, card 3-8).** Shonu's idea: strange exams may be hard, sensible
ones should not be. Proposal: the user does not list photos. The user says how much each worker's job counts, plus
how much "nothing here" counts. The photos inside each job come from the photos the network was built on.

**What I found (Lemma T51, Corollary T51.1).** This alone does not remove the hard puzzle of T43. That hard exam is
already an exam of this kind when its own photos are the base photos. Every mash-up lights up no worker, so all
mash-ups sit in "nothing here". The user can only say how much "nothing here" counts. Which mash-ups are in the
photos is fixed by the photos, and the puzzle lives there.

So the puzzle moves to the photos where the big network says "nothing". The big network never had to notice which
mash-ups were among them, because its answer on all of them is the same. The small network must notice.

What is proved: if whoever poses the problem may also choose the base photos, it stays hard (T51.1). If the base
photos are fixed by the network (the photos it was really built on), the old proof says nothing, and the question is
open: the hopeful case. A puzzle hidden inside the network instead, as workers for single mash-ups, makes the network
unclean (Lemma T53: nobody hears those workers, or one is heard backwards).

**Listening (Definition D49, Lemma T52, card 3-9).** Two kinds of sensible listener: teams (inputs light up together:
wheel, window, headlight for "car") and alternatives (inputs take turns: curves at different angles for "some
curve"). The strict test of card 3-7 punishes alternatives. Two fixes fail for one shared reason: when one
alternative is on, the others are silent, so "do the others agree right now?" gets no evidence from them.

| Fix | Alternatives | Random wiring (exact check, 400 units) |
|---|---|---|
| does this worker decide the listener? | full credit (1) | small but never below 0, for every unit |
| do the inputs that are on agree? | exactly 0 | a coin flip |

Next idea: score teams (on together more than chance) and alternatives (on together less than chance) both, against
a random-wiring baseline.

**Decisions waiting for Shonu.** Card 3-8: exam in the network's words, and what to ask of the "nothing" photos.
Card 3-9: allow teams and alternatives; accept a graded listening test.

## 19. The developer's zoom tree, and one test per kind of group (2026-10-27)

Shonu's new trunk (T4): the developer says which neurons go coarse, how many neurons each group becomes, and in what
order the groups degrade (a zoom tree). The rule computes the best small network at every level of the tree.

| Word | What it is in the network |
|---|---|
| group | a set of neurons in one layer that the developer wants replaced by fewer new neurons |
| pre-activation | a neuron's weighted sum before ReLU. ON when it is above 0 |
| class (b) | each new neuron reads only its own group's pre-activations (the choice written into T4) |
| class (r) | step by step: a coarser neuron reads only the coarse neurons of the level below |
| AND group | its members are ON together or OFF together on every photo of the mix |
| OR group | at most one member is ON on any photo |

**Step 1. Three ways to build a coarse neuron, and their order (Theorem T54, proved).**

```
level 0 (N):   c0 c36 c72 ... c324             10 curve neurons
level 1:       [c0 c36 c72] [...] [...]         3 coarse neurons
level 2:       [ the 3 level-1 neurons ]        1 coarser neuron

loss:  (a) reads anything  <=  (b) reads the 10 originals  <=  (r) reads the 3 level-1 neurons
```

Why it matters: each class can do everything the next one can. So "is step by step optimal?" means "does (r) lose
nothing against (b)?" (Hypothesis H-recur), and "does the developer's class lose anything?" means "(b) against (a)"
(Hypothesis H-group).

**Step 2. One step is cheap when it merges few neurons (Theorem T55, proved).**

```
one step:  children p1 p2  ->  new neuron relu(m1 p1 + m2 p2 + beta)
pieces:    one per ON/OFF pattern of the new neuron on the photos; few patterns when few children
```

The number of pieces grows like (number of photos) to the power (number of children x new neurons). Each piece is one
convex fit. Limit: each step's weights feed the next step, and nothing is proved yet about how long those numbers get
over many steps.

Why it matters: this is the computational payoff of Shonu's recursive idea. Small merges are polynomial. One big
merge is not (next step).

**Step 3. One big group handed over in one step is still NP-hard (Proposition T57, proved).**

```
detector network (T43):  detectors A B C D E F, one reader
developer:               group = {A B C D E F} -> 1 new neuron
the rule must still pick which detectors the neuron covers  ->  the clique puzzle
```

Why it matters: this corrects what I said before (that a given grouping makes T43 easy). It holds only when the
developer names exactly the detectors to cover. A big group in one step keeps the puzzle.

**Step 4. An AND group collapses exactly, from the weights alone (Theorem T58, proved, Lean).**

```
members on together or off together:   relu(z1), relu(z2), relu(z3) all pass through, or all output 0
so  a1 relu(z1) + a2 relu(z2) + a3 relu(z3)  =  relu(a1 z1 + a2 z2 + a3 z3)     when a1, a2, a3 >= 0
```

Votes of both signs need two new neurons (the positive votes and the negative votes). In general: rank + 1 neurons.
The check `model/tr_and.py` confirms it in exact fractions on 40 random networks, one with 2 hidden layers, and a
control where a non-AND group breaks it.

Why it matters: for AND groups the rule is plain linear algebra. No search, and the mix matters only through where the
members are on together.

**Step 5. The general test, and why OR groups pay (Proposition T59 and Corollary T59.1, proved).**

```
OR pair A, B:   photo a: A ON         photo b: B ON         blend photo (a+b)/2: both OFF, answer "nothing"
one new neuron ON at a and at b  ->  its pre-activation is affine, so it is ON at the blend too  ->  wrong there
```

Whether a group merges exactly into one neuron is one linear program. AND groups with one-signed votes always pass.
An OR group fails when the mix holds a blend of members' photos where the group is quieter (T59 part 2, for any output
weight and offset). On the detector network this is exact both ways: two detectors merge with no loss exactly when
their blend is not in the mix (T59.1 bad pairs fail, T59.2 good pairs pass, both proved).

Why it matters: this is the classifier Shonu asked for, written as a test a computer can run. It also shows where the
hard part sits: in which blends the mix contains.

**Step 6. Is step by step optimal? A first check (Conjecture D52, numeric).**

On the smallest detector network (4 detectors, 8 photos), `model/tr_tree_search.py` tried all 18 merge orders:

| Merge order | Result |
|---|---|
| starts with a good pair (A+D or B+C: blend not in the mix) | first merge costs nothing (proved: T59.2); reaches the optimum on every tie branch |
| starts with a bad pair (blend in the mix) | first merge costs one photo; reaches the optimum only if the tie is broken the right way |
| all 18 orders | each has some step-by-step path to the optimum |

Why it matters: H-recur holds on this instance, but the safe trees are the ones that merge exact pairs first. Finding
such pairs in general is the clique puzzle again. That matches Shonu's decision: the developer chooses the tree.

These numbers are a toy check, so the paper keeps them in Part C (H-recur row), not in the conjecture.

**Step 7. Which groups one neuron replaces for free: the cliques (Corollary T59.3, proved).**

```
detectors A B C D, mix of T43:    free pairs (blend not in the mix): A+D, B+C
group with no costly pair  ->  one neuron  relu(c' (z_A + z_D) + const)  equals the group on every photo
group with a costly pair   ->  no neuron of the group's class reaches zero loss
```

The proof is short because the sum of the members' pre-activations is linear in the photo. So its value at a blend
photo is the average of its values at the two photos.

Why it matters: on these networks, "which groups merge for free" is exactly "which sets are cliques". That is the hard
puzzle of T43, now with a precise place in the zoom.

**Step 8. When step by step loses nothing (Lemma T60 and Corollary T60.1, proved).**

```
child 1: coarse neuron with weights m1 on its members     child 2: weights m2
parent's best direct neuron: weights (w1 on child 1's members, w2 on child 2's members)
step by step can build it  <=>  w1 points along m1 and w2 along m2
a direction a child dropped never comes back at a later level
```

On the detector networks with T43's mixes, when every target group is free (a clique), every tree reaches the optimum
at every level.

Why it matters: this proves your recursive idea for the free case, along every tree. The open case is target groups
that must pay. There a merge must choose which members to keep, and part 3 says the dropped direction does not return.

**Step 9. Every level, not only the top (numeric).**

The tree search now compares step by step with the best direct build at every level. 16 of the 18 merge orders have
a branch that is optimal at every level. The 2 orders that pass through the grouping {A+C, B+D} lose at level 2 on
every branch. Proposition T61 (proved, exact) shows why:

```
best build at {A+C, B+D}:  relu(-20 z_A - 40 z_C - 22.5) + relu(-40 z_B - 20 z_D - 22.5) - 18   = N on every photo
each new neuron: ON on most photos, OFF toward its own members  ->  not a detector; only the pair works
step by step:    first fits A+C alone (B, D still separate) -> a different neuron -> copied -> level 2 cannot reach 0
```

Why it matters: the best neurons at one level need not extend the best neurons of the level before. Step by step with
copying can lose even when a perfect build exists. A variant that refits every group at every level avoids this, at
a higher cost.

**What the referee changed.** H-group first said "some grouping", which is always true with one big group, so it now
says "the grouping into N's families". It has content only when a family is low-rank, which is the real case. H-recur
now starts from N and merges two neurons per step (before, an empty zoom made it trivially true). The OR conjecture
first said the coarse neuron fires on all members' regions; T20 refutes that. Round 2 found that "one member's region"
also fails when the mix has no blend photo between members. So it now needs that blend photo, the whole layer as one
group, and the squared-error objective. H-recur now only claims targets whose groups each become one neuron.

**Open.** OR steps: a closed form for the least cost (Conjecture D53). Deeper layers: what "reads only its group"
means when the neurons below are also new (Remark B.W.3). H-group: is (b) as good as (a) for strictly clean networks?

## 20. The neurons that turn off for their own members (2026-10-27, evening)

Shonu asked what the T61 neurons are, and whether they mean coarsening is the wrong frame.

| Word | What it is in the network |
|---|---|
| presence unit | a new neuron that is ON only on photos where one of its group's members is ON (a coarse detector) |
| dial | a neuron ON on at least half of the weighted photos; its value, not its ON/OFF, carries the information |
| split unit | ON at one member's photos and OFF at another member's photos: the two members sit at its two ends |
| silent blend | the midpoint of two photos, in the mix, where N's answer is below the average of the two ends |
| switch | a neuron switches across two photos if it is ON at one and OFF at the other |

**Step 1. The rule that explains them: every silent blend must be paid by a switch (Lemma T62a, Corollary T62b,
proved, Lean).**

```
photos x and x', their blend in the middle:
   answer at the blend  -  average of the answers at x and x'   =   sum over neurons that SWITCH between x and x'
                                                                   of  -(output weight) x (smaller pre-activation) / 2
a neuron ON at both photos, or OFF at both, adds nothing
```

Why it matters: a dip at a blend can only come from a neuron with a positive output weight that turns OFF on one side.
This one identity contains Proposition T59 (one neuron ON at both ends cannot dip) and explains T61.

**Step 2. Why the family grouping needs split dials (Theorem T62, proved).**

```
conflicting segments of T61's instance form a cycle:   A --- B
                                                         |     |
                                                         C --- D
each neuron's sign flips an even number of times around the cycle, and one neuron cannot flip on all four sides
-> each of the two new neurons pays for exactly two sides
-> a neuron that fired only on its own members could not pay without breaking the answer at B or D
```

Result: in every perfect build at {A+C, B+D}, each new neuron fires at exactly one of its two members, is OFF at the
other, and fires on photos where none of its members fire. T61's neurons are split dials: the {A, C} neuron is ON at A,
OFF at C, highest at B (a member of the other group), and ON at three of the four blend photos.

Why it matters: these neurons are forced by the mix, not an accident of the solver.

**Step 3. What every grouping of the smallest instance uses (exploration, `model/tr_poles.py`).**

| Grouping | Kind of groups | Best build |
|---|---|---|
| {A+D, B+C} | free (cliques) | perfect; presence detectors suffice (and many other perfect builds exist) |
| one group of three, one detector kept | conflicting, alone | loses 1; its neuron is a presence detector of a clique inside the group |
| {A+C, B+D} | the two families, conflicting | perfect; every perfect build uses two split units (proved, T62 part 4); all 8 pieces found are split dials |
| {A+B, C+D} | conflicting, each inside one block | not perfect (best 2); split dials again |
| T20 | an OR pair | presence detector |
| T33 | one family | a dial ON where no member is ON |

**Step 4. The essay's reading (Hypothesis H-freefam, PENDING).**

```
continuous family (curves at all angles):   blend of a 30-degree and a 60-degree curve  ->  the 45-degree neuron fires
                                            -> no silent blend -> free -> a coarse DETECTOR ("curves & loops")
discrete alternatives (taxi, ambulance):    blend of a taxi and an ambulance photo  ->  "nothing"
                                            -> silent blend -> conflicting -> a CONTRAST neuron, alternatives at its two poles
```

The essay's words fit the second case: "taxi and ambulance sit on each other's negative poles" under one gate, and
"gate = region, value = distinction inside it" (essay idea I1). The small network keeps the family as an axis
("taxi rather than ambulance") instead of a detector.

Why it matters: coarsening is not wrong. But the coarse neuron of a family depends on whether the family is a
continuum or a set of alternatives on the mix. That can be checked with one linear program per group (Proposition T59).

**Step 5. The decision (card 3-13).** Options: (a) a kind of coarse neuron per kind of group; (b) the same, with the
rule checking each group against the mix itself; (c) coarsening is the wrong frame for conflicting groups. My
recommendation: (b), and to test H-freefam on InceptionV1's curve family next.

## 21. One object for all four kinds: where the circuit bends (2026-10-27, night)

Shonu asked for one method instead of a theorem per kind of coarse neuron, and for a circuit-level object that scales
better than looking at every neuron.

| Word | What it is in the network |
|---|---|
| line between two photos | the inputs (1 - t) x + t x', t from 0 to 1 |
| kink | a point on the line where the answer bends: some neuron, in any layer, crosses zero there |
| jump | how much the answer's slope changes at a kink (positive = bends upward) |
| dip | answer at a point of the line minus the straight-line average of the answers at the two ends |
| dip line | two photos of the mix and a point between them that is also in the mix, with a nonzero dip |
| coarse circuit | a smaller network that replaces a set of neurons (in any layers) and must answer the same on the mix |

**Step 1. The kink formula (Theorem T63, proved; one-bend identity in Lean; checked exactly on 300 random deep
networks).**

```
answer along the line:   straight pieces, bending at kinks t_1, t_2, ...
dip at the point lambda  =  - sum over kinks of  jump x K(position)        K = a tent: 0 at the ends, highest at lambda
```

Why it matters: the dip identity of last session (T62a) is not about single neurons. It is a fact about the whole
circuit's answer along lines, at any depth. Every dip must be made by upward bends somewhere on the line.

**Step 2. What one neuron of one layer can do on a line (Proposition T64, proved).**

```
a relu(l(x)) along a line:   flat, then a straight ramp (or the reverse)  ->  ONE bend at most, and monotone
```

Why it matters: the two-point case of T59 (one neuron cannot cover two members with a silent blend), T62a and T62b
each follow in one line.

**Step 3. A size bound that belongs to the circuit (Theorem T65, Corollary T65.1, proved).**

```
dip lines of the mix:   each needs a neuron (positive output weight) that bends across it
one-layer neuron = one flat cut through the input space  ->  size >= tau = the fewest cuts that cross every dip line
T20:  dip lines (0, 2) and (-2, 0) do not overlap  ->  no single cut crosses both  ->  width 1 cannot be exact
```

Why it matters: the bound is computed from N's answers on the mix, once per circuit, with no neuron-by-neuron fitting.
T20's old impossibility proof is exactly this count.

**Step 4. Circuits across layers can be far smaller (Proposition T66, proved, checked exactly up to k = 6).**

```
zigzag answer on 2^k + 1 points:   one layer needs >= 2^(k-1) - 1 neurons (disjoint dip lines)
tent-map circuit: k layers of 2 neurons   ->   exact with 2k neurons          (k = 6: 12 against 31)
```

Why it matters: a deeper neuron's zero set is bent, so it can cross many dip lines. A coarse circuit that may use
several layers is a real gain, not only a change of words. Whether clean networks need it is Conjecture D61.

**Step 5. The four kinds, read through dip lines (Remark B.X.1, corrected by the referee).**

| Kind | Reading | Status |
|---|---|---|
| AND (T58) | the members switch together on the mix | analogy only: on a line between two photos they may bend in different places |
| free (T59.2) | no dip line inside the group, so the bound says nothing | analogy only: T59.2 is a construction |
| conflicting, alone (T59.1) | a dip line between two members cannot be crossed by a neuron ON at both ends | consequence (T64 part 3) |
| conflicting, together (T62) | the four dip lines need crossings; a group's neuron crosses at most two | consequence (T62b and T62's sign argument) |

On T61's instance (exact): the clique build has both neurons bending across all four dip lines, each paying half of
every dip; the split-dial build has each dip line paid by exactly one neuron.

**What the referee changed.** Its count: these results are elementary (the kink formula holds for any piecewise-linear
function) plus a known depth separation; none yet says what a least coarse circuit is. The bound tau depends only on
N's answers on the mix. It is only a lower bound: on T61's instance one cut crosses all four dip lines, yet no single
neuron is exact. My first conjecture ("depth is needed exactly when tau is too large") was false for that reason, and
now reads: for families that are free on the mix, the bound is reached by detectors (Conjecture D61).

**Step 6. The decision (card 3-14).** Should the trunk coarsen circuits (neurons in any layers, with a budget) instead
of groups inside one layer? My recommendation changed to (a) for now: keep T4 and use dip lines as an analysis tool.
The only evidence that coarse circuits across layers help is the general T66; revisit once a clean network with such
a job is found.

## 22. What the math says the unit is (2026-10-27, unit branch)

Shonu asked: stop adding one theorem per kind of coarse neuron. Write down what any unit of coarsening must do, then
let the math find the unit, even if it is not a "circuit". Paper: `UNIT-BRANCH.md`. Lab log:
`research/clean/units/UNIT-LOG.md`.

| Word | What it is in the network |
|---|---|
| set S | some of N's neurons, in any layers, that the developer wants to coarsen together |
| box | the new neurons that replace S. They read what S's neurons read. S's readers read them instead |
| free box | box neurons are any relu(m . z_S + beta). They may fire where every neuron of S is OFF |
| covering box | box neurons are OFF wherever every neuron of S is OFF (in one layer: m >= 0, beta <= 0) |
| co-firing | two neurons are both ON on the same input of the mix |
| independence | the best box for S does not depend on what happens to T, and the losses add |

**Step 1. The demands (Definition DU5).**

```
U1  replace S by a box, get a normal layered network again
U2  each new neuron belongs to one set of N's own neurons (no new basis)
U3  a box talks to the rest only through S's readers
U4  independence: coarsening S and coarsening T do not affect each other
U5  coarsening a box again is still a box
```

Why it matters: U4 is what lets a developer zoom into one district and leave the rest alone. It is the demand that
decides the unit.

**Step 2. Which sets can be replaced at all (Lemma TU1, proved).**

```
S = {layer-2 neuron, layer-4 neuron}, and a layer-3 neuron v outside S sits between them
-> v reads S and S reads v  ->  the box would have to read its own output  ->  not one box
```

Why it matters: a "circuit" across layers is a usable unit only if it contains every neuron on the paths between its
neurons. One-layer groups always pass.

**Step 3. Linear neurons: independence means uncorrelated activity (Theorem TU4, proved; checked on 30 random cases).**

```
S's neurons and T's neurons:  correlated on the mix   ->  spare neurons of T predict S's error and cancel it  ->  I < 0
                              uncorrelated            ->  losses add, best boxes do not depend on each other
```

Why it matters: activity on the mix decides the unit, not wiring.

**Step 4. ReLU neurons with free boxes: no independence at all (Theorem TU10, proved; exact instance).**

```
gap |x| = relu(x) + relu(-x) on {-2, -1, 1, 2}
S = {relu(x)}  is OFF on x < 0, but its pre-activation x still varies there
box of S:  copy relu(x)  +  spare relu(-x)        <- reads S's NEGATIVE pre-activation, fires on T's inputs
T = {relu(-x)} removed  ->  the spare neuron does T's job  ->  joint loss 0, single loss of T = 5/4,  I = -5/4
```

Why it matters: an OFF neuron still carries information in how far below zero it is. A free coarse neuron can read it
and fix a neighbour's error. So, with free boxes, no two sets that cost something can be coarsened independently
(Corollary TU11, last hidden layer). T61's "split dials" (last session) are this effect: the {A, C} neuron is most ON at B.

**Step 5. Covering boxes: independence for neurons that never fire together, at any depth (Theorem TU5, proved;
checked on 50 random two-layer networks, error 0).**

```
input of district V:   V's neurons ON,  T's neurons OFF  ->  T's covering box OFF  ->  sends nothing
input of district A:   the reverse
-> at every input only one box acts  ->  losses add exactly  ->  best boxes are independent
```

Why it matters: this is the essay's "negative poles as gates in a circuit, not a neuron's opposite". A covering
neuron's negative pole can only keep it OFF.

**Step 6. The unit (Corollaries TU8 and TU11, Proposition TU9, proved).**

| Box class | The unit the math gives |
|---|---|
| covering on the mix | in the last hidden layer: a co-firing component (neurons joined by "ON on the same input"). With a hub neuron per district, these are the districts. Across layers it works too (TU5), if the set contains every neuron on the paths between its neurons and is not wired to the other sets. Units nest |
| free | no two sets that cost something are independent |

So the math does not pick "circuit" or "one-layer group" as such: it picks who fires together on the mix. Wiring
enters only through "closed under paths" and "not wired to the others".

**Step 7. The toy with gates (lab evidence, `toy_district.py`, `toy_pcover.py`).**

```
two districts on a circle, V (right arc) and A (left arc), each with a gate and 4 narrow neurons; 4 classes
coarsen V's 4 neurons and A's 4 neurons, 1 new neuron each

free boxes:          singles 0.120 + 0.126,  joint 0.067   <- each box does part of the other's job
strict covering:     singles 0.161 + 0.155,  joint = their sum exactly (error 0.0)
covering on the mix: singles 0.120 + 0.126   <- same as free: the free single boxes already stay in their district
```

Why it matters: my prediction that gates stop free help FAILED, and TU10 says why. But independence is almost free
here: the best box for one district alone already stays home. What independence forbids is only the help between
districts when both are coarsened at once. The strict covering rule (m >= 0) was too strict: it also bans negative
reads inside the district, which independence does not need.

**Step 8. How many open questions go away (UNIT-BRANCH.md D.2).**

| Verdict | Nodes |
|---|---|
| partly informed | R-circuit, R-deep (a box reads its in-boundary, no decode; still your choice) |
| absorbed | R-kinds (becomes card 3-15) |
| changes | S-onehidden, R-groupopt, R-recur, R-freefam, R-cleannet |
| unchanged | S-twoclass, N-onehidden, N-ny3, R-mixclean, R-jobmix, R-listen, R-explaw, R-fullrank, R-dialfit |

**The decision (card 3-15, R-indep).** Free coarse neurons and independence cannot both hold (TU10). Neurons that are
covering on the mix (OFF at the mix's inputs outside their district, any reads inside) keep independence and give a
unit (the district). The price is the cross-district help that free neurons would add: large on T61's network (no gate,
each neuron is its own district), almost zero for single districts on the gated toy. My recommendation: develop (a),
covering on the mix, as the branch. Keep the trunk as written until the price is known on a real network.

| Status | Items |
|---|---|
| proved | TU1, TU2, TU3, TU4, TU5, TU6, TU7, TU8, TU9, TU10, TU11 (Lean: summation, independence of a sum, covering composes) |
| measured (lab) | toy with gates, random checks |
| open | the converse of TU5 below the last layer (Conjecture DU1); deep linear networks (DU3); the price of covering on a real network |
| referee | 19 findings, one error (T59.2's unit is covering on the mix, not a fold); all fixed in the paper, not re-refereed |

## 23. A coarse group sees only what its readers saw (2026-10-27, night)

You read section 22 and said: if N is clean and well trained, it should already be mostly independent. If our math says
"not independent", we are looking at N the wrong way. And independence should be graded, like every part of clean.

| Word | What it is in the network |
|---|---|
| box | the new neurons that replace a group. Example: 2 new neurons in place of 10 curve neurons. They read what the 10 read. The 10's readers read them instead |
| silent | a group is silent on an image when all its neurons are OFF (output 0) |
| output-faithful box | the box's output depends on the image only through what the group's neurons SENT (their outputs after ReLU) |
| silence-preserving box | output-faithful, and it sends nothing on images where the group sent nothing |

**Step 1. The old result held for every network (Remark B.4).**

```
TU10 premises:  some neuron of S has an input value that changes from image to image   <- true for almost any net
                T's box is not perfect                                                  <- true for almost any net
-> "no independence" held for clean, messy and random networks alike
```

Why it matters: a result that does not depend on N tells us about our definition of the box, not about clean networks.

**Step 2. The free box read what N throws away.**

```
neuron u of S:      input value     output     what N's readers get
  image of S:          +3.0           3.0            3.0
  image of T:          -1.0           0              0
  image of T:          -5.0           0              0      <- readers cannot tell -1 from -5
free box neuron relu(-input):  1.0 and 5.0 on T's images     <- it can, and uses it to do T's job
```

Why it matters: the essay says a network is "built to be read ... by its downstream readers", so we should "become
another reader". A reader never sees the -1 or the -5.

**Step 3. First principle: the box sees only what the group sent (Definition DU6, Lemma TU12, proved).**

```
images where all of S is OFF:  S sent 0 on every one  ->  the box must give one constant there
images where S is ON:          S sent different things ->  the box may do anything
```

Why it matters: the reading of below-zero values is gone. But one channel is left: a constant on all the images where
S is silent.

**Step 4. The constant is still a strong channel (Theorem TU13, proved; lab check).**

```
S's box: one neuron fits S's own images, one constant neuron adds "+1.5 on every image"
         the first neuron cancels the +1.5 on S's own images
-> net effect: "+1.5 only where S is OFF" = "S is off, so answer like T"
```

| Instance | Free box interaction | Output-faithful interaction |
|---|---|---|
| \|x\| = relu(x) + relu(-x), T removed | -5/4 | -13/12 (lab) |
| circle toy, budgets (1,1) | -0.18 | 0.000 |
| circle toy, budgets (2,2) | -0.004 | -0.002 |
| circle toy, budgets (1,2) | -0.095 | -0.107 |
| T61 detector network | at most -1/8 at budgets (1,2) (TU6, proved) | -0.368 at (1,1), -0.267 at (1,2) |

The free numbers come from section 22's run, which uses a search, not a proof. At (1,2) the output-faithful joint loss
(0.020) is lower than the free one (0.027), although every output-faithful box is also a free box. So the free search
missed its best build there, and the free values are only upper bounds.

Why it matters: my prediction that this constant is harmless FAILED on two instances. The coarse group uses its own
silence as a vote for the other group's answer.

**Step 5. Second principle: silence stays silence (Definition DU7, Corollary TU15, proved).**

The essay: "Think of negative poles as gates in a circuit, not a neuron's opposite." A group's silence keeps inputs out.
It does not vote for other answers. So the box must send nothing where its group sent nothing.

```
output-faithful  +  silence stays silence   =   "covering on the mix" (section 22, option (a))
-> two groups that are never ON together:  losses add exactly, at every budget  (I = 0)
```

Why it matters: last session I offered "covering" as a choice for you. Now it follows from two readings of the essay.
It is not a decision any more.

**Step 6. Independence is graded (Theorem TU14, proved).**

Count a neuron's output as silent when it is below a threshold (the essay: weak activations can be removed "for almost
no cost"). Then the interaction of two groups is at most a sum of two kinds of terms:

| Term | What it is in the network | When it is 0 |
|---|---|---|
| strong co-firing | images where both groups are strongly ON | the groups never fire strongly on the same image |
| dropped weak writes | images where one group is strongly ON and the other only weakly ON; the coarse network drops the weak part | weak activations write nothing that matters |

The essay measured the second kind on InceptionV1's last layer: keeping only the strongest quarter of positive
activations keeps top-1 accuracy at 69.47% against 70.08%. This is your "72 vs 71" number.

Why it matters: this is your picture. A clean network is "mostly independent", and the size of "mostly" is a number:
the mass of strong co-firing plus the size of the dropped weak writes. It can be one entry of the cleanliness list
eps(N).

**Step 7. What the unit means in an image model (Conjecture CU1, not proved).**

The essay: "A car neuron provides a reusable judgment and a sheltered place for taxis, ambulances, and other
specialists to develop." And: "Smaller models may stop at coarser distinctions, with something closer to a general
"car" neuron."

```
district:   [car judgment neuron]  +  taxi, ambulance, convertible, station wagon specialists
            they fire strongly together on car photos, and are silent on dog photos
coarse:     one neuron that is ON on car photos = the district's judgment ("a general car neuron")
```

Conjecture CU1: for a clean network, the best one-neuron coarse version of a district is ON across the whole district
and reads it like its judgment neuron. Not checked yet.

**Step 8. Where connections enter (Conjecture CU2, not proved).**

The essay says groups found from connections predicted ablation damage "much better than comparable groupings of
activations". In a lower layer, two groups can interact only through neurons that read both of them. So the unit
would be fixed by shared readers AND strong co-firing together. Not proved yet.

| Status | Items |
|---|---|
| proved | Remark B.4, Lemma TU12, Theorems TU13 and TU14, Corollary TU15 |
| measured (lab) | the table in Step 4 |
| failed prediction | "the constant is harmless" (Step 4): failed on \|x\|, the toy at (1,2), and T61 |
| referee | 2 errors (a bound without projection; Conjecture CU2 false as first written), 6 gaps; all fixed, not re-checked |
| open | CU1 (the unit is the district's judgment), CU2 (shared readers), an N-only formula for the graded term |

## 24. Where the "help" really came from (2026-10-27, late night)

You asked: the "help" between coarsened groups does not look like any coarse neuron in the essay. What is it really?

| Word | What it is in the network |
|---|---|
| reader | a neuron (or logit) that reads the group. Here: the logits |
| reader's threshold | the reader's bias. It decides at which input value the reader's answer flips |
| contrast | a distinction with two sides, such as "upper versus lower" |
| dial | one neuron read against a threshold: above it, one answer; below it, the other |

**Step 1. Look at the helping neurons (lab, `help_anatomy.py`).**

```
original toy, both districts coarsened to 1 neuron each:
  V's new neuron:  ON below the horizon on BOTH arcs   ->  writes V1 and A2 (the two "lower" classes)
  A's new neuron:  ON above the horizon on BOTH arcs   ->  writes V2 and A1 (the two "upper" classes)
```

Why it matters: the help looked like a shared "up versus down" neuron. That was a real reading, but it was not the main
cause. Step 3 shows why.

**Step 2. One new neuron can push only one way (Theorem TU19, proved).**

```
4 members: 2 "lower" (write V1), 2 "upper" (write V2)   --coarsen-->   1 new neuron u, u >= 0
logit V1 minus logit V2 receives:  u x (one fixed number)
   u > 0  ->  always the same sign        u = 0  ->  a tie
-> "upper versus lower" is impossible, unless the reader's threshold moves
```

Our builds kept every reader's bias at N's value. So the best build moved the threshold through a back door: another
group's new neuron is ON with one fixed value everywhere outside its own group, and that fixed value acts as the
missing threshold.

Why it matters: the "help" is not a new feature. It is a reader's threshold, smuggled through another group.

**Step 3. Open the front door: let each reader re-set its threshold (lab, `toy_bias.py`).**

| Toy | Help, thresholds frozen | Help, readers re-fit their bias |
|---|---|---|
| original circle toy | -0.18 | +0.006 |
| sheltered districts | -0.239 | +0.0001 |
| gate in front | -0.203 | +0.0001 |
| shared pose units | -0.052 | -0.003 |

Why it matters: in all four toys, with free coarse neurons and no special rule, the groups become independent up to
small terms. This matches your prior: the networks were not the problem. A frozen threshold in our definition was.
But the referee proved that this is not true in general (Proposition TU21's example, Step 4).

**Step 4. What is left, and what it means (Remark B.6, Proposition TU20, proved).**

| Source | What it is in the network | After readers re-fit their bias |
|---|---|---|
| frozen threshold | another group's fixed value acts as the reader's missing offset | the part that is the same on every image is gone |
| silence used as an opposite | a group's spare neuron sends a fixed value only where its group is silent: "T is off, so answer like S" | stays (proved example, TU21). It is worth at most one more neuron to the group it helps (TU18). "Silence stays silence" (DU7) forbids it |
| a rebuilt unit | relu(x) and relu(-x) read the same x with opposite signs. Put in different groups, one group's box rebuilds the other half. Exact example: I = -11/16 (TU20) | stays. Sheltered groups cannot do it (TU17) |
| a shared reader bias | a reader of both groups has one bias in the joint build | new, small here, either sign |

Why it matters: every source is something we can name in the network: a threshold, a silence read as "the other
answer", or the other half of a dial. None of them needed a hidden feature. The list is not proved complete.

**Step 5. What a district's coarse neuron is (lab, `cu1_dial.py`; Conjecture CU1, not proved).**

```
district V: 2 "lower" members, 2 "upper" members
best single coarse neuron (sheltered toy):
     +1.16 x lower member 1  +1.16 x lower member 2
     -1.16 x upper member 1  -1.16 x upper member 2   + 1.56
  -> ON across the whole district, high on lower images, low on upper images
  -> the reader's threshold decides: above it "V1", below it "V2"
```

This is a dial with siblings on both poles. The essay describes the same shape: "an ambulance neuron might have a taxi
on its negative pole", and the small model's neuron has "cars, trains, and carts" on its positive side and "boats,
planes" on its negative side.

Why it matters: this is the interpretable unit. Coarsening a family of siblings gives one dial over the family. The
siblings split between its two poles, and the reader's threshold sets the boundary.

| Status | Items |
|---|---|
| proved | TU17 (sheltered inputs), TU18 (a judgment unit gains nothing from a constant), TU19 (one unit is one-sided), TU20 (a cut dial keeps its help) |
| measured (lab) | Steps 1, 3, 5 |
| failed predictions | sheltering removes help (no: the threshold did); a gate in front removes help (no); the original toy keeps help with re-fit readers (no, it is small); the dial is monotone at 0.9 (0.79 to 0.88) |
| my overclaim, corrected by the referee | "re-fit readers remove the help" is true in the four toys, not in general: silence used as an opposite survives |
| open | CU1 (the dial form, proved on no network yet); CU3 (independence with re-fit readers, in general); the referee's report on TU15 to TU20 |
| your decision | card 3-16: may readers re-fit their bias? My recommendation: yes, together with "silence stays silence" |

## 25. The same error on the weights side (2026-10-27, late night)

You decided: readers may move their biases (node R-readbias, DECIDED). You asked whether the same error exists for
weights.

**Step 1. A bias is a weight.**

```
reader = w_1 x (kept neuron 1) + w_2 x (kept neuron 2) + ... + b x (a neuron that is always 1)
```

So re-fitting the bias is re-fitting one weight. The builds still froze the readers' weights on every kept neuron
(here: the logits' weights on the two gates). Another group's box could stand in for a change of such a weight, if it
can compute that neuron's signal.

**Step 2. When this kind of help is impossible (Proposition TU22, proved).**

```
district V: inputs sheltered, a kept "judgment" neuron j_V (ON with one value on all of V's images, OFF elsewhere)
reader may re-weight j_V  ->  it can add any offset on exactly V's images
-> no other district's box has anything to add:  I = 0 at every budget
```

Why it matters: the essay's "car neuron" ("a reusable judgment and a sheltered place for taxis, ambulances") is what
makes a district independent, as long as it is kept and its readers may re-weight it.

**Step 3. The toys (lab, `toy_weights.py`).**

| Toy | Single loss, bias only -> bias + weights | Interaction at (1,1) / (2,2) / (1,2) |
|---|---|---|
| original | 0.0094 -> 0.0050 | +0.0086 / +0.0005 / -0.0006 |
| sheltered | ~0 -> ~0 | +0.0001 / 0.0000 / +0.0001 |
| gate in front | 0.0009 -> 0.0006 | +0.0001 / +0.0011 / +0.0001 |
| shared pose | 0.0105 -> 0.0101 | -0.0010 / +0.0017 / -0.0040 |

Why it matters: re-weighting the gates works as each district's judgment, and it halves the original toy's single loss.
No new help appears. The largest term is now a small cost (+0.0086): in the original toy each box leaks onto the other
arc, and the readers' shared weights cannot serve both groups. The shared-pose toy keeps a small help (-0.004),
through N's own pose neurons, which both groups' boxes can read and the logits do not read.

| Status | Items |
|---|---|
| proved | TU22 (a kept judgment neuron that readers may re-weight makes its district independent). After the referee: it is enough that a few of the district's own kept neurons together are flat across the district; any kind of coarse neuron |
| measured (lab) | Step 3 |
| failed predictions | original toy \|I\| <= 0.002 (got +0.0086, a cost); pose \|I\| <= 0.003 (got -0.004) |
| decided | node R-readweights: readers also re-fit their weights on kept neurons (Shonu: "yes", 2026-10-27) |

## 26. A mapped network, and the unit it gives (2026-10-28, Trunk T5)

You proposed that the math may assume a network that is already mapped, as long as the map is mathematics too. The
trunk is now T5: the rule reads N, the mix P, a map M, and the zoom. Finding the map is outside the math.

| Word | Meaning here |
|---|---|
| claim | one checkable sentence about N and P, with an error eps that anyone can compute; eps = 0 means the sentence is true |
| region | a set of images, for example "car photos" (a concept, given by the images it covers) |
| district | the neurons of one layer that may be ON only on one region's images |
| shelter | outside its region, a district's inputs do not change (the gate in front: "its negative side keeps the dogs out") |
| judgment | a kept neuron, or a few kept neurons, whose sum is 1 on the region's images and 0 elsewhere (the "car neuron") |

**Step 1. What a map says (Part A.7 of the paper).**

```
map of the last hidden layer:
  region "car photos":  district {taxi, ambulance}   shelter: their inputs are fixed on non-car photos
                        district {car neuron}         judgment: 1 on car photos, 0 elsewhere
  region "dog photos":  district {ear, snout}, district {dog neuron}, ...
each claim -> one error number; the list of errors is how clean N is (your "multiple metrics")
```

Why it matters: "clean" is now a list of checkable numbers about a given map, not something the rule must discover.

**Step 2. The map splits the zoom into regions (Theorem TM1, proved).**

```
zoom level:  group 1 = {taxi, ambulance} -> 1 new neuron      group 2 = {ear, snout} -> 1 new neuron
on a car photo:   dog neurons are OFF, group 2's new neuron sends one fixed vector (its inputs are fixed: shelter)
                  the readers cancel that vector with the car neuron (judgment) -> only group 1 matters here
on a dog photo:   the same with the roles swapped
result:           best loss of the level = best loss of group 1 on car photos + best loss of group 2 on dog photos
```

Why it matters: groups in different regions are exactly independent, under the reader rules you decided (readers
re-fit biases and weights). Switching one group costs only that group. This holds in the last hidden layer, for maps
that cover it (nodes S-lastlayer, S-cover, S-regiongroup, card 3-18).

**Step 3. Why the unit is a region, not an "edge tie" (Proposition TA12, proved).**

```
neuron v writes the same amount to every class  -> no probability depends on v (nothing reads it, in effect)
v is ON where group S's coarse neuron makes mistakes
readers may re-fit their weights on v   -> they start to read v and fix part of S's mistakes
so removing S and v together costs MORE than the two costs added (interaction > 0)
```

Why it matters: readers that re-learn can read anything that is ON. So the units that TM1 proves independent are
fixed by where neurons are ON (the regions), and every neuron that may be ON in a region belongs to it, read or not.
(TM1 shows that regions are enough; it does not show that every independent unit must be a region.) The essay's warning
about activations still holds for what a neuron MEANS (paper B.6). It does not decide the unit, because the readers
can re-learn.

**Step 4. A map can say too little (Theorem TM3, proved; card 3-19).**

```
map M_1:  one district = every neuron of the layer, region = every image   -> always true, error 0
T5 says:  all errors 0  ->  give the exact best coarse network (read: computable in polynomial time, as for T43)
T43's clique detectors:  that problem is NP-hard
-> under that reading T5 can be met only if P = NP
```

Why it matters: the map must say ENOUGH. A true but vague map carries no information, so it cannot make the hard
problem easy. Adding more claim types does not help by itself, because a map may leave them out. The fix is to ask the
exact clause only of maps that are complete for the zoom: for each group, the map must hold certain claims (TM1's
district, shelter and judgment claims, and a "no silent blend" claim inside the group, Conjecture CM2). On the clique
instances the big group's blend claim is false, so the conflict disappears. This is card 3-19, your decision.

**Step 5. What it looks like in InceptionV1 (measured, Part C).**

```
candidate map at the last hidden layer (mixed5b):  region "dog photos" (labels 151-268) | region "other photos"
median neuron: 41% of its activity is on the WRONG region (50% = no preference)
7 of 1024 neurons have less than 10% on the wrong region; none is dog-only enough to serve as a judgment
```

Why it matters: "dogs versus everything" is a bad map for this layer. Fur, eyes and texture are shared with many other
animals. The essay puts its districts (cars, dogs) in the middle layers 4c to 4e, and with finer regions. That map is
the next measurement. The theorem says what such a map must contain: districts, a gate in front, and a judgment.

| Status | Items |
|---|---|
| decided | Trunk T5 (node R-map: "Yes, draft T5", "Graded eps", plan approved) |
| proved | TM1 (map splits a grouping into regions, exact), TM3 (a proved conflict: T5 read with polynomial time needs P = NP; card 3-19), TM4 (no bound on the split in the claims' errors alone: a tiny shelter error can be amplified by a huge weight), TA12 (re-fit readers tie an unread neuron to S when its activity meets S's leftover error) |
| checked (exact toys) | `tm1_split.py`: split exact to 1e-16 on 200 box pairs; breaking shelter or removing the judgment breaks it. `ta12_sub.py`: I = +0.046, control 0 |
| measured | H-map on InceptionV1's last layer: the dog/other map has district error 0.39 |
| failed (and fixed) | the first two TM1 checks were too weak (an optimiser that missed minima); the box-by-box check replaced them |
| referee | REFEREE-MAP-2026-10-28.md, two rounds: TM1, TA12, TM3 parts 1 and 2 correct; TM3 part 2 is a conflict with T5 (I had first read it wrongly); CM1 and its repair CM1' false, replaced by TM4; claim errors made computable and unchanged by rescaling |
| open | the "no silent blend" claim (CM2), what a "complete map" must hold (card 3-19), a graded clause that survives TM4, deeper layers (S-lastlayer), maps that leave neurons out (S-cover), two groups in one region (S-regiongroup) |

## 27. Complete maps: a map that says enough (2026-10-28, later)

You decided: the exactness promise applies only to complete maps (card 3-19). The trunk text now says so. (I first
applied the same limit to the graded promise too. The referee showed that no theorem forces that, so I took it out
again. It is a question for you.)

| Word | Meaning here |
|---|---|
| complete map | a map that, for every group of the zoom, holds the claims the theorems need: district, shelter, judgment (not needed when the region is every image), every other neuron in a district or flat, and a blend claim |
| blend | an image whose group pre-activations are a mix of the pre-activations at other images, but where the group sends LESS to the output than at all of them, or MORE than at all of them |
| flat claim | a neuron outside every district takes one value on each region |

**Step 1. The blend claim (Definition DM7).**

```
group S = {detector A, detector B}
image a: A fires, output 2          image b: B fires, output 2
image m: halfway between a and b in S's pre-activations, output 0   <- a silent blend
one new neuron relu(m.z + beta) is monotone along that line: it cannot be 2, then 0, then 2
-> S cannot be folded into one neuron exactly
(the claim checks both directions: quieter AND louder; a first version checked only "quieter",
 and with the two output classes swapped the hard instances slipped through: referee round 3)
```

Why it matters: this is the claim that can fail on hard instances, so the hardness becomes a positive error instead
of a contradiction.

**Step 2. On T43's hard networks the blend claim decides everything (Theorem TM5, proved).**

```
clique detectors, T43's images:
  blend error of S = 0   <=>   S has no bad pair   <=>   S folds into one neuron exactly (T59.2's formula)
the big group of all detectors always holds a bad pair (two detectors of one block) -> blend error > 0
-> no complete error-0 map for it -> T5 promises nothing there -> the conflict of TM3 is gone (Corollary TM5.1)
```

Why it matters: with complete maps, the clique instances agree with T5. The groups whose blend claim holds are folded by
a formula. The groups whose blend claim fails are exactly the hard ones.

**Step 3. Neurons outside every district (Theorem TM1 part 4, proved).**

```
neuron f in no district, value 0.8 on every car photo, 0.3 on every dog photo
readers may re-weight f, but that only adds one constant per region -> the split of TM1 still holds
```

Why it matters: a map does not have to put every neuron in a district, as long as the leftover neurons are flat.

**Step 4. Small errors can still matter a lot (Proposition TM4, proved).**

```
kept neuron j1: on car photos (region A), and on one dog photo with tiny value delta
readers give j1 the weight 1/delta  ->  it fixes the dog group's mistake on that photo
district error of j1 -> 0, but the effect on the best loss stays the same size
```

Why it matters: free reader weights amplify any small leak. A rule cannot meet the graded promise just by splitting the
problem with TM1. Whether some other fast rule meets it is open. This is not yet a decision for you, because no theorem
forces one.

**Step 5. InceptionV1, dog photos against car photos (measured, then corrected).**

```
first reading:  typical neuron has ~30% of its activity on the wrong region (4e, 5b)  -> "most neurons are shared"
check:          Zoom In's car detector 4c:447: 1% on dog photos; strong firing: 0 of 16 images are dogs
                the 10 neurons that push hardest toward "car" or "dog" at 5b: mostly 0-6% wrong when firing strongly
correct reading: the 30% is the median over ALL neurons, and most of them serve the other ~980 classes
```

Why it matters: the neurons that decide dog versus car do look like districts. My first reading was wrong, and you were
right to doubt it. What remains true: a complete map must still put every neuron somewhere (a district or a flat
claim), and the claims do not yet weigh a neuron by how much the output depends on it. That is the real next gap: the
map's errors should count a neuron's activity in proportion to what its readers do with it (the edge reading of the
unit branch).

| Status | Items |
|---|---|
| decided | complete maps for the exactness promise (card 3-19 (a)) |
| proved | TM5 + Corollary TM5.1 (blend claim decides exactness on T43's instances), TM1 part 4 (flat leftover neurons), TM4 part 2 (tiny district error amplified) |
| checked (exact) | `model/tm5_blend.py`: both class orders, 1778 group checks, 0 exceptions; `units/tm1_split.py` v4: flat neuron 1.6e-16; `units/tm4_dist.py`: V_b = 0, L_T fixed (first run failed: my instance let N read j1; fixed in the paper and script) |
| measured | dog/car map: all-neuron median share 0.30 (4e), 0.27 (5b); the neurons that decide dog vs car are close to specific (4c:447: 0.01; top writers mostly <= 0.06 when firing strongly). My first reading ("most neurons are shared") was wrong |
| referee | round 3 found the one-sided blend claim (fixed: two-sided), a missing shelter claim in TM5.1's map (fixed), and that the graded limit was not forced (removed) |
| open | weighing a neuron's map error by how much the output depends on it; the blend claim for general networks and sizes (CM2); deeper layers (CM3); the graded promise (should it also need complete maps?) |

## 28. Small errors, small changes: the robust optimum (2026-10-28, later)

You said TM4's blow-up looked like an error in the math, not a design choice. It was. This section names the step
that failed, the corrected object, and the first real-network numbers.

| Word | Meaning here |
|---|---|
| leak | a neuron that is ON (even weakly) on photos outside its region |
| swing | how far one neuron can push the outputs: the spread of its logit weights (largest minus smallest) |
| clean table | the network the map says N is: leaks set to 0, sheltered inputs set to one value, flat neurons set to their mean, judgment sums set to exactly 1 |
| robust optimum | the best small network judged by its WORST loss over N, the clean table, and every mix between them |

**Step 1. What the least loss can see (Proposition TE1, proved).**

```
kept neuron j1:  values (1, delta, 0) on photos a, b, c          N's weight on j1 = 0
the small network's free readers:  weight about 1/delta on j1   -> photo b fixed exactly
delta = 0.3, 0.1, 0.01, 0.001:  same gain every time;  swing of j1 = 6.7, 20, 200, 2000   (te_robust.py, part B)
```

Why it matters: with free reader weights, the least loss depends only on the SPAN of the neurons' value vectors. It
sees where a neuron is on, never how much. Every claim error measures how much. So no error could ever bound it.
This is the error in the math: the trunk's object, "the least loss among all such networks", is blind to size.

**Step 2. The corrected object (Definition DE2, node R-wellposed).**

```
N's values ----- every mix ----- clean table
     \______________________________/
          the networks the map describes
small network is scored by its worst loss over all of them
```

Why it matters: the small network may not rely on anything the map calls an error, but it may rely on everything
else. This is the essay's ablation used as the definition: the quality must not depend on whether the weak leaks are
there. With every error 0, nothing changes (Proposition TE3).

**Step 3. The graded split (Theorem TE4, proved).**

```
clean split sum  <=  robust optimum  <=  clean split sum + E
E = sum over errors of (size of the error) x (swing of the neuron that carries it)
     linear in district errors; square root in shelter, judgment and flat errors (those are squared quantities)
```

Why it matters: small errors now give small changes, as you expected. The rule is TM1's formula run on the clean
table. No small network beats it robustly, and it loses at most E.

**Step 4. The checks (exact instances, `model/te_robust.py`, `model/te_robust_e.py`).**

```
TM4's two instances:  robust optimum = clean value for every delta; the old least loss = 0        (blow-up gone)
25 random instances (3 logits, 2 grouped regions, a box, a flat neuron):
    clean split <= robust optimum found <= worst loss of the formula <= clean split + E      all held
    E / eps constant as eps -> 0 (linear);  the robust optimum -> the clean value
```

A failure on the way, then fixed: the first definition gave the judgment neuron's values no clean value. The check
showed the clean problem itself reading the judgment neuron's tiny wiggle with weight 1/eps (Step 1 again, inside the
formula). The rule now: every value that a claim calls an error gets a clean value.

**Step 5. Real networks: the essay's weak-activation test on five ResNets (Part C, `pc_weak.py`).**

```
keep only the strongest quarter of each photo's positive activations in the last layer (about 40% of activity cut):
ResNet-18   KL 0.036   top-1 67.3 -> 66.5
ResNet-34   KL 0.026   top-1 70.0 -> 69.2
ResNet-50   KL 0.014   top-1 72.9 -> 72.8
ResNet-101  KL 0.012   top-1 74.0 -> 73.9
ResNet-152  KL 0.011   top-1 76.7 -> 76.3
```

Why it matters: N's own readers send little through weak activations, and less as the network grows (about 3x from
the smallest to the largest). This is term (c) of E, measured.

**Step 6. Real networks: does the dog/car map get cleaner with size? (Part C, `pc_scale.py`).**

```
                  what N sends through the leaks   KL if leaks removed   where neurons are active (wrong-region share)
ResNet-18          0.455                            0.337                 0.391
ResNet-152         0.390                            0.140                 0.401
```

Why it matters: in output units the bigger nets are cleaner (about 7 standard errors on the first column, 2.4x on the
second), as you predicted. In activity units they are not. And in absolute terms this map is far from clean: pooled
channels with a two-region majority rule are a crude map, and no top dog-vs-car neuron is specific.

**Step 7. The referee's counterexample: your graded clause is still not met by this rule (Proposition TE5, proved).**

```
kept neuron j:  1 and 1 + 1/W on photos x1, x1'      N reads j with weight W  -> N tells x1 from x1' by a 1/W difference
j leaks W^(-1/2) on photo x2;  N sends W x W^(-1/2) = sqrt(W) through that leak, and hides it with another neuron
every claim error -> 0, yet the robust optimum stays 0.0121 above the clean split (W = 1e2, 1e4, 1e6, te_robust.py F)
and the clean formula's own loss grows: 0.86, 9.9, 99.9
```

Why it matters: the fault is in N, not in the small network. N itself relies on a tiny value with a huge weight. The
old errors cannot see that, because they measure where neurons are on, not what N's readers do with them. The new
quiet claim (DM9) measures exactly that: what N's own readers receive through the leaks. Here it is large (2, 20,
200), so this is not a small-error case after all. It is the same quantity that H-scale measured (Step 6), and it is
the quantity that fell with network size.

| Status | Items |
|---|---|
| proved | TE0 (KL moves by at most the spread of the logit change; Lean `ce_osc`), TE1 (free readers see span, not size; swing ~1/delta), TE3, TE4 (graded split for the robust optimum), TE4.1 (TM4's blow-up gone), TE5 (graded clause not met without quiet claims) |
| new claim type | DM9 quiet claim (output units); with it, N's part of the bound is a claim error |
| checked (exact) | `te_robust.py` parts A-F all held; `te_robust_e.py` 25 rows all held after the judgment fix |
| measured | H-weak holds in the essay's form, falling with size; H-scale mixed: falls in output units, flat in activity units |
| referee | REFEREE-BE-2026-10-28.md: 2 errors (graded clause overclaimed -> TE5; judgment clean value not rescaling-safe -> fixed), 6 gaps, all fixed; not re-refereed |
| open | the clean small network's swings in the bound (they depend on the zoom, so they cannot be claim errors); TE0's bound is 50-130x loose on real logits; a finer real map; deeper layers (CM3) |
| your call | the corrected trunk wording (card 3-20) |

## 29. Size: does a larger network add up more error? (2026-10-28, latest; COARSE-THEORY.md B.N, card 3-21)

Shonu: a larger network has more neurons, so small errors could add up; but the essay says larger networks are easier
to read. Formalize "larger and cleaner gives a better result", so that small-network work carries to large networks.

**Step 1. The copy test (Proposition TN1, proved).**

```
neuron u  ->  n copies, each with a share lam_i of u's input weights, same output weights   (the network is the same)
function, least loss, robust optimum, clean split, E      -> all the same
district error, shelter error (sums over the neurons)     -> + (n - 1) x u's term, no limit in n
```

Why it matters: "errors add up with size" was a property of two of our error sums, not of the network. A bound
written in those sums would get worse with size for no reason. The copy test also found a fault in DE2: the judgment
clean value split its gap equally by neuron count. Fixed (shares by what each neuron carries); the check shows the old
rule misses the copy by up to 0.30.

**Step 2. Narrower neurons never hurt (Proposition TN2, proved).** Narrow kept neurons that add up to one broad one
(the essay's "bigger models learn narrower neurons") never make the robust optimum worse, and the fair measures do not
move.

**Step 3. What the bound counts (Theorem TN3, proved; Corollary TN4).**

```
E <= (kept mass) x (weighted mean deviation) + (box mass) x (weighted mean shelter deviation) + what N sends through the errors
the weights of each mean add to 1; copies and rescaling change none of the numbers; no number counts neurons
```

Corollary TN4: if the errors fall with size (H-scale, quiet form) and the write does not inflate (H-mass, new node
R-mass), the bound falls with size.

**Step 4. Checks.** `model/tn_copy.py`: 8 random exact instances, copies (n = 2, 3) of a judgment neuron, a district
neuron, a flat neuron and a group member, and splits of two kept neurons. Every check passed. Copies change V_b,
V_rob and the clean-table minimum by at most 1.2e-11. Splits change V_rob by -1.0e-2 to +8e-17 (never up). Lean
`Coarse/Scale.lean` builds.

**Step 5. The ResNet ladder (Part C, `pc_mass.py`).**

| | ResNet-18 | 34 | 50 | 101 | 152 |
|---|---|---|---|---|---|
| output spread O | 22.31 | 21.46 | 21.21 | 21.91 | 21.98 |
| what the leaks send / O (quiet form) | 0.455 | 0.422 | 0.403 | 0.390 | 0.390 |
| fair deviation (fair form) | 0.392 | 0.393 | 0.399 | 0.401 | 0.401 |
| leak cancellation at the logits | 9.6 | 10.8 | 13.3 | 13.8 | 13.3 |

Why it matters: H-mass part (1) holds (O does not grow). The quiet form of H-scale holds, the fair form does not. The
larger nets send less through their leaks because the leaks cancel more at the logits, not because channels fire more
in their own region. One coarse map, one draw.

| Status | Items |
|---|---|
| proved | TN1, TN2, TN3, TN4 (Lean `copy_mass`, `merge_osc`, `between_sum`, `mass_mean`, `osc_sum`) |
| changed | DE2 item 4 (judgment shares by carried activity); TE4's judgment constant |
| checked (exact) | `model/tn_copy.py` 8 of 8 instances |
| measured | H-mass part (1) holds; H-scale quiet form holds, fair form does not |
| open | H-mass part (2) (needs the clean small networks); the real gap, not only its bound; new layer kinds |
| your call | R-mass (card 3-21) |

**Step 6. Referee (REFEREE-BN-2026-10-28.md) and fixes.** 3 errors, 6 gaps. The main point: fix the error definitions,
do not work around them. Done: the district and shelter errors are now the largest neuron's term (not the sum); flat
claims take a set of neurons and the largest term; the blend-claim count (DM6 item 5) counts classes of proportional
neurons, not neurons. With these, a copy changes no error at all. Also fixed: TN3 part 3 (a dead term was missing),
TN4 re-stated on a "map form" of H-scale that is about N alone, the E claim for copies split out as Corollary TN1.1,
and the transfer and ResNet comments (ResNet's last block has a skip, so it is a new layer kind). Referee's rating:
4 bookkeeping results, 0 structural; the content is the counting fault in our definitions. Checks rerun: 8 of 8.

## 30. ResNet-18: the map, the merge rule and the zoom (2026-10-28, latest; COARSE-THEORY.md DM10, B.Y, Part C; card 3-22)

Shonu: "create a good map ... then apply this method ... I want the full compressible, expandable experience."

**Step 1. What ResNet-18's last layer looks like (Part C, measured on 35000 photos).**

```
last block:   z = conv path + skip   ->  relu  ->  mean over 7 x 7  ->  1000 logits
a channel:    on for ~860 of 1000 classes       (a district wants: off outside a few)
its nearest other channel: correlation 0.24     (a family wants: near 1)
its strongest 0.1 % of entries: ~61 classes     (a real concept: the cards page)
keep only that strong 0.1 %:    KL 4.9          (the outputs use the broad rest)
layer3, layer4.0, inner layers: the same picture
```

Why it matters: our claim types say a unit is off outside its region, or a near-copy of its family. ResNet-18's
channels are neither. So the maps were bad partly because, at the level of single channels and label classes, no good
map of these types exists for this network. Cards page: https://claude.ai/artifact/D4nGkYa9e6Xgz9jLk9r8Lt

**Step 2. The block with a skip (Definition DY1, Lemma TY1, proved).**

```
a new channel for group S:  relu(m . z_S + beta), same m at every position
in torchvision terms:       a conv2 row  +  a 1 x 1 projection of the skip  (torchvision already uses such projections)
TM1's proof never asks how z_S is computed  ->  TM1 holds for this block word for word
```

Why it matters: the theory now covers ResNet's last block. The "new layer kind" of section 29 was a definition gap.

**Step 3. The merge rule (Definition DY2, Theorem TY2, proved; new claim type DM10).**

```
proportional claim for a set Q:  "the channels of Q are copies of one another up to scale"
                                 error = (sum over Q of (output weight x scale) x distance from Q's mean direction)^2
merge rule:  Q  ->  one channel relu(z_Q)           z_Q = weighted mean of the normalised pre-activations
             its readers = sum of the members' scaled readers        (written down, no training)
bound:       loss <= (sum of sqrt(errors))^2 / 4      (Fact F11: KL <= a quarter of the squared logit change)
```

Why it matters: on maps that hold such near-copy claims for every group, the first rule in the paper whose bound
depends only on the claims' errors and is 0 when they are 0, for any number of logits, with no districts or regions. The exact check (`model/ty_merge.py`, 60 random instances x 3
noise levels) passed every case. Lean `Coarse/Merge.lean` builds.

**Step 4. The zoom on ResNet-18 (Part C; slider page https://claude.ai/artifact/Y22GZTSNSVKJ26U85GusJx).**

| channels kept | 512 | 256 | 128 | 64 | 32 | 16 | 8 |
|---|---|---|---|---|---|---|---|
| top-1 (full net 0.679) | 0.678 | 0.658 | 0.601 | 0.495 | 0.391 | 0.254 | 0.072 |
| agreement with the full net | 1.000 | 0.848 | 0.730 | 0.579 | 0.450 | 0.286 | 0.084 |
| KL, merge rule + re-fit readers | 0 | 0.131 | 0.432 | 1.022 | 1.631 | 2.421 | 4.021 |
| KL, keep the strongest channels | 0 | 0.124 | 0.419 | 1.046 | 2.066 | 3.156 | 4.137 |
| KL, random groups | 0 | 0.128 | 0.432 | 1.095 | 1.912 | 2.977 | 4.250 |
| KL, trained boxes (check) | | 0.098 | | 0.749 | | 2.002 | |

Why it matters: the compress-and-expand works mechanically: every size is stored once, and switching is a lookup
(TY3). But on this network the map adds little: down to ~100 channels the merges are no better than pruning or random
groups, and trained boxes beat the rule at every size checked. The proved bound holds but is 10^4 to 10^6 times too
large, because the channels are not near-copies.

| Status | Items |
|---|---|
| proved | TY1, TY2, TY3 (side branch under R-unitmap, R-propcomplete; Lean `relu_lip`, `merge_exact`, `merge_exact_sum`) |
| new definitions | DY1 (block with positions and a skip), DY2 (merge rule), DM10 (proportional claim) |
| checked (exact) | `model/ty_merge.py`: exact case 0, bound, rescaling, copies, re-fit, dynamic program |
| measured | ResNet-18 has no channel-level map of the A.7 types at any layer; the zoom numbers above. Corrected 2026-11-01: this claim is too strong. What is true: my scripts, with one fixed rule, label regions and these claim types, found none. Maps now come from agents (`map_agents/`) |
| open | the layers below the last one (card 3-12); a map type that fits broad channels (card 3-22 (c)) |
| your call | card 3-22: R-unitmap (what a map is made of), R-propcomplete (count merge maps as complete) |

**Step 5. Referee (REFEREE-BY-2026-10-28.md) and fixes.** The core proofs are correct. Rating: 0 structural results,
3 bookkeeping results. TY2 is a perturbation bound (relu is 1-Lipschitz, plus Fact F11) that holds for any layer with a
mean pool and a linear head. It says nothing about the best small network when the errors are large.

| Referee finding | Fix |
|---|---|
| hidden premise: the map must hold a merge partition of every group | now in the tags of TY2 and TY3, node R-propcomplete; TY2 and TY3 are a side branch until card 3-22 |
| "meets T5's graded promise" was overstated | now "on maps that hold merge partitions"; new TY2 part 6: a bound from the list of all proportional errors alone |
| the dynamic program solved "exactly k sets", the rule needs "at most k" | fixed (cost is not monotone; the referee's example); Part C's run used exact-k cuts, which TY2 still covers |
| channels as units and shared box weights were a hidden class choice | tag phrase, node R-unitmap |
| four false or wrong-reason sentences, an existence gap, bit-model wording | corrected in DM10, DY2, TY2 |

The referee's suggestion for the next claim type: a "pooled" proportional claim that compares the channels' pooled
outputs (what the logits read) instead of their pre-activations. It is exact strictly more often. Not adopted yet.

## 31. The stream: areas instead of layers (2026-10-29; COARSE-THEORY.md F12, F13, DM11 to DM13, B.V, D.15; card 3-23)

Shonu: "let's try this subspace idea ... we're thinking of regions of the model rather than orderings."

**Step 1. Three kinds of stream (Fact F12, from the code).**

```
ResNet      out = relu(out + identity)        a ReLU on every coordinate      -> channels are units
ConvNeXt    depthwise 7x7 filter, then norm  a filter on every coordinate    -> channels are half-units
ViT, GPT    norm -> linear in, linear out    nothing on single coordinates   -> only directions matter
```

Why it matters: "the stream has no special basis" is exactly true only for the third kind. That is the kind this step
studies, and the kind that most modern models have.

**Step 2. Rotation and the norm (Fact F13, proved, Lean `norm_rot`).**

```
rotate the stream by any Q, fold Q into every read and write  ->  the same network
neuron a -> neuron b through the stream = r_b . w_a            (an angle, unchanged by Q)
```

Why it matters: a map of the stream must be made of subspaces (areas), not coordinates. An area = the space a family
writes into and its readers read from (Definition DM11).

**Step 3. What a part nobody reads still does (Theorem TV1, proved, Lean `read_unchanged`, `temp_le`).**

```
every norm divides by |h|  ->  a direction no reader reads still changes the size every reader sees
drop it:  every read unchanged  <=>  its energy is the same on every input  (then fold it into the norm's constant)
else:     some read changes; keep it (one direction if it lies on a line: Corollary TV1.1), or pay a "temperature"
```

Why it matters: the essay's "single unread dimension" that scales everything through LayerNorm has varying energy. By
TV1 a cut must keep it. This is the one structural result of the step (referee).

**Step 4. Areas at the readout (Theorem TV2, Proposition TV3, proved; checked).**

```
areas that only the logits read, pairwise at right angles:
loss of the cut  <=  ( sum over areas of sqrt(readout error of that area) )^2 / 4      exact when all are 0
best k directions inside one area = top-k of SVD( what the logits read  x  the area's content^(1/2) )
an area never needs more than (number of logits - 1) directions
```

Why it matters: the best cut for the bound has a closed form. It weighs each direction by what the logits read from
it, not only by its energy (that is how it differs from SliceGPT). It is the best cut for the bound, not for the loss
itself.

**Step 4b. The catch (Lemma TV5, found by the referee, proved).**

```
"no block reads S"  <=>  S lies in K = the vectors that EVERY block read matrix sends to 0
a transformer MLP block has 4d read rows  ->  for generic weights K = {0}
```

Why it matters: Steps 3 and 4 are exact, but a real network meets their premise only on a null set of weights. In a
real network every direction is read a little. So these results are the exact end of a graded theory, and that graded
theory (Conjecture DV1: the bound grows with how much the blocks read the cut part) is the real next step.

**Step 4c. The graded case: blocks read the cut part a little (Theorem TV6, Proposition TV7, Corollary TV6.1, proved).**

```
cut a subspace D out of the stream; every block may still read D a little
leak of neuron i      = how much its read changes because D is gone        (computed from N alone)
error per area A      e_l  <=  (I + T_l) e_(l-1)  +  leak written into A    (I = the skip: errors are carried, not multiplied)
no norm:   T_l[A,B] > 0 only if a neuron of block l reads area B and writes area A   (the area graph)
           -> a leak that flows into a dead end of the area graph costs nothing
with norm: every area couples to every area (the norm divides all reads by the size of the whole stream)
loss  <=  1/4 ( what the logits lose directly  +  each leak carried to the logits )^2
```

Why it matters: this covers real networks, where every direction is read (K = {0}). It is the first bound in B.V that
uses the map's structure (which areas feed which). Two more results frame it:

```
TV7    one block writes a value of size delta taken from D; the logits read it with weight about 1/delta (norms on):
       every claim about every area has error <= delta^2 (also the rescaling-proof one), but the least loss stays
       0.218 for every delta  ->  NO bound from claims made one block at a time can exist
       (referees caught two earlier versions: one was only a rescaling, one had an incomplete map)
TV6.1  a new claim type (DM14, "carried leak"), whose error already includes N's gains:
       loss <= 1/4 (sum of sqrt(carried leak errors))^2      -> T5's graded form, in letter, for one fixed cut
TV8    if a computed check says the cut's stream error stays small, the norm couples each neuron by its largest read
```

The check (`model/tv_graded.py`, 40 dense random networks with K = {0}, 0 failures): the bound holds at every input
and for both a projection cut and a re-fit cut. A leak into a dead-end area adds exactly nothing. TV7's least loss stays
at 0.2181 while both block-read errors fall to about 1e-8. Honest numbers: the bound is loose, a median 10^4 times the
true loss without norms and 10^7 with them (worst-case Lipschitz steps ignore cancellation between neurons). A version
with operator norms is no tighter and loses the area routing. TV8 holds where its check passes (6 of 20 weak-leak
networks) but is about 3 times looser than TV6 there, because these random neurons read near their full strength.

**Step 5. The check (`model/tv_regions.py`, 40 planted networks, 0 failures).** The planted network has an input area,
a middle area, an output area with "dog", "cat" and "own" sub-areas, and one loud unread line, all rotated at random.
Every statement held to machine precision. One premise test: when a block reads a little of the cut area, the bound
fails badly (loss 0.3 to 2.6 against a bound near 0). So the premise "no block reads it" is needed.

| Status | Items |
|---|---|
| proved | F13, TV0, TV1, TV1.1, TV2, TV3, TV3.1, TV4, TV5 (side branch under S-purestream, R-streamregion, S-readoutarea, R-normconst) |
| referee | REFEREE-BV-2026-10-29.md: no false step; 1 structural result (TV1), 6 bookkeeping; scope gap (TV5) and overstatements fixed |
| proved later | TV6, TV7, TV6.1, TV8 (the graded case; referee rounds 2 and 3, all findings fixed; round-3 fixes not re-refereed) |
| open | how loose the bound is on clean networks; a tighter bound that keeps cancellation; attention and positions; ResNet and ConvNeXt streams |
| your call | card 3-23 |

## 32. One recursion over a tree of areas (2026-10-29; COARSE-THEORY.md B.H, D.16; card 3-24)

Your idea: a clean network gates. Earlier decisions are coarse, later ones finer, inside the area the earlier gate
opened. Zoom and depth are one recursion. Prove that it is optimal.

**Step 1. The object: a tree of areas (Definitions DH1 to DH3).**

```
layer 1:            [ car gate ]                     [ dog gate ]        <- areas: car photos | dog photos
                     /        \                            |
layer 2:      [ taxi ]      [ ambulance ]               [ dog ]          <- areas inside the car area
                 (+ J1)          (+ J2)                                  <- judgment units mark each area

group (one zoom level) = some children of one node, with everything under them
box   = k new neurons that read the group's top pre-activations, carried to the logits by copy neurons
```

Why it matters: one object holds both directions. A zoom step moves a group up the tree. Moving up the tree is also
moving to an earlier layer.

**Step 2. Local is optimal (Lemma TH0, Theorem TH1, proved).**

```
inside the car area, everything after the car gate is a function of the car gate's neurons      (TH0)
least loss of ANY grouping, at ANY mix of depths  =  sum over groups of one local problem each  (TH1)
switching one group changes only that group's box
```

Why it matters: this is your "local terms, not the whole network". It holds at every depth at once. The layer-L
splitting theorem TM1 is the special case. It needs one new claim, the branch claim: no neuron outside a subtree
reads it ("the dogs ... don't have to worry about cars"). Without it, removing the taxi neurons could switch on a
dog neuron they kept off.

**Step 3. Sibling merges nest (Proposition TH2 part 1, proved).**

```
along sibling merges (sizes not growing):  least loss of a finer level <= least loss of a coarser level
coarsest level exact  ->  restrict its neurons to each child's area  ->  every finer level exact
taxi + ambulance -> 1 "car" neuron;   the car neuron seen only on taxi photos = a valid taxi neuron
and the car neuron = (taxi neuron's pre-activation + ambulance neuron's pre-activation) - constant
```

Why it matters: "a smaller network is a coarser version of the larger one" is now a theorem for steps that merge
siblings. It generalizes T60.1 from the detector networks to any network with a tree map at error 0.

**Step 4. Lifts do not nest (Proposition TH2 part 2, proved, exact instance).**

```
x -> a = relu(x) -> g = relu(a_1) -> f_i = relu(g - i) -> gap 0, 1, 0, 1      (g = 1, 2, 3, 4)
box reading f's or g  (finer levels):   monotone in g  -> cannot make 0,1,0,1   -> lossy
box reading a = x     (coarsest level): relu(x_2) = 0,1,0,1                       -> exact
```

Why it matters: a depth step changes what the box can read. A later gate can be a bottleneck. So zoom and depth are
the same for the split (Step 2), not for nesting. The coarser network here is not a coarsening of the finer one.

**Step 5. Lossy merges: neuron by neuron is not optimal (Proposition TH3, proved, exact instance).**

```
child 1 (taxi):   signal on direction e_a (strength 4) and e_b (3)    -> its best 1 neuron keeps e_a
child 2 (amb.):   signal on e_b (3)                                    -> its best 1 neuron keeps e_b
parent, 1 neuron: best uses e_b on BOTH (6 > 4)                        -> least loss 2
built from the children's neurons: child 1 dropped e_b                 -> at least 3
```

Why it matters: being clean in TH1's sense does not make step by step optimal when a level must lose something.
What the parent needs from a child is a summary (its signal matrix), not the child's best neuron. Conjecture TH4: a
parent's least loss is a fast function of its children's summaries. The one-node problem stays hard (T57), so TH4
takes it as given.

**Step 6. Checks (exact fractions, 0 failures).**

| Script | Checks |
|---|---|
| `model/th_t61.py` | T61's instance has no tree map meeting TH1's premises (every error-0 region holds at least 5 of 8 points) |
| `model/th_tree.py` | a gated 2-layer network: all claims 0; TH1's assembly on 50 random choices; restriction exact; (r) identity; TH2 part 1 (a) on 50 lossy builds (first run failed: the script gave each judgment unit both groups' constants, not its own group's as in TH1's proof; fixed) |
| `model/th_lift.py` | the zigzag: finer levels at least 1/32 (D2, exact bound); coarsest exact |
| `model/th_break.py` | TH3: eigenvalues 4, 3 and 6, 4; values 3/2, 2, 3; numeric search agrees |

| Status | Items |
|---|---|
| proved | TH0, TH1, TH2, TH3 (side branch under R-tree, R-branch, S-plainff) |
| referee | REFEREE-BH-2026-10-29.md: no false step in any proof; 3 false statements as written, TH4 ill-posed, a hidden restriction (plain feed-forward networks, now node S-plainff); all fixed, not re-refereed. Content: TH0, TH1 bookkeeping; TH2, TH3 structural |
| open | TH4 (summaries passed up); the KL form of TH3; a graded TH1 (errors > 0); ResNet with the tree |
| your call | card 3-24 |

## 33. Summaries passed up the tree (2026-10-29; COARSE-THEORY.md B.P, D.16; card 3-26)

Your request: TH3 showed that naive recursion is not optimal. Pass up a summary instead, prove the same kind of
result, and keep the summary cheap and instant switching in the running.

**Step 1. What a node passes up (Definitions DP1, DP2).**

```
inside the node's area:   centred logits Y(x)  ~  B_w z_top(x)  +  (constant, judgment neurons)  +  r_w(x)
summary s_w = ( E_w ,  Sigma_w )
  E_w      = how much Y moves inside the area
  Sigma_w  = how much of that is a linear function of the top pre-activations, per output direction (n_y x n_y)
  eps_lin  = what is left over (r_w): the bending that no linear read copies
```

Why it matters: the table keeps every output direction. TH3's child dropped one direction because it passed up only
its best neuron.

**Step 2. One group: the table pins the least loss (Theorem TP1, Corollary TP1.1, proved).**

```
tail  = sum of the eigenvalues of (Sigma_1 + Sigma_2 + ...) below the k largest
Lin_S = 1/2 x tail + 1/2 x eps_lin
1/2 x tail   <=   least loss of ANY box   <=   Lin_S
box that reaches Lin_S: k neurons, always on, weights = top eigenvectors x each child's B_w
among boxes whose neurons stay on inside their areas, Lin_S is the least for ANY eps_lin   (TP1.1)
```

Why it matters: at eps_lin = 0 the least loss is a closed form (Eckart and Young), for boxes of any shape. With
error, the gap is at most eps_lin / 2. That bound is graded in the linear-read errors only (every other claim at
error 0), and it is for D2, not KL. When k is at least the rank of the summed table, the lower side is 0.

**Step 3. Merges add tables, and every level is least (Theorem TP2, proved).**

```
TH3 instance:   child 1: e_a 4, e_b 3      child 2: e_b 3
summary rule:   add -> e_a 4, e_b 6 -> keep e_b   -> loss 2   = the least
old rule:       child 1 cut to e_a 4 -> add -> e_a 4, e_b 3 -> keep e_a -> loss 3
cut-down tables lose at most half the trace they cut:   3 <= 2 + 3/2
```

Why it matters: the recursion now works in any order for sibling merges. In TH3's instance the old step-by-step rule
is the same rule on a cut-down table, and that explains TH3 with a number. A lift (a step to an earlier layer) does
NOT pass tables up: the higher node's table is computed again from N, in the same one-time pass. TH2 part 2's
instance: Lin = 1/20 at the finer levels, 0 at the coarsest.

**Step 4. Cost and switching (Proposition TP3, proved).**

| Task | Cost | Grows with N? |
|---|---|---|
| all summaries and build records | one pass of N over the inputs + one least-squares solve per node | once, up front |
| least loss and box directions of a level | add the tables, one eigen-solve: about n_y^3 | no |
| box weights of a level | linear in the width of the group's top and judgment neurons | yes, as any build of N's kind |
| change k in a group | longer or shorter prefix of one stored box | no |
| switch levels | select stored weights (as in TY3) | no |

Measured on a random 2-layer net (8 areas, 1024 neurons, 1000 logits, 3200 inputs; `model/tp_summary.py`): forward
pass 0.02 s, all 8 summaries 0.01 s, one level from summaries 0.04 s. These are timings on one laptop, not a proof.

**What is open.** A node whose neurons switch on and off inside its own area has a large eps_lin. There the least box
must bend, and that is the hard problem of T57. In 10 random instances with eps_lin > 0, a searched ReLU box beat the
summary build in 3. In one of them it reached the lower bound Lin - eps_lin / 2 exactly, so the bound can be tight.
(A search finds a box, not the least box, so these gains are at least what is shown.) KL is open.

**Check.** `model/tp_summary.py`: 149 checks, 0 failures (TH3 and TH2 numbers exact; 21 random instances).
Lean: `energy_split`, `trace_bound`, `topk_trunc`, `th3_summary` in `Coarse/Recur.lean`.


## 34. The stream, round 2: overlap and cancellation (2026-10-29; COARSE-THEORY.md F14, B.V.3, DM15, D.15; cards 3-23 answered, 3-25)

Shonu, on card 3-23: "residual streams are a little bit different ... you can still have overlaps and have meanings
not be crowded"; "you can create a quantum algorithm specifically in order to cancel out those terms".

**Step 1. An angle between two areas (Fact F14, Lemmas TV10, TV11; proved, Lean `gl_read`, `norm_scale_read`,
`crowd_no_read`).**

```
no norm:    change the stream's coordinates by ANY invertible A, fix reads (R A^-1) and writes (A W)  ->  same network
            so two areas at 45 degrees compute what two areas at 90 degrees compute
with norm:  the read becomes  s(x) * N's read,   s = sqrt((|h|^2 + eps) / (|A h|^2 + eps))   (one number per norm)
crowding:   a cut can drop D and keep the areas E whole, for every reader, iff D meets the SUM of E only in 0
            three lines in a plane meet pairwise in 0, yet you cannot drop one and keep the other two
```

Why it matters: your reading is right for the network's function. Overlap is free there, and crowding is the one
real limit. The paper's error measures still use right angles. Widening them is card 3-25.

**Step 2. The build cancels a leak (Theorem TV9, Propositions TV9.1, TV9.2; proved, Lean `ols_le`).**

```
a cut drops D. A neuron of N read  r . h  = r . (kept part) + r . (dropped part)
projection read (TV6):   r . (kept part)                      leak = - r . (dropped part)
TV9 read:                least-squares fit of N's read on the kept part
                         -> cancels every part of the leak that the kept content predicts on P

copy example (TV9.1):    D holds a copy of a kept direction -> projection cut loses log 2 - H, TV9 cut is exact
lower limit (TV9.2):     no cut recovers what the first write loses (TV7's c(s) is this limit)
```

Why it matters: this is your quantum picture with the cut as the algorithm. It gives a rule, and on random networks
it lowers the real loss (median 2.5 times, in 39 of 40). TV6's bound for it is not smaller, because TV6 takes sizes.

**Step 3. The network cancels too (Lemma TV13, Proposition TV14; proved).**

```
logit change of any cut  =  omega  +  sum over blocks l of  U' (I + J_L) ... (I + J_(l+1)) W_l S_l leak_l     (exact, TV13)
                                                             signs kept: a later block can undo an earlier leak
TV6   = the same sum with every sign and slope replaced by its size
TV14  = the same sum with N's OWN slopes (no run of the cut) + a remainder of second order in the leaks
```

Where TV6 loses, on random no-norm networks (median, total about 17,000 times the loss):

| Part | Factor |
|---|---|
| softmax step (Fact F11) | 6 |
| signs across blocks | 2.4 |
| signs inside the carrying | 31 |
| slopes set to their largest value, sizes per area | 33 |

TV14 against TV6, as the leaks shrink (one random network; t = the fraction of its leak):

| t | TV6 bound / loss | TV14 bound / loss |
|---|---|---|
| 1 | 1.1e5 | 4.9e6 |
| 0.1 | 5.6e4 | 2.4e4 |
| 0.01 | 5.2e4 | 274 |

Why it matters: about half of TV6's looseness (in orders of magnitude) is lost cancellation. TV14 keeps the signs.
It wins once the leaks are small, and it tends to the softmax step alone. Both bounds hold, so a rule uses the smaller.

**Step 4. The stream branch in T5's form (Definition DM15, Corollary TV6.2; proved).**

```
complete area map (DM15): per level and group S, an area V_S inside S with dim <= k_S, and the carried leak claims
rule:                     per level, the projection cut that keeps the V_S; store all levels; switch by lookup
promise 1 (all errors 0): loss 0, a closed form
promise 2 (errors > 0):   the rule's cut has loss <= (sum of sqrt(errors))^2 / 4
```

Why it matters: the stream branch now has both of T5's promises in form, as a side branch (laminar areas, node
S-laminar pending). It holds by construction: the map holds one certificate per level. The carried-leak errors hold
N's gains, so the graded bound is loose wherever TV6 is.

**Step 5. The three technical nodes, settled by the math.**

| Node | Outcome |
|---|---|
| S-readoutarea | DROPPED: it is the map's exactness condition |
| R-normconst | DECIDED by your delegation: re-setting a norm's constant = one more constant, unread direction (Lemma TV12); it widens the build class, so the referee kept it a decision |
| R-cutclaim | DROPPED: a claim names members of the map; completeness (DM15) carries the zoom |

| | |
|---|---|
| proved | F14, TV9, TV9.1, TV9.2, TV10, TV11, TV12, TV13, TV14, TV6.2, DM15 (definition) |
| checks | `model/tv_cancel.py`: 0 failures on 40 networks |
| referee | REFEREE-BVO-2026-10-29.md: no false step; 1 structural, 7 bookkeeping, 1 conjecture that was provable (now TV14). REFEREE-BVO2-2026-10-29.md: TV14 correct; 1 error (a round-1 "cost of one stream direction" claim, false by a 3-dimensional example, deleted); DM15 redefined. All fixed; round-2 fixes not re-refereed |
| open | DV3 (TV14 with norms); whether the remainder is small on trained networks; a map complete for every zoom at once (one claim per member); areas at an angle in the error measures; a Lean proof of the Taylor step |
| your call | card 3-25 (S-laminar: areas at an angle; S-smoothact: TV14 leaves out ReLU; S-nonorm: TV14 is proved without norms) |

## 35. The broken cases read as networks: reuse at a merge, linear areas at a lift (2026-10-29; COARSE-THEORY.md B.L; card 3-27)

Your point: a coarser network that is optimal exists and we understand how it forms, so the failures of TH3 and TH2
part 2 should be explained by what the network does. Worktree `smallnet`.

**Step 1. TH3 as a network.**

```
area 1: u_a -> e_a,  u_b -> e_b          area 2: u_c -> e_b
child 1 alone keeps e_a (4 > 3)          merged: e_b wins (3 + 3 = 6), because it is REUSED
```

Why it matters: the reuse shows only in the readers (equal logit weights on u_b and u_c). A child that decides from
its own area cannot see it.

**Step 2. Theorem TL1 (proved, sufficient only).** A merge built from the children's box neurons reaches the
summary value Lin when condition (K) holds: every direction the merge keeps is, inside each child's areas, one the
child's box writes. Least at error 0; within half the linear-read errors of least otherwise. (K) holds when every
child lost nothing, or when siblings write orthogonal output directions and the merge is no bigger than any child.
TH3 is where it fails.

**Step 3. TH2 part 2 as a network.**

```
x_1 -> a_1 -> g -> f_1, f_2, f_3 -> gap 0, 1, 0, 1     g, a_1, a_2 ON everywhere = identity
x_2 -> a_2 (no reader)                                  coarse box at G_3 reads past g
```

**Step 4. Theorem TL2 and Corollary TL2.1 (proved).** When the neurons a lift passes through are on inside their area
and the gate is flat there (the on-inside claim, Definition DL1), every finer neuron is a neuron one layer down. So a
lift whose box has room for every finer neuron never raises the least loss, for KL and D2. TH2 part 2 is this case: lifts not nesting
is normal. A lift by itself is not a coarsening. Only a smaller size is.

| | |
|---|---|
| proved | TL1 (sufficient condition (K), two cases, TH3 fails it), DL1 (definition), TL2, TL2.1 |
| checks | `model/tl_netread.py`: 0 failures. Orthogonal siblings: (K) and least in 40/40; graded (error > 0, sizes up to n_y + 1): reaches Lin in 40/40. Generic random weights: (K) in 2/40 (structure, not default) |
| Lean | `psd_parts_zero`, `weighted_sq_zero`, `relu_push_down`, `lift_compensate` in `Coarse/Netread.lean` |
| referee | REFEREE-BL-2026-10-29.md: constructions correct; 1 false converse (removed), a quantifier error and an undefined case for sizes >= n_y in TL1 (fixed), 1 proof gap in 2(b) (new argument), bookkeeping; one free gain: TL1 holds at any linear-read error, within half the errors of least. Fixes not re-refereed |
| open | a lift with fewer box neurons (the box must do the children's switching); a graded (K) |
| your call | card 3-27 (R-oninside) |

## 36. The stream, round 3: norms, every activation, areas at an angle (2026-10-29; COARSE-THEORY.md B.V.4, B.V.5, D.15; card 3-25 answered)

Shonu, on card 3-25: S-laminar "is certainly a development decision ... fine for you to make those decisions on your
own"; on smooth activations: "I just want a nice general method over these residual streams ... we don't need to place
those sorts of restrictions on anything"; "I'd like you to prove that norm's conjecture".

**Step 1. The norm is one more slope (Lemma TV15.2, Theorem TV15; proved, Lean `norm_hess_identity`, `norm_hess_le`,
`norm_hess_vec`).**

```
leak at block l
   |
[ norm slope  (I - n n^T) / |stream| ]   at N's stream with D removed      <- new
   |
[ cut's read r' ]
   |
[ activation slope ]                     at N's own pre-activation
   |
write -> next block -> ... -> [ norm slope ] -> logits

norm's curvature:  |second derivative| <= (2 / sqrt 3) / |stream|^2     (sharp; one identity in c and q)
```

Why it matters: real transformers have a norm before every read. TV14 did not apply to them. TV15 does, and the
conjecture DV3 is now a theorem.

**Step 2. A ReLU kink is a margin, not a ban (Definition DV4, Lemma TV15.1, Corollary TV15.3; proved, Lean
`relu_rem_pos`, `relu_rem_neg`).**

```
remainder modulus psi(t) = the largest error of "value + slope x change" for a change up to t

tanh, GELU     psi(t) <= (Lip' / 2) t^2              the old quadratic term
ReLU           psi(t) = max(t - |a|, 0)              0 while the change does not cross 0
any Lipschitz  psi(t) / t -> 0 at a differentiable point
```

Why it matters: one theorem now covers every Lipschitz activation. For ReLU, the slopes are N's own on/off pattern.
Without norms, when no window crosses a kink, the signed sum IS the logit change (rounding error 4e-15).

**Step 3. How tight it is (synthetic networks; Appendix).**

| Network (leaks scaled by t) | t = 1 | 0.01 | 0.0001 | old bound TV6 |
|---|---|---|---|---|
| ReLU, norm | 3e10 | 481 | 4.0 | 7e6 |
| GELU, norm | 1.5e11 | 284 | 4.1 | 1.2e7 |

The numbers are bound / true loss. The floor is the softmax step, about 3.6. At full leaks TV15 is worse than TV6,
so a rule uses the smaller of the two bounds.

**Step 4. Areas at an angle (Lemma TV16, Proposition TV17; proved).**

```
no norm:    a cut at an angle = the straight cut in other coordinates            -> nothing changes
norm:       they differ by one number per norm and input (Fact F14)
TV6:        holds for any direct split, with oblique projections                 -> angles allowed
energy:     per-area energies do not add at an angle (two 3-dimensional examples:
            every per-area claim at 0, every cut loses)                          -> one energy claim on the whole dropped part
```

Why it matters: your "subspace crossing" is now in the definitions. Without norms it is free. With norms, the map
makes its energy claim on everything a cut drops at once, not area by area.

| | |
|---|---|
| decided by me (your delegation) | S-laminar: widen; R-cutclass (new): a cut may drop any complement and keep the rest in any coordinates |
| dropped as moot | S-smoothact, S-nonorm (Theorem TV15 covers both) |
| proved | DV4, TV15.1, TV15.2, TV13.1, TV15, TV15.3, TV16, TV17 |
| checks | `model/tv_norm.py`: 0 failures, about 8,100 checks, 40 networks per case |
| referee | REFEREE-BV45-2026-10-29.md: TV15 correct; 3 errors and 2 gaps in the angle part and one corollary sentence, all fixed; not re-refereed |
| open | TV2, TV3.1, TV4 with one energy claim on the whole dropped part (at an angle, with norms); whether the remainder is small on trained networks; attention |

## 37. The stream, round 4: one map for every zoom (2026-10-29; COARSE-THEORY.md B.V.6, TV2.1, TV15 part 2)

You said: keep going until a decision is needed from you.

**Step 1. Claims that name one member (Definition DM16, Theorem TV18; proved, Lean `mulvec_mono`).**

```
before (TV6.2):  one claim per LEVEL: "dropping D_j loses nothing"        -> a new zoom needs new claims
now (TV18):      one claim per MEMBER and block: "member B's leak reaches the logits along no route of N"
                 for ANY set D of dropped members:  loss <= (sum over B in D of sqrt(claims of B))^2 / 4
```

Why it matters: T5 says no claim refers to the zoom. These claims do not, and one map serves every zoom.
The price: a member claim cannot see members cancelling each other, so its exact case is smaller.

**Step 2. The rule for every zoom (Corollary TV18.1; proved).**

```
one time:   compute the member claims (one pass of N on P)
per group:  pick which members to drop, least sum of sqrt(errors), keeping at most k_S directions
            = a small knapsack, exact, time (#members x dim S)
switching:  stored cut per level
exact:      when the claims of the members the map certifies are 0
```

**Step 3. Angles everywhere (Proposition TV18.2, Corollary TV2.1; proved).** With norms at an angle, each member's
claim also carries its cross energies with the other members. No premise is needed. TV2 (readout areas) works at an
angle with the oblique cut and one energy claim on the whole dropped part.

**Step 4. Trained networks: where the bound loses (TV15 part 2; synthetic, Appendix).**

| measure (norm form, GELU, 2 directions dropped) | bound / true loss |
|---|---|
| signed first-order sum alone | 4.8 |
| TV15 as before | 8.6e13 |
| TV15 part 2 (signed drift) | 47 |
| TV6 | 3.1e11 |

Why it matters: on trained networks the first-order signed sum already tracks the loss (within 3 to 7 times with
norms). The remainder bound is the weak link. Part 2 fixes it for small cuts, not for large ones.

| | |
|---|---|
| decided by me (your delegation) | R-normconst widened (a cut's norm constant may fall below N's, above 0; used by TV2.1 and TV17 part 1) |
| proved | DM16, TV18, TV18.1 (5 parts), TV18.2, TV2.1, TV15 part 2 |
| checks | `model/tv_member.py` 0 failures (about 4,160); `model/tv_norm.py` 0 failures (about 8,700) |
| referee | REFEREE-BV6-2026-10-29.md: inequalities correct; 1 error, 3 gaps fixed; not re-refereed |
| open | a tight remainder for large cuts; signed (TV15-type) claims that name one member; attention |

## 38. Deep layers: a clean subtree is linear in its top neurons (2026-10-29; COARSE-THEORY.md B.L.2; card 3-27)

Your ask: "continue until you can find the theorem which finds the optimal small network for deep layer cases".
Worktree `smallnet`.

**Step 1. What the network does inside a clean area (Theorem TL3, proved).**

```
inside an area:  every neuron that is ON acts as the identity      -> the layers are linear there
a gate:          pre-activation = kappa on its area, zeta off it   -> its area's indicator is LINEAR in the layer above
=> the whole subtree is an affine function of its top neurons, at any depth
```

Why it matters: depth adds no switching for the coarse network to make. The one-layer closed form (TP1) then holds
at every depth.

**Step 2. What this needs (Corollary TL3.2, proved).** Every split below the top must be a flat 0/1 direction of the
top layer: the top layer already carries the area code of every finer gate. On points in general position, with more
than twice as many points as top neurons, no split is left: the subtree is a stack of linear maps.

**Step 3. A lift adds one table (Proposition TL4, proved).** Parent table = children's tables + a between table of the
children's masses and mean outputs (the votes of the gates the lift swallows; law of total variance).

**Step 4. The optimal small network at every level (Corollary TL5, proved).**

```
per group: k neurons, ON over the whole area, read the top neurons along B^T q, write q
q = top-k eigenvectors of the group's table;  loss = 1/2 x (energy - sum of the top k eigenvalues)
tables: merge = add;  lift = add + between table;  switch = select stored weights
```

**Step 5. With errors (Theorem TL6, proved).** Each node's linear-read error is at most (weight norms, shelter values,
1 / gate margin) times the square of a sum of absolute defects. A weak gate amplifies. This bounds the gap to the least
only when the errors sit in on-inside claims and in shelter claims below the tops.

| | |
|---|---|
| proved | TL3, TL3.1, TL3.2, TL4, TL5, TL6 |
| checks | `model/tl_deep.py`: 0 failures (claims at 0; eps_lin = 0 at nine nodes; a switching control gives eps_lin > 0; TL4 to 1e-11; table = box at every level of a merge-and-lift zoom; no optimiser run beat the closed form; TL6's bound held at every noise level, loose by 10^4 to 10^9) |
| Lean | `gate_indicator`, `relu_sheltered`, `within_between`, `gate_err_identity`, `unit_err_identity` in `Coarse/Netread.lean` |
| referee | REFEREE-BL2-2026-10-29.md, two rounds. Round 1: TL3, TL4 correct; Conjecture TL6 false (normalised errors) -> replaced by Theorem TL6; scope (TL3.2). Round 2: TL6's proof correct; premise, gap claim and recursion input fixed. Round-2 fixes not re-refereed |
| open | a tight bound with errors; gates that read features varying inside their area; measuring the claims' errors on a real network (Part C) |
| your call | card 3-27 (R-oninside) |

## 39. One budget for neurons and stream, and the optimality audit (2026-10-31; COARSE-THEORY.md B.V.7, B.Y.2; cards 3-29, 3-30)

Your idea: a developer coarsens a capability, so the neuron cut and the stream cut should be one cut. Your bar: the
optimum that well-run distillation reaches, with errors that depend only on how clean the network is.

**Every result now carries a label.**

| Label | Meaning | Meets your bar? |
|---|---|---|
| OPT-KL | gives the least distillation loss, or a build within a cleanliness-only error of it, which reaches the least exactly when the network is clean, even where the least is above 0 | yes |
| EQ | proves two least losses are equal | yes |
| OPT-D2 | the least of a different loss (squared logits) | no |
| ONE-BUILD | a bound on one rule's loss, exact only when nothing is lost | no |

**Step 1. The stream is free once the neurons are coarsened (Theorem TV19.2; label EQ; proved).**

```
coarse network: each group of k neurons writes at most k stream directions
stream on P lives in:  E_N (first write + biases + kept neurons' writes)  +  the new neurons' writes
cut the stream to that span  ->  same neurons, same outputs, at every input       (any loss, any activation, norms too)
=> least loss with a narrower stream = least loss of the neuron zoom, once width >= dim E_N + sum k_S
   (both directions proved: cut a wide optimum down, or lift a narrow one up)
```

Why it matters: your "no extra loss from the stream" is exact, for the optimum itself. The stream adds no error of its
own. Its cost sits on the writers.

```
the limit (Proposition TV19.4):  a full first write (a transformer's embedding) fills the stream  ->  nothing is free
                                 until the embedding and the kept writers are coarsened too
read side (Proposition TV19.1):  without a norm, also drop what nobody reads (minimal realization); with a norm,
                                 only when that part's energy is the same at every input
```

**Step 2. In ResNet, a capability is one group (Theorem TY4; label EQ; proved by reading TH1's proof).**

```
ResNet block:  h_next = relu( h + conv-bn( relu( conv-bn(h) ) ) )     <- ReLU on every stream channel
=> a stream channel is a unit, like a neuron
capability = one subtree group: inner channels + stream channels of several blocks
the least KL loss = sum of one local problem per group     (exact, when the map's claims hold)
across a stage border (56 -> 28 -> 14 -> 7 positions) a value is carried by an average pool, which keeps its mean
```

Why it matters: the true optimum of a ResNet now splits into one small problem per capability. That is a
reduction, not a solution: each small problem is still open for KL (Step 3). No change to the trunk's wording is
needed there.

**Step 3. The audit: our deep closed forms are the optimum of the wrong loss (Proposition TY5; proved).**

```
a group replaced by nothing, one area (its gate value constant there):
  true optimum (KL):        logits = log( mean of N's softmax )
  our closed forms (D2):    logits = mean of N's logits
  extra loss:               KL( mean softmax || softmax(mean logits) )         exact
example with EVERY claim at error 0:  (0.9, 0.1) and (0.5, 0.5)  ->  0.7 vs 0.75  ->  extra loss 0.0064
any size: only a factor 1 / (2 mu), mu = smallest probability (<= 1/1000 on ImageNet)
```

Why it matters: Theorem TP1 and Corollary TL5 ("the optimal small network at every depth") are labelled OPT-D2. They
miss your bar even on a perfectly clean network. I chose D2 on 2026-10-25 (node R-obj). You never decided it. Card
3-30 asks you now.

**What meets your bar today, honestly.** Only two kinds of result give the KL optimum itself: cases where nothing
is lost, and TY5 part 1 (a group replaced by nothing). Every closed form for a group that keeps k >= 1 units is
OPT-D2. The EQ results reduce the optimum to small problems. They do not solve those problems.

| Existing result | Label |
|---|---|
| TM1, TH1, TY4 (the least splits into one problem per group) | EQ (a reduction of the optimum, not the optimum) |
| TV19.2 (the stream is free above the write budget) | EQ |
| TY5 part 1 (a group replaced by nothing: log of the mean softmax) | OPT-KL, closed form |
| TP1, TL5 (deep closed forms) | OPT-D2 |
| TY2 (merge rule), TV6, TV18 (stream cuts) | ONE-BUILD |

| | |
|---|---|
| proved | DV5, DV6, TV19, TV19.1, TV19.2, TV19.4; DY3, TY4, TY5 |
| referee | REFEREE-BV7-BY2-2026-10-31.md: 3 main gaps (TV19.2 was empty; copies across a stride; TY5's premise), 12 smaller; all fixed; round 2 below |
| checks | `model/tv_written.py`: 0 failures (about 1,160 checks, with the lift); `model/ty_stage.py`: 0 failures (about 1,450, with the stride copy) |
| Lean | `written_step`, `quotient_read` in `Coarse/Stream.lean`; `kl_split` in `Coarse/Merge.lean` |
| open | the KL optimum of one clean group with k >= 1 units (no closed form known); a graded form of TY4; a stream narrower than the write budget |
| your calls | card 3-29 (S-skipff, R-jointgroup), card 3-30 (R-obj) |

**Is the ResNet run ready for next session?**

| Piece | State |
|---|---|
| the split of the true optimum on a ResNet stage (TY4, EQ) | proved, exact case only |
| the cache | `cache/r18` has the last block only; a stage cache (both blocks of layer4, inner and stream) is about 4 GB more |
| what the run can measure | the claims' errors on layer4 (district, shelter, judgment, branch), and the gap between our build and a well-run distillation of the same small stage at each size |
| what it cannot yet do | test a closed form for the KL optimum: we have none for k >= 1 |
| known risk | the 2026-10-28 test found no clean map at ResNet-18's last block. A stage map may also be empty. That would be a result |

## 40. The theory moved to the least KL (2026-10-31; COARSE-THEORY.md B.I, D.17; card 3-30 answered, card 3-31)

You decided card 3-30: KL is the only target. Node R-obj is DROPPED. Every squared-loss (D2) result stays as a record,
labelled OPT-D2.

**What moved to OPT-KL, and what did not.** The answer depends on how many units a group keeps.

| Group keeps | Least KL | Label |
|---|---|---|
| 0 units (capability removed) | closed form at any depth: on each area, the log of N's mean probabilities. Its loss is the area's information I_w. Merge = add. Lift = add + one KL term per child (Theorem TI1) | OPT-KL |
| at least rank-of-table units (capability kept whole) | at most a quarter of the linear-read error, 0 when clean; TP1's box gets there (Theorem TI2) | OPT-KL |
| a mix of the two on one level | the removed groups' information, exact when clean, also when the loss is > 0 (Corollary TI2.1, node S-keepdrop). Only the linear-read errors are graded; every other claim must be exactly 0 | OPT-KL |
| between 0 and the rank (capability squeezed) | open. No closed form. The D2 table does not fix it (Proposition TI5). To second order it is the eigenvector answer in the Fisher metric of the area's mean output (Theorem TI4) | open |

```
squeezed case, smallest example (Proposition TI5 part 2): 3 classes, 1 unit, every claim at error 0
  mean output uniform        -> KL least ~ a^2/12   (every direction ties, as for D2)
  mean output (1/2,1/4,1/4)  -> KL least ~ a^2/16   along "class 1 against the rest"
                                a D2-least build along (0,1,-1): ~ 3a^2/32, 1.5x worse
```

**Plainly: the deep closed form for a squeezed group is still not the optimum.** TP1 and TL5 are the least of D2
there. I do not have a KL closed form for that case, and I expect none of the same kind (Conjecture DI1: it may be
NP-hard).

**A surprise from the check.** In the uniform example above, a unit that switches off on part of its area beats every
unit that stays on (about 1.2% lower at a = 1/2, 4.5% lower at a = 1; found by a local optimiser, not certified). For
D2 that never happens (TP1 part 1). If it holds (Conjecture DI2), the "one coarse neuron on over its whole area"
picture of TL5 belongs to D2, not to the KL optimum.

**The deep test network says the same.** On `model/tl_deep.py`'s 3-layer net, removing the whole top group costs
KL 0.58 at the optimum (TI1). TP1's one-unit D2 box costs KL 2.63: worse than keeping nothing. So for KL, the D2
closed form is not a good stand-in even as a build.

| | |
|---|---|
| proved | DI1 (table), TI0, TI1, TI2, TI2.1, TI3, TI4, TI5 |
| conjectures | DI1 (hardness of the squeezed case), DI2 (a least unit may switch) |
| checks | `model/ti_info.py`: 0 failures (chain rule exact in sympy; table recursion 4e-16; KL of TP1's box <= Lin/2 always, <= eps_lin/4 at k >= rank, also on the switching control). `model/ti_open.py`: 0 failures (means match at the least; L/a^2 -> 1/12 and 1/16; least line -> (2,-1,-1); D2 line -> 3/32) |
| Lean | `kl_chain` (new) and `kl_split` in `Coarse/Merge.lean` |
| referee | `REFEREE-BI-2026-10-31.md`: every core proof correct; 1 false summary sentence ("D2's directions are KL's only when the mean is uniform": wrong, it is "if"), 1 overclaim (TI2.1 does not yet meet T5's clauses: only the linear-read errors are graded, and completeness is not defined for those claims), 2 hidden premises, 1 unproved sentence; all fixed, not re-refereed |
| your call | card 3-31: may a level only remove or keep whole capabilities for now (rec yes), with "squeezing" as the next research step |

**For the ResNet example.** Under keep-or-drop levels, every level of a ResNet stage zoom is least for KL in closed
form (TY4 carries TI1 and TI2 to skips). So the next session can measure, on layer4, the claims' errors and the gap
between our keep-or-drop build and a well-run distillation of the same size. Squeezing a capability stays open.

## 41. Row 3: squeezing is packing (2026-10-31, night; ROADMAP.md session 1; COARSE-THEORY.md B.Q, D.18; card 3-31 answered, card 3-32)

You answered card 3-31: keep-or-drop "is the same as pruning", so fix row 3. You set four sessions to the ResNet goal
(`ROADMAP.md`) and five standing rules (exact is exact, graded is useful, trained nets are clean, no tradeoff you did
not choose, "not possible" needs a theorem).

**Step 1. The plan's first guess was wrong, and the test showed it.** The guess: a squeeze picks which
sub-capabilities keep their detail (a tree knapsack). The test `model/ts_sep.py`:

```
two areas, each confident in its own classes (dog area -> dog classes, car area -> car classes), one unit
                      outside mass   best unit for one area   one unit for BOTH areas
no confidence (m=0)       0.53            0.0546                    0.0505
medium (m=3)              0.055           0.0986                    0.0088
high (m=6)                0.003           0.1027                    0.0005
```

Read as a network: each area's own offset pushes the other area's classes down. Then one output direction means
"which breed" on dog photos and "which make" on car photos. The only cost is the probability each area still gives
to the other's classes.

**Step 2. What is proved (Theorem TQ1). Label: ONE-BUILD. This does NOT meet your bar yet.**

```
a squeezed group, each area moving its own classes in at most k directions, class sets disjoint:
   sum of each area's own least   <=   least of the group   <=   packed build   <=   sum of active-class errors
packed build: max (area's own rank) units, shared by all areas   (row 2 needed the rank of the whole group)
```

Why it matters: coarsening needs about as many units as the largest sub-capability, not their sum. This is the real
coarsening you asked for, not pruning. But plainly: the lower side is 0 here, so this bounds one build. It does not
give the least. The referee caught that I had labelled it OPT-KL.

**Step 3. Withdrawn: "row 3 is never exact".** I wrote that this was a proved fact of softmax. The referee showed
it was wrong. My proof only showed that MY error formula is never 0. Row 1 is exact with a loss above 0, so a
positive loss does not rule out an exact form. What is proved (Proposition TQ2 part 2): every squeezed group loses
something. Whether row 3 has an exact form is open, and it is the next step.

**Step 4. Is the graded bound useful?**

| Where | Bound / true least | Verdict |
|---|---|---|
| synthetic family, confidence growing (`model/ts_tree.py` E) | 3.58, 3.42, 3.41, 3.41 | first-order tight there (Conjecture DQ1, not yet proved) |
| ResNet-18, proxy areas (`pc_active.py`) | bound / area information 1.6 to 3.6 | NOT useful yet |

**Step 5. Realism on ResNet-18 (Part C, H-active).** Areas are a proxy: classes clustered by N's own confusions, an
area = the photos whose top class lies in a block (no unit-level map of ResNet-18 exists, 2026-10-28).

| Blocks | Outside mass | Packed build keeps of the information |
|---|---|---|
| 100 (k = 38) | 0.166 | 51% |
| 300 (k = 14) | 0.205 | 12% |

The premise holds: the areas are confident. But the plain packed build keeps little. Is that the network or my
construction? One fitted head of the same form (shared directions, a read and an offset per area; local fit, not
certified) answers it (`pc_active_opt.py`):

| Areas | Shared units | Fitted head keeps of the information |
|---|---|---|
| 300 | 14 | 72% |
| 300 | 2 | 40% |
| 100 | 38 | 90% |

So the network allows cheap squeezing. My packed build is the weak part: it lines up the shared directions in a fixed
way instead of fitting them to each other. The bound of TQ1 is about that fixed build, so on ResNet-18 it is not
useful yet. Next: an aligned packing, and its bound.

**Step 6. You said I had not tried different approaches. Here is every approach of this session.**

| # | Approach | Result |
|---|---|---|
| 1 | tree knapsack: pick which parts keep detail | refuted (TQ2 part 3) |
| 2 | packing, one canonical build | bound on one build (ONE-BUILD); weak on ResNet-18 |
| 3 | gating by the next layer's area-specific readers | does not apply at the top: each area still needs its own last-layer units, or shares and leaks; needs a build class outside DH3 (card 3-24 option (b)) |
| 4 | the clean limit, both sides (Theorem TQ3) | PROVED: as areas become confident, the least of a squeeze tends to the sum of each area's own-block least, from below and from above |
| 5 | per-area scales on ResNet-18 | close only 10% to 17% of the gap: scales are not the alignment |
| 5b | one full k x k alignment map per area, own directions from N (an SVD) | closes ALL of the gap in the complete case: 100 areas, 38 units keep 90%; 300 areas, 14 units keep 73%, as good as a head with everything fitted. The alignment is the one missing piece |
| 6 | lining up the shared directions | my first guess (balancing, from control theory) was WRONG (referee). The right closed form is PROVED (Lemma TQ4): pair each area's strongest direction with the other area's weakest. That it is the cost of packing, to second order, is Conjecture DQ2 |

**Step 7. The clean-limit sandwich (Theorem TQ3; label LIMIT, two-sided).**

```
sum over areas of (1 - outside mass / weight) x own-block least   <=   least of the squeeze
                                                                  <=   sum of own-block leasts + leakage (<= 2 D mu per area)
```

Checked (`model/ts_tq3.py`, 0 fails): one area squeezed below its own rank, so the least stays 0.083 > 0. The ratio
least / sum of own-block leasts is 1.47 (no confidence), 1.03 (m = 4), 1.0006 (m = 8). At m = 8 the two sides are
within 0.4% of the least. That is your "reach the optimum even where it is above 0", in the confident limit. It is not
yet at a finite, given confidence: there the leakage term is a bound, not the least.

What is left, as one finite problem: at finite confidence, how the shared directions line up (the leakage). The fit
inside each area is an exact convex program. Card 3-32 item 5 asks you whether "directions from one small fixed-size
problem plus an exact convex fit" counts as the trunk's "exact closed form".

| | |
|---|---|
| proved | DQ1 (a summary of N, not a claim), TQ1 parts 1 to 4 (part 2 is ONE-BUILD), TQ2 parts 1 to 3 (part 2: every row-3 least is > 0; part 3: the tree knapsack fails), TQ3 (the clean-limit sandwich, always-on boxes), TQ4 (reverse pairing) |
| conjecture | DQ1 (first-order tightness, comparable confidence), DQ2 (cost of sharing, second order, via TQ4), DQ3 (the recursion of TQ3) |
| checks | `model/ts_sep.py` 0 fails; `model/ts_tree.py` 0 fails (243 random instances, positivity, ratio 3.41) |
| Lean | `pack_step`, `phi_nonneg`, `phi_le_neg` in `Coarse/Pack.lean` |
| referee | round 2 `REFEREE-BQ2-2026-10-31.md`: TQ1, TQ2 parts 1 and 2, TQ3 parts 1 and 2 correct; 3 errors fixed (TQ2 part 3's network broke a shelter claim, now fixed; DQ2's balancing was wrong, replaced by the proved reverse pairing TQ4; the recursion Remark was overclaimed, now Conjecture DQ3). Round 1 `REFEREE-BQ-2026-10-31.md`: core proofs correct; 3 errors (the "never exact" claim; the OPT-KL label; DQ1 is not a claim), several gaps (TQ2 part 1 proof, the instance as a network, tightness fails for uneven confidence); all fixed in the paper, not re-refereed |
| dropped from the plan, said plainly | attainment for switching boxes (TQ2 part 2 uses a direct lower bound instead); the trained small-net instance of the L-sep test (not run) |
| your calls | card 3-32 (R-active, S-squeezecomplete, the meaning of "useful"; and the open question: an exact form for row 3) |
| next | the least of the packed form with the shared directions fitted to each other: find its exact form, or its bound with a matching lower bound |

## 42. Row 3: the leak is the whole squeeze (2026-11-01; ROADMAP.md session 1 continued; COARSE-THEORY.md B.Q, D.18; card 3-32 answered by delegation)

You asked two things: close row 3 before the roadmap goes on, and write into `ROADMAP.md` that it must end in the full
Cale experience on the whole of ResNet, with no detraction. The second is done (the goal, a new standing rule, session
3 and 4 rows, a hard gate in place of the old "numbers, not optima" line). The first is NOT done yet. Here is what
moved.

**Step 1. A finding, not a decision: there is no "below the top" squeeze.** In our builds every squeezed group writes
its coarse units straight to the logits (Definition DH3, item 4). So the spill between areas cannot be gated away
by a later layer. It is the problem itself. (Letting copies be gated per area would let k units act as one head per
area, so k would stop meaning size. That is not a direction.)

**Step 2. Your card 3-32.** You said these were math details. You were right. I decided them (DECIDED by delegation,
your quote in `DECISIONS.md`). Item 5 now reads: "exact" = a value with a PROVED matching lower bound. The number of
unknowns is not the test.

**Step 3. The main result: Theorem TQ5. Label LIMIT-1 (exact first-order term, both sides). It does NOT meet your bar.** Its scope: boxes that stay on, every area made more confident by the same amount, no rate.

```
a complete squeeze (each area's own movement fits in k units), areas made more confident by m:
    least of the squeeze  <=  e^-m x LEAK LEAST            (every m)
    least of the squeeze  >=  e^-m x LEAK LEAST - o(e^-m)  (as m grows)
LEAK LEAST = best way to share the k units: each area's movement spills onto the other areas' classes,
             each spill weighed by how likely those classes still are there
```

Why it matters: when areas are confident, the squeeze loses exactly the cost of the spill, and nothing else. The build
made from the leak least's best point is optimal to first order. Check `model/tq_leak.py`: 200 point checks of the
upper side, 0 fails; the ratio least / (e^-m x leak least) goes 0.25 -> 0.74 -> 0.96 -> 0.994 -> 0.999 as m = 1, 3, 5,
7, 9 (4 instances), never above 1.

What it does not do, plainly: the leak least is still a least over output spans (non-convex in the span, convex at a
fixed span). It is not a closed form. And it is a statement about the confident limit.

**Step 4. When does the leak least have a closed form?**

| Case | Result |
|---|---|
| two areas, small movement | reverse pairing (TQ4, 2026-10-31) |
| equal spills, each area moves in one direction | PROVED (Lemma TQ6): least = n^2/k - n times one constant; the k units hold the n areas as an equal-norm tight frame |
| three or more areas, unequal | the pairing does NOT carry over: `model/tq_ngauge.py` finds cases where mixing directions wins (by 0.1% and 0.4%) |
| many areas, unequal spills | Conjecture DQ4: hard (NP-hard in the number of areas). Route: Peeters' min-rank. Not proved, so not a fork |

**Step 5. ResNet-18 (Part C, H-active, proxy areas).** `pc_active_free.py`, 100 areas, 38 units:

| Question | Result | What it means |
|---|---|---|
| does freeing every direction help? | 2.3% better than N's own directions plus alignments | the own directions are nearly pinned. Good for a fixed-size form |
| what do the alignments look like? | not a pairing (reverse order 4% of the energy), not the identity (2%) | no simple claim to name |
| how far is ResNet-18 from the confident limit? | its true loss is 0.34 of TQ5's first-order term | FAR. By your rule, the confident limit is not the regime of this trained network |

So, said plainly: TQ5 is correct and checked, but on ResNet-18 it is the wrong regime. The object to solve next is
the least at the network's own confidence. TQ5's identity (the exact split of the KL into own block and spill, (*) in
its proof) holds at every confidence, so it is the starting point.

**Step 6. Your hint: the essay, and what clean structure gives for free.** The essay, in its words: "With less
room, the network doesn't fold unrelated things together; its neurons just cover coarser-grained concepts", "Fewer
neurons make generalizing neurons coarser, never fractured", and from its toy: "PCA-like solutions, combining related
inputs into shared representations". My packed builds fold unrelated details ("breed" on dogs, "make" on cars) into
one unit. The essay says trained small networks do not do that. They keep what the areas share.

The test on ResNet-18 (orientation only): keep the top shared directions of the whole group's movement (one eigenvector
problem, from N alone), then give each area its exact convex fit at that span (TI3 part 2).

| Units | Packed with alignments | Shared span, each area reads its whole movement |
|---|---|---|
| 38 | 0.097 of the information lost | 0.027 |
| 14 | 0.265 | 0.104 |

The control, same full reads, only the span differs: the canonical packed span loses 0.156 (38 units) and 0.200 (14
units). So the shared span wins by about 6 times and 2 times. Not yet tested: the aligned packed span with full reads.
This is orientation, not proof.

**Step 7. The mathematics of that hint (Proposition TQ7). Label SECOND-ORDER, graded. It does NOT meet your bar.**

```
if every area sees the classes through nearly the same Fisher metric (within 1 +- eta of one metric):
   the least, to second order, sits between (1 - eta) and (1 + eta) times one eigenvalue tail (closed form)
   the shared span is within (1 + eta)/(1 - eta) of the least
```

Check `model/tq_shared.py`: 120 instances, 0 fails (true ratio at most 1.15 where the bound allows 9).
Realism, plainly: on ResNet-18 the premise FAILS. The areas' metrics differ by a factor of about 10^8 (eta = 0.99999999).
So TQ7 is true, but it does not explain why the shared span wins on ResNet-18. That is now Question DQ6. Its candidate
claim: each area's own-block movement is built from the group's common patterns (the essay's reuse). What is missing is
a matching lower bound.

| | |
|---|---|
| proved | TQ5 (LIMIT-1), TQ6 (matrix fact), TQ7 (SECOND-ORDER), Definition DQ5 |
| conjecture | DQ4; Question DQ6 (why the shared span works); Question under DQ2 (pairing fails for 3+ areas, Part C) |
| checks | `model/tq_leak.py` 0 fails; `model/tq_ngauge.py` 0 fails; `model/tq_shared.py` 0 fails |
| Lean | `log_one_add_ge`, `sq_sum_le_card_fin` in `Coarse/Pack.lean` |
| referee | `REFEREE-BQ3-2026-11-01.md`: TQ5 correct after repairs (one false inequality in Step A, three gaps, all fixed); TQ6 correct; DQ4 was not well posed (fixed: promise problem, binary input, five gaps of its route now listed); the many-area Remark had only a script behind it, so it moved to Part C as a Question; TQ5 narrowed to "always-on boxes, along N_m, no rate"; all fixed, not re-refereed |
| dropped from the plan, said plainly | B5 (no case exists); the trained small-net instance (not run again); the certified global optimum for TQ5 (Part A certifies the upper side point by point; the lower side is checked only by a local optimiser) |
| row 3 closed? | NO. Open: (1) the least at the network's own confidence; (2) a lower bound that matches the shared span when the areas' metrics differ (DQ6), the most promising lead; (3) the leak least in closed form (or DQ4 proved); (4) an area squeezed below its own rank (DI1, DQ3) |
| your calls | none needed now |

## 43. Row 3: the shared span is the object; what the mathematics says so far (2026-11-02; plan i-think-we-are-agile-raccoon.md, second version; COARSE-THEORY.md B.Q, D.18)

You said: read the essay, use what clean structure gives for free, and "mathematics are the only foundations". You
also asked for heartbeats on long runs (now a 15-minute Monitor at every long launch; saved as a rule).

**Step 1. Is the shared span the least? (orientation, ResNet-18 proxy, full reads)**

| Units, start | At the start span | After free search over the span |
|---|---|---|
| 38, shared | 0.028 | 0.025 (best found) |
| 38, packed | 0.159 | 0.067 |
| 38, random | 0.111 | 0.062 |
| 14, shared | 0.108 | 0.097 (best found) |
| 14, packed | 0.203 | 0.158 |
| 14, random | 0.231 | 0.166 |

The shared span is in the basin of the best build found: free search improves it by only about 11%. Other starts stop
far higher. And 61% (38 units) and 54% (14 units) of the loss sits on the quarter of photos where N's top two classes
are closest: the essay's "fine distinctions between things the network is already confused about".

**Step 2. The mathematics.**

| Item | What it says | Label |
|---|---|---|
| Lemma TQ8 | At each point, the KL of a change is half its variance under N's own prediction, times a factor between g(-omega) and g(omega) (omega = the change's spread across classes). Exact, no area-mean metric. 20,000 checks, 0 fails | an inequality about KL itself |
| what TQ8 does NOT give, said plainly | the planned two-sided "Theorem TQ8" was FALSE as planned: a competitor can push very unlikely classes far down at almost no cost (true cost about 0.15 of the quadratic). So the quadratic least is a lower bound only over builds of bounded spread | statement revised |
| TI4's area-mean model on ResNet-18 | it prefers a different span (0.83 vs 0.39): the movement is not small there; the per-point form is the right one | orientation |
| Lemma TQ9 | a certificate: grid of output directions, exact dual values of the convex fits, a step from each grid point to its cell; a proved lower bound on the least | certificate |
| TQ9 in practice | valid but far too weak: on a 4-class hand-built net no cell was pruned in two rounds; it would need millions of cells. The trained small net was therefore not reached (said plainly, third time it slips; the reason is the weak certificate, not time) | negative, practical |

**Where row 3 stands.** NOT closed. The shared span is the right object on ResNet-18 (orientation), and the
mathematics now has the exact per-point tool (TQ8) and a valid certificate (TQ9). What is missing: (1) a lower bound
on the least that matches the shared span (a second-order certificate, or a claim under which the weighted low-rank
problem has a closed form); (2) why the shared span works when the areas' metrics differ (Question DQ6).

| | |
|---|---|
| proved | TQ8, TQ9 (and TQ7, TQ5, TQ6 from LADDER 42) |
| checks | `model/tq_point.py` 0 fails; `model/tq_cert.py` (ran, did not certify) |
| referee | `REFEREE-BQ4-2026-11-02.md`: TQ7 and TQ8 correct; TQ9 had two false steps (a missing on-inside premise, which made the certificate unsound without it; a wrong reason for 'no duality gap', now Rockafellar Cor. 31.2.1) and gaps (exactly feasible dual family, attainment, notation, 'always-on least'); TQ7 is NOT first-order tight (bound O(eta), true gap O(eta^2)); TQ8's lower side now carries the factor g(-omega*) and an explicit instance; all fixed, not re-refereed. Referee's verdict, said plainly: this round adds no structural result |
| your calls | none needed now |
| next | a second-order certificate (curvature of the least near its minimum) so the trained small net can be certified; then the claim of DQ6 |

## 44. Row 3: the "small dropped part" plan, checked against its own definition of done (2026-11-02; plan i-think-we-are-agile-raccoon.md, third version)

You asked for a plan that gets row 3 done. The plan fixed six pass/fail items in advance. Here is each one, plainly.

| # | Done means | Result | Pass? |
|---|---|---|---|
| D1 | a theorem: the least as a closed-form leading term plus a smaller remainder, from N | Theorem TQ10 PROVED and refereed: along a family where the dropped part shrinks by e, the least = e^2 Phi + O(e^3), Phi a convex weighted least-squares problem (normal equations), each point in N's own Fisher metric; covers switching boxes as a two-sided bound. `model/tq_small.py`: 25 instances, 0 fails, both ratios -> 1. But the remainder's constants grow with powers of 1/(smallest probability) | PARTLY (no explicit remainder; no Lean) |
| D2 | the remainder evaluated on ResNet-18 and below the leading term at 38 units | ResNet-18 is not in TQ10's regime: with the metric of the kept-only prediction, Phi = 0.120 against a best build of 0.025; with N's own metric, Phi = 0.022, close in VALUE, but the build from it loses 0.104 (the quadratic over-penalises cheap moves on unlikely classes). No explicit remainder bound exists yet | NO |
| D3 | a claim type with an error from N, tied to k_S by completeness | Definition DQ7, the rank claim (legal after the referee's fix: it names map nodes, not a zoom group); its graded use is Corollary TQ10.1 (ONE-BUILD). Completeness: not used (with r = k_S every row-3 group has error > 0) | PARTLY |
| D4 | certified optimum on a hand-built and a trained small net | the certificate (Lemma TQ9 with the new dual cell step) still did not certify: after the referee found an impossible rank condition in it (fixed), the rerun reached its cap (524,256 cells) with no usable bound in some cells. Orientation from it (hand-built net, 1 unit): shared span 0.047, TQ10 rule 0.032, best found 0.026. The trained net was stopped unrun | NO |
| D5 | covers switching boxes, or states the gap two-sided | TQ10 part 3: Phi_free (codes free at every point) <= least <= Phi | YES |
| D6 | the result is not on a side branch | 37 of 40 pending nodes decided by delegation (your 2026-11-01 words) or dropped as moot; the 3 directional ones are on card 3-33 (none blocks row 3) | YES |

**So row 3 is NOT done.** Two items fail or stay partial for one shared reason: every lower bound so far needs to "pin" the
best build's span near the shared span, and every pinning argument so far has constants with 1/(smallest probability),
which is useless on ResNet-18 (about 1e-8).

**What this round taught, in the network's terms.** On ResNet-18 the dropped part of the movement is LARGE as a
quadratic (47 times the true KL of the shared span) but CHEAP in KL: it is mostly big logit moves on classes the network
already finds unlikely. What is small is the tilt: the best span sits about 11% away from the shared span. So the right
object is Conjecture DQ8: keep the exact KL, linearise only the tilt. That is a convex problem (exact global minimum
with a dual certificate, at any size). Its two sides are already proved by a Lipschitz step. What is missing is exactly
one thing: a pinning lemma whose constants come from N's head classes, not its least likely ones.

| | |
|---|---|
| proved | TQ10 (LIMIT-2), Corollary TQ10.1 (ONE-BUILD), TQ9 part 4 (dual cell step, head form corrected), Definition DQ7 |
| conjecture | DQ8 (pinning with constants from N; then the convex Psi sandwiches the least) |
| checks | `model/tq_small.py` 0 fails; `model/tq_cert.py` see D4 |
| ResNet-18 (Part C) | TQ10's Phi 0.120 / 0.022 (two metrics) vs best found 0.025; builds 0.163 / 0.104 |
| referee | `REFEREE-BQ5-2026-11-02.md`: 6 errors (head-form rank condition impossible, which also broke the certificate script; DQ7 named a zoom group; "graded promise of TQ10" did not exist; TQ11 cited but absent; DQ8 mixed two tilts), gaps (eigen-gap unused, 1/s_min powers, scope of the family); all fixed, not re-refereed. Content: TQ10 is a regime statement |
| your calls | card 3-33 (R-mixclean, R-circuit, R-wellposed); none blocks row 3 |
| next | the pinning lemma of DQ8 with head-class constants; then compute Psi on ResNet-18 and its certificate |

## 45. Row 3 on the real object: what the trunk and the agents changed (2026-11-01; plan great-work-so-far-zippy-lagoon.md; COARSE-THEORY.md B.Q.9; card 3-33 answered)

You answered card 3-33, and you told me I was not using the trunk. You also told me to give mapping to agents. Here is what that changed.

| Word | Meaning here |
|---|---|
| one read | k new units read the last layer with ONE matrix for all photos: what a real k-unit network does |
| free codes | k numbers chosen freely for each photo: the most ANY network with k top units can do (Proposition TQ11) |
| proxy | my old test object: class clusters, each with its own read and its own offset |
| nat | the unit of KL. N holds 5.66 nats of information on these photos |

**Step 1. My row 3 numbers were not about a real network.**

```
proxy, "38 units":     photo -> which of 100 clusters?  (free)  -> offset of that cluster + V x read of that cluster
the trunk's box:       photo -> 512 pre-activations z -> 38 units relu(m . z + beta) -> 1000 logits     (one read for all photos)
free codes (a floor):  photo -> any 38 numbers at all -> 1000 logits
```

Why it matters: Proposition TQ11 part 3 proves that a proxy can lie below what any k-unit network can reach. TQ10's failure, the shared span and the DQ8 wall were all measured on it. The trunk's own words, "new units that read only that group's units", would have caught this. The cached "train" photos are also ImageNet validation crops, so P was never N's training set in row 3.

| Units | Logit PCA span alone | The trunk's box (one read of z; switching did not help, from two starts at each of 14 and 38 units) | On photos it did not see | Free codes, two kinds of start | Proxy |
|---|---|---|---|---|---|
| 14 | 3.05 | 1.23 | 1.49 | 0.18 | 0.23 |
| 38 | 1.47 | 0.34 | 0.63 | 0.047 | 0.06 |
| 64 | 0.745 | 0.139 (read only) | 0.34 | | |
| 128 | 0.160 | 0.033 (read only) | 0.11 | | |

All numbers are KL in nats, and all are fits: no rule produces them yet. A read of h, the values after the ReLU, gets 1.10, 0.28, 0.11, 0.022, but it adds a layer after N's last layer, so it is not the trunk's box (the referee caught this). My table in the plan used 1.47 for "real": that was the PCA span only.

**Step 2. Corrections to what I said before.** The shared span is NOT the object on a real head: it costs about 2.2 times the best at 38 units (read of h). My written prediction (probability spans would win) was also false: probability is the worst and logit the best of the closed forms. No closed-form span is close yet.

Good news from the same run (read of h): from four very different starts, random included, the search ends within 4% of one value. The landscape looks benign at full size (measured, not proved). The spans nearly nest: the 14-unit span sits inside the 38-unit span with cosines 0.97 falling to 0.77. Bad news: the best builds move logits about twice as much as N does.

**Step 3. Your circuit question: the box sits about 7 times above the floor of any k-number code at the top. That floor looks like class identity, not a circuit of N, so the gap between a group and N's own circuits is still unmeasured.** I told you "about 2%" and then "at least 12%" earlier today, and then, in this write-up's first draft, "group and circuit differ by 7 times". All three were wrong. The first two came from free-code fits that started from the one-read solution and got stuck. The third read an oracle as a circuit.

```
any network with k top units   >=  free codes            (TQ11 part 2; reached when the lower layers can be any size, part 5)
the trunk's box at k = 14:     1.23 nats, top-1 agreement with N 0.53
free codes at k = 14:          0.18 nats, top-1 agreement with N 0.95
```

Why it matters: a last-layer group is one linear read of z, plus a gate. That read cannot write what the 14 numbers need to be. Free codes reach the same value from a random start and from a start on a table of class rows (0.184 against a table least of 0.171), so the 14 numbers appear to encode which class. So the floor is the cost of WRITING class identity in 14 dimensions. The gap is the cost of COMPUTING it from z with one read and a gate. The layers below the floor may be any size, so it is an oracle, not a circuit of N. Whether N's own stack can compute it in fewer units is the open question: such a stack could recover anywhere between none and all of the gap.

Theorem TQ14: where each region of photos has collapsed to a point, the gap is exactly 0, and the whole squeeze is a table of K rows of probabilities, whatever the size of N. Label: EQ. The table's own least stays open. On this net the table with one row per top class has L_table(14) = 0.17, L_table(38) = 0.045 (close to the floor), but N holds 0.60 nats inside those rows, and its features are far from collapsed. The small CIFAR net is not in the collapse regime either: 40 epochs is not the terminal phase of training, and 34% of its feature variance stays inside classes. So it tests nothing about TQ14 part 6, either way. The network is not there yet; that is different from a wrong premise.

**Step 4. The agents read the network, and the essay holds up.** Six agents, about 60 channels, 550000 tokens in all. A separate reviewer tested them.

```
channel 179 (ResNet-18):  thin lines out from a hub (spokes, whiskers, spider legs)   vs   heavy solid masses
channel 214 (ResNet-50):  curl (wiry dog hair, corkscrew, mantis leg)
a class:                  a vote over many such channels (median 23 carry half of the margin)
```

| Check | Result |
|---|---|
| A reviewer predicts from the explanations alone, on classes the card never names | 23 of 24 pass. Percentile 0.73 on HIGH, 0.44 on NEAR |
| "Off outside the region" (our district claim) | fails everywhere. Channels are on for 80 to 98% of photos |
| Removing a channel's activity outside its region | costs far less than removing the channel |
| What feeds a last-layer channel | mostly its own skip. Real circuits need level 2 or deeper |

Why it matters: LADDER 30's "no map exists" was wrong. The channels are readable. My scripts and my claim types were the fault. These channels are graded votes of shared patterns, the essay's generalization, not exclusive districts. So the type of claim must change (a first one, in the trunk's own loss: Definition DQ12). Cost, an estimate: about 20 million tokens for ResNet-50's last block, about 150 million for all of its block channels.

**Step 5. The trunk's "inputs N was trained on".** Lemma TQ12, proved, Lean:

```
training objective(W') - training objective(W) = KL(N || W') + (lambda/2) |W' - W|^2
```

Why it matters: for a trained N, the teacher-student KL is the excess of N's own training objective. Your remark exactly: the goal is well formulated. On a small net whose whole P we hold, stationarity nearly holds (cosine 0.93). That net is overfit, so it is not a verdict on ImageNet nets. It does not by itself give the least of a squeeze.

**Step 6. The first closed form shape at a loss above 0, read from a failed conjecture.** I conjectured that a tight frame is the least of a symmetric table. Every random start beat it, and they all landed on one value, with this form:

```
6 classes, 3 units:   unit 1: class 1 = +3.4, class 2 = -3.4, the rest 0
                      unit 2: class 3 = +3.4, class 4 = -3.4, the rest 0
                      unit 3: class 5 = +3.4, class 6 = -3.4, the rest 0
```

One unit, two poles, two classes. Why it matters: it looks like the essay's last-layer neurons (a pattern on one pole, a close relative on the other), and it has a closed form that matches the search. It is a conjecture (DQ11), for K = 2k only.

**Step 7. A certified lower bound, the first in row 3.** Lemma TQ15: the least over spans is semiconcave, so a cell's values are bounded from its corners, with an error quadratic in the cell size.

```
4-class symmetric table, a = 0.6, 2 units:   pole build 0.047374,  certified lower bound 0.046188   (gap 2.5%, stopped at the cell cap)
```

Why it matters: the pole build is within 2.5% of a proved global optimum on this table. It is a check on a small table, not a method: the number of cells explodes with the number of angles of the span. The other certificates were still running when this was written; see Part C. And each corner's value is from a numerical fit, so to be fully rigorous a corner needs the dual value of TQ15 part 4.

**Where row 3 stands against the plan's definition of done.**

| # | Row 3 is done when | Result |
|---|---|---|
| D1 | Every ResNet-18 number is the loss of a real network of that size | YES for the box rows. Old proxy rows and the read of h are marked |
| D2 | The least with a matching lower bound on the real object | NO. Only the collapse case (TQ14), which reduces to a table whose least is open. Said plainly: the convex route of the plan fails, its best possible bound is 20% to 45% of the least on small tables (TQ13's check) |
| D3 | Certified optimum on hand-built and trained small nets | PARTLY. Within 2.5% on one symmetric table (TQ15). Not at float error, so this fails the bar as written |
| D4 | Our KL against a trained distillation at k = 14, 38, 64, 128 | NOT RUN. No rule exists yet. The values above are fits, which the trunk does not allow as the answer (label ONE-BUILD) |
| D5 | First-order tight bound | NOT MEASURED. TQ14 part 3 is a graded bound, with tightness not claimed: its error rho is near 1 on unlikely classes |
| D6 | Labels and form | YES. Referee report REFEREE-BQ9-2026-11-01.md, all findings fixed, not re-refereed |

**So row 3 is NOT done.** What moved: the object is real; the circuit question has a theorem and a floor, though not yet its answer; the trunk's training fact is in the paper; a closed form at a loss above 0 exists as a conjecture, with a stationarity proof and a 2.5% certificate; and the map is finally being read in the essay's way.

| | |
|---|---|
| Trunk check | used: "new units that read only that group's units" (TQ11, the box); "the inputs N was trained on" (TQ12, TQ14); "How M is found ... is outside the trunk" (map agents); the one-time computation at any cost (R-mixclean (e), TQ15); not yet used: the graded clause on a real map; the nested zoom, beyond the measured nesting and the nesting of tables |
| proved | TQ11 (5 parts), TQ12 (Lean: ts_point, ts_identity, ts_sandwich, for the algebra only), TQ13, TQ14 (6 parts), TQ15, TQ16. Not refereed yet: TQ11 part 5, TQ13 part 4, TQ14 part 3, TQ15 part 4, TQ16, all from the referee round |
| conjecture | DQ11 (the pole build is the least of the symmetric table with K = 2k) |
| checks | `model/ts_identity.py`, `model/tq13_check.py`, `model/tq14_table.py --only-a`: 0 fails; `model/tq14_table.py` part B: pole formula within 5e-15 of the search on all 6 tables |
| referee | `REFEREE-BQ9-2026-11-01.md`: 4 errors and many gaps, all fixed, not re-refereed; it also gave the graded form of TQ14 and showed my h reads add a layer |
| your calls | card 3-34: nothing you must answer. It shows the number, and says what I will do unless you say otherwise |
| next | test the hypothesis that N's own stack can compute what the top needs, with a circuit as a stack of groups, within the trunk as written (card 3-34) |

## 46. Row 3: the first exact theorems, and how far real networks are from them (2026-11-01; plan you-ve-done-a-lot-elegant-dongarra.md; COARSE-THEORY.md B.Q.9; card 3-35)

You said row 3 would change in this session. Here is what changed. Row 3 now has exact theorems. It is NOT closed, by your own rule, because real networks miss their premises by a wide margin. The decision is on card 3-35.

| Word | Meaning here |
|---|---|
| table | one row of probabilities over the classes for each region of photos |
| pole | one unit, one class on its plus side, one class on its minus side |
| span | the k directions in logit space that k units can write |
| complement | every logit direction that the k units cannot write |
| dual point | a set of made-up probabilities. Each one that passes a linear test gives a proved floor under the loss of every build |
| rho | the least probability the build gives anywhere, times that photo's weight |
| read | k units computed from z with one matrix, for all photos |

**Step 1. The first equality with the least, above 0.**

```
N's table: 2k classes, each region favours its own class
k units, poles:    unit 1: class 1 = +tau, class 2 = -tau, the rest 0     (one pair per unit)
                   loss = a closed form > 0
                   and NO build of k units, with any weights, does better. Proved.
```

Why it matters: this is the proof of Conjecture DQ11 (Theorem TQ17), and the first lower bound in row 3 that meets a build at a loss above 0. All five earlier sessions failed here. A fresh reviewer found the proof, and I shortened it.

Why it worked where the others failed:

```
old bounds:   measure a build in LOGITS. Push an unlikely class down by 20: looks huge, costs almost no KL.
              So each bound carried 1 / (smallest probability), about 1e8.
new bound:    measure in PROBABILITIES. The same push moves 1e-8 of probability: looks tiny, as it should.
              The rank enters through the complement, whose size is fixed: C - 1 - k directions.
```

The proof has one short identity at its centre. For a row of any projector onto a complement, the squared distance from the pole's row equals how far its partner entry falls short. The sum then closes row by row. Lean holds that identity. The rest is by hand.

**Step 2. Any size, any code, and the read.**

```
any table, any k, any build that is stationary:
   g     = what the units leave undone that the code class could still reach
   sigma = how strongly the units write
   rho   = the build's least probability, times the row's weight
   if g <= rho and g <= rho sigma / 2:    this build is the global least.  Proved (Theorem TQ18).
what z cannot compute at all does not enter the test.
```

Why it matters: no span is pinned. Both sides of the inequality scale with how far a competitor's span lies, so one test covers every span, near or far. That removes the wall of LADDER 42 to 45. It also proves its own build on a network when the test passes (Corollary TQ18.1), with no claim on the map. It is the first lower bound in this Part with the read in it.

**Step 3. The definition that blocked everything.** Completeness (Definition DM6) required "N has two logits" for a squeeze. So no ImageNet map could ever be complete. I rewrote it: a squeeze is complete with a pole claim, at any number of logits, with many groups. Node R-poleclaim, by delegation. The narrow scope that remains is yours, node S-polescope: only levels whose total written rank is C/2. Also said plainly: the claim's pair counts are the sizes, so it meets "No claim refers to a target size" in its letter only. That is on the card.

**Step 4. Exact is really exact, on hand-built tables.**

| Check | Tables | Worst fall below the theorem |
|---|---|---|
| symmetric, K = 4 to 12 | 20 | 5e-16 |
| pole family, not symmetric | 60 | 2e-16 |
| any k, free and read | 24 | 1e-16 |
| any k, a family of any code | 18 | 1e-16 |
| kept units, free and read | 16 | 2e-17 |
| many groups and kept units, the sum form | 1 | 2e-16 |
| control: premise broken | 15 | a fit beats the pole value on 8 |

Random starts are the competitors here, so this checks the theorem on the samples. It does not replace the proof.

**Step 5. What real networks do.** All numbers are KL in nats, on validation crops, not N's training photos.

```
the box from real random starts, k = 14, ResNet-18:   1.26, 1.28   (warm start: 1.23)    gates on 0.82
each unit:                                            half of a pole's weight sits on 60 classes
                                                      the top 2 classes hold 3% of its weight
```

Why it matters: the gap above the free-code floor is real. (Corrected in LADDER 49: 1.26 and 1.28 are under-converged; the read reaches 1.172 on these photos. Corrected in LADDER 47: I wrote here that the squeeze is "far from a pole". That was wrong. These are coarse poles, with relatives on each side. The floor below rules out only disjoint flat blocks.) A build with 2k disjoint flat class blocks cannot reach it: it must lose at least 2.33 nats at k = 14, by the chain rule. The box loses 1.23.

| Measure | ResNet-18 | ResNet-50 |
|---|---|---|
| read against free codes, k = 14 | 5.4 times | 3.5 times |
| k = 38 | 4.7 times | 2.5 times |
| k = 128 | 3.2 times | 1.7 times |
| test ratio of Theorem TQ18, read, k = 14 (should be <= 1) | 7e36 | 2e49 |
| pole span against a free fit, k = 500, best pairing | 160 times | 140 times |

Why it matters: the read gap shrinks on the wider network, as Theorem TQ14 part 6 suggests, but it stays above 1. And real networks miss the exact theorems by a vast margin. The trained CIFAR net misses too: test ratio 1e160 at k = 2, falling to 1e3 at k = 50. The cause has a name: the premise g <= rho. A dual point must stay a set of probabilities at every competitor span, and a build's tail reaches 1e-25 to 1e-80.

**Step 6. Lumping the tail away does not rescue it.** I expected the tiny tail to be the whole problem. So I wrote Conjecture DQ15: run the same test on each row's top classes only. The route said to measure before proving, so I did.

```
a typical row, what it leaves undone on its top classes / what those classes can give up:
   table, best case (top k + 1 classes):   about 50 to 140
   read, ResNet-18, top 15 classes:        about 2000
   premise, with a typical row's numbers:  500 times too small at best on a table, 30000 times or more on the read
```

Why it matters: a head of k + 1 classes already holds very small probabilities, and what a row leaves undone there is far larger than they are. So the linear shape of the dual point itself falls short, with or without the tail. By its own route, DQ15 as stated is the wrong conjecture for these networks. This is a measurement, not a theorem.

**Step 7. A surprise, and a wrong prediction of mine.** Theorem TQ17 part 5 (from the referee) says a pole's partner is no likelier than any other class. So the classes on one unit are ones the table rarely confuses.

```
the pole form, k = C/2, ResNet-50:   partner = most confused class     1.11 nats
                                     partner = least confused class    0.014 nats
```

I predicted under 3 times. It is 80 times. (Corrected in LADDER 47: I read this as packing against the essay, and blamed the read. Both were wrong. It compares two pairings inside one wrong form, and the real least is 100 times lower than either and keeps relatives together. And the exact class here is the essay's toy's world.)

**Where row 3 stands against the plan's definition of done.**

| # | Row 3 is done when | Result |
|---|---|---|
| D1 | Slot 3 theorem at a loss above 0 | YES, TQ17 and TQ18. Three referee rounds: every core proof held. Lean holds the row identity only, so the real content is by hand |
| D2 | Exact check | YES on hand-built tables, at float error. The trained small net is reported, and it misses the test |
| D3 | Slots 1 and 2 | YES at total rank C/2 only: pole claim, DM6 rewritten. I wrote a second claim type for TQ18. The referee showed that it names a build, so I removed it. The test now lives in the one-time computation |
| D4 | The real squeeze measured | YES, in Part C |
| D5 | A claim real networks nearly meet | NO. This is the failure that matters |
| D6 | Form | YES (lint) |

**So row 3 is NOT closed.** What moved: the lower bound exists, with no pinning, and with the read inside it. Its value carries no 1/(smallest probability), but its premise still holds the build's least probability, through rho. What is missing is a premise a trained network meets.

| | |
|---|---|
| Trunk check | used: "complete for the zoom" (DM6 item 5, the blocker); "new units that read only that group's units" (the read in TQ18, TQ18.1, the width test); "one-time computation" (the test of TQ18 lives there); "No claim refers to a build" (it removed my certificate claim); "P = inputs N was trained on" (Fact F8 gives t = 0, so part 5 applies). Not used: the nested zoom: one size per network does not nest |
| proved | TQ17 (6 parts), TQ17.1, TQ18 (6 parts), TQ18.1, TQ19. Labels: TQ17 the closed form of a table's least; TQ17.1 and TQ18.1 OPT-KL, exact case only; TQ18 a certificate of L_Z. No graded bound in the eps list of a network yet: that is session 2 |
| conjecture | DQ15 (the same test with the tail lumped); its premise misses real networks, so it is not the way to them as stated |
| checks | `model/tq17_pole.py`, `model/tq18_cert.py`, `model/tq18_defect.py`, `model/tq18_rows.py`: 0 fails. Graded part 5: bound / gap 12 to 53 on small tables, in a build's defects, not in N's cleanliness |
| referee | REFEREE-BQ10-2026-11-01.md, three rounds. Not re-refereed after the third round's fixes, which were to text, scope and one definition in part 6 |
| said plainly | card 3-34's stack is set aside, and I told you so in the plan. Three old certificate jobs were stopped, with your word. The exact claim needs real numbers: at rational logits and rational parameters its table error is never 0 (Lindemann-Weierstrass), so a network with float logits is reached only through a graded bound. And no graded clause of T5 holds yet, for one reason I can show: kept units with free weights make the least jump (TQ17.1's Comment) |
| your calls | card 3-35 |
| next | your call on card 3-35. If you keep row 3 open: a dual point that is not linear in the projector. I have no candidate |

## 47. Row 3: the contradiction with the essay was my error (2026-11-01; plan you-ve-done-a-lot-elegant-dongarra.md, second version; COARSE-THEORY.md B.Q.9)

You said the answer likely lies in the contradiction with the essay, and that networks work because they are interpretable. You were right that there was an error, and I made it. Row 3 is still NOT closed. The picture is much clearer.

| Word | Meaning here |
|---|---|
| coarse pole | one side of a unit: many related classes, with graded weights |
| toy's world | the essay's phrase: memories "related to nothing else", where packing is right |
| minor | a sub-table: a family of classes, with each photo's probabilities conditioned on that family |
| weighted low-rank | fitting a low-rank table when each entry has its own importance |

**Step 1. The units are the essay's units. I had read them wrong.**

```
ResNet-18, 14 units, fitted from a random start, then rotated:
 unit 0:   green mamba, vine snake, tree frog ...   vs   go-kart, school bus, tow truck, jeep ...
 unit 4:   rifle, cannon, missile, revolver ...     vs   poodles, terriers ...
 unit 11:  grey fox, coyote, red wolf, dhole ...    vs   cello, harp, banjo, violin ...
essay's small model:   cars, trains, carts          vs   boats, planes
```

Why it matters: this is "coarser, never fractured", on the least of the trunk's own object. I measured "60 classes on a pole" and called it "not a pole". A blind test backs this. A reader saw only class names and picked held-out members of real poles 79% of the time, against 43% on random groups. It said "no pattern" on all 6 random groups and on 1 of the 12 real poles. Shuffled controls of within-pole similarity sit at 0.00, against 0.04 to 0.11.

**Step 2. There is no contradiction. My theorem is about the toy's world.**

```
TQ17's family:   every other class equally unlikely  -> nothing is related to anything -> unrelated classes share a unit
real tables:     relatives are likelier than strangers                                  -> relatives share a pole
```

Why it matters: the essay says this itself: "memories that really are ends in themselves, related to nothing else. That's the toy's world, and there it does pack." So the theorem stands, and my Comment was wrong. At k = C/2, I forced one class per pole, then compared two pairings inside that wrong form. The real least there is 100 times lower than either, and it keeps relatives together.

My first explanation was also wrong: I blamed the table for ignoring the read. Free codes and the table hold relatives on one pole just as much as the read does.

**Step 3. A third error of mine, this one in the plan you approved.** I told you real networks are nearly clean in their own metric, and I proposed a test number. That number is not a test. Squared, it is about 2 x loss / write. So below 1 only says the loss is small.

```
a tree table:   number = 0.37     and the tree build is beaten by 92%
```

Why it matters: I retract that reading. What stays true: on a photo's top classes, the build's ratio to N is close at large k. That is the same fact as a small loss.

The comparison that does mean something is the strongest dropped direction against the weakest kept one.

| k | 14 | 38 | 64 | 128 |
|---|---|---|---|---|
| ResNet-18 read | 197 | 18 | 3.8 | 1.1 |
| ResNet-50 read | 821 | 18 | | |

Why it matters: this is the real distance from a gap test, and it says small k is far away. It has no proof behind it yet.

**Step 4. An exact case in the essay's world, and how narrow it is.**

```
one table with a family tree:  k = 3 units: 4 families, relatives on one pole, the least    (test passes)
                               k = 7 units: 8 families, same table, spans nest, the least   (test passes)
```

Why it matters: this is the first exact result that uses the nested zoom, and it has the essay's form (Proposition TQ21). Limit, said plainly: its premise forces fine distinctions below 1/K in probability, so no table confident in one class passes. When fine levels are stronger, flat family poles lose to graded ones, by up to 92%. Real units are graded.

**Step 5. The exact statement of the problem, and where the loss really is.**

```
any two builds:  D(l') - D(l) = sum of p_x [ KL(q_x || q'_x) - <s_x - q_x, l'_x - l_x> ]       (Lemma TQ20)
                                      cost of moving              gain on what was left undone
```

Why it matters: this has no tiny number in it. At second order it is weighted low-rank approximation, with one metric per photo. So it is hard unless a claim on N helps. Theorem TQ18 was hard to meet because it used one projector in plain distance.

The referee gave one route with a proof (Proposition TQ22). Conditioning on a family of classes moves tiny numbers into a photo's weight, where they do no harm. So the least is at least the sum of the family sub-tables' leasts.

| Families | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| share of the table's loss in the families, k = 14, ResNet-18 | 0.79 | 0.57 | 0.37 | 0.17 |
| share between families | 0.01 | 0.03 | 0.06 | 0.12 |

Why it matters: these are fits, so not yet a bound. They say the loss in small families is small, and the mass between families is small too. Most of the loss at fine scale is **the cost of sharing units across families**. The lower bound does not charge for that sharing yet. That cost is the object to solve: the essay's interference against reuse, on real tables. Related families hold more than random ones at every J, and my prediction there was wrong.

**Where row 3 stands against this plan's definition of done.**

| # | Row 3 is done when | Result |
|---|---|---|
| R1 | The contradiction resolved and scored | YES. Corrections written. Coherence against a control, held-out prediction scored. The blind test covers one fit |
| R2 | A certificate real networks nearly meet | NO. Not proved, not refuted. My candidate number was retracted, and the lumping form was withdrawn |
| R3 | Exact theorem with relatives on one pole | YES, in a narrow form: TQ21 |
| R4 | Form | see below |

**So row 3 is NOT closed.** What changed: I no longer think real networks are in the wrong class for an exact theorem because they are messy. They are readable. What is missing is a statement about the cost of sharing units across related families.

| | |
|---|---|
| Trunk check | used: "the essay" (line 701 settled the contradiction); "new units that read only that group's units" (I tested it as the cause, and it is not); nested zoom (TQ21, the first use in row 3; TQ22 gives a tree of minors); "a map of claims, found by agents" (the blind reader; the clusters in the minors row are a measure, not a map). Not used: "P = inputs N was trained on" beyond Fact F8 |
| proved | TQ20 (identity), TQ21 (certificate on a tree), TQ22 (a lower bound with no tiny number). No OPT-KL result is new here |
| conjecture | DQ15, rewritten: with confident families, a squeeze's cost beyond its minors is a leak term. Real tables are not confident in one family, so this is a route, not the target. The form in LADDER 46 is withdrawn |
| checks | `model/tq18_tree.py`: 0 falls where the test passes. Fits: `cache/map/fit_*` |
| referee | REFEREE-BQ10-2026-11-01.md, rounds 4 and 5: TQ20 and TQ21 parts 1 to 4 hold; all findings fixed; TQ22's source corrected; DQ15 rewritten after it was found empty. The last fixes (DQ13's sentence, TQ21 part 5, TQ22's citation, DQ15) were not re-refereed |
| your calls | none. Card 3-35 stays open, and I have stopped asking |
| said plainly | three times in one session I told you a thing with too much confidence, and then retracted it: "far from a pole", "the read causes it", "the test number". All three are corrected above |
| next | the cost of sharing units across related families: first on a small table with two related families where it can be solved, then the same number on real tables |

## 48. Row 3: two fan-out rounds, and what they established (2026-11-01; plan you-ve-done-a-lot-elegant-dongarra.md, third version; COARSE-THEORY.md B.Q.10; `rounds/`)

You asked for a purely mathematical run with agents fanned out. Two rounds are done: 18 agents, about 4.8 million tokens. **No slot is filled on a table with real structure. Row 3 is NOT closed.** Round 3 is smaller, and every track in it must meet a real table.

| Word | Meaning here |
|---|---|
| stranger family | a family of classes other than the one a photo belongs to |
| certificate | a test that proves one given build is the true least |
| walled program | at high confidence, each unlikely class must sit a fixed distance below the top. A build pays for going over |

**Step 1. Every certificate from round 1 dies on real tables.**

```
                                               misses by, on ResNet-18 and ResNet-50
moment certificate (E6, TZ3), always-on read:  2e6 to 2e9
patchwork certificate (D3, TZ1), family patches: 1e11 to 1e19 at the median
packing bound (C10, TZ4):                      reaches at most 1e-7 of the fit (it should reach 0.1)
cut bound (C4):                                dead at k = 14
```

Why it matters: these are the four tools that survived round 1's referees, on hand-built tables. They are real mathematics. None of them says anything about a network yet.

**Step 2. Where the loss really is on a real table, at k = 14.**

```
a photo of a dog, its 1000 probabilities:
  inside the dog family                   0.12 of the loss
  mass between families                   0.18
  inside STRANGER families (how N ranks
  teapots against kettles on a dog photo) 0.70
```

Why it matters: this is the sharper form of the sharing cost. Most of what 14 units cannot keep is N's ordering of classes it already finds unlikely, at probabilities of 1e-4 to 1e-2. Any bound or claim that looks only at a photo's own family misses it. Exact family tables (D2) need the own-family share to be about 1, and it is 0.12.

**Step 3. Real tables are not in the confident regime.** Mean tail mass on the regions is 0.31, top class 0.69. So the confident-limit results (TQ3, TQ5, TQ17, the walled program TZ14 and TZ15) do not describe this network. KL there is linear on 46% to 60% of the loss, exponential on 29% to 34%, and quadratic on only 8% to 26%. So neither the quadratic picture nor a pure limit fits.

**Step 4. The read is not the object either.**

```
a toy table:   the box with gates   = free codes   < the always-on read
```

Why it matters: Theorem TZ13 (round name J4.2) shows gates can remove the whole cost of the read. So a bound on the always-on read does not reach V(G). On real fits, gates did not help from the starts I tried, and the theorem says random starts can miss. There is still no lower bound on the box above the free-code floor, when features vary inside a class.

**What did survive, as mathematics.** All are in B.Q.10, with referee corrections. None is OPT-KL on a real class.

| Item | What it says | Label |
|---|---|---|
| TZ1, TZ2 | certificates with local projectors | certificate |
| TZ3, TZ9 | certificates for the read, with an average Fisher number | certificate |
| TZ4, TZ5 | a lower bound with no fixed competitor | lower bound |
| TZ6, TZ7, TZ8 | graded bounds in the largest log-ratio | graded; real regions have a largest log-ratio of 39 |
| TZ10 to TZ13 | the read's cost, exactly, on toy tables; gates | exact on toy tables only |
| TZ14 to TZ18 | the confident object: a walled low-rank program | limit, or toy tables |

Two results of round 2 are NOT in the paper yet, because no referee has read them, and they are the most novel: G4 (the first exact least with one budget of units shared across families) and K2 (an information floor that charges all rows for one budget).

| | |
|---|---|
| Trunk check | used: "new units that read only that group's units" (TZ11 to TZ13, the box against the read); "inputs N was trained on" (track E: training does not decide the squeeze); "complete for the zoom" and "graded clause" (slots); "one-time computation" (the certificates). Nested zoom: TZ1's patches, no result yet on real tables |
| proved | B.Q.10, by agents' hand proofs, with referee corrections. F2 (TZ19) is not refereed. No OPT-KL result on a table with real structure |
| checks | Part C rows, round 2 measurements 1 to 6; `rounds/round1_digest.txt`, `rounds/round2_digest.txt` |
| said plainly | my selection of what to referee trusted each agent's own count of slots. One agent claimed 2 and the referee cut it to 0, while the two most novel results went unread. I now choose. Cost is 2.4 million tokens per round, against my estimate of 1.4 million |
| your calls | none |
| next | round 3: referee G4 and K2 by my own choice; three tracks, each with a real-table measure in the same track; a track that returns only a limit or a toy-table result has failed |

## 49. Row 3: three fan-out rounds, where it stands, and why I stopped (2026-11-02; plan you-ve-done-a-lot-elegant-dongarra.md, third version; COARSE-THEORY.md B.Q.10; `rounds/`)

**Row 3 is NOT closed. No slot is filled on a table real networks are near. No theorem shows that one cannot be.** Three rounds: 25 agents, about 6.6 million tokens. I set a plain target for round 3, the first proved lower bound at k = 14 that reaches 10% of the fit, and said I would stop and tell you if it failed. It failed by about 85 times.

| Word | Meaning here |
|---|---|
| certified floor | a proved number that no build of k units can go below |
| encoder | the one fixed linear map from z to the k units |
| code term | the extra loss from using that encoder, against free codes at the same span |
| weighted low-rank | fitting a low-rank table when each entry has its own importance, with no shared scale |

**Step 1. The gap, in one line.**

```
ResNet-18, k = 14, class-row table:
   best proved floor   0.00029 nats
   best build found    0.245   nats      the truth lies between them: a factor of about 850
```

Why it matters: nothing here says which end is close. On hand-built tables the theorems are exact, but on a real table the mathematics barely constrains the least.

**Step 2. Why the bounds are small: one shared scale per block.**

```
a lower bound needs:   one projector over a block of photos  ->  one scale for each class across those photos
a real table has:      the same class differs by a factor up to e^39 between regions
so it must choose:     a tiny common weight on a big block,  or  small blocks with their own units each
```

Why it matters: a bound that keeps each photo's own scale is weighted low-rank fitting. It is hard in general, unless a claim on N helps, and no one in three rounds found one. Three things are proved about this: budget floors are empty at k >= 7 on 1000 classes (TZ24); a sum over blocks is capped by what those blocks' own leasts allow (TZ30); and on the best weights found, any certificate is capped near 1% to 1.6% of the fit. These limit methods. They are NOT a theorem that a slot cannot be filled.

**Step 3. The most useful real finding: the read's cost is not what z lacks.**

```
full-rank read of z, 512 outputs:   loses 0.004 nats on the fit rows, about 0.03 on held-out rows
14-unit read:                       loses 1.06
at that read's own span:            0.246 (free codes at that span) + 0.658 (the code term)
```

Why it matters: z can compute N almost perfectly, and the span the read picks is not far from a free fit's span. Most of the loss is that one fixed linear encoder cannot write what free units would give each photo. That is a sharper question than any earlier form of row 3.

**Step 4. Numbers of mine that were wrong.** A referee reran my fits in float64.

| I said | True |
|---|---|
| the box at k = 14 is 1.23, and 1.26 to 1.28 from random starts | under-converged. The read reaches 1.172 on the same 35000 photos. All are upper values |
| gates do not help | withdrawn. No run I made tested it: one run stalled at 60 iterations, and it had no gate in it. One hinge gains 3e-4. Whether gates help by a real amount is open |
| the floor is 0.18 | at most 0.166 on 12000 rows, and still falling |

**Step 5. What is true, in a table.**

| Object | State |
|---|---|
| Exact leasts, tables built from the answer | TQ17, TZ21: exact, with one shared budget of units; a real table is not of this form |
| Certificates | TQ18, TZ1, TZ3, TZ9: valid; real tables miss by 1e6 to 1e19 |
| Lower bounds with no fixed competitor | TZ4, TZ23, TZ25, TZ28: valid; best is 0.12% of the fit |
| The box above the floor | no certified bound at real size |
| Impossibility | none proved. Hardness stops at one rounding step (track K). Counting cannot give it |

**So row 3 is NOT closed, by your bar or by your roadmap.** I have not moved any label to fit that.

| | |
|---|---|
| Trunk check | used: "new units that read only that group's units" (Step 3, the encoder); "one-time computation" (certificates); "a map of claims" (none reached a real table); "inputs N was trained on" (track E, round 1: training does not decide the squeeze). Nested zoom: TZ21 and TZ1's patches only |
| proved | B.Q.10: TZ1 to TZ34, by agents' hand proofs with referee corrections. TZ19 not refereed. No OPT-KL result on a table with real structure |
| checks | Part C, round 2 and round 3 rows; `rounds/round3_digest.txt` |
| your calls | none are forced by a theorem. I stopped the rounds because I said I would. Whether to spend more is yours |
| next, my view | stop broad rounds. One narrow item is left that could end row 3 by your rule: the rounding step of the hardness route (track K). One open question could change the object: the encoder's code term |

## 50. Row 3: the general form is proved hard, and the claim that would escape it is still missing (2026-11-02; plan ok-i-want-you-fluttering-pond.md; COARSE-THEORY.md B.Q.11, D.19; card 3-36)

**Row 3 is still NOT closed. No slot is filled on a table real networks are near.** What changed: the wall is now a theorem. On networks whose every claim holds, the least of a squeeze is NP-hard. This proves Conjecture DI1. It says the general form has no fast exact rule, unless P = NP. It does not say a real network's least cannot be computed, and it does not explain why the three rounds failed. I told you in the plan that it would. That was wrong, and the referee caught it.

| Word | Meaning here |
|---|---|
| area | a set of photos where one judgment unit is on, with its own free offset for the logits |
| weight pattern | how much each class counts in each row's loss, to second order: the row's own probabilities |
| weighted fitting | the best low-rank matrix when each entry has its own importance |
| biclique | in a bipartite graph, a set of left and right nodes with every edge present. Finding the largest one is a standard hard problem |
| padding | adding heavy blocks to a problem so that extra units are forced to go there |

**Step 1. A network whose squeeze is exactly weighted fitting.**

```
for each area w:   2 photos,  one judgment unit h_w (on, flat),  one group unit s_w (1 on one photo, 3 on the other)
logits on area w:  own class b_w = B      shared class g = B - 1      other own classes = 0
                   tail class j = log W[w, j]  +- (a/2) M[w, j]       <- all that moves inside the area
squeeze:           the s units -> k = r new units, h units kept

least KL  =  (a^2 tau / 8m) x  min over rank-r X of  sum W[w, j] (M[w, j] - X[w, j])^2     within 4 alpha + 6 eta
```

Why it matters: the map's claims all hold on this network, and its squeeze is plain weighted fitting, with any weights I choose. Earlier sessions said this at second order. This is a two-sided proof. A class's cost and its gain carry the same small probability, so pushing a tiny class far gains nothing. That is what stopped the round 1 attempt.

**Step 2. Weighted fitting is the biclique problem, so the squeeze is hard.**

```
any bipartite graph  ->  M = its edges,  W = 1 on edges, d off them   (Gillis and Glineur 2011, Lemma 3: I read the source)
                     ->  weighted fit = edges outside the largest biclique, within 1/4
                     ->  least KL tells you the largest biclique
```

| Part of Theorem TQ23 | Network | Hard to compute |
|---|---|---|
| 1 | 1 unit, areas kept | V_b(1) |
| 2 | a table of 2m rows, no kept units, k = m | L_table(k), which equals V_b(k), L_free(k), L_on(k) there |
| 3 | r units, areas kept, with padding | V_b(r), for any r polynomial in the input |

Why it matters: your rule was "if it is not possible, show it mathematically". This shows it for the general form, on clean maps. It meets your rule only partly, as the next step says.

**Step 3. What the theorem does NOT say. These limits are the real result.**

```
excluded:      a fast rule accurate to about 1 part in 8 m n
NOT excluded:  a fast rule good within a factor of 2, or of 10
the instance:  every claim holds, but its tail patterns are arbitrary graphs. No training run made them
its regime:    tail mass and movement both tiny, so KL is quadratic. Real tables are quadratic on 8% to 26% of the loss
```

Why it matters: your bar asks for least KL exactly, so the first line matters for exactness. But round 3's target was only 10% of the fit, a factor of 10, and hardness says nothing against that. So this does not explain the certificate failures. Your rule also asks for a trained-like instance, and this one is not shown to be trained-like.

**Step 4. Reading the hard network, I looked for what real tables have instead. I did not find it.** I measured two candidate claims, both new here. Both name no build, size or zoom.

| Claim, class-row tables | ResNet-18 | ResNet-50 | Verdict |
|---|---|---|---|
| Product tail: strangers' mass within a factor 3 of one row scale x one column scale | .42 | .40 to .44 | misses badly |
| Tree weights, off the diagonal, rms log error | 1.43 | 1.44 | far from 0 |
| no tree | 1.81 | 1.82 | |
| shuffled tree (control) | 1.80 | 1.80 | no change, so the tree is real structure |

My plan said real tails are near a product shape. That reading of round 3 was wrong: most stranger mass is more than a factor 3 away. I predicted tree rms about 1.0 and mass at least .7. I got 1.43 and .63. Both predictions failed. By your rule, a premise trained networks miss badly is the wrong premise, so neither is the missing claim as stated. The tree here is one Ward clustering, not a map from agents. That is untested.

**Step 5. One small positive result, at one unit.**

```
wall program at k = 1, tree pattern, no scales:   does a best order keep each subtree together?
K = 4, 5, 6, all orders tried:   real tree 92 of 92      shuffled tree 68 of 92 (off by up to 9.8 times)
                                 with row and column scales 69 of 92 (off by up to 2.0 times)
```

Why it matters: this is Conjecture DQ17, proved at K = 3 only. It is the first sign that a tree cuts the search. But it is a limit, at one unit, on small tables. Real tables have scales, and the referee proved the statement false with scales, already at K = 3. Tractability at k >= 2 is NOT conjectured, because level tables sit inside the tree class and hold a packing problem.

**Numbers from the same run.** A span in closed form, from the singular vectors of the log table in the column scales' metric, with a convex fit at that span, reaches these. Label ONE-BUILD, upper values. I predicted under 1.5 times, and it failed.

| k | ResNet-18, closed span / fit | ResNet-50, closed span / fit |
|---|---|---|
| 14 | .541 / .248 | .489 / .188 |
| 38 | .169 / .066 | .134 / .043 |

**What the referees found in my drafts.** All fixed. Round 2 read the repairs: every proof in its scope holds, and it corrected comments only. Its own fixes were not re-refereed.

| I wrote | True |
|---|---|
| Proposition TQ24 reads the reduction's graph | it read the complement. Corrected with a new proof by dynamic programming |
| part 3 follows from the source's Remark 3 | the remark's sketch fails at rank 3: C8 has real rank 3, and an exact fit of one copy leaves 16 against 18. The referee gave a padding proof, now Lemma TQ23.c, for every polynomial r |
| hardness shows the searches of B.Q.10 could not succeed | only at one named accuracy, not at a factor of 10 |
| tree weights are a candidate for the exact promise | not earned. Restated at k = 1 |
| a tree would give a recursion, and a known theorem limits hardness at level tables | the recursion is not earned, and the theorem did not apply. Removed, and replaced by a short direct proof |

**Said plainly about the plan.** I did not run the certified global search on the small instance. The plan you approved called it cheap. That was wrong, because I ignored the loss scale. You can judge the reason from these numbers:

```
smallest useful case:   G3 at d = 10,  7 classes,  B = 14
loss scale Gamma:       about 1.1e-11
certificate's error:    (1/2) x 1450 x theta^2     (Lemma TQ15, from a logit spread of 14)
to resolve Gamma / 32:  theta about 2e-8
cells:                  5 angles, so about 2e38 before pruning
last run of this tool:  stopped at its cap of 120000 evaluations, at gap 2.5%
```

So "exact is really exact" is not met by this check in your sense, on this theorem. The exact check holds the proof's own build and fits from 6 starts per case against the proved interval. A fit can refute the lower side, and none did. It cannot prove it. Lean holds only two one-line inequalities. The proof is by hand, with two referee rounds. One advisor call inside the second round timed out, so only the referee read that round.

| | |
|---|---|
| Trunk check | used: "a map of claims" (M_N, every claim at 0); "complete for the zoom" (Corollary TQ23.2 limits any definition of it that promises a polynomial-time closed form); "one-time computation" (R-mixclean (e) stands, so the trunk is not contradicted); "new units that read only that group's units" (Step 1's box and two-point step); "the essay" (tree claim, toy's world). Not used: "inputs N was trained on", the nested zoom, the graded clause |
| proved | Lemmas TQ23.a, TQ23.b, TQ23.c, Theorem TQ23, Corollaries TQ23.1 to TQ23.3, Proposition TQ24. Labels: none of the four. These are limits and fill no slot. No OPT-KL result is new |
| conjecture | DQ17 |
| checks | `model/tq23_instance.py`, `model/tq23_constants.py`, `model/tq24_check.py`, `model/dq17_orders.py`: 0 fails on the corrected tests. The first full run reported 3 fails from my own test, which held a stalled fit against the upper value. The corrected test evaluates the build itself. `pc_real_weights.py`, `pc_real_tree.py`: predictions mostly failed, as shown |
| referee | `REFEREE-BQ11-2026-11-02.md`, two rounds |
| your calls | card 3-36. Nothing blocks on it |
| next, my view | the claim on real tables, from agents reading the network, not from one clustering. And the question this theorem leaves open: is a constant-factor certificate possible? It would be graded, so whether that counts is your call on the card |

## 51. Row 3, round 4: your end rule, a fixed factor proved hard, and no candidate passes (2026-11-02; plan i-need-you-to-tingly-orbit.md; COARSE-THEORY.md B.Q.12, D.20; card 3-37)

**Row 3 is NOT closed. Nothing passes on a real network. The impossibility side moved further than any earlier session, and it is not finished.** Your rule, on card 3-37: row 3 ends only by passing on real networks, or by a proof that it cannot.

| Word | Meaning here |
|---|---|
| vertex cover | a set of vertices of a graph that touches every edge. Finding the smallest one is a standard hard problem |
| Unique Games Conjecture | a widely used open conjecture in complexity theory. Results that rest on it are conditional |
| fixed metric | the loss (1/2) Var under N's own probabilities, which stay fixed. No softmax in the student |
| weight gap | the second singular value of the table of class weights, after scaling rows and columns. 0 means a plain product |

**Step 1. A fixed factor is now proved hard, on clean networks.** The referee found this. I had missed it, and I checked its proof.

```
any graph Q   ->   areas = vertices,  one unit k = 1,  every claim at error 0
                   own tail class: heavy.   Edge of Q: heavy, and must stay level.   Others: light
one unit can write its pattern only on areas with no edge between them
so the areas it gives up form a vertex cover

least KL  =  Gamma L vc(Q)  (1 + O(delta))
```

| Proposition | Excludes a fast rule within | Rests on |
|---|---|---|
| TQ30 | a factor below about 1.36 | P != NP (Dinur and Safra) |
| TQ30 | a factor below 2^(1/2) | P != NP (Khot, Minzer, Safra; that its last step is proved I have from a secondary source) |
| TQ30 | a factor below 2 | Unique Games Conjecture |
| TQ31, signed entries | every fixed factor | Unique Games Conjecture |

Why it matters: LADDER 50 could exclude only about 1 part in 8 m n. Now a factor of 1.36 is excluded outright. A certificate within a factor of 2 is excluded on these networks if the conjecture holds, and with signed entries so is any factor. So a claim that passes your test must leave these networks out.

**Step 2. What that does and does not settle.** Against your IMPOSSIBLE test:

| Line | State |
|---|---|
| I1: a theorem, every claim at 0 | met |
| I2: every fixed factor, unless P = NP | met only under the Unique Games Conjecture. Under P != NP alone, below about 1.36 |
| I3: trained-like | not done. I expect it can be done: labels sampled from the table make N a global minimum |
| I4: matches ResNet's tables | not met. Tail mass and movement are small here |
| I5: clean by the essay | not met. These are tables of graphs, and I expect them to fail this |

Why it matters: I must say this before you read Step 1 as an ending. This does not show a real network's least cannot be computed. A finished version would say what any passing claim must exclude.

```
what I tried first, and why it failed (Proposition TQ26):
   a completion theorem gives one bad entry, and a YES build pays on every hole of every row
   ratio below 1, always
the mistake:  I assumed a YES build fits every row. TQ30 puts the gap in the rows it gives up
```

**Step 3. Real networks: every screen missed its line.** Each line and prediction was fixed before its run.

| # | Candidate | Line | Result | Prediction |
|---|---|---|---|---|
| S1 | weight gap, ResNet-18 / -50 | at most .5, on .7 of the loss | .67 / .65 at .72 of the loss: misses | held |
| S2 | read curvature near a Kronecker product | ratio in [.5, 2] | .04 to over 1000: misses by far | held |
| S3 | table: a fixed-metric or closed-form value within 10% of the fit | within 10% | Q / fit .60 to .85, closed form 3 to 30 times too high: misses | Q within 30%: missed |
| S3b | the read: Q at least half the fit | at both k | .35 at k = 14, .62 at k = 38: misses at k = 14 | missed at k = 14 |
| S4 | a claim from agents | tree rms at most 1.0 | NOT RUN | |
| S5 | gates | gain above 10% | train 5.3%, sealed -13%, not converged, too few rows: not settled | train gain held; warm start missed |
| S6 | packing dual | at least 10% of the fit | already dead: round 2 measured 1e-7. My plan was wrong to list it | |

**Corrected 2026-11-03, row S5. I reported this number without saying what it covered.** The photo lists are sorted by class, and the script takes the first rows of each.

| Run | Fit rows | "Sealed" rows | Read: fit, sealed | Best gate: fit, sealed | Gain: fit, sealed |
|---|---|---|---|---|---|
| the one in row S5 | first 8000: classes 0 to 228 | first 3000: classes 0 to 299 | .346, 1.21 | .327, 1.37 | 5.4%, -13% |
| finished later, not reported until now | all 35000: all 1000 classes | first 3000: classes 0 to 299 | 1.172, 1.133 | 1.187, 1.153 | -1.2%, -1.7% |

The comparison inside each run is fair, since all of its builds use the same rows. But neither "sealed" number says anything about all 1000 classes. The 8000-row number also tests photos of classes 229 to 299, which that fit never saw at all. All runs stopped at the chunk limit, so gates stay unsettled. No gate beat the read on the full fit rows.

**Step 4. Three things the screens showed that I did not expect.**

```
1. weight gap past the first directions:   sigma_2 = .67,  sigma_15 = .25,  sigma_39 = .19     at the same point, both nets
2. fixed metric, warm against random start:   within 2% at every k, on tables and on the read
3. tail against rank:   rank 128 gives rms .2, and .99 of the mass within a factor 3      so the tail is structure, not noise
```

**Corrected the same day: lead (3) is WRONG, and I told you it in chat.** An agent warned of photo noise, and I measured it. `pc_r4_s4_noise.py`:

```
two independent halves of each class's photos (10 and 10):    rms between them    3.07 (ResNet-18), 3.26 (ResNet-50)
rank 14, fit on half 1:     KL on half 1   .237      KL on half 2   1.01
rank 38, fit on half 1:     KL on half 1   .056      KL on half 2    .93       (ResNet-50: .163, .958 and .034, .838)
```

Why it matters: rank 128's small number is mostly a fit to the noise of these 20 photos per class. It is not shown to be structure. Much of the .245 that three rounds tried to certify may be the same. A held-out score of about 1.0 sits close to the real read's 1.06, so the gap "7 times above the floor" may be partly an artifact of this small table. That is a reading of two numbers and not a proof. Part of the half-table score is half 2's own noise, which no build can fit. The trunk's object is the least on P, so noise is still part of that least, but a claim could not name it. My prediction of .9 missed by more than 3 times. An agent's rougher split-half estimate gave about .8 and another's about 1.5 on the full table, so the size of the noise on this measure is not settled, though the in-sample and held-out KL numbers above are direct. S4's line of 1.0 may be below the noise floor, so it was a poor line. Leads (1) and (2) rest on the same noisy tables, so they are weaker than I said. (2) is what a low-rank-plus-noise table would do anyway, so it does not show structure.

**What the referee found in my drafts.** All fixed.

| I wrote | True |
|---|---|
| a proof needs a constant mass of bad rows | false: TQ30 needs none |
| "no choice of weights helps" | proved for product hole weights only, and my reason was wrong |
| a Gaussian xi gives KL = half the variance | impossible on finitely many classes. A limit |
| a few entries do not move sigma_2 | false: one rare pair can hold it near 1 |
| the entry bound and the row norm in TQ26 | I mixed them. The conclusion stands after the repair |

Limits of the checks, said plainly. `model/tq30_cover.py` has 0 fails on 24 small graphs, and it can only refute. At network level the free fit stalled at a loss scale of 1e-11, so only the proof's own build is checked there. No certified global search was run. So "exact is really exact" is not met by this check. TQ30 and TQ31 as I wrote them are not re-refereed yet.

| | |
|---|---|
| Trunk check | used: "a map of claims" (M_N at error 0); "complete for the zoom" (M_N is not complete, said in TQ30); "one-time computation" (no time limit, so the trunk is not contradicted); "new units that read only that group's units" (the box of TQ23.a); "inputs N was trained on" (not yet: I3). Not used: the nested zoom, the graded clause |
| proved | Propositions TQ26, TQ28, TQ30, TQ31, Lemmas TQ27, TQ29. Labels: none of the four. These are limits and tools. No OPT-KL result is new. No slot is filled |
| conjecture | DQ21 |
| checks | Part C, Round 4 rows. `rounds/round4/` |
| referee | `REFEREE-BQ12-2026-11-02.md` |
| your calls | none now. The factor 2 in P4 stays as approved |
| next, my view | (a) re-referee TQ30 and TQ31; (b) I3 and I4 for TQ30, which are provable or a named miss; (c) finish S4 and score it held out; (d) move the real-network work off the 20-photo table and onto the read, on as many photos as the disk holds |

## 52. Row 3, round 4 continued: the hard family reaches ordinary losses, and the read looks limited by its k numbers (2026-11-02; plan i-need-you-to-tingly-orbit.md; COARSE-THEORY.md B.Q.12; card 3-37)

**Row 3 is NOT closed. Nothing passes on a real network.** The impossibility side now works at the tail mass a real table has. The real-network side has a new object, and no theorem on it yet.

| Word | Meaning here |
|---|---|
| sealed photos | photos no fit has seen |
| movement a | how far a class's logit moves between the two photos of an area in the hard network |
| decoder | what turns the k numbers into 1000 logits. The trunk's box uses one linear map |

**Step 1. The hard family no longer needs small numbers (Proposition TQ31, part 2).** The second referee round proposed this, and I checked it by hand.

```
before:  tail mass tiny, movement tiny, loss about 1/poly      ->  "that is not a real table"
now:     any fixed movement a, any fixed tail mass             ->  loss of ordinary size
         one split of KL:   KL = sum over classes of s_i psi(e_i),  every term >= 0,  so a wrong sign must pay somewhere
```

Checked on 30 small graphs: 0 fails, tails up to .63, best fit .006 to .16. Why it matters: "a real table has a big tail" no longer answers this theorem. It stays conditional on the Unique Games Conjecture, and its tables stay tables of graphs.

**Step 2. Your IMPOSSIBLE test now reads:**

| Line | State |
|---|---|
| I1 | met |
| I2 | below about 1.36 with no conjecture; every factor only under the Unique Games Conjecture |
| I3 | outputs reached by real training from random starts (KL at most 1e-5). The map's claims do not hold on the trained network: units do not line up with areas. With soft labels every network is a global minimum of its own objective, so that clause says little |
| I4 | tail mass and range met. The shape of the loss is not shown |
| I5 | not met, not tried. I expect these graph tables to fail it |

Why it matters: this is still not "impossible on real networks". A theorem about graph tables does not reach a real table unless real tables are shown to be in the family, and I expect they are not.

**Step 3. Real networks: the 14-number limit.**

```
ResNet-18, k = 14, 35000 photos to fit, 10000 sealed:              sealed KL     fit rows
   plain read: 14 numbers -> one linear map -> logits                 1.487        1.173
   same 14 numbers -> a 2-layer decoder                               1.495         .890
   encoder and 2-layer decoder trained together, from the read        1.513         .942
   encoder and 2-layer decoder trained together, from random          1.678        1.026
   for reference: 64 PCA numbers -> a 2-layer decoder                  .350         .023
```

Why it matters: on photos the fit did not see, no decoder I tried beats the plain linear read from k numbers. So the cost looks like what k linear numbers of z cannot hold, not how they are read. This is one network, one k, one decoder size, 35 photos per class, and these are upper values. It is not a floor. The decoders overfit, and more photos could change it. These are validation crops, not N's training photos.

**Step 4. What died.**

| Screen | Result |
|---|---|
| S4: trees from agents | held out 1.69 to 1.73, Ward 1.73, shuffled 2.06, line 1.0. My line may be below the noise floor. A directed low-rank form beat trees by .1 to .17 |
| S7: Gaussian information floor | empty: 14 dimensions hold 8 nats, N has 5.7 |
| S7: closed-form encoder | 1.8 to 6.9 times the read fit. Predicted 1.5 times |

**Step 5. Said plainly.** Your rule asks for a pass or a proof. After this round I have neither. A proof that covers real networks needs a family that holds them, and I do not know how to define one that a theorem can reach. A pass needs a floor on the read within a factor of 2, and every floor so far misses, the best by far. The information picture in Step 3 is the first thing that points at what the floor should be: the least over M of what k linear numbers of z leave out. The hard step is an upper bound on what any k linear numbers can hold, and the finite list P makes that quantity 0 unless the argument uses more than P alone (a generic projection is one-to-one on a finite list).

| | |
|---|---|
| Trunk check | used: "new units that read only that group's units" (the read; the decoders are not networks the trunk allows, and I say so); "inputs N was trained on" (I3, and N's training photos are not on this disk); "a map of claims" (M_N, and agents' trees). Not used: the nested zoom, the graded clause, the one-time computation |
| proved | Proposition TQ31 part 2, by hand, two readers. Labels: none of the four. No OPT-KL result is new. No slot is filled |
| checks | Part C. `model/tq31_anytail.py`, `model/tq31_trained.py`, `pc_r4_s4_eval.py`, `pc_r4_s7_info.py`, `pc_r4_s8_decoder.py`, `pc_r4_s8b_joint.py` |
| referee | `REFEREE-BQ12-2026-11-02.md`, two rounds. TQ31 part 2 has not had a third reader |
| your calls | none are forced. Two to think about: does I2 count under the Unique Games Conjecture? And is the 2-layer test above worth repeating on N's real training photos, which needs a download of about 2.5 GB for 20 per class? |

**Corrected the same day, LADDER 52 Step 1.** A third reader showed that "loss of ordinary size" was false on the instances the reduction builds: with filler columns, the loss falls with the graph's largest degree, and the movement sits on a small part of the tail. I had told you this in chat. The reader gave a repair, and I checked it by hand: one weight and one heavy-class size per area, no filler (Definition DQ24). Then V_b(1) is of the order of U / E, the share of edges no two-siding can cut.

```
36 small graphs, 0 fails:   tail about .34, more than .998 of it moving;   fit / (U / E) stays within a factor of about 1.6 at each setting;   fit 0 when U = 0
```

Why it matters: on a YES instance the loss is of order eta and on a NO instance of order eta^(1/2), a constant at fixed factor, and not 1/poly. The final text of the repair has my check and the reader's sketch, with no separate referee read yet. Everything in this family stays conditional on the Unique Games Conjecture and stays on tables of graphs.

## 53. Row 3: the essay test of the hard network, and what it says about "clean" (2026-11-02; plan i-need-you-to-tingly-orbit.md; COARSE-THEORY.md B.Q.12; cards 3-37, 3-39)

**Row 3 is NOT closed. Line I5 is not met, and the test showed something larger: the paper's clean numbers, restricted to the photo list, do not separate hard tables from clean ones (see the correction at the end of this section).** I expected the hard network to simply fail. It does fail as built. But the hardness does not need that build.

| Word | Meaning here |
|---|---|
| shared cause | one unit or route that explains why two classes are linked |
| side link | a class in a row's head that is not a relative of the row's class |
| build E | the agent's hand-built network that carries a hard table |

**Step 1. The hard network, read as a network.**

```
area       two input points, and only that area's two units are on there
h_w        writes the whole row of the table
s_w        toggles three classes between the two points
reading:   a lookup table with a small toggle. Nothing is a fold, a judgment or a family
```

Why it matters: this is what I expected, and by itself it would have said the hard family is not clean.

**Step 2. But a table of the same hard family can be carried cleanly by the paper's numbers, on three small instances.**

```
build E:   a family gate  ->  two shared poles  ->  a coarse reader  ->  class judgments
           table error 0 against a hard table (Proposition TQ33)
           every computed error of D42 and D45:  0
```

Why it matters: every hardness proof reads only the table, not how units carry it. So any principle about units can separate hard tables only if it forces something on the table. D42 and D45 do not.

**Step 3. What E still lacks.** Inside a family it reads well. Across families nothing causes a side link. When the agent adds a unit per link (build G), the errors rise to .45 to .60. On ResNet-18, eps_V is .91, the worst of all. So D42's V rewards a network for not reusing anything.

Why it matters: this may be the first real hint at the missing claim: **every link in the table has a shared cause in the network**. It would force something on the table. I have not defined it and it is not measured. It is on card 3-39.

**Step 4. How far hard tables are from real ones, by the numbers.**

| Number | Hard tables | ResNet-18 |
|---|---|---|
| side links that return | all | 3% |
| effective off-own entries in a row | 2 to 6 | 50 |
| share of a row's movement in one direction | 1 | .64 |
| triangles among mutual competitive links (tau) | 0 on the reduction's graphs | .24, null .003 |

Why it matters: real competition is mostly one way and spread over many classes. These hard tables are the reverse, by build. That is a direction for a claim. It does not make a real least computable.

**Step 5. The budget.** At k = 2 the fit on sparse hard graphs falls to about 0 (C5: .021 at k = 1, under 1e-5 at k = 2). So all of this hardness is at k = 1 for now, and I do not have it at k = 14. Your target is k = 14 and 38.

Why it matters: the IMPOSSIBLE side is further from real networks than the headline says. A hard family at k = 14 would need rows with heads longer than k, and padding, and that is not done.

| | |
|---|---|
| Trunk check | used: "a map of claims" (M_N, and how units carry the same table); "the essay" (the reading, and the failure of D42 and D45 to capture it); "new units that read only that group's units" (the free code against the real read, Step 5). Not used: the nested zoom, the graded clause |
| proved | Proposition TQ33 by the agent, checked by hand by me, not refereed. TQ34 is a computed check on small graphs. Labels: none of the four. No slot is filled |
| checks | Part C. `rounds/round4/i5/` |
| your calls | card 3-39 (recommend: clean means more than D42 and D45). Card 3-38 (recommend: keep I2 strict). I continue under both recommendations unless you say otherwise |
| next, my view | (a) referee TQ33 and the per-area text of TQ31. (b) Define "every link has a shared cause" and measure it on the read, with held-out photos. Screen it before any proof, and it must leave TQ30's, TQ31's and TQ33's graphs out. (c) Settle gates on enough rows, since P4 reads the box |

**Corrected the same day, after the referee's fourth round. I told you these wrongly in chat.**

| I said | True |
|---|---|
| every clean error of D42 and D45 is exactly 0 on the hand-built network | only with D42 restricted to the photo list. D42 as written integrates over the whole input box, and there its error is above 0. It also holds on three small instances, after two hand adjustments, in floating point |
| "an equally hard table" | those three instances are not hard: C4 has U = 0, and C5 has U = 1. Hardness is about the family, as n grows |
| Proposition TQ34 | withdrawn. It is a Part C row now |
| TQ33's hardness | not proved as first written: a constant was not bounded. Repaired with per-area weights, Definition DQ25. The repaired text has my check only |

Also fixed: constants in TQ31, and DQ24's reason. TQ31's core and K hold, now with four readers on the argument. Step 3's reading is unchanged: D42's V rewards not reusing anything, and the agent reads this build as clean inside a family, and not across families.

**A check on my own next step.** LADDER 53 Step 3 suggested the claim "every link in the table has a shared cause". As stated it does not separate: build G has one shared unit per edge and carries the same hard table. So a claim must also limit how many causes there are against how many links, so that each cause explains many links. I have no definition of this yet. Whatever it is, it gets a screen on the read, with held-out photos, before any proof.

## 54. Row 3: the new bar, tested on two real networks (2026-11-03; plan pasted-content-id-c43d-okay-we-ve-serialized-possum.md; COARSE-THEORY.md B.Q.13, C.R5; cards 3-40, 3-41)

**Row 3 is NOT closed. The bar is now yours: "given this particular goal the developer has, are we optimizing KL". No method passes every line on both networks.** Your idea about goals held up, in a sharper form than I expected. The closed-form rule does not yet earn the new label. Every number here is an upper value, on sealed photos.

| Word | Meaning here |
|---|---|
| goal | weights on the input photos. "Don't care" is weight 0 |
| read | k numbers, each a linear function of the last block's z, then one linear map to the 1000 logits |
| the rule | a span in closed form, then a convex fit of the write (Definition DG2, Lemma TG1) |
| logit start | top directions of N's logits, with a least-squares read, then training stopped on validation photos |

**Step 1. What the bar changed.** Trunk T5 became T6, with your approval. At a squeeze there are three lines.

```
(a)  on new photos from the goal:   KL  <=  1.3 x the best trained read
(b)  both sizes together cost less than training one level
(c)  one fixed calculation: same inputs, same network
```

Why it matters: this removes the proof that nothing is better, which took four rounds and did not arrive. A referee caught one claim of mine, and I withdrew it: "the hardness proofs limit only the old bar" was not proved. What is true is Proposition TG4: they do not stop a rule from passing, and a pass on those tables carries no promise of being near the least.

**Step 2. A goal only helps when it is a family.** Best trained read, ResNet-18:

```
                              k = 14     k = 38
all 1000 classes               1.508      .656
123 random classes             1.245      .628
dogs and cats, 123 classes      .402      .149
```

ResNet-50 shows the same order: 1.624, 1.290, .400 at k = 14, and .559, .544, .124 at k = 38.

Why it matters: the number of classes alone buys little. A group of related classes buys a lot. This fits the essay's picture of real networks. I have not ruled out that dog photos are easier for any read, apart from the goal.

**Step 3. The rule against the lines.**

| Line | ResNet-18 | ResNet-50 |
|---|---|---|
| (a) at most 1.3 | passes on all 12 runs. Ratios .94 to 1.30. All classes at k = 14 passes by .001 | fails on 6 of 12. Ratios 1.02 to 1.74. Dogs and cats pass (1.02, 1.04) |
| (b) cost | fails on 5 of 6 goals | fails on all 6 |
| (c) same numbers | passes, on 2 goals tested | passes, on 2 goals tested |
| dogs and cats beat random classes | yes, by 3.8 and 5.1 times | yes, by 5.2 and 5.1 times |

**Corrected the same day, before I reported this to you. The 6 failures depend on which reference counts.**

| Reference for line (a) | ResNet-18 | ResNet-50 |
|---|---|---|
| the starts your approved text names: random, and from the rule's build | 12 of 12 pass (.94 to 1.30) | 12 of 12 pass (.94 to 1.24) |
| the script's line, fixed before the run: those, plus the logit start | 12 of 12 pass | 6 of 12 pass |

I added the logit start to make the reference harder, and I did not tell you that it went past the approved words. I keep the stricter number as the headline, since a better read of the same form is known on those runs. Nothing here closes row 3: line (b) fails either way.

```
why (a) may fail on ResNet-50, not proved:
   12 fit photos per class, z has 2048 numbers
   a 123-class goal has 1476 rows, fewer than 2048
   whitening then fits the list for any z:    random goals, k = 14:   fit .29     sealed 2.12
```

Why it matters: the two networks are not compared on equal terms, since ResNet-18 has 30 fit photos per class. So "ResNet-50 fails" may be "too few photos fails". It does not explain why dogs and cats stay near 1 with the same row count. And on all classes the rule also costs something apart from noise: on ResNet-18 at k = 14 it is 1.762 against 1.159 on its own fit photos.

**Added the same day, the photo-count control. It finished after the order changed in LADDER 55, so it opens no new round.** ResNet-18 at 12 fit photos per class passes line (a) on all 12 runs, against the four starts. All classes reach 1.18 and 1.20, against ResNet-50's 1.38 and 1.44. So photo count alone does not explain ResNet-50's failures. The network, or rows against the size of z, still could. Numbers are in Part C.

**Step 4. What I did not plan to find.**

```
logit start, then short training on validation:
   lowest sealed KL on 22 of 24 runs, within 1.065 on the other 2
   no random start
   one level is quicker than one level from a random start on 11 of 12 runs at k = 14
   both sizes together cost more than one random-start level on 11 of 12 goal runs
```

Why it matters: this is training, so it is what line (b) was written to keep out. But at the last block a read is a small fit, seconds to minutes, so line (b) may say little here. It may say much more on a whole network. Its ratio of 1 is selected over four starts, so it is not an independent test. This is on card 3-41.

**Step 5. Things to be careful about.**

| Item | Said plainly |
|---|---|
| Rule below 1 on dogs and cats, ResNet-18 | not optimal. Trained reads overfit there (fit .12, sealed .40). Regularised training or more photos could beat it |
| Agreement with N | dogs and cats .75, .86 (ResNet-18), .79, .88 (ResNet-50). All classes .34, .62, and .35, .67. Passing a KL line does not mean good enough for an app |
| Off the goal | KL 4.5 to 6.8 on photos outside dogs and cats. That is what "don't care" costs |
| The read | always-on. Gated reads stay unsettled |
| The photos | ImageNet validation crops, not N's training photos |
| Middle layers | not touched. The convex picture holds at the last layer only |
| rule_fast | added after the fact. Same KL within 2%, and it does not fix line (b). My prediction on it missed |
| My predictions | dogs and cats did better than I predicted. On ResNet-18, line (a) passed everywhere where I predicted one failure, and line (b) failed where I predicted a pass. All in Part C |

| | |
|---|---|
| Trunk check | used: "or a developer's" inputs (now the center); "new units that read only that group's units" (the read); the new (a), (b), (c). Not used: the map, the nested zoom beyond one shared span, the graded clause |
| proved | Lemmas TG1, TG3, Propositions TG2, TG4. Labels: tools and a limit. The final wording of TG4's comment has my check only |
| checks | `model/tg_goal.py`, 0 fails; Lean `goal_exchange`, `goal_exchange_approx`; `rounds/round5/` |
| label of the rule | ONE-BUILD. No method earned OPT-GOAL |
| referee | `REFEREE-BQ13-2026-11-03.md`, two reads |
| your calls | card 3-41 |
| next, my view | test the logit start at one middle layer, against ordinary distillation of that level, before any trunk text changes. Run ResNet-18 at 12 fit photos per class, to separate the network from the row count |

## 55. A meta-analysis: the project drifted, and the map goes first (2026-11-03; plan pasted-content-id-1c9a-okay-before-serialized-snail.md; ROADMAP.md "The order changed"; node R-channeltree)

**Yes, the project drifted. Nothing a developer or consumer can use has come out since 2026-10-18. Row 3 is parked. A complete agent-made map of ResNet-18 is now the next job.** You said: "before we can make any new progress, we need to have actually good maps."

**Step 1. The root cause: the object you asked for was replaced by one that was easier to prove things about.** I did this several times.

```
you asked for:     parent channel -> child channels, collapse a child, the parent absorbs the error
what I worked on:  proxy areas  ->  squared logits  ->  tables of class rows  ->  k linear numbers read from one layer
```

Why it matters: each step gave real theorems, and each step moved further from your picture. The "optimizer things" of LADDER 54 are the end of this chain.

| Finding | Evidence |
|---|---|
| Your tree left the work | "parent" or "child" on 28 lines of LADDER 1-40, on 0 lines of LADDER 41-54. I counted this |
| Row 3 took everything | 14 sections in a row. Two fan-out rounds cost about 6.6M tokens, the map pilot 0.55M |
| The last session did not use the map | LADDER 54's trunk check: "Not used: the map, the nested zoom" |
| The map sat behind a gate that was never passed | ROADMAP session 3 waited for session 2, which never started. That gate came from my plan, and you approved it |
| No whole-network build in this track | Every ResNet build is the last block only |
| Cale is absent from the math | "Gemma" appears 0 times in the roadmap, LADDER, decisions and paper. I counted this |
| Each map request went elsewhere | 10-29 gave a script with one fixed rule. 10-31 put the map behind the gate. 11-01 gave a 20-minute pilot of about 60 channels, never extended |

| Shared causes | Evidence |
|---|---|
| The direction changed often | The meta-analysis counts at least 14 question changes in 15 days, some yours and some mine. I did not recount this |
| You pushed row 3 too | "close row 3 once and for all", 2026-11-02 |
| The rules favor theorems over reading | The math guards gate runs behind paper items, so looking at neurons is discouraged by the structure |

**Step 2. Two trees, and I chose yours.**

| | The paper's tree (DH1) | Your tree |
|---|---|---|
| A node is | a layer and a region of photos | a channel |
| An edge is | a smaller region inside a larger one | a parent channel that builds a child channel |
| On ResNet-18 | district errors .31 to .96, channels on for 80 to 98% of photos | never built. The ConvNeXt map found a parent for 39 of 41 sampled neurons |

Why it matters: the pilot's agents could read the channels, with 23 of 24 blind tests passed. So the reading worked, and the claim type failed. A map in the old vocabulary would have looked bad again. Node R-channeltree records this. The trunk does not change.

**Step 3. What "complete" will mean, said plainly.** The map will read and link every stream channel. It will not be "complete for the zoom" in the sense of DM6, which gives the exact closed form. DM6 covers the last layer only, and its item 5 fails for real squeezes here. The new parent claim is a first attempt to fix that, and I do not promise that it works.

**Step 4. The new guard.** Each section from this one on must have a "Cale line" row and a "Whole-network build" row that says yes or no. You said a note is too weak, so this is in `tools/ladder_lint.py`, with four new self-tests.

**Step 5. A correction that came in the same day.**

```
stage 4, at a channel's peak:    z = 18.45    the skip gives 1.32    the residual path of block 1 gives the rest
stages 1, 2, 3, at the peak:     the skip gives .70 to .77 of z
```

Why it matters: correction of LADDER 45 ("mostly its own skip") and card 3-34 (".88 skip"), both wrong for stage 4. My plan of today repeated the claim. The large "own" entry in the feeder table is the skip plus block 1 reading the same channel and amplifying it. So in stage 4 the job forms in block 1, and links inside the stage are real links. Details: `map_agents/resnet18/results/PREDICTIONS.md`.

The photo-count control finished. Its result is in LADDER 54 and Part C (C.R5): photo count alone does not explain ResNet-50's misses. Row 3 stays parked, and that is the last line on it.

| | |
|---|---|
| Trunk check | used: "a map M of N on P", "How M is found (by people, agents or a program) is outside the trunk". No trunk text changes |
| Cale line | nothing yet. This section sets the route, which is a reading and not a proven result: ResNet for one question, then SmolLM2's MLP width, then Gemma 3n width tiers |
| Whole-network build | no |
| label | none. This section holds no theorem |
| next | the better microscope, then a test slice of about 120 channels |

## 56. The ResNet-18 map: test slice. The channels read well at the top, and the tree you drew does not hold there (2026-11-03; plan pasted-content-id-1c9a-okay-before-serialized-snail.md; COARSE-THEORY.md B.CT, C.R6; card 3-42)

**The map works, and its shape is not your tree. At stage 4 the channels are readable and the links are causal. But one child has many weak parents, so a child that disappears has no one parent to take up the error. This is measured on ResNet-18 only, and 3 of my 4 pass lines failed.** All numbers are on sealed photos that no agent saw. These are validation crops, not N's training photos.

| Word | Meaning here |
|---|---|
| stream channel | one channel number through both blocks of a stage. `s4:179` is channel 179 of stage 4 |
| fill | set a channel to its mean, so it carries nothing |
| z | a channel's value before its ReLU |
| cut | the set of channels a smaller network keeps |

**Step 1. What the agents read.** Six agents read 106 of 960 channels, with 119 parent links. A seventh merged them into 17 families. About 1.2M tokens in all.

```
s1:29 tiny dark dot  ->  s2:71 dark dot  ->  s3:222 dark blotches on pale  ->  s4:35 spotted against plain coat
s3:118 thin lines fanning from a centre  ->  s4:242 hanging strands, s4:219 drape, s4:162 curls
s3:164 wheel on a painted body comes through, faces and daisies are turned away    (the closest thing to the essay's gate)
```

Why it matters: no channel is a car channel, a dog channel or a cat channel. Each one splits by a shape, a coat or a tone. This is the essay's picture of verbs and not nouns, found again on this network. Only the first chain runs from stage 1 to stage 4 among those read. Dog eyes and car lights share `s2:71`.

**Step 2. The lines I fixed before the runs.**

| Line | Pass | Result | Verdict |
|---|---|---|---|
| Specific enough to lose: a reviewer predicts firing from the words alone | 85% | 44%. s4 80%, s3 29%, s2 20%, s1 13% | Fails |
| Real parents: pick the true parent of 5 from the words alone | 70% | 57%, chance 20% | Fails |
| Parents act: a parent's removal moves the child 2 times more than controls | most | 92% of 119 links | Passes |
| Families act: name the classes that lose most | most of 15 | 5 of 15. Mean 4.9 of 10 named, chance 1 | Fails |

Why it matters: stage 4 is near the line, and the lower stages are far from it. I believe the test is weak below stage 4, since a stroke dial fires somewhere in nearly every class. That is plausible, and I have not shown it. So the failure stands as scored. The fix I propose is a crop-based blind test for stages 1 to 3, fixed before it runs.

**Step 3. The main finding: density.**

```
a stage 4 channel, on its 64 strongest sealed photos
   one listed parent removed                   ->  falls 3.8% of z
   all its listed parents removed together     ->  falls 6%
   only the top 4 of 256 feeders stay real     ->  97% of z^2 changes
   only the top 64 of 256 feeders stay real    ->  77% of z^2 changes
```

Why it matters: this is the direct test of "a child disappears and a parent takes up some of the error". At stage 4 a child is built from many small votes, so no single parent could take up its error. The tree agent reads the same thing from the cards: a graph with strong edges, and in places one parent pushes two children in opposite directions (`s3:17`, `s3:125`).

Two cautions. The feeders were chosen by input x gradient, which the agents found unreliable, and I ran no random set of 64 as a control. So what is measured is this: no 64 of its 256 inputs, chosen this way, carry a stage 4 channel. "Many weak parents" is my reading of that, and it may overstate the density. And the essay predicts exactly this for a small model, with narrower neurons at larger size. I have not measured the larger size.

**Step 4. A mistake of mine corrected.** LADDER 45 and card 3-34 said a last-layer channel is fed mostly by its own skip, about .88. My plan repeated it. It is wrong for stage 4.

```
stage 4, at a channel's peak:    z = 18.45    the skip gives 1.32    block 1's residual path gives the rest
stages 1 to 3, at the peak:      the skip gives .70 to .77 of z
```

**Step 5. The object in the paper, and what the referee did to it.** Definitions DC1 to DC3 write down channels, parent claims, the channel graph and cuts. Theorem TC1: when each kept channel's error is 0 on the exact kept set below it, every kept channel keeps N's value, and the whole cut loses only what its top layer loses. Label ONE-BUILD. The referee found TC2 false in one corner, and showed that no bound in the errors alone exists for this build (now Proposition TC3). It also said TC1 is close to a tautology, and the paper now says so. Its one piece of content: tests of one layer each certify the whole cut. The exact check has 0 fails. The fixed version has my check only.

```
keep only the dog and cat cards (22 + 20 channels):  median error .71 to 1.24     KL 5.88     agreement .001     random: 5.71 to 5.80
remove the vehicle cards (13 + 20 channels):         median error .02 to .04      KL .090     agreement .885     random: .090 to .124
```

Why it matters: this is the first whole-network number on this track, on the dogs and cats goal. The fill's means are over all classes, not the goal's photos, so the error column is not Definition DC2's error for this goal, which I did not run. The comparison with random is fair, since all use the same fill. The premise is far from true on a small cut and nearly true on a thin one. But on the thin cut the map did not clearly beat random: one seed ties it. That is LADDER 30's result again. It is a removal of about 4% of the channels of those two stages, so it is a small test. The network is not much smaller, since the lower stages and inner channels stay whole. So I do not call this the Step 6 build.

| | |
|---|---|
| Trunk check | used: "a map M of N on P", "a finite list of claims", "Each claim has one of the types named in the paper" (new type, node R-channeltree), "How M is found ... is outside the trunk", "D_P on the goal is measured". Not used: "new units", since a fill is not one. No trunk text changes |
| Cale line | nothing yet. If parents are many, a child's error has to go to many channels, and that decides the method carried to SmolLM2 and Gemma 3n's widths. Unknown today |
| Whole-network build | no. Whole-network fills were scored, but no smaller network was built |
| label | TC1, TC2 ONE-BUILD; TC3 a limit; TC4 a tool. Nothing earned OPT-GOAL or OPT-KL |
| checks | `model/tc_tree.py`, 0 fails; `REFEREE-BCT-2026-11-03.md`, one read of the first version; a bug in my fill (each layer needs its own mean) was fixed before these numbers, moving 93% to 92% |
| predictions | `map_agents/resnet18/results/PREDICTIONS.md`. Wrong: the skip at stage 4, the blind test at s3, s2, s1, the density, and beating random by 1.5 times. Right: real parents 50 to 75%, and which families pass |
| your calls | your read of the page, and card 3-42 |
| next, my view | ResNet-50's density first, then your neuron idea on the network with clearer parents, then the fill |

## 57. The map first: I withdraw "the tree does not hold", and I fix the microscope (2026-11-03; plan im-looking-at-the-playful-whisper.md; COARSE-THEORY.md C.R6, DC4; card 3-42 closed)

**LADDER 56's reading "many weak parents" and "the tree does not hold at the top" is withdrawn. It is a result about my microscope. Nothing else runs until the map is clean.** You said: "we cannot move forward until we get a clean mapping of the network, and i don't accept the network isn't clean, id rather guess the map is wrong".

| Word | Meaning here |
|---|---|
| inner channel | a channel between the two convolutions of a block, with its own ReLU |
| direct input | a channel one convolution below the child |

**Step 1. What I got wrong.**

```
what I measured at stage 4:

   s3 parent -> conv -> inner -> conv -> s4 block 0 -> conv -> inner -> conv -> s4 block 1 child
                 |______________________ 4 convolutions ______________________|
```

Why it matters: 0 of the 119 links lie inside a stage, although the skip is .07 of z at stage 4. Several sparse layers in a row can multiply into a dense result, so the density may come from my distance. The one-layer check tested this.

**Step 2. The one-layer check: distance was a large part of the old number.** All on sealed photos. These are validation crops, not N's training photos.

```
keep k inputs of a stage 4 channel, the rest at their mean:   share of z^2 that changes

                                        k = 4    k = 16    k = 64
old setup, four convolutions away        .97      .91       .77
new, one convolution away                .43      .15       .04
new, a random list                       .95      .92       .76
```

Why it matters: "no 64 inputs carry a child" came from the setup. With one convolution, 16 inputs of 513 come close, with a root mean square error of about .39 of z. The comparison is not like for like: the old setup kept 64 of 256 and crossed three ReLUs.

**Step 3. Where I was wrong, and the reference.** My counts failed: half of the summed effect takes 76 inputs at stage 4, and I predicted 30. Three-way bundles did not lower that count. Position inside the 3 x 3 window gave no large gain.

```
inputs for half of the effect, and what k = 16 leaves

ResNet-18 s4b1 stream from its inner channels         78 of 513     .156
ResNet-18 s4b1 inner from s4b0 stream                119 of 512     .456
ConvNeXt-Tiny stream direction from its neurons      109 of 3073    .117
ConvNeXt-Tiny neuron from its stream directions      156 of 768     .595
```

Why it matters: on this measure ResNet-18 looks about as dense as a network the essay calls clean. So my prediction of 30 was too tight, and the count is no defect in itself. This is a reference and it is not like for like: ConvNeXt's stream has no privileged basis, and it has more units and fewer photos here. It does not prove your tree. Agents and sealed tests must do that.

**Step 4. The old cards also fail a better test.** A reviewer sees a card's words and 12 sealed crops, and names the 6 that fire. A wrong card scores low (-.11 on 9 cards), so the test can tell cards apart.

| Stage | Old class test | New crop test | Line |
|---|---|---|---|
| s1 | 13% | 27% (chance 28%) | 85% |
| s2 | 20% | 64% | 85% |
| s3 | 29% | 48% | 85% |

Why it matters: the old test was only part of the problem. The old cards are not specific enough. Stage 1 sits at chance, which fits weak cards and also fits a test that cannot score a dial, so stage 1 gets a test with drawn images when a later wave reaches it.

**Step 5. A first look at inner channels, and its limit.**

```
s4b1i:191  firing to 4.0   a spoked wheel under a rider          dogs on the negative side
s4b1i:58   firing to 2.6   tabby faces                           dogs and blue birds on the negative side
4 random inner channels    firing to 1.0 to 2.1                  none reads at a glance
```

Why it matters: the old map read no inner channel, and two of them read at once. But I chose those two because they were strong inputs, and the random four did not read. So this is a lead, not a result. The words above are my reading, and no test has checked them.

**Step 6. What changed in the lines before the new cards.** "Parents carry" could not be passed by any card as written, so its measure changed. The crop test moved to 24 crops. The next wave covers stage 4's four layers only, since per-block nodes use up the token limit there. All of this is in `PREDICTIONS.md`, with my predictions, before any new card exists.

| | |
|---|---|
| Word | Meaning |
|---|---|
| wave | one round of agents over part of the network |

| | |
|---|---|
| Trunk check | used: "a map M of N on P", "each inside one hidden layer", "How M is found ... is outside the trunk". No trunk text changes |
| Cale line | nothing yet |
| Whole-network build | no |
| label | none. This section holds no theorem. Definition DC4 adds inner channels as nodes |
| checks | `map_agents/resnet18/tests/test_ev2.py`: the exact sum holds on 12 kinds of node, and a fill by hand matches it. Results: `results/one_layer_check.json`, `one_layer_map.json`, `reference_convnext.json`, `review2/old/` |
| next | agents read stage 4's four layers, then the sealed lines |
