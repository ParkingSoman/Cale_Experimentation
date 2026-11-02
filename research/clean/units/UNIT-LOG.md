# Unit branch: lab log

Paper: `research/clean/curves/UNIT-BRANCH.md`. Plan: `~/.claude/plans/oh-yeah-last-session-tingly-milner.md`.
This log is the record. It is not a source of proof.

## 2026-10-27: exact checks of Part B

| Script | Serves | Result |
|---|---|---|
| `u_tu5.py` | TU5 | 50 random two-branch networks (2 hidden layers, sign gate on x1, disjoint supports), random covering boxes across both layers: joint logits equal the right single build's logits at every point (max error 0.0); D adds to 3e-14 |
| `u_tu5.py` | TU3 | 200 random last-layer pairs that fire together, random free boxes: the bound holds in all; largest |I| / bound = 0.15 |
| `u_tu5.py` | TU7 | instance u = relu(x), v = relu(x - 1), P = {1/2, 2}, D2: D*(S, 0) = 17/4, D*(T, 2) = 0, joint 1/4, I = -4 (exact) |
| `u_tu4.py` | TU4 | 30 random linear instances: with Sigma_ST = 0 the cross term is 0 to 2e-14 and the joint equals the sum of singles; with correlated activity the closed form joint = single - |Pi|^2 holds to 2e-14 and |Pi|^2 > 0 in all 30 |
| `u_tu6.py` | TU6 | joint build at budgets (1, 2) exact on P (fractions); covering impossibility: z_A + z_C = -19/20 at A and at C; B, D off on A, midAC, C; numeric D2*({A, C}, 1) = 1/3 (search, not a proof), above the proved bound 1/8 |
| Lean | TU2, TU4, TU5, TU9 | `lean/coarse/Coarse/Unit.lean` (`silent_add`, `cross_zero_add`, `cover_compose`) compiles |

## 2026-10-27: Step 7 toy, a network with gates in front of districts (predictions written before the run)

Question (Conjecture DU2): when N has a gate in front of each district (Hypothesis H-district), do FREE coarse units
still help across districts, or does the free optimum keep to its own district (so independence holds without the
covering restriction)?

Network `toy_district.py`: input on the unit circle (plus a small radial jitter), two districts: "V" on the right arc,
"A" on the left arc. Last hidden layer: per district, one gate unit (on on the whole arc) and four member units
(narrow direction detectors along the arc), members of different districts never on together. Four classes: V1, V2
(left and right halves of the V arc), A1, A2. Members write to their half's class, gates to both classes of their
district. A control network `nogate` moves the arcs so that one V member and one A member fire together on some points.

Sets: S = V's four members, T = A's four members (gates kept). Budgets k in {1, 2} each. Least builds by Adam from
many starts (lab evidence, not Part B).

Predictions:
1. Covering class, gated network: I = 0 at every budget (Theorem TU5; any nonzero value is optimiser noise).
2. Free class, gated network: |I_free| is small compared with D*(S) + D*(T): below 10% of it. (DU2 says the free optimum
   is covering; if the free optimum has units that fire on the other arc, this prediction fails.)
3. Free minus covering loss on the gated network: below 10% of the covering loss. (DU2.)
4. Control without separation: |I_free| clearly larger than on the gated network (TU7).

### Results (2026-10-27, `toy_district.py`, `toy_district.json`; 16 Adam starts, 2500 steps each)

| Network | Class | Budget | D*(S) | D*(T) | joint | joint's boxes alone | additivity error of the joint's boxes |
|---|---|---|---|---|---|---|---|
| gated | free | (1,1) | 0.120 | 0.126 | 0.067 | 0.187, 0.242 | -0.36 |
| gated | free | (2,2) | 0.013 | 0.002 | 0.011 | 1.63, 1.30 | -2.92 |
| gated | free | (1,2) | 0.120 | 0.002 | 0.027 | 0.74, 1.80 | -2.52 |
| gated | covering | (1,1) | 0.161 | 0.155 | 0.328 | 0.173, 0.155 | 0.0 |
| gated | covering | (2,2) | 0.152 | 0.152 | 0.317 | 0.162, 0.155 | 0.0 |
| gated | covering | (1,2) | 0.252 | 0.087 | 0.345 | 0.161, 0.184 | 0.0 |
| control | free | (1,1) | 0.260 | 0.184 | 0.261 | 2.00, 0.96 | -2.70 |
| control | covering | (1,1) | 0.263 | 0.221 | 0.469 | 0.274, 0.222 | -0.026 |

Verdicts on the predictions:
1. HELD exactly: covering boxes on the gated network add with error 0.0 at every budget (TU5). The nonzero I estimates
   in the covering rows are optimiser noise: the single fits are worse than the joint's boxes used alone (for example
   0.252 against 0.161).
2. FAILED: free boxes help across gated districts strongly (joint 0.067 against singles 0.245 at (1,1)). In the joint
   optimum each box alone is far worse than in the pair: each box does part of the other's job. Theorem TU10 (written
   after the smoke run showed this) explains it: the members' negative pre-activations still vary on the other arc.
3. FAILED: strict covering boxes cost a lot (at (2,2): 0.15 against 0.013 and 0.002).
4. FAILED: the control without gates shows about the same free interaction as the gated network. Gates do not stop
   free help (TU10 says they cannot).

### Covering on P (Remark B.3), `toy_pcover.py`, `toy_pcover.json`

Boxes with any signs, constrained to be off at the points of P outside the set's support (penalty, then the leak is
checked and clipped; largest leak 8e-6).

| Budget | D*(S) | D*(T) |
|---|---|---|
| (1,1) | 0.120 | 0.126 (equal to the free single optima) |
| (2,2) | 0.026 | 0.002 (free: 0.013, 0.002) |

Reading: the free single optima are (almost) silent outside their own district already. Independence costs nothing in
the single problems here. What it forbids is the cross-district help of the joint problem. The price of strict
covering (m >= 0) comes from forbidding negative reads INSIDE the district, which independence does not ask for.

## 2026-10-27 (night): the output-faithful box (plan `~/.claude/plans/okay-nice-work-last-soft-peach.md`)

Shonu's reading of the last result: if N is clean and well trained, it should already be (mostly) independent, so a
result that says "not independent" means we are looking at N the wrong way; and independence should be a graded entry
of the cleanliness list, not a yes/no decision. Diagnosis: Theorem TU10's premises (a member's pre-activation separates
the points of P; T not exact) say nothing about cleanliness, so TU10 holds for every network. It is a statement about
the free box, which reads how far below zero an OFF member is. N's readers never receive that value.

### Essay, raw quotes (https://already-clean.goodfire.com/essay.md, fetched 2026-10-27; hedges kept)

| Section | Quote |
|---|---|
| Appendix, Weak Activations: Further Measurements (figure) | "Keeping roughly the strongest quarter of positive activations in Inception's final layer preserves **69.47% accuracy, versus 70.08% originally**. ConvNeXt's late MLP branches are similarly tolerant of weak-response removal, with their residual bypass intact." (This is the number Shonu remembered as 72% vs 71%.) |
| Do Weak Activations Matter? | "What we see is that you can ablate almost all the weak activations of the network for almost no cost. The places where removing them costs the most are some of the easiest to understand. ... In vision models, it's the early layers, which function like dials: the difference between slightly bright and medium bright matters, but for a later detector, like king penguin, low or very low evidence doesn't." |
| same | "what we lose when we cut weak activations ... it's usually fine distinctions between things the neural network is already confused about." |
| How I Think About Neurons | "A neuron is less about what it sees than about what it does to the flow coming in, and often its job is to gate a region of the network." |
| same | "The negative side of neurons matters almost as much as positives for understanding them, even in ReLU networks where the negative side is cut off. A neuron is built from votes ... The negative side never gets gradient on its own, but the individual votes do whenever the neuron is on" |
| same | "Smaller models may stop at coarser distinctions, with something closer to a general "car" neuron." / small model: "cars, trains, and carts in one neuron, with boats, planes, and a water-based ox-cart ... on its negative side." |
| What I Mean by Clean | "a network's parts are built to be read, and built upon, by its downstream readers ... become another reader. The more we read it the way its own readers do, the more interpretable I think it will look." |
| A City vs a Knot | "pale, wavy, and head-side ear neurons all feed one hanging-ears neuron, a natural place to start." / "The family turns out to be three separate routes" / "tethered abstraction: views that let us zoom out while keeping the larger patterns connected to the neurons and circuits that support them." |
| More on Circuit Ecology | "A car neuron provides a reusable judgment and a sheltered place for taxis, ambulances, and other specialists to develop." / "neurons aren't really for copying; they're for computing distinctions." |
| Edge Featurizers | "These motifs also predicted where ablation would matter much better than comparable groupings of activations: removing their neurons caused more damage on the images that used them. That makes connections a promising level of abstraction" |
| What Neurons Do Under Pressure (3b crush) | "The neurons stay interpretable and generalized, but coarser: one "curves & loops" neuron instead of ten curve detectors, two circle neurons instead of several circle types" |

Reading for the unit: "become another reader" says a coarse group should depend on the image only through what the
group SENDS to its readers (its outputs after ReLU), not through how far below zero its OFF members are. The ablation
quote says even weak positive outputs carry little; below-zero values are not sent at all.

### Predictions, written before `toy_faithful.py` runs

Class "output-faithful on P" (Definition DU6 to be written): the box's output is a function of the members' outputs on
P. Hand result (Lemma TU12 to be proved): when S's in-boundary values at P's points outside supp(S) affinely span the
read space, each box unit is either off on all of them or a constant unit. Both toys meet that (checked by hand for
T61's instance: B vs midAB forces m_A = 0, D vs midCD forces m_C = 0).

| Instance | Prediction |
|---|---|
| \|x\| on {-2,-1,1,2}, D2, S = relu(x) at budget 2, T = relu(-x) at budget 0 | N's logit bias fixed: I = -9/16 (help only through a constant: joint 11/16 vs 5/4). Free logit bias in every build: I = 0 (both 11/16) |
| circle toy, (1,1) | joint = singles' sum within 0.005 (singles = covering-on-P values 0.120, 0.126); constants not used |
| circle toy, (2,2), (1,2) | \|I\| < 0.005 |
| T61 instance (D2), ({A,C}:1, {B,D}:1) and (1,2) | \|I\| only from constants; I >= -(shift gains); I = 0 with a free logit bias |

### Results so far (`toy_faithful.py`, 8 Adam starts x every split of the budget into covering-on-P and constant units)

| Instance | DS | DT | joint | I | Prediction |
|---|---|---|---|---|---|
| \|x\|, N's bias fixed | 0 | 5/4 | 1/6 (0.16667) | -13/12 | FAILED (predicted -9/16) |
| \|x\|, free logit bias | 0 | 11/16 | 1/8 | -9/16 | FAILED (predicted 0) |
| circle toy (1,1) | 0.1199 | 0.1255 | 0.2454 | 3e-9 | HELD |
| circle toy (2,2) | 0.0095 | 0.0074 | 0.0149 | -0.0020 | HELD (\|I\| < 0.005) |

Why the \|x\| prediction failed (my error in the hand computation): I treated S's constant unit as a global logit
shift. The optimiser pairs it with a covering unit of S that cancels the constant on S's own inputs. The net effect is
a constant that acts ONLY where S is silent, which here is exactly T's support. So S's coarse version says "S is off,
so answer like T". That is a use of S's silence as an opposite. TU13's lower bound still holds: sigma_T = 5/4 - 1/8 =
9/8, so I >= -9/8, and -13/12 > -9/8.

Reading through the essay: "Think of negative poles as gates in a circuit, not a neuron's opposite." A coarse unit
that turns S's silence into a vote for T uses S's off state as an opposite. A second principle from the essay, "silence
stays silence" (a box sends nothing where its members sent nothing), removes this channel. Output-faithful + silence
= covering on P (TU12 with v = 0), and TU5 then gives I = 0 exactly. So the two essay principles together DERIVE the
covering-on-P class. The circle toy's free singles were already of this kind at (1,1).

### All results (`toy_faithful.py`, `toy_faithful.json`)

| Instance | Budgets | Logit bias | DS | DT | joint | I |
|---|---|---|---|---|---|---|
| \|x\| | (2, 0) | N's | 0 | 1.25 | 0.1667 | -1.083 |
| \|x\| | (2, 0) | free | 0 | 0.6875 | 0.125 | -0.5625 |
| circle toy | (1,1) | N's | 0.1199 | 0.1255 | 0.2454 | 0.0000 |
| circle toy | (2,2) | N's | 0.0095 | 0.0074 | 0.0149 | -0.0020 |
| circle toy | (1,2) | N's | 0.1199 | 0.0074 | 0.0199 | -0.1074 |
| circle toy | (1,1) | free | 0.0095 | 0.0087 | 0.0199 | +0.0016 |
| circle toy | (2,2) | free | 0.0063 | 0.0073 | 0.0150 | +0.0014 |
| circle toy | (1,2) | free | 0.0095 | 0.0073 | 0.0199 | +0.0031 |
| T61 instance (D2) | (1,1) | N's | 0.6246 | 0.6246 | 0.8809 | -0.368 |
| T61 instance (D2) | (1,2) | N's | 0.6246 | 0.5000 | 0.8572 | -0.267 |
| T61 instance (D2) | (1,1) | free | 0.5095 | 0.5095 | 0.8573 | -0.162 |
| T61 instance (D2) | (1,2) | free | 0.5095 | 0.4286 | 0.6963 | -0.242 |

Verdicts:
1. Output-faithful boxes alone do NOT give independence. The constant they send where their set is silent helps the
   other set strongly (\|x\|, toy (1,2), T61). Predictions for \|x\| and T61 FAILED. The toy predictions at (1,1) and
   (2,2) held.
2. With a free logit bias in every build, the toy's single losses drop by a factor of about 12 at budget 1 (0.12 ->
   0.0095): most of a single district's coarse loss was a missing constant. The shared bias then couples the pair a
   little (I between +0.0014 and +0.0031).
3. Silence-preserving boxes (DU7) need no run: TU15 part 2 gives I = 0 exactly on all three instances (supports are
   disjoint in each). Their single losses are the covering-on-P values (toy: 0.120, 0.126 at (1,1); 0.026, 0.002 at
   (2,2), from `toy_pcover.json`; T61: at least 1/2 per group of two detectors, TU6 part 4).
4. Reading of T61: {A, C} has no hub (each detector is its own district), so by CU1's picture it is not a unit. The large
   silence price there reflects a grouping that cuts across districts, not a failure of the principle.

Referee of B.5 (opus, 2026-10-27 night): 2 errors (TU14 norm bound without projection; DU5 false as written), 6 gaps,
3 wording items. All fixed in the paper (B.5 rewritten; TU16 added; DU5 restated with meeting units; DU4 under
H-district-theta and silence-preserving boxes). Report: `research/clean/curves/REFEREE-B5U-2026-10-27.md`. Fixes not
re-refereed.

## 2026-10-27 (late night): what the "help" is (Shonu: "what is that error really coming from?")

### Anatomy of the free joint build, circle toy (`help_anatomy.py`, `help_anatomy.json`)

| Budgets | Box unit | Fires on | Writes (largest) | Reading |
|---|---|---|---|---|
| (1,1), joint 0.056 | V's box | 24 of 24 points below the horizon (y < 0), on BOTH arcs; 6 above | V1 +1.07, A2 +1.09 | a shared "down" unit: V1 and A2 are the two lower classes |
| (1,1) | A's box | 24 of 24 points above the horizon, on both arcs; 4 below | V2 +1.02, A1 +0.86, V1 -0.73, A2 -0.72 | a shared "up" unit |
| (1,2), joint 0.039 | A's box, unit 1 | 40 of 48 points, both arcs | all writes about 0.1 | a bias unit (a constant) |

The four classes are (left or right) x (up or down). The kept gates say left or right. N computes up/down TWICE,
once inside each district (V1/V2 specialists and A1/A2 specialists). The coarse network computes it ONCE and shares it
across districts. So the "help" at (1,1) is a coarse network finding a distinction that both groups make. The
essay's word for this: "Cars and dogs might be grouped together because they share a pose." At (1,2) the help is a
default offset (a constant). Both are possible only because V's box can see A's images: the members of both districts
read the raw input x, so V's inputs vary across A's points (pre-activations -16 to -33 there).

### Predictions for `toy_shelter.py` (written before the run)

Two-layer versions of the toy. Layer 1: the two gates and, per district, the four narrow caps (silent outside their
arc). Layer 2: each district's four members read only their own district's layer-1 units. Boxes are FREE (any ReLU
units of the set's in-boundary).

| Network | Mix | Prediction |
|---|---|---|
| sheltered (members read only their own district's layer 1) | arcs only | free help only through a constant (Theorem TU17 to be written): I(1,1) near 0; I(1,2) clearly negative |
| sheltered | arcs + background points (no district on; equal mass) | \|I\| at every budget below 20% of the arcs-only \|I\| |
| shared pose (layer 1 also has up and down units, read by both districts' members) | arcs only | I(1,1) < -0.02; the helping unit's largest read weight is on the up or down unit of N |

### First result of `toy_shelter.py`: prediction FAILED, and why

sheltered, arcs only, (1,1): DS 0.118, DT 0.1225, joint 0.0017, I = -0.239 (predicted near 0). Each box's single unit is
on at every point of BOTH arcs. On the other arc its inputs are all 0, so it takes one constant value there (TU12 part
1, here automatic: the inputs are silent). Reading: a one-unit coarse district needs a unit that is ON across its whole
district (the district's "judgment", like a car neuron on for every car). The sheltered members read only narrow caps,
and a ReLU of caps cannot be on across the whole arc while off outside it (the caps are 0 at the arc's ends AND outside
it). So each box makes an always-on unit, and the other box absorbs its constant on the other arc. The pair builds two
district judgments out of each other's silence. The missing piece of N is a gate that the members read (principle Q4
of the main paper, "a gate in front"): its region marks the district.

### Added prediction (before the run): `q4` variants

Members also read their district's gate with a positive weight (member = relu(cap + 0.2 g - 0.56), still off where the
cap is 0). The box can read the gate.

| Network | Prediction |
|---|---|
| sheltered + q4, arcs only | single losses at (1,1) well below 0.118 (at most half); \|I\| at (1,1) below 10% of DS + DT |
| sheltered + q4, with background | the same |

### Results of `toy_shelter.py` and `toy_q4.py` (free boxes, readers' biases fixed at N's)

| Network | Mix | I(1,1) | I(2,2) | I(1,2) | DS, DT at (1,1) |
|---|---|---|---|---|---|
| sheltered | arcs | -0.239 | +0.000 | -0.118 | 0.118, 0.123 |
| sheltered | + background | -0.119 | +0.000 | -0.059 | 0.059, 0.061 |
| pose | arcs | -0.052 | +0.003 | -0.023 | 0.036, 0.044 |
| pose | + background | -0.024 | +0.002 | -0.012 | 0.018, 0.023 |
| q4 (gate in front) | arcs | -0.203 | +0.001 | -0.099 | 0.101, 0.105 |
| q4 | + background | -0.101 | +0.001 | | 0.050, 0.053 |

Predictions: sheltered "I(1,1) near 0" FAILED; background "\|I\| below 20%" FAILED (it halves, because the background
halves the arcs' weight, not because the channel weakens); q4 "single loss at most half" FAILED (0.101).

Why q4 failed (reasoning, then a test): one new ReLU unit writes along one vector, and only where it is on. In a
district with two classes (lower, upper) the build must make V1 - V2 positive on the lower points and negative on the
upper points. With one unit, V1 - V2 = u(x) (V_1 - V_2) has one sign wherever u > 0, and is 0 where u = 0, because the
logits' biases are N's (0 here) and the kept gate writes equally to V1 and V2. So one unit can say "more upper" but
never "upper versus lower". It needs an offset, and the readers (the logits) cannot re-fit theirs. The other district's
box supplies one through its constant. That is the second kind of help: a smuggled reader bias. The gate in front did
not help because the missing piece is the readers' offset, not a district marker. (`toy_faithful.py` already showed
this: with a free logit bias the toy's single losses fell from 0.12 to 0.0095.)

### Predictions for `toy_bias.py` (free boxes, every build may re-fit the readers' biases; written before the run)

| Network | Prediction |
|---|---|
| original toy (members read the raw input) | help stays (kind 1, a shared distinction): I(1,1) < -0.02, with box units that fire on both arcs on one side of the horizon |
| sheltered, q4 | help gone: \|I\| <= 0.01 at every budget, and single losses at (1,1) below 0.03 |
| pose | help stays smaller: I(1,1) < -0.005, with the helping unit reading N's up or down unit most |

### Results of `toy_bias.py` (free boxes; every build re-fits the readers' biases)

| Network | DS, DT at (1,1) | I(1,1) | I(2,2) | I(1,2) | Same network, readers' biases fixed: I(1,1) |
|---|---|---|---|---|---|
| original toy (members read the raw input) | 0.0094, 0.0087 | +0.0061 | +0.0022 | -0.0034 | -0.18 (free search, section 22) |
| sheltered | 0.0000, 0.0001 | +0.0001 | 0.0000 | +0.0001 | -0.239 |
| q4 (gate in front) | 0.0009, 0.0014 | +0.0001 | +0.0011 | +0.0009 | -0.203 |
| pose (shared up/down units) | 0.0105, 0.0179 | -0.0028 | +0.0019 | +0.0015 | -0.052 |

Verdicts:
1. sheltered and q4: HELD (\|I\| <= 0.01, singles below 0.03).
2. original toy: FAILED (predicted I(1,1) < -0.02). Almost all of its help was also the readers' offset: with re-fit
   biases each district's one unit already does its job alone (0.009), so there is little left to share. What remains
   is small in both directions: -0.0034 at (1,2); POSITIVE +0.006 at (1,1).
3. pose: FAILED in size (predicted < -0.005; got -0.0028).
4. New effect: positive I. In the joint build both groups share ONE set of reader biases (the logits are read by
   both), while each single build has its own. Two groups that want the readers' thresholds in different places
   compete. Small here (<= 0.006).

Reading. In these toys the "help" was almost entirely one thing: coarsening a group to one unit removes the contrast
its members made (TU19: one ReLU unit is one-sided), and the contrast needs the readers' thresholds to move. Our
builds froze the readers' biases at N's values, so the best build moved them through a back door: another group's new
unit, which takes one constant value outside its own group, acts as an offset. With the front door open (readers
re-fit their biases), the groups are independent up to small terms, even with free boxes and inputs shared by both
groups.

What a re-fit bias cannot remove (exact): the \|x\| instance. S = relu(x), T = relu(-x), T at budget 0, free logit
bias: D*(T, 0) = 11/16 (the best constant), and S's free box at budget 2, {relu(x), relu(-x)}, is exact, so I = -11/16.
Here S's inputs (x) carry T's whole job: relu(x) and relu(-x) are the two halves of one dial, x. The grouping cut the
dial in half. This kind of help is real information sharing, and I < 0 says "these two groups are one feature".

### Predictions for `cu1_dial.py` (Conjecture CU1: a district's one-unit coarse version is its judgment; written before the run)

Single build of district V at budget 1, readers re-fit their biases, every network of `toy_bias.py`.
1. The unit is on at >= 90% of V's points (on across the district, like a "car" neuron on for every car).
2. Its value is monotone along the arc: \|Spearman(value, angle)\| >= 0.9 on V's points (a dial: "how upper").
3. Its write is mostly the contrast: \|V2 - V1 component\| > the common (V1 + V2) component.

### Results of `cu1_dial.py` (district V's single build at budget 1, readers re-fit their biases)

| Network | loss | on at V's points | reads (N's units) | write V1, V2 | Reading |
|---|---|---|---|---|---|
| sheltered | 0.00005 | 92% | lower caps +1.16, +1.16; upper caps -1.16, -1.16; bias +1.56 | +1.10, -2.35 | a dial: lower half on the positive pole, upper half on the negative pole |
| q4 | 0.0009 | 96% | lower caps -1.66, -1.90; upper caps +1.90, +1.35; gate -0.23; bias +3.18 | -0.66, +1.36 | the same dial, sign flipped |
| pose | 0.0105 | 95% | caps +0.96, +0.43, -0.31, -1.86; up unit +0.30 | +0.76, -1.14 | the same dial, plus a little of N's pose unit |
| original | 0.0094 | 58% | raw input 6.87 x1 - 3.61 x2, bias -6.2 | +2.63, -2.84 | a one-sided "lower part" unit; the reader's bias makes "upper" the default |

Verdicts: prediction 1 (on at >= 90%) HELD for sheltered, q4, pose; FAILED for the original toy (58%). Prediction 2
(\|Spearman\| >= 0.9) FAILED everywhere (0.79 to 0.88: the caps are bumpy, the value is only roughly monotone).
Prediction 3 (contrast > common part) HELD everywhere.

Reading: the one-unit coarse version of a district of siblings is a CONTRAST DIAL over the district's own members:
positive weight on one subset of siblings, negative weight on the rest, on across the district, with the reader's
re-fit threshold deciding where the answer flips. The essay: "an ambulance neuron might have a taxi on its negative
pole"; the small model's last-layer neuron has "cars, trains, and carts" on its positive side and "boats, planes" on
its negative side. When the inputs are not sheltered (original toy), the same contrast is made by a one-sided unit
plus the reader's default. CU1 as written ("non-negative weight to the hub", "the judgment") is wrong in form: the
coarse unit is a dial with siblings on both poles, not a sum. The paper's CU1 is restated.

Referee of TU15-TU20 (opus, late night): 3 errors. Main one: "re-fit readers remove the help" (my B.6 (1) and CU3) is
FALSE in general. Counterexample: two points, T at budget 2 spends a spare unit on a constant that acts only where T
is silent (= at S's point), I = -(w h_s(a))^2 / 4 under re-fit biases. So "silence used as an opposite" survives
DU8; DU7 forbids it. In the four toys it did not appear, because the singles were already near 0. CU3 withdrawn;
TU21 (two-sided bounds) proved instead. Report: `research/clean/curves/REFEREE-B5U2-2026-10-27.md`.

## 2026-10-27 (late night): the weights side (Shonu: R-readbias DECIDED, "yea they can move their biases"; "im also wondering if the same error exists on the weights side")

A reader's bias is its weight on an always-on input. The builds also freeze the readers' weights on every KEPT unit
(here: the logits' weights on the two gates). A box of T can stand in for a change of such a weight on S's points if
it can compute that kept unit's signal there. Re-fitting those weights is the same move as re-fitting the bias.

### Predictions for `toy_weights.py` (before the run): free boxes, readers re-fit bias AND their weights on kept units

| Network | Prediction |
|---|---|
| original toy | single loss at (1,1) at least 30% below the bias-only 0.0094 (the gate is on across V's arc: re-weighting it is a district judgment); \|I\| <= 0.002 at every budget |
| sheltered, q4, pose | \|I\| stays <= 0.003 (nothing left to smuggle: a box cannot see the other district) |
| \|x\| instance (by hand, before the run) | unchanged: the kept unit relu(x) is 0 on T's points, so re-weighting it cannot replace T; I = -11/16 stays |

### Results of `toy_weights.py` (free boxes; readers re-fit bias and weights on the kept gates)

| Network | DS at (1,1): bias only -> bias + weights | I(1,1) | I(2,2) | I(1,2) |
|---|---|---|---|---|
| original | 0.0094 -> 0.0050 | +0.0086 | +0.0005 | -0.0006 |
| sheltered | 0.0000 -> 0.0000 | +0.0001 | +0.0000 | +0.0001 |
| q4 | 0.0009 -> 0.0006 | +0.0001 | +0.0011 | +0.0001 |
| pose | 0.0105 -> 0.0101 | -0.0010 | +0.0017 | -0.0040 |

Verdicts: original single at least 30% lower: HELD (47%). Original \|I\| <= 0.002: FAILED at (1,1) (+0.0086, a
COST, not help). sheltered and q4 \|I\| <= 0.003: HELD. pose \|I\| <= 0.003: FAILED at (1,2) (-0.0040).
Readings:
1. Re-weighting the kept gates acts as each district's judgment (the gate is on across its arc and off elsewhere): it
   halves the original toy's single loss. This is TU22's mechanism with a gate whose value varies, not a constant.
2. No cross-district help appears that bias re-fitting had not already removed. The largest term is now a cost
   (+0.0086 in the original toy): its boxes read the raw input, so each single box leaks onto the other arc, and in
   the joint build the two leaks meet (TU3's product term). The readers' shared weights cannot serve both.
3. The pose toy keeps a small help (-0.004 at (1,2)). Its pose units are kept but NOT read by the logits, so DU9 cannot
   re-weight them; a box can read them (they are in both districts' inputs). This is the rebuilt-unit channel through
   one of N's own shared units.
Answer to Shonu's question: yes, the same freeze exists on the weights side (TU22 says what re-fitting kept weights
buys), but in these toys it carried little help beyond the bias; its main effect is on the single losses.

Shonu DECIDED R-readweights ("yes"). Referee round 3 on DU9/TU22: true as stated; 1 gap and 7 wording items, fixed:
TU22 now needs no least builds, holds for any box class, and needs only that the flat function on a set's support be a
combination of the set's own kept units (several varying gates can do it). Report appended to
`research/clean/curves/REFEREE-B5U2-2026-10-27.md`.

## 2026-10-28: activations and edges (UNIT-BRANCH.md A.5, A.6, B.6, B.7, D.3)

Shonu asked why the essay calls activations misleading, as math. Read the full essay text again (scratchpad copy,
md5 0cf6c457...). Result: an activation alone drops a unit's in-votes (TA3) and its readers (TA1, TA2, TA4); the
computation is weights times ON/OFF gates along paths (TA5). Our co-firing unit (TU8) uses a new write of an activation,
so a unit that nothing reads still breaks U4 (TA6). Lean: `Coarse/Edge.lean`, builds, no sorry.

| Script | Serves | Prediction (written before the run) | Result |
|---|---|---|---|
| `ta6_unread.py` | TA6 | D2: I = 0 at budget 0, I = -49/20 at budget 1; D_P: I < 0; gamma_v != 0 | HELD (D_P: I = -0.2147, gamma_v first coordinate -0.343) |
| `ta7_reader.py` | TA7 | I_rem / eps^2 -> 0 (orthogonal writes, 3 logits), 1/3 (3 logits), 1/4 (2 logits); free-write gamma_v != 0 in the orthogonal case | HELD (-0.00033, 0.3330, 0.24999992 at eps = 0.001; gamma_v = (-0.243, 0.332, -0.089)) |
| `ta4_rank.py` | TA4 | activation order z, u, v; removal-cost order v, u, z; z (write (1, 1)) costs exactly 0 | HELD (KL costs 0.120, 0.0050, 0.0) |

New node R-edgetether (PENDING, card 3-17): write-tethered boxes (Definition DA2), Conjecture CU7.

Referee round 1 (REFEREE-EDGE-2026-10-28.md): 4 errors, all in B.7/D.3. Main one: a unit with W_out = 0 has no
out-boundary (DU1), so DU2 already forbids its box any write. TA6 now uses a softmax-unread unit with edges
(column alpha 1); `ta6_unread.py` re-run with column (1, 1): same numbers, HELD. TA6 fails under DU9 (readers re-weight
v). New Proposition TA8: with write-tethered boxes, help needs T's activity to meet S's error inside T's columns; a
softmax-unread set is exactly independent. CU7 is now a graded conjecture (orthogonal writes interact at eps^3).

## 2026-10-28: Trunk T5 (maps), Theorem TM1, Proposition TA12

Shonu proposed that the trunk may assume a mapped network (T5). COARSE-THEORY.md A.7 defines maps (claims with errors);
B.M proves TM1 (a map's regions split a grouping exactly, class (b) builds with free logits) and TM3 (where T43's
hardness sits). UNIT-BRANCH.md B.8 proves TA12 (with re-fit reader weights, a softmax-unread unit is tied to the sets it
fires with).

### Predictions (written before the runs)

| Script | Serves | Prediction |
|---|---|---|
| `ta12_sub.py` | TA12 | P = {1, 2, 3}, u = relu(x - 1/2) column (1, 0) = S at budget 0, v = relu(x - 3/2) column (1, 1). Readers re-fit bias and weights. I(S, 0; {v}, 0) > 0, gamma != 0 at the joint optimum. Control v' = relu(x - 7/2) (never on): I = 0 to solver precision |
| `tm1_split.py` | TM1 | Two regions A, B (3 points each) and one point in no region; 3 logits; S = district of A (3 units) at k = 1, T = district of B at k = 1, judgment units kept. V_b({S, T}) = L_S + L_T, V_b({S}) = L_S, V_b({T}) = L_T, so I = 0, all to optimiser precision (|I| <= 1e-5). Control 1 (shelter broken: one member of S also reads B's feature, still off on B; S at k = 2): I < 0 beyond 1e-4. Control 2 (no judgment units kept): |I| > 1e-4 |

### Results, first instance (3 points per region)

`ta12_sub.py`: HELD. Main: D*({v}, 0) = 0 (-7e-18), D*(S, 0) = 0.00186 (readers re-weight v), joint 0.0477, so
I = +0.0459; gamma = (-0.073, +0.073). Control (v never on): I = 7e-18.

`tm1_split.py`, first instance: main HELD but uninformative (L_S = 9e-18, L_T = 3.7e-6: with 3 points per region, one
box unit plus the kept units fit almost exactly). Shelter control: I = -3.70e-6 = -L_T exactly (the sign and the
mechanism held: S's box, reading uB through a1, did all of T's remaining job), but the threshold 1e-4 FAILED because
L_T itself is 3.7e-6. No-judgment control: I = -4.4e-7, FAILED (|I| > 1e-4). Reason: the instance is too easy to test
magnitudes. Fix: 8 points per region (next run), predictions unchanged in sign; thresholds restated relative to L_T:
shelter control I < -0.1 L_T; no-judgment control |I| > 1e-4 (unchanged, a real test).

### Results, second instance (8 points per region): the optimiser, not the theorem

main FAILED as a numerical check: V_T = 7.40e-5 < L_T = 1.71e-4 found by the optimiser. But every single build of T
gives a feasible point of L_T's problem (its terms on region B, with w_B = its bias), so the true L_T <= 7.40e-5: the
random-restart L-BFGS did not reach the infimum of L_T (errors ~1e-4, the size of the effects). The controls inherit the
same noise (shelter: I = +3.4e-5; no judgment: I = -0.0282, HELD, far above the noise).
Fix (rule 4: the statement is fine, the check was too weak): TM1's proof holds box by box. For FIXED boxes the logits are
the only free part, and that fit is convex (multinomial logistic with soft targets). New check `tm1_split.py` v3:
(a) for 200 random box pairs, J(C_S, C_T) = l_S(C_S) + l_T(C_T), single_S(C_S) = l_S(C_S), single_T(C_T) = l_T(C_T),
by Newton's method, predicted to 1e-9; (b) controls by the same fixed-box test: shelter broken -> max deviation > 1e-4;
no judgment -> max deviation > 1e-4; (c) infima by optimising the reduced (box-only) function, reported, not a test.

### Results, `tm1_split.py` v3 (box by box, Newton; 200 random box pairs)

| Case | max |J - l_S - l_T| | max |single_S - l_S| | max |single_T - l_T| | interaction of fixed boxes (min, max) | Verdict |
|---|---|---|---|---|---|
| main (claims hold) | 7.8e-17 | 6.9e-17 | 8.6e-17 | (-7e-17, +1.1e-16) | HELD |
| shelter broken (a1 also reads uB, still off on B) | 3.3e-3 | 2.1e-3 | 1.1e-16 | (-1.33e-3, +2.13e-3) | HELD (> 1e-4) |
| no judgment units kept | 0.128 | 0.038 | 0.124 | (-0.070, -0.030) | HELD (> 1e-4) |

Reading. With the map's claims true, the split is exact for every box, as the proof says (the infimum statement follows).
Breaking shelter breaks it for S's side only (T's single build is still exact: T's box is still sheltered), and the
fixed-box interaction takes both signs. Without kept judgment units every pair interacts (always negative here:
the groups lend each other offsets, the "silence used as an opposite" channel).

## 2026-10-28 (later): complete maps (Shonu: "the exactness promise applies only to complete maps")

Trunk T5 now promises exactness and bounds only for maps complete for the zoom (Definition DM6). New paper items:
DM7 blend claim, DM8 flat claim, DM6 complete map, TM1 part 4, TM4 part 2, TM5 and Corollary TM5.1, CM3.

### Predictions (written before the runs)

| Script | Serves | Prediction |
|---|---|---|
| `curves/model/tm5_blend.py` | TM5 | on detector networks with T43's mixes (kappa = 2, m = 2, H one edge; kappa = 2, m = 3, H a path; kappa = 3, m = 3, H a triangle; kappa = 3, m = 4, H a path, 300 random sets), for every set S tested: eps_blend(S) = 0 <=> S has no bad pair <=> T59 part 1's linear program is feasible; and when there is no bad pair, T59.2's unit reproduces t_S at every point (max error < 1e-9). No exception |
| `tm1_split.py` v4 | TM1 part 4 | add a unit f in no district, constant on A and on B (different constants) and varying on the no-region point: the split holds box by box (< 1e-9); control: f varies on region B -> deviation > 1e-4 |
| `tm4_dist.py` | TM4 part 2 | the instance of TM4 part 2 for delta = 0.1, 0.01, 0.001: V_b = 0 (< 1e-9), L_T fixed and > 0, eps_dist(j1) -> 0 |

`tm4_dist.py`, first run: FAILED on "L_T fixed" (0.0690, 0.0667, 0.0665 for delta = 0.1, 0.01, 0.001), though
V_b = 0 (< 4e-17) and L_T stays > 0.066. Cause: my instance gave j1 the gap weight 0.5, so N's own output at b moves
with delta. The paper's proof said "L_T does not depend on delta", which needs N's weight on j1 to be 0 (class (b)'s
free logits may still read j1). Fix: paper states N's logit weights on j1 are 0; script re-run with weight 0.

### Results (complete maps)

| Script | Result | Verdict |
|---|---|---|
| `curves/model/tm5_blend.py` | kappa=2 m=2: 15 sets (6 without bad pair); kappa=2 m=3: 63 (10); kappa=3 m=3: 511 (33); kappa=3 m=4: 300 sampled (3). Exceptions to "eps_blend = 0 <=> no bad pair <=> T59 LP feasible": 0. T59.2's unit error at most 7.1e-15 | HELD |
| `tm4_dist.py` (re-run, N's weight on j1 = 0) | V_b <= 4.3e-17 for delta = 0.1, 0.01, 0.001; L_T = 0.066492 at every delta; eps_dist(j1) = 0.091, 0.0099, 0.0010 | HELD |
| `tm1_split.py` v4 | flat unit f (0.8 on A, 0.3 on B): max deviation 1.6e-16 on 200 box pairs. f varying on B (constant on A): joint and single_T deviate from l_S + l_T by 1.75e-3 (single_S exact), but the fixed-box interaction J - single_S - single_T stays within 1.3e-16. Reading: such an f acts as one more kept unit of region B (it is constant on A), so it helps T equally in both builds; it breaks the formula with l_T, not the independence | HELD |

`curves/model/tm5_blend.py` v2 (after referee round 3: DM7 made two-sided; both logit orders): 1778 group checks,
0 exceptions, T59.2 error <= 7.1e-15. The referee's point: the one-sided claim gave the big group blend error 0 with the
logit rows swapped (vertex points are strict maxima), which would have brought TM3's conflict back.

### 2026-10-28: Shonu doubted the InceptionV1 map result; check `curves/pc_map3.py`

Prediction (written first): the old share measure would give Zoom In's car detector 4c:447 a share near the layer
median. FAILED: 4c:447 has share 0.010 (strong firing: 0 of 16 images are dogs). The top car-side and dog-side writers of
mixed5b are mostly <= 0.06 under strong firing. So the measure works; my reading was wrong: the layer medians (0.27 to
0.43) run over channels the dog/car output does not depend on. Withdrawn: "most neurons are shared between dogs and
cars" and "fur, eyes, texture are shared" (dog/other map). Remaining point: map errors do not weigh a unit by how much
the output depends on it.
