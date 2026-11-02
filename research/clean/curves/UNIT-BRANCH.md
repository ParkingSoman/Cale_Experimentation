# What is the unit of coarsening? A branch of the coarse theory

Branch of `COARSE-THEORY.md`, started 2026-10-27. Status: side branch under node R-circuit (PENDING). Nothing here is
a step toward Trunk T4 until Shonu decides the nodes it uses. Plan: `~/.claude/plans/oh-yeah-last-session-tingly-milner.md`.
Lab log: `research/clean/units/UNIT-LOG.md`. The rules of `CLAUDE.md` in this folder hold: Parts A, B and D carry no
measurements, every result has a hypothesis tag, a proof and a Lean line.

Shonu (2026-10-27): "we're just kind of adding and adding these theorems for very particular neuron-level things, when,
in fact, evidence likely suggests that we need a circuit-level view of things." And, on how to choose the new unit:
"figuring out what a general unit is and its properties are, and then using mathematics given that prior to discover
its identity", and "follow the math like a river rather than trying to force it if it points us to a different general
unit".

# Part 0: the question

Trunk T4 replaces groups of units of ONE hidden layer. Card 3-14 asks whether the thing replaced should instead be a
circuit. This branch does not start from "circuit". It asks what any unit of coarsening must do, writes those demands
as mathematics (U1 to U5, Definition DU5), and then proves which sets of N's units meet them (Part B). The named
candidates (one-layer group, circuit, district) are compared with the result afterwards.

Summary of what Part B proves (revised after the referee report REFEREE-UNIT-2026-10-27.md).
1. Substitution (U1): a set can be replaced by one box, with an acyclic result, exactly when it is convex in N's
   graph (Lemma TU1). One-layer groups are a special case.
2. Independence (U4) is the demand that decides the unit. With linear units and a constant metric, uncorrelated
   activity on the mix gives independence at every budget, and independence forces F C_S Sigma_ST = 0 (Theorem TU4).
   With ReLU units, free coarse units can "help" a neighbour's error, and then independence fails, also in a perfectly
   symmetric detector network (Proposition TU6, on Theorem T61's instance).
3. If every coarse unit is silent at the mix's inputs outside its own members' support (a box covering on P,
   Remark B.3; covering boxes, Definition DU3, are a special case), independence holds exactly, at every depth, for
   convex unlinked sets whose members never fire on the same input of the mix (Theorem TU5). In the last hidden layer,
   it fails when a unit of T is on at an input where removing S changes the output, unless the effects cancel
   (Proposition TU7).
4. So, for such boxes, the finest unit system of the last hidden layer is, apart from cancellation, the partition into
   **co-firing components** (Corollary TU8). Under Hypothesis H-district (a co-firing hub per district), these
   components are districts. Covering composes, so units nest (Proposition TU9).
5. Free coarse units leave no independence between sets that cannot be removed for free: a free unit can read the
   members' negative pre-activations and help an inexact neighbour, even when the two sets never fire together
   (Theorem TU10, Corollary TU11). Covering on P keeps independence and can cost loss (TU6 part 4). This is a fork
   forced by theorems: node R-indep (card 3-15).
6. Activations and edges (A.5, A.6, B.6, B.7, added 2026-10-28). Read as mathematics, the essay's distrust of
   activations is: an activation alone drops a unit's in-votes (TA3) and its readers (TA1, TA2, TA4), and it carries a
   gauge (rescaling, splitting); the network's computation is weights and ON/OFF gates along paths (TA5). DU2 tethers a
   box's writes only by graph, so the co-firing unit of item 4 can use a new write of an activation that changes no
   class probability: a softmax-unread unit still breaks U4 (TA6). In the removal interaction the readers enter as a
   factor next to co-firing (TA7). Write-tethered boxes (Definition DA2, node R-edgetether, PENDING) bring the readers
   into U4: help needs overlap inside the helper's columns, and orthogonal columns leave only a third-order interaction
(TA8). Conjecture CU7 is the deep version.

# Part A: starting points

## A.0 Setting

N is a plain layered ReLU network: input x in R^d, hidden layers 1, ..., L, unit u of layer l has pre-activation
z_u = sum over units j of layer l - 1 of W[u, j] h_j + b_u (h_j = x_j for l = 1) and output h_u = relu(z_u), and the
logits are y_N = sum over u of layer L of W[., u] h_u + c. A unit is **on** at x when z_u(x) > 0 and **off** otherwise.
The mix P is a finite list of inputs x with weights p_x > 0 summing to 1. For a network B with the same logits,
D_P(B) = sum over x of p_x KL(softmax(y_N(x)) || softmax(y_B(x))). With two logits, D2_P(B) = sum over x of
p_x (g_N(x) - g_B(x))^2, where g = y_1 - y_2 is the gap.

## A.1 Facts

**Fact FU1 (per-input sum).** D_P and D2_P are sums over the points of P of a term that depends only on y_B at that
point, and the term is 0 when y_B(x) = y_N(x). (Definition.)

**Fact FU2 (OFF units send nothing).** If h_u = 0 at x, every weight leaving u contributes 0 at x. (relu(t) = 0 for
t <= 0.)

**Fact FU3 (the curvature of KL in the logits).** For a probability vector q and logits y, the function
phi(delta) = KL(q || softmax(y + delta)) is smooth and convex with Hessian diag(s) - s s^T, s = softmax(y + delta). Its
largest eigenvalue is at most 1/2. Reason: for a unit vector v, v^T (diag(s) - s s^T) v is the variance of the values
v_i under the law s. By Popoviciu's inequality it is at most (max_i v_i - min_i v_i)^2 / 4, and
(v_i - v_j)^2 <= 2 (v_i^2 + v_j^2) <= 2.

## A.2 Definitions

**Definition DU1 (graph, paths, convex sets, boundaries).** N's graph has an edge j -> u when W[u, j] != 0. A path is
a chain of edges. A set S of hidden units is **convex** when every path that starts and ends in S stays in S. The
**in-boundary** of S is the set of units outside S (and input coordinates) with an edge into S. The **out-boundary**
is the set of units outside S (and logits) with an edge from S. Two sets S and T are **unlinked** when no path joins a
unit of S to a unit of T, in either direction.

**Definition DU2 (box, replacement, build).** Let S be a convex set of hidden units and k >= 0. A **box** for S of
size k is a set of k new ReLU units, wired as follows: a new unit reads only units of S's in-boundary and earlier new
units of the same box (the build is a feedforward network, not necessarily one with N's layers). Each edge of N from a member of S to a unit r of the out-boundary
is removed, and r may read any new unit of the box with any weight, in place of the members. All other weights and
biases are N's. The **build** N[S <- C] is N with S replaced by the box C. Two boxes for two disjoint sets give the
build N[S <- C_S, T <- C_T]. (Trunk T4's class (b) at one layer is the special case where S lies in one hidden layer
and the box is one layer of units relu(m^T z_S + beta). Lemma TU1 says why S must be convex.)

**Definition DU3 (covering box).** For S convex and given values of its in-boundary, let z_S be the pre-activations
that N's own weights give to the members of S from those values (computed layer by layer inside S). A box is
**covering** when, for every real value of its in-boundary (not only the values N attains), if every member of S is
off then every new unit of the box is off. For S inside one hidden layer and new units relu(m^T z_S - tau), the box is
covering when m >= 0 and tau >= 0; these are the folds of Theorem T29 of `COARSE-THEORY.md`, and, when the set where
all members are off is non-empty, that theorem says they are the only covering units read from z_S. A box is
**covering on P** when, at every x in P outside supp(S) (Definition DU4), its units are off in every build where S's
in-boundary takes N's values at x. A covering box is covering on P.

**Definition DU4 (support, silence).** The **support** of S on P is supp(S) = {x in P : some member of S is on at x in
N}. A replacement of S is **silent** at x (in a given build) when every unit of S's out-boundary receives from the box
at x exactly what it received from S's members at x in N.

**Definition DU5 (least loss, interaction, the five demands).** Fix a class of boxes (all boxes, covering boxes, or
boxes covering on P). D2* and the D2 interaction are defined the same way with D2_P in place of D_P.
1. D*(S, k) is the infimum of D_P(N[S <- C]) over boxes C of size k in the class. D*(S, k_S; T, k_T) is the infimum of
   D_P(N[S <- C_S, T <- C_T]) over pairs in the class.
2. The **interaction** is I(S, k_S; T, k_T) = D*(S, k_S; T, k_T) - D*(S, k_S) - D*(T, k_T). (It is the second
   difference of the set function "loss after replacing" of Sundararajan et al. (2020), with the empty replacement as
   baseline and D_P(N) = 0; literature table.)
3. The demands on a family of disjoint sets that a zoom would replace independently:
   - **U1 substitution**: each set can be replaced by a box and the result is a feedforward ReLU network (an acyclic
     graph of ReLU units; the layers of A.0 need not be kept).
   - **U2 tether**: each new unit belongs to one set of N's own units (no change of basis, no unit shared between sets).
   - **U3 boundary**: a box acts on the rest of the network only through the out-boundary of its set.
   - **U4 independence**: for every two sets S, T of the family and every budgets, I(S, k_S; T, k_T) = 0, and every
     pair of least boxes for S and T alone is a least pair for both together.
   - **U5 nesting**: a box of a box is a box of the union of the sets, in the same class.
   U2 and U3 hold by construction of Definition DU2. U1 is Lemma TU1. U4 is the demand that decides the unit. Its
   source is the zoom of a district without touching the rest ("switching ... costs time that does not grow with the
   size of N" in Trunk T4 asks it only for a chain of levels; asking it for every combination of levels of different
   sets is a new demand, node R-indep).
4. A **unit system** for the class is a partition of a set of hidden units into sets that pairwise meet U4. A family
   with one set meets U4 trivially; the content is in families of two or more sets that cannot be removed for free.

## A.3 Hypotheses (from the essay)

**Hypothesis H-district (a co-firing hub per district) (proposed; node R-district, PENDING).** The units of N's last
hidden layer split into districts. In each district one unit (its hub) is on at every input of P where some member of
the district is on. Members of different districts are never on at the same input of P.
Source in the essay: "a neuron in front of them acts as a gate: it responds positively to cars and negatively to dogs,
and its negative side keeps the dogs out. The dogs go on to their own area, which doesn't have to worry about cars." The
essay's gate sits UPSTREAM of the district ("in front of"). An upstream gate makes the second sentence true (members of
different districts are off together outside their gate's region) but does not by itself join a district's members
into one co-firing component; H-district adds a hub in the same layer (for example a general "car" unit next to the
taxi and ambulance units, as in the essay's "pale, wavy, and head-side ear neurons all feed one hanging-ears neuron",
where the hub is the reader one layer up). The essay does not claim that every unit sits in exactly one district.
Photos with a car and a dog together break the second sentence; Theorem TU3 prices that.

## A.4 What the essay says a unit is, and where it disagrees with the field

Quotes keep the essay's hedges.

| Essay sentence | What it asks of a unit | Where it enters |
|---|---|---|
| "a network's parts are built to be read, and built upon, by its downstream readers" | a unit is judged through what its readers receive | U3 (boundary); the metric of Fact FU3 is the readers' (the logits') sensitivity |
| "a neural network is almost a fractal of explanations, of coherent, useful groupings: at the macro scale, in its mechanisms, and at the micro scale" | units nest at every scale | U5 |
| "become another reader" | the coarse object is what a coarser reader would need | D_P with the developer's mix |
| "Every district, circuit and neuron is the network's own" | no change of basis | U2 |
| "views that let us zoom out while keeping the larger patterns connected to the neurons and circuits that support them" ("tethered abstraction") | each coarse unit maps back to a set of N's units | U2 |
| "It makes sense to think of cities in terms of districts, and I'm all for thinking in terms of groups of mechanisms" | the unit is a group of mechanisms | Corollary TU8 |
| "Think of negative poles as gates in a circuit, not a neuron's opposite" | a negative pole keeps inputs out of a region | covering boxes (DU3), Remark B.2 |
| "neurons aren't really for copying; they're for computing distinctions" | a coarse unit need not copy a member | the free class |

Where the essay disagrees with the field's idea of a circuit (literature table rows L1 to L7). In each row the branch
follows the essay.

| Field | Essay | Adopted here |
|---|---|---|
| A circuit is "a set of features, and the weighted edges that go between them", features being directions (Zoom In, L1) | "I'm now fairly skeptical of working in terms of features at all"; "the directions neurons write to and read from are a good place to start" | sets of N's own units (U2) |
| A circuit is "a subgraph of M responsible for some behavior", judged by faithfulness, completeness and minimality on one task (IOI, ACDC; L4, L5) | exploratory, not enumerative: "It starts with questions"; parts are built for their readers | the unit is fixed by N and the mix, not by one task metric. The mix plays the role of the task |
| Replacement models with units of their own, per prompt (attribution graphs, L6) | splitting the V unit "takes us further from what the neuron does"; "higher-level structures ... belong at the end, not the start" | no replacement units; boxes are tied to members (DU2, DU3) |
| A car feature stored across dog units for later read-out (Zoom In's packing example) | "nothing reads that combination"; it was shared pose | no packed units are assumed anywhere |
| Neurons as detectors (nouns) | "A neuron is less about what it sees than about what it does to the flow coming in" | the free class allows units that fire on non-members (TU6); covering forbids it (DU3) |

<!-- literature-table -->
| Row | Source | Object | Layers | Building blocks | How found |
|---|---|---|---|---|---|
| L1 | Olah et al. 2020, Zoom In (distill.pub/2020/circuits/zoom-in) | "A 'circuit' is a computational subgraph of a neural network. It consists of a set of features, and the weighted edges that go between them" | several | features (directions, often single neurons) of the network | reading weights between visualised features |
| L2 | Cammarata et al. 2021, Curve Circuits | the curve circuit, conv2d0 to mixed3b of InceptionV1 | five layers | the network's own neurons, in families | weights between families, synthetic stimuli, a hand-built copy |
| L3 | Elhage et al. 2021, transformer circuits | QK and OV circuits; "a sum where every term corresponds to an end-to-end path" | several | heads and weight matrices | exact algebra, attention frozen |
| L4 | Wang et al. 2022, IOI | "a circuit C is a subgraph of M responsible for some behavior"; faithfulness, completeness, minimality | several | heads, neurons | path patching on one task |
| L5 | Conmy et al. 2023, ACDC | subgraph; an edge is removed if its effect on a KL metric under corruption is below tau | several | components at a chosen grain | patching, greedy pruning |
| L6 | Ameisen et al. 2025, circuit tracing | attribution graph over cross-layer transcoder units | several | replacement-model units, not neurons | trained replacement, per-prompt attribution |
| L7 | Beckers and Halpern 2019; Geiger et al. 2021, 2025 | constructive abstraction: a partition of the low-level variables, each high-level variable a function of one cell, interventions commute | not fixed | clusters of the network's own variables (also rotated bases in the distributed form) | interchange interventions |
| L8 | Sundararajan et al. 2020, Shapley-Taylor index | second difference F(T u ij) - F(T u i) - F(T u j) + F(T) | n/a | set functions | definition |
| L9 | Kemeny and Snell (lumpability; via a secondary source) | a partition is lumpable when every state of a block has the same total probability of entering each block | n/a | Markov chains | definition |
| L10 | Telgarsky 2016 | deep narrow networks that shallow ones approximate only with exponentially many nodes | n/a | ReLU networks | proof |
| L11 | Theis, Korshunova, Tejani and Huszar 2018, "Faster gaze prediction with dense networks and Fisher pruning" | the Fisher information of a per-unit mask, used to score units for removal (recalled, not re-read) | one at a time | units (channels) of the network | second-order estimate |
<!-- /literature-table -->

Compatibility. L7 is the closest: a unit system (DU5 item 4) is the partition of a constructive abstraction, and U2 is
its "each high-level variable is a function of one cell", without rotated bases. L7 judges a partition by whether
interventions commute; this branch judges it by U4, whether least replacements commute. L8 is the interaction used in
DU5. L9 is the Markov analogue of a partition that a coarse reader can use without loss. L4 and L5 judge a subgraph by one
task metric; here the mix is the task, and the object is the whole network's loss. L6 is outside the chain (rule 5).

Lineage. The 2026-10-24 circuit unit (NEURON-MATH Part VI: an edge between clean families with a transported metric) and
Definitions D58 to D60 of `COARSE-THEORY.md` (a set of units in any layers, coarse circuits) are both kept as special
cases: D58's set is a set S of DU2 when it is convex (D60 left this open), and D60's coarse circuit is a box. What is
new here is the demand U4 and the identity it forces.

## A.5 Two readings of a unit (added 2026-10-28)

Shonu (2026-10-27): the essay says "activations are a false way of looking at things ... or just at least like a
misleading way, and that we should be looking at connections"; "we need to mathematically understand why are activations
misleading". A.5 and A.6 set up the two readings and tie each essay sentence to a statement. B.6 proves the statements.
B.7 reads this branch's own unit (TU7, TU8, TU10) through them.

**Definition DA1 (activation reading, in-votes, out-edges, edge reading).** Let S be a set of hidden units.
1. The **activation reading** of S on P is the list act_S = (h_u(x)) over u in S and x in P.
2. The **in-votes** of a unit u of layer l at x are c_{u,j}(x) = W[u, j] h_j(x), one per unit j of layer l - 1
   (h_j = x_j for l = 1). They add up to z_u(x) - b_u. This is the vote of Definition D44 item 1 of
   `COARSE-THEORY.md` for a dense layer.
3. The **out-edge** of S into a unit r of its out-boundary (or a logit) at x is e_{r,S}(x) = sum over u in S of
   W[r, u] h_u(x): what r receives from S. For one unit, e_{r,u}(x) is the vote a_{r,u}(x) of Definition D41 item 2,
   indexed by the voter instead of the reader. For a box C, e_{r,C}(x) is the sum over C's read units e of W[r, e] h_e(x).
4. The **edge reading** of S on P is the pair (the in-votes of S's members, the out-edges of S) at every x of P.
5. A set S of the last hidden layer is **softmax-unread** when P_perp W_out[., u] = 0 for every member u (P_perp as in
   Fact F11 of `COARSE-THEORY.md`): every member writes the same amount to every logit, so no logit difference, and no
   class probability, depends on S. A member may still have non-zero weights, and so edges in the sense of DU1.
Items 2 and 3 agree with the votes of D44 item 1 and D41 item 2 when N is in canonical form (D43), up to D41's
convention of reading through logit differences. The out-edge of a set or of a box is their sum. The use g_u of D44 (a
reader's weight times its gate) is the derivative form of the out-edges' effect. It is not needed below.

Comment (the discriminating test). Let c > 0. Multiply u's in-weights and bias by c and its out-weights by 1/c. The
function of N does not change (Fact F1 of `COARSE-THEORY.md`). Then h_u becomes c h_u, every out-edge of u stays the
same (Lemma TA2 part 2), u's in-votes are multiplied by c (their ratios stay the same), and the loss of every build stays
the same (relu(c z) = c relu(z) needs c > 0). Splitting u into two copies with out-weights a W[., u] and (1 - a) W[., u]
keeps every out-edge, and the loss of every build that replaces both copies or neither, and gives the activation reading
one more coordinate. So a quantity of the activation reading that changes under
these two moves is not a property of N's function. Supports, co-firing and correlations pass the rescaling test.
Variances, fixed thresholds and counts of coordinates do not. The canonical form (Definition D43) removes both moves.

## A.6 What the essay says about activations, sentence by sentence

Quotes keep the essay's hedges. The last column says whether a result of B.6 or B.7 proves the math reading ("proved"),
proves a statement next to it ("related"), or only discusses it in a comment ("discussed").

| # | Essay sentence | Math reading | Where |
|---|---|---|---|
| 1 | "It's tempting to think of a neuron's activation as just a number ... Everything feeding into it makes its own small semantic judgment, and the maximum is where they all agree most" | h_u is relu of the sum of its in-votes. The sum keeps one number out of n | proved: TA3 part 1 |
| 2 | "To see the shared pattern, you have to visualize what the neuron's inputs have in common, not the neuron on its own" (on 4e:55) | vote vectors with the same h_u can have different routes; two inputs of N have them when N attains such vectors (a premise about N and P). What inputs share for u is in the votes | proved: TA3 parts 1, 3 |
| 3 | "Looking only at the product hides how it was produced" (SwiGLU sign regimes) | one output value, many factorizations: a fibre of the map from causes to output | discussed: TA3 comment |
| 4 | "what fired weakly here may have fired strongly one layer back" | a small h_u can come from large in-votes that cancel (D41's cancelled share), when N attains such votes | proved for vote vectors: TA3 part 1 |
| 5 | "The negative side of neurons matters almost as much as positives for understanding them, even in ReLU networks where the negative side is cut off. A neuron is built from votes" | relu sends the whole half-space (sum of votes + b <= 0) to 0. The two poles differ in the in-votes, not in h_u | proved: TA3 part 2 |
| 6 | "In a real network, a representation matters only through what reads it"; "a network's parts are built to be read, and built upon, by its downstream readers" | the loss of any build depends on a set only through its out-edges | proved: TA1 |
| 7 | "nothing reads that combination" (the car-in-dogs circuit); "a single unread dimension" (the residual stream); the Social Security direction "is unused by the network" | the essay's object is a direction that no reader uses. The unit-level object nearest to it is a softmax-unread unit (DA1 item 5): it has an activation reading and changes no class probability. Activation statistics cannot tell it from a read unit | proved for the unit-level object: TA2 part 1, TA6 |
| 8 | "you can ablate almost all the weak activations of the network for almost no cost. The places where removing them costs the most are some of the easiest to understand" (the last layer; early layers "function like dials") | removing one last-layer unit costs at most (1/4) h^2 |P_perp w|^2, and to leading order (1/2) h^2 w^T H w: activation squared times the write's curvature. The essay's sentence is about cutting weak activations across the whole network; TA4 removes one unit | related: TA4 |
| 9 | "These motifs also predicted where ablation would matter much better than comparable groupings of activations" (edge motifs) | ablation effects are out-edge quantities, and two removals interact, to leading order, through activation overlap times reader overlap. Whether edge groupings predict ablation better is an empirical claim of the essay, not proved here | related: TA4, TA7 |
| 10 | "A neuron is less about what it sees than about what it does to the flow coming in, and often its job is to gate a region of the network"; "better understood as a verb than a noun ... it separates two subtle patterns at its positive and negative poles ... and pushes the flow of activations toward different neurons" | the output is a sum over paths of weight products times ON/OFF gates. A unit adds one gate to each path through it. The size of h_u comes from upstream | proved: TA5 |
| 11 | the old metrics were "rewarding explanations that could simulate activations rather than ones that were right, which often come apart"; "Simulation also rewards fine details of weak activations that matter little" | fitting the activation reading weights each input by h_u, not by what h_u does downstream | discussed: TA4 comment |

What the essay does not say. It does not say activations carry no information. It uses strong activations everywhere
("the strong positives and negatives matter a lot"; "In general I think what the neuron learns should be trusted, its
natural sparsity and activation ranges"). Read as mathematics, its claim is narrower: **the activation reading alone is a
lossy summary on both sides of a unit, and it carries a gauge.** On the input side it drops the in-votes (TA3). On the
output side it drops the readers (TA1, TA2). And it changes under rescaling and splitting, which leave N's function
alone (A.5). So it misleads when it is used as a unit's identity (rows 1 to 5) or as a unit's importance (rows 8, 9,
11). This paper adds a third use, its own: as the rule for grouping units (row 7, B.7); that reading is ours, not a
sentence of the essay.

The essay's own example, InceptionV1 4e:55, in the two readings (the essay's description, "Cars and Animals" and its
footnote 2; no measurement here):

```
in-votes (what feeds it)                 unit 4e:55                 out-edges (who reads it)
V-shaped flames     + \                                          / strong -> car units
V-shaped fox faces  +  >-- z = sum of votes --> relu --> h -----<
V-shaped water jets + /    round (O) shapes vote -               \ weak   -> cat path ("barely reaches the output")

activation reading: inputs with large h = cat faces and car fronts          -> "cat + car"
edge reading:       votes agree on a pointed V, the negative pole is an O,
                    the readers are car units                               -> "V versus O, read by cars"
```

**Definition DA2 (write-tethered box) (node R-edgetether, PENDING).** A box C for S is **write-tethered** when, for
every read unit e of C, the column of weights from e to S's out-boundary lies in the span of the members' columns
{W[., u] restricted to the out-boundary : u in S}. (A coarse set may merge and re-weight what its members already send.
It may not send along a direction that no member sends.) Source: "a representation matters only through what reads
it"; "become another reader". DU2 already tethers by graph: a box may only reach units of the out-boundary, so a unit
with no out-edge at all gets no write. In a dense layer the out-boundary is the whole next layer (or every logit), and
the graph tether imposes nothing. DA2 tethers by direction. For sets of the last hidden layer with two logits, modulo
the all-ones vector the columns span at most one direction, so DA2 changes the loss there only for softmax-unread sets
(as a constraint on weights it forbids more, but writes that differ by a multiple of 1 give the same softmax).

# Part B: theorems

## B.1 Substitution

**Lemma TU1 (a box is well defined exactly for convex sets) (under: none).** Let S be a set of hidden units.
1. If S is convex, the values of S's in-boundary at an input do not depend on the weights or units that replace S, so
   a box (Definition DU2) is a function from in-boundary values to out-boundary values, and every build N[S <- C] is
   acyclic: a feedforward ReLU network.
2. If S is not convex, some build N[S <- C] with a box of one unit has a directed cycle.
Proof. (1) Suppose an in-boundary unit j depends on S: a path runs from a member u of S to j, and j has an edge into a
member w of S. That path from u to w leaves S, against convexity. A directed cycle in a build must pass through a box
unit e (N's graph is acyclic, and the box's own edges go from earlier to later units). Follow the cycle from e: it
leaves the box at an out-boundary unit r that reads e, runs through units of N outside S, and comes back to the box
from an in-boundary unit j. In N, r reads a member of S and j feeds a member of S, so N has a path
(member) -> r -> ... -> j -> (member) that leaves S and comes back, against convexity. (2) A path from u in S to w in S
that leaves S has a segment r -> ... -> j whose units all lie outside S, with r reading a member of S (r is in the
out-boundary) and j feeding a member of S (j is in the in-boundary). Take a box with one unit e that reads j, and let
r read e (both allowed by DU2). Then e -> r -> ... -> j -> e is a directed cycle. QED.
Lean: none, a statement about graphs proved by hand.
Comment: a set inside one hidden layer is always convex, so Trunk T4's groups pass U1. A set across layers passes U1
only when it is closed under the paths between its members: a "circuit" in the sense of D58 qualifies as a unit only
when it is convex. When a box has more than one layer of units, the build may need more layers than N; U1 asks only for
a feedforward network.

## B.2 Additivity: when two replacements do not see each other

**Theorem TU2 (silence gives exact additivity) (under: none).** Let S and T be disjoint, convex and unlinked, and let
C_S, C_T be boxes for them. Suppose that at every x of P, the replacement of S or the replacement of T is silent in
the joint build N[S <- C_S, T <- C_T]. Then, at every x, the joint build's logits equal those of N[S <- C_S] or those
of N[T <- C_T], and
   D_P(N[S <- C_S, T <- C_T]) = D_P(N[S <- C_S]) + D_P(N[T <- C_T]).
Proof. Since S and T are unlinked, no unit downstream of T's out-boundary is in S or in S's in-boundary (a unit there
would give a path from T to S), so S's in-boundary values at x are the same in the joint build as in N[S <- C_S], and
S's box gives the same outputs; the same holds with S and T exchanged. At an x where T's replacement is silent, every
unit downstream of T receives what it receives in N[S <- C_S], so the logits equal y_{N[S <- C_S]}(x); and in
N[T <- C_T] alone T's replacement is silent at x too (its in-boundary is N's there), so y_{N[T <- C_T]}(x) = y_N(x).
The same with S and T exchanged. So at each x the loss term of the joint build is the loss term of one of the single
builds, and the other single build's term is 0 (Fact FU1). Summing over P gives the identity. QED.
Lean: `silent_add` in `lean/coarse/Coarse/Unit.lean` (the summation step: a pointwise identity with one term zero at
each point gives additivity of the sums); the graph step by hand.

**Theorem TU3 (the price of firing together) (under: S and T read only by the logits).** Let S and T be disjoint sets
of units whose only out-boundary is the logits (so they are convex and unlinked when they lie in the last hidden layer
or in parallel branches feeding only the logits). For boxes C_S, C_T, write delta_S(x) = y_{N[S <- C_S]}(x) - y_N(x)
and delta_T(x) likewise. Then the joint build has logits y_N + delta_S + delta_T, and
   | D_P(N[S <- C_S, T <- C_T]) - D_P(N[S <- C_S]) - D_P(N[T <- C_T]) | <= (1/2) sum over x of p_x |delta_S(x)| |delta_T(x)|.
Here |.| is the Euclidean norm. Consequently
-(1/2) E_P |delta_S^J| |delta_T^J| - eps <= I(S, k_S; T, k_T) <= (1/2) E_P |delta_S^*| |delta_T^*| + 2 eps
for any joint build J within eps of D*(S, k_S; T, k_T) and any single builds within eps of D*(S, k_S) and D*(T, k_T).
Proof. The logits are linear in the units of S, T and the box units, all of which only feed the logits, so the two
changes add. Fix x, write phi(delta) = KL(softmax(y_N(x)) || softmax(y_N(x) + delta)), a = delta_S(x), b = delta_T(x).
Then phi(0) = 0 and phi(a + b) - phi(a) - phi(b) + phi(0) = integral over s, t in [0, 1] of a^T H(s a + t b) b, with H
the Hessian of Fact FU3, whose norm is at most 1/2. Multiply by p_x and sum. For the consequence, apply the bound to
the joint near-optimum (giving the lower bound, since D*(S, k_S) and D*(T, k_T) are at most the single losses of its
two boxes) and to the pair of single near-optima (giving the upper bound). QED.
Lean: none, by hand (Fact FU3 and the integral form of the second difference).
Comment: the bound vanishes on every x where one of the two changes is zero. Interaction lives only on the inputs
where both replacements act.

## B.3 Linear units: independence is uncorrelated activity

For this section only (node S-linunits, PENDING: a partial result, not proposed as a restriction): N has one hidden
layer of LINEAR units, h = z = A x + a; the loss is D2_F(B) = sum over x of p_x |F^(1/2) (y_N(x) - y_B(x))|^2 with a
fixed positive semidefinite F. Two model choices are made here and are not consequences of D_P or DU2: F is constant
(the Hessian of Fact FU3 changes with x), and each box adds a free constant vector to the logits. A box for S of size
k is k linear units M z_S + m with any writes C'. With z~_S = z_S - E_P z_S and
Sigma_ST = E_P [z~_S z~_T^T] (the covariance of the two sets' activity on the mix), a box for S leaves the error
R_S z~_S plus a constant, with R_S = C_S - C' M, where C_S holds the members' writes, and rank(C' M) <= k.

**Theorem TU4 (independence of linear sets) (under: one hidden layer, linear units, constant metric, a free logit constant per box).**
1. If Sigma_ST = 0, then I(S, k_S; T, k_T) = 0 for all budgets, and the least pairs are exactly the pairs of least
   single boxes.
2. Let k_S be any budget, R*_S the error matrix of a least box for S at k_S, and k_T >= rank(C_T) + n_y. Then
   I(S, k_S; T, k_T) <= - |Pi|^2 <= 0, where Pi is the F-orthogonal projection of R*_S z~_S onto the functions G z~_T
   (G any n_y x |T| matrix), and I < 0 whenever F R*_S Sigma_ST != 0.
3. So if I = 0 for all budgets, then F C_S Sigma_ST = 0 and F C_T Sigma_TS = 0 (take k_S = 0, R*_S = C_S). If, in
   addition, F^(1/2) C_S has full column rank (the members' writes, seen through the metric, are linearly
   independent; this needs |S| <= rank F <= n_y), then Sigma_ST = 0.
Proof. The free constants make the loss of any pair equal to E_P |F^(1/2) (R_S z~_S + R_T z~_T)|^2 (the mean error is
removed exactly by the constant, in the single and the joint problems alike). Expanding, the loss is
E|F^(1/2) R_S z~_S|^2 + E|F^(1/2) R_T z~_T|^2 + 2 tr(R_S^T F R_T Sigma_TS). (1) If Sigma_ST = 0 the cross term is 0 for
every pair, so the joint loss is the sum of the single losses for every pair, its infimum is the sum of the infima,
and its minimisers are the pairs of minimisers. (2) With k_T >= rank(C_T) + n_y, every matrix C_T + G with rank(G) <= n_y
has rank at most k_T, so it equals C' M for some box, giving R_T = -G. The joint loss of (R*_S, -G) is
E|F^(1/2)(R*_S z~_S - G z~_T)|^2; its least value over G is E|F^(1/2) R*_S z~_S|^2 - |Pi|^2, while D*(T, k_T) = 0 (take
G = 0). So I <= -|Pi|^2. The projection is zero exactly when E[(G z~_T)^T F R*_S z~_S] = tr(G^T F R*_S Sigma_ST) = 0
for every G, that is when F R*_S Sigma_ST = 0. (3) At k_S = 0 the only box is removal, R*_S = C_S; then (2) gives
F C_S Sigma_ST = 0, and the same with S and T exchanged. If F^(1/2) C_S has a left inverse, F C_S Sigma_ST = 0 gives
F^(1/2) C_S Sigma_ST = 0 (multiply by F^(1/2) from the left: F = F^(1/2) F^(1/2)), hence Sigma_ST = 0. QED.
Lean: `cross_zero_add` in `lean/coarse/Coarse/Unit.lean` (one direction, for attained minima: if the joint loss is a sum
of a function of each box, a pair of single minimisers is a joint minimiser and the values add); the infimum form and
the matrix algebra by hand.
Comment (what the linear world says). Uncorrelated activity is enough for independence; correlated activity breaks it
whenever S's best error is visible against T's activity (part 2), because spare units of T can predict and cancel S's
error. Activity on the mix, not wiring, is what the two parts test. For sets larger than n_y the converse of part 1 is
not proved. Deep linear networks: Conjecture CU5.

## B.4 ReLU units: help, covering, and districts

**Proposition TU6 (free coarse units help each other: U4 fails in a perfectly symmetric detector network) (under: two
logits, one hidden layer).** Take the network and mix of Proposition T61 of `COARSE-THEORY.md`: Definition D40 with
kappa = 2 blocks and m = 2 directions, units A, B (block 1) and C, D (block 2), gap g_N = c (h_A + h_B + h_C + h_D) with
c = 2 / (1 - r) and no offset, and P the four vertices (gap 2) and the midpoints of AB, CD, AC, BD (gap 0), each with
weight 1/8; here r = 39/40 (the two directions have inner product 4/5). Boxes are as in Definition DU2 (N's logit
offset kept), with units of Trunk T4's class (b): relu(m^T z_S + beta), any m and beta. D2* is the least D2_P.
1. D2*({A, C}, 1) >= 1/8, and D_P*({A, C}, 1) >= kappa_0 / 8, where kappa_0 > 0 is the least two-logit KL between a
   target gap t in {0, 2} and a gap t' with |t' - t| = 1.
2. D2*({B, D}, 2) = 0 and D2*({A, C}, 1; {B, D}, 2) = 0.
3. Hence I({A, C}, 1; {B, D}, 2) <= -1/8 for D2 (and <= -kappa_0 / 8 for D_P): the two groups are not independent.
4. In covering boxes (Definition DU3) the two groups ARE independent (Theorem TU5 applies: their supports
   {vertex A, vertex C} and {vertex B, vertex D} are disjoint), and D2*_cov({A, C}, 1) >= 1/2.
The interaction in part 3 uses unequal budgets and a constant unit, which is not covering.
Proof. (1) On the three points A, midAC, C the units B and D are off: from Definition D40, at vertex A z_B = 4/5 - r and
z_D = -r, at vertex C z_B = -r and z_D = 4/5 - r, at midAC z_B = z_D = 2/5 - r, all negative since r = 39/40. So so the single build's gap is w relu(l(x)) with l affine along the segment from A to C. A ReLU of
an affine function is monotone along a segment, so the build's gap at midAC lies between its gaps at A and C. The
targets are 2, 0, 2. If the smaller end value is at least 1, the error at midAC is at least 1; otherwise the error at
that end exceeds 1. So some point of weight 1/8 has a gap error of at least 1: D2 >= 1/8. Two-logit KL is increasing
in the gap error on each side of the target, so that point's KL term is at least kappa_0, which is positive. (2) The box
of {B, D} made of B and D themselves is exact. For the joint build take Proposition T61's units
relu(-20 z_A - 40 z_C - 45/2) for {A, C}, and relu(-40 z_B - 20 z_D - 45/2) and relu(1) (m = 0, beta = 1, written with
weight -18) for {B, D}: T61's build, with its offset -18 carried by the constant unit. T61 shows it is exact on P.
(3) follows from (1) and (2). (4) A covering unit of class (b) for {A, C} is relu(m_A z_A + m_C z_C - tau) with m >= 0,
tau >= 0 (Theorem T29, whose premises hold here: the input domain is R^4 and the set where z_A and z_C are both
non-positive is non-empty). At vertex A, z_A = 1 - r and z_C = -r (the blocks are orthogonal); at vertex C the reverse. If it were on at
both, adding its two pre-activations would give (m_A + m_C)(1 - 2 r) - 2 tau > 0, impossible since r > 1/2. So at one
of the vertices A, C the build's gap is 0 (the unit is off, B and D are off, the offset is 0), against target 2: error
2, weight 1/8, D2 >= 1/2. QED.
Lean: none, by hand (monotone along a segment, as in Proposition T64 part 3 of `COARSE-THEORY.md`).
Comment (what the free optimum does). In T61's build the unit of group {A, C} is relu(-20 z_A - 40 z_C - 45/2): it
fires most at vertex B, a member of the other group. It is driven by the deep negative pre-activations of its own
members: the members' negative poles act as its positive input. It does part of {B, D}'s job. That is the "help" that
part 3 counts, and the reason the developer could not coarsen {A, C} without knowing what happens to {B, D}.

**Theorem TU5 (sets that never fire together are independent under covering boxes, at any depth) (under: none).**
Let S and T be disjoint, convex and unlinked sets of hidden units, in any layers, with disjoint supports on P. Let C_S
and C_T be boxes covering on P (Definition DU3; covering boxes are covering on P). Then for every x in P:
   the joint build N[S <- C_S, T <- C_T] has the logits of N[S <- C_S] if x in supp(S), of N[T <- C_T] if x in supp(T),
   and of N otherwise.
Hence D_P(joint) = D_P(N[S <- C_S]) + D_P(N[T <- C_T]) for every such pair, I = 0 for every budgets in the class, and
the least pairs are exactly the pairs of least single boxes (U4).
Proof. As in TU2, unlinked sets see N's in-boundary values in every build. Take x not in supp(S). In N all members of
S are off at x, and S's in-boundary takes N's values at x, so by Definition DU3 (covering on P) all units of C_S are
off. Members send nothing (Fact FU2) and box units send nothing, so S's replacement is silent at x. Likewise T's
replacement is silent at x when x is not in supp(T). Since the supports are disjoint, at every x at least one
replacement is silent, and TU2 applies; the case "neither support" has both silent and gives N's logits. The loss of a
pair is a sum of a function of C_S and a function of C_T, so the infimum is the sum of the infima and the minimisers
are the pairs of minimisers. QED.
Lean: `silent_add` (the summation step); the graph step by hand.
Comment: the premises are convex, unlinked, disjoint supports and covering on P. No symmetry, width or depth limit is
used. A box covering on P can only act where its own members act, so it cannot reach into a neighbour's region: "The
dogs go on to their own area, which doesn't have to worry about cars."

**Proposition TU7 (firing together breaks independence, also for covering boxes) (under: S and T in the last hidden
layer).** Let S and T be disjoint sets of the last hidden layer. For x in P let delta_S(x) = -sum over u in S of
W[., u] h_u(x) (the change of the logits when S is removed) and g_x = softmax(y_N(x) + delta_S(x)) - softmax(y_N(x))
(the gradient of the KL term at that change). If for some v in T the vector
   gamma_v = sum over x of p_x h_v(x) g_x
is not zero, then I(S, 0; T, |T| + 1) < 0 in the covering class (and in the free class).
Proof. At budget 0 the only box for S is removal, so D*(S, 0) = sum over x of p_x phi_x(delta_S(x)). At budget |T| + 1
the box made of the members of T themselves (relu(z_u), u in T: covering) and one more unit e relu(z_v) (covering,
e > 0) with write w, is exact when w = 0, so D*(T, |T| + 1) = 0. The joint loss with this box is
f(e) = sum over x of p_x phi_x(delta_S(x) + e w h_v(x)). Its derivative at e = 0 is gamma_v^T w; with w = -gamma_v it is
-|gamma_v|^2 < 0, so for small e > 0 the joint loss is below D*(S, 0). Hence I < 0. QED.
Lean: none, by hand (a one-sided derivative).
Comment: the spare unit writes to the logits, so TU7 needs the logits in T's out-boundary (true when no logit
weight on T is 0; added 2026-10-28, REFEREE-EDGE finding 21). gamma_v is a weighted sum over the inputs where v is on. It is zero when v is never on where S's removal
changes the softmax, and it can be zero by cancellation. Say the pair (S, T) has **no cancellation** when gamma_v != 0
for every v in T such that h_v(x) g_x != 0 at some x of P.

**Corollary TU8 (the unit, for boxes covering on P in the last hidden layer) (under: boxes covering on P; families
that partition the last hidden layer).** Call two units of the last hidden layer **co-firing** if they are both on at
some x of P, and call the connected components of this relation over the whole layer the **co-firing components**.
1. Any partition of the last hidden layer into unions of co-firing components is a unit system: it meets U1 to U5
   (U4 by TU5; U5 by TU9, since sets of one layer are convex).
2. Conversely, if two disjoint sets S, T of the last hidden layer meet U4 and (S, T) has no cancellation, then every
   unit v of T is off at every x of P where removing S changes the softmax (h_v(x) g_x = 0). So, without cancellation,
   a set that meets U4 with the rest of the layer is closed under co-firing at the inputs where it matters.
Proof. (1) Different components have disjoint supports; sets in one layer are convex and unlinked; TU5 and TU9. (2) is
TU7 in contrapositive with the definition of no cancellation. QED.
Lean: none, by hand.
Comment (what the river gives). The unit is not chosen from wiring (D58) and not from one layer as such (T4). It is a
group of units that fire together on the mix. A unit that is never on (dead on P) is its own component. Under
Hypothesis H-district the components are the districts, minus their dead members: the hub co-fires with every live
member of its district, and no member co-fires across districts. Without a hub, members that never fire together are
separate units (Proposition TU6 part 4: A, C, B and D are four components). A partition finer than the components can
also meet U4 when two units are joined only through units outside it; part 1 is about partitions of the whole layer.

**Proposition TU9 (covering composes: units nest) (under: none).** Let C be a covering box for S, and C' a covering box
for the set of C's units (taken as members). Then C' is a covering box for S. The union of covering boxes for disjoint
sets is a covering box for the union, when the union is convex.
Proof. If every member of S is off, every unit of C is off (C covers S), so every unit of C' is off (C' covers C). For
the union: if every member of S u T is off, all members of S are off and all members of T are off, so both boxes are
off. QED.
Lean: `cover_compose` in `lean/coarse/Coarse/Unit.lean` (the composition step as implications between "all off" statements
at one in-boundary value; the union part by hand).

**Theorem TU10 (free coarse units leave no independence) (under: S and T in the last hidden layer; free boxes of
class (b); a least box of T at k_T exists).** Let some member u of S have a pre-activation z_u that separates the
points of P (different points get different values). Then I(S, |S| + 1; T, k_T) < 0 unless D*(T, k_T) = 0. This holds also when S and T
never fire together.
Proof. Let C_T be a least box of T at k_T, with logit change delta_T(x) and g_x = softmax(y_N(x) + delta_T(x)) -
softmax(y_N(x)). For S take the box made of S's members themselves (exact) plus one spare unit f(x) = relu(a z_u(x) + b)
with write e w. The joint loss is F(e) = sum over x of p_x phi_x(delta_T(x) + e w f(x)), so F(0) = D*(T, k_T) and
F'(0) = w^T gamma_f with gamma_f = sum over x of p_x f(x) g_x. If gamma_f != 0 for some a, b, take w = -gamma_f: then
F(e) < D*(T, k_T) for small e > 0, while D*(S, |S| + 1) = 0, so I < 0. It remains to show that gamma_f = 0 for all a, b
forces D*(T, k_T) = 0. Sort the distinct values t_1 < ... < t_n of z_u on P. The functions 1 = relu(0 z_u + 1) and
relu(z_u - c_j) with t_j < c_j < t_(j+1) (j = 1, ..., n - 1) are, on the points of P, a triangular family that spans
all functions of z_u; since z_u separates the points of P, they span all functions on P. So gamma_f = 0 for all a, b
gives sum over x of p_x f(x) g_x = 0 for every function f on P, hence g_x = 0 at every x (p_x > 0). Then the softmax
of N[T <- C_T] equals N's on P and D*(T, k_T) = D_P(N[T <- C_T]) = 0. QED.
Lean: none, by hand (hinge functions span the functions on a finite set of reals).
Comment (the mechanism, in the network's words). A member that is OFF on the other district's inputs still has a
pre-activation there, and that value varies from input to input. A free coarse unit can read it, fire on the other
district's inputs, and correct the other district's error. The essay's gate decides whether a unit is ON; it does not
remove the information in how far below zero the unit is. A one-dimensional exact instance: N's gap |x| = relu(x) +
relu(-x) on P = {-2, -1, 1, 2} (uniform), S = {relu(x)}, T = {relu(-x)}. Each unit is its own district's gate and they
never fire together. D2*(T, 0) = 5/4, D2*(S, 2) = 0, and the joint build (copy of relu(x), plus relu(-z_S) = relu(-x)
written with weight 1, T removed) is exact: I = -5/4 for free boxes, while I = 0 for covering boxes (TU5).

**Corollary TU11 (free boxes: no independence between sets that cost something) (under: sets in the last hidden layer;
free boxes of class (b); least boxes exist at the budgets used).** Let S and T be disjoint sets of the last hidden layer,
each with a member whose pre-activation separates the points of P. If S and T meet U4, then D*(S, 0) = 0 and
D*(T, 0) = 0: both can be removed at no cost. So a family of two or more sets meets U4 only if every set in it can be
removed at no cost.
Proof. D* is non-increasing in the budget (a box can leave a unit unused by writing it with weight 0), so if
D*(T, 0) > 0 then T is inexact at budget 0, and TU10 (S at budget |S| + 1, T at budget 0) gives I < 0. The same with S
and T exchanged. QED.
Lean: none, by hand.
Comment: with free boxes, the developer cannot coarsen two costly sets independently; each combination of levels must
be solved jointly. The separation premise holds for generic weights when N has one hidden layer (then z_u is affine in
x). With two or more hidden layers it can fail on an open set of weights (inputs whose previous layer is all zero give
equal values). For boxes covering on P, TU5, TU8 and TU9 give units: the co-firing components.

**Remark B.1 (depth).** TU5 holds at any depth, so a unit may be a set across layers: a hub, its members, and lower
units that fire only inside the district, PROVIDED the set stays convex and unlinked from the other units of the family.
That is a condition on wiring, not on firing: in a dense network the lower units read by two districts link them. Lower
units that fire in two districts (the essay's shared pose units) co-fire with both; TU7 is stated for the last hidden
layer only, and the depth version of the converse is open (Conjecture CU4).

**Remark B.3 (how much covering is needed).** The proof of TU5 uses only that a box is silent at the points of P
outside its set's support. Call such a box **covering on P**. TU10's proof shows what fails without it: a unit that is
non-silent at a point of another set's support can be steered to help there. So "covering on P" suffices for
independence (with disjoint supports), and all of class (b) fails; neither result shows that independence needs exactly
"covering on P"; covering for every in-boundary value (Definition DU3, the folds of T29 at one layer) is the version
that does not depend on the mix. Covering on P allows negative reads inside the support, which DU3 forbids. Two
consequences follow at once from the definitions: a free least box of S that is silent at the points of P outside
supp(S) is also a least box covering on P (it lies in the smaller class); and the loss that independence costs a pair
is the free joint optimum's gain from help, which TU3 bounds by the inputs where both sets act. Theorem TU5, Corollary
TU8 and Proposition TU9 hold with "covering on P" in place of "covering" (their proofs use only silence on P).

**Remark B.2 (negative poles).** For a fold (one layer, m >= 0, tau >= 0), the members' negative pre-activations can
only push a box unit down, so they can keep it off but never turn it on. (Boxes covering on P may read negative values
inside the support.) This is the essay's "negative poles as gates in a
circuit, not a neuron's opposite". The free optimum of TU6 uses the opposite: its unit is driven on by its members'
negative pre-activations.

## B.5 Output-faithful boxes: a coarse set sees what its readers saw (added 2026-10-27, night; revised after REFEREE-B5U-2026-10-27)

**Remark B.4 (Theorem TU10 uses no principle of clean).** TU10's premises are: S and T lie in the last hidden layer,
boxes are free (class (b)), a least box of T at k_T exists, some member of S has a pre-activation that separates the
points of P, and that least box is not exact. The last two are properties of N, but they are generic (TU11's comment
says when separation can fail with depth), and none of the premises speaks of districts, gates, symmetry or any
principle of clean. So TU10 holds for clean and for unclean networks alike. Its mechanism is a property of the free
class: a free box reads its members' pre-activations below zero, a value that N's own readers never receive (Fact FU2).
The essay's view of a network is the opposite ("a network's parts are built to be read, and built upon, by its
downstream readers ... become another reader"). This section defines the classes of boxes that follow that view.

**Definition DU6 (output-faithful box, thresholded form) (node R-outfaith, PENDING).** Let S be a convex set and let
dS be its members with an edge to the out-boundary. For thresholds theta = (theta_u)_u >= 0 put
h^theta_u(x) = h_u(x) if h_u(x) > theta_u and 0 otherwise. The **read units** of a box are its units with an edge to
the out-boundary. A box C for S is **output-faithful on P at theta** when, for all x, x' in P with
h^theta_dS(x) = h^theta_dS(x'), every read unit takes the same value at x and x' (in builds where S's in-boundary takes
N's values at x and x'). At theta = 0 it is called output-faithful on P. The **weak set** is
O^theta_S = {x in P : h^theta_dS(x) = 0} (at theta = 0 it is P minus supp(S)), and the **strong support** is P minus
O^theta_S. A point is **strong for S** when it lies in the strong support.
Source: the essay ("become another reader"; "you can ablate almost all the weak activations of the network for almost
no cost"). theta > 0 reads weak positive outputs as if they were zero. theta = 0 reads only what N's readers receive.
(The map h -> h^theta is used only to say which points the box must treat alike; the box units are ReLU units.)

**Lemma TU12 (what an output-faithful box can do) (under: each part has its own tag below).**
1. (under: none) The read units of an output-faithful box at theta take one constant vector c_0 at every point of
   O^theta_S.
2. (under: S in one hidden layer; box units relu(m^T z_S + beta)) Let A_S be the affine hull of {z_S(x) : x in R^d}
   (all values S's pre-activations can take). If the points z_S(x), x in O^theta_S, affinely span A_S, then every read
   unit of an output-faithful box at theta is either off at every point of O^theta_S or constant on all of R^d.
3. (under: the premises of part 2) If moreover h^theta_dS separates the points of the strong support (different points
   give different vectors), then the output-faithful boxes at theta are exactly the boxes whose read units are each
   of one of these two kinds: off on O^theta_S, or constant.
Proof. (1) All points of O^theta_S have the same vector h^theta_dS = 0, so DU6 gives equal read-unit values there.
(2) Let a read unit have the constant value c_i on O^theta_S. If c_i = 0 it is off there. If c_i > 0, then
m^T z_S(x) + beta = c_i at every x of O^theta_S, so the affine function z -> m^T z is constant on the affine hull of
those points, which is A_S; since z_S(x) lies in A_S for every x, m^T z_S(x) is the same for every x in R^d and the
unit is constant. (3) Under separation, two different points of the strong support never have equal vectors, so DU6
asks nothing there, and on O^theta_S it asks exactly (1); read units of the two kinds meet (1), and by (2) every read
unit is of one of them. QED.
Lean: none, by hand (one line of linear algebra: an affine function constant on a spanning set is constant on its hull).
Comment: on the instance of TU10 (|x| on {-2, -1, 1, 2}) a unit relu(m x + beta) of S = {relu(x)} must take the same
value at -2 and -1, so it is off at both or m = 0. The spare unit relu(-x) of TU10 is not output-faithful.

**Theorem TU13 (output-faithful boxes interact only through their constants) (under: S and T read only by the
logits; S and T never on together on P; theta = 0).** For output-faithful boxes C of S and D of T, let
v_C = W_C c_0 (what C's read units write to the logits at the points of O_S, one vector by TU12 part 1),
Delta_C(x) the logit change of N[S <- C] at x in supp(S), and likewise v_D, Delta_D. Let R = P minus
(supp(S) u supp(T)) and phi_x as in TU3. For a vector v, v' is its part orthogonal to the all-ones vector.
1. (exact split) D_P(N[S <- C, T <- D]) = sum over supp(S) of p_x phi_x(Delta_C + v_D) + sum over supp(T) of
   p_x phi_x(Delta_D + v_C) + sum over R of p_x phi_x(v_C + v_D), and D_P(N[S <- C]) = sum over supp(S) of
   p_x phi_x(Delta_C) + sum over P minus supp(S) of p_x phi_x(v_C).
2. If v_C and v_D are multiples of the all-ones vector (a change the softmax does not see), the loss of that pair is
   the sum of the two single losses. (This is a statement about the pair; it does not give I = 0 over the whole class.)
3. Let sigma_S(k_S) = D*(S, k_S) - inf over output-faithful C of size k_S and w in R^{n_y} of sum over supp(S) of
   p_x phi_x(Delta_C + w) (what one free logit shift on S's own support, with no charge elsewhere, would gain), and
   sigma_T(k_T) likewise. Then -(sigma_S(k_S) + sigma_T(k_T)) <= I(S, k_S; T, k_T). And for any eps >= 0 and any
   single boxes C_e, D_e within eps of D*(S, k_S) and D*(T, k_T),
   I <= (1/2) [ sum over supp(S) of p_x |Delta'_Ce| |v'_De| + sum over supp(T) of p_x |Delta'_De| |v'_Ce| + sum over R
   of p_x |v'_Ce| |v'_De| ] + 2 eps.
Proof. (1) The logits are linear in the units of S, T and the box units, which only feed the logits. At x in supp(S)
every member of T is off (S and T are never on together), and D's read units take their constant values (TU12 part 1),
so T's replacement changes the logits by v_D - 0. The same with S and T exchanged, and at R both replacements add their
constants. In the single build N[S <- C], at x outside supp(S) the members send 0 and C adds v_C. (2) phi_x(delta + a 1)
= phi_x(delta) for every real a, since the softmax is unchanged when every logit moves by a. Substituting into (1)
gives joint = sum over supp(S) of p_x phi_x(Delta_C) + sum over supp(T) of p_x phi_x(Delta_D), which is the sum of
the two single losses. (3) Lower bound: drop the R term (it is >= 0) and treat v_D as a free shift w in the first sum
and v_C in the second: joint >= (D*(S, k_S) - sigma_S(k_S)) + (D*(T, k_T) - sigma_T(k_T)) for every pair, so the
infimum is too. Upper bound: D*(S, k_S; T, k_T) <= D_P(N[S <- C_e, T <- D_e]), and the single losses of C_e, D_e are at
most D* + eps each. Subtract the two single losses from (1): each of the three sums becomes a sum of second differences
phi_x(a + b) - phi_x(a) - phi_x(b) + phi_x(0). By shift invariance each such term is unchanged when a and b are
replaced by a' and b'; then bound it by (1/2) |a'| |b'| (Fact FU3, as in TU3). QED.
Lean: none, by hand (TU3's estimate and the shift invariance of the softmax).
Comment. Free boxes help a neighbour with ANY function on its support (TU10). Output-faithful boxes can help only with
one vector of n_y numbers, the same on every input outside their own set. The channel is a constant, not a reading of
the neighbour's inputs. It need not be small. On the instance of TU10 with T at budget 0, sigma_T(0) = 5/4 - 1/8 = 9/8
(the best free shift on T's support {-2, -1} is the mean gap 3/2), so part 3 allows I down to -9/8. A box of S at
budget 2 can realise much of it: one covering unit fits S's support, a constant unit adds a shift, and the covering
unit cancels that shift on supp(S), so the shift acts only where S is silent. The coarse version of S then says "S is
off, so answer as T would". That uses S's silence as an opposite.

**Theorem TU14 (graded independence: strong co-firing, weak writes, constants) (under: S and T read only by the logits;
boxes output-faithful at theta).** S and T may now be on together. For a box C of S let v_C = W_C c_0 (C's write at its
weak set, one vector by TU12 part 1), and let delta_S(x) be the logit change of N[S <- C] at x: delta_S(x) = Delta_C(x)
at points strong for S, and delta_S(x) = v_C - W_S h_S(x) at the others (the box writes its constant; the members'
weak outputs are no longer sent). Likewise delta_T. Then for every pair of boxes
   | D_P(N[S <- C, T <- D]) - D_P(N[S <- C]) - D_P(N[T <- D]) | <= (1/2) sum over x of p_x |delta'_S(x)| |delta'_T(x)|,
and at a point that is not strong for S, |delta'_S(x)| <= |v'_C| + |(W_S h_S(x))'|. So the bound is a sum over points
of products of two factors, one per set, of these kinds:
   (i) strong co-firing: at points strong for both sets, |Delta'_C| |Delta'_D|;
   (ii) dropped weak writes: any product with a factor |(W h(x))'| from a set that is not strong at x (this includes
        points strong for neither set);
   (iii) constants: any product with a factor |v'|.
Proof. The logits are linear in all the units involved, so the joint change is delta_S + delta_T. Fix x; by shift
invariance phi_x(a + b) - phi_x(a) - phi_x(b) + phi_x(0) does not change when a, b are replaced by a', b', and by Fact
FU3 it is at most (1/2) |a'| |b'| in absolute value (as in TU3). At a point not strong for S, every member of dS has
h^theta = 0, so C's read units take their constant values (TU12 part 1) and delta_S(x) = v_C - W_S h_S(x); projection
is linear and the triangle inequality gives the bound on |delta'_S(x)|. Multiply by p_x and sum. QED.
Lean: none, by hand (TU3).
Comment. At theta = 0, the factors of kind (ii) vanish (an OFF member writes nothing, Fact FU2) and (i) is the
co-firing of TU7. A larger theta moves points out of (i) and into (ii): weakly co-firing points no longer count as
co-firing, and the price is the weak writes that the coarse network drops. This is the graded form of independence:
the bound is 0 for sets that are never strongly on together, whose dropped weak writes are 0, and whose constants are
invisible; it grows with the mass of strong co-firing and with the size of the dropped weak writes. The bound is
evaluated at given boxes (the joint near-optimum for the lower side of I, single near-optima for the upper side, as in
TU3). An entry of eps(N) that uses N alone needs a bound on |Delta'| from N; that is open (Part D).

**Definition DU7 (silence-preserving box) (node R-outfaith, PENDING).** A box C for S is **silence-preserving on P at
theta** when it is output-faithful on P at theta and, at the points of the weak set O^theta_S, every unit r of S's
out-boundary receives 0 from the box: sum over read units e of W[r, e] c_0,e = 0 (it sends nothing where its members
sent nothing above threshold). For logit readers this is v_C = 0. Source: the essay, "Think of negative poles as gates
in a circuit, not a neuron's opposite": a set's silence keeps inputs out; it does not vote for other answers.

**Corollary TU15 (two reader principles give the covering class, and exact independence) (under: each part has its own
tag below).**
1. (under: the premises of TU12 part 3 at theta = 0) A silence-preserving box has read units that are off on
   P minus supp(S), plus constant read units whose total write to every reader is 0. It gives the same build as the box
   covering on P made of its non-constant read units, which is no larger.
2. (under: S and T read only by the logits; S and T never on together on P; theta = 0) Every pair of
   silence-preserving boxes has D_P(joint) = D_P(N[S <- C]) + D_P(N[T <- D]). So, within this class, I = 0 at every
   budget and the least pairs are the pairs of least single boxes (U4).
3. (under: S and T read only by the logits; theta >= 0) For silence-preserving boxes, TU14's bound has no factor of
   kind (iii); applied as in TU3 (the joint near-optimum for the lower side, single near-optima for the upper side, plus
   2 eps), I is bounded by (1/2) times the sum of the strong co-firing products (i) and the products with dropped weak
   writes (ii).
Proof. (1) By TU12 part 3 each read unit is off on P minus supp(S) or constant on R^d; the constant units' total write
is 0 by DU7, so they change no logit anywhere, and removing them leaves the build unchanged. (2) TU13 part 1 with
v_C = v_D = 0: the cross terms vanish and the R term is 0, for every pair in the class; the infimum of a sum of a
function of C and a function of D is the sum of the infima. (3) TU14 with v'_C = v'_D = 0, used as in TU3. QED.
Lean: none, by hand (a direct consequence of TU12 to TU14).
Comment. Last session's option (a) of card 3-15 offered "covering on P" as a choice. Here it follows from two
readings of the essay: a coarse set may depend on the input only through what its members sent (DU6), and its silence
stays silence (DU7). What is left is graded: strong co-firing and dropped weak writes (TU14).

**Proposition TU16 (sets whose descendants meet only at the logits) (under: none).** Let S and T be disjoint, convex
and unlinked, and suppose no hidden unit is a descendant (along N's edges) of both S and T. Then for any boxes C, D the
joint build's logits are y_N + delta_S + delta_T, where delta_S and delta_T are the logit changes of the single builds,
and the estimate of TU3 (with projected changes, as in TU14) holds for the pair.
Proof. Every hidden unit outside S and T is a descendant of at most one of them. A unit that descends from neither
takes N's value in all three builds. A unit that descends from S only takes, in the joint build, the value it takes in
N[S <- C] (its inputs come from S's box, from other descendants of S only, and from units that take N's values), and
likewise for T. The logits are linear in the last hidden layer, so the joint change at the logits is the sum of the two
single changes. The rest is TU3's estimate. QED.
Lean: none, by hand (induction over layers).

**Proposition TU17 (sheltered inputs make every box constant outside its set) (under: each part has its own tag below).** Say S is **sheltered on P** when its
in-boundary takes one fixed vector a_0 at every point of P outside supp(S) (for example, all its in-boundary units are
off there: the essay's "the dogs go on to their own area, which doesn't have to worry about cars").
1. (under: none) Every box of S, free or not, takes one constant value at all points of P outside supp(S).
2. (under: S and T sheltered, read only by the logits, never on together on P; theta = 0) The conclusions of TU13 hold
   for free boxes of S and T: they can help each other only through one constant vector each.
Proof. (1) A box's units are functions of its in-boundary values (Definition DU2), and these are a_0 at every point of
P outside supp(S). (2) TU13's proof uses output-faithfulness only through this constancy (its part 1 and TU12 part 1).
QED.
Lean: none, by hand (a function of a constant is constant).
Comment. So the help in which a box reads how the other set's images differ (TU10) needs S's inputs to vary on T's
images. Sheltered sets leave only the constant channel.

**Proposition TU18 (judgment units and the constant channel) (under: S read only by the logits; a class of boxes that
is closed under changing the write of a unit, such as the free, output-faithful or silence-preserving class).**
Call a read unit u a **judgment unit** of S when its value is a constant c > 0 at every point of supp(S) and 0 at every
other point of P.
1. (a least box of S of size k in the class exists, and has a judgment unit) Let C* be such a box. For every w in
   R^{n_y}, sum over supp(S) of p_x phi_x(Delta_C*(x) + w) >= sum over supp(S) of p_x phi_x(Delta_C*(x)).
2. (the class is the silence-preserving class at theta = 0, and a judgment unit of S can be built from S's in-boundary)
   sigma_S(k) <= D*(S, k) - D*(S, k + 1), with sigma_S as in TU13 part 3 (the infimum taken over this class).
Proof. (1) Change u's write by w / c. The logits move by w at the points of supp(S) and do not move elsewhere (u is 0
there). The new box is in the class and has size k, so its loss is at least D_P(N[S <- C*]); the two losses differ only
on supp(S), which gives the inequality. (2) Take any box C of size k in the class and any w. Add a judgment unit with
write w / c. The new box has size k + 1, is silence-preserving (the new unit is 0 off supp(S)), and its loss is
sum over supp(S) of p_x phi_x(Delta_C + w), since C sends nothing off supp(S). So D*(S, k + 1) <= that sum; take the
infimum over C and w. QED.
Lean: none, by hand (a perturbation of one write).
Comment. The constant channel gives a set at most what one more unit would give it, when that unit can be a judgment
unit. Such a unit needs an input that marks the set's support (a gate in front, principle Q4 of COARSE-THEORY.md A.5).

**Proposition TU19 (one unit is one-sided: a contrast needs its reader's threshold) (under: S read only by the logits;
a box of size 1).** Fix two logits c, c' and write d_B(x) = y_B,c(x) - y_B,c'(x). Suppose that on supp(S) every unit
outside S writes the same amount to c and c' (no kept unit separates c from c' there).
1. (readers' biases fixed at N's, and N's biases equal for c and c') For every box C = {u} of size 1, at every x in
   supp(S), d_B(x) = u(x) (V_c - V_c'), so d_B has one sign (or is 0) on all of supp(S). If N's d changes sign on
   supp(S) (S's job there is a two-sided contrast, such as "upper versus lower"), the build is wrong in sign or tied at
   every point of one side.
2. (under DU8) With a re-fit offset beta for d, d_B = u(x) dv + beta on supp(S), which is affine in u(x) and changes
   sign at most once as u(x) grows. The offset beta also acts at every point outside supp(S).
Proof. The logits are linear in the units; units outside S add equal amounts to c and c', the members are gone, and
the box adds u(x) times its write. u(x) >= 0. Part 2: the offset adds beta everywhere. QED.
Lean: none, by hand (sign of a product with a non-negative factor).
Comment. Without a re-fit reader offset, a set coarsened to one unit cannot express a contrast, and another set's box
can supply an offset through its constant (TU13). That is a reader's threshold, supplied through another set. A
contrast is a dial ("the difference between slightly bright and medium bright matters"): one unit read against a
threshold.

**Definition DU8 (builds with re-fit readers) (node R-readbias, DECIDED 2026-10-27).** In a build N[S <- C] (or a joint build),
every unit of S's out-boundary (and every logit, if S is read by the logits) may also change its bias. The other
weights and biases stay N's. Source: when a reader's inputs are replaced, its threshold was set for the old inputs.
A contrast made by one coarse unit is a threshold on that unit (TU19), so the reader must be able to move it.
Comment: this widens the class of DU2. In a joint build, a reader of both S and T has one bias, shared by both
replacements.

**Proposition TU20 (a re-fit reader does not remove the help of a rebuilt half) (under: DU8; D2; the instance of
TU10).** On P = {-2, -1, 1, 2} (uniform) with N's gap |x| = relu(x) + relu(-x), S = {relu(x)} and T = {relu(-x)},
free boxes and a re-fit logit bias: D2*(T, 0) = 11/16, D2*(S, 2) = 0, and the joint build is exact, so I = -11/16.
Proof. In T's single build at budget 0, T is removed, S keeps N's unit, and only the logit bias beta is free, so
the gap is relu(x) + beta. Its errors |x| - relu(x) - beta are 2 - beta and 1 - beta at x = -2, -1, and -beta at
x = 1, 2. The mean squared error ((2 - beta)^2 + (1 - beta)^2 + 2 beta^2) / 4 is least at beta = 3/4, with value
(25/16 + 1/16 + 18/16) / 4 = 11/16. S's single build at budget 2 can copy relu(x), so D2*(S, 2) = 0. In the joint build
with T at budget 0, S's free box {relu(x), relu(-x)} reads S's in-boundary x and reproduces |x| with beta = 0, so the
joint loss is 0 and I = 0 - 0 - 11/16. QED.
Lean: none, by hand (a one-variable quadratic).
Comment. Here S's inputs carry T's whole job: relu(x) and relu(-x) read the same x with opposite signs, and S's box
rebuilds T's unit. (A free box with enough units can fit any function on a finite P when its inputs separate the
points, TU10; the rebuilt half of a dial is the simplest case.)

**Proposition TU21 (sheltered sets with re-fit readers: two-sided bounds) (under: DU8; S and T sheltered, read only
by the logits, never on together on P; free boxes; theta = 0).** Let sigma_S(k_S) = D*(S, k_S) - inf over boxes C of
size k_S and w in R^{n_y} of sum over supp(S) of p_x phi_x(Delta_C + w), where Delta_C is the change with N's logit
bias, and likewise sigma_T(k_T). Then
   -(sigma_S(k_S) + sigma_T(k_T)) <= I(S, k_S; T, k_T) <= (1/2) E_P |delta'_S| |delta'_T| + 2 eps
for single builds within eps of D*(S, k_S) and D*(T, k_T), with delta_S, delta_T their logit changes (including their
re-fit biases).
Proof. Lower bound: by TU17 each box takes one constant outside its set. In a joint build with bias beta, the offset on
supp(S) is v_D + beta, which the lower bound treats as a free w; drop the term on the points outside both supports
(it is >= 0). So every joint build has loss at least (D*(S, k_S) - sigma_S) + (D*(T, k_T) - sigma_T). Upper bound: take
single builds (C, beta_S) and (D, beta_T) within eps, and the joint build (C, D, beta_S + beta_T). The logits are linear
in all the units and the bias, so the joint change is delta_S + delta_T at every point; TU3's estimate with projected
changes (as in TU14) gives the bound. QED.
Lean: none, by hand (TU3 and TU13 part 3).
Comment. DU8 does not remove the constant channel. Example: P = {a, t}, uniform, two logits, S = {s} on only at a,
T = {tau} on only at t, both sheltered. Removing S and re-fitting the bias gives D2*(S, 0) = (w h_s(a))^2 / 4 (best bias
w h_s(a) / 2). T at budget 2 can copy tau and add a constant unit relu(1) written v = w h_s(a), with tau's write lowered
by v / h_tau(t). The joint build is then exact with beta = 0, so I = -(w h_s(a))^2 / 4. The re-fit bias acts on all of P.
The box's constant acts only where T is silent, here exactly at S's point: "T is off, so answer as S would". DU7
forbids this use of silence.

**Definition DU9 (builds with fully re-fit readers) (node R-readweights, DECIDED 2026-10-27).** As DU8, and in
addition every reader of S may change its weights on the kept units it reads. "Kept" depends on the build: in a single
build of S, every unit outside S (including T's members) is kept; in a joint build, every unit outside S and T. DU8 is
the sub-class of DU9 in which no weight on a kept unit moves; a bias acts like a weight on a constant input. In a joint
build, a reader of both sets has one set of weights.

**Proposition TU22 (a set whose own kept units can mark its support cannot be helped) (under: DU9; S and T read only
by the logits, sheltered, never on together on P; any class of boxes, the same for single and joint builds; the
premises (a) and (b) below).** Let U be the units outside S and T read by the logits.
(a) Each unit of U is of one of three kinds: 0 at every point of P outside supp(S) (the set K_S); 0 at every point of P
    outside supp(T) (the set K_T); or one constant on supp(S) and one constant on supp(T) (any values elsewhere).
(b) The constant function 1 on supp(S) is, on supp(S), a linear combination of the units of K_S; likewise for T and
    K_T. (A judgment unit of S, a kept unit with one value c > 0 on supp(S) and 0 elsewhere on P, is the one-unit case.)
Then I(S, k_S; T, k_T) = 0 at every budget, and D*(S, k_S) = L_S, where L_S = inf over boxes C of size k_S, w in
R^{n_y} and weight changes A_S on K_S of sum over supp(S) of p_x phi_x(Delta_C + A_S h_{K_S} + w); likewise for T.
Proof. In any build the logit change at x is the boxes' changes plus beta plus the weight changes times the kept units.
(1) D*(S, k_S) >= L_S. In a single build of S, at x in supp(S): T's members are off (never on together), so their
re-weighting adds 0; units of K_T are 0; units of the third kind add one constant vector; the bias adds beta. So the
change on supp(S) is Delta_C + A_S h_{K_S} + (a constant), and the loss is at least its supp(S) part, which is at
least L_S.
(2) The construction. Given C, w, A_S for S and D, w', A_T for T, build the joint build with both boxes, bias
beta = -(v_C + v_D) (v_C, v_D the boxes' constants outside their sets, TU17 part 1), weight changes A_S on K_S and A_T on
K_T, no change on third-kind units, and, by (b), an added combination of K_S equal to (w + v_C) on supp(S) and an added
combination of K_T equal to (w' + v_D) on supp(T). At x in supp(S) the change is Delta_C + v_D + beta + A_S h_{K_S} +
(w + v_C) = Delta_C + A_S h_{K_S} + w. At x in supp(T) it is Delta_D + A_T h_{K_T} + w'. At the other points it is
v_C + v_D + beta = 0. So the joint loss is sum over supp(S) of p_x phi_x(Delta_C + A_S h_{K_S} + w) plus the same for T.
The same construction with only S's box (beta = -v_C) gives a single build of S whose loss is the supp(S) sum, so
D*(S, k_S) <= L_S; with (1), D*(S, k_S) = L_S.
(3) Joint lower bound: as in (1), at x in supp(S) the joint change is Delta_C + A h_{K_S} + (a constant), so the joint
loss is at least L_S + L_T. Joint upper bound: by (2), at most L_S + L_T. So D*(S, k_S; T, k_T) = L_S + L_T =
D*(S, k_S) + D*(T, k_T). QED.
Lean: the summation step is `silent_add` in `lean/coarse/Coarse/Unit.lean`, with f and g the per-point losses of the two
single builds of step (2) and h the per-point joint loss; the rest by hand.
Comment. A district's judgment (the essay's "car neuron", "a reusable judgment and a sheltered place for taxis,
ambulances, and other specialists") makes a district independent when it is kept and its readers may re-weight it:
the readers can then give the coarsened district any offset on exactly its own images, and no other district's box
has anything to add. Premise (a) says each kept unit belongs to one district, or is flat on both. Premise (b) can be
met by several kept units together (for example a few gates whose values vary but whose combination is flat on the
district). Whether units that vary on both supports break I = 0 is not settled here.

**Remark B.6 (what "help" is: the sources found so far).** Interaction between two coarsened sets that never fire
together comes from these sources. The list is not proved to be complete.
1. A frozen reader threshold (TU13, TU19). One coarse unit is one-sided, so a contrast needs its readers' threshold to
   move. With frozen reader biases, another set's box can move it through its constant. DU8 removes the part of this
   offset that is the same at every point of P.
2. Silence used as an opposite (TU13, TU18, TU21's comment). A box's constant acts only where its set is silent. If the
   mix holds little besides the two sets, "T is silent" nearly means "S's input", and T's spare units can give S an
   offset there. This survives DU8. It costs T at most what one more unit would give it (TU18 part 2, when a judgment
   unit can be built); DU7 forbids it.
3. A rebuilt unit (TU10, TU20). If S's inputs separate the points of T's support, a free box of S can rebuild part of
   T's job; the simplest case is the other half of a dial. Sheltered sets have no such channel (TU17).
4. A shared reader bias (DU8 joint builds). A reader of both sets has one bias in the joint build, while each single
   build moves it freely. This can make I larger or smaller.

## B.6 Activations and edges (added 2026-10-28; revised after REFEREE-EDGE-2026-10-28)

**Lemma TA1 (the loss sees a set only through its out-edges) (under: none).** Let S be a convex set of hidden units and
let C_1, C_2 be two boxes for S (Definition DU2; the box "S itself" is allowed). Fix the readers' biases and weights (in
classes with re-fit readers, DU8 and DU9, fix one choice for both builds). If e_{r,C_1}(x) = e_{r,C_2}(x) for every
unit r of S's out-boundary (and every logit S feeds) and every x in P, then y_{N[S <- C_1]}(x) = y_{N[S <- C_2]}(x) for
every x in P, and D_P(N[S <- C_1]) = D_P(N[S <- C_2]).
Proof. By Lemma TU1 part 1, S's in-boundary takes N's values at x in both builds. Order the units outside S
topologically. A unit not downstream of S has the same inputs in both builds. A unit r downstream of S sums what it
receives from the box, e_{r,C_i}(x), equal by premise, and what it receives from units outside S that come earlier in
the order, equal by induction. So every unit outside S and every logit takes the same value in both builds. Each term of
D_P depends only on y_B(x) (Fact FU1). QED.
Lean: `loss_eq_of_edges_eq` in `lean/coarse/Coarse/Edge.lean` (the summation step); the induction by hand.
Comment: "a representation matters only through what reads it". What happens inside the box, and the members'
activations in N, reach the loss only through the out-edges.

**Lemma TA2 (activations and out-edges determine each other in neither direction) (under: each part has its own tag
below).**
1. (under: u and v in the last hidden layer; two logits) (Same activation, different loss.) Let h_u(x) = h_v(x) at
   every input, let d_u, d_v be the gap weights (d = W_out[1, .] - W_out[2, .]), and E = sum over x of p_x h_u(x)^2 > 0.
   Then D2_P(N[u <- (empty)]) = d_u^2 E and D2_P(N[v <- (empty)]) = d_v^2 E, which differ when d_u^2 != d_v^2. If
   d_v = 0 != d_u (v is softmax-unread, DA1 item 5), then also D_P(N[v <- (empty)]) = 0 < D_P(N[u <- (empty)]).
2. (under: none) (Different activation, same loss.) Rescaling u (in-weights and bias times c > 0, out-weights times
   1/c) changes the activation reading and keeps every out-edge, hence, by TA1, the loss of every build. Splitting u into
   two copies (out-weights a W[., u] and (1 - a) W[., u]) does the same for every build that replaces both copies or
   neither.
So no quantity of the activation reading alone equals the removal cost of every set in every network, and a quantity
that changes under the moves of part 2 is not a property of N's function.
Proof. (1) The logits are linear in the last hidden layer, so removing u changes the gap at x by exactly -d_u h_u(x),
and D2 = sum over x of p_x (d_u h_u(x))^2 = d_u^2 E. With d_v = 0 removing v adds the same number to both logits at
every x, so the softmax and D_P do not change (Fact FU1). With d_u != 0 and E > 0, some x with p_x > 0 has
h_u(x) != 0, the gap changes there, and that KL term is positive (KL(q || q') > 0 for q' != q). (2) relu(c z) =
c relu(z) for c > 0, c h_u (w / c) = h_u w, and a w h_u + (1 - a) w h_u = w h_u; then TA1. QED.
Lean: `removal_cost_D2`, `same_act_diff_cost`, `rescale_edge` (the algebra, for c != 0; relu's positive homogeneity
by hand), `split_edge` in `lean/coarse/Coarse/Edge.lean`.

**Lemma TA3 (an activation forgets its votes and its negative pole) (under: none).** Let a unit u read n >= 2 units,
with bias b, and write s(c) = relu(c_1 + ... + c_n + b) for vote vectors c in R^n. Call the **route** of c the set of
indices of its largest entries (as in Definition D44).
1. For t > 0 the set {c : s(c) = t} is the affine hyperplane c_1 + ... + c_n = t - b, of dimension n - 1. For every
   such t it holds two vote vectors with disjoint routes: with K > |t - b|, c = (t - b) e_1 + K (e_1 - e_2) and
   c' = (t - b) e_2 + K (e_2 - e_1), whose routes are {1} and {2}.
2. {c : s(c) = 0} is the closed half-space c_1 + ... + c_n <= -b. Every vote vector in it, however far below the hinge,
   gives the same output, and by Fact FU2 every reader of u receives 0 from u there.
3. So two inputs of N whose vote vectors are such a pair (whether N and P have such inputs is a premise about N and P,
   named where it is used) have the same h_u and disjoint routes, and u's readers receive the same from u at both. What
   tells them apart, and what tells the two poles apart, is in the in-votes, which the activation reading does not hold.
Proof. (1) relu is one-to-one on (0, infinity), so s(c) = t > 0 exactly when the sum is t - b. The map c -> sum of c_i
is onto R with a kernel of dimension n - 1 (rank and nullity). The sums of c and c' are t - b. In c the first entry is
t - b + K > 0, the second is -K < 0 and the others are 0, so the first entry is the only largest one; c' is the same with
1 and 2 exchanged. (2) relu(r) = 0 exactly when r <= 0. (3) restates (1) and (2). QED.
Lean: `votes_same_sum` (two different vote vectors with one sum, n = 2) and `relu_neg_eq` (part 2) in
`lean/coarse/Coarse/Edge.lean`; the routes and the dimension count by hand.
Comment: the inputs where h_u is large get labels from their content ("cat faces and car fronts"); the votes say what
those inputs share for u ("a V"). Part 1 at the maximum is the essay's feature-visualization failure: raising h_u can
use a route that ordinary images rarely take. A gated unit g v (SwiGLU) has the same fibres: (-a)(-b) = a b.

**Lemma TA4 (removing a unit costs its out-edge through the curvature) (under: each part has its own tag below).** Let
w_u = W_out[., u], delta_x = -h_u(x) w_u, and phi_x(delta) = KL(softmax(y_N(x)) || softmax(y_N(x) + delta)).
1. (under: u in the last hidden layer) Removing u changes the logits at x by exactly delta_x, and
      D_P(N[u <- (empty)]) = sum over x of p_x h_u(x)^2 (integral over t in [0, 1] of (1 - t) w_u^T H_x(t delta_x) w_u),
   with H_x(z) = diag(s) - s s^T, s = softmax(y_N(x) + z). By Fact F11 of `COARSE-THEORY.md`,
   D_P(N[u <- (empty)]) <= (1/4) sum over x of p_x h_u(x)^2 |P_perp w_u|^2; in particular the cost is 0 when u is
   softmax-unread. Near t = 0 the integrand is w_u^T H_x(0) w_u: to leading order the cost is activation squared times
   the write's curvature at N's output.
2. (under: u and v in the last hidden layer; two logits) The activation does not order the costs: there are units u, v
   with h_u > h_v at every point of P and a smaller removal cost for u, in D2 and in D_P.
Proof. (1) The logits are linear in the last hidden layer. phi_x(0) = 0 and its gradient softmax(y + delta) -
softmax(y) is 0 at delta = 0, so Taylor's formula with integral remainder gives phi_x(delta) = integral over t of
(1 - t) delta^T H_x(t delta) delta, and delta^T H delta = h_u^2 w_u^T H w_u. The bound is Fact F11 applied to delta_x.
(2) One input, N's gap 0 there (offset chosen). u: h_u = 2, gap weight 1/10. v: h_v = 1, gap weight 1. The gap changes
are -1/5 and -1, so the D2 costs are 1/25 and 1; two-logit KL from gap 0 increases with the size of the gap change, so
the D_P costs are in the same order. QED.
Lean: `rank_flip` in `lean/coarse/Coarse/Edge.lean` (part 2, the D2 instance); part 1 by hand (Taylor with integral
remainder and Fact F11).
Comment: for a unit in a deeper layer, the first effect of removing it on a reader r is the change -e_{r,u}(x) of r's
pre-activation, passed on only where r is on or crosses its hinge (TA5). The cost is again set by out-edges and the
readers' gates. The essay's exceptions to cheap weak activations (early "dials", the last layer's "say or just think"
units) are places where this reader factor is large. Fitting activations (row 11 of A.6) weights each input by h_u
instead. Exact check: `research/clean/units/ta4_rank.py` (Appendix).

**Lemma TA5 (the path form: a unit contributes a gate) (under: none).** For an input x and a hidden unit u let
a_u(x) = 1 if z_u(x) > 0 and 0 otherwise (u's gate). Then h_u = a_u z_u, and, expanding layer by layer,
   y_N(x)_c = sum over paths i -> u_1 -> ... -> u_L -> c of W[c, u_L] ... W[u_1, i] x_i a_{u_1}(x) ... a_{u_L}(x)
            + sum over m and paths u_m -> ... -> u_L -> c of b_{u_m} W[u_(m+1), u_m] ... W[c, u_L] a_{u_m}(x) ... a_{u_L}(x)
            + (the logit offset)_c,
where u_k runs over layer k. The same sum, stopped at a unit u, gives h_u(x). Hence:
1. On each set of inputs where all gates are fixed, y_N is the affine map given by the sum over open paths (linear in
   x, plus the bias paths).
2. u enters each path through it by its in- and out-weights (fixed numbers) and by a_u(x). So the only way u's role
   depends on the input is a_u(x), a 0 or 1 fixed by the side of the hyperplane z_u = 0 of its in-votes.
3. The size of h_u(x) is the sum of the open paths that end at u: a quantity of the units and weights upstream of u.
Proof. relu(z) = 1[z > 0] z, so h_u = a_u (sum over j of W[u, j] h_j + b_u). Substitute the expansion of each h_j
(induction on the layer, h_j = x_j at layer 0) and distribute. (1) to (3) read the formula. QED.
Lean: `relu_gate` in `lean/coarse/Coarse/Edge.lean` (the one step); the expansion by hand (induction on the layer).
Comment: literature row L3 writes transformer logits as "a sum where every term corresponds to an end-to-end path"; TA5
is the same identity for a ReLU network, with the gates. In this form the two poles of u are the two sides of its
hyperplane, and the negative pole closes every path through u: "Think of negative poles as gates in a circuit, not a
neuron's opposite".

## B.7 This branch's unit, read through B.6 (added 2026-10-28; revised after REFEREE-EDGE-2026-10-28)

**Proposition TA6 (a softmax-unread unit still breaks U4) (under: S and v in the last hidden layer; boxes covering on
P, or boxes output-faithful and silence-preserving on P at theta = 0, or boxes covering on P with re-fit reader
biases).** Let v be a unit of the last hidden layer with W_out[., v] = alpha 1, alpha != 0 (v is softmax-unread, DA1 item 5,
and has an edge to every logit, so every logit is in {v}'s out-boundary). Let S be a set of the last hidden layer
without v, delta_S(x) = -(sum over u in S of W_out[., u] h_u(x)), g_x = softmax(y_N(x) + delta_S(x)) -
softmax(y_N(x)), and gamma_v = sum over x of p_x h_v(x) g_x. If gamma_v != 0, then
   I(S, 0; {v}, 0) = 0   and   I(S, 0; {v}, 1) < 0,
although removing v changes no class probability at any input. The same holds in the class of boxes that are
output-faithful and silence-preserving on P at theta = 0 (Definitions DU6, DU7), and in the class with re-fit reader
biases (DU8), with g_x taken at a least single build of S (one exists at budget 0: the loss is convex in the logit bias
and grows without bound off the all-ones direction).
Proof. Removing v adds -alpha h_v(x) to every logit, which leaves the softmax alone, so D*({v}, 0) = 0, the joint
removal build has the softmax of N[S <- (empty)], and I(S, 0; {v}, 0) = 0. At budget 1 the box made of v itself is
exact, so D*({v}, 1) = 0. The box with one unit relu(e z_v), e >= 0, and any write w to the logits (allowed, since the
logits are in the out-boundary) is a fold of v (Definition DU3 with m = e, tau = 0, so covering). Its
value e h_v(x) is a function of h_v alone, and v is the only member with an edge to the out-boundary (dS = {v}), so it is
output-faithful at theta = 0; it is 0 where v is off, so it is silence-preserving. With S removed and this box, the
logits change by delta_S(x) + e w h_v(x) - alpha h_v(x) 1, and the last term does not move the softmax, so the joint loss
is f(e) = sum over x of p_x phi_x(delta_S(x) + e w h_v(x)), with f(0) = D*(S, 0) and f'(0) = w^T gamma_v (the gradient
of phi_x at delta is softmax(y_N(x) + delta) - softmax(y_N(x))). With w = -gamma_v, f(e) < f(0) for small e > 0. So
D*(S, 0; {v}, 1) < D*(S, 0) = D*(S, 0) + D*({v}, 1). For DU8, run the same derivative at a least single build of S
(the joint build keeps its biases; v's box adds f'(0) = w^T gamma_v there). QED.
Lean: `exists_help` in `lean/coarse/Coarse/Edge.lean` (the D2 form: if sum p r h != 0, some new write of h lowers
sum p r^2); the KL form is the derivative above, by hand.
Comment (the frame). DU2 tethers a box by graph: a unit with no out-edge at all (W_out[., v] = 0) has an empty
out-boundary, and its box gets no write, so it is its own unit. But a unit that writes the same amount to every logit
has edges, and then DU2 lets its box write in any direction. The premise of TA6 uses v's activation and S's writes;
what v's own column does to the output (nothing) never enters. The helper is a NEW write of v's activation: a use that
N's readers do not make. TU7 counts the same kind of new use, so the co-firing components of TU8 contain units that no
class probability depends on (TU8's "no cancellation" premise excludes gamma_v = 0, not such units). TU10 is the
same root on the read side: a free box reads z_u < 0, which no reader of u receives (TA3 part 2). Remark B.6's sources 2
(silence used as an opposite) and 3 (a rebuilt unit) are the same root: the box gives an activation, or a silence, a use
that N's readers do not give it. Under DU9 the argument fails: the logits read v, so they may re-weight v in the single
build of S, and at a least DU9 single build the helper relu(e z_v) gains nothing (stationarity in v's weights). A
different covering unit of v, such as relu(z_v - tau) with tau > 0, may still gain, so whether I < 0 under DU9 is open.
DU9 gives v's activation a new write already in the single build, through N's own edges and in any direction. None of the
premises mention a principle of clean, so TA6 is about the class of boxes, not about N. Exact check:
`research/clean/units/ta6_unread.py` (Appendix).

**Proposition TA7 (activation overlap times reader overlap) (under: each part has its own tag below).** For disjoint
sets S and T and an input x, let a = delta_S(x) and b = delta_T(x) be the logit changes of removing S and of removing T.
1. (under: S and T in the last hidden layer) The removal interaction
      I_rem = D_P(N[S <- (empty), T <- (empty)]) - D_P(N[S <- (empty)]) - D_P(N[T <- (empty)])
   equals sum over x of p_x (a^T H_x b + R_x), with H_x = H_x(0) as in TA4 and
   |R_x| <= (sqrt 2 / 4) |a| |b| (|a| + |b|). The main term is
      a^T H_x b = sum over u in S and v in T of h_u(x) h_v(x) (W_out[., u]^T H_x W_out[., v]):
   activation overlap (both on at x) times reader overlap (their writes paired by the curvature).
2. (under: S and T in the last hidden layer; two logits) Every last-layer unit has one reader, the gap (Definition D41
   item 2), and W_out[., u]^T H_x W_out[., v] = s_1 s_2 d_u d_v with s = softmax(y_N(x)) and d the gap weights. So the
   out-edge of a unit is its activation times a fixed number, and for one unit in each set the main term is
   s_1 s_2 d_u d_v h_u(x) h_v(x): non-zero exactly when both units have d != 0 and both are on at x.
3. (under: S and T in the last hidden layer; three or more logits) Two units that are not softmax-unread can be on
   together with writes orthogonal in H_x: at a uniform softmax H_x = (1/3)(I - J/3), and the writes (1, -1, 0) and
   (1, 1, -2) give 0. Then the main term is 0 although the units co-fire; the remainder R_x need not be 0.
Proof. (1) As in TU3's proof, phi(a + b) - phi(a) - phi(b) + phi(0) = integral over s, t in [0, 1] of
a^T H(s a + t b) b, where H(z) is the covariance of the coordinates under the law softmax(y_N(x) + z). Its derivative in
a direction u is the third cumulant: d/dr a^T H(z + r u) b at r = 0 equals E[(u_Y - E u_Y)(a_Y - E a_Y)(b_Y - E b_Y)],
Y drawn from softmax(y_N(x) + z). Its absolute value is at most max_i |u_i - E u_Y| times
E|(a_Y - E a_Y)(b_Y - E b_Y)| <= sqrt(2) |u| times sqrt(Var a_Y Var b_Y) <= sqrt(2) |u| |a| |b| / 2 (a range is at
most sqrt 2 |u|; Var <= |.|^2 / 2 by Fact FU3). So |a^T (H(s a + t b) - H(0)) b| <= (sqrt 2 / 2) |s a + t b| |a| |b|,
and the integral of |s a + t b| over the unit square is at most (|a| + |b|) / 2. Multiply by p_x and sum. The
expansion of a^T H_x b uses that the logits are linear in the last hidden layer. (2) With s_1 + s_2 = 1,
w^T (diag(s) - s s^T) w' = s_1 s_2 (w_1 - w_2)(w'_1 - w'_2). (3) (1/3)(1 - 1 + 0) - (1/9)(0)(0) = 0. QED.
Lean: `two_logit_hessian` (part 2) and `ortho_instance` (part 3) in `lean/coarse/Coarse/Edge.lean`; part 1 by hand.
Comment (why this branch landed on co-firing). Two things hid the readers in B.4. First, writes free in direction: TU7's
helper picks its write along gamma, so the reader factor never has to be met (TA6). Second, two logits: in TU6 every
read unit has the same single reader, so part 2 makes even removal interactions a question of co-firing alone. With
three or more logits, in the removal interaction, co-firing AND reader overlap are both needed for a leading-order term
(part 3). Exact check: `research/clean/units/ta7_reader.py` (Appendix).

**Proposition TA8 (write-tethered boxes: help needs reader overlap) (under: S and T in the last hidden layer; boxes
covering on P; write-tethered boxes).** Let S and T be disjoint sets of the last hidden layer, and let span_T be the
span of T's columns {W_out[., u] : u in T}.
1. Suppose a least box C_S of S at k_S exists in the class (covering on P and write-tethered, DA2). (A logit outside
   T's out-boundary has entry 0 in every vector of span_T, so DU2 allows every tethered write.) Let g_x = softmax(y_{N[S <- C_S]}(x)) - softmax(y_N(x)), gamma_v = sum over x of
   p_x h_v(x) g_x, and Pi the orthogonal projection onto span_T. If Pi gamma_v != 0 for some v in T, then
   I(S, k_S; T, |T| + 1) < 0.
2. (under: S and T in the last hidden layer; write-tethered boxes) If T is softmax-unread, then for every budgets
   D*(T, k_T) = 0, I(S, k_S; T, k_T) = 0, and the least pairs are the pairs of least single boxes. (If every column of T
   is 0, T's out-boundary is empty and DU2 allows no write; the claim holds trivially.) This part does not use covering,
   so it holds for free tethered boxes too.
3. (under: S and T in the last hidden layer; boxes covering on P; write-tethered boxes; N's columns orthogonal in the
   curvature where S and T co-fire) Suppose W_S^T H_x W_T = 0 at every x of P where some member of S and some member of
   T are on (W_S, W_T the column matrices, H_x = H_x(0) of TA4). Then for every pair of boxes in the class, with a, b the
   logit changes of the two single builds,
      |D_P(N[S <- C_S, T <- C_T]) - D_P(N[S <- C_S]) - D_P(N[T <- C_T])| <= (sqrt 2 / 4) sum over x of
      p_x |a(x)| |b(x)| (|a(x)| + |b(x)|),
   and so -(that sum at a joint build within eps of D*(S, k_S; T, k_T)) - eps <= I(S, k_S; T, k_T) <= (that sum at single
   builds within eps of D*(S, k_S) and D*(T, k_T)) + 2 eps. The interaction is of third order in the logit changes.
Proof. (1) T's box: T's members with their own writes (tethered) and one spare unit relu(e z_v), e >= 0, with write w in
span_T (tethered; covering as in TA6). At e = 0 it is exact, so D*(T, |T| + 1) = 0. With C_S, the joint loss is
f(e) = sum over x of p_x phi_x(delta_{C_S}(x) + e w h_v(x)), with f(0) = D*(S, k_S) and f'(0) = w^T gamma_v. With
w = -Pi gamma_v, f'(0) = -|Pi gamma_v|^2 < 0, so f(e) < D*(S, k_S) = D*(S, k_S) + D*(T, |T| + 1) for small e > 0.
(2) Every column of T is a multiple of 1, so span_T is inside span(1), and a tethered box of T writes only multiples of
1, as do T's members. So in any build, T's replacement adds a multiple of 1 to the logits at each x, which leaves the
softmax alone. Hence D_P(N[T <- C_T]) = 0, and D_P(N[S <- C_S, T <- C_T]) = D_P(N[S <- C_S]) for every pair. The
infimum over pairs is D*(S, k_S), so I = 0, and a pair is least exactly when C_S is. (3) The logits are linear in the
last hidden layer, so the joint build's logits are y_N + a + b. a is a difference of writes of S's box and of S's
members, both in span_S (DA2), and likewise b is in span_T. At an x where S and T co-fire, a^T H_x b = 0 by the premise.
At any other x, one of the two sets has all members off, and its box is off there (covering on P), so a = 0 or b = 0.
TA7 part 1 then gives the per-pair bound, and the two-sided bound on I follows as in TU3 (the lower side at the joint
near-optimum, whose single losses are at least the single infima; the upper side at the single near-optima). QED.
Lean: none for the projection Pi and the KL form; `exists_help` is the D2 analogue of part 1's derivative step. Parts 2
and 3 by hand (softmax does not change when one number is added to every logit; TA7 part 1).
Comment: part 1 is TU7 with the spare unit's write kept inside T's columns. When T = {v} with v softmax-unread,
Pi gamma_v is 0, since every g_x sums to 0 and so gamma_v is orthogonal to 1: the helper of TA6 is gone (part 2). Two
sets can interact at first order where T's activity meets S's error inside T's own columns (part 1). When N's columns
of S and T are orthogonal in the curvature wherever they co-fire, the interaction is of third order (part 3). This is
TA7's "activation overlap times reader overlap" for re-fitted boxes. TA8 is stated for readers that keep N's biases and weights (DU2). With
re-fit readers (DU8, DU9, both DECIDED) a re-fit bias enters a and b and is not in span_S, so part 3's cross-term step
does not go through as written; TA8 under DU8 and DU9 is open. Results under DA2 stay a side branch until Shonu decides R-edgetether.

**Remark B.7 (DU6's threshold).** DU6 compares h_u with theta_u. Under the rescaling of A.5 the function stays and h_u
becomes c h_u, so the same function gets a different weak set unless theta_u becomes c theta_u. On the canonical form
(Definition D43) both rescaling and splitting are removed, and step (iii) of D43 deletes units whose out-weights are
all 0 (the units DU2 already tethers). A softmax-unread unit with alpha != 0 is not deleted. A threshold on the
out-edge |W[r, u]| h_u is rescaling-invariant but not splitting-invariant. So DU6 is meant on the canonical form.

**Remark B.8 (this branch's results in the two readings).**

| Result | What it reads | Reading |
|---|---|---|
| TU5 (disjoint supports, covering on P) | activation supports, and boxes that are off outside them | stated in the activation reading. It can be restated as an edge result (edge supports {x : some e_{r,S}(x) != 0}, boxes silent outside them) through TU2; that version is not written here |
| TU7, TU8 (co-firing components) | co-firing, and a write free in direction | activation reading plus a new use of a member's activation. A unit with no out-edge is its own unit (DU2); a softmax-unread unit with edges is tied in (TA6) |
| TU10, TU11 (free boxes) | members' pre-activations below 0 | a new read of what relu erases (TA3 part 2) |
| TU13, TU14 (output-faithful boxes) | the members' outputs h_dS | the read side follows the readers; the write side is free in direction (TA6 holds in DU6, DU7) |
| TU22 (readers re-weight kept units) | existing edges of N | an edge move along N's own edges, in any direction; it removes TA6's interaction (TA6 comment) |

## B.8 The unit under the decided reader rules, and under Trunk T5 (added 2026-10-28; revised after REFEREE-MAP-2026-10-28)

Note: the ledger (`DECISIONS.md`) checks the tags of `COARSE-THEORY.md` only; the tags of this file are not checked against nodes.

Trunk T5 (`DECISIONS.md` section 1) gives the rule a map of N. Theorem TM1 of `COARSE-THEORY.md` (Part B.M) proves
that a map's regions meet U4 for class (b) builds with free logits (Definition D50' there, the reading of the trunk's
builds used in B.W), under its premises: groups in the last hidden layer, a map that covers that layer, each region
meeting at most one group with its judgment units kept, and district, shelter and judgment claims at error 0. This
section says why, under the reader rules Shonu decided, the edge tie does not give a finer unit.

**Definition DA3 (edge tie) (under the frozen readers of DU2).** For units u, v of the last hidden layer put
   G[u, v] = sum over x of p_x h_u(x) h_v(x) W_out[., u]^T H_x W_out[., v] = sum over x of p_x h_u(x) h_v(x)
             Cov_{Y ~ s(x)}(W_out[Y, u], W_out[Y, v]),
with H_x = diag(s) - s s^T, s = softmax(y_N(x)). For sets, G(S, T) is the sum over u in S, v in T. By TA7 part 1 the
removal interaction is I_rem = G(S, T) plus a remainder of third order. G does not change under the rescaling of A.5,
and the two halves u1, u2 of a split unit (out-weights a W[., u] and (1 - a) W[., u] with a = 1/2) have
G[u1, u2] = G[u1, u1]. G is the Fisher information, under the model's own class law, of one gain per unit (multiply h_u
by theta_u, at theta = 1). Theis et al. 2018 (literature row L11) score units with a Fisher of this kind computed with
the data's labels (the empirical Fisher), and use its diagonal; the two agree in expectation when the labels are drawn
from the model's own outputs. In words: two units are tied on an input when both are on and, among the classes the network is still
weighing at that input, they vote alike or against each other.

**Proposition TA12 (with re-fit reader weights, a unit that no class probability depends on is tied to S when its
activity meets the leftover error, gamma != 0) (under: S and v in the last hidden layer; S read by the logits; DU9).** Let S be a set of the last hidden layer and v a unit of
that layer outside S with W_out[., v] = alpha 1, alpha != 0 (softmax-unread, with edges), and let some member of S have a
non-zero logit column (so the logits are readers of S and DU9 lets them re-weight kept units). In the class of DU9, suppose
the joint replacement of S at budget k_S and {v} at budget 0 has a least build B_0. Let
g_x = softmax(y_{B_0}(x)) - softmax(y_N(x)) and gamma = sum over x of p_x h_v(x) g_x. Then D*({v}, 0) = 0, and if
gamma != 0,
   I(S, k_S; {v}, 0) > 0.
Proof. Removing v moves every logit by -alpha h_v(x), so the softmax and D_P do not change: D*({v}, 0) = 0. In a single
build of S, v is kept and the logits may change their weights on it (DU9). Take B_0's box and B_0's reader weights and
bias, and keep v with the column alpha 1 + e w. Its logits are y_{B_0} + alpha h_v 1 + e w h_v, with the softmax of
y_{B_0} + e w h_v. Its loss is f(e) = sum over x of p_x phi_x(y_{B_0}(x) - y_N(x) + e w h_v(x)), with f(0) = D_P(B_0)
and f'(0) = w^T gamma (the gradient of phi_x is the difference of the softmaxes). With w = -gamma, f(e) < f(0) for small
e > 0. So D*(S, k_S) < D_P(B_0) = D*(S, k_S; {v}, 0), and I = D*(S, k_S; {v}, 0) - D*(S, k_S) - 0 > 0. QED.
Lean: `exists_help` in `lean/coarse/Coarse/Edge.lean` is the D2 form of the derivative step; the KL form by hand.
Comment. v's column alpha 1 never enters, only its activity h_v. The readers may start to read a unit that they did
not use before, so a unit that is on where S's coarse version errs is a resource for S, and removing both costs more
than the sum. So, under DU9, units are tied by co-activity with the other set's leftover error, read in any direction.
The write tether DA2 does not stop it: the new read is a change of the readers' weights, not a write of a box. In
class (b) of `COARSE-THEORY.md` D50' the logits are free, so the same holds there, also for alpha = 0.

**Remark B.9 (which object is the unit, by reader rule).**

| Readers | What ties two sets | The unit | Proved by |
|---|---|---|---|
| frozen (DU2), writes tethered (DA2) | both on at an input AND their writes overlap in the curvature there (edge tie) | components of the edge-tie graph; orthogonal writes leave a third-order interaction | TA7, TA8 |
| re-fit bias and weights (DU8, DU9, DECIDED) | a unit's activity meets the other set's leftover error (gamma != 0), in any direction | regions of a map meet U4 under TM1's premises: sets of inputs with the units that may be on there, a gate in front (shelter), and a kept judgment. TM1 shows regions are enough, not that a unit must be one | TA12; TM1 of `COARSE-THEORY.md` |

So, with the reader rules Shonu decided, the essay's warning about activations still holds for reading what a unit
means (TA1 to TA5), but the units that TM1 proves independent are fixed by where units are on and by the shelter of
their in-votes. The reason is the readers, not the
units: readers that re-learn may read anything that is on. The edge reading enters the region through the shelter
claim, which is about in-votes (DM3 of `COARSE-THEORY.md`: outside its region a district's inputs do not change).

# Part C: distance of real networks

Nothing measured yet. Literature figure (essay, appendix "Weak Activations: Further Measurements"; not our
measurement): keeping roughly the strongest quarter of positive activations in InceptionV1's final layer keeps top-1
accuracy at 69.47% against 70.08%. This is evidence about the cost of ONE replacement of the whole last layer by a
thresholded copy, measured by top-1 accuracy, not by KL. It is not a measurement of TU14's pairwise terms. (A
thresholded copy h -> h 1[h > theta] is not a ReLU unit; the nearest output-faithful ReLU unit is relu(z_u - theta_u).)
Hypothesis H-district needs one number: the share of the mix's weight on inputs where two districts' members fire
together (Theorem TU3 prices it). Hypothesis H-district-theta needs the same share with "fire" read as "strongly on at theta", for a stated theta
(for example each unit's own 75th percentile on P), plus the size of the weak writes dropped at that theta (TU14 (ii)). Lab instances of B.5 (|x|, the circle toy, T61's instance) are in
`research/clean/units/UNIT-LOG.md`.

# Part D: where the math leads

**CONJECTURE CU4 (firing together across layers) (under: boxes covering on P).** For sets S, T in any layers with no path
between them, U4 fails, apart from cancellation, whenever a unit of S and a unit of T fire together on an input where
one of them changes the logits.
Route: replace TU7's spare unit by a spare unit of T's box placed in T's top layer, and use the one-sided derivative
through the readers that are on at that input.
Lean: none, conjecture.

**CONJECTURE CU5 (deep linear networks) (under: linear units, constant metric).** In a linear network with several
hidden layers, two sets meet U4 at every budget only if no unit of one is correlated on the mix with a unit of the
other or with a unit on a path through it; so units close along the wiring that carries correlated activity.
Route: write each set's best error as a linear function of its in-boundary activity, and repeat TU4 part 2 with spare
units of T placed after T's in-boundary.
Lean: none, conjecture.

**Hypothesis H-district-theta (districts at a threshold) (proposed; node R-district, PENDING).** H-district of A.3 with
"on" read as "strong at theta" (Definition DU6): members of different districts are never strongly on at the same point
of P, and each district's hub is strong wherever a member is strong.

**CONJECTURE CU1 (the one-unit coarse version of a district is a contrast dial over its members) (under:
H-district-theta; DU8; least boxes exist).** Let a district's members split by the readers' contrast into two
subsets A+ and A- (for two logits c, c': members whose write to c - c' is positive, and the rest). Then a least box of
the district at budget 1 has one unit whose read gives non-negative weight to every member of A+ and non-positive
weight to every member of A- (read through the members' pre-activations or their inputs), which is on at every point
of the district's strong support, and whose write is along c - c'. The answer's flip point is set by the readers'
re-fit bias.
Route: TU19 says one unit can carry a contrast only with a reader offset, and only along one direction of the unit's
value. With DU8 the offset is free; the least unit's value must then order the district's points by N's contrast
(a one-variable regression of c - c' on the unit's value, T38 at width 1). Show that, among units of the members'
inputs, the order-preserving one has the sign pattern stated, when the members' supports are ordered along the
contrast (H-district-theta plus a one-dimensional order of the siblings, to be named). Check first on the toys of the
lab log (`cu1_dial.py`).
Lean: none, conjecture.

**CONJECTURE CU2 (where descendants meet) (under: boxes silence-preserving at theta).** Let S and T be disjoint, convex
and unlinked, in any layers. Call a hidden unit q a **meeting unit** if it descends from both. Then the interaction of S
and T is bounded by terms at meeting units only: at each point x, a term that is nonzero only if both replacements
change q's pre-activation at x and q is on (or crosses its hinge) at x in one of the builds; plus TU14's terms at the
logits. With no meeting unit, TU16 gives the logit-level bound.
Route: induction from the top layer down, as in TU16, with the error at each meeting unit split into the two single
changes plus a cross term; a ReLU's second difference is not O(|a| |b|) near its hinge (relu(a + b) - relu(a) - relu(b)
= -eps at a = eps, b = -eps), so the cross term must be bounded by the mass of points where both changes reach q and q
is near its hinge, not by a Lipschitz constant. This is the form in which the essay's edge motifs ("connections ...
predicted where ablation would matter much better than comparable groupings of activations") would enter: a set's
partners are fixed by where its descendants meet AND by strong activity there, not by activity alone.
Lean: none, conjecture.

Conjecture CU3 (withdrawn 2026-10-27, referee report REFEREE-B5U2-2026-10-27.md). It said that sheltered sets with re-fit
readers are independent up to a threshold conflict. False: TU21's comment gives a counterexample (silence used as an
opposite survives DU8). Its provable part is Proposition TU21.

## D.1 The earlier results, read through the unit

| Earlier result (`COARSE-THEORY.md`) | Reading here | Kind |
|---|---|---|
| T58 AND collapse | members that are on together form one co-firing component; the collapse happens inside one unit | special case inside a unit |
| T59.2 free-group detector | its presence unit has a positive offset when the group has two or more members, so it is NOT covering in the sense of DU3; it is covering on P (off at the mix's points outside the group's support) | special case (covering on P) |
| T59.1 clique detector | a covering unit can serve only members it can be on with; the dropped members are the price of covering | the price of covering |
| T61, T62 split dials | free units of {A, C} do part of {B, D}'s job (TU6); the groups are not independent | consistent with TU6 and TU10 |
| T29 covering units are exactly the folds | at one layer, the covering class of DU3 is the class of folds; folds suffice for independence (TU5), free units do not give it (TU10) | folds return as one class that gives independence |
| T32 to T35 covering folds are not optimal | one family and no neighbour: the free gain is a better fit inside the group, not help across groups | unchanged (the price of strict covering) |
| T60 step-by-step copy lemma, H-recur | covering: unlinked districts can be done in any order (TU5, TU9); free: no independence (TU10) | changed |
| T63 to T66 kinks and dip lines | bounds on one unit's least box; unchanged | unchanged |
| T9, T11, T16, T17, T39 gated folds | "one gate per family" and "the region is the gate's" are covering conditions | special case (covering) |
| T38 backfitting | an optimality condition inside one unit | unchanged |
| T42 transfer bound | holds for every class of boxes | unchanged |
| T43 hardness on detector networks | in D40 every member is its own co-firing component, so a group of several members is a choice of unit system; the hardness sits in which members one unit serves | unchanged |
| D45 routes and reads (A.6) | also an activity notion; the co-firing component is the activity notion that independence picks | related |
| D58 to D60 circuits | a circuit is a unit only when it is convex (TU1) and, for covering units, a union of co-firing components | changed |

## D.2 The open nodes of `DECISIONS.md`, read through the unit

| Node | Verdict | Why |
|---|---|---|
| R-edgetether | new (card 3-17) | the write side of "become another reader": a box sends only along its members' columns (DA2). TA6 is the reason; TA8 and CU7 are what it gives |
| R-outfaith | changes | DU6 and DU7 tether what a box reads; TA6 holds inside DU6 and DU7 for softmax-unread units with edges, so they do not tether the direction of what it writes |
| R-circuit | partly informed | in the last hidden layer, with boxes covering on P, the finest unit system is (apart from cancellation) the co-firing components (TU8); with free boxes, no two costly sets are independent (TU11). Convex wiring circuits pass U1 (TU1). The depth converse is Conjecture CU4 |
| R-deep | partly informed | a box reads its set's in-boundary (DU2), and covering ties it to its own members (DU3), so the box definition needs no per-layer decode; whether this reading is the right one for the trunk is still Shonu's choice. The free-box failure of U4 is proved only for the last layer |
| R-kinds | absorbed into R-indep | the kinds that stay in their district (covering on P) are presence and clique units; the split dials are free units that help (TU6); which kinds are allowed is the choice of card 3-15 |
| S-onehidden | changes | TU1, TU2, TU5, TU9 hold at any depth; TU7, TU8, TU10 are for the last hidden layer |
| R-groupopt | changes | covering on P: partitions into co-firing components meet U4; free: no two costly sets are independent (TU11) |
| R-recur | changes | covering: unlinked districts can be done step by step in any order; free: T61 is a failure and TU10 the reason |
| R-freefam | changes | "free" and "conflicting" groups become the price of covering inside one unit |
| R-cleannet | changes | Hypothesis H-district is a candidate principle stated through co-firing |
| S-twoclass | changes | with two logits every last-layer unit has one reader, so the reader overlap of TA7 cannot be seen; results proved with two logits (TU6) are silent about it |
| N-onehidden | unchanged | T20 and T22 concern the least box of one unit |
| N-ny3 | unchanged | T23 concerns attainment, not the unit |
| R-mixclean | unchanged | the hardness of T43 lives inside a unit |
| R-jobmix | unchanged | how the mix is built does not change the unit's definition (the mix enters through co-firing) |
| R-listen | unchanged | a read test for clean, not about the unit |
| R-explaw | unchanged | the base law for targets |
| R-fullrank | unchanged | a first-layer condition for folds |
| R-dialfit | unchanged | an InceptionV1 measurement |

Count (revised 2026-10-28): 2 partly informed (R-circuit, R-deep), 1 absorbed (R-kinds), 7 change (R-outfaith,
S-onehidden, R-groupopt, R-recur, R-freefam, R-cleannet, S-twoclass), 8 unchanged, 1 new node (R-edgetether, card
3-17). Card 3-15 (R-indep) is the fork of the earlier revision.

Conjecture CU6 (formerly DU2; withdrawn 2026-10-27). It said that under H-district the least covering build equals the least free
build. Its route had a false step: "the gate of each district makes the districts' errors visible only on their own
supports". A free box reads its members' pre-activations, and those still vary on the other district's inputs, so a
free unit can fire there and help (Theorem TU10, with its exact one-dimensional instance). The toy with gates of the lab
log (`research/clean/units/UNIT-LOG.md`) shows the same: free boxes help across gated districts.

## D.3 Where B.6 and B.7 point (added 2026-10-28; revised after REFEREE-EDGE-2026-10-28)

The common root of TA6, TU10 and Remark B.6's sources 2 and 3: our boxes may give an activation a read or a write that
N's readers do not give it. DU6 and DU7 closed the read side for what the box reads from its own set. DU2 tethers the
write side only by graph, which is empty in a dense layer. The essay's "become another reader" asks for both sides by
direction. Definition DA2 (node R-edgetether, PENDING) is the write side; Proposition TA8 gives its consequences in the
last hidden layer.

**CONJECTURE CU7 (the deep version of TA8) (under: boxes covering on P; write-tethered boxes; N's columns orthogonal
in the curvature where S and T co-fire).** Let S and T be disjoint, convex and unlinked sets in any layers. Replace
"orthogonal in H_x" by orthogonality in the pulled-back curvature J_q(x)^T H_x J_q(x) at every meeting unit q
(Conjecture CU2) that is on, or crosses its hinge, at an input where both replacements reach it. Then I(S, k_S; T, k_T)
is bounded, as in TA8 part 3, by a sum over those inputs of terms of third order in the changes at the meeting units,
plus terms at inputs where a meeting unit is near its hinge.
Route: TA8 part 3 at the logits; CU2's induction from the top layer down; at each meeting unit split the change into
the two single changes and a cross term; bound the cross term by the mass of inputs near the hinge (TA5: the gate is
the only input-dependent factor of the unit). Check first on a two-hidden-layer toy with a tethered helper.
Lean: none, conjecture.

What this gives the next step (the unit). Under the edge reading, two sets are tied when their activity overlaps on
the mix AND their readers overlap (TA7, TA8), and a box carries only N's existing directions (DA2). In the last layer
that is TA8; CU7 is the deep version. In deeper layers the readers are hidden units with gates, so the tie runs along paths that are
open at the same inputs (TA5): the essay's edge motifs, "recurring patterns of connections", read as sets of paths that
carry flow together. That definition is the next step; it is not written here.

# Appendix

Exact checks: `research/clean/units/` (see `UNIT-LOG.md`). Referee reports for A.5 to D.3: `REFEREE-EDGE-2026-10-28.md` (round 1: 4 errors, fixed; round 2: no errors, 2 gaps,
fixed; TA8 part 3 was added after round 2 and is not refereed). B.6 and B.7 (values moved here from Part B):
`ta4_rank.py` (TA4: KL removal costs 0.120, 0.0050 and 0 for the units of activation 1, 2 and 5; the activation order
and the cost order are reversed); `ta6_unread.py` (TA6, with v's column (1, 1): D2 interaction 0 at budget 0 and -49/20
at budget 1; D_P interaction -0.215 at budget 1); `ta7_reader.py` (TA7: I_rem / eps^2 at eps = 0.001 is -0.0003, 0.3330
and 0.2500 for orthogonal writes, (1, -1, 0) with (1, 0, -1), and two logits; the first goes to 0 like eps, the
third-order term). The lab log's pose toy (`UNIT-LOG.md`, `toy_weights.py`) saw TA6's channel in a trained toy: pose
units kept but not read by the logits, which a box can read.

Referee report: `REFEREE-UNIT-2026-10-27.md` (19 findings, one error). Fixed in this revision: U1 and TU1 now speak of
feedforward (acyclic) builds, and TU1 part 2 gives a directed cycle (findings 1, 2); TU3 slack 2 eps and Euclidean norm
(3); TU4 part 3's left-inverse step and the one-way statements, with |S| <= n_y named (4, 5); TU4's model choices in
its tag, the deep-linear claim moved to Conjecture CU5 (6); TU6 signs computed from D40, supports, T29's premises, the
note on unequal budgets (7); DU3 quantifies over all real in-boundary values and names T29's premise (8); TU10
"separates the points" (9); "no cancellation" defined, TU7 and TU8 part 2 restated (10); TU8 now for partitions of the
whole layer, dead units named (11); H-district renamed "a co-firing hub per district", the essay's upstream gate
discussed (12); TU11 restated as "no two costly sets are independent", with the depth caveat (13); TU9's union needs a
convex union, Remark B.1's wiring condition, Remark B.2 restricted to folds (14); Remark B.3 says "suffices" (15); D.1's
T59.2 row: covering on P, not DU3 (16, the error); D.1 rows T32 to T35, T29, T43, T61 (17); D.2 verdicts for R-circuit
and R-deep lowered to "partly informed" (18); Lean lines describe exactly what is formalised (19). TU5, TU8, TU9 now use
"covering on P", which the toy check `toy_pcover.py` uses too. The fixes are not re-refereed.
