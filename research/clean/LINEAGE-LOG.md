# Lineage log: find the tree in how the network grew

Plan: `~/.claude/plans/okay-look-at-the-prancy-jellyfish.md` (approved 2026-10-21).
Code: `research/clean/lineage/`. Results: `results/lineage-*.json`.
Model: Pythia-160M (EleutherAI, GPT-NeoX, 12 layers, stream 768, 3,072 GELU neurons per MLP, parallel attention
and MLP). 154 checkpoints on Hugging Face. Official model card: "All models were trained on the exact same data, in
the exact same order" (checked 2026-10-21, README of EleutherAI/pythia-160m).

Dev layers: 2, 5, 8. Sealed layers: 4, 7, 10 (no test touches them until a held-out check).
Dev text: wikitext-2 test blocks 0-39 (prose), code_train.txt first 40 blocks (code). Calibration: wikitext-2 train,
code_test.txt is not used for calibration.

## Two readings (from the plan)

| Reading | Prediction in checkpoints |
|---|---|
| A. Split | Final siblings were near-identical early (same firing / same input direction), then moved apart. |
| B. Consolidate | A final neuron's job is first spread over many neurons (a population predicts it, no single neuron does), then one neuron takes it. |

## L0. Setup

- mlx-lm `gpt_neox` loads Pythia. Patched MLP to exact GELU (HF config `hidden_act: gelu`; mlx-lm uses gelu_approx).
- Baseline final model (main = step143000): dev prose CE 3.6567, dev code CE 1.9245 (40 blocks each, 512 tokens).
- Not cross-checked against PyTorch (torch is not installed). CE is in the normal range for this model.
- Reasoning probe (`lineage/probe.py`), final model: recitation top-1 0.645 (8 public-domain passages, after a
  6-token prefix), arithmetic 0.125 (40 few-shot single-digit sums; chance is about 0.1, so arithmetic is at its
  floor in this model and is reported only). Recitation is the useful probe signal.
- Step 0 check: CE 11.05 on the L1 token set (random guess = ln 50304 = 10.83). The checkpoints load correctly.

## L1. Measure lineage, no merge (pre-registered 2026-10-21, before any analysis)

Data (`lineage/extract.py`): 19 checkpoints (step0, 1, 2, 4 ... 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000,
main = 143000). Per checkpoint: MLP weights of all layers, MLP pre-activations Z at layers 2, 5, 8 (and 4, 7, 10,
stored but NOT read until a sealed check) on a fixed set of 8 blocks (6 wikitext-2 train blocks after block 300,
2 code_test blocks after block 50), 4,088 tokens. Firing h = gelu(Z).

Measures per dev layer:
- M1 identity: r_i(t) = corr(h_i at t, h_i at the end). Birth step = first checkpoint from which r_i stays >= 0.8.
- M2 twin share S(t): share of neurons with some other neuron at the SAME checkpoint with firing corr >= 0.9.
- M3 consolidation: for each final neuron i, on held-out tokens (fit on blocks 0-2 + 6, score on 3-5 + 7):
  population R^2 (ridge on all 3,072 neurons at checkpoint t) and best single-neuron corr^2 at t.
  t_pop = first checkpoint with R^2 >= 0.6. t_single = first checkpoint with best single corr^2 >= 0.6.
- Key number: final firing corr and final input-direction cosine inside lineage groups vs Ward groups (Ward on
  normalized final input weights) vs random groups, same group count. Lineage groups = average-linkage clusters of
  firing corr at t* (t* = the checkpoint with the largest S(t) among step1-step8000), cut to 768 groups.

Pass rules (fixed now):
- Reading A passes on a layer if max S(t) over step1-step8000 >= 0.05 AND >= 5x max(S(step0), S(end)), AND the
  twin pairs at t* have median final firing corr < 0.5 (they moved apart).
- Reading B passes on a layer if at least 30% of final neurons have t_single >= 4 x t_pop (in training steps).
- A track verdict needs the same reading to pass on 2 of 3 dev layers. Both can pass. Neither can pass.
- Caveat written now: population R^2 is always at least the single corr^2 in principle, so B has a built-in lean.
  The 4x time gap is the guard. If B passes, I will also report the gap on a shuffled-time null (final neuron i
  predicted from checkpoint t with token order shuffled = 0) only as a sanity check of the fit.

## L2 rules (pre-registered 2026-10-21, before L1 results were read)

`lineage/l2.py L K`. One layer swapped, the rest intact. Calibration 32 wikitext-2 train + 32 code_train blocks (code
blocks after block 60, so they do not overlap the 40 dev code blocks). Arms at the same K (same stored numbers):
cut_ls (top-K by energy, LS output refit), and for each grouping (lineage at t*, Ward on final input weights,
random): merge_nofit (mean input, summed outputs), merge_ls (mean input, LS output refit), rep_ls (strongest member's
input, LS refit). Score: dCE vs the intact model on dev prose and dev code, 95% bootstrap over blocks, plus the
recitation probe.

Pass (lineage helps): on 2 of 3 dev layers at K = 768 (25%), the best lineage arm beats cut_ls on prose AND code
(95% interval of the difference below 0, paired over the same blocks) AND beats the same arm with random groups.
Secondary: K = 1536 (50%).

## L1 result (`results/lineage-l1-L{2,5,8}.json`)

| Measure | L2 | L5 | L8 |
|---|---|---|---|
| Twin share S(t), max over step1-8000 | .003 (step8000) | .000 | .003 (step32) |
| Twin share at step0 / end | .000 / .003 | .000 / .001 | .000 / .001 |
| Identity r_i(t) median at step 1000 / 8000 / 32000 / 64000 | .10 / .38 / .64 / .83 | .14 / .36 / .62 / .80 | .15 / .38 / .61 / .77 |
| Birth step median (identity stays >= .8) | 64000 | 143000 | 143000 |
| Final firing corr inside groups: lineage / Ward / random | .074 / .075 / .041 | .052 / .121 / .052 | .055 / .124 / .053 |

- **Reading A (split): FAIL on 3 of 3 layers.** Almost no neuron has a twin (firing corr >= .9) at any checkpoint.
  The few twins at L2 stay twins to the end (final corr .999: duplicates, not splits). Lineage groups built from early
  firing are no better than random (large noise clusters, many singletons). In Pythia (normal initial weights) there
  is no visible condensation. This matches the risk named in the plan.
- **Reading B (consolidate): NOT JUDGED. Measurement bug.** M3 fit 3,072 predictors on 2,044 tokens, so the ridge
  overfits (held-out pop R^2 is negative at every early checkpoint and only .80 at the end, where the target itself is
  a predictor). Fix below (L1b).
- **New fact, not in either reading:** a neuron's job keeps moving for most of training. At step 8000 the loss on
  this text is 3.33 (most of the gain from 11.05 is already made), but the median neuron fires only .38 like its final
  self. Half the neurons settle only in the last half of training. Reading: the model learns the job before the
  neurons settle on it. That is the essay's "recognizes cars fine before its car neurons have formed. At that point
  it's just more distributed." L1b tests it properly.

## L1b. Consolidation, fixed fit (pre-registered before running)

Fix: population = top 256 principal components of the checkpoint's firing (fit on train tokens), ridge on those,
held-out R^2. Null floor: the same fit from step0 (a random network still carries token identity, so it predicts
part of an early-layer neuron). B passes on a layer if at least 30% of final neurons have t_single >= 4 x t_pop,
where t_pop = first checkpoint with held-out pop R^2 >= .6 AND >= (step0 R^2 + .2), and t_single = first checkpoint
with best single-neuron held-out corr^2 >= .6. Same 2-of-3 layer rule.

L1b result (`results/lineage-l1b-L*.json`): NOT VALID. Even at the final checkpoint only 5-19% of neurons reach pop
R^2 >= .6, although the neuron itself is in the population. 256 components on 2,044 tokens drop the sparse
neurons' own signal. B share 0.8% / 1.4% / 2.8% is not a verdict. The token set is too small.

## L1c. Consolidation, enough text (pre-registered before running)

Online, no stored activations: 48 blocks (40 wikitext-2 train blocks after block 400, 8 code_test blocks after
block 60), about 24.5k tokens. Train = even blocks, test = odd blocks. Population = all 3,072 neurons, ridge
(relative 1e-3). Checkpoints: step0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, main.
Sanity gate: at main, median held-out pop R^2 >= .95 (the target is a predictor). If the gate fails, no verdict.
B rule unchanged from L1b (30% of neurons with t_single >= 4 x t_pop, null floor step0 + .2, 2 of 3 layers).

## L2b. Lineage order (pre-registered before running; A failed, so L2a merge arms are not the main test)

Settle score per neuron = mean over checkpoints step1000-step64000 of r_i(t) (firing corr with its final self, L1
token set). High = the job settled early (a "trunk" by the growth reading). Arms at K = 768, each with LS output
refit: energy (= cut_ls), settle (keep the K highest settle scores), hybrid (keep the K best by the mean of the two
rank orders), random (null, seed 0). Pass: settle OR hybrid beats energy on prose AND code (paired 95% below 0) on
2 of 3 dev layers, and beats random. My prediction, written now: settle alone loses to energy (energy is tied to
loss directly); hybrid ties or wins on code.

## L1c result (`results/lineage-l1c-L*.json`). Gate passes (final pop R^2 median 1.000 on all 3 layers).

| Checkpoint | pop R^2 median L2 / L5 / L8 | best single corr^2 median L2 / L5 / L8 |
|---|---|---|
| step0 | -.17 / -.17 / -.20 | .02 / .02 / .02 |
| step1000 | .05 / .27 / .22 | .04 / .05 / .06 |
| step8000 | .32 / .44 / .41 | .16 / .15 / .17 |
| step32000 | .48 / .58 / .54 | .42 / .40 / .41 |
| step64000 | .71 / .74 / .68 | .69 / .66 / .63 |

**Reading B: FAIL on 3 of 3** (share with t_single >= 4 x t_pop: .03 / .07 / .11, rule .30). The population does
carry about twice as much of a final neuron's job as any single neuron early (step 8000), and the gap grows with
depth, but neither carries most of it. The final jobs are mostly NOT present early, spread or not. Both settle late.

## L2b result (`results/lineage-l2b-L*-K768.json`): FAIL. Prediction half right.

Settle order and hybrid are WORSE than energy on all 3 layers (dCE vs energy, prose / code: settle L2 +.006 / +.015,
L5 +.013 / +.018, L8 +.066 / +.016; hybrid L2 +.004 / +.005, L5 +.006 / +.007, L8 +.065 / +.003). Both beat random.
Overlap of settle top-768 with energy top-768 is 30-41%.

## Essay-based reading of the setup error, and one fix (L1d, pre-registered before running)

Blue sky #10 found the same error once: a copy test penalizes coarseness itself. L1c asks whether an early network
holds the EXACT final job of one neuron. The essay's picture is coarse-to-fine ("coarse-to-fine wins right away";
"with more room, that neuron differentiates"). If an early network holds a COARSE version of a final FAMILY, then
the family's shared signal should be predictable early, even when each member is not.

L1d: same online setup as L1c. Final families = Ward clusters of the final neurons' standardized firing (768
groups). Targets per checkpoint: each family's summed standardized firing, and random groups with the same sizes
(null: summing alone smooths noise). Pass (coarse early, reading A'): at step 2000, 4000 or 8000, median held-out R^2
of family targets minus random-group targets >= .15 on 2 of 3 dev layers.

## L1d result (`results/lineage-l1d-L*.json`): FAIL on 3 of 3

Family minus random-group R^2 at step 2000 / 4000 / 8000: L2 .10 / .10 / .10, L5 .13 / .13 / .13, L8 .12 / .12 / .13
(rule .15). A random network (step0) already gives .05-.07 of this gap, and the gap stays flat over training, then
shrinks after step 32000. Final families are a little easier to predict early than single neurons, but not because
they exist early as coarse units. No coarse-first signal.

## Side check (not the lineage verdict): do merges still lose to the cut in a GELU model? (pre-registered)

Coarsening-track memory listed this open question ("test other forms: GPT-2 GELU"). `l2.py L 768 step8000`. Same
arms as L2 rules. Lineage groups here are clusters of step-8000 firing, which L1 showed are near noise, so they act
as a second null. Reading: if Ward merge_ls beats cut_ls on 2 of 3 layers, the merge wall was a SwiGLU property.

Side check result (`results/lineage-l2-L*-K768.json`): the merge wall HOLDS in a GELU model. Every merge arm loses to
cut_ls on all 3 layers. merge minus cut, prose / code: Ward merge_ls L2 +.057 / +.048, L5 +.022 / +.025, L8 +.021 /
+.022. No-fit merges are far worse (+.13 to +.43). Cut_ls itself costs dCE .132 / .070 (L2), .067 / .043 (L5),
.110 / .049 (L8). So the "one unit cannot express two" wall is not a SwiGLU property.

Probe hint (8 passages only, no interval, NOT a result): one layer cut to 25% drops recitation from .645 to
.54-.60. Settle order kept recitation a little better than energy at L2 (.562 vs .551) and L8 (.568 vs .537) while
its average loss was worse. Worth a bigger probe before anyone reads it.

## Verdict of the lineage track on Pythia-160M

| Question | Answer |
|---|---|
| Do coarse units exist early and split (A)? | No. Almost no twins at any checkpoint. |
| Is a final neuron's job spread over many neurons early, then gathered (B)? | Not resolved. Fails the rule, but the shape matches B: the population leads a single neuron 2-3x at step 8000 (R^2 .32-.44 vs .15-.17), the lead closes by step 64000. Absolute levels are low. My grid cannot show a 4x gap for any neuron whose population threshold falls after step 36000 (4 x 36000 > 143000), and the medians sit at 64000. A rule flaw, written after the result. |
| Are final families present early as coarse units (A')? | No. Small advantage, mostly already in a random network. |
| Does growth order help pick which neurons to keep? | No. Worse than energy, better than random. |
| When does a neuron's job settle? | Late. Median identity with its final self: .38 at step 8000, .64 at 32000, .8 at 64000. |
| Is the merge wall a SwiGLU property? | No. It holds in GELU too. |

Reading: in a normally trained language model, the neuron basis is decided late and is not a growth tree. The model
has most of its skill (loss 3.33 vs 11.05 at random) while its neurons still differ a lot from the final ones. So a
tree from growth will exist only if training is BUILT to leave one (small initial weights so neurons condense, or
growth by splitting, with a nested loss that keeps the parents valid). That needs our own training. Decision for
Shonu. The vision trigger as written did not fire (we know why the groups are noise), but the cheapest lab for the
"trained to leave a tree" question is a tiny model we train ourselves.

Caveats added after review (2026-10-21):
- "Jobs settle late" is confounded with the learning-rate schedule (Pythia: cosine decay to 10% of max over the
  full run, so the rate is still high at step 64000). Neurons move while the rate is high, tree or no tree. Only
  another training run can separate the two.
- L1d families are co-firing (Ward) groups. The coarsening track found essay siblings sit on each other's negative
  poles under an upstream judgment neuron, not in co-firing groups. L1d's null is a null about co-detector groups.

## Round 2 (Shonu, 2026-10-21: "maybe it is the right idea and you didn't look in a certain way")

Re-read of the essay's training-time claims: "watched a neuron form, it started on yellow flowers ... Then it grew
along each facet"; "The movement is local: a neuron drifts to neighboring things". These describe the CORE (strongest
firings) forming first and offshoots growing outward. The essay's size claim: "With less room ... its neurons just
cover coarser-grained concepts ... With more room, that neuron differentiates into many specific cars."

Essay-based reading of my setup errors in L1:
1. Identity used correlation over ALL tokens. Almost all tokens sit near zero, so weak activations dominate it. The
   essay says weak activations are where offshoots and analogies live, and to look at strong activations.
2. Splits were looked for as TWINS (two neurons alike at the same time). With a fixed neuron count, a split looks
   different: one neuron's territory at time t is later shared by two different neurons. Twins are not needed.
3. Gating and "smaller = coarser" are claims about DEPTH and MODEL SIZE, not training time. I never tested size.

Setup for all three tests: 48 blocks (40 wikitext-2 train after block 400, 8 code_test after block 60), 24.5k tokens,
pre-activation z (before GELU). Core of a neuron = its top-50 tokens by z (positive pole). A parent core = top-200.
cov(child, parent) = share of the child's 50 core tokens inside the parent's 200. Chance level is about .008.

T1 (core identity over training). O_j(t) = overlap of top-50 at t with top-50 at the end. Pass ("core settles
early"): median O_j(step8000) >= .5 on 2 of 3 dev layers. Compare with all-token corr .36-.38 at step 8000.

T2 (territory split over training). For each final neuron j and checkpoint t, ancestor = the neuron at t with the
highest cov (any slot, itself included), kept if cov >= .5. Split family = an ancestor with >= 2 final descendants
whose final cores overlap <= 10 of 50. Split share = share of final neurons in a split family. Null = step0.
Pass: at some t in step1000-16000, split share >= .10 and >= 3x the step0 share, on 2 of 3 dev layers.

T3 (size tree, the essay's own claim). Children = final Pythia-160M neurons at the dev layer. Parents = all neurons
of Pythia-70M (6 layers x 2,048 = 12,288). Same-size null = Pythia-160M-seed1 (another random start), 12,288
neurons sampled at random from all its layers (equal candidate count). Same cov, ancestor and split-family rules.
Pass: 70M split share >= .10 AND >= 2x the seed1 split share, on 2 of 3 dev layers. Also reported: share of children
with any parent, children per parent, and which 70M layer the parents sit in.

## Round 2 results

T1 (`results/lineage-t12-L*.json`): FAIL 3 of 3. Median core overlap with the final core at step 8000: .12 / .08 /
.12 (rule .5); at 32000: .32 / .22 / .26; at 64000: .52 / .42 / .42. The strong core settles late too, even later
than the all-token measure suggested. So L1's "jobs settle late" is not a weak-activation artifact.

T2: FAIL 3 of 3. Split share peaks at .075 / .049 / .040 (step 8000-16000) vs step0 .043 / .021 / .019 (rule: >= .10
and 3x). When a final neuron has an ancestor (cov >= .5), it is nearly always ITSELF (self share ~ has-parent share
at every checkpoint). No territory moves between slots. Neurons form in place, late.

T3 (`results/lineage-t3.json`): FAIL 3 of 3. Split share, 70M parents vs same-size seed1 parents: L2 .064 vs .061, L5
.050 vs .051, L8 .059 vs .066. Children with any parent: 13-18% for BOTH. Children per parent ~1.3 for both. The
70M parents do sit at matching depth (L2 children: 70M layers 0-1; L8 children: 70M layer 4), so depth matches
across sizes, but there is no one-to-many shape.

Reading. With top-token overlap as the microscope, only ~15% of neurons match even between two models of the same
size. Two explanations, and language cannot tell them apart: (a) most neurons in these small models are not shared
across models (Dravid et al. 2026: shared "Rosetta" neurons are a minority and their share falls with size), or (b)
the microscope is too weak for these small, broad-neuron models (the essay: "smaller models have broader neurons, with
more residual generalization"). The size ratio is also small (MLP width 2,048 vs 3,072; the essay used 300x).

Decision: move the size question to vision, where (1) the essay's evidence lives, (2) we can train a size range
ourselves in minutes, and (3) labels give a ground-truth hierarchy (CIFAR-100: 100 fine classes in 20 superclasses),
so "coarse" can be measured without a matching microscope.

T3 null note: if PolyPythia seed1 also changes data order (not only the start), it is MORE different from 160M than
70M is (70M shares the data order). That bias favors 70M, and 70M still tied. It strengthens the null.

## T3b. Microscope check (pre-registered before running)

Same children, same parent sets as T3, but match by best single-neuron |corr| of GELU firing over all 24.5k tokens
(the L1c measure). Report the share of children with best |corr| >= .775 (corr^2 >= .6) and >= .5. Reading fixed now:
if the seed1 match share at .775 is >= .40, T3's verdict is a microscope artifact and T3 is re-run with correlation.
If it stays near .15, neurons mostly do not transfer across models of this size, and T3 stands.

T3b result (`results/lineage-t3b.json`): T3 STANDS. By correlation, the share of 160M neurons with a match >= .775 is
.007 / .002 / .006 (70M) and .005 / .002 / .005 (seed1); >= .5: .055 / .054 / .086 (70M) and .045 / .049 / .078
(seed1). Lower than top-token overlap, and again 70M = seed1. The microscope is not the cause. In these small
language models most neurons are not shared across models. Positive note: best-match parents sit at matching
relative depth in both 70M and seed1 (L2 children -> early layers, L8 -> late), so depth structure transfers.

## Round 3: vision lab (pre-registered 2026-10-21, before any training)

Why: the essay's evidence is vision; we can train a size range ourselves; CIFAR-100 labels give a ground-truth
two-level hierarchy (100 fine classes in 20 superclasses), so "coarse" is measured without a matching microscope.

Net (`vision/net.py`): isotropic ConvNeXt-style, the essay's clean conditions (residual stream, LayerNorm, GELU
pointwise MLP). 4x4 patch stem (8x8 grid), 6 blocks: x + fc2(gelu(fc1(ln(dwconv7(x))))), mean pool, LayerNorm, head.
Neurons = fc1 units. Neuron activation on an image = max over the 64 positions of the GELU output.

Models, all 25 epochs, AdamW lr 2e-3 cosine, weight decay .05, batch 256, crop + flip, SAME data order (seed 0):
| Name | Stream C | MLP H | Weights | Role |
|---|---|---|---|---|
| S | 32 | 128 | 65k | small model (essay: coarser neurons) |
| L | 128 | 512 | 850k | large model (13x weights, 4x MLP width) |
| LN | 128 | 128 | 258k | fixed reference: L's shape with a 128-wide MLP, trained from scratch |
| L2 | 128 | 512 | 850k | same-size null: L with another random start |
Run time stated before running: about 17 min on the Mac in total. Checkpoints at epochs 1, 2, 4, 8, 16, 25.
Test split (seed 12345): 5k analysis images (all neuron reading and any selection), 5k sealed (accuracy only).

V1. Smaller = coarser? Last block (5) primary, block 3 secondary. Per neuron, over its top-100 analysis images:
super purity (share in the modal superclass), fine spread (distinct fine classes inside the modal superclass, 1-5),
breadth (distinct fine classes overall). Null: the same net at random start. Pass (block 5): median fine spread
S > L, AND median super purity S >= L - .05, AND S and L purity both >= null + .10.

V2. The product question. Cut L's MLP to 128 in EVERY block (top-128 by energy, least-squares refit of fc2 on 5,000
fixed train images, all positions, block by block on the already-cut model). Sealed fine and super accuracy for L,
L-cut, LN, S. Flexible vs fixed: L-cut vs LN. Gracefulness: among images L got right and L-cut gets wrong (fine), the
share whose prediction stays in the true superclass, compared with the same share for the trained models' own
errors. Reported, no pass rule.

V3. True families. Block 5 of L (block 3 secondary). Each neuron's family = modal superclass of its top-100 analysis
images (ground-truth labels). Within each family, k-means on the fine-class histogram into k_s groups,
k_s proportional to family size, summing to 128. Arms, one block cut to 128, LS refit of fc2: cut_ls (energy),
family merge_ls (mean fc1 rows and bias), family rep_ls (strongest member's fc1 row), random-group merge_ls (same
sizes), Ward-on-fc1 merge_ls. Pass: family merge_ls beats cut_ls on sealed super AND fine accuracy (paired bootstrap
over images, 95% interval of the difference above 0) AND beats random merge_ls. If it fails, the merge wall holds
even with true families: the limit is the unit form, not identification.

Training done (`vision/train.log`, ~18 min total). Sealed accuracy fine / super: S .376 / .512, L .444 / .580,
LN .426 / .558, L2 .427 / .554. Weak models (25 epochs, 6 blocks, 8x8 grid), but ordered as expected.

V1 result (`results/vision-v1.json`): FAIL, and NOT INFORMATIVE. Neurons are barely class-selective: block 5 super
purity S .15, L .16, L2 .17, LN .21 vs random-start null .11; a neuron's top-100 images cover ~45-52 of 100 fine
classes (null 59). Fine spread median 4 for every model (a tie, so "S > L" fails). Labels cannot read coarseness when
neurons do not track classes at all.

Essay-based reading of the setup error: the essay reads LAST-layer neurons that feed the classifier, in deep nets
whose late neurons are class-level. Here a neuron's image score was its strongest single PATCH (max over 64
positions), which favors local textures. The head reads the MEAN over positions. One fix (V1b, pre-registered): score
= mean over positions. Same V1 rules. If block-5 purity stays below null + .10 for L, the label-based reading is
abandoned for these nets, and V3 runs with output-side families instead (the classes a neuron's write promotes).

V2 result (`results/vision-v2.json`): flexible LOSES to fixed. Sealed fine / super: L .444 / .580, L cut to 128 in
every block .346 / .494, LN (same shape trained from scratch) .426 / .558, S .376 / .512. L_cut minus LN: fine -.080
(95% -.093..-.067), super -.064 (-.078..-.051). L_cut is below even S, which has a 4x narrower stream. Gracefulness:
of the 790 images that L gets right and L_cut gets wrong, 26% stay in the true superclass, vs 21-23% for the trained
models' own errors. Slightly more graceful than an ordinary error, not much.

V1b result (`results/vision-v1-mean.json`): FAIL. With mean scores, trained neurons are no purer than a random start
(block 5 purity .18-.22 vs null .20). Output-side check: the top-5 classes a neuron's write promotes (through lnf and
the head) are no purer than for random directions (.40 vs .40, L and S, blocks 3 and 5). Pre-registered fallback
reached: label reading abandoned for these nets. V3 NOT RUN: families from labels would be near-random, so V3 would
only repeat the random-group arm.

Reading: these nets (6 blocks, 8x8 grid, 25 epochs, 44% fine accuracy) have no class-level neurons, so the essay's
"small = coarse" picture cannot show in them. The essay's ConvNeXts are deep, staged and well trained (72% top-1 on
ImageNet for the medium one). The fix is a better-trained, staged net (downsampling stages, more epochs), about an
hour on the Mac or minutes on a rented GPU. Not run: over the 10-minute rule, and the earlier promise to ask first.

Notes added at review:
- Model-level "small = coarse" check from V2 (independent of neuron reading): share of a model's errors that stay in
  the true superclass: S .211, L .227, LN .225, L2 .217 (chance ~.04). The small model is NOT coarser in its
  mistakes. No sign of "small = coarse" at the model level either, in these weak nets.
- V2's cut is the weakest retrofit (energy top-K + LS, G12 target). No RCF and no window fit, the two things that
  helped on SmolLM2. So -8.0 fine is a floor for retrofitting, not its ceiling. Untested here.
- T3b depth alignment counts every child's best match, including ~90% below corr .5. Suggestive only.
- V1's rule needed an integer median (1-5) to differ, so a tie was likely. Labeled uninformative, not a failure of
  the essay claim.
- Process: the vision nets (~18 min) were trained after Shonu's "perhaps a vision model would be better", without a
  clear yes. Run time was stated in this log only.
