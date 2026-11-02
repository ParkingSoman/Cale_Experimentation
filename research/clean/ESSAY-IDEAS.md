# Essay ideas: concepts from "Neural Networks Are Already Clean" applied to coarsening

Written 2026-10-20 after a full read of the raw essay (943 lines,
https://already-clean.goodfire.com/essay.md). Rule (Shonu): use the essay for CONCEPTS to apply.
The methods below are ours. The essay proposes none of them.

Hedges to keep when quoting: the essay calls its picture a "leading hypothesis" ("maybe 85% sure").
The SwiGLU value/gate claim is for VISION models. The weak-activation accuracy figure is vision
(InceptionV1, ConvNeXt). The language-model claims rest on audits of Qwen3-32B and Kimi K3 and a
forthcoming companion paper. Its coarse layers (480 to 10, 1,024 to 13) were retrained from scratch.

Results of each idea go to CHILDREN-LOG.md, section "GATE FAMILIES".

## Ideas: essay concept, then how to apply it

Each row is an idea for a METHOD. The essay does not propose any of these methods.

| # | Essay concept (where) | Application to coarsening | New vs our log? |
|---|---|---|---|
| I1 | "A gate in front of the car detectors". Taxi and ambulance sit on each other's negative poles under one car gate. Gate = region, value = distinction inside it (How I Think About Neurons, SwiGLU note). | **Shared-gate merge.** For neurons whose gates match, `silu(g)` factors out: `silu(g)·u_a·d_a + silu(g)·u_b·d_b = silu(g)·(x·M)` with `M = Σ u_i d_iᵀ`. A rank-r version of `M`, fitted only where the gate is open, gives r real SwiGLU neurons with the shared gate. The merge is EXACT when gates are equal. This removes the "one half-space" wall that killed every earlier merge. | New. The gate-partner lead was logged and never used. |
| I2 | The upstream judgment IS the gate in front. Siblings build on one judgment (Circuit Ecology, "sheltered place for specialists"). | **Gate tying.** Replace each family member's gate by one shared gate (the best family gate, or the judgment neuron's read direction). Measure the cost. If tying is cheap, every judgment family becomes I1-mergeable, not only the 11-17%. | New. R2.8 showed frozen old gates cost almost nothing. Tying is the next step. |
| I3 | A neuron is a target. "Everything feeding into it makes its own small semantic judgment, and the maximum is where they all agree most" (Blind Maximum). | **Vote pruning (self-coarsening).** Split each neuron's gate and up reads into votes from upstream units (recipe tools). Keep only its broad votes (the judgment neuron's vote). Drop the specific votes. The taxi neuron becomes a car neuron. Siblings then share a gate and collapse through I1. | Partly tested: recipe gates were FITTED to data and tied. Pruning in weight space plus collapse is new. |
| I4 | Recursive generalization: a neuron "already on" for a neighbor is "first in line" to grow into it. An offshoot "arrives by its own route" (Geology / Cesium figure). | **Grafting.** Keep the trunk neuron whole. Add each removed sibling's distinctive route into the trunk's gate and up, but read through a SPARSE upstream neuron that fires only in the sibling's context. A sparse upstream unit is near zero elsewhere, so the graft adds an OR branch without moving the trunk's half-space. This copies how growth makes a coarse neuron. | New. K3 widened with a bias. Recipe parents replaced the gate. No test kept the trunk and grafted routes. |
| I5 | Neurons are analogizers. "If you're talking about voyaging on a boat, the model might judge that a bit Titanic-ish." Weak activations radiate out to related things (Trump, Titanic). | **The weak band is the parent.** Siblings' weak activations overlap where the coarse concept lives (voyages). Parent territory = positions where several siblings are weakly active together. Test: does this band match the judgment neuron's firing? It gives a parent TARGET without a trained layer, and a check for I2-I4. | New. |
| I6 | Readers. Parts are "built to be read by downstream readers". An unread "saliency" coordinate scales what LayerNorm passes (Residual Stream). | **Variance collapse.** A least-squares refit shrinks output variance. After RMSNorm, readers then see weaker signals and gates fall below threshold. That may be why every LS-based merge lands at the cut while fits win. Fix (REPAIR, Jordan et al. 2022): restore readers' pre-activation mean and variance after coarsening. | New diagnostic. Model folding (R0) used a variance fix. We never measured it. |
| I7 | Readers again: "the car readers do get a little cat signal ... many work together". Edge motifs predict ablation damage better than activation groupings. | **Orphaned readers.** When a specialist goes, its exclusive readers in later layers compute on missing input. Remove or mute those readers with it (a branch cut across layers). Test first: how much of a cut's damage comes from orphaned readers firing wrongly? | New. All tests so far cut one layer and froze its readers. |
| I8 | "Percentiles flatten ... whether you're looking at a dial, where the middle matters, or a detector, where it mostly doesn't." | **Type-aware coarsening.** Classify each neuron as dial (smooth range) or detector (cliff) from its raw activation shape. Dials add linearly, so low-rank merge should work for them. Detectors are OR-like, so keep or drop them whole. Earlier merges mixed both. | New. |
| I9 | The smallest ConvNeXt's neurons are coarse, and there "SAEs helped most": coarse units look like building blocks "grouped by context" (What Dictionaries Learn). | **Grouping depends on zoom level.** At the coarsest sizes, context groups (our NMF "topic bins") may be the right families. At mid sizes, job families (judgment keys) may be right. We tested topic bins only at one level and called them wrong. | New framing of an old result. |
| I10 | "Consolidating into a neuron is ... about buildability." Trunks exist because specialists build on them. Toy playground: two neurons reused by two tasks. | **Rank by fan-out.** Keep first the neurons that many later readers read (trunks). Drop first the neurons with one reader (specialists). A dial order, mechanical. | New score. Diverse-cut used judgment keys, not fan-out. |
| I11 | Double-negative regime: "a neuron can fire for two quite different reasons" (Sign Regimes). | **Separate sign regimes in every statistic** used for grouping and merging (gate open vs gate closed). Low cost, applies to I1-I9. | Hygiene fix. |
| I12 | The network is a city. Neurons are control flow that gate regions. Experts are "far more specialized than previously reported". | **Foveated zoom.** Keep the trunk always. Load a family's specialists only while its trunk fires. A dial in memory, not a smaller static layer. | From the first draft of this plan. Kept as one idea, not the main one. |

Recursive generalization (Shonu's lead) shows up three times: I4 copies growth to build a parent,
I5 uses the offshoot band as the parent's territory, and I3 runs growth backwards (remove the
specific votes, get the broad neuron back). It also explains a past result. Our trained parents read
as "dominant child broadened" (K2, A2). Under recursive generalization that is the expected shape of
a coarse neuron, not a caveat.

## My best bet

I1 + I2 + I3 form one method, and the essay supports each link:

1. Find the family under one judgment neuron (works today).
2. Give the family one gate: the shared gate, or a tied gate, or each member with its specific votes
   pruned (I2 / I3).
3. With one gate, `silu(g)` factors out, so the family's values merge exactly into r neurons (I1).
4. The size dial is r per family: r = family size gives the original, r = 1 gives one coarse neuron.
   The dial is continuous and nested by construction (PCA order).

The math wall from M1b ("one neuron cannot express two") does not apply inside a shared gate. That is
the main realization of this pass. I6 is the best cheap diagnostic, because it could explain all the
old failures at once.


## I13 (found 2026-10-20 night): route-capture fold (RCF)

| Essay concept (axiom) | Application |
|---|---|
| Circuit racing: "established routes have a head start in capturing residual error". | When layer-L children are removed, their output is residual error. Let the routes that already fire right before them capture it: layer L's attention heads (they gathered the context) and layer L-1's neurons. |
| Umbrella to bee: new behavior is "an off-ramp between two routes the network already has". | The capture is a linear off-ramp from an existing route into the removed children's write directions. No new gate, so the one-half-space wall never applies. |
| "Gate in front": the coarse judgment already exists upstream. | The coarse version of a child is written by its upstream route, not by a new unit in the same layer. |

Mechanics: one CV-ridge least-squares solve per layer. Features = [kept layer-L hiddens, layer L-1 hiddens, layer-L
attention head outputs]; target = the original layer-L output. The route parts are added to layer L-1's write matrix
and to layer L's attention output matrix: zero extra stored numbers for one coarse model. Needs mixed (prose + code)
calibration and a sink guard (original output at block position 0). Results: CHILDREN-LOG G6-G12.

## Later (product, not now): app-specific folds (Shonu, 2026-10-20)

RCF's choices depend on the calibration text: which children stay, and how each parent's "average child" write is
weighted. Real app traffic may differ. Decision for later work:

| Item | Decision |
|---|---|
| Default | Today's RCF calibration (broad prose + code). It is the generalist setting. |
| Main path | The developer sends example data. Cale picks the kept children and fits the parent writes for that app. |
| Control 1 | Keep the children that fire most on the developer's examples. |
| Control 2 | Keep chosen behavior sharp, let the rest be averaged. Through example sets ("must stay sharp" vs "may be averaged"), never neuron names or words. |
| Storage | Per-app sets cost storage. Shonu: that is a feature (per-app tuning), not a problem. |

Related open question: can app sets share most of their numbers (night report, "storage for a nested dial")?
