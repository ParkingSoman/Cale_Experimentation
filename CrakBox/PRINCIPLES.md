# What the essay says about mapping

This is the deep dive. It reads "Neural Networks Are Already Clean" for one question: how should a network be mapped, again and again, at a cost we can carry?

| Item | Value |
|---|---|
| Essay text | `research/clean/curves/map_agents/essay.md` |
| A quote written as `L401` | Line 401 of that file. `tests/test_quotes.py` checks every quote against its line |
| The rules that come from this study | `CONSTITUTION.md` |

The essay's numbers come from its own networks. Most of its claims are observations, and it hedges many of them. They are references for us, not promises about ResNet. It also says less about mapping method than this study may suggest: the method below is my reading.

## The one-paragraph answer

The essay does not offer an algorithm for mapping. It offers a laboratory: agents read a network's own parts, guess what each one does, try to break the guess, and keep what survives. Understanding compounds, so the map gets easier as it fills. The fixed part is a set of rules around the agents. The essay calls this a "mechanistic constitution". That is the strict and flexible split you asked for: the rules are fixed, and the reading is free.

L539: "Guided by a mechanistic constitution, agents propose and test explanations at many scales, each grounded in individual neurons, and keep, drop, or replace them with better ones."

## The fourteen principles

### 1. Become another reader

L469: "a network’s parts are built to be read, and built upon, by its downstream readers"

L469: "become another reader"

The test of a map is what the network itself does with the part. A human's sense of the part comes second.

L471: "We should be wary of reading it too much like a human"

| In the tool | Every card says what reads the node. A code-scored test checks the claim about readers |
|---|---|
| Today | Readers are a view. Nothing scores what a card says about them |

### 2. Ordinary investigation at scale beats a clever decomposition

L63: "scaling ordinary investigation beats engineering a clever new decomposition"

L513: "a general scientific agent moving between the neurons, data, and groups of both, not a fixed optimization"

L537: "Each step is simple"

So "automated mapping" does not mean a mapping algorithm. It means an automated lab. This also matches your rule of 2026-11-01. The essay reports that the process worked on modern convolutional networks, with its own hedge about language models.

L51: "Agents can map out neurons and circuits, build explorable microscopes and explore them, test explanations thoroughly"

| In the tool | Code ranks and computes. Agents make claims. No step groups units by one rule |
|---|---|
| Today | True in spirit, built as one-off scripts per round |

### 3. A constitution, with free judgment inside it

L539: "agents propose and test explanations at many scales"

| In the tool | `CONSTITUTION.md` holds the fixed rules. Workflows guide the reader, who may leave them with a reason |
|---|---|
| Today | A brief for each round, rewritten each time |

### 4. Explanations must be able to lose

L401: "The best criticism is specific to the conjecture."

L509: "guess what something is, try to break the guess, and keep what survives"

L393: "a second sees only that explanation and writes an example meant to activate it maximally"

L401: "they all take creativity and judgment rather than a summary statistic"

This principle changes our work the most. A general test can be passed by a wrong explanation.

L401: "could do well on many tests of the V neuron while completely missing what it really is"

L37: "rewarding explanations that could simulate activations rather than ones that were right, which often come apart"

L565: "So I’d rather rely on normal science than on any one metric."

Our crop test asks which crops fire, and its fire crops sit near .57 of the largest firing. That is close to a simulation score. So it stays a reported number and stops being the gate.

| In the tool | A critic attacks each claim with a specific test. The blind judge runs the center, near-miss and far-concept tests |
|---|---|
| Today | Agents grade their own predictions |

The essay's reference numbers, from a language model:

L641: "the new examples fired their neurons strongly 76% of the time; the near misses, 11%"

L647: "Those texts fired their neurons strongly about 1% of the time"

### 5. Many separate views point to one place

L43: "Any of these could break the explanation, yet they keep pointing to the same place."

Line 43 lists the views: the surrounding context, steering, upstream, downstream, the negative pole, and specific tests.

| In the tool | A verdict needs several tests that could have broken the claim. See "The views we have and lack" below |
|---|---|
| Today | Several views exist. No rule requires that they agree |

### 6. The crossword: the map gets easier as it fills

L39: "nearly empty and mostly filled in aren’t surprising places to be; they’re the natural anchors"

L45: "When you can see 2% of a network, you can’t tell where your tools are broken; when you can see most of it, the unexplained remainder points straight at them."

L49: "like working on one row of an otherwise empty crossword"

Our last round read 126 of 2048 stage 4 nodes, about 6%. By the essay's picture, that is near the hard anchor.

**What our data says, and its limit.** I tested this on the old cards. Reading order showed no effect (r = -.06, p about .5, inside each layer). The trend with neighbours was not significant (r = .13, p = .17), and channel strength confounds it. So the data does not show compounding. It is also a weak test: fill was 6%, and the agents barely shared cards. Compounding is the essay's claim, and it can fail here. Step 7 tests it, with a control arm.

| In the tool | Fill a connected scope in waves. Later readers see earlier cards |
|---|---|
| Today | Thin samples |

### 7. Motifs compound

L43: "you start to know what to expect from the next one, which is part of why the map fills in faster as it goes"

L487: "I’d take the motifs in a minute"

| In the tool | A library of motifs and workflows, kept with versions and records. It is the asset that makes the next map cheaper |
|---|---|
| Today | `NOTES2.md`, shared notes for one round |

### 8. A verb and not a noun, and both poles

L101: "the neuron is better understood as a verb than a noun"

L101: "Both human and automated interpreters seem strongly biased toward describing neurons as detectors for groups of nouns"

L103: "if I couldn’t use groups of nouns at all, how would I describe this neuron to someone else?"

L233: "The negative side of neurons matters almost as much as positives"

L353: "The neuron’s negative side matters about as much as its positive."

| In the tool | Required card fields, a required test of the negative pole, and a workflow for cards that read as noun lists |
|---|---|
| Today | In the brief. Our notes still lean on nouns, and no test scores the negative pole |

### 9. Treat each neuron by its role

L31: "I’m somewhat against trying to standardize things across all neurons."

L31: "treating each neuron and circuit differently, according to their motifs, depth, role etc."

L327: "it’s often what tells you whether you’re looking at a dial, where the middle matters, or a detector, where it mostly doesn’t"

| In the tool | Code sees a situation and selects a workflow. A dial gets an ordering test, and a detector gets a center test |
|---|---|
| Today | The same test for every node, which may be why stage 1 sat at chance |

### 10. Blame the microscope, and tune it to the architecture

L685: "blaming the network for the microscope"

L109: "The microscope just wasn’t tuned right"

L421: "find what the architecture is doing and work with it, rather than fighting the network"

L173: "Polysemanticity is often more a claim about the observer than the neuron."

L205: "I was making superposition the null hypothesis when it should have been the last hypothesis"

The essay's cases are specific: a saliency coordinate that nothing reads (421), a double-negative regime in one activation (377), and feature visualization that fell into a basin of dogs (91). Each one was found by looking harder at one network. So each new network needs a calibration step, and the list of what to check grows.

The essay also gives a reason for this stance:

L525: "the mistakes aren’t symmetric"

| In the tool | A calibration file per network. An open node names its suspect |
|---|---|
| Today | Your rule, "id rather guess the map is wrong", is in memory and in the brief |

### 11. Bigger networks have narrower neurons

L47: "bigger models learn narrower neurons that are easier to recognize in the first place"

L415: "bigger models tend to have more narrow neurons"

L501: "agents found labeled dog-ear neurons faster in the larger models"

So ResNet-18 may be the hardest network of its family to read. This is an inference from the essay, and it is not measured on ResNet. It gives a new view: a hard channel in ResNet-18 may have near-matches in ResNet-50 that read easily. The essay does not propose this view. Your rule from 2026-10-20 asks for applications like it.

| In the tool | A view of a channel's closest channels in ResNet-50, built when a stuck node needs it |
|---|---|
| Today | Absent on this track. `MAP-STATE.md` holds a related ConvNeXt result |

### 12. Fractal, with tethered zoomed-out views

L35: "start with a basic explanation, then zoom in when you want to understand more"

L505: "views that let us zoom out while keeping the larger patterns connected to the neurons and circuits that support them"

L505: "There should be many maps and many agents"

L519: "fog-of-war map filling in at several levels at once"

**This is the direct link to Cale.** The essay's zoom on a map and Cale's zoom on a network have the same shape. The map's levels become the trunk's groupings, and each level is tied to real channels.

| In the tool | Each understood node writes its claims with code-computed errors. Levels above the node come from those records |
|---|---|
| Today | Cards and claims are separate |

### 13. Flows of attribution are a level of their own

L399: "we also cluster the attribution flowing into the neuron"

L653: "predicted where ablation would matter much better than comparable groupings of activations"

L829: "We tend to favor more dynamic groupings, like clustering locally within one attribution graph or a group of them, rather than one global clustering"

L513: "they belong at the end, not the start"

So local clustering is allowed as evidence, inside one node or one graph. It never makes the map.

| In the tool | Bundles stay. A wider edge-motif view gets built only if a stuck node needs it |
|---|---|
| Today | `bundles` exists |

### 14. Watch it grow

L133: "When we retrained the model and watched a neuron form"

L213: "it’s easy to check if you have training checkpoints"

L819: "a specific shape in both semantic space and time"

| In the tool | A named gap |
|---|---|
| Today | Torchvision's ResNet-18 has no checkpoints, and ImageNet's training set is not on this disk |

## The views we have and lack

| View | Essay | Today | Next |
|---|---|---|---|
| Strong examples, in rings by share of the largest firing | 327 | Have | Keep |
| Negative pole | 233, 353 | Have | Add a test |
| Exact inputs, with signs | 233 | Have | Code proposes links |
| Readers | 43, 469 | Have | Add a code-scored test |
| Bundles of inputs | 147, 399 | Have | Keep. Three-way bundles did not lower the input count in LADDER 57 |
| Removal against controls | 233, 447 | Have | Keep. It passes too easily to be a line |
| One example followed through the network | 229 | Have, `trace` | Use it in workflows |
| Edits to an example | our own | Have | Good and cheap, per `NOTES2.md` |
| Drawn tests | our own | Have, partly | Works for line nodes. Fails for round bodies, per `NOTES2.md` |
| Feature visualization | 111, 113 | Set aside as faint | A faint view is a microscope debt (below) |
| Views of what a bundle's inputs share | 71, 79 | Lack | Build when needed |
| Natural-route visualization | 571 | Have, unproven | Test on known nodes |
| Steering: turn a node up, read what changes | 43, 191 | Lack | Vision's form: amplify a node, read the class shift |
| Near-matches in a larger network | 99, 503 | Lack | Principle 11 |
| Edge motifs from many graphs | 653 | Lack | Build when needed |
| Growth over training | 133, 685 | Lack | Named gap |

**On feature visualization.** The brief told agents not to spend calls on `fv`, since one view was faint. The essay reports the same symptom as a tool fault:

L111: "Once researchers added it, the same neurons came out crisp."

L71: "To see the shared pattern, you have to visualize what the neuron’s inputs have in common, not the neuron on its own."

L571: "bias the visualization toward the routes real images use"

I have not tested whether these fixes help on ResNet-18. It stays first in line among the tool debts.

## Where the essay pulls against this project

These tensions are real. The tool has to be honest about them.

| # | Tension | Essay | How the tool handles it |
|---|---|---|---|
| 1 | You ask for one defined way. The essay resists standardizing | L31, above | The process and the judge are standard. The reading and the tests per situation are not |
| 2 | Your clean is 100% of a scope, per node. The essay works differently | See below | See below |
| 3 | Tests can pass wrong cards | L401, above | The judge is necessary and not sufficient. The critic and your spot-check remain |
| 4 | The essay's blind maximum writes new text. Images are harder | See below | See below |
| 5 | Building a tool can replace looking | See below | Tools get built for named stuck nodes only |

**Tension 2, said plainly.** Your standard is stricter than the essay's own reports.

L519: "less like clearing a bar for each piece and more like a fog-of-war map filling in at several levels at once"

L515: "There’s always another weak activation to point to."

L637: "I don’t trust our current tools enough to turn it into a cleanliness score"

L29: "only ~2-4% are not yet understood"

L731: "I don’t fully understand these exceptions yet"

L213: "we don’t have to force an explanation we don’t have"

The essay's author, after much longer study, still reports open cases. So reaching 100% of a scope may be hard, and I do not know how hard. Three things follow.

1. A scope limits what counts, so the goal can be reached. You confirmed this limit on 2026-11-03: strong firings, both poles, and weaker ranges that code shows the goal depends on.
2. An open node stays open. Nobody may force an explanation to move the bar. That would break the bar and the essay's rule both.
3. If the limits are hit, the map stays "not done", and I show you what was tried.

The essay supports a chosen scope:

L479: "It is also broadly fine ignoring lots of the neural network"

The weak-firing limit rests on this finding about its networks, so code checks it on each scope:

L309: "you can ablate almost all the weak activations of the network for almost no cost"

**Tension 4, said plainly.** In a language model, a judge can write a new sentence, and it may beat every real example.

L395: "a neuron as a target, with meaning growing out from the center in every direction"

L399: "a blind maximum tests the neuron’s center, not its breadth"

On images, our drawn shapes do not fire round-body nodes. So our first form of this test is a choice from a large sealed pool. That is an adaptation, and it is weaker: a judge cannot beat the pool. It also means near misses and far concepts come from the pool, not from a judge's own writing. Wrong cards must fail a test before it counts. The first form of this test failed that check, so it was replaced (LOG, section 3).

**Tension 5, said plainly.** This is the danger of this project.

L525: "I spent most of a decade figuring out how to look at a model rather than looking at it"

So every round after setup must add judged nodes.

## What I did not get from the essay

| Item | Note |
|---|---|
| A numeric pass mark for a node | The essay gives none, and it resists them. Our marks come from calibration on wrong cards |
| What a "parent" is, as a definition | Your tree is your object. The essay offers the gate (225), growth (133) and coarse-to-fine differentiation (23). These fit a tree, and none of them defines one |
| Results on ResNet-18 | The essay does not report any |
| How much reading a network needs at a given size | The essay's agents read at very large scale. It gives no cost per neuron |
