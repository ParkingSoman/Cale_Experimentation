# One coarse neuron

Shonu's project, started 2026-11-21. Read this first, every session.

## The question

> "there is some combination, across all of the writing and the reading, where coarse = parent * children, where * is the operation we are trying to find." (Shonu, 2026-11-21)

## Rules

| Rule | Shonu's words |
|---|---|
| Scope | "if we EVER study anything more than a single coarse neuron along with its immediate children, then we are failing until i say otherwise" |
| Own project | "treat this VERY TINY system as its own project" |
| What to understand | "we need to understand reads, and writes" |
| The essay | "we need to periodically look at the essay" (local copy: `../curves/map_agents/essay.md`) |
| Form of the answer | "a particular method or simple principle ... rather than something very complicated", not "a hodgepodge of a bunch of crazy methods" |
| No hoped-for result | the error-cancelling idea stays in the roadmap |

Every table in `LOG.md` covers only the one parent, its immediate children, and the one coarse neuron.

## Reading rules (Shonu, 2026-11-21: "treat exceptions as things to be scrutinized ... concepts with no exceptions")

| # | Rule |
|---|---|
| 1 | Same tool on both poles. Positive pole: crops centred on the MAX position. Negative pole: crops centred on the MIN position (most negative input before the ReLU). Both by the same bands: top, strong, weak. |
| 2 | Every strong photo is explained: by the trunk (the main job, read on the TOP band), or by a branch traced back to the trunk (recursive generalization, essay lines 131-137). Do not stretch one abstract word over every tile. Weak activations usually do not matter (essay line 353): they are not where a job is defined. Name each strong photo that is not explained. |
| 3 | A reading has two layers, both required (essay lines 101-103 and footnote 3). (a) The PATTERN: pattern A against pattern B in shape, texture, layout and scale words, with no object nouns. It explains WHY things are grouped. (b) The GROUPS it brings in from upstream: noun examples, listed as examples. A noun list without (a) is the drift Shonu flagged; (a) without (b) is the "one-letter overcompression" of footnote 3 (noun-free words fail sibling tests, 2026-11-22). `code/verb_lint.py` checks that the "Verb (job)" column holds (a) with no nouns; (b) goes in the pole columns. |
| 4 | A claim across neurons holds for every neuron, or it is not made. The neuron that breaks it is the next thing to read. |
| 5 | A negative pole is a GATE, not an opposite (essay lines 233, 491: "Think of negative poles as gates in a circuit, not a neuron's opposite"). It holds things that were grouped with the positive EARLIER, which this neuron turns away. Check it with numbers before words: are the neuron's up-voting feeders (or its parent) on at its negative cells? If its down votes are many small ones, say so; do not name one concept for them. |
| 6 | A reading is a hypothesis until it passes a prediction test scored by code (a blind judge predicts held-out crops from the words, against a shuffled-words control). Read the negative side with a reader who has not seen the positive side. The words given to judges hold both layers of rule 3: the pattern first, then the groups as examples. Never revise words toward a judge without testing them again on NEW photos, and log each revision as a mild fit. Power (2026-11-22): one 12-target score has a standard error of about .14, so a test uses 24 targets per pole on two sheets with two different judges, all from one fixed band (ranks 0-190 of the test photos), with the same words for every member in a round. |
| 7 | The first tool for a job is the essay's intersection image (footnote 4): the aligned colour mean and edge mean of each pole's top cells (`code/oc_mean.py`). Crops come second, to check the shape the means show. Pixel statistics that average a whole window miss outlines; prefer the images. |

## Change of gate and scope (Shonu approved, 2026-11-22 afternoon)

| # | Change |
|---|---|
| 1 | A word-free bar is the gate: predictions in activations and weights, written before the run, scored by code on fresh photos. Floor B below (word readings + blind judges) stays as a record, not a gate |
| 2 | The scope may go past one coarse neuron: the second parent `s3:78`, and two more coarse neurons, so a rule is tested on more than one |
| 3 | This project is now week 1 of the hackathon roadmap (`docs/HACKATHON-ROADMAP-2026-11.md` at the repo root). Why: LOG "Session review" |

## Finding children (standing rule, Shonu adopted 2026-11-22 night; R1 replaced 2026-11-23)

Shonu: "the essay itself has a particular way of finding children ... you're kind of just like blankly applying a particular rate." Memory: `feedback-essay-children`. This replaces Floor A's membership rule below, which stays as a record.

Shonu, 2026-11-23: "children is what the next neurons are directly after a parent neuron from a neuron gating standpoint ... the directions it directly writes to ... the neurons which are reading the particular portions of that parent's write." This is the ONE child definition. Do not replace it without a named failure of it and Shonu's word.

| # | Rule | Source |
|---|---|---|
| R1 | Children = the next-layer neurons that read the parent's write, in weights. For a stream channel parent: the child score = the read weight on the parent's channel (bn1 folded, summed over 3 x 3) times the child's write norm (scale-free). The full signed list is the child set; a cut is made only where a step needs one (a replacement), written before its run. Firings check children; they never define them. Script `code/oc_children_w.py` | Shonu 2026-11-23 |
| R1 (old, record) | Dependence on single photos. It found the same top names (LOG "Children by the parent's write") | essay 231 |
| R2 | Group the per-photo routes. Each group is a facet. Groups are evidence for an agent to read, not a final answer | essay 147, 399 (tool); for children: my inference |
| R3 | "Strong" comes from each neuron's own raw activation curve, stated before the run with the curve shown. No cut that is the same for every neuron | essay 327; my inference for membership |
| R4 | Siblings: each carries the parent's route on its own strong photos; they rarely fire together; they feed a shared reader; together they are the cross-section of one circuit | essay 225 ("might"), 501; Shonu's circuit check |
| R5 | Co-firing may check that a coarse neuron covers its children. It never defines siblings | my inference (essay 225, N11) |
| R6 | A child set is closed only by a coverage rule written before the run | my inference |

## Floors (trial rule, Shonu 2026-11-22; Floor A's membership rule REPLACED by "Finding children" above)

Shonu: build an efficient, robust method for (A) having all the children of a neuron and (B) reading a neuron correctly; set a bar of "good"; once it passes, continue toward the operation; any inconsistency reopens A and B. Later, the operation's predicted KL becomes a check on readings. The bars below were written before their runs.

### A: all the children

Definition. Neuron j (A stage-4 block 0) is a child of parent P (a stage-3 channel) when both hold:

| Side | Test |
|---|---|
| Dependence (on j's own top photos) | P is among j's top 3 positive feeders, by the exact input split at j's peak cell, mean over j's top 50 photos |
| Drive (on P's top photos) | P's positive vote into j, counted only where j is on, is above the 90th percentile of all 512 neurons, on P's top 200 photos |

Bar A (all must pass):

| # | Bar |
|---|---|
| A1 | Stable: the child set from two disjoint halves of the photos agrees (Jaccard at least .8) |
| A2 | Not knife-edge: with "top 2" or "top 4" feeders in place of "top 3", the set changes by at most a quarter of its members; the log names every member that moves |
| A3 | Explained spread: the log states the share of P's used drive that the set holds, and the set holds more than any same-size set of neurons for which P is NOT a top-3 feeder |

### B: a correct reading

A reading is the TRUNK (the job on the top band, as a verb) plus named BRANCHES (each one local step from the trunk). Test items come from photos the reader never saw (the train-cache photos), at each neuron's peak cell.

| # | Bar |
|---|---|
| B1 | Positive side: a blind judge (a general-purpose Sonnet agent that follows `judge/JUDGE.md`; Shonu's standing grant, 2026-11-22) gets the words and a sheet of 36 crops: 12 of the neuron's own top cells and 24 top cells of its siblings (where the neuron itself is below its median). It ranks the 12 most likely to fire. Precision at 12 at least .75 (chance .33) |
| B2 | Control: the same sheet with a sibling's words. The own words must beat it by at least .3 |
| B3 | Negative side: only for neurons whose numbers show a gate (README rule 5). Same test on the negative cells, with words written by a reader who did not see the positive side |
| B4 | A neuron that fails is read again with a fixed tool, never with only new words |

## The object

| Role | Neuron |
|---|---|
| Parent | A = torchvision ResNet-18. Stage-3 output channel 69 (`s3:69`): lets radial patterns around a dark centre through. Positive side: wheels and shaggy faces. Negative pole: smooth-coated dogs. |
| Children | A's stage-4 block-0 neurons that read `s3:69` directly with the largest edge, measured on `s3:69`'s own peak positions. Step 1 fixes the list. |
| Coarse neuron | One stage-4 block-0 neuron of a narrow network that does `s3:69`'s job without splitting it. First choice: T0 (block 0 at 128 neurons, trained by KL to A, with A's block 1 fixed as its reader). Fallback: B k128 seed 1 n101. |

T0 is the first choice because its readers are A's block 1, the same readers as the children's. So the coarse write and the children's writes can be compared in one space.

## Essay lines for this object

| Line | Words |
|---|---|
| 23 | "With less room ... one car neuron covering many kinds of car. With more room, that neuron differentiates into many specific cars." |
| 225 | "a neuron in front of them acts as a gate: it responds positively to cars and negatively to dogs, and its negative side keeps the dogs out" |
| 469 | "a network's parts are built to be read, and built upon, by its downstream readers" |

## Steps

| Step | Content | State |
|---|---|---|
| 0 | Set up this project | done |
| 1 | Fix the triple (children by exact edge; the coarse neuron). Stop and show Shonu. | done, approved |
| 2 | Read each member the essay's way: job as a verb, both poles, feeders, readers, growth | done as floors A and B (LOG "Floors A and B"); night pass done (LOG "Picture of the system"); open O1, O3, O4 (O4 allowed since 2026-11-22 pm: second parent s3:78) |
| 2b | Closure: which A neurons are n101's system (LOG "Closure", 2026-11-22 evening) | done, bar NOT met: no set closes, all fail on the weight match only. The s3:69 family is about a quarter of n101's counterpart. s3:78's readers are not part of it. The co-firing group comes closest (weights .56, top photos .87). Waits for Shonu: the next object |
| 2c | Children the essay's way (LOG, same name, and its "Checks") | partly done: by dependence the core children come back (n305, n297, n163, n97, n23, then n108, n237), split by photo class; flow share was a measure artefact; n101 has 4 stable facets, 3 match the parent's groups. Open: a coverage rule for dependence, reader agents (step 4) |
| 2d | Writes: a translation through the readers (LOG, same name, and "Summary of the write translation") | done: effect at the shared head through each network's own block 1. B <-> B works (.41-.57) but is close to a weight projection; A <-> B weak every way it was measured (.13-.23 vs bar .20: not coarseness, not the yardstick, not context). Cause: A's block 1 reshapes A's writes, B's barely does. Children's summed effect vs n101: not shown (below random on every set; rests on the weak translation). Write side of `*` not nameable yet. Fork for Shonu: (a) wide B as the fine network, (b) fixed-reader coarse net with A |
| 2e | Read, one firing at a time (LOG, same name; Shonu 2026-11-23: no nouns, named neurons, close the read first) | done, NOT covered by the pre-written bar on either pole (bin c .71 / .93). Tool first: the in-a-group test had no power (random reads of the same firings cluster as tightly), and on most firings no A neuron is in its own strong half. After checks (LOG "Checks"): on each firing n101's input is s3:69 plus a second feeder that changes (s3:88; s3:86; s3:203 / s3:78; s3:167 / s3:189 / s3:122); the grouping of firings is about half input structure, so the subcircuits are a description; for s3:88 and s3:86 the A neurons that read s3:69 also read the same second feeder (n81, n108; n305, n57, n233); on a single firing no A neuron's split copies n101's (best .50, chance .49), the best matches spread over ~15 A neurons. Hypothesis to test on fresh photos: n101 = s3:69 + union of the A neurons' second feeders, per firing. Curl channels s3:167 / s3:189 / s3:122: no A reader. Animal / artifact split WITHDRAWN. Guard `code/label_gate.py` |
| 3 | Side by side, no fit: firing by case, reads, writes | next (hackathon week 1); writes move to the language model in week 2 |
| 4 | Name the operation. Each candidate has a "Derived from:" line and a prediction written before its test | next (hackathon week 1), gated by the word-free bar |

## Roadmap for later sessions (no work now)

1. The same operation one generation older: `s3:69` as the coarse neuron, with its own parent and its children.
2. The family is a circuit. Use the operation to build the whole circuit.
3. Shonu's hypothesis (recorded, not tested now): because neurons gate, the children's errors cancel each other.

## Where things run

| Item | Place |
|---|---|
| Activations | UVA cluster, `/bigtemp/fkm5yp/n6/clean/neuron/coarse` (caches, A, T0, B) |
| Crop sheets, weight reading | the Mac, numpy only (`../neuron/coarse/sheet18.py`) |
| Results | `results/` here |
