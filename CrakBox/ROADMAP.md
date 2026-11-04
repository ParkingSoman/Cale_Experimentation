# CrakBox roadmap

Shonu, 2026-11-04: "the first milestone being to achieve 100% interpretability for a small network which is actually useful, and then scaling up to larger and larger networks. we need to start small and get big, and break this up into multiple sessions."

Each session updates its row here, and adds an entry to `LOG.md`.

## Milestones

Each node count is an estimate until its milestone checks it.

| # | Milestone | Network | Nodes | Done when | State |
|---|---|---|---|---|---|
| 0 | Set up CrakBox | None | None | The tool works from this folder, and all tests pass | Tests pass. **First action of the next session, before any agent runs:** remove the link at `research/mapping`, because the guard's folder rules may not cover it. Then run `test_core`, and confirm that a Read of `guards/sealed_guard.py` is still blocked |
| 1 | Crack the first network | spaCy word tagger | About 500 | Every node is understood, on a progress bar | Not started |
| 2 | Show that it repeats | spaCy name finder, which has its own weights (to confirm) | About 500 | With the workflow library frozen, it reaches 100% with no new workflow, at the same cost per node | Not started |
| 3 | Go larger | ResNet-18: the parked 184 nodes, then more | 184, then up to about 3,900 | Set when milestone 2 ends. It needs a lower cost per node first | Parked at 2 of 184 |
| 4 | Language model | SmolLM2-135M, already on disk | Set later | Set when milestone 3 ends | Not started |
| 5 | Cale's own model | The prototype, or the planned model if it exists by then | Set later | Set when milestone 4 ends | Not started |

**The rule for going larger.** A network moves up only after the one before it reaches 100% and the frozen library repeats. Each milestone gets its sessions when the one before it ends.

**Cost per node is a main number on every result.** The last run cost about 46k tokens per node. Control stays with the step limits and exits in the constitution, not with a token budget.

| Network | Nodes | One first read at that rate |
|---|---|---|
| Tagger | About 500 | About 23M tokens |
| All of ResNet-18 | About 3,900 | About 180M tokens |

Text may cost less than images. This is not measured yet.

## Milestone 1: sessions

| Session | Job | Ends when | State |
|---|---|---|---|
| 1 | Add text support. Rebuild the tagger so that the tools can see each node, and confirm that it gives the same output as spaCy. Write the text form of the judge's test. Report the true node count | Tests pass, output matches, and the count is reported. No agent has read a node | **Next** |
| 2 | Set the marks on wrong descriptions. Run 10 nodes through the whole loop | Wrong descriptions lose, and cost per node is measured. Show Shonu the result, and write the token prediction for the fill | |
| 3 and on | Fill the network in waves, from the outputs down. Later readers see earlier cards. Each session ends with the bar, the open nodes and their suspects | The bar reaches 100%, or a limit in the constitution stops the work | |
| Last | Shonu spot-checks a random sample. The whole-network check runs. The library freezes for milestone 2 | Shonu accepts the result | |

### Session 1 inputs

| Use | File | Rule |
|---|---|---|
| Reading: readers and critics may see it | `research/tree/data/wikitext2_train.txt`, 10 MB | |
| Sealed: only scoring code may see it | `research/tree/data/wikitext2_test.txt`, 1.3 MB | A sealed copy must sit under the guard's protection before any reader runs. A test proves this |

Session 1 also writes the goal and its random control, as the constitution requires. The tagger gets its own Python setup in this folder, listed in `DISK.md`. Free disk stays above 15 GB.

### The edge test on the tagger

Shonu did not need to decide this, because the essay answers it (2026-11-04).

A blind judge reads only a node's description. The judge writes texts that should fire the node, and look-alikes that should not. Code measures both. The score is a win, a tie or a loss per node (essay, line 641). The test stays required.

Our ResNet-18 form picked from a fixed pool of photos and added a fixed pass mark, because a judge cannot write a new photo. The tagger takes written text, so it uses the essay's own form.

One check stays. Wrong descriptions must lose. On ResNet-18, half of the wrong items also scored above zero on the edge test, so a win alone may not separate right from wrong here either. The exact win rule and its false-pass rate come from calibration in session 2, before any node is judged. If wrong descriptions win too often, stop and show Shonu the numbers.

### The whole-network check

Node tests cannot show that the parts fit together. So code puts all the claims in place and measures how far the tagger's output moves on sealed text. It stays a reported number. Shonu decides at the end of milestone 1 if it becomes required. This check is my addition, and Shonu can remove it.

### Risks. Session 1 checks the first three

| Risk | Why it matters | What to do |
|---|---|---|
| The tagger shares its lower layers with the sentence parser | Some nodes may do work for the parser and not for the tagger. A tagger-only test would read this as a failed card | Check it. Count the parser as a reader of those nodes |
| Each node has 3 linear pieces and uses the largest (maxout) | One description may need to cover 3 regimes | The constitution already lists this as a suspect. A workflow may split a node by piece |
| Words enter through hashed tables | Early nodes have no plain input name | Show evidence as words in sentences |
| Small networks may have broader neurons (essay, lines 47 and 415) | Each node may be harder to read than in a large network | The bar reports it. A gap names a suspect in the map, the tools or the test |
| 100% is stricter than the essay's own reports | The essay's author still reports open cases (lines 29 and 731), so how hard this is stays unknown | An open node stays open. Nobody lowers a test to finish. If the limits stop the work, show Shonu what was tried |

## Calls made by delegation, which Shonu can change

| Call | Reason |
|---|---|
| The folder is `CrakBox/` at the top of the repo | The plain reading of "within this one" |
| Inside names stay: `mapper.py`, `map-reader` and the others | A rename adds risk and gives nothing |
| The ResNet-18 evidence tools stay in the curves track | Only milestone 3 needs them |
| Every log entry keeps the "Cale line" and "Whole-network build" rows. CrakBox milestones do not include a resize build | The old plan put the full Cale vision next. This roadmap makes understanding the milestones, and the map feeds Cale's resize work |
