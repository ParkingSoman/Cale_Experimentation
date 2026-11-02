# Curves track: math mode (read this first)

The goal: a mathematical argument, from facts known for sure, for how to derive an optimal smaller network from a
larger one. It should generalize across networks. InceptionV1 curves are the first example, not the subject.
The fixed form of that goal is the trunk T5 in `DECISIONS.md`. Every session serves the trunk.
Paper: `COARSE-THEORY.md`. `NEURON-MATH.md`, `PAPER.md` and `CURVES-LOG.md` are the record. Cite them as log, never as
proof. Shonu's reasons for these rules: memory note `feedback-math-method`.

## What the trunk gives (go through this BEFORE stating any theorem)

Shonu, 2026-11-01: "you're doing these theorems and you're not considering all the things that the trunk gives us ... and
then you realize, oh, a lot of things actually fit". A LADDER section states which lines it used ("Trunk check").

| Trunk words | What it gives | Question to ask |
|---|---|---|
| distillation loss D_P | teacher outputs on P, one fixed goal | Am I aiming at anything else? |
| P = inputs N was trained on | stationarity (Fact F8), labels, N's own objective | Does this premise follow from training? |
| a map of claims, found by people, agents or a program | agents find it with judgment; code only computes each claim's error; we may define claim types | Which claim would make this exact? Am I writing a map algorithm? (If yes, stop.) |
| complete for the zoom | we define it; a map can be made complete | Am I trying to prove more than the promise? |
| graded clause | real nets need only a bound in the errors | Is the exact case the one I need here? |
| nested zoom | a tree, the chain rule (TI0), reuse across levels | Can a level reuse the level below? |
| one-time computation | any cost, within modest hardware (R-mixclean (e)) | Is hardness really a wall here? |
| "new units that read only that group's units" | the build must be a real network | Is my number the loss of a real network of that size? |
| the essay | clean mechanisms, coarser never fractured, no groups of nouns | Is my object the network's own, or mine? |

## How results reach Shonu

| Item | Rule | Guard |
|---|---|---|
| Explanations (chat, LADDER, cards) | Real ML objects. A code-block diagram per step, then one "Why it matters" line. Each new word once, in a table. One analogy at most. A paper label never stands alone. Good examples: LADDER 30, `~/.claude/plans/id-like-you-to-tingly-harp.md` | `tools/ladder_lint.py` |
| Technical nodes | Mine. Do the math, record "DECIDED by delegation" with his quote and my reason, tell him in one line | |
| Directional forks | What the product promises, a narrowing left after the general attempt failed, or a theorem showing two parts of the trunk conflict. A card with a "Directional because" line | `tools/ladder_lint.py` |
| Anything the trunk settles | Do not ask | |
| Maps | Subagents read the network the essay's way and propose claims. Code computes each error. Never one algorithm for all networks (Shonu, 2026-11-01) | |

## The bar: optimality (Shonu, 2026-10-31; read before calling anything done)

Shonu: "this cannot ever be classified as 'done' or 'well executed' if we haven't confronted the optimality question.
for now, we are keeping the bar at ensuring we have found optimality with the bounds dependent on network
cleanliness."

The object is V(G): the least distillation loss E_P KL(softmax(y_N) || softmax(y_B)) over every allowed small network
B. That is what well-executed distillation converges to. No approximation stands in for it.

| Label | Meaning | Meets the bar? |
|---|---|---|
| OPT-KL | gives V(G), or a build with loss V(G) + gap, gap bounded by the claims' errors only, 0 when clean, and when every claim is 0 the build reaches V(G) even where V(G) > 0 | Yes |
| EQ | proves two least losses are equal | Yes |
| OPT-D2 | the least of another loss (squared centred logits, node R-obj) | No |
| ONE-BUILD | an upper bound on one rule's loss, exact only at loss 0 | No |

1. Every result in the paper, LADDER and chat carries one label.
2. A session, a Part or a track is not "done" while its main result is OPT-D2 or ONE-BUILD. Say so in plain words.
3. A move off OPT-KL is shown to Shonu in chat and LADDER, never only in the paper.
4. A blocked proof becomes a conjecture with a Route and a card, never a weaker rule called the optimum.

## The roadmap and Shonu's standing rules (2026-10-31)

Read `ROADMAP.md` first, every session. It holds the four sessions to the ResNet goal and their status.
Shonu's standing rules (his words in chat, 2026-10-31), on top of the bar:

| Rule | Meaning |
|---|---|
| "exact is really exact" | an exact theorem is checked against a certified global optimum on small nets, hand-built and trained |
| "graded is actually useful" | a bound counts only if first-order tight (bound / true gap stays bounded as errors -> 0); measured on trained nets |
| "well trained networks are already clean" | each new premise gets a realism check on a trained network in the session it enters |
| "no tradeoffs I haven't chosen" | every narrowing or change to "exact" goes on a card before the proof that uses it |
| "if it is not possible, show it mathematically" | a "not possible" needs a theorem on a clean, trained-like instance; a counterexample is first read as a network |

## The paper must stand without its measurements

| Part | What it may hold |
|---|---|
| A | FACTS (true of every network of this kind, with a proof or source), DEFINITIONS (interpretability objects as math objects), HYPOTHESES (named, for example H-clean) |
| B | Lemmas and theorems with full proofs. Each says which hypotheses it uses: "(under: none)" or "(under: H-clean, H-sym)" |
| C | The only place for numbers: the distance eps of InceptionV1 from each hypothesis |
| D | Where the math leads: conjectures, each with a "Route:" |
| Appendix | Triage of old results, referee reports |

## Rules

1. Start only from what is known for sure. "Clean" and every essay concept enter as named hypotheses.
2. Every Theorem, Lemma, Proposition and Corollary has a hypothesis tag, a Proof (or is a CONJECTURE with a Route),
   and a "Lean:" line. Lean is the default proof tool (`lean/coarse/Coarse/Model.lean`). Where Lean is not
   practical, prove the finite version in Lean and say why. Use sympy for identities.
3. Order of work: state, then prove, then check the proof on the exact synthetic instance (`model/`), then (only
   then) measure one eps on InceptionV1.
4. A failed proof or check goes back to the paper. Find the false step and fix the statement or the hypothesis.
   Never run a chain of probes. Two runs in a row without a new paper item = stop and write.
5. Literature enters only through the compatibility table in Part A (between the
   `<!-- literature-table -->` markers). Framings that split clean families into sparse directions (SAEs,
   transcoders, dictionary learning, superposition as a default) do not enter the chain of logic.
6. Interpretability objects (feature, family, clean, edge, circuit, coarse network) are defined as math objects.
   That is the main lever. A new idea gets a definition before it gets an experiment.
7. The trunk is fixed (`DECISIONS.md` section 1). Every restriction away from it is a node in `DECISIONS.md`.
   A new restriction (a hypothesis, a premise in a tag, a narrower class in a definition) needs a node BEFORE the proof
   that uses it. A technical node I decide by delegation, with his 2026-11-01 quote and my reason (see "How results reach
   Shonu"). A directional node is PENDING and gets a plain card, and only Shonu moves it to DECIDED, in his own words in
   the chat. Results under a PENDING or RETRO node are side branches: they stay as mathematics, but do not enter the D.1
   chain as steps toward the trunk.
8. Work from the general form. The object is the optimum itself (Part B.S). Folds, gates, families and other
   interpretability ideas are hypotheses about the optimum, never methods we must use (Shonu, 2026-10-26: "I don't
   ever want us to like say, hey, we have to do this method just because"). Ask first: what is the general form,
   and are we working generally? The essay tells us how real networks are. In this track it is the source for the
   definition of clean, not a source of methods.
   When a proof is blocked, the next step is a card for Shonu, not a new restriction.

## Guards (they run without being asked; the switch is the file `.math-mode` in this folder)

| Guard | File | Event |
|---|---|---|
| Paper linter | `tools/paper_lint.py` | After every Edit or Write to `COARSE-THEORY.md` |
| Run gate | `tools/run_gate.py` | Before every Bash call. A script runs only with a `# serves: <ID>` line naming a paper item. InceptionV1 scripts must serve an `H-` item |
| Turn reminder | `tools/turn_reminder.sh` | Every prompt |
| Stop gate | `tools/stop_gate.sh` | End of every turn: lint plus `sorry` check |
| Referee | `.claude/agents/math-referee.md` | Call at the end of each Part. Paste its report into the Appendix |
| Ledger | `tools/ledger.py` (via the paper linter) | Every tag phrase, item and node agrees with `DECISIONS.md`; the trunk text is locked by `tools/trunk.sha256`. `tools/ledger.py --report` prints each result's distance from the trunk |
| Self-test | `tools/test_guards.sh` | Run after any change to a guard |

Remove `.math-mode` to switch the guards off (for example, for Cale product work).
