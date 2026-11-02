---
name: feedback-research-method
description: "How to run research: session goal (map), smallest test, fix the step, priors+log, holdout keeper, flexible-vs-fixed check, vision first, specific first, work autonomously"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2ba42140-21a0-426d-8405-942d9986529c
  modified: 2026-10-21T17:58:40.748Z
---

Research-method rules from Shonu, merged 2026-10-21. Essay and interp-tool rules are in [[feedback-research-essay-rules]].

## Session goal 2026-10-21: build the map, do not stop
- Build an accurate coarse-to-fine map of ConvNeXt-Tiny. Inside that goal, do not stop to ask for decisions. Keep going until the studied neurons are deeply understood.
- **Why:** Shonu said "the overall goal right now is just mapping the network ... don't stop until you have an amazing understanding of them." State file: [[project-map-track]].

## Work autonomously in chunks (2026-10-20)
- "Don't ask for approval before doing a good chunk of work since I'm going to be away for a while", and "you should look into the essay when you hit issues."
- **Why:** he is often away; stopping at every fork wastes hours. He trusts the discipline (pre-registered rules, keeper, log).
- **How to apply:** chain planned steps (dev check -> diagnose -> essay reading -> one fix -> rerun -> keeper query with a written prediction). Stop only for truly his decisions (spending money, product direction, deleting things). Report once at the end.

## Smallest test first (2026-10-20)
- Default to the SMALLEST experiment that can test the hypothesis: fewest layers, parameters, arms, widths, calibration blocks or minutes that still give a clear answer (with a fair control). Single-layer ~20 s tests are the normal unit. Long runs come later, on a rented GPU. Treat a weak result in a regime (e.g. RCF weaker at wide widths) as a likely bug unless there is evidence it is natural.
- **Why:** I launched a 4.5 h six-arm whole-model run while two problems were open. Shonu stopped it: "in general i don't want you to default to long runs. that is how we kill speed, since you work very fast and incredibly well with these targeted experiments."
- **How to apply:** ask "what is the smallest version that answers this?" State the run time before anything over ~10 min, and prefer to shrink it. Whole-model or many-arm confirmations wait until the design is settled. Wait-loop trap: never wait on `pgrep -f <script name>` when another shell's text holds that name; wait on a saved PID (`$!`) (2026-10-20: a test waited forever). See [[project-cale-business-ideas]] (cloud GPU), [[lesson-rules]] L139.

## Fix the step, do not escalate (2026-10-19)
- When a step fails, find what is wrong in THAT step and fix it. Do not propose a larger run (more training, bigger model, from-scratch retrain) next. Escalation repeats the same mistake bigger.
- **Why:** Shonu on the coarse-neuron reproduction: "the default thing if a step fails would be to fix that step, not to make an even larger version of the same mistake." He wants fast iteration, forcing a result from the cheapest step.
- **How to apply:** in plans list "fixes inside this step" before any "next, bigger step". "We must find something here." Escalate only after naming why the step cannot answer the question, and ask first.

## Persist with priors and a running log (2026-10-19)
- Coarse-children search: keep `research/clean/CHILDREN-LOG.md` (priors table, guards, axis catalog, log). Read it first each session; update after every test.
- Work under strong stated priors (every narrower unit is a coarse parent in EVERY layer; a weak layer means a wrong axis or test, not "no children"). Finding children likely needs per-layer mapping, not one generalist method; look for field tools that make it easier.
- **Why:** he does not want a continuous failure mode of "looking at the wrong children or in the wrong direction". But priors are not settled: "be careful", do not overfit, science means proving yourself wrong.
- **How to apply:** each prior lists what would count against it. Use nulls, fix thresholds before looking, confirm on held-out text, log failures.

## Split by research question (2026-10-18/19)
- Round 1 held too much because stages were split by funding. One concrete thing per step, with one go/stop gate each. Ask "the single most important thing" to learn per stage. Explore for $0 first (interpretability, Mac-only, no training) before any rented GPU.
- **How to apply:** think seriously about "could it be free" ideas he raises, with reasoning in the chat, not only in a file.

## Holdout keeper (2026-10-19)
- He does not want a method that works "by happenstance". One keeper agent holds the held-out data; the implementer sees coarse feedback only. Many tries on the same eval blocks (ZOOM v0-v15 on dev blocks) select for noise.
- **How to apply:** protocol in `~/.claude/plans/ok-id-like-you-unified-blum.md`: keeper data in `~/Cale-holdout/` (implementer never reads it), new layers + new text blocks as held-out, pre-registered tests, coarse feedback buckets, query budget, final test once per method family. Audit the transcript for reads of the folder. Before a keeper query require effects on >= 3 dev layers (see [[project-research-history]]).

## Flexible vs fixed check (2026-10-20), standing
- Frequently compare whatever flexible method we use (one procedure that refits the trained network to ANY size) against a FIXED reference that spends resources to make that size good, at the same storage, across many circumstances (sizes, layers, text types). The test may be edited as methods change.
- Flexible MATCHES or is close to fixed -> very good: the method can serve every tier. Flexible BEATS fixed -> question the fixed reference first (under-trained? too short? wrong objective?) and strengthen it before trusting the win (2026-10-20 Q8: a 2-min trained layer lost to RCF everywhere, so it was not a ceiling). Fixed beats flexible -> gains left on the table; use ladder rungs (more freedom at equal storage) to find which part of the method misses them.
- **Why:** after the V2 ladder found RCF's fold class too narrow, Shonu: "we want to frequently touch base with whatever method we are using. How does it compare to if I spent resources to make that specific size good?" It answers "how do we know we capture the gains we can?".
- **How to apply:** rerun a small ladder/reference comparison after any method change; report flexible-vs-fixed in every results summary. Living doc: research/clean/FLEXIBLE-VS-FIXED.md.

## Vision first, good models only (2026-10-21)
- "We should now start testing with vision models first since it'll help a lot with giving you a second thing to look at during iteration and it will be easier to find bugs." And: "ensure we aren't wasting time by testing on a badly trained model."
- **Why:** our CIFAR nets (44% accuracy) had no class-level neurons; SmolLM2-135M / Pythia-160M are small models whose neurons the essay calls broad and messy. Null results there may be the model, not the method.
- **How to apply:** per mechanism: (1) reproduce in a good pretrained vision model (ImageNet ConvNeXt, the essay's architecture; look at images), (2) then run the same single-layer test on a well-trained language model. No verdicts from weak models.

## Specific first, several neurons (2026-10-21)
- Study a few specific neurons closely first, spread across depth, then derive a general method. Not whole-network rulers first, and not n=1.
- **Why:** Shonu stopped whole-network containment and edge maps: "you're moving in reverse; the powerful method was hyper-specific first, then a method that generalizes, using the vision model's ease of interpretability." Then stopped a single-neuron trace: "n=1 is not a good sample size ... pause and propose your next direction."
- **How to apply:** on a direction change, pause and propose the plan (sample, protocol, how hand findings become a method) before running. Use vision to look and to test with image edits. (The session-goal rule above says: inside the map goal, keep going.)

Related: [[project-research-history]], [[project-map-track]].
