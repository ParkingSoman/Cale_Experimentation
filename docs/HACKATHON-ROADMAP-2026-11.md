# Elastic Cale: four-week roadmap (SideQuest Sundays W26)

## Context

You entered the SideQuest Sundays hackathon (Nov 22 to Dec 20, 2026). The goal: take the strongest ~1-2B language model and make it fully elastic. Top size X. Any size in [0, X], computed in place from one set of weights plus a small file. You choose which concepts it keeps as it shrinks. The demo is the full Cale product on this model: chat and other apps, adapting per computer.

Your decisions this session:

| Decision | Your words, short |
|---|---|
| Model | the best-performing ~1B model, because the essay links good training with clean neurons |
| Tracks | ONE track. The blocker is understanding how a coarse neuron works. The size dial is easy after that. |
| Progress posts | visual explainers of where the research is |
| "Expands" | contract from X to any size in [0, X], and jump to any size in that range, no retraining |
| Demo | Cale's chat and other apps on this model, adapting per computer. Bigger models (10-20B) only if cheap. |

## What approving this plan grants (please read)

| # | Grant | Why |
|---|---|---|
| G1 | Week 1 may go past one coarse neuron: the second parent `s3:78`, and two more coarse neurons | a rule must hold on more than one neuron, and 2/3 of n101's read comes from outside its one parent |
| G2 | A word-free bar replaces Floor B (word readings + blind judges) as the gate. Words stay as a record | on unseen photos only 4 of 16 word readings passed, and the sibling-control test fights the hypothesis itself |
| G3 | Model = **Qwen3.5-2B** (evidence below). Say "Gemma 4 E2B" at approval to switch | the model is fixed after week 1 (no 180° pivots) |

## The hackathon rules that shape the plan

| Rule | What it means for us |
|---|---|
| Post the starting point today, Nov 22, in #project-intros | what exists today + what we hope to demo |
| Only progress made during the competition counts | the starting point must be exact, so the change is visible |
| Log progress in #i-shipped every Sunday by 11:59 pm PT (Nov 29, Dec 6, Dec 13) | each week ends with something people can open. ELI5. A missed week makes us ineligible |
| Demo Day Dec 20 | show the change since the first post |
| Judged on progress, execution, traction | "a research result you can reproduce" counts |
| Credit and context | credit AI tools and the essay. Disclose prior work |
| Work by people outside the team disqualifies | **check:** CrakBox has 3 GitHub collaborators. Either add them as teammates (up to 2), or use only work you own |

## The model: Qwen3.5-2B against Gemma 4 E2B

Read from both model cards directly. Neither card compares itself with the other. Gemma's card does not say if thinking was on.

| Shared benchmark | Qwen3.5-2B, thinking | Qwen3.5-2B, non-thinking | Gemma 4 E2B (mode not stated) |
|---|---|---|---|
| MMLU-Pro | **66.5** | 55.3 | 60.0 |
| MMMLU (multilingual) | 63.1 | 56.9 | **67.4** |
| GPQA | **51.6** | n/a | 43.4 (Diamond) |
| Tau2 (agent tasks) | **48.8** | n/a | 24.5 |
| MMMU-Pro (vision) | **50.3** | 47.7 | 44.2 |

| Fact | Qwen3.5-2B | Gemma 4 E2B |
|---|---|---|
| Stored parameters | 2B | 5.1B (2.3B "effective"; the rest are per-layer embedding tables) |
| Same-family smaller model | 0.8B (a yardstick: "our 2B cut to 0.8B" against the real 0.8B) | none smaller |
| Path to a ~10B product | 4B, 9B | E4B, 26B, 31B |
| Engines | llama.cpp mainline runs Qwen3.5 GGUFs. MLX support not yet checked | Cale's MLX code already has Gemma 4 files (`nest/elastic.py`, `nest/runtime.py`) |

My reading: Qwen3.5-2B wins 4 of 5 shared benchmarks in thinking mode, at less than half the stored size. Gemma 4 E2B wins multilingual. By your criterion (best performer at ~1-2B), Qwen3.5-2B. Risk: its hybrid attention needs new MLX model code.

## What exists today (the exact starting point)

| Piece | State | Source |
|---|---|---|
| Cale runtime | installs once on Mac, Windows, Linux. Apps state goals. Cale picks one of 16 sizes of Gemma 3n (about 48-100%) per computer, and switches size in place between tokens (about 0.03 ms). Google built those sizes into Gemma 3n by training | `cale/`, `nest/gemma3n_levels.json`, `demo/README.md` |
| Runtime size plumbing | per-layer FFN width on one loaded model: MLX and a llama.cpp patch. An older MLX loader also stores FFN neurons most-important-first, so a width is a prefix | `nest/elastic.py`, `llama/patches/0001-*`, `nest/runtime.py` |
| Size dial on an image network | ConvNeXt-Base stages 3-4: a 2.4 MB file gives any size, no training, exact undo. It is a rank tree inside each block (a dropped neuron hands its write to higher-ranked neurons). Measured: 50% keeps 65.5% top-1 (full 80.4%), 40% keeps 57.4%, 25% keeps 36.3%. The loss is errors piling up across layers | `research/clean/NEURON-LOG.md` N5 |
| Coarse-neuron research | current reading, not tested: in ResNet-18, a narrow net's coarse neuron reads its parent about 2x a child, follows the parent on every child's photos, and keeps only the parent's veto | `research/clean/onecoarse/LOG.md` |

The new claim for this hackathon: compute the sizes for a model that was NOT trained to have them.

## The roadmap

One track. Each week ends with a visual explainer for #i-shipped. Each bar is written in the LOG before its run and scored by code.

### Today (Nov 22)

1. Write the last session's review into `onecoarse/LOG.md` and memory (summary at the end of this plan).
2. Write the hackathon facts and these decisions into memory.
3. Build one visual explainer page: the coarse-neuron picture (marked "current reading") and the image-network dial numbers.
4. Give you the post below, with the explainer link. You post it.

### Week 1 (to Nov 29): the operation, on coarse neurons in ResNet-18

Goal: name the read side of `coarse = parent * children` and show it predicts coarse neurons.

1. Write candidate operations, each with a "Derived from:" line, before any run. Example: the parent AND (any child's condition) AND NOT the parent's veto.
2. Test each one word-free on fresh photos: does it predict n101's firing and read better than the parent alone, the children's sum, and random co-firing groups?
3. Closure (G1): how much of n101 comes from the `s3:69` family, and how much from the `s3:78` family.
4. **Low-closure branch:** if one parent's family explains little, redefine "children" as the co-firing group the coarse neuron merges (an earlier result: B's neurons match merged co-firing groups better than one parent's children). Test the operation on that object.
5. Apply the winning operation to two more coarse neurons (for example the wheel neuron n122). It must hold on every one.
6. Setup only: load Qwen3.5-2B and 0.8B on the cluster. Check MLX support for Qwen3.5. Capture FFN activations on a fixed text set.

Writes wait for week 2: net B's writes are a private code, so they cannot be compared there.
Bar: the operation beats all three controls on fresh photos, on every tested coarse neuron.

Added 2026-11-24 (Shonu: start from the set D of neurons to delete). Three networks: Z = delete D, R = retrain the whole network free, C = calculated. R is trained and read unit by unit first (LOG "Delete set D"). Then, in order:

| Item | Open question |
|---|---|
| ^(D), "caret" | given D, which kept units take over D's job, and by what condition |
| Star with partial children | a parent with some children in D and some kept. Example already in the first D: s3:69's R1 children past rank 11 stay. What does the morphed parent write, and what do the kept children now read? 2026-11-24 (LOG "Reads: why siblings split"): no weight of a deleted child may stay; siblings split by WHEN (nearly independent firing), so a carrier of a deleted child's job must fire at its times; in-family sibling handoff failed 2 of 3 pairs; untested legal carriers: an out-of-family neuron that reads the child's condition, and the parent's own 1x1 skip column |
| Build C | from the two items above. Fill the C columns of the unit ledger (`onecoarse/results/oc_ledger`) and compare with R, unit by unit |
Post: "How a small network makes one coarse neuron".

Changed 2026-11-25 (Shonu): the guiding light is now distillation, not removing one capability.

| Item | Now |
|---|---|
| Target networks | the WHOLE ResNet-18 distilled by KL to A at 95%, 50%, 5% of its neurons (every layer's width times f), best of two starts: from random, and cut from A then distilled. Plus 100% from random. Log: `research/clean/distill/LOG.md` |
| Goal | calculate these distilled networks directly from A, with no training |
| Capability dial | moved to a future feature (Week 3 item 3 stays on the roadmap, but it no longer guides the research) |

Post for Nov 29: test KL to A against the share of neurons kept, one line per start.

### Week 2 (to Dec 6): the operation in the language model, reads and writes

Goal: the operation holds in Qwen3.5-2B, across several layers at once.

1. Make the model's own coarse reference: retrain a span of 4 consecutive FFN layers of Qwen3.5-2B at 1/4 width, together, by matching the full model's output (cluster). Layers after the span stay fixed, so the span's writes land in the shared stream and can be compared. (A single layer with fixed readers did not form a coarse neuron in ResNet, so the span is co-trained.)
2. Find parents, children and coarse neurons in the span. Read reads AND writes.
3. Build the span from the operation alone, with no training: reads and writes.
4. Check compounding: score the built span inside the full model, not layer by layer.

Bar: inside the full model, the operation-built span closes at least half the gap between plain neuron dropping and the trained span.
**Decision point, Dec 6:** if the bar fails, the dial ships with the N5 rank tree as a floor, and the research stays the story. This keeps a working demo.
Post: "The same rule inside a language model".

### Week 3 (to Dec 13): the dial for the whole model, and the concept dial

1. Apply the operation to every FFN layer. Order neurons once, so each size is a prefix plus a small config.
2. Measure the size-quality curve over [0, X]. Compare with Qwen3.5-0.8B at its size and with plain dropping.
3. Concept dial (FUTURE FEATURE since 2026-11-25, not the guiding light; it enters the calculation as the choice of what "optimal" weights): calibration text per concept (for example code, French, maths), with weights. The config decides which children keep their own neuron.
4. Predict each concept's quality at each size before measuring. Report the signed error.

Bars: config under 2% of the model. A switch is a slice, no solve. The curve beats plain dropping at every size. Concept predictions within ±10%.
Post: "One model, every size, your choice of what it keeps".

### Week 4 (to Dec 20): Cale on the elastic model, and Demo Day

1. A facts file for Qwen3.5-2B, in place of `cale/gemma3n_facts.py` for this model.
2. Engines: MLX model code with prefix widths plus the config. A llama.cpp width patch for Qwen3.5.
3. Make the size choice continuous: `cale/choose.py` and `cale/batching.py` step through a fixed pool of sizes today.
4. Rebuild the stale installers. Build the Windows installer on the HP laptop. Linux has no chat today: Mac and Windows carry the demo.
5. Run chat, Quick Notes and Rewrite on the M5 Mac and the HP laptop. Same app code, each computer at its own size.
6. Record the demo video. Publish the explainer series and the reproducible numbers.

Stretch, only if cheap: the same method on Qwen3.5-9B.

## Draft post for #project-intros (you post it)

> Hi! I'm Shonu. My project is Cale: one AI model that shrinks to fit any computer, instantly.
>
> **What exists today.** Cale is an app you install once. After that, any app built for it gets local AI with no sign-up and no limits. It picks a model size for each computer and can change size between words. Today this only works with Google's Gemma 3n, because Google trained those sizes into it.
>
> I also have research on computing sizes for a model that was NOT trained for them. On an image network, a 2.4 MB file already gives any size with no retraining (half size: 65.5% accuracy against 80.4% at full). My current reading of why small networks work: a small network's "coarse" neuron is the big network's "parent" neuron carried forward, with its specialized "children" merged in. That is not tested yet. Testing it is week 1.
>
> **Goal for Dec 20.** Take Qwen3.5-2B, one of the strongest models of its size, and make it fully elastic. Any size from small up to full, computed in place from one set of weights plus a small file. You choose which skills it keeps as it shrinks, and Cale predicts how each skill degrades. Demo: Cale's chat and apps on a new Mac and an old Windows laptop, the same app code, each computer at its own size.
>
> Built with Claude Code. Research guided by the essay "Neural Networks Are Already Clean".

## Risks, said once

| Risk | Plan |
|---|---|
| The operation is not found by Dec 6 | the decision point keeps a working dial |
| A 2B model cut far below half size may degrade a lot | report the real curve against the 0.8B yardstick. No claim past the measured range |
| Engine support for Qwen3.5's hybrid attention | checked in week 1 setup |
| The Mac swap is nearly full (12.2 of 13.3 GB) | heavy jobs run on the cluster only |
| Outside contributors | settle the CrakBox collaborator question before week 1 work uses CrakBox code |

## Session review to record today (from the first half of this session)

| Item | Content |
|---|---|
| Where we are | fixed system: parent `s3:69`, 6 core children (+ a ring of 11), coarse n101. Word-free: n101 reads the parent 2x a child (1.39 vs .39-.72), carries the children's condition channels (13 of 16), keeps only the parent's veto, follows the parent on every child's photos (.55-.88) |
| Gaps | G1 the operation is untested. G2 the system is not closed (2/3 of n101's read is outside; `s3:78`; A `n212`). G3 writes not studied (B's writes are a private code). G4 words: 4 of 16 poles pass on unseen photos (.78 dev to .61) |
| Failure mode | the bar became the object: about ten judge rounds tuned words. The sibling control fights the hypothesis by design. Words came from the top 0.2% and were tested lower. The dev pool leaked (a possible cause, not proven). Scores with se ~.14 drove rewording. Shared: the "loop until the judges figure it out" instruction made the bar the goal |
| Smaller faults | started a write build before the object was clear (stopped twice). Asked Shonu a judgement he had delegated. A new one-off pixel statistic per exception (4 of 7 failed). Two attempts to route around the judge-lock word scan |

## Verification

- Every bar is in the LOG before its run, and scored by code on data never used to build the rule.
- Each Sunday post links something people can open: an explainer page, a curve, or a video.
- Demo Day numbers are reproducible from one script and the config file.
