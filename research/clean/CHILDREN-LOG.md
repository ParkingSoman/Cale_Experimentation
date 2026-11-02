# Children search log (running)

Goal: show that each unit of a narrower trained layer is a coarse version of a set of distinct units of the
wider layer (the essay's finding), in EVERY layer. Then find cheap ways to build such parents.

Read this file at the start of every session on this track. Add to it after every test.

## Working priors

These are assumptions we work under on purpose. They are not settled. Each one says what would count against it.

| # | Prior | Status | What would count against it |
|---|---|---|---|
| P1 | Every unit of a narrower trained layer is a coarse parent of distinct wider-layer units, in every layer. A weak layer means a wrong axis or a wrong test, not "no children". | Assumed (Shonu, essay) | Every axis in the catalog below fails at one layer, each with nulls, AND the trained layer clearly beats the cut there |
| P2 | The right axis depends on depth. Early layers: where a unit fires (input). Late layers: what it writes (output). Middle layers: both, or what later layers read. | Assumed. Literature support to check: MLP keys/values (Geva 2021), output-centric descriptions | Late layers stay weak on the output axis too |
| P3 | A child can be either pole of an old neuron. A SwiGLU neuron has two poles. The essay says siblings sit on each other's negative poles. | Assumed | Children almost never come from negative poles |
| P4 | Coarse means OR-like coverage: the parent fires wherever a child fires. It does not mean a weighted sum. Linear R² is not the test. | Adopted 2026-10-19 | none needed (definition) |
| P5 | The first position of a block (attention sink) and other huge-norm tokens must be excluded | Verified 2026-10-19: they faked R² = 1 at layer 28 | none |
| P6 | Frozen-rest training is enough to make coarse units. If one layer fails on every axis, suspect the training objective within this step (e.g. local MSE favors compression) before any bigger run. | Assumed | Global (KL) training changes nothing on any axis |

## Guards against overfitting

1. Every number comes with two nulls: untrained random neurons, and old neurons treated as parents.
2. Fix thresholds before looking. Current: child top-50, child if parent percentile >= 0.95, "hot" = child's top 1%, parent top-100.
3. Find children on one text set. Confirm on another set, or on held-out text.
4. Change one thing at a time. Write down every failed attempt here.
5. Every claim names its result file.
6. Once per session, try to break one prior on purpose.

## Axis catalog

| Axis | How a unit is profiled | Tool | Layers tested | Result (trained / random null / old-as-parent) | File |
|---|---|---|---|---|---|
| A1 Linear sum of activations | Sparse regression of the parent's activation on old activations | `r1.py analyze` | 3, 15, 28 | Dense mixes: 10 old neurons explain 41-66% (null 3-13%). Not the coarse test (P4). | `results/clean-r1-analyze.json` |
| A2 Input firing coverage | Text positions where a unit fires (top-50), both poles | `r1cover.py` | 3, 15, 28 (K=10, 48) | K=10 units with 2+ children: L3 0.80 / 0.30 / 0.42. L15 0.55 / 0.10 / 0.17. L28 0.45 / 0.00 / 0.10. Strong at L3, weaker with depth. | `results/clean-r1cover.json` |
| A3 Output steering coverage | Change in next-word log-probs when the unit's output is added (steering) | `r1out.py` | 28 (3, 15 running) | L28 K=10 poles with 2+ children: 0.60 / 0.05 / 0.05 (input axis gave 0.45). Up-cover 0.52 (input 0.35). | `results/clean-r1out.json` |
| A4 What later layers read (downstream effect) | Effect of the unit's output on later MLP / attention inputs. Field practice for middle layers (R0c). | not built. Next axis for L15. | | | |
| A7 Combined | Per unit: best pole, best axis. Same rule for nulls. | `r1combine.py` | | pending | `results/clean-r1combine.json` |
| A5 Ablation effect profile | Per-token loss change when the unit is zeroed | `r1read.py` | 3, 15, 28 | Parent-child ablation correlation 0.0-0.3 at L15 (single test) | `results/clean-r1read-*.json` |
| A6 Feature basis | Old units as SAE features (EleutherAI sae-SmolLM2-135M-64x, MLP layers 0,3,...,27) | not built | | | |

## Open issues in finding children

- DEFERRED (Shonu, 2026-10-19): check the impact on REASONING. Every result here is average next-word loss. Do it
  once the zoom idea works: copy/induction test, per-token-type loss (r3.py), LAMBADA / ARC-Easy / GSM8K subsets
  on SmolLM2-1.7B.

- Unexplained units (K=48): L15 27%, L28 38%. Next axes to try: a working downstream readout (per position, not
  mean over positions), and the blind test on L28 with the corrected output axis.
- Hub items (tokens or later neurons that react to everything) fake matches on output-type axes. Always scale items.
- A2 at L3: for most parents ONE dominant old neuron explains about 80% of the parent's top tokens. The other children show only in the downward check. Is the parent "dominant child + broadened", or a true OR of equals? Check the children's share of the parent's top tokens one by one.
- The reading files list too few children (cover order stops at the dominant one). List the downward-covered children too.
- The descriptions (explain + blind detection score) are not run yet.

## What the literature says (R0c, `research/clean/R0c-mapping-axes.md`)

- The essay gives no child-finding recipe. It showed coarse neurons with top natural images (input axis, vision). An SAE on the crushed layer could not pull the original neurons back out.
- The depth pattern (shallow patterns early, output-defined units late) has support (Geva 2021, Gurnee 2024), but it is a tendency.
- Gur-Arieh 2025: input-only descriptions miss output effects. Combining both works best.
- Middle layers: the field uses the downstream effect (transcoders, attribution graphs).
- Feature hedging: narrow dictionaries blend features. That is close to our 10-neuron case. Watch for it.
- R0c's caution: the literature does not promise strong children in every layer. Under P1 we count that only after every axis in the catalog fails at a layer, with nulls.

## HOW search (parent from children, no training run)

Shonu (2026-10-19): search many ways to build a parent from its children. Every few rounds, a "blue sky" review
names the error under the failed attempts, and the next round aims at it. Also test his hypothesis: one set of
children may allow SEVERAL valid parents, each better for a different task (memory: project-multiple-parents).

Families: verified ones (global layers L15/L28, K=48, coverage + hub filter). Code: `r2lib.py`, `r2round*.py`.

| Round | Aimed at | Rules | Result | File |
|---|---|---|---|---|
| R2.0 | first try | M1 rules H0-H6 vs trained parent | act match L28 H6 0.58, H0 0.42. SPAN bound 0.63 | `clean-r2how-*.json` |
| R2.1 | E1 gate direction | data gates: PROTO, PROBE, LOGIT (union of children's top tokens) | No gain: act 0.40, children-output error 0.88-0.90 vs H6 0.75 | `clean-r2round1-L28.json` |
| R2.2 | E4 target, E5 metric | Fisher-metric value/output (H0F, H6F, PROBEF), swap test on loss | All rules recover 0-5% of a trained unit's value (= deleting it). Zeroing one unit costs only +0.004. | `clean-r2round2-L28.json` |
| R2.3 | E6/E7 whole layer | All 48 parents by one rule (what = from trained layer), joint LS output refit, held-out CE (base 3.018) | CUT +0.160, KIDS-H0 +0.163, KIDS-H6 +0.164, KIDS-AVG +0.207, KIDS-PROBE +0.362. TRAINED gates+values with LS refit +0.126. TRAINED +0.093. No children rule beats the cut. Gap = gates/values 0.034 + outputs 0.033. | `clean-r2round3-L28.json` |
| R2.4 | E8/E9 loss-aware local fit | per-token loss gradients from the original model, fit only the layer (seconds-minutes) | FAILS. Cut start: plain fit prose +0.134, gradient-weighted +0.148. Started from the TRAINED layer, both local fits make it worse (+0.093 -> +0.15-0.17). No one-layer objective knows what the trained layer knows. | `clean-r2round4-L28.json` |
| R2.5 | blue sky #5: lookahead | short-tail distillation: cache h_L once, fit with KL through the layers AFTER L only | 30 s from cut: prose +0.114 (cut +0.160, global 20 min +0.093). About 7x more steps/s than global. | `clean-r2round5-L28.json` |
| R2.5b | MULTIPLE PARENTS | same start (cut), 3 min each, text = prose / 25 / 50 / 75% code / code | prose vs code loss above the original: prose-fit +0.110 / +0.208, 25% +0.117 / +0.097, 50% +0.124 / +0.083, 75% +0.127 / +0.076, code-fit +0.156 / +0.072. One knob moves the layer along a smooth trade-off. Same unit, prose-fit vs code-fit: firing corr 0.79 on prose, 0.69 on code (different units 0.03), children shared 50% (different units 0%). Same family, different parent. Weakness: units stay near their start neuron (median 1 child, global 2+). | `clean-r2round5-L28.json`, `clean-r1cover.json` (tags tail_*) |
| R2.6 | blue sky #6: start point | 5 starts, 1-min short-tail fit each (RANDOM did not run: script bug) | start -> after: CUT +0.160 -> +0.114, H6 +0.164 -> +0.115, H0 +0.163 -> +0.115, CLUSTER-AVG (no trained layer) +0.180 -> +0.120, AVG +0.207 -> +0.134. Merges do not help as a start. | `clean-r2round6-L28.json` |
| R2.7 | blue sky #7: cost at depth | L15, lookahead window 1 / 3 / 7 / end, 2 min each | cut +0.051, global (20 min) +0.033. w1 +0.042 (4297 steps), w3 +0.038 (1106), w7 +0.039 (590), end-KL +0.041 (182), control end-MSE +0.041 (244). Objective type does not matter. Steps do. A 3-layer window is best at fixed time: cost scales with the window, not the depth. | `clean-r2round7-L15-main.json`, `-endmse.json` |
| R2.8 | advisor: hybrid | frozen gates (D: gate+value frozen, UD: gate frozen), 1 min, L28 | CUT D +0.126, UD +0.116 (full +0.114). KIDS-H6 D +0.123, UD +0.116. RANDOM full +0.174. Frozen old-neuron gates cost almost nothing. Children-built gates = cut gates. The start matters vs random. | `clean-r2round8-L28.json` |

### Blue sky #8 / summary after R2.8 (reported to Shonu)
- No weights-only rule builds a parent from its children (8 rounds). In the neuron basis the trained parent is
  coarse in function but dense in weights.
- What works: old-neuron gates (cut is enough) + a short task-aware fit of values/outputs with a ~3-layer lookahead.
  Recovers ~50-70% of the cut-to-global gap in 1-3 min. The text mix is a smooth developer knob.
- Open questions asked: is a minutes-long local fit "not expensive"? Try a feature basis (SAE) for zero-training
  parents? Does coarseness itself matter, or only the trade-off?

### Blue sky #1 (after M1 and R2.0)
- E1: rules mix the children's gate weights. One SwiGLU gate is one half-space and may not reach the union of the
  children's regions. The trained parent may read a shared context direction that no child's gate owns.
- E2: scoring by "match the trained parent" assumes one right answer. If many parents are valid, score function.
- E3: child lists (<= 6 by percentile) are incomplete.

### Blue sky #2 (after R2.1: E1 was not the bottleneck)
- E4: every rule fits the children's SUMMED output in full detail. One neuron writes one direction, the children
  write several. Losing that detail is what "coarse" means, so the target is wrong. The trained parent keeps the
  part that matters for the loss. Fix: fit under the empirical Fisher metric of the layer output (loss-aware
  merging practice). One gradient pass, no training.
- E5: neither "match the trained parent" nor "children-output error" measures the model's loss. Fix: swap test.

### Blue sky #3 (after R2.2: no rule beats deleting the unit)
- E6: testing one parent at a time inside a co-trained layer is biased. Families share children, and the trained
  layer split each old neuron's job across several parents. A rebuilt parent duplicates work its neighbours do.
- E7: so judge whole layers: all parents by one rule, joint output refit, held-out loss.

### Blue sky #4 (after R2.3: children rules = cut at the layer level)
- Trained parents are coarse in FUNCTION (they cover their children) but DENSE in WEIGHTS (SPAN 0.52-0.63, gate
  weights not sparse in old gates). So children should define the target behaviour, not the building blocks.
- E8: every fit so far copies the old layer's output with all tokens and directions weighted equally. The trained
  layer optimizes the model's loss. Fix: loss-aware local fit with per-token gradients (first-order loss change),
  one gradient pass, then fit only the layer.
- E9: the LS output refit ignores which tokens matter. Same fix.
- Bonus: the gradient weighting depends on the text, so it gives different parents for prose and code from the
  same start. That is a direct test of the multiple-parents hypothesis (R2.5).

### Blue sky #5 (after R2.4: local objectives move AWAY from the trained layer)
- The trained layer's 0.04 advantage over any local fit comes from how its output is USED downstream. A one-layer
  objective cannot see that. Layer-wise compression handles it with a lookahead. At L28 only 2 layers follow, so
  fit with the true KL while running only the tail.
- Reframe: if the right objective makes coarse parents appear by themselves, then parents are defined by the
  OBJECTIVE (text + loss), not by the children's weights. "Several valid parents per child set" = different
  objectives. A developer's control = choosing and weighting objectives.

### Blue sky #6 (after R2.5b: fitted units stay near their start neuron)
- Short fits from the cut keep each unit close to one old neuron (median 1 child vs 2+ for global). Maybe a merged
  start (children-based) gives coarser, better parents in the same time. Tested in R2.6: no.

### Blue sky #7 (after R2.6: merges do not help even as a start)
- Six rounds agree: a coarse parent is not computed from the children's weights. It appears when the layer is fitted
  with the right objective. Children describe WHAT a parent covers (reading, checking). The HOW is a short fit with
  the right objective. Open engineering question: does the short fit stay cheap at depth (long tails)? Standard
  answer: a lookahead window (match the residual a few layers later). R2.7 measures how much window is needed.

## ZOOM-0/1: smallest test of a zoom configuration (Shonu: "biased prior it works, get it done, independent eval")

One layer (L28, 1536 neurons), K = 24 parents, one dial, no refit at any setting. Fast loop (~1-5 min/attempt):
cached inputs, loss through the 2 layers after L28 only, 20 held-out blocks (test 0-19) for iteration.
Final independent check later: unseen test blocks 60-99, full-model loss, code text, reviewer agent.
Criteria fixed before the runs: 1 steady (strict tolerance 1e-4; a practical 1e-3 version was ADDED after v7 and
is reported next to it), 2 beats a same-size cut with LS refit at 24/48/96/192/384/768, 3 full = original,
4 within 25% of a layer fitted per size. Code `zoom0.py`, `zoom1.py`. Results `clean-zoom0.json`, `clean-zoom1.json`.

| v | change (aimed at) | dial vs cut at 24/48/96/192/384/768 | criteria |
|---|---|---|---|
| 0 | families by |corr|, whole families open, static order | .126/.126/.119/.103/.093/.067 vs .175/.175/.139/.117/.093/.054 | 1 no (tiny rise), 2 no, 3 yes |
| 1 | nested fit under random openings (parents were tuned for one state) | .124/.124/.117/.108/.091/.061 | 1 yes, 2 no (768), 3 yes |
| 2 | families open one child at a time (whole-family steps waste neurons) | .281/.185/.152/.131/.105/.078 | worse: parent double-counts |
| 3 | parent FADES by family importance share still off | .159/.173/.149/.127/.095/.056 | 2 no |
| 4 | fade by family OUTPUT share still missing (RESID table) | .152/.154/.130/.111/.085/.053 vs cut .175/.148/.131/.118/.082/.050 | 1 yes, 2 no (48/384/768 by <= .007) |
| 5 | + faded output correction dD for originals (beta = share off, 0 at full) | .126/.136/.119/.101/.073/.045 | 1 no (24->32 +.006), **2 yes**, 3 yes |
| 6 | + 70% of nested steps on real dial states | .133/.136/.115/.099/.072/.044 | 1 no (48 bump), 2 yes, 3 yes |
| 7 | + gap-first return order | .134/.128/.113/.094/.076/.052 | 1 no (rises ~.0003), 2 no (768), 3 yes |
| 8 | nested 300 s, both orders | imp: .131/.132/.113/.097/.070/.042. gap: .133/.127/.112/.094/.074/.049 | both pass 1-3, BUT the steady sweep was every 64 neurons and missed a rise (imp 24->48). Sweep tightened to every 16 + test sizes. |
| 9 | gap, fine sweep, criterion 4 (per-size fit, same 330 s) | .134/.127/.113/.096/.075/.051; fitted .126/.119/.103/.092/.063/.043 | **4 yes** (first time), 3 yes, 2 no (768 by .0007; run-to-run spread ~.002), 1 no (max rise .0015) |
| 10A | + monotone fade table | .../.050 at 768 vs cut .050 | 2 yes (barely), 1 no (rises .0013 at 688/720/1120) |
| 10B | + 2nd correction scaled by own-family fade | .128/.125/.111/.095/.072/.049 | 2 yes, 3 yes, 1 no (max rise .0011 at 24->32, others <= .0005) |
| diag | rises happen where a big family's parent must stand in for an arbitrary missing subset of its children (one output direction cannot) | e.g. family 16 (114 kids): alpha .234 at 98 kids, then 0 | fix: more, smaller families (K = 48) |
| 11 | v10B with K = 48 | .116/.108/.091/.071/.046 at 48..768 vs cut .148/.131/.118/.082/.050 | 2 yes, 3 yes, 1 practical yes (max rise .0007), 1 strict no |
| 11 FINAL | independent check (`zoomfinal.py`): full model, fresh prose blocks 60-99, code text, dial rebuilt from file | prose dial .122/.117/.103/.074/.048, cut .160/.128/.113/.078/.045, fitted .121/.097/.088/.061/.035. code dial .214/.205/.198/.185/.155, cut .228/.224/.217/.185/.162, fitted .212/.205/.196/.185/.163 | prose: 1 strict yes (32-step sweep), 3 yes, 2 and 4 pass 48-384, FAIL at 768. code: 4 yes everywhere, 2 yes except a .0001 tie at 384. Dev edge at 768 did not hold on fresh data (selection on dev blocks). |
| review | adversarial reviewer (`ZOOM-REVIEW.md`) | no leak, no number-changing bug, dial code matches to 2e-8 in float64 | MAJOR: "no refit" misleading (the corrections ARE a size-scaled learned refit). Storage 1.7x the original layer (dD + eD = 1.77M params). No error bars. Fixed in zoomfinal: sizes < n, sweep from K, ref steps recorded, paired bootstrap, param counts. |
| 12 | rank-16 corrections (storage fix) | .119/.111/.094/.077/.052 | 2 no (768), 1 no. Only 821 nested steps (slower steps); compare at equal steps later. |
| 11 dev2 | v11 scored on dev blocks 20-39 (quick mode) | .110/.110/.090/.072/.047 vs cut .154/.123/.108/.078/.044 | the 768 loss is real (also on dev2), rise .002 |
| 13 | + learned beta schedule (17 knots x share off) | .111/.111/.093/.077/.046 (dev2) | no gain; knots settled ~1.2 everywhere: correction strength was NOT the bottleneck |
| 14 | + hybrid order (gap-first for 192 returns, then strongest-first; v8 showed strongest-first is better at 768) | .110/.110/.092/.075/.045 vs cut .154/.123/.108/.078/.044 (dev2) | closer at 768, still loses by .001; start rise .0027 |
| bs9 | blue sky #9: runtime closed-form LS refit from stored tables (A = H^T H, B) instead of learned corrections (`zoom_rls.py`) | v14: .110/.121/.105/.084/.050 | WORSE than stored corrections everywhere, rises .014. Loss-aware learned corrections beat local copying again. Dropped. |
| tau | parents switch off below a missing share tau (frees slots) | tau .1/.2/.3 at 768: .045/.048/.048 | worse at 768; tau .2 steadier (max rise .0005) but costs both ends. Parents earn their slots. |
| 15 | v14, nested 600 s, 90% dial-path steps | .110/.110/.090/.074/.045 vs cut .154/.123/.108/.078/.044 (dev2) | 768 still loses by .0007; start rise .0023. Stopped iterating here. |
| **15 FINAL** | independent check on UNUSED blocks 40-59 + code, full model, paired bootstrap 95% (`clean-zoomfinal-v15_final40.json`) | prose dial .124/.126/.116/.081/.051, cut .170/.139/.123/.086/.055, fitted .131/.111/.102/.073/.049. code dial .215/.207/.199/.191/.164, cut .228/.224/.217/.185/.162, fitted .212/.203/.196/.185/.163 | 3 PASS. 4 PASS at every size, prose and code (48: dial beats fitted on prose). 2: significant win prose 48-96, code 48-192; prose 192-768 ahead but within 95%; code 384 cut significantly better (-.006). 1: one rise .0011 at 48->64. Storage: corrections 1.77M = 67% of the layer (2.65M). |

### ZOOM verdict (2026-10-19)
VALIDATED (one layer, L28, SmolLM2-135M, independent eval): one zoom configuration, no per-size fitting, gives a
continuous dial from 48 neurons (3%) to the full layer; at the SMALL end it significantly beats the cut on prose and
code and matches per-size fitted layers; full setting = original exactly.
NOT validated: mid/large sizes are about even with the cut (code 384 worse); a .001 rise at the first step; storage
+67% for the learned corrections (rank-16 version lost quality); only one layer; reasoning deferred.
Wording: the corrections are a learned, dial-scaled refit. Say "no per-size fitting", never "no refit".
Caveats (advisor): criterion 4 favours the dial (dial parents got 30 s + 600 s of fitting, each per-size reference
330 s). Version 15 was selected on dev blocks 20-39; blocks 40-59 scored it once. The final steady sweep was every 32
neurons (development: 16), so the .0011 rise may be an undercount. Rank-16 corrections were only tested at equal
TIME (half the steps); an equal-STEP test is the cheap next experiment.


## COARSENING (from 2026-10-19, plan ~/.claude/plans/ok-id-like-you-unified-blum.md)

Method-agnostic. Goal: a method M that builds K coarse units for layer L from the base model and public text only
(no trained parents of L). Contract: `research/clean/coarse/contract.py`. Methods: `research/clean/coarse/methods/`.
Rule for this log: a failure is "method X on unit form Y failed". Never "merging does not work".
Tools: raw-neuron tools only. No SAEs, transcoders, crosscoders, circuit-tracer, Gemma Scope 2.

### Data split (fixed 2026-10-19, before any coarsening result)
| Tier | Contents | Who sees it |
|---|---|---|
| Dev | L15, L28 (and L3) trained parents, families, test blocks 0-59, wikitext train, code_train.txt, all earlier results | implementer |
| Validation | NEW layers L9, L21 (keeper trains global K=48 parents), test blocks 100-139 | keeper only |
| Final | NEW layers L6, L24, test blocks 140-199, code_test.txt (GPT-2 tier only when a method claims it) | keeper, once per method family |
Keeper files: `~/Cale-holdout/` (the implementer never reads it). Ledger: `~/Cale-holdout/ledger.json`.

### Pre-registered tests (fixed before any result)
| Test | Pass rule |
|---|---|
| Cost (primary) | full-model CE at K=48 vs cut_ls of EQUAL stored numbers, paired bootstrap 95% over blocks: bucket better / even / worse. Also bucket vs trained parents: reaches trained / between / at-or-below cut |
| Cover | share of units with 2+ children (R1 thresholds: child top 1% hot, parent percentile >= 0.95, hub guard 20%) >= 0.5 AND >= 3x the random-unit null |
| Shared write | median cosine of a unit's next-word effect with its children's mean effect >= 0.5 and above the random null (n/a if the form has no write direction) |
| Blur only | blind judge "broader category" (r1judge.py), final tier only |
Feedback back to the implementer: buckets and pass/fail only, which test failed first, on which tier. 3 validation
queries per method family per week. Final once per family.

### Keeper ready (2026-10-19). Validation buckets L9 / L21 (no numbers by design)
| Method | cost vs cut | vs trained | cover | shared write |
|---|---|---|---|---|
| cut_ls | even / even | at-or-below-cut | fail / fail | fail / fail |
| shortfit3 (3 min) | better / better | reaches-trained / reaches-trained | pass / pass | fail / fail |
| ref_trained (20 min global) | better / better | reaches-trained | fail / fail | pass / fail |
Keeper notes: cover drops near-copy children (activation corr >= 0.9 with the unit), else a cut unit is its own child.
Cover is noisy near its threshold (the real trained parents fail it). Cost stays the primary test.

### Baselines (dev, L28, K=48, test blocks 0-19)
- cut_ls (`methods/cut_ls.py`): base 3.184 -> 3.318 (+0.134), 82,944 numbers.

### Blue sky #10 (after K1, K2, K3, K4)
- Gates are not the problem. The dominant child already fires where its siblings fire (K2), and cut gates cost almost
  nothing (R2.8). Siblings have unrelated weights (K1), so there is no shared core to average.
- Every base-only rule so far picks the WRITE from local information (weights, co-firing, local least squares). Every
  one of them lands at the cut. Every method that sees how later layers USE the write (short fit with a window, global
  training) beats the cut.
- Under the failures: "coarse" is defined by the readers downstream, not by the children. A coarse write is the part of
  the family's write that later layers need when they cannot tell the siblings apart.
- Prize: 35-49% of the cut's damage per layer, additive (K4).
- K5 as first written (closed-form write under a downstream metric) was DROPPED before running: with gates and values
  fixed, any single global metric gives the same output refit as plain least squares, yet trained outputs beat that
  refit by 0.033 (R2.3). A per-token metric linearized at the original model was R2.4, and it made even the trained
  layer worse. Reading: later layers react NON-LINEARLY to a change this large (97% of a layer removed), so any write
  computed from the original model's local or linearized response misses. The write must be checked through the actual
  later layers: a short fit, or iterated re-linearization (which is a fit).
- Open choices for Shonu: (a) accept a short downstream fit as the coarsening mechanism (minutes; interpretability
  picks WHAT: units, families, dial order) and test it through the keeper; (b) change the setting the essay used:
  coarsen readers together with writers (several layers at once), or another unit form / model (GPT-2 GELU, OPT ReLU).


### TOY RECIPE (Shonu 2026-10-19: tiny by-hand test before automating; L15, 20 children in 4 groups + 3 decoys)
Missing middle step (hypothesis): children share a RECIPE (the earlier units that drive them when they fire), not
weights. Parent = shared recipe, turned back into weights. Recipe = exact direct-path split of each child's activation
(first-order: silu'(g)*up*gate part + silu(g)*up part) over embedding, 144 heads (L0-15), 23,040 neurons (L0-14),
at its top-50 positions minus at 500 random positions. Groups: dates (833+ 1077+ 116- 50+ 940- 994+), conversions
(655+ 168- 131+ 653+ 297- 865-), closing punctuation (16+ 410+ 717- 365+), names (1402+ 638+ 184- 646-), decoys 15+ 69- 235-.
Pass rules fixed before running:
1. Within-group recipe cosine >= 0.3 and >= 3x the cosine across groups.
2. NMF (k = 4) on the recipes puts children into their groups: adjusted Rand index >= 0.5.
3. A gate built only from a group's shared ingredients covers the group's children on held-out dev text (test blocks
   0-39): median child-hot percentile >= 0.9, and higher than the average-of-children gate (H1).

Toy results 1-3 (`clean-toy-recipe.json`, readable `research/clean/coarse/toy_label.md`):
- Rule 1 PASS: recipe cosine within 0.31, across 0.07, with decoys 0.14 (names 0.41, conversions 0.37, dates 0.25, punct 0.18).
- Rule 2 PASS: NMF k=4 ARI 0.76. Conversions and names exact; dates split into year digits vs month/season words (the
  latter joined punctuation).
- Rule 3: 3/4 by near-ties. Cover cannot separate methods: the average child gate (H1) covers 0.97-0.99 too.
  Precision (union-Jaccard with the children's hot sets) separates: recipe parent best for dates .47 (H1 .43, trained .30)
  and conversions .49 (.44, .29).
- Reading: neuron ingredients are semantic. Conversions: L12-14 neurons for "digit right after 'metres (' / 'feet (' /
  'pounds ('" plus head 11.0 reading the unit word 1-4 tokens back. Dates: "month after born/commissioned on", "year
  start after 'in'", head 11.5 reading the month name 0-2 tokens back. Names: Japanese / Indian name-piece neurons.
- Guard needed: many heads read the block's first token with weight 1.00 (attention sink = idle head). They appear in
  every group, like hub children. Exclude idle-head ingredients in the next version.
- One upstream neuron alone can BE the parent: 12.570- covers all six conversion children (0.99, precision 0.48, trained
  parent 0.29); 12.220+ for dates (0.97 / 0.36). Names have no single upstream parent (best 0.61): their parent is a
  combination. So two kinds: "parent already exists upstream" and "parent = combination of ingredients".
Rule 4 (fixed before running, toy end to end): replace the 20 children in L15 by 4 parents (gate from the recipe; value
and write fitted in closed form to the family's summed output). Pass if KL to the original on dev test blocks 0-39 is
lower than (a) keeping each family's dominant child with the same closed-form refit and (b) average-gate (H1) parents,
by paired bootstrap 95%.

Rule 4 PASS (`clean-toy-parents.json`, dev test blocks 0-39): KL to the original after removing the 20 children:
remove 0.00245, H1 parents 0.00201, dominant child + refit 0.00178, RECIPE parents 0.00161. Recipe minus dominant
-0.00017 (95% -0.00028 to -0.00007), minus H1 -0.00040. Recipe parents recover 34% of the removal damage, dominant
27%, H1 18%. CE is noisier: 2.9986 recipe vs 2.9984 dominant (KL is the cleaner measure here). [CORRECTED 2026-10-20, see HONESTY CHECK below] First time on this track
that a parent built from the children's side beats keeping the dominant child at the loss level. The new part is the
gate: it reads the family's shared UPSTREAM ingredients, not the children's weights. Caveats: 4 hand-picked families
(taken from trained-parent families), one layer, dev only, small effect (20 of 1536 neurons), idle-head ingredients
still included.

### Blue sky #11 (after K5 v1-v3, keeper query 1, K6)
- Toy (verified sibling families, 20 children inside a full layer): recipe parents beat the strongest child. Whole layer
  (families found by NMF on recipes): no reliable gain on dev, keeper L9 worse / L21 even, no gain after a fit.
- What changed between the two: the FAMILIES. Toy families were verified siblings (same job, different specifics).
  Whole-layer NMF families read as TOPIC bins (sports bios, ship specs, digits) with grab-bags of 26-34 members.
  Heads dominate the recipes, and heads carry context (what the text is about), so recipe NMF groups by context.
- Candidate error: family discovery groups by co-context, not by shared job. Parent construction may be fine.
- Two clean dev tests: (A) ORACLE families (trained-parent families, dev only) + recipe gates at whole-layer scale:
  tests construction alone. (B) family discovery from NEURON ingredients only (no heads): tests whether job-like
  families appear when context is removed.

### Essay reading #1 (Shonu: read the essay when stuck) and checks (`clean-essay-checks.json`, dev)
Read raw: Cars and Animals, Recursive Generalization, How I Think About Neurons, Under Pressure, Exploring Language
Models, Sign Regimes, Edge Featurizers, Motifs, Watching Neurons Develop. Claims that bear on coarsening (essay hedges kept):
- Neurons are gates / distinctions, not nouns; the negative pole keeps out things grouped earlier. Siblings (taxi,
  ambulance) sit on each other's negative poles; the coarse grouping is an EARLIER gate. Small models "stop at coarser
  distinctions, and end early".
- SwiGLU in VISION models: "the value tends to learn broader things, and gate more specific things".
- Double-negative regime fills the noisy middle of some neurons' positive pole; separate the four sign cases.
- Edge motifs (factoring attribution graphs = connections) predicted ablation damage much better than activation groupings.
Checks on SmolLM2 (rules fixed before running):
1. Value broad / gate specific: FAIL at L15 and L28. Value has more distinct top tokens for only 46% / 49% of neurons;
   gate top positions sit inside the value's high region for only 39% / 27% (so mostly the reverse: the value's top
   positions sit inside the gate's open region). In SmolLM2 the GATE looks like the broader half.
2. Neurons share values: FAIL. Value partner |corr| >= 0.6: 3% / 3% of neurons; gate partner: 17% / 11%. Neurons share
   GATES more than values.
3. Siblings on each other's negative poles (toy families, L15): passes by the letter (1.2% vs 0% of 84 pairs below p25)
   but FAILS in substance: siblings CO-FIRE (30% of pairs above p90 vs 1% random; median percentile 0.82 vs 0.49).
Reading: the vision claims about SwiGLU do not transfer to SmolLM2 as measured. A lead: groups of neurons that share a
gate (same broad region, different values) exist for 11-17% of neurons.

### Setup errors found by treating the essay as true (Shonu 2026-10-20: essay = source of truth)
Strict essay-style sibling search (`clean-siblings-essay-L15.json`): pairs of top-600 poles each below p10 at the
other's top-1%: 14 of 179,700 pairs (3% of poles), mostly punctuation hubs. Null 0. Ranked setup errors:
1. Our "children" (coverage) are redundant CO-DETECTORS, not the essay's children (alternatives inside one category that
   do not fire together). Evidence: siblings co-fire (check 3), parents read as "dominant child broadened". The toy won
   by merging co-detectors (real, small). The answer key is also not essay-coarse: one layer, rest frozen, residual
   bypass open (the essay's crushed layer had no bypass and the whole network retrained). "reaches-trained" only means
   "reaches a 20-min frozen-rest fit".
2. Siblings were searched across neurons. The essay reads them off ONE neuron's negative pole. In SwiGLU a strong
   negative needs gate open and value negative, so the two poles share the gate: the gate is a candidate parent.
   We never read a child's negative pole.
3. Check 1 used token counts (the noun-list lens) and a vision-only claim. Gate "broad" may just mean "often open":
   report gate-open share.
Next (pre-registered): POLES test on the 20 toy children: positive pole (g+u+), negative pole (g+u-), gate-only unit
(value constant). Blind describer, separate blind judge, decoys. (a) for >= half the children the negative pole reads as
a related alternative of the positive pole, clearly above decoy pairs; (b) for those, the gate-only unit reads as a
broader category containing both, clearly above decoys. Report gate-open share.

POLES test result (`clean-poles-test.json`, descriptions `research/clean/coarse/poles/desc.json`):
- Rule (a) FAILED: negative pole judged a related alternative for 10% of children = decoys 10%.
- Rule (b) passes by the letter on 1 of 2 qualifying children (meaningless); overall 15% real vs 0% decoys.
- Reading the descriptions (essay as truth): many negative poles share an ABSTRACT earlier grouping with the positive
  pole: 833+ date day/month vs unit word after a number (word after a number); 131+ conversion digit vs sports count
  (number giving a quantity); 653+ conversion opener vs "comprising / many" (quantity); 940- year digits vs
  "between X and Y" spans (time / range); 865- conversion digit vs time expressions. Matches the essay: the negative side
  "gates out things usefully grouped earlier", not an opposite. Setup error 1: the judge question asked for a
  taxi/ambulance-style category. Setup error 2: the gate was described by its top 12 positions, which coincide with the
  positive pole (essay: a maximum tests the center, not the breadth). The gate is OPEN on 15-60% of positions; both poles
  need it open; the open region is the candidate earlier grouping.
- REVISED test (written after seeing the result above, so it is a new test, not a rescue): describe each gate's OPEN
  REGION (moderate-strength open positions, both value signs, not the top); judge asks the essay's question (is there an
  earlier grouping containing both poles, and is the open region that grouping?). Decoys as before. Pass: real >= 50%
  and >= 2x decoys.

REVISED POLES result (`clean-poles-revised.json`, region descriptions `poles/desc_region.json`):
- Part A (essay question: is there an EARLIER grouping containing both poles?): real 60% vs decoys 30% (2x). 12 of 20
  children, e.g. 833+ "numeric expressions (dates vs measurements)", 653+ "quantity / measure expressions", 940- "date /
  time ranges", 638+ / 646- name fragments. Supports the essay's reading of negative poles (not opposites; things
  grouped earlier).
- Part B (is the gate's moderate OPEN REGION that grouping?): real 35% vs decoys 0%: FAILS the 50% rule. Region
  descriptions are mostly broad / generic syntax glue. The moderate open band (p50-p90 of open positions) is far broader
  than the grouping. The grouping lies between the gate's top (which reads like the positive pole) and its open region.
- Next candidate (essay mechanism, mechanizable with the recipe tools): the earlier grouping is an UPSTREAM unit active
  at BOTH poles' top positions ("the gate in front of taxi and ambulance"). Parent of a neuron = upstream units in the
  intersection of its positive-pole and negative-pole recipes. Families = neurons that share such a parent.

UPSTREAM PARENT test (pre-registered 2026-10-20, before running): for each toy child, gate recipes at its positive-pole
top positions (gate open, value with the pole's sign) and at its negative-pole top positions (gate open, value opposite),
each minus random positions. Shared parent score of an upstream unit = min(pos contribution, neg contribution) when both
> 0. Hub guard: a unit in the top-3 of more than 1/3 of children does not count. The top upstream NEURON per child is
described blind (top contexts, its pole). For the 12 children whose poles got a named earlier grouping (judge A), a
separate blind judge decides if the parent neuron's description would fire across that grouping. Decoy = another
child's parent neuron. Pass: real >= 50% and >= 2x decoys.

UPSTREAM PARENT result (`clean-upstream-parents.json`, `clean-upstream-judge.json`): PASS. Blind judge: the child's
top shared upstream neuron fires across the named earlier grouping for 7/12 children (58%) vs decoys 0/12. Examples:
653+ (quantity / measure expressions) -> 12.706 "digit inside a numeric quantity with a unit"; 168- (words predicting a
consequence) -> 12.1469 "result clause after so / such / enough"; 940- (date / time ranges) -> 10.317 "digit within a
year"; 410+ (punctuation at noun-phrase edge) -> 2.1180 "possessive / attaching punctuation". Misses: 646- (parent is
South-Asian name fragments, narrower than "person-name fragments"), 638+, 994+, 1077+, 365+. Activation check: the
parent is active at BOTH poles' positions for several children (131+ -3.1 / -2.8 sd, 653+ +2.7 / +1.6, 646- +3.3 /
+1.3), weak (< 1 sd) for others. Caveats: n = 12, one judge, some "yes" are loose. Supports the essay's mechanism: an
L15 neuron splits a grouping that an earlier unit already computes.

TOY RULE 5 (pre-registered 2026-10-20, before running): coarse parents from the upstream grouping. Families: 20 toy
children clustered into 4 groups (average-linkage, cosine) by their BOTH-POLE shared gate recipes. Parent gate: reads
the family's shared (both-pole) ingredients, written at the family's positive- and negative-pole positions. Value +
write: same closed form as rule 4, fitted to the family's summed output. Same removal set and dev blocks as rule 4.
Variants scored together: remove, dominant child (hand groups), recipe parents (hand groups, rule 4), upstream gate with
hand groups, upstream gate with own families. Pass: upstream (own families) KL lower than BOTH dominant and recipe
(hand groups), paired bootstrap 95%.

TOY RULE 5 result (`clean-toy-rule5.json`): FAILS by the letter. KL: remove .00245, dominant (hand) .00178, recipe (hand)
.00161, upstream gate (hand) .00158, upstream gate (own families) .00152. upstream_own beats dominant (95% -.00050..-.00003)
but only ties recipe (-.00027..+.00007). Sub-result: on the SAME hand families the upstream (both-pole) gate beats the
recipe (positive-pole) gate, small but significant (-.000027, 95% -.000045..-.000010). Own families are degenerate: one
family of 16 children + names (2) + two singletons. Both-pole recipes share generic ingredients, so clustering them merges
most children. Family discovery is still the open part; the upstream gate is the better gate.

Essay reading #2 (after rule 5): "A car neuron provides a reusable judgment and a sheltered place for taxis, ambulances,
and other specialists to develop." -> a family = specialists that build on the SAME specific upstream judgment neuron,
not neurons that share generic ingredients (what broke rule-5 clustering). Both-pole recipes are the same for both poles
of a neuron, so family members are whole neurons.
TOY RULE 5b (revision, pre-registered before running, same pass rule as rule 5): ingredients weighted by IDF over the 20
children (df = share of children with the ingredient in their top-50 shared ingredients; weight log(1/df)); cluster the
IDF-weighted both-pole recipes into 4 families (average-linkage, cosine). Pass: upstream (own families) KL lower than
dominant and recipe (hand groups), paired bootstrap 95%.

TOY RULE 5b result (`clean-toy-rule5b.json`): FAILS by the letter again. KL: upstream own .00150, upstream hand .00156,
recipe hand .00161, dominant .00178. upstream_own vs recipe 95% -.00030..+.00004 (tie). IDF made the gate gain cleaner:
upstream vs recipe gate on the same hand families -.000052 (95% -.000064..-.000041). Families 15 / 2 / 2 / 1: in essay
terms the 15 may be right (all split "numeric contexts": dates, conversions, counts, years); the hand groups were
co-detector groups. 20 hand-picked children is too small to decide families. Next: whole layer (recipe_v4).
WHOLE-LAYER v4 (pre-registered before running): candidates = top 6K NEURONS by output energy; both-pole gate recipes
(streaming, gate-only coefficients, positive-pole and negative-pole top positions each minus random); IDF over
candidates; family key = each neuron's top IDF-weighted upstream NEURON; tail mode (top K-12 cut neurons as-is, 12
parents for the 12 largest-energy tail families with >= 2 members); parent gate = upstream gate (shared IDF-weighted
ingredients at the family's both-pole positions); value + write closed form to the family's summed output; joint LS.
Dev pass: beats cut_ls at BOTH L15 and L28 (paired bootstrap 95%, dev test blocks 0-39).

HONESTY CHECK (2026-10-20, `clean-toy-parents-fair.json`): rule 4's "dominant child" baseline had its value refit to the
family, which hurts (diag V3). Fair baselines on the same run: dominant child intact .00197; dominant child intact with
ONLY its write refit by least squares to the family output .00171. Recipe parents .00161 vs that fair baseline: -0.00010,
95% -0.00024..+0.00007 = TIE. Correction: recipe parents do NOT beat the best dominant-child baseline; the rule-4 claim
"first time a children-side parent beats the dominant child" was against a handicapped baseline.

### Blue sky #12 (after v4 at K=48 / 384 / 768, toy rules 4-5b, honesty check)
- Construction door closed for THIS contract (one layer, rest frozen, a new SwiGLU unit fitted from local data): every
  version loses to keeping one real neuron intact, and loses more with more parent slots. Same result as R2.0-R2.8 and
  bs9. Consistent with the essay: its coarse layers were RETRAINED; it shows no coarse unit built by a local rule.
- Identification is now solved, blind-checked and base-model-only: upstream judgment neurons (58% vs 0% blind),
  both-pole shared recipes, families keyed by judgment (e.g. 10.317 years, 12.570 conversions).
- Essay reframe: if the parent judgment already exists upstream, dropping specialists IS coarsening; the question is
  ORDER (which neurons to drop first), not building new units.
- Next: (1) ZOOM synthesis: ZOOM-1's dial with v4's base-only families instead of trained-parent families (closes the
  contract if it matches ZOOM-1's small-end win). (2) family-diverse cut: one neuron per judgment family first, then by
  energy; judged on average CE AND code-text CE (breadth). Saliency fix deprioritized (<= 1% of gate variation).

Correction to a plan premise: ZOOM-1 families do NOT come from trained parents. zoom1.py fits its K parents by a 30 s
short-tail fit from the cut (fit_tail), assigns each original neuron to the parent it matches best by |activation corr|,
then a nested fit. So ZOOM-1 already meets the coarsening contract (base model + public text + declared minutes of fit).
FAMILY-DIVERSE CUT (pre-registered before running): judgment keys (recipe_v4 both-pole IDF keys) for the top min(4K,
1536) neurons; select by cut score but take at most one neuron per judgment key in a first pass (neurons without a key
always allowed), then fill to K by cut score; LS refit. Judged on dev prose (test blocks 0-39) AND dev code
(code_train.txt). Pass: at K=384 and K=768 on both L15 and L28, prose CE not worse than cut (95% interval includes 0
or below) AND code CE better than cut (95% interval below 0).

FAMILY-DIVERSE CUT result (`clean-k-dev.json`, text = prose / code): FAILS the pre-registered rule. L28 code better than
cut at K384 (-0.0135, 95% -0.018..-0.009; 0.172 vs 0.186) and K768 (-0.006, -0.010..-0.002); L28 prose tie at 384,
worse at 768 (+0.0036). L15: prose tie / worse (+0.0032 at 768), code tie. Reading: calibration is prose; energy ranking
favors prose specialists; one-per-judgment-family keeps general coverage (code). A breadth vs prose-fit trade-off.
NULL (pre-registered before running): random swap of the SAME number of neurons (78-206) from ranks K..4K into the cut,
3 seeds. Pass: diverse_cut beats every seed of the random-swap null on L28 code at K=384 (95% interval below 0).

NULL result: PASS (`clean-k-dev.json`, L28 K384 code). diverse_cut 0.172 vs random-swap seeds 0.207 / 0.183 / 0.190 (plain
cut 0.186); diverse minus each seed -0.034 / -0.011 / -0.017, all 95% intervals below 0. Random swaps are WORSE than the
plain cut; judgment-family swaps are better. The breadth gain comes from the upstream-judgment structure.
KEEPER PROTOCOL EXTENSION (logged before use, 2026-10-20): eval.py gains --K and --text {prose,code}. Validation code =
code_test.txt blocks AFTER the first 40 (final tier keeps the first 40 untouched). Calibration runs of cut_ls with the new
options are exempt from budget.
KEEPER QUERY (family "diverse-cut", query 1 of 3), prediction written before the run, K=384: L21 (late, like L28) code
BETTER than cut, prose EVEN; L9 (early, like L15) code EVEN, prose EVEN.

DIVERSE CUT across the dial (dev, `clean-k-dev.json`), diverse minus cut: L28 code K48 +0.0028 (worse), K192 -0.0024
(better, 95% -0.0047..-0.0001), K384 -0.0135, K768 -0.0060; L28 prose K48 +0.0019 (worse), K192 / K384 tie, K768 +0.0036
(worse). L15 code tie at 48 / 192 / 384 / 768; L15 prose tie except K768 worse. Breadth gain lives in the MIDDLE of the
dial at the late layer (where ZOOM-1 only tied the cut); at the small end energy ranking wins.

KEEPER (validation, family "diverse-cut" query 1 of 3, diverse_cut.py sha256 bf079d3a..., recipe_v4.py 6dc5363e...),
K=384: L21 prose EVEN, code EVEN (predicted code better); L9 prose WORSE, code WORSE (predicted even). Calibration
cut_ls K384 even everywhere. The dev L28 breadth gain did not transfer.
LESSON (two keeper misses, recipe_v3p12 and diverse_cut, both carried by dev L28 gains): L28 sits two layers before the
output and behaves unlike L21. A dev effect seen at L28 only is not evidence for late layers in general. Before the next
keeper query, check on MORE dev layers (allowed: 0-5, 7, 8, 10-14, 16-20, 22, 23, 25-29; never 6, 9, 21, 24) and require
the effect at >= 3 of them, incl. one in 22-27.

### Tools (Step 1, 2026-10-19, `research/clean/tools/`)
- nnsight 0.8.0 PASS: HF base CE 3.0545 vs MLX 3.0547; zeroing L28 neurons 0-299 +0.0536 vs +0.0534.
- delphi 0.1.3 PASS (dry run, stubs for vllm/faiss/sparsify in `pilots/t2_stubs.py`): raw neurons via LatentCache on `model.layers.L.mlp.down_proj`, transcode=True, identity encoder. Clamps negatives: run the negative pole as a second pass. OpenRouter client built in. Notes: `tools/DELPHI-NOTES.md`.
- NeuronScope 0.1.7: works on gpt2, cannot load SmolLM2, no single-neuron ablation. Adds nothing over nnsight. Dropped.
- GRIFFIN (SwiGLU, training-free) and MoEfication (ReLU-only router) read: `tools/SKIP-NOTES.md`.

### Rounds
| # | Method family | Idea | Dev result | Keeper verdict |
|---|---|---|---|---|
| K1 | family shared core (unit form: family block, rank dial) | 2a gate FAILED its pre-registered rule at L28/L15/L3 (`clean-k1-spectrum.json`): sibling gate/value/output rows are as unrelated as random neurons (rank@90% ratio 0.99-1.00, top-1 share 0.23-0.26 vs null 0.22-0.25). Only firing is shared (activation top-1 0.38-0.40 vs 0.23-0.25). Data-weighted view (pre-activation corr): gate |corr| 0.14-0.24 vs null 0.07-0.13, about 2x random but small. Reading: siblings are distinct detectors. A coarse parent is a NEW direction, not a subspace the children share. This closes K1 on SwiGLU SmolLM2 only. | not submitted |
| K2-check | gate broadening (dominant child, wider gate) | NOT closed (`clean-k2-broaden-check.json`). At a sibling's top-1% positions the dominant child's gate pre-activation sits at median percentile 0.92 / 0.94 / 0.87 (L28/L15/L3; random neuron 0.46-0.50), its hidden at 0.95 / 0.96 / 0.94. 40-58% of sibling pairs above 0.9. Caveat: children were listed BY parent coverage, so some circularity. Reading with K1: the dominant child already fires where its siblings fire, with unrelated weights. That is why cut gates are good (R2.8). The coarse part is the WRITE (what the family writes when it fires), which is defined downstream. | not submitted |
| K3 | base-only families + gate widening (SwiGLU + gate bias, LS write) | FAILED on dev (`clean-k3-basefam.json`). Families from the cut's 48 neurons (coverage >= 0.95, hub guard): only 34% / 22% of parent poles have 2+ children (L28 / L15); overlap with trained families 16% / 16% (null 1% / 2%). So trained parents are mostly NOT the dominant child broadened. Caveat: matches are moderate (|corr| trained unit vs best cut neuron median 0.65 L28 / 0.62 L15, range 0.34-0.90; the best match among ALL 1536 neurons is the same cut neuron). Trained units start at the cut and drift away from it. Widening (alpha 0 / 0.5 / 1): L28 +0.134 / +0.134 / +0.139, L15 +0.065 / +0.066 / +0.066. No gain. | not submitted |
| K4-ceiling | how big is the prize? | `clean-k4-twolayer.json`, dev blocks 0-39: L15 cut +0.054 / trained +0.028; L28 cut +0.132 / trained +0.086; both cut +0.186 / trained +0.115. Gains ADD (no compounding, no loss). Trained parents (one layer at a time, rest original) save 35-49% of the cut's damage per layer. | n/a |
| K5 v1 | recipe parents, whole layer (`methods/recipe_v1.py`): top 6K poles, streaming recipes, hub-ingredient guard (27 dropped), NMF K families, recipe gate + family-fit value, joint LS | dev check FAILED (`clean-k-dev.json`): L15 +0.0595 vs cut +0.0544 (worse, 95% +0.002..+0.008), L28 +0.134 vs +0.132 (tie). Diagnosis (`clean-k-diag-v1.json`, same families, one swap at a time): L28 recipe gate beats dominant gate with the same refit by 0.010 (.1342 vs .1445); L15 neutral (.0595 vs .0584). The loss is WHICH units get slots: family dominants as-is .0587/.1380 vs cut .0543/.1321; only 27-28/48 overlap with the cut's top neurons. NMF made families of up to 24 and spent slots on minor ones. Family-fit value helps only with a broad (recipe) gate. | not submitted |
| K5 v2 | seeded families (cut's top-K neurons are seeds, members by recipe cosine >= 0.2) | dev FAILED: L15 +0.0576 vs cut +0.0544 (95% +0.001..+0.005), L28 +0.1309 vs +0.1321 (-0.0012, crosses 0). One L15 seed took 37 members: its parent blurred an important neuron. Next (v3, fixed before running): top K-P neurons as-is, P recipe parents for families of the remaining strong poles, P = 12 and 24 (like the toy: parents stand in for removed children). | not submitted |
| K5 v3 | tail parents: top K-P neurons as-is, P recipe parents for NMF families of the remaining strong poles | dev (`clean-k-dev.json`): P=12: L15 +0.0551 vs cut +0.0544 (tie, 95% -0.0006..+0.0021), L28 +0.1299 vs +0.1321 (better, -0.0023, 95% -0.0043..-0.0005). P=24: L15 +0.0583 (worse), L28 +0.1295. Pre-registered dev rule (beat cut at BOTH) NOT met. L28 gain = about 5% of the cut-to-trained gap; the 3-min fit reaches trained on sealed layers. Families are huge (12 bins for ~250 poles: sizes up to 33-34). KEEPER QUERY 1 (recipe family), prediction written before the run: L21 better than cut, L9 even. | KEEPER (validation, query 1 of 3 for family 'recipe'; method sha256 61a08b0a..., recipe_v1.py 2915533c...): cost vs cut L9 WORSE, L21 EVEN; vs trained at-or-below-cut both; cover fail, shared write fail. Both one bucket below the prediction. The dev L28 gain did not transfer. |
| K6 | recipe + short fit (`methods/recipe_fit3.py`): v3p12 gates frozen, 3-min window fit of values/writes. Control `cutfit3_frozen` (cut gates frozen, same fit) | dev FAILED (`clean-k-dev.json`). dCE / median children / share 2+: L28 recipe_fit3 +0.1379 / 2 / .60, cutfit3_frozen +0.1328 / 3 / .65, shortfit3 +0.1310 / 5 / .75, cut_ls +0.1321 / 2 / .52. L15 recipe_fit3 +0.0378 / 1 / .38, cutfit3_frozen +0.0374 / 1 / .43, shortfit3 +0.0345 / 1 / .44, cut_ls +0.0544 / 0 / .24. Recipe gates do not help after a fit (L28 worse than frozen cut gates by .005, L15 tie) and do not add coarseness. Note: the 3-min fit itself makes units coarser than the cut (children metric with near-copy exclusion), and gains little at L28 on dev (+.131 vs +.132) but a lot at L15 (+.035 vs +.054). | not submitted |
| K7 v4 | upstream-grouping parents, whole layer (`methods/recipe_v4.py`): both-pole gate recipes, IDF, family key = top upstream judgment neuron, tail mode P=12 | dev FAILED (`clean-k-dev.json`): L15 +0.0560 vs cut +0.0544 (worse, 95% +0.0003..+0.0029), L28 +0.1321 vs +0.1321 (tie). Families now small and essay-like (2-10 members; keys incl. 10.317 years, 12.570 conversions). Ceiling check: the 12 families' members carry 4.5% / 4.2% of the layer's output energy, the 12 cut neurons they replace 2.9% / 1.8%, all dropped neurons 80% / 45% (L15 / L28). At K=48 a parent must stand in for ~30 neurons; specialist families (2-10) are the granularity of MID sizes. Next (pre-registered): v4 at K=384 and 768 with P=K/4, pass = beats cut_ls at both L15 and L28 (where ZOOM-1 only tied the cut). | not submitted |
| K7b v4 mid sizes | same, K=384 / 768, P=K/4 | dev FAILED, worse with more parents (`clean-k-dev.json`): K384 L15 +0.0241 vs cut +0.0234 (tie), L28 +0.0820 vs +0.0747 (worse, +0.005..+0.010); K768 L15 +0.0121 vs +0.0095 (worse), L28 +0.0563 vs +0.0416 (worse, +0.010..+0.021). A unit fitted locally to a family's summed output loses to keeping one real neuron intact. Identification (families, upstream judgment) now follows the essay; construction of a NEW unit is the bottleneck at every size. Saliency check: coordinate 247 (L15) and 17 / 507 (L28) carry 54% / 46% / 32% of the gates' resting level but <= 1% of their variation (a bias channel, the essay's 'saliency knob'). | not submitted |

## Log

- 2026-10-19: R1 trained K=10/48/192 at L3/15/28 (local, from cut and random). Global KL runs at L15 K=10, 48.
  Local training beats cut by 0.005-0.015 nats (`clean-r1-train.json`). Global L15 K=10: cut +0.066, global +0.045.
- 2026-10-19: A7 (`r1combine.py`, K=10, per unit best pole; pass = 2+ children AND children explain >= 50% of the
  unit's top items). Depth axis picked BEFORE results (P2). Trained / random / old-as-parent:
  L3 input 1.00 / 0.10 / 0.75. L15 input 0.70 / 0.10 / 0.30. L28 output 0.80 / 0.00 / 0.05.
  Notes: the L3 output axis does not discriminate (random null 0.80). The L3 old-as-parent null is high (0.75):
  old L3 neurons co-fire a lot. Only 10 units per cell, so each unit = 10 points. Running: downstream axis at L15, K=48.
- 2026-10-19: BLIND SEMANTIC TEST (`r1family.py`, `r1judge.py`, `results/clean-r1judge-*.json`). Describer, scorer and
  judge are separate agents. The judge picks real children over decoy old neurons as the set the parent is a broader
  category of. L3 input: 7/7 families with children (p = 0.008), detection score 0.86. L15 input: 6/7 (p = 0.06),
  detection 0.89, parents coherent 86%. Read examples: L15 "digit starting a converted value in brackets" over
  siblings "after knots/tons", "after kg/metres", "after metres". L15 "start of a year in a date" over year-digit
  variants. L3 "unit after a number" over distance/speed/length units. Bug found and fixed: L3 F8-F10 had no children
  (script did not let parents share children). They were graded as right by a guess, so 10/10 became 7/7.
  Caveat: the parent is often its dominant child broadened to siblings, not an even OR.
- 2026-10-19: K=48 (more units): pass share L3 input 0.85 (nulls 0.10 / 0.75), L15 any axis 0.73-0.77 (0.10 / 0.45),
  L28 any axis 0.69 (0.00 / 0.25). Not "every unit" yet: 15-30% of units are still unexplained.
- 2026-10-19: L28 output blind test: judge 8/8 (p = 0.004), detection 0.85. BUT F5-F8 are an artifact: parent and
  children all read "mix of rare word fragments and code-like tokens". Steering pushes many units toward the same
  junk tokens (hub items), so they match on junk. Real coarse family: F1 parent "closing punctuation" over closing
  quote / closing ")" / colon / closing "]" or "}". F2 (word stems) is parent ~= child, not broader.
- 2026-10-19: Fix: output axis with item scaling (`r1out.py --z`, `results/clean-r1out-z.json`). Per unit, K=48,
  pass on input OR output-z, trained / random / old-as-parent: L3 0.88 / 0.12 / 0.80. L15 0.73 / 0.06 / 0.40.
  L28 0.62 / 0.00 / 0.20. The earlier L28 output numbers (0.69-0.80) were inflated by the junk hub.
  Note: at L3 old neurons also cover each other (0.80), so L3 coverage alone barely separates trained from old.
  The L3 evidence rests on the blind semantic test (7/7).
- 2026-10-19: A8 virtual weights (`r1virt.py`, `results/clean-r1virt.json`): which later components read the unit's
  output direction. Coverage, K=48, 2+ children: L28 0.60 / random 0.07 / old 0.00. At L3/L15 ~0 even for old
  neurons: coverage (top-5% threshold) does not fit a dense linear axis. Linear-axis test (positive sparse mix of
  <= 5 old poles, R2 >= 0.5, 2+ children >= 10% each, fixed before looking): L28 0.46 / 0 / 0. L3 0.10, L15 0.06
  (nulls 0-0.05). Combined per unit, any of input / output-z / virtual: L28 0.96 / 0.00 / 0.20.
- 2026-10-19: L28 output-z blind test, K=48: judge 20/21 (p = 1e-5), detection 0.98. BUT the judge called 19 of 20
  "same", not "broader", and F2-F9 are again "code and symbol fragments" with the same 4 old children.
  Cause found: log-prob change. A strong push near the output flattens the distribution, so the rarest (junk)
  tokens gain most in log-prob. Any flattening unit looks like a junk promoter. Item scaling cannot remove it.
  Fix: probability change (`r1out.py --target probs`). Running.
- 2026-10-19: probs readout (`clean-r1out-probs.json`): too weak, 0 children even for old neurons at every layer.
  Dropped.
- 2026-10-19: A9 direct logit attribution (`r1virt.py --reader dla`, `clean-r1dla.json`), K=48: coverage 2+ trained
  0.25-0.36 vs random 0.07-0.08 vs old 0.03-0.07. Sparse mix does NOT discriminate here (random 0.40-0.42 pass):
  5 of 3072 directions fit almost anything in this 576-dim space. Stopped adding axes. Next: blind test on L28
  virtual-axis families, described by DLA words (linear, no flattening artifact).
- 2026-10-19: Second judge with a "broader / same / unclear" field (`*_judge2_out.json`): same choices (L3 7/7,
  L15 6/7). Relation among right picks: L3 broader 2, same 5. L15 broader 2, same 4. So most parents read as "about
  the same as one or more children": the dominant child broadened to siblings. A clearly wider category is the
  minority (4 of 13). Caveat: descriptions may be too coarse to show the small differences between siblings.
- 2026-10-19: L28 virtual-axis blind test, K=48 local: judge 20/34 (p = 0.20), NOT significant. Cause: HUB CHILDREN.
  The same 4 old neurons (function words, spacing/punctuation, both/their, prepositions) are "children" of almost
  every parent. They carry a generic output direction that many later components read. Guard: a child under > 20%
  of all trained unit poles is a hub and does not count (fixed before looking).
- 2026-10-19: P6 confirmed at L28: the LOCAL objective (copy the layer's output) makes units compress generic
  directions. The GLOBAL (KL) layer-28 K=48 units: input axis 2+ children 0.65 of poles (local 0.31), up-cover 0.53
  (local 0.25). Per unit, no hubs: input 0.71 / random 0.00. Virtual axis fell to 0.17 (fewer hub matches).
  Use global layers for L28 from now on. Blind test on L28 global input families running (34 families).
- 2026-10-19: L28 GAP CLOSED. Global L28 K=48, input axis, blind test (`clean-r1judge-L28_input_K48.json`): judge
  34/34 (p = 6e-11), relation broader 21 / same 9 / unclear 4, detection 0.92, parents coherent 65%. No hub children
  (max 6 of 34 parents share one child). Examples: "last digits of numbers in ship specs" over digits in
  comma/decimal numbers, in measurements, in brackets. "First piece of a proper name" over word-start pieces and
  name pieces. Global training gives clearly BROADER parents (62%) than local layers at L3/L15 (~30%).
  Next: same blind test on global L15 K=48 (running), then the "how" step on global parents.
- 2026-10-19: Global L15 K=48 blind test (`clean-r1judge-L15g_input_K48.json`): judge 30/30 (p = 9e-10), detection 0.81.
- 2026-10-19: "Broader vs same" depends a lot on the judge. One judge rated ALL families (78, shuffled across layers,
  `results/clean-r1relation.json`): broader / same / unrelated: L3 5/1/1, L15 local 3/3/1, L15 global 15/14/1,
  L28 global 23/6/4. About 3 children fall inside each parent's scope. Use one judge for any comparison.
- 2026-10-19: FIRST "HOW" RESULT (`r2how.py`, `results/clean-r2how-L*-K48-global.json`). Each M1 rule builds one
  parent from the verified children only (target = the children's summed output). It is compared with the trained
  parent, which it never sees. Median activation match with the trained parent:
  L28: H0 0.42, H1 0.39, H3/H4 0.43, H6 0.58, SPAN bound 0.63. L15: H0 0.32, H2 0.38, H6 0.34, SPAN bound 0.52.
  Every rule fires on the children (child cover 0.77-0.94, trained 0.97-0.98), so the coarse "fires on all children"
  part is easy. Matching the trained parent is not. Even SPAN (the best mix of the children's own gates and values,
  fitted to the trained parent) reaches only 0.52-0.63. So the trained parent uses more than its <= 6 listed
  children: the children list is incomplete, or the parent also uses directions no old neuron owns.
- 2026-10-19: L15 failing units (12 of 48): 8 have exactly ONE child. NOT copies: their best activation correlation
  with any old neuron is 0.29-0.59. They are new units that the input axis cannot read. Try the probs output axis.
- 2026-10-19: Are failing units just tiny? K=48 pass rate, big half vs small half by output share:
  L3 0.83 vs 0.88, L15 0.79 vs 0.79, L28 0.88 vs 0.50. Only at L28 are failures mostly small units. Failing units
  carry 12-18% of each layer's output. So at L3/L15 the gap is a real axis/test problem (P1), not unit size.
- 2026-10-19: A4 downstream v1 is DEGENERATE: zero children even among old neurons. Item scaling did not fix it.
  Profiles are nearly unrelated across units (mean |cos| 0.07). The mean change over positions probably washes out
  position-dependent effects. Needs a different readout (open).
- 2026-10-19: Coverage cannot tell coarse from polysemantic (a union of unrelated children also passes).
  The description step (explain, then blind score, then "is the parent a broader category?") is the real test of P1.
- 2026-10-19: Global (KL, frozen rest, 20 min) CE, base 3.018 (`results/clean-r1-global.json`): L3 K10 cut 3.139 /
  local 3.129 / global 3.106. L15 K10 3.084 / 3.082 / 3.063. L15 K48 3.071 / 3.065 / 3.051. L28 K48 3.160 / 3.145 /
  3.111. Global recovers 25-40% of the cut's cost.
- 2026-10-19: A1 fails as a coarse test by definition (P4). A2 finds a coarse signature at L3 that weakens with
  depth. Per P1/P2, read that as an axis problem. Next: A3 at L28 and L15.

## GATE FAMILIES (from 2026-10-20, plan ~/.claude/plans/we-ve-been-plugging-away-snazzy-duckling.md)

Essay reading #3 (full text, all sections). Shonu: use the essay for CONCEPTS to apply, not as a recipe.
Idea catalog: `research/clean/ESSAY-IDEAS.md` (I1-I12). Main bet: I1 + I2 + I3.
Key math (I1): for SwiGLU neurons that share a gate g, silu(g) factors out:
sum_i silu(g) u_i(x) d_i = silu(g) * (x^T M), M = sum_i u_i d_i^T. A rank-r version of M gives r real SwiGLU
neurons with the shared gate. M1b's wall ("one neuron cannot express the sum of two") does not apply inside a
shared gate. Lead from `clean-essay-checks.json`: 17% (L15) / 11% (L28) of neurons have a gate partner (|corr| >= 0.6).

G1 (pre-registered 2026-10-20, before running). Dev layers 2, 8, 12, 15, 18, 23, 26, 28 (never 6, 9, 21, 24).
Calibration text, 20k positions (block-first positions dropped). Statistic: corr of silu(gate pre-activation)
over positions (the factor that must be shared). Groups: average-linkage on that corr, cut at 0.6 (main) and 0.45.
Per group of size n >= 2, local merge test (closed form, no gradient fit):
  - representative gate = the member with the largest output energy;
  - merged output = silu(g_rep(x)) * x^T N, N = rank-r solution of min E||F(x) - silu(g_rep) x^T N||^2
    (F = the group's true summed output), via full LS then rank-r under the metric E[s^2 x x^T];
  - baseline at the SAME unit count r: keep the r strongest members, LS refit of their writes to F.
  - report relative error of F for merge vs baseline, r = 1 and r = n/2 (rounded up).
Null: random groups with the same sizes, same test.
Pass (both): (a) groups at 0.6 cover >= 10% of neurons in >= 5 of the 8 layers; (b) median over groups of
(baseline error - merge error) at r = 1 is > 0, and larger than the same difference for the random-group null,
in >= 5 of the 8 layers.

G1 diagnostic (`clean-g1-tie.json`, L15 / L28): raw SwiGLU gates are NOT shared. Each neuron's best gate partner:
median silu-gate corr 0.44 / 0.46; tie error (replace its silu(g) by its partner's, best scale) median 0.62 / 0.68;
tie error <= 0.1 for only 4.3% / 0.5% of neurons. The "11-17% gate partner" figure counted the gates' resting state.
L15 G1 (0.6 groups): merge ties keep-strongest at r = 1 (0.437 vs 0.441) and loses at r = n/2 (0.41 vs 0.25).
Reading (essay as truth): the shared "gate in front" is UPSTREAM (the judgment neuron), not a gate in layer L. Each
specialist's own gate = judgment vote + specific votes (I3). So test the shared judgment gate directly (G3).

G3 (pre-registered 2026-10-20, before running). Layers 12, 15, 26. Families: recipe_v4 judgment keys (keys_for) on
the top 400 neurons by output energy; families = same key, size 2-12. Calibration text, 15k fit / 5k eval positions.
Per family, relative error of its summed output F on eval positions, at unit count r = 1 and r = ceil(n/2):
  - BASE: keep the r strongest members whole, LS write.
  - JREAD: one shared gate silu(a z + b), z = the residual at L read along the judgment neuron's write direction
    (norm weight undone); a, b by grid on fit rows; value + write = rank-r LS under that gate (r SwiGLU neurons that
    share one gate).
  - JACT (upper bound, not buildable as is): same with z = the judgment neuron's true activation.
  - NULL: JREAD with a random neuron from the key's layer as the "judgment".
Pass: median (BASE - JREAD) at r = 1 > 0 AND > median (BASE - NULL) at r = 1, on >= 2 of the 3 layers.

G4 (pre-registered 2026-10-20, before running): variance collapse after LS refits (I6). Layers 12, 15, 26; K = 48
and 384; cut_ls built on calibration blocks. Measures: per-coordinate std ratio of the coarse layer's output vs the
original (median), output energy ratio, median std ratio of layer L+1 gate pre-activations. Fixes: REPAIR-style
per-coordinate affine (coarse output mean/std matched to the original's on calibration; 2d stored numbers) and a
single global scale. CE on dev blocks 0-39, paired bootstrap 95%. Pass: a fix recovers >= 20% of cut_ls's damage
(dCE vs original) with a 95% interval below 0 vs cut_ls, on >= 2 of 3 layers at one K.

G4 result (`clean-g4.json`): FAIL, clean negative. The LS refit does shrink the layer output (per-coordinate std
ratio 0.47-0.51 at K48, 0.80-0.82 at K384) but the next layer's gate pre-activation std is unchanged (0.98-1.04):
the residual stream carries the rest. Restoring the spread HURTS everywhere: REPAIR adds 7-43% of the cut's damage,
a global scale 7-45% (all intervals above 0). Variance collapse is not why LS merges land at the cut. The removed
spread is noise the refit was right to drop.

G1 result (`clean-g1.json`): FAIL by the letter. (a) 0.6-groups cover >= 10% in 4 of 8 layers (L15 .105, L18 .182,
L23 .134, L28 .111; L2 .019, L8 .022, L12 .051, L26 .076). (b) passes in 8 of 8: merge beats keep-strongest in
77-100% of groups, null gain ~0. But the gain is tiny: median 0.001-0.013 of the family's output energy.
Reading under the axioms (Shonu 2026-10-20: the essay is law; tests check OUR method only): the method used a gate
from layer L itself. The essay's shared gate is the upstream judgment. Next: G3.
G3 speed change (before any G3 result was seen): 10k positions (5k fit / 5k eval) and a 6 x 4 grid for (a, b),
because the first version needed ~18k large solves per layer. Pass rule unchanged.

G3 L15 (`clean-g3.json`): FAIL. 60 judgment families (sizes 2-5). r = 1 relative error of the family output:
keep-strongest 0.46, JREAD 0.69, JACT (true judgment activation as gate) 0.93, null 0.72. JREAD wins 0/60.
Reading under the axioms (method error, not essay error): the test scores a coarse unit by COPYING the family's
summed output. A coarse "car" unit is not meant to reproduce "taxi" output on taxis. It must give readers what they
need when they cannot tell the siblings apart. A copy test penalizes coarseness itself (same as blue sky #10).
RULE from here: coarse units are judged by downstream loss (dev CE), never by a local copy error.

G5 GRAFT (pre-registered 2026-10-20, before running). Method `coarse/methods/graft_v1.py` (I4, recursive
generalization as an axiom): trunks = cut top-K; siblings = next 2K neurons, each to the trunk "first in line"
(largest relative |hidden| at the sibling's top-50 positions); route = upstream neuron in L-6..L-1, selective for
the sibling's top positions and readable at L (|corr(read, activation)| >= 0.5); graft = route read added to the
trunk's gate and up so that at the sibling's top positions they reach FRAC = 0.5 of the trunk's own top level (up
sign chosen so the trunk's write points along the sibling's); joint LS writes. Same stored numbers as cut_ls.
Judged ONLY by dev CE (`dev_eval.py`, test blocks 0-39), paired bootstrap 95% vs cut_ls.
Pass: graft - cut_ls interval below 0 on >= 2 of layers 12, 15, 26 at K = 48 or K = 384 (prose). If pass: confirm
on >= 3 more dev layers incl. one in 22-27 and on code text before any keeper query.

G5 debug (graft_v1, L15 K48: +0.095 vs cut +0.054, WORSE). Implementation errors found: one broad trunk (1254, fires
everywhere) took 78 of 94 grafts; value grafts leaked into trunk cores (core corr down to -0.74). graft_v2 adds axiom
guards: "first in line" by selectivity above the trunk's own baseline; <= 3 offshoots per trunk; route must be
selective for the sibling and not the trunk core; skip a graft if core corr < 0.9. Only 7 grafts survive at L15 K48
and they barely move the trunk in sibling territory (1.51 -> 1.70). CE check running (same pass rule).

G6 ROUTE CAPTURE (pre-registered 2026-10-20, before running). Axioms: circuit racing ("established routes have a head
start in capturing residual error") + umbrella->bee (new behavior = an off-ramp between existing routes). A removed
sibling's output is residual error; let the routes that already fire there capture it. Candidates = all neurons of
layers L-6..L-1 (their activations already exist). Selection: top R by single-regressor energy on the cut's LS
residual. Joint LS of [kept layer-L hiddens, route activations] -> original layer-L output.
Arms (layers 12, 15, 26; K = 48 and 384; dev CE blocks 0-39; paired bootstrap 95% vs cut_ls):
  - SKIP: K/2 kept neurons + 3K/2 route writes (a route write = d numbers) = the cut's stored numbers exactly.
  - FOLD: K kept neurons + R = K routes folded into the upstream neurons' own write columns (0 extra numbers).
  - FOLD3: same with R = 3K.
Pass: SKIP or FOLD better than cut_ls (interval below 0) on >= 2 of 3 layers at one K. FOLD arms are judged with all
side effects on the layers in between included (real forward pass).

G3 L12: FAIL like L15 (JREAD wins 0/57; copy test, see rule above). G5 graft_v2: L12 +0.0560 vs cut +0.0552 (worse,
95% +0.0002..+0.0017), L15 tie. Grafting through dense residual reads: FAIL as built.
G6 (pre-registered): FAIL by the letter. L15 fold3 better (-0.0032, 95% -0.0062..-0.0002); L12 skip/fold3 blow up
(+0.42); L26 fold worse (+0.008/+0.016), skip tie; K384: ties / slightly worse.
G6b exploratory (`clean-g6b.json`, L15 K48): fold scales with R: 3K +0.0512, 10K +0.0437, 30K +0.0405 (cut +0.0544).
Routes only among upstream trunks: 3K tie, all trunks blow up (+0.45). Refit iterations after folding: worse.
Cross-validated ridge + spike filter: L12 still broken. SKIP-FREE arm (K kept + R route writes at L, no side
effects, extra storage): L15 30K +0.031 (43% of the cut's damage recovered; trained parent +0.028), L26 30K +0.087
(23%), L12 broken (+0.17).
BUG FOUND (position 0): fits drop each block's first position (attention sink), but the dev eval runs it. A route
(or ANY new unit) never constrained at position 0 can write garbage into the sink that every later token reads.
Masking route writes at position 0: L12 skip-free 3K +0.0508, 30K +0.0380 (cut +0.0551): works. This may have hurt
OLD construction methods too (new gates never saw position 0). Fair fix ("sink guard"): keep the original layer
output at position 0 for every arm incl. the cut. In Cale, position 0 is always the same chat-template token, so
that output is a constant per layer.

G7 ROUTE CAPTURE, SINK-SAFE (pre-registered 2026-10-20, before running). Sink guard on every arm (cut included) and
on every upstream layer whose weights a fold changes. Layers 3, 12, 15, 18, 26 (dev). K = 48 and 384. CV ridge.
Arms: CUT (cut_ls); SKIP (K/2 kept + 3K/2 route writes: equal stored numbers); FOLD10 / FOLD30 (K kept + 10K / 30K
routes folded into upstream write columns: 0 extra numbers); SKIPFREE30 (upper bound, extra storage).
Pass: FOLD30 or SKIP better than CUT (95% interval below 0) on >= 3 of the 5 layers incl. L26, at K = 48.
Secondary: same at K = 384.
Sink check of OLD methods (`clean-sinkcheck.json`, L15 / L26, K48): the sink guard changes nothing (recipe_v4 +0.0560 /
+0.1169, graft_v1 +0.0954 / +0.1544, same with and without). The position-0 bug only hit ROUTE writes (upstream
neurons with huge position-0 values). Old construction failures were not caused by it.

G7 result (`clean-g7.json`, K48, sink guard on every arm): FAIL by the letter (FOLD30 better on 2 of 5, not L26).
dCE (share of the cut's damage recovered): L3 cut .1159 / fold30 .1195 (-3%) / skipfree30 .1046 (10%); L12 .0552 /
.0532 (4%, tie) / .0380 (31%); L15 .0544 / .0372 (32%) / .0309 (43%); L18 .0665 / .0571 (14%) / .0419 (37%); L26
.1134 / .1112 (2%, tie) / .0869 (23%). SKIP at equal storage loses or ties everywhere. SKIPFREE (10x the coarse
layer's storage, an upper bound) wins on all 5 layers. Reading: the routes carry the signal; folding it into upstream
writes loses most of it to side effects in the layers between. Next (G8): penalize folded directions those layers read.

G8 (exploratory, `clean-g8.json`, K48, R=30K from L-6..L-1): reader penalty (mu * sum c_r P c_r, P = in-between
readers) helps a little: L12 .0532 -> .0505 (mu 1), L26 .1112 -> .1062, L15 .0372 -> .0367 (mu .1).
BACK = 1 (routes = layer L-1 only) is much better and needs no penalty: L12 .0461, L26 .0923 (R=20K); with ALL 1536
neurons of L-1 as routes (R=32K): L3 .1099 (5% of the cut's damage recovered), L12 .0437 (21%), L15 .0338 (38%), L18
.0474 (29%), L26 .0888 (22%); all better than the cut (95% below 0). Zero extra stored numbers for one coarse model,
~12 s per layer, one CV-ridge LS. (L15 3-min fit ~ .035, 20-min trained parent .028.) Axiom reading: circuit racing,
the closest route has the biggest head start; layer L-1's neurons write the coarse version of the removed children.
BACK = 1 was chosen after seeing L12/L26, so these 5 layers are not clean evidence. Name: PLF (previous-layer fold).
Storage caveat for the NESTED dial: the folded L-1 write matrix differs from the original (a 1536 x 576 delta per
coarsened layer, 1/3 of a layer) unless stored low-rank or shared across sizes.

G9 PLF CONFIRMATION (pre-registered 2026-10-20, before running). Method fixed: keep the cut's top-K at L; routes = all
neurons of L-1; joint CV-ridge LS of [kept L hiddens, L-1 hiddens] -> original L output; the route part is added to
layer L-1's write matrix; sink guard on L and L-1. Fresh dev layers 1, 10, 20, 23, 27. K = 48 and 384. Text = prose
(test blocks 0-39) and code (code_train.txt).
Pass: better than cut_ls (95% below 0) on >= 4 of 5 layers at K48 prose AND >= 3 of 5 at K48 code. Secondary: K384.
If pass: one keeper query (needs a protocol extension: the method changes layer L-1 too; logged before use).

G10 (exploratory, `clean-g10.json`): ATTENTION FOLD (AF) = routes are layer L's attention head outputs (input of
o_proj); the route part is added to o_proj (0 extra numbers; attention is never coarsened, so AF works when every MLP
is coarsened). K48 prose, prose calibration: L15 cut .0544 / PLF .0338 / AF .0369 / AF+PLF .0265 (51%, beats the
20-min trained parent .028); L12 .0552 / .0437 / .0455 / .0382 (31%); L26 .1134 / .0888 / .1082 / .0868 (24%).
BUT on code text the prose-calibrated folds lose a little (L15 code: cut .0649, AF+PLF .0698; L26 .1100 / .1122).
MIXED calibration (64 prose + 64 code blocks from code_train.txt chars 400k+, no overlap with the dev code blocks)
fixes it: L15 prose cut .0574 / AF+PLF .0324 (44%); code .0468 / .0285 (39%).
Method name: RCF (route-capture fold) = cut top-K + AF + PLF, mixed calibration, CV ridge, sink guard.

G11 RCF CONFIRMATION (pre-registered 2026-10-20, before running). Layers 4, 7, 11, 16, 22, 25 (no CE test touched them
before). K = 48 and 384. Mixed calibration. Arms: CUT (same mixed calibration), AF (whole-model compatible), RCF
(AF + PLF). Text = dev prose blocks 0-39 and dev code (code_train.txt first 400k chars).
Pass: RCF better than CUT (95% below 0) on >= 5 of 6 layers for prose AND for code at K48. Secondary: K384; AF alone.
G9 result (`clean-g8.json` tags _g9): FAIL. PLF alone, prose calibration, K48 prose: L1 WORSE (+.049; layer 0 MLP is a
bad fold target), L10 better (-.016), L20 tie, L23 better (-.029), L27 better (-.029) = 3 of 5. K48 code: better only
at L23; L20 worse (+.015). K384 prose: better L10 / L23 / L27, tie L20, worse L1. Both failure causes (prose-only
calibration, layer-0 target) were found in G10 before this result was read, and G11 already fixes them.

G12 WHOLE-MODEL (pre-registered 2026-10-20, before running). `gate/g12.py`. Coarsen the MLP width of layers 1-29
except 6, 9, 21, 24 (25 layers; layer 0 intact) to frac = 0.5 and 0.25, first to last; each layer fitted on inputs
from the already-coarsened model, target = the original MLP on those inputs. Mixed calibration (32 prose + 32 code
blocks). Arms: CUT (top-K + LS), AF (+ attention fold at every coarsened layer), RCF (AF + fold into the previous
layer's KEPT neurons when that layer is coarsened). Scored by whole-model dev CE, prose and code, paired bootstrap.
Pass: RCF better than CUT (95% below 0) at both fracs on prose and code.
G12 result (`clean-g12.json`): FAIL. Whole model (25 layers), dCE prose / code: frac .5 CUT .804 / .432, AF .846 /
.455, RCF .845 / .463; frac .25 CUT 1.397 / .881, AF 1.459 / .886, RCF 1.416 / .869. The per-layer fold gains vanish.
Suspected method error: the target (original MLP on drifted inputs) only copies the MLP; it never repairs the drift
from earlier layers. Next: drift-correcting target Y* = h_orig(after layer L) - h_coarse(before MLP L) (standard in
layer-wise compression). Same arms, same pass rule (G12b, logged before running).

G11 result (`clean-g10.json` tags *_g11): PASS. RCF (AF + PLF, mixed calibration) vs CUT, share of the cut's damage
recovered, * = 95% below 0. K48 prose: L4 11%*, L7 18%*, L11 34%*, L16 38%*, L22 36%*, L25 28%* (6/6). K48 code:
31%*, 29%*, 40%*, 44%*, 41%*, 29%* (6/6). K384 prose: L4 -3% (tie), 7%*, 22%*, 26%*, 22%*, 16%* (5/6); code: 3%
(tie), 5% (tie), 15%*, 18%*, 26%*, 6%* (4/6). AF alone: K48 7-21% prose, 8-28% code, all significant.

KEEPER PROTOCOL EXTENSION (logged before use, 2026-10-20): a method may return a layer with
`.extra = [(owner, attr, module), ...]`; the evaluator sets owner.attr = module before scoring and restores it after
(RCF changes layer L's o_proj and layer L-1's down_proj). Stored numbers of the coarse layer = the cut's (3dK); the
extras add none for one coarse model. rcf_v1 dev check reproduces G11 (L16 K48 +0.0418, cut_ls +0.0650; restore exact).
KEEPER QUERY (family "route-capture", query 1 of 3), rcf_v1.py sha256 39b6fcd9d08af25b...
Prediction written before the run, K=48: L9 prose BETTER, code BETTER; L21 prose BETTER, code BETTER; vs trained:
"between" on both layers. K=384 (secondary): L9 even or better, L21 better.
G12b (drift-correcting target, frac .5): FAIL. dCE prose / code: CUT .816 / .450, RCF .809 / .478, AF .849 / .514.
RCF vs CUT prose -.007 (95% -.015..+.003), code +.028 (worse); AF worse on both. The dc target made the cut a little
worse too. Next facts needed: (1) single-layer RCF at 50% width; (2) does the gain compose where it is strong (middle
block 11-20 at K192), with a PLF-only arm (no attention change).
KEEPER VERDICT (validation, family "route-capture", rcf_v1 sha 39b6fcd9..., ledger: 4 entries; budget 2 of 3 used,
counted per (method, K)). K48: L9 prose BETTER / reaches-trained, L21 prose BETTER / reaches-trained, L9 code BETTER /
reaches-trained, L21 code BETTER / reaches-trained. K384: L9 prose EVEN, L21 prose BETTER, L9 code BETTER, L21 code
BETTER. Cover FAIL (expected: RCF keeps original neurons whole and builds no new parent units); shared write n/a.
Restore sanity passed. Prediction was "between"; the result is one bucket better. FIRST method on this track that
reaches the 20-min trained parents on sealed layers and sealed text with no gradient fit (one LS solve per layer).
Open: whole-model composition (G12/G12b fail at 50% width), nested dial storage.

G13 (exploratory): (1) single layer at 50% width (K768, mixed cal): RCF L16 cut .0186 -> .0146 (22%), L22 .0274 ->
.0241 (12%). (2) MIDDLE BLOCK layers 11-20 all coarsened to K192 (12.5%), sequential: dCE prose / code CUT .731 /
.593, PLF-only .666 / .539, RCF .589 / .448. RCF vs CUT -.142 (95% -.157..-.127) / -.145 (-.163..-.128): gains
COMPOSE across 10 layers. So G12's failure likely comes from early (1-8) / late layers where single-layer folds were
weak or harmful (L3 worse, L4 ties, layer-0 target catastrophic). Next (G14): whole model, RCF only on layers >= 10.
Nested-dial storage (`clean-g10.json` tags _rank/_rrr, K48 prose, mixed cal): RCF fold deltas truncated to rank r,
share of the cut's damage recovered L16 / L22: full 38% / 36%; r128 30% / 28%; r64 20% / 19%; r16 13% / 9%.
Reduced-rank regression (project onto top-r fitted output directions) is no better (r128 28% / 27%). Per dial
setting, rank 128 costs ~0.42M numbers (16% of a layer). The fold is spread over many directions (many small
routes). Free for one coarse model; a real cost for a many-setting nested dial.
G14 (whole model, folds only on layers >= 10, cut below): still no gain. frac .5 RCF vs CUT prose +.007 (95% -.000..+.015),
code +.016 (worse); PLF-only prose -.004 (tie), code +.012 (worse); frac .25 RCF prose tie, code -.016 (better).
Block 11-20 at K192 composed (-.14). Separating width vs earlier-coarsening vs late layers: (a) block 11-20 at K768,
(b) layers 10-29 at K768 with layers 1-8 intact.
G15: block 11-20 at K768 (50%), 32+32 cal: RCF vs CUT prose tie, code +.010 (worse); layers 10-29 at K768: prose tie,
code +.022 (worse). Calibration size test on block 11-20 K768: 64+64 blocks: CUT .196 / RCF .190 prose (-.006, 95%
-.011..-.001), code tie; 128+128 blocks: CUT .193 / RCF .178 prose (-.015, -.019..-.011), code .111 / .104 (-.007,
-.012..-.002). Reading: the whole-model failure at 50% width was mostly a calibration-starved fit (thousands of
regressors per layer, 64 blocks). More text -> more gain. Next: whole model 25 layers at 50% with 128+128 blocks.
G16: whole model 25 layers at 50%, 128+128 cal: CUT .739 / .390, RCF .737 / .388 (prose -.002, 95% -.010..+.006;
code -.002, -.014..+.009): TIE. More text lowered the cut's own cost (.80 -> .74). The block gain does not survive
whole-model coarsening. Next: folds only on 10-20 (cut elsewhere), and on 10-29.
G17: whole model 50%, 128+128 cal, RCF only on layers 10-20: prose -.0098 (95% -.016..-.003), code +.002 (tie);
RCF on 10-29: prose -.0098 (-.016..-.004), code -.003 (tie). Folds on early layers (1-8) cancel the gain. Whole-model
gain is real on prose but only ~1.3% of the cut's damage (per layer: 20-40%). Biggest open problem.
KEEPER BASELINE CORRECTION (budget-exempt, ledger entry cut_ls_mix sha f430c692...): the keeper's cut_ls and trained
parents use prose-only calibration, rcf_v1 uses prose + code, so the earlier CODE buckets were confounded. New fair
baseline `coarse/methods/cut_ls_mix.py` (same mixed calibration). rcf_v1 vs cut_ls_mix: BETTER in all 8 cells (L9,
L21 x K48, K384 x prose, code). cut_ls_mix vs cut_ls: code better (4/4), prose worse 3 / even 1. Route-capture
validation budget left: 1 of 3.
G18: whole model 50%, drift-correcting target, 128+128 cal: CUT_dc .718 / .404, RCF_dc .699 / .426. RCF_dc vs CUT_dc
prose -.019 (95% -.030..-.009), code +.022 (worse); vs the plain cut prose -.040 (better), code +.036 (worse).
A prose/code trade-off, not a fix. Night report: research/clean/NIGHT-REPORT-2026-10-20.md.

## RCF SCALE + CHANNELS (from 2026-10-20, plan ~/.claude/plans/okay-so-i-let-proud-taco.md)

Goal: make RCF work for the whole model, and one fair test of "channels" (new up/write pairs on a kept trunk's gate,
2/3 of a neuron each). Order: A0 -> B1 -> A1 -> A2. Rules: dev layers only, judge by dev dCE (prose 40 test blocks,
code 40 blocks of code_train.txt chars 0-400k), paired bootstrap 95%, keeper not queried this session.

A0 PER-LAYER MAP (pre-registered 2026-10-20, before running). Descriptive, no pass rule. `gate/a0.py`.
Single-layer CUT (top-K by energy + LS) vs RCF (rcf_v1 form: kept L hiddens + L-1 hiddens + L attention head
outputs -> original L output, CV ridge; PLF skipped at L = 1) at K768 (50%) and K384 (25%), every layer 1-29 except
6, 9, 21, 24. Mixed calibration 128 prose + 128 code blocks (g12 mixed_cal). Sink guard on all arms.
Outputs per layer: dCE cut / rcf, share recovered = 1 - rcf/cut, rcf - cut 95% interval.
Uses: CEILING = mean share recovered over the 25 layers (per K, per text). HELP set = layers with rcf - cut 95%
below 0 on prose AND code at that K. HURT set = 95% above 0 on either text. A2 folds only on the HELP set.

B1 CHANNELS (pre-registered 2026-10-20, before running). `gate/b1.py`. Shonu's idea: extra neurons that cancel the
leak / keep children sharp. Sane general form: a CHANNEL = new (up u, write d) pair that reuses a KEPT trunk's gate:
output silu(g_i x) (u . x) d. Stored 2d numbers (2/3 of a neuron); the gate is shared, not stored again.
Method rcf_ch: (1) RCF at K as in A0. (2) Residual R = Y - RCF fit on calibration rows. (3) Per kept trunk i: F_i =
silu(g_i x) * x; full LS of R on F_i; rank-r under the metric F_i^T F_i (G1 closed form) gives candidate channels and
their gains sigma^2 (rows subsampled 1 in 4 for this step). (4) Top E candidates by gain across all trunks. (5) One
joint CV-ridge refit of [kept hiddens, channel activations, L-1 hiddens, L attention outputs] -> Y; the route parts
fold as in RCF. Same mixed calibration as A0 (128 + 128).
Control (the only one that counts): EQUAL STORED NUMBERS, RCF at K' = K + ceil(2E/3).
Layers 12, 15, 16, 22. Budgets: K48 E18 (K' 60), K48 E72 (K' 96), K384 E144 (K' 480).
Screen (pre-filter, not a verdict): held-out calibration rows (last 20% of each half held out of the channel
selection and the refit), relative output error. If rcf_ch is worse than RCF@K' locally at EVERY layer and budget,
stop before dCE.
Pass: rcf_ch - RCF@K' 95% below 0 on >= 3 of 4 layers, prose AND code, at one budget.
Fail: log "channels as built, fail" and stop. No bigger run of the same idea.

A1 PLACEMENT CHECK (pre-registered 2026-10-20, before running). `gate/a1.py`. Block 11-20 (no keeper layers),
K768, sequential as g12, 128 + 128 calibration. Arms: CUT; RCF fold (g12 placement: route parts into layer L-1's
write matrix and layer L's o_proj); RCF oracle (same fitted route parts written AFTER MLP L, so layer L never reads
them; extra storage, upper bound only). Reading: gain = CUT - arm (dCE).
Rule: if oracle gain >= 1.5 x fold gain on prose AND code, the stale-input effect is real -> run RCF refit (hold the
L-1 fold fixed, recapture layer L inputs, refit only kept writes and the attention fold to the local drift-correcting
target Y_pre - Cp hp - Wo (z' - z)); refit passes if better than fold (95% below 0) on prose and code. Otherwise go
to A2 with the fold placement.

A2 DOWNSTREAM WINDOW (pre-registered 2026-10-20, before running). `gate/a2.py`. Whole model (layers 1-29 except 6,
9, 21, 24), sequential. Per layer: closed-form start, then 1.5 min Adam (lr 3e-4, batch 4) on 64 prose + 64 code
calibration blocks, loss = relative MSE of the residual after layer L+3 vs the teacher (ORIGINAL layers L..L+3 on the
same drifted input of layer L), position 0 excluded. Arms: CUT-W (free: kept writes only; the fairness control) and
RCF-W (free: kept writes + L-1 fold + attention fold; folds only on the A0 HELP set at that width, cut elsewhere).
frac 0.5 and 0.25. Sanity first: at 0 minutes, RCF-W on block 11-20 must equal A1 fold.
Pass: RCF-W - CUT-W 95% below 0 on prose AND code at one frac, AND the share of CUT-W's damage recovered is >= 1/3
of the A0 ceiling (mean single-layer share at that K, per text).

A0 result (`results/clean-a0.json`, single layer, 128 + 128 cal). Share of the cut's damage recovered by RCF, mean
over 25 layers: K768 (50%) prose 6.0% / code 5.5%; K384 (25%) prose 13.1% / code 11.5%. (K48 in G11: 11-44%.)
HELP (95% below 0 on prose AND code): K768 {19, 22, 27, 28, 29}; K384 {7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 22, 26,
27, 28, 29}. HURT (95% above 0 on either text): K768 {2, 3, 4, 14, 23}; K384 none.
Reading: the gain shrinks with width at EVERY layer. At 50% the removed half is weak, diffuse output and the routes
capture little of it. So the whole-model 50% result (G17: ~1.3%) is mostly a low ceiling (~6%), not mainly a
composition failure: composition kept about 1/4 of the ceiling. Layers 1-5 lose at 50% (L2-L4 significantly); at
25% early layers are mostly ties, not losses. Code results at K768 are noisy (L14, L20, L23 lose on code only).
CEILINGS for A2: frac .5 prose 6.0% / code 5.5%; frac .25 prose 13.1% / code 11.5%.

B1 result (`results/clean-b1.json`): FAIL by the letter. rcf_ch is WORSE than equal-storage RCF@K' in all 12 cells
(4 layers x 3 budgets), prose and code, all 95% intervals above 0 (+.001 to +.010). Screen also fails everywhere.
Diagnosis (method error): the channel selection overfits. Candidates claim 7-54% of RCF's residual in-sample, but
held-out error barely moves (L12 K48: E18 .4379, E72 .4375). Each trunk's full LS has 576 x 576 free numbers, fit
on the rows where its gate is open, with ridge 1e-3. The chosen u vectors are mostly noise, so the refit gives them
little weight.
B1b (pre-registered 2026-10-20, before running; one fix of the step, same test): (1) channel up vectors restricted to
the top 64 principal directions of the layer input (64 x 576 per trunk); (2) candidates fit on train rows 0::4,
GAIN MEASURED on held-out train rows 2::4 (write refit on the fit rows, gain on the held-out rows); top E by held-out
gain. Everything else as B1 (layers, budgets, control, pass rule). If B1b fails too: channels as built, fail. Stop.
B1b result (`results/clean-b1b.json`): FAIL. Honest selection works as a fix of the step (held-out error now drops
with channels: L12 K48 .4379 -> .4348 / .4299), but channels capture only 1-5% of RCF's residual on held-out rows and
rcf_ch is still WORSE than equal-storage RCF@K' in all 12 cells, prose and code (+.0007 to +.0071, all 95% above 0).
Verdict: channels on kept trunk gates, as built, fail. A whole kept child is worth more than 1.5 channels. Stopped
per the pre-registration.
Reading (tree terms): RCF's leftover sits where REMOVED children fired. A kept trunk's gate is mostly closed there
(taxi's gate does not open on trucks), so a linear read under that gate cannot reach it. Same lesson as G1: SmolLM2's
gates are not shared enough. Untested variant (not run, logged only): channels under UPSTREAM route gates (layer
L-1 neurons), where the "car" gate would sit.

A1 result (`results/clean-a1.json`, block 11-20, K768, 128 + 128 cal). dCE prose / code: CUT .1932 / .1107; RCF fold
.1780 / .1042 (gain .0152 / .0064, both 95% below 0; reproduces G15); RCF oracle .1633 / .0933 (gain .0299 / .0174).
Oracle / fold gain = 2.0x prose, 2.7x code: >= 1.5x on both, so the stale-input effect is REAL. Writing the folds
before MLP L (into L-1's write matrix and L's o_proj) loses about half the gain, because layer L reads them.
Oracle recovers 15.5% / 15.7% of the cut's block damage. Next: RCF refit (pre-registered above).
A1 refit result: dCE prose / code .1751 / .1026. Refit - fold: prose -.0028 (95% -.0052..-.0004), code -.0017
(-.0035..+.0004, tie). FAIL by the letter (needs both). It closes only ~1/5 of the fold-to-oracle gap. The stale
part is not only the kept writes and the attention fold: the folds also change which kept neurons fire and how
attention L attends. A2's window fit trains the folds through the real path, so it can see this effect.

A2 sanity: 0 minutes, block 11-20, frac .5: dCE .1780 / .1042 = A1 fold exactly. PASS.
A2 smoke (exploratory, block 11-20, frac .5, 0.5 min per layer): CUT-W .1557 / .1016 (CUT closed form .1932 / .1107);
RCF-W .1347 / .0870 (fold closed form .1780 / .1042; oracle .1633 / .0933). RCF-W - CUT-W: prose -.0210 (95%
-.0245..-.0172), code -.0147 (-.0190..-.0101). The window fit helps both arms; the fold adds ~14% of CUT-W's damage
on top. The fold is now worth MORE than in closed form (block gain .015 / .006 -> .021 / .015).
Note: A0's HELP set at K768 is only {19, 22, 27, 28, 29}, but folds on 11-20 help a lot once window-fitted. The
pre-registered A2 runs keep the HELP rule. EXPLORATORY extra arm (logged before running): RCF-W with folds on ALL
coarsened layers, both fracs.
A2 run order: frac .5 CUT-W, frac .5 RCF-W (HELP), frac .25 CUT-W, frac .25 RCF-W (HELP), then exploratory
RCF-W (all), frac .5 and .25. 1.5 min per layer.

A2 whole-model chain STOPPED by Shonu (4.5 h too long). New order: resolve the two open problems with quick
single-layer iteration first. (1) RCF weaker at wide K: treat as a likely bug. (2) The fold disturbs its own layer.
Hypothesis: same bug. The disturbance is read by the KEPT neurons of layer L, and there are more of them at wide K.

Q1 (pre-registered 2026-10-20, before running). `gate/q1.py`. Single layer, layers 12, 16, 22, 27; K 48, 384, 768;
128 + 128 cal. One RCF fit per (layer, K); arms differ only in where the route parts are written: fold, oracle (both
after MLP L), oracle_p (only the L-1 fold moved), oracle_a (only the attention fold moved). Share of the cut's damage.
Reading rule: if the oracle's share at K768 is >= 2x the fold's AND the oracle's share falls with K much less than
the fold's (oracle K768/K48 ratio >= 1.5x the fold's), the width problem is mainly the disturbance. oracle_p vs
oracle_a says which fold does the harm.
Q1 result (`results/clean-q1.json`). Share of the cut's damage, fold -> oracle (prose / code):
K48: L12 26->33 / 40->43, L16 38->41 / 45->50, L22 38->41 / 44->47, L27 34->35 / 38->41.
K384: L12 12->20 / 14->24, L16 28->31 / 18->37, L22 24->29 / 26->35, L27 20->22 / 23->29.
K768: L12 17->25 / 11->28, L16 19->25 / 3->37, L22 14->20 / 34->46, L27 14->19 / 17->23.
(1) The disturbance grows with width (oracle - fold ~3 points at K48, 5-34 at K768). Code suffers most.
(2) oracle_p ~ oracle in every cell: the L-1 fold (written before attention L AND MLP L) causes almost all of it.
The attention fold's placement costs little.
(3) Reading rule: NOT met (oracle K768 >= 2x fold only on code at L12, L16; oracle also falls with K: prose
33-41% -> 19-25%). So the disturbance is a real bug but only PART of the width problem.
Next, smallest versions (Shonu 2026-10-20: default to the smallest test, no long runs on the Mac):
Q2 (pre-registered): zero-storage fixes, layers 16, 22, K 384, 768. Arms: fold, f3_1/f3_2 (install, recapture
layer L's real inputs, solve [hk', hp, z'] -> D h + Wo (z - z'); one more round), f1 (15 s Adam of Dk, Cp, Ca
through the real layer L, window 0). Pass: a fix reaches >= 2/3 of the oracle's gain over fold (Q1 numbers) in
>= 3 of the 4 cells, prose AND code.
Q3 (pre-registered): is the rest natural? Local, no dCE. Layers 16, 22; K 48, 384, 768. Held-out R^2 of the REMOVED
neurons' summed output predicted from the routes (L-1 hiddens + attention outputs), cal 128+128 vs 256+256 (data
starvation check). If R^2 falls with K about as much as the oracle share does, the rest is natural (weak neurons are
less predictable). If R^2 is flat, or the train/held-out gap grows with K, it is a fixable bug (overfit / starved).
Q2 result (`results/clean-q2.json`). Share of the cut's damage (prose / code), fold | f3_1 | f3_2 | f1 (Q1 oracle):
L16 K384 28/18 | 29/29 | 30/31 | 34/35 (31/37); L16 K768 19/3 | 22/8 | 23/9 | 24/20 (25/37);
L22 K384 24/26 | 26/30 | 26/28 | 29/37 (29/35); L22 K768 14/34 | 13/37 | 11/33 | 17/37 (20/46).
f1 reaches >= 2/3 of the oracle's gain on prose 3/4, code 2/4: FAIL by the letter (code). f3 helps a little only.
Caveat: f1 also fits, so its gain is not only the placement fix. Control needed (Q4).
Q3 result (`results/clean-q3.json`). Held-out R^2 of the removed output from the routes (n128 / n205):
L16 K48 .469/.479, K384 .420/.430, K768 .384/.395; L22 K48 .482/.491, K384 .417/.427, K768 .386/.396.
Train-test gap ~.05 at every K; more data adds ~.01. Not starved, not a growing overfit.
R^2 falls ~20% from K48 to K768, but the oracle's loss share falls 39-51% on prose. Reading: part of the width drop
is natural (weak neurons are a bit less predictable), part is OBJECTIVE MISMATCH: the local MSE treats all positions
and directions alike, and at wide K what it captures matters less for the loss. Consistent with f1 and the A2 window
fit working.
Q4 (pre-registered): f1 control. cut_f1 = the cut with the same 15 s fit of its kept writes (no routes). Cells as
Q2. Reading: f1 - cut_f1 vs fold - cut (closed form). If the fold's value grows with the fit, the fit fixes the
disturbance; if not, f1's gain is just the fit.
Q5 (pre-registered): loss-weighted rows. Row weight w_t = squared norm of d(next-word loss)/d(MLP L output) at
position t (one backward pass on the cal blocks, original model), normalized to mean 1; also sqrt weights. Closed
form, fold and oracle placement. Layers 16, 22, K768. Pass: weighted fold beats plain fold (95% below 0) on prose AND
code in both layers.
Q4 result (`results/clean-q4.json`): cut_f1 (the cut + the same 15 s fit, window 0) recovers -5..+3% of the cut's
damage: nothing. f1 vs fold: L16 K384 35/34 vs 28/18, K768 23/20 vs 19/3; L22 K384 29/37 vs 24/26, K768 17/37 vs
14/34. So f1's gain is ALL from fitting the folds through the real layer. f1 = a working zero-storage fix for the
disturbance (code at L16 K768: 3% -> 20%).
Q5 result (`results/clean-q5.json`): FAIL. Loss-weighted rows do not help (w - plain: 4 of 4 ties or worse; full
weights hurt L22 prose +.0028). Note why a FIXED output weighting cannot help: the unconstrained LS solution is the
same for any fixed output metric. Only the window fit (later layers judge the error) changes what is fitted.
Q6 (pre-registered): does the window objective remove the width drop? Single layer, layers 16, 22, K 48 and 768.
Arms: fold (closed form), fw3 = f1 with a 3-layer window (15 s), cut_w3 = the cut + the same window fit (control).
Reading: share of the cut's damage recovered by fw3 relative to cut_w3. If the K768 share is >= 2/3 of the K48
share, the width drop was mostly the objective; otherwise it is mostly natural.
Q6 result (`results/clean-q6.json`, window 3, 15 s). RCF-W share of CUT-W's damage (prose / code): L16 K48 51/53,
K768 25/29; L22 K48 43/49, K768 13/39. The window fit makes RCF much stronger at every width, but K768 keeps only
30-80% of the K48 strength (rule: >= 2/3 -> mostly objective). Mostly NOT the objective; the drop remains.
Q7 (pre-registered): route-aware choice of kept neurons. Hypothesis (design bug): RCF keeps the top-K by energy,
ignoring the routes. Keep instead the top-K by UNEXPLAINED energy (|h_j - route prediction| on held-out rows x
|d_j|), so the routes repair what they can and the kept neurons hold what they cannot. Closed form, layers 16, 22,
K 48 and 768, fold and oracle placement, same stored numbers. Pass: route-keep fold beats energy-keep fold (95% below
0) on prose AND code at K768 in both layers.
Q7 result (`results/clean-q7.json`): FAIL. Route-aware keep vs energy keep, fold, K768: L16 prose +.0006 (tie), code
+.0008 (tie); L22 prose +.0001 (tie), code +.0008 (tie). The two rules pick 85-89% of the same neurons at K768. At K48
(overlap 58-67%): L22 prose better (-.0059), L16 prose worse (+.0012), code ties. About half of each neuron's energy
is predictable from the routes (median .49 on both layers).

WIDTH-DROP VERDICT (Q1-Q7). Suspects tested, smallest tests: (1) the fold disturbing its own layer: REAL, grows with
K, fixed by f1 (a 15 s true-path fit, zero storage); (2) data starvation: no; (3) position weighting: no (and a fixed
output weighting cannot help in principle); (4) wrong objective: the 3-layer window helps at EVERY width (RCF-W vs
CUT-W ~43-53% at K48) but K768 still keeps only 30-80% of that; (5) keep rule ignoring routes: no.
Reading: after fixing (1), the rest looks mostly natural. At 50% width the removed half is weaker and a little less
predictable from the routes (held-out R^2 -20%), and what the routes capture matters less for the loss.
Report: research/clean/RCF-REPORT-2026-10-20.md.
Q8 CEILING (pre-registered 2026-10-20, before running). Is the width drop natural? Reference = a free trained fit
(shortfit3: cut start, ALL of G, U, D trained, 3-layer window, 2 min, mixed calibration interleaved prose/code).
Layers 16, 22; K 48 and 768. Share of the closed-form cut's damage. Rule: NATURAL if the trained share at K768 is
<= 2/3 of its K48 share on both layers (prose and code averaged); otherwise a real gap remains for RCF to close.
Q8 result (`results/clean-q8.json`). Trained (2 min, all G U D free, window 3) share of the closed-form cut's damage,
prose / code: L16 K48 36/37, K768 31/34; L22 K48 25/18, K768 11/23. K768/K48 (mean of texts): L16 .89, L22 .79.
By the letter: NOT natural (> 2/3 on both layers). BUT the reference is not a ceiling: RCF-W (Q6 f1, 15 s, window 3)
beats it in all 8 cells (vs the same closed-form cut: L16 K48 56/59, K768 36/34; L22 K48 47/52, K768 16/42). So Q8
cannot settle "natural". What it does show: at K768 RCF-W matches or beats a 2-min trained layer; at K48 it beats it
by 20-34 points. RCF's "width drop" is mostly RCF being unusually strong at narrow widths, not weak at wide ones.
A real ceiling needs a much longer trained reference (GPU-scale run, later).

W0 WHOLE MODEL, FIXED DESIGN (pre-registered 2026-10-20, before running; Shonu approved ~30 min). `gate/a2.py`,
frac .5, every layer 1-29 except 6, 9, 21, 24, sequential, 15 s Adam per layer, window 3, 64 + 64 fit blocks.
Arms: CUT-W (kept writes only) and RCF-W (kept writes + L-1 fold + attention fold, folds on ALL coarsened layers).
Pass: RCF-W - CUT-W 95% below 0 on prose AND code. Report the share of CUT-W's damage recovered.
W0 CANCELLED by Shonu before any result: first a concrete account of the width effect and a check at widths above 50%.

W1 WIDTH ACCOUNT (pre-registered 2026-10-20, before running). `gate/w1.py`. Goal (Shonu): a concrete mechanism for
RCF's width drop, not a hypothesis. Single layer, layers 16, 22, K 48, 192, 384, 768, 1152, 1344; ORACLE placement
(the disturbance is excluded here; Q1 measured it). Exact identity on the dev blocks:
s_loss = 1 - (kappa_r / kappa_c)(1 - s_loc), kappa = dCE per unit error energy. So the drop is split into (i) s_loc
(how much error energy RCF removes) and (ii) kappa_r / kappa_c (whether the leftover costs more loss per unit).
(i) is explained by a commonality split of the removed output (kept-only / routes-only / shared / neither; RCF can
only add routes-only). (ii) by injecting the repair alone (kappa_rho) and the cross term. Half-cut injection checks
whether loss is quadratic in the error.
Reading rules: the factor whose change K48 -> K1344 accounts for more of the drop in log(1 - s_loss) is "the" cause.
If routes-only falls while shared rises, the cause of (i) is overlap with the kept neurons (not a bug).
W2 TREND ABOVE 50% (pre-registered): L16, L22 at K1152 (75%): CUT-W and RCF-W (15 s, window 3, `q2.py --q4`) and the
2-min trained layer (`q8.py`). Concern (Shonu): RCF falls below training as width grows. Report all three vs the
closed-form cut, next to K48 and K768.
W1 result (`results/clean-w1.json`). (ii) kappa_r / kappa_c is <= ~1 in 22 of 24 cells (.26-1.03): the leftover
costs the SAME or LESS loss per unit than the cut's error. Loss weighting is NOT the cause; at wide K on code it even
helps RCF. All of the drop is (i) the local share s_loc: K48 39-44% -> K1344 14-16%.
(i) is explained exactly by the commonality split. s_loc ~= routes_only / (routes_only + neither) holds within .04
in all 12 (layer, K) cells. Removed output, L16 (L22 alike), K48 -> K1344: routes-only .38 -> .08, shared .08 -> .24,
kept-only .005 -> .07, neither .53 -> .61. Route-predictable total (routes-only + shared) .47 -> .32.
MECHANISM: the cut's LS refit of the kept neurons already captures whatever the kept neurons and the routes BOTH
know (shared). With more kept neurons, the shared part grows and the part only the routes know shrinks. RCF can add
only that part. Plus a smaller effect: weaker removed neurons are a bit less route-predictable (.47 -> .32 total).
The "neither" part (no linear fold of kept neurons + routes reaches it) is 52-61% at EVERY width.
Verdict: NOT a bug. RCF reaches its linear ceiling at every width (measured s_loc ~ predicted). The only remaining
headroom is the "neither" part. Half-cut injections give .14-.32 of the full cut's loss (quadratic would be .25):
roughly quadratic, so the identity reading is sound (except K1344 code, where dCE is tiny).
W2 result (`results/clean-w2.json`, `clean-q8.json`, K1152). Share of the closed-form cut's damage, prose / code:
RCF-W L16 33/23, L22 17/25; trained (2 min) L16 29/33, L22 9/10; CUT-W L16 11/9, L22 1/-9. RCF-W beats training in 3
of 4 cells (not L16 code). Compared with K768 (RCF-W L16 36/34 vs trained 31/34; L22 16/42 vs 11/23): no clear trend
of RCF falling below training; noisy, since the cut's damage at K1152 is tiny (.003-.008). The closed-form fold HURTS
at L16 K1152 code (-4%, the disturbance), and f1 repairs it (23%).
W3 (pre-registered): what reaches the "neither" part? Local held-out R^2 with extra sources: layer L-2 hiddens and
layer L-1 attention outputs (both foldable at zero storage), and the MLP input x itself (linear; costs a 576 x 576
map). Layers 16, 22; K48, K768. Reading: a source is worth building only if it cuts "neither" by >= .05 absolute.
W3 result (`results/clean-w3.json`). "neither" share of the removed output (lower = more reachable), base -> +l2 /
+z1 / +x / +all: L16 K48 .529 -> .464 / .515 / .493 / .438; L16 K768 .564 -> .515 / .556 / .541 / .498; L22 K48
.522 -> .444 / .509 / .492 / .424; L22 K768 .569 -> .517 / .560 / .554 / .505.
Only layer L-2's neurons pass the bar (-.05 to -.08); L-1 attention (-.01) and x (-.02/-.03) do not.
Predicted effect (s_loc ~ routes_only / (routes_only + neither)): K768 ~.19 -> ~.26, K48 ~.42 -> ~.49.
Next (proposed, not run): "two-back RCF": routes from L-1 AND L-2 neurons (both folded into existing write matrices,
zero storage), fitted through the real path (f1) so the disturbance through layer L-1 is handled. Smallest test:
L16, L22 at K48 and K768, oracle placement first, then fold + f1.
W4 ANY-LAYER ROUTES (pre-registered 2026-10-20, before running; Shonu's idea: any earlier neuron may be a route,
pick the best). `gate/w4.py`. Local only. Odd positions only (memory). Score every MLP neuron of layers 1..L-2 by the
leftover energy it explains alone (calibration rows); arms add them to today's RCF features; held-out "neither"
share on dev rows. Arms: l2_full (1536), l2_top512, any_top512 / 1536 / 4096 (global ranking, 512 candidates per
layer). Layers 16, 22; K48, K768. Reading: any-layer selection is worth building only if any_top1536 beats l2_full
(same count) by >= .03 "neither", or any_top4096 cuts "neither" by >= .05 more than l2_full.
W4 result (`results/clean-w4.json`). "neither": base / l2_full / any_top1536 / any_top4096: L16 K48 .531 / .468 /
.467 / .431; L16 K768 .567 / .519 / .518 / .488; L22 K48 .526 / .449 / .446 / .407; L22 K768 .574 / .524 / .523 /
.493. FAIL both bars: any-layer selection TIES all of L-2 at equal count (diff <= .003), and 4096 selected routes
cut only .031-.042 more than l2_full. Selected neurons come mostly from L-2 and L-3, thinly back to layer 1. Far
routes would also disturb every layer in between. Verdict: two-back (L-1 + L-2) is the sensible route set.
A linear-route floor remains: ~.40-.50 of the removed output is not linear in ANY earlier neurons (the removed
neuron's own gate on its own input). Reaching it needs nonlinear units, i.e. storage.
W5 (pre-registered 2026-10-20, before running). Why can't earlier routes reach "neither"? Claim: the removed
neurons' PRE-activations (g.x, u.x: linear reads of the stream) are reachable by routes; their OUTPUTS (after the
gate nonlinearity) are not. Test: L16, K768, removed neurons. Held-out R^2 (dev rows) from routes (L-1 + L-2 hiddens
+ L attention outputs; no x) of (a) gate pre-activations, (b) up pre-activations, (c) neuron outputs h_j, each
averaged over removed neurons weighted by output energy. Claim holds if (a), (b) >= .8 and (c) <= .6.
W5 result (`results/clean-w5.json`, L16 K768): held-out R^2 from routes (L-1 + L-2 hiddens + L attention):
gate pre-activations .884, up pre-activations .868, neuron outputs .431 (energy-weighted; medians alike). CLAIM HOLDS.
The routes know almost everything the removed neurons READ; the unreachable part is the removed neuron's own gate
bend at layer L. A fold adds only linear mixes, so no route depth reaches it. Reaching it needs a new nonlinear unit
(gate direction + write = storage). The cheapest such unit tried (B1 channels) lost to whole kept neurons.
Design: two-back RCF for the reachable part (zero storage) + kept neurons for the rest.

EFFICIENCY AUDIT (Shonu 2026-10-20: "how do we know we capture the gains we can?"). Run before two-back RCF.
V1 BUCKET AUDIT (pre-registered, before running). `gate/v1.py`, L16, K48 and K768, oracle placement. The leftover
splits exactly per removed neuron (LS is linear in the target). Per removed neuron: pre_r2 = min(R^2 of gate and up
pre-activations from RCF's features), out_r2 = R^2 of its output. BEND = pre_r2 >= .8 (input reachable, output not),
MISSING = pre_r2 < .8 (input not reachable). Each group's leftover is injected alone (dCE). Reading: if BEND holds
most of the leftover's loss, the bucket is mostly "needs a new unit"; if MISSING does, better routes could help.
V2 EFFICIENCY LADDER (pre-registered, before running; ~25 min). `gate/v2.py`, L16, K48 and K768. Zero extra stored
numbers in every rung; start = closed-form RCF (A); 3-layer window from the input of layer L-1. B = RCF's own weights,
15 s. C = same weights, 150 s / 300 s. D = everything in layers L-1 and L except norms, 150 s / 300 s. Share of the
closed-form cut's damage. Readings (per K, mean of prose and code): C - B >= 5 points -> our fast fit leaves gains
(method bug); D - C >= 5 points -> the fold class is too narrow; if C(300) - C(150) >= 3 points or D(300) - D(150) >= 3
points, that rung has not levelled off and its gap is a lower bound.
V1 result (`results/clean-v1.json`, L16). Input-reading R^2 median .82 (K48) / .85 (K768); output R^2 median .43 both.
K48: BEND 1077 neurons, 75% of leftover energy, dCE .0281 / .0201 of .0354 / .0248 (~80% of the loss); MISSING 411,
25% energy, .0072 / .0042 (~20%). K768: BEND 755 of 768 neurons, 99% energy, ~all the loss. Caveat: the .8 threshold
is arbitrary and the MISSING group sits just below it (q10 .77), so "missing" means "somewhat less reachable".
Reading: the leftover is almost all bend novelty (input readable, output not), most clearly at wide K.
V2 result (`results/clean-v2.json`, L16). Share of the closed-form cut's damage, prose / code:
K48: A 38/45, B 56/60, C150 58/59, C300 57/58, D150 66/65, D300 67/64.
K768: A 19/3, B 37/33, C150 38/35, C300 38/32, D150 61/40, D300 51/36.
(1) C - B ~ 0 (-1..+2): the 15 s fit already gets everything RCF's own weights can give. No fitting bug.
(2) D - C = +6..+9 at K48; +5..+23 at K768 (both checkpoints >= 5 on the mean): the FOLD CLASS IS TOO NARROW.
Letting the other zero-storage weights near layer L move (gates, ups, attention) gains more. So the "neither" bucket
is not all untouchable at zero storage: existing bends can be re-aimed to cover part of the removed bends.
(3) D has not levelled off and is unstable at K768 (prose 61 -> 51 from 150 s to 300 s while its fit loss still fell:
overfitting the 128 fit blocks). Its gap is real at both checkpoints but its size is uncertain.
Next (proposed): split D to find which weights carry the gain: D_gate = C + kept G, U at L; D_prev = C + L-1 MLP G,
U; D_attn = C + attention q, k, v (L-1, L) + L-1 o_proj. L16, K48 and K768, 150 s each.
V3 SPLIT OF RUNG D (pre-registered 2026-10-20, before running). `gate/v2.py --rungs C,Dgate,Dprev,Dattn --secs 150`.
L16, K48 and K768, 150 s each, same start and window as V2. Dgate = C + kept G, U at L; Dprev = C + L-1 MLP G, U;
Dattn = C + q, k, v of L-1 and L + L-1 o_proj. Reading: an arm "carries" the gain if it reaches >= 2/3 of
(D150 - C150) from V2 on the mean of prose and code, at both K.
S1 SHARED L-1 GATES (pre-registered 2026-10-20, before running; runs after V3). `gate/s1.py`. Shonu's flexible-vs-
fixed rule: per-width gate changes are the "fixed" side. Test: one SHARED set of layer L-1 gate/up changes for K48
and K768 together (per width: kept writes, L-1 write matrix, o_proj fold). 300 s joint (two forwards per step, so
about the same steps per width as V3's 150 s). L16. Reading per width: kept fraction = (S1 - C150) / (Dprev150 -
C150), mean of prose and code, V3 numbers. Shared is "flexible enough" if the kept fraction >= 2/3 at both widths.
V3 result (`results/clean-v3.json`, L16, 150 s). Share of the closed-form cut's damage, prose / code:
K48: C 58/60, Dgate 59/61, Dprev 65/64, Dattn 61/62 (V2 D150 66/65). Dprev carries ~85% of D's gain at K48.
K768: C 39/35, Dgate 47/42, Dprev 46/40, Dattn 40/41 (V2 D150 61/40, D300 51/36). At K768 the gain is split:
kept gates +8/+7 and L-1 gates +7/+5 each; neither reaches 2/3 of D150's gain (D150 itself is unstable).
Reading: with few kept neurons the parents' (L-1) gates do the work; with many, re-aiming the kept gates pays too.
Attention adds a little, mostly on code at K768.
S1 result (`results/clean-s1.json`, L16, 1331 joint steps). Shared L-1 gates, share prose / code: K48 65/63 (V3 Dprev
65/64, C 58/60): kept fraction ~.87. K768 43/33 (Dprev 46/40, C 39/35): kept fraction ~.09 (code below C). FAIL at
K768. Suspected test error: the joint loss sums relative losses, and K48's is ~3x K768's, so the shared gates mostly
serve K48. S1b (pre-registered, one fix of the step): each width's loss divided by its start value (mean of 8 fixed
batches), same everything else, same reading rule.
S1b result (balanced, 859 joint steps: fewer than S1's 1331, a confound). K48 63/62 (kept ~.61), K768 46/34 (prose
all kept, code none; kept ~.4). FAIL by the letter at both widths. Sharing one set of L-1 gate changes across widths
costs part of the gain, mostly code at wide K. Not pursued further now.
G (pre-registered 2026-10-20, before running). RCF-G = closed-form RCF start + window fit with RCF's weights AND the
kept G, U at L AND layer L-1's G, U (= Dgate + Dprev, no attention). Gcut = same free weights, start from the cut
(folds start at zero). `v2.py --rungs ... --secs 150`, checkpoints 75 s / 150 s. L22: C, Dgate, Dprev, G, Gcut (also
replicates V3); L16: G, Gcut. K48 and K768.
Readings (mean of prose and code): G - C >= 5 on both layers and widths -> add gate fitting to the method. G - Gcut
>= 3 -> the closed-form RCF start matters beyond the weight class; else RCF is mainly a good start. G150 - G75 >= -3 ->
stable. L22 replicates V3 if Dprev >= Dgate at K48.
T1 TWO-BACK RCF (pre-registered 2026-10-20, before running; runs after G). `gate/t1.py`. Routes = L-1 AND L-2 MLP
hiddens + L attention. Arms: A1 (one-back closed form), B1 (A1 + 15 s window fit), A2o (two-back, oracle placement,
upper bound), A2 (two-back fold placement), B2 (A2 + 15 s fit, L-2 write matrix also free). Window from the input of
L-2 to L+3. L16, L22; K48, K768. Pass: B2 - B1 >= 3 points (mean of prose and code) in >= 3 of 4 cells. W3's
prediction: two-back raises the local share by ~.07, i.e. roughly +5..+7 points of the loss share.
G result (`results/clean-g-l22.json`, `clean-g-l16.json`). Share prose / code (150 s unless noted):
L22 K48: C 48/53, Dgate 49/55, Dprev 58/64, G 59/65 (75 s 58/64), Gcut 57/62 (75 s 51/55).
L22 K768: C 19/39, Dgate 23/46, Dprev 27/34, G 28/47 (75 s 28/47), Gcut 28/42 (75 s 26/40).
L16 K48: G 67/64 (75 s 65/65), Gcut 64/64 (75 s 62/62); L16 K768: G 55/34 (75 s 50/40), Gcut 50/41 (75 s 51/37).
(1) G - C (mean): L22 +11.5 / +8.5, L16 +6.5 / +7.5: PASS (>= 5 everywhere). Add gate fitting to the method.
(2) G - Gcut at 150 s: +2.5, +2.5, +1.5, -1.0: below 3 everywhere. Once the gates move, the closed-form RCF start
mainly speeds convergence (at 75 s the gap is up to +8). The route idea stays in the class (Gcut learns its folds
through the free L-1 write matrix and o_proj); the closed form just starts it there.
(3) Stable: G150 - G75 >= -3 on the mean in all cells. (4) L22 replicates V3 (Dprev > Dgate at K48; split at K768).
G ~ full D at K48 (L16 D150 66/65), below D150 at K768 (D150 itself unstable).
G75 (pre-registered 2026-10-20, runs after T1): RCF-G at K1152 (75%), L16 and L22, 150 s (`v2.py --rungs C,G`).
Compare with W2 at K1152 (RCF-W L16 33/23, L22 17/25; 2-min trained L16 29/33, L22 9/10). Shonu's trend concern: does
the method fall below training as width grows? Reading: G >= trained on the mean of prose and code on both layers ->
no sign of the trend.
T1 result (`results/clean-t1.json`). Share prose / code: L16 K48 A1 38/45, B1 54/58, A2o 48/56, A2 40/47, B2 58/62;
L16 K768 A1 19/3, B1 34/35, A2o 32/37, A2 8/-19, B2 28/17; L22 K48 A1 38/44, B1 46/52, A2o 50/57, A2 43/49, B2 54/59;
L22 K768 A1 14/34, B1 19/43, A2o 28/57, A2 17/20, B2 5/44.
Potential is real: closed-form oracle two-back beats one-back by +10..+23 (more than W3's local prediction). But B2 -
B1: +4/+4, -6/-18, +8/+7, -14/+1: FAIL (2 of 4 cells). At K768 the L-2 fold disturbs layers L-1 and L (many kept
neurons read it) and 15 s cannot repair it; even closed-form A2 is below A1 there.
T1b (pre-registered, one fix of the step; runs after G75): the failing cells (L16, L22 at K768) with a 150 s fit for
both B1 and B2. Pass: B2 - B1 >= 3 (mean) in both cells.
G75 result (`results/clean-g75.json`, K1152). Share prose / code: L16 C150 45/12, G75s 58/21, G150 52/37 (trained 2-min
29/33); L22 C150 9/30, G75s 34/24, G150 -11/21 (trained 9/10). Reading rule FAILS on L22: G150 < trained, and G is
UNSTABLE at 75% width (L22 prose 34 -> -11 from 75 s to 150 s, below the closed form). At wide widths the cut's damage
is tiny (.003-.008) and fitting gates on 128 blocks overfits. Shonu's trend concern is partly real: the fitted method
needs a stopping rule.
ES (pre-registered, runs after T1b): RCF-G with EARLY STOPPING on held-out calibration blocks (16 prose cal[64:80] +
16 code cal[192:208], never used for fitting or scoring): window loss checked every 25 steps, best weights kept, 300 s
cap. Cells: L16 K768, L22 K768, L16 K1152, L22 K1152 (the unstable ones). Pass: G-ES >= max(G75s, G150) - 2 (mean)
in >= 3 of 4 cells, and never below C150.
T1b result (`results/clean-t1-150s.json`, 150 s, K768): L16 B1 40/39, B2 38/21 (-2/-18); L22 B1 17/38, B2 24/44
(+7/+6). FAIL (needs both). Two-back helps at narrow widths (T1 K48: +4..+8) but is unreliable at wide widths: the
L-2 fold disturbs layers L-1 and L, and a longer fit repairs it on L22 but not on L16 code. Stopped per the rule.
Not tried (possible later): two-back inside RCF-G with early stopping; a reader penalty on the L-2 fold (G8 idea).

LIQUID track (2026-10-20, separate session): shrinking attention heads, head size and stream width. Own log `research/clean/LIQUID-LOG.md`, report `research/clean/LIQUID-REPORT-2026-10-20.md`. Code `research/clean/liquid/` imports rcf_v1/contract, never edits them.
ES result (`results/clean-es.json`, RCF-G + early stopping, cap 240 s, not the 300 s written above). Share prose / code
(mean) vs the pre-registered bar max(G75s, G150) - 2: L16 K768 51/43 (46.8) vs 43.0 PASS; L16 K1152 63/35 (48.9) vs
42.5 PASS; L22 K768 28/47 (37.4) vs 35.5 PASS; L22 K1152 25/24 (24.4) vs 27.0 FAIL by 2.6. Above C150 on the mean in
all 4 cells (L22 K1152 code alone is 24 vs C150 30). 3 of 4 cells -> PASS by the letter of the rule.
Reading: early stopping removes the collapse (L22 K1152 prose -11 -> 25) and beats the 2-min trained layer at K1152
on both layers (L16 63/35 vs 29/33, L22 25/24 vs 9/10). It does not reach the lucky 75 s point on L22 K1152: the
held-out check is 32 blocks and the damage there is tiny, so the stopping signal is noisy. RCF-G + ES is the current
best single-layer method at every tested width. Stop step (in `raw`): 400, 1125, 675, 675 of ~2850-3095 steps (L16 K768, L16 K1152, L22 K768, L22 K1152). The held-out check fired early in every cell, so the stop did act. One run per cell.

## TREE FOLD (2026-10-21; idea: research/clean/LINEAGE-LOG.md end, memory project-depth-chain-idea)

Shonu's picture (essay figures): the coarse-to-fine tree runs along DEPTH. A child neuron's parent is the earlier
neuron that drives it most (the essay's sports-car trace: "the neuron that contributes most to the next one").
A smaller model = the chain cut early. Coarsening a child = remove it; its parent carries the child's average output.

TF1 (pre-registered before running). `gate/tf1.py`. SmolLM2-135M, dev layers 11, 16, 22 (never 6, 9, 21, 24).
K = 48 (primary) and 384. Calibration 64 prose + 64 code blocks (as rcf_v1). Dev: prose test blocks 0-39, code
code_train.txt first 400k chars. Sink guard on every changed module.
Parent of a removed child c = the layer L-1 neuron p with the largest |attribution| to c on c's top 0.5% calibration
tokens by |h_c|: attr(p, c) = mean over those tokens of h_p / rms(stream) * d_p . (gamma * dh_c/dx), with
dh_c/dx = silu'(g_c) u_c G_c + silu(g_c) U_c (activation x gradient through the RMSNorm scale; the norm's
mean-removal is ignored). Fold: alpha_c = least squares of h_c on h_p (no intercept). Layer L-1 down_proj column p
gains alpha_c * d_c (the child's OWN write vector). Stored per child: one parent index + one number. Nested by
construction (a smaller size only adds more children's terms).
Arms. Refit regime (kept down_proj refit by LS to Y minus the fold's prediction): cut_ls, tree1 (1 parent, attribution),
tree3 (3 parents by attribution, 3 numbers), tree1_pred (parent = best single predictor of h_c), rcf_plf (dense fold
over all L-1 neurons, CV ridge), rcf_v1 (dense L-1 + attention; reference ceiling). Nested regime (kept down_proj
untouched): cut_raw, tree1_raw, tree3_raw, rcf_plf_raw.
Score: share of the cut's damage won back = (dCE cut - dCE arm) / dCE cut, same regime's cut.
Pass: tree1 wins back >= 50% of rcf_plf's gain (refit regime), prose AND code, on >= 2 of 3 layers at K = 48.
Run time stated: about 15-20 min for both K on 3 layers.
TF1 result (`results/clean-tf1.json`): FAIL on 3 of 3 layers, both K. tree1 / rcf_plf gain, prose / code: K48 L11
.02 / .04, L16 .09 / .10, L22 .12 / .04; K384 between -.08 and .11. One attributed parent explains 1-4% of the
removed output (fold R^2 .014-.041); three parents 3-8%. rcf_v1 reproduces G11 (L16 K48: 38% / 44%).

Essay-based readings (before the next step): (1) "Everything feeding into it makes its own small semantic judgment":
a neuron is built from MANY votes; the essay's chain shows only the top contributor on ONE image. (2) The gate in
front may sit several layers back (recipe_v4 found judgment neurons at layers 10-12 for L15 children), not in L-1.

TF-DIAG (pre-registered, no CE). Held-out fold R^2 of the removed output (fit on even cal blocks, score on odd) with
the top-n parents per child by attribution, n = 1, 3, 10, 30, 100, from (a) layer L-1 only, (b) layers L-6..L-1
pooled; plus the dense ridge on (a) and (b). K48, layers 11, 16, 22. Fixed reading:
- if pooled top-1 R^2 >= 3x the L-1 top-1 R^2: the parent sits further upstream -> TF2 with multi-layer parents;
- else if some n <= 10 reaches >= 50% of the dense R^2 (same pool): a few-parent fold is viable -> TF2 with that n;
- else: upstream support is spread over many neurons; a one-parent linear fold is not the right form of the idea.
TF-DIAG result (`results/clean-tf-diag.json`): third branch. Held-out R^2 of the removed output, parents from L-6..L-1:
L11 top1 .024, top10 .090, top100 .206, dense .607; L16 .040 / .128 / .242 / .480; L22 .060 / .179 / .341 / .668.
Pooled top-1 is only 1.2-1.6x the L-1 top-1 (rule 3x). Top-10 reaches 19-27% of dense (rule 50%). Top-1 parents sit
mostly in L-1 and L-2. Upstream support is spread over many neurons.
TF-CTX (descriptive, `results/clean-tf-ctx.json`): over a child's top 327 tokens, the most common top contributor is
top on only 10% (L16) / 13% (L22) of them; 131 / 110 different neurons take the top spot. The top contributor on a
token carries 0.6-0.8% of the total |attribution|. Caveat: direct path from MLP neurons only (attention and
embeddings not included); RMSNorm scale only.
Reading: the essay's chain is a PER-INPUT path (its figure follows ONE image). In SmolLM2 the edges of that path change
from token to token, and each step is one small vote among many. There is no fixed one-parent tree to fold along. A
context-dependent parent, linearized, is the dense fold, which is what RCF already does. "Coarsen to your parent" in
this model means "let whichever upstream route is active carry you", not "fold into one fixed neuron".
TF-BIG (well-trained LM check, `gate/tf_big.py`, `results/clean-tf-big.json`; L-1 parents only, 200 sampled removed
children, K = 3%, 24 + 24 calibration blocks). Top parent share of |attribution| (x uniform) / top-10 R^2 / dense R^2:
SmolLM2-1.7B L9 .0080 (x66) / .063 / .410; L13 .0074 (x61) / .064 / .379; L18 .0103 (x84) / .108 / .451.
SmolLM2-135M (same script) L11 .019 (x29) / .064 / .424; L16 .019 (x29) / .121 / .465; L22 .023 (x35) / .141 / .482.
Modal top parent .11-.15 everywhere. The better-trained, larger model is about 2x more concentrated relative to chance,
but top-10 / dense is the same (.15-.29). Model quality does not change the tree-fold verdict.
Correction: the earlier "vision is 10x more concentrated" compared vision L-1 (.083) with 135M pooled over 6 layers
(.006-.008). Matched (L-1, x uniform): vision x256 vs LM x29-84, i.e. 3-8x.
