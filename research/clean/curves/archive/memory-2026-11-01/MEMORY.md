# Memory index

## Current focus
- [ROADMAP](project-curves-track.md) — 2026-11-01: roadmap must END with full Cale on WHOLE ResNet-50 (no detraction); row 3 still open (TQ5 limit only; ResNet-18 far from limit, KL/Lambda 0.34); READ research/clean/curves/ROADMAP.md FIRST: 4 sessions to the ResNet optimum (1 squeezing row 3, 2 tight graded bounds in ResNet form, 3 ResNet-18, 4 instant zoom R18/R50)
- [Curves track](project-curves-track.md) — NEWEST (2026-10-31 late, worktree `kl`): THEORY ON KL, R-obj DROPPED by Shonu; B.I: TI1 drop groups (information table, chain rule) + TI2 kept-whole groups (<= eps_lin/4) are OPT-KL, keep-or-drop levels exact (S-keepdrop, card 3-31 PENDING); squeezed groups 0<k<rank OPEN (TI4 Fisher-metric 2nd order, TI5 D2 tables do not fix KL, DI1 hardness, DI2 least unit may switch); before (2026-10-31, worktree `joint`, merged): ONE BUDGET + OPTIMALITY AUDIT: TV19.2 EQ (stream free above dim E_N + sum k_S), TY4 EQ (ResNet stage split, capability = subtree group), TY5: TP1/TL5 are OPT-D2, miss the KL bar even when clean; cards 3-29, 3-30 PENDING; before (2026-10-29 night, worktree `smallnet`, anchor-merged): DEEP LAYERS B.L.2: TL3 clean subtree (on-inside + sheltered flat gates) is affine in its top -> TP1 closed form at any depth (TL5), lifts add a between table (TL4), TL3.2 scope (splits must be flat 0/1 directions of the top layer; generic points -> no splits), TL6 graded bound (weight norms, 1/gate margin; loose); LADDER 38; before (2026-10-29 evening, worktree `normproof`, anchor-merged): DV3 PROVED as TV15 (norms + every Lipschitz activation, ReLU margin), angles TV16/TV17, R-cutclass; card 3-25 answered; round 4: zoom-free member claims TV18/TV18.1 (one map, every zoom), card 3-28 attention PENDING; NEVER whole-file copy or diff3 the paper, anchor-merge; before (2026-10-29 latest, worktree `smallnet`, merged): broken cases read as networks, paper B.L (TL1 merge step least under (K) = no dropped reused direction; TL2 lift through on-inside units re-expresses finer units), card 3-27 R-oninside PENDING; before (2026-10-29 late, worktree `summary`, merged into main): summaries up the tree, paper B.P (TP1 sandwich, TP2 sibling merges add tables, TP3 cost), card 3-26 R-linread PENDING; TWO TRACKS (2026-10-29): stream round 2 (worktree `stream`: card 3-23 answered, trunk amended for areas, F14/TV9-TV13 overlap + cancellation, DV3 next, card 3-25 S-laminar) and tree of areas, paper B.H (TH1 split any depth, TH2 sibling merges nest / lifts don't, TH3 lossy merge breaks step by step), card 3-24 PENDING; before (2026-10-29): stream as areas, paper B.V (TV1 norm, TV2 readout split, TV3 SVD), card 3-23 PENDING; before (2026-10-28): ResNet-18 map+zoom (no channel-level map exists; B.Y merge rule TY2 bound; card 3-22 PENDING); before: size B.N (copy test TN1, TN3 no unit count, card 3-21 R-mass PENDING); before: robust optimum (B.E: TE1 span-blindness, DE2, TE4 graded split, card 3-20 PENDING); ResNet ladder: H-weak holds, H-scale mixed; before: Trunk T5 mapped network, TM1/TM5, complete maps DECIDED
- [Map track](project-map-track.md) — before curves: derive track D-series (DERIVE-REPORT-2026-10-22.md); before that: accurate coarse-to-fine map of ConvNeXt-Tiny; state in research/clean/MAP-STATE.md; do not stop to ask; small models (atto/nano) = second reference

## How Shonu wants me to work
- [Working style](feedback-user-working-style.md) — a "contradiction I see" note = find the form from the network, not explain hardness (2026-10-29); limited ML background; real ML objects + a diagram and "why it matters" per step, few analogies (2026-10-27); on a new direction explain first, then a ~4-step session (2026-10-29); own ideas; Cale equation; attack the root layer
- [One recursion](feedback-one-recursion.md) — 2026-10-29: zoom and depth recursion are ONE idea (tree of areas, node R-tree)
- [No unchosen tradeoffs](feedback-no-unchosen-tradeoffs.md) — 2026-10-31: exact=exact (certified check), graded=first-order tight, trained nets are clean (realism check), card before any narrowing, "not possible" needs a clean theorem
- [Decide, don't quiz](feedback-decide-dont-quiz.md) — once Shonu gives a direction, pick the option myself; limits go inside the plan
- [Testing rules](feedback-testing-rules.md) — Claude in Chrome only; never hit the real Cale (47821); hidden-zoo review
- [Disk hygiene](feedback-disk-hygiene.md) — shared near-full disk; DISK.md ledger; build, score, delete; keep >15 GB free
- [Lesson rules](lesson-rules.md) — durable corrections from past sessions (some from other projects)

## Research rules
- [Error = gap to the optimum](feedback-error-meaning.md) — rule vs least small build; 2026-10-31: THE OPTIMUM IS THE POINT, label every result OPT-KL / EQ / OPT-D2 / ONE-BUILD, flag any move off it loudly
- [General form first](feedback-general-form-first.md) — 2026-10-26: optimum is the object; folds/gates = hypotheses; clean = property of N alone, many principles, eps a list
- [Derive the unit](feedback-derive-the-unit.md) — 2026-10-27: properties as math first, then prove which object has them; follow the math
- [Mapped network](feedback-mapped-network.md) — 2026-10-28: trunk T5 reads a map of N (typed claims, graded eps); finding the map is outside the math
- [Wrong-question check](feedback-wrong-question-check.md) — 2026-10-27/28: a fork forced by a theorem true for every N is our definition's fault; a blow-up is an ill-posed object to fix, not a decision card; clean principles are graded
- [Trunk + decisions](feedback-trunk-decisions.md) — 2026-11-01: technical nodes (premises, summaries, certificates) are MINE, decide by delegation, bring only directional forks; fixed trunk T*; every NARROWING restriction = a node Shonu decides; card, not a new hypothesis, when blocked; 2026-10-29: widening/technical scope nodes are mine (do the math, decide); prove the general case before any premise that leaves out ReLU/norms
- [Math paper stands alone](feedback-math-paper-standalone.md) — 2026-10-24 night: COARSE-THEORY.md; facts -> definitions -> named hypotheses; Lean; numbers only in Part C; enforced by hooks + `.math-mode`
- [Quick iteration](feedback-quick-iteration.md) — loops of 1-3 min on a small cache; no hour-long sweeps while iterating; 2026-11-02: every long run gets a 15-min heartbeat Monitor at launch (alive, CPU, last result; crash/finish at once)
- [Research method](feedback-research-method.md) — map goal; smallest test; fix the step; keeper; flexible-vs-fixed; vision first; specific first
- [Derive like a mathematician](feedback-derive-like-mathematician.md) — THE LOOP (verify each failure reason); NO real-image shortcuts; exactness test = target; 2026-10-23 template; mix is a declared input; DERIVED/MIX-DERIVED/FITTED/TRAINED
- [Essay + interp rules](feedback-research-essay-rules.md) — essay is law and inspiration; read it when stuck; real interp tools

## Research history
- [All tracks](project-research-history.md) — nested Gemma, ocean ladder (both superseded), Round 1, tree, clean, coarsening/RCF, zoom, liquid, lineage
- [Curves history](project-curves-history.md) — curves track before the T5 map work: curves crush, weights-only, exactness, trunk resets, unit branch

## Cale product
- [Vision + equation](project-cale-vision.md) — success equation; one runtime + one model; Gemma 3n is the prototype final model
- [Design rules](feedback-cale-design-rules.md) — developer confidence, no machine constants, API surface, tokens/s goals, overshoot
- [Consumer demo](project-consumer-demo.md) — installers, Quick Notes + Rewrite, demo rules, GPU fix, traps
- [Sameness](project-sameness.md) — same feel on every computer; fixed-time tests; ship prepared; install numbers
- [Phases 3-5](project-cale-phases-3-5.md) — Gemma 3n nest, relative goals, Mac+Win+Linux (done)
- [Phase 6](project-phase6.md) — honest API, GPU, goals, prompt reading 1.6-1.8x faster (steps 5-7 open)
- [Business ideas](project-cale-business-ideas.md) — LoRA retraining subscription, device-testing service, cloud GPU later

## References
- [HP test bench](reference-hp-testbench.md) — ssh hp, hp job script, heartbeats, traps, click Cale dialogs there
- [GitHub + Vercel](reference-github-vercel.md) — ParkingSoman repo and gh login; cale-rewrite site deploy
