---
name: feedback-cale-design-rules
description: "Cale design rules from Shonu: developer confidence, no machine constants, absorb OS cost, API surface, no natural-language API, tokens/s goals, overshoot is error (merged)"
metadata:
  type: feedback
---

## Developer confidence (2026-10-13)
- Quote: "A developer needs to feel confident that the code they wrote works the way they expect it to on every machine." Design principle for Cale and for how I build.
- Why: a rare hardware case happens every time on that hardware. A silent trade-off makes the app behave differently on some laptops and breaks "write once, run everywhere".
- Apply: first look hard for a way to avoid the trade-off at no extra resources. If real, optimize both sides, then let the developer choose explicitly (documented goal/setting); Cale reports in the reply what it did. Never decide a behavior-changing trade-off per machine in silence.
- Weak-machine rule (2026-10-13): never label a fix "HP-specific"; a future bigger Matryoshka model puts strong machines in the same spot. Each fix is general or a developer choice. Never hard-code size counts, layer counts, device names.
- Floor behavior: when even the smallest size cannot meet the goals: predict before running, goal `priority`, `on_miss` (best_effort / refuse / wait / try; "try" runs anyway below the floor). A safety refusal (would crash/freeze the computer) overrides every setting and is a documented error.
- Test on odd hardware too (weak GPU MX150, slow iGPU UHD 620, old drivers; later CUDA on a strong GPU).

## No per-machine constants (2026-10-13, rule 15, docs/CONSTANTS.md)
- Quote: "you are using a lot of per machine constants. ... i want this to work on many environments well (remember the equation). you should think of ways to extract these variables from the machines Cale is running on without hardcoding stuff."
- Before writing a number ask if it depends on the computer. Then get it from the OS/device/engine (e.g. `cale_offload_min_batch`, `cale_n_batch`) or measure and keep it (first-load speed, `PieceTuner`, observed base memory). A policy number goes in one named place and is reported. Update `docs/CONSTANTS.md`. Trap: the engine's compute-buffer report is a reservation, not touched memory (over-predicted by 0.6 GB). Rule 7 was loosened: speeds measured once (see tuning in [[project-sameness]]; `CALE_NO_MEASURE=1` turns it off).

## Cale absorbs OS cost (2026-10-12)
- When I proposed one engine everywhere (llama.cpp + Metal): **no.** WE absorb the cost of optimizing for different OSes (per-device only if forced). App developers must not think about it. Faster runtime = more app types work well.
- Keep the fastest engine per OS (MLX Mac, llama.cpp elsewhere). Use generic code where nothing is lost. Expect many OS-specific parts; not a smell. Put shared logic above the engine interface (one grammar library on logits, one chat template, one scheduler). Do not re-propose collapsing engines.

## API surface principle (2026-10-15)
- Exposing every local knob makes local apps harder than cloud apps. Relative capabilities are the first-class primitive (one app works on many machines); absolute stay as a legacy escape hatch; Cale figures out the rest. Auto mode by default, deeper control when asked.
- Apply: four-question test in `docs/research/api-surface-principle.md` before exposing any field. Name capabilities by outcome, not mechanism. Give an explanation of each choice instead of a control.

## No natural-language API (2026-10-19)
- Developer control of Cale (zoom, compression, goals) is structured: calibration data sets with numeric weights, built artifacts (e.g. "zoom profile" file like a LoRA), numeric/enum goals. Never apps passing natural-language goals ("keep code sharp") at runtime.
- Why: it should not feel like prompting Cale; no separate model interprets intent. An optional agent may live OUTSIDE the runtime to help build a profile; its output is a plain artifact. Show config-style code, e.g. `cale.load(profile="x.zoom", goals={...})`.

## Speed goals in tokens/s (2026-10-12)
- Answer length is unknowable, so "finish in N seconds" is wrong. Goals: `min_tokens_per_s`, `max_first_token_s`, `memory_pct` (counting context), plus `min_quality`, `power` (my additions; he may drop them). No timing run at first launch with heuristics. Predictor: bandwidth x efficiency / bytes per token; time the first 8-16 tokens of the first real answer; adjust with live CPU/GPU load, power source, heat; step the size down between tokens if live speed misses. Research 2026-10-12: spec-only +-30-50%, after one run +-10%.

## Overshoot is an error (2026-10-15)
- Quote: "scoring better than the goals is a type of error because it means you could've used a more intelligent slice. It cuts both ways. ... The whole point is accurate calibration."
- Report headroom (actual/goal) beside met/missed. Flag "wasted slice" when headroom is well above 1 AND a larger size was allowed. Judge predictors and the adjuster by signed error. Scorecards must not call "faster than goal" a pass without checking the size.

Related: [[project-cale-vision]], [[project-cale-phases-3-5]], [[feedback-user-working-style]].
