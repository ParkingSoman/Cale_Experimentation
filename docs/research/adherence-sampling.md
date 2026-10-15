# Adherence and sampling: what we have not researched yet

Date: 2026-10-15. Research only. No code changed. Nothing was run on the HP.
Tags: **[M]** measured, with the file. **[C]** read from code. **[G]** guess, not measured.

Shonu asked two things.

1. Can quantization help speed? Where did "16-bit ≈ 4-bit" come from?
2. Can Cale sample everywhere, all the time, without hurting the app, so predictions get more exact?

The goal behind both is the Cale equation: (app types that work well) × (environments). A prediction that is wrong makes an app miss its goal on one machine. So sample quality is a direct lever on the equation.

---

## A. The "16-bit ≈ 4-bit" result

### Where it came from

| Question | Answer | Source |
|---|---|---|
| Which model? | **Gemma 4 12B** (Google's QAT build `gemma-4-12B-it`). Not Gemma 3n. | `FINDINGS.md` line 195 on, memory `project-ocean-ladder.md` |
| Which test? | MMLU-Pro, n=1000, same questions, layer by layer. | `FINDINGS.md` "Step 1 gate result" |
| Numbers | 16-bit original 53.8. 8-bit 52.8. QAT 4-bit 53.1. z=0.72 [M] | `FINDINGS.md`, `results/ladder-gate.log` |
| How strong? | One benchmark, one model, n=1000. The gap (0.7 points) is inside the noise. Good for "no visible loss on knowledge questions". It says nothing about long context, JSON output, or code. | [M] |
| Why 4-bit held up | Google trained the model with 4-bit in mind (QAT). Plain 4-bit (no QAT) lost 13.5 MMLU-Pro points against 8-bit on the same model. | `FINDINGS.md` line 61 [M] |

**So the result does not carry over to Gemma 3n.** Shonu's doubt is right.

### What exists for Gemma 3n

| Fact | Source |
|---|---|
| We ship Q4_0 GGUF from unsloth (`gemma-3n-E4B-it-Q4_0.gguf`, about 4.4 GB). | `cale/install.py:39` [C] |
| Gemma 3n Q4_0 full size is bit-identical to stock llama.cpp at full size, and the sliced E4B behaves like Google's E2B. | `results/xplat/llama-gates.json` [M] |
| No 16-bit vs Q4_0 comparison for Gemma 3n exists in the repo. | grep of `FINDINGS.md`, `docs/`, `results/` [M, absence] |
| The Gemma 3n weights are not QAT as far as I know. Q4_0 on a non-QAT model usually loses more than QAT 4-bit. | web search found no Gemma 3n quant numbers [G] |
| The 3n MMLU-Pro harness gave inverted results earlier (E2B > E4B) until the prompt was fixed. So any new quant test must reuse the fixed harness. | `FINDINGS.md` line 357 [M] |

### Does a smaller format make the HP faster?

Decode on a CPU is memory-bound. Bytes read per token set the speed. `cale/gemma3n_facts.py` uses 4.5 bits per weight [C]. Measured clean decode on the HP is 7.7 tok/s (Windows) and 7.1 (WSL), full size [M: `results/xplat/bakeoff.json`].

| Format | Bytes per token vs Q4_0 | What the CPU can do with it | Guess for HP decode |
|---|---|---|---|
| Q8_0 | about 1.9x more | has a repack path | about 1.9x slower [G] |
| Q4_0 (today) | 1.0 | repack on gives 2x: 7.7 vs 3.8 tok/s [M: `FINDINGS.md` GPU table] | 7.7 |
| Q4_K_M | about 1.05x more | repack exists in newer llama.cpp for some K types. Not checked for b11558 | about the same, maybe slower [G] |
| Q3_K / Q2_K | 0.75x / 0.6x | slower kernels, less repack | at most 1.2x faster, with a quality cost [G] |
| IQ4_NL | same as Q4_0 | has repack | about the same [G] |

**Conclusion [G]:** going below 4.5 bits buys little on this CPU. Going above costs a lot. The repack path is worth more (2x) than any bit-width change. So Q4_0 is the right place on the HP. The open question is quality (Q4_0 on a non-QAT model), not speed.

Quantization can still help in two other places:

1. **KV cache type.** Doc Reader holds about 7,900 tokens of context. A Q8_0 or Q4_0 KV cache cuts memory reads per token at long context. llama.cpp supports `cache-type-k/v`. Gemma 3n has only 20 layers with KV [C: `gemma3n_facts.py`], so the gain is small. [G]
2. **Mac (MLX).** MLX 4-bit g64 is what we ship. A QAT-style exact grid helped Gemma 4 (47.5 vs 45.0) [M: `FINDINGS.md` line 86]. Nothing equivalent was built for 3n.

### The measurement that would settle it (Mac, light, not run)

| Test | Cost | What it decides |
|---|---|---|
| `llama-perplexity` on Gemma 3n E4B: BF16 vs Q8_0 vs Q4_0 vs Q4_K_M, 200 chunks of a fixed text | about 30 min on the Mac, no HP | does Q4_0 lose quality on 3n? |
| Same four on the fixed MMLU-Pro 3n harness, n=300 | about 1 h | same, on the real task |
| `llama-bench` of the four on the HP, run when the HP is idle | 10 min of HP time | the true speed price of each format |

**Decision for Shonu:** only run the HP bench after the scorecard job ends. Quality first, speed second.

---

## B. What Cale samples today

| # | Sampling point | Cost | Feeds | Source |
|---|---|---|---|---|
| 1 | Memory speed probe at engine start | about 0.15 s, no model | `bandwidth` prior for predictor | `predict.py` header, `engine.py` [C] |
| 2 | `cale tune` probes at install: environment, disk, working memory, threads, sizes, prefill curve, context, batch, GPU pieces, sustained, cooldown, drift | minutes (quick) to many minutes (max). No app waits. | `table` (speed per size), `pre_curve`, `dec_ctx`, `sustained`, `batch_eps`, placement | `cale/tuning.py` [C] |
| 3 | "Drift" probe: first measurement again at the end | one call | `extra["noise"]`. The same computer moves 10-25% between early and late readings | `tuning.py`, `FINDINGS.md` line 997 [M] |
| 4 | Every finished answer: tokens/s, first-word, prompt speed | free (counters already exist) | `corr_dec`, `corr_pre`, `by_level`, `fixed` fit | `Predictor.observe` [C] |
| 5 | Disk read speed of every model load | free | `disk_bps` | `observe()` [C] |
| 6 | CPU busy share, other programs only, cached 0.5 s | a few syscalls | `live_factor()`: `1 − 0.5·busy`, floor 0.35 | `oslayer.cpu_busy_other` [C] |
| 7 | Battery and "hot" (5 s cache) | one OS call | `power: light` goal | `oslayer.power_state` [C] |
| 8 | Heat model: recent work adds heat, idle cools it | free | `sustained` throttle factor | `Predictor.note_busy` [C] |
| 9 | Closed loop inside an answer: every few tokens compare speed with goal, step one size down or up | free | size choice, no learning | `engine._controller` [C] |
| 10 | Live activity: per piece timestamps | free | Hub meters | `cale/activity.py` [C] |
| 11 | Memory use after each answer | one OS call | `_grow_base` | `engine._grow_base` [C] |
| 12 | Memory pressure level | one OS call | size cap | `pressure.py` [C] |
| 13 | Predicted vs real check offline | a test run | proof | `scripts/predict_check.py`, `cale/tests/relative.py` [C] |

### Gaps against Shonu's list

| Sampling point he named | Today | Gap |
|---|---|---|
| **During model download** | `fetch_model.py` counts bytes only for the progress bar. The network speed is not saved. | Free. Record bytes/s and stalls per minute. Gives network speed for the demo ("download takes N min") and a free disk-write speed. |
| **At install** | `cale tune` exists. It is one-shot. | The tune result is trusted for the machine's whole life. Nothing re-checks it after a Windows or driver update (this session's HP changed OS). |
| **Between answers (idle gaps)** | Nothing. | Biggest gap. Idle time is free CPU. See D. |
| **Inside answers** | Only the closed loop (raw 16-token rate, thrown away after use) and per-piece times in `activity.py` (memory only, last 200 calls). | Per-token times are not stored. The shape of an answer (fast start, slow tail, thermal drop) is lost. |
| **Context length effect** | `dec_ctx` is one number from install. | Learned never. Doc Reader sees real 8k context. |
| **Contention** | `1 − 0.5·busy`, floor 0.35. | Measured slowdown on the HP was 6x (9.5 → 1.6 tok/s, `results/demo/hp-windows.json` "together") while CPU busy peaked at 0.63. The model says at most 0.5x–0.65x. [M] |
| **Paused time inside `decode_s`** | `decode_s = end − t_first` (`backends/llama.py:147`). A lower-tier answer paused by the new scheduler keeps counting. | The predictor learns "slow" from an answer that was only waiting. See C, H3. [C] |
| **Bad samples** | Skipped only if tokens < 9, size switched, or batch average ≥ 1.05. | Contention, paused time, throttling, and battery are not excluded. They are divided out by `live_factor`, which is too weak (above). The result is a permanently pessimistic `corr_dec`. [C + M] |
| **Per-app and per-goal data** | None. | A prediction for "JSON sorter" and for "7,900-token reader" use the same table as short plain text. |

---

## C. The open HP puzzle, from existing data only

### The data

| Item | Value | Source |
|---|---|---|
| Clean HP decode, full size, Windows | **7.7 tok/s** (bench), 7.2 (our build) | `bakeoff.json`, `FINDINGS.md` "Windows speed hunt" [M] |
| Clean HP decode, small sizes | 9.5-13.7 tok/s | `llama-gates.json`, `FINDINGS.md` line 784 [M] |
| Earlier "unexplained" note | measured 3.7 vs predicted 6.5 | memory `project-consumer-demo.md` |
| Quick Notes, size 49%, alone, first app | **9.5 tok/s** (min 9.05, max 11.3) | `hp-windows.json` [M] |
| Doc Reader, size 100%, context ~7,900 | **1.2-1.4 tok/s** | `hp-windows.json` [M] |
| Meeting Recap, size 48% | **1.37 tok/s**, first word 17.9 s | same [M] |
| Sorter, size 64% | **1.21-1.31 tok/s** | same [M] |
| Quick Notes while Sorter runs | **1.43-1.69 tok/s** (size 48%) | same, "together" [M] |
| Doc Reader read of 7,824 tokens | **112 s real vs 295 s predicted** (70 vs 26 tok/s) | same [M] |
| HP state after run | on AC, battery 100%, clock 1792/1992 MHz idle, McAfee `mcshield` 503 CPU-seconds, Defender running | ssh check this session [M] |

**What the shape says.** Four different apps, at 48%, 49% to 100% size, all sit near 1.3-1.6 tok/s. Size changes the clean speed by 1.7x (7.7 vs 13). So the cause is **not size-dependent**. One fast number exists: Quick Notes, the first app run, before the others. This points to a **state that appeared after the first app and stayed**.

Also: decode is 5x slower than the table while prompt reading is 2.7x faster than the table. The two errors go in opposite directions. So they likely have **different causes**.

### Ranked hypotheses [all G]

| # | Hypothesis | Why it fits | Cheapest test (HP idle, after the job) |
|---|---|---|---|
| H1 | **Another heavy program is running from the second app on.** Examples: a leftover Sorter, earlier check scripts, an app the scorecard opened, Windows post-update work, antivirus scan. | Slow from app 2 on. CPU busy 0.55-0.63 in "together". Past notes: "Sorter works by itself whenever it runs". | Before the run, list top CPU processes and the Hub's running calls. Run Quick Notes **last** too. If it is slow at the end as well, the state persists. 2 min. |
| H2 | **Weights paged out or scanned.** Windows drops pages of the mapped model file under pressure, or `mcshield` scans each page-in. Each token then reads weights from disk. | Speed is flat across sizes (disk-bound, not size-bound). `FINDINGS.md` already saw 347,000 page reads/s on the HP. `mcshield` used 503 CPU-s. | `Get-Counter '\Memory\Page Reads/sec'` while one answer runs. Repeat with `--mlock` (lock weights in RAM). 5 min. |
| H3 | **Measurement artifact: paused time counts as decode time.** The new tier scheduler pauses lower tiers. `decode_s` includes the pause. | Explains Sorter (lowest tier) at 1.29. Does not explain Doc Reader. | Subtract pause time in the engine (or log `paused_s`), rerun Sorter alone. 15 min of code. [C: `backends/llama.py:147`] |
| H4 | **Long context.** Doc Reader holds 7,900 tokens and Recap a long transcript. `dec_ctx` is a single number from install. | Fits Doc Reader and Recap. Does not fit Quick Notes together (short context). | Same size, 100 vs 8,000 tokens of context, plain answer, two calls. 3 min. Read `dec_ctx` from the HP `predict.json`. |
| H5 | **New shim or new placement after the OS update.** The Windows update may have changed the GPU driver (device order swapped before, `FINDINGS.md` line 793). A split placement made smaller sizes 2.5-3x slower than CPU. The shim is also new today. | Known failure mode in this repo. | `cale status` placement line, then `scripts/gpu_check.py`. A/B: old DLL (saved at the scratchpad `win-old.zip`) on one Doc Reader question. 10 min. |
| H6 | **Throttling.** i7-8550U is a 15 W part. Heat plus McAfee could hold it near base clock. | `sustained` exists as a probe. Idle clock looked normal, but **load clock was not sampled**. | Log `CurrentClockSpeed` once a second during one answer. Free. |
| H7 | **Predictor learned a bad number and now chooses badly.** A slow stretch (any of H1-H3) pulled `corr_dec` down. | `observe()` divides only by the weak `live_factor`. | Read `corr_dec` in the HP `predict.json`. If far below 1.0, reset it and rerun. 1 min. |

**Prediction error for the Doc Reader (295 s vs 112 s).** Reasons I can see [C/G]:

- The table prompt speed was measured at install, before patch 0003 (prompt tokens skip layers 20-34). `FINDINGS.md` says 1.6-1.8x faster. The `key` has `prompt_path: 2`, so the table may have been remeasured; I did not check on the HP.
- `corr_pre` moves 35% per answer, so one run barely corrects it.
- The MX150 reads big prompts about 3x faster than the CPU (58.7 vs 19.2 tok/s, `FINDINGS.md` line 78). A `pre_curve` that was measured with the card on a short prompt may not match 7,800 tokens.

The first test to run is **H1 + H2 together**. They are cheap, and they explain "flat across sizes".

---

## D. Design: continuous, near-free sampling

### Rules

1. **Never add latency to an answer.** Every sample is a counter that already exists, or runs in an idle gap.
2. **Strict budget per point** (below). Over budget → the sampler turns itself off for that point and says so in the log.
3. **No text.** Only numbers, sizes, and times. Same rule as `activity.py`.
4. **Reject first, learn second.** A bad sample teaches harm (see H7).
5. **One small file per machine and engine.** Rolling. Easy to delete. Easy to ship to Shonu for a funding slide ("data from N machines").

### Sampling points and budgets

| Point | What is sampled | Overhead budget | Why free |
|---|---|---|---|
| Download | bytes/s every second, stalls, disk write speed | under 0.1% CPU | Counter already in `fetch_model.py`. |
| Install | existing `cale tune` plus its `drift` result | already paid | Add: **re-tune trigger** when OS build, GPU driver, or CPU model changes (see below). |
| Per token, inside an answer | timestamp of each token (from `activity.piece`, already called). Keep only 8 summary numbers per answer: first-8-token speed, middle speed, last-16 speed, slowest gap, paused time, context length at start, tier changes. | one `perf_counter` per token (about 100 ns). Total under 0.01%. | Timestamps exist. Only the reduction is new. |
| llama.cpp counters | `llama_perf_context` gives eval and prompt-eval times and counts. | one call after each answer | Built into llama.cpp [web: search result, field names not verified; check `llama.h` in b11558]. |
| Between answers (idle gap, 1-30 s) | host load, battery, clock, memory pressure, page reads | one sample per 2 s while Cale is idle. About 20 µs each. | CPU is idle by definition. Stop at the first request. |
| Long idle (minutes) | a **mini probe**: 8 tokens decode and 64 tokens prefill at the smallest size. Median of 3 across the day. | at most 2 s of CPU per 10 min, only when load is under 10%, on AC, cool, and no app waits | This is the "drift" probe, repeated. It catches OS updates, driver changes, and throttling without waiting for a user to hit them. |
| Contention | OS pressure: Linux PSI `/proc/pressure/cpu` ("some" avg10) is exact for "someone wanted the CPU and waited". Windows: processor queue length and `CallNtPowerInformation(ProcessorInformation)` (current vs limit MHz). macOS: thermal state and memory pressure level. | few syscalls | These are the direct readings. `1 − 0.5·busy` is a guess. [G: APIs from my knowledge, not verified on these machines] |

### Storage

`~/.cale/samples.jsonl` (rolling, last 2,000 answers and last 500 idle probes, about 300 KB).

One record per answer:
`t, engine, level, ctx_tokens, prompt_read, prompt_tps, first8_tps, mid_tps, last16_tps, paused_s, switches, load_other, psi_cpu, mhz_ratio, on_battery, hot, mem_pressure, goal_met`.

No prompt text. No answer text. No app names (tier only).

### How the predictor uses it

| Today | Change |
|---|---|
| One `corr_dec` per machine, weight 0.35. | One `corr_dec` **per (engine, level)** and a separate **per-context bucket** (0-1k, 1k-4k, 4k+). Weighted by recency (half-life of about 20 answers). |
| `live_factor = 1 − 0.5·busy`. | Fit `live_factor` from the samples: pairs of (load, ratio real/table). The slope is learned per machine. It can say "6x" when 6x is true. |
| Bad samples divided out. | Bad samples **rejected**: load_other above 0.25, paused time above 10%, on battery, hot, mem_pressure above normal. Kept in a separate list as "under contention" data. |
| `prefill`: 3 numbers from install. | `prefill_tps` as a function of prompt length and GPU/CPU path, from answers with `prompt_read`. Median of last 9 per bucket. |
| No uncertainty. | Every prediction carries a band (P25-P75 of recent ratio). Apps get "7-9 tok/s" and goals can use the **low end**. |
| One-shot tune. | Re-tune trigger: OS build, GPU driver version, CPU model, or `drift` over 25%. Runs at idle only. |

The **contention table** is a new asset. It says how a goal behaves when another app is busy. That is exactly the Cale equation for the "Chat + Sorter" demo. With it, the scheduler can say before the call: "Quick Notes will get 1.6 tok/s while the Sorter runs. Pause the Sorter?" The current tier scheduler only reacts.

### Own ideas

1. **Prediction scorecard in the Hub.** Show predicted vs real for the last 20 answers. Shonu can see accuracy tighten. It is also a funding chart: "error falls from ±40% to ±8% in 30 answers".
2. **Machine fingerprint.** Hash (CPU, RAM, GPU, driver, OS build, engine build). A new hash triggers a quick re-tune. Same hash shares samples. Later: a tiny opt-in "fleet table" of fingerprints and speeds, so a new machine starts from a neighbour's numbers, not a prior.
3. **Draft/speculative hooks.** Draft acceptance rate is a free per-answer signal about the content type (math vs prose). Gemma 3n has no drafter in our build [C], so this is for later.
4. **Sample-based quality guard.** If idle probes show output length or repetition drifting, flag it. Cheap early warning for a bad kernel (like repack off).
5. **Goal budget accounting.** Count how often each goal was met by app type. This is the first "app types that work well" number for the equation, per environment.
6. **Pause-aware timing.** Subtract paused time in `decode_s` today. This is a bug fix and needs no research.

### What stays out

- No timing run at first launch. Shonu said no (memory `feedback-tokens-per-second.md`). The idle mini probe starts only after the first real answer and only when the computer is quiet.
- No telemetry leaves the machine. Fleet idea 2 is opt-in and later.

---

## E. Ranked list

| Rank | Item | Expected gain | Risk | Effort | Needs Shonu |
|---|---|---|---|---|---|
| 1 | **Find the HP slowdown** (H1, H2, H6, H7 tests, 10 min total on idle HP) | high: 5x on 4 of 5 apps | none | small | no (HP is a test bench) |
| 2 | **Subtract paused time in `decode_s`** | medium: stops wrong learning for low tiers | low | tiny | no |
| 3 | **Reject bad samples** (load, pause, battery, hot) before `observe()` | high: stops permanent pessimism | low | small | no |
| 4 | **Per-token summary + `samples.jsonl`** (D table) | enabler for 5-9 | low | small-medium | yes: file name and size, and "no app names" rule |
| 5 | **Better contention reading** (PSI on Linux, queue length and MHz on Windows, thermal on macOS) | high for the "Chat + Sorter" demo | medium: OS calls differ | medium | no |
| 6 | **Idle mini probe + re-tune trigger** (OS or driver change) | high: protects against the update we just saw | medium: must never steal CPU | medium | yes: how much CPU idle probing may use |
| 7 | **Per-context and per-level corrections** | medium: fixes Doc Reader class errors | low | small | no |
| 8 | **Quant quality test on Gemma 3n** (perplexity BF16/Q8/Q4_0/Q4_K_M on Mac) | unknown: tells if Q4_0 costs quality on 3n | none (Mac, light) | small | yes: OK to run 1-2 h on the Mac |
| 9 | **HP speed bench of the four formats** | low expected (guess: no faster than Q4_0) | none | small | no, after the scorecard |
| 10 | **KV-cache type test** (Q8_0 K/V at long context) | small [G] | low: quality | small | yes: it changes quality (needs a gate, `docs/PLAN.md` row "Quant type") |
| 11 | **Hub prediction scorecard** | pitch value | none | small | no |
| 12 | **Fleet table** (opt-in) | long-term | privacy | large | yes: policy |

### Claims to double-check before relying on them

- Format speed table in A (all [G]). Test with `llama-bench` before saying it to anyone.
- OS counters in D (PSI, `CallNtPowerInformation`, thermal state): from memory, not checked on these machines.
- `llama_perf_context` field names: not verified for b11558.
- Whether the HP `predict.json` was re-measured after patch 0003: not checked (HP not touched).

### Sources

- Repo: `FINDINGS.md`, `results/xplat/bakeoff.json`, `results/xplat/llama-gates.json`, `results/demo/hp-windows.json`, `cale/predict.py`, `cale/tuning.py`, `cale/engine.py`, `cale/oslayer.py`, `cale/backends/llama.py`, `cale/activity.py`, `cale/fetch_model.py`, `cale/gemma3n_facts.py`, `docs/PLAN.md`, memory files `project-ocean-ladder.md`, `feedback-tokens-per-second.md`, `project-consumer-demo.md`.
- Web (indirect only; no Gemma 3n numbers found): [Gemma 3 QAT thread, r/LocalLLaMA](https://red.applefritter.com/r/LocalLLaMA/comments/1jr89mc/gemma_3_qat), [GGUF vs GPTQ vs EXL2 guide](https://www.kunalganglani.com/blog/llm-quantization-gguf-gptq-exl2.md), [llama.cpp `llama-context.h` mirror](https://git.mitjafelicijan.com/llmnpc/tree/llama.cpp/src/llama-context.h).
