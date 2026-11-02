# Numbers in the code, and where each comes from

Rule 15 of `docs/PRINCIPLES.md`: a number that depends on the computer is asked or measured, never written. This file lists what is left.

Kinds:

| Kind | Meaning |
|---|---|
| **Model fact** | A property of Gemma 3n, the final model of this prototype (rule 5). Same on every computer. |
| **Measured** | Cale gets it from the computer, the device or the engine. The number in the code is only a first guess, or there is none. |
| **Policy** | A limit someone may want to change. It does not depend on the computer. One place, a name, reported. |
| **Open** | Depends on the computer and is still written in the code. A known gap, with the way to close it. |

## Measured (done)

| What | Where | How it is measured |
|---|---|---|
| Answer and prompt speed of each size | `cale/predict.py`, `cale/engine.py` `tune` | Measured when Cale is installed (`cale tune`), with a full-length prompt piece and a few answer tokens, and saved with the key of the computer and model. Every real answer then corrects it. If nothing was measured, the first load does a short version, once. `PRIORS` is used only before any measurement. Mac: first estimate 0% off. |
| Working memory of a prompt piece | `engine.tune` | The most memory the process holds while the longest piece is read, minus the weights, the base and the context. Saved. Mac (MLX): 809 MB, against the old 500 MB guess. |
| Number of threads | `cale/tuning.py` | Answers and prompts are measured separately among half the cores, the cores, twice the cores and the fast cores. A different count is chosen only if clearly better (15% once, 8% when alternating runs agree). Otherwise the number of cores. |
| Speed of each prompt piece length (a GPU that reads prompts) | `engine.tune`, `batching.PieceTuner` | Tried at install, saved. No app pays for trying. |
| Memory speed | `cale_membw` in the shim | A 0.15 s probe, no model. |
| Memory held besides the weights | `engine._sample_base`, `_grow_base` | The largest of five readings of the process, raised whenever an answer finds more in use. A prediction is never under. |
| Cost of an extra answer in a batch (`batch_eps`) | `predict.observe_step` | Learned from real steps. `BATCH_EPS` is only the start. |
| Disk speed | `predict.observe` | Learned from real reads. `DISK_PRIOR` is only the start. |
| The shortest prompt piece a GPU takes | shim `cale_offload_min_batch` | Asked of the GPU backend itself. |
| The longest prompt piece | shim `cale_n_batch` | Asked of the engine. |
| The best prompt piece for a GPU that reads prompts | `batching.PieceTuner` | Tries the sizes between those two, keeps the fastest within the time budget. If the CPU piece wins, the GPU is not used for prompts. |
| Which device runs the model, and whether every layer fits in it | `cale/backends/devices.py`, shim `cale_fit_all` | From device type, and from llama.cpp's own projection of the memory (no weights loaded; about 4 s on a 2 GB card). `WORKING_BYTES` is used only if the library cannot project. The memory left free in the projection is the policy `fit_margin_mb` (300 MB). |
| Prompt piece next to running answers | `batching._step` | From the measured prompt speed and the policy `STEP_SECONDS`. |
| Time that memory pressure is ignored after a load | `engine._settling` | Ignored only while the process memory is still changing (more than 2% between two looks). `LOAD_GRACE_S` (30 s) is only the upper limit. |

## Policy

| What | Value | Where |
|---|---|---|
| Time one prompt piece may take for an app that leaves to be noticed | 6 s (`CALE_MAX_PIECE_S`) | `batching.MAX_PIECE_S` |
| Time of a prompt piece next to running answers | 0.75 s (1.5 s alone, on the CPU) | `batching.STEP_SECONDS` |
| Share of RAM a setup may need before Cale refuses it, and the memory that must stay free now | 85%, 300 MB | `choose.SAFE_FRACTION`, `choose.SAFE_MARGIN` |
| Answers that run together | 4 (`CALE_SLOTS`) | `backends/llama.py` | A policy, not a machine number: the context memory is one shared pool, so an extra slot costs only a little state, and the benefit (nobody waits for another's whole answer) does not depend on the computer. Whether batching also adds speed is measured (`batch_eps`). |
| Longest wait, lease, idle time, poll times | `cale/goals.py`, `engine.py`, `server.py` | |
| Request limits (schema size, pictures, sound) | `constrain.py`, `media.py` | |
| How far a hold lets the size drop under pressure | `engine.WARN_CAP` | |
| Largest size for `power: "light"` | 75% | `gemma3n_facts.LIGHT_MAX_PCT` |
| Last prompt piece (tokens) from which the last token is read alone, when no speeds are measured yet | 16 | `gemma3n_facts.HOLD_LAST_DEFAULT` | Used only until `cale tune` (or the first load) has measured this computer. After that the rule `gemma3n_facts.hold_last_min` reads the measured prompt and answer speeds. A wrong value costs at most one extra pass over the weights per prompt. |
| Slowest text flow that still reads as flowing (the chat's starting goal, and where the slider sits when the person has not moved it) | 7 tokens/s | `cale/anythingllm/feel.js` `READING_TPS`, `cale/chat.py` `GOALS`. Source: average adult reading is about 4 to 5 words/s, which is about 5 to 7 tokens/s (Brysbaert 2019, a review of 190 studies; Xiao and Yang 2025). Below 7, text arrives slower than people read. Text far above reading speed gives no benefit, so a larger size is better than more speed. |
| Margins of the slider ends | 0.98 of the fastest allowed size, 0.9 of the largest | `feel.js` `QUICK_MARGIN`, `SMART_MARGIN`. A goal exactly at a predicted speed would be missed by a small error. These keep both ends inside what the sizes reach. |
| Fixed-time hardware test: how long one answer-speed reading runs | at least 2.0 s and at least 32 answer tokens, at most 7.0 s | `cale/tuning.py` `WINDOW_S`, `MIN_TOKENS`, `WINDOW_MAX_S`. A reading is a window of time, not a fixed amount of work: the test took 12 s on a Mac and 311 s on a 4-core laptop when it was a fixed number of tokens. With windows it is 31 s and about 110 s. The floor of 32 tokens exists because a 5 tokens/s laptop makes only 10 tokens in 2 s, and the speed then varied by 32% between two runs (5.4% with the floor). The cap keeps a slower computer's test bounded: fewer tokens and a less exact reading, never a longer test. |
| Fixed-time hardware test: settling before the first reading | windows of 2 s (16 tokens at least), until two in a row agree within 8%, at most 3 | `cale/tuning.py` `SETTLE_S`, `SETTLE_AGREE`, `SETTLE_MAX`. The computer is brought to its working state (clocks up, caches full). If it does not settle, it is measured as it is and the noise is recorded. |
| Fixed-time hardware test: prompt lengths | answer readings use 48 tokens. The table's prompt is about 1.0 s of reading (48 to 1024 tokens). The long prompt of the prefill curve is about 2.0 s. The two pieces of the working-memory reading are about 1.5 s each. | `cale/tuning.py` `PROMPT_MIN`, `PROMPT_MAX`, `PROMPT_S`, `PROMPT_LONG_S`, `MEMORY_PIECE_S`. Time sets the length, so a slow reader reads fewer tokens and a fast one more. |
| Fixed-time hardware test: the graphics-card piece reading | at most 12 s | `cale/tuning.py` `GPU_PIECES_S` |
| Window of the placement probe (CPU against graphics card) | a prompt piece of about 4 s of reading (it was 8 s) | `cale/tune.py` `probe` |

## Model facts

Layer count, the first layer that shares a KV cache, bytes per weight, the context limit, bytes of KV per token, the size ladder: `cale/gemma3n_facts.py`, `nest/levels3n.py`.

Also a model fact, and an estimate: `gemma3n_facts.ATTENTION_SHARE` (0.25), the work of a layer apart from its feed-forward part, in units of a half-width feed-forward part (about 12M against 50M weights). It gives the share of the prompt work that sits in layers 20 to 34 (0.43 for the full size). Measured against it: layers 20 to 34 took 40% of a 512-token piece on the Mac (MLX), and skipping them made prompts 1.66 to 1.8 times faster on three CPUs. Prompt reading skips those layers when no output is wanted (`llama/patches/0003`).

## Open: still written in the code, depend on the computer

| What | Value | Where | How to close it |
|---|---|---|---|
| Working memory for reading a prompt, **when `cale tune` has not run** | 500 MB | `gemma3n_facts.SCRATCH_BYTES` | **Tried and reverted:** the llama engine reports its compute buffers (shim `cale_compute_bytes`), but that is memory reserved, not memory touched. Using it over-predicted by 0.6 GB on Linux, so memory goals chose smaller sizes than they needed. Next: the peak of the process while the first prompt is read (poll during the warm-up), on both engines. |
| Speed priors before the model is loaded | `predict.PRIORS` | `cale/predict.py` | Save the speeds in `predict.json` on every computer (done after the first load). Until then a computer that never loaded the model shows the prior as an "estimate". |
| Memory the MLX engine keeps for activations | 1.5 GB | `nest/runtime.py` `RESERVE_GB` | **Not used by the product.** Only research scripts (`scripts/bench.py`, `serve.py`, `stream_speed.py`) pass a RAM budget to `NestedGemma.load`. The product loads everything and sizes memory from the measured working memory (`cale tune`: 809 to 946 MB on the Mac). Checked 2026-10-13. |
| Memory pressure levels | `cale/pressure.py`, `oslayer.py` | | Calibrate per OS with a stress test. Not done: it needs a computer that can be pushed into swap safely. |

## Rules of choice in `cale tune` (policies, not machine numbers)

| Policy | Value | Where | Why this value |
|---|---|---|---|
| A choice needs a clear gain | 1.15 (one reading), 1.08 (alternating runs) | `tuning.GAIN`, `tuning.NOISE` | The same computer measured twice differs by 10 to 25% (the drift probe). |
| Reference call that picks a way of running | 512 prompt and 128 answer tokens | `tuning.REFERENCE` | A typical app call. Changing it changes only which card wins when prompts and answers disagree. |
| Answers may be this much slower on a card | 0.8 times the CPU speed | `tuning.ANSWER_FLOOR` | A card must not make answers clearly worse. Reported in `placement.json`. |
| Sustained test length | 120 s | `tuning.py` | A fanless Mac slowed only after 93 s. |
| Probe a card with this share of CPU speed | 0.5 (it was 0.3) | `tuning.SKIP_BELOW` | A way whose first reading (a short prompt) is under half of the CPU's is dropped after that reading. Measured on a laptop with four ways, the long-piece speed was 1.9x (CUDA), 1.5x, 1.0x and 1.0x the short one: never more than twice, so a way under half of the CPU cannot win. |
| Time budget of the placement test | 120 s | `tuning.PLACEMENT_BUDGET_S` | No new way of running is STARTED if it would end after this (judged by the longest trial so far; the first way is always tried). Ways are tried in the order most likely to win: a card with its own memory before an integrated one, its own vendor's driver first. Each way costs a model load (20 to 34 s onto a graphics card), so the number of ways is what grows the test. Without a budget the placement test took about 295 s on a laptop with a card (three ways on two cards); with the budget, the lighter probe and the dropping of hopeless ways it took about 234 s with a budget of 180 s. |
| Verify a thread choice by alternating runs | always, 3 runs each | `tuning.threads` | One reading can be off by 10 to 25%, so no gain skips the check. |
| A loss smaller than this is "inside the noise" and not saved (context slope) | `NOISE` - 1 = 8% | `tuning.context` | The predictor keeps its prior instead. |
| Working memory | the larger of two readings | `tuning.working_memory` | One reading can be low. |
| Sustained speed | not measured if the computer was hot at the start | `tuning.sustained` | A hot computer has slowed already. |
| Heat model recovery fraction clamp | 0.05 to 0.95 | `tuning.cooldown` | Keeps a bad reading from giving an infinite or zero time constant. |
| Rest before the cooldown reading | 40 s | `tuning.cooldown` | Long enough to see recovery on a laptop, short enough to run at install. |
| Predictor weights `BATCH_EPS` 0.7, `FIRST_WEIGHT`, `LATER_WEIGHT`, and `PRIORS` | see `cale/predict.py` | `predict.py` | **Open:** tuned on the Mac and the HP. Used only until `cale tune` has measured. |
| Card choice when CUDA and Vulkan see the same card | CUDA is listed, Vulkan hidden | `devices.one_per_card` | **Open:** on the HP, Vulkan answered faster (7.7 against 6.1 tokens/s) but CUDA read prompts far faster. A neutral rule would measure both. |

## Added 2026-10-15 (adherence and sampling work)

| What | Where | Kind | Source or how it is replaced |
|---|---|---|---|
| Cost of other programs' CPU use: `LOAD_ALPHA` 1.0 (speed = 1 / (1 + alpha * busy)) | `cale/predict.py` | Measured, learned | The prior comes from one sweep (`results/demo/sweep-windows.json`, HP 4 cores: busy 0.5 gives 0.61-0.68x, busy 1.0 gives 0.5x). Each computer learns its own alpha from answers that ran alone (3 quiet answers, then every busy one). |
| Pause time is not decode time | `cale/batching.py`, `cale/predict.py` | Measured | Pause intervals are timed per answer. |
| Trial load for a small or failing graphics card: `CANARY_FREE_BELOW` 3 GB, `CANARY_TTL_S` 600, `TROUBLE_S` 24 h | `cale/backends/llama.py` | Policy | The free memory comes from the device. The 3 GB line is a policy for "small card". Reported in the log. |
| CPU after a GPU failure: `GPU_COOLDOWN_S` 600 (`CALE_GPU_COOLDOWN_S`) | `cale/batching.py` | Policy | Then the GPU is tried again. |
| Speed-goal adjuster: `STALL_S` 3 s, `STALL_RATIO` 0.92, up needs 3x the time (`CALE_ADJUST_MODE`, `CALE_STALL_S`, `CALE_STALL_RATIO`) | `cale/batching.py` | Policy | 0.92 is the old rule's own number. The new rule only changes what is measured: steps in which nobody read a prompt. Compare old and new on a machine before keeping. |
| Passive samples: `KEEP` 4000 lines, `HOST_SAMPLE_S` 600 | `cale/samples.py`, `cale/engine.py` | Policy | About 1 MB per computer. Numbers only. |
| Model objects survive a restart: `MAX_AGE_S` 24 h, `REVIVE_LEASE_S` 600 | `cale/objstore.py` | Policy | Documented in `docs/API.md`. |
| Re-measure when the placement is stale: `REPLACE_IDLE_S` 180 (`CALE_REPLACE_IDLE_S`), `REPLACE_EVERY_S` 1 day | `cale/engine.py` | Policy | Only while nobody uses Cale. An arriving app stops it at once. |
| Smoothing of other programs' CPU use: `BUSY_TAU_S` 6 s | `cale/engine.py` | Policy | A 2 s reading swung predictions by 25%. |
| Overshoot: `OVERSHOOT` 1.3 | `cale/engine.py` | Policy | Beating every goal by more than this with a larger size allowed is a calibration error. |
