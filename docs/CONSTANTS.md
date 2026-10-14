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

## Model facts

Layer count, the first layer that shares a KV cache, bytes per weight, the context limit, bytes of KV per token, the size ladder: `cale/gemma3n_facts.py`, `nest/levels3n.py`.

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
| Probe a card with this share of CPU speed | 0.5 | `CALE_PROBE_SKIP_BELOW` | A card under half the CPU on 64 tokens is dropped to save time. |
