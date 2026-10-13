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
| Answer and prompt speed of each size | `cale/predict.py`, `cale/engine.py` `_measure_speed` | The first load on a computer reads a short prompt and answers a few tokens, once, and saves the result with the key of the computer and model. Then every real answer corrects it. `PRIORS` is used only before the model is loaded (the `/v1/models` view). Mac: first estimate 0% off. |
| Memory speed | `cale_membw` in the shim | A 0.15 s probe, no model. |
| Memory held besides the weights | `engine._sample_base`, `_grow_base` | The largest of five readings of the process, raised whenever an answer finds more in use. A prediction is never under. |
| Cost of an extra answer in a batch (`batch_eps`) | `predict.observe_step` | Learned from real steps. `BATCH_EPS` is only the start. |
| Disk speed | `predict.observe` | Learned from real reads. `DISK_PRIOR` is only the start. |
| The shortest prompt piece a GPU takes | shim `cale_offload_min_batch` | Asked of the GPU backend itself. |
| The longest prompt piece | shim `cale_n_batch` | Asked of the engine. |
| The best prompt piece for a GPU that reads prompts | `batching.PieceTuner` | Tries the sizes between those two, keeps the fastest within the time budget. If the CPU piece wins, the GPU is not used for prompts. |
| Which device runs the model | `cale/backends/devices.py` | From device type and free memory, never from a name or index. |
| Prompt piece next to running answers | `batching._step` | From the measured prompt speed and the policy `STEP_SECONDS`. |

## Policy

| What | Value | Where |
|---|---|---|
| Time one prompt piece may take for an app that leaves to be noticed | 6 s (`CALE_MAX_PIECE_S`) | `batching.MAX_PIECE_S` |
| Time of a prompt piece next to running answers | 0.75 s (1.5 s alone, on the CPU) | `batching.STEP_SECONDS` |
| Share of RAM a setup may need before Cale refuses it | 85% | `choose.SAFE_FRACTION` |
| Longest wait, lease, idle time, poll times | `cale/goals.py`, `engine.py`, `server.py` | |
| Request limits (schema size, pictures, sound) | `constrain.py`, `media.py` | |
| How far a hold lets the size drop under pressure | `engine.WARN_CAP` | |
| Largest size for `power: "light"` | 75% | `gemma3n_facts.LIGHT_MAX_PCT` |

## Model facts

Layer count, the first layer that shares a KV cache, bytes per weight, the context limit, bytes of KV per token, the size ladder: `cale/gemma3n_facts.py`, `nest/levels3n.py`.

## Open: still written in the code, depend on the computer

| What | Value | Where | How to close it |
|---|---|---|---|
| Working memory for reading a prompt | 500 MB | `gemma3n_facts.SCRATCH_BYTES` | llama.cpp prints the compute buffer size when it makes the context. Capture that line in the shim. On MLX, read the peak memory of the first prompt. |
| Memory a GPU needs besides the weights | 800 MB | `backends/devices.WORKING_BYTES` | Let llama.cpp's own fit say whether every layer fits, and read the answer. |
| The safety margin left free | 300 MB | `choose.SAFE_MARGIN` | The OS reserve: Linux `min_free_kbytes`, macOS page free target, a share of RAM on Windows. |
| Speed priors before the model is loaded | `predict.PRIORS` | `cale/predict.py` | Save the speeds in `predict.json` on every computer (done after the first load). Until then a computer that never loaded the model shows the prior as an "estimate". |
| Number of threads | physical cores | `backends/llama.py` `default_threads` | Probe memory speed at 1, 2, 4 ... threads and stop where it stops growing. Prompts want more threads than answers, so this needs two probes. |
| Memory the MLX engine keeps for activations | 1.5 GB | `nest/runtime.py` `RESERVE_GB` | Read the peak memory of the first prompt. |
| Time that memory pressure is ignored after a load | 30 s | `engine.LOAD_GRACE_S` | Use the measured load time. |
| Number of answers that run together | 4 | `backends/llama.py` `CALE_SLOTS` | From the memory left after the weights and from the measured batch cost. |
| Memory pressure levels | `cale/pressure.py`, `oslayer.py` | | Calibrate per OS with a stress test. |
