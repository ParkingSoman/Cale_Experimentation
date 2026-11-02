# Getting Cale ready for a computer (`cale tune`)

Cale measures the computer once, when it is installed. The person sees one line: "Getting Cale ready for this computer. This takes
a few minutes and happens once." No app ever waits for a measurement, and nothing here is written for one computer.

```
cale tune                    the default profile (full)
cale tune --profile quick    seconds on a fast Mac, about two minutes on a weak laptop
cale tune --profile max      everything, including the long tests
```

The installer runs `cale tune` before it starts Cale. If it fails, Cale says so in one line and measures a little at the first load
instead. What it finds is kept in the Cale home folder (`predict.json`, `placement.json`, `engine-options.json`, and a full report in
`tune-report.json`: every probe, its seconds, and what it found).

## The rules every probe follows

1. Use the real model and the real engine. A made-up test misses the code path that real answers use.
2. Warm up first and throw the warm-up away: clocks rise, caches fill, memory settles. (Measured on a fanless Mac: the first reading was
   40% too slow.) The warm-up is itself a fixed-time step: short windows are repeated until two in a row agree within 8%, at most three.
3. **Fixed time, not fixed work (since 2026-10-18).** A reading is a window of time, not a fixed number of tokens. It runs at least 2 s and
   until 32 answer tokens were seen, and never more than 7 s. Prompt lengths are set by time too (about 1 s of reading, never a fixed
   count). So the test takes about the same time on a slow laptop as on a fast Mac: a fast computer fits more tokens into the window, a slow
   one gets a less exact reading, never a longer test. Before, the same test took 12 s on a Mac and 311 s on a 4-core laptop. The floor of
   32 tokens is there because a 5 tokens/s laptop makes only 10 tokens in 2 s, and the speed then varied by 32% between two runs (5.4% with the floor).
   For the numbers that matter, take the median of several windows.
4. Never change a setting unless the new one is clearly better. The same computer measured twice moves by 10 to 25% (the drift probe
   shows it), so a choice made from one run needs a gain of 15%, and one verified by alternating runs needs 8%. The other speed may be
   12% lower and still count as "not worse".
5. A graphics card is tried in its own process. A driver that fails or hangs cannot take the installer with it.
6. Record the power state and the cores the numbers were measured on.
7. Check it with `scripts/tune_check.py`: it runs the real `cale tune` several times and reports the time, the spread between runs, and the error
   against a slow measurement of every size (`scripts/sizes_accuracy.py`).

## What is measured

Seconds are measured on the three test computers (Mac: M-series, 24 GB, MLX. HP Windows: i7-8550U 4 cores, MX150 with CUDA. HP Linux:
the same laptop in WSL, CPU only). `full` runs everything in the "full" column; `max` runs all of it.

| Probe | What it measures | What it decides | Profile | Mac | HP Windows | HP Linux |
|---|---|---|---|---|---|---|
| placement | CPU, and each graphics card (CUDA, Vulkan), on a real prompt piece, each in its own process | Which way of running is fastest. A card is used only if a reference call (512 prompt tokens, 128 answer tokens) finishes 15% sooner and its answers are at least 80% as fast as the CPU's. CUDA versus Vulkan is decided by this, not by a rule. | full (llama) | - | about 3 min | - |
| environment | power (plugged or battery), cores, core types, RAM | what the numbers were measured on | quick | 0 s | 0 s | 0 s |
| disk | cold read of the model file, no OS cache | how long reading weights back takes (size changes under a memory goal) | quick | 0.2 s | 1 s | 1 s |
| working_memory | the most memory the process holds while the longest prompt piece is read. **Fixed time:** the longest piece is started and stopped after 3 s (the memory of a piece is taken when its first pass starts); under 16 MB is not saved | memory goals and the safety limit (replaces a guess of 500 MB) | quick | 7 s (855 MB) | 14 to 24 s (the settling is in this figure) | 35 s (200 to 490 MB) |
| threads | threads for answers and for prompts, separately, among half the cores, the cores, twice the cores and the fast cores of a hybrid CPU | thread counts; chosen only if clearly better, always checked by alternating runs (the default count is measured first, after a warm-up at that count) | full (CPU answers) | - | 15 s | 100 s |
| sizes | prompt and answer speed at four spread-out sizes (median of runs). **Four since 2026-10-18** (was five): `scripts/sizes_accuracy.py` measured all 16 sizes and built tables from subsets. Four predicted the others as well as five: Mac 1.3% against 1.6% mean error (3.9% against 2.8% at worst), HP 2.4% against 2.5% (10.7% against 10.8%). Three cost about 2 points on the Mac. | the table the predictor reads. Sizes between are read off by cost. | quick | 6 s | 53 s | 41 s |
| prefill_curve | prompt speed by prompt length | first-token time for long prompts | quick | 2 s | 19 s | 29 s |
| context | answer speed with a long context | how much answer speed is lost per 1000 tokens of context (two readings; a loss inside the noise is not saved) (Windows: between 2.5% and 24% per 1000 on different runs, so noisy). **Moved to `max` on 2026-10-18:** on the Mac (4096 tokens) and on the 4-core HP (918 tokens) it found no loss and saved nothing, for 6 s and 62 s. | max | 3 s | 25 s (62 s on 2026-10-18) | 34 s |
| gpu_pieces | speed of each prompt piece length when a card reads prompts | the piece length (no app pays for trying) | full (card) | - | 18 s | - |
| sustained | speed after two minutes of work against the first seconds | slowdown of hot laptops (Mac fanless: 0.89 after 93 s; Windows with the card: 0.87 after 40 s; Linux: 0.96 to 0.99). **Moved to `max` on 2026-10-18:** it was 115 to 120 s of every install (84% of a Mac install). Cale now learns the slowdown from real long answers that start on a cool computer (`predict.observe_pace`, saved in `predict.json` as `sustained`, `n_sus`). Limit: answers of 30 to 40 s show a smaller slowdown than the 2-minute test did (a Mac measured 3% in 30 s against 11% in 2 minutes), so very long work is still caught by the live speed controller, not by the prediction. | max | 115 s | 120 s | 120 s |
| drift | the first measurement repeated at the end | how noisy this computer is | quick | 1 s | 7 s | 9 s |
| batch | speed of 2, 3 and 4 answers at once | the cost of each extra answer (the engine also learns it while running) | max | - | 35 s | 31 s |
| cooldown | speed after a rest | how fast a hot computer recovers (the time constant of the heat model) | max | - | - | 47 s |
| engine options | flash attention, and passes of 1024 and 2048 tokens (card only), each in its own process, judged against two readings of the default | engine settings; kept only if clearly better | max | - | about 2 min | no effect (CPU) |

Total, with the default profile (measured 2026-10-18, `scripts/tune_check.py`; the install timeline is `scripts/timeline_report.py --anchor install_runtime`):

| | Fixed work (before) | Fixed time (now) |
|---|---|---|
| Mac, the hardware test | 131 s (12 s without the 120 s heat test) | **23 s in every run** (probes: working memory 7, sizes 14, prefill curve 2) |
| Mac, the whole install (model linked, not downloaded) | 156 s | **35 s** |
| HP Windows, the hardware test (CPU engine) | 422 s, then 311 s without the heat and context tests | 64 to 119 s (the threads step is 15 s when no thread count is clearly better, 45 s when one has to be confirmed) |
| Speed of each size against real answers (`scripts/predict_check.py`) | Mac 0 to 2% | **Mac within 1 to 2%** (gate 25% and 10%) |

The test is the same length every time on the Mac, because a reading is a window of time. On a slower computer a window holds fewer tokens, so the reading is less exact, but the test is never longer than its caps. The caps are in `docs/CONSTANTS.md` ("Fixed-time hardware test").

What is left on the HP, in seconds: threads 15 to 48 (worth it when it finds more threads: 26% faster answers on one run, 5% on another, which is why a change needs a clear gain), sizes 24 to 43, working memory 14 to 24 (this includes the settling of the computer), the repack of the model file 59 to 64, and the unpack of the chat (56 s for 12,194 files; it was 251 s for 61,000).

## How the results are used

* **Predictions.** `cale/predict.py` reads the measured table. A real answer then moves a correction factor gently. Answer speed
  predictions on the Mac are 7% off before the first answer and 9% after (gate: 25% and 10%). A hot laptop is predicted slower
  (a heat model: recent work builds heat, idle time cools it, the sustained slowdown scales the speed).
* **Settings.** Threads, flash attention and pass sizes are read when the model loads.
* **Placement.** `devices.choose` uses `placement.json` while the cards and the driver are the same as when it was measured.
* **Memory.** The measured working memory replaces a constant in the memory goals and the safety limit. It is raised if an answer shows
  more in use.

## Measured, then dropped

| Idea | Why it was dropped or limited |
|---|---|
| Flash attention on a CPU | no difference (15.7 and 6.79 tokens/s default, 16.0 and 6.74 off, 15.7 and 6.85 on). An early "win" was noise. The probe now runs only with a graphics card. |
| Stopping the sustained test early | a fanless Mac showed no slowdown for 90 seconds and then lost 11%. An early stop would have called it steady. |
| Prompt pieces longer than 512 on a CPU | cost memory, gain nothing |

## Not measured, and why

| Idea | Reason |
|---|---|
| Speculative decoding (a smaller size drafts, or a copy of text from the prompt) | Needs an engine feature first. A rough estimate from the research: the smaller size is only about twice as cheap, so with 70% accepted the gain is about 1.0 times; a prompt-copy draft helps summaries, questions on a given text and edits (2 to 4 times on a strong GPU) and nothing on open writing. On a CPU that is limited by arithmetic (the HP), extra tokens in one pass cost about as much as separate ones. Worth building and then measuring per computer. |
| Thread affinity and pinning on hybrid CPUs, `--poll`, process priority, Windows EcoQoS | The research says the effect can be large (3 times, in one report, for P-cores only on a 12th-generation Intel), but none of the test computers has a hybrid CPU, so a change could not be checked. Core types are detected and the fast-core count is one candidate in the thread probe. |
| KV-cache quantization | A quality choice for the developer, not a speed measurement. |
| NUMA, huge pages, CUDA graphs | Aimed at servers, or no evidence of an effect on a laptop. |
| Memory pressure levels | Needs a computer that can be pushed into swap safely. |

## What the research said (2026-10-13)

Three research agents searched the web. Evidence was thin: most numbers come from blogs, forum posts and issue threads. Points that shaped
the design:

* Answer speed is limited by memory speed, prompt speed by arithmetic. Extra threads can hurt answers (4 beat 7 on a 7-core CPU; a
  12-thread Apple chip fell below 6 threads) but help prompts. So the two thread counts are measured separately.
* Sustained speed of thin laptops can differ from burst speed a lot (reports from 5% on a fanless Mac to half on a phone), and no
  rigorous data exists for LLMs. So Cale measures its own, over two minutes, and keeps a heat model.
* A graphics card must be tested in a separate process: Windows resets a stuck graphics driver after about 2 seconds of a hung job.
* llama.cpp's own `--fit` reads free memory and sets layers. It does not time anything. Ollama sizes by free memory too. Neither
  measures speed. (This is where Cale goes further.)
* Disk speed must be read without the OS cache (Windows `FILE_FLAG_NO_BUFFERING`, Linux `posix_fadvise`, macOS `F_NOCACHE`).

## What the HP showed (2026-10-13, Windows, `max` profile)

| Way of running | Prompt tokens/s | Answer tokens/s | Reference call vs CPU |
|---|---|---|---|
| CPU | 17.5 | 7.14 | 1.0 times |
| CUDA, card reads prompts | 82.5 | 6.09 | 1.73 times sooner (chosen) |
| Vulkan on the same card | 20.1 | 7.68 | 1.12 times (not clearly better) |
| Vulkan on the Intel graphics, all layers | 7.3 | 4.4 | dropped after 64 tokens |

Two findings changed the design:

1. The first rule asked that answers be no more than 12% slower. CUDA answers were 15% slower on that run (about 5% on others). A rule built from
   one speed alone threw out a way that was 1.7 times better for a normal call. The rule now compares the time of a whole reference call.
2. One reading of the default was low by chance (5.9 against 7.0 answers). Every change then looked like a win. The options probe now
   reads the default twice and compares against the mean.

The context probe gave 2.5% loss per 1000 tokens on this run (2403 tokens of context) and 23% on the Vulkan run (678 tokens). Earlier runs gave 19 to 24%. The probe is noisy at short context, so the predictor keeps the number but the answer correction fixes what it gets wrong.

## Failure handling (from the overfit review, 2026-10-13)

* A card whose probe fails is never called "slower than the CPU". If any probe failed and nothing else won, no `placement.json` is saved. The report lists it under `problems`, and the installer says that some measurements did not finish.
* A saved "whole model on the card" decision is checked against free card memory at every load. If the memory is not free, Cale uses the card for prompts only, or the CPU, and says so in the reason.
* A saved device that is not found now is named in the reason.
* An old `placement.json` is deleted at the start of every `cale tune`.
* The "full" way is tried first on a card that may hold the model. If it does not win, the prompts-only way is tried too.
* Engine options (flash attention, pass size) are tried only by the `max` profile. A later `full` run uses the defaults and says so in the report.
* The sustained test is skipped when the computer was already hot at the start. Working memory is read twice and the larger reading is kept.
