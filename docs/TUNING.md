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
   40% too slow.)
3. For the numbers that matter, take the median of several runs.
4. Never change a setting unless the new one is clearly better. The same computer measured twice moves by 10 to 25% (the drift probe
   shows it), so a choice made from one run needs a gain of 15%, and one verified by alternating runs needs 8%. The other speed may be
   12% lower and still count as "not worse".
5. A graphics card is tried in its own process. A driver that fails or hangs cannot take the installer with it.
6. Record the power state and the cores the numbers were measured on.

## What is measured

Seconds are measured on the three test computers (Mac: M-series, 24 GB, MLX. HP Windows: i7-8550U 4 cores, MX150 with CUDA. HP Linux:
the same laptop in WSL, CPU only). `full` runs everything in the "full" column; `max` runs all of it.

| Probe | What it measures | What it decides | Profile | Mac | HP Windows | HP Linux |
|---|---|---|---|---|---|---|
| placement | CPU, and each graphics card (CUDA, Vulkan), on a real prompt piece, each in its own process | Which way of running is fastest. A card is used only if prompts are 15% faster and answers are not clearly slower. CUDA versus Vulkan is decided by this, not by a rule. | full (llama) | - | see below | - |
| environment | power (plugged or battery), cores, core types, RAM | what the numbers were measured on | quick | 0 s | 0 s | 0 s |
| disk | cold read of the model file, no OS cache | how long reading weights back takes (size changes under a memory goal) | quick | 0.2 s | 1 s | 1 s |
| working_memory | the most memory the process holds while the longest prompt piece is read | memory goals and the safety limit (replaces a guess of 500 MB) | quick | 1 s (850 MB) | 6 s | 35 s (200 to 490 MB) |
| threads | threads for answers and for prompts, separately, among half the cores, the cores, twice the cores and the fast cores of a hybrid CPU | thread counts; chosen only if clearly better, checked by alternating runs | full (CPU answers) | - | 15 s | 100 s |
| sizes | prompt and answer speed at five spread-out sizes (median of runs) | the table the predictor reads. Sizes between are read off by cost. | quick | 6 s | 53 s | 41 s |
| prefill_curve | prompt speed by prompt length | first-token time for long prompts | quick | 2 s | 19 s | 29 s |
| context | answer speed with a long context | how much answer speed is lost per 1000 tokens of context (Windows with the card: 19 to 24% per 1000) | full | 3 s | 25 s | 34 s |
| gpu_pieces | speed of each prompt piece length when a card reads prompts | the piece length (no app pays for trying) | full (card) | - | 18 s | - |
| sustained | speed after two minutes of work against the first seconds | slowdown of hot laptops (Mac fanless: 0.89 after 93 s; Windows with the card: 0.87 after 40 s; Linux: 0.96 to 0.99) | full | 115 s | 120 s | 120 s |
| drift | the first measurement repeated at the end | how noisy this computer is | quick | 1 s | 7 s | 9 s |
| batch | speed of 2, 3 and 4 answers at once | the cost of each extra answer (the engine also learns it while running) | max | - | 35 s | 31 s |
| cooldown | speed after a rest | how fast a hot computer recovers (the time constant of the heat model) | max | - | - | 47 s |
| engine options | flash attention, and passes of 1024 and 2048 tokens (card only), each in its own process | engine settings; kept only if clearly better | max | - | see below | no effect (CPU) |

Total, with the default profile: about 2 minutes on the Mac, about 4.5 minutes on the HP with the card plus the placement comparison,
and about 6.5 minutes on the HP without a card (the weakest case).

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
