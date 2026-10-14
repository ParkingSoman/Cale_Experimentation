# STATUS (2026-10-14): prompt reading (the next direction, chosen by Shonu after the overview)

Name: the Cale equation = (app types that work well) x (environments each one runs in). The biggest limit on it was slow prompt reading on CPU-only laptops (about 17 tokens/s on the HP), which fails every long-prompt app type.

| Step | State |
|---|---|
| 1 Measure the ceiling | Done. Layers 20 to 34 are skipped for every prompt token except the last: 1.66 to 1.80 times faster prompts on HP Windows, Mac CPU. MLX already did it for all pieces but the last. `FINDINGS.md`, "Prompt reading". |
| 2 llama.cpp patch | Done: `llama/patches/0003-cale-prefill-kv-exit.patch` + shim option key 4 (the last token alone). The early exit changes no number (bit for bit). |
| 3 MLX | Done: the same rule in `nest/gemma3n.py` `_prefill`. First token 1.4 to 1.7 times sooner for prompts of 70 to 400 tokens. |
| 4 Gate | Done on Mac, HP Windows, HP Linux, Linux arm64 (container): all suites pass (`prefill_exit` is new), installer end to end with new archives. Zoo long-document app on HP Linux: was TIMEOUT at 280 s, now OK in 185 to 194 s. With CUDA on HP Windows: 35 s. |
| 5 Saved prompts on disk | Not started. Decision first: reuse a saved prompt across sizes? (Found: sizes that agree in layers 0 to 19 share the memory exactly, 9 groups for 16 sizes. See FINDINGS.) |
| 6 Read ahead | Not started. |
| 7 Read at a smaller size, answer at full size | Not started. First result looks promising (KL 0.0001 to 0.033, 3 prompts). Developer choice only. |

# STATUS (2026-10-13, end of the session that ran this plan)

Install-time tuning (`cale tune`, see `docs/TUNING.md`) is done and tested on Mac, HP Windows (installer end to end, CUDA chosen), HP Linux (installer end to end) and the arm64 container. All suites pass on all of them. The only failure is a known container artefact (`reside e4b reads everything back`, 7 GB container).

| Stage | State |
|---|---|
| 2b GPU | Done, with option D (the GPU reads prompts, the CPU answers; see FINDINGS). Per-size GPU placement is not needed with D (0 layers on the GPU). CUDA done (installer adds the CUDA 12.4 backend, driver check, prompts 4.7 times faster than the CPU on the HP). Which way to run is measured at install (`cale tune`), not guessed. |
| 3 Gemma facts | Done (`cale/gemma3n_facts.py`). The small-drafter idea was rejected by arithmetic. |
| 4 Goals in tokens/s, floor | Done: no timing run, `on_miss` best_effort/refuse/wait/try, `priority`, 507 safety limit. Prediction accuracy 6-16%. |
| 5 Several apps at once | Done on llama (slots, scheduler). MLX stays one at a time. Known limits 1 to 5 fixed. Limits 6 to 13 are in `FINDINGS.md` ("Review of stages 3 to 8"). |
| 6 Cloud-API gap | Done: tools, JSON schema, prompt cache, pictures and sound (Mac only). `logprobs` on both engines. Not done: pictures and sound on llama. |
| 7 Any language, no screen | Done: leases, tickets, `cale allow`, Node client, `cale` command. |
| 8 More computers | Linux arm64 tested in a container. Windows arm64 builds, not tested. The installer unpacks engine archives for windows-x86_64, linux-x86_64, linux-arm64. |

The stage order changed: stage 6 was built before stage 5.

# Cale master plan (Phase 6 onward), rewritten 2026-10-13

This file replaces `~/.claude/plans/you-were-finishing-up-splendid-willow.md`. It keeps that plan's stages 3-8 and replaces its stage 2b with what the GPU experiments of 2026-10-12/13 measured.

## 1. Why Cale exists (purpose)

Shonu's goal (2026-10-12):

> Success = (number of app types that work **well**) × (number of consumer environments each runs in).
> App types: measure the gap between Cale and what developers can do with cloud AI.
> Environments: operating systems × the compute each computer has.
> Write once, run everywhere.

- Cale is one runtime plus one strong nested ("Matryoshka") model. The model downloads once. Cale picks a size for each computer and each request.
- Gemma 3n is the prototype model. Later Shonu plans his own laptop model.
- For this prototype, Gemma 3n is the final model. Cale is tuned for it alone. That tests "optimize for one model only", which general tools (Ollama, LM Studio) cannot do.
- Cale absorbs per-OS cost: the fastest engine per OS stays (MLX on Mac, llama.cpp on Windows and Linux). Shared logic sits above the engine interface.
- Rules for agents live in the repo: `docs/PRINCIPLES.md`.

**The weak-machine rule (Shonu, 2026-10-13).** The HP is a weak machine for Gemma 3n. A future, much stronger Matryoshka model will put strong machines in the same spot at its larger settings. So no fix is "for the HP". Every fix is either:

- **General**: Cale decides. The app sees the same behavior everywhere. Only speed differs, and Cale reports it.
- **Developer choice**: the behavior can differ (prompt speed vs answer speed, quality vs speed, memory vs speed, what to do when no size can meet the goals). The developer picks it explicitly. Cale reports what ran.

Size counts, layer counts and device names are never hard-coded. Logic works for any number of sizes, larger and smaller than today's 16.

How work is checked: a **hidden app zoo**. A reviewer agent reads only the public docs and writes 12 probe apps I never see. It scores "app types that work well × environments". A second reviewer reads each stage's diff and asks: does this fit the demo apps or one machine?

## 2. What is done

| Phase | Result |
|---|---|
| 1-2 | Nested Gemma: one file, many sizes, instant size switch |
| 3 | Cale runs Gemma 3n E4B (E2B nested). Open WebUI and Aider demos. |
| 4 | Memory holds and goals (deadline, memory share). 2.88 -> 1.41 GB. |
| 5 | Mac + Windows + Linux. llama.cpp engine with a width patch (`llama/patches/`), model objects API, installer, tests on all 3. Commit `a7f3fb7`. |
| 6, stage 0-1 | `docs/PRINCIPLES.md`, memory notes, hidden app zoo. Baseline 11 of 36 cells (`results/zoo/BASELINE.md`). |
| 6, stage 2a | Honest API (strict fields, `stop`/`seed`/`top_p`, one 32768 context limit, engine recovery, wait time counted, memory goal caps size, client disconnect stops work). Windows OpenMP fix (decode 4.1 -> 7.2 tok/s). Zoo 11 -> 18 of 36. All suites pass on Mac, HP Linux, HP Windows. Last commit `e936772`. |

## 3. Stage 2b: what the GPU experiments found (2026-10-12/13)

### Why the experiments were run

The first GPU test said "the GPU makes answers slower than the CPU". The old plan proposed a custom hybrid for that. Shonu rejected it:

- A GPU slower than the CPU looks like a bug, not a fact.
- A hybrid tuned to one 2 GB card is an overfit.

He was right.

### What was measured

HP laptop (i7-8550U, 16 GB, MX150 2 GB, Intel UHD 620). Official llama.cpp b11558 Windows builds, stock Gemma 3n E4B Q4_0. 512-token prompt. Tokens per second.

| Setup | Prompt | Answer |
|---|---|---|
| CPU only, repack on | 19.2 | 7.7 |
| CPU only, repack off | 11.8 | 3.8 |
| MX150, 0 layers, old setup (host buffer on) | 58.7 | 3.9 |
| MX150, 0 layers, `--no-host` | 21.1 | 7.4 |
| MX150, 0 layers, `--no-host`, no op offload | 18.9 | 7.4 |
| MX150, 16 layers, `--no-host` | 31.1 | 8.2 |
| **MX150, auto-fit (`-fitt 300`), `--no-host`** (run 3 times, two days) | **28.5** | **9.2** |
| MX150, 24+ layers by hand | out of GPU memory | - |
| Intel UHD 620, 2018 driver | crash at load | - |
| Intel UHD 620, driver 31.0.101.2145, 0 layers | 11.2 | 3.7 |
| Intel UHD 620, new driver, all layers | 11.6 | 4.4 |

### What it means

1. **The cause of the slowdown.** When a GPU backend is loaded, llama.cpp puts the CPU's weights in a "host" buffer that the GPU can copy from. Weights there are not repacked, so the fast CPU code does not run. Answers drop to the "repack off" speed (3.8). The setting `no_host` fixes it.
2. **A real trade-off remains for weights that stay on the CPU.** A weight is stored once. Repacked (fast CPU answers) or plain (the GPU can copy it to read prompts fast). Not both.
3. **Auto-fit works.** llama.cpp's own fit code (`common/fit.cpp`, `common_fit_params`) found a working layer count. My hand-picked counts ran out of memory.
4. **"Fits" is not "faster".** The Intel UHD 620 now runs, but loses to the CPU in every setup. It reports 7.4 GB free, so fit alone would put everything on it.
5. **GPU numbers change.** After the driver update, Vulkan0 and Vulkan1 swapped. Cale must pick a device by what it is (name, shared memory or not, size), never by its number.
6. **A strong GPU that holds the whole model has no trade-off.** That is the general case. The HP shows only the weak-GPU case.

### The trade-off, and Shonu's decision (2026-10-13)

| Setup for weak GPUs | Prompt | Answer |
|---|---|---|
| A. Auto-fit + `no_host` | 28.5 | 9.2 |
| B. 0 layers, host buffer on | 58.7 | 3.9 |

Shonu: find a way to get both with no extra resources. If there is none, optimize both and **let the developer choose** the trade-off. "A developer needs to feel confident that the code they wrote works the way they expect it to on every machine." (Memory: `feedback-developer-confidence`.)

**Why "both" is blocked today (read in b11558 source):**

| Code | Effect |
|---|---|
| `ggml-cpu/repack.cpp`: repack buffer type has `is_host = nullptr` | `ggml-backend.cpp:1113`: the scheduler offloads an op only if its weight buffer is host. So prompt matmuls on repacked weights never go to the GPU. |
| `ggml-cpu/repack.cpp:5178`: `get_tensor = nullptr` | A repacked weight cannot be read back in plain form |

The repack is a fixed byte reordering (Q4_0 blocks interleaved for AVX2). It can be undone.

**Plan for "both" (option D):** keep one repacked copy. When the scheduler sends a prompt matmul to the GPU, undo the reordering into a small pinned staging buffer (about 20 MB, one weight at a time) and upload that. Estimated cost: one read of the CPU-resident weights per 512-token piece, about 0.3 s against about 9 s of GPU work (estimate, not measured). Predicted result on the HP: prompt near 55-59, answer near 9.2.

**If D fails its gate:** ship A as default, B as an explicit developer choice. The choice is made through documented goals (`max_first_token_s` vs `min_tokens_per_s`, stage 4) or one explicit setting until stage 4 exists. The reply always says which setup ran.

### The general form of this trade-off (Shonu's question, 2026-10-13)

Shonu asked: does this trade-off exist on every machine when the weights must stream? Yes.

- Prompt reading handles hundreds of tokens per pass over the weights. Moving weights from slow memory is shared by all of them.
- Answering handles one token per pass. Moving weights is paid again for every token.
- So prompts tolerate weights in slow memory. Answers do not.

| Machine | Fast memory | Slow memory |
|---|---|---|
| Weak GPU (MX150) | GPU memory | RAM |
| Any laptop, model bigger than free RAM | RAM | disk |
| Mac | RAM | disk |
| Strong GPU that holds the model | GPU memory | none (no trade-off) |

Cale already has a hidden instance. `llama/shim/cale_llama.cpp`: repacked weights are fast but fixed in RAM. A memory-mapped file can be paged out (memory goals work) but uses the slow CPU code. A tight memory goal can make answers slower in silence. This breaks the developer-confidence rule.

Three general levers, applied to every tier:

| Lever | What | Stage |
|---|---|---|
| Format | Store the weights on disk already repacked for this CPU (convert once at install or first run). Pages stream and still use the fast code. Option D gives the GPU plain weights when it needs them. | 2b |
| Size | Nested model: read the prompt at full size (streaming is cheap there), answer at the largest size that fits in fast memory. Needs the "size switch between tokens" check and a quality check. | 3 |
| Prediction | Before the call, predict speed for each tier and size. If a quality floor forces streaming, the developer sees the predicted speed and chooses. The reply says what ran. | 4 |

Measure first in stage 2b: answer speed with a memory-mapped (not repacked) file against repacked, on HP Windows and HP Linux, to know how big the hidden instance is.

### Stage 2b work (after the discussion)

0. **Housekeeping (blocked in plan mode):**
   - Copy `C:\cale\jobs\gpu3.log` to `gpu7.log` and the body scripts into `results/gpu/`. Label gpu5 correctly (the device numbers swapped).
   - FINDINGS entry with the table above.
   - Add the weak-machine rule (section 1) to `docs/PRINCIPLES.md`, and give the overfit reviewer the section 3b sorting as its checklist.
   - Add Shonu's principle to `docs/PRINCIPLES.md`: a developer must trust their code behaves as expected on every machine. No silent per-hardware trade-off. Avoid it, else make it a documented developer choice and report what ran.
   - Copy this plan into the repo (`docs/PLAN.md`) so agents can read it (lesson L124). Update memory `project-phase6-progress.md` to point at it.
1. **Risk check first: the width patch on the GPU.** A Cale size uses a view of part of each FFN weight. Vulkan matrix multiply on a view of a quantized weight is not verified. Building Cale's library with Vulkan needs the Vulkan SDK on the HP (a setup cost). Gate: every Cale size gives the same answer on GPU and CPU (same seed, greedy).
2. **Build:** `GGML_BACKEND_DL` with CPU variants plus Vulkan, from b11558. Ship the official Vulkan DLL from the same tag. Files: `llama/build.ps1`, `llama/build.sh`, `llama/CMakeLists.txt`.
3. **Shim:** `llama/shim/cale_llama.cpp` hard-codes `n_gpu_layers = 0`. Add device choice, layer count and `no_host`. `common_fit_params` lives in `common/`, not the core library. Link it or copy the small part needed.
4. **Option D patch ("both speeds"):** a new patch in `llama/patches/` on b11558.
   - Repack buffer: add a "read back plain" function (inverse of the Q4_0 interleave, for the types Cale's file uses).
   - Scheduler: allow op offload for weights that can be read back plain. Copy through a pinned staging buffer.
   - First a test that a weight read back equals the original file bytes, for every repacked type in Cale's file.
   - Gate on the HP, same session, 3 runs: prompt at least 50 tok/s and answer at least 9.0 tok/s, with no rise in resident RAM over setup A (measure peak working set). Same answers as CPU only (greedy, same seed).
   - If the gate fails: setup A by default, B as a developer choice (see above). Record why D failed.
   - Then the format lever for RAM vs disk: a file stored pre-repacked for this CPU, memory-mapped. Gate: answer speed within 5% of repacked-in-RAM, and pages can still be dropped (memory goals still work). The file is CPU-specific (AVX2 interleave differs from ARM), so it is made on the machine, not shipped.
5. **Device choice in Cale:** `cale/backends/`. Rules:
   - Pick by device properties, never by index.
   - Use a GPU only when predicted or measured speed beats the CPU. Fit only sets the layer count.
   - Fit assumes system RAM is unlimited. Cale's memory goal still caps the part on the CPU.
   - Any GPU load error falls back to the CPU. Never crash.
6. **Gate:**
   - With a GPU loaded, answers are never slower than CPU only (HP Windows).
   - On the HP: MX150 used (with D: prompt at least 50, answer at least 9.0. Without D: setup A, plus B on request). UHD 620 not used.
   - The reply and `/cale/status` say which device and setup ran. A developer can read it.
   - All suites in `cale/tests/` pass on Mac, HP Linux, HP Windows. Zoo re-run. Overfit review of the diff.
   - Linux GPU: built, not tested (WSL2 has no real GPU driver path).

## 3b. Review of past and planned work: general fix or developer choice

Source: a read-only review of the repo (2026-10-13). No code checks for the HP or Mac by name. These items were tuned on one machine.

### General fixes (Cale decides; same behavior everywhere)

| Item (where) | Tuned on | General fix | Stage |
|---|---|---|---|
| Repack lost when a GPU is loaded | HP MX150 | `no_host`, plus option D (both speeds) | 2b |
| GPU never used (`cale_llama.cpp:93`) | HP MX150/UHD 620 | Device by properties and predicted speed. Auto-fit layers to free GPU memory. Never by index. | 2b |
| OpenMP off on Windows only (`build.ps1`) | one i7-8550U | Ship CPU variants with `GGML_BACKEND_DL`. Choose OpenMP or not per CPU by a short check at first run, or keep the faster one after measuring on 2+ CPUs | 2b |
| AVX2-only binary, x86 only (`CMakeLists.txt:17-24`, `install.py:68`) | HP | CPU variants (SSE, AVX2, AVX-512, ARM) loaded at run time | 2b, 8 |
| Repacked file fixed in RAM vs mapped file slow | HP | Pre-repacked file on disk, made on this machine | 2b |
| Threads = physical cores (`backends/llama.py:27`) | HP 4 cores | Detect performance cores on hybrid CPUs | 2b |
| Stale defaults `n_threads=4, n_ctx=8192` (`llamalib.py:31`) | HP | Remove. One source of truth. | 2b |
| `MARGIN = 1.25` (`profile.py:14`) | Mac GPU | Learn per machine from real misses | 4 |
| `WORK_BYTES = 400 MB` (`profile.py:15`) | Mac | KV bytes per token × context + measured compute buffer (Gemma facts) | 3, 4 |
| One short prompt timed; prompt speed per size guessed; never corrected (`profile.py:17, 55-67, 102-107`) | Mac | Prompt speed as a function of length and size, corrected from every call | 4 |
| Fixed timeouts 60/120/600 s (`server.py:214`, `client.py:36, 99`) | Mac speed | Scale from the prediction. Docs say how to set the client timeout. | 4 |
| `LOAD_GRACE_S = 30` from load start (`engine.py:39`) | Mac | Count from load end | 4 |
| Memory pressure thresholds (`oslayer.py:120-158`) | Mac carried to Linux/Windows | Calibrate with a stress test per OS | 4 |
| Quant type Q4_0 (`install.py:37`) | AVX2 repack | Keep one quant on all machines for equal quality. Change only with a quality gate. | 8 |
| WSL `total_ram` is the VM's RAM (`oslayer.py:60`) | HP WSL2 | Correct inside a VM. Document it. | 7 |

### Developer choices (behavior can differ, so the developer states it)

| Trade-off | Comes up when | How the developer states it | Stage |
|---|---|---|---|
| Prompt speed vs answer speed | Weights do not fit in fast memory (weak GPU today, big model tomorrow) and option D fails or streaming is needed | `max_first_token_s` and `min_tokens_per_s`, plus `priority` (below) | 2b default, 4 |
| Quality vs speed | No size meets both `min_quality` and a speed goal | `priority` | 4 |
| Memory vs speed | Fast setup needs fixed RAM, or weights must stream from disk | `memory_pct`, `allow_disk` (exists) | 4 |
| No setup meets the goals at all | Machine too weak for even the smallest size | `on_miss` (extended below) | 4 |
| Hold default `memory_pct = 25` (`server.py:382`) | Any machine | No hidden default: documented, and reported in the reply | 4 |

## 3c. When the machine is too weak for every size (the "floor")

**Today** (`choose.py:46-82`, `engine.py:263-352`, `server.py:117`): `on_miss` is `"smallest"` (default: run the smallest or fastest size, reply says `goal_met=false`) or `"refuse"` (422 `goal_missed` with an estimate). Memory goals are checked when the object is created. Deadline goals only per call. No way to ask "can this computer meet my goals?" before a call.

**New design (stage 4, built size-count-agnostic):**

1. **Ask before you run.** Creating a model object returns a prediction for every goal: met or not, predicted first-token time, tokens/s, memory, chosen size. A dry-run form of the same call exists for apps that want to check at start-up. `/v1/models` shows the machine's floor: the fastest and the smallest setup and what they achieve.
2. **`priority`**: the order in which goals may be given up when they conflict. Example: `["quality", "first_token", "speed", "memory"]`. Default documented.
3. **`on_miss` when nothing meets the goals**, chosen by the developer:

| Value | Behavior |
|---|---|
| `"best_effort"` (today's `"smallest"`, default) | Run the setup that misses least, by `priority`. Reply says what missed and by how much. |
| `"refuse"` | 422 with the floor numbers, before any work |
| `"wait"` | Queue until resources free up (other apps, memory pressure), up to `max_wait_s`, then act as `"refuse"` |
| `"try"` | Run anyway, even below the floor: every lever allowed (stream weights from disk, shorter context, any size). It may be very slow. The reply gives the predicted and actual numbers. |

**Safety limit (no setting turns it off).** Cale refuses any setup it predicts would crash or freeze the computer: out of memory, heavy swapping, or the OS killing other apps. This holds even with `"try"`. The reply is a separate, documented error (for example 507 `unsafe_for_this_computer`) with the numbers: what was needed, what is free. It is listed in the public docs, so developers know it can happen and can handle it.

4. **Levers below the smallest size**, used only when the developer allows them, each reported: stream weights from disk (`allow_disk`, exists), shorter context, smaller answer cap. Future Matryoshka sizes below today's smallest slot into the same ladder with no API change.
5. **Install and status say it once.** If this computer is below the floor for typical goals, `cale status` and the installer say so with numbers.

Gate: a zoo app on the HP with impossible goals gets the documented behavior for each `on_miss` value, including `"try"`. A request that would exhaust memory gets `unsafe_for_this_computer` under every `on_miss` value, and the computer stays responsive. The same app on the Mac gets met goals. No code path names a size.

## 4. Stages after 2b (from the old plan, unchanged in intent)

| Stage | Goal | Main points |
|---|---|---|
| 3 | Optimize for Gemma 3n on purpose | Gemma facts in one module (`cale/gemma3n_facts.py`): bytes per token per size, KV bytes per token, skipped layers. Try Gemma-only speedups: small size drafts for the full size, size switch between tokens, shared prompt cache across sizes. Keep a speedup only if it measures faster on Mac and HP with the same quality. |
| 4 | Goals in tokens per second, no timing run, and the floor (section 3c) | Includes the general fixes and developer choices marked stage 4 in section 3b. Goals: `min_tokens_per_s`, `max_first_token_s`, `memory_pct` (counting context), `min_quality`, `power`. `deadline_s` goes. No hidden `max_tokens` default. Predict speed from bandwidth ÷ bytes per token, correct from the first answer's first tokens, closed-loop size step-down. Here the weak-GPU choice B can serve first-token goals. Gate: first estimate ±25%, after first answer ±10%, goal met in 95% of answers. |
| 5 | Many apps at once | Batch several answers. One size per batch, the largest that meets every live goal. Fair order. Report conflicts. Gate: 2 and 4 zoo apps at once. |
| 6 | Close the gap with cloud APIs | `tools` and tool messages, `response_format` JSON schema (grammar), `logprobs`, prompt cache across turns, images and audio input on both engines. Gate: zoo tool, JSON, long-context apps on all environments. |
| 7 | Any app language, any computer setup | Model objects over plain HTTP with leases. `cale` command on the PATH. Python and JavaScript reference clients. Approval without a screen (`cale allow NAME`). Installer carries its own runtime. Gate: zoo Node.js and `curl` apps. Headless Linux approves from a terminal. |
| 8 | More computers | arm64 (Snapdragon, Pi 5 class). CPUs without AVX2. 8 GB RAM computer runs a useful size. |

Each stage ends with its gate, the zoo scorecard, a FINDINGS entry, and a commit.

## 5. Open items and questions

| Item | Status |
|---|---|
| Weak-GPU trade-off | Decided: try D (both speeds). If it fails, A default and B as a developer choice. |
| NVIDIA driver on the HP | Updated to 576.80 on Shonu's yes. CUDA versus Vulkan is measured on each computer at install. Old drivers still work (Vulkan or CPU). |
| Intel driver on the HP | Done: 24.20.100.6344 (2018) -> 31.0.101.2145. Installer exit 1000, HP restarted, ssh returned. |
| Linux Allow dialog click | Verified on a virtual screen (Xvfb, zenity, xdotool): Allow gives a key, Don't Allow gives 403. WSLg itself does not work over ssh. |
| Open WebUI and Aider on the HP since strict fields | Open WebUI through the API: connect, objects, chat verified. The page click-through did not finish (the ssh link drops). Aider re-run on the Mac against a test server: one edit, correct code written. Open WebUI is not installed on the Mac (disk), so its page was not clicked. |
| Mac disk | 28 GB free after Shonu deleted stale BlinkFix files (was 15 GB). Check again at the end of each session. |

## 6. Verification (every stage)

| Check | How |
|---|---|
| Unit and API tests | `cale/tests/*` on Mac, HP Windows, HP Linux (`scripts/hp`, 5-minute heartbeats) |
| Speed | `llama-bench` or `scripts/xplat_bench.py` streaming decode. 3 runs. CPU-only run in the same session as the comparison. |
| Hidden app zoo | Reviewer agent scorecard |
| Overfit review | Second reviewer reads the diff against `docs/PRINCIPLES.md` |
| Demo apps | Open WebUI and Aider still work |
| Disk | `df -h` on the Mac. Big builds on the HP. `DISK.md` ledger. |
