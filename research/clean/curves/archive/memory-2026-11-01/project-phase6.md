---
name: project-phase6
description: "Phase 6 (make Cale general, 2026-10-12 to 10-14): done list, measured findings, GPU data, tuning, prompt-reading plan (steps 1-4 done) (merged)"
metadata:
  type: project
---

Master plan `docs/PLAN.md`. Cale vision: [[project-cale-vision]]. Earlier phases: [[project-cale-phases-3-5]].

## Phase 6 progress

MASTER PLAN (2026-10-13, approved): `docs/PLAN.md` in the repo (copy of `~/.claude/plans/you-were-previously-executing-golden-shamir.md`). It replaces the older splendid-willow plan. Stage 2b was redone: the GPU slowdown was a setup bug (`no_host` fixes it), next is option D (both speeds), the weak-machine rule, and the `on_miss` floor design. See also [[feedback-cale-design-rules]]. Driven by [[project-cale-vision]]. Branch `phase5-cross-platform`. Baseline zoo scorecard: `results/zoo/BASELINE.md` (11 of 36 cells before; Mac 5/12, HP Linux 3/12, HP Windows 3/12).

**Done (stage 0, 1, 2a, part of 2b), all suites pass on Mac, HP Linux, HP Windows** (`cale/tests/`: engine_features, honest, e2e, relative, objects, oslayer, backend_llama):
- strict request fields (400 by name), `stop` / `seed` / `top_p` on both engines, one public context limit (32768), clean `context_length_exceeded`, engine failure -> 500 `engine_error` + reload, wait time counted (`waited_s`, `goal_met_actual`), memory goal caps object size, a client that leaves stops its work (`nest/cancel.py`).
- Security review caught: `/cale/status` (no key) must never show raw error text. Details go to the private log.

**Measured findings worth remembering:**
- Windows decode was 40-50% slower than Linux on the same HP CPU because of the **mingw OpenMP runtime**. `llama/build.ps1` now builds with `-DGGML_OPENMP=OFF`: decode 4.1 -> 7.2 (full), 6.2 -> 12.3 (smallest) tok/s. Linux keeps OpenMP (equal either way). NOT the cause: power mode, EcoQoS, priority, threads, pinning, memory, huge pages.
- MLX arrays belong to the thread that made them. A request that stops early leaves half-computed arrays and every later request fails with "no Stream(gpu) in current thread". Fix: `NestedGemma3n.materialize()` on early exit. (This was the zoo's "wedged engine".)
- HP prompt reading is ~15 tok/s (CPU). MX150 via Vulkan (official llama.cpp b11558, stock model): prompt 59 tok/s with 0 layers on the GPU, answers no faster; full offload fails (2 GB card). Intel UHD 620 (driver from 2018) crashes at load. Idea not built yet: runtime-loaded backends (`GGML_BACKEND_DL`) + official Vulkan/CUDA DLLs, hybrid "GPU reads the prompt, CPU answers".
- Benchmark that is steady: `scripts/xplat_bench.py` `stream_decode_tok_s` (the older difference-of-two-runs number is +-40% noisy).

**Next (see plan):** finish 2b (GPU design, Intel driver question for Shonu), stage 3 (Gemma facts in one module `cale/gemma3n_facts.py`, Gemma-only speedups), stage 4 (tokens/s + first-token goals, hardware-based prediction, no first-launch timing), stage 5 (several apps at once), stage 6 (tools, JSON schema, prompt cache, images/audio), stage 7 (any language, no-screen approval, installer without Python), stage 8 (arm64).

**Still not verified:** Linux Allow dialog click on WSLg (display stopped answering after `wsl --shutdown`; code path fixed and unit-tested). Open WebUI / Aider demos not re-run since strict field checking.

See [[reference-hp-testbench]], [[project-cale-phases-3-5]].

**2026-10-13 status:** step 0 of stage 2b (logs in `results/gpu/`, `docs/PRINCIPLES.md` rules 11-14, FINDINGS entry) done. Next: width patch on Vulkan check, GGML_BACKEND_DL build, shim device choice, option D patch. HP Intel driver now 31.0.101.2145.

**2026-10-13 later:** commit `bfcafd4`. GPU backends load at run time (`CALE_BACKENDS_DL`), width patch fixed for CPU/GPU split (scheduler looks through views). OPEN: MX150 fit is made for full size, so smaller sizes are 2.5-3x slower than CPU; Intel iGPU slower than CPU always. Next: per-size placement by measured/predicted speed, wire into `cale/backends` + installer, run all suites on the DL build, option D.

**2026-10-13 end of session (stages 3-8 built):** goals in tokens/s with no timing run (`cale/predict.py`, accurate 6-16%), `on_miss` best_effort|refuse|wait|try plus 507 `unsafe_for_this_computer`, several answers at once on llama (`cale/batching.py`; on the HP it gives fairness, not speed), tools + JSON schema + per-app prompt cache, pictures on MLX only (`cale/media.py`, `capabilities.images` in `/v1/models`; llama gives 400 `unsupported_content` since no mmproj file exists and making one needs the 16 GB original weights), `cale allow`, tickets, leases, Node client, installer unpacks `llama/dist/<os>-<arch>.zip|tar.gz`, Linux arm64 tested in a colima container, Windows arm64 only built. NOT done: audio, `logprobs`, option D (both speeds), CUDA test, per-size GPU placement. Lessons: Windows NUL looks like a tty (`oslayer.is_terminal`); a test file named `concurrent.py` shadowed the stdlib; containers can hold less RAM than the host (cgroup limit read in `oslayer`); a Linux build must not need libgomp. Details in `FINDINGS.md` "Phase 6 stages 3 to 8". Run suites with `scripts/run_suites.py MODEL --only NAME` (first arg is the model path, not a suite).

**Late 2026-10-13 additions:** the second zoo (`~/Cale_Zoo2`, 14 apps written from the public docs only) found real bugs, now fixed: Windows library never saw `CALE_SLOTS` (own CRT environment copy) so Windows ran one answer at a time (now `cale_set_slots`); MLX prompt pieces longer than the 512 local window lost facts in long prompts (`prefill_step`); several same-role turns gave 500 (`merge_same_role`); unknown size now 404; `cale stop` was missing; `cale allow` used `isatty` (Windows NUL). Lesson: env vars set in Python after start are NOT seen by a mingw-built DLL on Windows. Pass config by function. Known unfixed limits are listed in FINDINGS "Review of stages 3 to 8". Mac colima VM (`~/.colima`, 52 GB, shared with other projects) is the big disk user, not ours to delete.

**Later on 2026-10-13 (after a Stop hook said the plan was not fully executed):** built option D (patch `llama/patches/0002`, the GPU reads prompts while weights stay repacked in RAM; rule in `cale/backends/devices.py`; HP: prompt 33 vs 16 tokens/s, 1064-token prompt first token 22.7 s vs 75 s; answers unchanged), sound input on MLX (`input_audio` WAV, 30 s), batching limits 1-5, Open WebUI end-to-end, Allow dialog checked with a stand-in zenity (real dialog still unverified: no zenity in WSL). Lessons: (1) **check the exit code of every build in a job script**: my script hid `cmake configure failed` and I measured the old library twice; cmake -S must match the cached source dir (build from `C:\cale\code7`); (2) GPU prompt pieces must be long (256-512 tokens) or the copy of the weights eats the gain; (3) read the slot count only after the model is loaded (the crowd test skipped its batching checks for months). Still open: logprobs, pictures/sound on llama, NVIDIA/CUDA test, known limits 6-13 in FINDINGS, a real Linux Allow dialog click.


## Update 2026-10-13 (install-time tuning)
- Rule 7 changed by Shonu's direction: Cale measures once at install (`cale tune`, profiles quick/full/max, default full, a few minutes), not at app load. Docs: `docs/TUNING.md`.
- Placement (CPU vs each card, each in its own process) is chosen by the time of a reference call (512 prompt + 128 answer tokens), not by answer speed alone. On the HP, CUDA wins 1.73x.
- Engine options (flash, batch) are judged against two readings of the default: one low reading fooled the probe.
- Pictures/sound on the llama engine: sound impossible with llama.cpp tag b11558 (audio tensor not mappable). Vision mmproj conversion unfinished and distribution of the gated weights unsolved. Left undone on purpose.

## Prompt-reading plan (approved 2026-10-14)

Plan file: `~/.claude/plans/you-ve-done-a-lot-purrfect-pretzel.md`. Why: on CPU-only laptops prompts read at ~17 tok/s, so long-prompt app types fail (biggest limit on the Cale equation). Shonu chose this direction and said: Mac and HP only for now, cloud machines later.

Key finding: Gemma 3n layers 20-34 reuse KV, so they matter only for the last prompt token. Both engines run all 35 layers for every prompt token. llama.cpp `gemma3n.cpp` has a TODO for it. Estimate 1.6-1.7x (not measured).

Steps: (1) measure ceiling with width-0 on layers 20-34, no new code; (2) llama patch 0003; (3) MLX `_prefill`; (4) gate; (5) disk prompt cache; (6) read-ahead; (7) smaller size for reading (developer choice, measure quality first).

Open decision for Shonu before step 5: llama keeps saved prompt after a size change, MLX does not. Pick exact (same size only) or reuse across sizes (reported). The "no last session task is unfinished" overview: Stop-hook loop asked for items needing hardware we lack; nothing half-built.

See [[feedback-user-working-style]], [[project-phase6]].

**STATUS 2026-10-14 end (commit 0ac5348, DISK 6ef60ca): steps 1-4 DONE.** Layers 20-34 skipped for prompt tokens with no output (llama patch 0003 + shim option key 4 + MLX `_prefill`), rule `gemma3n_facts.hold_last_min` from measured speeds. Prompts 1.6-1.8x faster on HP Windows/Linux CPU and Mac CPU; MLX first token 1.4-1.7x sooner for 70-400 token prompts. Early exit is bit exact; reading the last token alone moves scores like any batch-shape change. All suites pass on Mac (MLX), HP Windows, HP Linux; arm64 container passes except the known 7 GB artefact. New engine archives in `llama/dist/`. Zoo long-document app: HP Linux TIMEOUT -> OK (185-194 s), HP Windows with CUDA OK in 35 s. The Mac's real installed Cale was NOT upgraded (Shonu decides).
**Next (not started):** step 5 saved prompts on disk (decide first: reuse across sizes; sizes that agree in layers 0-19 share memory exactly, 9 groups of 16), step 6 read-ahead, step 7 read small / answer big (KL 0.0001-0.033 on 3 prompts, measure properly, developer choice only). Reviewer idea: gather output rows after layer 19 so mixed steps skip too (bigger patch). `cale tune` could measure the shared-layer share instead of `ATTENTION_SHARE`.
