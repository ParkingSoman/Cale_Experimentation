---
name: project-cale-phases-3-5
description: "Cale build history phases 3-5 (2026-10-09 to 10-11): Gemma 3n nest, holds + relative goals, Mac+Windows+Linux; numbers, gotchas, what is not done (merged)"
metadata:
  type: project
---

Phase 2 (home-made 12B slicing) is SUPERSEDED: see [[project-research-history]] sections 1-2. Next phase: [[project-phase6]].

## Phase 3 - Gemma 3n + two real apps (2026-10-09)

Shonu's call (2026-10-09): drop the home-made 12B width slicing. Use Gemma 3n, which Google trained as a nest (E2B inside E4B, 16 official sizes). Keep the same story: one runtime, one loaded model, instant switch between a smart/slow and a fast/small size. Demo = two REAL open-source apps that already mix a small and a large cloud model, each pointed at Cale instead. Nemotron Elastic 30B was considered and parked (16.8 GB file, ~21 GB peak on this 24 GB Mac; MLX has no runtime-slicing code for it).

**Built (plan: ~/.claude/plans/okay-i-think-we-ve-functional-quail.md):**
- `nest/gemma3n.py` NestedGemma3n + `nest/levels3n.py` (pure-Python size table + `pick`) + `nest/gemma3n_levels.json` (Google's table). A size = per layer half/full FFN (8192/16384 channels) + skip layers 20-24. Switch ~0.03 ms.
- Cale API names: `gemma-3n:smart` (100%), `gemma-3n:fast` (48%), `gemma-3n:<48-100>` (nearest trained size under the memory cap). Default answer length 1024 tokens.
- App 1 `apps/open-webui/cale_webui.py` (+ `cale.js` card, port 8810): unmodified Open WebUI 0.11.4, "Get Cale" install card with progress, connect, then answers=smart, titles/follow-ups/tags/autocomplete=fast.
- App 2 `apps/aider/cale_aider.py`: asks once "Use Cale?", main=smart, weak model (commit messages)=fast, edit format whole.
- `cale/install.sh` has CALE_PROGRESS lines, `cale/fetch_model.py` (HF download with byte progress), CALE_MODEL_SRC (hard-link a local folder).

**Gate results:** C: runtime at full size = plain mlx-vlm, bit-exact (8/8 outputs, logits diff 0.0). A: official E2B is NOT bit-identical to an E4B slice (646 of 1469 tensors differ, separate post-training) but behaves the same on real text (loss 2.200 vs 2.200, top-1 vs E4B 0.807 vs 0.808).

**Gotchas that cost time:**
- Gemma 3n puts all probability on `**` after "The answer is", so the old letter-logit MMLU-Pro harness is noise (E4B 30.5 vs official E2B 29.7). Use GSM8K (E4B 77, E2B 73, n=100) plus KL-vs-E4B on held-out text (monotone) instead.
- Layers 20+ share KV from earlier layers: scoring without a cache is wrong.
- mlx-vlm's stream path adds a second BOS if the template text already has one.
- Open WebUI sends no max_tokens; Cale must default high.
- Starlette StaticFiles refuses symlinks that point outside the folder; hard-link instead.
- Never `pkill -f cale.server`: it also kills the installed service. Kill by port.
- Old 12B models and demo/inbox were deleted on 2026-10-09 with Shonu's OK. Only `models/gemma-3n-e4b-4bit` remains (a second 5.5 GB copy lives in Cale's own home after the Hugging Face install test). Disk got to 13 GB free once from swap; keep >15 GB.
- Open WebUI offline mode sets HF_HUB_OFFLINE, and the installer inherits it: strip it. snapshot_download has two byte counters (network + disk): count only the "Reconstructing" one.
- The real Hugging Face install ran in a browser (card, progress, byte-identical files). The Mac Allow dialog was never clicked in any test.

See [[project-research-history]] (superseded phase 2), [[project-research-history]], [[feedback-disk-hygiene]].

## Phase 4 - holds + relative goals (2026-10-10)

Shonu's idea (2026-10-10): a "small" or "large" model is relative to the Mac, so offer relative capabilities next to absolute sizes. Two layers like low vs high level languages: **holds** (app loads a size into RAM, it stays until the app unloads or quits; smaller sizes are free; a bigger size reads missing weights from disk for that call, never refuses) and **goals** (`gemma-3n:auto` + `cale_deadline_s`, `/cale/load {"memory_pct"}`). Shonu's hypothesis: relative goals make better local apps; absolute stays for completeness. Garbage-collection-style automatic unloading is a "maybe later".

**Decisions Shonu made:** speed goal = total time only (needs max_tokens); memory goal = % of total RAM; disk use is a per-call yes/no in relative mode; on a missed goal the app picks per call (default run smallest and report, or refuse with an estimate); hold ends when the app quits (not on memory pressure).
**My proposal, not Shonu's call:** an app with no hold gets an implicit full-size hold (10 min idle, ends on critical pressure) so Open WebUI/Aider-style callers keep working.

**Numbers (M5 24 GB):** per-layer embedding tables moved to disk (row reads): 4.20 -> 2.88 GB full, 2.12 GB at 75%, 1.41 GB at 48%, under 4% slower, text identical. Full size still bit-exact (gate C 8/8). Details and bugs: FINDINGS.md "Phase 4". Code: `cale/choose.py`, `cale/profile.py`, `cale/engine.py`, `cale/client.py`, `nest/gemma3n.py`; test `cale/tests/relative.py`; dev server `scripts/cale_dev_server.sh HOME` (port 47899).

**Gotchas:** MLX arrays belong to the thread that evaluates them: evaluate before storing and warm up at load. `np.take` on a memmap reads the whole array. Test with a short `CALE_IDLE_S` only when the test does not rely on pressure timing. Never run the dev server on 47821; kill by port. Shonu asked me to test through the real Open WebUI and Aider myself (done, in a browser and CLI, on the dev server). The installed Cale at 47821 still has the old code until `./cale/install.sh` is run.

See [[project-cale-phases-3-5]], [[feedback-disk-hygiene]].

## Phase 5 - Mac + Windows + Linux (2026-10-11), DONE

Goal (Shonu, 2026-10-11): one Cale runtime on Mac, Windows and Linux that adapts to each computer's power; both demo apps (Open WebUI, Aider) run on all three and state relative goals. Generic tools first, OS-specific code only where generic loses something, no per-OS forks. Plan: `~/.claude/plans/pasted-content-id-d6c3-make-compressed-dream.md`. **Status: implemented and tested on all three.** Full write-up: FINDINGS.md "Phase 5", `llama/README.md`, `cale/README.md`, `demo/README.md`.

**Decisions Shonu made:** fastest engine per OS, patch llama.cpp so ONE GGUF switches sizes (no two-file story; the stock E2B GGUF on the HP is only a test reference). Apps start Cale on demand (no service/scheduled task/Startup item on the HP). HP demo one OS at a time. Apps make a "model object" with parameters; no goals hidden in the model name or per-call JSON.

**Engines:** Mac = MLX. Windows + Linux = llama.cpp on the CPU with our patch (`llama/`, pinned b11558). MLX on a CPU is unusable (9.7 s for one 256-token matmul). Vulkan: MX150 out of memory, Intel UHD 620 crashes. Windows MLX needs `mlx-cpu` explicitly.

**Built:** `cale/oslayer.py` (all OS code), `cale/service.py` (start/stop with no service manager; server exits itself after 10 idle min in on-demand mode), `cale/backends/` (`llama.py`, `llamalib.py`), model objects (`POST /cale/models`, `cale.client.model(...)`, `ensure_running()`; ids per app and name, stable: `answers`, `tasks`, `code`, `commit`), `cale/install.py` + `install.sh/.ps1` + `uninstall.py` (installs, downloads, then converts the GGUF with `llama/prepare_gguf.py`), apps rewritten (Open WebUI shows a badge with the size Cale chose), `llama/build.sh` + `build.ps1` (portable GCC, no admin), built libs in `llama/dist/`, `scripts/hp` + `scripts/hp_forward.sh` + `scripts/demo_drive.py`.

**Prepared GGUF:** each `down` matrix stored as two halves (3-D tensor) so repacking stays on. About +33% prompt speed, a held small size frees 1.14 GB. Gates: full size loss 1.9417 vs stock 1.9384, E2B slice 2.1078 vs official 2.1053, greedy answers identical. Not bit-exact any more.

**Tests (all pass):** Mac, Windows, Linux (WSL2): `e2e.py --model` 37/37, `relative.py`, `objects.py`; `backend_llama.py` on Windows and Linux; `oslayer.py` on all three. Installer end to end on Windows and Linux (real download + convert: about 4 min on Windows). The real installer WAS run on the Mac as an upgrade (2026-10-11, 22:11): new code runs under the LaunchAgent, old app keys still valid. I also replaced Cale's own duplicate 5.5 GB model copy with hard links (freed 5.5 GB). The Mac demo ran with the installed Cale and the REAL Allow dialog (clicked through UI scripting, System Events): Open WebUI card flow and Aider both worked.

**Seen on the HP (i7-8550U, 4 cores):** the same apps ran 63%/69% sizes with goals met on Linux, 48%/49% on Windows (answers goal not met, honestly reported), 100% on the Mac.

**Not done / limits:** On Windows and Linux the Allow dialog was never clicked by a person (SSH has no screen; tests use `CALE_AUTO_APPROVE`; `cale/tests/allow_dialog.py` is there for a person at the desktop). On the Mac it was clicked by script, not by a hand. llama.cpp on Metal WAS compared: MLX is as fast or faster (prompt 873 vs 744 tok/s, answer 35-37 vs 36.8) and uses less memory, so the Mac keeps MLX. No Linux arm64 / Windows arm64 build. Small sizes write poor code in Aider (model quality, not plumbing; the `code` object has a long deadline so it gets a big size). Mac disk: 17 GB free at the end (it had dropped to 12 GB from other use; the dedupe above helped). Committed 2026-10-11 as `a7f3fb7` on branch `phase5-cross-platform` (live key, secrets and webui.db are git-ignored). Follow-up: [[project-cale-vision]] (Phase 6 plan). On the HP the smallest size runs 6.0 tok/s on Windows vs 12.4 on Linux (same CPU): unexplained, Phase 6 stage 2.

**Gotchas that cost time (details in [[reference-hp-testbench]]):** SSH to the HP stalls under full-speed downloads (Wi-Fi), use `scripts/hp`; Windows venv `python.exe` is a launcher (kill the tree); WSL and Windows share localhost ports (stop one side first); `fseek` is 32-bit on Windows; repacked weights are in ordinary memory (drop = empty, restore from file) while weights of a type that cannot be repacked stay mapped (read-only).

See [[project-cale-phases-3-5]], [[project-cale-phases-3-5]].
