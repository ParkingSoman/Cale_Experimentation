# Disk ledger

The disk is shared with another project and is nearly full. Every large file
in this project is listed here with the reason it exists. Remove anything that
is not listed.

## Rules

1. At most one 8-bit and one 4-bit nest build exist at a time. Build, score, then delete or replace.
2. Never keep two copies of the same weights in different formats "just in case".
3. Dataset downloads go to a cache that is deleted after the samples are written to `data/`.
4. Check `df -h ~` before any step that writes more than 5 GB. Stop if free space would drop below 15 GB.
5. Never delete files outside this folder, except HF cache entries this project created.

## Current large files

| Path | Size | Why it exists | Delete when |
|---|---|---|---|
| `~/Library/Application Support/Cale/models/gemma-3n-e4b/` | 5.5 GB | Cale's own copy, downloaded from Hugging Face by the install test on 2026-10-09 (byte-identical to `models/gemma-3n-e4b-4bit`, checked by SHA-256). It is a second copy, not a link. | `./cale/uninstall.sh`. To save 5.5 GB, reinstall with `CALE_MODEL_SRC=models/gemma-3n-e4b-4bit` (hard links). |
| `models/gemma-3n-e4b-4bit` | 5.5 GB | Phase 3 main model: Gemma 3n E4B, 4-bit MLX. E2B and the in-between sizes are nested inside it (MatFormer). | Replaced by a newer build |
| `apps/open-webui/.venv` | 1.8 GB | App 1 of the demo: Open WebUI 0.11.4 in its own venv (not a source clone) | When the demo is retired |
| `apps/aider/.venv` | 0.6 GB | App 2 of the demo: Aider 0.86.2 in its own venv | When the demo is retired |
| `results/g3n-*.json`, `results/bench/g3n-*.json` | <1 MB | Phase 3 measurements | Keep |
| `data/gemma3n-slicing` | 0.04 MB | Google's official size table (source of `nest/gemma3n_levels.json`) | Never needed again after gate A |

## Removed

| Path | Size | Date | Reason |
|---|---|---|---|
| `models/g12-nest` | 7.3 GB | 2026-10-09 | Old 12B one-file build. Replaced by Gemma 3n. Shonu agreed. |
| `models/qg4` | 7.0 GB | 2026-10-09 | Same weights as `g12-nest`. Shonu agreed. |
| `models/gemma-4-e4b-it-4bit`, `models/gemma-4-e2b-it-4bit` | 8.1 GB | 2026-10-09 | Gemma 4 baselines from phase 1. Shonu agreed. |
| `models/qat-assistant-4bit` | 0.27 GB | 2026-10-09 | Drafter for the old 12B. |
| `demo/inbox` | 29 MB | 2026-10-09 | Old demo app, needed the 12B Cale. Replaced by Open WebUI and Aider. Not in git. |
| `models/tmp-gemma-3n-e2b-4bit` | 4.2 GB | 2026-10-09 | Gate A done: our E2B slice matches the official E2B on real text (loss 2.200 vs 2.200) |
| `models/nest8-id` | 12 GB | 2026-10-01 | Original-order copy. Reorder checked exact, so it was no longer needed. |
| `models/gemma-4-12B-it-4bit` | 6.3 GB | 2026-10-01 | Same as our own 4-bit build at full width |
| `data/cache` + HF dataset entries | 0.14 GB | 2026-10-01 | Samples already saved in `data/` |
| `models/nest8` | 12 GB | 2026-10-01 | 8-bit tests done; 4-bit QAT is the main test model now |
| `models/nestq4e8` | 6.8 GB | 2026-10-01 | QAT standard-4-bit test done; exact-grid build is better |
| `models/qg4-id` | 7.0 GB | 2026-10-01 | Ceiling confirmed equal to the nested build at full width |
| `models/nest4e8` | 6.8 GB | 2026-10-01 | Superseded by QAT builds |
| `models/nest4` | 6.3 GB | 2026-10-01 | `nest4e8` scored almost the same (answer drift 0.683 vs 0.704) |
| `models/gemma-4-12B-it-qat-bf16` | 22 GB | 2026-10-01 | `qg4` holds the QAT weights; not needed by the ladder (plan decision) |
| `models/q4e8` | 6.8 GB | 2026-10-01 | Encoder comparison done; exact-grid `qg4` is the 4-bit build |
| `models/gemma-4-12B-it-bf16` | 22 GB | 2026-10-01 | Precision ladder dropped (ocean ≈ QAT 4-bit); Shonu agreed. Re-download if needed. |

## Phase 5 additions (2026-10-11)

| Path | Size | Reason |
|---|---|---|
| `llama/dist/` | 18 MB | Built Windows and Linux libraries. Needed by the installer. Keep. |
| `/private/tmp/claude-501/xplat/llama.cpp` | 0.2 GB | llama.cpp source clone used to make the patch. Delete when the patch is final. |
| HP laptop `C:\cale\`, `~/cale-test` (WSL), `~/cale-apps` (WSL) | about 40 GB | Models (two copies per OS), builds, Open WebUI and Aider environments. Test laptop only. |
