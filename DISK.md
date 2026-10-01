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
| `models/gemma-4-12B-it-bf16` | 22 GB | Source for every build. Full-precision reference and statistics runs. | End of phase 1, after Shonu agrees |
| `models/nest8` | 12 GB | 8-bit reordered 12B. Structure experiments without 4-bit noise. | Replaced by a newer 8-bit build |
| `models/gemma-4-e4b-it-4bit` | 4.8 GB | Baseline: the bar to beat | End of phase 1 |
| `models/gemma-4-e2b-it-4bit` | 3.3 GB | Baseline | End of phase 1 |

## Removed

| Path | Size | Date | Reason |
|---|---|---|---|
| `models/nest8-id` | 12 GB | 2026-10-01 | Original-order copy. Reorder checked exact, so it was no longer needed. |
| `models/gemma-4-12B-it-4bit` | 6.3 GB | 2026-10-01 | Same as our own 4-bit build at full width |
| `data/cache` + HF dataset entries | 0.14 GB | 2026-10-01 | Samples already saved in `data/` |
