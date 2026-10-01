# Findings log (phase 1: nested Gemma 4 12B, no training)

Metric: "drift" = KL divergence of next-token predictions vs. the bf16 12B, on 100
held-out samples (10 sources: web, code, math, 8 languages, chat, 5 image types).
"ans" = only positions inside the model's answer turn. Lower is closer to the 12B.

## 2026-10-01

### Infrastructure
- bf16 12B runs layer by layer over all samples at 4.6 GB peak RAM (`nest/stream.py`). This gives a true full-precision reference.
- Neuron reordering is exact (relative difference ~5e-5, summation order only).
- Live width switching on the loaded model works. Decode speed falls almost in step with width.
- A transposed down_proj layout makes slicing free, but MLX's kernel for it is 3.1-3.3x slower at prefill. Switched to the standard layout. A future file format should store each layer's down_proj as fixed tiers (contiguous blocks).

### Bug found upstream
- mlx-vlm 0.7.3 `gemma4_unified/gemma4_unified.py` `Model.__call__` does not pass `mm_token_type_ids` to the language model. Image tokens then get causal attention instead of the bidirectional attention the 12B was trained with. Our code passes it (`nest/io.py: hidden`, `nest/gen.py`). With the fix, 8-bit image drift fell from ~1.5 to 0.03-0.08.

### Statistics (bf16, 320 calibration samples, 113k tokens)
- MLP activity is concentrated: top 25% of neurons hold 77-100% of score energy per layer.
- Text and image use different neurons in early layers: top-25% overlap is 14% in layers 0-1, ~55% in layers 7-10, ~65% late.
- Layer 11 multiplies the stream norm by ~370 (massive activations). Cosine similarity between layer input and output is then dominated by a few huge dimensions and is misleading.

### Results

| Model / level | drift (all) | drift (ans) | top-1 agree (ans) |
|---|---|---|---|
| 12B 8-bit full | 0.071 | 0.051 | 0.950 |
| 12B 8-bit, 90% MLP width | 0.642 | 0.478 | 0.791 |
| 12B 8-bit, 75% MLP width | 1.469 | 1.217 | 0.650 |
| 12B 8-bit, skip 4 layers | 4.351 | 3.585 | 0.359 |
| 12B 8-bit, skip 8 layers | 6.582 | 5.865 | 0.183 |
| 12B 8-bit, skip 12 layers | 15.04 | 14.36 | 0.019 |
| 12B 4-bit full | 1.279 | 0.704 | 0.746 |
| 12B 4-bit, 90% MLP width | 1.577 | 1.004 | 0.682 |
| E4B 4-bit (separate model) | 6.330 | 3.313 | 0.611 |
| E2B 4-bit (separate model) | 6.287 | 4.015 | 0.595 |

(E4B/E2B: 13 of 100 samples skipped because their image tokenization differs.)

### Conclusions so far
1. Layer skipping without training fails on this model, even for 4 layers. Middle layers change the stream by 20-100% of its norm. Cosine similarity hid this.
2. Width cuts without repair cost a lot: 10% width ≈ 0.48 answer drift. Mean compensation barely helps (layer-local error 0.180 → 0.153 at best), so the damage is lost information, not a bias shift.
3. Per-layer MLP output error grows from ~5% (90% width) to ~12-25% (50% width), and drift grows roughly with the square of it.
4. Plain 4-bit (MLX affine, group 64) costs more than a 10% width cut at 8-bit. Precision is a lever worth improving.
5. Drift to the 12B is not quality. E4B is a different model. Benchmarks decide the E4B bar.

### Benchmarks (letter-choice, no chain-of-thought; MMLU-Pro n=200, MMStar n=100; SE ~3.4 and ~5 points)

| Model / level | approx. weights | MMLU-Pro | MMStar |
|---|---|---|---|
| 12B 8-bit full | 12.7 GB | 49.0 | 54 |
| 12B 8-bit 90% width | ~11.8 GB | 38.5 | 56 |
| 12B 8-bit 75% width | ~10.4 GB | 24.0 | 41 |
| 12B 4-bit (8-bit word table) full | 6.8 GB | 35.5 | 54 |
| 12B 4-bit 90% width | ~6.3 GB | 31.0 | 50 |
| 12B 4-bit 75% width | ~5.6 GB | 18.0 | 29 |
| E4B 4-bit | 4.8 GB | 29.5 | 41 |
| E2B 4-bit | 3.3 GB | 20.0 | 35 |

- 8-bit word table vs plain 4-bit: answer drift 0.704 → 0.683. The 4-bit loss is in the decoder layers.
- Plain MLX 4-bit costs 13.5 MMLU-Pro points vs 8-bit; images unaffected. Next lever: Google's QAT weights.
- 12B 4-bit at 90% width beats E4B on both tests but at ~6.3 GB vs 4.8 GB.

### Calibration-only repair (layer-local, bf16, fit on 160 calib samples, eval on 50 held-out)
Relative MLP output error:

| layer, width | none | rank-64 linear-in-x patch | full linear-in-x | full LS re-solve of kept down_proj |
|---|---|---|---|---|
| 2, 13824 | 0.046 | 0.036 | 0.031 | - |
| 2, 11520 | 0.092 | 0.067 | 0.057 | 0.057 |
| 11, 11520 | 0.069 | 0.053 | 0.050 | 0.052 |
| 24, 11520 | 0.152 | 0.129 | 0.120 | 0.126 |
| 40, 11520 | 0.123 | 0.113 | 0.105 | 0.116 |

- Rank-64 patch (0.5M params/layer, strictly nested) removes 8-27% of the error. Expected drift reduction ~20-45%.
- Full LS re-solve is no better than a linear patch: math-only repair has a low ceiling here.

### QAT weights (Google's gemma-4-12B-it QAT, MLX copy) — 2026-10-01
QAT weights sit on a 16-level grid per block of 32 input weights. MLX's quantizer estimates the step from (max-min)/15, which is wrong when blocks leave levels unused (most do: ~12.8 distinct values per block). `nest/gridq.py` detects the step from value gaps: 0.39% error vs 6% for mx.quantize g32. All 332 weight tables matched the grid.

| Model / level | MMLU-Pro | MMStar | drift (ans) |
|---|---|---|---|
| QAT, MLX 4-bit g64 (+8-bit word table), full | 45.0 | 53 | 0.488 |
| same, 90% width | 46.0 | 53 | 0.764 |
| same, 75% width | 29.0 | 41 | - |
| QAT, exact-grid 4-bit g32, full (7.0 GB) | 47.5 | 55 | 0.367 |

- QAT + exact grid: 4-bit nearly matches the original at 8-bit (49.0 / 54).
- 75% width on QAT 4-bit ties E4B (29.0 vs 29.5, 41 vs 41) at ~5.6 GB vs 4.8 GB.
- Drift vs the original bf16 includes the QAT model's own difference (weights differ 16-51% per matrix), so benchmarks matter more than drift for QAT builds.

### Nested exact-grid QAT build (`models/qg4`, 7.0 GB)
284 tables exact-grid, 48 (reordered down_proj) MLX g32.

| level | MMLU-Pro | MMStar | drift (ans) |
|---|---|---|---|
| full | 48.5 | 57 | 0.396 |
| 90% width | 42.5 | 56 | 0.682 |
| 75% width | 25.0 | 44 | 1.350 |

- Nesting costs nothing at full width (ceiling, original order: 47.5 / 55).
- The cliff sits between 90% and 75% width on knowledge (MMLU-Pro); images hold up better.
- Noise: MMLU-Pro n=200 (±3.4), MMStar n=100 (±5). Final claims need larger sets.

### Width curve, nested exact-grid QAT 4-bit (`qg4`); size = weights the level uses (g32 = 5 bits/weight)

| level | size | MMLU-Pro | MMStar |
|---|---|---|---|
| full | 7.5 GB | 48.5 | 57 |
| u90 | 7.0 | 42.5 | 56 |
| u85 | 6.7 | 37.0 | 56 |
| u80 | 6.4 | 32.0 | 47 |
| u75 | 6.2 | 25.0 | 44 |
| u70 | 5.9 | 24.5 | 33 |
| u65 | 5.7 | 18.0 | 28 |
| u60 | 5.4 | 16.0 | 27 |
| e90 (equal-loss) | 7.0 | 40.0 | 58 |
| e75 (equal-loss) | 6.2 | 28.5 | 43 |
| E4B 4-bit | 5.2 | 29.5 | 41 |
| E2B 4-bit | 3.6 | 20.0 | 35 |

- Equal-loss allocation ≈ uniform (within noise).
- At E4B's footprint (~u55) the slice is far below E4B. The slice ties E4B only at ~6.2-6.4 GB (+20-25%).
- Smooth to ~u85, steep cliff below. Width cuts alone (no training) do not meet the equal-RAM E4B bar.
- QAT grid is Q4_0-symmetric (zero on grid, levels -8..7): llama.cpp Q4_0 stores it exactly at 4.5 bits/weight; MLX needs 5.

### Speed per level (round-robin, ratios vs full; other session also used the GPU)

| level | size | decode vs full | prefill vs full |
|---|---|---|---|
| full | 7.5 GB | 1.00 (≈8 tok/s under contention) | 1.00 |
| u90 | 7.0 | 1.07 | 1.05 |
| u75 | 6.2 | 1.19 | 1.19 |
| u60 | 5.4 | 1.34 | 1.39 |
| u50 | 4.8 | 1.48 | 1.49 |
| E4B (separate run) | 5.2 | ≈2.7 | ≈2.4 |
| E2B (separate run) | 3.6 | ≈4.5 | ≈4.3 |

Width slicing speed is bandwidth-bound: MLP is ~71% of bytes, so u50 caps at ~1.5x.

### Google's QAT MTP drafter (`mlx-community/gemma-4-12B-it-qat-assistant-4bit`, 270 MB)

| level | plain tok/s | with drafter | speedup | accepted share of drafts | identical output |
|---|---|---|---|---|---|
| full | 11.9 | 22.5 | 1.94x | 65.8% | 4/6 |
| u90 | 12.1 | 24.4 | 1.96x | 69.9% | 2/6 |
| u75 | 14.3 | 25.7 | 1.80x | 56.2% | 2/6 |

- Full 12B + drafter ≈ plain E4B speed, at full 12B quality, for +270 MB.
- Not bit-identical (4/6): multi-token verify sums in a different order than single-token decode, which flips near-ties. Same model quality.

### Harness check (first 20 prompts; what models write first)
All models sometimes open with prose instead of a bare letter (12B most often: ~5/20 MMLU-Pro, 7/20 MMStar). Letter scoring at the first answer position was noisy for all models and slightly unfair to the 12B. Fixed: the answer now starts with "The answer is" and the scorer compares tokens " A".." J". Earlier MMLU-Pro/MMStar numbers above used the old harness; re-run queued.

### Corrected harness ("The answer is" prefix, " A".." J" tokens) — supersedes letter-choice numbers above

| model / level | size (MLX) | MMLU-Pro (n=200) | MMStar (n=100) |
|---|---|---|---|
| 12B qg4 full | 7.5 GB | 50.5 | 69 |
| u90 | 7.0 | 43.5 | 58 |
| u80 | 6.4 | 39.5 | 50 |
| u75 | 6.2 | 35.0 | 48 |
| u60 | 5.4 | 22.5 | 27 |
| E4B 4-bit | 5.2 | 33.0 | 39 |
| E2B 4-bit | 3.6 | 24.0 | 32 |

- u75 beats E4B on both tests at +1 GB. Crossover ≈ u70 (~5.9 GB). Gap to the equal-RAM bar ≈ 0.7 GB (~15%).
- QAT reference for clean drift on QAT builds: `results/ref-qat-bf16`.

### Final comparison on larger sets (MMLU-Pro n=1000 ±1.5; MMStar n=300 ±2.9) — supersedes n=200 E4B numbers

| model / level | size (MLX) | MMLU-Pro | MMStar |
|---|---|---|---|
| qg4 (exact grid, 5 bit) full | 7.5 GB | 53.0 | 59.7 |
| qg4 u75 | 6.2 | 31.9 | 54.0 |
| qg4 u70 | 5.9 | 28.8 | 44.7 |
| qg4 u65 | 5.7 | 25.0 | 40.7 |
| q4e8 (MLX g64, 4.5 bit, 8-bit word table) full | 6.8 | 51.3 | 56.7 |
| q4e8 u75 | 5.6 | 31.5 | 50.0 |
| q4e8 u70 | 5.4 | 26.6 | 46.0 |
| q4e8 u65 | 5.1 | 22.0 | 39.7 |
| E4B 4-bit | 5.2 | 38.6 | 47.0 |
| E2B 4-bit | 3.6 | 29.9 | 34.3 |

GSM8K (first 50, greedy, 320 tokens): 12B full 78%, E4B 60%.

- E4B was under-rated at n=200 (33.0 → 38.6). Lesson: n=200 is too small for decisions.
- At E4B's footprint the slice loses clearly on knowledge (22.0 vs 38.6). u75 wins on images (54 vs 47) but loses on knowledge (31.9 vs 38.6).
- Verdict: phase-1 success bar (≥ E4B at E4B's RAM, no training) is NOT met. Width cuts without training fall off too fast below ~u85.
- The 0.5-bit exact-grid premium buys ~1-3 points at equal width; near E4B's size the cheaper g64 encoder is the better trade.
- Exact-verify flag in mlx-vlm applies only to its own verify path, not the MTP drafter path; drafter output stays 4/6 identical (summation order), speedup 1.93x.
- qg4 MMLU-Pro (n=1000): u85 41.0, u80 37.7 → crossover with E4B (38.6) at ≈u81 (≈6.5 GB exact grid, ≈5.9 GB with g64) vs E4B 5.2 GB.
- IFEval (first 30, strict prompt-level): 12B full 76.7%, E4B 66.7%.

## 2026-10-01 — Phase 2a: precision ladder built up from QAT (plan: `~/.claude/plans/okay-given-the-current-eventual-quokka.md`)

### QAT is the original, rescaled (found before Step 1)

| Table (examples, layers 5/20/47) | Original size ÷ QAT size | Cosine | Plain difference ÷ original | After per-row scale |
|---|---|---|---|---|
| gate_proj, q_proj, k_proj | 1.6–2.2 | 0.995 | 37–54% | ~10% |
| o_proj | 2.1–2.6 | 0.995 | 52–62% | 10–12% |
| down_proj | 3.5–3.8 | 0.991 | 71–74% | 12–13% |
| layer 0 (all tables) | 1.06–1.23 | 0.99 | 13–23% | 11–15% |

- QAT moved most of each weight table's scale into the norm weights. 194 of 292 unquantized tensors (norms, `layer_scalar`) differ, by up to 25%.
- So "original = QAT4 + Δ" needs one extra number per output row: `original ≈ row_scale · A + B + C`. Code: `nest/planes.py`.
- Consequence for the nest file: norms need two copies (QAT for the 4-bit tier, original for 8/16). Tiny bytes.
- Neuron order check: `deq(qg4)` vs QAT bf16 in `perm-wanda` order: 0.35% error on grid tables, 7–8% on `down_proj` (the 48 tables where the grid encoder lost to MLX's).

| Rung | Weight error vs original (all 332 tables, smoke run) | Plain RTN for comparison |
|---|---|---|
| 4-bit (A alone) | median 45% (it is QAT, a different fine-tune) | plain 4-bit g64: 9–12% |
| 8-bit (row_scale·A + B) | median 1.0%, max 1.6% | plain 8-bit g64: 0.55% |
| 16-bit (+ C) | median 0.01% | — |

Without the row scale, the 8-bit rung error was 4–7%: almost as bad as plain 4-bit.
