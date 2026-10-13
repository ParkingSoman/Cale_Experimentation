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

### Step 1 gate result (MMLU-Pro n=1000, layer by layer, same questions) — ladder dropped

| Rung | MMLU-Pro |
|---|---|
| 16-bit ocean (original bf16) | 53.8 |
| 8-bit (row_scale·A + B) | 52.8 |
| 4-bit (QAT, dense control `r4`) | 53.1 |
| 4-bit (QAT, `qg4` cached bench) | 53.0 |

| Pair | Points | A right / B right | z |
|---|---|---|---|
| ocean − QAT 4-bit | +0.7 | 51 / 44 | +0.72 |
| 8-bit − QAT 4-bit | −0.3 | 43 / 46 | −0.32 |
| ocean − 8-bit | +1.0 | 17 / 7 | +2.04 |
| harness check: r4 − qg4 bench | +0.1 | 10 / 9 (same answer on 96.9%) | +0.23 |

- Gate outcome: ocean ≈ QAT 4-bit. Google's QAT 12B at 4 bits matches the original 16-bit on MMLU-Pro. Going up from QAT has no quality to climb to.
- The decomposition works (8-bit rung within ~1 point of the ocean), but it buys nothing here.
- Not tested: the ocean on GSM8K/IFEval (generation is too slow layer by layer). MMStar was skipped.
- Decision (Shonu, 2026-10-01): drop the precision ladder as an axis. Focus on width, speed (drafter), one file, instant knobs, API.
- Side findings kept: macOS uncached reads need F_NOCACHE and a page-aligned buffer (5.7 GB/s; unaligned was cached and 1.6 GB/s). Plane runtime math checked against dense rebuild (0.2–0.4% = bf16 output rounding).

## 2026-10-01 — Phase 2b: one file, instant knobs (width + drafter), and where it helps

### The file: `models/g12-nest/model.nest` (7.76 GB)
- One file holds the QAT 4-bit 12B (exact grid, MLP neurons in importance order), its norms, config, tokenizer and Google's QAT MTP drafter (270 MB, loaded from the file's bytes).
- `down_proj` is stored as column blocks [0, 85%), [85%, 90%), [90%, 100%): every width reads contiguous bytes.
- Check (`scripts/check_nest.py`): every table, norm and drafter byte is bit-identical to `models/qg4` / `models/qat-assistant-4bit`.
- Runtime `nest/runtime.py` (`NestedGemma`): `set(width=, draft=)` takes ~1 ms (median 0.3–1.2 ms; worst seen 21 ms) and reads no file bytes. Server: `scripts/serve.py` (OpenAI-style, plus `width` and `draft`). Both tested.
- Letter scores vs `qg4`: identical at 85% width. At 100%, the 3-block `down_proj` sum gives small kernel-order noise (max 0.4 log-prob on 16 prompts). Summing in float32 did not help in the model and cost 11% decode / 20% prefill, so bf16 sums are kept.

### Quality per tier (same items for every tier)

| Tier | GB read | MMLU-Pro (n=1000) | MMStar (n=300) | GSM8K (n=50) | IFEval (n=30) |
|---|---|---|---|---|---|
| 12B w100 | 7.47 | 53.4 | 60.0 | 78 | 76.7 |
| 12B w90 | 6.94 | 44.1 | 58.7 | 84 | 80.0 |
| 12B w85 | 6.67 | 41.0 | 60.0 | 86 | 76.7 |
| 12B w100 + drafter | 7.47 + 0.27 | (= w100) | — | 74 | 80.0 |
| 12B w85 + drafter | 6.67 + 0.27 | (= w85) | — | 86 | 76.7 |
| E4B 4-bit | 5.2 | 38.6 | 47.0 | 60 | 66.7 |

- The file reproduces `qg4` at full width: MMLU-Pro 53.4 vs 53.0, same answer on 971/1000 (9 vs 5 split of the rest).
- Width cuts cost knowledge (MMLU-Pro −12 points at 85%) but not images, math or instruction following on these sets (all differences within noise: GSM8K ±6, IFEval ±8).
- The drafter does not change quality beyond noise.

### Speed (round-robin medians; another session shared the GPU)

| Tier | Prefill tok/s (512 tokens) | Decode, random text (separate runs for E4B and qg4) | Effective tok/s, IFEval-like (same rounds) | Effective tok/s, GSM8K-like (same rounds) |
|---|---|---|---|---|
| 12B w100 | 276 | 10.4 | 10.0 | 10.5 |
| 12B w85 | 302 | 11.7 | 11.4 | 11.7 |
| 12B w100 + drafter | 257 | 20.4 | 12.1 | 21.1 |
| 12B w85 + drafter | 295 | 22.1 | 16.0 | 21.1 |
| E4B | 750 | 17.1 | 27.9 | 21.3 |
| `qg4` (control, same weights) | 261 | 10.7 | — | — |

- "Effective" = generated tokens / whole-call seconds on held-out prompts (`scripts/content_speed.py`). The drafter's gain depends on the text: ~2x on math, only ~1.2–1.4x on instruction-following text.
- E4B: 17.1 tok/s decode in a separate run (`speed.py`), 27.9 effective in the same rounds as the 12B. A plain model's decode speed does not depend on the text, so the gap comes from GPU sharing between runs (phase 1 measured 21.4). Timings from separate runs differed by up to 1.6x; only same-round comparisons are valid. E4B was tested without a drafter.
- The one-file layout costs ~3% decode vs `qg4` (10.4 vs 10.7 tok/s).

### Load and switch cost (`scripts/stack_cost.py`, cold = page cache evicted, medians of 3)

| Setup | Cold load | Warm load | Switch between tiers |
|---|---|---|---|
| one file (all tiers + drafter) | 6.4 s | 5.8 s | ~1 ms, 0 bytes read |
| 12B `qg4` (mlx-vlm) | 6.5 s | 4.5 s | — |
| E4B (mlx-vlm) | 4.5 s | 3.3 s | — |
| 12B + E4B stack, one at a time | — | — | 3.3–4.5 s reload per model change |

The one file reads through uncached I/O, so its "warm" load is no faster than cold.

### Reads from SSD (`scripts/nocache_check.py`)
- Uncached reads stay uncached (mincore 0% resident after reading; 2nd read 1.07x the 1st). 5.9 GB/s with 16 MB chunks; 6.5–7 GB/s with 2–4 threads; 1 MB single-thread 3.3 GB/s.
- Reads overlap GPU work: together 0.33 s vs max 0.26 s / sum 0.42 s (partial overlap).
- `down_proj` as 3 blocks vs one matrix: +24% per call at decode (934 vs 756 µs), none at prefill. In the whole model this is ~3%.

### Low RAM: stream part of the file (`--ram 8.3`, 6.8 GB budget)

| Tier | Answer time, one pass | Decode tok/s | SSD GB per pass |
|---|---|---|---|
| w100, all in RAM | 1.85 s | 10.4 | 0 |
| w100, tails streamed | 1.94 s | 3.8 | 0.67 |
| w85, partly streamed | 1.87 s | 5.9 | 0.48 |

- One-pass calls (classify, pick, score) barely slow down when part of the model streams. Long generation is 2–3x slower.
- Design gap: gate/up tables are kept whole, so their 15% width tails take RAM even at 85%. Splitting them into row blocks would let w85 sit fully in RAM at a 6.8 GB budget (cost: more matmuls at full width).

### Mixed workload, 100 calls (`scripts/mix_sim.py`; routing perfect by construction)

| Mix | One file, routed: time / quality | E4B only | 12B+E4B both in RAM (13 GB) | 12B+E4B one at a time |
|---|---|---|---|---|
| 100% easy | 1253 s / 76.7 | 717 s / 66.7 | 717 s / 66.7 | 717 s / 66.7 |
| 80% easy | 1132 s / 74.1 | 695 s / 63.2 | 703 s / 66.1 | 836 s / 66.1 (34 reloads) |
| 50% easy | 950 s / 70.2 | 661 s / 58.0 | 683 s / 65.2 | 886 s / 65.2 (52 reloads) |
| 20% easy | 769 s / 66.3 | 628 s / 52.8 | 662 s / 64.3 | 787 s / 64.3 (32 reloads) |
| 0% easy | 648 s / 63.7 | 606 s / 49.3 | 648 s / 63.7 | 648 s / 63.7 |

Routed = easy → w85 + drafter, hard long → w100 + drafter, hard one-pass → w100. Quality = mean of IFEval / GSM8K / MMLU-Pro over the calls.
- Simulator check against real 20-call runs through the runtime (routed one file only): predicted 158 s; two runs measured 177 s and 188 s (−10.5% and −16%; bar was ±10%, missed). Warming every tier first did not close the gap; the two runs differ by 11 s on their own (shared GPU). Read the simulator's times as 10–16% optimistic. A first version that used random-text speeds was −29% off.

### Verdict

| Situation | Best setup | By how much |
|---|---|---|
| All hard calls | the 12B alone (nesting not exercised) | +14 points vs E4B at +7% time; a stack's E4B would be 5 GB of dead weight |
| RAM fits ~8 GB, mostly easy calls | E4B is faster | E4B 1.6–1.75x faster, but 10–11 points lower quality |
| RAM fits 13 GB, mixed calls | 12B+E4B both in RAM is fastest | 1.2–1.6x faster than the one file, 2–8 points lower quality |
| RAM fits only ~8 GB, mixed calls | one file: higher quality than the one-at-a-time stack (2–8 points) | the stack is up to 1.35x faster on easy-heavy mixes; equal or slower when ≥80% of calls are hard |
| Less than ~8 GB RAM | one file with streaming for one-pass calls | answer time +5%; generation 2–3x slower |

- Routing gained nothing measurable here. At 50% easy, routed = 950 s / 70.2 and "always w85 + drafter" = 948 s / 70.1. Routing one-pass calls to w100 is a real gain (+12.4 MMLU-Pro points, n=1000, significant), but GSM8K at n=50 happens to favor w85 by 12 points (86 vs 74, noise), and in the blended score the two cancel. The tiers differ too little in speed (15–30%) for instant switching to save time.
- Quality and RAM (corrected wording): E4B has the most score per GB (50% mix: 58.0 / 5.2 GB = 11.2 per GB, one file 70.2 / 7.8 GB = 9.0). The one file gives the highest quality at ~8 GB, and it is the only setup with instant switching below 13 GB. Its quality comes from being the 12B; nesting adds extra modes at 0 extra RAM (85% width reads a 6.67 GB subset of the same 7.47 GB).

### Framing (Shonu, 2026-10-01): this phase is a proof of concept for instant mode switching
- The goal is that a developer switches one loaded model between "different models" (modes) in milliseconds, instead of unloading one model and loading another (seconds).
- Measured: mode switch ~1 ms median (0.3–1.2 ms; worst seen 21 ms), 0 bytes read. Model swap in a stack: 3.3–4.5 s warm, 4.5–6.5 s cold. That is ~3,000–6,000x faster.
- Today's modes are too similar for the switch to save time or add quality in a workload (routing ≈ always w85 + drafter). Training (healing) is the planned step that makes the modes differ enough for switching to matter.
- Where it loses: raw speed on easy, chat-like calls. The width knob adds only ~15–30% speed. A real small model (E4B) is ~2x faster on that content.
- The drafter is the strongest speed lever (up to 2x on math-like text). The width knob is weak for speed and costs knowledge.

Caveats: another session shared the GPU, RAM and SSD (medians reported). Small n for GSM8K (±6) and IFEval (±8). Routing is perfect by construction. E4B had no drafter. Python engine, one reader thread, not tuned. SSD speed differs between Macs.

## 2026-10-09 — Phase 3: Gemma 3n (trained nesting) + two real apps (plan: `~/.claude/plans/okay-i-think-we-ve-functional-quail.md`)

Shonu's change: stop slicing the 12B ourselves. Gemma 3n was trained so that E2B lives inside E4B (MatFormer), and Google publishes 16 official sizes. One 4-bit E4B file (`models/gemma-3n-e4b-4bit`, 5.5 GB) now serves all of them. The demo is two real open-source apps (Open WebUI, Aider) that each already use a small and a large cloud model. Both now call Cale.

### What a size is
Per layer, the FFN runs on its first 8192 channels (half) or all 16384 (full). The two cheapest sizes also skip layers 20–24 (they own no KV cache, so skipping is exact). Switching only changes which blocks run: 0.04 ms, nothing is copied (`nest/gemma3n.py`, size table in `nest/gemma3n_levels.json`).

### Gates
| Gate | Result |
|---|---|
| C: full size = plain mlx-vlm E4B | Pass, exact. 8/8 greedy outputs identical, first-step score difference 0.0. A first version split `down_proj` into two blocks and summed them. That added rounding noise (score diff up to 0.875, outputs drifted). The fix keeps the original matrix for full width. |
| A: our E2B slice = official E2B | Pass on behaviour, not on bits. 646 of 1469 tensors differ from a slice of E4B (even attention weights that were not cut): Google post-trained the two checkpoints separately. On 75 held-out chat/web/code/math samples: reference-text loss 2.200 (ours) vs 2.200 (official), top-1 agreement with E4B 0.807 vs 0.808, KL to E4B 0.478 vs 0.481. |
| B: size curve | See table. `results/g3n-curve.json` (served by Cale in `GET /v1/models`). |

### The letter-score benchmark does not work on Gemma 3n
Old harness (answer starts "The answer is", compare letter scores): official E2B 29.7, E4B 30.5, 95% size 31.3 on MMLU-Pro n=1000. Smaller beating bigger means the test is broken. Cause: after "The answer is" Gemma 3n puts probability 1.0 on `**`, so the letters sit in the tail. Two other prompts (ask for "only the letter", or start with "The answer is **") gave E4B 28.5/31.5, 75% size 28.0/29.5, E2B 22.0/29.0 at n=200. E4B at ~30 versus Google's 50.6 is a gap of 20 points that 4-bit rounding cannot explain. Reading one letter token straight after the question is not valid for this model, so none of these MMLU-Pro numbers mean anything. Used instead: GSM8K (model reasons, then answers) and closeness to E4B on held-out text. A scorer that reads the first token without a KV cache is also wrong here (layers 20+ reuse earlier K/V).

One more oddity, not explained: on the held-out reference texts the 30-layer sizes (e2b, e1.96) score a lower loss (2.20) than the 35-layer sizes (2.35–2.50), even E4B. Skipping layers cannot make a model better at text. The references are machine-written, so loss on them is not a quality measure here. Only KL to E4B is used as a closeness measure.

### Size curve (one loaded model, 4.2 GB in memory; same RAM at every size)
| Size | % of E4B | KL to E4B | top-1 same as E4B | tok/s (clean run) | speed-up | GSM8K (n=100) |
|---|---|---|---|---|---|---|
| e4b | 100 | 0 | 1.00 | 37.2 | 1.00x | 77 |
| e3.79-layer | 95 | 0.009 | 0.98 | 38.5 | 1.03x | – |
| e3.59-layer | 90 | 0.047 | 0.95 | 40.0 | 1.08x | – |
| e3.24-block | 81 | 0.076 | 0.93 | 42.9 | 1.15x | 80 |
| e3.18-layer | 80 | 0.196 | 0.88 | 43.3 | 1.16x | – |
| e2.98-block | 75 | 0.205 | 0.88 | 44.4 | 1.19x | – |
| e2.73-block | 69 | 0.252 | 0.86 | 45.4 | 1.22x | 74 |
| e2.49-block | 63 | 0.392 | 0.82 | 48.0 | 1.29x | – |
| e2b | 48 | 0.478 | 0.81 | 56.9 | 1.53x | 73 |

All 16 sizes are in `results/g3n-curve.json`. Dominated (another size is no bigger, no farther from E4B, no slower): e2.98-layer, e2.54-layer, e1.96-layer.
- The small size is only 1.5x faster. Most of the per-token cost does not shrink with the FFN: attention, the 262k-word output layer and the per-layer embeddings. Do not promise more.
- Quality cost on GSM8K is small (77 → 73, noise ±4). On held-out text, the first ~19% of compute (down to 81%) costs little (KL ≤ 0.08), then there is a step (KL 0.20 at 80%), then a steady slope.
- RAM does not drop with size: every size reads the same file. GSM8K tok/s in the table of the run were lower (27–45) because the GPU was shared at the time. Only the clean speed run is in the table.

### What was built
| Piece | Where |
|---|---|
| Runtime | `nest/gemma3n.py`, `nest/levels3n.py` |
| Cale (server, engine, installer, model download with progress) | `cale/` (`install.sh`, `fetch_model.py`) |
| App 1: Open WebUI 0.11.4, unmodified, plus a "Get Cale" card | `apps/open-webui/cale_webui.py`, `cale.js`, port 8810 |
| App 2: Aider 0.86.2, unmodified, plus a launcher that offers Cale | `apps/aider/cale_aider.py` |
| Demo steps | `demo/README.md` |

### Observed, not assumed (Cale log, one `[chat]` line per answer)
- Open WebUI: answers `asked=gemma-3n:smart ran=gemma-3n`; chat title (9 tokens), follow-up ideas (81) and tags (24) `asked=gemma-3n:fast ran=gemma-3n:48`.
- Aider: the code edit `asked=gemma-3n:smart`; the commit message `asked=gemma-3n:fast ran=gemma-3n:48`.
- A second app with its own key adds no memory (resident stays 4.2 GB) and downloads nothing. Tests: `cale/tests/e2e.py` 37 passed, `cale/tests/pressure.py` all passed.

### Bugs found on the way
- Open WebUI sends no length limit, and Cale's default of 256 tokens cut answers short. Default is now 1024.
- A request under a memory cap could run a size above the cap ("nearest to 60%" picked 63%). Now: the largest size at or under the cap.
- Cap changes lower compute only. They do not free memory (all sizes read the same weights). The old 12B runtime freed about 2 GB.

### Not done / not verified
- Mac Allow dialog: see below. The real Hugging Face download did run (2026-10-09, after Shonu approved deleting the old models). See the next section.
- Autocomplete (as-you-type) is switched on in Open WebUI but no request for it was seen in Cale's log. Title, follow-ups and tags were seen.
- Aider's `[Y/n]` question was tested with the answer n. The answer Y calls the same key code that ran in the auto-approve tests.
- The Mac "Allow" dialog was not clicked in any test (tests used the auto-approve test mode). The final install is in normal mode.
- Only 75 held-out text samples were available for KL. GSM8K covers 4 of 16 sizes at n=100.
- No Aider test beyond a one-function edit in a tiny repo. Aider is told to use whole-file edits because a small model handles diffs badly.
- Gemma 3n has no speculative-decoding drafter here, so the 2x drafter gain from phase 2 does not exist in this build.

### 2026-10-09 (later): the real Hugging Face flow, run in a browser
Shonu approved deleting the old models and the Inbox demo (22 GB of models, 29 MB demo), so there was room to test the real download. Deleted: `models/g12-nest`, `qg4`, both Gemma 4 baselines, `qat-assistant-4bit`, `demo/inbox`. Free disk went from 21 GB to 44 GB.

Flow: uninstall Cale, start `cale_webui.py` with no local model folder, open the page, click "Get Cale". Result: the card showed "Downloading the model: 1.9 of 5.9 GB" with a matching bar. The download took about one minute. The state then moved to "Connect". The files are byte-identical to the repo copy (SHA-256 of both safetensors files), and the model loads and answers from the installed runtime (E4B and E2B sizes).

Two bugs found by this test, both fixed:
| Bug | Cause | Fix |
|---|---|---|
| The download failed at once with "offline mode is enabled" | Open WebUI's offline mode sets `HF_HUB_OFFLINE`. The installer, started from Open WebUI, inherited it. | `apps/open-webui/cale_webui.py` removes the offline variables from the installer's environment. |
| The progress passed 100% (8.9 of 5.9 GB) | `snapshot_download` makes two byte counters (network bytes and bytes written to disk). My counter added both. | `cale/fetch_model.py` counts only the disk counter and caps at the total. The page caps the bar. |

Still not done: nobody clicked the Mac Allow dialog (tests used auto-approve mode). Cale is now installed in normal mode with no app keys, so the first Connect click shows the real dialog.

## Phase 4 (2026-10-10): holds and relative goals

Shonu's idea: sizes should be relative to the Mac, not fixed. An app states a goal (finish in N seconds, use at most X% of RAM) and Cale picks the size. An app can also load a size into RAM and keep it there until it unloads it or quits (the "C-like" low level). Relative goals are the "high level". Plan: `~/.claude/plans/okay-i-want-to-cuddly-eagle.md`.

### What was built
| Part | File |
|---|---|
| FFN states per layer (0 none, 1 half, 2 full), `reside`, `ensure` (read missing weights from disk for one call), `settle` | `nest/gemma3n.py` |
| Per-layer embedding tables read from the model file, row by row | `nest/gemma3n.py` `MappedEmbedding` |
| Per-Mac profile (speed per size, bytes per size, disk speed) | `cale/profile.py` |
| Size choice for a deadline or a memory goal | `cale/choose.py` (+ `cale/tests/choose.py`) |
| Holds, implicit hold, pressure rules | `cale/engine.py` |
| `POST /cale/load`, `POST /cale/unload`, `gemma-3n:auto`, `cale_*` goal fields, headers | `cale/server.py`, `cale/client.py` |
| Aider and Open WebUI use a memory goal | `apps/aider/cale_aider.py`, `apps/open-webui/cale_webui.py` |

### Measured (M5, 24 GB, `results/g3n-residency.json`, `results/g3n-ple.json`)
| Check | Result |
|---|---|
| Full size, new storage vs plain mlx-vlm (gate C) | 8 of 8 outputs identical (still bit-exact) |
| Memory with every size resident, per-layer tables in RAM | 4.20 GB |
| Memory with the per-layer tables on disk | 2.88 GB (full), 2.12 GB (75%), 1.41 GB (48%) |
| Per-layer tables on disk: speed | e4b 34.9 to 33.8 tok/s, e2b 57.2 to 56.2 tok/s (under 4% slower), text identical |
| Held smallest, ask for full size | reads about 1.98 GB from disk, 0.18 s (file in the Mac's cache), memory back to the hold level after |
| Predicted RAM vs measured | exact for 75% (2.12 vs 2.12 GB) and 48% (1.41 vs 1.41 GB) |
| Killed app | hold freed in 0.5 s |
| Critical pressure with a hold | hold and memory stay |
| Old tests (`e2e.py` 22 checks, `pressure.py`) | all pass |
| New test `cale/tests/relative.py` | all pass |
| Aider (real run, own code unchanged) | memory goal 40% held the full size, goal 8% held the 49% size; edit applied and committed both times |
| Open WebUI (real run in a browser) | answers, title, follow-ups work; hold of the 49% size at an 8% goal; the answer (asked as `smart`) read 1.9 GB from disk |

### Bugs found on the way
| Bug | Fix |
|---|---|
| MLX arrays still lazy at load belong to the thread that first evaluates them. A hold made in one thread broke answers in another ("no Stream(gpu, 1)"). | Evaluate all weights at load and run one warm-up answer in the loading thread. Evaluate every array before storing it. |
| `np.take` on a memory map read the whole 1.2 GB table for one row (22 ms per token, speed halved). | Plain row indexing: 0.35 ms. |
| In the full state the half-width gate/up are free views, but they were counted as real bytes. All smaller-size memory predictions were 0.4-0.7 GB too low. | Count real bytes only. Profile version 2. |
| `http.client` hands its socket to the response, so `close()` did not end a hold. | `cale/client.py` uses a plain socket. |

### Caveats
- The disk speed in the profile was measured with the file in the Mac's cache (11 GB/s). Real answers correct it, but the first deadline with `cale_allow_disk` may be too optimistic.
- The installed Cale (port 47821) still runs the old code. Run `./cale/install.sh` to update it.
- Loading the model still needs about 4.2 GB for a moment, before the per-layer tables are moved to disk. A hold with a very small goal can pass its goal during the load.
- The Mac Allow dialog was again not clicked (tests used auto-approve).

## Phase 5 (2026-10-11): Cale on Mac, Windows and Linux (plan: `~/.claude/plans/pasted-content-id-d6c3-make-compressed-dream.md`)

Goal: one runtime on Mac, Windows and Linux that adapts to each computer. Both demo apps run on all three and say what they need (goals), not a fixed size.

### Stage 0: which engine on Windows and Linux (`results/xplat/bakeoff.json`)

Test computer: HP laptop, Intel i7-8550U (4 cores, 8 threads, AVX2), 15.9 GB RAM (7.8 GB inside WSL2), Wi-Fi only.

| Engine | Result |
|---|---|
| MLX on the CPU | Unusable. One 2048x16384 matrix multiply for a 256-token prompt took 9.7 s (about 1.8 GFLOP/s, one core). 4-bit was no faster than 32-bit. The model warm-up did not finish in 7 minutes. |
| llama.cpp on the CPU, 4 threads | Windows E4B: 18.6 prompt / 7.7 answer tok/s. E2B: 38.9 / 13.7. Linux (WSL2) E4B: 16.9 / 7.1. E2B: 33.5 / 12.3. |
| llama.cpp on Vulkan | The 2 GB NVIDIA MX150 ran out of memory. The Intel UHD 620 crashed on load. No numbers. |

Decision: Windows and Linux use llama.cpp on the CPU. The Mac keeps MLX, and that was measured (same Mac M5, stock llama.cpp b11558 on Metal against our MLX engine):

| Mac M5, E4B, full size | Prompt | Answer | Model memory |
|---|---|---|---|
| llama.cpp on Metal | 744 tok/s | 36.8 tok/s | 4.09 GB |
| Cale on MLX | 873 to 880 tok/s | 35.0 to 37.1 tok/s | 2.88 GB |
| Cale on MLX, smallest size | 1577 tok/s | 60 tok/s | 2.88 GB |

MLX is as fast or faster and also has the instant small size.
The size knob matters on this laptop: E2B answers 1.7 times faster than E4B.

### The patched llama.cpp (`llama/`)

A 50-line patch gives llama.cpp the matryoshka sizes: per layer, the feed-forward block uses its first N channels (a view, no copy) or the layer is skipped (layers 20 to 24 only).

| Gate | Result |
|---|---|
| C: full size equals stock llama.cpp | Identical. 3 of 3 greedy answers, loss difference 0.0 on every sample. |
| A: E4B sliced to the E2B size vs Google's E2B file | Loss 2.1049 vs 2.1053. Greedy answers identical 3 of 3. Same result as MLX in phase 3. |

Repacking (the second step, `llama/prepare_gguf.py`): llama.cpp can repack Q4 weights for faster CPU math. A view of a repacked weight is only valid for whole row groups, and a repacked weight cannot be mapped from the file. The `gate` and `up` matrices are fine (a size is a block of whole rows). The `down` matrix is cut by columns, so its first half is spread over every row. The fix: store each `down` matrix as two halves (a 3-D tensor) in the file. A one-time conversion (67 s on the test laptop, bytes only reordered, 812 other tensors unchanged) does it. The installer runs it after the download.

| Same laptop, Linux in WSL2, 541-token prompt | Prompt | Answer | Memory |
|---|---|---|---|
| stock file, no repacking: full / smallest size | 11.3 / 22.3 tok/s | 3.2 / 8.4 tok/s | 3.3 GB |
| prepared file, repacking: full / smallest size | 15.2 / 29.5 tok/s | 4.0 / 12.4 tok/s | 3.4 GB |
| Windows, prepared file: full / smallest size | 15.9 / 30.8 tok/s | 3.4 / 6.0 tok/s | 3.1 GB |

About a third faster on prompts. A held small size now frees 1.14 GB of the weights (0.85 GB with the stock file), because the second half of `down` can go too. Answer speeds are noisy (difference of two single runs).

Quality with the prepared file: full size loss 1.9417 against 1.9384 for stock llama.cpp, identical greedy answers (no longer bit-exact: repacking uses other kernels). E4B sliced to the E2B size: 2.1078 against 2.1053 for Google's E2B file, identical greedy answers.

Two bugs found on the way: `cale_restore` wrote into a read-only mapped weight (the `down` weights are a 4-bit type that cannot be repacked, so they stay mapped: skip those), and `fseek` takes a 32-bit offset on Windows (the file is over 2 GB: use `_fseeki64`). After a repacked load the original file pages stay resident (5.2 GB): give them back at once (`cale_drop_file_mapping`, 3.4 GB after).

### What was built
| Piece | Where |
|---|---|
| Runtime | `nest/gemma3n.py`, `nest/levels3n.py` |
| Cale (server, engine, installer, model download with progress) | `cale/` (`install.sh`, `fetch_model.py`) |
| App 1: Open WebUI 0.11.4, unmodified, plus a "Get Cale" card | `apps/open-webui/cale_webui.py`, `cale.js`, port 8810 |
| App 2: Aider 0.86.2, unmodified, plus a launcher that offers Cale | `apps/aider/cale_aider.py` |
| Demo steps | `demo/README.md` |

### Observed, not assumed (Cale log, one `[chat]` line per answer)
- Open WebUI: answers `asked=gemma-3n:smart ran=gemma-3n`; chat title (9 tokens), follow-up ideas (81) and tags (24) `asked=gemma-3n:fast ran=gemma-3n:48`.
- Aider: the code edit `asked=gemma-3n:smart`; the commit message `asked=gemma-3n:fast ran=gemma-3n:48`.
- A second app with its own key adds no memory (resident stays 4.2 GB) and downloads nothing. Tests: `cale/tests/e2e.py` 37 passed, `cale/tests/pressure.py` all passed.

### Bugs found on the way
- Open WebUI sends no length limit, and Cale's default of 256 tokens cut answers short. Default is now 1024.
- A request under a memory cap could run a size above the cap ("nearest to 60%" picked 63%). Now: the largest size at or under the cap.
- Cap changes lower compute only. They do not free memory (all sizes read the same weights). The old 12B runtime freed about 2 GB.

### Not done / not verified
- Mac Allow dialog: see below. The real Hugging Face download did run (2026-10-09, after Shonu approved deleting the old models). See the next section.
- Autocomplete (as-you-type) is switched on in Open WebUI but no request for it was seen in Cale's log. Title, follow-ups and tags were seen.
- Aider's `[Y/n]` question was tested with the answer n. The answer Y calls the same key code that ran in the auto-approve tests.
- The Mac "Allow" dialog was not clicked in any test (tests used the auto-approve test mode). The final install is in normal mode.
- Only 75 held-out text samples were available for KL. GSM8K covers 4 of 16 sizes at n=100.
- No Aider test beyond a one-function edit in a tiny repo. Aider is told to use whole-file edits because a small model handles diffs badly.
- Gemma 3n has no speculative-decoding drafter here, so the 2x drafter gain from phase 2 does not exist in this build.

### 2026-10-09 (later): the real Hugging Face flow, run in a browser
Shonu approved deleting the old models and the Inbox demo (22 GB of models, 29 MB demo), so there was room to test the real download. Deleted: `models/g12-nest`, `qg4`, both Gemma 4 baselines, `qat-assistant-4bit`, `demo/inbox`. Free disk went from 21 GB to 44 GB.

Flow: uninstall Cale, start `cale_webui.py` with no local model folder, open the page, click "Get Cale". Result: the card showed "Downloading the model: 1.9 of 5.9 GB" with a matching bar. The download took about one minute. The state then moved to "Connect". The files are byte-identical to the repo copy (SHA-256 of both safetensors files), and the model loads and answers from the installed runtime (E4B and E2B sizes).

Two bugs found by this test, both fixed:
| Bug | Cause | Fix |
|---|---|---|
| The download failed at once with "offline mode is enabled" | Open WebUI's offline mode sets `HF_HUB_OFFLINE`. The installer, started from Open WebUI, inherited it. | `apps/open-webui/cale_webui.py` removes the offline variables from the installer's environment. |
| The progress passed 100% (8.9 of 5.9 GB) | `snapshot_download` makes two byte counters (network bytes and bytes written to disk). My counter added both. | `cale/fetch_model.py` counts only the disk counter and caps at the total. The page caps the bar. |

Still not done: nobody clicked the Mac Allow dialog (tests used auto-approve mode). Cale is now installed in normal mode with no app keys, so the first Connect click shows the real dialog.

## Phase 4 (2026-10-10): holds and relative goals

Shonu's idea: sizes should be relative to the Mac, not fixed. An app states a goal (finish in N seconds, use at most X% of RAM) and Cale picks the size. An app can also load a size into RAM and keep it there until it unloads it or quits (the "C-like" low level). Relative goals are the "high level". Plan: `~/.claude/plans/okay-i-want-to-cuddly-eagle.md`.

### What was built
| Part | File |
|---|---|
| FFN states per layer (0 none, 1 half, 2 full), `reside`, `ensure` (read missing weights from disk for one call), `settle` | `nest/gemma3n.py` |
| Per-layer embedding tables read from the model file, row by row | `nest/gemma3n.py` `MappedEmbedding` |
| Per-Mac profile (speed per size, bytes per size, disk speed) | `cale/profile.py` |
| Size choice for a deadline or a memory goal | `cale/choose.py` (+ `cale/tests/choose.py`) |
| Holds, implicit hold, pressure rules | `cale/engine.py` |
| `POST /cale/load`, `POST /cale/unload`, `gemma-3n:auto`, `cale_*` goal fields, headers | `cale/server.py`, `cale/client.py` |
| Aider and Open WebUI use a memory goal | `apps/aider/cale_aider.py`, `apps/open-webui/cale_webui.py` |

### Measured (M5, 24 GB, `results/g3n-residency.json`, `results/g3n-ple.json`)
| Check | Result |
|---|---|
| Full size, new storage vs plain mlx-vlm (gate C) | 8 of 8 outputs identical (still bit-exact) |
| Memory with every size resident, per-layer tables in RAM | 4.20 GB |
| Memory with the per-layer tables on disk | 2.88 GB (full), 2.12 GB (75%), 1.41 GB (48%) |
| Per-layer tables on disk: speed | e4b 34.9 to 33.8 tok/s, e2b 57.2 to 56.2 tok/s (under 4% slower), text identical |
| Held smallest, ask for full size | reads about 1.98 GB from disk, 0.18 s (file in the Mac's cache), memory back to the hold level after |
| Predicted RAM vs measured | exact for 75% (2.12 vs 2.12 GB) and 48% (1.41 vs 1.41 GB) |
| Killed app | hold freed in 0.5 s |
| Critical pressure with a hold | hold and memory stay |
| Old tests (`e2e.py` 22 checks, `pressure.py`) | all pass |
| New test `cale/tests/relative.py` | all pass |
| Aider (real run, own code unchanged) | memory goal 40% held the full size, goal 8% held the 49% size; edit applied and committed both times |
| Open WebUI (real run in a browser) | answers, title, follow-ups work; hold of the 49% size at an 8% goal; the answer (asked as `smart`) read 1.9 GB from disk |

### Bugs found on the way
| Bug | Fix |
|---|---|
| MLX arrays still lazy at load belong to the thread that first evaluates them. A hold made in one thread broke answers in another ("no Stream(gpu, 1)"). | Evaluate all weights at load and run one warm-up answer in the loading thread. Evaluate every array before storing it. |
| `np.take` on a memory map read the whole 1.2 GB table for one row (22 ms per token, speed halved). | Plain row indexing: 0.35 ms. |
| In the full state the half-width gate/up are free views, but they were counted as real bytes. All smaller-size memory predictions were 0.4-0.7 GB too low. | Count real bytes only. Profile version 2. |
| `http.client` hands its socket to the response, so `close()` did not end a hold. | `cale/client.py` uses a plain socket. |

### Caveats
- The disk speed in the profile was measured with the file in the Mac's cache (11 GB/s). Real answers correct it, but the first deadline with `cale_allow_disk` may be too optimistic.
- The installed Cale (port 47821) still runs the old code. Run `./cale/install.sh` to update it.
- Loading the model still needs about 4.2 GB for a moment, before the per-layer tables are moved to disk. A hold with a very small goal can pass its goal during the load.
- The Mac Allow dialog was again not clicked (tests used auto-approve).

## Phase 5 (2026-10-11): Cale on Mac, Windows and Linux (plan: `~/.claude/plans/pasted-content-id-d6c3-make-compressed-dream.md`)

Goal: one runtime on Mac, Windows and Linux that adapts to each computer. Both demo apps run on all three and say what they need (goals), not a fixed size.

### Stage 0: which engine on Windows and Linux (`results/xplat/bakeoff.json`)

Test computer: HP laptop, Intel i7-8550U (4 cores, 8 threads, AVX2), 15.9 GB RAM (7.8 GB inside WSL2), Wi-Fi only.

| Engine | Result |
|---|---|
| MLX on the CPU | Unusable. One 2048x16384 matrix multiply for a 256-token prompt took 9.7 s (about 1.8 GFLOP/s, one core). 4-bit was no faster than 32-bit. The model warm-up did not finish in 7 minutes. |
| llama.cpp on the CPU, 4 threads | Windows E4B: 18.6 prompt / 7.7 answer tok/s. E2B: 38.9 / 13.7. Linux (WSL2) E4B: 16.9 / 7.1. E2B: 33.5 / 12.3. |
| llama.cpp on Vulkan | The 2 GB NVIDIA MX150 ran out of memory. The Intel UHD 620 crashed on load. No numbers. |

Decision: Windows and Linux use llama.cpp on the CPU. The Mac keeps MLX, and that was measured (same Mac M5, stock llama.cpp b11558 on Metal against our MLX engine):

| Mac M5, E4B, full size | Prompt | Answer | Model memory |
|---|---|---|---|
| llama.cpp on Metal | 744 tok/s | 36.8 tok/s | 4.09 GB |
| Cale on MLX | 873 to 880 tok/s | 35.0 to 37.1 tok/s | 2.88 GB |
| Cale on MLX, smallest size | 1577 tok/s | 60 tok/s | 2.88 GB |

MLX is as fast or faster and also has the instant small size.
The size knob matters on this laptop: E2B answers 1.7 times faster than E4B.

### The patched llama.cpp (`llama/`)

A 50-line patch gives llama.cpp the matryoshka sizes: per layer, the feed-forward block uses its first N channels (a view, no copy) or the layer is skipped (layers 20 to 24 only).

| Gate | Result |
|---|---|
| C: full size equals stock llama.cpp | Identical. 3 of 3 greedy answers, loss difference 0.0 on every sample. |
| A: E4B sliced to the E2B size vs Google's E2B file | Loss 2.1049 vs 2.1053. Greedy answers identical 3 of 3. Same result as MLX in phase 3. |

Cost: repacking is off (a view of a repacked weight is wrong, and repacked weights cannot be mapped). That costs about 33% on prompts and 24% on answers here (stock: 16.3 / 7.0 tok/s, no repack: 11.0 / 5.3). The fix is a file layout that splits the `down` matrix in two. Not done yet. See `llama/README.md`.

### What was built

| Part | Detail |
|---|---|
| `cale/oslayer.py` | All OS-specific code in one file: home folder, RAM, memory pressure (macOS sysctl, Linux pressure-stall file, Windows low-memory event), file lock, Allow dialog (osascript, MessageBox, zenity), detached start. |
| `cale/service.py` | Start and stop Cale with no service manager. Mac keeps its LaunchAgent. Others: a detached process that ends itself after 10 idle minutes. |
| `cale/backends/` | `llama.py` (same interface as the MLX model), `llamalib.py` (ctypes), `__init__.py` picks the engine. The engine and profile no longer import MLX. |
| Profile | Times three sizes and fits a line for the rest. Minutes saved on a slow CPU. |
| Model objects | `POST /cale/models`, `cale.client.model(...)`. An app states a size or goals (deadline, answer length, memory share, allow disk, on miss). The object id is the model name. One object per app and name, so ids are stable (`answers`, `tasks`). `GET /v1/models` shows the size each object ran at last. |
| Installer | `cale/install.py` (standard library only) with `install.sh` and `install.ps1` as thin starters. Gets `uv` and Python, installs the engine for this OS, downloads or links the model (and on Windows and Linux converts it, see above), installs the library, makes the `cale` command. |
| Apps | Open WebUI and Aider use model objects, run on all three OSes, and Open WebUI shows a badge with the size Cale chose. |

### Test results

| Suite | Mac (MLX) | Windows (llama.cpp) | Linux, WSL2 (llama.cpp) |
|---|---|---|---|
| `oslayer.py` | pass | pass | pass |
| `backend_llama.py` | not applicable | all pass | all pass |
| `e2e.py --model` (37 checks) | 37 of 37 | 37 of 37 | 37 of 37 |
| `relative.py` (holds and goals) | all pass | all pass | all pass |
| `objects.py` (model objects) | all pass | all pass | all pass (one check failed once in an installed run and passed on rerun) |
| install, status, on-demand restart, uninstall | upgrade of the installed Cale: pass (the old app keys still worked) | pass | pass |
| Allow dialog | Allow gives a key, Don't Allow gives 403, the replace-key warning shows (clicked by script through the UI) | called over SSH: times out and says no. A person must click it: `cale/tests/allow_dialog.py` | no screen over SSH: refuses at once. A person must click it. |

Adaptation seen: the same model object picked the full size at normal speed and the smallest size on a 3x slower computer, on all three systems.

### The demo, run for real on the laptop (Open WebUI card flow, then Aider)

Open WebUI started with no Cale installed. The "Get Cale" install, connect, chat and a title task were driven by `scripts/demo_drive.py` (the same routes the card calls). The same app, with the same goals (`answers`: 600 tokens in 90 s; `tasks`: 60 tokens in 30 s), got different sizes on each computer.

| Computer | Install (Get Cale to connected) | `answers` ran | `tasks` ran |
|---|---|---|---|
| Mac M5 (MLX) | already installed | 100% size, goal met, 1.6 s | 100% size, goal met |
| Linux in WSL2 (4 CPU threads) | 1 min (model file linked, then converted) | 63% size, goal met (predicted 90 s) | 69% size, goal met |
| Windows (4 CPU threads) | 4 min (real 4.4 GB download, then converted) | 48% size, goal NOT met (predicted 127 s), still answered | 49% size, goal met |

Aider (objects `code` and `commit`) found the running Cale on Linux and Windows with no download and no extra memory, wrote a correct `hello.py` and (on Linux, which has git) made two commits with sensible messages. At small sizes Aider's edit format sometimes fails (the model writes `path/to/hello.py`). That is a quality limit of small sizes. For code the app says "time matters less" (a 600 s deadline), and then it runs a big size.

### Gotchas that cost time

| Gotcha | Detail |
|---|---|
| SSH link to the laptop | Both computers are on Wi-Fi. A full-speed download on the laptop stalls SSH. The laptop must download big files itself, and jobs must run detached (`scripts/hp`). |
| Windows children of SSH | A `Start-Process` child dies when the SSH session ends. Start jobs with WMI (`Win32_Process.Create`). |
| Windows venv `python.exe` | It is a launcher that starts the real Python as a child. Kill the whole tree (`taskkill /T /F`), or an orphan server keeps the port. |
| Windows port reuse | With `SO_REUSEADDR`, two servers can share one port. The server turns it off on Windows. |
| Windows `os.replace` | A virus scanner can hold a new file for a moment. Retry. |
| MLX on Windows | Needs the separate `mlx-cpu` package (it holds `mlx.dll`). |
| `/mnt/c` in WSL | Copying through it ran at about 2 MB/s. Download inside WSL instead. |
| Linux memory count | `mincore` reports the shared page cache. Use `/proc/self/pagemap` to count what this process has mapped. |
| Allow dialog | Cannot show for a process started over SSH on Windows. Tests use `CALE_AUTO_APPROVE`. |

## Phase 6 (2026-10-12): make Cale general (plan: `~/.claude/plans/you-were-finishing-up-splendid-willow.md`)

Goal: success = (app types that work well) x (environments each one runs in). Check with a **hidden app zoo**: 12 probe apps
written by a reviewer agent that read only `docs/`. Scorecards: `results/zoo/BASELINE.md` (before) and the table below (after stage 2a).

### The app zoo, before and after stage 2a

| App | Mac before | Mac after | HP Linux before | HP Linux after |
|---|---|---|---|---|
| agent-tools | FAILS (tools ignored in silence) | FAILS (honest 400) | FAILS | FAILS (honest 400) |
| json-extract | FAILS | FAILS (honest 400) | FAILS | FAILS (honest 400) |
| long-doc-qa | WELL | WELL | FAILS | FAILS (a 5k prompt takes about 6 min on this CPU) |
| summarizer | WELL | WELL | FAILS | FAILS (14 min, prompt reading) |
| two-apps-at-once | POOR (said "met" when it was not) | WELL (both goals met) | POOR | POOR (now honest: `goal_met=false`, but a 4 s goal waited 32 s) |
| node-app, curl-script, streaming-backend | WELL | WELL | WELL | WELL |
| low-memory-app | POOR | WELL | POOR | WELL |
| stop-and-sampling | FAILS | WELL | FAILS | WELL |
| battery-background-job | FAILS | FAILS | FAILS | FAILS (no goal for it yet) |
| image-input | FAILS | FAILS | FAILS | FAILS (honest 400) |
| **Works well** | **5 of 12** | **8 of 12** | **3 of 12** | **5 of 12** |

### What changed (all tests pass on Mac, HP Linux, HP Windows)

| Gap the zoo found | Fix |
|---|---|
| Unsupported fields got 200 and were ignored | 400 by name (`unsupported_parameter`, `unknown_parameter`) |
| `stop`, `seed`, `top_p` ignored | Implemented on both engines. Unseeded sampling now differs per call (MLX random state is per thread). |
| Context limit differed per computer, a 45k prompt broke the Mac | MLX reads the prompt in pieces (33k tokens: 23 s, 6 GB peak). One public limit (32768), shown in `/v1/models`. Clean `context_length_exceeded`. |
| Engine "wedged" after one bad call | Cause: MLX arrays belong to the thread that made them. A request that stopped early left half-computed arrays, and every later request failed with "no Stream(gpu) in current thread". `materialize()` finishes them. Engine failure now returns 500 `engine_error` and reloads. |
| `Goal-Met: true` while the app waited in line | Wait time counts (`waited_s`, `total_s`, `goal_met_actual`) |
| A memory goal said 48% but the call ran at 100% | The object's size caps its calls |
| An abandoned request kept the engine busy for minutes | A closed connection stops the work, even in the middle of a long prompt (`nest/cancel.py`, shim abort callback) |
| Windows 40-50% slower than Linux on the same CPU | The mingw OpenMP runtime. Build without it: decode 4.1 -> 7.2 (full), 6.2 -> 12.3 (smallest) tok/s |
| Error details visible on `/cale/status` (security review) | Details only in the private log |

### Windows speed hunt (HP, i7-8550U, Gemma 3n prepared file, 229-token prompt)

| Tried | Full size decode | Smallest decode |
|---|---|---|
| baseline (OpenMP) | 4.1 | 6.2 |
| efficiency mode off, above-normal priority | same | same |
| 3 or 8 threads, 4 threads pinned to separate cores | same | same |
| OMP_WAIT_POLICY / GOMP_SPINCOUNT | same | same |
| official `llama-bench` (stock model, MSVC build) | 7.7 | - |
| **our build without OpenMP** | **7.2** | **12.3** |
| Linux (WSL), OpenMP / without | 6.7 / 6.2 | 10.4 / 10.9 |

### GPU on the HP (official llama.cpp b11558 Vulkan tool, stock model)

| Setup | Prompt (tok/s) | Answer (tok/s) |
|---|---|---|
| CPU only | 18 | 7.7 |
| MX150, 0 layers on GPU (it only does the big prompt matmuls) | 59 (512-token prompt) | 4.4 |
| MX150, 16 layers | 51 | 4.7 |
| MX150, all layers | out of memory (1.9 GB free, model 4.1 GB) | - |
| Intel UHD 620, driver from 2018 | crash at load (access violation) | - |

A weak GPU makes prompt reading 3.3x faster and answers no faster. A strong GPU is untested (the cloud GPU idea).

### Not done yet

Linux Allow dialog click on WSLg (display stopped answering after `wsl --shutdown`; the code path was fixed and unit-tested). Open WebUI and Aider not re-run since strict field checking. GPU support in Cale itself. Stages 3 to 8 of the plan.

## Phase 6 stage 2b: the GPU "slowdown" was a setup bug (2026-10-13)

Question (Shonu): a GPU that makes answers slower than the CPU looks like a bug. Is it?
Answer: yes. Logs and scripts: `results/gpu/` (gpu3 to gpu7). Plan: `docs/PLAN.md`.

HP laptop (i7-8550U, MX150 2 GB, Intel UHD 620). Official llama.cpp b11558 `llama-bench`, stock Gemma 3n E4B Q4_0, 512-token prompt, tokens/s.

| Setup | Prompt | Answer |
|---|---|---|
| CPU only, repack on | 19.2 | 7.7 |
| CPU only, repack off | 11.8 | 3.8 |
| MX150, 0 layers, host buffer on (the old test) | 58.7 | 3.9 |
| MX150, 0 layers, `--no-host` | 21.1 | 7.4 |
| MX150, 16 layers, `--no-host` | 31.1 | 8.2 |
| MX150, auto-fit (`-fitt 300`), `--no-host` (3 runs, 2 days) | 28.5 | 9.2 |
| MX150, 24+ layers chosen by hand | out of GPU memory | - |
| Intel UHD 620, 2018 driver | crash at load | - |
| Intel UHD 620, driver 31.0.101.2145 (0 layers / all layers) | 11.2 / 11.6 | 3.7 / 4.4 |

Cause: with a GPU backend loaded, CPU weights go in the GPU's host buffer. That buffer is not repacked, so answers run at the "repack off" speed. `no_host` fixes it.
Still a trade-off: repacked weights cannot be offloaded to the GPU for prompts (`repack.cpp`: `is_host = nullptr`, `get_tensor = nullptr`; scheduler offload rule at `ggml-backend.cpp:1113`).
Device numbers swapped after the Intel driver update (Vulkan0 was the MX150 before, the Intel after). gpu5 labels "intel"/"nvidia" are reversed for runs M and N. Select devices by properties, never by index.
The Intel driver changed on the HP: 24.20.100.6344 (2018) to 31.0.101.2145. Installer exit code 1000, then a restart. The NVIDIA driver is still 472.19 (2021), CUDA not tested.

## Phase 6 stage 2b, part 2: GPU backends load at run time, and the width patch works on a CPU/GPU split (2026-10-13)

What was built: `llama/CMakeLists.txt` option `CALE_BACKENDS_DL=ON` (CPU variants for every instruction set plus GPU backends as separate libraries, found next to `cale_llama`). `llama/shim/cale_llama.cpp`: `cale_devices` (devices listed by type, name, memory), `cale_load_dev` (device, layer count or llama.cpp's own fit, `no_host`), `cale_placement`. `cale/backends/llamalib.py`: `devices()` and the same options on `Lib`. The official Windows `ggml-vulkan.dll` (built with MSVC) loads into our MinGW build: no Vulkan SDK was needed on the HP.

**A bug the GPU exposed in our width patch.** With part of the model on a GPU, every size smaller than full crashed (null function pointer). Cause: a size is a view of a weight, and a view has no memory yet when the scheduler picks a device. So the operation using it was not pinned to the weight's device. The scheduler then copied a view of a *repacked* weight to the GPU, and a repacked weight cannot be read back (`get_tensor` is null). Found with a crash trace in the shim (`CALE_CRASH_TRACE=1`, `scripts/resolve_crash_offsets.py`). Fix: `ggml/src/ggml-backend.cpp` in the patch now looks through a view to the weight it views. A first fix (setting the view's memory by hand) broke the CPU: it skipped the step that prepares a repacked view, and gave `nan`. Kept only as a lesson.

Check: `scripts/gpu_check.py` (HP Windows, `.cale.gguf`, 178-token prompt, 4 threads, tokens/s):

| Setup | Size | Quality (NLL) | Prompt | Answer |
|---|---|---|---|---|
| CPU | e4b / e2.49 / e2b | 0.918 / 1.003 / 1.011 | 15.9 / 25.0 / 30.4 | 6.5 / 9.5 / 11.8 |
| Intel iGPU, everything on it | e4b / e2.49 / e2b | 0.916 / 1.011 / 1.012 | 10.3 / 15.7 / 19.8 | 4.1 / 5.7 / 6.6 |
| MX150, fit chose 13 layers | e4b / e2.49 / e2b | 0.918 / 0.999 / 1.007 | 21.0 / 20.1 / 28.5 | 6.7 / 4.0 / 3.8 |

Quality matches the CPU within 0.004 everywhere. **But speed is not right yet:** on the MX150 the fit is made for the full size. Smaller sizes skip layers 20-24, which sit on the GPU, so the answer gets 2.5-3x slower than the CPU (4.0 and 3.8 against 9.5 and 11.8). The Intel iGPU is slower than the CPU at every size. So a placement must be chosen per size, by measured or predicted speed, and "GPU used" cannot be a load-time decision alone. This is now a stage 2b item.

**The HP link.** Findings and changes:

| Finding | Evidence | Change |
|---|---|---|
| The network path is fine | Tailscale direct, 6 ms | none |
| The laptop stalls for tens of seconds when a model loads (347,000 page reads per second) | HP monitor log vs ssh log, 39 s hang at 00:49 | jobs run at BelowNormal priority (`scripts/hp`) |
| A shared ssh connection dies silently and the next call waits 50 s | `read from master failed` | no shared connection. `~/.ssh/config`: probe every 5 s, give up after 20 s. A fresh login is 0.5 s |
| A failed connection looked like "link down" | `hssh` gave up at once | `hssh` retries 4 times when ssh exits 255 |
| The Wi-Fi card may be turned off by Windows | `PnPCapabilities` was 16 | set to 24, adapter restarted |

After the changes, ssh calls during a run with three model loads: 30 took 1 s, 9 took under 1 s, 1 took 2 s, none failed.

## Phase 6 stages 3 to 8 (2026-10-13)

Tests: every suite in `cale/tests/` through `scripts/run_suites.py` (Mac MLX, Mac llama, HP Windows, HP Linux, Linux arm64 in a container).

**Goals in tokens per second, no timing run (stages 3, 4).** Cale predicts speed from memory speed and the bytes one token reads. It corrects the prediction from the first answer.

| Computer | First estimate off by | After the first answer off by |
|---|---|---|
| Mac (MLX) | 16% at most | 10% at most |
| HP Windows | 14% | 6% (loaded computer: 1% first, 10% after) |
| Linux | 16% | 4% |

The gate was 25% and 10%. A small model that drafts for the big one was rejected before building it. The nested small size is only about 1.8 times faster, so the drafter cannot pay for itself.

**Several answers at once (stage 5).** Cale runs up to 4 answers together on the llama engine. Order is fair and sizes stay exact.

| Computer | One answer | Four answers together |
|---|---|---|
| HP CPU | 7.3 tokens/s | 8.6 tokens/s total |
| Mac CPU | - | 1.8 times the throughput of one |

On a weak CPU, batching gives fairness, not speed. The MLX engine stays one answer at a time.

**Tools, JSON, prompt cache (stage 6).** `tools`, `tool_choice`, `response_format` (`json_object`, `json_schema`) work on both engines. The guide checks every token, so the JSON is always valid. The prompt cache is kept per app after a security review found that one app could learn about another app's prompts through timing.

**Pictures (stage 6).** MLX only. A picture is 256 prompt tokens. A 768 by 768 picture costs about 1.3 s of reading. The tower that reads pictures (about 0.3 GB) loads for the call and is freed after it. The test picture (red circle, the word HELLO) was described correctly. The llama engine answers `400 unsupported_content`. Reason: llama.cpp can read Gemma 3n pictures, but no ready file exists. Making one needs the original 16 GB weights from Hugging Face. Audio input: not done on either engine.

**No screen, any language (stage 7).** `cale allow NAME`, connect tickets, leases and a Node client work. A Windows bug found by the tests: the empty device NUL says it is a terminal. `cale allow` then waited for typing that never came. Fixed with `oslayer.is_terminal` (asks the console).

**More computers (stage 8).**

| Finding | Evidence | Change |
|---|---|---|
| Linux arm64 works | All server suites pass in an Ubuntu 24.04 arm64 container | installer accepts arm64 |
| The Linux library needed `libgomp` | The container had no OpenMP | `GGML_OPENMP=OFF`, static C++ runtime on Linux |
| A container can hold less memory than the host | `--memory 5g` test | Cale reads the container limit. Full size gives `507 unsafe_for_this_computer`. A goal call works. No out-of-memory kill. |
| A package named `concurrent` hid the Python one | test crashed on import | renamed `crowd.py` |
| The installer left out `cancel.py` and `stops.py` | `install_check` | both added to the file list |
| Windows arm64 | the library builds (cross compiler) | **built, not tested**: no such computer here |

**Zoo 2 on the Mac found three real bugs (2026-10-13).** The zoo is 14 new probe apps, written by a reviewer who saw only the public docs.

| Bug | Evidence | Fix |
|---|---|---|
| Several tool results in a row gave `500 engine_error` | the Gemma chat template needs alternating turns | `tools.merge_same_role` joins same-role turns |
| A fact in a prompt of about 5000 tokens was missed on the Mac engine only | the llama engine found `7391-LANTERN`. MLX said `8814-MARBLE` with pieces of 2048 tokens, and was right with 512 | the prompt is read in pieces of at most the local window (512). It was also 10% faster |
| An unknown size such as `gemma-3n:nope` gave 400, the docs said 404 | zoo `curl` app | now `404 model_not_found` |

Demo apps: Aider and Open WebUI still sent the removed `deadline_s`. Both now use `min_tokens_per_s` and `max_first_token_s`. Aider was run end to end against a test Cale and fixed a bug in a file.
