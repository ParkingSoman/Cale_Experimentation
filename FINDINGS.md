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
