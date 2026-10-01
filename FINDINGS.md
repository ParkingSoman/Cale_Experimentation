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
