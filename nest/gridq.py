"""Exact 4-bit encoding for QAT weights that already sit on a 16-level grid.

Google's QAT Gemma 4 weights hold, per block of 32 consecutive input weights,
values on 16 evenly spaced levels. MLX's own quantizer derives the step from
(max - min) / 15, which is wrong whenever a block leaves its top or bottom
level unused. Here the step is detected from the gaps between values, refined
by least squares, and written in MLX's affine format (w = scale * q + bias).

Result on QAT weights: ~0.5% relative error, vs ~6-7% for mx.quantize.
"""
import mlx.core as mx
import numpy as np

BITS = 4
LEVELS = 2 ** BITS - 1


def _encode_rows(wb):
    """wb: (n_blocks, B) float32. Returns q (uint8), scale, bias (float32)."""
    mn = wb.min(1, keepdims=True)
    srt = np.sort(wb, 1)
    dif = np.diff(srt, axis=1)
    tol = 1e-7 * np.abs(srt).max(1, keepdims=True)
    step = np.where(dif > tol, dif, np.inf).min(1, keepdims=True)
    span = wb.max(1, keepdims=True) - mn
    fallback = ~np.isfinite(step) | (span / np.where(np.isfinite(step), step, 1) > LEVELS + 0.5)
    step = np.where(fallback, span / LEVELS, step)
    step = np.where(step == 0, 1e-8, step)
    q = np.round((wb - mn) / step)
    num = ((wb - mn) * q).sum(1, keepdims=True)
    den = (q * q).sum(1, keepdims=True)
    step = np.where(den > 0, num / np.maximum(den, 1e-30), step)
    # Round scale and bias to bf16 (how MLX stores them), then re-pick levels against them.
    step_b = np.array(mx.array(step).astype(mx.bfloat16).astype(mx.float32))
    mn_b = np.array(mx.array(mn).astype(mx.bfloat16).astype(mx.float32))
    step_b = np.where(step_b == 0, 1e-8, step_b)
    q = np.clip(np.round((wb - mn_b) / step_b), 0, LEVELS).astype(np.uint8)
    return q, step_b, mn_b


def _pack(q):
    """Pack uint8 4-bit values (rows, cols) into uint32 words, 8 values per word, low nibble first."""
    rows, cols = q.shape
    q = q.reshape(rows, cols // 8, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    return (q << shifts).sum(-1).astype(np.uint32)


def quantize_grid(w, group_size=32, row_chunk=4096):
    """w: mx.array (out, in). Returns (packed uint32, scales bf16, biases bf16) in MLX layout."""
    out_dim, in_dim = w.shape
    packs, scales, biases = [], [], []
    for r in range(0, out_dim, row_chunk):
        wb = np.array(w[r:r + row_chunk].astype(mx.float32))
        n = wb.shape[0]
        q, s, b = _encode_rows(wb.reshape(-1, group_size))
        packs.append(_pack(q.reshape(n, in_dim)))
        scales.append(s.reshape(n, in_dim // group_size))
        biases.append(b.reshape(n, in_dim // group_size))
    return (mx.array(np.concatenate(packs)),
            mx.array(np.concatenate(scales)).astype(mx.bfloat16),
            mx.array(np.concatenate(biases)).astype(mx.bfloat16))
