"""Precision ladder planes: build the original weights up from the QAT 4-bit weights.

QAT 4-bit (plane A) is the original weight table rescaled (1.6x to 3.8x per
table; the scale moved into the norm weights) plus a small change. So:

    original ~= row_scale * A + B + C

  A          QAT 4-bit, exactly as stored in models/qg4 (g32)
  row_scale  one float per output row, least-squares fit of the original to A
  B          4-bit (g64) rounding of  original - row_scale * A
  C          8-bit (g64) rounding of what B missed

Tiers: 4-bit = A (with QAT norms). 8-bit = row_scale * A + B. 16-bit = all three.
8-bit and 16-bit tiers use the original norms.

Without row_scale the difference is 40-75% of the original and the 8-bit tier
error is 4-7%. With it, the difference is ~10% and the 8-bit tier error ~1-1.5%.
"""
import mlx.core as mx

A_GROUP, A_BITS = 32, 4
B_GROUP, B_BITS = 64, 4
C_GROUP, C_BITS = 64, 8


def deq(parts, group, bits):
    w, s, b = parts
    return mx.dequantize(w, s, b, group_size=group, bits=bits).astype(mx.float32)


def quant(x, group, bits):
    """Quantize in the dtype the file stores (bf16 scales and biases)."""
    return mx.quantize(x.astype(mx.bfloat16), group_size=group, bits=bits)


def rel(x, ref):
    return (mx.linalg.norm(x.reshape(-1)) / mx.linalg.norm(ref.reshape(-1))).item()


def make_planes(orig, a_parts, with_c=True):
    """orig: original weight (out, in), already in the same neuron order as A.
    a_parts: (packed, scales, biases) of plane A.
    Returns dict with a (dequantized A, float32), row_scale, b_parts, c_parts and errors."""
    o = orig.astype(mx.float32)
    a = deq(a_parts, A_GROUP, A_BITS)
    s = mx.sum(o * a, axis=1, keepdims=True) / mx.maximum(mx.sum(a * a, axis=1, keepdims=True), 1e-30)
    s = s.astype(mx.bfloat16).astype(mx.float32)
    sa = s * a
    b_parts = quant(o - sa, B_GROUP, B_BITS)
    r8 = sa + deq(b_parts, B_GROUP, B_BITS)
    out = {"a": a, "row_scale": s, "b_parts": b_parts, "r8": r8,
           "err4": rel(o - a, o), "err8": rel(o - r8, o)}
    if with_c:
        c_parts = quant(o - r8, C_GROUP, C_BITS)
        r16 = r8 + deq(c_parts, C_GROUP, C_BITS)
        out.update(c_parts=c_parts, r16=r16, err16=rel(o - r16, o))
    return out
