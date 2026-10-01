"""Micro-benchmark: can a width slice of loaded quantized MLP weights run at full speed?"""
import time, statistics
import mlx.core as mx

H, I, B, G = 3840, 15360, 4, 64
def q(w): return mx.quantize(w, group_size=G, bits=B)
x = mx.random.normal((1, 1, H)).astype(mx.bfloat16)
gate = q(mx.random.normal((I, H)).astype(mx.bfloat16) * 0.02)
up = q(mx.random.normal((I, H)).astype(mx.bfloat16) * 0.02)
down = q(mx.random.normal((H, I)).astype(mx.bfloat16) * 0.02)         # normal layout, groups along I
downT = q(mx.random.normal((I, H)).astype(mx.bfloat16) * 0.02)        # transposed layout, groups along H
mx.eval(gate, up, down, downT)

def qmm(x, p, transpose=True, rows=None):
    w, s, b = p
    if rows is not None:
        w, s, b = w[:rows], s[:rows], b[:rows]
    return mx.quantized_matmul(x, w, s, b, transpose=transpose, group_size=G, bits=B)

def mlp_copy(x, n, parts):  # physically sliced copies (ideal speed)
    g, u, d = parts
    a = mx.multiply(qmm(x, g), qmm(x, u))
    return qmm(a, d)

def mlp_view_T(x, n):  # gate/up row slice + transposed down row slice, all views
    a = mx.multiply(qmm(x, gate, rows=n), qmm(x, up, rows=n))
    return qmm(a, downT, transpose=False, rows=n)

def bench(f, reps=200):
    for _ in range(10): mx.eval(f())
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(reps): mx.eval(f())
        ts.append((time.perf_counter() - t) / reps * 1e6)
    return statistics.median(ts)

for frac in (1.0, 0.75, 0.5, 0.25):
    n = int(I * frac) // 64 * 64
    parts = (tuple(a[:n] for a in gate), tuple(a[:n] for a in up),
             q(mx.dequantize(*down, group_size=G, bits=B)[:, :n]))
    mx.eval(parts)
    t_copy = bench(lambda: mlp_copy(x, n, parts))
    t_view = bench(lambda: mlp_view_T(x, n))
    print(f"width {frac:.2f}: physical copy {t_copy:7.1f} us | live view (down transposed) {t_view:7.1f} us")
