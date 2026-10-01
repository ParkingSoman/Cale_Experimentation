"""Chain 16 MLP layers per eval to measure bandwidth-bound time per layer."""
import time, statistics
import mlx.core as mx
H, I, B, G, L = 3840, 15360, 4, 64, 16
def q(w): return mx.quantize(w, group_size=G, bits=B)
def rnd(*s): return mx.random.normal(s).astype(mx.bfloat16) * 0.02
layers = [dict(gate=q(rnd(I, H)), up=q(rnd(I, H)), down=q(rnd(H, I)), downT=q(rnd(I, H))) for _ in range(L)]
mx.eval(layers)
x0 = mx.random.normal((1, 1, H)).astype(mx.bfloat16)

def qmm(x, p, transpose=True, rows=None):
    w, s, b = p
    if rows is not None: w, s, b = w[:rows], s[:rows], b[:rows]
    return mx.quantized_matmul(x, w, s, b, transpose=transpose, group_size=G, bits=B)

def run_copy(copies):
    x = x0
    for c in copies:
        a = qmm(x, c["gate"]) * qmm(x, c["up"])
        x = x + qmm(a, c["down"]) * 0.01
    return x

def run_view(n):
    x = x0
    for c in layers:
        a = qmm(x, c["gate"], rows=n) * qmm(x, c["up"], rows=n)
        x = x + qmm(a, c["downT"], transpose=False, rows=n) * 0.01
    return x

def bench(f, reps=30):
    for _ in range(5): mx.eval(f())
    ts = []
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(reps): mx.eval(f())
        ts.append((time.perf_counter() - t) / reps / L * 1e6)
    return statistics.median(ts)

for frac in (1.0, 0.75, 0.5, 0.25):
    n = int(I * frac) // 64 * 64
    copies = [dict(gate=tuple(a[:n] for a in c["gate"]), up=tuple(a[:n] for a in c["up"]),
                   down=q(mx.dequantize(*c["down"], group_size=G, bits=B)[:, :n])) for c in layers]
    mx.eval(copies)
    mb = 3 * H * n * B / 8 / 1e6
    tc, tv = bench(lambda: run_copy(copies)), bench(lambda: run_view(n))
    print(f"width {frac:.2f}: {mb:5.1f} MB/layer | copy {tc:6.1f} us ({mb/tc*1e3:5.0f} GB/s) | view {tv:6.1f} us ({mb/tv*1e3:5.0f} GB/s)")
    del copies
