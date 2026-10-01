"""Prefill-shaped matmul speed: standard layout vs transposed (nest) layout for down_proj."""
import time, statistics, mlx.core as mx
H, I, G = 3840, 15360, 64
for bits in (8, 4):
    for M in (1, 512):
        a = mx.random.normal((M, I)).astype(mx.bfloat16)
        std = mx.quantize(mx.random.normal((H, I)).astype(mx.bfloat16) * .02, group_size=G, bits=bits)
        tr = mx.quantize(mx.random.normal((I, H)).astype(mx.bfloat16) * .02, group_size=G, bits=bits)
        mx.eval(a, std, tr)
        def f_std(): return mx.quantized_matmul(a, *std, transpose=True, group_size=G, bits=bits)
        def f_tr(): return mx.quantized_matmul(a, *tr, transpose=False, group_size=G, bits=bits)
        res = {}
        for name, f in (("standard", f_std), ("transposed", f_tr)):
            for _ in range(5): mx.eval(f())
            ts = []
            for _ in range(7):
                t = time.perf_counter()
                for _ in range(20): mx.eval(f())
                ts.append((time.perf_counter() - t) / 20 * 1e3)
            res[name] = statistics.median(ts)
        print(f"{bits}-bit, {M:4d} tokens: standard {res['standard']:.2f} ms | transposed {res['transposed']:.2f} ms | ratio {res['transposed']/res['standard']:.2f}")
