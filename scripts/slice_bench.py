"""Step 0: which down_proj layout slices best? Times one 12B down_proj (15360 -> 3840).

A  one table, full matmul (baseline, width 100 only)
A3 today's three column blocks (13056 / 768 / 1536), summed
B  same table, strided column slice: w[:, :n/8], s[:, :n/32]
C  stored transposed and quantized along the output axis: row prefix w[:n]

Run:  PYTHONPATH=. .venv/bin/python scripts/slice_bench.py
"""
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

I, H, G, BITS = 15360, 3840, 32, 4
WIDTHS = [1.0, 0.75, 0.5]
RUNS = 9


def q(w):
    return mx.quantize(w, group_size=G, bits=BITS)


def qmm(x, parts, transpose):
    w, s, b = parts
    return mx.quantized_matmul(x, w, s, b, transpose=transpose, group_size=G, bits=BITS)


def timeit(fn, reps):
    mx.eval(fn())                       # warm up
    ts = []
    for _ in range(RUNS):
        t = time.perf_counter()
        for _ in range(reps):
            mx.eval(fn())
        ts.append((time.perf_counter() - t) / reps * 1000)
    return statistics.median(ts)


def main():
    mx.random.seed(0)
    W = (mx.random.normal((H, I)) * 0.02).astype(mx.bfloat16)        # down_proj: (out, in)
    A = q(W)
    C = q(W.T)                                                         # (in, out), quantized along out
    edges = [0, 13056, 13824, 15360]
    blocks = [q(W[:, edges[k]:edges[k + 1]]) for k in range(3)]
    mx.eval(*A, *C, *[p for b in blocks for p in b])
    out = {}
    for tokens, reps, label in ((1, 50, "decode"), (512, 5, "prefill")):
        x = mx.random.normal((1, tokens, I)).astype(mx.bfloat16)
        mx.eval(x)
        for wd in WIDTHS:
            n = int(round(wd * I / 64) * 64)
            xs = x[..., :n]
            w, s, b = A
            res = {}
            if wd == 1.0:
                res["A"] = timeit(lambda: qmm(x, A, True), reps)
            nb = sum(1 for c in edges[1:] if c <= n)
            if nb:
                def a3():
                    y = None
                    for k in range(nb):
                        t = qmm(x[..., edges[k]:edges[k + 1]], blocks[k], True)
                        y = t if y is None else y + t
                    return y
                res["A3"] = timeit(a3, reps)
            res["B"] = timeit(lambda: qmm(xs, (w[:, :n // 8], s[:, :n // 32], b[:, :n // 32]), True), reps)
            cp = tuple(mx.contiguous(p) for p in (w[:, :n // 8], s[:, :n // 32], b[:, :n // 32]))
            mx.eval(*cp)
            res["Bc"] = timeit(lambda: qmm(xs, cp, True), reps)     # B with the copy cached
            res["C"] = timeit(lambda: qmm(xs, (C[0][:n], C[1][:n], C[2][:n]), False), reps)
            out[f"{label}:w{int(wd * 100)}"] = {k: round(v, 4) for k, v in res.items()}
            print(label, f"w{int(wd * 100)}", {k: round(v, 4) for k, v in res.items()}, "ms", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/slice-bench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
