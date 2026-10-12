"""Step 4: check that streamed reads really come from the SSD, and measure the width-block overhead.

| Check | Pass |
|---|---|
| F_NOCACHE reads leave no pages in the page cache (mincore) | resident fraction ~0 after reading |
| 2nd uncached read of a region is as slow as the 1st | ratio ~1 |
| SSD speed by chunk size and thread count | GB/s table |
| reads overlap GPU work | together ~ max(read, compute), not the sum |
| down_proj as 3 column blocks vs one matrix (decode and prefill shapes) | overhead stated |

usage: nocache_check.py NEST_FILE   -> results/nocache-check.json
"""
import ctypes
import fcntl
import json
import mmap
import os
import sys
import threading
import time

import mlx.core as mx

from nest.nestfile import NestFile
from nest.offload import F_NOCACHE

libc = ctypes.CDLL("libc.dylib", use_errno=True)
PAGE = mmap.PAGESIZE
nf = NestFile(sys.argv[1])
out = {}


def resident_fraction(off, n):
    """Fraction of the file range [off, off+n) in the page cache (mincore on a private mapping)."""
    a0 = off // PAGE * PAGE
    mm = mmap.mmap(nf.fd, n + off - a0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=a0)
    npages = (len(mm) + PAGE - 1) // PAGE
    vec = (ctypes.c_char * npages)()
    base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))
    libc.mincore(base, ctypes.c_size_t(len(mm)), vec)
    frac = sum(1 for v in vec.raw if v & 1) / npages
    del base
    mm.close()
    return frac


def evict(off, n):
    a0 = off // PAGE * PAGE
    fd = os.open(nf.path, os.O_RDWR)
    mm = mmap.mmap(fd, n + off - a0, offset=a0)
    base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))
    libc.msync(base, ctypes.c_size_t(len(mm)), 0x0002)   # MS_INVALIDATE
    mm.close()
    os.close(fd)


def nocache_read(off, n, chunk, threads=1):
    fds = []
    for _ in range(threads):
        fd = os.open(nf.path, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        fds.append(fd)
    parts = [(off + i, min(chunk, n - i)) for i in range(0, n, chunk)]

    def work(k):
        buf = mmap.mmap(-1, chunk)          # page-aligned: required for uncached reads
        for j in range(k, len(parts), threads):
            o, c = parts[j]
            os.preadv(fds[k], [memoryview(buf)[:c]], o)
        buf.close()

    t = time.perf_counter()
    ths = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    dt = time.perf_counter() - t
    for fd in fds:
        os.close(fd)
    return dt


A = nf.header["regions"]["A"]
base = nf.data_start + A[0]
GB = 1 << 30

# 1-2. Uncached reads stay uncached.
off, n = base, GB
evict(off, n)
r0 = resident_fraction(off, n)
t1 = nocache_read(off, n, 16 << 20)
r1 = resident_fraction(off, n)
t2 = nocache_read(off, n, 16 << 20)
out["uncached"] = {"resident_before": r0, "resident_after_read": r1, "first_s": t1, "second_s": t2,
                   "gbps_first": n / t1 / 1e9, "ratio_second_first": t2 / t1}
print(out["uncached"], flush=True)

# 3. SSD speed by chunk size and threads (fresh 256 MB region per cell; the file is ~7.8 GB).
speed = {}
o = base + GB
CELL = GB // 4
for chunk in (1 << 20, 4 << 20, 16 << 20, 64 << 20):
    for th in (1, 2, 4):
        evict(o, CELL)
        dt = nocache_read(o, CELL, chunk, th)
        speed[f"{chunk >> 20}MB x{th}"] = round(CELL / dt / 1e9, 2)
        o += CELL
out["ssd_gbps"] = speed
print(speed, flush=True)

# 4. Overlap: GPU matmul loop while reading 1 GB.
w = mx.random.normal((8192, 8192)).astype(mx.bfloat16)
x = mx.random.normal((512, 8192)).astype(mx.bfloat16)
mx.eval(w, x)


def gpu(k):
    for _ in range(k):
        mx.eval(x @ w)


gpu(3)
t = time.perf_counter(); gpu(40); g = time.perf_counter() - t
evict(o, GB)
r = nocache_read(o, GB, 16 << 20, 2)
o += GB
evict(o, GB)
th = threading.Thread(target=nocache_read, args=(o, GB, 16 << 20, 2))
t = time.perf_counter(); th.start(); gpu(40); th.join(); both = time.perf_counter() - t
out["overlap"] = {"gpu_s": g, "read_s": r, "together_s": both, "max_s": max(g, r), "sum_s": g + r}
print(out["overlap"], flush=True)

# 5. Width-block overhead: down_proj (3840 x 15360) as one 4-bit matrix vs 3 column blocks.
O, I = 3840, 15360
cuts = [0, 13056, 13824, 15360]
wf = (mx.random.normal((O, I)) * 0.02).astype(mx.bfloat16)
q1 = mx.quantize(wf, group_size=32, bits=4)
blocks = [tuple(mx.contiguous(a[:, c0 // s: c1 // s]) for a, s in zip(q1, (8, 32, 32))) for c0, c1 in zip(cuts, cuts[1:])]
mx.eval(q1, blocks)


def qmm(x, q):
    return mx.quantized_matmul(x, *q, transpose=True, group_size=32, bits=4)


mm = {}
for T in (1, 512):
    x = mx.random.normal((1, T, I)).astype(mx.bfloat16)
    cases = {"one matrix, full width": lambda: qmm(x, q1),
             "3 blocks, full width": lambda: qmm(x[..., :13056], blocks[0]) + qmm(x[..., 13056:13824], blocks[1]) + qmm(x[..., 13824:], blocks[2]),
             "1 block, 85% width": lambda: qmm(x[..., :13056], blocks[0])}
    for name, fn in cases.items():
        for _ in range(10):
            mx.eval(fn())
        ts = []
        for _ in range(7):
            t = time.perf_counter()
            for _ in range(50):
                mx.eval(fn())
            ts.append((time.perf_counter() - t) / 50 * 1e6)
        mm[f"T={T} {name}"] = round(sorted(ts)[3], 1)
out["down_proj_us"] = mm
print(mm, flush=True)
json.dump(out, open("results/nocache-check.json", "w"), indent=1)
