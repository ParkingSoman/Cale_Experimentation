"""Step 6a: what a switch costs. Load time (cold and warm) and time to the first answer,
for the one-file 12B and for the models a 12B + E4B stack would reload.

Cold = the model's files are evicted from the page cache first (msync MS_INVALIDATE,
checked with mincore). Each load runs in a fresh process. Medians over --reps.

usage: stack_cost.py [--reps 3]   -> results/stack-cost.json
"""
import argparse
import ctypes
import json
import mmap
import os
import statistics
import subprocess
import sys

libc = ctypes.CDLL("libc.dylib", use_errno=True)
SETUPS = {
    "nest 12B (one file)": ("nest", "models/g12-nest/model.nest"),
    "12B qg4 (mlx-vlm)": ("vlm", "models/qg4"),
    "E4B 4-bit (mlx-vlm)": ("vlm", "models/gemma-4-e4b-it-4bit"),
}
CHILD = r'''
import sys, time, json
t0 = time.perf_counter()
kind, path = sys.argv[1], sys.argv[2]
if kind == "nest":
    from nest.runtime import NestedGemma
    M = NestedGemma.load(path); model, proc = M.model, M.processor
else:
    from nest.io import load
    model, proc = load(path)
t1 = time.perf_counter()
from nest.gen import generate
generate(model, proc, {"user": "Say hello."}, max_tokens=8)
t2 = time.perf_counter()
print(json.dumps({"load_s": t1 - t0, "first_answer_s": t2 - t0}))
'''


def files(path):
    if os.path.isfile(path):
        return [path]
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")]


def evict(path):
    for f in files(path):
        fd = os.open(f, os.O_RDWR)
        n = os.fstat(fd).st_size
        mm = mmap.mmap(fd, n)
        base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))
        libc.msync(base, ctypes.c_size_t(n), 0x0002)    # MS_INVALIDATE
        del base
        mm.close()
        os.close(fd)


def resident(path):
    tot = res = 0
    for f in files(path):
        fd = os.open(f, os.O_RDONLY)
        n = os.fstat(fd).st_size
        mm = mmap.mmap(fd, n, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        pages = (n + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vec = (ctypes.c_char * pages)()
        base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))
        libc.mincore(base, ctypes.c_size_t(n), vec)
        del base
        res += sum(1 for v in vec.raw if v & 1)
        tot += pages
        mm.close()
        os.close(fd)
    return res / tot


def child(kind, path):
    env = dict(os.environ, PYTHONPATH=".")
    out = subprocess.run([sys.executable, "-c", CHILD, kind, path], capture_output=True, text=True, env=env)
    line = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not line:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(line[-1])


ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
res = {}
for rep in range(args.reps):
    for name, (kind, path) in SETUPS.items():
        evict(path)
        r0 = resident(path)
        cold = child(kind, path)
        warm = child(kind, path)
        res.setdefault(name, []).append({"cold": cold, "warm": warm, "resident_before_cold": r0})
        print(rep, name, f"cold load {cold['load_s']:.1f}s (first answer {cold['first_answer_s']:.1f}s), "
              f"warm load {warm['load_s']:.1f}s; resident before cold {r0:.2f}", flush=True)
out = {}
for name, rs in res.items():
    out[name] = {f"{c}_{k}": statistics.median(r[c][k] for r in rs) for c in ("cold", "warm")
                 for k in ("load_s", "first_answer_s")}
    out[name]["resident_before_cold_max"] = max(r["resident_before_cold"] for r in rs)
print(json.dumps(out, indent=1))
json.dump(out, open("results/stack-cost.json", "w"), indent=1)
