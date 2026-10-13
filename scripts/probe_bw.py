"""Print this computer's memory read speed as Cale's predictor sees it. usage: python scripts/probe_bw.py [LLAMA_LIB]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.platform == "darwin":
    from nest.gemma3n import memory_bandwidth
    probe = memory_bandwidth
else:
    from cale.backends.llama import default_threads
    from cale.backends.llamalib import memory_bandwidth
    lib = sys.argv[1] if len(sys.argv) > 1 else None
    n = default_threads()
    probe = lambda: memory_bandwidth(n, lib)                    # noqa: E731
for _ in range(3):
    t = time.perf_counter()
    bw = probe()
    print(f"{bw / 1e9:.1f} GB/s  ({time.perf_counter() - t:.2f} s)")
