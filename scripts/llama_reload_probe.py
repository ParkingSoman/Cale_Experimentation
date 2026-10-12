"""Does dropping memory work after the model is closed and loaded again in the same process?
usage: llama_reload_probe.py MODEL.gguf [LIB]   Prints process memory and the OS count after each step."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cale.backends.llama import LlamaNested  # noqa: E402

model, lib = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
for round_ in range(1, 4):
    m = LlamaNested.load(model, lib_path=lib)
    a = m.active_bytes() / 1e9
    m.reside(["e2b"])
    b, rb = m.active_bytes() / 1e9, m.resident_now() / 1e9
    time.sleep(2)
    c, rc = m.active_bytes() / 1e9, m.resident_now() / 1e9
    print(f"round {round_}: after load {a:.2f} GB | after reside e2b {b:.2f} (os-count {rb:.2f}) | 2 s later {c:.2f} (os-count {rc:.2f})", flush=True)
    m.close()
print("PROBE_DONE", flush=True)
