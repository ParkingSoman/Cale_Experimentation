"""When the engine fails while the GPU is in use, answers that have not said a word start again on the CPU (cale/batching.py _engine_failed).
Run: python cale/tests/gpu_retry.py     (no model, no engine, no GPU: the engine is a stand-in)
"""
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import batching  # noqa: E402
from cale.engine import EngineError  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


class Fake(batching.BatchMixin):
    def __init__(self, placement):
        self.cv = threading.Condition()
        self.inbox, self.active = [], []
        self.m = SimpleNamespace(placement=placement)
        self.gpu_off_until, self._gpu_env_was = 0.0, None
        self.events, self.unloaded = [], []

    def log(self, text):
        self.events.append(text)

    def _unload(self, why):
        self.unloaded.append(why)


def request(out_tokens):
    r = batching.Request([], 16, 0.0, "app", 100, None, 1.0, None, None, time.time(), False, None, None)
    r.stream = SimpleNamespace(out=out_tokens)
    return r


os.environ.pop("CALE_GPU", None)
e = Fake("device=Vulkan1;gpu_layers=0;no_host=1; chosen because x")
silent, talking, waiting = request(0), request(5), request(0)
waiting.stream = None
e.active, e.inbox = [silent, talking], [waiting]
e._engine_failed(OSError("native out of memory"))
check("an answer that said nothing is started again", silent in e.inbox and silent.stream is None and silent.retried and silent.q.empty())
check("an answer that already spoke gets the error", isinstance(talking.q.get_nowait(), EngineError) and talking not in e.inbox)
check("a waiting request is kept, not failed", waiting in e.inbox and waiting.q.empty())
check("the CPU is used for a while", os.environ.get("CALE_GPU") == "off" and e.gpu_off_until > time.time())
check("the model is unloaded for the new try", e.unloaded == ["engine error"])
check("the older request goes first", e.inbox.index(silent) < e.inbox.index(waiting))
e.active = [silent]
silent.stream = SimpleNamespace(out=0)
e._engine_failed(OSError("again"))
check("a second failure of the same answer is an error, not a loop", isinstance(silent.q.get_nowait(), EngineError))

os.environ.pop("CALE_GPU", None)
c = Fake("device=cpu;gpu_layers=0;no_host=1; chosen because the CPU")
r1 = request(0)
c.active = [r1]
c._engine_failed(OSError("cpu failure"))
check("on the CPU a failure stays an error", isinstance(r1.q.get_nowait(), EngineError) and not c.inbox and "CALE_GPU" not in os.environ)
d = Fake("device=Vulkan1;gpu_layers=0")
r2 = request(0)
d.active = [r2]
d._engine_failed(RuntimeError("debug: injected engine failure"))
check("the debug failure is still an error on a GPU placement", isinstance(r2.q.get_nowait(), EngineError))
os.environ.pop("CALE_GPU", None)

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
