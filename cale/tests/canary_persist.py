"""A passed trial load of a small graphics card is kept on disk for a day, so the next start of Cale does not repeat it (cale/backends/llama.py canary_ok).
Run: python cale/tests/canary_persist.py   No graphics card, no model: the trial is a fake that counts its calls.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["CALE_HOME"] = tempfile.mkdtemp(prefix="cale-canary-test-")
for k in ("CALE_CANARY_CHILD", "CALE_NO_CANARY"):
    os.environ.pop(k, None)
from cale import paths, tuning  # noqa: E402
from cale.backends import llama  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


calls = []
verdict = {"ok": True}
tuning.canary = lambda env, path, lib, log=print: (calls.append(env), verdict["ok"])[1]

dev = {"name": "Vulkan1", "free": 1 << 30}                      # a card with 1 GB free: under the 3 GB line, so it gets a trial
sig = [[["Card", "desc", 2048]], None]
other_sig = [[["Card", "desc", 2048]], "12.6"]                  # the driver changed


def ask(**kw):
    return llama.canary_ok(dev, kw.get("layers", 0), "m.gguf", None, kw.get("sig", sig))


check("first start: a trial runs and passes", ask() is True and len(calls) == 1, calls)
check("a file says it passed", (paths.HOME / "gpu-canary.json").exists())
llama._CANARY.clear()                                           # a new process: nothing in memory
check("next start: no trial (kept on disk)", ask() is True and len(calls) == 1, calls)
llama._CANARY.clear()
check("a different driver: a new trial", ask(sig=other_sig) is True and len(calls) == 2, calls)
llama._CANARY.clear()
check("a different way of using the card: a new trial", ask(layers=999) is True and len(calls) == 3, calls)

# old enough: trusted for a day only
llama._CANARY.clear()
f = paths.HOME / "gpu-canary.json"
d = json.loads(f.read_text())
d["at"] = time.time() - llama.VERIFIED_S - 60
f.write_text(json.dumps(d))
check("after a day: a new trial", ask(layers=999, sig=json.loads(json.dumps(d["what"][1]))) is True and len(calls) == 4, calls)

# a failure is never kept as a pass, and trouble forces trials
llama._CANARY.clear()
verdict["ok"] = False
check("a failing card: the trial runs and fails", ask(sig=other_sig, layers=999) is False and len(calls) == 5, calls)
check("a failure removes the saved pass", not f.exists())
check("a failure marks trouble", (paths.HOME / "gpu-trouble.json").exists())
llama._CANARY.clear()
verdict["ok"] = True
check("in trouble: every load gets a trial, even with a saved pass", ask(sig=other_sig, layers=999) is True and len(calls) == 6, calls)
check("a pass ends the trouble", not (paths.HOME / "gpu-trouble.json").exists())
check("...and is kept again", f.exists())

# a big card needs no trial at all
big = {"name": "Big", "free": 8 << 30}
n = len(calls)
check("a card with plenty of free memory: no trial", llama.canary_ok(big, 999, "m.gguf", None, sig) is True and len(calls) == n)
# no signature given (an old caller): works as before, nothing is saved
llama._CANARY.clear(); f.unlink(missing_ok=True)
check("without a signature: the trial runs and nothing is saved", llama.canary_ok(dev, 0, "m.gguf", None) is True and not f.exists())

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
