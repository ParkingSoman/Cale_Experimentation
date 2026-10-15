"""The trial load that guards the graphics card (cale/backends/llama.py canary_ok, cale/tuning.py canary).
Run: python cale/tests/canary.py     (from the repo folder or the installed runtime; needs the llama engine library and, for the last two checks, the model:
CALE_MODEL=/path/to/gemma-3n-e4b.gguf). No graphics card is needed: the trial is run with the CPU and with a model that cannot load.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["CALE_HOME"] = tempfile.mkdtemp(prefix="cale-canary-test-")
from cale import paths, tuning  # noqa: E402
from cale.backends import llama  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


calls = []
real = tuning.canary
tuning.canary = lambda env, path, lib, log=print, timeout=300: (calls.append(env), False)[1]
big = {"name": "BigCard", "free": 8 << 30}
small = {"name": "SmallCard", "free": 1 << 30}
check("a card with plenty of free memory gets no trial", llama.canary_ok(big, 999, "m.gguf", None) is True and not calls)
check("a small card gets a trial, and a failed trial says no", llama.canary_ok(small, 0, "m.gguf", None) is False and calls == ["force:prompts:SmallCard"], calls)
check("a failed trial is remembered for a day", (paths.HOME / "gpu-trouble.json").exists())
check("...so even a big card gets a trial now", llama.canary_ok(big, 999, "m.gguf", None) is False and calls[-1] == "force:full:BigCard", calls)
llama._CANARY.clear()
tuning.canary = lambda env, path, lib, log=print, timeout=300: (calls.append(env), True)[1]
check("a passed trial says yes", llama.canary_ok(small, -1, "m.gguf", None) is True and calls[-1] == "SmallCard", calls)
check("...and clears the day-long trouble mark", not (paths.HOME / "gpu-trouble.json").exists())
t = len(calls)
check("a passed trial is trusted for a while (no second trial)", llama.canary_ok(small, -1, "m.gguf", None) is True and len(calls) == t)
os.environ["CALE_CANARY_CHILD"] = "1"
check("the trial child never starts another trial", llama.canary_ok(small, 0, "m.gguf", None) is True and len(calls) == t)
del os.environ["CALE_CANARY_CHILD"]
tuning.canary = real

# the whole graphics path is tried in a child before this process touches a card
calls.clear()
tuning.canary = lambda env, path, lib, log=print, timeout=300: (calls.append(env), True)[1]
card = {"type": "GPU", "name": "Card0", "description": "A card", "total": 2 << 30, "free": 2 << 30, "handle": 1}
igpu = {"type": "IGPU", "name": "Chip0", "description": "A chip", "total": 8 << 30, "free": 8 << 30, "handle": 2}
llama._CANARY.clear()
check("path: a computer with no card of its own needs no trial", llama.gpu_path_ok([igpu], "m.gguf", None) is True and not calls)
check("path: the CPU setting needs no trial", (os.environ.update(CALE_GPU="off"), llama.gpu_path_ok([card], "m.gguf", None))[1] is True and not calls)
os.environ.pop("CALE_GPU", None)
check("path: a card of its own gets one trial the first time", llama.gpu_path_ok([card], "m.gguf", None) is True and calls == ["auto"], calls)
check("path: a pass is written down for a day", (paths.HOME / "gpu-verified.json").exists())
llama._CANARY.clear()
check("path: after a restart the written pass is trusted (no second trial)", llama.gpu_path_ok([card], "m.gguf", None) is True and len(calls) == 1, calls)
llama._CANARY.clear()
card2 = dict(card, total=4 << 30)
check("path: other cards or another driver mean a new trial", llama.gpu_path_ok([card2], "m.gguf", None) is True and len(calls) == 2, calls)
tuning.canary = lambda env, path, lib, log=print, timeout=300: (calls.append(env), False)[1]
llama._CANARY.clear()
card3 = dict(card, total=6 << 30)
check("path: a failed trial says no", llama.gpu_path_ok([card3], "m.gguf", None) is False)
check("path: ...and is remembered for a few minutes (no new trial)", llama.gpu_path_ok([card3], "m.gguf", None) is False and len(calls) == 3, calls)
check("path: ...and the trouble mark is written", (paths.HOME / "gpu-trouble.json").exists())
tuning.canary = real
llama._CANARY.clear()

# listing the devices is guarded by a child process too
from cale.backends import llamalib  # noqa: E402
lib_here = os.environ.get("CALE_LLAMA_LIB")
if lib_here and Path(lib_here).is_file():
    llamalib._ENUM.clear()
    check("the device list works in a child process on a healthy computer", llamalib.enumeration_safe(lib_here) in (True, False) and lib_here in llamalib._ENUM)
    llamalib._ENUM[lib_here] = False
    check("a failed listing means no devices (the CPU), not a crash", llamalib.devices(lib_here) == [])
    llamalib._ENUM.clear()
    os.environ["CALE_CANARY_CHILD"] = "1"
    check("the child itself is not guarded again", llamalib.enumeration_safe(lib_here) is True)
    del os.environ["CALE_CANARY_CHILD"]
else:
    print("   (no CALE_LLAMA_LIB: the device listing checks are skipped)")

model = os.environ.get("CALE_MODEL") or str(paths.MODEL)
if Path(model).is_file():
    t0 = time.time()
    check("a real trial on the CPU passes", tuning.canary("off", model, os.environ.get("CALE_LLAMA_LIB"), log=print) is True)
    print(f"   (the trial took {time.time() - t0:.1f} s)")
    bad = Path(tempfile.mkdtemp()) / "not-a-model.gguf"
    bad.write_bytes(b"GGUF" + b"\0" * 64)
    check("a model that cannot load fails the trial, and the caller lives", tuning.canary("off", str(bad), os.environ.get("CALE_LLAMA_LIB"), log=lambda t: print("   " + t[:160])) is False)
else:
    print(f"   (no model file at {model}: the two real trials are skipped)")

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
