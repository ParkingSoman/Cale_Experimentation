"""The model download starts at the beginning of the install (Windows and Linux) and runs while the other steps run (cale/install.py prefetch_model).
Run: python cale/tests/install_prefetch.py   No network, no model.
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import install  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


home = Path(tempfile.mkdtemp(prefix="cale-prefetch-test-"))
(home / "models").mkdir()
calls = []


def slow_download(url, dest):
    calls.append((url, threading.current_thread().name))
    time.sleep(0.6)
    Path(dest).write_bytes(b"gguf")


real = install.download
install.download = slow_download
try:
    t = time.time()
    pre = install.prefetch_model(home, "llama")
    check("it returns at once, with the download still running", pre is not None and time.time() - t < 0.3 and pre[0].is_alive())
    time.sleep(0.1)
    check("the download runs in another thread", calls and calls[0][1] != threading.current_thread().name, calls)
    install.wait_for(pre)
    check("wait_for waits for the download", not pre[0].is_alive() and (home / "models" / "gemma-3n-e4b.download.gguf").exists())
    check("the completion marker is written", (home / "models" / "gemma-3n-e4b.download.complete").exists())
    check("a finished download is not started again", install.prefetch_model(home, "llama") is None)

    home2 = Path(tempfile.mkdtemp(prefix="cale-prefetch2-test-")); (home2 / "models").mkdir()
    check("the Mac does not prefetch (its download needs Python's packages)", install.prefetch_model(home2, "mlx") is None)
    import os
    os.environ["CALE_MODEL_SRC"] = "somewhere"
    try:
        check("a local model file means no download", install.prefetch_model(home2, "llama") is None)
    finally:
        del os.environ["CALE_MODEL_SRC"]

    # a download that fails stops the install, with the same message as before
    def failing(url, dest):
        install.die("the model download failed: no network")

    install.download = failing
    home3 = Path(tempfile.mkdtemp(prefix="cale-prefetch3-test-")); (home3 / "models").mkdir()
    pre = install.prefetch_model(home3, "llama")
    stopped = None
    try:
        install.wait_for(pre)
    except SystemExit as e:
        stopped = e.code
    check("a failed download stops the install (not silently)", stopped == 1, stopped)
    check("no completion marker after a failure", not (home3 / "models" / "gemma-3n-e4b.download.complete").exists())

    def crashing(url, dest):
        raise OSError("disk full")

    install.download = crashing
    home4 = Path(tempfile.mkdtemp(prefix="cale-prefetch4-test-")); (home4 / "models").mkdir()
    pre = install.prefetch_model(home4, "llama")
    stopped = None
    try:
        install.wait_for(pre)
    except SystemExit as e:
        stopped = e.code
    check("any other error also stops the install", stopped == 1, stopped)
finally:
    install.download = real

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
