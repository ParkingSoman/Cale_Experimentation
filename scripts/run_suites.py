"""Run Cale's test suites on this computer and print one summary. Works on Mac, Windows and Linux.

usage: python scripts/run_suites.py MODEL [--lib PATH_TO_cale_llama] [--only honest,objects,...] [--keep-home]
  MODEL  a model folder (Mac, MLX) or a .gguf file (Windows, Linux)
  --lib  the cale_llama library to test (sets CALE_LLAMA_LIB). Default: whatever the backend finds.

Suites that load the model themselves run first, then a test server (port 47899, its own home) is started for the
suites that talk to a running Cale. The installed Cale is not touched. Exit code 1 if any check failed.
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OWN_MODEL = ["oslayer", "choose", "constrain", "install_check", "engine_features", "backend_llama"]     # these load the model or need no server
SERVER = ["honest", "relative", "objects", "goals", "structured", "promptcache", "images", "logprobs", "crowd", "anylang", "e2e"]
SERVER_ENV = {"anylang": {"CALE_HEADLESS": "1", "CALE_DEV_APPROVE": "0", "CALE_APPROVE_WAIT_S": "8"}}    # a server with no screen: the person answers in a terminal
PORT = 47899


def run(name, extra, env, timeout):
    t = time.time()
    try:
        p = subprocess.run([PY, str(ROOT / "cale" / "tests" / f"{name}.py"), *extra], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
        out, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, code = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), -9
        out += "\nTIMEOUT"
    lines = out.splitlines()
    passed = sum(1 for l in lines if l.startswith("PASS"))
    failed = [l for l in lines if l.startswith("FAIL")]
    return {"name": name, "passed": passed, "failed": failed, "code": code, "seconds": round(time.time() - t), "tail": lines[-6:]}


def port_open():
    try:
        socket.create_connection(("127.0.0.1", PORT), 1).close()
        return True
    except OSError:
        return False


def wait_ready(timeout):
    """True when a server answers a real request (not only when the port accepts a connection)."""
    import urllib.request
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/cale/status", timeout=3).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def wait_closed(timeout):
    end = time.time() + timeout
    while port_open() and time.time() < end:
        time.sleep(0.5)


def stop_server(proc):
    """End a server process and everything it started. On Windows terminate() can leave the real Python behind its launcher."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_with_server(name, env, keep_home):
    """One suite against its own fresh server and home: suites leave holds and loaded sizes behind, and must not affect each other."""
    wait_closed(30)                                      # the last server must be gone before the next one starts
    home = tempfile.mkdtemp(prefix=f"cale-{name}-")
    log = open(Path(home) / "server.log", "w")
    env = dict(env, **SERVER_ENV.get(name, {}), CALE_TEST_HOME=home)
    srv = subprocess.Popen([PY, str(ROOT / "scripts" / "cale_dev_server.py"), home], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(180):
            return {"name": name, "passed": 0, "failed": [f"FAIL server did not start (see {Path(home) / 'server.log'})"], "code": 1, "seconds": 0, "tail": []}
        return run(name, ["--base", f"http://127.0.0.1:{PORT}"] + (["--model"] if name == "e2e" else []), env, 2400)
    finally:
        stop_server(srv)
        log.close()
        wait_closed(30)
        if not keep_home:
            shutil.rmtree(home, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lib")
    ap.add_argument("--only")
    ap.add_argument("--keep-home", action="store_true")
    a = ap.parse_args()
    mac = sys.platform == "darwin" and os.environ.get("CALE_BACKEND") != "llama"          # a Mac normally runs MLX; CALE_BACKEND=llama tests the llama.cpp engine here
    env = dict(os.environ, CALE_MODEL=a.model, PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT))
    if a.lib:
        env["CALE_LLAMA_LIB"] = a.lib
    only = set(a.only.split(",")) if a.only else None
    results = []
    for name in OWN_MODEL:
        if only and name not in only:
            continue
        extra = [a.model] + ([a.lib] if (a.lib and name == "backend_llama") else [])
        if name in ("oslayer", "choose", "constrain", "install_check"):
            extra = []
        if name == "backend_llama" and mac:
            continue                                     # the llama.cpp backend is not used on a Mac
        results.append(run(name, extra, env, 1800))
        print(f"{name}: {results[-1]['passed']} passed, {len(results[-1]['failed'])} failed ({results[-1]['seconds']} s)", flush=True)
    for name in [n for n in SERVER if not only or n in only]:
        results.append(run_with_server(name, env, a.keep_home))
        print(f"{name}: {results[-1]['passed']} passed, {len(results[-1]['failed'])} failed ({results[-1]['seconds']} s)", flush=True)
    print("\n== summary ==")
    bad = 0
    for r in results:
        print(f"{r['name']:>16}: {r['passed']:3d} passed, {len(r['failed'])} failed, exit {r['code']}")
        for f in r["failed"]:
            print("      " + f[:160])
        if r["code"] and not r["failed"]:
            print("      (no FAIL line, exit code", r["code"], ") last lines:", *r["tail"], sep="\n         ")
        bad += len(r["failed"]) + (1 if r["code"] and not r["failed"] else 0)
    print("ALL PASS" if not bad else f"{bad} PROBLEM(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
