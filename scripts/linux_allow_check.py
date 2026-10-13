"""Click Cale's real Linux Allow dialog, both ways, on a virtual screen. Needs Linux with Xvfb, zenity and xdotool installed
(apt-get install xvfb zenity xdotool). No person and no desktop are needed.

usage: python scripts/linux_allow_check.py MODEL.gguf [--lib PATH_TO_libcale_llama.so]
It starts a test Cale (port 47898, its own home, no auto-approve) with DISPLAY on a virtual screen. For each answer ("allow",
"deny") an app asks to connect, the dialog appears, the script clicks the button, and checks what Cale says. Then the key from the
"allow" run must work for a chat call, and the "deny" run must have given no key.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--lib")
a = ap.parse_args()
PORT, DISPLAY = 47898, ":99"
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


for tool in ("Xvfb", "zenity", "xdotool"):
    if not shutil.which(tool):
        sys.exit(f"{tool} is not installed (apt-get install xvfb zenity xdotool)")

home = tempfile.mkdtemp(prefix="cale-allow-")
env = dict(os.environ, DISPLAY=DISPLAY, GDK_BACKEND="x11", CALE_HOME=home, CALE_PORT=str(PORT), CALE_BACKEND="llama",
           CALE_MODEL=a.model, CALE_DEBUG="1", CALE_AUTO_APPROVE="0", PYTHONUNBUFFERED="1")
env.pop("WAYLAND_DISPLAY", None)
if a.lib:
    env["CALE_LLAMA_LIB"] = a.lib
xvfb = subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", "1280x800x24"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
env["PYTHONPATH"] = str(ROOT)
srv = subprocess.Popen([sys.executable, "-m", "cale.server"], cwd=ROOT, env=env, stdout=open(os.path.join(home, "server.log"), "w"), stderr=subprocess.STDOUT)


def call(path, body=None, key=None, timeout=60):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def xdo(*args):
    return subprocess.run(["xdotool", *args], env=env, capture_output=True, text=True, timeout=20).stdout.strip()


def dialog():
    """The Allow dialog window, once it is on the screen: (id, x, y, width, height)."""
    for _ in range(60):
        ids = xdo("search", "--name", "^Cale$").split()
        if ids:
            geo = dict(l.split("=") for l in xdo("getwindowgeometry", "--shell", ids[0]).splitlines() if "=" in l)
            return ids[0], int(geo["X"]), int(geo["Y"]), int(geo["WIDTH"]), int(geo["HEIGHT"])
        time.sleep(0.5)
    return None


def answer(app, allow):
    out = {}
    t = threading.Thread(target=lambda: out.update(zip(("status", "body"), call("/cale/connect", {"app": app}, timeout=90))))
    t.start()
    win = dialog()
    check(f"the Allow dialog for '{app}' appears on the screen", win is not None)
    if win:
        wid, x, y, w, h = win
        shot = subprocess.run(["xdotool", "getwindowname", wid], env=env, capture_output=True, text=True).stdout.strip()
        check("its title is Cale", shot == "Cale", shot)
        bx = x + (int(w * 0.75) if allow else int(w * 0.25))      # the buttons are the bottom row: Don't Allow on the left, Allow on the right
        by = y + h - 28
        xdo("mousemove", str(bx), str(by))
        time.sleep(0.3)
        xdo("click", "1")
    t.join(timeout=100)
    return out


try:
    for _ in range(90):
        try:
            call("/cale/status", timeout=3)
            break
        except Exception:
            time.sleep(1)
    got = answer("Click Allow App", True)
    check("clicking Allow gives a key", got.get("status") == 200 and got.get("body", {}).get("api_key"), got)
    key = got.get("body", {}).get("api_key")
    if key:
        s, r = call("/v1/chat/completions", {"model": "gemma-3n:fast", "max_tokens": 8, "messages": [{"role": "user", "content": "Say hello."}]}, key, timeout=300)
        check("the key works for a chat call", s == 200 and r["choices"][0]["message"]["content"], r)
    got = answer("Click Deny App", False)
    check("clicking Don't Allow gives no key (403 denied)", got.get("status") == 403 and "api_key" not in got.get("body", {}), got)
    check("no dialog is left on the screen", not xdo("search", "--name", "^Cale$"))
finally:
    srv.terminate()
    xvfb.terminate()
    time.sleep(1)
    shutil.rmtree(home, ignore_errors=True)

print("\nall passed" if not FAIL else f"\nFAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
