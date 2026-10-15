"""The Cale Hub: one page that shows Cale at work. Standard library only.

  python -m cale.hub            serve the Hub and open it in the browser
  python -m cale.hub --setup    the same, and install Cale first (the one-download installers use this)

The browser cannot call Cale (Cale refuses requests with an Origin header), so this small server does it,
using the Hub's own key. The page talks only to this server. It shows numbers, never prompt text.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent                                     # the folder that holds cale/, demo/, ...
sys.path.insert(0, str(SRC))
from cale import keys, oslayer, paths  # noqa: E402

HUB_PORT = int(os.environ.get("CALE_HUB_PORT", "47800"))
CALE_URL = os.environ.get("CALE_URL", f"http://{paths.HOST}:{paths.PORT}")
APPS_DIR = Path(os.environ.get("CALE_DEMO_APPS") or (SRC / "demo" / "apps" if (SRC / "demo" / "apps").exists() else paths.HOME / "apps"))

# --- setup (the installer, run as a child process) -----------------------------------------------------------------
SETUP = {"running": False, "stage": "", "pct": 0, "done": 0, "total": 0, "error": None, "finished": False, "since": None}
STAGES = {"runtime": "Putting Cale on this computer", "python": "Getting ready", "model": "Downloading the AI model",
          "convert": "Preparing the model", "tune": "Fitting Cale to this computer", "service": "Starting Cale", "done": "Done"}


def installed():
    """The installer finished. (A Cale that answers also counts: see state().)"""
    return (paths.HOME / "installed.ok").exists()


def run_setup():
    """Run cale/install.py and turn its CALE_PROGRESS lines into SETUP. Runs in a thread."""
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    env["CALE_PROGRESS"] = "1"
    tail = []
    try:
        p = subprocess.Popen([sys.executable, str(SRC / "cale" / "install.py")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env, cwd=str(SRC))
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith("CALE_PROGRESS "):
                u = json.loads(line[len("CALE_PROGRESS "):])
                SETUP["stage"] = u["stage"]
                if u["stage"] == "model" and u.get("total"):
                    SETUP.update(done=u["done"], total=u["total"], pct=min(90, round(25 + 65 * u["done"] / u["total"])))
                elif "pct" in u:
                    SETUP["pct"] = max(SETUP["pct"], u["pct"])
            else:
                tail = (tail + [line])[-6:]
        if p.wait() != 0:
            SETUP["error"] = " ".join(tail[-3:]) or f"the installer stopped (code {p.returncode})"
        else:
            SETUP.update(pct=100, stage="done", finished=True)
    except Exception as e:
        SETUP["error"] = f"{type(e).__name__}: {e}"
    finally:
        SETUP["running"] = False


def start_setup():
    if SETUP["running"] or installed():
        return
    SETUP.update(running=True, error=None, finished=False, pct=0, stage="runtime", since=time.time())
    threading.Thread(target=run_setup, daemon=True).start()


# --- talking to Cale ------------------------------------------------------------------------------------------------
def cale_get(path, key=None, timeout=4):
    req = urllib.request.Request(CALE_URL + path, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_computer = {"at": 0, "value": None}


def host_numbers():
    total, avail = oslayer.total_ram(), oslayer.available_ram()
    try:
        busy = round(oslayer.cpu_busy(), 3)
    except Exception:
        busy = None
    return {"os": {"darwin": "Mac", "win32": "Windows"}.get(sys.platform, "Linux"), "cpus": os.cpu_count(),
            "ram_total_gb": round(total / 1e9, 1), "ram_free_gb": round(avail / 1e9, 1), "cpu_busy": busy,
            "power": oslayer.power_state()}


def state():
    out = {"installed": installed(), "setup": {**SETUP, "label": STAGES.get(SETUP["stage"], "")}, "host": host_numbers(),
           "cale": None, "activity": None, "computer": None, "apps": apps_list()}
    try:
        out["cale"] = cale_get("/cale/status")
        out["installed"] = True                                   # an answering Cale is installed, wherever its files are
    except (urllib.error.URLError, OSError, ValueError):
        return out                                                # Cale is not running (yet)
    if time.time() - _computer["at"] > 20 or _computer["value"] is None:
        try:
            _computer.update(at=time.time(), value=cale_get("/cale/computer")["this_computer"])
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            pass
    out["computer"] = _computer["value"]
    try:
        out["activity"] = cale_get("/cale/activity", keys.hub_key())
    except urllib.error.HTTPError as e:
        if e.code == 401:                                         # the key file and the key store disagree: make a fresh key
            try:
                (paths.HOME / "hub.key").unlink()
                out["activity"] = cale_get("/cale/activity", keys.hub_key())
            except Exception:
                pass
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return out


# --- the demo apps --------------------------------------------------------------------------------------------------
RUNNING = {}                                                      # app id -> Popen


def registry():
    try:
        return json.loads((APPS_DIR / "registry.json").read_text())["apps"]
    except (FileNotFoundError, ValueError, KeyError):
        return []


def port_open(port):
    import socket
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((paths.HOST, port)) == 0


def _expand(template):
    return Path(template.replace("{src}", str(SRC)).replace("{home}", str(paths.HOME)).replace("{apps}", str(APPS_DIR)))


def _first_existing(templates):
    for t in templates:
        if _expand(t).exists():
            return str(_expand(t))
    return None


def launch_argv(spec):
    """The command that starts an app that lives outside demo/apps (Open WebUI): its Python and its launcher file, or None."""
    py, script = _first_existing(spec["launch"]["python"]), _first_existing(spec["launch"]["script"])
    return [py, script] if py and script else None


def app_present(spec):
    """Is the app really here? For an outside app: do its program and launcher exist."""
    if "launch" in spec:
        return launch_argv(spec) is not None
    return (APPS_DIR / spec["id"]).exists()


def apps_list():
    out = []
    for a in registry():
        out.append({k: a.get(k) for k in ("id", "name", "cale_app", "port", "values", "line", "language", "settings")}
                   | {"up": port_open(a["port"]), "present": app_present(a)})
    return out


def node_path():
    """Node for the Node demo app: the copy that came with Cale, else one on PATH."""
    import shutil
    for cand in (paths.HOME / "bin" / "node", paths.HOME / "bin" / "node.exe", SRC / "bin" / "node", SRC / "bin" / "node.exe"):
        if cand.exists():
            return str(cand)
    return shutil.which("node") or "node"


def launch(app_id):
    spec = next((a for a in registry() if a["id"] == app_id), None)
    if not spec:
        return 404, {"error": "no such app"}
    if port_open(spec["port"]):
        return 200, {"url": f"http://{paths.HOST}:{spec['port']}/"}
    env = {**os.environ, "CALE_URL": CALE_URL, "CALE_HOME": str(paths.HOME)}
    env["PATH"] = str(paths.HOME / "bin") + os.pathsep + env.get("PATH", "")     # so an app can run `cale start`
    if "launch" in spec:                                          # an app that lives outside demo/apps (Open WebUI)
        argv = launch_argv(spec)
        if argv is None:
            return 404, {"error": "This app is not installed on this computer."}
        cwd = str(Path(argv[1]).parent)
        if (SRC / "cale" / "install.py").exists():
            env["CALE_SRC"] = str(SRC)
    else:
        argv = [node_path(), "app.mjs"] if spec.get("language") == "node" else [sys.executable, "app.py"]
        cwd = str(APPS_DIR / spec["id"])
    env["CALE_NO_BROWSER"] = "1"
    try:
        RUNNING[app_id] = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           stdin=subprocess.DEVNULL, **({"creationflags": 0x08000000} if oslayer.WIN else {}))
    except OSError as e:
        return 500, {"error": f"Could not start it: {e}"}
    for _ in range(80):                                           # up to 20 s for the app to listen
        if port_open(spec["port"]):
            return 200, {"url": f"http://{paths.HOST}:{spec['port']}/"}
        time.sleep(0.25)
    return 504, {"error": "The app did not start in time."}


# --- the web server -------------------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _guard(self):
        if (self.headers.get("Host") or "").split(":")[0] not in (paths.HOST, "localhost"):    # DNS rebinding
            self._send(403, {"error": "bad host"})
            return False
        return True

    def do_GET(self):
        if not self._guard():
            return
        try:
            if self.path in ("/", "/index.html"):
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, state())
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._guard():
            return
        if self.headers.get("Origin") and self.headers["Origin"].split("//")[-1] not in (f"{paths.HOST}:{HUB_PORT}", f"localhost:{HUB_PORT}"):
            return self._send(403, {"error": "other sites may not do this"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/setup":
                start_setup()
                return self._send(200, {"ok": True})
            if self.path == "/api/open":
                code, out = launch(str(body.get("id", "")))
                return self._send(code, out)
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(404, {"error": "not found"})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--setup" in argv and not installed():
        start_setup()
    server = ThreadingHTTPServer((paths.HOST, HUB_PORT), Handler)
    url = f"http://{paths.HOST}:{HUB_PORT}/"
    print(f"Cale Hub: {url}", flush=True)
    if "--no-browser" not in argv and not os.environ.get("CALE_NO_BROWSER"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for p in RUNNING.values():
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
