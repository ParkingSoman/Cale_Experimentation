"""App 1: Open WebUI with Cale as its AI.

Open WebUI itself is not changed. This file starts it and adds:
  - /api/cale/status, /api/cale/install, /api/cale/connect  (what the "Get Cale" card calls)
  - one <script> in index.html that shows the card until Cale is installed and connected

When Cale is connected, Open WebUI gets this setup (the same split it uses with cloud AI). Each model is a
Cale "model object": the app says what it needs, Cale picks the size that fits this computer.
  answers  -> deadline CALE_ANSWER_DEADLINE s for a CALE_ANSWER_TOKENS token answer, and at most CALE_MEMORY_PCT % of the RAM
  tasks    -> titles, follow-up ideas, tags, autocomplete: deadline CALE_TASK_DEADLINE s for CALE_TASK_TOKENS tokens
A fast computer runs both at the full size. A weaker one runs smaller sizes. Open WebUI is told nothing about that.

Run:  <venv>/bin/python apps/open-webui/cale_webui.py        (Windows: <venv>/Scripts/python.exe)
Settings: CALE_URL (default http://127.0.0.1:47821), CALE_SRC (the folder that holds cale/install.py),
          CALE_MODEL_SRC (a local model to link in instead of downloading), PORT (default 8810)
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
CALE_URL = os.environ.get("CALE_URL", "http://127.0.0.1:47821").rstrip("/")
CALE_SRC = Path(os.environ.get("CALE_SRC", HERE.parent.parent))
KEY_FILE = Path(os.environ.get("DATA_DIR", HERE / "data")) / "cale-key"
APP_NAME = os.environ.get("CALE_APP_NAME", "Open WebUI")                   # the name the Allow question shows
os.environ.setdefault("CALE_PORT", str(urlparse(CALE_URL).port or 47821))   # the cale package finds Cale at the same address
sys.path.insert(0, str(CALE_SRC))
# What this app needs from a model, in its own words. Cale turns that into a size, per call, for this computer.
ANSWERS = dict(name="answers", deadline_s=float(os.environ.get("CALE_ANSWER_DEADLINE", "90")),
               max_tokens=int(os.environ.get("CALE_ANSWER_TOKENS", "600")),
               memory_pct=float(os.environ.get("CALE_MEMORY_PCT", "40")))
TASKS = dict(name="tasks", deadline_s=float(os.environ.get("CALE_TASK_DEADLINE", "30")),
             max_tokens=int(os.environ.get("CALE_TASK_TOKENS", "60")))

os.environ.setdefault("DATA_DIR", str(HERE / "data"))
os.environ.setdefault("FRONTEND_BUILD_DIR", str(HERE / "build"))
os.environ.setdefault("WEBUI_AUTH", "False")                  # one local user, no login screen
os.environ.setdefault("ENABLE_OLLAMA_API", "False")
os.environ.setdefault("ENABLE_EVALUATION_ARENA_MODELS", "False")   # hide the built-in "arena-model" entry
os.environ.setdefault("OFFLINE_MODE", "True")                 # no model downloads, no update checks
os.environ.setdefault("ENABLE_VERSION_UPDATE_CHECK", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("DO_NOT_TRACK", "True")
os.environ.setdefault("SCARF_NO_ANALYTICS", "True")
os.environ.setdefault("OPENAI_API_BASE_URLS", CALE_URL + "/v1")
os.environ.setdefault("TASK_MODEL_EXTERNAL", TASKS["name"])
os.environ.setdefault("ENABLE_AUTOCOMPLETE_GENERATION", "True")


def link_or_copy(src, dst):
    """A hard link when the disk allows it (no extra space), else a copy (another drive, or a file system without links)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_frontend():
    """A copy of the web app (hard links, no extra disk) whose index.html also loads the Cale card."""
    src = Path(find_spec("open_webui").submodule_search_locations[0]) / "frontend"
    dst = Path(os.environ["FRONTEND_BUILD_DIR"])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=link_or_copy, ignore=shutil.ignore_patterns("index.html"))
    html = (src / "index.html").read_text()
    tag = '<script src="/cale-ui/cale.js" defer></script>'
    (dst / "index.html").write_text(html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag)
    (dst / "cale-ui").mkdir()
    link_or_copy(HERE / "cale.js", dst / "cale-ui" / "cale.js")


# --- talking to Cale -------------------------------------------------------------------------
def cale_get(path, key=None, timeout=3):
    req = urllib.request.Request(CALE_URL + path, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def read_key():
    try:
        return KEY_FILE.read_text().strip()
    except OSError:
        return ""


INSTALL = {"running": False, "stage": "", "pct": 0, "done": 0, "total": 0, "error": None}
INSTALL_LOCK = threading.Lock()


def cale_installed():
    from cale import oslayer
    return (oslayer.home_dir() / "runtime" / "cale" / "server.py").exists()


def run_install():
    """Run cale/install.py and turn its CALE_PROGRESS lines into INSTALL. Runs in a thread."""
    script = CALE_SRC / "cale" / "install.py"
    # Open WebUI's offline mode sets these. The installer must be free to download the model.
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    env["CALE_PROGRESS"] = "1"
    tail = []
    try:
        p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith("CALE_PROGRESS "):
                u = json.loads(line[len("CALE_PROGRESS "):])
                INSTALL["stage"] = u["stage"]
                if u["stage"] == "model" and u.get("total"):
                    INSTALL.update(done=u["done"], total=u["total"], pct=min(95, round(25 + 70 * u["done"] / u["total"])))
                elif "pct" in u:
                    INSTALL["pct"] = u["pct"]
            else:
                tail = (tail + [line])[-6:]
        if p.wait() != 0:
            INSTALL["error"] = " ".join(tail[-3:]) or f"the installer stopped (code {p.returncode})"
    except Exception as e:                                    # the card shows this text
        INSTALL["error"] = f"{type(e).__name__}: {e}"
    finally:
        INSTALL["running"] = False


def state():
    """missing | installing | needs_connect | ready | failed"""
    if INSTALL["running"]:
        return "installing"
    if INSTALL["error"]:
        return "failed"
    try:
        cale_get("/cale/status")
    except (urllib.error.URLError, OSError, ValueError):
        return "starting" if cale_installed() else "missing"      # installed: keep_models starts it
    key = read_key()
    if not key:
        return "needs_connect"
    try:
        cale_get("/v1/models", key)
        return "ready"
    except (urllib.error.URLError, OSError, ValueError):
        return "needs_connect"


# --- routes ------------------------------------------------------------------------------------
def live_models():
    """What Cale chose for this app's models, for the card: [{"id", "last": {"size_pct", "goal_met", ...}}]."""
    key = read_key()
    if not key:
        return []
    try:
        data = cale_get("/v1/models", key)["data"]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []
    return [{"id": m["id"], **(m.get("cale") or {})} for m in data if m.get("owned_by") == APP_NAME]


async def status_route(request):
    from starlette.responses import JSONResponse
    from cale import oslayer
    return JSONResponse({"state": state(), "install": dict(INSTALL), "models": live_models(),
                         "download_gb": 5.5 if oslayer.MAC else 4.4})


async def install_route(request):
    from starlette.responses import JSONResponse
    with INSTALL_LOCK:
        if not INSTALL["running"]:
            INSTALL.update(running=True, stage="starting", pct=0, done=0, total=0, error=None)
            threading.Thread(target=run_install, daemon=True).start()
    return JSONResponse({"ok": True})


async def connect_route(request):
    """Ask Cale for a key (the computer shows its Allow question), then point Open WebUI at Cale."""
    import asyncio
    from starlette.responses import JSONResponse

    def ask():
        req = urllib.request.Request(CALE_URL + "/cale/connect", data=json.dumps({"app": APP_NAME}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=150) as r:
            return json.loads(r.read())

    try:
        got = await asyncio.to_thread(ask)
    except urllib.error.HTTPError as e:
        why = json.loads(e.read() or b"{}").get("error", {}).get("message", f"Cale said {e.code}")
        return JSONResponse({"ok": False, "error": why}, status_code=200)
    except (urllib.error.URLError, OSError) as e:
        return JSONResponse({"ok": False, "error": f"Cale did not answer ({e})"}, status_code=200)
    key = got["api_key"]
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key)
    try:
        KEY_FILE.chmod(0o600)                       # no effect on Windows: the user's own folder is private there
    except OSError:
        pass
    await point_open_webui_at_cale(request, key)
    return JSONResponse({"ok": True})


async def point_open_webui_at_cale(request, key):
    from open_webui.models.config import Config
    from open_webui.routers.openai import clear_openai_model_cache
    await Config.upsert({
        "openai.enable": True,
        "openai.api_base_urls": [CALE_URL + "/v1"],
        "openai.api_keys": [key],
        "openai.api_configs": {"0": {"enable": True, "tags": [], "prefix_id": "", "model_ids": [ANSWERS["name"], TASKS["name"]],
                                     "connection_type": "local", "auth_type": "bearer"}},
        "task.model.external": TASKS["name"],
        "task.model.default": TASKS["name"],
        "task.autocomplete.enable": True,
        "task.title.enable": True,
        "task.follow_up.enable": True,
        "task.tags.enable": True,
        "ui.default_models": ANSWERS["name"],
    })
    await clear_openai_model_cache(request)


def keep_models():
    """While Open WebUI runs: make sure Cale is running, then make this app's two model objects.
    The objects end by themselves when Open WebUI quits. If Cale restarts, make them again."""
    from cale.client import ensure_running, model
    while True:
        if cale_installed():
            try:
                ensure_running()
            except Exception as e:
                print(f"[cale] could not start Cale: {e}", flush=True)
        key = read_key()
        if key:
            made = []
            try:
                for spec in (ANSWERS, TASKS):
                    m = model(CALE_URL + "/v1", key, **spec)
                    m.__enter__()
                    made.append(m)
                    print(f"[cale] {m.id}: Cale runs the {m.info.get('size_pct', 'auto')}% size now; the goal is "
                          f"{ {k: v for k, v in spec.items() if k != 'name'} }", flush=True)
                made[0].closed.wait()
            except Exception as e:
                print(f"[cale] no model yet: {e}", flush=True)
            finally:
                for m in made:
                    m.close()
        time.sleep(10)


def make_secrets():
    """Open WebUI wants two secret keys. Make them once and keep them with its data."""
    import secrets
    f = Path(os.environ["DATA_DIR"]) / "secrets.json"
    have = json.loads(f.read_text()) if f.exists() else {}
    for name in ("WEBUI_SECRET_KEY", "OAUTH_SESSION_TOKEN_ENCRYPTION_KEY"):
        have.setdefault(name, secrets.token_urlsafe(32))
        os.environ.setdefault(name, have[name])
    f.write_text(json.dumps(have))
    try:
        f.chmod(0o600)
    except OSError:
        pass


def main():
    Path(os.environ["DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    make_secrets()
    build_frontend()
    from starlette.routing import Route
    from open_webui.main import app
    app.router.routes[0:0] = [Route("/api/cale/status", status_route, methods=["GET"]),
                              Route("/api/cale/install", install_route, methods=["POST"]),
                              Route("/api/cale/connect", connect_route, methods=["POST"])]
    threading.Thread(target=keep_models, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8810")), log_level="warning")


if __name__ == "__main__":
    main()
