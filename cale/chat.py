"""Cale's chat window server. Run:  python -m cale.chat

The window program (cale/host/mac/CaleWindow.swift) starts this with --server-exec / --server-arg and shows the page it
announces. What this does, in order:
  1. makes sure Cale is running;
  2. asks Cale for a "chat" model object (goals, not a size) and keeps it with a hold;
  3. prepares a data folder for the chat window (chat history, documents, settings) under Cale's home;
  4. starts the chat window server (AnythingLLM, rebranded, see cale/anythingllm/) and its document reader with Cale's Node;
  5. prints  READY <port> <token>  on one line when the page answers. The token is a secret for this launch: the window opens
     http://127.0.0.1:<port>/?t=<token> once, and the server then wants the cookie it sets. Anything else (a web page in a
     browser, another program) gets 403. See cale/anythingllm/preload.js;
  6. stays alive, keeps the hold, and ends everything when its parent ends or when it gets SIGTERM.

Files (all under Cale's home, see cale/paths.py):
  chat/        the program: server/, collector/, template.db, models-seed/, preload.js   (installed by cale/install.py)
  chat-data/   what the person makes: storage/ (database, documents, search index), hotdir/, secrets.json
  logs/chat.log

Settings (environment, all optional; for tests): CALE_CHAT_BUNDLE, CALE_CHAT_DATA, CALE_NODE, CALE_CHAT_PORT,
  CALE_CHAT_TPS (default 3), CALE_CHAT_MEMORY_PCT (40), CALE_CHAT_TOKENS (1024), CALE_CHAT_CONTEXT (8192).
"""
import json
import os
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import client, keys, oslayer, paths

BUNDLE = Path(os.environ.get("CALE_CHAT_BUNDLE") or paths.HOME / "chat")
DATA = Path(os.environ.get("CALE_CHAT_DATA") or paths.HOME / "chat-data")
APP_NAME = "Cale"                    # the name Cale knows this window by (first-party: no Allow question, like the Hub)
OBJECT_NAME = "chat"
WORKSPACE = "Chat"
START_TIMEOUT_S = 120

# What the chat needs, in its own words. Cale turns this into a size on each computer (cale/goals.py).
GOALS = dict(name=OBJECT_NAME, min_tokens_per_s=float(os.environ.get("CALE_CHAT_TPS", "3")),
             memory_pct=float(os.environ.get("CALE_CHAT_MEMORY_PCT", "40")), max_tokens=int(os.environ.get("CALE_CHAT_TOKENS", "1024")))
CONTEXT_TOKENS = int(os.environ.get("CALE_CHAT_CONTEXT", "8192"))

# Settings written into the chat window's database before each start.
SETTINGS = {"custom_app_name": "Cale", "meta_page_title": "Cale", "onboarding_complete": "true",
            "footer_data": "[]", "limit_user_messages": "false"}


def log(msg):
    print(f"[chat] {msg}", file=sys.stderr, flush=True)


def say_ready(port, token=""):
    """READY <port> <token>. The window program fills {port} and {token} in its address. (A server for other apps prints just READY <port>.)"""
    print(f"READY {port} {token}".rstrip(), flush=True)


def node_path():
    for c in (os.environ.get("CALE_NODE"), str(paths.HOME / "bin" / ("node.exe" if oslayer.WIN else "node")), shutil.which("node")):
        if c and Path(c).exists():
            return c
    raise SystemExit("Cale's chat needs Node, and it is not installed.")


def free_port():
    with socket.socket() as s:
        s.bind((paths.HOST, 0))
        return s.getsockname()[1]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


# --- the data folder --------------------------------------------------------------------------
def _link(path, target):
    """path -> target (a link), so everything the person makes lives in the data folder and the program folder stays unchanged."""
    path = Path(path)
    if path.is_symlink():
        if Path(os.readlink(path)) == Path(target):
            return
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, path, target_is_directory=True)


def secrets_for_chat():
    f = DATA / "secrets.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        s = {k: secrets.token_urlsafe(40) for k in ("JWT_SECRET", "SIG_KEY", "SIG_SALT")}
        oslayer.private_file(f, json.dumps(s))
        return s


def seed_settings(db):
    con = sqlite3.connect(db, timeout=30)
    try:
        for label, value in SETTINGS.items():
            con.execute("INSERT INTO system_settings (label, value) VALUES (?, ?) "
                        "ON CONFLICT(label) DO UPDATE SET value = excluded.value", (label, value))
        con.commit()
    finally:
        con.close()


def prepare_data():
    """Make the data folder, link it into the program folder, start from the empty database, seed the settings."""
    if not (BUNDLE / "server" / "index.js").exists():
        raise SystemExit(f"Cale's chat is not installed (no {BUNDLE / 'server' / 'index.js'}).")
    storage, hotdir = DATA / "storage", DATA / "hotdir"
    for d in (storage, hotdir, storage / "documents", storage / "models"):
        d.mkdir(parents=True, exist_ok=True)
    _link(BUNDLE / "server" / "storage", storage)
    _link(BUNDLE / "collector" / "hotdir", hotdir)
    db = storage / "anythingllm.db"
    if not db.exists():
        shutil.copyfile(BUNDLE / "template.db", db)
    seed = BUNDLE / "models-seed"
    if seed.is_dir():                                      # the file that turns text into numbers: copied once, so documents work offline
        for src in seed.rglob("*"):
            dst = storage / "models" / src.relative_to(seed)
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif not dst.exists():
                shutil.copyfile(src, dst)
    seed_settings(db)


# --- Cale: the key and the hold ---------------------------------------------------------------
def chat_key():
    """Cale's own window gets a key without a question (the person installed Cale to chat). Kept in a private file."""
    f = paths.HOME / "chat.key"
    try:
        key = f.read_text().strip()
        if keys.check(key) == APP_NAME:
            return key
    except OSError:
        pass
    key = keys.create(APP_NAME)
    paths.HOME.mkdir(parents=True, exist_ok=True)
    oslayer.private_file(f, key)
    return key


class Holder(threading.Thread):
    """Keeps the chat model object alive. If Cale restarts, it asks again. `id` is the object's name for chat calls."""

    def __init__(self, base_url, key):
        super().__init__(daemon=True)
        self.base_url, self.key = base_url, key
        self.id, self.info, self.error = None, None, None
        self.ready, self.stop = threading.Event(), threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                with client.model(self.base_url, self.key, **GOALS) as m:
                    self.id, self.info, self.error = m.id, m.info, None
                    self.ready.set()
                    while not self.stop.is_set() and not m.closed.wait(1):
                        pass
            except (client.CaleError, OSError, ValueError, KeyError) as e:
                self.error = str(e)
                self.ready.set()
                log(f"Cale did not give the chat object ({e}). Trying again.")
            self.stop.wait(3)


# --- the servers ------------------------------------------------------------------------------
class Servers:
    def __init__(self, node, model_id, key, port=None, cookie=None):
        self.node, self.model_id, self.key = node, model_id, key
        self.token = secrets.token_urlsafe(24)                    # the one-time code in the window's first address: new for every start
        self.cookie = cookie or secrets.token_urlsafe(24)         # the cookie secret: kept across restarts, so an open window keeps working
        self.port = port or int(os.environ.get("CALE_CHAT_PORT") or 0) or free_port()
        self.collector_port = free_port()
        self.procs = []
        self.logfile = None

    def env(self):
        paths.LOGS.mkdir(parents=True, exist_ok=True)
        e = dict(os.environ)
        for k in [k for k in e if k.startswith(("GENERIC_OPEN_AI", "LLM_", "EMBEDDING_", "VECTOR_", "OPEN_AI", "AUTH_"))]:
            e.pop(k)                                         # nothing from the person's shell may change the chat window
        e.update(secrets_for_chat())
        e.update(
            NODE_ENV="production", SERVER_PORT=str(self.port), COLLECTOR_PORT=str(self.collector_port),
            STORAGE_DIR=str(BUNDLE / "server" / "storage"), DISABLE_TELEMETRY="true", CHECKPOINT_DISABLE="1",
            LLM_PROVIDER="generic-openai", GENERIC_OPEN_AI_BASE_PATH=f"http://{paths.HOST}:{paths.PORT}/v1",
            GENERIC_OPEN_AI_MODEL_PREF=self.model_id, GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT=str(CONTEXT_TOKENS),
            GENERIC_OPEN_AI_API_KEY=self.key, GENERIC_OPEN_AI_MAX_TOKENS=str(GOALS["max_tokens"]),
            EMBEDDING_ENGINE="native", VECTOR_DB="lancedb", CALE_PARENT_PID=str(os.getpid()),
            CALE_CHAT_TOKEN=self.token, CALE_CHAT_COOKIE=self.cookie, CALE_CHAT_MAIN_PORT=str(self.port), CALE_CHAT_COLLECTOR_PORT=str(self.collector_port))
        return e

    def start(self):
        paths.LOGS.mkdir(parents=True, exist_ok=True)
        self.logfile = open(paths.LOGS / "chat.log", "ab")
        env = self.env()
        pre = ["-r", str(BUNDLE / "preload.js")]
        for sub in ("collector", "server"):
            self.procs.append(subprocess.Popen([self.node, *pre, "index.js"], cwd=BUNDLE / sub, env=env, stdin=subprocess.DEVNULL,
                                               stdout=self.logfile, stderr=subprocess.STDOUT))

    def alive(self):
        return all(p.poll() is None for p in self.procs)

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        end = time.time() + 5
        for p in self.procs:
            try:
                p.wait(max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []

    def request(self, path, body=None, timeout=3):
        """A call to the chat server, with the cookie it wants (its own calls need the secret too)."""
        req = urllib.request.Request(f"http://{paths.HOST}:{self.port}{path}", headers={"Cookie": f"cale_chat={self.cookie}"},
                                     data=None if body is None else json.dumps(body).encode())
        if body is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def get(self, path, timeout=3):
        return self.request(path, timeout=timeout)

    def wait_ready(self, timeout=START_TIMEOUT_S):
        end = time.time() + timeout
        while time.time() < end:
            if not self.alive():
                raise SystemExit("The chat window server stopped while starting. See " + str(paths.LOGS / "chat.log"))
            try:
                if self.get("/api/ping").get("online"):
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.4)
        raise SystemExit("The chat window server did not start in time. See " + str(paths.LOGS / "chat.log"))

    def ensure_workspace(self):
        """A first workspace, so the person can type at once."""
        try:
            if self.get("/api/workspaces").get("workspaces"):
                return
            self.request("/api/workspace/new", {"name": WORKSPACE}, timeout=20)
            self.request(f"/api/workspace/{WORKSPACE.lower()}/update", {"chatMode": "chat"}, timeout=20)     # plain chat: no tool "agent" mode, which small models handle badly
        except (urllib.error.URLError, OSError, ValueError) as e:
            log(f"could not make the first workspace ({e})")


# --- main -------------------------------------------------------------------------------------
def main():
    node = node_path()
    parent = os.getppid()
    try:
        client.ensure_running()
    except client.CaleError as e:
        print(f"Cale is not running and did not start: {e}", file=sys.stderr, flush=True)
        return 1
    prepare_data()
    holder = Holder(f"http://{paths.HOST}:{paths.PORT}", chat_key())
    holder.start()
    holder.ready.wait(90)
    if holder.id is None:
        print(f"Cale did not accept the chat: {holder.error}", file=sys.stderr, flush=True)
        return 1
    log(f"chat object {holder.id}: {holder.info}")

    servers = Servers(node, holder.id, chat_key())
    done = threading.Event()

    def bye(*_):
        done.set()

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, bye)
    try:
        servers.start()
        servers.wait_ready()
        servers.ensure_workspace()
        say_ready(servers.port, servers.token)
        while not done.is_set():
            if parent > 1 and (not pid_alive(parent) or os.getppid() != parent):
                break                                        # the window program is gone
            if holder.id and holder.id != servers.model_id:  # Cale gave the chat a new name (it restarted): start the servers again with it
                log("chat object changed; restarting")
                servers.stop()
                servers = Servers(servers.node, holder.id, servers.key, port=servers.port, cookie=servers.cookie)     # same port and cookie secret: the open window keeps working
                servers.start()
                servers.wait_ready()
                servers.ensure_workspace()
                say_ready(servers.port, servers.token)             # a new one-time code; the window program reloads its page with it
            if not servers.alive():
                print("The chat window server stopped. See " + str(paths.LOGS / "chat.log"), file=sys.stderr, flush=True)
                return 1
            done.wait(1)
    finally:
        holder.stop.set()
        servers.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
