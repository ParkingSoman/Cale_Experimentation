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
import hashlib
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

from . import client, keys, oslayer, paths, timeline

BUNDLE = Path(os.environ.get("CALE_CHAT_BUNDLE") or paths.HOME / "chat")
DATA = Path(os.environ.get("CALE_CHAT_DATA") or paths.HOME / "chat-data")
APP_NAME = "Cale"                    # the name Cale knows this window by (first-party: no Allow question, like the Hub)
OBJECT_NAME = "chat"
WORKSPACE = "My chats"
START_TIMEOUT_S = 120
NO_WINDOW = 0x08000000 if oslayer.WIN else 0         # CREATE_NO_WINDOW: a child program that is not started from a console must not open one


FILE_WAIT_S = 20                      # how long to trust the ready file before asking the port as well (a server without the preload never writes it)


class KillWithMe:
    """Windows: a Job Object that ends its programs when this program ends for any reason (also when it is killed, where SIGTERM never
    comes). Elsewhere the servers check their parent (CALE_PARENT_PID) and this does nothing."""

    def __init__(self):
        self.job = None
        if not oslayer.WIN:
            return
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        job = k.CreateJobObjectW(None, None)
        if not job:
            return

        class Basic(ctypes.Structure):
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", wintypes.DWORD), ("c", ctypes.c_size_t), ("d", ctypes.c_size_t),
                        ("e", wintypes.DWORD), ("f", ctypes.c_size_t), ("g", wintypes.DWORD), ("h", wintypes.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in "abcdef"]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", Counters), ("p", ctypes.c_size_t), ("j", ctypes.c_size_t), ("pm", ctypes.c_size_t), ("pk", ctypes.c_size_t)]

        info = Extended()
        info.Basic.LimitFlags = 0x2000                                   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):    # JobObjectExtendedLimitInformation
            self.job, self._k, self._ct = job, k, ctypes

    def add(self, proc):
        if self.job:
            h = self._k.OpenProcess(0x0100 | 0x0001, False, proc.pid)     # PROCESS_SET_QUOTA | PROCESS_TERMINATE
            if h:
                self._k.AssignProcessToJobObject(self.job, h)
                self._k.CloseHandle(h)


KILL_WITH_ME = KillWithMe()

# What the chat needs, in its own words. Cale turns this into a size on each computer (cale/goals.py).
GOALS = dict(name=OBJECT_NAME, min_tokens_per_s=float(os.environ.get("CALE_CHAT_TPS", "7")),     # 7: text flows at reading speed (docs/CONSTANTS.md). The slider (cale/anythingllm/feel.js) then sets it for this computer.
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
    return oslayer.pid_alive(pid)            # on Windows os.kill(pid, 0) would END the process: never use it there


def _is_link(path):
    """A symbolic link, or on Windows a directory junction (it has the reparse-point flag)."""
    path = Path(path)
    if path.is_symlink():
        return True
    if oslayer.WIN:
        try:
            return bool(os.lstat(path).st_file_attributes & 0x400)       # FILE_ATTRIBUTE_REPARSE_POINT
        except (OSError, AttributeError):
            return False
    return False


def _drop_link(path):
    try:
        os.rmdir(path) if oslayer.WIN else os.unlink(path)               # a junction ends with rmdir and leaves its target alone
    except OSError:
        os.unlink(path)


# --- the data folder --------------------------------------------------------------------------
def _link(path, target):
    """path -> target (a link), so everything the person makes lives in the data folder and the program folder stays unchanged."""
    path = Path(path)
    if _is_link(path):
        if os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(target)):
            return
        _drop_link(path)
    elif path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if oslayer.WIN:                                                      # a junction needs no special right (a symbolic link does)
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(target)], capture_output=True, text=True, creationflags=NO_WINDOW)
        if r.returncode:
            raise OSError(f"could not link {path} to {target}: {(r.stdout + r.stderr).strip()}")
    else:
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
    # Everything below is the same on every start of a bundle. Do it once (the install does, through --warm) and skip it when the marker still fits:
    # on a computer with virus scanners, links and many small file writes are what a person would wait for.
    marker = DATA / ".prepared"
    stamp = hashlib.sha1((json.dumps(SETTINGS, sort_keys=True) + str(BUNDLE.resolve()) + str((BUNDLE / "bundle.json").stat().st_mtime if (BUNDLE / "bundle.json").exists() else 0)).encode()).hexdigest()
    try:
        if marker.read_text() == stamp and (storage / "anythingllm.db").exists() and (BUNDLE / "server" / "storage").exists() and (BUNDLE / "collector" / "hotdir").exists():
            return
    except OSError:
        pass
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
    marker.write_text(stamp)


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
        self.reader_started = False
        self.ready_file = DATA / f"ready-{self.port}"

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
            CALE_CHAT_TOKEN=self.token, CALE_CHAT_COOKIE=self.cookie, CALE_CHAT_MAIN_PORT=str(self.port), CALE_CHAT_COLLECTOR_PORT=str(self.collector_port),
            NODE_COMPILE_CACHE=str(paths.HOME / "cache" / "node-compile"),     # Node keeps its compiled code of the big packed files: the next start skips the compile
            CALE_CHAT_READY_FILE=str(self.ready_file),         # preload.js writes it when the chat server listens (see wait_ready)
            CALE_FEEL_FILE=str(DATA / "feel.json"))              # where the slider keeps its position (cale/anythingllm/feel.js)
        return e

    def start(self):
        """Start the chat server. The document reader starts AFTER the chat is ready (start_reader): it is only needed when a document is added, and starting both at
        once made them compete for the processor and the virus scanners at the moment a person waits (CALE_READER_EARLY=1 starts it first, for a test)."""
        paths.LOGS.mkdir(parents=True, exist_ok=True)
        self.logfile = open(paths.LOGS / "chat.log", "ab")
        self.ready_file.unlink(missing_ok=True)
        self.env_ = self.env()
        if os.environ.get("CALE_READER_EARLY") == "1":
            self.start_reader()
        self._spawn("server")

    def start_reader(self):
        if self.reader_started:
            return
        self.reader_started = True
        self._spawn("collector")

    def _spawn(self, sub):
        pre = ["-r", str(BUNDLE / "preload.js")]
        p = subprocess.Popen([self.node, *pre, "index.js"], cwd=BUNDLE / sub, env=self.env_, stdin=subprocess.DEVNULL,
                             stdout=self.logfile, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        KILL_WITH_ME.add(p)
        self.procs.append(p)

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
        """Wait until the chat server answers. The server writes a file when it listens, and this waits for that file. Asking the port instead is slow on Windows:
        a connection to a port nobody listens on yet takes 2 s to be refused there (6 ms on a Mac), so one early try cost every start 2 s."""
        start = time.time()
        end = start + timeout
        while time.time() < end:
            if not self.alive():
                raise SystemExit("The chat window server stopped while starting. See " + str(paths.LOGS / "chat.log"))
            if not self.ready_file.exists() and time.time() - start < FILE_WAIT_S:
                time.sleep(0.01)
                continue
            try:
                if self.get("/api/ping").get("online"):
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.05)
        raise SystemExit("The chat window server did not start in time. See " + str(paths.LOGS / "chat.log"))

    def ensure_workspace(self):
        """A first workspace, so the person can type at once."""
        try:
            if self.get("/api/workspaces").get("workspaces"):
                return
            made = self.request("/api/workspace/new", {"name": WORKSPACE}, timeout=20)
            slug = (made.get("workspace") or {}).get("slug") or "my-chats"
            self.request(f"/api/workspace/{slug}/update", {"chatMode": "chat"}, timeout=20)     # plain chat: no tool "agent" mode, which small models handle badly
        except (urllib.error.URLError, OSError, ValueError) as e:
            log(f"could not make the first workspace ({e})")


# --- main -------------------------------------------------------------------------------------
def warm():
    """Start the chat servers once and stop them, so Node's compile cache is filled before the first open (the install calls this). Needs no AI."""
    prepare_data()
    servers = Servers(node_path(), OBJECT_NAME, chat_key())
    servers.start()
    try:
        servers.wait_ready()
        servers.start_reader()
        time.sleep(1.5)
    finally:
        servers.stop()
    return 0


def main():
    if "--warm" in sys.argv:
        return warm()
    timeline.mark("chat_start")
    node = node_path()
    parent = os.getppid()
    # Cale starts while the chat servers start: the servers need Cale only when a message is sent (the window never waits for the AI).
    cale_up = []

    def start_cale():
        try:
            client.ensure_running()
        except client.CaleError as e:
            cale_up.append(e)

    starter = threading.Thread(target=start_cale, daemon=True)
    starter.start()
    prepare_data()
    timeline.mark("data_ready")
    key = chat_key()
    holder = Holder(f"http://{paths.HOST}:{paths.PORT}", key)
    # The window must never wait for the AI to load. The chat servers start now, under the name Cale gives a chat object when that name is free.
    # A message sent while the AI still loads waits inside Cale (server.py _wait_pending). If Cale gave another name, the loop below restarts the servers with it.
    timeline.mark("model_object_asked")
    servers = Servers(node, OBJECT_NAME, key)
    done = threading.Event()

    def bye(*_):
        done.set()

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, bye)
    try:
        timeline.mark("servers_start")
        servers.start()
        starter.join()
        if cale_up:
            print(f"Cale is not running and did not start: {cale_up[0]}", file=sys.stderr, flush=True)
            return 1
        timeline.mark("cale_reachable")
        holder.start()
        servers.wait_ready()
        timeline.mark("servers_ready")
        servers.ensure_workspace()
        say_ready(servers.port, servers.token)
        timeline.mark("ready_said")
        servers.start_reader()                               # now: the chat is ready, the document reader can start
        timeline.mark("reader_started")
        logged = False
        while not done.is_set():
            if holder.id and not logged:
                logged = True
                timeline.mark("model_object_ready")
                log(f"chat object {holder.id}: {holder.info}")
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
                servers.start_reader()
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
