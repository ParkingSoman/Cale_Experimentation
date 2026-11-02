"""A small helper for Python apps: keep a size of the model in RAM for as long as a `with` block runs.

    from cale.client import hold
    with hold(base_url, api_key, size="smart") as h:        # or memory_pct=25
        print(h.info)                                       # {"size_pct": 100, "ram_gb": 4.2, ...}
        ...                                                 # chat calls here find the weights in RAM

The hold is one open connection. Closing it (leaving the block, or the app dying) ends the hold.
It uses a plain socket on purpose: http.client hands the socket to the response and cannot close it early.
"""
import json
import socket
import urllib.error
import threading
from urllib.parse import urlparse


class CaleError(Exception):
    def __init__(self, status, body):
        super().__init__(f"Cale refused ({status}): {body}")
        self.status, self.body = status, body


class hold:
    path = "/cale/load"

    def __init__(self, base_url, api_key, size=None, memory_pct=None, on_miss="smallest"):
        u = urlparse(base_url)
        self.host, self.port = u.hostname, u.port
        self.key = api_key
        self.spec = {"size": size} if size is not None else {"memory_pct": memory_pct, "on_miss": on_miss}
        self.sock = self.file = None
        self.info = None
        self.closed = threading.Event()        # set when the hold is over, whoever ended it

    def __enter__(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=600)
        body = json.dumps(self.spec).encode()
        head = (f"POST {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\nAuthorization: Bearer {self.key}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n")
        self.sock.sendall(head.encode() + body)
        self.file = self.sock.makefile("rb")
        status = int(self.file.readline().split()[1])
        while self.file.readline().strip():            # skip the headers
            pass
        if status != 200:
            text = self.file.read().decode(errors="replace")
            self.close()
            raise CaleError(status, text)
        self.info = json.loads(self.file.readline())
        threading.Thread(target=self._drain, daemon=True).start()      # read the heartbeats so they never pile up
        return self

    def _drain(self):
        try:
            while self.file.readline():
                pass
        except (OSError, ValueError):
            pass
        self.closed.set()

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None


class model(hold):
    """A model object: parameters for the model an app wants, instead of a fixed size.

        with model(base_url, key, name="answers", min_tokens_per_s=15, max_first_token_s=3, memory_pct=40) as m:
            client.chat.completions.create(model=m.id, messages=[...])     # Cale picks the size per call
            print(m.info["prediction"])                                    # what Cale predicts for these goals on this computer

    Give either size= (exact: "smart", "fast" or a percent) or goals (see cale/goals.py):
      min_tokens_per_s, max_first_token_s, memory_pct, min_quality (0 to 1), power ("normal" or "light").
    priority=["memory", "quality", "first_token", "speed"] says which goal to keep when they conflict (most important first).
    on_miss says what to do when no size meets the goals: "best_effort" (default), "refuse", "wait" (up to max_wait_s) or "try".
    allow_disk=True lets a call read missing weights from disk. max_tokens is the length limit of the object's answers.
    The object lives while the `with` block runs, or until the app quits.
    """
    path = "/cale/models"

    def __init__(self, base_url, api_key, name=None, size=None, max_tokens=None, **goals):
        super().__init__(base_url, api_key)
        given = {"name": name, "size": size, "max_tokens": max_tokens, **goals}
        self.spec = {k: v for k, v in given.items() if v is not None}

    def __enter__(self):
        super().__enter__()
        self.id = self.info["id"]
        return self


def predict(base_url, api_key, prompt_tokens=200, max_tokens=256, **goals):
    """Ask before you run: can this computer meet these goals, which size would Cale use, and what would every size do?
    Nothing runs. Returns the JSON reply of POST /cale/predict (see cale/server.py)."""
    import json
    import urllib.request
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    body = json.dumps({"prompt_tokens": prompt_tokens, "max_tokens": max_tokens, **goals}).encode()
    req = urllib.request.Request(root + "/cale/predict", data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise CaleError(e.code, e.read().decode(errors="replace"))


def ensure_running(wait=60):
    """Start Cale if it is not running (no service needed). Raises CaleError if it does not come up.
    Works when this file is copied into an app on its own: it then runs the `cale` command."""
    try:
        from . import service
    except ImportError:                                  # a standalone copy of this file
        return _start_with_command(wait)
    ok, msg = service.start(wait)
    if not ok:
        raise CaleError(0, msg)
    return msg


def _start_with_command(wait):
    import os
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request
    base = os.environ.get("CALE_URL", "http://127.0.0.1:47821").rstrip("/")

    def up():
        import socket
        from urllib.parse import urlparse
        u = urlparse(base)
        try:                                              # Windows takes 2 s to refuse a connection to a port nobody listens on: ask with a short wait first
            socket.create_connection((u.hostname, u.port or 80), timeout=0.25).close()
        except OSError:
            return False
        try:
            with urllib.request.urlopen(base + "/cale/status", timeout=3):
                return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    if up():
        return "Cale is running."
    exe = shutil.which("cale")
    if exe is None:                                       # the installer's own folders, for computers where `cale` is not on PATH
        home = os.environ.get("CALE_HOME")
        roots = [home] if home else {"darwin": ["~/Library/Application Support/Cale"], "win32": [os.path.expandvars("%LOCALAPPDATA%\\Cale")]}.get(
            sys.platform, ["~/.local/share/cale"])
        for r in roots:
            for name in ("bin/cale", "cale.cmd", "bin/cale.cmd"):
                cand = os.path.join(os.path.expanduser(r), name)
                if os.path.exists(cand):
                    exe = cand
                    break
    if exe is None:
        raise CaleError(0, "Cale is not installed on this computer, or the `cale` command cannot be found.")
    subprocess.run([exe, "start"], capture_output=True, timeout=wait + 10)
    end = time.time() + wait
    while time.time() < end:
        if up():
            return "Cale is running."
        time.sleep(0.1)
    raise CaleError(0, "Cale did not start in time.")
