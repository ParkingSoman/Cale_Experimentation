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

        with model(base_url, key, name="answers", deadline_s=30, max_tokens=400, memory_pct=40) as m:
            client.chat.completions.create(model=m.id, messages=[...])     # Cale picks the size per call

    Give either size= (exact: "smart", "fast" or a percent) or goals: deadline_s (with max_tokens) and/or memory_pct.
    allow_disk=True lets a call read missing weights from disk. on_miss="refuse" makes a missed goal an error
    instead of running the smallest size. The object lives while the `with` block runs, or until the app quits.
    """
    path = "/cale/models"

    def __init__(self, base_url, api_key, name=None, size=None, deadline_s=None, max_tokens=None, memory_pct=None,
                 allow_disk=None, on_miss=None):
        super().__init__(base_url, api_key)
        given = {"name": name, "size": size, "deadline_s": deadline_s, "max_tokens": max_tokens,
                 "memory_pct": memory_pct, "allow_disk": allow_disk, "on_miss": on_miss}
        self.spec = {k: v for k, v in given.items() if v is not None}

    def __enter__(self):
        super().__enter__()
        self.id = self.info["id"]
        return self


def ensure_running(wait=60):
    """Start Cale if it is not running (no service needed). Raises CaleError if it does not come up."""
    from . import service
    ok, msg = service.start(wait)
    if not ok:
        raise CaleError(0, msg)
    return msg
