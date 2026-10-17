"""Quick Notes: a notes page with ghost-text autocomplete, powered by Cale.

Run:  python app.py        (set CALE_URL if Cale is not at http://127.0.0.1:47821)
Standard library only. The browser talks to this tiny server; this server talks to Cale.
"""
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ["CALE_APP_DATA"]) if os.environ.get("CALE_APP_DATA") else HERE / "data"   # an installed app keeps its files outside its bundle
KEY_FILE = DATA / "cale-key"
NOTES_FILE = DATA / "notes.txt"
PORT = 8821
CALE_URL = os.environ.get("CALE_URL", "http://127.0.0.1:47821").rstrip("/")
APP_NAME = "Quick Notes"
CONTEXT_CHARS = 1500          # how much text before the cursor we send

SYSTEM = "Continue the text with only the next few words. Plain text, no quotes."   # short on purpose: every word of it is read again on each call


class State:
    """Everything the app knows about Cale. One lock guards the key and the model object."""
    lock = threading.RLock()
    key = None
    model = None            # a cale_client.model, held open while the app runs
    error = None            # plain-language problem to show, or None
    current = None          # the Cale connection of the newest ghost request
    current_id = 0


# --- cale-snippet-start (the Developer view shows exactly these lines) ---
def cale_make_model(key):
    # One "model object": goals instead of a fixed size. Cale picks the size per call.
    return cale_client.model(CALE_URL, key, name="autocomplete", max_first_token_s=1, max_tokens=20)


def cale_complete(conn, key, text):
    # Plain OpenAI-style chat call. Closing `conn` early makes Cale stop the work.
    body = json.dumps({"model": "autocomplete", "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]})
    conn.request("POST", "/v1/chat/completions", body, {"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    r = conn.getresponse()
    data = json.loads(r.read())
    return r.status, r.getheader("X-Cale-Size"), r.getheader("X-Cale-Goal-Met"), data
# --- cale-snippet-end ---

import cale_client  # noqa: E402  (a copy of cale/client.py, unchanged)


def cale_request(method, path, body=None, key=None, timeout=130):
    req = urllib.request.Request(CALE_URL + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def cale_up():
    try:
        return cale_request("GET", "/cale/status", timeout=3)[0] == 200
    except (OSError, ValueError):
        return False


def get_key():
    DATA.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
        if k:
            return k
    return ask_for_key()


def ask_for_key():
    print("Asking for access. If a question appears on this computer, choose Allow.")
    status, body = cale_request("POST", "/cale/connect", {"app": APP_NAME})
    if status != 200 or "api_key" not in body:
        raise RuntimeError("Quick Notes was not allowed to start. Close it, open it again, and choose Allow.")
    DATA.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(body["api_key"])
    return body["api_key"]


def setup_cale():
    """Make Cale ready: running, key, model object. Sets State.error to a plain message on failure."""
    with State.lock:
        State.error = None
        try:
            if not cale_up():
                try:
                    cale_client.ensure_running()
                except Exception:
                    pass
                if not cale_up():
                    State.error = "Quick Notes is not ready yet. Please wait a moment."
                    return False
            key = get_key()
            for attempt in range(2):
                try:
                    if State.model:
                        State.model.close()
                    State.model = cale_make_model(key).__enter__()
                    State.key = key
                    return True
                except cale_client.CaleError as e:
                    if e.status == 401 and attempt == 0:
                        KEY_FILE.unlink(missing_ok=True)
                        key = ask_for_key()
                        continue
                    State.error = "Quick Notes could not start. Close it and open it again."
                    return False
        except Exception as e:  # never show a stack trace in the UI
            State.error = "Quick Notes is not ready yet. Please wait a moment."
            return False


def warm_up():
    """Ask for one tiny answer when the app opens, so the model is in memory before the person types."""
    try:
        host = urlparse(CALE_URL)
        conn = http.client.HTTPConnection(host.hostname, host.port, timeout=180)
        try:
            cale_complete(conn, State.key, "Hello")
        finally:
            close_conn(conn)
    except Exception:
        pass


def ghost(text):
    """Ask Cale for a short continuation. Returns a dict for the browser."""
    with State.lock:
        State.current_id += 1
        my_id = State.current_id
        old = State.current
        State.current = None
    if old is not None:
        close_conn(old)            # typing again cancels the old request
    if not State.model and not setup_cale():
        return {"error": State.error}
    t0 = time.perf_counter()
    for attempt in range(2):
        host = urlparse(CALE_URL)
        conn = http.client.HTTPConnection(host.hostname, host.port, timeout=60)
        with State.lock:
            if State.current_id != my_id:
                return {"cancelled": True}
            State.current = conn
        try:
            status, size, goal, data = cale_complete(conn, State.key, text[-CONTEXT_CHARS:])
        except (OSError, http.client.HTTPException, ValueError):
            with State.lock:
                if State.current_id != my_id or State.current is not conn:
                    return {"cancelled": True}
            return {"error": "One moment. Keep typing."}
        finally:
            close_conn(conn)
        if status in (401, 404) and attempt == 0:      # key rejected or model object gone: set up again
            if status == 401:
                KEY_FILE.unlink(missing_ok=True)
            State.model = None
            if not setup_cale():
                return {"error": State.error}
            continue
        break
    ms = round((time.perf_counter() - t0) * 1000)
    if status != 200:
        msg = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
        return {"error": "No suggestion this time. Keep typing."}
    out = (data["choices"][0]["message"]["content"] or "").replace("**", "")
    cale = data.get("cale", {})
    return {"text": out, "ms": ms, "size": size, "goal_met": goal,
            "total_s": cale.get("total_s"), "goals_actual": cale.get("goals_actual")}


def close_conn(conn):
    try:
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        conn.close()
    except OSError:
        pass


def developer_snippet():
    src = Path(__file__).read_text(encoding="utf-8").split("\n")
    a = next(i for i, l in enumerate(src) if l.startswith("# --- cale-snippet-start"))
    b = next(i for i, l in enumerate(src) if l.startswith("# --- cale-snippet-end"))
    return "\n".join(src[a + 1:b])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif p == "/favicon.ico":
            self.send(204, b"", "image/x-icon")
        elif p == "/api/ready":
            self.send(200, {"ready": State.model is not None, "problem": None if State.model or not State.error else State.error})
        elif p == "/api/notes":
            text = NOTES_FILE.read_text(encoding="utf-8") if NOTES_FILE.exists() else ""
            self.send(200, {"text": text, "snippet": developer_snippet(), "cale": CALE_URL})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        body = self.read_json()
        try:
            if p == "/api/notes":
                DATA.mkdir(parents=True, exist_ok=True)
                NOTES_FILE.write_text(body.get("text", ""), encoding="utf-8")
                self.send(200, {"ok": True})
            elif p == "/api/ghost":
                self.send(200, ghost(body.get("text", "")))
            elif p == "/api/cancel":
                with State.lock:
                    State.current_id += 1
                    old, State.current = State.current, None
                if old is not None:
                    close_conn(old)
                self.send(200, {"ok": True})
            else:
                self.send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self.send(200, {"error": "Something went wrong. Keep typing."})


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=lambda: setup_cale() and warm_up(), daemon=True).start()
    window = bool(os.environ.get("CALE_APP_WINDOW"))               # started by the app's own window: free port, no browser
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", 0 if window else PORT), Handler)
    except OSError:
        print("Port %d is in use. Is Quick Notes already running?" % PORT)
        return
    srv.daemon_threads = True
    port = srv.server_address[1]
    url = "http://127.0.0.1:%d/" % port
    if window:
        print("READY %d" % port, flush=True)
        parent = os.getppid()                                       # the window program: if it is gone (even a force quit), stop too

        def watch():
            while os.getppid() == parent:
                time.sleep(2)
            os._exit(0)
        threading.Thread(target=watch, daemon=True).start()
    else:
        print("Quick Notes is at " + url + "  (Cale: " + CALE_URL + ")")
    if not (window or os.environ.get("QN_NO_BROWSER") or os.environ.get("CALE_NO_BROWSER")):
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
