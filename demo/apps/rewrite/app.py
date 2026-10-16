"""Rewrite: paste a sentence, pick a tone, watch the new version appear. Powered by Cale.

Run:  python app.py        (set CALE_URL if Cale is not at http://127.0.0.1:47821)
Standard library only. The browser talks to this tiny server; this server talks to Cale.
"""
import http.client
import json
import os
import socket
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ["CALE_APP_DATA"]) if os.environ.get("CALE_APP_DATA") else HERE / "data"   # an installed app keeps its files outside its bundle
KEY_FILE = DATA / "cale-key"
PORT = 8825
CALE_URL = os.environ.get("CALE_URL", "http://127.0.0.1:47821").rstrip("/")
APP_NAME = "Rewrite"

TONES = {
    "friendlier": "Rewrite the text so it sounds warmer and friendlier. Keep the meaning.",
    "shorter": "Rewrite the text so it is shorter. Keep the meaning.",
    "clearer": "Rewrite the text so it is clearer and simpler. Keep the meaning.",
    "formal": "Rewrite the text so it sounds polite and professional. Keep the meaning.",
}
SUFFIX = " Reply with ONLY the rewritten text. No quotes, no explanation."


class State:
    lock = threading.RLock()
    key = None
    model = None
    error = None


# --- cale-snippet-start (the Developer view shows exactly these lines) ---
def cale_make_model(key):
    # A model object: a feel instead of a fixed size. Cale picks the size on each computer.
    return cale_client.model(CALE_URL, key, name="rewriter", min_tokens_per_s=6, max_tokens=120)


def cale_stream(conn, key, tone, text):
    # Plain OpenAI-style streamed chat call. Closing `conn` early makes Cale stop the work.
    body = json.dumps({"model": "rewriter", "stream": True, "temperature": 0.4, "messages": [
        {"role": "system", "content": TONES[tone] + SUFFIX}, {"role": "user", "content": text}]})
    conn.request("POST", "/v1/chat/completions", body, {"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    return conn.getresponse()
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


def ask_for_key():
    status, body = cale_request("POST", "/cale/connect", {"app": APP_NAME})
    if status != 200 or "api_key" not in body:
        raise RuntimeError("not allowed")
    DATA.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(body["api_key"])
    return body["api_key"]


def get_key():
    if KEY_FILE.exists() and KEY_FILE.read_text().strip():
        return KEY_FILE.read_text().strip()
    return ask_for_key()


def setup_cale():
    with State.lock:
        State.error = None
        try:
            if not cale_up():
                try:
                    cale_client.ensure_running()
                except Exception:
                    pass
                if not cale_up():
                    State.error = "Rewrite is not ready yet. Please wait a moment."
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
                    State.error = "Rewrite could not start. Close it and open it again."
                    return False
        except Exception:
            State.error = "Rewrite was not allowed to start. Close it, open it again, and choose Allow."
            return False


def warm_up():
    """Ask for one tiny answer when the app opens, so the model is in memory before the person clicks."""
    try:
        host = urlparse(CALE_URL)
        conn = http.client.HTTPConnection(host.hostname, host.port, timeout=180)
        try:
            cale_stream(conn, State.key, "shorter", "Hello there.").read()
        finally:
            conn.close()
    except Exception:
        pass


def developer_snippet():
    src = Path(__file__).read_text(encoding="utf-8").split("\n")
    a = next(i for i, l in enumerate(src) if l.startswith("# --- cale-snippet-start"))
    b = next(i for i, l in enumerate(src) if l.startswith("# --- cale-snippet-end"))
    return "\n".join(src[a + 1:b])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"           # the stream ends when the connection closes

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

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif p == "/favicon.ico":
            self.send(204, b"", "image/x-icon")
        elif p == "/api/ready":
            self.send(200, {"ready": State.model is not None, "problem": None if State.model or not State.error else State.error,
                            "snippet": developer_snippet()})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/rewrite":
            return self.send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        text, tone = (body.get("text") or "").strip()[:2000], body.get("tone", "friendlier")
        if not text or tone not in TONES:
            return self.send(200, {"error": "Type or paste a sentence first."})
        if not State.model and not setup_cale():
            return self.send(200, {"error": State.error})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        host = urlparse(CALE_URL)
        conn = http.client.HTTPConnection(host.hostname, host.port, timeout=120)

        def out(obj):
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()
        try:
            r = cale_stream(conn, State.key, tone, text)
            if r.status in (401, 404):                       # key rejected or model object gone: ask the next click to set up again
                if r.status == 401:
                    KEY_FILE.unlink(missing_ok=True)
                State.model = None
                return out({"error": "One moment. Press the button again."})
            if r.status != 200:
                return out({"error": "No rewrite this time. Try again."})
            size = r.getheader("X-Cale-Size")
            buf = b""
            while True:
                chunk = r.readline()
                if not chunk:
                    break
                line = chunk.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                ev = json.loads(payload)
                piece = (ev.get("choices") or [{}])[0].get("delta", {}).get("content")
                if piece:
                    out({"t": piece})
                if ev.get("cale"):
                    out({"done": True, "size": size, "cale": ev["cale"]})
            out({"end": True})
        except (BrokenPipeError, ConnectionResetError):
            pass                                              # the browser left: closing conn below stops Cale
        except (OSError, http.client.HTTPException, ValueError):
            try:
                out({"error": "One moment. Press the button again."})
            except OSError:
                pass
        finally:
            try:
                conn.sock and conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=lambda: setup_cale() and warm_up(), daemon=True).start()
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print("Port %d is in use. Is Rewrite already running?" % PORT)
        return
    srv.daemon_threads = True
    url = "http://127.0.0.1:%d/" % PORT
    print("Rewrite is at " + url)
    if not (os.environ.get("QN_NO_BROWSER") or os.environ.get("CALE_NO_BROWSER")):
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
