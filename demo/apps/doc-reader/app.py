"""Private Doc Reader: ask questions about a long private document, always with Cale's FULL size (gemma-3n:smart).

Run:  python app.py        (opens http://127.0.0.1:8822)
Cale's address comes from the environment variable CALE_URL (default http://127.0.0.1:47821).
Python standard library only. The browser cannot call Cale (Cale refuses an Origin header), so this tiny server does it.
"""
import http.client
import json
import os
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
KEY_FILE = DATA / "cale-key"
APP_NAME = "Private Doc Reader"
PORT = 8822
CALE_URL = os.environ.get("CALE_URL", "http://127.0.0.1:47821").rstrip("/")
_u = urlparse(CALE_URL)
CALE_HOST, CALE_PORT = _u.hostname or "127.0.0.1", _u.port or 47821
SIZE = "gemma-3n:smart"          # always the full size: this developer never trades quality for speed
ANSWER_TOKENS = 700              # room reserved for each answer
SYSTEM = ("You answer questions about the private document below. Use only the document. "
          "Quote the exact number, period or clause when you can, and name the section. "
          "If the document does not say, answer: \"The document does not say.\"\n\n--- DOCUMENT ---\n")
WARMUP = "Read the document above carefully and wait for my questions. Reply with the single word: Ready."

from cale_client import CaleError, ensure_running, hold as cale_hold, predict as cale_predict   # noqa: E402  (copied, unchanged)


# ---------------------------------------------------------------- talking to Cale
class Friendly(Exception):
    """An error with a plain sentence for the person. `code` is machine-readable."""
    def __init__(self, message, code="error", status=500, extra=None):
        super().__init__(message)
        self.message, self.code, self.status, self.extra = message, code, status, extra or {}


_key_lock = threading.Lock()
_key = {"value": None}


def http_json(method, path, body=None, key=None, timeout=300):
    conn = http.client.HTTPConnection(CALE_HOST, CALE_PORT, timeout=timeout)
    try:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        conn.request(method, path, body=data, headers=headers)
        r = conn.getresponse()
        raw = r.read()
        try:
            return r.status, json.loads(raw or b"{}")
        except ValueError:
            return r.status, {"raw": raw[:300].decode(errors="replace")}
    except (OSError, http.client.HTTPException):
        raise Friendly("Cale is not running on this computer. Start it with the command: cale start --wait", "cale_down", 503)
    finally:
        conn.close()


# CALE-CODE-BEGIN  (the Cale calls this app makes; shown in "Developer view")
def connect():                                   # 1. ask the person to allow this app, keep the key
    st, r = http_json("POST", "/cale/connect", {"app": APP_NAME, "wait_s": 120})
    if st == 200: KEY_FILE.write_text(r["api_key"])
    return st, r

def predict_read(key, prompt_tokens):            # 2. before reading: how long will this document take at full size?
    p = cale_predict(CALE_URL, key, prompt_tokens=prompt_tokens, max_tokens=ANSWER_TOKENS)
    return next(s for s in p["sizes"].values() if s["size_pct"] == 100)["first_token_s"]

HOLD = None

def keep_full_size(key):                         # 3. keep the full size in memory while a document is open: Cale will not shrink it
    global HOLD                                  #    for this app, even when the computer is busy. (Without it, a memory squeeze can
    if HOLD is None or HOLD.closed.is_set():     #    make Cale answer with a smaller size.)
        try:
            HOLD = cale_hold(CALE_URL + "/v1", key, size="smart")
            HOLD.__enter__()
        except CaleError:
            HOLD = None                          # Cale could not promise it: carry on, the footer shows the size that ran

def read_document(key, doc_text):                # 4. read once (1 token out); Cale keeps the reading in its prompt cache
    keep_full_size(key)
    return chat(key, [SYSTEM_MSG(doc_text), WARMUP_MSG], max_tokens=3, stream=False)

def ask(key, history):                           # 5. the history only grows, so Cale reuses the reading
    return chat(key, history, max_tokens=ANSWER_TOKENS, stream=True)   # model = gemma-3n:smart, never smaller
# CALE-CODE-END


def SYSTEM_MSG(text):
    return {"role": "system", "content": SYSTEM + text}


WARMUP_MSG = {"role": "user", "content": WARMUP}


def get_key(force=False):
    with _key_lock:
        if _key["value"] and not force:
            return _key["value"]
        if not force and KEY_FILE.exists():
            _key["value"] = KEY_FILE.read_text().strip()
            if _key["value"]:
                return _key["value"]
        DATA.mkdir(exist_ok=True)
        st, r = connect()
        if st == 200 and r.get("api_key"):
            _key["value"] = r["api_key"]
            return _key["value"]
        if st in (403,):
            raise Friendly("Cale did not allow this app. Answer the Allow question on this computer, then try again.", "denied", 403)
        if st == 202:
            raise Friendly("Cale is waiting for you to allow this app. Run: cale allow \"%s\"" % APP_NAME, "pending", 202)
        if st == 429 or st == 409:
            raise Friendly("Cale is busy asking about another app. Try again in a moment.", "busy", 429)
        raise Friendly("Cale would not give this app a key (%s)." % st, "no_key", 502)


def explain(status, body):
    """Turn a Cale error into one plain sentence. Never show a raw error."""
    err = body.get("error", {}) if isinstance(body, dict) else {}
    code = err.get("code") or ""
    if code == "context_length_exceeded":
        return Friendly("This document is too long for Cale on this computer.", "too_long", 400, {"detail": find_numbers(body)})
    if code == "unsafe_for_this_computer" or status == 507:
        return Friendly("This computer does not have enough free memory right now to read this at full size. Close other apps and try again, or use the first part of the document.", "unsafe", 507)
    if status == 429:
        return Friendly("Cale is busy with another app. Try again in a moment.", "busy", 429)
    if status == 401:
        return Friendly("Cale did not accept this app's key.", "bad_key", 401)
    if status >= 500:
        return Friendly("Cale had a problem and will reload the model. Try again.", "engine", 502)
    return Friendly("Cale could not answer (%s)." % (err.get("message") or status), code or "error", 502)


def find_numbers(obj, out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("prompt_tokens", "max_tokens", "max_context_tokens") and isinstance(v, int):
                out[k] = v
            else:
                find_numbers(v, out)
    return out


def chat(key, messages, max_tokens, stream, holder=None):
    """One chat call. Returns (response, connection) for stream=True, else the parsed JSON. Retries once on 401."""
    for attempt in (0, 1):
        conn = http.client.HTTPConnection(CALE_HOST, CALE_PORT, timeout=900)
        try:
            body = {"model": SIZE, "messages": messages, "max_tokens": max_tokens, "stream": stream}
            conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            if holder is not None:
                holder["conn"] = conn
            r = conn.getresponse()
        except (OSError, http.client.HTTPException):
            conn.close()
            if holder is not None and holder.get("cancelled"):
                raise Friendly("Stopped.", "cancelled", 499)
            raise Friendly("Cale is not running on this computer. Start it with the command: cale start --wait", "cale_down", 503)
        if r.status == 401 and attempt == 0:
            r.read(); conn.close()
            key = get_key(force=True)
            continue
        if r.status != 200:
            raw = r.read(); conn.close()
            try:
                parsed = json.loads(raw or b"{}")
            except ValueError:
                parsed = {}
            raise explain(r.status, parsed)
        if stream:
            return r, conn
        try:
            return json.loads(r.read())
        except (OSError, http.client.HTTPException, ValueError):
            raise Friendly("The reading was stopped before it finished.", "cancelled", 499)
        finally:
            conn.close()


# ---------------------------------------------------------------- the one document in memory
LOCK = threading.Lock()          # one reading or answer at a time
DOC = {"name": None, "text": "", "words": 0, "est_tokens": 0, "real_tokens": None, "truncated": False,
       "read": False, "history": [], "last_prompt_tokens": 0, "predicted_s": None}
_ctx = {"max": None}


def max_context(key=None):
    if _ctx["max"]:
        return _ctx["max"]
    st, r = http_json("GET", "/cale/computer")
    if st == 200 and r.get("max_context_tokens"):
        _ctx["max"] = int(r["max_context_tokens"])
    else:
        _ctx["max"] = 8192            # guess if Cale does not say (it normally does)
    return _ctx["max"]


def est_tokens(text):
    """No tokenizer is public. About 4.2 characters per token on English text; use 3.9 to lean safe."""
    return int(len(text) / 3.9) + 40


def budget_tokens():
    return max_context() - ANSWER_TOKENS - 400      # room for answer, question, history of one turn


def first_part(text, limit_tokens):
    chars = int(limit_tokens * 3.9)
    if len(text) <= chars:
        return text
    cut = text[:chars]
    p = max(cut.rfind("\n\n"), cut.rfind(". "))
    return cut[:p + 1] if p > chars * 0.7 else cut


def load_doc(name, text, use_first_part):
    key = get_key()
    budget = budget_tokens()
    full_est = est_tokens(text)
    too_long = full_est > budget
    if too_long and not use_first_part:
        return {"ok": False, "too_long": True, "words": len(text.split()), "est_tokens": full_est,
                "limit_tokens": budget, "limit_words": int(budget / 1.45)}
    truncated = too_long
    if truncated:
        text = first_part(text, budget)
    est = est_tokens(text)
    try:
        try:
            secs = predict_read(key, est)
        except CaleError as e:
            if e.status == 401:
                key = get_key(force=True)
            elif e.status < 500:
                raise
            secs = predict_read(key, est)             # one retry: a new key, or a hiccup in Cale
    except CaleError as e:
        raise explain(e.status, safe_json(e.body))
    except (OSError, http.client.HTTPException, StopIteration):
        raise Friendly("Cale is not running on this computer. Start it with the command: cale start --wait", "cale_down", 503)
    DOC.update(name=name, text=text, words=len(text.split()), est_tokens=est, real_tokens=None, truncated=truncated,
               read=False, history=[], last_prompt_tokens=0, predicted_s=secs)
    return {"ok": True, "name": name, "words": DOC["words"], "est_tokens": est, "predicted_s": round(secs, 1),
            "truncated": truncated, "limit_tokens": budget}


def safe_json(s):
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return {}


# ---------------------------------------------------------------- HTTP handler
class Handler(BaseHTTPRequestHandler):
    server_version = "DocReader/1"

    def log_message(self, *a):
        pass

    def send_json(self, obj, status=200):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 20_000_000:
            raise Friendly("That file is too big for this app.", "big", 413)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise Friendly("The page sent something this app could not read.", "bad", 400)

    def start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                raw = (HERE / "static" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif path == "/api/status":
                self.send_json(self.status())
            elif path == "/api/sample":
                self.send_json({"name": "lease.md", "text": (HERE / "sample" / "lease.md").read_text(encoding="utf-8"),
                                "questions": SAMPLE_QUESTIONS})
            elif path == "/api/code":
                src = Path(__file__).read_text(encoding="utf-8")
                m = re.search(r"# CALE-CODE-BEGIN[^\n]*\n(.*?)# CALE-CODE-END", src, re.S)
                self.send_json({"code": m.group(1).strip("\n") if m else ""})
            else:
                self.send_json({"error": "not found"}, 404)
        except Friendly as e:
            self.send_json({"error": e.message, "code": e.code}, 200)
        except (BrokenPipeError, ConnectionError):
            pass

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/load":
                b = self.body()
                name = str(b.get("name") or "document")
                if not name.lower().endswith((".txt", ".md", ".markdown", ".text")) and not b.get("pasted"):
                    raise Friendly("This app reads .txt and .md files. Paste the text instead, or save the file as .txt.", "type", 400)
                text = str(b.get("text") or "").replace("\r\n", "\n").strip()
                if len(text) < 20:
                    raise Friendly("The document is empty. Paste some text or choose a .txt or .md file.", "empty", 400)
                self.send_json(load_doc(name, text, bool(b.get("use_first_part"))))
            elif path == "/api/read":
                self.read()
            elif path == "/api/ask":
                self.ask(self.body())
            else:
                self.send_json({"error": "not found"}, 404)
        except Friendly as e:
            try:
                self.send_json({"ok": False, "error": e.message, "code": e.code, **e.extra}, 200)
            except (BrokenPipeError, ConnectionError):
                pass
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception:                                  # never show a stack trace in the page
            try:
                self.send_json({"ok": False, "error": "Something went wrong inside the app. Try again.", "code": "internal"}, 200)
            except (BrokenPipeError, ConnectionError):
                pass

    def status(self):
        out = {"cale": False, "approved": False, "max_context_tokens": None, "message": ""}
        try:
            st, r = http_json("GET", "/cale/status", timeout=5)
            out["cale"] = st == 200
        except Friendly:
            try:
                ensure_running()                      # starts Cale when the cale package is installed alongside
                out["cale"] = True
            except Exception:
                out["message"] = "Cale is not running on this computer. Start it with the command: cale start --wait"
                return out
        try:
            get_key()
            out["approved"] = True
            out["max_context_tokens"] = max_context()
        except Friendly as e:
            out["message"] = e.message
        return out

    # -------- reading: one short call that makes Cale read the whole document and keep it in its prompt cache
    def read(self):
        if not DOC["text"]:
            raise Friendly("Choose a document first.", "nodoc", 400)
        if not LOCK.acquire(blocking=False):
            raise Friendly("Cale is still working on the last request. Wait a moment.", "busy", 429)
        try:
            self.start_sse()
            key = get_key()
            result, holder = {}, {}

            def work():
                try:
                    for _ in range(2):
                        try:
                            result["r"] = read_document(key, DOC["text"])
                            return
                        except Friendly as e:
                            if e.code == "too_long" and not result.get("trimmed"):
                                # the estimate was low: Cale told us the real size. Cut to fit and say so.
                                real = e.extra.get("detail", {}).get("prompt_tokens")
                                ratio = (budget_tokens() / real) if real else 0.8
                                DOC["text"] = first_part(DOC["text"], int(est_tokens(DOC["text"]) * ratio * 0.95))
                                DOC["truncated"] = True
                                DOC["words"] = len(DOC["text"].split())
                                result["trimmed"] = True
                                continue
                            result["err"] = e
                            return
                except Exception:
                    result["err"] = Friendly("Reading failed. Try again.", "internal", 500)

            t0 = time.time()
            th = threading.Thread(target=work, daemon=True)
            th.start()
            while th.is_alive():
                th.join(0.5)
                try:
                    self.sse({"t": round(time.time() - t0, 1), "predicted_s": DOC["predicted_s"]})
                except (BrokenPipeError, ConnectionError, OSError):
                    return                              # the page left; Cale stops reading when its connection closes
            secs = time.time() - t0
            if "err" in result:
                e = result["err"]
                self.sse({"error": e.message, "code": e.code})
                return
            r = result["r"]
            u = r.get("usage", {})
            DOC["real_tokens"] = u.get("prompt_tokens")
            DOC["last_prompt_tokens"] = u.get("prompt_tokens", 0)
            ans = (r["choices"][0]["message"].get("content") or "Ready.").strip() or "Ready."
            DOC["history"] = [SYSTEM_MSG(DOC["text"]), WARMUP_MSG, {"role": "assistant", "content": ans}]
            DOC["read"] = True
            waited = (r.get("cale") or {}).get("waited_s") or 0
            self.sse({"done": True, "seconds": round(secs, 1), "predicted_s": DOC["predicted_s"], "waited_s": waited,
                      "prompt_tokens": DOC["real_tokens"], "truncated": DOC["truncated"], "words": DOC["words"],
                      "size": (r.get("cale") or {}).get("size")})
        except Friendly as e:
            try:
                self.sse({"error": e.message, "code": e.code})
            except (BrokenPipeError, ConnectionError, OSError):
                pass
        finally:
            LOCK.release()

    # -------- asking: stream the answer
    def ask(self, b):
        q = str(b.get("question") or "").strip()
        if not q:
            raise Friendly("Type a question first.", "empty", 400)
        if not DOC["read"]:
            raise Friendly("Read the document first.", "notread", 400)
        if not LOCK.acquire(blocking=False):
            raise Friendly("Cale is still working on the last request. Wait a moment.", "busy", 429)
        conn = None
        try:
            key = get_key()
            hist = DOC["history"]
            # keep the history inside the context limit: drop the oldest answered questions (this costs a re-read)
            dropped = False
            while len(hist) > 3 and (DOC["last_prompt_tokens"] + est_tokens(q) + ANSWER_TOKENS + 200) > max_context():
                del hist[3:5]
                DOC["last_prompt_tokens"] = DOC["real_tokens"] or DOC["est_tokens"]
                dropped = True
            msgs = hist + [{"role": "user", "content": q}]
            sent_before = DOC["last_prompt_tokens"]
            t0 = time.time()
            holder = {}
            r, conn = chat(key, msgs, ANSWER_TOKENS, True, holder)
            self.start_sse()
            size_hdr = r.headers.get("X-Cale-Size")
            t_first, text, usage, cale, finish = None, [], {}, {}, None
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("cale"):
                    cale = ev["cale"]
                for ch in ev.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if piece:
                        if t_first is None:
                            t_first = time.time()
                        text.append(piece)
                        self.sse({"d": piece})
            t_end = time.time()
            answer = "".join(text).strip()
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            first_s = (t_first - t0) if t_first else None
            inferred = False
            if cached is None:                         # the streamed usage has no cached_tokens (see README, friction notes)
                inferred = True
                # a first word within a second or two of a prompt this long can only mean the reading was reused
                fast = first_s is not None and first_s < max(2.0, 0.25 * (DOC["predicted_s"] or 8)) and not dropped
                cached = min(sent_before, pt) if fast else 0
            tps = None
            if t_first and ct > 1 and t_end > t_first:
                tps = (ct - 1) / (t_end - t_first)
            DOC["history"] = msgs + [{"role": "assistant", "content": answer}]
            DOC["last_prompt_tokens"] = pt + ct
            self.sse({"done": True, "size": size_hdr or cale.get("size"), "first_s": None if first_s is None else round(first_s, 2),
                      "tps": None if tps is None else round(tps, 1), "prompt_tokens": pt, "completion_tokens": ct,
                      "reused": cached, "reused_inferred": inferred, "finish": finish, "dropped_history": dropped,
                      "waited_s": cale.get("waited_s"), "total_s": round(t_end - t0, 1)})
        except Friendly as e:
            if conn is None:
                raise
            try:
                self.sse({"error": e.message, "code": e.code})
            except (BrokenPipeError, ConnectionError, OSError):
                pass
        except (BrokenPipeError, ConnectionError, OSError):
            pass                                       # the page left; closing our connection makes Cale stop
        finally:
            if conn is not None:
                conn.close()
            LOCK.release()


SAMPLE_QUESTIONS = [
    "How much is the late fee, and when does it apply?",
    "How much notice must the tenant give before moving out?",
    "What is the pet deposit, and when is it refunded?",
]


def main():
    DATA.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:%d" % PORT
    print("Private Doc Reader is running at " + url)
    print("Cale address: " + CALE_URL)
    try:
        if not os.environ.get("CALE_NO_BROWSER"):
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
