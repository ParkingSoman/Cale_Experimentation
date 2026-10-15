"""Background Sorter: sorts a folder of .txt notes into one table, quietly, using Cale.
Standard library only. Run:  python app.py   (CALE_URL sets Cale's address)."""
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import date as _date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cale_client import CaleError, ensure_running, model

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PORT = 8823
CALE_URL = os.environ.get("CALE_URL", "http://127.0.0.1:47821").rstrip("/")
APP_NAME = "Background Sorter"
KINDS = ["receipt", "meeting_note", "idea", "todo", "email", "other"]
MAX_CHARS = 4000          # a file is cut to this many characters before it is sent
GAP_S = 0.5               # a short rest between files, so the computer stays free

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": KINDS},
        "date": {"type": ["string", "null"], "maxLength": 10},
        "title": {"type": "string", "maxLength": 80},
        "amount": {"type": ["number", "null"]},
        "summary": {"type": "string", "maxLength": 240},
    },
    "required": ["kind", "date", "title", "amount", "summary"],
    "additionalProperties": False,
}
RULES = ("You sort a person's notes. Answer with JSON only. kind: receipt (a purchase, bill or order, with a total), "
         "meeting_note, idea, todo (task lists and shopping lists), email (has From/To/Subject), other (empty or unclear). "
         "date: the date written in the text as YYYY-MM-DD, or null. title: at most 8 words, in English. "
         "amount: the total paid or due as a number for receipts, else null. summary: one sentence, in English.")


# --- CALE START (the Cale code of this app; the Developer view shows these lines) ---
def post_json(path, body, key=None, timeout=600):
    req = urllib.request.Request(CALE_URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), r.headers

def open_sorter(key):   # a model object: gentle goals; Cale picks the size for every call
    return model(CALE_URL, key, name="sorter", power="light", memory_pct=28, max_tokens=220,
                 on_miss="best_effort", priority=["memory", "quality", "first_token", "speed"])

def ask_cale(key, model_id, text):   # the reply must fit SCHEMA
    rf = {"type": "json_schema", "json_schema": {"name": "note", "schema": SCHEMA}}
    msgs = [{"role": "system", "content": RULES}, {"role": "user", "content": text}]
    body, headers = post_json("/v1/chat/completions", {"model": model_id, "messages": msgs, "response_format": rf}, key)
    return body, headers.get("X-Cale-Size"), headers.get("X-Cale-Goal-Met")
# --- CALE END ---


def developer_view():
    src = Path(__file__).read_text(encoding="utf-8")
    m = re.search(r"# --- CALE START.*?\n(.*?)# --- CALE END ---", src, re.S)
    return m.group(1).strip("\n") if m else ""


def say_error(e):
    """(code, plain message) for an error from Cale. Never a stack trace."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            err = json.loads(e.read()).get("error", {})
        except Exception:
            err = {}
        return e.code, (err.get("code") or ""), (err.get("message") or "")
    return 0, "", str(e)


class Sorter:
    def __init__(self):
        self.lock = threading.RLock()
        self.paused = False
        self.folder = str((HERE / "sample").resolve())
        self.rows = {}                    # full path -> row
        self.status, self.message = "starting", "Starting"
        self.version = 0
        self.key = None
        self.mobj = None
        self.active_s = 0.0
        self.done_session = 0
        self.last_size = None
        self.goal_calls = 0
        self.goal_met_calls = 0
        self.last_memory = None
        self.current = None
        self.hash_cache = {}
        self.fail_counts = {}
        self.stop = threading.Event()
        self.load()

    # ---- files on disk ----
    def load(self):
        try:
            self.rows = json.loads((DATA / "sorted.json").read_text(encoding="utf-8")).get("rows", {})
        except Exception:
            self.rows = {}

    def save(self):
        DATA.mkdir(exist_ok=True)
        with self.lock:
            rows = dict(self.rows)
            folder = self.folder
        payload = json.dumps({"version": 1, "rows": rows}, indent=1, ensure_ascii=False)
        tmp = DATA / "sorted.json.tmp"
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, DATA / "sorted.json")
        cols = ["file", "kind", "date", "title", "amount", "summary", "seconds", "size", "error"]
        tmp = DATA / "sorted.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.table(folder, rows):
                w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
        os.replace(tmp, DATA / "sorted.csv")

    def table(self, folder, rows=None):
        rows = self.rows if rows is None else rows
        out = [dict(r, path=p) for p, r in rows.items() if r.get("folder") == folder]
        return sorted(out, key=lambda r: r["file"].lower())

    def scan(self):
        """All .txt files of the folder: [(path, hash)]. Hash is cached by size and time."""
        out = []
        try:
            files = sorted(p for p in Path(self.folder).iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        except OSError:
            return None
        for p in files:
            try:
                st = p.stat()
                k = (str(p), st.st_mtime_ns, st.st_size)
                h = self.hash_cache.get(k)
                if h is None:
                    h = hashlib.sha1(p.read_bytes()).hexdigest()
                    self.hash_cache[k] = h
                out.append((p, h))
            except OSError:
                continue
        return out

    def pending(self):
        files = self.scan()
        if files is None:
            return None
        todo = []
        for p, h in files:
            r = self.rows.get(str(p))
            if r and r.get("hash") == h and r.get("folder") == self.folder:
                continue
            if self.fail_counts.get((str(p), h), 0) >= 3:
                continue
            todo.append((p, h))
        return todo

    # ---- state for the page ----
    def set(self, status, message):
        with self.lock:
            if (self.status, self.message) != (status, message):
                self.status, self.message = status, message
                self.version += 1

    def snapshot(self):
        with self.lock:
            rows = self.table(self.folder)
            files = self.scan()
            total = len(files) if files else 0
            done = sum(1 for r in rows if not r.get("error"))
            fpm = (self.done_session / self.active_s * 60) if self.active_s > 5 and self.done_session else None
            return {
                "status": self.status, "message": self.message, "paused": self.paused, "folder": self.folder,
                "version": self.version, "rows": rows, "total": total, "done": done,
                "failed": sum(1 for r in rows if r.get("error")),
                "files_per_min": round(fpm, 1) if fpm else None, "size": self.last_size,
                "goal_calls": self.goal_calls, "goal_met_calls": self.goal_met_calls, "memory_pct_actual": self.last_memory,
                "current": self.current, "gentle": "Running gently: low power mode",
                "cale_url": CALE_URL,
            }

    # ---- talking to Cale ----
    def get_key(self):
        kf = DATA / "cale-key"
        if self.key:
            return self.key
        try:
            k = kf.read_text(encoding="utf-8").strip()
            if k:
                self.key = k
                return k
        except OSError:
            pass
        DATA.mkdir(exist_ok=True)
        body, _ = post_json("/cale/connect", {"app": APP_NAME, "wait_s": 0}, timeout=30)
        ticket = None
        while True:
            if body.get("api_key"):
                kf.write_text(body["api_key"], encoding="utf-8")
                self.key = body["api_key"]
                return self.key
            ticket = body.get("ticket", ticket)
            self.set("waiting", "Waiting for you to allow Background Sorter in Cale. " + (body.get("message") or ""))
            if self.stop.wait(2):
                raise SystemExit
            try:
                with urllib.request.urlopen(CALE_URL + "/cale/connect/status?ticket=" + (ticket or ""), timeout=30) as r:
                    body = json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    self.set("waiting", "Cale says no. Allow Background Sorter in Cale, then press Resume.")
                    self.paused = True
                    raise SystemExit
                raise

    def forget_key(self):
        self.key = None
        self.close_model()
        try:
            (DATA / "cale-key").unlink()
        except OSError:
            pass

    def close_model(self):
        if self.mobj is not None:
            try:
                self.mobj.close()
            except Exception:
                pass
            self.mobj = None

    def ready(self):
        key = self.get_key()
        if self.mobj is not None and self.mobj.closed.is_set():
            self.mobj = None
        if self.mobj is None:
            m = open_sorter(key)
            m.__enter__()
            self.mobj = m
        return key, self.mobj.id

    def wait_msg(self, status, msg, secs):
        self.set(status, msg)
        end = time.time() + secs
        while time.time() < end and not self.stop.is_set():
            if self.paused:
                break
            time.sleep(0.25)

    def handle_error(self, e, attempt):
        """Turn any failure into a plain waiting message. Returns 'retry' or 'skip' (this file cannot be sorted)."""
        wait = min(10 * (2 ** min(attempt, 4)), 120)
        if isinstance(e, CaleError):
            code, ecode, msg = e.status, "", ""
            try:
                err = json.loads(e.body).get("error", {})
                ecode, msg = err.get("code") or "", err.get("message") or ""
            except Exception:
                msg = str(e.body)[:120]
        else:
            code, ecode, msg = say_error(e)
        if code == 401:
            self.forget_key()
            self.set("waiting", "Cale did not accept the key. Asking for a new one.")
            return "retry"
        if code == 404:
            self.close_model()
            return "retry"
        if code == 400 and ecode == "context_length_exceeded":
            return "shorter"
        if code in (400, 413):
            return "skip"
        if code == 507 or ecode == "unsafe_for_this_computer":
            self.wait_msg("waiting", f"This computer is low on memory. Waiting {wait} s, then trying again.", wait)
        elif code in (422, 429):
            self.wait_msg("waiting", f"Cale is busy or cannot meet the gentle goals now. Waiting {wait} s, then trying again.", wait)
        elif code == 500:
            self.wait_msg("waiting", f"Cale had a problem. Waiting {wait} s, then trying again.", wait)
        else:
            # no answer at all: Cale is probably not running
            self.set("waiting", "Cale is not answering. Trying to start it.")
            try:
                ensure_running()
            except Exception:
                pass
            self.close_model()
            self.wait_msg("waiting", f"Cale is not running here ({CALE_URL}). Waiting {wait} s, then trying again.", wait)
        return "retry"

    # ---- the worker ----
    def validate(self, o):
        if not isinstance(o, dict) or set(o) != {"kind", "date", "title", "amount", "summary"}:
            raise ValueError("wrong keys")
        if o["kind"] not in KINDS:
            raise ValueError("bad kind")
        if not isinstance(o["title"], str) or not isinstance(o["summary"], str):
            raise ValueError("bad text")
        a = o["amount"]
        if a is not None and (isinstance(a, bool) or not isinstance(a, (int, float))):
            raise ValueError("bad amount")
        d = o["date"]
        if d is not None:
            try:
                if not (isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)):
                    raise ValueError
                _date.fromisoformat(d)
            except ValueError:
                o["date"] = None          # the schema cannot force a real date; keep the table clean
        return o

    def sort_one(self, p, h):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        text = text[:MAX_CHARS]
        row_base = {"file": p.name, "folder": self.folder, "hash": h}
        if not text.strip():
            self.finish(p, dict(row_base, kind="other", date=None, title="(empty file)", amount=None,
                                summary="The file is empty.", seconds=0.0, size=None, error=None))
            return
        attempt = 0
        while not self.stop.is_set():
            if self.paused:
                return
            t0 = time.time()
            try:
                self.set("running", "Sorting " + p.name)
                key, mid = self.ready()
                body, size, goal = ask_cale(key, mid, text)
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("answer cut short")
                obj = self.validate(json.loads(choice["message"]["content"]))
                secs = round(time.time() - t0, 2)
                cale = body.get("cale") or {}
                mem = (cale.get("goals_actual") or {}).get("memory") or {}
                with self.lock:
                    self.last_size = size or self.last_size
                    if goal is not None:
                        self.goal_calls += 1
                        self.goal_met_calls += 1 if str(goal).lower() == "true" else 0
                    if mem.get("actual_pct") is not None:
                        self.last_memory = mem["actual_pct"]
                self.finish(p, dict(row_base, **obj, seconds=secs, size=size, error=None))
                self.done_session += 1
                return
            except (ValueError, KeyError, IndexError, TypeError) as e:
                self.fail_counts[(str(p), h)] = self.fail_counts.get((str(p), h), 0) + 1
                if self.fail_counts[(str(p), h)] >= 3:
                    self.finish(p, dict(row_base, kind="other", date=None, title=p.stem, amount=None, summary="",
                                        seconds=None, size=None, error="Cale's answer could not be used (" + str(e)[:60] + ")"))
                return
            except SystemExit:
                return
            except Exception as e:
                r = self.handle_error(e, attempt)
                attempt += 1
                if r == "skip":
                    self.fail_counts[(str(p), h)] = 3
                    self.finish(p, dict(row_base, kind="other", date=None, title=p.stem, amount=None, summary="",
                                        seconds=None, size=None, error="Cale refused this file: " + say_error(e)[2][:80]))
                    return
                if r == "shorter":
                    text = text[: max(200, len(text) // 2)]

    def finish(self, p, row):
        with self.lock:
            self.rows[str(p)] = row
            self.version += 1
        self.save()

    def run(self):
        last = time.time()
        while not self.stop.is_set():
            now = time.time()
            busy = self.status in ("running", "waiting")
            if busy and not self.paused:
                self.active_s += now - last
            last = now
            if self.paused:
                self.set("paused", "Paused. Press Resume to continue.")
                time.sleep(0.3)
                continue
            todo = self.pending()
            if todo is None:
                self.set("idle", "This folder cannot be read. Check the path.")
                time.sleep(1)
                continue
            if not todo:
                self.current = None
                self.set("done", "All files are sorted. Watching the folder for changes.")
                time.sleep(1)
                continue
            p, h = todo[0]
            self.current = p.name
            self.sort_one(p, h)
            self.current = None
            time.sleep(GAP_S)

    # ---- controls ----
    def set_paused(self, v):
        with self.lock:
            self.paused = v
            self.version += 1

    def set_folder(self, path):
        p = Path(path.strip().strip('"')).expanduser()
        if not p.is_dir():
            return False
        with self.lock:
            self.folder = str(p.resolve())
            self.version += 1
        self.save()
        return True


SORTER = Sorter()


class Handler(BaseHTTPRequestHandler):
    server_version = "BackgroundSorter"

    def log_message(self, *a):
        pass

    def send(self, code, data, ctype="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = (data if isinstance(data, str) else json.dumps(data)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def host_ok(self):
        return self.headers.get("Host", "") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def do_GET(self):
        if not self.host_ok():
            return self.send(403, {"error": "bad host"})
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self.send(200, SORTER.snapshot())
        if path == "/api/code":
            return self.send(200, {"code": developer_view()})
        if path == "/data/sorted.csv" or path == "/data/sorted.json":
            f = DATA / path.split("/")[-1]
            if f.exists():
                return self.send(200, f.read_bytes(), "text/csv; charset=utf-8" if f.suffix == ".csv" else "application/json")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.host_ok():
            return self.send(403, {"error": "bad host"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if self.path == "/api/pause":
            SORTER.set_paused(True)
        elif self.path == "/api/resume":
            SORTER.set_paused(False)
        elif self.path == "/api/folder":
            if not SORTER.set_folder(str(body.get("path", ""))):
                return self.send(400, {"error": "That folder does not exist."})
        else:
            return self.send(404, {"error": "not found"})
        self.send(200, {"ok": True})


def main():
    DATA.mkdir(exist_ok=True)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"Port {PORT} is in use. Is Background Sorter already running? Open http://127.0.0.1:{PORT}")
        sys.exit(1)
    httpd.daemon_threads = True
    threading.Thread(target=SORTER.run, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"Background Sorter is at {url}  (Cale: {CALE_URL})  Press Ctrl+C to stop.")
    if not (os.environ.get("NO_BROWSER") or os.environ.get("CALE_NO_BROWSER")):
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SORTER.stop.set()
        SORTER.close_model()


if __name__ == "__main__":
    main()
