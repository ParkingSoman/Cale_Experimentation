"""Cale local API: an OpenAI-style server on 127.0.0.1 that runs one Gemma 3n model at any of its trained sizes.

| Endpoint | Key needed | Job |
|---|---|---|
| GET  /cale/status | no  | is Cale running, is the model loaded, memory pressure |
| POST /cale/connect | no  | an app asks for a key. The computer shows an Allow dialog. |
| GET  /v1/models | yes | the two named sizes (smart, fast), the size range, the measured quality curve |
| POST /v1/chat/completions | yes | chat. "model": "gemma-3n:smart", "gemma-3n:fast", "gemma-3n:<percent>" or "gemma-3n:auto". stream supported. |
| POST /cale/load | yes | keep a size in RAM until this connection closes. {"size": ...} or {"memory_pct": ...} |
| POST /cale/models | yes | make a model object: {"name", "size", or goals: "min_tokens_per_s", "max_first_token_s", "memory_pct", "min_quality", "power", plus "priority", "on_miss", "max_wait_s", "allow_disk", "max_tokens"}. Reply stays open. Use the returned id as the model name. |
| POST /cale/predict | yes | ask before you run: the same goal fields, plus "prompt_tokens" and "max_tokens". Says whether this computer can meet them, which size Cale would use, and what every size would do. Runs nothing. |
| POST /cale/unload | yes | end a hold early. {"hold": id} |

"gemma-3n:auto" lets Cale choose the size. Goal fields: "cale_min_tokens_per_s", "cale_max_first_token_s", "cale_memory_pct", "cale_min_quality", "cale_power", "cale_priority", "cale_on_miss", "cale_max_wait_s", "cale_allow_disk".

No browser access on purpose: there are no CORS headers, requests whose Host is not 127.0.0.1
are refused, and every route refuses a request that carries an Origin header (so web pages cannot use it).
Debug switches (CALE_DEBUG=1 only): CALE_AUTO_APPROVE=1 skips the dialog, CALE_IDLE_S=n sets the idle time,
POST /cale/debug/pressure forces a memory pressure level.
"""
import json
import math
import os
import secrets
import select
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nest.cancel import Cancel, Cancelled
from nest.levels3n import MAX_CONTEXT, load_levels, pct_range, pick

from . import approval, backends, keys, oslayer, paths, pressure
from .engine import ContextTooLong, Engine, EngineError, GoalMissed, Unsafe, Unsupported
from . import choose, constrain, goals as goals_mod, tools as tools_mod

BASE = "gemma-3n"
LEVELS = load_levels()
MIN_PCT, MAX_PCT = pct_range(LEVELS)
ALIASES = {"smart": MAX_PCT, "fast": MIN_PCT}
MAX_TOKENS = MAX_CONTEXT              # the context decides what really fits (the engine checks it)
DEBUG = os.environ.get("CALE_DEBUG") == "1"
CONNECT_LOCK = threading.Lock()      # one Allow dialog at a time
CONNECT_LOCK_META = threading.Lock()  # guards the table of waiting requests
OBJECTS = {}                         # model objects: (app, id) -> {"spec", "created"}. See _model_object.
OBJ_LOCK = threading.Lock()
OBJECT_FIELDS = {"family", "name", "size", "max_tokens"} | set(goals_mod.FIELDS)


class ApiError(Exception):
    def __init__(self, status, code, message, extra=None, param=None):
        super().__init__(message)
        self.status, self.code, self.extra, self.param = status, code, extra, param


def parse_model(name):
    base, _, size = str(name).partition(":")
    if base != BASE:
        raise ApiError(404, "model_not_found", f"Unknown model '{name}'. Use '{BASE}:smart', '{BASE}:fast', '{BASE}:auto' or '{BASE}:<{MIN_PCT}-{MAX_PCT}>'.")
    if not size:
        return MAX_PCT
    if size == "auto":
        return "auto"
    if size in ALIASES:
        return ALIASES[size]
    try:
        pct = float(size)
        if not math.isfinite(pct):
            raise ValueError
        return int(min(max(round(pct), MIN_PCT), MAX_PCT))
    except (ValueError, OverflowError):
        raise ApiError(404, "model_not_found", f"Unknown model '{name}'. Use '{BASE}:smart', '{BASE}:fast', '{BASE}:auto' or '{BASE}:<{MIN_PCT}-{MAX_PCT}>'.")


def model_id(pct):
    return BASE if pct >= 100 else f"{BASE}:{pct}"


def clean_messages(msgs):
    if not isinstance(msgs, list) or not msgs:
        raise ApiError(400, "invalid_messages", "'messages' must be a non-empty list.")
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            raise ApiError(400, "invalid_messages", "Each message must be an object.")
        role, c = m.get("role"), m.get("content")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant"):
            raise ApiError(400, "invalid_messages", f"Role '{role}' is not supported.")
        if isinstance(c, list):
            if any(p.get("type") != "text" for p in c):
                raise ApiError(400, "unsupported_content", "Only text content is supported for now.")
            c = "\n".join(p.get("text", "") for p in c)
        if not isinstance(c, str):
            raise ApiError(400, "invalid_messages", "Message content must be text.")
        out.append({"role": role, "content": c})
    return out


def lp_content(entries):
    """Engine log-probability entries -> the OpenAI `logprobs.content` list."""
    def piece(tid):
        raw = ENGINE.piece_bytes(tid)
        return {"token": raw.decode("utf-8", "replace"), "bytes": list(raw)}
    out = []
    for e in entries:
        out.append({**piece(e["id"]), "logprob": e["logprob"],
                    "top_logprobs": [{**piece(t), "logprob": lp} for t, lp in e["top"]]})
    return out


def number(req, key, lo, hi, default, integer=False):
    v = req.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ApiError(400, f"invalid_{key}", f"'{key}' must be a number from {lo} to {hi}.", param=key)
    return int(v) if integer else float(v)


def flag(req, key, default=False):
    v = req.get(key, default)
    if not isinstance(v, bool):
        raise ApiError(400, f"invalid_{key}", f"'{key}' must be true or false.", param=key)
    return v


def goals_from(req, prefix=""):
    """Goals from a request body, checked. Raises ApiError."""
    try:
        return goals_mod.parse(req, prefix)
    except goals_mod.GoalError as e:
        raise ApiError(400, e.code, str(e), param=e.param)


# What /v1/chat/completions accepts. Anything else gets a clear 400. Cale never takes a field and ignores it in silence.
CHAT_FIELDS = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "seed", "stop", "stream",
               "tools", "tool_choice", "parallel_tool_calls", "response_format", "logprobs", "top_logprobs"}
CALE_FIELDS = {"cale_" + f for f in goals_mod.FIELDS} | {"cale_deadline_s"}    # Cale's own goal fields (with model gemma-3n:auto). deadline_s is refused with its own message.
NO_EFFECT_FIELDS = {"user", "stream_options", "store", "metadata"}      # accepted: they change nothing Cale does
ZERO_ONLY_FIELDS = {"frequency_penalty", "presence_penalty"}            # accepted only as 0
NOT_YET_FIELDS = {"functions", "function_call", "logit_bias", "modalities", "audio", "prediction", "reasoning_effort"}


def check_chat_fields(req):
    """Reject fields Cale does not support. A field set to null counts as absent."""
    for k, v in req.items():
        if v is None or k in CHAT_FIELDS or k in NO_EFFECT_FIELDS or k in CALE_FIELDS:
            continue
        if k in ZERO_ONLY_FIELDS and v == 0:
            continue
        if k == "n" and v == 1:
            continue
        if k in NOT_YET_FIELDS or k in ZERO_ONLY_FIELDS or k == "n":
            raise ApiError(400, "unsupported_parameter", f"Cale does not support '{k}' yet. Remove it from the request, or use a supported value.", param=k)
        raise ApiError(400, "unknown_parameter", f"Cale does not know the field '{k}'. Supported fields: {', '.join(sorted(CHAT_FIELDS | CALE_FIELDS))}.", param=k)


def chat_sampling(req):
    """top_p, seed and stop from a chat request, checked. Returns (top_p, seed, stop)."""
    caps = backends.supports()
    top_p = number(req, "top_p", 0.0, 1.0, 1.0)
    if top_p < 1.0 and not caps["top_p"]:
        raise ApiError(400, "unsupported_parameter", "This engine build does not support 'top_p' yet.", param="top_p")
    seed = req.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ApiError(400, "invalid_seed", "'seed' must be an integer.", param="seed")
    stop = req.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop is not None:
        if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(x, str) and 0 < len(x) <= 100 for x in stop):
            raise ApiError(400, "invalid_stop", "'stop' must be a string or a list of up to 4 non-empty strings (100 characters at most).", param="stop")
    return top_p, (None if seed is None else seed % (2 ** 32)), stop


def object_spec(req):
    """Check the body of POST /cale/models. A size is exact. Goals (see cale/goals.py) let Cale choose."""
    goals = goals_from(req)                           # first: a removed field gets its own message, not "unknown field"
    unknown = sorted(set(req) - OBJECT_FIELDS)
    if unknown:
        raise ApiError(400, "invalid_model_object", f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(OBJECT_FIELDS))}.")
    family = req.get("family", BASE)
    if family != BASE:
        raise ApiError(404, "model_not_found", f"Unknown family '{family}'. Only '{BASE}' exists.")
    spec = {}
    if "size" in req:
        pct = parse_model(f"{BASE}:{req['size']}")
        if pct == "auto":
            raise ApiError(400, "invalid_model_object", "A size is a number or smart/fast. To let Cale choose, give goals (min_tokens_per_s, max_first_token_s, memory_pct, min_quality, power).")
        spec["size_pct"] = pct
    max_tokens = number(req, "max_tokens", 1, MAX_TOKENS, None, True)
    if "size_pct" in spec and (goals_mod.is_set(goals) or any(k in req for k in goals_mod.FIELDS)):
        raise ApiError(400, "invalid_model_object", "A size is exact. Do not combine it with goals.")
    if "size_pct" not in spec and not goals_mod.is_set(goals):
        raise ApiError(400, "invalid_model_object", "Give a size, or at least one goal: min_tokens_per_s, max_first_token_s, memory_pct, min_quality or power=\"light\".")
    spec.update(goals if "size_pct" not in spec else {}, max_tokens=max_tokens)
    return spec


def object_slug(name):
    s = "".join(c if c.isalnum() else "-" for c in str(name or "model").lower()).strip("-")
    return (s or "model")[:30]


def ask_user(app, replacing=False):
    """Ask the person: allow this app? With a screen, the Allow dialog (oslayer.ask_allow). Without one, the request waits for
    `cale allow NAME` typed in a terminal on this computer (cale/approval.py)."""
    if DEBUG and os.environ.get("CALE_AUTO_APPROVE") == "1":
        return True
    if oslayer.has_screen():
        answer = oslayer.ask_allow(app, replacing)
        if answer is not None:
            return answer                                  # a person answered (or the question timed out: that is a no)
    return approval.wait_for_terminal(app, replacing, float(os.environ.get("CALE_APPROVE_WAIT_S", 300)))


CONNECTS = {}                        # ticket -> {"app", "state": pending|allowed|denied, "key", "done": Event, "since"}
CONNECT_MAX = 16                     # waiting requests at most: one app cannot fill Cale's memory with them
LEASES = {}                          # hold id -> {"app", "ttl", "expires"} for holds made with lease_s
MAX_LEASE_S = 3600


def _decide_connect(ticket):
    """Runs in its own thread: ask the person, then make the key if they allowed."""
    it = CONNECTS[ticket]
    allowed = False
    try:
        if oslayer.has_screen():
            with CONNECT_LOCK:                                     # one dialog at a time
                allowed = ask_user(it["app"], it["replacing"])
        else:
            allowed = ask_user(it["app"], it["replacing"])
    except Exception as e:
        print(f"[connect] asking failed ({type(e).__name__}); counting that as no", flush=True)
    if allowed:
        it["key"] = keys.create(it["app"])
    it["state"] = "allowed" if allowed else "denied"
    it["done"].set()


def _purge_connects():
    now = time.time()
    for t in [t for t, it in CONNECTS.items() if now - it["since"] > 900 or (it["done"].is_set() and now - it["since"] > 300)]:
        CONNECTS.pop(t, None)


ENGINE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 120                        # a stalled client cannot hold a thread for ever

    # --- helpers ----------------------------------------------------------------------
    def _json(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, e):
        self.close_connection = True      # the body may be unread: never reuse this connection
        err = {"message": str(e), "type": "invalid_request_error", "code": e.code, "param": e.param}
        if e.extra:
            err["cale"] = e.extra
        self._json(e.status, {"error": err}, {"Connection": "close"})

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or self.headers.get("Transfer-Encoding"):
            raise ApiError(400, "bad_length", "Send a Content-Length header and no chunked body.")
        if n > 4_000_000:
            raise ApiError(413, "too_large", "Request body is too large.")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise ApiError(400, "invalid_json", "Body is not valid JSON.")
        if not isinstance(body, dict):
            raise ApiError(400, "invalid_json", "The body must be a JSON object.")
        return body

    def _guard(self):
        if self.headers.get("Origin") or self.headers.get("Sec-Fetch-Site"):
            raise ApiError(403, "browser_not_allowed", "Web pages cannot use Cale. Call it from the app's own backend.")
        host = (self.headers.get("Host") or "").split(":")[0]
        if host != paths.HOST:
            raise ApiError(403, "bad_host", f"Use http://{paths.HOST}:{paths.PORT} (not '{host}').")

    def _auth(self):
        auth = self.headers.get("Authorization") or ""
        app = keys.check(auth[7:].strip() if auth.lower().startswith("bearer ") else "")
        if app is None:
            raise ApiError(401, "invalid_api_key", "Missing or invalid Cale key. Ask Cale for one at /cale/connect.")
        return app

    def log_message(self, fmt, *a):
        print(f"[http] {self.address_string()} {fmt % a}", flush=True)

    # --- routes -----------------------------------------------------------------------
    def do_GET(self):
        try:
            self._guard()
            if self.path == "/cale/status":
                return self._json(200, {"ok": True, "version": "0.1", "model": BASE, **ENGINE.status()})
            if self.path.startswith("/cale/connect/status"):
                from urllib.parse import parse_qs, urlparse
                ticket = (parse_qs(urlparse(self.path).query).get("ticket") or [""])[0]
                return self._connect_reply(ticket, "")
            if self.path == "/cale/computer":                          # like /cale/status: no key. What this computer can do.
                return self._json(200, {"model": BASE, "max_context_tokens": ENGINE.known_context(), "this_computer": ENGINE.this_computer()})                  # no key: it never loads the model
            if self.path == "/v1/models":
                app = self._auth()
                curve = None
                if paths.CURVE.exists():
                    curve = json.loads(paths.CURVE.read_text())
                data = [{"id": f"{BASE}:{name}", "object": "model", "owned_by": "local", "size_pct": pct}
                        for name, pct in ALIASES.items()]
                data.append({"id": f"{BASE}:auto", "object": "model", "owned_by": "local", "size_pct": None})
                with OBJ_LOCK:
                    data += [{"id": oid, "object": "model", "owned_by": app, "size_pct": None,
                              "cale": {"spec": o["spec"], "last": o.get("last")}}
                             for (owner, oid), o in OBJECTS.items() if owner == app]
                this_mac = ENGINE.this_computer()
                return self._json(200, {"object": "list", "data": data, "cale": {
                    "base": BASE, "max_context_tokens": ENGINE.known_context(), "capabilities": {"images": bool(backends.supports().get("images")), "audio": bool(backends.supports().get("audio")), "tools": True, "json_schema": True}, "sizes": {"min": MIN_PCT, "max": MAX_PCT, "aliases": ALIASES,
                                            "available": sorted({l["pct"] for l in LEVELS}),
                                            "usage": f"{BASE}:<percent> or {BASE}:auto", "goals": list(goals_mod.FIELDS)}, "curve": curve,
                    "this_computer": this_mac, "this_mac": this_mac,       # this_mac: the first name, kept so old apps still work
                    "holds": ENGINE.status()["holds"]}})
            raise ApiError(404, "not_found", "Unknown path.")
        except ApiError as e:
            self._error(e)
        except Exception as e:
            print(f"[http] error {type(e).__name__}: {e}", flush=True)
            self.close_connection = True

    def do_POST(self):
        self.arrived = time.time()                 # time spent waiting in line counts toward an app's first-token goal
        try:
            self._guard()
            if self.path == "/cale/connect":
                return self._connect()
            if self.path == "/v1/chat/completions":
                app = self._auth()
                return self._chat(self._body(), app)
            if self.path == "/cale/load":
                return self._hold(self._auth(), self._body())
            if self.path == "/cale/models":
                return self._model_object(self._auth(), self._body())
            if self.path == "/cale/predict":
                self._auth()
                return self._predict(self._body())
            if self.path == "/cale/lease":
                return self._lease(self._auth(), self._body())
            if self.path == "/cale/unload":
                app = self._auth()
                ended = ENGINE.release(str(self._body().get("hold", "")), app)
                return self._json(200, {"ended": ended})
            if DEBUG and self.path == "/cale/debug/slow":
                ENGINE.slow = number(self._body(), "factor", 0.1, 100, 1.0)
                return self._json(200, {"ok": True, "slow": ENGINE.slow})
            if DEBUG and self.path == "/cale/debug/ram":
                gb = self._body().get("gb")
                ENGINE.debug_ram = None if gb is None else int(number({"gb": gb}, "gb", 0.1, 4096, 1.0) * 1e9)
                return self._json(200, {"ok": True, "ram_bytes": ENGINE.debug_ram})
            if DEBUG and self.path == "/cale/debug/pool":
                m = ENGINE.m
                n = self._body().get("tokens")
                if m is not None and hasattr(m, "pool_limit"):
                    m.pool_limit = None if n is None else int(n)
                return self._json(200, {"ok": True, "pool_tokens": None if m is None else getattr(m, "pool_limit", None)})
            if DEBUG and self.path == "/cale/debug/fail":
                ENGINE.debug_fail = True
                return self._json(200, {"ok": True, "next_call_fails": True})
            if DEBUG and self.path == "/cale/debug/pressure":
                pressure.set_override(self._body().get("level"))
                return self._json(200, {"ok": True, "pressure": pressure.name()})
            raise ApiError(404, "not_found", "Unknown path.")
        except ApiError as e:
            self._error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:                  # keep serving whatever happens
            print(f"[http] error {type(e).__name__}: {e}", flush=True)
            try:
                self._json(500, {"error": {"message": "Internal error. The details are in Cale's log.", "type": "server_error", "code": "internal"}})
            except Exception:
                pass

    def do_DELETE(self):
        try:
            self._guard()
            app = self._auth()
            if self.path.startswith("/cale/models/"):
                return self._lease(app, {"id": self.path[len("/cale/models/"):], "end": True})
            raise ApiError(404, "not_found", "Unknown path.")
        except ApiError as e:
            self._error(e)

    def _lease(self, app, req):
        """Renew or end a lease. {"id": object id} or {"hold": hold id}; "end": true ends it now."""
        end = flag(req, "end")
        if "id" in req:
            oid = str(req["id"])
            with OBJ_LOCK:
                obj = OBJECTS.get((app, oid))
                if obj is None or not obj.get("lease"):
                    raise ApiError(404, "no_lease", f"No leased model object '{oid}' for this app.")
                if end:
                    OBJECTS.pop((app, oid), None)
                    hid = obj["lease"].get("hid")
                else:
                    obj["lease"]["expires"] = time.time() + obj["lease"]["ttl"]
                    return self._json(200, {"id": oid, "expires_in_s": obj["lease"]["ttl"]})
            if hid:
                ENGINE.release(hid)
            return self._json(200, {"id": oid, "ended": True})
        if "hold" in req:
            hid = str(req["hold"])
            with OBJ_LOCK:
                lease = LEASES.get(hid)
                if lease is None or lease["app"] != app:
                    raise ApiError(404, "no_lease", f"No leased hold '{hid}' for this app.")
                if not end:
                    lease["expires"] = time.time() + lease["ttl"]
                    return self._json(200, {"hold": hid, "expires_in_s": lease["ttl"]})
                LEASES.pop(hid, None)
            ENGINE.release(hid, app)
            return self._json(200, {"hold": hid, "ended": True})
        raise ApiError(400, "invalid_lease", "Send {\"id\": <model id>} or {\"hold\": <hold id>}, and \"end\": true to end it.")

    def _connect(self):
        body = self._body()
        app = str(body.get("app", "")).strip()
        app = "".join(ch for ch in app if ch.isprintable() and ch not in '"\\')[:40]
        if not app:
            raise ApiError(400, "invalid_app", "Send {\"app\": \"Your App Name\"}.")
        wait_s = number(body, "wait_s", 0, 600, 120)
        with CONNECT_LOCK_META:
            _purge_connects()
            if sum(1 for it in CONNECTS.values() if it["state"] == "pending") >= CONNECT_MAX:
                raise ApiError(429, "busy", "Too many apps are waiting for the user to answer. Try again later.")
            if oslayer.has_screen() and CONNECT_LOCK.locked():
                raise ApiError(429, "busy", "Another app is waiting for the user to answer. Try again later.")
            if any(it["app"] == app and it["state"] == "pending" for it in CONNECTS.values()):
                # Never hand a second asker the first one's ticket: it could collect the first one's key. The first asker polls its own ticket.
                raise ApiError(409, "already_waiting", f"An app named '{app}' is already waiting for the person's answer. Wait for it, or use another name.")
            ticket = "t-" + secrets.token_urlsafe(16)
            CONNECTS[ticket] = {"app": app, "state": "pending", "key": None, "done": threading.Event(), "since": time.time(),
                                "replacing": app in keys.apps()}
            threading.Thread(target=_decide_connect, args=(ticket,), daemon=True).start()
        it = CONNECTS[ticket]
        if wait_s > 0:
            it["done"].wait(wait_s)
        return self._connect_reply(ticket, app)

    def _connect_reply(self, ticket, app):
        it = CONNECTS.get(ticket)
        if it is None:
            raise ApiError(404, "unknown_ticket", "No such request. Ask again with POST /cale/connect.")
        if it["state"] == "allowed":
            key, it["key"] = it["key"], None                       # the key is handed out once
            if key is None:
                raise ApiError(410, "already_delivered", "The key was handed out already. Ask again with POST /cale/connect.")
            CONNECTS.pop(ticket, None)
            return self._json(200, {"app": it["app"], "api_key": key, "base_url": f"http://{paths.HOST}:{paths.PORT}/v1"})
        if it["state"] == "denied":
            CONNECTS.pop(ticket, None)
            raise ApiError(403, "denied", "The user did not allow this app.")
        how = (f"Ask the person to type, in a terminal on this computer:  cale allow \"{it['app']}\"" if not oslayer.has_screen()
               else "The person must answer the question on the screen.")
        return self._json(202, {"status": "pending", "ticket": ticket, "poll": f"/cale/connect/status?ticket={ticket}", "message": how},
                          {"Retry-After": "2"})

    def _log_chat(self, app, req, ran, done):
        """One line per answer: which app asked for what, and which size ran."""
        print(f"[chat] app={app!r} asked={req.get('model', BASE)} ran={model_id(ran)} "
              f"tokens={done.get('tokens')} seconds={done.get('seconds')} disk_mb={done.get('disk_mb', 0)}", flush=True)

    def _note_done(self, app, name, obj, done):
        if obj:
            with OBJ_LOCK:
                last = OBJECTS.get((app, name), {}).get("last")
                if last is not None:
                    last.update(seconds=done.get("seconds"), tokens=done.get("tokens"))

    def _hold(self, app, req):
        """Keep a size in RAM. The reply stays open: one JSON line now, then a blank line every second.
        The hold ends when the app closes the connection, quits, or calls /cale/unload."""
        has_size, has_mem = "size" in req, "memory_pct" in req
        if has_size == has_mem:
            raise ApiError(400, "invalid_hold", "Send either {\"size\": \"smart\"|\"fast\"|<percent>} or {\"memory_pct\": <1-100>}.")
        if has_size:
            spec = {"size_pct": parse_model(f"{BASE}:{req['size']}")}
            if spec["size_pct"] == "auto":
                raise ApiError(400, "invalid_hold", "A hold needs a size, not 'auto'. Use memory_pct to let Cale choose.")
        else:
            spec = {"memory_pct": number(req, "memory_pct", 1, 100, 25), "on_miss": goals_from({"on_miss": req.get("on_miss", "best_effort")})["on_miss"]}
        lease = number(req, "lease_s", 5, MAX_LEASE_S, None)
        try:
            hid, event, info = ENGINE.hold(app, spec)
        except GoalMissed as e:
            raise ApiError(422, "goal_missed", str(e), e.info)
        if lease is not None:                                       # plain HTTP: no open connection. The hold lasts `lease_s` seconds after its last renewal.
            with OBJ_LOCK:
                LEASES[hid] = {"app": app, "ttl": lease, "expires": time.time() + lease}
            return self._json(200, {**info, "lease": {"expires_in_s": lease, "renew": "POST /cale/lease", "end": "POST /cale/unload"}})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self._stay_open(info, event, lambda: ENGINE.release(hid))

    def _predict(self, req):
        """Ask before you run: what would Cale do with these goals on this computer? Nothing is run. Same fields as a model object, plus prompt_tokens."""
        goals = goals_from(req)
        extra = sorted(set(req) - set(goals_mod.FIELDS) - {"prompt_tokens", "max_tokens"})
        if extra:
            raise ApiError(400, "unknown_parameter", f"Unknown field(s): {', '.join(extra)}. Allowed: {', '.join(sorted(set(goals_mod.FIELDS) | {'prompt_tokens', 'max_tokens'}))}.")
        prompt = number(req, "prompt_tokens", 0, MAX_CONTEXT, 200, True)
        answer = number(req, "max_tokens", 1, MAX_TOKENS, 256, True)
        return self._json(200, ENGINE.predict(goals, prompt, answer))

    def _stay_open(self, info, event, cleanup):
        """Send one JSON line, then a blank line every second until `event` is set or the app closes its end."""
        try:
            self.wfile.write(json.dumps(info).encode() + b"\n")
            self.wfile.flush()
            while not event.is_set():
                if select.select([self.connection], [], [], 1.0)[0]:        # the app closed its end
                    try:
                        if self.connection.recv(1, socket.MSG_PEEK) == b"":
                            break
                    except OSError:
                        break
                self.wfile.write(b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            cleanup()

    def _model_object(self, app, req):
        """Make a model object: a named set of parameters (a size, or goals) that the app then uses as a model name.
        The reply stays open like a hold. The object ends when the app closes the connection or quits.
        A size or a memory goal also keeps that size in RAM for as long as the object lives."""
        lease = number(req, "lease_s", 5, MAX_LEASE_S, None)
        spec = object_spec({k: v for k, v in req.items() if k != "lease_s"})
        hold_spec, hid, event, info = None, None, threading.Event(), {}
        if "size_pct" in spec:
            hold_spec = {"size_pct": spec["size_pct"]}
        elif spec["memory_pct"] is not None:
            hold_spec = {"memory_pct": spec["memory_pct"], "on_miss": spec["on_miss"]}
        if hold_spec:
            try:
                hid, event, info = ENGINE.hold(app, hold_spec)
            except GoalMissed as e:
                raise ApiError(422, "goal_missed", str(e), e.info)
        if "size_pct" not in spec:                              # ask before you run: can this computer meet the goals?
            expect = spec.get("max_tokens") or 256
            pr = ENGINE.predict({k: v for k, v in spec.items() if k in goals_mod.FIELDS}, 200, expect)
            info["prediction"] = pr
            if not pr["goal_met"] and spec["on_miss"] == "refuse":
                if hid:
                    ENGINE.release(hid)
                raise ApiError(422, "goal_missed", "No size meets these goals on this computer. See error.cale for what Cale predicts.", pr)
        oid = object_slug(req.get("name"))
        with OBJ_LOCK:
            if (app, oid) in OBJECTS:                     # the same name is alive already: this one gets a suffix
                oid = f"{oid}-{uuid.uuid4().hex[:4]}"
            OBJECTS[(app, oid)] = {"spec": spec, "created": time.time(),
                                   "max_pct": info.get("size_pct") if "memory_pct" in (hold_spec or {}) else None,   # a memory goal also caps the size this object's calls run at
                                   "memory_goal_met": info.get("goal_met") if "memory_pct" in (hold_spec or {}) else None}
        print(f"[model] {app!r} made {oid}: { {k: v for k, v in spec.items() if v is not None} }", flush=True)
        if lease is not None:                                       # plain HTTP: no open connection. The object lasts `lease_s` seconds after its last use or renewal.
            with OBJ_LOCK:
                OBJECTS[(app, oid)]["lease"] = {"ttl": lease, "expires": time.time() + lease, "hid": hid}
            return self._json(200, {"id": oid, "object": "cale.model", "family": BASE, "spec": {k: v for k, v in spec.items() if v is not None},
                                    **info, "lease": {"expires_in_s": lease, "renew": "POST /cale/lease  (a chat call with this id renews it too)",
                                                      "end": "POST /cale/lease with \"end\": true, or DELETE /cale/models/" + oid}})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def cleanup():
            with OBJ_LOCK:
                OBJECTS.pop((app, oid), None)
            if hid:
                ENGINE.release(hid)
            print(f"[model] {oid} ended", flush=True)

        self._stay_open({"id": oid, "object": "cale.model", "family": BASE, "spec": {k: v for k, v in spec.items() if v is not None},
                         **info}, event, cleanup)

    def _tool_message(self, content, finish, done, tools, choice, parallel, start, msgs):
        """An answer given tools: a message with tool_calls if the model called tools, else the text. If the model tried and got it
        wrong (auto mode), ask once more with the answer guided to a valid call. Returns (message, finish_reason, done)."""
        calls, kind = tools_mod.parse_calls(content, tools)
        if kind == "broken" and choice == "auto" and finish != "length":
            gen2, _ = start(tools_mod.tool_guide(tools, "required", parallel))
            text2, done2 = [], {}
            for ev in gen2:
                if "delta" in ev:
                    text2.append(ev["delta"])
                else:
                    done2 = ev["done"]
            done2["tokens"] = done2.get("tokens", 0) + done.get("tokens", 0)
            done2["seconds"] = round(done2.get("seconds", 0) + done.get("seconds", 0), 3)
            done2["total_s"] = round(done2.get("total_s", 0) + done.get("total_s", 0), 3)
            content, finish, done = "".join(text2), done2.get("finish", "stop"), {**done, **done2}
            calls, kind = tools_mod.parse_calls(content, tools)
        if kind == "calls":
            return {"role": "assistant", "content": None, "tool_calls": calls}, "tool_calls", done
        return {"role": "assistant", "content": content}, finish, done

    def _stream_tools(self, gen, hdr, info, cid, created, label, expect, tools, choice, parallel, start, msgs, app, name, obj, req):
        """A streamed answer given tools. Cale has to read the whole answer before it knows if it is a tool call, so it streams at the end."""
        text, done = [], {}
        try:
            for ev in gen:
                if "delta" in ev:
                    text.append(ev["delta"])
                else:
                    done = ev["done"]
        except EngineError:
            raise ApiError(500, "engine_error", "The engine failed and was reset. Try again. The details are in Cale's log.")
        message, finish, done = self._tool_message("".join(text), done.get("finish", "stop"), done, tools, choice, parallel, start, msgs)
        self._note_done(app, name, obj, done)
        self._log_chat(app, req, done["size_pct"], done)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        for k, v in hdr.items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True

        def chunk(delta, fin=None, extra=None):
            o = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": label(done["size_pct"]),
                 "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            o.update(extra or {})
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if message.get("tool_calls"):
                chunk({"role": "assistant", "content": None})
                for i, c in enumerate(message["tool_calls"]):
                    chunk({"tool_calls": [{"index": i, "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": ""}}]})
                    chunk({"tool_calls": [{"index": i, "function": {"arguments": c["function"]["arguments"]}}]})
            else:
                chunk({"role": "assistant", "content": ""})
                if message["content"]:
                    chunk({"content": message["content"]})
            chunk({}, finish, {"usage": {"prompt_tokens": done["prompt_tokens"], "completion_tokens": done["tokens"],
                                         "total_tokens": done["prompt_tokens"] + done["tokens"]},
                               "cale": {**info, "size": done["size_pct"], "seconds": done["seconds"], "waited_s": done.get("waited_s"),
                                        "total_s": done.get("total_s"), "goal_met_actual": done.get("goal_met_actual"),
                                        "goals_actual": done.get("goals_actual"), "size_changes": done.get("switches") or None}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _watch_client(self, cancel, done):
        """While an answer is made, notice a client that closed its connection. The work stops (the engine checks `cancel`)."""
        conn = self.connection
        while not done.wait(0.5):
            try:
                if select.select([conn], [], [], 0)[0] and conn.recv(1, socket.MSG_PEEK) == b"":
                    cancel.set()
                    return
            except (OSError, ValueError):
                cancel.set()
                return

    def _chat(self, req, app="?"):
        cancel, done = Cancel(), threading.Event()
        threading.Thread(target=self._watch_client, args=(cancel, done), daemon=True).start()
        try:
            return self._chat_inner(req, app, cancel)
        except Cancelled:
            self.close_connection = True                 # the client is gone: there is nobody to answer
            print(f"[chat] app={app!r} client left: work stopped", flush=True)
        except constrain.DeadEnd as e:
            raise ApiError(422, "constraint_dead_end", f"{e}. Try a simpler schema, or relax a limit such as minLength or minItems.")
        finally:
            done.set()

    def _chat_inner(self, req, app, cancel):
        check_chat_fields(req)
        name = req.get("model", BASE)
        with OBJ_LOCK:
            obj = OBJECTS.get((app, name)) if isinstance(name, str) else None     # only this app's own objects exist for it
        spec = obj["spec"] if obj else None
        if obj and obj.get("lease"):                                 # using an object keeps its lease alive
            with OBJ_LOCK:
                obj["lease"]["expires"] = time.time() + obj["lease"]["ttl"]
        pct = ("auto" if "size_pct" not in spec else spec["size_pct"]) if obj else parse_model(name)
        try:
            tools = tools_mod.parse_tools(req.get("tools"))
            choice = tools_mod.parse_choice(req.get("tool_choice"), tools)
            rf_guide, rf_note = tools_mod.response_guide(req.get("response_format"))
        except tools_mod.ToolError as e:
            raise ApiError(400, e.code, str(e), param=e.param)
        parallel = req.get("parallel_tool_calls", True)
        if not isinstance(parallel, bool):
            raise ApiError(400, "invalid_parallel_tool_calls", "'parallel_tool_calls' must be true or false.", param="parallel_tool_calls")
        lp_flag, lp_top = req.get("logprobs"), req.get("top_logprobs")
        if lp_flag is not None and not isinstance(lp_flag, bool):
            raise ApiError(400, "invalid_logprobs", "'logprobs' must be true or false.", param="logprobs")
        if lp_top is not None and (isinstance(lp_top, bool) or not isinstance(lp_top, int) or not 0 <= lp_top <= 20):
            raise ApiError(400, "invalid_top_logprobs", "'top_logprobs' must be a whole number from 0 to 20.", param="top_logprobs")
        if lp_top is not None and not lp_flag:
            raise ApiError(400, "invalid_top_logprobs", "'top_logprobs' needs 'logprobs': true.", param="top_logprobs")
        logprobs_k = (lp_top or 0) if lp_flag else None
        use_tools = bool(tools) and choice != "none"
        if logprobs_k is not None and (use_tools or rf_guide is not None):
            raise ApiError(400, "unsupported_parameter", "Cale cannot report 'logprobs' together with 'tools' or 'response_format' yet. Use one of them.", param="logprobs")
        if use_tools and rf_guide is not None:
            raise ApiError(400, "unsupported_parameter", "Cale cannot combine 'tools' and 'response_format' in one call yet. Use one of them.", param="response_format")
        msgs = tools_mod.convert_messages(req.get("messages"), ApiError, images_ok=bool(backends.supports().get("images")), audio_ok=bool(backends.supports().get("audio")))
        guide = None
        if use_tools:
            msgs = tools_mod.with_system(msgs, tools_mod.tool_prompt(tools, choice, parallel))
            if choice != "auto":
                guide = tools_mod.tool_guide(tools, choice, parallel)
        elif rf_guide is not None:
            msgs, guide = tools_mod.with_system(msgs, rf_note), rf_guide
        limit_key = "max_completion_tokens" if req.get("max_completion_tokens") is not None else "max_tokens"
        asked = req.get(limit_key)
        if asked is None and obj and spec.get("max_tokens"):
            asked = spec["max_tokens"]                   # the object's answer length
        max_tokens = number({limit_key: asked}, limit_key, 1, MAX_TOKENS, MAX_TOKENS, True)    # no limit given: the context decides
        temp = number(req, "temperature", 0.0, 2.0, 0.0)
        top_p, seed, stop = chat_sampling(req)
        stream = bool(req.get("stream"))
        goal = None
        call_goals = any(k.startswith("cale_") for k in req)
        if call_goals:
            call = goals_from(req, "cale_")              # also refuses the removed cale_deadline_s with its own message
        if obj and call_goals:
            raise ApiError(400, "goal_in_object", "This model object holds its own goals. Do not send cale_* fields with it.")
        if obj and pct == "auto":
            goal = {"goals": {k: spec.get(k) for k in goals_mod.FIELDS}, "expect_tokens": spec.get("max_tokens"), "max_pct": obj.get("max_pct") or 100}
        elif pct == "auto":
            goal = {"goals": call if call_goals else goals_mod.parse({})}
        elif call_goals:
            raise ApiError(400, "goal_needs_auto", f"The cale_* goal fields need model '{BASE}:auto'.")
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        def start(g):
            """Begin an answer (with guide g). Returns (generator, plan). Engine problems become API errors."""
            gen = ENGINE.chat_stream(msgs, max_tokens, temp, app, 100 if pct == "auto" else pct, goal,
                                     top_p=top_p, seed=seed, stop=stop, arrived=getattr(self, "arrived", None),
                                     max_tokens_explicit=asked is not None, cancel=cancel, guide=g, logprobs=logprobs_k)
            try:
                return gen, next(gen)["plan"]
            except GoalMissed as e:
                gen.close()
                raise ApiError(422, "goal_missed", str(e), e.info)
            except Unsupported as e:
                gen.close()
                raise ApiError(400, "unsupported_parameter", str(e), param="logprobs")
            except Unsafe as e:
                gen.close()
                raise ApiError(507, "unsafe_for_this_computer", str(e) + " Cale does not run what would freeze or crash the computer. Use a shorter prompt, or free memory.", e.info)
            except ContextTooLong as e:
                gen.close()
                raise ApiError(400, "context_length_exceeded", str(e), e.info, param="messages")
            except EngineError as e:
                raise ApiError(500, "engine_error", "The engine failed and was reset. Try again. The details are in Cale's log.")

        gen, plan = start(guide)
        expect = plan["size_pct"]
        if obj:
            with OBJ_LOCK:                               # what this object ran at, for apps that show it
                if (app, name) in OBJECTS:
                    OBJECTS[(app, name)]["last"] = {"size_pct": expect, "goal_met": plan["goal_met"],
                                                    "predicted": plan["predicted"], "at": int(time.time())}
        hdr = {"X-Cale-Requested-Size": str(pct), "X-Cale-Size": str(expect), "X-Cale-Disk-MB": str(plan["disk_mb"]),
               "X-Cale-Waited-S": str(plan["waited_s"])}
        mem_met = obj.get("memory_goal_met") if obj else None
        if mem_met is not None:
            hdr["X-Cale-Memory-Goal-Met"] = str(mem_met).lower()
        pr = plan["predicted"] or {}
        hdr["X-Cale-Predicted-Tokens-Per-S"] = str(pr.get("tokens_per_s"))
        hdr["X-Cale-Predicted-First-Token-S"] = str(pr.get("first_token_s"))
        hdr["X-Cale-Confidence"] = plan["confidence"]
        if plan["goal_met"] is not None:
            hdr["X-Cale-Goal-Met"] = str(plan["goal_met"]).lower()
            if plan["unmet"]:
                hdr["X-Cale-Unmet"] = ",".join(plan["unmet"])
        info = {"requested_size": pct, "predicted": plan["predicted"], "goal_met": plan["goal_met"], "unmet": plan["unmet"],
                "confidence": plan["confidence"], "waited_s": plan["waited_s"], "memory_goal_met": mem_met,
                "answer_limit": plan["max_tokens"]}
        label = (lambda p: name) if obj else model_id          # a model object reports its own id as the model

        if not stream:
            text, done, lps = [], {}, []
            try:
                for ev in gen:
                    if "delta" in ev:
                        text.append(ev["delta"])
                    elif "logprobs" in ev:
                        lps.extend(ev["logprobs"])
                    else:
                        done = ev["done"]
            except EngineError as e:
                raise ApiError(500, "engine_error", "The engine failed and was reset. Try again. The details are in Cale's log.")
            message, finish = {"role": "assistant", "content": "".join(text)}, done.get("finish", "stop")
            if use_tools:
                message, finish, done = self._tool_message(message["content"], finish, done, tools, choice, parallel, start, msgs)
            actual = done.get("size_pct") or expect
            if done.get("goal_met_actual") is not None:          # now the answer is finished: say what really happened
                hdr["X-Cale-Goal-Met"] = str(done["goal_met_actual"]).lower()
                info["goal_met_actual"] = done["goal_met_actual"]
                info["goals_actual"] = done.get("goals_actual")
            if done.get("switches"):
                info["size_changes"] = done["switches"]
            hdr["X-Cale-Total-S"] = str(done.get("total_s"))
            self._note_done(app, name, obj, done)
            self._log_chat(app, req, actual, done)
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": label(actual),
                "choices": [{"index": 0, "message": message, "finish_reason": finish,
                             **({"logprobs": {"content": lp_content(lps)}} if logprobs_k is not None else {})}],
                "usage": {"prompt_tokens": done.get("prompt_tokens", 0), "completion_tokens": done.get("tokens", 0),
                          "total_tokens": done.get("prompt_tokens", 0) + done.get("tokens", 0),
                          "prompt_tokens_details": {"cached_tokens": done.get("cached_tokens", 0)}},
                "cale": {**info, "size": actual, **{k: done.get(k) for k in ("seconds", "first_token_s", "waited_s", "total_s", "switch_ms", "drafter", "disk_mb", "disk_s")},
                           "answers_sharing_the_step": done.get("batch_avg", 1.0)},
            }, hdr)

        if use_tools:
            return self._stream_tools(gen, hdr, info, cid, created, label, expect, tools, choice, parallel, start, msgs, app, name, obj, req)
        try:
            first = next(gen, None)            # a load or input failure here still gets a normal error reply
        except EngineError as e:
            raise ApiError(500, "engine_error", "The engine failed and was reset. Try again. The details are in Cale's log.")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        for k, v in hdr.items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True

        def send(raw):
            self.wfile.write(b"data: " + raw + b"\n\n")
            self.wfile.flush()

        def chunk(delta, finish=None, extra=None, lp=None):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": label(expect),
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish, **({"logprobs": {"content": lp_content(lp)}} if lp else {})}]}
            obj.update(extra or {})
            send(json.dumps(obj).encode())

        def events():
            if first is not None:
                yield first
            yield from gen

        try:
            chunk({"role": "assistant", "content": ""})
            for ev in events():
                if "delta" in ev:
                    chunk({"content": ev["delta"]})
                elif "logprobs" in ev:
                    chunk({}, None, None, ev["logprobs"])
                else:
                    d = ev["done"]
                    self._note_done(app, name, obj, d)
                    self._log_chat(app, req, d["size_pct"], d)
                    chunk({}, d["finish"], {
                        "model": label(d["size_pct"]),
                        "usage": {"prompt_tokens": d["prompt_tokens"], "completion_tokens": d["tokens"],
                                  "total_tokens": d["prompt_tokens"] + d["tokens"]},
                        "cale": {**info, "size": d["size_pct"], "seconds": d["seconds"], "waited_s": d.get("waited_s"),
                                 "total_s": d.get("total_s"), "goal_met_actual": d.get("goal_met_actual"),
                                 "goals_actual": d.get("goals_actual"), "size_changes": d.get("switches") or None,
                                 "drafter": d["drafter"], "disk_mb": d.get("disk_mb"), "disk_s": d.get("disk_s")}})
            send(b"[DONE]")
        except (BrokenPipeError, ConnectionResetError, Cancelled):
            pass
        except Exception as e:                 # headers are sent already: report inside the stream
            print(f"[http] stream error {type(e).__name__}: {e}", flush=True)
            try:
                send(json.dumps({"error": {"message": "Internal error. The details are in Cale's log.", "type": "server_error"}}).encode())
                send(b"[DONE]")
            except Exception:
                pass
        finally:
            gen.close()                        # releases the generation lock if the client left early


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server share a busy port, so two Cales could answer one port.
    # Off there: a second Cale (two apps starting it at once) stops with "address in use". Elsewhere it only speeds up restarts.
    allow_reuse_address = not oslayer.WIN


def reap_leases():
    """Every second: end the leased model objects and holds that nobody renewed or used."""
    while True:
        time.sleep(1)
        now = time.time()
        gone = []
        with OBJ_LOCK:
            for key, o in list(OBJECTS.items()):
                if o.get("lease") and o["lease"]["expires"] < now:
                    gone.append(o["lease"].get("hid"))
                    OBJECTS.pop(key, None)
                    print(f"[model] {key[1]} lease ended", flush=True)
            for hid, l in list(LEASES.items()):
                if l["expires"] < now:
                    gone.append(hid)
                    LEASES.pop(hid, None)
        for hid in gone:
            if hid:
                ENGINE.release(hid)


def exit_when_idle(limit=None):
    """On-demand mode (no service manager restarts Cale): end the process after `limit` seconds with no hold, no
    model object and no answer. The next app that needs Cale starts it again (cale.client.ensure_running)."""
    limit = limit or float(os.environ.get("CALE_EXIT_IDLE_S", 600))
    start = time.time()
    while True:
        time.sleep(15)
        with OBJ_LOCK:
            objects = bool(OBJECTS)
        with ENGINE.mu:
            holds = bool(ENGINE.holds)
        answering = not ENGINE.lock.acquire(blocking=False)      # an answer (or a model load) is running
        if not answering:
            ENGINE.lock.release()
        if objects or holds or answering:
            start = time.time()
        elif time.time() - max(start, ENGINE.last_used) > limit:
            print(f"[cale] idle for {int(limit)} s: exiting. Apps start Cale again when they need it.", flush=True)
            os._exit(0)


def main():
    global ENGINE
    paths.HOME.mkdir(parents=True, exist_ok=True)
    ENGINE = Engine(paths.MODEL, idle_s=float(os.environ.get("CALE_IDLE_S", 600)) if DEBUG else 600)
    srv = Server((paths.HOST, paths.PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=reap_leases, daemon=True).start()
    if os.environ.get("CALE_ON_DEMAND") == "1":
        threading.Thread(target=exit_when_idle, daemon=True).start()
    print(f"[cale] listening on http://{paths.HOST}:{paths.PORT} model={paths.MODEL}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
