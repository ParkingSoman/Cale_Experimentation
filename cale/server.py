"""Cale local API: an OpenAI-style server on 127.0.0.1 that runs one Gemma 3n model at any of its trained sizes.

| Endpoint | Key needed | Job |
|---|---|---|
| GET  /cale/status | no  | is Cale running, is the model loaded, memory pressure |
| POST /cale/connect | no  | an app asks for a key. The computer shows an Allow dialog. |
| GET  /v1/models | yes | the two named sizes (smart, fast), the size range, the measured quality curve |
| POST /v1/chat/completions | yes | chat. "model": "gemma-3n:smart", "gemma-3n:fast", "gemma-3n:<percent>" or "gemma-3n:auto". stream supported. |
| POST /cale/load | yes | keep a size in RAM until this connection closes. {"size": ...} or {"memory_pct": ...} |
| POST /cale/models | yes | make a model object: {"name", "size" or "deadline_s"+"max_tokens" and/or "memory_pct", "allow_disk", "on_miss"}. Reply stays open. Use the returned id as the model name. |
| POST /cale/unload | yes | end a hold early. {"hold": id} |

"gemma-3n:auto" lets Cale choose the size. Goal fields: "cale_deadline_s" (needs max_tokens), "cale_allow_disk", "cale_on_miss".

No browser access on purpose: there are no CORS headers, requests whose Host is not 127.0.0.1
are refused, and every route refuses a request that carries an Origin header (so web pages cannot use it).
Debug switches (CALE_DEBUG=1 only): CALE_AUTO_APPROVE=1 skips the dialog, CALE_IDLE_S=n sets the idle time,
POST /cale/debug/pressure forces a memory pressure level.
"""
import json
import math
import os
import select
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nest.levels3n import MAX_CONTEXT, load_levels, pct_range, pick

from . import backends, keys, oslayer, paths, pressure
from .engine import ContextTooLong, Engine, EngineError, GoalMissed
from . import choose

BASE = "gemma-3n"
LEVELS = load_levels()
MIN_PCT, MAX_PCT = pct_range(LEVELS)
ALIASES = {"smart": MAX_PCT, "fast": MIN_PCT}
MAX_TOKENS = MAX_CONTEXT              # the context decides what really fits (the engine checks it)
DEFAULT_TOKENS = 1024                # when an app sends no limit (Open WebUI does not)
DEBUG = os.environ.get("CALE_DEBUG") == "1"
CONNECT_LOCK = threading.Lock()      # one Allow dialog at a time
OBJECTS = {}                         # model objects: (app, id) -> {"spec", "created"}. See _model_object.
OBJ_LOCK = threading.Lock()
OBJECT_FIELDS = {"family", "name", "size", "deadline_s", "max_tokens", "memory_pct", "allow_disk", "on_miss"}


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
        raise ApiError(400, "invalid_model", f"Size '{size}' is not a number.")


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


def number(req, key, lo, hi, default, integer=False):
    v = req.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ApiError(400, f"invalid_{key}", f"'{key}' must be a number from {lo} to {hi}.")
    return int(v) if integer else float(v)


def flag(req, key, default=False):
    v = req.get(key, default)
    if not isinstance(v, bool):
        raise ApiError(400, f"invalid_{key}", f"'{key}' must be true or false.")
    return v


def on_miss(req, key="cale_on_miss"):
    v = req.get(key, "smallest")
    if v not in ("smallest", "refuse"):
        raise ApiError(400, f"invalid_{key}", f"'{key}' must be \"smallest\" or \"refuse\".")
    return v


# What /v1/chat/completions accepts. Anything else gets a clear 400. Cale never takes a field and ignores it in silence.
CHAT_FIELDS = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "seed", "stop", "stream"}
CALE_FIELDS = {"cale_deadline_s", "cale_allow_disk", "cale_on_miss"}    # Cale's own goal fields (with model gemma-3n:auto)
NO_EFFECT_FIELDS = {"user", "stream_options", "store", "metadata"}      # accepted: they change nothing Cale does
ZERO_ONLY_FIELDS = {"frequency_penalty", "presence_penalty"}            # accepted only as 0
NOT_YET_FIELDS = {"tools", "tool_choice", "functions", "function_call", "response_format", "logprobs", "top_logprobs",
                  "logit_bias", "parallel_tool_calls", "modalities", "audio", "prediction", "reasoning_effort"}


def check_chat_fields(req):
    """Reject fields Cale does not support. A field set to null counts as absent."""
    for k, v in req.items():
        if v is None or k in CHAT_FIELDS or k in NO_EFFECT_FIELDS or k in CALE_FIELDS:
            continue
        if k in ZERO_ONLY_FIELDS and v == 0:
            continue
        if k == "n" and v == 1:
            continue
        if k == "response_format" and isinstance(v, dict) and v.get("type") == "text":
            continue
        if k in ("logprobs", "parallel_tool_calls") and v is False:
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
    """Check the body of POST /cale/models. A size is exact. Goals (deadline_s, memory_pct) let Cale choose."""
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
            raise ApiError(400, "invalid_model_object", "A size is a number or smart/fast. To let Cale choose, give deadline_s or memory_pct.")
        spec["size_pct"] = pct
    deadline = number(req, "deadline_s", 0.1, 3600, None)
    memory = number(req, "memory_pct", 1, 100, None)
    max_tokens = number(req, "max_tokens", 1, MAX_TOKENS, None, True)
    if "size_pct" in spec and (deadline is not None or memory is not None):
        raise ApiError(400, "invalid_model_object", "A size is exact. Do not combine it with deadline_s or memory_pct.")
    if "size_pct" not in spec and deadline is None and memory is None:
        raise ApiError(400, "invalid_model_object", "Give a size, or a goal: deadline_s (needs max_tokens) or memory_pct.")
    if deadline is not None and max_tokens is None:
        raise ApiError(400, "max_tokens_needed", "A deadline needs 'max_tokens': Cale cannot predict the time of an answer of unknown length.")
    spec.update(deadline_s=deadline, memory_pct=memory, max_tokens=max_tokens,
                allow_disk=flag(req, "allow_disk"), on_miss=on_miss(req, "on_miss"))
    return spec


def object_slug(name):
    s = "".join(c if c.isalnum() else "-" for c in str(name or "model").lower()).strip("-")
    return (s or "model")[:30]


def ask_user(app, replacing=False):
    """Show the Allow dialog (see oslayer.ask_allow)."""
    if DEBUG and os.environ.get("CALE_AUTO_APPROVE") == "1":
        return True
    return oslayer.ask_allow(app, replacing)


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
                prof = ENGINE.saved_profile()
                this_mac = None
                if prof:
                    this_mac = {"measured_at": prof["measured_at"], "disk_gb_per_s": round(prof["disk_bps"] / 1e9, 2),
                                "sizes": {l["id"]: {"size_pct": l["pct"], **{k: prof["levels"][l["id"]][k] for k in ("decode_tps", "prefill_tps")},
                                                    "ram_gb": round(choose.model_bytes(prof, choose.need(l)) / 1e9, 2)}
                                          for l in ENGINE.levels}}
                return self._json(200, {"object": "list", "data": data, "cale": {
                    "base": BASE, "max_context_tokens": ENGINE.known_context(), "sizes": {"min": MIN_PCT, "max": MAX_PCT, "aliases": ALIASES,
                                            "available": sorted({l["pct"] for l in LEVELS}),
                                            "usage": f"{BASE}:<percent> or {BASE}:auto"}, "curve": curve,
                    "this_computer": this_mac, "this_mac": this_mac,       # this_mac: the first name, kept so old apps still work
                    "holds": ENGINE.status()["holds"]}})
            raise ApiError(404, "not_found", "Unknown path.")
        except ApiError as e:
            self._error(e)
        except Exception as e:
            print(f"[http] error {type(e).__name__}: {e}", flush=True)
            self.close_connection = True

    def do_POST(self):
        self.arrived = time.time()                 # time spent waiting in line counts toward an app's deadline
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
            if self.path == "/cale/unload":
                app = self._auth()
                ended = ENGINE.release(str(self._body().get("hold", "")), app)
                return self._json(200, {"ended": ended})
            if DEBUG and self.path == "/cale/debug/slow":
                ENGINE.slow = number(self._body(), "factor", 0.1, 100, 1.0)
                return self._json(200, {"ok": True, "slow": ENGINE.slow})
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

    def _connect(self):
        app = str(self._body().get("app", "")).strip()
        app = "".join(ch for ch in app if ch.isprintable() and ch not in '"\\')[:40]
        if not app:
            raise ApiError(400, "invalid_app", "Send {\"app\": \"Your App Name\"}.")
        if not CONNECT_LOCK.acquire(blocking=False):
            raise ApiError(429, "busy", "Another app is waiting for the user to answer. Try again later.")
        try:
            allowed = ask_user(app, replacing=app in keys.apps())
        finally:
            CONNECT_LOCK.release()
        if not allowed:
            raise ApiError(403, "denied", "The user did not allow this app.")
        self._json(200, {"app": app, "api_key": keys.create(app), "base_url": f"http://{paths.HOST}:{paths.PORT}/v1"})

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
            spec = {"memory_pct": number(req, "memory_pct", 1, 100, 25), "on_miss": on_miss(req, "on_miss")}
        try:
            hid, event, info = ENGINE.hold(app, spec)
        except GoalMissed as e:
            raise ApiError(422, "goal_missed", str(e), e.info)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self._stay_open(info, event, lambda: ENGINE.release(hid))

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
        spec = object_spec(req)
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
        oid = object_slug(req.get("name"))
        with OBJ_LOCK:
            if (app, oid) in OBJECTS:                     # the same name is alive already: this one gets a suffix
                oid = f"{oid}-{uuid.uuid4().hex[:4]}"
            OBJECTS[(app, oid)] = {"spec": spec, "created": time.time(),
                                   "max_pct": info.get("size_pct") if "memory_pct" in (hold_spec or {}) else None,   # a memory goal also caps the size this object's calls run at
                                   "memory_goal_met": info.get("goal_met") if "memory_pct" in (hold_spec or {}) else None}
        print(f"[model] {app!r} made {oid}: { {k: v for k, v in spec.items() if v is not None} }", flush=True)
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

    def _chat(self, req, app="?"):
        check_chat_fields(req)
        name = req.get("model", BASE)
        with OBJ_LOCK:
            obj = OBJECTS.get((app, name)) if isinstance(name, str) else None     # only this app's own objects exist for it
        spec = obj["spec"] if obj else None
        pct = ("auto" if "size_pct" not in spec else spec["size_pct"]) if obj else parse_model(name)
        msgs = clean_messages(req.get("messages"))
        limit_key = "max_completion_tokens" if req.get("max_completion_tokens") is not None else "max_tokens"
        asked = req.get(limit_key)
        if asked is None and obj and spec.get("max_tokens"):
            asked = spec["max_tokens"]                   # the object's answer length
        max_tokens = number({limit_key: asked}, limit_key, 1, MAX_TOKENS, DEFAULT_TOKENS, True)
        temp = number(req, "temperature", 0.0, 2.0, 0.0)
        top_p, seed, stop = chat_sampling(req)
        stream = bool(req.get("stream"))
        goal = None
        if obj and any(k in req for k in ("cale_deadline_s", "cale_allow_disk", "cale_on_miss")):
            raise ApiError(400, "goal_in_object", "This model object holds its own goals. Do not send cale_* fields with it.")
        if obj and pct == "auto":
            goal = {"deadline_s": spec["deadline_s"], "allow_disk": spec["allow_disk"], "on_miss": spec["on_miss"],
                    "expect_tokens": spec.get("max_tokens"), "max_pct": obj.get("max_pct") or 100}
        elif pct == "auto":
            deadline = number(req, "cale_deadline_s", 0.1, 3600, None)
            if deadline is not None and asked is None:
                raise ApiError(400, "max_tokens_needed", "A deadline needs 'max_tokens': Cale cannot predict the time of an answer of unknown length.")
            goal = {"deadline_s": deadline, "allow_disk": flag(req, "cale_allow_disk"), "on_miss": on_miss(req)}
        elif any(k in req for k in ("cale_deadline_s", "cale_allow_disk", "cale_on_miss")):
            raise ApiError(400, "goal_needs_auto", f"The cale_* goal fields need model '{BASE}:auto'.")
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        gen = ENGINE.chat_stream(msgs, max_tokens, temp, app, 100 if pct == "auto" else pct, goal,
                                 top_p=top_p, seed=seed, stop=stop, arrived=getattr(self, "arrived", None),
                                 max_tokens_explicit=asked is not None)
        try:
            plan = next(gen)["plan"]
        except GoalMissed as e:
            gen.close()
            raise ApiError(422, "goal_missed", str(e), e.info)
        except ContextTooLong as e:
            gen.close()
            raise ApiError(400, "context_length_exceeded", str(e), e.info, param="messages")
        except EngineError as e:
            raise ApiError(500, "engine_error", "The engine failed and was reset. Try again. The details are in Cale's log.")
        expect = plan["size_pct"]
        if obj:
            with OBJ_LOCK:                               # what this object ran at, for apps that show it
                if (app, name) in OBJECTS:
                    OBJECTS[(app, name)]["last"] = {"size_pct": expect, "goal_met": plan["goal_met"],
                                                    "predicted_s": plan["predicted_s"], "at": int(time.time())}
        hdr = {"X-Cale-Requested-Size": str(pct), "X-Cale-Size": str(expect), "X-Cale-Disk-MB": str(plan["disk_mb"]),
               "X-Cale-Waited-S": str(plan["waited_s"])}
        mem_met = obj.get("memory_goal_met") if obj else None
        if mem_met is not None:
            hdr["X-Cale-Memory-Goal-Met"] = str(mem_met).lower()
        if plan["predicted_s"] is not None:
            hdr["X-Cale-Predicted-S"] = str(plan["predicted_s"])
            hdr["X-Cale-Goal-Met"] = str(plan["goal_met"]).lower()
        info = {"requested_size": pct, "predicted_s": plan["predicted_s"], "goal_met": plan["goal_met"],
                "waited_s": plan["waited_s"], "memory_goal_met": mem_met}
        label = (lambda p: name) if obj else model_id          # a model object reports its own id as the model

        if not stream:
            text, done = [], {}
            try:
                for ev in gen:
                    if "delta" in ev:
                        text.append(ev["delta"])
                    else:
                        done = ev["done"]
            except EngineError as e:
                raise ApiError(500, "engine_error", "The engine failed and was reset. Try again. The details are in Cale's log.")
            actual = done.get("size_pct") or expect
            if "goal_met_actual" in done:                      # now the answer is finished: say what really happened
                hdr["X-Cale-Goal-Met"] = str(done["goal_met_actual"]).lower()
                info["goal_met_actual"] = done["goal_met_actual"]
            hdr["X-Cale-Total-S"] = str(done.get("total_s"))
            self._note_done(app, name, obj, done)
            self._log_chat(app, req, actual, done)
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": label(actual),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                             "finish_reason": done.get("finish", "stop")}],
                "usage": {"prompt_tokens": done.get("prompt_tokens", 0), "completion_tokens": done.get("tokens", 0),
                          "total_tokens": done.get("prompt_tokens", 0) + done.get("tokens", 0)},
                "cale": {**info, "size": actual, **{k: done.get(k) for k in ("seconds", "waited_s", "total_s", "switch_ms", "drafter", "disk_mb", "disk_s")}},
            }, hdr)

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

        def chunk(delta, finish=None, extra=None):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": label(expect),
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
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
                                 "drafter": d["drafter"], "disk_mb": d.get("disk_mb"), "disk_s": d.get("disk_s")}})
            send(b"[DONE]")
        except (BrokenPipeError, ConnectionResetError):
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
    if os.environ.get("CALE_ON_DEMAND") == "1":
        threading.Thread(target=exit_when_idle, daemon=True).start()
    print(f"[cale] listening on http://{paths.HOST}:{paths.PORT} model={paths.MODEL}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
