"""Websites may use Cale only through one narrow path. Everything else stays refused.

usage: .venv/bin/python cale/tests/web_origin.py        (no model, no real Cale, no dialog: a stub engine on a spare port, a fresh CALE_HOME)
Checks: foreign / null / file / http origins are refused; preflight headers; the Allow question shows the site's address;
a key made for a site works only from that site; an app key does not work from a browser and a site key does not work without its origin;
one open question per site; cooldown after "Don't Allow"; apps without an Origin behave as before; streamed answers carry the CORS headers.
"""
import http.client
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

HOME = tempfile.mkdtemp(prefix="cale-web-test-")
os.environ["CALE_HOME"] = HOME
os.environ["CALE_DEBUG"] = "1"
os.environ["CALE_AUTO_APPROVE"] = "1"
os.environ["CALE_WEB_DEV_ORIGINS"] = "http://127.0.0.1:5555"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import server  # noqa: E402

FAIL = []
SITE, OTHER = "https://rewrite.example.com", "https://other.example.org"


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


class StubEngine:
    """Just enough of cale.engine.Engine for the routes under test. No model is loaded."""
    def __init__(self):
        self.replace_stop = threading.Event()
        self.mu = threading.Lock()
        self.holds = {}

    def status(self):
        return {"loaded": False, "holds": [{"hold": "h1", "app": "Some Other App", "size_pct": 50, "age_s": 1}], "pressure": "normal"}

    def known_context(self):
        return 4096

    def this_computer(self):
        return {"ram_gb": 16}

    def predict(self, goals, prompt_tokens, answer_tokens, cap=100):
        return {"goal_met": True, "size_pct": 50}

    def release(self, hid, app=None):
        return False

    def chat_stream(self, msgs, max_tokens, temp, app="?", size_pct=100, goal=None, **kw):
        yield {"plan": {"size_pct": 50, "disk_mb": 0, "waited_s": 0, "predicted": {"tokens_per_s": 9, "first_token_s": 1}, "confidence": "high",
                        "goal_met": True, "unmet": [], "max_tokens": 50}}
        yield {"delta": "Hello"}
        yield {"delta": " there"}
        yield {"done": {"size_pct": 50, "finish": "stop", "prompt_tokens": 3, "tokens": 2, "seconds": 0.1, "drafter": False}}


server.ENGINE = StubEngine()
server.Handler.log_message = lambda self, fmt, *a: None      # quiet
srv = server.Server(("127.0.0.1", 0), server.Handler)
srv.daemon_threads = True
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

ASKED = []                                            # what the person was asked: the text shown in the Allow question
_real_ask = server.ask_user


def recording_ask(app, replacing=False):
    ASKED.append(app)
    return _real_ask(app, replacing)                  # CALE_DEBUG=1 + CALE_AUTO_APPROVE=1: counts as "Allow"


server.ask_user = recording_ask


def call(method, path, body=None, origin=None, key=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    h = {"Host": host or f"127.0.0.1:{PORT}"}
    if origin is not None:
        h["Origin"] = origin
    if key:
        h["Authorization"] = "Bearer " + key
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    c.request(method, path, data, h)
    r = c.getresponse()
    raw = r.read()
    try:
        js = json.loads(raw) if raw else {}
    except ValueError:
        js = {"raw": raw.decode(errors="replace")}
    out = (r.status, js, {k.lower(): v for k, v in r.getheaders()})
    c.close()
    return out


def code(res):
    return (res[1].get("error") or {}).get("code")


def reset_web_state():
    server.WEB_STATE["until"].clear()
    server.WEB_STATE["asked"].clear()


# ---- apps (no Origin) behave as before ----------------------------------------------------------------------------
s, js, h = call("POST", "/cale/connect", {"app": "Plain App"})
check("app connect works without Origin", s == 200 and js["api_key"].startswith("cale-"), f"{s} {js}")
APP_KEY = js["api_key"]
check("app answer has no CORS headers", "access-control-allow-origin" not in h)
check("app: status still full", call("GET", "/cale/status")[1].get("holds") is not None)
check("app: key works (predict)", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, key=APP_KEY)[0] == 200)
check("app: no key -> 401", call("POST", "/cale/predict", {"min_tokens_per_s": 6})[0] == 401)
check("app: /cale/computer works without key", call("GET", "/cale/computer")[0] == 200)
check("app: bad Host still 403", call("GET", "/cale/status", host="localhost:%d" % PORT)[0] == 403)
check("app: Sec-Fetch-Site without Origin still 403", call("GET", "/cale/status", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403)
check("app: name starting web: is refused", call("POST", "/cale/connect", {"app": "Web:https://rewrite.example.com"})[0] == 400)
check("app: debug route still works", call("POST", "/cale/debug/slow", {"factor": 1.0})[0] == 200)

# ---- refused origins -----------------------------------------------------------------------------------------------
for bad in ("null", "file://", "http://evil.example", "http://127.0.0.1:9999", "https://localhost", "https://evil.example/path",
            "https://EVIL.example", "https://evil.example.", "https://user@evil.example", "", "chrome-extension://abc", "https://a b.example"):
    for path, method, body in (("/cale/status", "GET", None), ("/cale/connect", "POST", {}), ("/cale/predict", "POST", {"min_tokens_per_s": 6})):
        res = call(method, path, body, origin=bad, key=APP_KEY)
        check(f"origin {bad!r} -> 403 on {path}", res[0] == 403 and "access-control-allow-origin" not in res[2], f"{res[0]}")
    check(f"origin {bad!r} preflight -> 403", call("OPTIONS", "/cale/connect", origin=bad, headers={"Access-Control-Request-Method": "POST"})[0] == 403)
check("dev origin is refused unless listed", call("GET", "/cale/status", origin="http://127.0.0.1:5556")[0] == 403)
check("listed dev origin is accepted", call("GET", "/cale/status", origin="http://127.0.0.1:5555")[0] == 200)
import socket  # noqa: E402
raw = socket.create_connection(("127.0.0.1", PORT))
raw.sendall(f"GET /cale/status HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nOrigin: {SITE}\r\nOrigin: http://evil.example\r\nConnection: close\r\n\r\n".encode())
check("two Origin headers -> 403", raw.recv(100).split(b" ")[1] == b"403")
raw.close()
check("OPTIONS without Origin -> 403", call("OPTIONS", "/cale/connect", headers={"Access-Control-Request-Method": "POST"})[0] == 403)

# ---- preflight --------------------------------------------------------------------------------------------------------
s, js, h = call("OPTIONS", "/v1/chat/completions", origin=SITE, headers={"Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type", "Access-Control-Request-Private-Network": "true"})
check("preflight 204", s == 204, str(s))
check("preflight: exact origin, not *", h.get("access-control-allow-origin") == SITE)
check("preflight: Vary Origin", "origin" in h.get("vary", "").lower())
check("preflight: private network", h.get("access-control-allow-private-network") == "true")
check("preflight: methods", all(m in h.get("access-control-allow-methods", "") for m in ("GET", "POST", "DELETE", "OPTIONS")))
check("preflight: headers", "authorization" in h.get("access-control-allow-headers", "").lower() and "content-type" in h.get("access-control-allow-headers", "").lower())
check("preflight with PUT -> 403", call("OPTIONS", "/cale/connect", origin=SITE, headers={"Access-Control-Request-Method": "PUT"})[0] == 403)
check("preflight with bad Host -> 403", call("OPTIONS", "/cale/connect", origin=SITE, host="evil.example", headers={"Access-Control-Request-Method": "POST"})[0] == 403)

# ---- a site with no key ---------------------------------------------------------------------------------------------
s, js, h = call("GET", "/cale/status", origin=SITE)
check("site: status answers with CORS headers", s == 200 and h.get("access-control-allow-origin") == SITE and "origin" in h.get("vary", "").lower())
check("site: status shows nothing about other apps", "holds" not in js and js.get("ok") is True, str(js))
for path, method, body in (("/cale/predict", "POST", {"min_tokens_per_s": 6}), ("/v1/models", "GET", None), ("/cale/computer", "GET", None),
                           ("/v1/chat/completions", "POST", {"messages": []}), ("/cale/models", "POST", {"min_tokens_per_s": 6}),
                           ("/cale/activity", "GET", None), ("/cale/load", "POST", {"size": "fast"})):
    res = call(method, path, body, origin=SITE)
    check(f"site without key -> 401 on {path}", res[0] == 401 and res[2].get("access-control-allow-origin") == SITE, str(res[0]))
check("site: app key from a browser origin -> 403", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, origin=SITE, key=APP_KEY)[0] == 403)
check("site: debug route -> 403", call("POST", "/cale/debug/slow", {"factor": 1.0}, origin=SITE)[0] == 403)

# ---- connect from a site: the question shows the address --------------------------------------------------------------
reset_web_state()
ASKED.clear()
s, js, h = call("POST", "/cale/connect", {"app": "Totally Your Bank", "wait_s": 20}, origin=SITE)
check("site connect -> 200", s == 200 and js.get("api_key", "").startswith("cale-"), f"{s} {js}")
check("question shows the site address, not the name the page sent", ASKED == ["rewrite.example.com"], str(ASKED))
check("answer names the address", js.get("app") == "rewrite.example.com" and js.get("origin") == SITE)
check("connect answer has CORS headers", h.get("access-control-allow-origin") == SITE)
WEB_KEY = js["api_key"]
rec = json.loads(Path(HOME, "keys.json").read_text())["apps"]
check("key record is bound to the origin", rec.get("web:" + SITE, {}).get("origin") == SITE and "Totally Your Bank" not in rec)
check("key file holds only a hash", WEB_KEY not in Path(HOME, "keys.json").read_text())

# ---- keys are bound to the origin -------------------------------------------------------------------------------------
check("web key from its own origin works", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, origin=SITE, key=WEB_KEY)[0] == 200)
check("web key from another https origin -> 403", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, origin=OTHER, key=WEB_KEY)[0] == 403)
check("web key from an http origin -> 403", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, origin="http://evil.example", key=WEB_KEY)[0] == 403)
check("web key from null origin -> 403", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, origin="null", key=WEB_KEY)[0] == 403)
check("web key without Origin -> 403", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, key=WEB_KEY)[0] == 403)
s, js, h = call("GET", "/v1/models", origin=SITE, key=WEB_KEY)
check("web key: /v1/models works, holds of other apps hidden", s == 200 and js["cale"]["holds"] == [], str(s))
check("web key: /cale/computer works", call("GET", "/cale/computer", origin=SITE, key=WEB_KEY)[0] == 200)
check("web key cannot read the Hub's activity", call("GET", "/cale/activity", origin=SITE, key=WEB_KEY)[0] == 403)
check("app key still works", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, key=APP_KEY)[0] == 200)

# ---- model object + streamed answer from the site ---------------------------------------------------------------------
s, js, h = call("POST", "/cale/models", {"name": "rewriter", "min_tokens_per_s": 6, "max_tokens": 120, "lease_s": 30}, origin=SITE, key=WEB_KEY)
check("site makes a model object (goals)", s == 200 and js.get("id") == "rewriter", f"{s} {js}")
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
c.request("POST", "/v1/chat/completions", json.dumps({"model": "rewriter", "stream": True, "messages": [{"role": "user", "content": "x"}]}),
          {"Host": f"127.0.0.1:{PORT}", "Origin": SITE, "Authorization": "Bearer " + WEB_KEY, "Content-Type": "application/json"})
r = c.getresponse()
body = r.read().decode()
check("streamed answer has CORS headers", r.status == 200 and r.getheader("Access-Control-Allow-Origin") == SITE and "Origin" in (r.getheader("Vary") or ""), str(r.status))
check("streamed answer exposes the speed headers", "X-Cale-Size" in (r.getheader("Access-Control-Expose-Headers") or ""))
check("streamed answer arrives", "Hello" in body and "[DONE]" in body)
c.close()
check("the object is the site's: an app cannot use it", call("POST", "/v1/chat/completions", {"model": "rewriter", "messages": [{"role": "user", "content": "x"}]}, key=APP_KEY)[0] != 200)
s, js, h = call("DELETE", "/cale/models/rewriter", origin=SITE, key=WEB_KEY)
check("DELETE from the site works", s == 200 and h.get("access-control-allow-origin") == SITE, f"{s} {js}")
# the open-connection form (like cale.client.model): first line arrives, then the page keeps the connection
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
c.request("POST", "/cale/models", json.dumps({"name": "held", "min_tokens_per_s": 6}),
          {"Host": f"127.0.0.1:{PORT}", "Origin": SITE, "Authorization": "Bearer " + WEB_KEY, "Content-Type": "application/json"})
r = c.getresponse()
first = json.loads(r.fp.readline())
check("open-connection model object: CORS headers and first line", r.status == 200 and r.getheader("Access-Control-Allow-Origin") == SITE and first.get("id") == "held")
c.close()
time.sleep(0.5)

# ---- a second connect replaces the key (as for apps) -----------------------------------------------------------------
s, js, h = call("POST", "/cale/connect", {"wait_s": 20}, origin=SITE)
check("same site asks again -> new key", s == 200 and js["api_key"] != WEB_KEY, str(s))
check("the old key of that site stops working", call("POST", "/cale/predict", {"min_tokens_per_s": 6}, origin=SITE, key=WEB_KEY)[0] == 401)

# ---- tickets belong to their caller ---------------------------------------------------------------------------------
reset_web_state()
GATE = threading.Event()
ANSWER = {"yes": False}


def slow_ask(app, replacing=False):
    ASKED.append(app)
    GATE.wait(20)
    return ANSWER["yes"]


server.ask_user = slow_ask
ASKED.clear()
s, js, h = call("POST", "/cale/connect", {"wait_s": 0}, origin=OTHER)
check("slow question -> 202 pending with a ticket", s == 202 and "ticket" in js, f"{s} {js}")
ticket = js.get("ticket", "")
check("one open question per site: second ask is refused (409, or 429 while a dialog is open)", call("POST", "/cale/connect", {"wait_s": 0}, origin=OTHER)[0] in (409, 429))
time.sleep(0.3)
check("only one question was shown", ASKED == ["other.example.org"], str(ASKED))
check("another site cannot use the ticket", call("GET", f"/cale/connect/status?ticket={ticket}", origin=SITE)[0] == 404)
check("an app cannot use the ticket", call("GET", f"/cale/connect/status?ticket={ticket}")[0] == 404)
s, js, h = call("GET", f"/cale/connect/status?ticket={ticket}", origin=OTHER)
check("the site can poll its own ticket (202)", s == 202 and h.get("access-control-allow-origin") == OTHER, str(s))
ANSWER["yes"] = False
GATE.set()
for _ in range(50):
    s, js, h = call("GET", f"/cale/connect/status?ticket={ticket}", origin=OTHER)
    if s != 202:
        break
    time.sleep(0.1)
check("Don't Allow -> 403 denied with CORS headers", s == 403 and code((s, js)) == "denied" and h.get("access-control-allow-origin") == OTHER, f"{s} {js}")
s, js, h = call("POST", "/cale/connect", {"wait_s": 0}, origin=OTHER)
check("cooldown after Don't Allow -> 429 with Retry-After", s == 429 and code((s, js)) == "asked_recently" and int(h.get("retry-after", 0)) > 0, f"{s} {js} {h.get('retry-after')}")
check("cooldown is 10 minutes", 590 < int(h.get("retry-after", 0)) <= 601, h.get("retry-after"))
check("cooldown does not block another site", call("POST", "/cale/connect", {"wait_s": 0}, origin=SITE)[0] in (202, 200))
GATE.set()
time.sleep(0.3)
server.WEB_STATE["until"][OTHER] = time.time() - 1
server.ask_user = recording_ask
s, js, h = call("POST", "/cale/connect", {"wait_s": 20}, origin=OTHER)
check("after the cooldown the site may ask again", s == 200, f"{s} {js}")

# ---- global limit (a site with many sub-domains) -----------------------------------------------------------------------
reset_web_state()
server.ask_user = lambda app, replacing=False: False
codes = []
for i in range(server.WEB_MAX_ASKS + 1):
    codes.append(call("POST", "/cale/connect", {"wait_s": 20}, origin=f"https://s{i}.evil.example")[0])
check("many sub-domains: first questions are asked, then 429", codes[:server.WEB_MAX_ASKS] == [403] * server.WEB_MAX_ASKS and codes[-1] == 429, str(codes))
check("apps are not limited by the web limit", call("POST", "/cale/connect", {"app": "Another App", "wait_s": 20}, key=None)[0] == 403)   # the stub says no: 403, not 429
server.ask_user = recording_ask
reset_web_state()

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nAll checks passed.")
sys.exit(1 if FAIL else 0)
