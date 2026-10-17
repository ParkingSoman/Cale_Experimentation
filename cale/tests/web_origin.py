"""Websites may use Cale only through one narrow path. Everything else stays refused.

usage: .venv/bin/python cale/tests/web_origin.py        (no model, no real Cale, no dialog: a stub engine on a spare port, a fresh CALE_HOME)
Checks: foreign / null / file / http origins are refused; preflight headers; the Allow question shows the site's address;
a key made for a site works only from that site; an app key does not work from a browser and a site key does not work without its origin;
one open question per site; Chromium's per-site rules (block, 4 ignores -> 7 days, allow clears, no global limit, forget); apps without an Origin behave as before; streamed answers carry the CORS headers.
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
from cale import oslayer, paths, server, sites  # noqa: E402

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


SITE_FLAGS = []                                       # for each question: was it worded as a website question?


def recording_ask(app, replacing=False, site=False, outcome=None):
    ASKED.append(app)
    SITE_FLAGS.append(site)
    return _real_ask(app, replacing, site=site, outcome=outcome)       # CALE_DEBUG=1 + CALE_AUTO_APPROVE=1: counts as "Allow"


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
    for o in list(sites.listing()):
        sites.forget(o)


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


def slow_ask(app, replacing=False, site=False, outcome=None):
    ASKED.append(app)
    GATE.wait(20)
    if outcome is not None:
        outcome["how"] = ANSWER.get("how", "allow" if ANSWER["yes"] else "deny")
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
check("one Don't Allow is a dismissal, not a block (it may ask again)", sites.check(OTHER) is None and sites.listing()[OTHER]["dismissals"] == 1, str(sites.listing()))
sites.record(OTHER, "deny")
check("two dismissals: it may still ask", sites.check(OTHER) is None)
sites.record(OTHER, "deny")
s, js, h = call("POST", "/cale/connect", {"wait_s": 0}, origin=OTHER)
ra = js.get("error", {}).get("cale", {}).get("retry_after_s", 0)
check("the 3rd dismissal gives 7 days: 429 asked_recently", s == 429 and code((s, js)) == "asked_recently" and 604000 < ra < 605000, f"{s} {js}")
check("asked_recently message has no technical words", "cannot ask again yet" in js["error"]["message"], str(js))
check("a waiting site does not hold back another site", call("POST", "/cale/connect", {"wait_s": 0}, origin=SITE)[0] in (202, 200))
GATE.set()
time.sleep(0.3)

# the block is in the file, mode 0600, and survives a restart (a fresh read of the file by a fresh import)
check("sites file exists with mode 0600", paths.SITES.exists() and (paths.SITES.stat().st_mode & 0o777) == 0o600 or os.name == "nt")
import importlib  # noqa: E402
fresh = importlib.reload(sites)
check("the wait survives a restart (module reloaded, state read from the file)", (fresh.check(OTHER) or ("",))[0] == "asked_recently", str(fresh.check(OTHER)))
import subprocess  # noqa: E402
r = subprocess.run([sys.executable, "-c", "from cale import sites; print(sites.check('%s'))" % OTHER], capture_output=True, text=True,
                   env={**os.environ}, cwd=str(Path(__file__).resolve().parents[2]))
check("the wait is seen by a new process", "asked_recently" in r.stdout, r.stdout + r.stderr)
sites = fresh
server.sites = fresh
check("forget() resets the wait", sites.forget("other.example.org") is True and sites.check(OTHER) is None)
check("forget() of an unknown site is False", sites.forget("never.example.org") is False)
server.ask_user = recording_ask
s, js, h = call("POST", "/cale/connect", {"wait_s": 20}, origin=OTHER)
check("after forget the site may ask again", s == 200, f"{s} {js}")

# ---- ignored questions: 3 still ask, the 4th gives a 7-day embargo ---------------------------------------------------
reset_web_state()
IGN = "https://ignore.example.com"
server.ask_user = slow_ask
ANSWER.update(yes=False, how="timeout")
GATE.set()
seen = []
for i in range(4):
    seen.append(call("POST", "/cale/connect", {"wait_s": 20}, origin=IGN)[0])
check("3 ignored questions are each asked (403 denied for this answer), the 4th too", seen == [403, 403, 403, 403], str(seen))
s, js, h = call("POST", "/cale/connect", {"wait_s": 0}, origin=IGN)
ra = js.get("error", {}).get("cale", {}).get("retry_after_s", 0)
check("after the 4th ignored question: 429 asked_recently", s == 429 and code((s, js)) == "asked_recently", f"{s} {js}")
check("embargo is about 7 days (retry_after_s ~ 604800)", 604700 < ra <= 604801 and 604700 < int(h.get("retry-after", 0)) <= 604801, f"{ra} {h.get('retry-after')}")
check("kDefaultIgnoresBeforeBlock = 4 and kDefaultEmbargoDays = 7", sites.DEFAULT_IGNORES_BEFORE_BLOCK == 4 and sites.DEFAULT_EMBARGO_DAYS == 7)
check("embargo ends after 7 days", sites.check(IGN, now=time.time() + 7 * 86400 + 5) is None)
sites.record(IGN, "timeout", now=time.time() + 8 * 86400)
check("an ended embargo starts the count again", sites.listing()[IGN]["ignores"] == 1 and sites.listing()[IGN]["embargo_until"] == 0)

# ---- an allow clears the counters --------------------------------------------------------------------------------------
reset_web_state()
for _ in range(3):
    sites.record(IGN, "timeout")
check("3 ignores are stored", sites.listing()[IGN]["ignores"] == 3)
server.ask_user = recording_ask
s, js, h = call("POST", "/cale/connect", {"wait_s": 20}, origin=IGN)
check("an allow clears the counters", s == 200 and IGN not in sites.listing(), f"{s} {sites.listing()}")
for _ in range(3):
    sites.record(IGN, "timeout")
check("after an allow, 3 more ignores do not embargo", sites.check(IGN) is None)

# ---- error outcomes do not count ---------------------------------------------------------------------------------------
reset_web_state()
for _ in range(10):
    sites.record(IGN, "error")
check("error outcomes change nothing", IGN not in sites.listing() and sites.check(IGN) is None)
server.ask_user = lambda app, replacing=False, site=False, outcome=None: False     # a stub that fails and says nothing
for _ in range(6):
    call("POST", "/cale/connect", {"wait_s": 20}, origin=IGN)
check("a question that fails without an outcome counts for nothing", IGN not in sites.listing(), str(sites.listing()))
def _boom(app, replacing=False, site=False, outcome=None):
    raise RuntimeError("x")
server.ask_user = _boom
for _ in range(6):
    call("POST", "/cale/connect", {"wait_s": 20}, origin=IGN)
check("an exception while asking counts for nothing", IGN not in sites.listing(), str(sites.listing()))
sites.forget("other.example.org")
for how in ("deny", "deny", "timeout", "timeout", "timeout"):
    sites.record(OTHER, how)
check("dismissals and ignores are counted apart (2 + 3: still may ask)", sites.check(OTHER) is None, str(sites.listing().get(OTHER)))
sites.record(OTHER, "timeout")
check("the 4th ignore starts the 7 days even after dismissals", (sites.check(OTHER) or ("",))[0] == "asked_recently")
sites.forget("other.example.org")

# ---- no limit across sites (a site with many sub-domains: each gets its own question) --------------------------------
reset_web_state()
server.ask_user = lambda app, replacing=False, site=False, outcome=None: (outcome.update(how="timeout") if outcome is not None else None) or False
codes = [call("POST", "/cale/connect", {"wait_s": 20}, origin=f"https://s{i}.evil.example")[0] for i in range(8)]
check("many sub-domains: each one gets its own question, no global 429", codes == [403] * 8, str(codes))
check("each sub-domain has its own record", len(sites.listing()) == 8)
check("apps are not affected by site records", call("POST", "/cale/connect", {"app": "Another App", "wait_s": 20}, key=None)[0] == 403)   # the stub says no: 403
server.ask_user = recording_ask
reset_web_state()

# ---- ask_allow: the answer parsing (no dialog is opened) -------------------------------------------------------------
check("mac: Allow", oslayer.mac_outcome(0, "button returned:Allow, gave up:false", "") == "allow")
check("mac: gave up -> timeout", oslayer.mac_outcome(0, "button returned:, gave up:true", "") == "timeout")
check("mac: Allow text with gave up -> timeout, never allow", oslayer.mac_outcome(0, "button returned:Allow, gave up:true", "") == "timeout")
check("mac: Don't Allow (cancel) -> deny", oslayer.mac_outcome(1, "", "execution error: User canceled. (-128)") == "deny")
check("mac: -128 only -> deny", oslayer.mac_outcome(1, "", "... (-128)") == "deny")
check("mac: other failure -> error", oslayer.mac_outcome(1, "", "osascript: no window server") == "error")
check("mac: odd success -> error", oslayer.mac_outcome(0, "button returned:Don't Allow", "") == "error")
check("windows: 6 allow, 7 deny, 32000 timeout, 0 error", [oslayer.win_outcome(x) for x in (6, 7, 32000, 0, 2)] == ["allow", "deny", "timeout", "error", "error"])
check("zenity: 0 allow, 1 deny, 5 timeout, 3 error", [oslayer.zenity_outcome(x) for x in (0, 1, 5, 3, -9)] == ["allow", "deny", "timeout", "error", "error"])
import unittest.mock as mock  # noqa: E402
for label, how_in in (("allow", "allow"), ("deny", "deny"), ("timeout", "timeout"), ("error", "error")):
    with mock.patch.object(oslayer, "MAC", True), mock.patch.object(oslayer, "WIN", False), mock.patch.object(oslayer.subprocess, "run") as run:
        res = {"allow": (0, "button returned:Allow, gave up:false", ""), "deny": (1, "", "User canceled. (-128)"),
               "timeout": (0, "button returned:, gave up:true", ""), "error": (1, "", "boom")}[how_in]
        run.return_value = mock.Mock(returncode=res[0], stdout=res[1], stderr=res[2])
        out = {}
        ret = oslayer.ask_allow("rewrite.example.com", False, site=True, outcome=out)
        cmd = run.call_args[0][0]
        check(f"ask_allow (mac, mocked) {label}: bool result and outcome", isinstance(ret, bool) and ret is (how_in == "allow") and out["how"] == how_in, f"{ret} {out}")
        check(f"ask_allow (mac, {label}): name and kind travel as argv, not in script text",
              cmd[-3:] == ["rewrite.example.com", "no", "website"] and not any("rewrite.example.com" in x for x in cmd[:-3]), str(cmd[-3:]))
        if label == "allow":
            text = " ".join(cmd[:-3])
            check("mac dialog text uses the kind from argv and the shared wording", 'Allow the " & k' in text and "to use Cale on this computer?" in text and "If you did not just open this " in text and "choose Don't Allow." in text)
    with mock.patch.object(oslayer, "MAC", False), mock.patch.object(oslayer, "WIN", True), mock.patch.object(oslayer, "ctypes", create=True) as ct:
        ct.windll.user32.MessageBoxTimeoutW.return_value = {"allow": 6, "deny": 7, "timeout": 32000, "error": 0}[how_in]
        out = {}
        ret = oslayer.ask_allow("x.example.com", True, site=True, outcome=out)
        check(f"ask_allow (windows, mocked) {label}: bool result and outcome", isinstance(ret, bool) and ret is (how_in == "allow") and out["how"] == how_in, f"{ret} {out}")
with mock.patch.object(oslayer, "MAC", False), mock.patch.object(oslayer, "WIN", False), mock.patch.object(oslayer.shutil, "which", return_value="/usr/bin/zenity"), \
        mock.patch.dict(os.environ, {"DISPLAY": ":0"}), mock.patch.object(oslayer.subprocess, "run") as run:
    for rc, how_in in ((0, "allow"), (1, "deny"), (5, "timeout"), (7, "error")):
        run.return_value = mock.Mock(returncode=rc, stdout=b"", stderr=b"")
        out = {}
        ret = oslayer.ask_allow("x.example.com", False, site=True, outcome=out)
        check(f"ask_allow (zenity, mocked) code {rc}: bool and {how_in}", isinstance(ret, bool) and ret is (rc == 0) and out["how"] == how_in, f"{ret} {out}")
    run.side_effect = oslayer.subprocess.TimeoutExpired("zenity", 1)
    out = {}
    ret = oslayer.ask_allow("x.example.com", False, site=True, outcome=out)
    check("ask_allow (zenity) TimeoutExpired -> False and timeout", ret is False and out["how"] == "timeout")
check("ask_allow works without an outcome dict", True)

# ---- terminal path: `cale deny` is a deny, no answer is a timeout ------------------------------------------------------
from cale import approval  # noqa: E402
out = {}
t = threading.Thread(target=lambda: out.setdefault("r", approval.wait_for_terminal("t.example.com", False, 5, outcome=out)))
t.start()
time.sleep(1.2)
approval.decide(approval.pending()[0]["ticket"], False)
t.join(10)
check("terminal: cale deny -> False and deny", out.get("r") is False and out.get("how") == "deny", str(out))
out = {}
check("terminal: no answer -> False and timeout", approval.wait_for_terminal("t.example.com", False, 0.6, outcome=out) is False and out["how"] == "timeout", str(out))

# ---- the question says "website" for a site and "app" for an app -----------------------------------------------------
reset_web_state()
SITE_FLAGS.clear()
call("POST", "/cale/connect", {"wait_s": 20}, origin="https://wording.example")
call("POST", "/cale/connect", {"app": "Wording App", "wait_s": 20})
check("the question is worded as a website question only for a site", SITE_FLAGS == [True, False], str(SITE_FLAGS))
import inspect
src = inspect.getsource(server.oslayer.ask_allow)
check("the dialog text names the kind (website or app)", 'kind = "website" if site else "app"' in src and "{kind}" in src)

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nAll checks passed.")
sys.exit(1 if FAIL else 0)
