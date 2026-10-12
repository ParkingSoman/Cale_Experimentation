"""Drive the Open WebUI demo without a browser: the same routes the "Get Cale" card and the chat page call.
Works on Mac, Windows and Linux. Start Open WebUI first (apps/open-webui/cale_webui.py).

usage: python scripts/demo_drive.py [OWUI_URL]      (default http://127.0.0.1:8810)
Steps: status -> install (if Cale is missing) -> connect -> wait for the model objects -> chat -> show what size Cale chose.
"""
import json
import sys
import time
import urllib.error
import urllib.request

base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8810").rstrip("/")


def call(path, body=None, token=None, timeout=900, method=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    req = urllib.request.Request(base + path, data=None if body is None and method != "POST" else json.dumps(body or {}).encode(),
                                 headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def status():
    try:
        return call("/api/cale/status", timeout=8)
    except (urllib.error.URLError, OSError, ValueError):
        return {"state": "no answer", "install": {}, "models": []}


def stamp(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


s = status()
stamp(f"card state: {s['state']}")
if s["state"] == "missing":
    stamp("click: Get Cale")
    call("/api/cale/install", method="POST")
    last = None
    while True:
        s = status()
        i = s.get("install", {})
        line = f"{s['state']} {i.get('stage', '')} {i.get('pct', '')}%" + (f" {i['done'] / 1e9:.2f}/{i['total'] / 1e9:.2f} GB" if i.get("total") else "")
        if line != last and (s["state"] != "installing" or i.get("pct", 0) % 10 == 0 or s["state"] != "installing"):
            stamp(line)
            last = line
        if s["state"] in ("needs_connect", "failed", "ready"):
            break
        time.sleep(3)
    if s["state"] == "failed":
        sys.exit(f"install failed: {s['install'].get('error')}")
if s["state"] in ("starting", "no answer"):
    for _ in range(60):
        time.sleep(3)
        s = status()
        if s["state"] not in ("starting", "no answer"):
            break
if s["state"] == "needs_connect":
    stamp("click: Connect Open WebUI to Cale (the computer asks Allow; in test mode it is approved)")
    r = call("/api/cale/connect", method="POST")
    if not r.get("ok"):
        sys.exit(f"connect failed: {r}")
for _ in range(80):
    s = status()
    if s["state"] == "ready" and len(s["models"]) >= 2:
        break
    time.sleep(3)
stamp("models: " + ", ".join(m["id"] for m in s["models"]))

tok = call("/api/v1/auths/signin", {"email": "admin@localhost", "password": "admin"})["token"]
t = time.time()
r = call("/api/chat/completions", {"model": "answers", "stream": False,
                                   "messages": [{"role": "user", "content": "In two sentences, why is the sky blue?"}]}, tok)
stamp(f"chat 'answers' took {time.time() - t:.1f} s: " + r["choices"][0]["message"]["content"].strip()[:160].replace("\n", " "))
t = time.time()
r = call("/api/v1/tasks/title/completions", {"model": "tasks", "stream": False,
                                             "messages": [{"role": "user", "content": "In two sentences, why is the sky blue?"},
                                                          {"role": "assistant", "content": "Because of Rayleigh scattering."}]}, tok)
stamp(f"title 'tasks' took {time.time() - t:.1f} s: " + r["choices"][0]["message"]["content"].strip()[:120].replace("\n", " "))
for m in call("/api/cale/status")["models"]:
    last = m.get("last") or {}
    print(f"  {m['id']:8} ran the {last.get('size_pct')}% size, goal {'met' if last.get('goal_met') else 'NOT met'}, "
          f"predicted {last.get('predicted_s')} s, took {last.get('seconds')} s for {last.get('tokens')} tokens", flush=True)
print("DEMO_DRIVE_DONE", flush=True)
