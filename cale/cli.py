"""The `cale` command. Standard library only, so it starts fast.

  cale status         is Cale running, is the model loaded
  cale apps           apps that hold a key
  cale revoke NAME    take a key away
  cale tune                                measure this computer again (done once when Cale is installed)
  cale start [--wait] | stop | restart     --wait: return when Cale answers
  cale logs           follow the log
  cale computer       what Cale predicts for this computer (speed and memory of every size)
  cale pending        apps that wait for you to allow them (no screen on this computer)
  cale allow NAME     allow one (type this in a terminal on this computer). cale deny NAME says no.
  cale status --json  the same as status, for programs
"""
import sys
import time

from . import approval, keys, oslayer, paths, service


def status(as_json=False):
    s = service.status()
    if as_json:
        import json
        print(json.dumps({"running": s is not None, **(s or {})}))
        return 0 if s is not None else 1
    if s is None:
        print("Cale is not running. Type: cale start")
        return 1
    print("Cale is running.")
    print(f"  model loaded : {'yes' if s['loaded'] else 'no (it loads when an app first asks)'}")
    if s["loaded"]:
        print(f"  memory in use: {s['resident_gb']} GB, size cap {s['cap_pct']}%")
    print(f"  memory state : {s['pressure']}")
    for e in s["events"][-5:]:
        print(f"  {e}")
    return 0


def apps():
    held = keys.apps()
    if not held:
        print("No app has a key yet.")
    for a, t in held.items():
        print(f"  {a}  (since {time.strftime('%Y-%m-%d %H:%M', time.localtime(t))})")
    return 0


def revoke(name):
    ok = keys.revoke(name)
    print(f"Removed the key for '{name}'." if ok else f"No app named '{name}'.")
    return 0 if ok else 1


def start(wait=False):
    ok, msg = service.start(60 if wait else 5)
    print(msg)
    return 0 if ok else 1


def stop():
    ok = service.stop()
    print("Cale is stopped." if ok else "Cale did not stop. Look at the log: " + str(paths.LOGS / "cale.log"))
    return 0 if ok else 1


def computer():
    """What Cale predicts for this computer: the speed and memory of every size. Starts Cale if needed. Loads the model on first use.
    The command uses its own key (named "cale command"), like any app, because loading the model is not free for anyone to ask."""
    import json
    import urllib.request
    ok, msg = service.start(60)
    if not ok:
        print(msg)
        return 1
    key = keys.create("cale command")
    print("Asking Cale (the model loads first if it is not loaded yet)...", flush=True)

    def call(path, body=None):
        req = urllib.request.Request(f"http://{paths.HOST}:{paths.PORT}{path}", data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())
    call("/cale/predict", {"prompt_tokens": 200, "max_tokens": 256})          # loads the model
    d = call("/v1/models")["cale"]
    c = d.get("this_computer")
    if not c:
        print("Cale has nothing to show yet.")
        return 1
    print(f"Engine placement : {c.get('placement')}")
    print(f"Memory speed     : {c.get('memory_gb_s')} GB/s     disk: {c.get('disk_gb_per_s')} GB/s     sure: {c.get('confidence')}")
    print(f"Context limit    : {d.get('max_context_tokens')} tokens")
    print(f"{'size':>14} {'%':>4} {'answer tok/s':>13} {'prompt tok/s':>13} {'needs GB':>9} {'quality':>8}")
    for sid, v in c["sizes"].items():
        print(f"{sid:>14} {v['size_pct']:>4} {v['decode_tps']:>13} {v['prefill_tps']:>13} {v.get('needs_gb', ''):>9} {v.get('quality', ''):>8}")
    f = c.get("floor") or {}
    if f:
        print(f"The most this computer can do: fastest {f['fastest']['size']} at {f['fastest']['tokens_per_s']} tok/s; smallest needs {f['smallest']['needs_gb']} GB.")
    return 0


def pending():
    p = approval.pending()
    if not p:
        print("No app is waiting.")
    for x in p:
        print(f"  {x['app']}  (waiting {int(time.time()) - x['since']} s)  ticket {x['ticket']}")
    return 0


def answer(name, allow):
    import os
    verb = "allow" if allow else "deny"
    if not (oslayer.is_terminal(sys.stdin) and oslayer.is_terminal(sys.stdout)) and os.environ.get("CALE_ALLOW_NON_INTERACTIVE") != "1":
        print(f"cale {verb} must be typed in a terminal, so that a person answers. (CALE_ALLOW_NON_INTERACTIVE=1 skips this check, for tests.)")
        return 2
    found = approval.matches(name)
    if not found:
        print(f"No app named '{name}' is waiting. Type: cale pending")
        return 1
    if len(found) > 1:
        print(f"{len(found)} requests are waiting under the name '{name}'. Use the ticket of the one you mean:")
        for x in found:
            print(f"  cale {verb} {x['ticket']}     ({x['app']}, waiting {int(time.time()) - x['since']} s)")
        return 1
    x = found[0]
    if allow and oslayer.is_terminal(sys.stdin):                       # the question shows what is waiting, not what was typed
        if input(f'Allow "{x["app"]}" (ticket {x["ticket"]}) to run the local AI model on this computer? [y/N] ').strip().lower() not in ("y", "yes"):
            print("Not allowed.")
            return 1
    approval.decide(x["ticket"], allow)
    print(f"{'Allowed' if allow else 'Denied'} '{x['app']}'.")
    return 0


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    cmd = a[0] if a else "status"
    if cmd == "status":
        return status("--json" in a)
    if cmd == "apps":
        return apps()
    if cmd == "revoke" and len(a) == 2:
        return revoke(a[1])
    if cmd == "start":
        return start("--wait" in a)
    if cmd == "tune":
        from . import tune
        return tune.main()
    if cmd == "computer":
        return computer()
    if cmd == "pending":
        return pending()
    if cmd in ("allow", "deny") and len(a) == 2:
        return answer(a[1], cmd == "allow")
    if cmd == "stop":
        return stop()
    if cmd == "restart":
        if stop() != 0:
            return 1
        return start(True)
    if cmd == "logs":
        oslayer.follow(paths.LOGS / "cale.log")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
