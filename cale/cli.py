"""The `cale` command. Standard library only, so it starts fast.

  cale status         is Cale running, is the model loaded
  cale apps           apps that hold a key
  cale revoke NAME    take a key away
  cale start | stop | restart
  cale logs           follow the log
"""
import sys
import time

from . import keys, oslayer, paths, service


def status():
    s = service.status()
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


def start():
    ok, msg = service.start()
    print(msg)
    return 0 if ok else 1


def stop():
    ok = service.stop()
    print("Cale stopped. Type: cale start" if ok else "Cale did not stop.")
    return 0 if ok else 1


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    cmd = a[0] if a else "status"
    if cmd == "status":
        return status()
    if cmd == "apps":
        return apps()
    if cmd == "revoke" and len(a) == 2:
        return revoke(a[1])
    if cmd == "start":
        return start()
    if cmd == "stop":
        return stop()
    if cmd == "restart":
        stop()
        return start()
    if cmd == "logs":
        oslayer.follow(paths.LOGS / "cale.log")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
