"""Approving an app without a screen. Standard library only.

With a screen, Cale shows the Allow dialog (oslayer.ask_allow). Without one (a server, an SSH session, a minimal Linux), the request waits
and the person answers in a terminal on the same computer:

    cale pending          the apps that are waiting
    cale allow NAME       allow one (asks you to confirm, in a terminal)
    cale deny NAME

How the two sides talk: the server writes a small note in Cale's private folder (`pending/TICKET.json`). The command writes `TICKET.allow` or
`TICKET.deny` next to it. No new network door is opened. Anyone who can write in that folder can already read the keys, so the folder's
permissions are the protection, as for `keys.json`.
"""
import json
import os
import secrets
import time

from . import paths

POLL_S = 0.5


def folder():
    d = paths.HOME / "pending"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def wait_for_terminal(app, replacing=False, seconds=300):
    """Wait for `cale allow` or `cale deny` for this app. True only for allow. No answer in `seconds` counts as no."""
    d = folder()
    ticket = secrets.token_hex(4)
    note = d / f"{ticket}.json"
    note.write_text(json.dumps({"ticket": ticket, "app": app, "replacing": replacing, "since": int(time.time())}))
    print(f"[connect] {app!r} is waiting. On this computer, in a terminal, type:  cale allow \"{app}\"", flush=True)
    try:
        end = time.time() + seconds
        while time.time() < end:
            if (d / f"{ticket}.allow").exists():
                return True
            if (d / f"{ticket}.deny").exists() or not note.exists():
                return False
            time.sleep(POLL_S)
        print(f"[connect] no answer for {app!r} in {seconds} s. Counting that as no.", flush=True)
        return False
    finally:
        for suffix in (".json", ".allow", ".deny"):
            (d / f"{ticket}{suffix}").unlink(missing_ok=True)


def pending():
    """The apps that wait for an answer: [{"ticket", "app", "replacing", "since"}], oldest first."""
    out = []
    for f in folder().glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    return sorted(out, key=lambda x: x.get("since", 0))


def decide(name, allow):
    """Answer the waiting request(s) of the app called `name` (or with that ticket). Returns how many were answered."""
    n = 0
    for p in pending():
        if name.lower() in (p["app"].lower(), p["ticket"]):
            (folder() / f"{p['ticket']}.{'allow' if allow else 'deny'}").write_text("1")
            n += 1
    return n
