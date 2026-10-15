"""Model objects outlive a restart of Cale.

An object is a named set of goals that an app made (POST /cale/models). It lives in Cale's memory, so a restart of Cale (an update, a crash,
a reboot) used to break every open app with "Unknown model". Now the spec of each live object is also written to a small file. The first call
that names an object Cale no longer has brings it back from that file. The app notices nothing.

Only the spec is kept (the goals and the name): no prompt, no answer. An object nobody used for MAX_AGE_S is forgotten.
"""
import json
import time

MAX_AGE_S = 24 * 3600          # POLICY: how long a remembered object can be brought back
REVIVE_LEASE_S = 600           # POLICY: a brought-back object lives this long after its last use (as a leased object)


def save(path, live, dormant):
    """Write the objects. `live` and `dormant`: {(app, id): {"spec": ..., "created": ...}}. Live ones win. Never raises."""
    try:
        rows = {}
        for src in (dormant, live):
            for (app, oid), o in src.items():
                rows[(app, oid)] = {"app": app, "id": oid, "spec": o["spec"], "created": o.get("created") or time.time()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(rows.values())), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def load(path, now=None):
    """-> {(app, id): {"spec", "created"}} for the objects young enough to bring back."""
    now = time.time() if now is None else now
    try:
        out = {}
        for r in json.loads(path.read_text(encoding="utf-8")):
            if now - float(r.get("created") or 0) < MAX_AGE_S:
                out[(r["app"], r["id"])] = {"spec": r["spec"], "created": r["created"]}
        return out
    except Exception:
        return {}
