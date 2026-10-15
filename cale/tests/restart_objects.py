"""Model objects survive a restart of Cale (cale/objstore.py, cale/server.py _persist_objects and _revive). Run: python cale/tests/restart_objects.py
No model, no engine. Objects made from goals need no engine to come back.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["CALE_HOME"] = tempfile.mkdtemp(prefix="cale-objects-test-")
from cale import objstore, paths, server  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


spec = {"min_tokens_per_s": 5.0, "memory_pct": None, "on_miss": "best_effort"}
with server.OBJ_LOCK:
    server.OBJECTS[("App A", "recap")] = {"spec": spec, "created": time.time()}
server._persist_objects()
f = paths.HOME / "objects.json"
check("the object spec is written to disk", f.exists() and json.loads(f.read_text())[0]["id"] == "recap")
check("only the spec is kept (no prompt, no answer)", set(json.loads(f.read_text())[0]) == {"app", "id", "spec", "created"})

# a restart: memory is empty, the file is still there
with server.OBJ_LOCK:
    server.OBJECTS.clear()
server.DORMANT = None
check("after a restart the object is not alive", ("App A", "recap") not in server.OBJECTS)
obj = server._revive("App A", "recap")
check("the first call that names it brings it back", obj is not None and obj["spec"] == spec and obj["lease"]["ttl"] == objstore.REVIVE_LEASE_S, obj)
check("it is alive now", ("App A", "recap") in server.OBJECTS)
check("another app cannot use it", server._revive("App B", "recap") is None)
check("an unknown name stays unknown", server._revive("App A", "nothing") is None)

# ending
with server.OBJ_LOCK:
    server.OBJECTS.pop(("App A", "recap"), None)                 # the app ends it (or its lease runs out)
server._persist_objects(forget=("App A", "recap"))
with server.OBJ_LOCK:
    server.OBJECTS.clear()
server.DORMANT = None
check("an object that ended is not brought back", server._revive("App A", "recap") is None)

# too old
f.write_text(json.dumps([{"app": "App A", "id": "old", "spec": spec, "created": time.time() - objstore.MAX_AGE_S - 5}]))
server.DORMANT = None
check("an object nobody used for a day is forgotten", server._revive("App A", "old") is None)

# a broken file is not a problem
f.write_text("not json")
server.DORMANT = None
check("a broken file means no objects, not a crash", server._revive("App A", "x") is None)

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
