"""A call that names a model object still being made waits for it (cale/server.py _model_object, _wait_pending). Run: python cale/tests/pending_object.py
No model, no engine. The chat starts before the AI has loaded, so its first message must wait, not fail.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["CALE_HOME"] = tempfile.mkdtemp(prefix="cale-pending-test-")
from cale import server  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


# 1. no object and nothing pending: no wait, no object
t = time.time()
check("no object, nothing pending: returns at once", server._wait_pending("App A", "chat") is None and time.time() - t < 0.2)

# 2. an object being made: the call waits, then gets the object
made = {}


def slow_made(self, app, req):
    time.sleep(0.6)                                          # the AI loads
    with server.OBJ_LOCK:
        server.OBJECTS[(app, "chat")] = {"spec": {}, "created": time.time()}
        made["event"] = server.PENDING.pop((app, "chat"), None)
    if made["event"]:
        made["event"].set()                                  # what _model_object_made does when the object exists
    time.sleep(0.5)                                          # the connection stays open, like a real object
    return "done"


fake = SimpleNamespace(_model_object_made=lambda app, req: slow_made(None, app, req))
th = threading.Thread(target=lambda: server.Handler._model_object(fake, "App A", {"name": "chat"}))
th.start()
time.sleep(0.15)
check("the object is listed as pending while it is made", ("App A", "chat") in server.PENDING)
t = time.time()
obj = server._wait_pending("App A", "chat")
waited = time.time() - t
check("the call waits for the object", obj is not None and waited >= 0.3, (obj, waited))
check("it goes on as soon as the object exists, not when the connection ends", waited < 0.9, waited)
th.join()
check("pending is cleared", ("App A", "chat") not in server.PENDING)

# 3. an error while the object is made: nobody waits for ever
server.OBJECTS.clear()


def failing(app, req):
    time.sleep(0.4)
    raise server.ApiError(422, "goal_missed", "no size meets these goals")


fake = SimpleNamespace(_model_object_made=failing)
err = []
th = threading.Thread(target=lambda: err.append(_catch(fake)))


def _catch(f):
    try:
        server.Handler._model_object(f, "App B", {"name": "chat"})
    except server.ApiError as e:
        return e.code


th.start()
time.sleep(0.1)
t = time.time()
check("an error leaves no object", server._wait_pending("App B", "chat") is None)
check("the waiting call is released when the make fails", time.time() - t < 1.0)
th.join()
check("the error reaches the app", err == ["goal_missed"], err)
check("pending is cleared after an error", ("App B", "chat") not in server.PENDING)

# 4. the same name alive already: no pending entry, so no wait
with server.OBJ_LOCK:
    server.OBJECTS[("App C", "chat")] = {"spec": {}, "created": time.time()}
seen = []
fake = SimpleNamespace(_model_object_made=lambda app, req: seen.append(("App C", "chat") in server.PENDING))
server.Handler._model_object(fake, "App C", {"name": "chat"})
check("a name that is alive is not marked pending", seen == [False], seen)

# 5. one app's wait does not catch another app's call
th = threading.Thread(target=lambda: server.Handler._model_object(SimpleNamespace(_model_object_made=lambda a, r: time.sleep(0.5)), "App D", {"name": "chat"}))
th.start()
time.sleep(0.1)
check("another app's call does not wait", server._wait_pending("App E", "chat") is None)
th.join()

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
