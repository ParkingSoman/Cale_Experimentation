"""What Cale is doing right now, and what it did lately. Numbers only: no prompt text, no answer text.

The Cale Hub reads this to show one card per app. One request runs in one thread, so the record
of the running call is found through a thread-local value."""
import collections
import itertools
import threading
import time

KEEP = 200                                   # finished calls kept
_LOCK = threading.Lock()
_DONE = collections.deque(maxlen=KEEP)
_RUNNING = {}
_IDS = itertools.count(1)
_HERE = threading.local()


def begin(app, model):
    """A call starts. Call end() when it finishes, even when it fails."""
    rec = {"id": next(_IDS), "app": app, "model": str(model)[:80], "t0": time.time(), "pieces": 0,
           "size_pct": None, "first_token_s": None}
    with _LOCK:
        _RUNNING[rec["id"]] = rec
    _HERE.rec = rec


def size(size_pct):
    rec = getattr(_HERE, "rec", None)
    if rec:
        rec["size_pct"] = size_pct


def piece():
    """One streamed piece of the answer arrived. A piece is about one token, not exactly."""
    rec = getattr(_HERE, "rec", None)
    if rec:
        now = time.time()
        if rec["pieces"] == 0:
            rec["first_token_s"] = round(now - rec["t0"], 3)
            rec["t_first"] = now
        rec["t_last"] = now
        rec["pieces"] += 1


def finish(size_pct, done, goal_met=None):
    """The answer is complete. `done` is the engine's final record."""
    rec = getattr(_HERE, "rec", None)
    if rec:
        rec.update(final={"size_pct": size_pct, "tokens": done.get("tokens"), "seconds": done.get("seconds"),
                          "waited_s": done.get("waited_s"), "prompt_tokens": done.get("prompt_tokens"),
                          "goal_met": done.get("goal_met_actual", goal_met), "disk_mb": done.get("disk_mb", 0)})


def _speed(rec, tokens, secs):
    """Answering speed: tokens after the first one, over the time between the first and the last piece.
    The engine's own `seconds` also holds the time to read the prompt, which would make a long prompt look slow."""
    if tokens and tokens > 2 and rec.get("t_last") and rec["t_last"] - rec["t_first"] > 0.2:
        return round((tokens - 1) / (rec["t_last"] - rec["t_first"]), 2)
    return round(tokens / secs, 2) if tokens and secs else None


def end(outcome="ok"):
    rec = getattr(_HERE, "rec", None)
    _HERE.rec = None
    if not rec:
        return
    now = time.time()
    fin = rec.get("final") or {}
    tokens, secs = fin.get("tokens"), fin.get("seconds")
    out = {"id": rec["id"], "app": rec["app"], "model": rec["model"], "at": int(rec["t0"]), "outcome": outcome,
           "size_pct": fin.get("size_pct", rec["size_pct"]), "tokens": tokens, "seconds": secs,
           "total_s": round(now - rec["t0"], 3), "first_token_s": rec["first_token_s"],
           "waited_s": fin.get("waited_s"), "prompt_tokens": fin.get("prompt_tokens"),
           "goal_met": fin.get("goal_met"), "disk_mb": fin.get("disk_mb", 0),
           "tokens_per_s": _speed(rec, tokens, secs)}
    with _LOCK:
        _RUNNING.pop(rec["id"], None)
        _DONE.append(out)


def snapshot(limit=50):
    """Newest finished calls first, plus the calls that run now."""
    now = time.time()
    with _LOCK:
        running = [{"id": r["id"], "app": r["app"], "model": r["model"], "size_pct": r["size_pct"],
                    "running_s": round(now - r["t0"], 2), "pieces": r["pieces"], "first_token_s": r["first_token_s"],
                    "pieces_per_s": round(r["pieces"] / max(now - r["t0"] - (r["first_token_s"] or 0), 0.001), 2)
                    if r["pieces"] > 1 else None}
                   for r in _RUNNING.values()]
        done = list(_DONE)[-limit:][::-1]
    return {"running": running, "recent": done}
