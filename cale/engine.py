"""The engine: one model in memory, kept as the apps ask, shrunk or dropped under memory pressure.

What sits in RAM is the union of two things:

| Kind of hold | Who makes it | How long it lasts |
|---|---|---|
| Explicit hold | an app calls `/cale/load` and keeps the connection open | until the app unloads it or the app quits |
| Implicit hold | any app that has no explicit hold and just asks for answers: the full size stays in RAM | 10 minutes after its last request, or until critical memory pressure |

Nothing else frees an explicit hold. Memory pressure never does.

| Rule | Behaviour |
|---|---|
| Lazy load | the model loads on the first request or hold, not at login |
| Idle | no implicit use for IDLE_S seconds: the implicit hold ends. If no hold is left, the model is dropped. |
| Pressure warn | cap the size a request runs at (less compute and heat). Explicit holds stay in RAM. |
| Pressure critical | the implicit hold ends. Explicit holds stay. |
| Pressure normal for RECOVER_S | give the cap back |
| Pressure in LOAD_GRACE_S after a load | ignored (the load itself causes it) |

The monitor thread never touches the model while a request runs. It records what it wants
(`want_cap`, `want_unload`, `want_resident`) and the change happens under the generation lock, between requests.
"""
import gc
import math
import threading
import time
import uuid

from nest.levels3n import load_levels, pick

from . import backends, choose, pressure
from . import profile as prof_mod

IDLE_S = 600
WARN_CAP = 0.6
RECOVER_S = 180                # long, so a cap does not flip back and forth
LOAD_GRACE_S = 30              # loading the model raises pressure by itself: ignore it at first
POLL_S = 2
SAVE_EVERY = 10                # save the profile after this many real answers


class GoalMissed(Exception):
    """The app asked for a goal no size can meet and said to refuse. `info` has what Cale predicts."""

    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


class Engine:
    def __init__(self, model_path, idle_s=IDLE_S):
        self.path = str(model_path)
        self.idle_s = idle_s
        self.lock = threading.RLock()          # one generation at a time
        self.mu = threading.Lock()             # guards the hold table
        self.m = None
        self.cap = 1.0
        self.want_cap = None
        self.want_unload = False
        self.want_resident = False
        self.implicit = False
        self.implicit_last = 0.0
        self.holds = {}                        # id -> {"app", "level", "since", "event"}
        self.levels = load_levels()
        self.prof = None
        self.slow = 1.0                        # debug only: pretend this computer is this many times slower
        self.seen = 0
        self.last_used = time.time()
        self.normal_since = time.time()
        self.load_s = None
        self.loaded_at = 0.0
        self.events = []                       # short log of what the engine did on its own
        threading.Thread(target=self._monitor, daemon=True).start()

    # --- state ------------------------------------------------------------------------
    def log(self, msg):
        self.events = (self.events + [f"{time.strftime('%H:%M:%S')} {msg}"])[-20:]
        print(f"[engine] {msg}", flush=True)

    def status(self):
        resident = ffn = None
        m = self.m                              # the monitor thread may unload the model while we read
        if m is not None:
            try:
                resident = round(m.active_bytes() / 1e9, 2)
                ffn = round(m.ffn_bytes() / 1e9, 2)
            except Exception:
                resident = ffn = None
        with self.mu:
            holds = [{"hold": k, "app": h["app"], "size_pct": h["pct"], "age_s": round(time.time() - h["since"])}
                     for k, h in self.holds.items()]
        return {"loaded": m is not None, "cap_pct": round(self.cap * 100),
                "resident_gb": resident, "ffn_gb": ffn, "pressure": pressure.name(), "load_s": self.load_s,
                "idle_s": round(time.time() - self.last_used), "implicit_hold": self.implicit,
                "holds": holds, "events": self.events}

    def held_ids(self):
        with self.mu:
            ids = [h["level"] for h in self.holds.values()]
        return ids + (["e4b"] if self.implicit else [])

    def app_has_hold(self, app):
        with self.mu:
            return any(h["app"] == app for h in self.holds.values())

    # --- load, cap, unload (call with the lock held) ----------------------------------
    def _load(self):
        t = time.perf_counter()
        self.m = backends.load_model(self.path)
        self.cap = 1.0
        self.want_cap = None
        self.loaded_at = time.time()
        self.normal_since = self.loaded_at
        self.load_s = round(time.perf_counter() - t, 1)
        self.log(f"loaded model in {self.load_s}s")

    def _unload(self, why):
        if self.m is None:
            return
        m, self.m = self.m, None
        self.want_cap = None
        m.close()
        gc.collect()
        self.log(f"unloaded model ({why})")

    def _resident_now(self):
        """Make RAM match the holds: free what no hold needs, or drop the model if no hold is left."""
        ids = self.held_ids()
        if not ids:
            self._unload("no hold needs it")
        elif self.m is not None:
            freed = self.m.reside(ids)
            if freed:
                self.log(f"resident weights changed by {-freed / 1e9:+.2f} GB (holds: {len(ids)})")

    def _apply_pending(self):
        if self.want_unload:                   # critical pressure or idle: the implicit hold ends
            self.want_unload = False
            self.want_cap = None
            self.implicit = False
            self.want_resident = True
        if self.want_resident:
            self.want_resident = False
            self._resident_now()
        if self.want_cap is not None and self.m is not None:
            cap, self.want_cap = self.want_cap, None
            t = time.perf_counter()
            try:
                self.m.set_cap(cap)
            except Exception as e:             # tables may be half changed: drop the model, it reloads clean
                self.log(f"cap change failed ({type(e).__name__}: {e})")
                self._unload("failed cap change")
                return
            self.cap = cap
            self.log(f"cap set to {round(cap * 100)}% in {time.perf_counter() - t:.1f}s")

    # --- monitor ----------------------------------------------------------------------
    def _monitor(self):
        while True:
            time.sleep(POLL_S)
            try:
                self._tick()
            except Exception as e:             # the monitor must never die
                self.log(f"monitor error: {type(e).__name__}: {e}")

    def _tick(self):
        lvl, now = pressure.level(), time.time()
        if self.m is not None and now - self.loaded_at < LOAD_GRACE_S:
            lvl = 1
        if lvl == 1:
            if self.want_cap is not None and self.want_cap < 1.0:
                self.want_cap = None           # pressure went away before the cap was applied
            if self.m is not None and self.cap < 1.0 and now - self.normal_since >= RECOVER_S:
                self.want_cap = 1.0
        else:
            self.normal_since = now
            if lvl >= 4:
                self.want_unload = True
            elif self.cap > WARN_CAP:
                self.want_cap = WARN_CAP
        idle = self.m is not None and self.implicit and now - self.implicit_last > self.idle_s
        if (self.want_cap is not None or self.want_unload or self.want_resident or idle) and self.lock.acquire(blocking=False):
            try:
                if idle and time.time() - self.implicit_last > self.idle_s:    # re-check: a request may have just ended
                    self.want_unload = True
                self._apply_pending()
            finally:
                self.lock.release()

    # --- profile ----------------------------------------------------------------------
    def _profile(self):
        """What this computer measured. Measures once (about 20 s) when no saved profile fits. Call with the lock held."""
        if self.prof is None:
            self.prof = prof_mod.load(self.m)
            if self.prof is None:
                self.log("measuring this computer (about 20 s, once)")
                saved = list(self.m.resident)
                self.m.reside_states([2] * self.m.n_layers)
                self.prof = prof_mod.measure(self.m)
                prof_mod.save(self.prof)
                self.m.reside_states(saved)
                fit = [l for l in self.levels if all(n <= r for n, r in zip(choose.need(l), saved))]
                self.m.set(max(fit, key=lambda l: l["pct"]))      # a size that still runs from what is in RAM
                self.log("profile saved")
        return self.prof

    def profile_view(self):
        """The profile as the chooser sees it: debug slowdown applied."""
        p = self._profile()
        if self.slow == 1.0:
            return p
        view = dict(p, levels={k: dict(v, decode_tps=v["decode_tps"] / self.slow, prefill_tps=v["prefill_tps"] / self.slow)
                               for k, v in p["levels"].items()})
        return view

    def saved_profile(self):
        """The profile if it exists already (in memory or on disk). Never measures."""
        if self.prof is not None:
            return self.prof
        try:
            import json
            return json.loads(prof_mod.PROFILE.read_text())
        except (OSError, ValueError):
            return None

    # --- holds ------------------------------------------------------------------------
    def hold(self, app, spec):
        """Keep a size in RAM for `app`. spec: {"size_pct": n} or {"memory_pct": n, "on_miss": "smallest"|"refuse"}.
        Returns (hold id, event that is set when the hold ends, info). Raises GoalMissed."""
        with self.lock:
            self._apply_pending()
            if self.m is None:
                self._load()
            info = {}
            if "memory_pct" in spec:
                prof = self._profile()
                r = choose.for_memory(self.levels, prof, prof["key"]["total_ram"], spec["memory_pct"])
                info = {"goal": "memory", "goal_met": r["met"], "budget_gb": round(r["budget"] / 1e9, 2),
                        "needs_gb": round(r["bytes"] / 1e9, 2)}
                if not r["met"] and spec.get("on_miss") == "refuse":
                    raise GoalMissed(f"No size fits in {spec['memory_pct']}% of this computer's memory. "
                                     f"The smallest needs {info['needs_gb']} GB.", info)
                level = r["level"]
            else:
                level = pick(self.levels, spec["size_pct"])
            hid = uuid.uuid4().hex[:8]
            event = threading.Event()
            with self.mu:
                self.holds[hid] = {"app": app, "level": level["id"], "pct": level["pct"], "since": time.time(), "event": event}
            self._resident_now()
            info.update(hold=hid, size=level["id"], size_pct=level["pct"],
                        ram_gb=round(self.m.active_bytes() / 1e9, 2), pressure=pressure.name())
            self.log(f"hold {hid} for {app!r}: size {level['pct']}%, RAM {info['ram_gb']} GB")
            return hid, event, info

    def release(self, hid, app=None):
        """End a hold. Only its owner may, when `app` is given. Returns True if a hold ended."""
        with self.mu:
            h = self.holds.get(hid)
            if h is None or (app is not None and h["app"] != app):
                return False
            del self.holds[hid]
        h["event"].set()
        self.want_resident = True
        if self.lock.acquire(blocking=False):
            try:
                self._apply_pending()
            finally:
                self.lock.release()
        self.log(f"hold {hid} ended")
        return True

    # --- requests ---------------------------------------------------------------------
    def chat_stream(self, messages, max_tokens, temperature, app="?", size_pct=100, goal=None):
        """Generator. First {"plan": ...} (the size chosen), then {"delta"} pieces, then {"done"}.
        Holds the generation lock for the whole answer.

        size_pct: the absolute size asked for. goal: None, or {"deadline_s": float|None, "allow_disk": bool,
        "on_miss": "smallest"|"refuse"} to let Cale choose. Raises GoalMissed before the first event."""
        with self.lock:
            self.last_used = time.time()
            try:
                self._apply_pending()
                if self.m is None:
                    self._load()
                if not self.app_has_hold(app):
                    self.implicit_last = time.time()
                    if not self.implicit:
                        self.implicit = True
                        self._resident_now()
                cap = round(self.cap * 100)
                plan = {"goal_met": None, "predicted_s": None}
                if goal is None:
                    lv = pick(self.levels, size_pct, cap)
                elif goal["deadline_s"] is None:                   # no deadline: the largest size that needs no disk
                    pool = [l for l in self.levels if l["pct"] <= cap + 0.5 and not self.m.missing(l)]
                    lv = max(pool or [min(self.levels, key=lambda l: l["pct"])], key=lambda l: (l["pct"], l["mmlu_pt"]))
                else:
                    n = self.m.prompt_len(messages)
                    r = choose.for_deadline(self.levels, self.profile_view(), list(self.m.resident), n,
                                            goal.get("expect_tokens") or max_tokens,    # a model object says how long its answers are
                                            goal["deadline_s"], goal["allow_disk"], cap)
                    lv = r["level"]
                    plan.update(goal_met=r["met"], predicted_s=round(r["predicted_s"], 2))
                    if not r["met"] and goal["on_miss"] == "refuse":
                        raise GoalMissed(f"No size can finish in {goal['deadline_s']} s on this computer. "
                                         f"The fastest is predicted at {plan['predicted_s']} s.", plan)
                disk_mb = round(sum(self.m.layer_read[i] for i, _ in self.m.missing(lv)) / 1e6)
                plan.update(size_pct=lv["pct"], id=lv["id"], disk_mb=disk_mb)
                yield {"plan": plan}
                for ev in self.m.chat_stream(messages, max_tokens=max_tokens, width=lv, temperature=temperature):
                    if "done" in ev:
                        self._learn(ev["done"])
                    yield ev
            finally:
                self.last_used = time.time()
                if self.implicit:
                    self.implicit_last = time.time()
                self._apply_pending()

    def _learn(self, d):
        """Fold a real answer into the profile, if there is one."""
        if self.prof is None or self.slow != 1.0:
            return
        prof_mod.observe(self.prof, d["tier"], d["tokens"], d["seconds"], d["prompt_tokens"],
                         d.get("disk_mb", 0) * 1e6, d.get("disk_s", 0.0))
        self.seen += 1
        if self.seen % SAVE_EVERY == 0:
            prof_mod.save(self.prof)
