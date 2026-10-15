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
import contextlib
import gc
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

from nest.cancel import Cancelled
from nest.levels3n import MAX_CONTEXT, load_levels, pick

from . import backends, choose, constrain, oslayer, paths, pressure
from . import gemma3n_facts as facts
from . import goals as goals_mod
from .batching import BatchMixin, BatchPred, Request, tier_of
from .turns import TurnQueue
from .predict import Predictor

IDLE_S = 600
WARN_CAP = 0.6
RECOVER_S = 180                # long, so a cap does not flip back and forth
LOAD_GRACE_S = 30              # loading the model raises pressure by itself: ignore it at first
POLL_S = 2


class GoalMissed(Exception):
    """The app asked for goals no size can meet and said to refuse (or to wait, and the wait ran out). `info` has what Cale predicts."""

    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


class Unsupported(Exception):
    """The request asks for something this engine setting cannot do (the server answers 400)."""


class Defer(Exception):
    """Batched path only: the goals cannot be met yet and the app said `wait`. The scheduler tries this request again a little later,
    and the answers that are already running go on meanwhile."""


class Unsafe(Exception):
    """Cale predicts this call would freeze or crash the computer, so it does not run it, whatever the app asked. `info` has the numbers."""

    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


class ContextTooLong(Exception):
    """The prompt (plus the answer length asked for) does not fit this computer's context. `info` has the numbers."""

    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


class EngineError(Exception):
    """The engine failed during a call. Cale has unloaded the model and reloads it on the next call."""


class Engine(BatchMixin):
    def __init__(self, model_path, idle_s=IDLE_S):
        self.path = str(model_path)
        self.idle_s = idle_s
        self.lock = threading.RLock()          # one generation at a time
        self.turns = TurnQueue()               # who is next for that lock (see turns.py)
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
        self.pred = None                       # the speed predictor (cale/predict.py), made when the model loads
        self.mach = None                       # what this computer's copy of the model weighs (see choose.py)
        self._slow = 1.0                       # debug only: pretend this computer is this many times slower
        self.debug_ram = None                  # debug only: pretend this computer has this many bytes of RAM
        self.last_used = time.time()
        self.normal_since = time.time()
        self.load_s = None
        self.loaded_at = 0.0
        self.events = []                       # short log of what the engine did on its own
        self.last_error = None                 # the last engine failure, for /cale/status
        self.debug_fail = False                # debug only: make the next call fail inside the engine
        self.running_cancel = None             # the Cancel of the request that is running now
        self._batch_init()
        threading.Thread(target=self._monitor, daemon=True).start()

    @property
    def slow(self):
        return self._slow

    @slow.setter
    def slow(self, v):
        self._slow = v
        if self.pred is not None:
            self.pred.slow = v

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
        return {"loaded": m is not None, "placement": getattr(m, "placement", None), "slots": getattr(m, "n_slots", 1) if m is not None else None,
                "cap_pct": round(self.cap * 100),
                "resident_gb": resident, "ffn_gb": ffn, "pressure": pressure.name(), "load_s": self.load_s,
                "idle_s": round(time.time() - self.last_used), "implicit_hold": self.implicit,
                "holds": holds, "events": self.events, "last_error": self.last_error}

    def held_ids(self):
        with self.mu:
            ids = [h["level"] for h in self.holds.values()]
        return ids + ([max(self.levels, key=lambda l: l["pct"])["id"]] if self.implicit and self.levels else [])

    def piece_bytes(self, tid):
        """The bytes a token stands for (for `logprobs`). The model must be loaded: an answer that reports log-probabilities loaded it."""
        return self.m.piece_bytes(tid)

    def max_context(self):
        """The longest prompt plus answer this computer accepts, in tokens. Needs the model loaded."""
        return int(getattr(self.m, "max_context", 0) or MAX_CONTEXT)

    def known_context(self):
        """The context limit in tokens. Exact once the model is loaded, the engine's default before that."""
        return self.max_context() if self.m is not None else backends.context_limit()

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
        self._learn_machine()
        self.log(f"loaded model in {self.load_s}s")

    def _learn_machine(self):
        """What this computer's copy of the model weighs, and how fast its memory reads. No model timing: the memory probe takes about 0.15 s."""
        m = self.m
        try:
            bw = float(m.memory_bandwidth())
        except Exception as e:
            print(f"[engine] memory probe failed ({type(e).__name__})", flush=True)
            bw = 0.0
        key = {"engine": backends.name(), "total_ram": oslayer.total_ram(), "placement": getattr(m, "placement", None),
               "model_bytes": sum(f.stat().st_size for f in m.model_files()),
               "prompt_path": 2}      # how a prompt is read: 2 = the last token alone, layers without their own memory skipped. Speeds saved for another way are not used.
        base0 = self._sample_base()
        self.log(f"base memory {base0 / 1e9:.2f} GB (the process holds {m.active_bytes() / 1e9:.2f} GB, FFN weights {m.ffn_bytes() / 1e9:.2f} GB)")
                                    # before the warm-up: working memory is added separately (scratch_bytes, context)
        old = self.pred
        self.pred = Predictor(backends.name(), bw, key, paths.HOME / "predict.json", levels=self.levels)
        if old is not None and old.key == key:
            self.pred.dec_full, self.pred.pre_full, self.pred.fixed = old.dec_full, old.pre_full, old.fixed     # a reload keeps what was learned
            self.pred.n_dec, self.pred.n_pre, self.pred.disk_bps = old.n_dec, old.n_pre, old.disk_bps
        self.pred.slow = self._slow
        self.pred.busy = oslayer.cpu_busy_other
        if hasattr(m, "hold_last_speeds"):                  # the llama engine reads the last prompt token alone when it pays (needs measured speeds)
            m.hold_last_speeds = lambda lv: (self.pred.prefill_tps(lv), self.pred.decode_tps(lv)) if self.pred is not None and self.pred.table else None
        self.mach = {"layer_bytes": m.layer_bytes, "layer_read": m.layer_read, "base_bytes": base0, "scratch_bytes": int((self.pred.extra or {}).get("scratch_bytes") or 0),   # measured by `cale tune` (0 = not measured: facts.SCRATCH_BYTES)
                     "total_ram": key["total_ram"], "disk_bps": self.pred.disk_bps}
        if self.pred.n_dec == 0 and self.pred.n_pre == 0 and not getattr(self, "_tuning", False):       # `cale tune` did not run for this computer and model (a test server, new hardware): measure now, once
            self._measure_speed()

    def _sample_base(self):
        """Memory this process holds besides the FFN weights: the largest of a few readings. The OS reports process memory with a little noise,
        and a prediction must never be under what is used. (It is raised again whenever an answer finds more in use: see _grow_base.)"""
        m, seen = self.m, []
        oslayer.give_back_heap()                          # memory the loader freed but the allocator kept is not in use
        for _ in range(5):
            seen.append(int(m.active_bytes() - m.ffn_bytes()))
            time.sleep(0.05)
        return max(seen)

    def _grow_base(self, d=None):
        """After an answer: if the process holds more than the whole prediction assumed (weights, base, working memory, the context that
        was used), the base grows by the difference. A prediction is never under what is used, and nothing is counted twice."""
        if getattr(self, "active", None):
            return                                        # other answers are running: their working memory is not part of the base
        try:
            oslayer.give_back_heap()                      # working memory that was freed but kept by the allocator is not in use
            held = int(self.m.active_bytes() - self.m.ffn_bytes())
        except Exception:
            return
        ctx = int((d or {}).get("prompt_tokens", 0)) + int((d or {}).get("tokens", 0))
        expected = self.mach["base_bytes"] + (self.mach.get("scratch_bytes") or facts.SCRATCH_BYTES) + facts.kv_bytes(ctx)
        if held > expected:
            self.log(f"base memory raised by {(held - expected) / 1e9:.2f} GB (the process holds {held / 1e9:.2f} GB besides the weights, {expected / 1e9:.2f} GB were predicted)")
            self.mach["base_bytes"] += held - expected

    def _run_prompt(self, tokens, max_tokens=16, width=None):
        """Read a prompt of about `tokens` tokens and answer a few: the `done` record. For measuring."""
        m = self.m
        unit = "The sea is wide and deep. "
        n_unit = max(1, int(m.prompt_len([{"role": "user", "content": unit}])))
        text = unit * max(1, int(tokens / max(n_unit - 8, 1)))
        msgs = [{"role": "user", "content": text + "Write the whole numbers from 1 to 60, separated by spaces."}]
        done = None
        for ev in m.chat_stream(msgs, max_tokens=max_tokens, width=width or max(self.levels, key=lambda l: l["pct"]), temperature=0.0):
            done = ev.get("done", done)
        m.clear_cache()
        return done

    def tune(self, profile="max", say=lambda text: None, report_path=None):
        """Measure this computer once, when Cale is installed (`cale tune`), so no app ever waits for it. See cale/tuning.py for what is
        measured. What the engine needs is saved in predict.json; the full report (every probe, its cost and what it found) in `report_path`."""
        from .tuning import PLACEMENT_NOTE, Tuner, choose_engine_options, choose_placement, probe_cpu
        PLACEMENT_NOTE.clear()
        self.options_report = {"note": "engine options are tried only by the max profile; the defaults are used"} if profile != "max" else {}
        if backends.name() == "llama":                           # which way of running is fastest: decided first, in separate processes
            (paths.HOME / "engine-options.json").unlink(missing_ok=True)
            (paths.HOME / "placement.json").unlink(missing_ok=True)   # an old decision must not steer a computer whose new measurement failed
            decision = None
            try:
                decision = choose_placement(self.path, os.environ.get("CALE_LLAMA_LIB"), paths.HOME, say, self.log)
            except Exception as e:
                self.log(f"tune: could not compare the ways of running ({type(e).__name__}: {str(e)[:120]})")
            if profile == "max":
                try:
                    if decision is None:                         # no graphics card: the CPU alone is the way of running
                        decision = {"use": "cpu", "mode": None, "measured": {"cpu": probe_cpu(self.log)}}
                    self.options_report = choose_engine_options(decision, paths.HOME, say, self.log)
                except Exception as e:
                    self.log(f"tune: could not try the engine options ({type(e).__name__}: {str(e)[:120]})")
        with self.lock:
            self._apply_pending()
            if self.m is not None:
                self._unload("tune")
            self.pred = None
            try:
                (paths.HOME / "predict.json").unlink()          # measure again, whatever was saved
            except OSError:
                pass
            self._tuning = True                                  # the load must not do its own short measurement
            try:
                self._load()
            finally:
                self._tuning = False
        self._tuning = True
        try:
            report = Tuner(self, profile, say, self.log).run()
        finally:
            self._tuning = False
        self.pred.save()
        data = report.to_json()
        data["engine_options"] = self.options_report
        data["problems"] = list(PLACEMENT_NOTE) + [f"{p['probe']}: {str(p.get('result'))[:80]}" for p in data.get("probes", []) if not p.get("ok", True)]
        if report_path:
            Path(report_path).write_text(json.dumps(data, indent=1))
        with self.lock:
            self._unload("tuned")
        return data

    def _measure_speed(self):
        """The first time Cale loads the model on a computer: read a short prompt and answer a few tokens, and keep the speeds.
        They replace the first guess (PRIORS in predict.py, tuned on three test computers), so a prediction on any computer starts from
        what this computer does. The result is saved with the key of this computer and model: it is done once, not at every load.
        A computer where a GPU reads long prompts is measured with a piece as long as the pieces it will read."""
        m = self.m
        if os.environ.get("CALE_NO_MEASURE") == "1":
            return
        try:
            piece = getattr(m, "prompt_piece_hint", 0) or 64
            unit = "The sea is wide and deep. "
            n_unit = max(1, int(m.prompt_len([{"role": "user", "content": unit}])) - 0)
            full = max(self.levels, key=lambda l: l["pct"])
            t0 = time.perf_counter()
            done = None
            while True:                       # a prompt that takes under half a second says little: read a longer one (up to the engine's piece)
                text = unit * max(1, int(piece / max(n_unit - 8, 1)))
                msgs = [{"role": "user", "content": text + "Write the whole numbers from 1 to 60, separated by spaces."}]
                for ev in m.chat_stream(msgs, max_tokens=16, width=full, temperature=0.0):
                    done = ev.get("done", done)
                m.clear_cache()
                if not done or done["first_token_s"] >= 0.5 or piece >= 512:
                    break
                piece *= 4
            if done:
                self.pred.observe(done)
                self._grow_base(done)
                self.log(f"measured this computer in {time.perf_counter() - t0:.1f}s: {self.pred.summary()}")
        except Exception as e:
            self.log(f"speed measurement skipped ({type(e).__name__}: {str(e)[:120]})")

    def machine(self):
        """The machine record with the disk speed as learned so far."""
        return {**self.mach, "disk_bps": self.pred.disk_bps}

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

    def _settling(self, now):
        """Loading the model raises memory pressure by itself. Ignore pressure while the process is still growing (measured: its memory
        changed by more than 2% since the last look), but never longer than LOAD_GRACE_S after the load, and at least two looks."""
        since = now - self.loaded_at
        if since >= LOAD_GRACE_S:
            return False
        try:
            rss = self.m.active_bytes()
        except Exception:
            return since < 2 * POLL_S
        last, self._rss_last = getattr(self, "_rss_last", None), rss
        if since < 2 * POLL_S or last is None:
            return True
        return abs(rss - last) > 0.02 * max(last, 1)

    def _tick(self):
        if getattr(self, "_tuning", False):
            return                                     # `cale tune` is measuring: nothing may change the size or unload the model under it
        lvl, now = pressure.level(), time.time()
        if self.m is not None and self._settling(now):
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

    # --- what this computer does --------------------------------------------------------
    def this_computer(self, load=False):
        """Predicted speed and memory of every size, for /v1/models. From the live predictor, else from the file saved last time.
        load=True loads the model first if it is not loaded (so the answer is complete)."""
        if load and self.m is None:
            with self.lock:
                self._apply_pending()
                if self.m is None:
                    self._load()
        p = self.pred
        if p is not None and self.mach is not None:
            m = self.machine()
            sizes = {l["id"]: {"size_pct": l["pct"], "decode_tps": round(p.decode_tps(l), 2), "prefill_tps": round(p.prefill_tps(l), 1),
                               "ram_gb": round(choose.model_bytes(m, choose.need(l)) / 1e9, 2),
                               "needs_gb": round(choose.memory_needed(m, l, facts.HOLD_CONTEXT) / 1e9, 2), "quality": round(facts.quality(l), 3)}
                     for l in self.levels}
            fast, small = max(sizes.items(), key=lambda kv: kv[1]["decode_tps"]), min(sizes.items(), key=lambda kv: kv[1]["ram_gb"])
            work = ((self.mach.get("scratch_bytes") or facts.SCRATCH_BYTES) + facts.kv_bytes(facts.HOLD_CONTEXT)) / 1e9
            return {"confidence": p.confidence, "memory_gb_s": round(p.bandwidth / 1e9, 1), "disk_gb_per_s": round(p.disk_bps / 1e9, 2),
                    "working_gb": round(work, 3),            # what a memory goal adds to the weights: working memory for a prompt and the context of a hold
                    "placement": getattr(self.m, "placement", None), "sizes": sizes,
                    "floor": {"fastest": {"size": fast[0], "tokens_per_s": fast[1]["decode_tps"]},
                              "smallest": {"size": small[0], "needs_gb": small[1]["needs_gb"]}}}
        try:
            import json
            d = json.loads((paths.HOME / "predict.json").read_text())
            return {"confidence": "saved", "disk_gb_per_s": round(d.get("disk_bps", 0) / 1e9, 2), "sizes": d.get("sizes") or None}
        except (OSError, ValueError):
            return None

    # --- holds ------------------------------------------------------------------------
    def hold(self, app, spec):
        """Keep a size in RAM for `app`. spec: {"size_pct": n} or {"memory_pct": n, "on_miss": "smallest"|"refuse"}.
        Returns (hold id, event that is set when the hold ends, info). Raises GoalMissed."""
        with self.mu:
            self.holds_waiting = getattr(self, "holds_waiting", 0) + 1       # answers that run in a burst let this hold in first
        try:
            self.lock.acquire()
        finally:
            with self.mu:
                self.holds_waiting -= 1
        with contextlib.ExitStack() as _x:
            _x.callback(self.lock.release)
            self._apply_pending()
            if self.m is None:
                self._load()
            info = {}
            if "memory_pct" in spec:
                r = choose.for_memory(self.levels, self.machine(), self.mach["total_ram"], spec["memory_pct"])
                info = {"goal": "memory", "goal_met": r["met"], "budget_gb": round(r["budget"] / 1e9, 2),
                        "needs_gb": round(r["bytes"] / 1e9, 2)}
                if not r["met"] and spec.get("on_miss") in ("refuse", "wait"):
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
    def chat_stream(self, messages, max_tokens, temperature, app="?", size_pct=100, goal=None,
                    top_p=1.0, seed=None, stop=None, arrived=None, max_tokens_explicit=True, cancel=None, guide=None, logprobs=None):
        """Generator. First {"plan": ...} (the size chosen), then {"delta"} pieces, then {"done"}.
        Holds the generation lock for the whole answer.

        size_pct: the absolute size asked for. goal: None, or {"goals": cale/goals.py dict, "max_pct": size cap from the object's
        memory goal, "expect_tokens": how long the object's answers are} to let Cale choose.
        arrived: time.time() when the request came in, so time spent waiting in line counts toward first_token goals.
        max_tokens_explicit: False when the app gave no limit: then the limit shrinks to fit the context instead of failing.
        cancel: a nest.cancel.Cancel. The server sets it when the client closes its connection. The work then stops.
        guide: a cale.constrain.Guide: every token must fit it (JSON mode, JSON schema, tool calls).
        logprobs: None, or how many of the most likely tokens to report next to each token. Events {"logprobs": [{"id", "logprob", "top": [(id, logprob)]}]} then come with the text.
        Raises GoalMissed, Unsafe, ContextTooLong before the first event, Cancelled if the client left, and EngineError if the engine fails."""
        if self.m is None:
            with self.lock:                                         # to know what the engine can do, it must be loaded
                self._apply_pending()
                if self.m is None:
                    self._load()
        if self.can_batch():
            yield from self._batched_chat_stream(messages, max_tokens, temperature, app, size_pct, goal, top_p, seed, stop, arrived,
                                                 max_tokens_explicit, cancel, guide, logprobs)
            return
        if logprobs is not None and not getattr(self.m, "supports_logprobs", False):
            raise Unsupported("This engine setting runs one answer at a time and cannot report log-probabilities. Start Cale with CALE_SLOTS above 1.")
        with self.turns.take(tier_of(goal), cancel), self.lock:
            if cancel:
                cancel.check()                                      # the client gave up while waiting in line: do not start
            waited = max(0.0, time.time() - arrived) if arrived else 0.0
            self.last_used = time.time()
            try:
                self._apply_pending()
                if self.m is None:
                    self._load()
                if cancel:
                    cancel.on_set(lambda: self._interrupt_for(cancel))
                    self.running_cancel = cancel
                if not self.app_has_hold(app):
                    self.implicit_last = time.time()
                    if not self.implicit:
                        self.implicit = True
                        self._resident_now()
                lv, plan, max_tokens, goals, n_prompt, answer_tokens, controller = self._plan_request(
                    messages, max_tokens, max_tokens_explicit, size_pct, goal, waited, cancel, with_controller=True)
                yield {"plan": plan}
                if self.debug_fail:
                    self.debug_fail = False
                    raise RuntimeError("debug: injected engine failure")
                for ev in self.m.chat_stream(messages, max_tokens=max_tokens, width=lv, temperature=temperature,
                                             top_p=top_p, seed=seed, stop=stop, cancel=cancel, controller=controller, guide=guide, owner=None if seed is not None else app, **({} if logprobs is None else {"logprobs": logprobs})):
                    if "done" in ev:
                        d = ev["done"]
                        d["waited_s"] = round(waited, 3)
                        d["total_s"] = round(waited + d["seconds"], 3)
                        if goals is not None and goals_mod.is_set(goals):
                            self._judge(d, goals, waited)
                        self._learn(d)
                    yield ev
            except (GoalMissed, Unsafe, ContextTooLong, Cancelled, constrain.DeadEnd):
                raise
            except Exception as e:                                  # the engine may be left half-way: start clean on the next call
                # The details go to Cale's own log only. /cale/status needs no key, and error text can hold paths and sizes.
                self.last_error = f"{type(e).__name__} at {time.strftime('%H:%M:%S')}"
                print(f"[engine] engine failed ({type(e).__name__}: {str(e)[:300]})", flush=True)      # the private log only: self.log() shows in /cale/status
                self.log(f"engine failed ({type(e).__name__}); unloading the model")
                self._unload("engine error")
                raise EngineError(type(e).__name__) from e
            finally:
                self.running_cancel = None
                self.last_used = time.time()
                if self.implicit:
                    self.implicit_last = time.time()
                self._apply_pending()

    def _plan_request(self, messages, max_tokens, explicit, size_pct, goal, waited, cancel, n_active=1, with_controller=False, defer_since=None):
        """Everything decided before an answer starts: how long the prompt is, the context check, the size, the safety check.
        Returns (size, plan, max_tokens, goals, prompt tokens, expected answer tokens, closed-loop controller or None).
        Raises GoalMissed, Unsafe, ContextTooLong. `n_active`: how many answers share the step (the speed prediction is for that crowd)."""
        pred = self.pred if n_active <= 1 else BatchPred(self.pred, n_active)
        n_prompt = self.m.prompt_len(messages)
        limit = self.max_context()
        if n_prompt + 1 > limit or (explicit and n_prompt + max_tokens > limit):
            raise ContextTooLong(f"This request needs {n_prompt} prompt tokens plus {max_tokens} answer tokens. "
                                 f"This computer accepts {limit} in total.",
                                 {"prompt_tokens": n_prompt, "max_tokens": max_tokens, "max_context_tokens": limit})
        max_tokens = min(max_tokens, limit - n_prompt)
        cap = min(round(self.cap * 100), round(goal.get("max_pct", 100))) if goal else round(self.cap * 100)
        goals = goal["goals"] if goal else None
        plan = {"goal_met": None, "unmet": [], "predicted": None, "waited_s": round(waited, 3), "confidence": self.pred.confidence}
        controller = None
        answer_tokens = goal.get("expect_tokens") or max_tokens if goal else max_tokens
        if goal is None:
            lv = pick(self.levels, size_pct, cap)
        elif not goals_mod.is_set(goals):                  # no goals: the largest size that needs no disk
            pool = [l for l in self.levels if l["pct"] <= cap + 0.5 and not self.m.missing(l)]
            lv = max(pool or [min(self.levels, key=lambda l: l["pct"])], key=lambda l: (l["pct"], l["mmlu_pt"]))
        else:
            res = self._decide(goals, n_prompt, answer_tokens, waited, cap, cancel, pred, defer_since)
            lv = res["row"]["level"]
            plan.update(goal_met=res["met"], unmet=res["unmet"], predicted=self._predicted(res["row"]))
            if with_controller and goals["min_tokens_per_s"] is not None:
                controller = self._controller(goals, lv, cap)
        self._check_safe(lv, n_prompt, answer_tokens, waited)
        disk_mb = round(sum(self.m.layer_read[i] for i, _ in self.m.missing(lv)) / 1e6)
        plan.update(size_pct=lv["pct"], id=lv["id"], disk_mb=disk_mb, prompt_tokens=n_prompt, max_tokens=max_tokens)
        if plan["predicted"] is None:
            plan["predicted"] = self._predicted(choose.row(lv, pred, self.machine(), self._ctx(n_prompt, answer_tokens, waited),
                                                           goals or goals_mod.parse({}), list(self.m.resident)))
        return lv, plan, max_tokens, goals, n_prompt, answer_tokens, controller

    def _batched_chat_stream(self, messages, max_tokens, temperature, app, size_pct, goal, top_p, seed, stop, arrived, explicit, cancel, guide, logprobs=None):
        """chat_stream when answers run together: the scheduler does the work, this generator hands out its events."""
        if cancel:
            cancel.check()
        req = Request(messages, max_tokens, temperature, app, size_pct, goal, top_p, seed, stop, arrived, explicit, cancel, guide, logprobs)
        yield from self._batched_stream(req)

    def _interrupt_for(self, cancel):
        """Stop a long step inside the engine, but only if this request is the one running now."""
        m = self.m
        if self.running_cancel is cancel and m is not None and hasattr(m, "interrupt"):
            m.interrupt()

    # --- choosing --------------------------------------------------------------------------
    def _ctx(self, n_prompt, answer_tokens, waited):
        return {"prompt_tokens": n_prompt, "answer_tokens": answer_tokens, "waited_s": waited,
                "total_ram": self.debug_ram or self.mach["total_ram"],
                "available_ram": None if self.debug_ram else oslayer.available_ram()}

    @staticmethod
    def _predicted(row):
        return {"tokens_per_s": round(row["decode_tps"], 2), "first_token_s": round(row["first_token_s"], 2),
                "memory_gb": round(row["memory_bytes"] / 1e9, 2), "quality": round(row["quality"], 3)}

    def predict(self, goals, prompt_tokens, answer_tokens, cap=100):
        """Ask before you run: what would Cale do with these goals on this computer? Nothing runs and nothing is refused: the
        answer says whether the goals can be met, which size Cale would use, and what every size would do. Loads the model if needed."""
        if self.m is None:
            with self.lock:
                self._apply_pending()
                if self.m is None:
                    self._load()
        ctx = self._ctx(prompt_tokens, answer_tokens, 0.0)
        power = oslayer.power_state() if goals["power"] == "light" else {}
        light = goals["power"] == "light" and bool(power.get("on_battery") or power.get("hot"))
        res = choose.for_goals(self.levels, self.pred, self.machine(), goals, ctx, list(self.m.resident), cap, light,
                               goals["allow_disk"] or goals["on_miss"] == "try")
        out = {"confidence": self.pred.confidence, "sizes": self._table(res["rows"]), "unsafe": res["unsafe"],
               "floor": (self.this_computer() or {}).get("floor")}
        if res["unsafe"]:
            return {**out, "goal_met": False, "unmet": [], "chosen": None, **self._unsafe_info(res["rows"])}
        return {**out, "goal_met": res["met"], "unmet": res["unmet"],
                "chosen": {"size": res["row"]["level"]["id"], "size_pct": res["row"]["level"]["pct"], **self._predicted(res["row"])}}

    def _decide(self, goals, n_prompt, answer_tokens, waited, cap, cancel=None, pred=None, defer_since=None):
        """Choose a size for these goals. Applies on_miss: best_effort runs the best size, try also allows disk, refuse and wait raise GoalMissed."""
        power = oslayer.power_state() if goals["power"] == "light" else {}
        light = goals["power"] == "light" and bool(power.get("on_battery") or power.get("hot"))
        allow_disk = goals["allow_disk"] or goals["on_miss"] == "try"
        t0 = time.time()
        while True:
            ctx = self._ctx(n_prompt, answer_tokens, waited + (time.time() - t0))
            res = choose.for_goals(self.levels, pred or self.pred, self.machine(), goals, ctx, list(self.m.resident), cap, light, allow_disk)
            if res["unsafe"]:
                raise Unsafe("No size can run this call safely on this computer.", self._unsafe_info(res["rows"]))
            if res["met"] or goals["on_miss"] != "wait" or time.time() - t0 >= goals["max_wait_s"]:
                break
            if defer_since is not None:                          # the scheduler is waiting for us: give the lock back instead of sleeping on it
                if time.time() - defer_since[0] >= goals["max_wait_s"]:
                    break
                raise Defer()
            if cancel:
                cancel.check()                                  # the app left: stop waiting, let the next app have the lock
            time.sleep(0.5)
        if not res["met"] and goals["on_miss"] in ("refuse", "wait"):
            best = self._predicted(res["row"])
            info = {"goal_met": False, "unmet": res["unmet"], "predicted": best, "sizes": self._table(res["rows"]),
                    "floor": self.this_computer().get("floor") if self.this_computer() else None}
            raise GoalMissed(f"No size meets {', '.join(res['unmet'])} on this computer"
                             f"{f' (waited {waited:.1f} s in line)' if waited >= 0.5 else ''}. "
                             f"The best size is predicted at {best['tokens_per_s']} tokens/s, first token in {best['first_token_s']} s, "
                             f"{best['memory_gb']} GB.", info)
        return res

    @staticmethod
    def _table(rows):
        return {r["level"]["id"]: {"size_pct": r["level"]["pct"], "tokens_per_s": round(r["decode_tps"], 2), "first_token_s": round(r["first_token_s"], 2),
                                   "memory_gb": round(r["memory_bytes"] / 1e9, 2), "quality": round(r["quality"], 3), "safe": r["safe"]} for r in rows}

    def _unsafe_info(self, rows):
        small = min(rows, key=lambda r: r["memory_bytes"])
        return {"needed_gb": round(small["memory_bytes"] / 1e9, 2), "ram_gb": round((self.debug_ram or self.mach["total_ram"]) / 1e9, 1),
                "available_gb": round(oslayer.available_ram() / 1e9, 2), "smallest_size": small["level"]["id"]}

    def _check_safe(self, lv, n_prompt, answer_tokens, waited):
        """The safety limit. It holds for every size, whatever the app asked, even with on_miss=try."""
        r = choose.row(lv, self.pred, self.machine(), self._ctx(n_prompt, answer_tokens, waited), goals_mod.parse({}), list(self.m.resident))
        if not r["safe"]:
            raise Unsafe(f"Size {lv['pct']}% would not fit safely on this computer with this prompt.", self._unsafe_info([r]))

    def _controller(self, goals, start, cap):
        """Between tokens: if the live answer speed falls under the goal, step to the next smaller size, and back up when there is room.
        Only sizes already in RAM, under the cap, are used. The key/value cache stays valid because a size changes only the FFN width
        and skipped layers (checked with scripts/switch_check.py)."""
        target = goals["min_tokens_per_s"]
        pool = sorted([l for l in self.levels if l["pct"] <= cap + 0.5 and not self.m.missing(l)
                       and (goals["min_quality"] is None or facts.quality(l) >= goals["min_quality"])], key=lambda l: -l["pct"])
        mem = goals["memory_pct"]
        if mem is not None:
            budget = self.mach["total_ram"] * mem / 100
            pool = [l for l in pool if choose.memory_needed(self.machine(), l, facts.HOLD_CONTEXT) <= budget] or pool[-1:]
        state = {"n": 0, "t": 0.0, "switch": 0}

        def control(n, elapsed, level):
            dn, dt = n - state["n"], elapsed - state["t"]
            state["n"], state["t"] = n, elapsed
            if dt <= 0 or n - state["switch"] < 16:
                return None
            speed = dn / dt
            i = next((k for k, l in enumerate(pool) if l["id"] == level["id"]), None)
            if i is None:
                return None
            if speed < 0.92 * target and i + 1 < len(pool):                 # too slow: one size smaller
                state["switch"] = n
                return pool[i + 1]
            if i > 0 and speed > 1.3 * target:                              # room to spare: one size larger, if it would still keep up
                up = pool[i - 1]
                if speed * self.pred.time_factor(level) / self.pred.time_factor(up) >= 1.1 * target and n - state["switch"] >= 32:
                    state["switch"] = n
                    return up
            return None
        return control

    def _judge(self, d, goals, waited):
        """After the answer: what the goals were, what really happened, and whether each was met."""
        out = {}
        tokens, decode_s = int(d.get("tokens") or 0), float(d.get("decode_s") or 0.0)
        if goals["min_tokens_per_s"] is not None:
            tps = (tokens - 1) / decode_s if tokens >= 2 and decode_s > 0 else None
            out["speed"] = {"goal": goals["min_tokens_per_s"], "actual": None if tps is None else round(tps, 2),
                            "met": None if tps is None else tps >= goals["min_tokens_per_s"]}
        if goals["max_first_token_s"] is not None:
            ft = waited + float(d.get("first_token_s") or 0.0)
            out["first_token"] = {"goal": goals["max_first_token_s"], "actual": round(ft, 2), "met": ft <= goals["max_first_token_s"]}
        if goals["memory_pct"] is not None:
            used = self.m.active_bytes()
            out["memory"] = {"goal_pct": goals["memory_pct"], "actual_pct": round(100 * used / self.mach["total_ram"], 1),
                             "met": used <= self.mach["total_ram"] * goals["memory_pct"] / 100}
        if goals["min_quality"] is not None:
            q = min((facts.quality(l) for l in self.levels if l["id"] in {d.get("start_tier"), d.get("tier")}), default=1.0)
            out["quality"] = {"goal": goals["min_quality"], "actual": round(q, 3), "met": q >= goals["min_quality"]}
        d["goals_actual"] = out
        known = [v["met"] for v in out.values() if v["met"] is not None]
        d["goal_met_actual"] = all(known) if known else None

    def _learn(self, d):
        """Fold a real answer into what Cale knows about this computer's speed."""
        self._grow_base(d)
        if self.pred is not None:
            self.pred.note_busy(float(d.get("seconds") or 0.0))
            self.pred.observe(d)
