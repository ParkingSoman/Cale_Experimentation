"""Several answers at once. Part of the engine (a mixin), used when the engine can run answers together (llama.cpp with slots).

Decoding one token reads every weight once, whatever the number of answers in the step, so a step with 4 answers costs little more than a
step with 1. One scheduler thread runs the steps. It holds the generation lock while any answer is running, so holds, unloads and the
monitor wait for the burst to end, as they did when answers ran one at a time.

| Rule | Why |
|---|---|
| One size for all answers in a step | the model runs at one width per step. The size is the smallest any running answer's goals need. |
| An exact size (`gemma-3n:100`) is kept | an app that names a size gets it. An answer with a different exact size waits until the step is free of other exact sizes. |
| The prompt of a new answer is read a piece at a time, next to the others' tokens | nobody waits for a whole long prompt |
| A new answer starts when a slot is free, in the order it arrived | fair order |
| Each answer's goals are checked as if `n` answers share the step | `BatchPred` below |
"""
import os
import queue
import threading
import time

from nest.cancel import Cancelled

from . import choose, constrain, goals as goals_mod, oslayer
from .backends.llama import PoolFull
from . import gemma3n_facts as facts

STEP_SECONDS = 0.75         # POLICY, not a measurement: a step that reads a prompt piece should take about this long, so the answers next to it keep moving
CHECK_EVERY = 8             # steps between looks at the speed goals
IDLE_POLL_S = 0.05
# A GPU that reads prompts gets the weights of the model again for every piece, so a piece must be long to pay for that. How long, and from
# which length the GPU takes the work, depends on the computer: the GPU backend says the shortest piece it takes (lib.offload_min_batch), the
# engine says the longest (its batch size), and PieceTuner finds the best length between them by trying. The one number left is a policy:
# how long a piece may take, because one piece cannot be stopped and an app that leaves holds the next app up for that long.
MAX_PIECE_S = float(os.environ.get("CALE_MAX_PIECE_S", "6"))


class PieceTuner:
    """Picks the length of a prompt piece for a GPU that reads prompts, by trying. Used when a prompt is read alone: next to running
    answers the pieces stay short (below what the GPU takes) so the answers keep moving.

    Candidates: the engine's batch size and its halves down to the shortest piece the GPU backend takes, and one piece just below that
    (the CPU does it). Each is tried once, longest first, going down only while a piece takes longer than the budget. Then the candidate
    with the best speed among those that fit the budget is used. If it is the CPU piece, the GPU does not pay on this computer."""

    def __init__(self, shortest, longest, budget_s=MAX_PIECE_S):
        sizes, n = [], max(longest, 1)
        while n >= max(shortest, 1):
            sizes.append(n)
            n //= 2
        self.gpu_sizes = sizes or [max(longest, 1)]
        self.cpu_size = max(shortest - 1, 1)               # one token under what the GPU takes
        self.speed = {}                                    # piece length -> tokens per second seen
        self.budget = budget_s

    def _fits(self, size):
        return size in self.speed and size / max(self.speed[size], 1e-9) <= self.budget

    def next_size(self):
        for size in self.gpu_sizes:
            if size not in self.speed:
                return size
            if self._fits(size):
                break                                      # a piece within the budget: shorter GPU pieces will not be faster
        if self.cpu_size not in self.speed:
            return self.cpu_size
        fits = [s for s in self.speed if self._fits(s)] or [min(self.speed, key=lambda s: s / max(self.speed[s], 1e-9))]
        return max(fits, key=lambda s: self.speed[s])

    def seen(self, size, tokens, seconds):
        if seconds > 0 and tokens > 0:
            v = tokens / seconds
            old = self.speed.get(size)
            self.speed[size] = v if old is None else 0.7 * old + 0.3 * v

    def uses_gpu(self):
        return bool(self.speed) and self.next_size() > self.cpu_size


class BatchPred:
    """The predictor as one of `n` answers that share a step sees it: every token takes `cost` times as long."""

    def __init__(self, pred, n):
        self.pred, self.n = pred, n
        self.cost = pred.batch_cost(n)
        self.confidence = pred.confidence

    def decode_tps(self, level):
        return self.pred.decode_tps(level) / self.cost

    def prefill_tps(self, level):
        return self.pred.prefill_tps(level)

    def first_token_s(self, level, prompt_tokens, disk_s=0.0):
        return disk_s + prompt_tokens / self.prefill_tps(level) + 1.0 / self.decode_tps(level)

    def time_factor(self, level):
        return self.pred.time_factor(level)


class Request:
    """One call waiting for, or in, the scheduler."""

    def __init__(self, messages, max_tokens, temperature, app, size_pct, goal, top_p, seed, stop, arrived, explicit, cancel, guide, logprobs=None):
        self.messages, self.max_tokens, self.temperature, self.app = messages, max_tokens, temperature, app
        self.size_pct, self.goal, self.top_p, self.seed, self.stop = size_pct, goal, top_p, seed, stop
        self.arrived = arrived or time.time()
        self.explicit, self.cancel, self.guide = explicit, cancel, guide
        self.logprobs = logprobs
        self.q = queue.Queue()
        self.abandoned = False          # the consumer stopped reading (the generator was closed)
        self.stream = None
        self.level = None               # the size this answer was planned for
        self.fixed = False              # an exact size was asked for
        self.goals = None
        self.waited = 0.0
        self.max_pct = 100
        self.steps = 0                  # steps this answer took part in once it was producing tokens
        self.share = 0                  # sum of the number of answers in those steps (to know how crowded it was)
        self.started = None
        self.defer_since = None         # [time] when an app that said `wait` first could not be served
        self.retry_at = 0.0


class BatchMixin:
    """Methods added to Engine. State: self.inbox, self.active, self.cv, self.runner."""

    def _batch_init(self):
        self.inbox, self.active = [], []
        self.cv = threading.Condition()
        self.runner = None
        self.batch_level = None

    # --- the consumer side: a generator like Engine.chat_stream's ----------------------------------------------------
    def can_batch(self):
        m = self.m
        return m is not None and getattr(m, "can_batch", False)

    def _batched_stream(self, req):
        """Submit `req` and yield its events: the plan, the deltas, the done record. Exceptions of the scheduler are raised here."""
        with self.cv:
            self.inbox.append(req)
            if self.runner is None or not self.runner.is_alive():
                self.runner = threading.Thread(target=self._runner_loop, daemon=True, name="cale-batch")
                self.runner.start()
            self.cv.notify_all()
        try:
            while True:
                ev = req.q.get()
                if isinstance(ev, BaseException):
                    raise ev
                yield ev
                if "done" in ev:
                    return
        finally:
            req.abandoned = True                 # a closed generator (the client left) frees the slot at the next step
            with self.cv:
                self.cv.notify_all()

    def piece_tuner(self):
        if getattr(self, "_tuner", None) is None or getattr(self, "_tuner_for", None) is not self.m:
            self._tuner = PieceTuner(self.m.offload_min, self.m.batch_tokens)
            self._tuner_for = self.m
        return self._tuner

    # --- the scheduler thread -----------------------------------------------------------------------------------------
    def _runner_loop(self):
        while True:
            with self.cv:
                if not self.inbox and not self.active:
                    self.cv.wait(timeout=30)
                    if not self.inbox and not self.active:
                        self.runner = None       # nothing to do for a while: the thread ends, the next request starts a new one
                        return
            if self.draining():
                time.sleep(0.05)                 # a hold or the monitor is waiting for the lock: let it in first
            try:
                self._serve()
            except Exception as e:               # the scheduler must never die with requests waiting
                print(f"[engine] scheduler error {type(e).__name__}: {str(e)[:200]}", flush=True)
                self._fail_all(e)

    def _serve(self):
        with self.lock:
            try:
                self._apply_pending()
                if self.m is None:
                    self._load()
                while True:
                    self._admit()
                    if not self.active:
                        if self.draining():
                            break                       # let the monitor or a hold have the lock; the runner comes back for the waiting requests
                        with self.cv:
                            if not self.inbox:
                                break
                        time.sleep(IDLE_POLL_S)         # waiting requests that cannot start together with the running ones
                        continue
                    self._step()
            except Exception as e:
                self._engine_failed(e)
            finally:
                self.last_used = time.time()
                if self.implicit:
                    self.implicit_last = time.time()
                if self.m is not None and not self.active:
                    try:
                        self.m.settle()
                    except Exception:
                        pass
                self.batch_level = None
                self._apply_pending()

    def _engine_failed(self, e):
        self.last_error = f"{type(e).__name__} at {time.strftime('%H:%M:%S')}"
        print(f"[engine] engine failed ({type(e).__name__}: {str(e)[:300]})", flush=True)
        self.log(f"engine failed ({type(e).__name__}); unloading the model")
        err = None
        from .engine import EngineError
        err = EngineError(type(e).__name__)
        for r in list(self.active):
            r.q.put(err)
        self.active.clear()
        with self.cv:
            waiting, self.inbox = self.inbox, []
        for r in waiting:
            r.q.put(err)
        self._unload("engine error")

    def _fail_all(self, e):
        from .engine import EngineError
        err = EngineError(type(e).__name__)
        for r in list(self.active):
            r.q.put(err)
        self.active.clear()
        with self.cv:
            waiting, self.inbox = self.inbox, []
        for r in waiting:
            r.q.put(err)

    # --- admitting new answers ------------------------------------------------------------------------------------------
    def draining(self):
        """Something must be applied that cannot happen while answers run (memory pressure, a hold). No new answer starts until it has."""
        return bool(self.want_unload or self.want_cap is not None or getattr(self, "holds_waiting", 0) > 0)

    def _admit(self):
        from .engine import Defer
        m = self.m
        if self.draining():
            return
        with self.cv:
            waiting = list(self.inbox)
        for r in waiting:
            if r.abandoned or (r.cancel is not None and r.cancel.is_set()):
                self._drop_waiting(r)
                if not r.abandoned:
                    r.q.put(Cancelled())
                continue
            if r.retry_at > time.time():
                continue                                 # an app that said `wait`: it is tried again in a moment. Others may start meanwhile.
            if len(self.active) >= m.n_slots or not m.free_slots():
                break                                    # in order: nobody jumps the queue
            if not self._compatible(r):
                break
            self._drop_waiting(r)
            try:
                self._start(r)
            except Defer:
                r.retry_at = time.time() + 0.5
                with self.cv:
                    self.inbox.insert(0, r)
            except PoolFull:                                # the shared context is full: wait for a running answer to finish
                with self.cv:
                    self.inbox.insert(0, r)
                break
            except Exception as e:
                from .engine import ContextTooLong, EngineError, GoalMissed, Unsafe
                if isinstance(e, (GoalMissed, Unsafe, ContextTooLong, Cancelled, constrain.DeadEnd)):
                    r.q.put(e)                              # the app gets the same error a one-at-a-time answer would give
                else:
                    r.q.put(EngineError(type(e).__name__))  # whoever asked must always get an answer, even when the engine fails
                    raise

    def _drop_waiting(self, r):
        with self.cv:
            if r in self.inbox:
                self.inbox.remove(r)

    def _compatible(self, r):
        """May this request start next to the running answers? An exact size cannot share a step with a different exact size."""
        if r.goal is not None or not self.active:
            return True
        fixed = {x.level["id"] for x in self.active if x.fixed}
        want = self._exact_level(r)
        return not fixed or want["id"] in fixed

    def _exact_level(self, r):
        from nest.levels3n import pick
        return pick(self.levels, r.size_pct, round(self.cap * 100))

    def _start(self, r):
        """Plan the request, set the step's size, open its slot. Raises what chat_stream raises before the first event."""
        m = self.m
        r.waited = max(0.0, time.time() - r.arrived)
        self.last_used = time.time()
        if not self.app_has_hold(r.app):
            self.implicit_last = time.time()
            if not self.implicit:
                self.implicit = True
                self._resident_now()
        n_after = len(self.active) + 1
        if r.goal is not None and r.defer_since is None:
            r.defer_since = [time.time()]
        lv, plan, max_tokens, goals, n_prompt, answer_tokens, _ = self._plan_request(
            r.messages, r.max_tokens, r.explicit, r.size_pct, r.goal, r.waited, r.cancel, n_active=n_after, defer_since=r.defer_since)
        r.goals, r.max_tokens, r.fixed = goals, max_tokens, r.goal is None
        r.max_pct = min(round(self.cap * 100), round(r.goal.get("max_pct", 100))) if r.goal else round(self.cap * 100)
        r.level = lv
        target = self._batch_target(lv, r)
        if target["id"] != lv["id"]:                      # the others are running at another size: this answer runs at the step's size
            plan.update(size_pct=target["pct"], id=target["id"])
            if plan.get("goal_met") is not None or r.goals is not None:
                row = choose.row(target, BatchPred(self.pred, n_after), self.machine(), self._ctx(n_prompt, answer_tokens, r.waited),
                                 r.goals or goals_mod.parse({}), list(m.resident))
                plan["predicted"] = self._predicted(row)
                if goals_mod.is_set(r.goals):
                    plan["unmet"] = [g for g, ok in row["checks"].items() if not ok]
                    plan["goal_met"] = not plan["unmet"]
                    from .engine import GoalMissed, Unsafe
                    if not row["safe"]:
                        raise Unsafe(f"Size {target['pct']}% would not fit safely next to the answers that are running.", self._unsafe_info([row]))
                    if plan["unmet"] and r.goals["on_miss"] in ("refuse", "wait"):       # joining a step that runs at another size must not break what the app asked for
                        info = {"goal_met": False, "unmet": plan["unmet"], "predicted": plan["predicted"], "sizes": None, "floor": None}
                        raise GoalMissed(f"The answers that are running hold this computer at the {target['pct']}% size. "
                                         f"{', '.join(plan['unmet'])} cannot be met next to them.", info)
        if not m.make_room(n_prompt + min(max_tokens, 512) + 8 * len(self.active) + 8):
            if not self.active and not m.make_room(n_prompt + 8):
                from .engine import ContextTooLong
                limit = self.max_context()                     # alone and still no room: waiting would never help
                raise ContextTooLong(f"The prompt of {n_prompt} tokens leaves no room for an answer in the {limit} tokens this computer accepts.",
                                     {"prompt_tokens": n_prompt, "max_tokens": max_tokens, "max_context_tokens": limit})
            if self.active:
                raise PoolFull("the shared context is full of running answers")      # before the plan is sent: the request just waits
        read, read_s = m.ensure(target)                     # weights that a hold let go of are read back from disk now
        self._set_level(target)
        plan["disk_mb"] = round(read / 1e6)
        r.disk_mb, r.disk_s = round(read / 1e6), round(read_s, 3)
        r.q.put({"plan": plan})
        if self.debug_fail:
            self.debug_fail = False
            raise RuntimeError("debug: injected engine failure")
        r.stream = m.open_stream(r.messages, max_tokens, r.temperature, r.top_p, r.seed, r.stop, r.guide,
                                  None if r.seed is not None else r.app, r.logprobs)       # a seeded answer must not depend on what an earlier answer left in the cache
        r.started = time.perf_counter()
        self.active.append(r)

    def _batch_target(self, lv, r):
        """The size the step runs at once `r` joins: an exact size wins, else the smallest size any running answer needs."""
        if r.goal is None:
            return lv
        fixed = [x.level for x in self.active if x.fixed]
        if fixed:
            return fixed[0]
        cur = self.batch_level
        if cur is None or not self.active:
            return lv
        return lv if lv["pct"] < cur["pct"] else cur

    def _set_level(self, level):
        """The step runs at this size from now on. (The weights it needs must be in memory: the caller reads them with m.ensure.)"""
        if self.batch_level is None or self.batch_level["id"] != level["id"]:
            old = self.batch_level
            for r in self.active:                          # every running answer sees the change in its reply (`size_changes`), whoever caused it
                if old is not None and r.stream.t_first is not None:
                    r.stream.switches.append({"at_token": r.stream.out, "from": old["id"], "to": level["id"]})
            self.m.set(level)
            self.batch_level = level

    # --- one step ---------------------------------------------------------------------------------------------------------
    def _step(self):
        m = self.m
        self._reap()
        if not self.active:
            return
        decoding = [r for r in self.active if r.stream.t_first is not None]
        # A piece of prompt cannot be stopped once started, so its length is set by time: next to other answers about STEP_SECONDS,
        # alone about twice that. On a slow computer that is a few tokens, so an app that leaves is noticed quickly.
        secs = STEP_SECONDS if decoding else 2 * STEP_SECONDS
        n_batch = getattr(m, "batch_tokens", 512)                                # the engine's own limit for one pass
        budget = max(1, min(n_batch, int(self.pred.prefill_tps(self.batch_level) * secs)))
        tuner = self.piece_tuner() if getattr(m, "gpu_reads_prompts", False) else None
        piece_n = None
        if tuner is not None:
            if decoding:
                budget = min(budget, max(m.offload_min - 1, 1))         # next to running answers: short pieces, below what the GPU takes, so the answers keep moving
            else:
                piece_n = tuner.next_size()
                budget = piece_n
        reading = [r for r in self.active if r.stream.t_first is None]
        before = m.lib.slot_tokens(reading[0].stream.slot) if tuner is not None and not decoding and len(reading) == 1 else None
        t = time.perf_counter()
        finished = m.step([r.stream for r in self.active], budget)
        dt = time.perf_counter() - t
        if before is not None:                                                    # a piece read alone: what it taught about this computer
            got = m.lib.slot_tokens(reading[0].stream.slot) - before
            if got >= max(piece_n // 2, 1):
                tuner.seen(piece_n, got, dt)
        n = len(decoding)
        for r in decoding:
            r.steps += 1
            r.share += n
        self._flush(finished)
        if n >= 1 and not any(r.stream.t_first is None for r in self.active):
            self.pred.observe_step(n, dt, self.batch_level)
        self.steps_done = getattr(self, "steps_done", 0) + 1
        if self.steps_done % CHECK_EVERY == 0:
            self._adjust()

    def _reap(self):
        """Stop answers whose client left."""
        for r in list(self.active):
            if r.abandoned or (r.cancel is not None and r.cancel.is_set()):
                self.m.close_stream(r.stream)
                self.active.remove(r)
                if not r.abandoned:
                    r.q.put(Cancelled())

    def _flush(self, finished):
        for r in self.active:
            for text in r.stream.take():
                r.q.put({"delta": text})
            if r.logprobs is not None:
                lps = r.stream.take_logprobs()
                if lps:
                    r.q.put({"logprobs": lps})
        for st in finished:
            r = next((x for x in self.active if x.stream is st), None)
            if r is None:
                continue
            self.active.remove(r)
            if not self.active:                               # the last answer is done: give back what a hold had let go of, before the reply goes out
                with self.cv:
                    waiting = bool(self.inbox)
                if not waiting:
                    try:
                        self.m.settle()
                    except Exception:
                        pass
            if st.error is not None:                          # this answer failed on its own: only its app hears about it
                from .engine import EngineError
                print(f"[engine] one answer failed ({type(st.error).__name__}: {str(st.error)[:200]}); the others go on", flush=True)
                r.q.put(st.error if isinstance(st.error, constrain.DeadEnd) else EngineError(type(st.error).__name__))
                continue
            d = st.finish_up(self.batch_level)
            for text in st.take():
                r.q.put({"delta": text})
            if r.logprobs is not None:
                lps = st.take_logprobs()
                if lps:
                    r.q.put({"logprobs": lps})
            d["waited_s"] = round(r.waited, 3)
            d["total_s"] = round(r.waited + d["seconds"], 3)
            d["disk_mb"], d["disk_s"] = r.disk_mb, r.disk_s
            d["batch_avg"] = round(r.share / r.steps, 2) if r.steps else 1.0
            if r.goals is not None and goals_mod.is_set(r.goals):
                self._judge(d, r.goals, r.waited)
            self._learn(d)
            r.q.put({"done": d})

    # --- keeping speed goals while answers run ------------------------------------------------------------------------------
    def _adjust(self):
        """Every few steps: if a running answer's speed goal is missed, the whole step moves one size down. If everyone has room, one up."""
        if not self.active or self.batch_level is None or any(r.fixed for r in self.active):
            return
        n = len(self.active)
        goals = [r for r in self.active if r.goals and r.goals.get("min_tokens_per_s") is not None]
        if not goals:
            return
        now = time.perf_counter()
        last = getattr(self, "_adj", None)
        self._adj = (now, [r.stream.out for r in goals])
        if last is None or len(last[1]) != len(goals):
            return
        dt = now - last[0]
        if dt <= 0:
            return
        speeds = [(r.stream.out - o) / dt for r, o in zip(goals, last[1])]
        cap = min(r.max_pct for r in self.active)
        floor_q = max((r.goals["min_quality"] for r in self.active if r.goals and r.goals.get("min_quality") is not None), default=None)
        budgets = [self.mach["total_ram"] * r.goals["memory_pct"] / 100 for r in self.active if r.goals and r.goals.get("memory_pct") is not None]
        pool = sorted([l for l in self.levels if l["pct"] <= cap + 0.5 and not self.m.missing(l)
                       and (floor_q is None or facts.quality(l) >= floor_q)
                       and (not budgets or choose.memory_needed(self.machine(), l, facts.HOLD_CONTEXT) <= min(budgets))], key=lambda l: -l["pct"])
        i = next((k for k, l in enumerate(pool) if l["id"] == self.batch_level["id"]), None)
        if i is None:
            return
        slow = any(sp < 0.92 * r.goals["min_tokens_per_s"] for r, sp in zip(goals, speeds) if r.stream.t_first is not None)
        if slow and i + 1 < len(pool):
            self._record_switch(pool[i + 1])
        elif i > 0 and all(sp > 1.3 * r.goals["min_tokens_per_s"] for r, sp in zip(goals, speeds)):
            up = pool[i - 1]
            ratio = self.pred.time_factor(self.batch_level) / self.pred.time_factor(up)
            if all(sp * ratio >= 1.1 * r.goals["min_tokens_per_s"] for r, sp in zip(goals, speeds)):
                self._record_switch(up)

    def _record_switch(self, level):
        if self.m.missing(level):                          # only sizes whose weights are in memory (a switch must not stall every answer)
            return
        self._set_level(level)
