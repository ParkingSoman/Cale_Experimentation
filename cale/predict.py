"""Predict how fast each size runs on this computer, without timing the model. Pure Python.

How: producing one answer token reads every active weight once, so its speed is memory speed divided by the bytes read.
Gemma's facts (gemma3n_facts.py) give the bytes of every size. The computer's memory speed is probed in about 0.15 s
(the engine does that, no model needed). Reading a prompt is limited by arithmetic and scales with the same sizes.
Then every real answer corrects the numbers (a weighted average, so the first answer counts most), and what was learned
is saved, so the next start is already right. Nothing is ever timed on purpose: the first start uses priors, flagged
as `estimate`.

Two unknowns per computer are learned: the speed of the full size for answers (`dec_full`) and for prompts (`pre_full`).
A third, the part of per-token time that does not shrink with the size (`fixed`), starts from a prior per engine and
is fitted once answers at two different sizes have been seen.
"""
import json
import math
import os
import statistics
import time

from . import gemma3n_facts as facts

# Priors per engine, from the three test computers (see FINDINGS, Phase 6 stage 4). They matter only until the first answer.
PRIORS = {
    # eff: answer speed x bytes per token / probed memory speed. Mac M-series 34.2 tok/s x 2.24 GB / 130 GB/s = 0.59.
    # HP i7-8550U 6.9 tok/s x 2.24 GB / 28.7 GB/s = 0.54. prefill_ratio: prompt speed / answer speed (Mac 790/34.2, HP 17/6.9).
    # fixed: the share of per-token time that does not shrink with the size (Mac 0.33, HP 0.135, from the e4b and e2b speeds).
    "mlx": {"eff": 0.62, "prefill_ratio": 23.0, "fixed": 0.27},
    "llama": {"eff": 0.54, "prefill_ratio": 2.5, "fixed": 0.135},
}
KEEP = 9               # answers kept per size for the median
BATCH_EPS = 0.7        # each extra answer in a step adds this share of the time of one. Measured: Mac CPU 0.4-0.6, HP CPU 0.8-0.9 (llama.cpp, 2 and 4 answers). Learned per computer after that.
DISK_PRIOR = 1.0e9     # bytes per second until a real read has been seen (a modest SSD)
FIRST_WEIGHT = 0.8     # the first real answer replaces the prior almost fully
LATER_WEIGHT = 0.35    # later answers move the number gently


class Predictor:
    def __init__(self, engine, bandwidth, key=None, path=None, levels=None):
        self.engine = engine if engine in PRIORS else "llama"
        self.levels = levels or facts.levels()
        self.full = facts.full_level(self.levels)
        self.path, self.key = path, key or {}
        pr = PRIORS[self.engine]
        self.bandwidth = float(bandwidth or 0.0)
        bw = self.bandwidth if self.bandwidth > 0 else 20e9            # unknown memory speed: assume a modest laptop
        self.dec_full = bw * pr["eff"] / facts.bytes_per_token(self.full)
        self.pre_full = self.dec_full * pr["prefill_ratio"]
        self.fixed = pr["fixed"]
        self.n_dec = self.n_pre = 0
        self.disk_bps, self.disk_samples = DISK_PRIOR, []
        self.batch_eps, self.batch_n = BATCH_EPS, 0
        self.extra = {}                                                 # measured at install by `cale tune`: scratch_bytes, threads, piece_speed
        self.table = {}                                                 # measured at install: level id -> {"dec": tokens/s, "pre": tokens/s} for a short prompt
        self.pre_curve = []                                             # measured at install, full size: [(prompt tokens, prompt tokens/s)] by length
        self.dec_ctx = 0.0                                              # measured at install: share of answer speed lost per 1000 tokens of context
        self.sustained = 1.0                                            # measured at install: speed after minutes of work / speed in the first seconds (1 = no throttling)
        self.corr_dec = self.corr_pre = 1.0                             # what real answers say about the measured table (1 = it is right)
        self.heat, self.heat_at = 0.0, None                             # 0 = cool, 1 = as hot as after the sustained test: recent work heats, idle time cools
        self.by_level = {}                                              # level id -> recent decode speeds (tokens/s)
        self.slow = 1.0                                                 # debug only: pretend this computer is this many times slower
        self.busy = lambda: 0.0                                         # share of the CPU other programs use now (set by the engine)
        self._load()

    # --- what is predicted ----------------------------------------------------------------------
    def time_factor(self, level):
        """Time per answer token at this size, relative to the full size."""
        return self.fixed + (1.0 - self.fixed) * facts.cost(level, self.full)

    def heat_now(self, now=None):
        now = time.time() if now is None else now
        if self.heat_at is None:
            return self.heat
        return self.heat * math.exp(-max(0.0, now - self.heat_at) / float(self.extra.get("cool_s", 120.0)))

    def note_busy(self, seconds, now=None):
        """The engine worked for `seconds`: the computer is that much hotter. A laptop slows down once it is hot (`sustained`)."""
        now = time.time() if now is None else now
        self.heat = min(1.0, self.heat_now(now) + seconds / max(float(self.extra.get("burst_s", 30.0)), 1.0))
        self.heat_at = now

    def live_factor(self):
        """Other programs using the computer now slow both memory and arithmetic, and a hot laptop is slower than a cool one. 1.0 = idle and cool."""
        throttle = 1.0 - (1.0 - self.sustained) * self.heat_now() if self.sustained < 0.999 else 1.0
        return max(0.35, 1.0 - 0.5 * min(max(self.busy(), 0.0), 1.0)) * throttle / self.slow

    def _table_value(self, level, key):
        """The measured speed of this size, or between the two nearest measured sizes (by cost), or None if nothing was measured."""
        if not self.table:
            return None
        if level["id"] in self.table:
            return self.table[level["id"]][key]
        pts = sorted(((facts.cost(l, self.full), self.table[l["id"]][key]) for l in self.levels if l["id"] in self.table))
        c = facts.cost(level, self.full)
        lo = [p for p in pts if p[0] <= c]
        hi = [p for p in pts if p[0] >= c]
        if lo and hi:
            (c0, v0), (c1, v1) = lo[-1], hi[0]
            return v0 if c1 == c0 else v0 + (v1 - v0) * (c - c0) / (c1 - c0)
        c0, v0 = (lo[-1] if lo else hi[0])
        return v0 * (c0 / c) if key == "pre" else v0

    def decode_tps(self, level, ctx=0, answer_tokens=None):
        """Answer speed at this size, with `ctx` tokens already in the context and (if given) over an answer of that many tokens, which
        slows to the sustained speed once the burst is over."""
        t = self._table_value(level, "dec")
        if t is None:
            return self.dec_full / self.time_factor(level) * self.live_factor()
        v = t * self.corr_dec * self.live_factor() / (1.0 + self.dec_ctx * ctx / 1000.0)
        if answer_tokens and self.sustained < 0.999:
            burst = float(self.extra.get("burst_s", 20.0)) * v                 # tokens produced before the speed settles
            n = float(answer_tokens)
            if n > burst:
                return n / (burst / v + (n - burst) / (v * self.sustained))
        return v

    def prefill_tps(self, level, tokens=None):
        """Prompt speed at this size. With `tokens`, the speed for a prompt of that length (long prompts are a little slower per token)."""
        t = self._table_value(level, "pre")
        if t is None:
            return self.pre_full / facts.cost(level, self.full) * self.live_factor()
        v = t * self.corr_pre * self.live_factor()
        if tokens and len(self.pre_curve) >= 2:
            ref = self._curve(self.table_prompt)
            v *= self._curve(tokens) / ref if ref else 1.0
        return v

    table_prompt = 256                                                         # the prompt length the table was measured at

    def _curve(self, n):
        """Measured prompt speed (full size) at a prompt length: straight lines between the measured points, flat beyond them."""
        pts = sorted(self.pre_curve)
        if n <= pts[0][0]:
            return pts[0][1]
        for (n0, v0), (n1, v1) in zip(pts, pts[1:]):
            if n <= n1:
                return v0 + (v1 - v0) * (n - n0) / (n1 - n0)
        return pts[-1][1]

    def first_token_s(self, level, prompt_tokens, disk_s=0.0):
        """Seconds from the start of the work to the first token: reading the prompt, then one token."""
        return disk_s + prompt_tokens / self.prefill_tps(level, prompt_tokens) + 1.0 / self.decode_tps(level, prompt_tokens)

    def batch_cost(self, n):
        """Time of one step with n answers, relative to a step with one."""
        return 1.0 + self.batch_eps * max(n - 1, 0)

    def observe_step(self, n, step_s, level):
        """A step of n answers took step_s: learn how much each extra answer costs. Needs a settled solo speed to compare with."""
        if n < 2 or self.n_dec < 2 or step_s <= 0:
            return
        solo_s = 1.0 / self.decode_tps(level)
        eps = max(0.0, min(2.0, (step_s / solo_s - 1.0) / (n - 1)))
        w = 0.1
        self.batch_eps = (1 - w) * self.batch_eps + w * eps
        self.batch_n += 1

    @property
    def confidence(self):
        return "estimate" if self.n_dec == 0 or self.n_pre == 0 else "learned"

    # --- learning -------------------------------------------------------------------------------
    def observe(self, done):
        """Fold one finished answer (the `done` record of the engine) into the numbers. Answers that changed size are skipped."""
        if self.slow != 1.0:
            return
        level = next((l for l in self.levels if l["id"] == done.get("tier")), None)
        if level is None:
            return
        disk_bytes = float(done.get("disk_mb") or 0) * 1e6
        if disk_bytes > 50_000_000 and float(done.get("disk_s") or 0) > 0:
            self.disk_samples = (self.disk_samples + [disk_bytes / float(done["disk_s"])])[-KEEP:]
            self.disk_bps = statistics.median(self.disk_samples)
        live = self.live_factor()
        tokens = int(done.get("tokens") or 0)
        decode_s = float(done.get("decode_s") or 0.0) - float(done.get("paused_dec_s") or 0.0)      # an answer held back for another app was waiting, not decoding
        if tokens >= 9 and decode_s > 0 and not done.get("switches") and float(done.get("batch_avg") or 1.0) < 1.05:
            tps = (tokens - 1) / decode_s
            hist = self.by_level.setdefault(level["id"], [])
            hist.append(tps)
            del hist[:-KEEP]
            w = FIRST_WEIGHT if self.n_dec == 0 else LATER_WEIGHT
            if self.table and self._table_value(level, "dec"):
                ctx = int(done.get("prompt_tokens") or 0)
                want = self._table_value(level, "dec") / (1.0 + self.dec_ctx * ctx / 1000.0)
                self.corr_dec = (1 - LATER_WEIGHT) * self.corr_dec + LATER_WEIGHT * (tps / live / want)    # a measured table moves gently
            else:
                implied = tps * self.time_factor(level) / live
                self.dec_full = (1 - w) * self.dec_full + w * implied
            self.n_dec += 1
            self._fit_fixed()
        prompt = int(done.get("prompt_tokens") or 0) - int(done.get("cached_tokens") or 0)      # only the tokens that were really read
        first = float(done.get("first_token_s") or 0.0) - float(done.get("disk_s") or 0.0) - float(done.get("paused_pre_s") or 0.0)
        if prompt >= 32 and first > 0 and float(done.get("batch_avg") or 1.0) < 1.05:
            read_s = first - 1.0 / max(self.decode_tps(level), 1e-6)
            if read_s > 0:
                w = FIRST_WEIGHT if self.n_pre == 0 else LATER_WEIGHT
                if self.table and self._table_value(level, "pre"):
                    base = self._table_value(level, "pre")
                    if len(self.pre_curve) >= 2:
                        base *= self._curve(prompt) / self._curve(self.table_prompt)
                    self.corr_pre = (1 - LATER_WEIGHT) * self.corr_pre + LATER_WEIGHT * (prompt / read_s / live / base)
                else:
                    implied = prompt / read_s * facts.cost(level, self.full) / live
                    self.pre_full = (1 - w) * self.pre_full + w * implied
                self.n_pre += 1
        self._save()

    def _fit_fixed(self):
        """Once answers at two clearly different sizes are known, solve for the part of per-token time that does not shrink."""
        pts = [(l, statistics.median(self.by_level[l["id"]])) for l in self.levels if self.by_level.get(l["id"])]
        if len(pts) < 2:
            return
        (a, ta), (b, tb) = max(pts, key=lambda p: facts.cost(p[0], self.full)), min(pts, key=lambda p: facts.cost(p[0], self.full))
        ca, cb = facts.cost(a, self.full), facts.cost(b, self.full)
        if ca - cb < 0.2:
            return
        r = tb / ta                          # speed ratio small over large: (f + (1-f) ca) / (f + (1-f) cb)
        den = r - r * cb - 1 + ca
        if abs(den) > 1e-6:
            self.fixed = min(0.6, max(0.02, (ca - r * cb) / den))

    # --- saving ---------------------------------------------------------------------------------
    def _load(self):
        if not self.path:
            return
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if d.get("key") != self.key or d.get("version") != 1:
            return
        self.dec_full, self.pre_full, self.fixed = d["dec_full"], d["pre_full"], d["fixed"]
        self.n_dec, self.n_pre = d.get("n_dec", 0), d.get("n_pre", 0)
        self.disk_bps = d.get("disk_bps", self.disk_bps)
        self.extra = d.get("extra") or {}
        self.table = d.get("table") or {}
        self.table_prompt = int(d.get("table_prompt", self.table_prompt))
        self.pre_curve = [tuple(p) for p in (d.get("pre_curve") or [])]
        self.dec_ctx, self.sustained = float(d.get("dec_ctx", 0.0)), float(d.get("sustained", 1.0))
        self.corr_dec, self.corr_pre = float(d.get("corr_dec", 1.0)), float(d.get("corr_pre", 1.0))
        self.batch_eps = float(d.get("batch_eps", self.batch_eps))

    def save(self):
        self._save(force=True)

    def _save(self, force=False):
        if not self.path or (not force and (self.n_dec + self.n_pre) % 3 not in (1, 0)):
            return
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "key": self.key, "dec_full": self.dec_full, "pre_full": self.pre_full,
                                       "fixed": self.fixed, "n_dec": self.n_dec, "n_pre": self.n_pre, "disk_bps": self.disk_bps,
                                       "sizes": self.snapshot(), "extra": self.extra, "table": self.table, "table_prompt": self.table_prompt, "pre_curve": self.pre_curve,
                                       "dec_ctx": self.dec_ctx, "sustained": self.sustained, "corr_dec": self.corr_dec, "corr_pre": self.corr_pre,
                                       "batch_eps": self.batch_eps,
                                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
            os.replace(tmp, self.path)
        except OSError:
            pass

    def snapshot(self):
        """Predicted speed of every size, kept in the saved file so /v1/models can show it before the model is loaded."""
        return {l["id"]: {"size_pct": l["pct"], "decode_tps": round(self.decode_tps(l), 2), "prefill_tps": round(self.prefill_tps(l), 1)}
                for l in self.levels}

    def summary(self):
        return {"confidence": self.confidence, "memory_gb_s": round(self.bandwidth / 1e9, 1),
                "decode_tps_full": round(self.decode_tps(self.full), 2), "prefill_tps_full": round(self.prefill_tps(self.full), 1),
                "answers_learned_from": self.n_dec}
