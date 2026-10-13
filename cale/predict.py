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
        self.by_level = {}                                              # level id -> recent decode speeds (tokens/s)
        self.slow = 1.0                                                 # debug only: pretend this computer is this many times slower
        self.busy = lambda: 0.0                                         # share of the CPU other programs use now (set by the engine)
        self._load()

    # --- what is predicted ----------------------------------------------------------------------
    def time_factor(self, level):
        """Time per answer token at this size, relative to the full size."""
        return self.fixed + (1.0 - self.fixed) * facts.cost(level, self.full)

    def live_factor(self):
        """Other programs using the computer now slow both memory and arithmetic. 1.0 = idle."""
        return max(0.35, 1.0 - 0.5 * min(max(self.busy(), 0.0), 1.0)) / self.slow

    def decode_tps(self, level):
        return self.dec_full / self.time_factor(level) * self.live_factor()

    def prefill_tps(self, level):
        return self.pre_full / facts.cost(level, self.full) * self.live_factor()

    def first_token_s(self, level, prompt_tokens, disk_s=0.0):
        """Seconds from the start of the work to the first token: reading the prompt, then one token."""
        return disk_s + prompt_tokens / self.prefill_tps(level) + 1.0 / self.decode_tps(level)

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
        tokens, decode_s = int(done.get("tokens") or 0), float(done.get("decode_s") or 0.0)
        if tokens >= 9 and decode_s > 0 and not done.get("switches"):
            tps = (tokens - 1) / decode_s
            hist = self.by_level.setdefault(level["id"], [])
            hist.append(tps)
            del hist[:-KEEP]
            implied = tps * self.time_factor(level) / live
            w = FIRST_WEIGHT if self.n_dec == 0 else LATER_WEIGHT
            self.dec_full = (1 - w) * self.dec_full + w * implied
            self.n_dec += 1
            self._fit_fixed()
        prompt, first = int(done.get("prompt_tokens") or 0), float(done.get("first_token_s") or 0.0) - float(done.get("disk_s") or 0.0)
        if prompt >= 32 and first > 0:
            read_s = first - 1.0 / max(self.decode_tps(level), 1e-6)
            if read_s > 0:
                implied = prompt / read_s * facts.cost(level, self.full) / live
                w = FIRST_WEIGHT if self.n_pre == 0 else LATER_WEIGHT
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

    def _save(self):
        if not self.path or (self.n_dec + self.n_pre) % 3 not in (1, 0):
            return
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "key": self.key, "dec_full": self.dec_full, "pre_full": self.pre_full,
                                       "fixed": self.fixed, "n_dec": self.n_dec, "n_pre": self.n_pre, "disk_bps": self.disk_bps,
                                       "sizes": self.snapshot(),
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
