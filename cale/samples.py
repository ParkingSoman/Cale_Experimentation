"""Passive samples: numbers about what happened, kept on this computer only, to make Cale's predictions better over time.

Rules (docs/research/adherence-sampling.md):
- Passive only. A sample is a reading of something that happened anyway (a finished call, a timer, a counter). It never runs the model
  and never waits for anything, so it cannot make an app miss a goal or change the size Cale picks.
- Numbers only. No prompt, no answer, no app name (so one app's data cannot leak to another app that reads this file). Tiers and sizes are numbers.
- Small. One short line per event. The file keeps the newest KEEP lines.
- Never raises. A full disk or a locked file loses a sample, nothing else.
"""
import json
import queue
import threading
import time

KEEP = 4000                 # POLICY: lines kept (about 1 MB)
TRIM_EVERY = 500            # writes between looks at the file size


class Samples:
    def __init__(self, path):
        self.path = path
        self.mu = threading.Lock()
        self.writes = 0
        self.recent = []                      # the newest lines in memory (the predictor and the Hub read these, not the file)
        self.q = queue.SimpleQueue()          # lines waiting to be written: the caller never waits for the disk
        self.writer = None

    def record(self, kind, **fields):
        """Add one line. `fields` are numbers, booleans or short words. Returns the line, or None if it could not be kept. Never waits for the disk."""
        try:
            line = {"t": round(time.time(), 1), "kind": kind}
            line.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fields.items() if v is not None})
            with self.mu:
                self.recent.append(line)
                del self.recent[:-200]
                if self.writer is None:
                    self.writer = threading.Thread(target=self._write_loop, daemon=True)
                    self.writer.start()
            self.q.put(json.dumps(line, separators=(",", ":")))
            return line
        except Exception:
            return None

    def _write_loop(self):
        while True:
            text = self.q.get()
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
                self.writes += 1
                if self.writes % TRIM_EVERY == 0:
                    self._trim()
            except Exception:
                pass                          # a full disk or a locked file loses a sample, nothing else

    def flush(self, timeout=2.0):
        """Wait until the lines written so far are on disk (a short-lived process such as the installer needs this before it exits)."""
        end = time.time() + timeout
        while time.time() < end and not self.q.empty():
            time.sleep(0.01)
        time.sleep(0.05)

    def _trim(self):
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if len(lines) > 2 * KEEP:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text("\n".join(lines[-KEEP:]) + "\n", encoding="utf-8")
                tmp.replace(self.path)
        except Exception:
            pass

    def read(self, kind=None, limit=KEEP):
        """The newest lines from the file (all kinds, or one)."""
        try:
            out = []
            for l in self.path.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    d = json.loads(l)
                except ValueError:
                    continue
                if kind is None or d.get("kind") == kind:
                    out.append(d)
            return out
        except Exception:
            return []


def step_summary(dts):
    """Time of one step (about one token) over an answer: count, median, 95th percentile, worst, in milliseconds. A stall shows in the worst and the 95th."""
    if not dts:
        return None
    s = sorted(dts)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]          # noqa: E731
    return {"n": len(s), "p50": round(1000 * pick(0.5), 1), "p95": round(1000 * pick(0.95), 1), "max": round(1000 * s[-1], 1)}
