"""Who goes next when several apps wait for an engine that answers one at a time (MLX on the Mac).

The same rule as in batching.tier(): an app that asked for speed goes before an app that did not, and an app that said `power: "light"`
goes last. Inside a tier the order of arrival counts. A running answer is never stopped: the rule only decides who starts next."""
import contextlib
import itertools
import threading


class TurnQueue:
    def __init__(self):
        self.cv = threading.Condition()
        self.waiting = []              # (tier, order of arrival)
        self.busy = False
        self._order = itertools.count()

    @contextlib.contextmanager
    def take(self, tier, cancel=None):
        """Wait for the turn. `cancel` (a nest.cancel.Cancel) ends the wait when the client leaves."""
        me = (tier, next(self._order))
        with self.cv:
            self.waiting.append(me)
            try:
                while self.busy or min(self.waiting) != me:
                    if cancel is not None:
                        cancel.check()
                    self.cv.wait(0.1)
                self.busy = True
            finally:
                self.waiting.remove(me)
                self.cv.notify_all()
        try:
            yield
        finally:
            with self.cv:
                self.busy = False
                self.cv.notify_all()
