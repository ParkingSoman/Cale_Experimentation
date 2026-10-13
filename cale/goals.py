"""What an app can ask Cale for. Pure Python. The server parses a request here, the chooser (choose.py) uses the result.

| Goal | Meaning | Where it is judged |
|---|---|---|
| `min_tokens_per_s` | answer at least this fast (tokens per second) | predicted before, measured after, kept during by stepping the size |
| `max_first_token_s` | the first token within this many seconds of the call arriving (waiting in line, reading the prompt) | predicted before, measured after |
| `memory_pct` | stay under this share of the computer's RAM, counting the weights and the context | before |
| `min_quality` | never use a size that repeats less than this share (0 to 1) of the full size's next-word choices | before |
| `power` | `"light"`: on battery or when hot, run a smaller size | before |

When goals pull in different directions, `priority` says which to keep (most important first).
When no size meets the goals, `on_miss` says what to do: `best_effort` (run the best size and say what missed),
`refuse` (422, nothing runs), `wait` (try again for `max_wait_s`, then refuse), `try` (run anyway, every lever allowed).
Nothing overrides the safety limit: Cale never runs what it predicts would freeze or crash the computer.
"""
import math

SPEED, FIRST_TOKEN, MEMORY, QUALITY = "speed", "first_token", "memory", "quality"
DEFAULT_PRIORITY = (MEMORY, QUALITY, FIRST_TOKEN, SPEED)
MAX_WAIT_S = 60
ON_MISS = ("best_effort", "refuse", "wait", "try")
ON_MISS_ALIASES = {"smallest": "best_effort"}          # the name of the same behavior before Phase 6
GOAL_FIELDS = ("min_tokens_per_s", "max_first_token_s", "memory_pct", "min_quality")
OPTION_FIELDS = ("power", "priority", "on_miss", "max_wait_s", "allow_disk")
FIELDS = GOAL_FIELDS + OPTION_FIELDS


class GoalError(Exception):
    def __init__(self, code, message, param=None):
        super().__init__(message)
        self.code, self.param = code, param


def _num(req, key, name, lo, hi):
    v = req.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise GoalError(f"invalid_{name}", f"'{key}' must be a number from {lo} to {hi}.", key)
    return float(v)


def parse(req, prefix=""):
    """Goals from a request body. prefix is "" for a model object and "cale_" for the fields of one call.
    Returns a dict with every field present (None when not given). Raises GoalError."""
    if f"{prefix}deadline_s" in req:
        raise GoalError("unsupported_parameter",
                        f"'{prefix}deadline_s' was removed: Cale cannot know how long an answer will be. State what you need instead: "
                        f"'{prefix}min_tokens_per_s' (answer speed) and/or '{prefix}max_first_token_s' (time until the first token).",
                        f"{prefix}deadline_s")
    g = {"min_tokens_per_s": _num(req, prefix + "min_tokens_per_s", "min_tokens_per_s", 0.1, 10000),
         "max_first_token_s": _num(req, prefix + "max_first_token_s", "max_first_token_s", 0.05, 3600),
         "memory_pct": _num(req, prefix + "memory_pct", "memory_pct", 1, 100),
         "min_quality": _num(req, prefix + "min_quality", "min_quality", 0.0, 1.0)}
    power = req.get(prefix + "power", "normal")
    if power not in ("normal", "light"):
        raise GoalError("invalid_power", f"'{prefix}power' must be \"normal\" or \"light\".", prefix + "power")
    g["power"] = power
    om = ON_MISS_ALIASES.get(req.get(prefix + "on_miss", "best_effort"), req.get(prefix + "on_miss", "best_effort"))
    if om not in ON_MISS:
        raise GoalError("invalid_on_miss", f"'{prefix}on_miss' must be one of: {', '.join(ON_MISS)}.", prefix + "on_miss")
    g["on_miss"] = om
    # At most 60 s: while Cale waits it holds the generation lock, so other apps wait too.
    g["max_wait_s"] = _num(req, prefix + "max_wait_s", "max_wait_s", 0.0, MAX_WAIT_S) if req.get(prefix + "max_wait_s") is not None else 30.0
    allow = req.get(prefix + "allow_disk", False)
    if not isinstance(allow, bool):
        raise GoalError("invalid_allow_disk", f"'{prefix}allow_disk' must be true or false.", prefix + "allow_disk")
    g["allow_disk"] = allow
    pr = req.get(prefix + "priority")
    if pr is None:
        g["priority"] = DEFAULT_PRIORITY
    else:
        names = {"speed": SPEED, "first_token": FIRST_TOKEN, "memory": MEMORY, "quality": QUALITY}
        if not isinstance(pr, list) or not pr or len(set(pr)) != len(pr) or any(x not in names for x in pr):
            raise GoalError("invalid_priority", f"'{prefix}priority' must be a list of distinct names from: {', '.join(names)}.", prefix + "priority")
        g["priority"] = tuple(pr) + tuple(x for x in DEFAULT_PRIORITY if x not in pr)         # what is not named comes last, in the default order
    return g


def is_set(g):
    """True if any goal (or a light power setting) is given: then Cale chooses the size."""
    return g is not None and (any(g.get(k) is not None for k in GOAL_FIELDS) or g.get("power") == "light")


def given(g):
    """The goals that are set, as {name: value}, for replies."""
    return {k: g[k] for k in GOAL_FIELDS + ("power",) if g.get(k) is not None and not (k == "power" and g[k] == "normal")}
