"""What Cale remembers about each website that asked to use it. Standard library only.

The rules are the ones Chromium uses for permission questions (components/permissions/permission_decision_auto_blocker.cc).
They are counted for each origin on its own. There is no limit across sites.
  - "Don't Allow" is remembered until the person resets it (`cale forget SITE`). It survives restarts.
  - A question that gets no answer counts as ignored. After IGNORES_BEFORE_BLOCK ignored questions, the origin waits EMBARGO_DAYS days.
  - An allow clears every count for that origin.
Chromium also counts "dismissals" (3, then 7 days). Cale's question has no separate dismiss button, so that is not built.
"""
import json
import time

from . import oslayer, paths

# Names and values from Chromium: permission_decision_auto_blocker.cc
DEFAULT_IGNORES_BEFORE_BLOCK = 4     # kDefaultIgnoresBeforeBlock
DEFAULT_EMBARGO_DAYS = 7             # kDefaultEmbargoDays
EMBARGO_S = DEFAULT_EMBARGO_DAYS * 86400

BLOCKED = "blocked"                  # the person pressed "Don't Allow"


def _read():
    try:
        data = json.loads(paths.SITES.read_text())
        if isinstance(data.get("sites"), dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
        pass
    return {"sites": {}}


def _write(data):
    paths.HOME.mkdir(parents=True, exist_ok=True)
    oslayer.private_file(paths.SITES, json.dumps(data, indent=1))     # mode 0600, written to a temp file and swapped in


def _locked():
    """One writer at a time, across the server and the `cale` command."""
    return oslayer.file_lock(paths.HOME / "sites.lock")


def normalize(site):
    """'rewrite.example.com' and 'https://rewrite.example.com/' both mean the origin 'https://rewrite.example.com'."""
    site = str(site).strip().rstrip("/")
    return site if "://" in site else "https://" + site


def check(origin, now=None):
    """None if this origin may ask. Else (code, retry_after_s): ("site_blocked", None) or ("asked_recently", seconds to wait)."""
    now = time.time() if now is None else now
    rec = _read()["sites"].get(origin)
    if not rec:
        return None
    if rec.get("state") == BLOCKED:
        return "site_blocked", None
    until = rec.get("embargo_until", 0)
    if until > now:
        return "asked_recently", int(until - now) + 1
    return None


def record(origin, how, now=None):
    """Remember the person's answer. how: "allow", "deny", "timeout" or "error". "error" changes nothing."""
    if how not in ("allow", "deny", "timeout"):
        return
    now = time.time() if now is None else now
    with _locked():
        data = _read()
        sites = data["sites"]
        rec = sites.get(origin, {})
        if how == "allow":
            sites.pop(origin, None)                                  # an allow clears the counts
        elif how == "deny":
            sites[origin] = {"state": BLOCKED, "ignores": 0, "embargo_until": 0, "since": int(now)}
        elif rec.get("state") != BLOCKED:
            if rec.get("embargo_until", 0) and rec["embargo_until"] <= now:
                rec = {}                                             # an ended embargo starts the count again (Chromium does the same)
            ignores = int(rec.get("ignores", 0)) + 1
            until = now + EMBARGO_S if ignores >= DEFAULT_IGNORES_BEFORE_BLOCK else 0
            sites[origin] = {"state": "open", "ignores": ignores, "embargo_until": until, "since": int(rec.get("since", now))}
        _write(data)


def forget(site):
    """Reset everything about one site. Takes 'rewrite.example.com' or the full origin. True if it was known."""
    origin = normalize(site)
    with _locked():
        data = _read()
        found = data["sites"].pop(origin, None) is not None
        if found:
            _write(data)
    return found


def listing():
    """{origin: {"state", "ignores", "embargo_until", "since"}} for every site Cale remembers."""
    return dict(_read()["sites"])
