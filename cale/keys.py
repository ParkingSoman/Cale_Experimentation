"""App keys. Only a hash of each key is stored. A key belongs to one app name.
A key for a website also records the website's Origin. It works only on requests that carry that same Origin.
A key for an app has no origin. It works only on requests that carry no Origin."""
import hashlib
import json
import secrets
import time

from . import oslayer, paths


def _read():
    try:
        return json.loads(paths.KEYS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"apps": {}}


def _write(data):
    paths.HOME.mkdir(parents=True, exist_ok=True)
    oslayer.private_file(paths.KEYS, json.dumps(data, indent=1))


def _locked():
    """One writer at a time, across the server and the `cale` command."""
    return oslayer.file_lock(paths.HOME / "keys.lock")


def _hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def create(app, role=None, origin=None):
    """Make a new key for the app. Any older key for the same app stops working.
    role="hub" is for the Cale Hub only: it may read what every app does.
    origin is set for a website (see cale/server.py): the key then works only for requests from that Origin."""
    key = "cale-" + secrets.token_urlsafe(24)
    with _locked():
        data = _read()
        data["apps"][app] = {"sha256": _hash(key), "created": int(time.time()), **({"role": role} if role else {}), **({"origin": origin} if origin else {})}
        _write(data)
    return key


def role(app):
    return _read()["apps"].get(app, {}).get("role")


HUB_APP = "Cale Hub"


def hub_key():
    """The Hub's key. Made without a question, because the Hub is part of Cale. Kept in a private file."""
    f = paths.HOME / "hub.key"
    try:
        key = f.read_text().strip()
        if check(key) == HUB_APP and role(HUB_APP) == "hub":
            return key
    except FileNotFoundError:
        pass
    key = create(HUB_APP, role="hub")
    paths.HOME.mkdir(parents=True, exist_ok=True)
    oslayer.private_file(f, key)
    return key


def check(key):
    """App name for a valid key, else None. The file is read each time, so a revoke acts at once."""
    h = _hash(key or "")
    for app, rec in _read()["apps"].items():
        if secrets.compare_digest(rec["sha256"], h):
            return app
    return None


def lookup(key):
    """(app name, origin) for a valid key, else None. origin is None for an app key."""
    h = _hash(key or "")
    for app, rec in _read()["apps"].items():
        if secrets.compare_digest(rec["sha256"], h):
            return app, rec.get("origin")
    return None


def apps():
    return {a: r["created"] for a, r in _read()["apps"].items()}


def revoke(app):
    with _locked():
        data = _read()
        found = data["apps"].pop(app, None) is not None
        _write(data)
    return found
