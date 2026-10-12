"""App keys. Only a hash of each key is stored. A key belongs to one app name."""
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


def create(app):
    """Make a new key for the app. Any older key for the same app stops working."""
    key = "cale-" + secrets.token_urlsafe(24)
    with _locked():
        data = _read()
        data["apps"][app] = {"sha256": _hash(key), "created": int(time.time())}
        _write(data)
    return key


def check(key):
    """App name for a valid key, else None. The file is read each time, so a revoke acts at once."""
    h = _hash(key or "")
    for app, rec in _read()["apps"].items():
        if secrets.compare_digest(rec["sha256"], h):
            return app
    return None


def apps():
    return {a: r["created"] for a, r in _read()["apps"].items()}


def revoke(app):
    with _locked():
        data = _read()
        found = data["apps"].pop(app, None) is not None
        _write(data)
    return found
