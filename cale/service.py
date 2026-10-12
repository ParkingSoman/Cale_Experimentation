"""Start and stop Cale. Standard library only, so apps can import it without loading the model code.

On a Mac where the installer made a LaunchAgent, launchd keeps Cale running. On every other computer
Cale runs as a plain detached process that this module starts, and an app can start it on demand.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import oslayer, paths

PLIST = os.path.expanduser(f"~/Library/LaunchAgents/{paths.LABEL}.plist")
PIDFILE = paths.HOME / "cale.pid"


def use_launchd():
    return oslayer.MAC and os.path.exists(PLIST)


def _domain():
    return f"gui/{os.getuid()}"


def _launchctl(*a):
    return subprocess.run(["launchctl", *a], capture_output=True, text=True)


def status(timeout=3):
    """The /cale/status answer as a dict, or None if Cale does not answer."""
    try:
        with urllib.request.urlopen(f"http://{paths.HOST}:{paths.PORT}/cale/status", timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def running():
    return status() is not None


def _launchd_loaded():
    return _launchctl("print", f"{_domain()}/{paths.LABEL}").returncode == 0


def _command():
    """(python, env) for the server. Uses the installed copy if there is one, else the folder this file is in."""
    py = oslayer.venv_python(paths.HOME / "venv")
    runtime = paths.HOME / "runtime"
    if not py.exists():
        py, runtime = Path(sys.executable), Path(__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=str(runtime), CALE_HOME=str(paths.HOME), PYTHONUNBUFFERED="1",
               CALE_ON_DEMAND="1")      # no service manager: Cale ends itself when nothing has used it for a while
    return str(py), env


def start(wait=60):
    """Make sure Cale is running. Returns (ok, message). Waits until it answers, up to `wait` seconds."""
    if running():
        return True, "Cale is already running."
    if use_launchd():
        if _launchd_loaded():
            r = _launchctl("kickstart", f"{_domain()}/{paths.LABEL}")
        else:
            r = _launchctl("bootstrap", _domain(), PLIST)
            if r.returncode:
                time.sleep(2)
                r = _launchctl("bootstrap", _domain(), PLIST)
        if r.returncode:
            return False, f"Could not start Cale: {r.stderr.strip() or 'launchctl failed'}"
    else:
        py, env = _command()
        pid = oslayer.spawn_detached([py, "-m", "cale.server"], paths.LOGS / "cale.log", env)
        paths.HOME.mkdir(parents=True, exist_ok=True)
        PIDFILE.write_text(str(pid))
    for _ in range(int(wait * 2)):
        if running():
            return True, "Cale is running."
        time.sleep(0.5)
    return False, f"Cale did not answer within {wait} s. Look at the log: {paths.LOGS / 'cale.log'}"


def stop():
    if use_launchd():
        _launchctl("bootout", f"{_domain()}/{paths.LABEL}")
        for _ in range(20):                     # bootout is not instant
            if not _launchd_loaded():
                break
            time.sleep(0.5)
        return True
    try:
        pid = int(PIDFILE.read_text())
    except (OSError, ValueError):
        return not running()
    oslayer.kill_pid(pid)
    for _ in range(40):
        if not running():
            break
        time.sleep(0.25)
    try:
        PIDFILE.unlink()
    except OSError:
        pass
    return not running()
