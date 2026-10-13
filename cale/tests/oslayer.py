"""Tests for cale/oslayer.py and the modules that use it. Runs on Mac, Windows and Linux. No model needed.

usage: python cale/tests/oslayer.py          (from the folder that holds the `cale` folder)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale import oslayer  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def run_py(code, env_extra=None, timeout=60):
    env = dict(os.environ, PYTHONPATH=str(ROOT), **(env_extra or {}))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout)


def main():
    print(f"os: {oslayer.NAME}  python {sys.version.split()[0]}", flush=True)

    # 1. Folder choice, checked for every OS from one computer.
    h = Path("/home/u")
    check("home mac", oslayer.home_dir({}, "darwin", h) == h / "Library" / "Application Support" / "Cale")
    check("home linux", oslayer.home_dir({}, "linux", h) == h / ".local" / "share" / "cale")
    check("home linux xdg", oslayer.home_dir({"XDG_DATA_HOME": "/x"}, "linux", h) == Path("/x/cale"))
    check("home windows", oslayer.home_dir({"LOCALAPPDATA": "C:/L"}, "win32", h) == Path("C:/L/Cale"))
    check("home override", oslayer.home_dir({"CALE_HOME": "/c"}, "win32", h) == Path("/c"))

    # 2. Pressure rules, checked from sample text.
    calm = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0"
    busy = "some avg10=22.50 avg60=3.00 avg300=1.00 total=9\nfull avg10=1.00 avg60=0.00 avg300=0.00 total=1"
    bad = "some avg10=50.00 avg60=3.00 avg300=1.00 total=9\nfull avg10=14.00 avg60=0.00 avg300=0.00 total=1"
    check("psi calm = 1", oslayer.level_from_psi(calm) == 1)
    check("psi busy = 2", oslayer.level_from_psi(busy) == 2)
    check("psi bad = 4", oslayer.level_from_psi(bad) == 4)
    check("free 50% = 1", oslayer.level_from_free(50, 100) == 1)
    check("free 8% = 2", oslayer.level_from_free(8, 100) == 2)
    check("free 2% = 4", oslayer.level_from_free(2, 100) == 4)
    check("windows event = 4", oslayer.win_level(True, 10) == 4)
    check("windows load 95 = 2", oslayer.win_level(False, 95) == 2)
    check("windows load 50 = 1", oslayer.win_level(False, 50) == 1)

    # 3. Real readings on this computer.
    t, a, r = oslayer.total_ram(), oslayer.available_ram(), oslayer.process_rss()
    print(f"     total {t/1e9:.1f} GB, available {a/1e9:.1f} GB, this process {r/1e6:.0f} MB", flush=True)
    check("total ram > 1 GB", t > 1e9)
    check("available ram in range", 0 < a <= t)
    check("process rss > 5 MB", r > 5e6)
    check("pressure level is 1, 2 or 4", oslayer.pressure_level() in (1, 2, 4))

    # 4. The lock keeps two processes from writing at once: each adds 1 to a file 40 times.
    with tempfile.TemporaryDirectory() as d:
        counter, lock = Path(d) / "n.txt", Path(d) / "x.lock"
        counter.write_text("0")
        code = ("from cale import oslayer\nfrom pathlib import Path\nimport time\n"
                f"c=Path(r'{counter}')\n"
                "for _ in range(40):\n"
                f"    with oslayer.file_lock(r'{lock}'):\n"
                "        n=int(c.read_text()); time.sleep(0.002); c.write_text(str(n+1))\n")
        procs = [subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=str(ROOT)))
                 for _ in range(2)]
        for p in procs:
            p.wait(timeout=120)
        check("lock: 2 processes x 40 adds = 80", counter.read_text() == "80", counter.read_text())

        # 5. Private file, then the key store in a fresh Cale home.
        f = Path(d) / "secret.json"
        oslayer.private_file(f, json.dumps({"a": 1}))
        check("private file roundtrip", json.loads(f.read_text()) == {"a": 1})
        if not oslayer.WIN:
            check("private file mode 600", (f.stat().st_mode & 0o777) == 0o600, oct(f.stat().st_mode))
        code = ("from cale import keys, paths\n"
                "k = keys.create('Demo')\n"
                "assert keys.check(k) == 'Demo' and keys.check('nope') is None\n"
                "assert 'Demo' in keys.apps() and keys.revoke('Demo') and keys.check(k) is None\n"
                "print('KEYS_OK', paths.HOME)\n")
        r = run_py(code, {"CALE_HOME": str(Path(d) / "home")})
        check("keys: create, check, revoke", "KEYS_OK" in r.stdout, (r.stdout + r.stderr)[-300:])

    # 6. Detached process: start, see it, stop it.
    with tempfile.TemporaryDirectory() as d:
        pid = oslayer.spawn_detached([sys.executable, "-c", "import time; time.sleep(60)"], Path(d) / "log.txt")
        time.sleep(1.0)
        check("spawned process is alive", oslayer.pid_alive(pid))
        oslayer.kill_pid(pid)
        for _ in range(40):
            if not oslayer.pid_alive(pid):
                break
            time.sleep(0.25)
        check("killed process is gone", not oslayer.pid_alive(pid))

    # 7. Importing the service and pressure modules works here, and a quiet port answers "not running".
    r = run_py("from cale import service, pressure, cli, predict, goals\n"
               "print('IMPORT_OK', pressure.name(), service.status(timeout=1))",
               {"CALE_PORT": "47993", "CALE_HOME": tempfile.gettempdir() + "/cale-oslayer-test"})
    check("service, pressure, cli, predict, goals import", "IMPORT_OK" in r.stdout and "None" in r.stdout,
          (r.stdout + r.stderr)[-300:])

    # 8. Linux Allow dialog: a dialog tool that never closes counts as "no". It must not crash Cale (it did once, on WSLg).
    if not oslayer.WIN:
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "zenity"
            fake.write_text("#!/bin/sh\nsleep 30\n")
            fake.chmod(0o755)
            r = run_py("from cale import oslayer\noslayer.MAC = oslayer.WIN = False\n"
                       "print('ANSWER', oslayer.ask_allow('Test App', seconds=1))",
                       {"PATH": f"{d}{os.pathsep}{os.environ.get('PATH', '')}", "DISPLAY": ":99"}, timeout=40)
            check("a dialog tool that never closes counts as no", "ANSWER False" in r.stdout, (r.stdout + r.stderr)[-300:])

    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
