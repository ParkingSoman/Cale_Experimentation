#!/usr/bin/env python3
"""Remove Cale: the service (Mac), the command link and the install folder. Run:  python cale/uninstall.py
Only folders and files Cale made are removed. A model that was hard-linked from elsewhere stays at its source."""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cale import oslayer  # noqa: E402


def main():
    home = oslayer.home_dir()
    if str(home) in ("", "/", str(Path.home())) or not (home / "runtime" / "cale" / "server.py").exists():
        print(f"Not removing '{home}': it is not a Cale install folder.", file=sys.stderr)
        return 1
    from cale import service
    service.stop()
    if oslayer.MAC:
        subprocess.run(["launchctl", "bootout", f"gui/{__import__('os').getuid()}/ai.cale.runtime"], capture_output=True)
        (Path.home() / "Library" / "LaunchAgents" / "ai.cale.runtime.plist").unlink(missing_ok=True)
    link = Path.home() / ".local" / "bin" / "cale"
    if link.is_symlink() and Path(link.resolve()) == (home / "bin" / "cale").resolve():
        link.unlink()
    for sub in ("runtime", "venv", "models", "logs", "bin", "lib"):
        shutil.rmtree(home / sub, ignore_errors=True)
    for f in ("keys.json", "keys.lock", "keys.tmp", "curve.json", "profile.json", "predict.json", "cale.pid", "uv.tar.gz", "uv.zip"):
        (home / f).unlink(missing_ok=True)
    try:
        home.rmdir()
    except OSError:
        pass
    print("Cale removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
