"""Show Cale's real Allow question on this computer and say what happened. Run it at the desktop (it needs a screen).

usage: python cale/tests/allow_dialog.py [SECONDS]        (default 60: after that the question counts as "no")
Click Allow, click Don't Allow, or wait. The script prints which one happened.
Over SSH with no screen it must say "no" quickly and never crash. That is also a pass.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import oslayer  # noqa: E402

seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
print(f"System: {oslayer.NAME}. Showing the Allow question for up to {seconds} s...", flush=True)
t = time.time()
yes = oslayer.ask_allow("Dialog Test", replacing=False, seconds=seconds)
print(f"Answer: {'ALLOW' if yes else 'NO (Don\'t Allow, or no answer, or no screen)'} after {time.time() - t:.1f} s")
