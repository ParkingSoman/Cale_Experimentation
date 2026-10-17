"""Run the Cale installer and report plain progress. Standard library only.

One place turns the installer's CALE_PROGRESS lines into a percent and a plain label.
The Hub (developer view) calls run(). The native setup windows run this file and read its lines:

  python cale/setup_runner.py
      PCT <0-100> <label>      progress, for a bar and one line of text
      DONE                     Cale is installed
      ERROR <one plain line>   the installer stopped
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent                 # the folder that holds cale/

STAGES = {"runtime": "Putting Cale on this computer", "chat": "Setting up your chat", "python": "Getting ready", "model": "Getting Cale's AI ready",
          "convert": "Preparing it", "tune": "Fitting Cale to this computer", "service": "Starting Cale", "done": "Done"}


def percent(update, current=0):
    """The bar position for one progress update. The model download is most of the bar (25 to 90)."""
    if update["stage"] == "model" and update.get("total"):
        return min(90, round(25 + 65 * update["done"] / update["total"]))
    return max(current, update["pct"]) if "pct" in update else current


def run(on_update):
    """Run cale/install.py. Call on_update(stage, pct, label) on every change. Returns None, or a plain error line."""
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    env["CALE_PROGRESS"] = "1"
    tail, pct = [], 0
    p = subprocess.Popen([sys.executable, str(SRC / "cale" / "install.py")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, cwd=str(SRC))
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("CALE_PROGRESS "):
            u = json.loads(line[len("CALE_PROGRESS "):])
            pct = percent(u, pct)
            on_update(u["stage"], pct, STAGES.get(u["stage"], ""), u)
        else:
            tail = (tail + [line])[-6:]
    if p.wait() != 0:
        return " ".join(tail[-3:]) or f"the installer stopped (code {p.returncode})"
    on_update("done", 100, STAGES["done"], {})
    return None


def main():
    last = [-1, ""]

    def show(stage, pct, label, extra):
        if (pct, label) != tuple(last):
            last[:] = [pct, label]
            print(f"PCT {pct} {label}", flush=True)

    try:
        error = run(show)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    print("DONE" if error is None else "ERROR " + error.replace("\n", " "), flush=True)
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())
