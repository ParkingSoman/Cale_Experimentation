"""Run ONE torch command at a time on the Mac, for many agents at once (FANOUT fast loop, 2026-11-26).

Usage: curves/.venv/bin/python -I distill/fanout/torchq.py --who P4 -- <command ...>
It waits for the lock file fanout/.torch.lock (fcntl, released even if the process dies), waits while free swap
is under 400 MB, runs the command with a timeout (default 600 s), prints its output, and logs one line to
fanout/torchq.log (who, wait seconds, run seconds, exit code, command).
"""
import argparse
import fcntl
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, ".torch.lock")
LOG = os.path.join(HERE, "torchq.log")


def free_swap_mb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"free = ([\d.]+)M", out)
    return float(m.group(1)) if m else 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    t0 = time.time()
    with open(LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        while free_swap_mb() < 400:
            if time.time() - t0 > 900:
                print("torchq: free swap under 400 MB for 15 min, giving up", file=sys.stderr)
                sys.exit(3)
            time.sleep(10)
        t1 = time.time()
        try:
            r = subprocess.run(cmd, timeout=a.timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -9
            print(f"torchq: timeout after {a.timeout} s", file=sys.stderr)
        t2 = time.time()
    with open(LOG, "a") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')}\t{a.who}\twait {t1 - t0:.0f}s\trun {t2 - t1:.0f}s\texit {code}\t{' '.join(cmd)[:300]}\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
