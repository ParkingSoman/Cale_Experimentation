"""Run torch commands for many agents: up to 3 at once, with a memory guard (FANOUT, Shonu 2026-11-26: "3 slots, swap guard").

Usage: curves/.venv/bin/python -I distill/fanout/torchq.py --who P4 [--long] [--timeout 600] -- <command ...>

Short jobs share 3 slots (lock files .torch.slot0-2). --long jobs (over about 2 min, e.g. input generators) use their
own single lane (.torch.long), so short tests never queue behind them. A long job also counts toward the memory guard.
Before a job starts it waits until free swap >= 1,000 MB AND system memory free >= 30% (memory_pressure). The guard is
checked while holding a slot, so at most one job starts per check. Locks are fcntl locks: released when the process dies.
Each finished job adds one line to torchq.log: time, who, lane/slot, wait s, run s, exit code, command.
"""
import argparse
import fcntl
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "torchq.log")
GATE = os.path.join(HERE, ".torch.gate")
N_SLOTS = 2                                    # 2 short + 1 long = 3 jobs (15:51: back up from 1; memory guard gates each start)
MIN_SWAP_MB = 1000
MIN_FREE_PCT = 35
MAX_SWAP_USED_MB = 8000                         # macOS grows the swap file, so 'free' stays small; 'used' is the real signal


def free_swap_mb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"free = ([\d.]+)M", out)
    return float(m.group(1)) if m else 1e9


def used_swap_mb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def free_mem_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else 100


def take_slot(long):
    names = ["long"] if long else [f"slot{i}" for i in range(N_SLOTS)]
    while True:
        for nm in names:
            fh = open(os.path.join(HERE, f".torch.{nm}"), "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return nm, fh
            except BlockingIOError:
                fh.close()
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", required=True)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    t0 = time.time()
    slot, fh = take_slot(a.long)
    with open(GATE, "w") as g:                      # one job passes the memory guard at a time
        fcntl.flock(g, fcntl.LOCK_EX)
        while free_mem_pct() < MIN_FREE_PCT or used_swap_mb() > MAX_SWAP_USED_MB:   # free swap dropped: macOS resizes the file
            if time.time() - t0 > 3600:
                print("torchq: memory guard held for 30 min, giving up", file=sys.stderr)
                sys.exit(3)
            time.sleep(5)
        time.sleep(2)                               # let the previous job allocate before the next check
    t1 = time.time()
    try:
        code = subprocess.run(cmd, timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        code = -9
        print(f"torchq: timeout after {a.timeout} s", file=sys.stderr)
    t2 = time.time()
    fh.close()
    with open(LOG, "a") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')}\t{a.who}\t{slot}\twait {t1 - t0:.0f}s\trun {t2 - t1:.0f}s\texit {code}\t{' '.join(cmd)[:300]}\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
