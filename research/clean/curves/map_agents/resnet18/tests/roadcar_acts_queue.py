# serves: H-map
"""roadcar: run ev.py acts for parent-child pairs one after the other. Output goes to stdout."""
import subprocess
import sys

EV = "map_agents/resnet18/ev.py"
for pair in sys.argv[1:]:
    p, c = pair.split(">")
    r = subprocess.run([".venv/bin/python", EV, "acts", p, c], capture_output=True, text=True)
    print(pair, "|", " ".join(r.stdout.strip().splitlines()[-2:]), flush=True)
