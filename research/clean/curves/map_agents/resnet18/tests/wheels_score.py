# serves: H-map
"""wheels: percentile of each channel's largest firing on the transport classes (ev.py score). Read-only."""
import subprocess, sys
OUT = "/private/tmp/claude-501/-Users-shonusengupta-Cale-Experimentation/161e5945-0233-44f5-9176-bd85aff8b740/scratchpad/wheels_score.txt"
HI = {"trains": "466,547,565,820", "bikes": "444,671,880", "boats": "472,554,814,914", "air": "404,895"}
ALL = ",".join(HI.values())
with open(OUT, "w") as f:
    for ch in sys.argv[1:]:
        r = subprocess.run([".venv/bin/python", "map_agents/resnet18/ev.py", "score", ch, "--high", ALL, "--near", "7"],
                           capture_output=True, text=True)
        f.write("### " + ch + "\n" + r.stdout[-900:] + r.stderr[-200:] + "\n")
        f.flush()
