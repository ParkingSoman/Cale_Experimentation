# serves: H-map
"""wheels: run ev.py read-only commands for several channels, one text file per channel. No grouping logic."""
import subprocess, sys
PY = ".venv/bin/python"
EV = "map_agents/resnet18/ev.py"
OUT = "/private/tmp/claude-501/-Users-shonusengupta-Cale-Experimentation/161e5945-0233-44f5-9176-bd85aff8b740/scratchpad/"
for ch in sys.argv[1:]:
    txt = ""
    for cmd in (["sheet", ch], ["sheet", ch, "--band", "neg"], ["parents", ch], ["parents", ch, "--band", "neg"],
                ["readers", ch], ["siblings", ch], ["job", ch]):
        r = subprocess.run([PY, EV] + cmd, capture_output=True, text=True)
        txt += "### " + " ".join(cmd) + "\n" + r.stdout[-2500:] + r.stderr[-300:] + "\n"
    open(OUT + ch.replace(":", "_") + ".txt", "w").write(txt)
