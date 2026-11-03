# serves: H-map
"""wheels: s3 sheets, parents and acts tests (read-only ev.py calls). Output to the scratchpad."""
import subprocess
OUT = "/private/tmp/claude-501/-Users-shonusengupta-Cale-Experimentation/161e5945-0233-44f5-9176-bd85aff8b740/scratchpad/wheels_s3.txt"
CH = ["s3:33", "s3:158", "s3:98", "s3:71", "s3:46", "s3:217", "s3:7", "s3:191"]
ACTS = [("s3:33", "s4:56"), ("s3:33", "s4:168"), ("s3:158", "s4:168"), ("s3:98", "s4:168"), ("s3:71", "s4:56"),
        ("s3:217", "s4:456"), ("s3:191", "s4:456"), ("s3:33", "s4:474")]


def run(cmd):
    r = subprocess.run([".venv/bin/python", "map_agents/resnet18/ev.py"] + cmd, capture_output=True, text=True)
    return "### " + " ".join(cmd) + "\n" + r.stdout[-1800:] + r.stderr[-200:] + "\n"


with open(OUT, "w") as f:
    for ch in CH:
        for cmd in (["sheet", ch], ["sheet", ch, "--band", "neg"], ["parents", ch], ["readers", ch]):
            f.write(run(cmd))
            f.flush()
    for p, c in ACTS:
        f.write(run(["acts", p, c]))
        f.flush()
