#!/usr/bin/env python3
"""PreToolUse gate on Bash: no experiment run without a written paper item.

Active only while research/clean/curves/.math-mode exists.
A Python script under research/clean/curves (outside tools/ and .venv/) may run only if one of its
first 20 lines reads "# serves: <ID>" and <ID> appears in COARSE-THEORY.md.
A script that touches InceptionV1 (mentions lucent / inception) must serve a hypothesis
(ID starting "H-"): the only InceptionV1 numbers the paper takes are Part C eps values.
Inline python (-c or heredoc) that imports torch or lucent from the curves folder is blocked:
put it in a script with a "# serves:" line.
Exit 2 with a reason on stderr blocks the call.
"""
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.dirname(HERE)
PAPER = os.path.join(CURVES, "COARSE-THEORY.md")
MARKER = os.path.join(CURVES, ".math-mode")
SERVES = re.compile(r"^#\s*serves:\s*([A-Za-z0-9_.'-]+)")


def block(msg):
    sys.stderr.write("run gate (math mode): " + msg + "\n")
    sys.exit(2)


def check_script(path):
    rel = os.path.relpath(path, CURVES)
    if rel.startswith("..") or rel.startswith("tools" + os.sep) or rel.startswith(".venv"):
        return
    try:
        full = open(path).read()
    except OSError:
        block(f"{rel} does not exist when the gate checks it. Write the script first (with its "
              f"'# serves:' line), then run it in a separate command.")
    head = full.splitlines()[:20]
    ids = [SERVES.match(l).group(1) for l in head if SERVES.match(l)]
    if not ids:
        block(f"{rel} has no '# serves: <ID>' line in its first 20 lines. Write the paper item "
              f"(theorem check or Part C eps) first, then name it in the script.")
    if not os.path.exists(PAPER):
        block("COARSE-THEORY.md does not exist yet. Write the paper item before any run.")
    paper = open(PAPER).read()
    for i in ids:
        if i not in paper:
            block(f"{rel} serves '{i}', which is not in COARSE-THEORY.md.")
    if re.search(r"lucent|inception", full, re.I) and not any(i.startswith("H-") for i in ids):
        block(f"{rel} touches InceptionV1 but serves {ids}. InceptionV1 runs may only serve a "
              f"hypothesis eps (an ID starting 'H-') in Part C.")


def main():
    if not os.path.exists(MARKER):
        return 0
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (data.get("tool_input") or {}).get("command", "")
    cwd = data.get("cwd") or os.getcwd()
    if "python" not in cmd:
        return 0
    in_curves = os.path.abspath(cwd).startswith(CURVES) or "research/clean/curves" in cmd
    if in_curves and re.search(r"python[\d.]*\s+(-I\s+)?(-c|-\s*<<|<<)", cmd) \
            and re.search(r"\b(import|from)\s+(torch|lucent)", cmd):
        block("inline Python that imports torch or lucent. Put it in a script with a '# serves:' line.")
    try:
        toks = shlex.split(cmd, posix=True)
    except ValueError:
        toks = cmd.split()
    here = cwd
    for k, t in enumerate(toks):
        if t == "cd" and k + 1 < len(toks):  # follow "cd dir && python x.py"
            d = os.path.expanduser(toks[k + 1])
            here = d if os.path.isabs(d) else os.path.join(here, d)
        elif re.search(r"(^|/)python[\d.]*$", t):
            # the script is the first argument after the interpreter's flags; "-c" and "-" are inline code
            j = k + 1
            while j < len(toks) and toks[j].startswith("-") and toks[j] not in ("-c", "-"):
                j += 1
            if j >= len(toks) or toks[j] in ("-c", "-") or not toks[j].endswith(".py"):
                continue
            t = toks[j]
            p = t if os.path.isabs(t) else os.path.join(here, t)
            p = os.path.abspath(p)
            if p.startswith(CURVES + os.sep):
                check_script(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
