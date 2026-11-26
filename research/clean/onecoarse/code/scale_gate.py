#!/usr/bin/env python3
"""PreToolUse guard: no change of scale in the one-coarse project without Shonu's word (2026-11-23).

1. Cluster submit or push tools: answer "ask", so Shonu sees a prompt for every cluster job.
2. Bash that runs a Python script in research/clean/onecoarse/code/: the script needs a line
   "# scale: neurons=...; net=...; photos=...; where=..." in its first 20 lines, and every value must be listed in
   the scale-allow block of research/clean/onecoarse/SCOPE.md. Otherwise the call is blocked.
3. Inline Python (-c or heredoc) that names onecoarse and imports torch is blocked: write a script with a scale line.
Guards in code/ (this file, label_gate.py, verb_lint.py) are exempt.
Self-test: python3 scale_gate.py --self-test
"""
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCOPE = os.path.join(ROOT, "SCOPE.md")
EXEMPT = {"scale_gate.py", "label_gate.py", "verb_lint.py", "plan_grant.py"}
CLUSTER = ("mcp__uva-gpu__submit_job", "mcp__uva-tools__gpu_run_best", "mcp__uva-tools__gpu_push")
LINE = re.compile(r"^#\s*scale:\s*(.+)$")
REQUEST = ("Post a SCALE-UP REQUEST block (From / To / Why / Cost, see onecoarse/SCOPE.md) and wait for Shonu.")


def allowed(session_id=None):
    """SCOPE.md's block, plus every grant that an approved plan gave THIS session (plan_grant.py, 2026-11-24)."""
    txt = open(SCOPE).read()
    m = re.search(r"<!--\s*scale-allow(.*?)-->", txt, re.S)
    if not m:
        return None
    out = {}
    for ln in m.group(1).strip().splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            out[k.strip()] = {x.strip() for x in v.split(",") if x.strip()}
    sys.path.insert(0, HERE)
    try:
        from plan_grant import session_allow
        for k, v in session_allow(session_id).items():
            out.setdefault(k, set()).update(v)
    except Exception:
        pass
    return out


def parse_line(s):
    out = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = {x.strip() for x in v.split(",") if x.strip()}
    return out


def check_decl(name, line, session_id=None):
    """Return None if one scale line is inside the allowed scope, else the reason."""
    ok = allowed(session_id)
    if ok is None:
        return "SCOPE.md has no scale-allow block."
    decl = parse_line(line)
    for key in ("neurons", "net", "photos", "where"):
        if key not in decl:
            return f"{name}: the scale line names no '{key}='. Name all four axes."
        extra = decl[key] - ok.get(key, set())
        if extra:
            return (f"{name} declares {key}={sorted(extra)}, outside SCOPE.md and this session's approved-plan "
                    f"grants. SCALE-UP. {REQUEST} Put a scale-allow block in the plan; approval grants it.")
    return None


def check_script(path, session_id=None):
    """Return None if the script may run, else the reason."""
    name = os.path.basename(path)
    if name in EXEMPT:
        return None
    try:
        head = open(path).read().splitlines()[:20]
    except OSError:
        return f"{name} does not exist when the guard checks it. Write it first (with its '# scale:' line)."
    lines = [LINE.match(l).group(1) for l in head if LINE.match(l)]
    if not lines:
        return f"{name} has no '# scale:' line in its first 20 lines. {REQUEST}"
    return check_decl(name, lines[0], session_id)


OC_JOB = re.compile(r"onecoarse|\boc_\w+\.py")
OUTBOX = os.path.expanduser("~/uva-gpu/outbox")


def check_cluster(tool, inp, session_id=None):
    """A onecoarse cluster job must carry a '# scale:' line (as a comment line in its command, like '# rung:'), inside
    the allowed scope. gpu_push of an oc_*.py file is checked by the file's own scale line. Other projects pass."""
    if tool == "mcp__uva-tools__gpu_push":
        name = str(inp.get("name", ""))
        p = os.path.join(OUTBOX, name)
        if re.search(r"\boc_\w+\.py$", name) and os.path.isfile(p):
            return check_script(p, session_id)
        return None
    cmd = str(inp.get("command", ""))
    if not OC_JOB.search(cmd):
        return None
    lines = [m.group(1) for m in re.finditer(r"(?m)^\s*#\s*scale:\s*(.+)$", cmd)]
    if not lines:
        return ("onecoarse cluster job without a '# scale:' line. Add the script's scale line as a comment line "
                "in the command (next to '# rung:').")
    for ln in lines:
        r = check_decl(f"cluster job ({tool})", ln, session_id)
        if r:
            return r
    return None


def scripts_in(cmd, cwd):
    try:
        toks = shlex.split(cmd, posix=True)
    except ValueError:
        toks = cmd.split()
    here, found = cwd, []
    for k, t in enumerate(toks):
        if t == "cd" and k + 1 < len(toks):
            d = os.path.expanduser(toks[k + 1])
            here = d if os.path.isabs(d) else os.path.join(here, d)
        elif re.search(r"(^|/)python[\d.]*$", t):
            j = k + 1
            while j < len(toks) and toks[j].startswith("-") and toks[j] not in ("-c", "-"):
                j += 1
            if j < len(toks) and toks[j].endswith(".py"):
                p = toks[j] if os.path.isabs(toks[j]) else os.path.join(here, toks[j])
                found.append(os.path.abspath(p))
    return found


def decide(data):
    """Return (kind, reason): kind in {None, 'ask', 'block'}."""
    tool = data.get("tool_name", "")
    sid = data.get("session_id")
    if tool in CLUSTER:
        # Shonu, 2026-11-24: "when i approve a plan, i allow you to run the tool". No prompt per job. The job's scale
        # line is checked against SCOPE.md plus the approved plan's grant (plan_grant.py).
        r = check_cluster(tool, data.get("tool_input") or {}, sid)
        return ("block", "Scale guard: " + r) if r else (None, "")
    if tool != "Bash":
        return None, ""
    cmd = (data.get("tool_input") or {}).get("command", "")
    cwd = data.get("cwd") or os.getcwd()
    if "python" not in cmd:
        return None, ""
    if re.search(r"python[\d.]*\s+(-I\s+)?(-c|-\s*<<|<<)", cmd) and re.search(r"\b(import|from)\s+torch", cmd) \
            and ("onecoarse" in cmd or os.path.abspath(cwd).startswith(ROOT)):
        return "block", "Scale guard: inline Python with torch in onecoarse. Write a script with a '# scale:' line."
    for p in scripts_in(cmd, cwd):
        if p.startswith(HERE + os.sep):
            r = check_script(p, sid)
            if r:
                return "block", "Scale guard: " + r
    return None, ""


def self_test():
    import tempfile
    good = "# scale: neurons=s3:69, B:n101; net=B128aug_s1; photos=saved; where=mac\n"
    cases = [("t_none.py", "print(1)\n", True), ("t_good.py", good, False),
             ("t_big.py", "# scale: neurons=A:all512; net=A; photos=saved; where=mac\n", True),
             ("t_clu.py", "# scale: neurons=s3:69; net=A; photos=saved; where=cluster\n", True)]
    made, bad = [], 0
    for name, body, want_block in cases:
        p = os.path.join(HERE, name)
        open(p, "w").write(body)
        made.append(p)
        kind, why = decide({"tool_name": "Bash", "tool_input": {"command": f"python3 {p}"}, "cwd": ROOT})
        got = kind == "block"
        bad += got != want_block
        print(("ok  " if got == want_block else "BAD ") + name, kind, why[:90])
    kind, _ = decide({"tool_name": "Bash", "tool_input": {"command": "python3 -c 'import torch' # onecoarse"},
                      "cwd": ROOT})
    bad += kind != "block"
    print(("ok  " if kind == "block" else "BAD ") + "inline torch", kind)
    # cluster jobs (2026-11-24): checked against SCOPE.md plus this session's approved-plan grant
    import plan_grant
    sid = "selftest-scale-gate"
    gfile = os.path.join(plan_grant.GRANTS, sid + ".json")
    big = "# rung: Q=x; N=1\n# scale: neurons=s3:69; net=A; photos=held; where=cluster\npython oc_x.py\n"
    cases = [("cluster, other project", {"command": "python crak.py"}, None),
             ("cluster, onecoarse, no scale line", {"command": "python oc_x.py"}, "block"),
             ("cluster, onecoarse, out of scope", {"command": big}, "block")]
    for name, inp, want in cases:
        kind, why = decide({"tool_name": "mcp__uva-gpu__submit_job", "tool_input": inp, "session_id": sid})
        bad += kind != want
        print(("ok  " if kind == want else "BAD ") + name, kind, why[:80])
    plan = "<!-- scale-allow\nneurons: s3:69\nnet: A\nphotos: held\nwhere: cluster\n-->"
    plan_grant.handle({"tool_name": "ExitPlanMode", "hook_event_name": "PostToolUse", "session_id": sid,
                       "tool_response": {"plan": plan}})
    kind, why = decide({"tool_name": "mcp__uva-gpu__submit_job", "tool_input": {"command": big}, "session_id": sid})
    bad += kind is not None
    print(("ok  " if kind is None else "BAD ") + "cluster, granted by approved plan", kind, why[:80])
    kind, _ = decide({"tool_name": "mcp__uva-gpu__submit_job", "tool_input": {"command": big},
                      "session_id": "another-session"})
    bad += kind != "block"
    print(("ok  " if kind == "block" else "BAD ") + "cluster, another session's grant ignored", kind)
    os.remove(gfile)
    for p in made:
        os.remove(p)
    print("self-test", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


def main():
    if "--self-test" in sys.argv:
        return self_test()
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    kind, why = decide(data)
    if kind == "ask":
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                                 "permissionDecisionReason": why}}))
        return 0
    if kind == "block":
        sys.stderr.write(why + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
