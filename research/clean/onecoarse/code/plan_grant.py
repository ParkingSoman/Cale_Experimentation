#!/usr/bin/env python3
"""PostToolUse hook on ExitPlanMode: an approved plan grants its own scope (Shonu, 2026-11-24: "when i approve a plan,
i allow you to run the tool"; "we have to figure out this guard bullshit").

When Shonu approves a plan, Claude Code runs this hook with the approved plan text in tool_response.plan. If that text
holds a block

    <!-- scale-allow
    neurons: ...
    net: ...
    photos: ...
    where: ...
    -->

this hook appends it to onecoarse/grants/<session_id>.json. scale_gate.py then allows, in that session only, the
SCOPE.md block plus every grant of that session. The block is read from the text Shonu saw (tool_response.plan), never
from the plan file on disk. The grants folder, this file and scale_gate.py are on the CrakBox lock list: no session can
write a grant by hand. Every call is logged to grants/_hook_log.jsonl (keys only, no plan text) for diagnosis.

Usage: python3 plan_grant.py --hook        (stdin = hook JSON)
       python3 plan_grant.py --self-test
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GRANTS = os.path.join(os.path.dirname(HERE), "grants")
BLOCK = re.compile(r"<!--\s*scale-allow(.*?)-->", re.S)
SID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def parse_block(text):
    """All scale-allow blocks in the text, merged: {axis: sorted list of items}."""
    out = {}
    for m in BLOCK.finditer(text or ""):
        for ln in m.group(1).strip().splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                out.setdefault(k.strip(), set()).update(x.strip() for x in v.split(",") if x.strip())
    return {k: sorted(v) for k, v in out.items()}


def approved_plan(data):
    """The approved plan text, or None. Only tool_response counts (it holds the plan only after approval)."""
    if data.get("tool_name") != "ExitPlanMode" or data.get("hook_event_name", "PostToolUse") != "PostToolUse":
        return None, None
    resp = data.get("tool_response")
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except ValueError:
            return None, None
    if not isinstance(resp, dict) or not isinstance(resp.get("plan"), str):
        return None, None
    return resp["plan"], resp.get("filePath")


def handle(data, grants=GRANTS):
    """Write a grant if the call is an approved plan with a scale-allow block. Return the grant or None."""
    os.makedirs(grants, exist_ok=True)
    sid = str(data.get("session_id", ""))
    plan, path = approved_plan(data)
    allow = parse_block(plan) if plan else {}
    with open(os.path.join(grants, "_hook_log.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "session_id": sid,
                            "tool_name": data.get("tool_name"), "event": data.get("hook_event_name"),
                            "keys": sorted(data), "response_type": type(data.get("tool_response")).__name__,
                            "plan_found": plan is not None, "block_found": bool(allow)}) + "\n")
    if not allow or not SID.match(sid):
        return None
    gfile = os.path.join(grants, sid + ".json")
    rec = json.load(open(gfile)) if os.path.exists(gfile) else {"session_id": sid, "grants": []}
    g = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "plan_path": path, "allow": allow}
    rec["grants"].append(g)
    with open(gfile, "w") as f:
        json.dump(rec, f, indent=1)
    return g


def session_allow(sid, grants=GRANTS):
    """Union of every grant of this session: {axis: set}. Empty if none."""
    out = {}
    if not SID.match(str(sid or "")):
        return out
    gfile = os.path.join(grants, sid + ".json")
    if not os.path.exists(gfile):
        return out
    try:
        rec = json.load(open(gfile))
    except (OSError, ValueError):
        return out
    for g in rec.get("grants", []):
        for k, v in g.get("allow", {}).items():
            out.setdefault(k, set()).update(v)
    return out


def self_test():
    import tempfile
    bad = 0
    d = tempfile.mkdtemp()
    plan = "x\n```\n<!-- scale-allow\nneurons: s3:69, A:children\nnet: A\nphotos: held, test\nwhere: mac, cluster\n-->\n```\n"

    def check(name, ok):
        nonlocal bad
        bad += not ok
        print(("ok  " if ok else "BAD ") + name)
    g = handle({"tool_name": "ExitPlanMode", "hook_event_name": "PostToolUse", "session_id": "s1",
                "tool_response": {"plan": plan, "filePath": "/p.md"}}, d)
    check("approved plan with block -> grant", g is not None and g["allow"]["where"] == ["cluster", "mac"])
    check("session s1 sees it", "cluster" in session_allow("s1", d).get("where", set()))
    check("session s2 does not", session_allow("s2", d) == {})
    g = handle({"tool_name": "ExitPlanMode", "hook_event_name": "PostToolUse", "session_id": "s3",
                "tool_response": {"plan": "no block here", "filePath": "/p.md"}}, d)
    check("plan without block -> no grant", g is None and session_allow("s3", d) == {})
    g = handle({"tool_name": "ExitPlanMode", "hook_event_name": "PreToolUse", "session_id": "s4",
                "tool_input": {"plan": plan}}, d)
    check("PreToolUse (before approval) -> no grant", g is None and session_allow("s4", d) == {})
    g = handle({"tool_name": "ExitPlanMode", "hook_event_name": "PostToolUse", "session_id": "s5",
                "tool_input": {"plan": plan}, "tool_response": {}}, d)
    check("plan only in tool_input -> no grant", g is None)
    g = handle({"tool_name": "ExitPlanMode", "hook_event_name": "PostToolUse", "session_id": "../evil",
                "tool_response": {"plan": plan}}, d)
    check("bad session id -> no grant", g is None and not os.path.exists(os.path.join(d, "..", "evil.json")))
    g = handle({"tool_name": "Write", "hook_event_name": "PostToolUse", "session_id": "s6",
                "tool_response": {"plan": plan}}, d)
    check("other tool -> no grant", g is None)
    print("self-test", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


def main():
    if "--self-test" in sys.argv:
        return self_test()
    try:
        data = json.load(sys.stdin)
        handle(data)
    except Exception as e:  # a hook must never break the session; the log shows the failure
        try:
            os.makedirs(GRANTS, exist_ok=True)
            with open(os.path.join(GRANTS, "_hook_log.jsonl"), "a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": repr(e)}) + "\n")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
