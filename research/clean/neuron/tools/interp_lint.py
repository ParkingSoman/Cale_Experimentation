"""Interp guard for research/clean/NEURON-LOG.md (Shonu's standing rule, 2026-11-21).

Applies to top sections "## N<k>" with k >= 11. Inside them, every "###" subsection whose title contains
"result" must:
  1. hold a neuron table: a markdown table whose header names the neuron, its verb (job), both poles
     (a cell with "pos" and a cell with "neg", or one cell "poles"), its feeders and its readers;
  2. (every line of N11+) no verdict word such as "messy" or "polysemantic" about a neuron unless the same line names
     the tool gap ("tool gap: ..."). A reading that fails is the microscope's failure (Shonu, 2026-11-21).
  3. if it mentions a fit (least squares, ridge, regression, optimal transport, OLS, lstsq), hold a line
     starting "Derived from:" that names the reading the fit carries out.

Usage: python3 interp_lint.py [path]      -> prints problems, exit 1 if any
       python3 interp_lint.py --hook      -> PostToolUse hook: reads the tool call on stdin, lints only when
                                             NEURON-LOG.md was touched, exit 2 (shown to Claude) on problems
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.normpath(os.path.join(HERE, "..", "..", "NEURON-LOG.md"))
BANNED = re.compile(r"\b(messy|polysemantic|scattered|incoherent|mixed bag|no clear (?:job|theme|pattern)|not clean|is noise)\b", re.I)
FIT = re.compile(r"least[- ]squares|\bridge\b|regression|optimal[- ]transport|\bOLS\b|lstsq", re.I)


def header_ok(cells):
    c = [x.lower() for x in cells]
    has = lambda *ws: any(all(w in x for w in ws) for x in c)
    poles = has("poles") or (any("pos" in x for x in c) and any("neg" in x for x in c))
    return has("neuron") and (has("verb") or has("job")) and poles and has("feed") and has("read")


def tables(lines):
    for i in range(len(lines) - 1):
        if lines[i].lstrip().startswith("|") and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            yield [x.strip() for x in lines[i].strip().strip("|").split("|")]


def lint(text):
    probs = []
    secs = re.split(r"(?m)^(## .*)$", text)
    for k in range(1, len(secs), 2):
        m = re.match(r"## N(\d+)", secs[k])
        if not m or int(m.group(1)) < 11:
            continue
        for ln in secs[k + 1].splitlines():
            if BANNED.search(ln) and "tool gap" not in ln.lower():
                probs.append(f"{secs[k][:12]}: a neuron verdict word ({BANNED.search(ln).group(0)!r}) with no 'tool gap' on the line: {ln[:90]}")
        subs = re.split(r"(?m)^(### .*)$", secs[k + 1])
        for j in range(1, len(subs), 2):
            title, body = subs[j], subs[j + 1]
            lines = body.splitlines()
            if "result" in title.lower():
                if not any(header_ok(h) for h in tables(lines)):
                    probs.append(f"{secs[k][:12]} / {title}: no neuron table (neuron, verb, both poles, feeders, readers)")
                if FIT.search(body) and not re.search(r"(?m)^\s*\**Derived from:", body):
                    probs.append(f"{secs[k][:12]} / {title}: mentions a fit but has no 'Derived from:' line")
    return probs


def main():
    if "--hook" in sys.argv:
        try:
            call = json.load(sys.stdin)
        except Exception:
            return 0
        path = (call.get("tool_input") or {}).get("file_path", "")
        if not path.endswith("NEURON-LOG.md"):
            return 0
        probs = lint(open(LOG).read())
        if probs:
            print("interp_lint (standing rule 2026-11-21):\n" + "\n".join(probs), file=sys.stderr)
            return 2
        return 0
    path = sys.argv[1] if len(sys.argv) > 1 else LOG
    probs = lint(open(path).read())
    print("\n".join(probs) if probs else "interp_lint: ok")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
