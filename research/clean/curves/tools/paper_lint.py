#!/usr/bin/env python3
"""Lint COARSE-THEORY.md so the paper stands without its measurements.

Usage:
  paper_lint.py [PAPER]          lint a file (exit 0 clean, 1 on problems)
  paper_lint.py --hook           PostToolUse hook: read the tool call on stdin,
                                 lint only when the paper was edited, exit 2 on problems
Rules (Parts A, B, D and the preamble; Part C and the Appendix are exempt):
  L1  no measured-claim words: CHECKED, HOLDS, MEASURED, and no **Statement items
  L2  no numbers from experiments: decimals like .0491 or 0.28 and the words dev / sealed
  L3  every Theorem / Lemma / Proposition / Corollary header has a hypothesis tag "(under: ...)"
  L4  every such item has a "Proof" (or is a CONJECTURE with a "Route:") and a "Lean:" line;
      a named Lean theorem must exist in lean/coarse/Coarse/*.lean
  L5  incompatible framings (SAE, autoencoder, transcoder, dictionary, superposition,
      polysemantic) only inside the literature table markers
  L6  once Part C exists, every declared hypothesis H-xxx has an entry there
  K1-K7  the decision ledger DECISIONS.md agrees with the paper (see tools/ledger.py); runs on the real paper only
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.dirname(HERE)
PAPER = os.path.join(CURVES, "COARSE-THEORY.md")
LEDGER = os.path.join(CURVES, "DECISIONS.md")
sys.path.insert(0, HERE)
import ledger  # noqa: E402
LEAN_DIR = os.path.join(CURVES, "lean", "coarse", "Coarse")

LIT_OPEN, LIT_CLOSE = "<!-- literature-table -->", "<!-- /literature-table -->"
ITEM = re.compile(r"^\*\*(Theorem|Lemma|Proposition|Corollary)\b[^*]*\*\*")
ANY_ITEM = re.compile(r"^\*\*(Theorem|Lemma|Proposition|Corollary|Definition|Fact|Hypothesis|Conjecture|Statement)\b")
HYP_DECL = re.compile(r"^\*\*Hypothesis (H-[A-Za-z0-9-]+)")
MEASURED = re.compile(r"\b(CHECKED|HOLDS|MEASURED)\b")
STATEMENT = re.compile(r"^\*\*Statement\b")
DOT_DECIMAL = re.compile(r"(?<![\w.])(0)?\.\d{2,}\b")  # .0491 and 0.28 alike
RUN_WORDS = re.compile(r"\b(dev|sealed)\b", re.I)
INCOMPATIBLE = re.compile(r"\b(SAEs?|sparse auto-?encoders?|auto-?encoders?|transcoders?|dictionar(y|ies)|"
                          r"superposition|polysemantic\w*)\b", re.I)
LEAN_NAME = re.compile(r"Lean:\s*`?([A-Za-z_][\w.']*)`?")


def part_of(heading, current):
    m = re.match(r"^#\s+(Part\s+([A-Z])|Appendix)", heading)
    if not m:
        return current
    return "Appendix" if m.group(1) == "Appendix" else m.group(2)


def lean_names():
    names = set()
    if os.path.isdir(LEAN_DIR):
        for fn in os.listdir(LEAN_DIR):
            if fn.endswith(".lean"):
                with open(os.path.join(LEAN_DIR, fn)) as f:
                    for line in f:
                        m = re.match(r"\s*(?:private\s+)?(theorem|lemma|def|structure)\s+([\w.']+)", line)
                        if m:
                            names.add(m.group(2))
    return names


def lint(text):
    lines = text.splitlines()
    problems = []
    part, in_code, in_lit = "pre", False, False
    tagged = []  # (line_no, part) for every line, to find item bodies later
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
        if s == LIT_OPEN:
            in_lit = True
        elif s == LIT_CLOSE:
            in_lit = False
        if line.startswith("# "):
            part = part_of(line, part)
        tagged.append(part)
        if part in ("C", "Appendix"):
            continue
        if MEASURED.search(line):
            problems.append(f"L1 line {i}: measured-claim word '{MEASURED.search(line).group(1)}' in Part {part}")
        if STATEMENT.match(line):
            problems.append(f"L1 line {i}: '**Statement' item in Part {part} (use Theorem/Conjecture/Hypothesis)")
        if not in_code:
            m = DOT_DECIMAL.search(line)
            if m:
                problems.append(f"L2 line {i}: experiment-style number '{m.group(0)}' in Part {part} (numbers go to Part C)")
            m = RUN_WORDS.search(line)
            if m:
                problems.append(f"L2 line {i}: run word '{m.group(0)}' in Part {part}")
        if not in_lit:
            m = INCOMPATIBLE.search(line)
            if m:
                problems.append(f"L5 line {i}: framing '{m.group(0)}' outside the literature table")

    names = lean_names()
    for i, line in enumerate(lines):
        if tagged[i] in ("C", "Appendix") or not ITEM.match(line):
            continue
        header = ITEM.match(line).group(0)
        if "(under:" not in header:
            problems.append(f"L3 line {i+1}: item has no hypothesis tag '(under: ...)': {header[:70]}")
        body = [line]
        for j in range(i + 1, len(lines)):
            if ANY_ITEM.match(lines[j]) or lines[j].startswith("#"):
                break
            body.append(lines[j])
        btxt = "\n".join(body)
        conjecture = "CONJECTURE" in header
        if conjecture:
            if "Route:" not in btxt:
                problems.append(f"L4 line {i+1}: CONJECTURE without a 'Route:' line")
        elif not re.search(r"(^|\n)\*{0,2}Proof\b", btxt):
            problems.append(f"L4 line {i+1}: item has no Proof (or mark it CONJECTURE with a Route:)")
        if "Lean:" not in btxt:
            problems.append(f"L4 line {i+1}: item has no 'Lean:' line (a theorem name, or 'Lean: none, <reason>')")
        else:
            m = LEAN_NAME.search(btxt)
            if m and m.group(1) not in ("none", "pending") and m.group(1).split(".")[-1] not in names:
                problems.append(f"L4 line {i+1}: Lean name '{m.group(1)}' not found in {LEAN_DIR}")

    if any(p == "C" for p in tagged):
        c_text = "\n".join(l for l, p in zip(lines, tagged) if p == "C")
        for i, line in enumerate(lines):
            m = HYP_DECL.match(line)
            if m and m.group(1) not in c_text:
                problems.append(f"L6 line {i+1}: hypothesis {m.group(1)} has no eps entry in Part C")
    return problems


def ledger_problems():
    if not os.path.exists(LEDGER):
        return ["K0: DECISIONS.md is missing (the decision ledger)"]
    return ledger.check(open(LEDGER).read(), open(PAPER).read())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--hook":
        try:
            data = json.load(sys.stdin)
        except Exception:
            return 0
        ti = data.get("tool_input", {}) or {}
        if data.get("tool_name") == "Bash":
            # a shell command that names the paper may have edited it
            cmd = ti.get("command") or ""
            if ("COARSE-THEORY" not in cmd and "DECISIONS.md" not in cmd) or not os.path.exists(PAPER):
                return 0
        else:
            path = ti.get("file_path") or (data.get("tool_response") or {}).get("filePath") or ""
            if os.path.abspath(path) not in (PAPER, LEDGER) or not os.path.exists(PAPER):
                return 0
        probs = lint(open(PAPER).read()) + ledger_problems()
        if probs:
            sys.stderr.write("COARSE-THEORY.md lint failed. Fix the paper before going on:\n"
                             + "\n".join(probs) + "\n")
            return 2
        return 0
    path = sys.argv[1] if len(sys.argv) > 1 else PAPER
    probs = lint(open(path).read())
    if os.path.abspath(path) == PAPER:
        probs += ledger_problems()
    print("\n".join(probs) if probs else "paper lint: clean")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
