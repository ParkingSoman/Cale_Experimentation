#!/usr/bin/env python3
"""Guard for how results and decisions reach Shonu (added 2026-11-01, his request: "reconfigure yourself").

Each LADDER.md section numbered FIRST_SECTION or later must have:
  - a code-block diagram of what happens in the network,
  - a "Why it matters" line,
  - a "Trunk check" row or line (which trunk words the work used, or why none applied).
Each DECISIONS.md card numbered 3-FIRST_CARD or later must have a "Directional because" line, unless its heading says
it is ANSWERED by delegation.

Usage: ladder_lint.py [LADDER.md] [DECISIONS.md]. Exit 0 when clean, 1 with one line per problem.
"""
import os
import re
import sys

FIRST_SECTION = 45
FIRST_CARD = 34
HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.dirname(HERE)


def sections(text, pattern):
    """Split text at headings that match pattern. Yields (number, heading, body). Fenced code is not scanned."""
    hits = []
    fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            fence = not fence
        m = None if fence else re.match(pattern, line)
        if m:
            hits.append((int(m.group(1)), line.strip(), pos))
        pos += len(line)
    for i, (num, head, start) in enumerate(hits):
        end = hits[i + 1][2] if i + 1 < len(hits) else len(text)
        yield num, head, text[start:end]


def check_ladder(text):
    probs = []
    for num, _head, body in sections(text, r"^## (\d+)\. "):
        if num < FIRST_SECTION:
            continue
        if "```" not in body:
            probs.append(f"LADDER {num}: no code-block diagram of what happens in the network")
        if not re.search(r"why it matters", body, re.I):
            probs.append(f"LADDER {num}: no 'Why it matters' line")
        if not re.search(r"trunk check", body, re.I):
            probs.append(f"LADDER {num}: no 'Trunk check' row (which trunk words did this use, or why none applied)")
    return probs


def check_cards(text):
    probs = []
    for num, head, body in sections(text, r"^### 3-(\d+)\. "):
        if num < FIRST_CARD:
            continue
        if re.search(r"ANSWERED .*by delegation", head):
            continue
        if not re.search(r"directional because", body, re.I):
            probs.append(f"card 3-{num}: no 'Directional because' line. If it is technical, decide it by delegation instead")
    return probs


def main(argv):
    ladder = argv[1] if len(argv) > 1 else os.path.join(CURVES, "LADDER.md")
    cards = argv[2] if len(argv) > 2 else os.path.join(CURVES, "DECISIONS.md")
    probs = []
    if os.path.exists(ladder):
        probs += check_ladder(open(ladder, encoding="utf-8").read())
    if os.path.exists(cards):
        probs += check_cards(open(cards, encoding="utf-8").read())
    for p in probs:
        print(p)
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
