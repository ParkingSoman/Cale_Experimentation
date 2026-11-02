#!/usr/bin/env python3
"""Self-test for tools/ledger.py: one good case and one bad case per rule. Prints PASS/FAIL, exits with the failures."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger  # noqa: E402

PAPER = """# Part A: starting points
**Hypothesis H-gate.** Shared gates.
# Part B: theorems
**Theorem T9 (the fold) (under: H-gate; depth 1).** Text.
Proof. x. QED.
# Part D: where the math leads
**Conjecture D8 (under: H-gate; CONJECTURE).** Text.
Route: x.
"""
LEDGER = """<!-- trunk -->
**Trunk T\\*.** The best small network for every size.
<!-- /trunk -->
<!-- nodes -->
| ID | Kind | Axis | What | Carried by | By | Added | Status |
|---|---|---|---|---|---|---|---|
| R-gate | CONV | gates | ReLU -> shared gate | H-gate | Shonu chose it (2026-10-24) | 2026-10-24 | DECIDED |
| R-depth1 | CONV | depth | many -> one | T9 | me | 2026-10-25 | RETRO |
<!-- /nodes -->
<!-- aliases -->
| Phrase | Node |
|---|---|
| H-gate | R-gate |
| depth 1 | R-depth1 |
<!-- /aliases -->
<!-- paths -->
| Item | Path | Note |
|---|---|---|
| Theorem T9 | R-gate, R-depth1 | |
| Conjecture D8 | R-gate | |
<!-- /paths -->
"""
fails = 0
tmp = tempfile.mkdtemp()
sha = os.path.join(tmp, "trunk.sha256")
open(sha, "w").write(ledger.trunk_hash(LEDGER))


def case(name, led, pap, code):
    global fails
    probs = ledger.check(led, pap, sha_path=sha)
    hit = [p for p in probs if p.startswith(code)] if code else probs
    ok = (not probs) if code is None else bool(hit)
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" (got: {probs})"))
    fails += 0 if ok else 1


case("ledger: good ledger is clean", LEDGER, PAPER, None)
case("ledger: unlisted tag phrase fails (K1)", LEDGER, PAPER.replace("depth 1", "Gaussian input"), "K1")
case("ledger: new item with no path row fails (K2)",
     LEDGER, PAPER + "**Theorem T99 (under: H-gate).** x\nProof. x\n", "K2")
case("ledger: tag node missing from path fails (K3)", LEDGER.replace("| Theorem T9 | R-gate, R-depth1 |",
     "| Theorem T9 | R-gate |"), PAPER, "K3")
case("ledger: path with unknown node fails (K4)", LEDGER.replace("| Conjecture D8 | R-gate |",
     "| Conjecture D8 | R-gate, R-new |"), PAPER, "K4")
case("ledger: hypothesis without alias fails (K5)",
     LEDGER, PAPER.replace("# Part B", "**Hypothesis H-new.** x\n# Part B"), "K5")
case("ledger: DECIDED without Shonu fails (K6)", LEDGER.replace("Shonu chose it (2026-10-24)", "me"), PAPER, "K6")
case("ledger: new RETRO node fails (K6)", LEDGER.replace("| me | 2026-10-25 | RETRO |",
     "| me | 2026-10-26 | RETRO |"), PAPER, "K6")
case("ledger: new PENDING node passes", LEDGER.replace("| me | 2026-10-25 | RETRO |",
     "| me | 2026-10-26 | PENDING |"), PAPER, None)
case("ledger: changed trunk fails (K7)", LEDGER.replace("every size", "width 1"), PAPER, "K7")
print(f"ledger failures: {fails}")
sys.exit(fails)
