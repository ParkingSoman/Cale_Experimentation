#!/usr/bin/env bash
# UserPromptSubmit hook (math mode only): restate the paper rules on every prompt, against late drift.
CURVES="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$CURVES/.math-mode" ] || exit 0
cat <<'EOF'
[math mode: research/clean/curves/CLAUDE.md is the rule set]
0. THE BAR (Shonu, 2026-11-03, card 3-40): "given this particular goal the developer has, are we optimizing KL". A goal is weights on the inputs. At a squeeze: on sealed inputs from the goal, at most 1.3 times the best trained network of the same form; one computation for the whole zoom costs less than training one level; one fixed calculation. No proof of "least of all networks" is required at a squeeze, so do NOT chase floors. Dropped and kept-whole groups keep their exact promise. Label every result OPT-GOAL / OPT-KL / EQ / OPT-D2 / ONE-BUILD.
1. The paper (COARSE-THEORY.md) must stand with every measurement deleted. Numbers only in Part C. Every theorem says "(under: ...)".
2. Prove first (hand + Lean + sympy), check on the exact small instance, then a real network. A failed proof or check goes back to the paper. Never a chain of probes.
3. No SAE / dictionary / superposition framing. Literature enters through the compatibility table.
4. BEFORE ANY THEOREM: go through "What the trunk gives" in CLAUDE.md. Shonu: "you're doing these theorems and you're not considering all the things that the trunk gives us".
5. REAL NETWORK: is each number the loss of a network the trunk allows, at that size? Is my object the network's own, or groups of nouns? Maps come from agents reading N, never from one algorithm.
6. EXPLAIN (chat, LADDER, cards): real ML objects, a code-block diagram per step, one "Why it matters" line, each new word once in a table, one analogy at most. A paper label never stands alone.
7. ROUTING: technical = decide it myself by delegation and tell Shonu in one line. Only directional forks or a theorem conflict go on a card. Never ask what the trunk settles.
EOF
python3 "$CURVES/tools/ledger.py" --open 2>/dev/null || echo "(ledger unreadable: research/clean/curves/DECISIONS.md)"
