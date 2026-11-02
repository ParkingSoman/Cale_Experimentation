#!/usr/bin/env bash
# Stop hook (math mode only): the paper may not end a turn broken.
# Lints COARSE-THEORY.md and looks for `sorry` in Model.lean. On a problem it blocks the stop once
# with the reason. If the stop hook already fired this turn (stop_hook_active), it lets the stop
# through and shows the problem to the user instead, so it can never loop.
set -uo pipefail
CURVES="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$CURVES/.math-mode" ] || exit 0
payload="$(cat)"
active="$(printf '%s' "$payload" | jq -r '.stop_hook_active // false')"
probs=""
if [ -f "$CURVES/COARSE-THEORY.md" ]; then
  out="$(python3 "$CURVES/tools/paper_lint.py" "$CURVES/COARSE-THEORY.md")" || probs="$out"
fi
out="$(python3 "$CURVES/tools/ladder_lint.py")" || probs="$probs
$out"
model="$CURVES/lean/coarse/Coarse/Model.lean"
if [ -f "$model" ] && grep -nE '\bsorry\b' "$model" >/dev/null; then
  probs="$probs
Model.lean still has sorry at: $(grep -nE '\bsorry\b' "$model" | cut -d: -f1 | tr '\n' ' ')"
fi
[ -z "$probs" ] && exit 0
if [ "$active" = "true" ]; then
  jq -n --arg p "$probs" '{systemMessage: ("Math paper still has problems:\n" + $p)}'
  exit 0
fi
jq -n --arg p "$probs" '{decision: "block", reason: ("Before ending the turn, fix the math paper (or say plainly why it stays open):\n" + $p)}'
