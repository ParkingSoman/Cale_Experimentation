"""An attack that could not have failed counts for nothing, either way. This finds each current attack whose images all stayed under the power mark, and sends the same card back to a critic. Usage: void_weak_attacks.py RUN [--apply]. Without --apply it only prints."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mapper.state import ATTACK_POWER, Store  # noqa: E402

run = sys.argv[1]
apply = "--apply" in sys.argv
s = Store(os.path.join(ROOT, "runs", run, "state.db"))
kept, voided = [], []
for (node, state) in s.db.execute("SELECT id,state FROM nodes WHERE state IN ('attacked','reading') ORDER BY id").fetchall():
    moves = s.history(node)
    last = next(((k, d) for k, d in reversed(moves) if k in ("claimed->attacked", "claimed->reading", "reading->claimed", "attacked->claimed")), None)
    if last is None or last[0] not in ("claimed->attacked", "claimed->reading"):
        continue  # this node has no current attack
    steps = last[1]["critic"]["steps"]
    (kept if s.attack_has_power(steps) else voided).append((node, state))
    if apply and not s.attack_has_power(steps):
        s.void_attack(node, f"no image reached {ATTACK_POWER} of the largest firing")
print(json.dumps(dict(mark=ATTACK_POWER, kept=kept, voided=voided, applied=apply), indent=1))
s.close()
