---
id: gate
version: 1
priority: 60
trigger: reader_mix > 0.45 and reader_mix < 0.55 and neg_ratio > 0.9
step_limit: 6
exits: claim, switch:negative_pole, switch:clear_detector, open
written_for: s3:164, a wheel on a painted body comes through, while faces and daisies are turned away
---
Readers split on sign, and both poles are strong. The node may let one thing through and keep a related thing out (essay line 225). The negative side here is often not an opposite: it keeps out what was grouped with it earlier.

1. `deep NODE`. What is on each side? What did earlier layers group them by?
2. `readers NODE`. Which readers read it with a plus, and which with a minus? Read those readers' words if they have cards.
3. Write the gate: what passes, and what related thing is kept out. Predict a reader that should fall when this node fills, with `acts NODE READER` if it is a direct input.
