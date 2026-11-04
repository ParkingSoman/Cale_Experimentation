---
id: dial
version: 1
priority: 30
trigger: on_share > 0.9 and cliff > 0.15
step_limit: 6
exits: claim, switch:clear_detector, switch:tool_looks_wrong, open
tests: ordering
replaces_poles: yes
written_for: s4b1:32, a broad dial on for 97% of photos, where a smooth gradient gives nothing
---
The node is on for almost every example, and its meaning changes slowly. Here the middle matters. Essay line 327. Crop sheets cannot score this well, so the judge uses a fixed set of drawn items.

1. `deep NODE`. Which property grows along the rings: stroke, width, angle, spacing, colour, tone or texture?
2. `edit NODE`. Which edits keep it, and which kill it? Gray kills colour. Blur kills fine detail. Rotation kills orientation.
3. Write the prediction: which property, at which value, fires it most. Draw two to four images under `runs/drawn/` that differ only in that property, and run `probe NODE IMAGE...`.
4. `inputs NODE` and `readers NODE`, to see what it passes along.

Your words must let a judge rank drawn lines, bars, dots, rings, checks, arcs, plain colours and edges by how much they fire it. If the drawn items do not fire it at all, switch to `clear_detector`, and say so.
