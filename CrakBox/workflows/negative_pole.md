---
id: negative_pole
version: 1
priority: 50
trigger: neg_ratio > 1.3
step_limit: 6
exits: claim, switch:gate, switch:clear_detector, open
written_for: s4b1i:460, soft rounded heads on one side, with spokes and radial things strongly on the other
---
The negative side is stronger than the positive. Its job may live there. Essay lines 233, 353 to 357.

1. `deep NODE`. Read the negative pole first. What is shared at full and half strength?
2. `inputs NODE --band neg`. Which inputs push it down?
3. `readers NODE`. What do readers do with this node's firing? Does the positive side matter to them at all?
4. Write the distinction as the line between the two sides. Predict one item for each side, and test with `probe` or `edit`.

The motif file lists a radial negative pole seen on many inner channels. Check for it, and do not assume it.
