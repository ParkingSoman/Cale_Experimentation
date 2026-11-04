---
id: tool_looks_wrong
version: 1
priority: 999
trigger: largest < 0
step_limit: 6
exits: claim, switch:clear_detector, open
written_for: s4b0:42, where a probe gave one constant .21 for all drawn images, so a border artefact or a global feature may be at work
---
Code never picks this one. A reader switches here when a view looks wrong: a faint picture, one constant value, a coordinate that swamps the rest. Essay lines 111 to 113, 421, 685: blame the microscope first.

1. Say what looks wrong, and what you would expect instead.
2. Compare views: `fv NODE --natural` against `deep NODE`. Trust real photos when they differ. `fv` is slow, so use it once.
3. Try the suspect directly: the border (probe a blank image), a constant fill, the frame's edges.
4. If a new view would settle it, end the node open with suspect `tool`, and describe that view. The scientist reads this record.
