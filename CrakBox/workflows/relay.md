---
id: relay
version: 1
priority: 20
trigger: top_input_share > 0.25
step_limit: 4
exits: claim, switch:clear_detector, open
written_for: s4b1:226 and its inner channel, one mass passed along a chain with little change
---
One input dominates. This node may pass a job along. A relay is cheap to read: its parent's words mostly fit, so check what changed.

1. Read the top input's card or notes if they exist. Otherwise `deep TOPINPUT`.
2. `deep NODE`. What did this node add or drop against its parent? Look at the strangest ring members.
3. `acts TOPINPUT NODE`. How much does the child fall without it?
4. Write what changed. If nothing did, say that, and the words may stay close to the parent's. A different reader must write them.
