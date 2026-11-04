---
id: position_code
version: 1
priority: 10
trigger: position_share > 0.4
step_limit: 5
exits: claim, switch:clear_detector, open
written_for: a frame-position code seen in the ConvNeXt map (research/clean/MAP-STATE.md); no ResNet node yet
---
Its strong firings sit at one place in the frame. Part of its job may be where, and not what.

1. `deep NODE`. Check the heat maps first. Is the place fixed across photos?
2. `edit NODE`. Flip and rotation move the place. Zoom moves it too. Tile shuffles keep local content and break the layout.
3. Say where, and what at that place. A judge sees crops at the peak, so describe what the crop shows, too.
