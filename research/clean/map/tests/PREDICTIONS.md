# Early band predictions (written 2026-10-21 BEFORE any synthetic test ran)

Normalised by the deep-view natural max/min: 1 n351 5.49/-4.52, 3 n623 3.01/-6.06, 4 n499 4.51/-7.27, 5 n135 3.67/-5.81.
Statistic: max of fc1 pre-activation over the central 60% of the map.

| Neuron | Prediction | Control |
|---|---|---|
| 1 n351 | A thin (1-2 px) dark vertical line on a light ground gives > 50% of max. A 45 deg line gives < 25%. Vertical beats horizontal by > 1.5x. | Same line blurred (sigma 3) < 10%. Crossed 45 deg lines (mesh) give a negative value. |
| 3 n623 | Over a hue x saturation x lightness grid of uniform patches, the peak is blue / blue-violet, at LOW-to-mid saturation, > 50% of max. Orange/brown patches are negative. | A neutral gray patch is near 0 (between -25% and +25% of max). |
| 4 n499 | Uniform patches: a warm (yellow/orange), low-saturation patch gives > 25% of max. Saturated blue gives < -50% of abs(min). | Neutral gray near 0 (gray edit killed it). |
| 5 n135 | One oblique orientation of a light/dark edge or grating gives > 50% of max. The mirror orientation gives < 25%. Dense vertical gratings give a negative value. | If no single angle wins, the card says "unclear, low confidence". |
