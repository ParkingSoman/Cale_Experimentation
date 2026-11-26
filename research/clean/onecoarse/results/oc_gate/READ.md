# Gate maps: A (= C, WC) against the trained builds L and R4C

2,000 Broden held-set-1 photos x 49 cells. Checks: A's block 0 on held_s3 = held_s4b0 (max rel err 4.6e-04); C and WC gate maps = A's at every cut: True.

## 1. How much the gates change, and where the new-on cells fall

new share = new-on cells / A's on cells; lost share = lost cells / A's on cells (per kept unit, median; pooled in brackets).
D-cells = cells where at least one deleted unit fired in A. Ratio = share of new-on cells on D-cells / share of all A-off cells on D-cells (pooled over units; 95% CI, photos resampled).

| Cut | Build | Kept | D-cells share of all cells | new share median [pooled] | lost share median [pooled] | new-on on D-cells | chance | ratio [CI] |
|---|---|---|---|---|---|---|---|---|
| 11 | L | 501 | 0.75 | 0.012 [0.013] | 0.018 [0.017] | 0.817 | 0.749 | 1.09 [1.09, 1.10] |
| 11 | R4C | 501 | 0.75 | 0.011 [0.011] | 0.017 [0.017] | 0.812 | 0.749 | 1.09 [1.08, 1.09] |
| 64 | L | 448 | 1.00 | 0.030 [0.030] | 0.068 [0.068] | 0.999 | 0.998 | 1.00 [1.00, 1.00] |
| 64 | R4C | 448 | 1.00 | 0.022 [0.023] | 0.076 [0.076] | 0.999 | 0.998 | 1.00 [1.00, 1.00] |
| 192 | L | 320 | 1.00 | 0.036 [0.041] | 0.115 [0.120] | 1.000 | 1.000 | 1.00 [1.00, 1.00] |
| 192 | R4C | 320 | 1.00 | 0.024 [0.026] | 0.141 [0.144] | 1.000 | 1.000 | 1.00 [1.00, 1.00] |
| 384 | L | 128 | 1.00 | 0.505 [0.502] | 0.016 [0.021] | 1.000 | 1.000 | 1.00 [1.00, 1.00] |
| 384 | R4C | 128 | 1.00 | 0.280 [0.265] | 0.050 [0.057] | 1.000 | 1.000 | 1.00 [1.00, 1.00] |

### 1b. Graded measures (ADDED AFTER RUN 1: at cuts 64+ some deleted unit fires on ~every cell, so the yes/no D-cell test above has no room to show anything)

D-strength ratio = mean of A's summed deleted firing on new-on cells / the same on j's A-off cells (pooled; 95% CI, photos resampled). Near miss = mean percentile of j's A pre-activation on its new-on cells among all its A-off cells (per unit, median; chance .50, 1.00 = the new cells are the ones j almost fired on in A).

Margin-matched (ADDED AFTER RUN 2; a small read change opens near-miss cells first by construction): D-strength on new-on cells / on near-miss cells that STAYED off (A pre-activation >= the 10th pct of the new-on cells'). Per unit; median and share of units > 1. Above 1 = among cells equally close to firing, training opened the ones where the deleted units fired more.

| Cut | Build | D-strength ratio [CI] | near miss: median [10th pct] | margin-matched ratio: median [share > 1] |
|---|---|---|---|---|
| 11 | L | 1.43 [1.42, 1.45] | 0.994 [0.991] | 1.38 [0.75] (501 units) |
| 11 | R4C | 1.39 [1.37, 1.40] | 0.995 [0.992] | 1.38 [0.77] (500 units) |
| 64 | L | 1.27 [1.26, 1.28] | 0.983 [0.972] | 1.20 [0.85] (448 units) |
| 64 | R4C | 1.28 [1.27, 1.29] | 0.984 [0.975] | 1.22 [0.88] (447 units) |
| 192 | L | 1.18 [1.17, 1.19] | 0.973 [0.958] | 1.12 [0.77] (318 units) |
| 192 | R4C | 1.21 [1.20, 1.22] | 0.976 [0.964] | 1.16 [0.82] (318 units) |
| 384 | L | 1.10 [1.10, 1.10] | 0.906 [0.860] | 1.07 [0.76] (128 units) |
| 384 | R4C | 1.11 [1.11, 1.12] | 0.932 [0.897] | 1.08 [0.73] (128 units) |

## 2. Which knob moved the gate: threshold (bn1 shift) or read (bn1 scale x conv1)

New-on cells, pooled over kept units, as a share of the full build's new-on cells.

| Cut | Build | threshold only (A's read) | read only (A's threshold) |
|---|---|---|---|
| 11 | L | 0.03 | 1.02 |
| 11 | R4C | 0.02 | 1.04 |
| 64 | L | 0.02 | 1.06 |
| 64 | R4C | 0.02 | 1.14 |
| 192 | L | 0.01 | 1.06 |
| 192 | R4C | 0.02 | 1.19 |
| 384 | L | 0.02 | 0.99 |
| 384 | R4C | 0.01 | 1.09 |

### 2b. Gain or tilt? (ADDED AFTER REVIEW; bar fixed before the run)

alpha = the trained read's projection on A's read / |A's read|^2 (one number per unit). Gain-only gate = alpha x A's read + trained shift. Overlap = new-on cells the gain-only gate also opens / the build's new-on cells (pooled). Bar: >= .5 = the knob is one number per neuron; <= .2 = a real tilt; between = both.

| Cut | Build | alpha median [10th, 90th] | gain-only overlap | gain-only new-on / build new-on | verdict |
|---|---|---|---|---|---|
| 11 | L | 1.000 [0.999, 1.002] | 0.01 | 0.02 | tilt |
| 11 | R4C | 1.000 [0.998, 1.002] | 0.00 | 0.01 | tilt |
| 64 | L | 1.003 [0.998, 1.010] | 0.01 | 0.02 | tilt |
| 64 | R4C | 1.000 [0.995, 1.005] | 0.00 | 0.01 | tilt |
| 192 | L | 0.998 [0.990, 1.007] | 0.00 | 0.01 | tilt |
| 192 | R4C | 0.989 [0.980, 0.995] | 0.00 | 0.00 | tilt |
| 384 | L | 0.995 [0.982, 1.007] | 0.02 | 0.02 | tilt |
| 384 | R4C | 0.963 [0.945, 0.975] | 0.01 | 0.02 | tilt |

### 2c. Null for 'opened where the deleted units fired' (ADDED AFTER REVIEW; bar fixed before the run)

Per unit j, on its new-on cells vs its near-miss cells that stayed off: ratio of the MEAN firing per deleted unit, and of the mean firing per kept unit (j left out), and of the input patch norm. Bar: deleted-specific only if median (deleted ratio / kept ratio) > 1.05 and > 1 for >= 60% of units; otherwise 'busy cells open'.

| Cut | Build | deleted ratio (median) | kept ratio (median) | input-norm ratio (median) | deleted / kept: median [share > 1] | verdict |
|---|---|---|---|---|---|---|
| 11 | L | 1.38 | 1.02 | 1.06 | 1.33 [0.72] | deleted-specific |
| 11 | R4C | 1.38 | 1.02 | 1.06 | 1.34 [0.73] | deleted-specific |
| 64 | L | 1.20 | 1.02 | 1.07 | 1.16 [0.86] | deleted-specific |
| 64 | R4C | 1.22 | 1.01 | 1.07 | 1.17 [0.89] | deleted-specific |
| 192 | L | 1.12 | 1.00 | 1.08 | 1.12 [0.91] | deleted-specific |
| 192 | R4C | 1.16 | 1.00 | 1.08 | 1.15 [0.93] | deleted-specific |
| 384 | L | 1.07 | 1.08 | 1.07 | 1.01 [0.60] | busy cells open |
| 384 | R4C | 1.08 | 1.07 | 1.06 | 1.03 [0.60] | busy cells open |

## 3. Gate partners: whose cells do a kept unit's new-on cells fall on?

count partner = the deleted unit on whose A on-cells most of j's new-on cells fall. lift = that share / the same share on j's A-off cells. lift partner = the deleted unit with the largest lift (>= 20 new cells). Units with >= 20 new-on cells only.
Agreement = the partner equals the 8.5 / jobtest partner (deleted unit whose firing j's trained firing matches best, L).

| Cut | Build | Units | count partner: median share | median lift | count partner = jobtest | lift partner = jobtest | chance of a match (1 / #D) |
|---|---|---|---|---|---|---|---|
| 11 | L | 501 | 0.34 | 1.77 | 0.30 | 0.36 | 0.091 |
| 11 | R4C | 500 | 0.33 | 1.70 | 0.32 | 0.40 | 0.091 |
| 64 | L | 448 | 0.44 | 1.89 | 0.17 | 0.25 | 0.016 |
| 64 | R4C | 447 | 0.44 | 1.83 | 0.18 | 0.27 | 0.016 |
| 192 | L | 318 | 0.87 | 0.98 | 0.01 | 0.20 | 0.005 |
| 192 | R4C | 318 | 0.86 | 0.97 | 0.00 | 0.22 | 0.005 |
| 384 | L | 128 | 0.87 | 0.97 | 0.00 | 0.16 | 0.003 |
| 384 | R4C | 128 | 0.87 | 0.97 | 0.00 | 0.21 | 0.003 |

Note: the jobtest partner comes from L's firing in both rows (no R4C jobtest exists at 11 and 64).

## 4. Named units (L), read from the data with vecread.gate_card

gate_card known-answer test: a label's own cells read back as that label first for 32 of 45 common cell-level labels (bar .8); random 5% of cells give top lifts [1.39, 1.29, 1.25] (should be near 1). Low power: cards on a few hundred cells (n5's new-on cells: 284) are weak.

### Cut 11: n5 -> n464

A on-cells 7994, new-on 284, lost 353. Count partner n464 (share 0.48, lift 10.68); lift partner n464 (lift 10.68). Rank of n464 among 11 deleted: by count 1, by lift 1.

- old on-cells of n5 (7994 cells, 1658 photos): swivel chair 3.05, body 2.98, hair 2.8, armchair 2.73, bird 2.69, bed 2.57 | photo-level: living_room-s 1.46, conference_room-s 1.37, bedroom-s 1.32, street-s 1.08
- NEW-ON cells of n5 (284 cells, 228 photos): grey-c 1.0, white-c 0.98, brown-c 0.93, black-c 0.87 | photo-level: 
- A on-cells of n464 (5206 cells, 1435 photos): muzzle 3.37, head 2.56, bird 2.28, motorbike 2.01, wheel 1.87, food 1.72 | photo-level: bathroom-s 0.99, conference_room-s 0.89, dining_room-s 0.86, living_room-s 0.75
- shared top-5 cell-level concepts, NEW-ON of n5 and on-cells of n464: none; with n5's old cells: none

### Cut 64: n339 -> n476

A on-cells 13842, new-on 1801, lost 708. Count partner n314 (share 0.42, lift 1.57); lift partner n476 (lift 3.18). Rank of n476 among 64 deleted: by count 4, by lift 1.

- old on-cells of n339 (13842 cells, 1913 photos): armchair 2.33, arm 2.03, chair 1.82, seat 1.59, bed 1.52, bicycle 1.49 | photo-level: skyscraper-s 1.43, dining_room-s 1.32, bedroom-s 1.27, conference_room-s 1.22
- NEW-ON cells of n339 (1801 cells, 1100 photos): car 1.38, floor 1.21, black-c 1.2, grass 1.2, building 1.09, brown-c 0.97 | photo-level: dining_room-s 2.13, skyscraper-s 1.97, conference_room-s 1.71, street-s 1.58
- A on-cells of n476 (13509 cells, 1886 photos): swivel chair 2.42, truck 2.22, armchair 1.97, car 1.76, chair 1.68, bush 1.47 | photo-level: street-s 1.76, conference_room-s 1.17, skyscraper-s 1.16, highway-s 1.13
- shared top-5 cell-level concepts, NEW-ON of n339 and on-cells of n476: ['car']; with n339's old cells: none

### Cut 192: n430 -> n114

A on-cells 17094, new-on 1277, lost 1359. Count partner n260 (share 0.76, lift 0.85); lift partner n352 (lift 4.27). Rank of n114 among 192 deleted: by count 2, by lift 32.

- old on-cells of n430 (17094 cells, 1891 photos): signboard 2.36, work surface 1.99, horse 1.92, headboard 1.84, sky 1.71, bird 1.68 | photo-level: bedroom-s 1.8, dotted 1.77, conference_room-s 1.45, living_room-s 1.22
- NEW-ON cells of n430 (1277 cells, 864 photos): ceiling 2.31, sky 2.11, painted 1.96, blue-c 1.51, wall 1.35, wood 1.27 | photo-level: bedroom-s 1.74, kitchen-s 1.26, street-s 0.6
- A on-cells of n114 (41373 cells, 1907 photos): sky 1.41, painted 1.39, building 1.33, wall 1.32, house 1.31, white-c 1.3 | photo-level: kitchen-s 1.7, dining_room-s 1.53, bathroom-s 1.52, bedroom-s 1.52
- shared top-5 cell-level concepts, NEW-ON of n430 and on-cells of n114: ['painted', 'sky', 'wall']; with n430's old cells: ['sky']

### Cut 384: n430 -> n501

A on-cells 17094, new-on 8029, lost 403. Count partner n260 (share 0.83, lift 0.92); lift partner n393 (lift 3.09). Rank of n501 among 384 deleted: by count 70, by lift 54.

- old on-cells of n430 (17094 cells, 1891 photos): signboard 2.36, work surface 1.99, horse 1.92, headboard 1.84, sky 1.71, bird 1.68 | photo-level: bedroom-s 1.8, dotted 1.77, conference_room-s 1.45, living_room-s 1.22
- NEW-ON cells of n430 (8029 cells, 1859 photos): horse 1.72, hair 1.63, ceiling 1.57, sky 1.54, arm 1.5, painted 1.48 | photo-level: highway-s 1.59, bedroom-s 1.21, blotchy 1.15, dining_room-s 1.1
- A on-cells of n501 (19668 cells, 1941 photos): drawer 2.52, pane 2.39, chest of drawers 2.27, painting 1.96, sky 1.81, windowpane 1.63 | photo-level: skyscraper-s 1.48, kitchen-s 1.37, bedroom-s 1.32, bathroom-s 1.17
- shared top-5 cell-level concepts, NEW-ON of n430 and on-cells of n501: ['sky']; with n430's old cells: ['horse', 'sky']

### Cut 384: n76 -> n374

A on-cells 5684, new-on 3020, lost 192. Count partner n260 (share 0.77, lift 0.86); lift partner n393 (lift 5.08). Rank of n374 among 384 deleted: by count 4, by lift 12.

- old on-cells of n76 (5684 cells, 1441 photos): muzzle 3.22, ceiling 3.11, sky 2.5, white-c 1.82, blue-c 1.81, painted 1.77 | photo-level: highway-s 2.46, skyscraper-s 1.29, blotchy 1.23, street-s 0.81
- NEW-ON cells of n76 (3020 cells, 1387 photos): ceiling 3.83, sky 2.47, painted 2.05, blue-c 1.65, pink-c 1.58, white-c 1.49 | photo-level: highway-s 2.12, bathroom-s 1.29, kitchen-s 1.22, bedroom-s 1.19
- A on-cells of n374 (19524 cells, 1949 photos): ceiling 2.04, sky 1.84, painted 1.7, granite 1.59, tile 1.34, wood 1.33 | photo-level: blotchy 2.44, highway-s 1.42, street-s 1.02, skyscraper-s 0.93
- shared top-5 cell-level concepts, NEW-ON of n76 and on-cells of n374: ['ceiling', 'painted', 'sky']; with n76's old cells: ['blue-c', 'ceiling', 'sky']

## 5. Predictions (written in the plan before the run)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | new-on cells fall on D-cells above chance at every cut (L) | ratios 1.09, 1.00, 1.00, 1.00 | FAIL |
| P2 | new share grows with the cut (L, median 384 > 11) | 0.012, 0.030, 0.036, 0.505 | pass |
| P3 | lost share median <= .1 at every cut, both builds | L 0.018, 0.068, 0.115, 0.016; R4C 0.017, 0.076, 0.141, 0.050 | FAIL |
| P4 | n339's count partner is n476 and the cards share a top-5 concept | rank 4, shared ['car'] | FAIL |
| P5 | the read moves the gate more than the threshold (every cut and build) | read-only / threshold-only: 11L 1.02/0.03, 11R 1.04/0.02, 64L 1.06/0.02, 64R 1.14/0.02, 192L 1.06/0.01, 192R 1.19/0.02, 384L 0.99/0.02, 384R 1.09/0.01 | pass |
