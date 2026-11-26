# VECMATH 10: the read tilt, read as vectors, and the stream-error rule

P0 (A's 512 kernels through reads()): top z median 3.817, leftover median 0.799, pass True

## Checks

| Cut | known answer: share above own shuffle 95th (bar .9) | stream error^2 per cell at C | q along a (share removed) |
|---|---|---|---|
| 11 | 0.86 | 0.65 | 0.01 |
| 64 | 0.85 | 4.34 | 0.01 |
| 192 | 0.82 | 14.99 | 0.01 |
| 384 | 0.86 | 42.30 | 0.01 |

## P1: is the tilt reproducible (same unit, L against R4C)

| Cut | units | same-unit cosine, median | mismatched median | mismatched 99th | share of units above mismatched 99th |
|---|---|---|---|---|---|
| 11 | 501 | 0.830 | 0.006 | 0.316 | 1.00 |
| 64 | 448 | 0.829 | 0.017 | 0.266 | 1.00 |
| 192 | 320 | 0.857 | 0.035 | 0.337 | 1.00 |
| 384 | 128 | 0.838 | 0.054 | 0.412 | 1.00 |

## Size of the tilt

| Cut | build | alpha median (read metric) | tilt length / A read length, median | bn1 shift change, median abs |
|---|---|---|---|---|
| 11 | L | 1.000 | 0.027 | 0.000 |
| 11 | R4C | 1.000 | 0.025 | 0.000 |
| 64 | L | 1.006 | 0.084 | 0.001 |
| 64 | R4C | 1.002 | 0.081 | 0.002 |
| 192 | L | 0.995 | 0.124 | 0.001 |
| 192 | R4C | 0.983 | 0.119 | 0.003 |
| 384 | L | 1.000 | 0.232 | 0.001 |
| 384 | R4C | 0.958 | 0.200 | 0.005 |

## P3-P5: does the tilt point along the rule q

| Cut | build | cos(t, q) median | shuffle median | share of units above own shuffle 95th | neighbour control median | share cos(t,q) > neighbour |
|---|---|---|---|---|---|---|
| 11 | L | 0.138 | 0.077 | 0.68 | 0.007 | 0.73 |
| 11 | R4C | 0.125 | 0.067 | 0.65 | -0.002 | 0.73 |
| 64 | L | 0.153 | 0.095 | 0.80 | 0.031 | 0.74 |
| 64 | R4C | 0.136 | 0.085 | 0.76 | 0.015 | 0.76 |
| 192 | L | 0.146 | 0.077 | 0.92 | 0.030 | 0.72 |
| 192 | R4C | 0.115 | 0.034 | 0.94 | 0.013 | 0.73 |
| 384 | L | 0.336 | 0.258 | 0.86 | 0.097 | 0.81 |
| 384 | R4C | 0.294 | 0.197 | 0.91 | 0.061 | 0.82 |

## P6: does the rule open and close the right cells

Predicted read = a + |t| x unit(q), A's threshold. Hit = share of training's new-on (lost) cells that the predicted read also opens (closes). Pooled over units.

| Cut | build | training new-on / onA | rule: hit new | shuffle: hit new | neighbour: hit new | training lost / onA | rule: hit lost | shuffle: hit lost | neighbour: hit lost | rule opens / training opens |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | L | 0.013 | 0.639 | 0.636 | 0.463 | 0.017 | 0.380 | 0.395 | 0.497 | 7.73 |
| 11 | R4C | 0.011 | 0.636 | 0.632 | 0.467 | 0.017 | 0.376 | 0.392 | 0.497 | 8.39 |
| 64 | L | 0.030 | 0.699 | 0.691 | 0.498 | 0.068 | 0.369 | 0.383 | 0.495 | 12.65 |
| 64 | R4C | 0.023 | 0.702 | 0.693 | 0.513 | 0.076 | 0.365 | 0.381 | 0.495 | 16.16 |
| 192 | L | 0.041 | 0.953 | 0.948 | 0.705 | 0.120 | 0.130 | 0.138 | 0.273 | 22.32 |
| 192 | R4C | 0.026 | 0.948 | 0.943 | 0.707 | 0.144 | 0.116 | 0.127 | 0.270 | 34.13 |
| 384 | L | 0.502 | 0.899 | 0.937 | 0.589 | 0.021 | 0.109 | 0.090 | 0.325 | 4.24 |
| 384 | R4C | 0.265 | 0.932 | 0.952 | 0.603 | 0.057 | 0.119 | 0.084 | 0.304 | 6.92 |

Added after run 1 (bar unchanged): the same test with the mean shift removed from each predicted tilt (_zm).

Hit = share of training's new-on (lost) cells the predicted read also opens (closes). Opens = cells it opens / cells training opens.

| Cut | build | mean shift / spread: training tilt, rule | hit new: rule / shuffle / neighbour / random | hit lost: rule / shuffle / neighbour / random | opens: rule / shuffle / neighbour / random |
|---|---|---|---|---|---|
| 11 | L | 0.006, 0.111 | 0.41 / 0.38 / 0.31 / 0.29 | 0.34 / 0.31 / 0.30 / 0.28 | 1.29 / 1.31 / 1.24 / 1.14 |
| 11 | R4C | 0.005, 0.103 | 0.41 / 0.39 / 0.31 / 0.29 | 0.33 / 0.30 / 0.30 / 0.28 | 1.41 / 1.43 / 1.36 / 1.25 |
| 64 | L | 0.026, 0.377 | 0.46 / 0.43 / 0.32 / 0.29 | 0.33 / 0.30 / 0.31 / 0.27 | 1.75 / 1.78 / 1.62 / 1.56 |
| 64 | R4C | 0.032, 0.362 | 0.47 / 0.44 / 0.33 / 0.30 | 0.32 / 0.28 / 0.30 / 0.26 | 2.25 / 2.28 / 2.09 / 2.00 |
| 192 | L | 0.060, 0.599 | 0.53 / 0.50 / 0.35 / 0.29 | 0.30 / 0.23 / 0.28 / 0.25 | 2.06 / 2.15 / 1.92 / 1.72 |
| 192 | R4C | 0.082, 0.582 | 0.59 / 0.54 / 0.37 / 0.30 | 0.27 / 0.21 / 0.26 / 0.24 | 3.16 / 3.30 / 2.94 / 2.63 |
| 384 | L | 0.270, 1.253 | 0.27 / 0.27 / 0.22 / 0.20 | 0.51 / 0.38 / 0.38 / 0.34 | 0.35 / 0.36 / 0.33 / 0.30 |
| 384 | R4C | 0.134, 1.070 | 0.35 / 0.34 / 0.25 / 0.22 | 0.46 / 0.35 / 0.35 / 0.31 | 0.57 / 0.59 / 0.54 / 0.48 |

## Added run 3: does the tilt point at the deleted units' reads?

Per unit: best cosine of the tilt with a read (its a_j part removed), deleted units against other kept units, same set size m.

| Cut | build | m | best cos with a deleted read (median) | best cos with a kept read (median) | share of units deleted > kept |
|---|---|---|---|---|---|
| 11 | L | 11 | 0.249 | 0.140 | 0.74 |
| 11 | R4C | 11 | 0.215 | 0.137 | 0.73 |
| 64 | L | 64 | 0.273 | 0.203 | 0.85 |
| 64 | R4C | 64 | 0.265 | 0.199 | 0.84 |
| 192 | L | 192 | 0.307 | 0.271 | 0.71 |
| 192 | R4C | 192 | 0.318 | 0.273 | 0.79 |
| 384 | L | 127 | 0.286 | 0.242 | 0.80 |
| 384 | R4C | 127 | 0.296 | 0.248 | 0.84 |

## P2 and concepts: what the tilt reads as (vecread s3k, whitened pattern, hub z)

| Cut | vector | top z median | leftover median |
|---|---|---|---|
| 11 | A | 3.81 | 0.80 |
| 11 | t_L | 3.64 | 0.84 |
| 11 | t_R4C | 3.40 | 0.85 |
| 11 | q | 4.19 | 0.68 |
| 64 | A | 3.78 | 0.80 |
| 64 | t_L | 3.66 | 0.84 |
| 64 | t_R4C | 3.51 | 0.85 |
| 64 | q | 4.23 | 0.67 |
| 192 | A | 3.80 | 0.80 |
| 192 | t_L | 3.92 | 0.80 |
| 192 | t_R4C | 3.83 | 0.81 |
| 192 | q | 4.16 | 0.68 |
| 384 | A | 4.11 | 0.78 |
| 384 | t_L | 4.56 | 0.75 |
| 384 | t_R4C | 4.34 | 0.78 |
| 384 | q | 4.23 | 0.68 |

Most common concepts in the top 5 (share of units). A = A's reads (= C's). t = the tilt.

### Cut 11 (501 kept units)

| vector | side | top concepts (share of units) |
|---|---|---|
| A | up | potholed 0.12, stained 0.11, crystalline 0.10, marbled 0.10, blotchy 0.09, wrinkled 0.08 |
| A | down | zigzagged 0.08, chequered 0.08, dotted 0.07, perforated 0.07, honeycombed 0.07, muzzle 0.06 |
| D | up | chandelier 0.27, bubbly 0.27, pitted 0.18, blotchy 0.18, stained 0.18, sprinkled 0.18 |
| D | down | chequered 0.36, paisley 0.18, meshed 0.18, dishwasher 0.18, skin 0.18, grid 0.18 |
| t_L | up | nose 0.12, fan 0.10, paisley 0.10, chandelier 0.10, swirly 0.09, flower 0.07 |
| t_L | down | paisley 0.09, swirly 0.09, interlaced 0.07, stained 0.06, crosshatched 0.05, grooved 0.05 |
| t_R4C | up | nose 0.10, paisley 0.08, chandelier 0.08, fan 0.07, swirly 0.07, zigzagged 0.06 |
| t_R4C | down | paisley 0.06, swirly 0.06, interlaced 0.05, zigzagged 0.05, spiralled 0.05, crosshatched 0.05 |
| q | up | table 0.16, forest-broadleaf-s 0.11, plaything 0.11, painted 0.10, kitchen island 0.10, ottoman 0.10 |
| q | down | tree 0.37, forest-broadleaf-s 0.23, painted 0.19, field-wild-s 0.18, wallpaper 0.16, matted 0.14 |
| t_L vs q | up shared (>= 1 of top 5) | same unit 0.15, other unit 0.05 |
| t_R4C vs q | up shared (>= 1 of top 5) | same unit 0.13, other unit 0.05 |

### Cut 64 (448 kept units)

| vector | side | top concepts (share of units) |
|---|---|---|
| A | up | potholed 0.12, stained 0.10, crystalline 0.10, marbled 0.10, blotchy 0.09, wrinkled 0.08 |
| A | down | chequered 0.08, zigzagged 0.08, dog 0.07, muzzle 0.07, dotted 0.07, head 0.06 |
| D | up | stained 0.14, pitted 0.12, blotchy 0.12, chandelier 0.09, crystalline 0.09, potholed 0.09 |
| D | down | chequered 0.12, zigzagged 0.11, dotted 0.11, perforated 0.11, meshed 0.09, honeycombed 0.09 |
| t_L | up | nose 0.09, fan 0.09, wheel 0.08, chandelier 0.07, eye 0.06, muzzle 0.06 |
| t_L | down | alley-s 0.05, brick 0.05, stratified 0.04, wood 0.04, bubbly 0.04, paisley 0.04 |
| t_R4C | up | nose 0.09, wheel 0.07, fan 0.07, chandelier 0.07, muzzle 0.05, bubbly 0.05 |
| t_R4C | down | swirly 0.05, alley-s 0.05, tent 0.05, skin 0.04, paisley 0.04, freckled 0.04 |
| q | up | table 0.24, ottoman 0.15, chair 0.14, kitchen island 0.12, cup 0.11, footboard 0.11 |
| q | down | tree 0.43, forest-broadleaf-s 0.36, wallpaper 0.25, painted 0.21, field-wild-s 0.21, forest-needleleaf-s 0.18 |
| t_L vs q | up shared (>= 1 of top 5) | same unit 0.21, other unit 0.04 |
| t_R4C vs q | up shared (>= 1 of top 5) | same unit 0.18, other unit 0.04 |

### Cut 192 (320 kept units)

| vector | side | top concepts (share of units) |
|---|---|---|
| A | up | potholed 0.12, marbled 0.12, stained 0.11, crystalline 0.10, wrinkled 0.09, blotchy 0.09 |
| A | down | zigzagged 0.09, chequered 0.07, muzzle 0.07, honeycombed 0.07, dog 0.06, grid 0.06 |
| D | up | potholed 0.12, stained 0.11, blotchy 0.10, crystalline 0.10, pitted 0.08, smeared 0.07 |
| D | down | chequered 0.10, perforated 0.09, dotted 0.09, meshed 0.07, tvmonitor 0.06, honeycombed 0.06 |
| t_L | up | chandelier 0.08, cat 0.07, wheel 0.07, dog 0.06, muzzle 0.06, ear 0.06 |
| t_L | down | bedclothes 0.07, skin 0.05, freckled 0.05, sky 0.05, blue-c 0.05, plaza-s 0.04 |
| t_R4C | up | chandelier 0.08, cat 0.08, flower 0.06, palm 0.06, fan 0.06, ear 0.05 |
| t_R4C | down | bedclothes 0.07, alley-s 0.06, staircase-s 0.05, laminate 0.05, skin 0.05, attic-s 0.05 |
| q | up | table 0.43, ottoman 0.27, body 0.23, cup 0.21, kitchen island 0.19, pot 0.18 |
| q | down | forest-broadleaf-s 0.71, tree 0.63, field-wild-s 0.42, forest-needleleaf-s 0.36, wallpaper 0.35, painted 0.34 |
| t_L vs q | up shared (>= 1 of top 5) | same unit 0.16, other unit 0.04 |
| t_R4C vs q | up shared (>= 1 of top 5) | same unit 0.11, other unit 0.03 |

### Cut 384 (128 kept units)

| vector | side | top concepts (share of units) |
|---|---|---|
| A | up | stained 0.11, crystalline 0.10, potholed 0.10, smeared 0.10, marbled 0.08, stratified 0.08 |
| A | down | zigzagged 0.12, dog 0.09, honeycombed 0.09, ball 0.09, chequered 0.09, dotted 0.08 |
| D | up | potholed 0.13, blotchy 0.11, stained 0.11, marbled 0.10, crystalline 0.10, wrinkled 0.08 |
| D | down | chequered 0.08, perforated 0.07, dotted 0.07, zigzagged 0.07, muzzle 0.06, head 0.06 |
| t_L | up | dog 0.10, cat 0.09, body 0.09, airplane 0.09, head 0.09, ear 0.07 |
| t_L | down | tree 0.12, marbled 0.11, brick 0.10, forest-needleleaf-s 0.09, ground 0.09, mountain-s 0.09 |
| t_R4C | up | cat 0.11, silver screen 0.08, sheep 0.08, wheel 0.08, head 0.08, fan 0.07 |
| t_R4C | down | forest-needleleaf-s 0.10, castle-s 0.09, mountain-s 0.08, tree 0.07, plaza-s 0.07, wood 0.07 |
| q | up | ottoman 0.33, table 0.30, pot 0.25, kitchen island 0.24, body 0.20, sky 0.16 |
| q | down | tree 0.69, forest-broadleaf-s 0.55, forest-needleleaf-s 0.31, creek-s 0.31, wallpaper 0.28, field-wild-s 0.22 |
| t_L vs q | up shared (>= 1 of top 5) | same unit 0.22, other unit 0.11 |
| t_R4C vs q | up shared (>= 1 of top 5) | same unit 0.22, other unit 0.07 |

## Named units

### Cut 11, n5

| vector | up (top 5) | down (top 5) | top z | data side: top IoU cells (examples) |
|---|---|---|---|---|
| A | back pillow, cushion, foot, fabric, pillow | tvmonitor, perforated, elevator / elevator door, galley-s, door frame / doorframe | 4.62 | person 0.033, leg 0.02, fabric 0.02 |
| r_L | back pillow, cushion, foot, fabric, pillow | tvmonitor, elevator / elevator door, perforated, galley-s, door frame / doorframe | 4.63 | person 0.032, leg 0.021, fabric 0.019 |
| t_L | interlaced, paisley, eye, swirly, jacuzzi-indoor-s | gauzy, crystalline, crosshatched, pleated, stained | 5.51 | interlaced 0.073, zigzagged 0.061, paisley 0.056 |
| t_R4C | interlaced, paisley, eye, swirly, food | gauzy, pleated, crosshatched, grooved, alley-s | 4.83 | interlaced 0.069, zigzagged 0.059, paisley 0.048 |
| q | vault, wallpaper, painted, brick, ceiling | ashcan, bag, person, flowerpot, vase | 4.23 | striped 0.066, zigzagged 0.066, banded 0.057 |
| deleted n464 (A read) | interlaced, paisley, food, eye, swirly | dishwasher, chequered, tvmonitor, screen, cage / bird cage | 3.74 | head 0.036, interlaced 0.034, cat 0.028 |
| tilt L: closest deleted read | n464 (cos +0.641) | | | |
| tilt R4C: closest deleted read | n464 (cos +0.667) | | | |
| cos(t, q) L / R4C | +0.461 / +0.427 | shuffle 95th +0.387 | | |

### Cut 64, n339

| vector | up (top 5) | down (top 5) | top z | data side: top IoU cells (examples) |
|---|---|---|---|---|
| A | arcades, gravestone, back, bus_interior-s, armchair | muzzle, fan, chandelier, nose, landing_deck-s | 5.13 | chair 0.021, person 0.015, car 0.015 |
| r_L | arcades, gravestone, back, bus_interior-s, chair | muzzle, fan, chandelier, landing_deck-s, nose | 5.76 | car 0.023, chair 0.021, wheel 0.013 |
| t_L | wheel, car, van, motorbike, arcades | bedclothes, torso, pink-c, neck, fabric | 6.42 | car 0.096, wheel 0.048, street-s 0.026 |
| t_R4C | car, wheel, van, motorbike, arcades | pink-c, torso, bedclothes, attic-s, arm | 5.60 | car 0.108, wheel 0.043, street-s 0.028 |
| q | van, car, wheel, motorbike, gate | attic-s, youth_hostel-s, corridor-s, staircase-s, wood | 6.09 | car 0.107, wheel 0.043, street-s 0.034 |
| deleted n476 (A read) | van, arcades, car, plaza-s, street-s | neck, torso, cat, bell, jacuzzi-indoor-s | 5.64 | car 0.102, street-s 0.033, bus 0.023 |
| tilt L: closest deleted read | n476 (cos +0.485) | | | |
| tilt R4C: closest deleted read | n476 (cos +0.549) | | | |
| cos(t, q) L / R4C | +0.645 / +0.650 | shuffle 95th +0.433 | | |

### Cut 192, n430

| vector | up (top 5) | down (top 5) | top z | data side: top IoU cells (examples) |
|---|---|---|---|---|
| A | striped, white-c, chequered, zigzagged, paper | bumpy, scaly, pottedplant, braided, waffled | 5.09 | zigzagged 0.045, person 0.032, white-c 0.027 |
| r_L | striped, chequered, white-c, paper, zigzagged | bumpy, scaly, pottedplant, honeycombed, frilly | 4.89 | sky 0.047, zigzagged 0.04, white-c 0.03 |
| t_L | painting, shelves, brown-c, waffled, nursery-s | snow, river, ice rink, sea, water | 3.98 | sky 0.114, blue-c 0.063, highway-s 0.023 |
| t_R4C | painting, waffled, bookcase, television, poster | ice rink, snow, crosswalk, road, ground | 4.54 | dog 0.028, head 0.026, painting 0.025 |
| q | ottoman, fireplace, cup, living_room-s, computer | forest-broadleaf-s, tree, creek-s, forest-needleleaf-s, river-s / lake-natural-s | 3.75 | car 0.02, head 0.013, person 0.013 |
| deleted n114 (A read) | wall, shelves, frame, waiting_room-s, kitchen-s | skin, bird, food, torso, ceramic | 4.39 | sky 0.065, white-c 0.029, blue-c 0.028 |
| tilt L: closest deleted read | n114 (cos +0.470) | | | |
| tilt R4C: closest deleted read | n405 (cos +0.391) | | | |
| cos(t, q) L / R4C | +0.236 / +0.284 | shuffle 95th +0.229 | | |

### Cut 384, n430

| vector | up (top 5) | down (top 5) | top z | data side: top IoU cells (examples) |
|---|---|---|---|---|
| A | striped, white-c, chequered, zigzagged, paper | bumpy, scaly, pottedplant, braided, waffled | 5.09 | zigzagged 0.045, person 0.032, white-c 0.027 |
| r_L | striped, white-c, chequered, banded, zigzagged | scaly, bumpy, pottedplant, frilly, braided | 4.84 | sky 0.114, blue-c 0.061, white-c 0.032 |
| t_L | sky, lighthouse-s, arrival_gate-outdoor-s, blue-c, highway-s | countertop, work surface, deck, baggage_claim-s, ground | 3.53 | sky 0.136, blue-c 0.075, white-c 0.026 |
| t_R4C | ear, horse, bird, cow, dog | ice rink, grey-c, laminate, baggage_claim-s, countertop | 4.02 | dog 0.044, head 0.042, torso 0.028 |
| q | sky, painted, blue-c, arrival_gate-outdoor-s, lighthouse-s | pedestal, paper, clothing_store-s, chair, box | 4.17 | sky 0.138, blue-c 0.075, white-c 0.026 |
| deleted n501 (A read) | tvmonitor, keyboard, screen, drawer, computer | dog, cat, tail, cow, bird | 3.87 | sky 0.111, blue-c 0.063, white-c 0.023 |
| tilt L: closest deleted read | n374 (cos +0.467) | | | |
| tilt R4C: closest deleted read | n418 (cos +0.353) | | | |
| cos(t, q) L / R4C | +0.782 / +0.363 | shuffle 95th +0.335 | | |

### Cut 384, n76

| vector | up (top 5) | down (top 5) | top z | data side: top IoU cells (examples) |
|---|---|---|---|---|
| A | freckled, blotchy, marbled, stained, smeared | grid, waffled, chequered, woven, meshed | 5.47 | head 0.055, sky 0.054, freckled 0.052 |
| r_L | freckled, neck, smeared, nose, stained | grid, chequered, woven, waffled, meshed | 5.21 | sky 0.091, blue-c 0.055, head 0.041 |
| t_L | spiralled, water tower, television, pleated, eye | earth, sand, ground, flower, dirt track | 3.20 | sky 0.115, blue-c 0.068, highway-s 0.023 |
| t_R4C | water tower, black-c, spiralled, tower, leather | ice rink, ground, flower, sand, earth | 3.88 | person 0.036, head 0.03, dog 0.023 |
| q | sky, arrival_gate-outdoor-s, blue-c, lighthouse-s, apartment_building-outdoor-s | embankment, grass, bush, field, green-c | 3.05 | sky 0.139, blue-c 0.076, white-c 0.027 |
| deleted n374 (A read) | marbled, blotchy, stained, mountain-s, cracked | computer, tvmonitor, washer, stove, monitor | 3.82 | sky 0.129, blue-c 0.07, white-c 0.026 |
| tilt L: closest deleted read | n284 (cos +0.529) | | | |
| tilt R4C: closest deleted read | n284 (cos +0.319) | | | |
| cos(t, q) L / R4C | +0.665 / +0.365 | shuffle 95th +0.360 | | |
