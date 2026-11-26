# VECMATH 11: which kept neurons aim at a cut neuron, and why

## Q1: does j's aim at d rise when d is cut?

All pairs: median rise of aim (c1 -> c2), d entering the cut against d kept at both. Null = labels permuted.

| Cuts | build | j | entering | rise, entering | rise, staying | difference | null 99th |
|---|---|---|---|---|---|---|---|
| 11->64 | L | 448 | 53 | 0.002 | -0.005 | 0.007 | 0.003 |
| 11->64 | R4C | 448 | 53 | 0.002 | -0.005 | 0.007 | 0.004 |
| 64->192 | L | 320 | 128 | -0.000 | -0.008 | 0.008 | 0.003 |
| 64->192 | R4C | 320 | 128 | 0.008 | -0.001 | 0.009 | 0.003 |
| 192->384 | L | 128 | 192 | -0.017 | -0.018 | 0.001 | 0.007 |
| 192->384 | R4C | 128 | 192 | -0.008 | -0.011 | 0.003 | 0.007 |

Top partner only (added after the run): j's best-aimed d at c2, its aim at c1 and c2. Staying = the same pick among other kept neurons, same set size.

| Cuts | build | entering partner: c1 -> c2 | staying pick: c1 -> c2 | extra rise |
|---|---|---|---|---|
| 11->64 | L | 0.052 -> 0.258 | 0.067 -> 0.198 | 0.075 |
| 11->64 | R4C | 0.053 -> 0.258 | 0.064 -> 0.193 | 0.075 |
| 64->192 | L | 0.119 -> 0.295 | 0.115 -> 0.247 | 0.043 |
| 64->192 | R4C | 0.089 -> 0.303 | 0.106 -> 0.248 | 0.073 |
| 192->384 | L | 0.115 -> 0.279 | 0.121 -> 0.242 | 0.042 |
| 192->384 | R4C | 0.082 -> 0.290 | 0.100 -> 0.248 | 0.060 |

n339 aim at n476: cut 11 (n476 kept) -0.055 -> cut 64 (n476 cut) 0.485 (L); -0.016 -> 0.549 (R4C)

## Q2, Q5 and the rest: who aims (Spearman of best aim with each cause; shuffle 99th)

| Cut | build | parent read (s3:69) | write norm | beta_C mass | co-firing with D | self-correlation (L) |
|---|---|---|---|---|---|---|
| 11 | L | 0.274 (0.101) | -0.002 (0.104) | 0.379 (0.101) | 0.321 (0.101) | -0.500 (0.103) |
| 11 | R4C | 0.267 (0.102) | -0.033 (0.101) | 0.525 (0.106) | 0.477 (0.098) |  |
| 64 | L | 0.125 (0.103) | 0.048 (0.102) | 0.313 (0.107) | 0.105 (0.112) | -0.164 (0.107) |
| 64 | R4C | 0.114 (0.104) | -0.003 (0.109) | 0.418 (0.103) | 0.229 (0.110) |  |
| 192 | L | 0.034 (0.136) | 0.078 (0.133) | 0.142 (0.128) | 0.099 (0.131) | -0.065 (0.126) |
| 192 | R4C | 0.040 (0.134) | 0.130 (0.130) | 0.222 (0.124) | 0.053 (0.129) |  |
| 384 | L | 0.078 (0.210) | 0.054 (0.193) | 0.212 (0.197) | 0.142 (0.205) | -0.165 (0.208) |
| 384 | R4C | 0.198 (0.193) | 0.083 (0.198) | 0.273 (0.201) | 0.074 (0.195) |  |

## Q3, Q4 and the rest: which partner (is j's partner the top-ranked cut neuron on each score?)

Hit / chance (chance = 1/|D|). Hits = number of kept neurons. Rank pct = median percentile of the partner on that score (1 = top).

| Cut | build | shared parent | whole-read similarity | beta_C | co-firing | shared parent within the read-similarity fifth | known answer (planted) |
|---|---|---|---|---|---|---|---|
| 11 | L | 1.1x (51; pct 0.60) | 2.6x (117; pct 0.80) | 3.7x (167; pct 0.80) | 3.1x (142; pct 0.80) | 1.0x | 1.00 |
| 11 | R4C | 1.4x (62; pct 0.60) | 2.9x (133; pct 0.80) | 4.1x (188; pct 0.90) | 3.2x (146; pct 0.80) | 1.1x | 1.00 |
| 64 | L | 2.3x (16; pct 0.67) | 7.0x (49; pct 0.83) | 11.9x (83; pct 0.90) | 9.1x (64; pct 0.86) | 1.7x | 1.00 |
| 64 | R4C | 2.9x (20; pct 0.66) | 8.6x (60; pct 0.87) | 16.6x (116; pct 0.94) | 11.0x (77; pct 0.87) | 1.8x | 1.00 |
| 192 | L | 3.6x (6; pct 0.66) | 6.6x (11; pct 0.71) | 12.6x (21; pct 0.84) | 9.6x (16; pct 0.76) | 2.8x | 1.00 |
| 192 | R4C | 6.6x (11; pct 0.69) | 13.8x (23; pct 0.83) | 31.2x (52; pct 0.95) | 19.8x (33; pct 0.87) | 4.2x | 1.00 |
| 384 | L | 3.0x (1; pct 0.54) | 6.0x (2; pct 0.74) | 6.0x (2; pct 0.85) | 6.0x (2; pct 0.82) | 2.4x | 1.00 |
| 384 | R4C | 3.0x (1; pct 0.63) | 6.0x (2; pct 0.83) | 21.0x (7; pct 0.94) | 18.0x (6; pct 0.88) | 2.4x | 1.00 |

## Vecread: A's reads of the strongest and weakest aimers (R4C), share of the group with the concept in its top 5

Reference: A's 512 read kernels (vecread README K5b hubs): potholed .12, stained .11, crystalline .10, marbled .10.

| Cut | group | best aim (median) | top concepts |
|---|---|---|---|
| 11 | top 50 | 0.500 | stained 0.14, marbled 0.12, blotchy 0.12, matted 0.10, pitted 0.10, arcades 0.08 |
| 11 | bottom 50 | 0.054 | potholed 0.14, gauzy 0.12, marbled 0.12, crystalline 0.10, smeared 0.10, veined 0.10 |
| 64 | top 44 | 0.424 | stained 0.16, blotchy 0.16, potholed 0.14, matted 0.11, marbled 0.09, smeared 0.09 |
| 64 | bottom 44 | 0.155 | potholed 0.20, crystalline 0.16, blotchy 0.14, stained 0.11, marbled 0.11, smeared 0.09 |
| 192 | top 32 | 0.432 | potholed 0.16, crystalline 0.16, pitted 0.12, amusement_park-s 0.12, warehouse-indoor-s 0.09, fibrous 0.09 |
| 192 | bottom 32 | 0.218 | wrinkled 0.16, potholed 0.12, stained 0.12, matted 0.12, frilly 0.09, stratified 0.09 |
| 384 | top 12 | 0.484 | smeared 0.25, gauzy 0.25, potholed 0.25, crystalline 0.17, marbled 0.17, apartment_building-outdoor-s 0.17 |
| 384 | bottom 12 | 0.246 | dome 0.25, crystalline 0.25, chandelier 0.17, potholed 0.17, stained 0.17, poster 0.08 |

## The 5 strongest aimers per cut (R4C)

### Cut 11

| j -> partner | aim | j's A read | j's tilt | partner's A read | data side: j / tilt / partner | shared parent channel | j's read on s3:69 | self (L) |
|---|---|---|---|---|---|---|---|---|
| n154 -> n464 | 0.81 | pole, bridge-s, water, concrete | nose, swirly, paisley, interlaced | interlaced, paisley, food, eye | wheel, bicycle / head, interlaced / head, interlaced | s3:49 (0.021) | 0.028 | 0.997 |
| n86 -> n23 | 0.72 | hair, smeared, stained, ear | porous, pitted, sprinkled, blotchy | pitted, porous, blotchy, stained | person, head / person, head / person, head | s3:225 (0.016) | -0.105 | 1.000 |
| n5 -> n464 | 0.67 | back pillow, cushion, foot, fabric | interlaced, paisley, eye, swirly | interlaced, paisley, food, eye | person, leg / interlaced, zigzagged / head, interlaced | s3:96 (0.018) | 0.020 | 0.997 |
| n65 -> n464 | 0.64 | horse, vase, cow, saddle | interlaced, nose, zigzagged, paw | interlaced, paisley, food, eye | torso, chair / interlaced, zigzagged / head, interlaced | s3:49 (0.040) | -0.011 | 0.998 |
| n18 -> n297 | 0.62 | marbled, fibrous, matted, wrinkled | chandelier, wheel, flower, paisley | stained, blotchy, smeared, marbled | mountain, orange-c / wheel, painting / wheel, painting | s3:45 (0.018) | 0.028 | 0.999 |

### Cut 64

| j -> partner | aim | j's A read | j's tilt | partner's A read | data side: j / tilt / partner | shared parent channel | j's read on s3:69 | self (L) |
|---|---|---|---|---|---|---|---|---|
| n328 -> n33 | 0.57 | dome, chimney, tower, bell | jacket, tent, escalator, curtain | tent, escalator, umbrella, jacket | floor, sidewalk / person, torso / floor, carpet | s3:207 (0.091) | -0.183 | 0.992 |
| n154 -> n464 | 0.56 | pole, bridge-s, water, concrete | nose, rock, interlaced, lacelike | interlaced, paisley, food, eye | wheel, bicycle / interlaced, honeycombed / head, interlaced | s3:49 (0.021) | 0.028 | 0.990 |
| n339 -> n476 | 0.55 | arcades, gravestone, back, bus_interior-s | car, wheel, van, motorbike | van, arcades, car, plaza-s | chair, person / car, wheel / car, street-s | s3:23 (0.042) | -0.127 | 0.985 |
| n114 -> n197 | 0.52 | wall, shelves, frame, waiting_room-s | skyscraper, woven, stairway, skyscraper-s | stairway, blind, drawer, skyscraper-s | sky, white-c / lined, banded / lined, pleated | s3:47 (0.010) | 0.042 | 0.998 |
| n185 -> n509 | 0.51 | stained, wood, marbled, alley-s | rock, stratified, brick, ruins | stratified, ruins, rock, potholed | orange-c, wood / head, stratified / building, rock | s3:222 (0.024) | -0.102 | 0.992 |

### Cut 192

| j -> partner | aim | j's A read | j's tilt | partner's A read | data side: j / tilt / partner | shared parent channel | j's read on s3:69 | self (L) |
|---|---|---|---|---|---|---|---|---|
| n258 -> n451 | 0.58 | sidewalk, ground, road, snow | fence, wood, double door, wardrobe | central reservation, grass, fence, ground | road, sidewalk / brown-c, wood / grass, road | s3:108 (0.011) | -0.039 | 0.988 |
| n293 -> n236 | 0.57 | fence, stratified, potholed, crystalline | street-s, grass, tile, park-s | ground, grass, water, tile | white-c, grass / grass, road / grass, ground | s3:108 (0.046) | -0.068 | 0.981 |
| n339 -> n476 | 0.54 | arcades, gravestone, back, bus_interior-s | car, wheel, van, motorbike | van, arcades, car, plaza-s | chair, person / car, wheel / car, street-s | s3:23 (0.042) | -0.127 | 0.975 |
| n7 -> n393 | 0.52 | chequered, book, warehouse-indoor-s, bookcase | freckled, flecked, cat, sheep | sheep, castle-s, pasture-s, matted | chequered, building / sky, blue-c / sky, blue-c | s3:226 (0.011) | -0.279 | 0.982 |
| n390 -> n472 | 0.51 | tree, ocean-s, forest-broadleaf-s, forest-needleleaf-s | museum-indoor-s, kitchen island, floor, wood | carpet, tile, concrete, ground | person, torso / floor, ground / torso, grass | s3:158 (0.061) | -0.007 | 0.984 |

### Cut 384

| j -> partner | aim | j's A read | j's tilt | partner's A read | data side: j / tilt / partner | shared parent channel | j's read on s3:69 | self (L) |
|---|---|---|---|---|---|---|---|---|
| n296 -> n374 | 0.63 | honeycombed, grid, chequered, perforated | mountain-s, bridge-s, river-s / lake-natural-s, mountain_snowy-s | marbled, blotchy, stained, mountain-s | chequered, honeycombed / sky, blue-c / sky, blue-c | s3:137 (0.011) | -0.179 | 0.884 |
| n359 -> n510 | 0.54 | crystalline, smeared, stained, blotchy | flower, lacelike, paisley, mountain | bubbly, paisley, bumpy, plastic-clear | brown-c, yellow-c / paisley, lacelike / paisley, swirly | s3:11 (0.010) | -0.093 | 0.948 |
| n372 -> n401 | 0.53 | woven, zigzagged, apartment_building-outdoor-s, book | coast-s, golf_course-s, ocean-s, beach-s | coast-s, beach-s, sea, ocean-s | zigzagged, woven / sky, blue-c / sea, water | s3:246 (0.010) | -0.107 | 0.847 |
| n268 -> n446 | 0.51 | apartment_building-outdoor-s, building_facade-s, building, balcony | golf_course-s, valley-s, cloud, mountain-s | waffled, field-wild-s, field-cultivated-s, valley-s | person, building / sky, blue-c / sky, zigzagged | s3:129 (0.010) | -0.153 | 0.859 |
| n119 -> n501 | 0.50 | smeared, highway-s, painted, gauzy | field-wild-s, field-cultivated-s, valley-s, pasture-s | tvmonitor, keyboard, screen, drawer | sky, blue-c / sky, blue-c / sky, blue-c | s3:59 (0.013) | -0.127 | 0.970 |
