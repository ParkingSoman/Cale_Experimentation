# Widening controls and leftover plan items (oc_vs_widen_ctl.py)

Note: the read bar in oc_vs_read.py first used raw read kernels; it was changed to whitened patterns (the vecread rule) after that first run. With raw kernels, the 384 rows read .22 / .46 / .76 (fail); n339 at 64 passed with both metrics (.98).

## 1. Rank of the named unit among all kept units, for the same partner

| Cut | Unit -> partner | Build | Write rank among kept | Read rank among kept |
|---|---|---|---|---|
| 11 | n5 -> n464 | L | 1.00 | 0.99 |
| 11 | n5 -> n464 | R4C | 1.00 | 0.99 |
| 64 | n339 -> n476 | L | 1.00 | 1.00 |
| 64 | n339 -> n476 | R4C | 1.00 | 1.00 |
| 192 | n430 -> n114 | L | 0.85 | 0.99 |
| 192 | n430 -> n114 | R4C | 0.97 | 0.90 |
| 384 | n430 -> n501 | L | 0.83 | 0.95 |
| 384 | n76 -> n374 | L | 0.84 | 0.88 |
| 384 | n213 -> n501 | L | 0.92 | 0.98 |
| 384 | n430 -> n501 | R4C | 0.91 | 0.70 |
| 384 | n76 -> n374 | R4C | 0.97 | 0.82 |
| 384 | n213 -> n501 | R4C | 0.97 | 0.83 |

## 2. Kept units that pass write + read + keeps-job with some partner (top 5% of the deleted on both)

| Cut | Build | Real | Shuffled partner labels, mean (max of 20) |
|---|---|---|---|
| 11 | L | 193 of 501 | 44.2 (max 58) |
| 11 | R4C | 183 of 501 | 47.4 (max 60) |
| 64 | L | 353 of 448 | 106.0 (max 120) |
| 64 | R4C | 346 of 448 | 105.8 (max 125) |
| 192 | L | 276 of 320 | 131.4 (max 144) |
| 192 | R4C | 273 of 320 | 133.1 (max 144) |
| 384 | L | 125 of 128 | 83.3 (max 95) |
| 384 | R4C | 121 of 128 | 85.8 (max 95) |

## 3. Readers' Fisher G (stream after block 0), top 5 eigenvectors read as writes

Top-20 share of trace 0.134; top-5 share 0.060.

| Rank | Share | + pole (up 3) | - pole (down 3) | Leftover |
|---|---|---|---|---|
| 1 | 0.023 | file cabinet +0.05, airport_terminal-s +0.04, elevator / elevator door +0.04 | building_facade-s -0.04, painted -0.04, stained -0.04 | 0.98 |
| 2 | 0.016 | painted +0.18, wallpaper +0.12, concrete +0.12 | sculpture -0.18, bag -0.15, person -0.15 | 0.92 |
| 3 | 0.008 | mirror +0.14, blind +0.13, windowpane +0.12 | floor -0.20, carpet -0.19, granite -0.17 | 0.88 |
| 4 | 0.007 | crystalline +0.12, smeared +0.11, blotchy +0.10 | museum-indoor-s -0.11, pink-c -0.11, nursery-s -0.10 | 0.96 |
| 5 | 0.006 | white-c +0.07, grey-c +0.06, painted +0.06 | basket -0.10, paw -0.10, pot -0.09 | 0.97 |

Check: n430's A write reads zigzagged, chequered, interlaced (tool used for all 512 A writes in K5b; top cosine 0.32, leftover 0.83).