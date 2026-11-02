# Review results (blind held-out class test)

Pass = mean HIGH percentile above mean NEAR by at least 0.1. Chance: both near 0.5. Each row: 4 HIGH and 4 NEAR classes, none named in the card; picks in review_predictions.md, written before scoring.

| Net | Channel | Explanation | HIGH pct | NEAR pct | Result |
|---|---|---|---|---|---|
| resnet18 | 120 | band curving parallel to itself | 0.84 | 0.43 | pass |
| resnet18 | 147 | self-lit warm glow on dark | 0.75 | 0.54 | pass |
| resnet18 | 161 | small sharp thing on blur (macro) | 0.90 | 0.30 | pass |
| resnet18 | 179 | thin lines out from a hub vs thick solid | 0.73 | 0.47 | pass |
| resnet18 | 222 | many same-size touching cells | 0.74 | 0.53 | pass |
| resnet18 | 224 | warm yellow-tan soft pieces | 0.74 | 0.43 | pass |
| resnet18 | 273 | emitted light against darkness | 0.65 | 0.40 | pass |
| resnet18 | 327 | curved loop or arc (handle) | 0.76 | 0.47 | pass |
| resnet18 | 371 | body with limbs radiating from a centre | 0.72 | 0.47 | pass |
| resnet18 | 401 | round dots tiled, and open blue water | 0.64 | 0.30 | pass |
| resnet18 | 416 | dark matte-glossy fine ridges | 0.70 | 0.63 | fail |
| resnet18 | 510 | water meeting a built shore edge | 0.95 | 0.36 | pass |
| resnet50 | 214 | tight curls vs flat layers | 0.82 | 0.46 | pass |
| resnet50 | 257 | thin upright stalks packed in parallel | 0.55 | 0.36 | pass |
| resnet50 | 364 | extremity pointing out of a body | 0.77 | 0.40 | pass |
| resnet50 | 544 | soft irregular pitted matte surface | 0.65 | 0.53 | pass |
| resnet50 | 969 | broad flat slab edging over a gap | 0.80 | 0.54 | pass |
| resnet50 | 972 | body hanging from a straight bar | 0.76 | 0.58 | pass |
| resnet50 | 1113 | sleek short coat vs fluffy long fur | 0.88 | 0.25 | pass |
| resnet50 | 1256 | curved banded cylinder | 0.78 | 0.43 | pass |
| resnet50 | 1699 | flat blades or petals fanning from a centre | 0.62 | 0.49 | pass |
| resnet50 | 1852 | thin straight light lines in a grid | 0.56 | 0.32 | pass |
| resnet50 | 1882 | curved glossy mirror with highlight | 0.64 | 0.45 | pass |
| resnet50 | 1934 | coarse parallel straw strands | 0.68 | 0.48 | pass |

## Counts
- resnet18: 11/12 pass; mean HIGH 0.759, mean NEAR 0.444
- resnet50: 12/12 pass; mean HIGH 0.708, mean NEAR 0.441
- overall: 23/24 pass; mean HIGH 0.733, mean NEAR 0.442

## Notes
- One fail: resnet18 416 (dark matte-glossy fine ridges): HIGH 0.70 vs NEAR 0.63. NEAR classes pop bottle (0.79) and ping-pong ball (0.65) fired as much as the ridged-texture picks.
- Narrow passes (gap 0.11 to 0.19): resnet50 544 (pitted matte surface), 1699 (soft radial petals; sea anemone and sundial near chance), 257 (stalks; broom, thatch, paintbrush near 0.5), 1882 (glossy mirror; only gong high).
- Weak spots inside passing rows: resnet18 401 (sea lion 0.31 low, the water facet holds for whales and reef), resnet50 972 (ground apes 0.5-0.76 close to climbers), resnet18 120 (rock python 0.51).
- Pattern: explanations that name only a surface texture or a glossy/matte quality (416, 544, 1882, 1699) are the weakest. Explanations built on a clear opposite (pale vs dark coat 1113, water vs shore 510, macro vs full-frame 161, curl vs flat 214) win by 0.3 to 0.7.
- Caveat: channels drawn at random (seed 7) from those whose card states a distinction. Picks are one reviewer's, 4 per side, so each row is noisy (about +-0.1).
