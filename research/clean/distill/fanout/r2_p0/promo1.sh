#!/bin/sh
# P0 round 2 PROMOTE: floor (noisecut) on CLIP pink vs CLIP self input, full 1,000 test photos, .95 and .85.
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
$PY -I distill/fanout/board.py --net clip --rule noisecut --f 0.95,0.85 --who P0 --note "r2 P0: floor on shared pink (85% baseline for the input test)" --input distill/fanout/cache/input_pink_n256_s1.pt 2>&1 | grep '^{'
$PY -I distill/fanout/board.py --net clip --rule noisecut --f 0.95,0.85 --who P0 --note "r2 P0: floor on CLIP self-made input (make_input K32, self-InfoNCE proxy)" --input distill/fanout/cache/clip_self_n256_k32_s1.pt 2>&1 | grep '^{'
