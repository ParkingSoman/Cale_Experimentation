#!/bin/sh
cd /Users/shonusengupta/Cale_Experimentation/research/clean || exit 1
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
for r in r2_p1p3_a r2_p1p3_d r2_p1p3_p; do
  $PY -I distill/fanout/board.py --net clip --rule $r --f 0.95,0.85,0.5 --who P1-P3 --input distill/fanout/cache/input_pink_n256_s1.pt --note "r2 P1-P3: CLIP MLP-only, P4 fold on merged firings; a = no merge, d = R1 (weighted by noise RMS) OR-merge, p = shape-kept null (R1 owners permuted over deleted units)" 2>&1 | grep -v Warn
done
