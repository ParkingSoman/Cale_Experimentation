#!/bin/sh
# P0 round 2: P6 ps-pca stream-only, bases from: null input (quick first, the deciding row), then full 1,000 for pink, self, null.
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
C=distill/fanout/cache
for t in clip_self_n256_k32_s1 clip_null_n256_k32_s1 input_pink_n256_s1; do
  $PY -I distill/fanout/r2_p0_ps.py --input $C/$t.pt --f 0.85,0.7 --stride 1 2>&1 | grep ROW
done
