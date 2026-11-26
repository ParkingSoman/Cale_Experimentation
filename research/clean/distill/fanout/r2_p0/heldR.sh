#!/bin/sh
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
for t in r18_r2act_n256_s1 r18_r2color_n256_s1 r18_r2ent_n256_s1 r18_r2k8_n256_s1; do
  $PY -I distill/fanout/board.py --net r18 --rule noisecut --f 0.95,0.85 --split held --who P0 --note "r2 P0: TUNING SIGNAL (held split), input $t" --input distill/fanout/cache/$t.pt 2>&1 | grep '^{'
done
