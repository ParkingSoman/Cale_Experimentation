#!/bin/sh
# P0 round 2 batch A: four ResNet input variants (one knob each), then the floor quick at .95,.85 on each.
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
$PY -I distill/fanout/r2_p0_mkinput.py --tag r2act --target act 2>&1 | grep -v Warn | tail -2
$PY -I distill/fanout/r2_p0_mkinput.py --tag r2color --color 2>&1 | grep -v Warn | tail -2
$PY -I distill/fanout/r2_p0_mkinput.py --tag r2ent --target ent 2>&1 | grep -v Warn | tail -2
$PY -I distill/fanout/r2_p0_mkinput.py --tag r2k8 --K 8 2>&1 | grep -v Warn | tail -2
for t in r18_r2act_n256_s1 r18_r2color_n256_s1 r18_r2ent_n256_s1 r18_r2k8_n256_s1 r18_self_n256_k32_s1 input_pink_n256_s1; do
  $PY -I distill/fanout/board.py --net r18 --rule noisecut --f 0.95,0.85 --quick --who P0 --input distill/fanout/cache/$t.pt 2>&1 | grep '^{'
done
