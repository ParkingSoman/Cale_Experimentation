#!/bin/sh
# P0 round 2 batch B. (1) ResNet tuning signal: floor on the HELD split, full 1,000 photos, .95,.85, each input.
# (2) ResNet input check (reading only). (3) CLIP null generator (make_input, K32) and CLIP act K8. (4) CLIP floor quick + CLIP check.
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
C=distill/fanout/cache
for t in input_pink_n256_s1 r18_self_n256_k32_s1 r18_r2act_n256_s1 r18_r2color_n256_s1 r18_r2ent_n256_s1 r18_r2k8_n256_s1; do
  $PY -I distill/fanout/board.py --net r18 --rule noisecut --f 0.95,0.85 --split held --who P0 --note "r2 P0: TUNING SIGNAL (held split), input $t" --input $C/$t.pt 2>&1 | grep '^{'
done
$PY -I distill/fanout/check_input.py --net r18 --inputs $C/r18_r2act_n256_s1.pt,$C/r18_r2color_n256_s1.pt,$C/r18_r2ent_n256_s1.pt,$C/r18_r2k8_n256_s1.pt 2>&1 | tail -30
$PY -I distill/fanout/make_input.py --net clip --kind null 2>&1 | grep -v Warn | tail -1
$PY -I distill/fanout/r2_p0_mkinput.py --net clip --tag r2actk8 --target act --K 8 2>&1 | grep -v Warn | tail -2
for t in clip_null_n256_k32_s1 clip_r2actk8_n256_s1; do
  $PY -I distill/fanout/board.py --net clip --rule noisecut --f 0.95,0.85 --quick --who P0 --input $C/$t.pt 2>&1 | grep '^{'
done
$PY -I distill/fanout/check_input.py --net clip --inputs $C/input_pink_n256_s1.pt,$C/clip_self_n256_k32_s1.pt,$C/clip_null_n256_k32_s1.pt,$C/clip_r2actk8_n256_s1.pt 2>&1 | tail -30
