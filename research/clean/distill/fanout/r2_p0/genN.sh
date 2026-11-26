#!/bin/sh
# P0 round 2: the nulls (same recipe, shuffled-weights network makes the images). CLIP null (make_input K32), then the ResNet variant nulls.
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
$PY -I distill/fanout/make_input.py --net clip --kind null 2>&1 | grep -v Warn | tail -1
for args in "--tag r2k8null --K 8" "--tag r2entnull --target ent" "--tag r2colornull --color" "--tag r2actnull --target act"; do
  $PY -I distill/fanout/r2_p0_mkinput.py --kind null $args 2>&1 | grep -v Warn | tail -2
done
