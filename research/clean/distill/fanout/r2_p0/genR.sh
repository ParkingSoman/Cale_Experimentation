#!/bin/sh
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
for args in "--tag r2act --target act" "--tag r2color --color" "--tag r2ent --target ent" "--tag r2k8 --K 8"; do
  $PY -I distill/fanout/r2_p0_mkinput.py $args 2>&1 | grep -v Warn | tail -2
done
$PY -I distill/fanout/make_input.py --net clip --kind null 2>&1 | grep -v Warn | tail -1
$PY -I distill/fanout/r2_p0_mkinput.py --net clip --tag r2actk8 --target act --K 8 2>&1 | grep -v Warn | tail -2
