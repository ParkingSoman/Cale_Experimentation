#!/bin/sh
# P1-P3 r2: run several quick board rows in ONE torchq lock. Usage: sh batch.sh "FLAGS" rule1 rule2 ...
# FLAGS e.g. "--f 0.95,0.85,0.5 --quick"
cd /Users/shonusengupta/Cale_Experimentation/research/clean || exit 1
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
FLAGS="$1"; shift
for r in "$@"; do
  $PY -I distill/fanout/board.py --net clip --rule "$r" $FLAGS --who P1-P3 --input distill/fanout/cache/input_pink_n256_s1.pt 2>&1 | grep -v Warn
done
