#!/bin/zsh
# Step 5 quality grid on the one-file 4-bit build: widths 100/90/85, with and without drafter.
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
N=models/g12-nest/model.nest
L=results/bench-nest.log
$P scripts/bench.py $N nest-L --tiers 1.0,0.9 --tasks mmlupro1k,mmstar300 >> $L 2>&1
$P scripts/bench.py $N nest-L --tiers 0.85 --tasks mmstar300 >> $L 2>&1
$P scripts/bench.py $N nest-gen --tiers 0.9,0.85,1.0:d,0.85:d --tasks gsm8k --limit 50 >> $L 2>&1
$P scripts/bench.py $N nest-gen --tiers 0.9,0.85,1.0:d,0.85:d --tasks ifeval --limit 30 >> $L 2>&1
echo done > results/queue13.done
