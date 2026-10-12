#!/bin/zsh
# Rest of queue13 (stopped by the tool's 2 h limit): w85+draft GSM8K, then IFEval for all tiers.
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
N=models/g12-nest/model.nest
L=results/bench-nest.log
$P scripts/bench.py $N nest-gen --tiers 0.85:d --tasks gsm8k --limit 50 >> $L 2>&1
$P scripts/bench.py $N nest-gen --tiers 0.9,0.85,1.0:d,0.85:d --tasks ifeval --limit 30 >> $L 2>&1
echo done > results/queue14.done
