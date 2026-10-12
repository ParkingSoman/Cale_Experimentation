#!/bin/zsh
# Timing session (Steps 4 and 6). Starts only after queue14 is done: no timing while benchmarks run.
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
L=results/timing.log
while [ ! -f results/queue14.done ]; do sleep 30; done
$P scripts/nocache_check.py models/g12-nest/model.nest >> $L 2>&1
$P scripts/stream_speed.py nest --reps 3 --gen 96 >> $L 2>&1
$P scripts/stream_speed.py nest-ram8 --ram 8.3 --tiers 0.85,1.0 --reps 3 --gen 32 >> $L 2>&1
$P scripts/speed.py models/gemma-4-e4b-it-4bit e4b-now --reps 5 >> $L 2>&1
$P scripts/speed.py models/qg4 qg4-now --reps 5 >> $L 2>&1
$P scripts/stack_cost.py --reps 3 >> $L 2>&1
$P scripts/mix_direct.py >> $L 2>&1
echo done > results/queue15.done
