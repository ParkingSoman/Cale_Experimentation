#!/bin/zsh
# Step 1 gate: ocean / 8-bit rung / 4-bit control, layer by layer.
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
$P scripts/bench_layerwise.py ladder-gate --tasks mmlupro1k,mmstar300 --tokens 50000 > results/ladder-gate.log 2>&1
echo done > results/queue12.done
