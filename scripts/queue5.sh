#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
until [ -f results/queue4.done ]; do sleep 30; done
$P scripts/speed.py models/qg4 qg4 --nest --level results/levels/speed1.json > results/speed-qg4.log 2>&1
$P scripts/speed.py models/gemma-4-e4b-it-4bit e4b > results/speed-e4b.log 2>&1
$P scripts/speed.py models/gemma-4-e2b-it-4bit e2b > results/speed-e2b.log 2>&1
$P scripts/bench.py models/qg4 qg4 --nest --level results/levels/width2.json --tasks mmlupro,mmstar > results/bench-qg4-w2.log 2>&1
echo done > results/queue5.done
