#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
until [ -f results/queue8.done ]; do sleep 30; done
$P scripts/bench.py models/gemma-4-e4b-it-4bit e4b-v2 --tasks mmlupro,mmstar > results/bench-e4b-v2.log 2>&1
$P scripts/bench.py models/gemma-4-e2b-it-4bit e2b-v2 --tasks mmlupro,mmstar > results/bench-e2b-v2.log 2>&1
$P scripts/bench.py models/qg4 qg4-v2 --nest --level results/levels/key1.json --tasks mmlupro,mmstar > results/bench-qg4-v2.log 2>&1
echo done > results/queue9.done
