#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
$P scripts/bench.py models/qg4 qg4-gen --nest --tasks ifeval --limit 30 > results/bench-qg4-ifeval.log 2>&1
$P scripts/bench.py models/gemma-4-e4b-it-4bit e4b-gen --tasks ifeval --limit 30 > results/bench-e4b-ifeval.log 2>&1
$P scripts/bench.py models/qg4 qg4-L --nest --level results/levels/key3.json --tasks mmlupro1k > results/bench-qg4-L2.log 2>&1
echo done > results/queue11.done
