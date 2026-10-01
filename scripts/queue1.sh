#!/bin/zsh
# Sequential runs, one model in memory at a time.
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
$P scripts/score.py models/nest8 mix1-n8 --nest --level results/levels/mix1.json --per-source 10 > results/mix1-n8.log 2>&1
$P scripts/score.py models/nest4 mix1-n4 --nest --level results/levels/mix1-4bit.json --per-source 10 > results/mix1-n4.log 2>&1
$P scripts/score.py models/gemma-4-e4b-it-4bit e4b-4bit --per-source 10 > results/e4b.log 2>&1
$P scripts/score.py models/gemma-4-e2b-it-4bit e2b-4bit --per-source 10 > results/e2b.log 2>&1
echo QUEUE1 DONE > results/queue1.done
