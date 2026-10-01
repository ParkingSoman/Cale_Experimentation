#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
T=mmlupro,mmstar
$P scripts/score.py models/nest4e8 mix1-n4e8 --nest --level results/levels/mix1-4bit.json --per-source 10 > results/mix1-n4e8.log 2>&1
$P scripts/bench.py models/nest4e8 n4e8 --nest --level results/levels/mix1.json --only full,u90,u75 --tasks $T > results/bench-n4e8.log 2>&1
$P scripts/bench.py models/nest8 n8 --nest --level results/levels/mix1.json --only full,u90,u75 --tasks $T > results/bench-n8.log 2>&1
$P scripts/bench.py models/gemma-4-e4b-it-4bit e4b --tasks $T > results/bench-e4b.log 2>&1
$P scripts/bench.py models/gemma-4-e2b-it-4bit e2b --tasks $T > results/bench-e2b.log 2>&1
$P scripts/bench.py models/nest4 n4 --nest --tasks $T > results/bench-n4.log 2>&1
echo done > results/queue2.done
