#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
until [ -f results/queue3.done ]; do sleep 30; done
rm -rf models/nestq4e8; echo "$(date) deleted models/nestq4e8" >> results/queue4.log
$P scripts/score.py models/qg4-id qg4-id --nest --per-source 10 > results/qg4-id.log 2>&1
$P scripts/bench.py models/qg4-id qg4-id --nest --tasks mmlupro,mmstar > results/bench-qg4-id.log 2>&1
$P -m nest.build models/gemma-4-12B-it-qat-bf16 models/qg4 4 results/perm-wanda.npz - grid 32 >> results/queue4.log 2>&1
$P scripts/score.py models/qg4 qg4 --nest --level results/levels/mix1.json --per-source 10 > results/qg4.log 2>&1
$P scripts/bench.py models/qg4 qg4 --nest --level results/levels/mix1.json --tasks mmlupro,mmstar > results/bench-qg4.log 2>&1
echo done > results/queue4.done
