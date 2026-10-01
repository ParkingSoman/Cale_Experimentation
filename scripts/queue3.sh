#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
until [ -f results/queue2.done ] && grep -q "^done" results/download-qat.log; do sleep 30; done
rm -rf models/nest8
echo "$(date) deleted models/nest8" >> results/queue3.log
$P scripts/local_repair.py > results/local_repair.log 2>&1
$P -m nest.build models/gemma-4-12B-it-qat-bf16 models/nestq4e8 4 results/perm-wanda.npz '{"language_model.model.embed_tokens": 8}' >> results/queue3.log 2>&1
$P scripts/score.py models/nestq4e8 mix1-q4e8 --nest --level results/levels/mix1.json --per-source 10 > results/mix1-q4e8.log 2>&1
$P scripts/bench.py models/nestq4e8 q4e8 --nest --level results/levels/mix1.json --only full,u90,u75 --tasks mmlupro,mmstar > results/bench-q4e8.log 2>&1
echo done > results/queue3.done
