#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
until [ -f results/queue6.done ]; do sleep 30; done
$P scripts/check_letters.py models/qg4 --nest > results/letters-qg4.log 2>&1
$P scripts/check_letters.py models/gemma-4-e4b-it-4bit > results/letters-e4b.log 2>&1
$P scripts/check_letters.py models/gemma-4-e2b-it-4bit > results/letters-e2b.log 2>&1
$P scripts/make_reference.py results/ref-qat-bf16 - models/gemma-4-12B-it-qat-bf16 > results/ref-qat.log 2>&1
echo done > results/queue7.done
