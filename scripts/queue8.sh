#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
until [ -f results/queue7.done ]; do sleep 30; done
$P scripts/spec_test.py 3 > results/spec.log 2>&1
echo done > results/queue8.done
