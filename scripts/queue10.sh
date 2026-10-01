#!/bin/zsh
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
P=.venv/bin/python
L=results/queue10.log
$P -m nest.build models/gemma-4-12B-it-qat-bf16 models/q4e8 4 results/perm-wanda.npz '{"language_model.model.embed_tokens": 8}' >> $L 2>&1
$P scripts/bench.py models/q4e8 q4e8-L --nest --level results/levels/key2.json --tasks mmlupro1k,mmstar300 > results/bench-q4e8-L.log 2>&1
$P scripts/bench.py models/qg4 qg4-L --nest --level results/levels/key2.json --tasks mmlupro1k,mmstar300 > results/bench-qg4-L.log 2>&1
$P scripts/bench.py models/gemma-4-e4b-it-4bit e4b-L --tasks mmlupro1k,mmstar300,gsm8k,ifeval > results/bench-e4b-L.log 2>&1
$P scripts/bench.py models/gemma-4-e2b-it-4bit e2b-L --tasks mmlupro1k,mmstar300 > results/bench-e2b-L.log 2>&1
$P scripts/bench.py models/qg4 qg4-gen --nest --tasks gsm8k,ifeval --limit 50 > results/bench-qg4-gen.log 2>&1
$P scripts/spec_test.py 2 --exact > results/spec-exact.log 2>&1
echo done > results/queue10.done
