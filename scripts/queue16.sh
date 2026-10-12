#!/bin/zsh
# Step 2a: quality at any-size widths on the fully resident .nest file (joined down_proj).
# New points first (80..50), then 100 as a check against nest-L (53.4 / 60.0). Accuracy only: speed is measured alone later.
cd /Users/shonusengupta/Cale_Experimentation
export PYTHONPATH=.
.venv/bin/python scripts/bench.py models/g12-nest/model.nest any-L --tiers 0.8,0.7,0.6,0.5,1.0 --tasks mmlupro1k,mmstar300 >> results/bench-any.log 2>&1
echo done > results/queue16.done
