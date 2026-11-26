#!/bin/sh
# P0 round 2: the CLIP null input (make_input K32, shuffled-weights network makes the images). ResNet variant nulls dropped: no variant beat pink on the held split.
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
$PY -I distill/fanout/make_input.py --net clip --kind null 2>&1 | grep -v Warn | tail -1
