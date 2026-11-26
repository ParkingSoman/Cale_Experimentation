#!/bin/sh
# WRITE r3 promotion: stream(act)+MLP(weight) cut, heads full: off / plain fold / kslice / kslice roll null / open; .95,.85 full.
P=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
for r in r3_write_sm_kslice r3_write_sm_ksroll r3_write_sm_off r3_write_sm_fold; do
  $P -I distill/fanout/board.py --net clip --rule $r --f 0.95,0.85 --who WRITE --input distill/fanout/cache/input_pink_n256_s1.pt --note "r3 WRITE: stream+MLP cut, heads full; kslice / roll null / off / plain fold"
done
