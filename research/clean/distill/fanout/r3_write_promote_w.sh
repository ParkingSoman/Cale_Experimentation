#!/bin/sh
# WRITE r3 promotion: whole CLIP cut (stream+heads activity, MLP weight): off / kslice fold / kslice roll null; .95,.85 full.
P=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
for r in r3_write_w_kslice r3_write_w_ksroll r3_write_w_off; do
  $P -I distill/fanout/board.py --net clip --rule $r --f 0.95,0.85 --who WRITE --input distill/fanout/cache/input_pink_n256_s1.pt --note "r3 WRITE: whole cut, MLP fold with kept rows on the slice (kslice) / its roll null / off"
done
