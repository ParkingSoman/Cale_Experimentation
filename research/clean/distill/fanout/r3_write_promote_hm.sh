#!/bin/sh
# WRITE r3 promotion: heads + MLP cut, stream full; off / fold / roll null, full 1,000 test photos.
P=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
for r in r3_write_hm_off r3_write_hm_fold r3_write_hm_roll; do
  $P -I distill/fanout/board.py --net clip --rule $r --f 0.95,0.85 --who WRITE --input distill/fanout/cache/input_pink_n256_s1.pt --note "r3 WRITE: heads(act)+MLP(weight) cut, stream full; MLP fold off/on/roll-null"
done
