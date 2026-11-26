#!/bin/sh
# WRITE r3 promotion: CLIP MLP-only weight order at .7/.5: off (mix0), fold (r2 wfold), same-form roll null; port check m_fold .85.
P=/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python
for r in r2_p5_mix0_mlp r2_p4_wfold_mlp r3_write_m_roll; do
  $P -I distill/fanout/board.py --net clip --rule $r --f 0.7,0.5 --who WRITE --input distill/fanout/cache/input_pink_n256_s1.pt --note "r3 WRITE: MLP-only weight order, fold off/on/roll-null at .7/.5"
done
$P -I distill/fanout/board.py --net clip --rule r3_write_m_fold --f 0.85 --who WRITE --input distill/fanout/cache/input_pink_n256_s1.pt --note "r3 WRITE: port check of r3_write_lib vs r2_p4_wfold_mlp full .85 (.776)"
