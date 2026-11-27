#!/bin/bash
# Loop 1 board rows (PAGE v1). Full 1,000 test photos, scoring only. Sequential through torchq.
cd /Users/shonusengupta/Cale_Experimentation/research/clean
PY=curves/.venv/bin/python
Q="$PY -I distill/fanout/torchq.py --who TREE --long --timeout 1800 --"
R18IN=distill/fanout/cache/r18_self_n256_k32_s1.pt
CLIN=distill/fanout/cache/clip_self_n256_k32_s1.pt
$Q $PY -I distill/fanout/board.py --net r18 --rule tree_v1 --f 0.95,0.85,0.5,0.05 --input $R18IN --who TREE --note "loop1 page v1"
for r in tree_v1_fire tree_v1_write tree_v1_rand; do
  $Q $PY -I distill/fanout/board.py --net r18 --rule $r --f 0.95,0.85,0.5 --input $R18IN --who TREE --note "loop1 page v1 loo"
done
$Q $PY -I distill/fanout/board.py --net clip --rule tree_v1 --f 0.95,0.85,0.5,0.25,0.05 --input $CLIN --who TREE --note "loop1 page v1"
for r in tree_v1_fire tree_v1_write tree_v1_rand; do
  $Q $PY -I distill/fanout/board.py --net clip --rule $r --f 0.95,0.85,0.5 --input $CLIN --who TREE --note "loop1 page v1 loo"
done
echo LOOP1 DONE
