#!/bin/bash
# N11 job 3: re-capture A and every B with peak positions, then each block-0 neuron's stage-3 feeders (path18).
set -e
cd /bigtemp/fkm5yp/n6/clean/neuron/coarse
for t in A B32aug_s0 B32aug_s1 B128aug_s0 B128aug_s1 B512aug_s0; do
  python capture18.py $t
  python path18.py $t
done
cd ../results/coarse/r18
tar czf /bigtemp/fkm5yp/n11in/job3_small.tar.gz path_*.npz read_B32aug_s0.npz read_B32aug_s1.npz read_B128aug_s0.npz read_B128aug_s1.npz
cp read_A.npz /bigtemp/fkm5yp/n11in/read_A3.npz
cp read_B512aug_s0.npz /bigtemp/fkm5yp/n11in/read_B512_3.npz
echo JOB3_DONE
