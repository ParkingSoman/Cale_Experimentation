#!/bin/bash
# N11 job 5: train the one-level reference (block 0 at 128, block 1 = A), then the multi-level comparison.
set -e
cd /bigtemp/fkm5yp/n6/clean/neuron/coarse
python train18c.py --k 128 --seed 0 --blocks 0 --jpeg_dir /bigtemp/fkm5yp/n6 2>&1 | grep -v Warning
python multilevel18.py
cd ../results/coarse/r18 && tar czf /bigtemp/fkm5yp/n11in/job5_out.tar.gz multilevel.json multilevel_rows.npz rule_pos_*.npz B128aug_s0_b0.json B128aug_s0_b0.log
echo JOB5_DONE
