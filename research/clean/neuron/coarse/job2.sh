#!/bin/bash
# N11 job 2 on the cluster: B at k = 128, 512 (seed 0) and seed 1 at k = 32, 128, each read (capture, worth, edges).
set -e
cd /bigtemp/fkm5yp/n6/clean/neuron/coarse
for ks in "128 0" "512 0" "32 1" "128 1"; do
  read k s <<< "$ks"
  python train18c.py --k $k --seed $s --jpeg_dir /bigtemp/fkm5yp/n6 2>&1 | grep -v Warning
  python capture18.py B${k}aug_s$s
  python worth18.py B${k}aug_s$s
  python edge18.py B${k}aug_s$s
done
python capture18.py A
cd ../results/coarse/r18
tar czf /bigtemp/fkm5yp/n11in/job2_out.tar.gz B*aug_s*.json B*aug_s*.log B*aug_s*.pt worth_B*.npz edges_B*.npz read_B32aug_s1.npz read_B128aug_s0.npz read_B128aug_s1.npz
cp read_B512aug_s0.npz read_A.npz /bigtemp/fkm5yp/n11in/
echo JOB2_DONE
