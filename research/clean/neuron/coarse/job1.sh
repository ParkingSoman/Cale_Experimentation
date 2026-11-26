#!/bin/bash
# N11 job 1 on the cluster: smoke (spread photos) with a gate, then B k=32 seed 0, its reading, and the worth of A and B.
set -e
cd /bigtemp/fkm5yp/n6/clean && tar xzf /bigtemp/fkm5yp/n11in/code18d.tar.gz
cd neuron/coarse
python train18c.py --k 32 --smoke --jpeg_dir /bigtemp/fkm5yp/n6 2>&1 | grep -v Warning
python - <<'PY'
import json, sys
r = json.load(open("../results/coarse/r18/B32aug_s0_smoke.json"))
log = open("../results/coarse/r18/B32aug_s0_smoke.log").read()
acc = float(log.split("no mixup): ")[1].split()[0])
print("GATE smoke held KL", r["held_kl"], "A aug top-1", acc)
sys.exit(0 if (r["held_kl"] < 4.5 and acc > 0.4) else 3)
PY
python train18c.py --k 32 --seed 0 --jpeg_dir /bigtemp/fkm5yp/n6 2>&1 | grep -v Warning
python capture18.py B32aug_s0
python worth18.py A
python worth18.py B32aug_s0
python edge18.py B32aug_s0
cd ../results/coarse/r18 && tar czf /bigtemp/fkm5yp/n11in/job1_out.tar.gz B32aug_s0*.json B32aug_s0*.log B32aug_s0.pt read_B32aug_s0.npz worth_A.npz worth_B32aug_s0.npz edges_B32aug_s0.npz
echo JOB1_DONE
