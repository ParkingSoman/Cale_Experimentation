set -x
../../curves/.venv/bin/python build.py --mode seq --ot_writes --tag seq_otw
../../curves/.venv/bin/python build.py --mode lens --ridge_lens 1e-4 --ot_lens --ot_writes --tag lens_ot
../../curves/.venv/bin/python build.py --mode lens --ridge_lens 1e-4 --ot_lens --tag lens_otl
../../curves/.venv/bin/python build.py --mode base --ot_writes --tag base_otw
