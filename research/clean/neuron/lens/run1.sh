set -x
../../curves/.venv/bin/python build.py --mode seq --tag seq
../../curves/.venv/bin/python build.py --mode mean --tag mean
../../curves/.venv/bin/python build.py --mode lens --ridge_lens 1e-3 --tag lens_r1e-3
../../curves/.venv/bin/python build.py --mode lens --ridge_lens 1e-4 --tag lens_r1e-4
../../curves/.venv/bin/python build.py --mode lens --ridge_lens 1e-2 --tag lens_r1e-2
../../curves/.venv/bin/python build.py --mode lens --ridge_lens 1e-1 --tag lens_r1e-1
