"""L1 test 4 sweep: law basis, stream width fixed at 737 (nodes at f). See rules/l1_stream.py. Photo use: none."""
from rules import l1_stream as L

vit = L.make(mode="law", kfix=737)
