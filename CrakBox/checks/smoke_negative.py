"""Smoke run, second look at the negative pole: the same cards, with the negative side's words alone. Wrong items too, built the same way. Compared with the first form in the log."""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)
from mapper.state import Store  # noqa: E402

run = "resnet18.smoke"
net = cmd.get_net(run)
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
cards = s.all_cards()
b = cmd.bench(run)
nodes = ["s4b1i:68", "s4b1:277", "s4b1:35", "s4b1:400", "s4b1i:153", "s4b1i:393", "s4b0:327", "s4b0:99"]
real, wrong = [], []
for n in nodes:
    real.append(b.build_pole(net, n, "neg", cards))
    wrong.append(b.build_pole(net, n, "neg", cards, wrong=True))
print(json.dumps(dict(real=len(real), wrong=len(wrong))))
s.close()
