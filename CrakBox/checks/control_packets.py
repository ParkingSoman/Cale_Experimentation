"""Step 6: blind packets for the ConvNeXt control. Both poles of each claimed card, and one wrong item per card, on a pole taken in turn. No critic and no link tests here, since this run measures words against the judge."""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)
from mapper import judge as judgelib  # noqa: E402
from mapper.state import Store  # noqa: E402

run = "convnext_tiny"
net = cmd.get_net(run)
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
cards = s.all_cards()
b = cmd.bench(run)
problems = []
for i, node in enumerate(sorted(cards)):
    for pole in net.poles(node):
        try:
            b.build_pole(net, node, pole, cards)
        except (judgelib.ShortPool, judgelib.Leak) as e:
            problems.append(str(e))
    try:
        b.build_pole(net, node, net.poles(node)[i % 2], cards, wrong=True)
    except (judgelib.ShortPool, judgelib.Leak) as e:
        problems.append(str(e))
print(json.dumps(dict(cards=len(cards), items=len(b.key), problems=problems), indent=1))
s.close()
