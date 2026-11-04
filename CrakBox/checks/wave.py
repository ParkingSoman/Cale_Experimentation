"""Plan one wave of a run. Code assigns nodes to reader agents, so that both ends of a link have different authors, and writes each task file. It dumps every card so far, for later readers to read.

Colour: a layer's position mod 3. Links go 1 or 2 layers down, so linked layers never share a colour. Each agent holds one colour.
Model: dial and relay nodes go to the cheaper model. Their tests are simple and the smoke run read none. Others go to the stronger one, which leaned ahead in the smoke run on 4 nodes each. A decision by delegation, and the log reports both.
Usage: wave.py RUN NAME PER_AGENT LAYER..."""
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

CHEAP = {"dial", "relay"}
run, name, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
wanted = sys.argv[4:]
net = cmd.get_net(run)
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
layers = net.layers()
out = os.path.join(cmd.run_dir(run), "cards")
os.makedirs(out, exist_ok=True)
for node, card in s.all_cards().items():
    with open(os.path.join(out, node.replace(":", "_") + ".json"), "w") as f:
        json.dump({**{k: v for k, v in card.items() if not k.startswith("_")}, "judged": cmd.judged_note(s, node)}, f, indent=1)
groups = {}
for row in s.db.execute("SELECT id,layer,workflow FROM nodes WHERE state='reading' ORDER BY layer,id"):
    node, layer, workflow = row
    if layer not in wanted:
        continue
    colour = layers.index(layer) % 3
    model = "sonnet" if workflow in CHEAP else "opus"
    groups.setdefault((colour, model), []).append(node)
plan = []


class A:
    pass


for (colour, model), nodes in sorted(groups.items()):
    for start in range(0, len(nodes), per):
        part = nodes[start:start + per]
        author = f"{name}c{colour}{model[0]}{start // per}"
        a = A()
        a.net, a.args = run, ["reader", author] + part
        cmd.cmd_task(a, s)
        plan.append(dict(author=author, model=model, nodes=len(part), task=os.path.join(cmd.run_dir(run), "tasks", f"reader_{author}.md")))
print(json.dumps(plan, indent=1))
os.makedirs(os.path.join(cmd.run_dir(run), "work"), exist_ok=True)
s.close()
