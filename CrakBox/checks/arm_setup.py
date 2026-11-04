"""The near-view test, fixed in LOG before it runs. The stage 4 nodes that failed their positive pole get a fresh read. Code pairs them by first-read center and tosses a coin in each pair.
Both arms: the same model, step limit and feedback. One difference: the tool arm has the near view. No workflow slot is spent, since the suspect is a tool.
Usage: arm_setup.py RUN. Prints the arms and the task files."""
import importlib.util
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)
from mapper.state import Store  # noqa: E402

SEED = 20261104
run = sys.argv[1]
net = cmd.get_net(run)
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
report = json.load(open(os.path.join(ROOT, "checks", f"report_{run}.json")))["nodes"]
failed = []
for (node,) in s.db.execute("SELECT id FROM nodes WHERE state='open' ORDER BY id").fetchall():
    verdict = next(d["verdict"] for k, d in reversed(s.history(node)) if k == "judging->open")
    if "pole:pos" in verdict["failed"]:
        failed.append((report[node]["pole:pos"]["center_norm"], node))
failed.sort()
rng = random.Random(SEED)
tool, control = [], []
for j in range(0, len(failed) - 1, 2):
    a, b = failed[j][1], failed[j + 1][1]
    if rng.random() < .5:
        a, b = b, a
    tool.append(a)
    control.append(b)
if len(failed) % 2:
    control.append(failed[-1][1])
folder = cmd.run_dir(run)
with open(os.path.join(folder, "arms.json"), "w") as f:
    json.dump(dict(tool=tool, control=control, seed=SEED), f, indent=1)
with open(os.path.join(folder, "tool_bars.json"), "w") as f:
    json.dump({n: ["near"] for n in control}, f, indent=1)

# every card so far, marked with what its blind judge said, so that a reader does not lean on words that failed
out = os.path.join(folder, "cards")
for node, card in s.all_cards().items():
    with open(os.path.join(out, node.replace(":", "_") + ".json"), "w") as f:
        json.dump({**{k: v for k, v in card.items() if not k.startswith("_")}, "judged": cmd.judged_note(s, node)}, f, indent=1)

layers = net.layers()
NOTE = """

This node has an earlier card, and it failed its blind judge. Its file is {root}/runs/{run}/cards/NODE.json, with the colon written as an underscore. Read it, and read the node afresh. The judge could not separate the node's firings from pictures of similar things using those words: the sheet holds strong places of the node's most similar nodes, where this node stays quiet. So your words must say what separates them. Do not only reword the card. Tools that you may not have had before: photo, which saves a real strong photo for you to edit."""
HELP = net.evidence_help()
CLAUSE = ", near NODE (shows where this node fires, above strong places of its most similar nodes where it stays quiet. These are what your words must separate it from. Prints a PNG path, then open it with Read)"
assert CLAUSE in HELP, "the near tool's help text changed"
plan = []
for arm, nodes in (("T", tool), ("C", control)):
    groups = {}
    for node in nodes:
        groups.setdefault(layers.index(net.layer_of(node)) % 3, []).append(node)
    for colour, part in sorted(groups.items()):
        author = f"w1{arm}c{colour}"
        for node in part:
            s.retool(node, "the judge failed its edge score, and readers had no view of the near items")
            s.move(node, "reading")
        a = type("A", (), {})()
        a.net, a.args = run, ["reader", author] + part
        cmd.cmd_task(a, s)
        path = os.path.join(folder, "tasks", f"reader_{author}.md")
        text = open(path).read()
        if arm == "C":
            assert CLAUSE in text
            text = text.replace(CLAUSE, "")
        text = text.replace("Your final message, at most 80 words", NOTE.format(root=ROOT, run=run).strip() + "\n\nYour final message, at most 80 words")
        open(path, "w").write(text)
        plan.append(dict(arm=arm, author=author, nodes=part, task=path))
print(json.dumps(dict(tool=tool, control=control, plan=plan), indent=1))
s.close()
