"""Build every blind packet a run's waiting nodes need: poles or a dial's ordering, parent, reader, routes, and the mixed-in wrong items. One process, so the network loads once.
Usage: build_packets.py RUN [POOL_RUN]. A node that cannot have a test keeps that test missing, and the reason is printed. It says nothing about the network."""
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

run = sys.argv[1]
pool = sys.argv[2] if len(sys.argv) > 2 else ""
net = cmd.get_net(run)
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
cards = s.all_cards()
if pool:
    other = Store(os.path.join(cmd.run_dir(pool), "state.db"))
    cards = {**other.all_cards(), **cards}
    other.close()
b = cmd.bench(run)
waiting = [r[0] for r in s.db.execute("SELECT id FROM nodes WHERE state IN ('attacked','judging') ORDER BY id")]
made, problems = 0, []


def attempt(label, fn):
    """Returns True when this call made a new item. So a second run of this script adds no more wrong items for items it already has."""
    global made
    try:
        before = len(b.key)
        fn()
        made += len(b.key) - before
        return len(b.key) > before
    except (judgelib.ShortPool, judgelib.Leak) as e:
        problems.append(f"{label}: {e}")
    except SystemExit as e:
        problems.append(f"{label}: {e}")
    return False


for node in waiting:
    if s.state(node) == "attacked":
        s.move(node, "judging")
    card = s.card(node)
    w = cmd.find_workflow(s.workflow(node))
    if w is not None and w.replaces_poles:
        if attempt(f"{node} ordering", lambda: b.build_ordering(net, node, cards)) and b.want_wrong_item():
            attempt(f"{node} ordering wrong", lambda: b.build_ordering(net, node, cards, wrong=True))
    else:
        for pole in net.judged_poles(node):
            if attempt(f"{node} {pole}", lambda p=pole: b.build_pole(net, node, p, cards)) and b.want_wrong_item():
                attempt(f"{node} {pole} wrong", lambda p=pole: b.build_pole(net, node, p, cards, wrong=True))
        # a pole that is not judged stays reported, on about one node in five
        if "neg" in net.poles(node) and "neg" not in net.judged_poles(node) and sum(map(ord, node)) % 5 == 0:
            attempt(f"{node} neg sample", lambda: b.build_pole(net, node, "neg", cards))
    for kind in ("parent", "reader"):
        if kind == "parent" and not net.has_inputs(node):
            continue
        # the same truth for a real item and its wrong item
        if attempt(f"{node} {kind}", lambda k=kind: b.build_link(net, k, node, cmd.link_truth(net, s, node, k, cards), cards)) and b.want_wrong_item():
            attempt(f"{node} {kind} wrong", lambda k=kind: b.build_link(net, k, node, cmd.link_truth(net, s, node, k, cards), cards, wrong=True))
    for r in card.get("routes", []):
        attempt(f"{node} route {r['id']}", lambda r=r: b.build_pole(net, node, "pos", cards, route=set(r["inputs"]), route_id=r["id"]))
print(json.dumps(dict(nodes=len(waiting), items_made=made, problems=problems), indent=1))
s.close()
