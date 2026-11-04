"""The code pass for a whole run, in one process, so the network loads once. Then code routes each node. No tokens. Usage: make_sheets.py RUN SCOPE_FILE [--read]"""
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)
from mapper import sheets as sheetlib  # noqa: E402
from mapper import workflow as wf  # noqa: E402
from mapper.state import Store  # noqa: E402

run, scope_file = sys.argv[1], sys.argv[2]
net = cmd.get_net(run)
scope = json.load(open(scope_file))
s = Store(os.path.join(cmd.run_dir(run), "state.db"))
s.set_scope(scope, [(n, net.layer_of(n)) for n in scope["nodes"]])
lib = cmd.library()
folder = os.path.join(cmd.run_dir(run), "sheets")
counts = {}
start = time.time()
for i, node in enumerate(scope["nodes"]):
    if s.state(node) != "queued":
        continue
    _, sheet = sheetlib.write(net, node, folder)
    w = wf.route(sheet["features"], lib, s.tried(node))
    s.route(node, w.id)
    s.move(node, "reading")
    counts[w.id] = counts.get(w.id, 0) + 1
    print(i, node, w.id, round(time.time() - start), flush=True)
print(json.dumps(counts))
s.close()
