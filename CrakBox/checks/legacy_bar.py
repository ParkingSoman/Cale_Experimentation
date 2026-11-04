"""Step 3: the old cards under the new rules, in a run of their own (resnet18.legacy).
  legacy_bar.py import     load the cards, count what can reach a judge at all
  legacy_bar.py build      a fixed random sample: 20 nodes, both poles, real and wrong items. Prints judge batches, with no node names
  legacy_bar.py report     after judge-collect: pass marks from wrong items only, then real items against them
Old cards faced no critic, so none can be understood. This measures the words and the tests."""
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from mapper import card as cardlib  # noqa: E402
from mapper import judge as judgelib  # noqa: E402
from mapper.state import Store  # noqa: E402

RUN = "resnet18.legacy"
OLD = os.path.join(ROOT, "..", "research", "clean", "curves", "map_agents", "resnet18", "cards2")
SIZE = 20


def tools():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_cards():
    cards = {}
    for path in sorted(glob.glob(os.path.join(OLD, "*.jsonl"))):
        for line in open(path):
            if line.strip():
                d = json.loads(line)
                d["_author"] = os.path.basename(path)[:-6]
                cards[d["node"]] = d
    return cards


def do_import():
    from networks import get
    cmd = tools()
    net = get("resnet18")
    s = Store(os.path.join(cmd.run_dir(RUN), "state.db"))
    cards = load_cards()
    b = cmd.bench(RUN)
    patterns = net.leak_patterns()
    leak, bad, parents, independent, carded = [], [], 0, 0, 0
    for node, c in cards.items():
        try:
            cardlib.check(c, legacy=True, poles=net.poles(node))
        except cardlib.BadCard as e:
            bad.append(str(e))
            continue
        layer = net.layer_of(node)
        s.add_nodes([(node, layer)])
        if s.state(node) == "queued":
            s.route(node, "legacy")
            s.move(node, "reading")
            s.claim(node, c["_author"], {k: v for k, v in c.items() if not k.startswith("_")})
        found = judgelib.leaks(dict(words=dict(a=c.get("distinction"), b=c.get("positive"), d=c.get("negative"))), list(net.layers()) + [node], patterns)
        if found:
            leak.append((node, found[:2]))
        parent = ((c.get("main_parent") or {}).get("node") or "").replace(" (skip)", "")
        if parent:
            parents += 1
            if parent in cards:
                carded += 1
                independent += cards[parent]["_author"] != c["_author"]
    print(json.dumps(dict(cards=len(cards), rejected=bad, words_that_leak=len(leak), leak_examples=leak[:8], parents_named=parents, parents_with_a_card=carded, parents_with_an_independent_card=independent), indent=1))
    print(json.dumps(s.status()["states"]))
    s.close()


def do_build():
    from networks import get
    cmd = tools()
    net = get("resnet18")
    s = Store(os.path.join(cmd.run_dir(RUN), "state.db"))
    cards = s.all_cards()
    b = cmd.bench(RUN)
    patterns = net.leak_patterns()
    clean = []
    for node, c in sorted(cards.items()):
        ok = all(not judgelib.leaks(dict(words=judgelib.words_for(c, p)), list(net.layers()) + [node], patterns) for p in net.poles(node))
        if ok:
            clean.append(node)
    rng = random.Random(3)
    sample = rng.sample(clean, SIZE)
    items, short = [], []
    for node in sample:
        for pole in net.poles(node):
            for wrong in (False, True):
                try:
                    if wrong:
                        # a wrong item's words must be free of leaks too
                        pool = {n: cards[n] for n in clean}
                        pool[node] = cards[node]
                        items.append(b.build_pole(net, node, pole, pool, wrong=True))
                    else:
                        items.append(b.build_pole(net, node, pole, cards))
                except (judgelib.ShortPool, judgelib.Leak) as e:
                    short.append(str(e))
    rng.shuffle(items)
    per = 16
    batches = [items[i:i + per] for i in range(0, len(items), per)]
    out = os.path.join(cmd.run_dir(RUN), "batches.json")
    with open(out, "w") as f:
        json.dump(dict(batches=batches, packets=b.packets, answers=b.packets.replace("packets_", "answers_")), f)
    print(json.dumps(dict(clean_cards=len(clean), sampled=SIZE, items=len(items), batches=len(batches), short=short, batches_file=out), indent=1))
    s.close()


def do_report():
    cmd = tools()
    s = Store(os.path.join(cmd.run_dir(RUN), "state.db"))
    b = cmd.bench(RUN)
    wrong, real, pairs = [], [], []
    for item, k in b.key.items():
        answers = {j: a for j, a in s.answers(item).items() if "ranked" in a}
        if len(answers) < 2 or not k["test"].startswith("pole"):
            continue
        scores = b.judge_scores(item, answers)
        (wrong if k["wrong_card"] else real).append((k, scores))
    cal = judgelib.calibrate([sc for _, sc in wrong])
    marks = cal["marks"]
    votes = [judgelib.vote([judgelib.passes(x, marks) for x in sc]) for _, sc in real]
    wvotes = [judgelib.vote([judgelib.passes(x, marks) for x in sc]) for _, sc in wrong]
    mean = lambda rows, f: float(np.mean([x[f] for _, sc in rows for x in sc]))  # noqa: E731
    fields = ("center", "center_norm", "edge", "far", "fire_rate", "near_rate", "random_rate")
    agree = judgelib.agreement([tuple(sc[:2]) for _, sc in real + wrong], marks)

    def separation(field, sign):
        """How often a real judgment beats a wrong judgment on this score. .5 means no separation."""
        a = np.array([x[field] for _, sc in real for x in sc])
        c = np.array([x[field] for _, sc in wrong for x in sc])
        return float(np.mean((sign * a[:, None] > sign * c[None, :]) + .5 * (a[:, None] == c[None, :])))

    report = dict(calibration=cal, real_items=len(real), wrong_items=len(wrong),
                  real_mean={f: round(mean(real, f), 3) for f in fields}, wrong_mean={f: round(mean(wrong, f), 3) for f in fields},
                  separation=dict(center_norm=round(separation("center_norm", 1), 3), edge=round(separation("edge", 1), 3), far=round(separation("far", 1), 3)),
                  real_pass=sum(v is True for v in votes), real_fail=sum(v is False for v in votes), real_split=sum(v is None for v in votes),
                  wrong_pass=sum(v is True for v in wvotes), wrong_fail=sum(v is False for v in wvotes), wrong_split=sum(v is None for v in wvotes),
                  judge_agreement=round(agree, 3), bar=s.status()["understood"], scope=s.status()["scope"])
    print(json.dumps(report, indent=1))
    with open(os.path.join(ROOT, "checks", "legacy_bar_report_ranked.json"), "w") as f:
        json.dump(report, f, indent=1)
    s.close()


if __name__ == "__main__":
    {"import": do_import, "build": do_build, "report": do_report}[sys.argv[1]]()
