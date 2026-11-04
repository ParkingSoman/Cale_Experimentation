#!/usr/bin/env python3
"""The mapper command. The tool hands out jobs and checks what comes back. Each command is one legal move.

 A run
  init NET SCOPE.json                  start a run: nodes, goal, control goal, threshold, firings. All must be named
  status NET                           the bar, the open list, the cost
  next NET STATE --n N                 nodes waiting in a state. Above 10 started, only after a smoke batch passes
 A reader
  sheet NET NODE                       the code pass for one node, no tokens
  route NET NODE                       code picks a workflow
  read NET NODE                        start reading
  note NET AUTHOR TEXT                 append one line to the shared notes. Agents share this file, so it is never written whole
  predict NET NODE AUTHOR TEXT         put a reader's prediction on file. Tools that test (edit, probe, acts) refuse to run before it
  revise NET FILE AUTHOR               an author corrects its own card, before any critic has it
  switch NET NODE WORKFLOW REASON      leave a workflow with a written reason. It counts toward the limit of 3
  ev NET NODE ROLE TOOL ...            one evidence call. It is logged and counted, and the step limit holds
  card NET FILE AUTHOR                 hand in a card. A bad card is rejected
  open NET NODE SUSPECT TEXT           end a node open, with its stuck record
 A critic
  critic-plan NET NODE CRITIC FILE     put the prediction on file before the test
  critic NET NODE CRITIC FILE          hand in the result. It must cite logged calls made after the prediction
 The judge
  judge-build NET NODE [--wrong]       build one blind packet per pole of the node, plus sometimes a wrong item. Add --pole pos to build one
  judge-link NET NODE KIND [--wrong]   build a parent or reader packet
  judge-route NET NODE ROUTE_ID        build a route packet, with the card's other routes filled
  judge-ordering NET NODE [--wrong]    build a dial's packet of drawn items
  judge-tasks NET PER [--limit N]      write a task file per judge for unanswered items. Two judges per item, a third where they split. Prints the files
 The tasks
  task NET ROLE AUTHOR NODE...         write one reader's or critic's task file. A task carries its whole job, since a role's file loads only once per session
  judge-answer NET ITEM JUDGE FILE     store one answer. An author cannot judge its own words, and an answer cannot change
  judge-collect NET                    store every answer that judges wrote to the answers folder. A folder's name is the judge
  judge-calibrate NET                  fix pass marks from wrong cards only
  verdict NET NODE                     decide from settled votes. Only this command can make a node understood"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from mapper import card as cardlib  # noqa: E402
from mapper import claims as claimlib  # noqa: E402
from mapper import judge as judgelib  # noqa: E402
from mapper import sheets as sheetlib  # noqa: E402
from mapper import workflow as wf  # noqa: E402
from mapper.state import IllegalMove, LimitHit, Store  # noqa: E402

CLAIM_EXAMPLES = 1000
TEST_TOOLS = {"edit", "probe", "acts"}


def runs_root():
    return os.environ.get("MAPPER_RUNS", os.path.join(ROOT, "runs"))


def run_dir(net):
    return os.path.join(runs_root(), net)


def token(net):
    return hashlib.sha256(net.encode()).hexdigest()[:10]


def bench(net):
    """Packets sit in a folder whose name does not say which network this is."""
    return judgelib.Bench(run_dir(net), packets=os.path.join(runs_root(), "packets_" + token(net)))


def get_net(name):
    """A run name may hold a label after a dot, so one network can have several runs."""
    from networks import get
    return get(name.split(".")[0])


def library():
    return wf.load(os.environ.get("MAPPER_WORKFLOWS", os.path.join(ROOT, "workflows")))


def find_workflow(workflow_id):
    for w in library():
        if w.id == workflow_id:
            return w
    return None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def marks_of(b):
    path = os.path.join(b.keys, "marks.json")
    if not os.path.exists(path):
        raise SystemExit("no pass marks yet. Run judge-calibrate")
    return load_json(path)


def cmd_init(a, s):
    net = get_net(a.net)
    scope = load_json(a.args[0])
    nodes = scope.get("nodes") or []
    for n in nodes:
        if n not in net.nodes(net.layer_of(n)):
            raise SystemExit(f"{n} is not a node of this network")
    s.set_scope(scope, [(n, net.layer_of(n)) for n in nodes])
    print(f"{len(nodes)} nodes in scope")


def cmd_sheet(a, s):
    path, sheet = sheetlib.write(get_net(a.net), a.args[0], os.path.join(run_dir(a.net), "sheets"))
    print(path)
    print(json.dumps(sheet["features"], indent=1))


def cmd_route(a, s):
    node = a.args[0]
    path = os.path.join(run_dir(a.net), "sheets", sheetlib.file_name(node))
    if not os.path.exists(path):
        raise SystemExit(f"{node} has no sheet yet. Run: mapper.py sheet {a.net} {node}")
    w = wf.route(load_json(path)["features"], library(), s.tried(node))
    if w is None:
        s.move(node, "open", suspect="motif", record="no workflow fits this node: the stuck process starts")
        print("open: no workflow fits")
    else:
        s.route(node, w.id)
        print(w.id)


def cmd_ev(a, s, rest):
    node, role, tool = a.args[0], a.args[1], a.args[2]
    w = find_workflow(s.workflow(node))
    if w is None:
        raise SystemExit(f"{node} is not in a workflow. Route it first")
    if role == "reader" and tool in TEST_TOOLS and s.reader_prediction(node) is None:
        raise SystemExit(f"{node}: put your prediction on file before a test. Run: predict {a.net} {node} YOUR_NAME TEXT")
    used = s.steps_used(node, role)
    if used >= w.step_limit:
        raise SystemExit(f"{node}: {w.step_limit} steps used by the {role}. Take an exit")
    bars = os.path.join(run_dir(a.net), "tool_bars.json")
    if os.path.exists(bars) and tool in load_json(bars).get(node, []):
        raise SystemExit(f"{node}: the tool {tool} is not part of your task")
    command, cwd = get_net(a.net).evidence(tool, a.args[3:] + rest)
    done = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    out = done.stdout + done.stderr
    step = s.record_step(node, role, " ".join(a.args[2:] + rest), out, w.step_limit)
    print(f"step {step} ({used + 1} of {w.step_limit})")
    print(out)


def cmd_card(a, s):
    card = load_json(a.args[0])
    net = get_net(a.net)
    cardlib.check(card, poles=net.poles(card["node"]))
    filed = s.reader_prediction(card["node"])
    if filed is None:
        raise SystemExit(f"{card['node']}: no prediction on file. Run predict before your test, and before this card")
    card["prediction"] = filed["prediction"]
    s.claim(card["node"], a.args[1], card)
    print("claimed")


def cmd_revise(a, s):
    card = load_json(a.args[0])
    cardlib.check(card, poles=get_net(a.net).poles(card["node"]))
    filed = s.reader_prediction(card["node"])
    card["prediction"] = filed["prediction"] if filed else card["prediction"]
    s.revise(card["node"], a.args[1], card)
    print("revised")


def cmd_critic(a, s):
    node, critic = a.args[0], a.args[1]
    result = load_json(a.args[2])
    if not isinstance(result.get("broke"), bool) or not result.get("observed"):
        raise SystemExit("an attack needs 'observed' and a true or false 'broke'")
    s.conclude_attack(node, critic, result.get("steps"), result["observed"], result["broke"])
    print("broke: back to the reader" if result["broke"] else "held")


def start_judging(s, node):
    state = s.state(node)
    if state == "attacked":
        s.move(node, "judging")
    elif state != "judging":
        raise SystemExit(f"{node} is {state}. Only a card that held against its critic goes to the judge")


def cmd_judge_build(a, s):
    node = a.args[0]
    net = get_net(a.net)
    if not a.wrong:
        start_judging(s, node)
    cards = s.all_cards()
    b = bench(a.net)
    poles = [a.pole] if a.pole else net.poles(node)
    made = []
    for pole in poles:
        if a.wrong:
            made.append(b.build_pole(net, node, pole, cards, wrong=True))
            continue
        made.append(b.build_pole(net, node, pole, cards))
        if b.want_wrong_item():
            try:
                made.append(b.build_pole(net, node, pole, cards, wrong=True))
            except judgelib.ShortPool:
                pass  # no other card in this layer yet. The real item stands, and later items carry the wrong share
    print("\n".join(made))


def link_truth(net, s, node, kind, cards):
    """Truth comes from code. For a parent: the card's main parent, which must be a real input. A parent outside the scope never gets words, so the truth is then the node's strongest input inside the scope.
    For a reader: the node's strongest reader that can be described."""
    if kind == "parent":
        parent = (s.card(node).get("main_parent") or {}).get("node")
        if not parent:
            raise SystemExit(f"{node}: its card names no parent")
        parent = net.canonical(parent)
        in_scope = {r[0] for r in s.db.execute("SELECT id FROM nodes")}
        if parent in in_scope or parent in cards:
            return parent
        for name in judgelib.strong_links(net, "parent", node, 20):
            if name in in_scope:
                return name
        raise SystemExit(f"{node}: none of its 20 strongest inputs is in the scope")
    for name, _ in net.readers(node, "read", 20):
        if net.words_of(name) is not None or name in cards:
            return name
    raise SystemExit(f"{node}: none of its strongest readers has a card yet. Read them first")


def cmd_judge_link(a, s):
    node, kind = a.args[0], a.args[1]
    if kind not in ("parent", "reader"):
        raise SystemExit("kind is parent or reader")
    net = get_net(a.net)
    if not a.wrong:
        start_judging(s, node)
    b = bench(a.net)
    cards = s.all_cards()
    for run in [r for r in a.pool.split(",") if r]:
        other = Store(os.path.join(run_dir(run), "state.db"))
        cards = {**other.all_cards(), **cards}
        other.close()
    print(b.build_link(net, kind, node, link_truth(net, s, node, kind, cards), cards, wrong=a.wrong))


def cmd_judge_route(a, s):
    node, route_id = a.args[0], a.args[1]
    start_judging(s, node)
    routes = {r["id"]: r for r in s.card(node).get("routes", [])}
    if route_id not in routes:
        raise SystemExit(f"{node}: its card has no route {route_id}")
    print(bench(a.net).build_pole(get_net(a.net), node, "pos", s.all_cards(), route=set(routes[route_id]["inputs"]), route_id=route_id))


def cmd_judge_ordering(a, s):
    node = a.args[0]
    net = get_net(a.net)
    if not a.wrong:
        start_judging(s, node)
    print(bench(a.net).build_ordering(net, node, s.all_cards(), wrong=a.wrong))


JUDGE_TASK = """You judge claims about parts of a network. This task holds your whole job. Follow it even if other instructions you loaded differ.

You see only words and numbered items. You do not know which part it is, and some words are deliberately wrong. Answer from what you see. A program scores your answer, and nobody reads your reasons. Do not try to find out what network this is. Judge each packet alone.

Work on one item at a time, and finish it before you read the next. Never read two sheets in one step: images can come back in another order. Each sheet is stamped at its top with its item id, so check it. For each item below, read {packets}/ITEM.json. It has "test".
- "pole": read the sheet it names. Rank the {rank} items that fit the words best, the strongest fit first. Name {rank} different numbers. Some items only look like the words: leave those out when you can tell.
- "ordering": read the sheet it names. From the words alone, rank the {order} items most likely to fire, strongest first. Name {order} different numbers.
- "link": no sheet. Pick the one numbered candidate that fits the words best.

Write one file per item to {answers}/{judge}/ITEM.json, with exactly {{"item": "ITEM", "ranked": [numbers]}} or {{"item": "ITEM", "pick": number}}.

Items:
{items}

Your final message says only how many items you answered."""


def cmd_judge_tasks(a, s):
    """Items with no answers go to two judges. A split goes to a third. Judges get fresh names, so no author can be one, and each item goes to a judge only once."""
    import random
    b = bench(a.net)
    per = int(a.args[0])
    path = os.path.join(b.keys, "marks.json")
    marks = load_json(path)["marks"] if os.path.exists(path) else None
    two, one = [], []
    for item in b.key:
        answers = s.answers(item)
        if not answers:
            two.append(item)
        elif len(answers) == 1:
            one.append(item)
        elif len(answers) == 2 and marks is not None and judgelib.vote([judgelib.judge_passed(x, marks) for x in b.judge_scores(item, answers)]) is None:
            one.append(item)
    if a.limit:
        two = two[:a.limit]
    rng = random.Random(len(b.key))
    tag = len(os.listdir(b.packets)) % 100000
    files = []
    for side, items in (("a", two), ("b", two), ("c", one)):
        shuffled = list(items)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), per):
            judge = f"j{tag}_{side}_{start // per}"
            text = JUDGE_TASK.format(packets=b.packets, answers=b.packets.replace("packets_", "answers_"), judge=judge, rank=judgelib.N_RANK, order=judgelib.N_ORDER, items=" ".join(shuffled[start:start + per]))
            task = os.path.join(b.packets, f"task_{judge}.md")
            with open(task, "w") as f:
                f.write(text)
            files.append(task)
    print(json.dumps(dict(two_judges=len(two), third_judge=len(one), tasks=files), indent=1))


READER_TASK = """You read parts of a trained network, and write down what each one does, so that a blind judge can test your words. This task holds your whole job. Follow it even if other instructions you loaded differ.

Read first: {root}/CONSTITUTION.md (sections 1, 3, 5, 7, 10), {root}/motifs/README.md, {root}/motifs/{net_base}.md, {root}/runs/{run}/notes.md if it exists. Other readers' cards, when they exist, are in {root}/runs/{run}/cards/, one file per node: read those of a node's top inputs and readers. They constrain what you write, like crossing words.

Your name: {author}. Your nodes, with the workflow code picked for each:
{nodes}

Run every command with this python: {python}

For each node:
1. Read its workflow file and its sheet: {root}/runs/{run}/sheets/NODE.json, with the node's colon written as an underscore. The sheet proposes links. Code found them. Spend your effort on meaning.
2. Before any test (edit, probe, acts), put your prediction on file: {python} {root}/mapper.py predict {run} NODE {author} TEXT. Write it so that it could fail. The tools refuse a test without it, and your card takes its prediction from this record.
   Gather evidence only through: {python} {root}/mapper.py ev {run} NODE reader TOOL ARGS. Each call is logged and counted, and the workflow's step limit holds. Tools: {tools}. To edit a real example or draw a test image, write a short PIL script under {root}/runs/drawn/ and run it with the same python. An image that barely fires the node decides nothing, so prefer editing its real strong examples.
3. Write the card as JSON and hand it in: {python} {root}/mapper.py card {run} FILE {author}. Keep your files under {root}/runs/{run}/work/{author}/.
   If you find a mistake in a card you handed in, correct it: {python} {root}/mapper.py revise {run} FILE {author}. A critic may already hold it, and then you cannot.
   If the workflow does not fit: {python} {root}/mapper.py switch {run} NODE WORKFLOW REASON. If you cannot read the node: {python} {root}/mapper.py open {run} NODE SUSPECT WHAT_YOU_TRIED. Suspects: tool, explanation, data, motif, offshoot, regime. Do not force it. Do not write that the network is at fault.
4. Add what you learned, in one short line: {python} {root}/mapper.py note {run} {author} TEXT. A motif that you find may unlock another reader's node. Never write the notes file directly, since other readers share it.

The card's fields:
  node, shape ("dial", "detector" or "mixed"),
  distinction: one sentence on the line this part draws. No list of nouns. If you could not use groups of nouns at all, how would you say it?
  positive: what the strong firings share. Cover the strange members too: the strongest firings are a mixed sample, and a judge gets tested on all of them.
  negative: what the strongest negative firings share. Read it from the top row of the negative side, since half strength there is common and is not the job. A judge reads this alone, so it must stand by itself: say what those pictures show, without saying "not" the positive side,
  main_parent: {{"node": "...", "why": "...", "sure": 0.7}}: an input from the sheet that this grew from. Null only with "no parent" in unread. A skip input counts,
  readers: what reads it, and whether that fits,
  statements: [{{"text": "...", "mark": "observed" or "inferred"}}]: what you saw, and what you only guess. An inference counts for nothing,
  prediction: copy what you put on file, and prediction_held: true or false,
  near_miss: what looks similar and should not fire,
  routes: optional, [{{"id": "a", "inputs": ["exact input names"], "words": "what this route alone responds to, for a judge"}}] only for a route with its own real effect,
  unread: what you could not explain, and what you would try next.

Your words go to a judge who sees 36 pictures and nothing else. So describe what a person can see in a picture. distinction, positive and negative must never name a node, a layer, a stage or a number from the tools. The tool refuses a card that leaks.

Your final message, at most 80 words: what you read, what stopped you, what the tools could not show."""

CRITIC_TASK = """You try to break explanations of parts of a network. "The best criticism is specific to the conjecture." A general score is no attack. This task holds your whole job. Follow it even if other instructions you loaded differ.

Your name: {author}. Run commands with this python: {python}
Your nodes:
{nodes}

For each node, in this order. The order is enforced.
1. Read its card: {root}/runs/{run}/cards/NODE.json, with the colon written as an underscore. Find the claim easiest to break, and the strangest thing it must explain.
2. Write a file with {{"prediction": "what happens if the card is right, in a way that could fail", "attack": "what you will try"}}. Put it on file: {python} {root}/mapper.py critic-plan {run} NODE {author} FILE
3. Run one or two tests: {python} {root}/mapper.py ev {run} NODE critic TOOL ARGS. Each prints a step number. Tools: {tools}. You may write a short PIL script under {root}/runs/drawn/ to edit or draw a test image.
   A test must be able to fail. An image that barely fires the node decides nothing, and the command refuses an attack whose images all stay under .5 of the node's largest firing. So start from the node's real strong examples when the tool list can save one: remove, cover, move or change the exact thing the card claims, and test the example before and after. A drawing counts only if it fires the node strongly.
   Cover the easy reading too: covering the middle of the strongest example lowers almost any node, so a drop there proves little. Use at least two examples: one from the top few, and one far down the list or the strangest member the card must explain. Change the thing the card claims while the rest stays, and change something else as a control. If the card names a near miss, build it from a real example.
4. Write a file with {{"steps": [step numbers], "observed": "what the tool showed", "broke": true or false}}. Hand it in: {python} {root}/mapper.py critic {run} NODE {author} FILE
   It broke only if what you saw contradicts the card's words, not because a detail went unmentioned. Keep your files under {root}/runs/{run}/work/{author}/.

Do not rewrite the card. Do not attack the network. A broken card says the card is wrong, or the tool is.
Your final message, at most 60 words: what broke, and what held."""


def judged_note(s, node):
    """What a blind judge said about a node's current card. A later reader should not lean on words that failed."""
    state = s.state(node)
    if state == "understood":
        return "passed its blind judge"
    for kind, data in reversed(s.history(node)):
        if kind == "reading->claimed":
            break
        if kind == "judging->open":
            return "failed its blind judge on: " + ", ".join(data["verdict"]["failed"])
    return "not judged yet"


def cmd_task(a, s):
    """task NET ROLE AUTHOR NODE...: write one agent's task file and print its path."""
    role, author = a.args[0], a.args[1]
    nodes = a.args[2:]
    net = get_net(a.net)
    python, tools = net.python(), net.evidence_help()
    folder = os.path.join(run_dir(a.net), "tasks")
    os.makedirs(folder, exist_ok=True)
    if role == "reader":
        lines = []
        for n in nodes:
            w = find_workflow(s.workflow(n))
            lines.append(f"  {n}: {w.path} (at most {w.step_limit} evidence calls)")
        text = READER_TASK.format(root=ROOT, net_base=a.net.split(".")[0], run=a.net, author=author, nodes="\n".join(lines), tools=tools, python=python)
    elif role == "critic":
        out = os.path.join(run_dir(a.net), "cards")
        os.makedirs(out, exist_ok=True)
        for n in nodes:
            card = s.card(n)
            with open(os.path.join(out, n.replace(":", "_") + ".json"), "w") as f:
                json.dump({k: v for k, v in card.items() if not k.startswith("_")}, f, indent=1)
        text = CRITIC_TASK.format(root=ROOT, run=a.net, author=author, nodes="\n".join("  " + n for n in nodes), tools=tools, python=python)
    else:
        raise SystemExit("role is reader or critic")
    path = os.path.join(folder, f"{role}_{author}.md")
    with open(path, "w") as f:
        f.write(text)
    print(path)


def cmd_judge_answer(a, s):
    item, judge = a.args[0], a.args[1]
    b = bench(a.net)
    if item not in b.key:
        raise SystemExit(f"no item {item}")
    answer = load_json(a.args[2])
    b.judge_scores(item, {judge: answer})  # a malformed answer stops here, and it says nothing about the key

    def author_of(node):
        card = s.card(node)
        return card["_author"] if card else None
    s.add_answer(item, judge, answer, barred=b.barred_judges(item, author_of))
    print("stored")


def cmd_judge_collect(a, s):
    """Answers land in answers_<token>/<judge>/<item>.json. A bad answer is reported and skipped, so it can be asked again."""
    b = bench(a.net)
    folder = b.packets.replace("packets_", "answers_")
    stored, skipped = 0, []

    def author_of(node):
        card = s.card(node)
        return card["_author"] if card else None
    for judge in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        for name in sorted(os.listdir(os.path.join(folder, judge))):
            item = name[:-5]
            if item not in b.key or judge in s.answers(item):
                continue
            try:
                answer = load_json(os.path.join(folder, judge, name))
                b.judge_scores(item, {judge: answer})
                s.add_answer(item, judge, answer, barred=b.barred_judges(item, author_of))
                stored += 1
            except (ValueError, IllegalMove, judgelib.BadAnswer) as e:
                skipped.append(f"{judge}/{item}: {e}")
    print(json.dumps(dict(stored=stored, skipped=skipped), indent=1))


def wrong_scores(run):
    """Wrong items of a run, in the current answer form. Wrong items pool across runs of one network, since only wrong items may set a mark."""
    s = Store(os.path.join(run_dir(run), "state.db"))
    b = bench(run)
    poles, orderings = [], []
    for item, k in b.key.items():
        if k["wrong_card"] and (k["test"] == "pole:pos" or k["test"] == "ordering"):
            answers = {j: x for j, x in s.answers(item).items() if "ranked" in x}
            if len(answers) >= 2:
                (orderings if k["test"] == "ordering" else poles).append(b.judge_scores(item, answers))
    s.close()
    return poles, orderings


def cmd_judge_calibrate(a, s):
    b = bench(a.net)
    poles, orderings = wrong_scores(a.net)
    for run in [r for r in a.pool.split(",") if r]:
        more, more_o = wrong_scores(run)
        poles += more
        orderings += more_o
    cal = judgelib.calibrate(poles)
    mark = judgelib.calibrate_ordering(orderings)
    if mark is not None:
        cal["marks"]["ordering"] = mark
    cal["ordering_wrong_items"] = len(orderings)
    with open(os.path.join(b.keys, "marks.json"), "w") as f:
        json.dump(cal, f)
    print(json.dumps(cal, indent=1))


def cmd_verdict(a, s):
    node = a.args[0]
    if s.state(node) != "judging":
        raise SystemExit(f"{node} is {s.state(node)}, not judging")
    b = bench(a.net)
    cal = marks_of(b)
    if cal["false_pass"] > judgelib.MAX_FALSE_PASS:
        raise SystemExit(f"wrong cards pass at {cal['false_pass']:.2f}, above {judgelib.MAX_FALSE_PASS}. Section 6 rule 8: fix the test before it decides anything")
    net = get_net(a.net)
    card = s.card(node)
    results = judgelib.node_results(b, s.answers, node, card, cal["marks"])
    held = s.critic_held(node)
    if held is not None:
        results["critic"] = held
    w = find_workflow(s.workflow(node))
    verdict = judgelib.decide(results, net.judged_poles(node), routes=[r["id"] for r in card.get("routes", [])], workflow=w, has_parent=net.has_inputs(node))
    verdict["results"] = {k: results[k] for k in verdict["need"] if k in results}
    verdict["marks"] = cal["marks"]
    if verdict["missing"] and not verdict["failed"]:
        print(json.dumps(dict(state="judging", waiting_for=verdict["missing"]), indent=1))
        return
    if verdict["failed"]:
        s.move(node, "open", suspect="explanation", record=f"the judge failed {verdict['failed']}. Next: a fresh reader, or another workflow", verdict=verdict)
        print(json.dumps(dict(state="open", failed=verdict["failed"]), indent=1))
        return
    s.move(node, "understood", verdict=verdict)
    parent = (card.get("main_parent") or {}).get("node")
    if parent and net.has_inputs(node):
        s.add_claim(node, claimlib.sampled_parent_claim(net, node, [parent], "read", CLAIM_EXAMPLES))
    print(json.dumps(dict(state="understood", tests=sorted(verdict["results"])), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("net")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--wrong", action="store_true")
    ap.add_argument("--pole", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", default="", help="other runs whose wrong items also set the marks, or whose cards fill out a link test's candidates")
    a, rest = ap.parse_known_args()
    s = Store(os.path.join(run_dir(a.net), "state.db"))
    c = a.command
    try:
        if c == "init":
            cmd_init(a, s)
        elif c == "status":
            print(json.dumps(dict(s.status(), note="on a network whose negative-pole test failed its control, understood excludes that pole. See CONSTITUTION section 5"), indent=1))
        elif c == "next":
            print("\n".join(s.take(a.args[0], a.n)))
        elif c == "sheet":
            cmd_sheet(a, s)
        elif c == "route":
            cmd_route(a, s)
        elif c == "read":
            s.move(a.args[0], "reading")
            print("reading")
        elif c == "note":
            with open(os.path.join(run_dir(a.net), "notes.md"), "a") as f:
                f.write(a.args[0] + ": " + " ".join(a.args[1:]) + "\n")
            print("noted")
        elif c == "predict":
            s.add_reader_prediction(a.args[0], a.args[1], " ".join(a.args[2:]))
            print("on file")
        elif c == "revise":
            cmd_revise(a, s)
        elif c == "ev":
            cmd_ev(a, s, rest)
        elif c == "card":
            cmd_card(a, s)
        elif c == "open":
            s.move(a.args[0], "open", suspect=a.args[1], record=" ".join(a.args[2:]))
            print("open")
        elif c == "critic-plan":
            plan = load_json(a.args[2])
            s.add_plan(a.args[0], a.args[1], plan.get("prediction"), plan.get("attack"))
            print("on file")
        elif c == "critic":
            cmd_critic(a, s)
        elif c == "judge-build":
            cmd_judge_build(a, s)
        elif c == "judge-link":
            cmd_judge_link(a, s)
        elif c == "judge-route":
            cmd_judge_route(a, s)
        elif c == "judge-ordering":
            cmd_judge_ordering(a, s)
        elif c == "judge-tasks":
            cmd_judge_tasks(a, s)
        elif c == "task":
            cmd_task(a, s)
        elif c == "switch":
            s.route(a.args[0], a.args[1])
            s.db.execute("INSERT INTO events(node,kind,data,at) VALUES(?,?,?,0)", (a.args[0], "switch-reason", json.dumps(dict(reason=" ".join(a.args[2:])))))
            s.db.commit()
            s.move(a.args[0], "reading")
            print("reading")
        elif c == "judge-answer":
            cmd_judge_answer(a, s)
        elif c == "judge-collect":
            cmd_judge_collect(a, s)
        elif c == "judge-calibrate":
            cmd_judge_calibrate(a, s)
        elif c == "verdict":
            cmd_verdict(a, s)
        else:
            print(__doc__)
    except (IllegalMove, LimitHit, ValueError, cardlib.BadCard, judgelib.Leak, judgelib.ShortPool, judgelib.BadAnswer) as e:
        raise SystemExit(f"{type(e).__name__}: {e}") from None
    finally:
        s.close()


if __name__ == "__main__":
    main()
