"""The core, on a fake network. No torch, no photos, no agents. Run: python CrakBox/tests/test_core.py"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

from fake import FakeNet  # noqa: E402
from guards.sealed_guard import decide as guard  # noqa: E402
from mapper import card as cardlib  # noqa: E402
from mapper import claims, judge, workflow  # noqa: E402
from mapper.state import IllegalMove, LimitHit, Store  # noqa: E402

WORKFLOW = """---
id: {id}
version: 1
priority: {priority}
trigger: {trigger}
step_limit: 3
exits: claim, open
tests: {tests}
replaces_poles: {replaces}
---
Read the sheet. Answer the question.
"""


def make_workflow(id="clear", priority=10, trigger="largest > 0", tests="", replaces="no"):
    return workflow.parse(WORKFLOW.format(id=id, priority=priority, trigger=trigger, tests=tests, replaces=replaces))


def good_card(node, **change):
    c = dict(node=node, distinction=f"draws line number {node.split(':')[1]} against the rest", positive="strong side", negative="weak side", shape="detector",
             main_parent=dict(node="bot:1", why="largest input", sure=.7), readers="the next layer uses it", statements=[dict(text="fires on stripes", mark="observed")],
             prediction="a drawn stripe fires it", near_miss="a dotted line", unread="nothing yet")
    c.update(change)
    return c


FULL_VERDICT = dict(understood=True, failed=[], missing=[], results={"pole:pos": True})


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "state.db")
        self.s = Store(self.path)
        self.s.add_nodes([("top:1", "top"), ("top:2", "top"), ("bot:1", "bot")])

    def to_judging(self, node="top:1", author="reader-a"):
        s = self.s
        s.route(node, "clear")
        s.move(node, "reading")
        s.claim(node, author, good_card(node))
        s.add_plan(node, "critic-a", "it should fire on X", "try X")
        step = s.record_step(node, "critic", "probe", "out", 3)
        s.conclude_attack(node, "critic-a", [step], "it fired", False)
        s.move(node, "judging")

    def test_an_attack_must_be_able_to_fail(self):
        s, node = self.s, "top:1"
        s.route(node, "clear")
        s.move(node, "reading")
        s.claim(node, "reader-a", good_card(node))
        s.add_plan(node, "critic-a", "it should fire on X", "try X")

        def step(command, output):
            return s.record_step(node, "critic", command, output, 9)
        weak = step("probe a b", "b: largest value 0.40, 0.08 of the largest firing on the reading photos (5.00). Lowest value -1.00, 0.20 of the largest negative firing")
        strong = step("probe a c", "c: largest value 4.00, 0.80 of the largest firing on the reading photos (5.00). Lowest value -1.00, 0.20 of the largest negative firing")
        low = step("probe a d", "d: largest value 0.40, 0.08 of the largest firing on the reading photos (5.00). Lowest value -4.00, 0.80 of the largest negative firing")
        real = step("edit a", "top 20 photos, share kept")
        with self.assertRaises(IllegalMove):
            s.conclude_attack(node, "critic-a", [weak], "x", False)
        with self.assertRaises(IllegalMove):
            s.conclude_attack(node, "critic-a", [weak], "x", True)  # a weak test breaks nothing either
        self.assertEqual(s.state(node), "claimed")
        self.assertTrue(s.attack_has_power([weak, strong]))
        self.assertTrue(s.attack_has_power([low]))
        self.assertTrue(s.attack_has_power([weak, real]))

    def test_a_tool_fix_gives_one_read_and_spends_no_slot(self):
        s, node = self.s, "top:1"
        s.route(node, "clear")
        s.move(node, "reading")
        s.add_reader_prediction(node, "reader-a", "it should fire on X")
        s.record_step(node, "reader", "deep a", "out", 2)
        s.record_step(node, "reader", "deep a", "out", 2)
        with self.assertRaises(LimitHit):
            s.record_step(node, "reader", "deep a", "out", 2)
        with self.assertRaises(IllegalMove):
            s.retool(node, "a reading node is not open")
        s.claim(node, "reader-a", good_card(node))
        s.move(node, "open", suspect="explanation", record="the judge failed it")
        with self.assertRaises(IllegalMove):
            s.retool(node, "")
        s.retool(node, "the tool had no view of near items")
        s.move(node, "reading")
        self.assertEqual(s.tried(node), ["clear"])
        self.assertEqual(s.workflow(node), "clear")
        self.assertIsNone(s.reader_prediction(node))
        self.assertEqual(s.steps_used(node, "reader"), 0)
        s.record_step(node, "reader", "deep a", "out", 2)
        s.claim(node, "reader-b", good_card(node))
        s.move(node, "open", suspect="explanation", record="it failed again")
        with self.assertRaises(LimitHit):
            s.retool(node, "a second fix")
        s.route(node, "other")
        self.assertEqual(s.tried(node), ["clear", "other"])

    def test_a_voided_attack_counts_for_nothing(self):
        s, node = self.s, "top:1"
        s.route(node, "clear")
        s.move(node, "reading")
        s.claim(node, "reader-a", good_card(node))
        with self.assertRaises(IllegalMove):
            s.void_attack(node, "no attack yet")
        s.add_plan(node, "critic-a", "it should fire on X", "try X")
        s.conclude_attack(node, "critic-a", [s.record_step(node, "critic", "edit a", "out", 9)], "held", False)
        self.assertTrue(s.critic_held(node))
        s.void_attack(node, "it could not fail")
        self.assertEqual(s.state(node), "claimed")
        self.assertIsNone(s.critic_held(node))
        s.add_plan(node, "critic-b", "it should fire on Y", "try Y")
        s.conclude_attack(node, "critic-b", [s.record_step(node, "critic", "edit a", "out", 9)], "broke", True)
        self.assertFalse(s.critic_held(node))
        self.assertEqual(s.state(node), "reading")
        s.void_attack(node, "it could not fail")
        self.assertEqual(s.state(node), "claimed")
        self.assertIsNone(s.critic_held(node))

    def test_the_bar_counts_only_verdicts(self):
        self.to_judging()
        self.s.move("top:1", "understood", verdict=FULL_VERDICT)
        st = self.s.status()
        self.assertEqual((st["understood"], st["scope"]), (1, 3))
        self.assertEqual(st["layers"]["top"], dict(scope=2, understood=1))

    def test_understood_needs_a_verdict(self):
        self.to_judging()
        for bad in (None, {}, dict(understood=False, results={"x": True}), dict(understood=True, results={}), dict(understood=True, failed=["x"], results={"x": True}),
                    dict(understood=True, missing=["x"], results={"x": True}), dict(understood=True, results={"x": False})):
            with self.assertRaises(IllegalMove, msg=str(bad)):
                self.s.move("top:1", "understood", verdict=bad)
        self.assertEqual(self.s.state("top:1"), "judging")

    def test_every_illegal_move_is_refused(self):
        """Walks the whole table: from each state, every state not listed must be refused."""
        from mapper.state import MOVES
        walk = {"queued": [], "routed": ["routed"], "reading": ["routed", "reading"], "open": ["routed", "open"]}
        for i, (state, path) in enumerate(walk.items()):
            node = f"n:{i}"
            self.s.add_nodes([(node, "n")])
            for step in path:
                if step == "routed":
                    self.s.route(node, "w0")
                elif step == "reading":
                    self.s.move(node, "reading")
                elif step == "open":
                    self.s.move(node, "open", suspect="tool", record="x")
            self.assertEqual(self.s.state(node), state)
            for to in MOVES:
                if to not in MOVES[state]:
                    with self.assertRaises(IllegalMove, msg=f"{state}->{to}"):
                        self.s.move(node, to, suspect="tool", record="x", verdict=FULL_VERDICT)
                    self.assertEqual(self.s.state(node), state)

    def test_no_skipping_the_critic(self):
        self.s.route("top:1", "clear")
        self.s.move("top:1", "reading")
        with self.assertRaises(IllegalMove):
            self.s.move("top:1", "judging")
        self.s.claim("top:1", "a", good_card("top:1"))
        with self.assertRaises(IllegalMove):
            self.s.move("top:1", "judging")

    def test_critic_needs_a_prediction_first(self):
        s = self.s
        s.route("top:1", "clear")
        s.move("top:1", "reading")
        s.claim("top:1", "reader-a", good_card("top:1"))
        early = s.record_step("top:1", "critic", "probe", "out", 3)
        with self.assertRaises(IllegalMove):
            s.conclude_attack("top:1", "critic-a", [early], "x", False)  # no plan
        s.add_plan("top:1", "critic-a", "p", "a")
        with self.assertRaises(IllegalMove):
            s.conclude_attack("top:1", "critic-a", [early], "x", False)  # the test ran before the prediction
        late = s.record_step("top:1", "critic", "probe", "out", 3)
        reader_step = s.record_step("top:1", "reader", "probe", "out", 3)
        with self.assertRaises(IllegalMove):
            s.conclude_attack("top:1", "critic-a", [reader_step], "x", False)  # another role's call
        with self.assertRaises(IllegalMove):
            s.conclude_attack("top:1", "critic-b", [late], "x", False)  # another critic
        with self.assertRaises(IllegalMove):
            s.conclude_attack("top:1", "critic-a", [], "x", False)  # no call
        s.conclude_attack("top:1", "critic-a", [late], "x", False)
        self.assertEqual(s.state("top:1"), "attacked")

    def test_author_cannot_attack_its_own_card(self):
        s = self.s
        s.route("top:1", "clear")
        s.move("top:1", "reading")
        s.claim("top:1", "same", good_card("top:1"))
        s.add_plan("top:1", "same", "p", "a")
        step = s.record_step("top:1", "critic", "probe", "out", 3)
        with self.assertRaises(IllegalMove):
            s.conclude_attack("top:1", "same", [step], "x", False)

    def test_a_broken_card_needs_a_new_attack(self):
        s = self.s
        s.route("top:1", "clear")
        s.move("top:1", "reading")
        s.claim("top:1", "a", good_card("top:1"))
        s.add_plan("top:1", "c", "p", "a")
        step = s.record_step("top:1", "critic", "probe", "out", 3)
        s.conclude_attack("top:1", "c", [step], "x", True)
        self.assertEqual(s.state("top:1"), "reading")
        self.assertIs(s.critic_held("top:1"), False)
        s.claim("top:1", "a", good_card("top:1", distinction="better words"))
        self.assertIsNone(s.critic_held("top:1"))

    def test_open_needs_a_suspect_and_a_record(self):
        with self.assertRaises(IllegalMove):
            self.s.move("top:1", "open", record="tried things")
        with self.assertRaises(IllegalMove):
            self.s.move("top:1", "open", suspect="motif")
        with self.assertRaises(IllegalMove):
            self.s.move("top:1", "open", suspect="the network is bad", record="x")
        self.s.move("top:1", "open", suspect="motif", record="tried the sheet")
        self.assertEqual(self.s.status()["open"][0]["suspect"], "motif")

    def test_three_workflows_then_stays_open(self):
        s = self.s
        for w in ("a", "b", "c"):
            s.route("top:1", w)
            s.move("top:1", "open", suspect="tool", record="tried " + w)
        with self.assertRaises(LimitHit):
            s.route("top:1", "d")
        self.assertEqual(s.state("top:1"), "open")

    def test_same_workflow_twice_is_refused(self):
        self.s.route("top:1", "a")
        self.s.move("top:1", "open", suspect="tool", record="x")
        with self.assertRaises(IllegalMove):
            self.s.route("top:1", "a")

    def test_step_limit(self):
        s = self.s
        s.route("top:1", "clear")
        for _ in range(3):
            s.record_step("top:1", "reader", "deep", "out", 3)
        with self.assertRaises(LimitHit):
            s.record_step("top:1", "reader", "deep", "out", 3)

    def test_a_made_up_role_gets_no_steps(self):
        self.s.route("top:1", "clear")
        with self.assertRaises(IllegalMove):
            self.s.record_step("top:1", "reader2", "deep", "out", 3)

    def test_each_role_has_its_own_steps(self):
        s = self.s
        s.route("top:1", "clear")
        for _ in range(3):
            s.record_step("top:1", "reader", "deep", "out", 3)
        self.assertIsNotNone(s.record_step("top:1", "critic", "probe", "out", 3))

    def test_a_new_workflow_gets_new_steps(self):
        s = self.s
        s.route("top:1", "a")
        for _ in range(3):
            s.record_step("top:1", "reader", "deep", "out", 3)
        s.move("top:1", "open", suspect="tool", record="x")
        s.route("top:1", "b")
        self.assertIsNotNone(s.record_step("top:1", "reader", "deep", "out", 3))

    def test_smoke_gate_counts_nodes_already_started(self):
        s = self.s
        s.add_nodes([(f"x:{i}", "x") for i in range(30)])
        with self.assertRaises(LimitHit):
            s.take("queued", 50)
        for node in s.take("queued", 10):
            s.route(node, "clear")
        with self.assertRaises(LimitHit):
            s.take("queued", 1)  # repeated small batches do not walk past the gate
        s.set_smoke_passed("10 nodes ran")
        self.assertGreater(len(s.take("queued", 50)), 10)

    def test_scope_must_be_named(self):
        with self.assertRaises(IllegalMove):
            self.s.set_scope(dict(nodes=["a:1"]), [("a:1", "a")])
        self.s.set_scope(dict(nodes=["a:1"], goal="g", control_goal="c", threshold=.1, firings="strong"), [("a:1", "a")])
        self.assertEqual(self.s.scope()["goal"], "g")
        self.assertEqual(self.s.state("a:1"), "queued")

    def test_an_answer_cannot_change_and_an_author_cannot_judge(self):
        self.s.add_answer("item", "j1", dict(pick=1))
        with self.assertRaises(IllegalMove):
            self.s.add_answer("item", "j1", dict(pick=2))
        self.assertEqual(self.s.answers("item")["j1"], dict(pick=1))
        with self.assertRaises(IllegalMove):
            self.s.add_answer("item2", "reader-a", dict(pick=1), barred={"reader-a"})

    def test_resume_after_a_crash(self):
        """A process dies in the middle of a run. A new process sees every finished move and no half move."""
        code = (f"import sys; sys.path.insert(0, {ROOT!r}); from mapper.state import Store; import os\n"
                f"s = Store({self.path!r}); s.route('top:2', 'clear'); s.move('top:2', 'reading')\n"
                "s.db.execute(\"UPDATE nodes SET state='understood' WHERE id='top:2'\")\n"  # not committed
                "os._exit(1)\n")
        self.s.close()
        subprocess.run([sys.executable, "-c", code])
        s = Store(self.path)
        self.assertEqual(s.state("top:2"), "reading")
        self.assertEqual(s.tried("top:2"), ["clear"])
        self.assertEqual([k for k, _ in s.history("top:2")], ["queued->routed", "routed->reading"])
        self.s = s

    def test_a_card_and_its_move_are_one_transaction(self):
        self.to_judging()
        self.s.move("top:1", "open", suspect="tool", record="x")
        with self.assertRaises(IllegalMove):
            self.s.claim("top:1", "someone", good_card("top:1", distinction="unjudged words"))
        self.assertNotEqual(self.s.card("top:1")["distinction"], "unjudged words")


class TestCards(unittest.TestCase):
    def test_good_card(self):
        cardlib.check(good_card("top:1"))

    def test_rejects(self):
        for bad in (dict(prediction=""), dict(shape="blob"), dict(statements=[]), dict(statements=[dict(text="x", mark="guess")]),
                    dict(distinction="this is polysemantic"), dict(unread="many weak parents"), dict(main_parent=None), dict(routes=[dict(id="a")])):
            with self.assertRaises(cardlib.BadCard, msg=str(bad)):
                cardlib.check(good_card("top:1", **bad))

    def test_no_parent_with_a_record_is_allowed(self):
        cardlib.check(good_card("top:1", main_parent=None, unread="no parent: acts on the top 5 inputs moved nothing against controls"))

    def test_a_node_with_one_pole_owes_no_negative_words(self):
        c = good_card("raw:1")
        del c["negative"]
        cardlib.check(c, poles=("pos",))
        with self.assertRaises(cardlib.BadCard):
            cardlib.check(c)

    def test_inference_counts_for_nothing(self):
        c = good_card("top:1", statements=[dict(text="a", mark="observed"), dict(text="b", mark="inferred")])
        self.assertEqual([s["text"] for s in cardlib.tested(c)], ["a"])

    def test_legacy(self):
        cardlib.check(dict(node="top:1", distinction="d", positive="p", negative="n"), legacy=True)


class TestWorkflows(unittest.TestCase):
    def test_route_by_priority_and_tried(self):
        lib = sorted([make_workflow("dial", 5, "cliff < .3"), make_workflow("clear", 10)], key=lambda w: w.priority)
        self.assertEqual(workflow.route(dict(cliff=.1, largest=2), lib).id, "dial")
        self.assertEqual(workflow.route(dict(cliff=.9, largest=2), lib).id, "clear")
        self.assertEqual(workflow.route(dict(cliff=.1, largest=2), lib, ("dial",)).id, "clear")
        self.assertIsNone(workflow.route(dict(cliff=.1, largest=2), lib, ("dial", "clear")))

    def test_trigger_runs_no_code(self):
        for bad in ("__import__('os').system('true')", "largest.real > 0", "[1][0] > 0", "largest + 1 > 0"):
            with self.assertRaises(workflow.BadWorkflow, msg=bad):
                workflow.evaluate(bad, dict(largest=1))

    def test_unknown_feature_is_an_error(self):
        with self.assertRaises(workflow.BadWorkflow):
            workflow.evaluate("nothing > 0", dict(largest=1))

    def test_every_workflow_can_end_open(self):
        with self.assertRaises(workflow.BadWorkflow):
            workflow.parse(WORKFLOW.format(id="x", priority=1, trigger="largest > 0", tests="", replaces="no").replace("exits: claim, open", "exits: claim"))

    def test_only_an_ordering_replaces_poles(self):
        with self.assertRaises(workflow.BadWorkflow):
            make_workflow(replaces="yes")
        self.assertTrue(make_workflow(tests="ordering", replaces="yes").replaces_poles)


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.net = FakeNet()
        self.dir = tempfile.mkdtemp()
        self.cards = {n: good_card(n) for n in self.net.nodes("top")[:6]}
        self.bench = judge.Bench(self.dir)

    def build(self, wrong=False, node="top:0"):
        return self.bench.build_pole(self.net, node, "pos", self.cards, wrong=wrong)

    def packet(self, item):
        with open(os.path.join(self.dir, "packets", item + ".json")) as f:
            return json.load(f)

    def test_sheet_holds_the_right_pool(self):
        k = self.bench.key[self.build()]
        self.assertEqual(len(k["shares"]), 36)
        self.assertEqual((k["kinds"].count("fire"), k["kinds"].count("near"), k["kinds"].count("random")), (12, 12, 12))
        quiet = [s for s, kind in zip(k["shares"], k["kinds"]) if kind != "fire"]
        fire = [s for s, kind in zip(k["shares"], k["kinds"]) if kind == "fire"]
        self.assertLess(max(quiet), judge.QUIET)
        self.assertGreaterEqual(min(fire), .5)

    def test_scoring_on_known_answers(self):
        k = self.bench.key[self.build()]
        shares = np.array(k["shares"])
        order = np.argsort(-shares)
        best = dict(ranked=[int(j) + 1 for j in order[:12]])
        s = judge.score_pole(k, best)
        self.assertAlmostEqual(s["center"], shares[order[:6]].mean())
        self.assertAlmostEqual(s["center_norm"], 1.0)
        self.assertEqual((s["fire_rate"], s["near_rate"], s["random_rate"]), (1.0, 0.0, 0.0))
        self.assertEqual((s["edge"], s["far"]), (1.0, 1.0))
        kinds = np.array(k["kinds"])
        near = [int(j) + 1 for j in np.where(kinds == "near")[0]]
        s2 = judge.score_pole(k, dict(ranked=near))
        self.assertLess(s2["center_norm"], 0)
        self.assertEqual((s2["edge"], s2["far"]), (-1.0, 0.0))
        rand = [int(j) + 1 for j in np.where(kinds == "random")[0]]
        self.assertEqual(judge.score_pole(k, dict(ranked=rand))["far"], -1.0)
        self.assertAlmostEqual(judge.chance_center(shares).mean(), shares.mean(), places=1)

    def test_random_rankings_score_near_zero(self):
        k = self.bench.key[self.build()]
        rng = np.random.default_rng(0)
        got = [judge.score_pole(k, dict(ranked=[int(j) + 1 for j in rng.permutation(36)[:12]])) for _ in range(800)]
        for field in ("center_norm", "edge", "far"):
            self.assertLess(abs(np.mean([g[field] for g in got])), .03, field)

    def test_answers_must_rank_twelve_different_items(self):
        k = self.bench.key[self.build()]
        for bad in (dict(ranked=[1] * 12), dict(ranked=list(range(1, 12))), dict(ranked=list(range(1, 14))), dict(ranked=list(range(0, 12))), dict(), dict(ranked=[str(i) for i in range(1, 13)])):
            with self.assertRaises(judge.BadAnswer, msg=str(bad)):
                judge.score_pole(k, bad)

    def test_packet_is_blind(self):
        """No node name, layer, author, evidence or truth in anything a judge reads."""
        for wrong in (False, True):
            item = self.build(wrong, node=f"top:{1 if wrong else 0}")
            packet = self.packet(item)
            self.assertEqual(set(packet), {"item", "test", "sheet", "words", "count", "rank"})
            self.assertEqual(set(packet["words"]), {"job", "this_side"})
            neg = self.packet(self.bench.build_pole(self.net, f"top:{4 if wrong else 5}", "neg", self.cards, wrong=wrong))
            self.assertEqual(set(neg["words"]), {"this_side"})
            self.assertTrue(re.fullmatch(r"item_[0-9a-f]{12}", packet["item"]))
            self.assertEqual(os.path.basename(packet["sheet"]), item + ".png")
            text = json.dumps(packet)
            for word in ("share", "kind", "wrong", "author", "main_parent", "fire"):
                self.assertNotIn(word, text)

    def test_a_card_that_names_its_place_is_refused(self):
        """The scan must be able to fail. A reader's words can leak a node's place, so the words get scanned too."""
        for words in ("this is top:0", "TOP:0 for short", "it reads from the bot layer", "built from BOT:3", "it is unit 179", "channel 7 feeds it"):
            cards = dict(self.cards)
            cards["top:0"] = good_card("top:0", distinction=words)
            with self.assertRaises(judge.Leak, msg=words):
                self.bench.build_pole(self.net, "top:0", "pos", cards)
        self.assertFalse(judge.leaks(dict(words=dict(job="a bottle on a stop sign")), ["top", "bot"]))

    def test_a_link_packet_is_scanned_too(self):
        cards = {n: dict(good_card(n), _layer=self.net.layer_of(n)) for n in self.net.nodes("top")[:12] + self.net.nodes("bot")[:12]}
        cards["bot:7"]["distinction"] = "leaks bot:7"
        with self.assertRaises(judge.Leak):
            for seed in range(12):  # bot:7 is among the candidates in most draws
                judge.Bench(tempfile.mkdtemp(), seed=seed).build_link(self.net, "parent", "top:0", "bot:1", cards)

    def test_each_item_differs_and_wrong_share_is_near_a_quarter(self):
        """One new Bench per item, as one command per item gives."""
        cards = {n: good_card(n) for n in self.net.nodes("top")[:13] if n != "top:3"}  # top:3 is the spike node
        layouts, wrong_items, total = set(), 0, 0
        for n in sorted(cards):
            b = judge.Bench(self.dir)
            k = b.key[b.build_pole(self.net, n, "pos", cards)]
            layouts.add(tuple(k["kinds"]))
            total += 1
            for _ in range(3):
                b = judge.Bench(self.dir)
                if b.want_wrong_item():
                    wk = b.key[b.build_pole(self.net, n, "pos", cards, wrong=True)]
                    layouts.add(tuple(wk["kinds"]))
                    wrong_items += 1
                b = judge.Bench(self.dir)
                k = b.key[b.build_pole(self.net, n, "neg", cards, wrong=True)]
                layouts.add(tuple(k["kinds"]))
                total += 1
        self.assertEqual(len(layouts), total + wrong_items)
        self.assertGreater(wrong_items, 3)
        self.assertLess(wrong_items, 24)

    def test_rebuilding_cannot_shop_for_an_easier_sheet(self):
        a = self.build()
        self.assertEqual(judge.Bench(self.dir).build_pole(self.net, "top:0", "pos", self.cards), a)
        changed = dict(self.cards)
        changed["top:0"] = good_card("top:0", distinction="other words")
        self.assertNotEqual(judge.Bench(self.dir).build_pole(self.net, "top:0", "pos", changed), a)

    def test_wrong_card_uses_other_words(self):
        item = self.build(True)
        k = self.bench.key[item]
        self.assertTrue(k["wrong_card"])
        self.assertNotEqual(k["words_from"], "top:0")
        self.assertEqual(self.cards[k["words_from"]]["distinction"], self.packet(item)["words"]["job"])

    def test_key_survives_a_new_bench(self):
        self.assertIn(self.build(), judge.Bench(self.dir).key)

    def test_a_broad_pole_samples_only_its_strong_tail(self):
        values = np.linspace(0, 10, 2000)  # a quarter and more of all examples sit above half of the largest
        pool = judge.strong_pool(values, 10.0)
        self.assertEqual(len(pool), 40)
        self.assertGreaterEqual(values[pool].min(), 9.7)
        narrow = np.zeros(2000)
        narrow[:20] = 10.0
        self.assertEqual(len(judge.strong_pool(narrow, 10.0)), 20)

    def test_a_spike_node_is_refused_and_not_faked(self):
        with self.assertRaises(judge.ShortPool):
            judge.build_pole_sheet(self.net, "top:3", "pos", "sealed", np.random.default_rng(0))

    def test_votes_not_medians(self):
        """Three judges who each fail must fail, even when a median across fields would pass."""
        marks = dict(center=.5, edge=.3, far=.3)
        a = dict(center_norm=.9, edge=.0, far=.9)
        b = dict(center_norm=.9, edge=.9, far=.0)
        c = dict(center_norm=.1, edge=.9, far=.9)
        self.assertTrue(judge.passes(judge.medians([a, b, c]), marks))
        self.assertIs(judge.vote([judge.passes(s, marks) for s in (a, b, c)]), False)

    def test_vote(self):
        self.assertIsNone(judge.vote([]))
        self.assertIsNone(judge.vote([True]))
        self.assertIsNone(judge.vote([True, False]))
        self.assertIs(judge.vote([True, True]), True)
        self.assertIs(judge.vote([False, False]), False)
        self.assertIs(judge.vote([True, False, True]), True)
        self.assertIs(judge.vote([True, False, False]), False)

    def test_third_judge_and_agreement(self):
        marks = dict(center=.5, edge=.3, far=.3)
        yes, no = dict(center_norm=.8, edge=.5, far=.5), dict(center_norm=.2, edge=.5, far=.5)
        self.assertTrue(judge.need_third([yes, no], marks))
        self.assertFalse(judge.need_third([yes, yes], marks))
        self.assertEqual(judge.agreement([(yes, yes), (yes, no)], marks), .5)

    def test_calibration_states_its_rate_on_held_out_items(self):
        rng = np.random.default_rng(0)
        items = [[dict(center_norm=float(c), edge=float(e), far=float(f)) for c, e, f in rng.normal(0, .2, (2, 3))] for _ in range(60)]
        cal = judge.calibrate(items)
        self.assertTrue(cal["held_out"])
        self.assertEqual(cal["stated_on"], 30)
        self.assertGreaterEqual(cal["false_pass_upper"], cal["false_pass"])
        self.assertLess(cal["false_pass"], .2)
        self.assertLess(cal["quantile"], .95)  # three tests share the work, so each mark need not sit at the top
        self.assertFalse(judge.calibrate(items[:6])["held_out"])
        with self.assertRaises(ValueError):
            judge.calibrate([[]])

    def test_a_split_counts_as_a_pass_in_the_stated_rate(self):
        items = [[dict(center_norm=1.0, edge=1.0, far=1.0), dict(center_norm=-1.0, edge=-1.0, far=-1.0)] for _ in range(4)]
        marks = dict(center=0.0, edge=0.0, far=0.0)
        self.assertEqual(judge.rate_of(items, marks), 4)

    def test_upper_range(self):
        self.assertAlmostEqual(judge.clopper_upper(0, 22), 1 - .05 ** (1 / 22), places=4)

    def test_link_packet(self):
        cards = {n: dict(good_card(n), _layer=self.net.layer_of(n)) for n in self.net.nodes("top")[:12] + self.net.nodes("bot")[:12]}
        item = self.bench.build_link(self.net, "parent", "top:0", "bot:1", cards)
        k = self.bench.key[item]
        packet = self.packet(item)
        self.assertEqual(len(packet["candidates"]), judge.LINK_CHOICES)
        self.assertEqual(self.bench.judge_scores(item, {"a": dict(pick=k["true_number"])}), [dict(correct=True)])
        self.assertEqual(self.bench.judge_scores(item, {"a": dict(pick=k["true_number"] % 8 + 1)}), [dict(correct=False)])
        text = json.dumps(packet)
        self.assertNotIn("true_number", text)
        self.assertNotIn("_layer", text)

    def test_a_real_link_is_not_a_wrong_choice(self):
        cards = {n: dict(good_card(n), _layer=self.net.layer_of(n)) for n in self.net.nodes("top")[:12] + self.net.nodes("bot")}
        real = judge.strong_links(self.net, "parent", "top:0")
        self.assertGreater(len(real), 2)
        item = self.bench.build_link(self.net, "parent", "top:0", real[0], cards)
        shown = {c["job"] for c in self.packet(item)["candidates"]}
        for other in real[1:]:
            self.assertNotIn(cards[other]["distinction"], shown)

    def test_a_small_layer_is_widened_and_the_chance_stays(self):
        cards = {n: dict(good_card(n), _layer=self.net.layer_of(n)) for n in self.net.nodes("bot")[:3] + self.net.nodes("top")[:12] + self.net.nodes("raw")[:12]}
        item = self.bench.build_link(self.net, "parent", "top:0", "bot:1", cards)
        self.assertEqual(len(self.packet(item)["candidates"]), judge.LINK_CHOICES)
        self.assertEqual(self.bench.key[item]["chance"], 1 / judge.LINK_CHOICES)

    def test_a_wrong_choice_never_shares_the_node_author(self):
        # the truth's layer is short, so the pool widens to the layer that holds the node's own author
        cards = {n: dict(good_card(n), _layer=self.net.layer_of(n), _author="other") for n in self.net.nodes("bot")[:8]}
        cards.update({n: dict(good_card(n), _layer="raw", _author="third") for n in self.net.nodes("raw")[:12]})
        for n in self.net.nodes("top")[:12]:
            cards[n] = dict(good_card(n, distinction=f"written by the same author, number {n[4:]}"), _layer="top", _author="me")
        item = self.bench.build_link(self.net, "parent", "top:0", "bot:1", cards)
        self.assertNotIn("written by the same author", json.dumps(self.packet(item)["candidates"]))

    def test_a_wrong_link_item_uses_words_that_do_not_really_link(self):
        cards = {n: dict(good_card(n), _layer=self.net.layer_of(n)) for n in self.net.nodes("top") + self.net.nodes("bot")}
        # every node here has the same strongest inputs, so no words can serve as a wrong item for one of them
        truth = judge.strong_links(self.net, "parent", "top:0")[0]
        with self.assertRaises(judge.ShortPool):
            self.bench.build_link(self.net, "parent", "top:0", truth, cards, wrong=True)
        # an input that is strong for no node leaves every other card free to serve
        item = self.bench.build_link(self.net, "parent", "top:0", "bot:9", cards, wrong=True)
        source = self.bench.key[item]["words_from"]
        self.assertNotEqual(source, "top:0")
        self.assertNotIn("bot:9", judge.strong_links(self.net, "parent", source))

    def test_a_link_needs_words_from_an_independent_reader(self):
        cards = {n: dict(good_card(n), _layer=self.net.layer_of(n), _author="b") for n in self.net.nodes("bot")}
        cards["top:0"] = dict(good_card("top:0"), _layer="top", _author="b")
        with self.assertRaises(judge.ShortPool):
            self.bench.build_link(self.net, "parent", "top:0", "bot:1", cards)
        cards["top:0"]["_author"] = "a"
        self.assertTrue(self.bench.build_link(self.net, "parent", "top:0", "bot:1", cards))

    def test_outputs_have_their_own_words(self):
        item = self.bench.build_link(self.net, "reader", "top:0", "logit:1", {"top:0": dict(good_card("top:0"), _layer="top")})
        self.assertIn("output number", json.dumps(self.packet(item)["candidates"]))

    def test_route_values_use_the_exact_sum_on_both_poles(self):
        ex = np.arange(5)
        D, names, base, true = self.net.inputs("top:0", "sealed", ex, ex * 0)
        got = judge.route_values(self.net, "top:0", "pos", {"bot:0", "bot:2"}, "sealed", ex, ex * 0)
        self.assertTrue(np.allclose(got, np.maximum(base + D[:, 0] + D[:, 2], 0)))
        neg = judge.route_values(self.net, "top:0", "neg", {"bot:0", "bot:2"}, "sealed", ex, ex * 0)
        self.assertTrue(np.allclose(neg, np.maximum(-(base + D[:, 0] + D[:, 2]), 0)))
        self.assertTrue(np.allclose(judge.route_values(self.net, "top:0", "pos", set(names), "sealed", ex, ex * 0), np.maximum(true, 0)))

    def test_route_sheet_scores_the_route_on_its_quiet_items_too(self):
        tiles, rows = judge.build_pole_sheet(self.net, "top:0", "pos", "sealed", np.random.default_rng(0), route={"bot:0"})
        ex = np.array([r["example"] for r in rows])
        rv = judge.route_values(self.net, "top:0", "pos", {"bot:0"}, "sealed", ex, ex * 0)
        fire = [r for r in rows if r["kind"] == "fire"]
        self.assertGreaterEqual(min(r["share"] for r in fire), .5)
        ratio = [r["share"] / v for r, v in zip(rows, rv) if v > 0]
        self.assertTrue(np.allclose(ratio, ratio[0]))

    def test_battery_packet_and_score(self):
        net = FakeNet()
        cards = {n: good_card(n) for n in net.nodes("top")[:4]}
        item = self.bench.build_ordering(net, "top:0", cards)
        k = self.bench.key[item]
        order = np.argsort(-np.array(k["values"]))
        best = judge.score_battery(k, dict(ranked=[int(j) + 1 for j in order[:8]]))
        worst = judge.score_battery(k, dict(ranked=[int(j) + 1 for j in order[-8:]]))
        self.assertGreater(best["rho"], .5)
        self.assertLess(worst["rho"], -.2)
        self.assertTrue(judge.judge_passed(best, dict(ordering=.3)))
        self.assertFalse(judge.judge_passed(worst, dict(ordering=.3)))
        with self.assertRaises(judge.BadAnswer):
            judge.score_battery(k, dict(ranked=[1, 2]))

    def test_a_node_the_battery_cannot_reach_is_refused(self):
        net = FakeNet()
        net.battery_values = lambda node: np.zeros(12)
        with self.assertRaises(judge.ShortPool):
            self.bench.build_ordering(net, "top:0", {"top:0": good_card("top:0")})

    def test_ordering(self):
        self.assertAlmostEqual(judge.score_ordering([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0)
        self.assertAlmostEqual(judge.score_ordering([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0)
        self.assertEqual(judge.score_ordering([1, 1, 1, 1, 1], [10, 20, 30, 40, 50]), 0.0)  # a prediction that says nothing
        with self.assertRaises(ValueError):
            judge.score_ordering([1, 2, 3], [1, 2, 3])

    def test_decide(self):
        full = {"pole:pos": True, "pole:neg": True, "parent": True, "reader": True, "critic": True}
        poles = ("pos", "neg")
        self.assertTrue(judge.decide(full, poles)["understood"])
        self.assertTrue(judge.decide(dict(full, **{"pole:neg": False}), ("pos",))["understood"])  # a network whose negative test failed its control judges one pole
        got = judge.decide(dict(full, parent=False), poles)
        self.assertEqual((got["understood"], got["failed"]), (False, ["parent"]))
        missing = dict(full)
        del missing["critic"]
        got = judge.decide(missing, poles)
        self.assertEqual((got["understood"], got["missing"]), (False, ["critic"]))
        self.assertEqual(judge.decide(dict(full), poles, routes=("a",))["missing"], ["route:a"])

    def test_a_node_that_reads_the_raw_input_can_be_understood(self):
        got = judge.decide({"pole:pos": True, "reader": True, "critic": True}, ("pos",), has_parent=False)
        self.assertTrue(got["understood"])

    def test_workflow_can_add_and_only_a_dial_can_replace(self):
        full = {"pole:pos": True, "pole:neg": True, "parent": True, "reader": True, "critic": True}
        self.assertEqual(judge.decide(full, ("pos", "neg"), workflow=make_workflow(tests="drawn"))["missing"], ["drawn"])
        dial = make_workflow(tests="ordering", replaces="yes")
        self.assertTrue(judge.decide({"parent": True, "reader": True, "critic": True, "ordering": True}, ("pos", "neg"), workflow=dial)["understood"])
        self.assertEqual(judge.decide({"parent": True, "critic": True, "ordering": True}, ("pos", "neg"), workflow=dial)["missing"], ["reader"])


class TestScope(unittest.TestCase):
    def test_trace_is_closed_and_reaches_the_raw_input(self):
        from mapper import scope
        net = FakeNet()
        nodes, info = scope.trace(net, np.arange(200), 4, log=lambda *_: None)
        self.assertEqual({net.layer_of(n) for n in nodes}, {"top", "bot", "raw"})
        kept = set(nodes)
        for n in nodes:
            if net.has_inputs(n):
                names, weight = scope.strongest_input(net, n, np.arange(200))
                self.assertIn(net.canonical(names[int(np.argmax(weight))]), kept)
        self.assertGreaterEqual(len(nodes), 12)


class TestClaims(unittest.TestCase):
    def test_error_is_zero_exactly_when_all_inputs_stay(self):
        net = FakeNet()
        ex = np.arange(100)
        pos = np.zeros(100, dtype=int)
        full = claims.parent_claim(net, "top:0", [f"bot:{j}" for j in range(6)], "sealed", ex, pos)
        some = claims.parent_claim(net, "top:0", ["bot:0"], "sealed", ex, pos)
        self.assertEqual(full["eps"], 0.0)
        self.assertGreater(some["eps"], 0.0)

    def test_a_non_input_is_refused(self):
        with self.assertRaises(ValueError):
            claims.parent_claim(FakeNet(), "top:0", ["bot:99"], "sealed", np.arange(5), np.zeros(5, dtype=int))

    def test_a_sampled_claim_says_it_is_an_estimate(self):
        self.assertTrue(claims.sampled_parent_claim(FakeNet(), "top:0", ["bot:0"], "read", 50)["estimate"])


class TestGuard(unittest.TestCase):
    def call(self, tool_input, tool="Bash", **extra):
        return guard(dict(tool_name=tool, tool_input=tool_input, **extra))

    def test_blocks_sealed_data_and_keys(self):
        for text in ("cache/map18/sealed/s4b1_hx.npy", "cache/r18/ix_sealed.npy", "CrakBox/runs/resnet18/sealed_keys/key.json"):
            self.assertIsNotNone(self.call(dict(file_path=text), "Read"), text)
            self.assertIsNotNone(self.call(dict(command=f"cat {text}")), text)
            self.assertIsNotNone(self.call(dict(pattern="x", path=text), "Grep"), text)

    def test_tricky_commands_are_blocked(self):
        for command in ("python3 -c \"import json; print(json.load(open('runs/x/sealed_keys/key.json')))\"", "grep -r mapper.py judge-score runs/x/sealed_keys", "cat runs/x/seal*/key.json", "cat runs/x/sea*_keys/key.json", "cat runs/x/s?aled_keys/k*"):
            self.assertIsNotNone(self.call(dict(command=command)), command)

    def test_a_search_may_not_sweep_the_runs_folder(self):
        self.assertIsNotNone(self.call(dict(pattern="true_number", path="CrakBox/runs/resnet18"), "Grep"))
        self.assertIsNotNone(self.call(dict(pattern="**/*.json", path="CrakBox/runs"), "Glob"))
        self.assertIsNone(self.call(dict(pattern="x", path="CrakBox/mapper"), "Grep"))

    def test_allows_ordinary_work_for_the_main_session(self):
        self.assertIsNone(self.call(dict(command="CrakBox/mapper.py verdict resnet18 s4b1:1")))
        self.assertIsNone(self.call(dict(file_path="CrakBox/CONSTITUTION.md"), "Read"))

    def test_writing_about_a_path_is_not_reaching_it(self):
        self.assertIsNone(self.call(dict(file_path="CrakBox/LOG.md", new_string="a note on cache/map18/sealed", old_string="x"), "Edit"))
        self.assertIsNotNone(self.call(dict(file_path="runs/x/sealed_keys/key.json", content="{}"), "Write"))

    def test_a_judge_touches_only_its_packets_and_answers(self):
        role = dict(agent_type="map-judge")
        self.assertIsNone(self.call(dict(file_path="CrakBox/runs/packets_ab12/item_1.png"), "Read", **role))
        self.assertIsNone(self.call(dict(file_path="CrakBox/runs/answers_ab12/item_1.json"), "Write", **role))
        for path in ("CrakBox/runs/resnet18/state.db", "CrakBox/CONSTITUTION.md", "research/clean/curves/map_agents/resnet18/cards2/east.jsonl"):
            self.assertIsNotNone(self.call(dict(file_path=path), "Read", **role), path)
        self.assertIsNotNone(self.call(dict(command="ls"), "Bash", **role))

    def test_a_reader_shell_runs_only_the_mapper(self):
        role = dict(agent_type="map-reader")
        ok = "research/clean/curves/.venv/bin/python CrakBox/mapper.py ev resnet18 s4b1:1 reader deep s4b1:1"
        self.assertIsNone(self.call(dict(command=ok), **role))
        for bad in ("cat CrakBox/CONSTITUTION.md", "python3 -c 'print(1)'", ok + "; cat x", ok + " | tee y", "python3 CrakBox/mapper.py verdict resnet18 s4b1:1", "python3 CrakBox/mapper.py judge-answer resnet18 i j f"):
            self.assertIsNotNone(self.call(dict(command=bad), **role), bad)
        for path in ("CrakBox/runs/packets_ab/item_1.png", "CrakBox/runs/resnet18/state.db"):
            self.assertIsNotNone(self.call(dict(file_path=path), "Read", **role), path)

    def test_a_drawing_script_is_read_before_it_runs(self):
        d = tempfile.mkdtemp()
        folder = os.path.join(d, "CrakBox", "runs", "drawn")
        os.makedirs(folder)
        clean = os.path.join(folder, "lines.py")
        with open(clean, "w") as f:
            f.write("from PIL import Image\nImage.new('RGB', (9, 9)).save('x.png')\n")
        dirty = os.path.join(folder, "peek.py")
        with open(dirty, "w") as f:
            f.write("import glob\nprint(glob.glob('*'))\n")
        role = dict(agent_type="map-critic", cwd=d)
        self.assertIsNone(self.call(dict(command=f"python3 {clean}"), **role))
        self.assertIsNotNone(self.call(dict(command=f"python3 {dirty}"), **role))
        self.assertIsNotNone(self.call(dict(command=f"python3 {os.path.join(folder, 'missing.py')}"), **role))


class TestCoreNamesNoNetwork(unittest.TestCase):
    def test_scan(self):
        bad = re.compile(r"resnet|convnext|inception|imagenet|\bphoto|import torch|\(skip\)|clean.{0,3}curves|class_names", re.I)
        files = [os.path.join(ROOT, "mapper", n) for n in os.listdir(os.path.join(ROOT, "mapper")) if n.endswith(".py")] + [os.path.join(ROOT, "mapper.py")]
        self.assertGreater(len(files), 7)
        for path in files:
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    self.assertIsNone(bad.search(line), f"{path}:{i}: {line.strip()}")


if __name__ == "__main__":
    unittest.main(verbosity=1)
