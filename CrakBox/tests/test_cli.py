"""The mapper command, run as an agent runs it: one process per command. On the fake network from queued to understood and to open, then a short real-network check.
Run: python CrakBox/tests/test_cli.py"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GUARD = os.path.join(ROOT, "guards", "sealed_guard.py")
WORKFLOW = """---
id: clear
version: 1
priority: 10
trigger: largest > 0
step_limit: 2
exits: claim, open
---
Read the sheet.
"""


AUTHOR = dict(top="reader-a", bot="reader-b", raw="reader-c", s4b1i="reader-a")


def card_for(node, parent="bot:1"):
    return dict(node=node, distinction=f"draws line number {node.split(':')[1]} against the rest", positive="strong side", negative="weak side", shape="detector",
                main_parent=dict(node=parent, why="largest input", sure=.7) if parent else None, readers="the next layer uses it",
                statements=[dict(text="fires on stripes", mark="observed")], prediction="a drawn stripe fires it", near_miss="a dotted line",
                unread="nothing yet" if parent else "no parent: it reads the raw input")


class Run:
    def __init__(self, net="fake"):
        self.dir = tempfile.mkdtemp()
        self.net = net
        os.makedirs(os.path.join(self.dir, "wf"))
        with open(os.path.join(self.dir, "wf", "clear.md"), "w") as f:
            f.write(WORKFLOW)
        self.env = dict(os.environ, MAPPER_RUNS=os.path.join(self.dir, "runs"), MAPPER_WORKFLOWS=os.path.join(self.dir, "wf"), MAPPER_TEST="1")
        self.files = 0

    def __call__(self, *args):
        return subprocess.run([sys.executable, "-W", "ignore", os.path.join(ROOT, "mapper.py"), args[0], self.net, *args[1:]], capture_output=True, text=True, env=self.env)

    def ok(self, *args):
        done = self(*args)
        assert done.returncode == 0, f"{args}: {done.stderr}"
        return done.stdout

    def file(self, data):
        self.files += 1
        path = os.path.join(self.dir, f"f{self.files}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def key(self):
        with open(os.path.join(self.dir, "runs", self.net, "sealed_keys", "key.json")) as f:
            return json.load(f)

    def scope(self, nodes):
        return self.file(dict(nodes=nodes, goal="all classes", control_goal="random classes", threshold=.01, firings="strong firings and both poles"))

    def read_to_claim(self, node, parent="bot:1"):
        self.ok("sheet", node)
        self.ok("route", node)
        self.ok("read", node)
        self.ok("predict", node, AUTHOR[node.split(":")[0]], "a drawn stripe fires it")
        self.ok("card", self.file(card_for(node, parent)), AUTHOR[node.split(":")[0]])

    def attack(self, node, broke=False, critic="critic-a"):
        self.ok("critic-plan", node, critic, self.file(dict(prediction="p", attack="a")))
        line = self.ok("ev", node, "critic", "probe", "x").split("\n")[0]
        step = int(line.split()[1])
        self.ok("critic", node, critic, self.file(dict(steps=[step], observed="o", broke=broke)))

    def answer(self, item, knows, judges=("j1", "j2")):
        k = self.key()[item]
        for n, j in enumerate(judges):
            if k["test"] in ("parent", "reader"):
                pick = k["true_number"] if knows else (k["true_number"] % 8) + 1
                self.ok("judge-answer", item, j, self.file(dict(pick=pick)))
            else:
                shares = np.array(k["shares"])
                # a judge who does not know ranks the weakest first. A random ranking can pass by chance, and item ids are random, so this test failed on some runs
                order = np.argsort(-shares) if knows else np.argsort(shares)
                self.ok("judge-answer", item, j, self.file(dict(ranked=[int(x) + 1 for x in order[:12]])))


class TestWholeLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r = Run()
        cls.r = r
        top = [f"top:{i}" for i in range(14)]
        bot = [f"bot:{i}" for i in range(12)]
        cls.nodes = top + bot + ["raw:0"]
        r.ok("init", r.scope(cls.nodes))
        with_smoke = os.path.join(r.dir, "runs", "fake", "state.db")
        sys.path.insert(0, ROOT)
        from mapper.state import Store
        s = Store(with_smoke)
        s.set_smoke_passed("test")
        s.close()
        for n in top + bot:
            r.read_to_claim(n)
        r.read_to_claim("raw:0", parent=None)
        # pass marks come from wrong items, answered by judges who cannot know
        for i in range(24):
            for item in r.ok("judge-build", f"top:{5 + i % 9}", "--wrong", "--pole", "pos").split():
                r.answer(item, False)
        cls.calibration = json.loads(r.ok("judge-calibrate"))

    def test_1_calibration_has_held_out_items(self):
        self.assertTrue(self.calibration["held_out"])
        self.assertGreaterEqual(self.calibration["wrong_items"], 20)
        self.assertLessEqual(self.calibration["false_pass"], .1)

    def test_2_sheets_differ_between_commands(self):
        layouts = {tuple(k["kinds"]) for k in self.r.key().values() if "kinds" in k}
        self.assertGreater(len(layouts), 20)

    def test_3_no_judge_until_the_critic_held(self):
        r = self.r
        self.assertNotEqual(r("judge-build", "top:1").returncode, 0)  # claimed, not attacked
        self.assertNotEqual(r("verdict", "top:1").returncode, 0)

    def judge_all(self, node, knows=True, poles=("pos", "neg"), reader=True, parent=True):
        r = self.r
        items = []
        for pole in poles:
            items += r.ok("judge-build", node, "--pole", pole).split()
        if parent:
            items += r.ok("judge-link", node, "parent").split()
        if reader:
            items += r.ok("judge-link", node, "reader").split()
        key = r.key()
        real = [i for i in items if not key[i]["wrong_card"]]
        wrong = [i for i in items if key[i]["wrong_card"]]
        for i in wrong:
            r.answer(i, False)
        return real, knows

    def test_4_a_right_card_ends_understood_and_writes_its_claim(self):
        r = self.r
        node = "top:1"
        r.attack(node)
        real, _ = self.judge_all(node)
        waiting = json.loads(r.ok("verdict", node))
        self.assertEqual(waiting["state"], "judging")  # nothing answered yet
        first = real[0]
        r.answer(first, True, judges=("j1",))
        self.assertEqual(json.loads(r.ok("verdict", node))["state"], "judging")  # one answer settles nothing
        r.answer(first, True, judges=("j2",))
        for item in real[1:]:
            r.answer(item, True)
        out = json.loads(r.ok("verdict", node))
        self.assertEqual(out["state"], "understood", out)
        self.assertEqual(out["tests"], ["critic", "parent", "pole:neg", "pole:pos", "reader"])
        st = json.loads(r.ok("status"))
        self.assertGreaterEqual(st["understood"], 1)
        from mapper.state import Store
        s = Store(os.path.join(r.dir, "runs", "fake", "state.db"))
        self.assertEqual(s.claims()[0]["node"], node)
        self.assertTrue(s.claims()[0]["estimate"])
        s.close()

    def test_5_a_wrong_answering_judge_ends_open_with_a_suspect(self):
        r = self.r
        node = "top:2"
        r.attack(node)
        real, _ = self.judge_all(node)
        for item in real:
            r.answer(item, False)
        out = json.loads(r.ok("verdict", node))
        self.assertEqual(out["state"], "open")
        self.assertTrue(out["failed"])
        st = json.loads(r.ok("status"))
        self.assertIn(node, [o["node"] for o in st["open"]])
        self.assertEqual([o for o in st["open"] if o["node"] == node][0]["suspect"], "explanation")

    def test_6_a_split_waits_for_a_third(self):
        r = self.r
        node = "top:3"  # a spike node: its sheet is refused
        r.attack(node)
        refused = r("judge-build", node, "--pole", "pos")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("ShortPool", refused.stderr)
        node = "top:4"
        r.attack(node)
        real, _ = self.judge_all(node)
        for item in real[1:]:
            r.answer(item, True)
        r.answer(real[0], True, judges=("j1",))
        r.answer(real[0], False, judges=("j2",))
        out = json.loads(r.ok("verdict", node))
        self.assertEqual(out["state"], "judging", out)  # a split settles nothing
        self.assertEqual(len(out["waiting_for"]), 1)
        r.answer(real[0], True, judges=("j3",))
        self.assertEqual(json.loads(r.ok("verdict", node))["state"], "understood")

    def test_7_a_judge_cannot_be_the_author_or_change_an_answer(self):
        r = self.r
        node = "bot:2"
        r.attack(node)
        item = r.ok("judge-build", node, "--pole", "pos").split()[0]
        k = r.key()[item]
        shares = np.array(k["shares"])
        order = np.argsort(-shares)
        ans = r.file(dict(ranked=[int(x) + 1 for x in order[:12]]))
        barred = r("judge-answer", item, "reader-b", ans)
        self.assertNotEqual(barred.returncode, 0)
        self.assertIn("cannot judge", barred.stderr)
        r.ok("judge-answer", item, "j1", ans)
        again = r("judge-answer", item, "j1", ans)
        self.assertNotEqual(again.returncode, 0)
        bad = r.file(dict(ranked=[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        self.assertIn("BadAnswer", r("judge-answer", item, "j2", bad).stderr)

    def test_8_a_broken_card_goes_back_and_needs_a_new_attack(self):
        r = self.r
        node = "bot:5"
        r.attack(node, broke=True)
        self.assertNotEqual(r("judge-build", node).returncode, 0)
        r.ok("card", r.file(dict(card_for(node), distinction="better words on the second try")), "reader-a")
        self.assertNotEqual(r("judge-build", node).returncode, 0)
        r.attack(node)
        self.assertEqual(r("judge-build", node, "--pole", "pos").returncode, 0)

    def test_9_a_node_with_no_inputs_can_be_understood(self):
        r = self.r
        node = "raw:0"
        r.attack(node)
        items = r.ok("judge-build", node, "--pole", "pos").split()
        items += r.ok("judge-link", node, "reader").split()
        for item in items:
            r.answer(item, not r.key()[item]["wrong_card"])
        self.assertEqual(json.loads(r.ok("verdict", node))["state"], "understood")

    def test_a_leaky_card_cannot_reach_a_judge(self):
        r = self.r
        node = "bot:4"
        r.attack(node, broke=True)
        r.ok("card", r.file(dict(card_for(node), distinction="built from top:1 as far as I can see")), "reader-a")
        r.attack(node)
        self.assertIn("Leak", r("judge-build", node, "--pole", "pos").stderr)

    def test_packets_sit_in_a_neutral_folder(self):
        folders = glob.glob(os.path.join(self.r.dir, "runs", "packets_*"))
        self.assertEqual(len(folders), 1)
        self.assertNotIn("fake", os.path.basename(folders[0]))
        for path in glob.glob(os.path.join(folders[0], "*.json"))[:20]:
            with open(path) as f:
                text = f.read()
            for word in ("top:", "bot:", "fake", "reader-", "wrong", "share"):
                self.assertNotIn(word, text.replace(self.r.dir, ""), path)


class TestRevise(unittest.TestCase):
    def test_an_author_corrects_before_a_critic_has_it(self):
        r = Run()
        r.ok("init", r.scope(["top:0"]))
        r.read_to_claim("top:0")
        fixed = r.file(dict(card_for("top:0"), distinction="better words"))
        self.assertNotEqual(r("revise", fixed, "someone-else").returncode, 0)
        r.ok("revise", fixed, "reader-a")
        r.ok("critic-plan", "top:0", "critic-a", r.file(dict(prediction="p", attack="a")))
        self.assertIn("critic already has", r("revise", fixed, "reader-a").stderr)


class TestLimits(unittest.TestCase):
    def test_bad_scope_and_step_limit(self):
        r = Run()
        self.assertNotEqual(r("init", r.file(dict(nodes=["top:0"]))).returncode, 0)
        self.assertNotEqual(r("init", r.scope(["top:99"])).returncode, 0)
        r.ok("init", r.scope(["top:0", "top:1"]))
        refused = r("route", "top:0")
        self.assertIn("no sheet", refused.stderr)
        r.ok("sheet", "top:0")
        self.assertEqual(r.ok("route", "top:0").strip(), "clear")
        r.ok("read", "top:0")
        self.assertIn("prediction on file", r("ev", "top:0", "reader", "probe", "x").stderr)
        self.assertIn("no prediction on file", r("card", r.file(card_for("top:0")), "reader-a").stderr)
        r.ok("ev", "top:0", "reader", "deep", "x")
        r.ok("ev", "top:0", "reader", "deep", "x")
        self.assertIn("2 steps used", r("ev", "top:0", "reader", "deep", "x").stderr)
        r.ok("ev", "top:0", "critic", "deep", "x")
        self.assertNotEqual(r("ev", "top:0", "reader2", "deep", "x").returncode, 0)

    def test_verdict_refuses_when_wrong_cards_pass_too_often(self):
        r = Run()
        r.ok("init", r.scope(["top:0"]))
        keys = os.path.join(r.dir, "runs", "fake", "sealed_keys")
        os.makedirs(keys, exist_ok=True)
        with open(os.path.join(keys, "marks.json"), "w") as f:
            json.dump(dict(marks=dict(center=0, edge=0, far=0), false_pass=.5), f)
        r.ok("sheet", "top:0")
        r.ok("route", "top:0")
        r.ok("read", "top:0")
        r.ok("predict", "top:0", "reader-a", "it fires on stripes")
        r.ok("card", r.file(card_for("top:0")), "reader-a")
        r.attack("top:0")
        r.ok("judge-build", "top:0", "--pole", "pos")
        self.assertIn("fix the test", r("verdict", "top:0").stderr)

    def test_no_marks_no_verdict(self):
        r = Run()
        r.ok("init", r.scope(["top:0"]))
        r.ok("sheet", "top:0")
        r.ok("route", "top:0")
        r.ok("read", "top:0")
        r.ok("predict", "top:0", "reader-a", "it fires on stripes")
        r.ok("card", r.file(card_for("top:0")), "reader-a")
        r.attack("top:0")
        r.ok("judge-build", "top:0", "--pole", "pos")
        self.assertIn("no pass marks", r("verdict", "top:0").stderr)


class TestRealNetwork(unittest.TestCase):
    def test_a_node_through_the_command(self):
        r = Run("resnet18")
        r.ok("init", r.scope(["s4b1i:110", "s4b1:175"]))
        status = json.loads(r.ok("status"))
        self.assertEqual((status["understood"], status["scope"]), (0, 2))
        r.ok("sheet", "s4b1i:110")
        self.assertEqual(r.ok("route", "s4b1i:110").strip(), "clear")
        r.ok("read", "s4b1i:110")
        out = r.ok("ev", "s4b1i:110", "reader", "inputs", "s4b1i:110")
        self.assertIn("inputs to reach 50%", out)
        self.assertNotEqual(r("ev", "s4b1i:110", "reader", "rm", "x").returncode, 0)


class TestHookEntry(unittest.TestCase):
    """The guard, as the harness calls it: JSON on stdin, exit 2 blocks."""

    def call(self, payload):
        return subprocess.run([sys.executable, GUARD], input=json.dumps(payload), capture_output=True, text=True).returncode

    def test_exit_codes(self):
        self.assertEqual(self.call(dict(tool_name="Read", tool_input=dict(file_path="cache/map18/sealed/y.npy"))), 2)
        self.assertEqual(self.call(dict(tool_name="Bash", tool_input=dict(command="cat runs/x/sealed_keys/key.json"))), 2)
        self.assertEqual(self.call(dict(tool_name="Read", tool_input=dict(file_path="CrakBox/README.md"))), 0)

    def test_bad_input_does_not_block(self):
        done = subprocess.run([sys.executable, GUARD], input="not json", capture_output=True, text=True)
        self.assertEqual(done.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
