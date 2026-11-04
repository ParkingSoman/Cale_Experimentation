"""The ResNet-18 adapter, against the real network and the real stores. Run: python CrakBox/tests/test_adapter.py"""
import json
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mapper import claims, judge, sheets  # noqa: E402
from networks import get  # noqa: E402

NET = get("resnet18")
KINDS = ["s4b1:179", "s4b1i:179", "s4b0:179", "s4b0i:179", "s3b1:11", "s3b1i:11", "s3b0:11", "s2b1:9", "s1b1:7", "s1b0:3", "s1b0i:3"]


def top(node, split="read", count=16, pole="pos"):
    v, p = NET.peaks(node, pole, split)
    o = np.argsort(-v)[:count]
    return o, p[o], v[o]


class TestAdapter(unittest.TestCase):
    def test_shape_of_the_network(self):
        self.assertEqual(len(NET.layers()), 17)
        self.assertEqual(sum(len(NET.nodes(x)) for x in NET.layers()), 3904)
        self.assertEqual(NET.layer_of("s4b1i:3"), "s4b1i")

    def test_exact_sum_on_every_kind(self):
        for node in KINDS:
            ex, pos, _ = top(node)
            D, names, base, true = NET.inputs(node, "read", ex, pos)
            self.assertEqual(D.shape, (len(ex), len(names)))
            miss = np.abs(base + D.sum(1) - true).max() / (np.abs(true).mean() + 1e-6)
            self.assertLess(miss, 2e-3, node)

    def test_peak_agrees_with_the_field(self):
        for node in ("s4b1:179", "s4b1i:179", "s2b1:9"):
            ex, pos, v = top(node)
            self.assertLess(np.abs(NET.value_at(node, "pos", "read", ex, pos) - v).max() / v.max(), 2e-3, node)
            ex, pos, v = top(node, pole="neg")
            self.assertTrue((v > 0).all())
            self.assertLess(np.abs(NET.value_at(node, "neg", "read", ex, pos) - v).max() / v.max(), 2e-3, node)

    def test_stem_has_one_pole_and_no_inputs(self):
        self.assertEqual(NET.poles("s0:12"), ("pos",))
        with self.assertRaises(ValueError):
            NET.inputs("s0:12", "read", np.array([0]), np.array([0]))

    def test_readers_match_the_real_layer(self):
        """The largest reader's change, against the real convolution with this node at its mean."""
        import ev2
        import net as rn
        node = "s3b1:11"
        name, signed = NET.readers(node, "read", 1)[0]
        self.assertTrue(name.startswith("s4b0i:"))
        c = int(name.split(":")[1])
        v, _ = NET.peaks(node, "pos", "read")
        ex = np.sort(np.argsort(-v)[:32])
        x = NET._p("read").tensor(ex)
        bl = rn.stages(NET.m)[4][0]
        with torch.no_grad():
            _, o = rn.forward(NET.m, x)
            h = o[(3, 1)]["h"]
            real = bl.bn1(bl.conv1(h))[:, c]
            h2 = h.clone()
            h2[:, 11] = ev2.mean_of("s3b1_hm")[11]
            change = (real - bl.bn1(bl.conv1(h2))[:, c]).flatten(1)
            picked = change.gather(1, change.abs().argmax(1, keepdim=True))[:, 0].mean()
        self.assertLess(abs(float(picked) - signed), 1e-3 + 1e-2 * abs(signed))

    def test_top_layer_is_read_by_the_logits(self):
        self.assertTrue(NET.readers("s4b1:179", "read", 3)[0][0].startswith("logit:"))

    def test_output_effect(self):
        self.assertGreater(NET.output_effect("s4b1:179", "read", top("s4b1:179", count=8)[0]), 0)
        self.assertGreater(NET.output_effect("s4b1i:179", "read", top("s4b1i:179", count=8)[0]), 0)
        self.assertGreater(NET.output_effect("s0:12", "read", top("s0:12", count=8)[0]), 0)

    def test_hand_written_forward_matches_the_real_network(self):
        """A fill to a channel's own constant, written out two ways: the adapter's forward, and the real forward with the head read by hand."""
        import ev2
        import net as rn
        x = NET._p("read").tensor(np.array([0, 1]))
        node = dict(s=4, b=1, inner=False, c=0)
        with torch.no_grad():
            _, o = rn.forward(NET.m, x)
            h = o[(4, 1)]["h"].clone()
            h[:, 0] = ev2.mean_of("s4b1_hm")[0]
            want = rn.head(NET.m, h)
            got = NET._forward_fill(x, node)
        self.assertLess(float((want - got).abs().max()), 1e-4)

    def test_similar_stays_in_the_layer(self):
        got = NET.similar("s4b1i:179", "read", 5)
        self.assertEqual(len(set(got)), 5)
        self.assertTrue(all(n.startswith("s4b1i:") and n != "s4b1i:179" for n in got))

    def test_features(self):
        f = sheets.features(NET, "s4b1i:179")
        for k in ("largest", "on_share", "cliff", "inputs_for_half", "n_inputs", "top_input_share", "output_effect", "neg_ratio"):
            self.assertIn(k, f)
        self.assertEqual(f["n_inputs"], 512)
        print(f)

    def test_claims_on_real_inputs(self):
        ex = np.arange(60)
        pos = np.random.default_rng(0).integers(0, 49, 60)
        names = NET.inputs("s4b1:179", "read", ex[:2], pos[:2])[1]
        full = claims.parent_claim(NET, "s4b1:179", [n for n in names if not n.endswith(" (skip)")], "read", ex, pos)
        self.assertEqual(full["eps"], 0.0)
        one = claims.parent_claim(NET, "s4b1:179", [names[0]], "read", ex, pos)
        self.assertGreater(one["eps"], 0.0)

    def test_real_pole_sheet(self):
        """A sheet from the reading split, so no sealed number enters this test's output. judge.Bench refuses any split but sealed."""
        d = tempfile.mkdtemp()
        for node in ("s4b1:179", "s3b1i:11", "s1b1:7"):
            for pole in NET.poles(node):
                t0 = time.time()
                tiles, rows = judge.build_pole_sheet(NET, node, pole, "read", np.random.default_rng(0))
                judge.draw_sheet(tiles, os.path.join(d, "x.png"))
                self.assertEqual(len(rows), 36)
                fire = [r["share"] for r in rows if r["kind"] == "fire"]
                quiet = [r["share"] for r in rows if r["kind"] != "fire"]
                self.assertLess(max(quiet), judge.QUIET)
                print(f"{node} {pole}: {time.time() - t0:.1f}s, fire {min(fire):.2f} to {max(fire):.2f}, quiet up to {max(quiet):.2f}")

    def test_real_sealed_packet(self):
        """One packet per kind of node, from sealed examples, in a temporary run. It prints timings only, so no sealed content reaches a reader of this output."""
        d = tempfile.mkdtemp()
        bench = judge.Bench(d)
        cards = {n: dict(distinction="a line between two things", positive="one side", negative="the other side") for n in ("s4b1:179", "s4b1:12", "s3b1i:11", "s3b1i:40", "s1b1:7", "s1b1:9", "s0:12", "s0:13")}
        for node, pole in (("s4b1:179", "pos"), ("s3b1i:11", "neg"), ("s1b1:7", "pos"), ("s0:12", "pos")):
            t0 = time.time()
            item = bench.build_pole(NET, node, pole, cards, wrong=False)
            with open(os.path.join(d, "packets", item + ".json")) as f:
                packet = json.load(f)
            self.assertEqual(set(packet), {"item", "test", "sheet", "words", "count", "rank"})
            self.assertEqual(packet["count"], 36)
            self.assertTrue(os.path.exists(packet["sheet"]))
            self.assertEqual(len(bench.key[item]["shares"]), 36)
            print(f"sealed packet for one {NET.layer_of(node)} node: {time.time() - t0:.1f}s")

    def test_a_card_that_leaks_is_refused_on_the_real_layer_names(self):
        bench = judge.Bench(tempfile.mkdtemp())
        cards = {"s4b1:179": dict(distinction="built from s4b1i:243", positive="a", negative="b")}
        with self.assertRaises(judge.Leak):
            bench.build_pole(NET, "s4b1:179", "pos", cards, wrong=False)

    def test_bench_refuses_the_reading_split(self):
        with self.assertRaises(ValueError):
            judge.Bench(tempfile.mkdtemp()).build_pole(NET, "s4b1:179", "pos", {}, split="read")


if __name__ == "__main__":
    unittest.main(verbosity=1)
