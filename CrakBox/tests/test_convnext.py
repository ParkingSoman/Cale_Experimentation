"""The second network through the same core, with only the adapter different. Run: python CrakBox/tests/test_convnext.py"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from mapper import judge, sheets, workflow  # noqa: E402
from networks import get  # noqa: E402

NET = get("convnext_tiny")


class TestConvNeXt(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(NET.layers(), ["c17"])
        self.assertEqual(len(NET.nodes("c17")), 3072)

    def test_peak_agrees_with_the_field(self):
        v, p = NET.peaks("c17:100", "pos", "read")
        top = np.argsort(-v)[:16]
        self.assertLess(np.abs(NET.value_at("c17:100", "pos", "read", top, p[top]) - v[top]).max() / v.max(), 2e-3)
        v, p = NET.peaks("c17:100", "neg", "read")
        top = np.argsort(-v)[:16]
        self.assertLess(np.abs(NET.value_at("c17:100", "neg", "read", top, p[top]) - v[top]).max() / v.max(), 2e-3)

    def test_exact_sum(self):
        v, p = NET.peaks("c17:100", "pos", "read")
        top = np.argsort(-v)[:16]
        D, names, base, true = NET.inputs("c17:100", "read", top, p[top])
        self.assertEqual(D.shape, (16, 768))
        self.assertEqual(len(names), 768)
        self.assertLess(np.abs(base + D.sum(1) - true).max() / np.abs(true).mean(), 2e-3)

    def test_sheet_features_and_router(self):
        f = sheets.features(NET, "c17:100")
        self.assertIn("output_effect", f)
        self.assertGreater(f["output_effect"], 0)
        print(f)
        lib = workflow.load(os.path.join(ROOT, "workflows"))
        self.assertIsNotNone(workflow.route(f, lib))

    def test_real_sheet_and_blind_packet(self):
        tiles, rows = judge.build_pole_sheet(NET, "c17:100", "pos", "read", np.random.default_rng(0))
        self.assertEqual(len(rows), 36)
        self.assertLess(max(r["share"] for r in rows if r["kind"] != "fire"), judge.QUIET)

    def test_leak_patterns(self):
        self.assertTrue(judge.leaks(dict(words=dict(job="this is neuron 100")), [], NET.leak_patterns()))
        self.assertTrue(judge.leaks(dict(words=dict(job="it reads c17s:45")), [], NET.leak_patterns()))
        self.assertFalse(judge.leaks(dict(words=dict(job="a striped cat on grass")), [], NET.leak_patterns()))

    def test_evidence_command(self):
        command, cwd = NET.evidence("deep", ["c17:100"])
        self.assertTrue(command[1].endswith("ev.py"))
        with self.assertRaises(ValueError):
            NET.evidence("rm", [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
