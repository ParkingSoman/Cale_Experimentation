"""Unit tests for cale/choose.py. Run: .venv/bin/python -m cale.tests.choose"""
from cale import choose
from nest.levels3n import load_levels

LEVELS = load_levels()
N = 35
FULL, SMALL = LEVELS[0], LEVELS[-1]
assert FULL["id"] == "e4b" and SMALL["id"] == "e2b"

GB = 1e9
PROF = {
    "margin": 1.0, "work_bytes": 0.4 * GB, "base_bytes": 1.2 * GB, "disk_bps": 2 * GB,
    "layer_bytes": [[0, 0.028 * GB, 0.066 * GB]] * N, "layer_read": [0.0566 * GB] * N,
    "levels": {l["id"]: {"decode_tps": 30 + 0.5 * (100 - l["pct"]), "prefill_tps": 600 + 10 * (100 - l["pct"])} for l in LEVELS},
}
ALL = [2] * N
SMALLEST = choose.need(SMALL)


def test_need():
    assert choose.need(FULL) == [2] * N
    assert choose.need(SMALL).count(0) == 5


def test_union():
    assert choose.union_need([SMALL, FULL]) == [2] * N
    assert choose.union_need([SMALL]) == SMALLEST


def test_disk_bytes():
    assert choose.disk_bytes(PROF, FULL, ALL) == 0
    assert choose.disk_bytes(PROF, SMALL, SMALLEST) == 0
    d = choose.disk_bytes(PROF, FULL, SMALLEST)
    assert d == PROF["layer_read"][0] * sum(1 for a, b in zip(choose.need(FULL), SMALLEST) if a > b)


def test_deadline_picks_largest_that_fits():
    loose = choose.for_deadline(LEVELS, PROF, ALL, 100, 100, deadline_s=60)
    assert loose["level"]["id"] == "e4b" and loose["met"]
    tight = choose.for_deadline(LEVELS, PROF, ALL, 100, 100, deadline_s=choose.predict_s(PROF, SMALL, 100, 100) + 0.01)
    assert tight["met"] and tight["level"]["pct"] < 100


def test_deadline_miss_returns_fastest():
    r = choose.for_deadline(LEVELS, PROF, ALL, 100, 1000, deadline_s=0.01)
    assert not r["met"] and r["level"]["id"] == "e2b"


def test_disk_needs_permission():
    no = choose.for_deadline(LEVELS, PROF, SMALLEST, 100, 100, deadline_s=60, allow_disk=False)
    assert no["level"]["id"] == "e2b" and no["disk_bytes"] == 0
    yes = choose.for_deadline(LEVELS, PROF, SMALLEST, 100, 100, deadline_s=60, allow_disk=True)
    assert yes["level"]["id"] == "e4b" and yes["disk_bytes"] > 0


def test_disk_time_counts():
    slow = dict(PROF, disk_bps=0.001 * GB)
    r = choose.for_deadline(LEVELS, slow, SMALLEST, 100, 100, deadline_s=10, allow_disk=True)
    assert r["level"]["id"] != "e4b"


def test_memory_goal():
    ram = 24 * GB
    assert choose.for_memory(LEVELS, PROF, ram, 50)["level"]["id"] == "e4b"
    small = choose.model_bytes(PROF, SMALLEST) + PROF["work_bytes"]
    r = choose.for_memory(LEVELS, PROF, ram, 100 * small / ram)
    assert r["met"] and r["level"]["id"] == "e2b"
    miss = choose.for_memory(LEVELS, PROF, ram, 1)
    assert not miss["met"] and miss["level"]["id"] == "e2b"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
