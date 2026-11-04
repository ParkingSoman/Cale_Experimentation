# CrakBox

CrakBox cracks the black box. It maps a neural network until every part in scope is understood, and it starts small and grows. Shonu named it on 2026-11-04.

It is one tool, strict enough to not flounder and flexible enough to adapt to each network. Read in this order.

| File | Holds |
|---|---|
| `ROADMAP.md` | The milestones and the sessions. **Start here** |
| `LOG.md` | What each round did, in plain words. The end of it is the latest state |
| `CONSTITUTION.md` | The fixed rules |
| `PRINCIPLES.md` | What the essay says about mapping. The deep dive |

Plans: `~/.claude/plans/im-realizing-this-neural-quirky-crystal.md` is the current one. `~/.claude/plans/im-realizing-now-that-fluffy-swing.md` built the tool.

**Cale line.** Each network that Cale resizes needs a map, so the cost of a map is paid once per model. The map's levels become the zoom's groupings. This route is a reading of the evidence, not a proven result.

## Status

| Milestone | State |
|---|---|
| 0: set up CrakBox | Tests pass. **First action of the next session, before any agent runs:** remove the link at `research/mapping`, because the guard's folder rules may not cover it. `ROADMAP.md` has the check |
| 1: the spaCy word tagger to 100% | Not started. `ROADMAP.md` lists its sessions |
| ResNet-18 scope v1 | **Parked at 2 of 184**, and nothing is lost. It returns as milestone 3 |

## What the tool has done so far

| Step of the first plan | Work | State |
|---|---|---|
| 1 | Essay study and constitution | Done. Shonu approved both on 2026-11-03 |
| 2 | Core tool, with tests | Done on the fake network. The adapter and evidence calls are tested on the real one. See `LOG.md` for what stays open |
| 3 | Import the 135 old cards, calibrate pass marks, first bar | Done. 0 of 134 understood. The judge separates right words from wrong ones |
| 4 | Workflows v0 | Done. 10 workflows and 2 motif files |
| 5 | Smoke run on 10 nodes | Done. 8 judged, 0 understood |
| 6 | Positive control on ConvNeXt-Tiny | Done. No gap against ResNet-18, so the limit sits in the tests or the cards |
| 7 | Scope v1 of ResNet-18 | Parked at 2 of 184. Stage 4 is read. A test fixed in advance failed its mark. The bar excludes the negative pole |
| 8 | Rinse-and-repeat test | Moved to milestone 2 |

## Run the checks

Run these from the top of the repo.

```
research/clean/curves/.venv/bin/python CrakBox/tests/test_quotes.py
research/clean/curves/.venv/bin/python -W ignore CrakBox/tests/test_core.py
research/clean/curves/.venv/bin/python -W ignore CrakBox/tests/test_cli.py
research/clean/curves/.venv/bin/python -W ignore CrakBox/tests/test_adapter.py
research/clean/curves/.venv/bin/python -W ignore CrakBox/tests/test_convnext.py
```

| File | Tests | Needs |
|---|---|---|
| `test_quotes.py` | essay quotes, constitution names no network | nothing |
| `test_core.py` | store, cards, workflows, judge, claims, guard | no network, about 1 second |
| `test_cli.py` | `mapper.py`, one process per command, whole loop | fake network, plus one real node, about 2 minutes |
| `test_adapter.py` | exact sums, readers, sheets, sealed packets | the real network, about 2 minutes |
| `test_convnext.py` | the second network through the same core | the real network. Run it in its own process |

## What sits outside this folder

| Item | Where | Note |
|---|---|---|
| Agent roles: `map-reader`, `map-critic`, `map-judge`, `map-scientist` | `.claude/agents/` | Out of git. A new checkout needs them copied |
| Guard setting | `.claude/settings.json` | Out of git. A new checkout needs it copied. The guard itself is `guards/sealed_guard.py` |
| ResNet-18 evidence tools, and the Python setup | `research/clean/curves/` | Milestone 3 needs the tools. The adapter wraps them |
| The essay | `research/clean/curves/map_agents/essay.md` | |
