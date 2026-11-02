# Brief: does TQ31's hard network pass the essay? (card 3-37, line I5)

Your job: read the hard network the way the essay reads networks, and judge whether it is clean. Be honest in either direction. A verdict specific enough to lose is the goal. The main session expects it to fail, so look hard for what would make it pass, and report what you find.

Network: Proposition TQ31 and Definition DQ18 in `COARSE-THEORY.md`. Build small instances with `model/tq31_anytail.py` and `model/tq31_trained.py` as a start (copy code to your folder). Read the actual tables and units on a few graphs.

Principles: `map_agents/essay.md` (lines 33-35, 69-105, 145-161, 225-251, 399-403, 465-482 first). Part A's Definitions D42, D45, D20 in `COARSE-THEORY.md`. The working notes of what real networks do: `rounds/round4/s4/agent3/REPORT.md`.

Questions.
1. Compute the error of each principle of D42 and D45 that can be computed on this network. Say which ones have no meaning here.
2. Read it as a network. What is an area here? What does a unit do? What is the rest of the class table made of? Is it "a fold", "a judgment", "a family", a "sheltered place for specialists", or is it just a lookup? Does coarsening make sense: is there anything a coarser version of it would be?
3. Which sharp differences are there between this network's table and what `rounds/round4/s4/agent3/REPORT.md` reads on ResNet-18: a row's head holds a family and several unrelated directed side links?
4. The essential question: can a graph's hard pattern be carried by a network that does pass the principles? Try to build one, on a small instance: same hard table, or a table with the same hardness, but with units that read as clean. If you cannot, state what stops you. If a principle separates hard tables from clean ones, state it as a candidate claim: what numbers it names, what its error is, that the error is 0 exactly when it holds, that it names no build, size or zoom, and whether TQ30's and TQ31's graphs have it at error 0 or not. Measure the same error on ResNet-18's real read if it can be computed on rows of `cache/map` (photo rows, not only the class-row table: the 20-photo table holds a lot of photo noise, and half-tables differ by rms about 3).

Rules: Python by `.venv/bin/python`, with first line `# serves: TQ31` in any script. Inline Python may not import torch. Work only in `rounds/round4/i5/`. Do not edit the paper, `DECISIONS.md`, `LADDER.md`, `ROADMAP.md`, or existing scripts. Stop at about 300000 tokens. Do not claim more than you measured.

Hand in `rounds/round4/i5/REPORT.md`, in short plain sentences, and return its text as your reply.
