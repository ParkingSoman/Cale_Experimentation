# S8 blind study brief (read research/clean/map/STUDY-BRIEF.md first for method, tools and card format)

You study 5 neurons. For each one:
1. Understand it: distinction (positive pole, negative pole, one sentence), ONE prediction written before testing,
   a test with numbers (synthetic images, class subsets, edits; under 2 min compute each). Deep views of the neuron
   and of all its candidates are in research/clean/map/deep/.
2. Blind lineup: research/clean/map/s8/candidates.json gives C1..C9 -> [block, neuron] for your neuron. Sources are
   hidden. Decide which candidate (or none) is a COARSER version whose region contains this neuron's job (either
   pole). You may test candidates directly (is it on at the neuron's top spots, synthetic stimuli on both). Rank
   your top 3, best first, or "none". Name the link type (refinement / near-twin relay / color dial -> thing /
   child on the parent's off side / part -> whole / sibling competitor / position relay / other).
3. Card: research/clean/map/cards/b{block}_n{neuron}.md (STUDY-BRIEF format) + a line "BLIND PICK: C? (C?, C?)" or
   "BLIND PICK: none", + a line "LINK: <type>", + "VERDICT: yes / partly / no" (finer split inside a parent region).
4. Append your picks to research/clean/map/s8/picks_<band>.json as {"b_n": ["C?","C?","C?"] or []}.

Blind rules (strict): do NOT open results/vision-s8-key.json, results/vision-s6-key.json, results/vision-s5-rules.json,
research/clean/map/network_map.json, or the S5-S8 sections of research/clean/VISION-LOG.md. No whole-model parent
ranking; judge only the 9 given candidates. Never read ~/Cale-holdout. Other agents run tests on the same Mac at the
same time; keep runs short.
Final reply under 150 words: one line per neuron (distinction, pick, link, confidence).
