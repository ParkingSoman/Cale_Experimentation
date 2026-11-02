# Brief for a neuron study agent

Goal: an accurate coarse-to-fine map of ConvNeXt-Tiny (ImageNet). You deeply understand a few neurons and write one
card per neuron. Read `research/clean/MAP-STATE.md` first (5 min). Work in `research/clean/vision2/` (Python, MLX).

## Inputs you have
- Deep views: `research/clean/map/deep/b{block}_n{neuron}.png` + `.json`. Rows: crops at 100/75/50/25/10% of the
  neuron's max (does the meaning hold or radiate as it weakens?), strongest negative, half negative, heat maps on the
  top-6 full images (red = positive, blue = negative), and the edit fingerprint in the title (response after edit /
  before, median over its top-20 images: flip, rot90, rot180, gray, blur1, blur3, shuffle 2x2/4x4/8x8 tiles, hue
  roll, invert, zoomout, zoomin). 1.0 = no change, ~0 = killed, >1 = stronger.
- Upstream / downstream: `results/vision-s2-cards.json` ("up": exact direct inputs as [block, neuron, share], top 5
  positive then 2 most negative; "down": readers [block, neuron, strength]). Candidate B (territory parent) for each
  anchor: `results/vision-s3-lineup-key.json` -> "cands" -> "B".
- Older single-panel views: `research/clean/vision2/figs/s2/*.png`.
- To view any other neuron: `cd research/clean/vision2 && python view.py BLOCK,NEURON [...]` (~30 s for a few).

## Method (essay: a neuron is a distinction; describe what its inputs share, not nouns)
1. Look at every row of the deep view. Write the positive pole, negative pole, and the distinction in one sentence
   each. Avoid noun lists. Ask "if I could not use nouns, how would I describe it?"
2. Write ONE testable prediction before testing (e.g. "responds to vertical thin lines regardless of color; a
   synthetic vertical line on gray drives it to >50% of max, an oblique one does not").
3. Test it. Cheap tests that work: synthetic images drawn with PIL (lines at angles, stripes at frequencies, color
   patches, circles vs points, V vs O), class subsets of the 50k val images, crops, edits. Keep each test under 2
   minutes of compute. Put scripts in `research/clean/map/tests/`. Report numbers.
4. Parent: view the top inputs and candidate B. Which one is a COARSER version whose region contains this neuron's
   job (either pole)? Name the link type: refinement of the same pattern / color dial -> thing of that color /
   near-twin relay / child on the parent's off side / part -> whole / sibling competitor (strongest negative input) /
   other (describe). "None found" is a valid answer.
5. Children: view the readers. Are they finer splits inside this neuron's region?
6. Confidence: high / medium / low, and what is still unclear.

## Output
One card per neuron: `research/clean/map/cards/b{block}_n{neuron}.md` with sections: Distinction, Evidence (bands,
heat, edits), Test (prediction, result), Parent (+ link type), Children, Coarse-to-fine verdict (is it a finer split
inside its parent's region: yes / partly / no), Confidence. Under 300 words per card. Plain English, short sentences.

## Rules
- Never read `~/Cale-holdout`. Do not edit files outside `research/clean/map/` and `research/clean/vision2/` (you
  may add new scripts in vision2 or map/tests; do not change existing scripts).
- Do not study the held-out neurons: 2 n194, 9 n851, 12 n122, 15 n1613.
- Your final reply: under 150 words, the card paths and one line per neuron (distinction + parent + confidence).
