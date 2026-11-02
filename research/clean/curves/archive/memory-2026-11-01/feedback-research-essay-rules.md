---
name: feedback-research-essay-rules
description: "How to use the Already Clean essay (axioms, inspiration, read when stuck) and real interp methods; merged 2026-10-21"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2ba42140-21a0-426d-8405-942d9986529c
  modified: 2026-10-21T17:58:25.253Z
---

Rules from Shonu on the "Neural Networks Are Already Clean" essay (https://already-clean.goodfire.com/essay.md) and on interp tools. Raw numbers are always reported honestly.

## 1. The essay is law (2026-10-20)
- First rule: if our results disagree with the essay, treat the essay as the source of truth and assume our setup has an error (definitions, metrics, sign regimes, which units we call siblings, primitive reading tools).
- Update ("Claude Notes", 2026-10-20): the essay's concepts (recursive generalization, gates in front, negative poles, analogizer weak activations, circuit racing, etc.) are base AXIOMS. We build a product, not test the essay. Do not spend steps checking whether an essay concept holds in our model (e.g. "are gates shared?"). Every test asks only whether OUR method, derived from an axiom, works. A bad number blames the method, measurement or setup; then iterate. Lean into the newer concepts with a deliberate bias.
- Reminder (evening, liquid track): "automatically assume the essay as the gospel and work around that idea." Applies to EVERY track, including attention, head size, stream width (research/clean/LIQUID-LOG.md). When a liquid-track gate fails, write the essay-based reading of our setup error in the log and test one fix before moving on.
- Keep the essay's own hedges when quoting (e.g. claim made only for vision models; its attention claim is a "low-confidence prediction").
- **Why:** the essay's story is that apparent mess is the microscope, not the network ("blaming the network for the microscope"). Discrepancies of 2026-10-20 ("value broad/gate specific" failed; siblings co-fired instead of sitting on each other's negative poles) were more likely our definitions.
- **How to apply:** per discrepancy, list concrete setup errors that could cause it, test the most likely one, log it. The essay sets the hypothesis; the data still decides whether the fix worked.

## 2. Essay as inspiration, not recipe (2026-10-20)
- Do not hunt for its training recipe or conclude "the essay retrained coarse layers, so construction needs training". Read ALL sections with equal weight (language-model section and appendix too) and ask which concepts could make coarser neurons from detailed ones. Methods the essay never proposes are the point. "Keyword: application."
- **Why:** my first plan on 2026-10-20 said the essay "closes the construction door" and pivoted to foveated selection. He rejected that as copying, not inspiration.
- **How to apply:** per essay concept write "concept -> application to coarsening -> first cheap test". Catalog: research/clean/ESSAY-IDEAS.md.
- Also (2026-10-20): do not take earlier sessions' negative results as fundamental blockers; they may have implementation errors. Do not re-fix old strategies for their own sake, but when a promising idea depends on an old failure, re-check that failure before giving up on the idea.

## 3. Read the essay when stuck (2026-10-19)
- Whenever stuck, read more of the essay. It already covers realizations we keep rediscovering (neurons doing different jobs than first thought: firing on commas, the word before a feeling, not on the noun).
- **Why:** I kept iterating on methods (recipe v1-v3) when the essay may already explain why our families were wrong (topic bins vs real jobs).
- **How to apply:** at every blue-sky review, or after any failed keeper query, download the raw essay text to the scratchpad and read the relevant sections yourself (not a tool summary). Quote what it says, with hedges, in the log.

## 4. Use the field's real interp methods (2026-10-19)
- Before any step that reads, labels, compares or groups neurons/features, name the interp field's standard method for that question (look it up; interp is a small fast field) and use it, or write why not. Examples: autointerp explain + simulate scoring (not eyeballing top-activating tokens), attribution / activation patching for causal effect, the essay's own reading method (function as a verb, both poles, what it writes). SAEs / transcoders / crosscoders were named at first, but see the map/coarsening rule: neuron-basis tools only, no SAE-era tools.
- **Why:** I have a bias to hand-roll primitive observations. Primitive methods (top-activating examples, raw-neuron basis) are what made networks look messy and polysemantic. The essay came from moving past them.
- **How to apply:** every research plan gets a "Methods check" line per analysis step: field method, tool, source. Pure math checks (span, regression, loss) may stay hand-written. Installing PyTorch tools is fine. Report which method each claim rests on.

Related: [[feedback-research-method]], [[project-research-history]], [[project-map-track]].
