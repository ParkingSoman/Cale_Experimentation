---
name: feedback-math-paper-standalone
description: "Curves math paper must stand with all measurements deleted; facts -> definitions -> named hypotheses; Lean default; enforced by hooks in .math-mode, not by memory"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ece91aac-6818-4b8a-b706-34b0daefdff8
  modified: 2026-10-25T01:57:48.582Z
---

2026-10-24 night, Shonu: the math paper had become "a math motivated lab notebook". Many propositions were
experiments with a theorem label. Rule: if the paper cannot stand with every experimental result deleted, we are
not thinking mathematically. Conjectures must be proved, and so must things the engineering already showed.

**Why:** I checked his suspicion that the method weakened over sessions. It held. PAPER.md (2026-10-23) had 47 formal
results vs 23 measured Statements. NEURON-MATH.md (2026-10-24) had 41 vs 56, plus 72 REFUTED. The late "proofs" were
bookkeeping (convex quadratic root, E[ab] split, chain rule). Root cause: no mathematical model of the object, so
only identities could be proved and everything with content was measured.

**How to apply:**
- New paper `research/clean/curves/COARSE-THEORY.md`. Part A: FACTS (known for sure), DEFINITIONS (interpretability
  objects as math objects: feature, family, clean, edge, circuit, coarse network), HYPOTHESES (named, e.g. H-clean;
  "clean" and essay ideas are hypotheses, never starting points). Part B: theorems tagged "(under: ...)". Part C:
  only place for numbers (eps per hypothesis). Part D: room to extrapolate toward optimal subnetworks.
- Lean by default (finite version when the continuous one is impractical); sympy for identities; exact synthetic
  instance checks proofs. No InceptionV1 run before the theorem it serves is proved.
- Literature only through a compatibility table. No SAE / dictionary / superposition framing in the chain of logic.
- Shonu's goal: one mathematical argument for the eventual result that generalizes across networks.
- He said a memory note is too weak against RL-trained engineering habits. The rules are enforced by project hooks
  (`.claude/settings.json`, scripts in `research/clean/curves/tools/`, switch file `research/clean/curves/.math-mode`)
  and the `math-referee` agent. Rules file: `research/clean/curves/CLAUDE.md`. Plan:
  `~/.claude/plans/good-work-so-far-rustling-nautilus.md`.

Related: [[feedback-derive-like-mathematician]], [[feedback-research-essay-rules]], [[project-curves-track]].
