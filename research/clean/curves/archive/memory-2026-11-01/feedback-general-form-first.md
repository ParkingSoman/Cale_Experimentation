---
name: feedback-general-form-first
description: "Curves math: the optimum itself is the object; folds/gates/families are Shonu's hypotheses, never required methods; clean = property of N alone, multi-principle, eps is a list"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b4804902-d457-42e5-bb76-185d714bce80
  modified: 2026-10-26T17:11:18.258Z
---

2026-10-26, Shonu: folds, gates and other interp ideas "are really just my hypotheses for the ultimate thing I'm
really interested in ... what is actually the optimal subnetwork given a large network". "I don't ever want us to
like say, hey, we have to do this method just because." He wants "a mathematics mindset. What is a general form? Are
we working generally?" not an engineering mindset attached to one solution. Now rule 8 of
`research/clean/curves/CLAUDE.md`.

On clean (same day): not "a stronger clean" but "a good version of what a clean network is", one that reflects how
real networks are (source: the essay). Clean is a property of N ALONE: no mix P ("i don't want closed solutions to
depend on how an individual would want to orient the network"; P = the mix is still a trunk input because T24 shows
the optimum depends on it). Clean is "an object which satisfies a bunch of principles", and eps is a LIST of metrics
("multiple metrics ... which the error bounds can depend on"). His conjecture: a good definition of clean makes the
NP-hardness go away.

**Why:** sessions drifted into one solution route (gated folds) and framed forks as "restrict clean further". He saw
that as attachment to a method.

**How to apply:** state the general object first; mark each section's lens (general optimum vs method hypothesis);
when a definition seems to need data, bring that to Shonu as a finding. Scopes [[feedback-research-essay-rules]]
("essay is law") for the math track: the essay describes networks, it does not prescribe methods. Related:
[[feedback-trunk-decisions]], [[project-curves-track]].

2026-10-26 later (Revision 2 of A.5): Shonu: clean must describe REAL networks, because the essay's title says networks
"are already clean"; a definition under which clean is rare is wrong. Clean from N and X only, NOT training data
("you should be able to determine if a network is clean by interpreting the neurons and circuits themselves ...
rather than needing extra data ... using foundational mathematical principles"). Take the math from the essay, which
"does go against some standard interp methodologies" (e.g. "a blind maximum tests the neuron's center, not its
breadth": judge a unit on its whole target, not its peak). The essay never mentions symmetry: do not make exact
symmetry a principle of clean (Theorem T44: it holds only on a null set). The perfectly clean set must CONTAIN a non-empty open set (C6; the set itself need not be open) and be a weighted share in [0, 1] (C7, C8).
Check every new principle against the essay's own clean examples (e.g. the ambulance unit with a taxi on its
negative pole): D42's first V penalised that inhibition, which the essay calls clean (comment (h) of D42).

2026-10-26 evening: EXPLORATION FIRST. Shonu: "when we say clean, we really mean interpretable. And when we say
interpretable, we're actually pointing to a sort of engineering-ish method where we're exploring the neurons
themselves ... the first principle is actually exploration." Do not invent cleanliness measures (D42's V/G/B were
"very particular measures"). Explore real networks the way an interpreter does (essay's moves: poles, no-noun
descriptions, upstream build vs downstream use "line up", explanations that can lose), log every move, THEN
formalize. Packing must be judged in the network's own vocabulary (4e:55 "cats and cars" = V). Inputs rule: decided
AFTER Phase 1 (C1 stays for the paper until then). Plan: ~/.claude/plans/so-it-seems-that-declarative-ladybug.md.

