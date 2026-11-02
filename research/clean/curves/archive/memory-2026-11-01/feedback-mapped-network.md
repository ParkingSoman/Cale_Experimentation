---
name: feedback-mapped-network
description: "2026-10-28: the trunk may assume a network already mapped by interpretability work; finding the map is outside the math, every map term must still be a math object with a graded eps"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8223c980-b79d-45c7-b9f1-eae3350d13b8
  modified: 2026-10-28T05:08:27.438Z
---

Shonu (2026-10-28, on the edge-unit plan): "we could assume within the trunk that we already know how neurons are
grouped ... assume that we're given a network which is already mapped"; "we can't just say, hey, this is mapped and not
have a mathematical theory attached to that all the way to the fundamentals"; "I don't want to hamper the effectiveness
of this prior that a network is clean and well-trained ... by also making us create an algorithm for determining how to
interpret a network." He chose: draft Trunk T5 with a map input, graded eps per claim.

**Why:** our hardness results (T22, T43, T57) put the difficulty in the SEARCH for groupings; people and agents do that
search today. Requiring the rule to also find the map wastes the clean prior.

**How to apply:** a map = finite list of typed claims about N and P, each with a poly-time eps (0 iff the claim holds),
no claim mentions builds, sizes or the zoom (otherwise the map could state the answer). Concepts enter as extensions
(weighted subsets of P). Do not design unit-finding algorithms; prove what a claim must say and what it buys.
Related: [[project-curves-track]], [[feedback-trunk-decisions]], [[feedback-general-form-first]].
