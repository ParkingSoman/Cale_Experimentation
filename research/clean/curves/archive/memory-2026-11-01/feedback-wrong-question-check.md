---
name: feedback-wrong-question-check
description: "2026-10-27/28: before handing Shonu a fork, check whether the forcing theorem depends on N at all; a blow-up (tiny cause, fixed effect) is an ill-posed object to fix, not a decision; principles of clean are graded"
metadata:
  node_type: memory
  type: feedback
  originSessionId: fbc141f7-bffd-41d8-9b2a-9db6296e09d2
  modified: 2026-10-28T01:23:48.464Z
---

Shonu (2026-10-27, on card 3-15 R-indep): "if a network is well-trained, then it is already independent. And if we're
getting that result that it isn't independent, then we're looking at the network the wrong way ... the fact that we're
even asking or in a position to ask those questions, I think is kind of contrary to my theory". Also: every principle
of clean is a gradient ("if cleanliness is a gradient ... all of the components of cleanliness are gradients too"),
and a found unit must have a concrete interpretation in a real network (image model), not only a math name.

**Why:** TU10 (free boxes kill independence) held for EVERY network, clean or random, so it was about our box
definition (reading below-zero pre-activations N throws away), not about N. I turned it into a decision for him.

**How to apply:**
- Before writing a fork card, list the forcing theorem's premises. If none mention cleanliness / N's structure, the
  fork is about our definitions: fix the definition (essay-first) instead of asking Shonu.
- Make each principle an entry of the eps list (graded), never a yes/no choice.
- For any new unit/object, say what it is concretely in an image model (top images, districts) before asking him
  to decide on it.

**Second case (2026-10-28, TM4 blow-up):** I offered a "weight limit" card (cap / margin claim / keep). Shonu: "i suspect
this shouldn't be a 'decision' in the traditional sense ... this seems to me like a mathematical error in actually
finding the optimal subnetworks given a larger network, and we should treat this as the error it likely is". Right
again: TE1 showed the free-head least loss is blind to size for every N. Treating it as an error led to the robust
optimum (B.E), which needs no free parameter. How to apply: when a blow-up or discontinuity appears (tiny cause, fixed
effect), first look for the object that is ill-posed (not continuous in N) and correct it; offer a card only for the
final wording of the trunk.

Related: [[feedback-general-form-first]], [[feedback-trunk-decisions]], [[feedback-research-essay-rules]].

**Copy test (2026-10-28, B.N).** Replace one unit by n copies that share its output (in-weights times lam_i). N does not
change. Any measure that moves is counting units, not the network. It found that DM2, DM3, DM8 and DM6 item 5 (and the
DE2 judgment split) counted units. Run it on every new error, constant or condition before using it.
