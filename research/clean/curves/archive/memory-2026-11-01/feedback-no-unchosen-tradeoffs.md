---
name: feedback-no-unchosen-tradeoffs
description: "2026-10-31 standing rules for the curves track: exact is exact, graded is useful (first-order tight), trained nets are clean (realism check), no tradeoff Shonu has not chosen, 'not possible' needs a theorem on a clean instance"
metadata:
  type: feedback
---

Shonu (2026-10-31): "i just want to ensure that exact is really exact, and we are ensuring graded is actually useful.
the essay doesn't just say networks could be clean, it specifically says well trained networks are already clean ...
i don't want there to be tradeoffs i haven't chosen ... if that is not possible, you can't just say it, you must
mathematically show it, and im likely going to attack that using clean objections as well ... this is absolutely
genuinely possible given our new understanding that networks are already clean".

**Why:** sessions kept surprising him with new hurdles and with narrowings he never chose (S-keepdrop read to him
as pruning). Loose bounds (1e4 to 1e9) were called graded results.

**How to apply:**
- Exact theorem -> check against a certified global optimum (search + dual lower bound), hand-built AND trained nets.
- Graded bound -> must be first-order tight; report bound / true gap on trained nets.
- New premise -> realism check on a trained net (model/ trained nets, cache/r18) in the same session.
- Any narrowing, any change to "exact" -> card BEFORE the proof (card lists every tradeoff of the session).
- Never say "not possible" without a theorem on a clean, trained-like instance; read a counterexample as a network first.
- Roadmap: research/clean/curves/ROADMAP.md (4 sessions; he cut my 8 to 4: keep roadmaps compact).
See [[feedback-error-meaning]], [[feedback-trunk-decisions]], [[project-curves-track]].
