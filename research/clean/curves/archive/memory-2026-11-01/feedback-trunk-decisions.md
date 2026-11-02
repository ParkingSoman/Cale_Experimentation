---
name: feedback-trunk-decisions
description: "Curves math: the trunk goal is fixed; every restriction is Shonu's decision, recorded as a node in DECISIONS.md, never added to get a proof through"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2042dd7f-ae97-4b25-853c-aa58b52b1a9f
  modified: 2026-10-29T20:24:44.336Z
---

The curves theory has ONE fixed goal (the trunk T*, `research/clean/curves/DECISIONS.md` section 1, hash-locked by
`tools/trunk.sha256`): best whole network of every size k, any new units, KL on the image mix, sizes nest. Every step
away from it (a hypothesis, a premise inside a tag, a narrower class inside a definition) is a NODE in DECISIONS.md.
A new node starts PENDING; only Shonu moves it to DECIDED, in his own words in chat. Results under undecided nodes are
side branches, not steps toward the trunk.

**Why:** 2026-10-25 Shonu saw the theory drift: rev 7's open item (D8) sat 8 restrictions from the goal, most of them
added by me inside plans he approved as a whole. He wants to keep the goal constant and make each branch choice
himself, after a plain explanation of the math.

**How to apply:** when a proof is blocked, write a plain card for Shonu (meaning, gain, loss, my recommendation) and
ask; do not add a restriction. Run `tools/ledger.py --report` to see each result's distance. "(under: none)" only
means "nothing beyond the class of its Part" -- the path says the rest. This overrides [[feedback-decide-dont-quiz]]
for restrictions of the theory (not for routine engineering choices). Related: [[project-curves-track]],
[[feedback-math-paper-standalone]].

**A fork must be forced by a theorem (2026-10-25, late night).** Shonu rejected my R-root options (structure /
approximation / search) and my draft trunk wording: "things you threw out there and asked me to approve, rather than
taking this to an actual fork in the road ... I want to go where the math takes us." Rule: work at the trunk's general
form and prove things there until a THEOREM shows two parts of the goal conflict. Only then bring a card, and its
options are exactly that theorem's premises (to escape a theorem, one premise must go). A belief or an unproved
hardness is not a fork. Do not narrow the question to make progress; a partial positive result is recorded as partial,
not proposed as a restriction.

**Ambition first, peel only when forced (2026-10-25, late night).** Shonu: "we're trying for very ambitious theorems
and then only peeling out what we need, absolutely have to, when making these decisions." Do not drop a goal property
(e.g. instant switching) from the trunk because it looks hard; keep it until a theorem forces it out. He rejected my
plan twice for dropping instant switching and for a single k per layer (wants k_1..k_L). Clean is a spectrum: at
eps(N) = 0 he wants an exact closed form with NO error bound; at eps > 0 a bound that depends only on eps. That is not
"accepting error bounds".

**Justify from the project's deep goals, not from design rules (2026-10-25, late night).** Asked for "a theorem that
folds only is the right choice", I first argued from design principles (the build must use N's neurons). Shonu: that
is circular ("based off of principles which themselves look like pretty design principles"). He wants each deep goal
("no training: the training was already done", "download once, fold many times", "fold up something that exists")
stated as formal mathematics, with the choice DERIVED from it as a theorem.

**Technical scope nodes are mine; prove the general case first (2026-10-29, card 3-25).** Shonu: "S-Laminar is
certainly a development decision ... completely fine for you to make those decisions on your own ... you could see
what it actually does or what it doesn't do in terms of mathematics before ... asking me ... when you give me a
decision like that, it's not really like something that's very productive for me to respond to". On S-smoothact (a
smooth-activation premise that left out ReLU): "I just want a nice general method over these residual streams ... we
don't need to place those sorts of restrictions on anything".
**How to apply:** a SCOPE node that widens the class, or is proof bookkeeping, gets its math worked out and I decide
it (DECIDED by delegation with his quote, like R-normconst and S-laminar). A premise that leaves out common networks
(ReLU, norms, attention) is a last resort: first try to prove the general case (S-smoothact and S-nonorm were removed
that way, Theorem TV15). Only a restriction that NARROWS the class after the general attempt failed goes to him as a
card, with options from a theorem.

**Most card items are mine to settle (2026-11-01).** Shonu left card 3-32 unanswered and said: "some of the decisions
you asked of me are more mathematical details you should've ironed out rather than directional decisions i needed to
make, which is why i haven't answered them." Rule: a node that only fixes a technical premise, a summary's form, the
meaning of a bound's test, or what counts as a certificate is MINE: decide it, record it in DECISIONS.md as "DECIDED
by delegation" with his quote and my reason, and tell him in one line. Bring him only directional forks: what the
product promises (which zooms, which networks, what "exact" means to a developer), or a THEOREM that shows two parts
of the trunk conflict. When unsure, decide and flag, do not ask.
