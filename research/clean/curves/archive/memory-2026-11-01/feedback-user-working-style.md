---
name: feedback-user-working-style
description: "Who Shonu is and how he wants me to work: plain explanations, own ideas, Cale equation, attack the root layer (merged)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9d06e420-9dd2-4fb1-807e-0b6e19dc3744
  modified: 2026-10-29T20:44:42.557Z
---

## Background (user)
- Limited ML background, learns fast when things are explained well. Wants trade-offs surfaced and explained so we decide together, and wants me to try to break through trade-offs, not just report them. Prior projects: ported Gemma 4 (E2B/E4B/12B) to ONNX Runtime/browser in ~/BlinkFix and ~/BlinkFix-models.

## Own ideas, centered on the Cale equation (2026-10-14)
- Quote: "when you think of recommendations, i want you to also have your own ideas. center everything around the Cale equation, and let's name it the Cale equation going forward."
- Call (app types that work well) x (environments) "the Cale equation". Every recommendation says what it gives the equation. Always add ideas of my own beyond what he named. Say plainly what is an estimate and what is measured. Keep overviews short and simple (plain-english style).
- Why: out of step after two days of fast work; wants a partner who thinks about direction, not a builder who follows a list.

## Attack the underlying layer (2026-10-15)
- Quote: "you don't need to ask me to add something to the research report. we should always have a bias for researching any solution which can solve the problem underneath my problems (not just taking the issues I say at face value, but figuring out the common thread or more important thing underneath my complaints, and attacking that layer aggressively)."
- Why: complaints are symptoms. Example: "probing may hurt goals" and "adapt inside answers" share one root: Cale's predictions are not accurate enough. Fix = better measurement, not patches per symptom.
- How to apply: name the root cause in one line per request, then research and build at that layer. Add findings to `docs/research/` without asking. Still ask before outward-facing or hard-to-undo actions.

Related: [[project-cale-vision]], [[feedback-research-method]], [[lesson-rules]].

## Real ML objects + a diagram per step (2026-10-27; replaces the 2026-10-26 "everyday pictures" rule)
- History: 2026-10-26 night he said "you need to explain all of that simpler" (paper words in short sentences were not enough), so I moved to one everyday picture (workers on floors, exam). 2026-10-27 that also failed: "the explanations are using so many analogies im having an even harder time following ... it would be better if you still sorta explained and connected things to what is actually happening on the ML side and sorta drew a picture for me at every step what is happening and why it is important. this should be a sort of standing rule".
- Confirmed the same day after the 8-step summary in ~/.claude/plans/id-like-you-to-tingly-harp.md: "the way you explained things here was so nicely done (great job!) so id like you to continue explaining things just like this".
- How to apply (chat answers, LADDER sections, DECISIONS.md cards):
  - Use the real ML objects: neurons, weights, ReLU ON/OFF, layers, outputs, KL, the mix. Explain each word once, in a table, where it first appears.
  - Each step: a small text diagram (code block) of what happens in the network, then one "Why it matters" line.
  - One analogy at most, as a side note. Never let an analogy replace the real object.
  - Say what is proved, what is measured, what is open (a table works well). Name the theorem for each claim.
  - At most 2-3 questions, one sentence each.
- See [[project-curves-track]].

## Understand first, then a small session (2026-10-29)
- He rejected an 8-step plan for the residual stream: "we're doing a little bit too much in this one session", and asked for background first: "before we stage a plan, before we do anything", "what is the before ... and what is the after".
- The explainer that worked: what the object is in the network (with a code-block diagram), how it looks in real models (a table from the actual code), then "before / after clean", each point marked as essay or my reading.
- He thinks out loud in long notes and says I may ignore them. Do not ignore them: answer each nugget (right / needs a fix / already the trunk's target), with the math.
- How to apply: on a new direction, explain first and end the turn without a plan; then a plan of about 4 steps for one session.

## A "contradiction I see" note is a push, not a question (2026-10-29)
- He wrote that hardness clashes with how understandable small networks are. I answered with an explainer of why both
  are true; he rejected it: "I'm not trying to get you to say okay explain this to me and explain why it's impossible
  ... I truly believe this exists and we just need to find the right form of it ... look at the network".
- How to apply: take the existence of the interpretable optimum as given. Read the failing instances as networks
  (who reads whom, which neurons are on, which routes N uses) and turn each reading into a theorem. Do not lead with
  why the problem is hard.

## Technical nodes are mine; his analogies are leads (2026-10-29)
- On card 3-23's technical nodes (norm constants, readout areas, cut claims): "they don't really seem like decisions, they kind of seem like things that you could work out with the map itself ... explore the math for what's good for our ultimate goal". Settle such nodes by proof (drop as moot, with his quote) instead of putting them on a card. Only scope and trunk-wording questions go to him.
- His quantum-computing picture ("operations affect other operations, but you can create a quantum algorithm ... to cancel out those terms") pointed at a real open problem (TV6 loses cancellation). Treat his analogies as leads: find the math object they name, keep within Rule 5 (no superposition framing).
- He wants each work stream in its own worktree when two sessions run in parallel.

## Compact roadmaps (2026-10-31)
Shonu: "i don't think it needs to take 8 sessions". Merge sessions; do not stretch a roadmap's count. State
dependencies and walls up front, never as later surprises. See [[feedback-no-unchosen-tradeoffs]].
