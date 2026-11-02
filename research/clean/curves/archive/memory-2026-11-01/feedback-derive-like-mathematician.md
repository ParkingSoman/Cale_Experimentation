---
name: feedback-derive-like-mathematician
description: "For the coarse-neuron equation: investigate the object first (images + numbers), derive; never open with candidate equations to test"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 666d8617-a084-4cce-90bf-0a4bfce453e5
  modified: 2026-10-22T20:54:33.921Z
---

When looking for the equation that makes a coarse neuron from its parent and children, work like a mathematician, not an engineer. First study the object itself: real coarse neurons (atto/nano twins, solved parents), their fixed properties, both images and numbers, special and exact cases. Form conjectures from what you see. Derive the equation from properties that held. Do not start a plan with candidate equations to test.

**Why:** 2026-10-22 Shonu rejected a plan that listed three candidate equations up front: "mathematicians don't just throw out an equation ... they derive it ... look at this coarse neuron ... investigate the circle to derive pi." He also rejected Gauss-Newton fits with knobs chosen on held-out images as "training with another skin".

Training is NOT a ceiling. Shonu: training is an approximation of what the true equation does ("like saying a Taylor polynomial is the upper bound for the true cosine function"). The large trained model already holds the information. Aim for the equation to BEAT training. Never frame training or GN as the upper bound in plans; treat them as references to pass. Raw numbers are still reported honestly.

Strong references first, then map the approximation (Shonu, 2026-10-22): "we won't be able to see if it's better than training if the training itself is bad." Build a well-tuned training reference at the RIGHT structure (lr sweep, enough data, early stopping). Then study the trained solution like a best-fit line: if the slope is 2.96, guess 3. Look for clean constants and simple forms in what training changed; let those drive the derivation. (Same day I wrongly claimed a derived arm beat training; the reference structure was weak, a plain block drop beat both.)

Interpret every number (Shonu, 2026-10-22): numbers from the trained solution are only useful if each one has a concrete meaning in this network (which neuron, what it detects in images, why it moved). Work like physics discovery: think about what the objects mean while running small side experiments; connect each attempt rigorously to what it means for the network and its real-world job.

Three sources of new ideas (Shonu, 2026-10-22, curves track): (1) whenever a method fails, understand very deeply WHY it failed (the main source); (2) normal derivation; (3) periodically look at the images themselves. Example that worked: the derived curve neuron failed -> traced to the write (regression shrinks it) -> variance-keeping write.

**How to apply:** plans for derivation work open with an investigation phase (observe, measure invariants, look at pictures, toy/exact cases), then conjectures with falsifying checks, then the derivation, then one cold test. No output-loss fitting, no values tuned on images. Related: [[feedback-research-method]], [[feedback-research-essay-rules]], [[project-map-track]].

## Process rules (2026-10-22 night, plan ~/.claude/plans/okay-i-want-you-optimized-perlis.md)
Why no training, in Shonu's words: (1) practical, the large network already holds the info and an equation may beat
training; (2) combinatorial, there are many ways to coarsen some families and keep others fine, and training would
need one run per combination. TEST for any method: it gives ANY coarsening choice in seconds from fixed properties
of the large network (weights, its own activations and gradients, computed once), no per-choice optimization.
Gradients of the INTACT large network are allowed (Shonu said yes).
- Training has two jobs only: a ruler (best known solution at size k) and a microscope (swap read/write/branch).
  Never a headline. "Derived start + training beats random start" was a wrong headline (2026-10-22).
- Label every number DERIVED / CHOSEN (menu size) / READ-OFF. Only DERIVED counts.
- A failure goes back to the derivation: name the false assumption, re-derive. No patch on the old build.
- Audit 2026-10-22: after the plan's first derivation failed, sessions ran ~28 tests, 1/4 knob menus, each diagnosis
  patched the same skeleton. Fix: THEORY.md one-pager rewritten every 3 cards; 3 cards with no change -> pictures +
  literature, then a new conjecture. Dev comparisons paired on all 5000 dev images.

## Weights-only rule (2026-10-23, plan ~/.claude/plans/first-thing-i-want-enchanted-sutherland.md)
Shonu: the fitted builds (least squares on 600-35000 images, Fisher write) still depend on calibration data, so
they do not meet the bar. "Training is an approximation of the actual elegant mathematical formula." Work ~80% as a
math paper (definitions, lemmas, proofs, exact cases), ~20% grounding on real weights/activations.
- DECIDED: the final formula Phi reads the WEIGHTS ONLY. Images may only CHECK a hypothesis.
- Labels now: DERIVED (weights only) / FITTED (statistics or least squares on images, the old "derived") / TRAINED.
- The 2026-10-22 permission to use intact-network gradients on images is superseded.
- Paper: research/clean/curves/PAPER.md; Phi in phi.py must not import the data loader.

## The template session (2026-10-23, Shonu: "one of those sessions where I'm really feeling proud of it")
Shonu asked that the 2026-10-23 weights-only session (PAPER.md) be THE way to work on these problems until solved:
- Paper form: definitions, theorems with proofs, exact cases. Numbers only ground or falsify a statement.
- A written prediction before every check; small run budget (~6 runs + 1 cold test); paired dev on all 5000 images.
- Honest retractions in the paper itself (e.g. C3 "1.5% in span" retracted). A failure goes back to the theory.
- A label on every number. Proved negative results count (Theorem 5 blindness).
- Keep the math rigorous even when Shonu makes a product-level decision.

## Mix decision (2026-10-23, plan ~/.claude/plans/i-d-like-you-to-squishy-yao.md)
Shonu turned Theorem 5 into a product lever: the image mix is a DECLARED INPUT, not a hole. Cale ships a generalist
build (mix = the large model's own training mix); developers later declare a list of capabilities (images, what the
app reads from the output, weight). Developers never name neurons; the math gives each family its effective mix.
He wants swapping mixes and switching a family coarse/fine to be quick: Compile (averages per mix piece, once) +
Assemble (seconds, no images).
Labels now: DERIVED (weights only) / MIX-DERIVED (weights + closed-form averages over the declared mix) / FITTED
(anything chosen on dev images, or an iterative pass on the output loss) / TRAINED.
- 2026-10-23 drift: B1 (data hunger) failed and I ran 4 diagnostic pokes in a row (eigen-directions, effective
  sample size). Shonu: "we are going more into engineering work rather than the previous mathematical method which
  was already working." Rule: define Phi on the POPULATION moments of the mix; estimation noise is a later,
  separate (statistics) issue. After a failed check, the next step is a statement in the paper, not another probe.

## Beat-training rules (2026-10-23 afternoon, plan ~/.claude/plans/okay-i-d-like-you-enumerated-wreath.md)
- 100% mathematical reasoning (Shonu): "if you don't have mathematical reasoning for 100% of the way, this method is
  never going to work." Every part of a claimed build follows from a stated theorem or exact case. No CHOSEN knob.
- Simple-rule margin (audit after T4b): every written prediction names its simple-rule baselines (uniform, energy,
  kept members, naive top variance) and a margin (2 paired se), fixed before the number. Reaching the twin ceiling is
  not enough. Same slip as the 2026-10-20 recipe-parents result.
- Headline bar for "beats training" (Shonu chose): the BEST KNOWN trained network at the same size, not random-start
  distillation. Random start is reported next to it, never as the headline. Gap split: KL(formula) - KL(train) =
  A(formula) - O(train) - S(train); a formula wins only where training's optimizer or data gap exceeds the formula's
  approximation gap. Exactness test: training started AT the formula should not improve it.

## Data-free generalist (2026-10-23 evening, plan ~/.claude/plans/i-want-you-to-majestic-milner.md)
Shonu restarted the session: the GENERALIST build may use NO image data (read, write, mix). Reason: "when you add in
data, the next question automatically becomes, okay, what data are you adding?" Bar = "how the network would have been
if it was just trained to be smaller". Developer mixes may use images later. Images only CHECK.
- Supersedes the morning MIX decision for the generalist. Fisher/lsq builds are FITTED references.
- He thought Theorem 5 only hits the write. Corrected: it covers every plain average (read too, C5/I4b). But B0 (read
  barely depends on which natural images) supports trying the network's OWN generated inputs (feature vis).
- Fallback order he accepted: weights (conjectures A-D) -> self-generated inputs (E) -> natural-image law (1/f^2,
  dead leaves) ONLY with his OK. Network: InceptionV1 only. Labels DERIVED / LAW-DERIVED / FITTED / TRAINED.
- Do not re-ask: he wants to know "every number the network gives off the bat" (wiring, symmetries, spectra, probe
  responses, generated inputs, implied density, class similarity, mean pixel 117).

## Two problems (2026-10-23 night, same plan file, rewritten)
Shonu split the work: P1 = a mathematical formula that makes inputs from ANY trained network (no image set, no
tuned loop; "I don't want us to be beholden to specific images and the training cycle again"); P2 = the original
coarse-network problem, solved with P1's inputs. P2's needs set P1's axioms. He accepted: a density defined from
the network alone + a standard sampler counts as a formula, IF doubling the sampler's steps changes no result (> 2 se
-> CHOSEN). Start point: p(x,y) = p_theta(y|x) p(x); the network fixes the conditional; only p(x) is open (Theorem 5).

## Errors -> mathematical reason (Shonu, 2026-10-23 night, repeated: "you are becoming divorced from the math paper")
"Whenever you find errors you still need to bias yourself towards mathematical thinking ... errors means you have to
figure out the mathematical reason why you are wrong, which will naturally take you to the correct next steps."
Trigger: I ran 4 optimizer tweaks in a row on the stationarity generator (lr, init, sum-to-1, warm start) instead of
asking WHY its objective collapses. **How to apply:** after ANY failed run, stop running. Write in PAPER.md the
statement that explains the failure (what object, which identity, a proof or exact case). Only then choose the next run,
and it must follow from that statement. Drift gets worse late in long sessions: re-read this rule when a session is long.
- 2026-10-23 night decision: either selection rule (max entropy or smallest set) is fine; REBUILDING REAL TRAINING
  IMAGES FROM THE WEIGHTS IS FINE ("we are still using the network's own properties"). His conjecture: the network
  knows what it is unsure about -> make those unsure inputs. Plan: free-coefficient KKT reconstruction (Theorem 23).

## Shonu's interrogation method (2026-10-23 night; THE way to treat any gap claim)
"Looking at every single piece within this mathematical puzzle where we're saying the network isn't enough and
deciding, hey, I truly believe the network is enough and we're likely not thinking of the problem in the right way, or
we are, but there's certain mathematical directions that are tiny that we just didn't go down." Strong prior: "the
network itself already had so much computational effort put into it, and we don't need anything other than the network
itself and what the network itself can produce in order to create mathematically optimal subnetworks."
**How to apply:** for every "we need outside data" statement, list the assumption it rests on; question the broad
framing (e.g. Theorems 5/24 assume ANY stationary point, but the weights are the point TRAINING REACHED); and list the
small skipped directions (e.g. the Neural Feature Ansatz). Give each one a cheap check before conceding a gap.
- 2026-10-23 night, THIRD drift warning ("You're again drifting from math based stuff"): after Theorem 28.1 (coverage)
  I ran E17 -> E20 as four fix-variants (plain-mean constant, refit read, difference estimator, temperature), each a
  patch, none derived. Pattern: a success (E17) triggers a chain of variants. Rule: after ANY result, success or
  failure, write the statement it supports or refutes and what it implies BEFORE the next run; at most one run per
  statement. If two runs in a row were not derived from a written statement, stop and write the paper section.

## Theorem-first + "trained = imperfect example" (2026-10-24 early, plan ~/.claude/plans/okay-so-the-last-harmonic-stallman.md)
Shonu rejected the pessimistic framing ("data is needed, Theorem 34.3") and asked for:
- View a trained small neuron as ONE IMPERFECT EXAMPLE of the true coarse neuron: "the methods you choose implicitly
  make the method worse than training, or subservient to training." Never let a method's design assume training is the
  target; split each trained solution into (projection on the theory's structure) + (remainder = training error until
  proved otherwise).
- Build the paper like a proof: state the TARGET THEOREM first, then a lemma tree with status + dependencies; when a
  check refutes a statement, mark dependents SUSPENDED and fix the theory before building on it. Lean/sympy welcome.
- Encourage myself with real reasons it is possible (paper section 1: measure-over-neurons frame, Cor. 2 metric,
  impossibility only for exact stationarity, channel-3 alignment, Haar measure). Failures = wrong definition, not
  "impossible". Paper: research/clean/curves/NEURON-MATH.md.

## Exactness rule + THE LOOP (2026-10-24, plan ~/.claude/plans/okay-for-this-session-twinkly-gosling.md)
- NO REAL-IMAGE SHORTCUTS for the generalist. Last session I counted the Prop. 11.1 formula built from 2000 real images
  (.0647, "beats a training run") as acceptable. Shonu: that is information outside the network; once you use real image
  statistics "it's sort of like a spiral where you start kind of replicating training". FITTED builds are microscopes
  only, never a result and never an intermediate step of the generalist.
- THE TARGET = the exactness test: build with the formula, then train starting at it. If training improves it, training
  set a new ceiling. Goal: a formula so good that training on top of it does nothing, proved as a theorem.
- Fairness (his point): a formula cannot know a data set's image mix. So: generalist (what Cale ships) = weights only,
  with a theorem that training bringing no extra distribution information cannot improve it. Developer = generalist + a
  quick closed-form re-mix on a small sample of their images (children of the coarse neuron fire in proportion to that
  mix); training on their images should not improve that either. Definition used: "no extra information" = training on
  a mix D whose sufficient statistic m(D) equals the network's own m_theta. The theorem has content only if m is small.
- THE LOOP (Shonu asked that the plan encode it exactly): declare the target theorem; every step is a ledger statement
  it needs; state -> prove what can be proved (Lean in research/clean/curves/lean/coarse, sympy, exact toy) -> written
  prediction -> one check. On failure: REFUTED, dependents SUSPENDED, write the mathematical REASON as a statement, then
  VERIFY THE REASON ITSELF (exact case, Lean/sympy proof, or a measurement only the reason predicts) before repairing.
  "That ensures every step of the way is built on something truthful."

## Belief + strongest form + math first (2026-10-24 evening, plan ~/.claude/plans/last-session-made-quite-lively-flute.md)
Shonu: "I notice better performance when you genuinely believe it is possible ... I believe you can sanewash me and
get to an endpoint I can agree with." And: "start from mathematical-based reasoning ... when you do engineering-based
stuff the results are not as good."
**How to apply:** write plans from genuine belief that the goal is possible (list real reasons). When his argument is
loose, restate it in its strongest precise form first (the endpoint he can agree with), then keep honest caveats INSIDE
that frame (pushback is fine and wanted, but not as the headline). Phase 1 of every plan = derivation in the paper;
observations (images, numbers) are the world we check against, never the driver of the next step.
