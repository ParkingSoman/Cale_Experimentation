---
name: feedback-error-meaning
description: "Error" in the curves math = rule's build vs the OPTIMAL small network of the same size, never vs the big network
metadata:
  type: feedback
---

When Shonu says "error" or "bound" for a coarse-graining rule, he means D(rule's build) - D(least build of the same
widths). Paying some cost against the big network is expected and fine. (2026-10-26: "when i talk about error i don't
mean copying the large network exactly ... i mean error in what the optimal small network would be given the large
network.")

**Why:** I called T39.1 (gate fold copies N exactly) an "exact formula" and "least <= L_G" a bound. Both measure the
cost against N. In T39.1 the optimum pays no cost, so it never tests the rule.

**How to apply:** a result counts toward the trunk only if it compares the rule with the least build in a case where
the least build pays a cost (width below what N needs). Report the rule's gap to the least build. Treat
"the build equals N" cases as easy cases, not as wins. See [[project-curves-track]], [[feedback-trunk-decisions]].

## The optimum is the whole point (2026-10-31)
Shonu: "the number one thing im trying to get at through all of this stuff is simply to find the optimum through which
well executed distillation is trying to get to on a larger network through mathematical calculations ... i want to
ensure we aren't allowing approximations ... very explicit to me if we ever have to move on from that idea of
optimality with error bounds which depend on how clean the network is."

**Why:** the track drifted off the optimum twice without a loud flag: the deep closed forms (TP1, TL5) are least for
the squared-logit loss D2, not KL (node R-obj, RETRO, never his decision); the merge rule TY2 and the stream cuts TV6,
TV18 bound one build's loss and are exact only at loss 0.

**How to apply:** label every result: OPT-KL (gives V(G), or a gap to V(G) bounded by claims' errors, 0 when clean),
EQ (two least losses equal), OPT-D2 (optimum of a different loss: off-trunk), ONE-BUILD (upper bound on one rule:
off-trunk). Say any OPT-D2 or ONE-BUILD result to him in plain words in chat and LADDER, never only in the paper.
A blocked proof becomes a conjecture with a Route and a card, never a weaker rule called the optimum.

Where the bar is written: research/clean/curves/CLAUDE.md section "The bar: optimality" and rule 0 of tools/turn_reminder.sh (prints every prompt). Shonu (2026-10-31): "this cannot ever be classified as done or well executed if we havent confronted the optimality question".

2026-10-31 (later): Shonu DROPPED R-obj on card 3-30: "specifically to find the optimal KL ... for everything, including deep networks". D2 results are records only (OPT-D2).
