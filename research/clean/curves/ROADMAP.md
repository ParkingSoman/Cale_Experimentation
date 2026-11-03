# Roadmap: the optimum of every level of a ResNet zoom, by calculation

Shonu points here. Read it first, every session. Each session updates its row and the status table.
Approved 2026-10-31 (plan `~/.claude/plans/okay-last-session-i-jiggly-aho.md`).

## The goal

A developer defines a zoom on a ResNet. The rule computes, from the large network alone, the network with the least
distillation loss (KL) at each level. It does no training. After one computation up front, every level comes out
instantly. When the map is clean, the answer is exact. When it is nearly clean, a proved bound says how far, and the
bound is close to the true gap.

The Cale vision for this roadmap (Shonu, 2026-10-31): "instant computation given a defined zoom". ResNet is the
target. Language models come later. No app, no runtime, no consumer screens in this roadmap.

The end condition (Shonu, 2026-10-31, night): the roadmap "HAVE to end in a calculable subnetwork zoom sort of thing
(the full cale experience) for resnet, this needs to be able to be done for the full size of the model. no
tradeoffs, no ifs buts or coconuts, no "but wait!"s, just the full cale experience on resnet. if the roadmap ends in
ANY sort of detraction from that incredibly strong thing, then we haven't completed it."

If the roadmap ends in any detraction from this, it is not complete. What "full size, no detraction" means:

| Must hold | Not allowed |
|---|---|
| the whole network: stem, all four stages, skips, channel units | only the last block, or only the top layer |
| ResNet-50 (and the same code on ResNet-101 and ResNet-152) | ResNet-18 only |
| a real map: claims of N with measured errors | proxy areas from class clusters |
| every level of any zoom the developer defines, on a map made complete for it (trunk T5, card 3-19) | a zoom picked to suit the theory |
| the least KL at each level (OPT-KL), with a first-order-tight bound | a bound 1e4 too large, or a trained head as the answer |
| one computation up front, then every switch instant | retraining or refitting per level |

## Shonu's standing rules (2026-10-31)

| Rule | What it means in the work |
|---|---|
| "exact is really exact" | An exact result is a proved equality with the least KL. It is checked against a certified global optimum (search plus a dual lower bound) on small networks, hand-built and trained. A gap above float error means the theorem is wrong. |
| "graded is actually useful" | A bound counts only if it is first-order tight: as the claims' errors go to 0, bound / true gap stays bounded by a known constant. The ratio is measured on trained networks. Bounds 1e4 to 1e9 too large (TL6, TV6) do not count. |
| "well trained networks are already clean" | Every new premise gets a realism check on a trained network in the session it enters. A premise that trained networks miss badly is the wrong premise. The fix is a better premise, not a weaker result. |
| "no tradeoffs I haven't chosen" | Every narrowing, every change to the meaning of "exact", and every definition that limits which zooms are exact goes on a card before the proof that uses it. |
| "if it is not possible, show it mathematically" | "Not possible" needs a theorem that holds on a clean, trained-like network. A counterexample is first read as a network to find the missing claim. |
| THE BAR (CLAUDE.md) | every result labelled OPT-KL / EQ / OPT-D2 / ONE-BUILD |
| "no detraction at the end" (2026-10-31, night) | the roadmap ends only with the full Cale experience on the whole of ResNet-50 (table under "The goal"). No fallback ending counts. |

## How row 3 ends (Shonu, 2026-11-02; card 3-37)

Shonu: "this does not have any utility unless it passes on real networks. i can understand if we prove its mathematically
impossible to do so but we haven't proved it either way, and so we need to work until that happens."

Row 3 ends in one of two ways. The full tests P1 to P4 and I1 to I5 are on card 3-37.

| End | In one line |
|---|---|
| PASS | a claim with an exact theorem and a tight bound, and on ResNet-18 and -50 at k = 14 and 38, on the real box, the rule's build is at most 2 times its proved floor |
| IMPOSSIBLE | a theorem: unless P = NP, no fast rule is within any fixed factor, on a family that is trained, clean and matches ResNet's measured tables |

The written end in the table below ("row 3 has an exact theorem, checked, refereed") does not count by itself.

## Where we start (2026-10-31, after LADDER 40)

| Row | A group keeps | Least KL | Label |
|---|---|---|---|
| 1 | 0 units (capability removed) | closed form: log of N's mean probabilities on each area (TI1) | OPT-KL |
| 2 | rank or more units (kept whole) | within a quarter of the linear-read error (TI2) | OPT-KL |
| 3 | between 0 and rank (squeezed) | open: this roadmap's first job | none yet |

## The four sessions

| # | Session | Deliverable | Checked by | Ends when |
|---|---|---|---|---|
| 1 | Row 3: squeezing | card 3-32 (every tradeoff of the session); the separation claim; its realism check on trained nets and ResNet-18; the L-sep test; then the theorem: squeezed least = a tree knapsack over information tables (Branch A), or the form read from a clean counterexample | `model/ts_sep.py`, `model/ts_tree.py` against a certified global optimum; Lean; math-referee | row 3 has an exact theorem, checked, refereed |
| 2 | Bounds from cleanliness, in ResNet form | graded forms of TH1, TI1, TI2 and the squeeze theorem, first-order tight; completeness for the new claims; all of it carried to skips and channel units (TY4). Known wall: TM4, TE5. Route: the quiet claim DM9 | nets with claims slightly broken (hand-built and trained): bound >= gap always, ratio measured | T5's graded clause met with a tight bound |
| 3 | ResNet-18 | a REAL tree map of the whole network, found by map agents (claims of N with errors computed by code, not proxy areas; the pilot says start at stage 3, since feeders at the last two levels are mostly the skip; about 9000 tokens per channel, so about 20 million tokens for ResNet-50's last block, an estimate; needs N's real training photos for any H-stat or collapse claim, about 2.5 GB for 20 per class) at two grains, all four stages; every claim's error measured (Part C); every level built by the rule; a trained distillation of the same size as reference. The one-time computation stores summaries (information tables, per-node tables), not activation caches, so it fits the disk at full size | our KL <= trained KL + bound at every level | ResNet-18 zoom table in Part C and LADDER, whole network |
| 4 | The Cale vision on ResNet, full size | a developer-defined zoom on the WHOLE of ResNet-50 (every stage, every skip, every channel, all 1000 classes); one computation up front; every level instant, with its least KL and its bound | switch time measured flat across ResNet-18, -50, -101, -152; session 3's check at every level | every level of the zoom comes out instantly, computed by the rule from N alone, no training, OPT-KL within a first-order-tight bound |

Dependency, a hard gate: session 3 does not start until session 2 meets the bar. There is no fallback. The 2026-10-28
test found no clean channel map at ResNet-18's last block. That test was a script with one fixed rule and label regions, so it
shows less than it said (corrected 2026-11-01). Maps come from agents who read the network the essay's way, never from one
algorithm (Shonu, 2026-11-01: "you need to have subagents figure that stuff out"). A real map and session 2 stay the critical path.

## Status

| # | State | Date | Result | Shonu decided |
|---|---|---|---|---|
| 1 | in progress (row 3 has exact theorems; real networks miss their premises; NOT closed by the standing rule; your call on card 3-35); continued 2026-11-01 (plans `i-think-we-are-agile-raccoon.md`, `you-ve-done-a-lot-elegant-dongarra.md`) | 2026-11-01 | 2026-10-31: seven approaches (LADDER 41): tree knapsack false (TQ2.3); every row-3 least > 0 (TQ2.2); clean-limit sandwich TQ3; reverse pairing of two areas TQ4; canonical packing TQ1 (ONE-BUILD). 2026-11-01 (LADDER 42): every squeezed group writes straight to the logits (DH3), so the leak between areas is the whole problem; Theorem TQ5 (LIMIT-1, two-sided): as areas get confident, the least of a complete squeeze is e^-m times the leak least of Definition DQ5 (a least over output spans of convex problems), and the build from its minimiser is optimal to first order; Lemma TQ6: equal spills -> closed form, the units hold the areas as an equal-norm tight frame; the two-area pairing does NOT extend to 3+ areas (model/tq_ngauge.py); Conjecture DQ4: the leak least is hard in the number of areas. Essay hint (Shonu 2026-11-01): keep what areas SHARE. Proposition TQ7 (SECOND-ORDER, not refereed yet): one Fisher metric within 1 +- eta -> the closed-form shared span is within (1+eta)/(1-eta) of the least; premise FAILS on ResNet-18 (eta = 1 - 1e-8). Yet on ResNet-18 (orientation) the shared span with full reads loses 0.027 / 0.104 of the information (38 / 14 units) against 0.156 / 0.200 for the packed span with the same reads. Open, said plainly: why (Question DQ6: a claim that areas' own movements are built from shared patterns, plus a matching lower bound) ; the least at the network's own confidence; the leak least in closed form; an area squeezed below its own rank (DI1, DQ3). 2026-11-02 (LADDER 43): the shared span is the object on ResNet-18 (free search gains 11%, other starts end 2.5x higher; 54-61% of loss on near-ties); Lemma TQ8 (per-point KL sandwich) and Lemma TQ9 (certificate, valid but impractical) proved and refereed (REFEREE-BQ4); the planned two-sided TQ8 was false and was revised. 2026-11-02 (LADDER 44, plan 'small dropped part' with a 6-item definition of done): D5, D6 pass (37 of 40 pending nodes decided by delegation, 3 directional on card 3-33); D1, D3 partial (Theorem TQ10 proved and refereed, LIMIT-2, constants implicit; rank claim DQ7; Corollary TQ10.1 ONE-BUILD); D2, D4 FAIL (ResNet-18 not in TQ10's regime; certificate did not certify). NOT DONE. 2026-11-01 (LADDER 45, plan great-work-so-far-zippy-lagoon.md): the proxy areas were not real networks (TQ11), and the shared span is not the object on a real head; the trunk's real box is now measured (1.23 and 0.34 nats at 14 and 38 units), about 7 times above the free-code floor (0.18, 0.047), so one last-layer read cannot compute what k top numbers need to hold; the floor looks like class identity, an oracle, and a circuit through N is unmeasured (card 3-34); TQ12 ties the KL to N's own training objective; TQ14: under collapse a squeeze is a table of K rows and a group equals a circuit; map agents read the channels (23 of 24 held-out tests pass), and the district claim type is wrong for this layer; TQ16 and DQ11: pole build, a closed form at a loss above 0 (stationary, proved; least, conjecture; TQ15 certifies it within 2.5% on one small table). The pinning route DQ8 and the convex route TQ13 do not close it. NOT DONE (see LADDER 45 for D1 to D6). 2026-11-01 (LADDER 46): DQ11 is proved (Theorem TQ17): the first equality with a table's least above 0, with a matching lower bound, on a family that holds tables that are not symmetric; completeness at C logits now exists (DM6 item 5, Corollary TQ17.1: many groups, at total written rank C/2 only, node S-polescope); Theorem TQ18: a test of two numbers proves a build is the global least at any k and any code, with a read and kept units in view; Corollary TQ18.1 carries it to V_b(k). Checks at float error, on hand-built tables. Three referee rounds: every core proof held; many definition and scope errors found and fixed. Real networks miss: test ratio 1e23 to 1e78 on ResNet-18 and -50, pole form over 100 times above a free fit; lumping the tail (DQ15) also misses, by about 500 times at best on a typical row. The read gap is smaller on ResNet-50 (3.5 times against 5.4 times at k = 14) and stays above 1. Corrected (LADDER 47): no contradiction with the essay. TQ17's family is the essay's toy's world, and real fits hold relatives on one pole (blind test 0.79 against 0.43). "Over 100 times above a free fit" stays true of one-class poles at k = C/2, which is the wrong form on real tables. Lemma TQ20 states the exact global test; Proposition TQ21 is an exact case with relatives on one pole at several cuts of one table, narrow; Proposition TQ22 gives a lower bound by family sub-tables, with no tiny number. (LADDER 48) Two fan-out rounds: no slot filled on a real class; 19 results in B.Q.10; every certificate misses real tables by huge factors; at k = 14, 0.70 of the table's loss is N's ordering inside stranger families; real tables are not confident (mean tail 0.31); gates can remove the read's whole cost on toy tables, so the read is not the object. (LADDER 49) Three rounds, 25 agents, about 6.6M tokens: still NOT closed. Best proved floor at k = 14 on ResNet-18's table is .00029 against a fit of .245. No theorem shows a slot cannot be filled. New: the read's cost is the encoder, not z: a full read loses under 1% of the k = 14 read's loss. Box numbers corrected: read 1.172, not 1.23. Whether gates help is open. Rounds stopped as promised. Card 3-35 stays open and needs no answer. Card 3-34's stack is set aside. (LADDER 50) Theorem TQ23 proves DI1: on networks whose every claim holds, the squeeze least is NP-hard, at about 1 part in 8 m n, so the general form has no fast exact rule unless P = NP. NOT excluded: a fast rule within a factor of 2 or 10; so this does not explain the certificate failures. The instance is not shown to be trained-like. Two candidate claims measured on real tables, and both miss: product tail (.42 of stranger mass within a factor 3), one Ward tree (rms 1.43 against 1.81, control 1.80). Row 3 is still NOT closed: no slot filled on a table real networks are near. Card 3-36 is open and needs no answer yet (LADDER 51) Your end rule is on card 3-37: row 3 ends only by passing on real networks or by a proof that it cannot. Round 4: no candidate passes. Proposition TQ30: on clean networks, a fast rule within a factor below about 1.36 gives P = NP, at k = 1; Proposition TQ31: every fixed factor, under the Unique Games Conjecture. So IMPOSSIBLE's line I1 and, with no conjecture, I2 up to about 1.36 are proved; I3, I4, I5 are not. The tables are tables of graphs. Found: the 20-photo class-row table holds a large share of photo noise (half tables differ by rms about 3; a rank-14 fit scores .24 on its half and 1.0 on the other), so it is a poor stand-in for N's structure. (LADDER 52) TQ31 (every fixed factor, under the Unique Games Conjecture) now holds at any movement and any tail mass, on graph tables, in a per-area form with loss of the order of U / E: three readers on its core. With no conjecture the proved factor stays about 1.36, at k = 1. On sealed photos, no 2-layer decoder from 14 linear numbers beats the plain read on ResNet-18 (1.49 to 1.51 and 1.68 against 1.487): the read looks limited by its k numbers. That is a measurement and not a floor, and whether a gated box differs is open, since the gate run did not converge. Card 3-38 is open and needs no answer yet. Row 3 stays open | card 3-31: (c); card 3-32: answered by delegation 2026-11-01; card 3-33: answered 2026-11-01; card 3-36: open |
| 2 | not started | | | |
| 3 | not started | | | |
| 4 | not started | | | |
