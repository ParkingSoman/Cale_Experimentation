# Referee report 17: Part B.O, D.10, H-dialfit (2026-10-25)

Referee: the math-referee agent. It read only the paper (and DECISIONS.md rows), not the log.

| Item | Verdict | Reason | Fix applied |
|---|---|---|---|
| Lemma T36 | proof OK; tag gap | assumed D20 and an invariant mix under "none" | premise weakened to N's equivariance on the support of P (D20 implies it); builds = D15; path R-symN |
| Definition D28 | gap | subgroup only for invariant P; read from parameters in T37's comment | stated in T36's setting; defined on values on P; "picks out a direction" removed |
| Lemma T37 comment | overclaim | T32's build has mix points on kinks; T32 and T34 do not prove their builds least; "explains" unproved | comment restricted to T20 and T33 (proved least); "explains" removed |
| Theorem T38 | proof OK, citation gap; title overclaim | Clarke chain rule recalled, with unnamed facts; for k >= 2 a unit fits the partial residual, not N | proof replaced by one-sided derivatives and a nearest-point argument (self-contained, stronger); retitled as a backfitting fixed point; "a finite mix" in the tag; part 1 gloss split D2 / KL |
| Corollary T38.1 | gap | uniqueness needs a span condition; the average needs mass off A; w = 0; no-kink premise droppable | restated with kink weights lambda and 1 - lambda, w not 0, the mass condition, the span condition for uniqueness; "any affine function" removed; KL version stated for D_P |
| Corollary T38.2 | error; tag gap | with c fixed the sign is +1, so least only among sign +1 builds; identity metric, no pooling and the bias shift d are narrowings | conclusion fixed (sign +1, T38 with fixed signs); conditions made explicit; d = 0 case added; tag two logits, one hidden layer, a finite mix |
| Corollary T38.3 | proof OK; title and comment overclaim | the condition is a normal equation with correlated errors, not a fit of a target; the log was cited in Part B | retitled; E_P weights and the d-condition added; log sentence removed |
| Remark B.O.2 | error for k >= 2; overclaims | "best fit on each region" and "average where no unit fires" fail unless regions are laminar; T32 not proved least; "what distillation converges to"; detector-departure claim | restricted to width 1 (and laminar regions); T33 as the example; "what every least build satisfies"; Definition D31 (dial, detector) by firing mass; symmetry breaking cited from T20 |
| Clarke, Palais rows | gap / minor | load-bearing recalled row; Palais hypotheses "none" | Clarke row marked "not used"; Palais row lists a compact group and a smooth invariant function |
| D29 / D30 status | overclaim and error | a solver cannot refute an equality without a certificate, nor a strict inequality at all; T32's mix has no silent point | "numerically contradicted" / "no gap found at 1/10^6"; T32 sentence removed; T38 called a necessary condition |
| H-dialfit and Part C | gap | "close" undefined; slope 1 prediction missing; R^2 measures t; held-out vs in-sample; readers pooled; no d in the crush; node text stale | eps_dial = 1 - corr named; slope reported; R^2 labelled diagnostic; held-out split described; pooling and d = 0 stated as failed premises; node R-dialfit text updated |

Not changed: the D2 route claim of D29 (the referee found it correct). Open from the report: an exact D2 certificate for
D29 before calling it refuted; H-dialfit with T38.3's condition and the trained objective's second-order metric G.
