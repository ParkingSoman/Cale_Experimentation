# Referee report: Part B.Q, Definition DQ1, Hypothesis H-active, Conjecture DQ1 (D.18), 2026-10-31

(Written by the math-referee agent; saved here by the main session because the referee has no write tool.)

## Summary
- Lean: `lake build` passes (9071 jobs); only style and `push_neg` deprecation warnings; no `sorry` in `Coarse/`. `pack_step`, `phi_nonneg`, `phi_le_neg` state exactly what is used; they cover one-line steps only.
- The core proofs are correct: every step of TQ1 part 2 (|(U_w' c)_i| <= |c|_2, the d_x formula, d = 0 on K_w, convexity-at-endpoints, pack_step) and the compactness proof of TQ2 part 2.
- The errors are in labels and conclusions: "row 3 is never exact" does not follow from TQ2; the OPT-KL label of TQ1 is not honest (ONE-BUILD); DQ1 does not meet the trunk's "eps = 0 exactly when the claim holds"; TQ2 part 1's proof uses objects of the old definition; TQ1's bound is not first-order tight in general (finding 9).

## Findings
| # | Item | Verdict | Reason and fix |
|---|---|---|---|
| 1 | TQ2 title; B.Q item 4; card 3-32 items 4, 5 | ERROR | TQ2 part 1 is a property of DQ1; part 2 shows L_S > 0. Neither shows row 3 has no exact closed form (TI1 is exact with L_S > 0). By the track's rule the "never exact" is our definition's fault. Retitle; "is any claim type exact for row 3?" is open; card item 5 becomes a real question. |
| 2 | Labels TQ1 parts 2, 4 | ERROR | 0 <= L_S <= packed <= sum eps_act; lower side 0 in the packing case; eps_act = 0 only when the loss is 0: ONE-BUILD. |
| 3 | DQ1 vs trunk | ERROR | eps_act = 0 iff Y = 0, independent of K and of the outside mass; no statement is measured. State a claim or call eps_act a summary (like DI1); then TQ1 is a bound in N-computed quantities, not T5's graded clause. |
| 4 | Proof of TQ2 part 1 | GAP | Uses Delta, phibar, b of the old form; rho_K can be 0 with Y != 0. Two-case proof: (a) rho_K(x) > 0 somewhere; (b) Pi_K Y = 0, Y != 0: some outside Y_i non-constant (Y centred). Add a Lean lemma phi(t) = 0 only at t = 0. |
| 5 | TQ2 part 3 instance | GAP | Give a network: inputs (sigma, c), g_1 = relu(sigma), g_2 = relu(-sigma), u_1 = relu(c + 2 + 10(sigma - 1)), u_2 = relu(c + 2 - 10(sigma + 1)); logits m(e1+e2) g_1 + m(e3+e4) g_2 + a(u_1 - 2 g_1)(e1-e2)/sqrt2 + a(u_2 - 2 g_2)(e3-e4)/sqrt2; districts {u_w, g_w}, region A_w, J_w = {g_w}, T_w = {u_w}. |
| 6 | TQ2.3 step "(1 - mu_2) J_2"; "Delta_max" | MINOR | use min_x s_x(K_2) >= 1 - mu_2 / p_min; D_A. |
| 7 | Notation | MINOR | r_w and Pi_w each have two meanings; rename r^K_w, Pi^K_w. |
| 8 | TQ1 tag; part 4; status | GAP | part 4 grades eps_lin: tag; cite TI2 part 2 and TH1, not TI2.1 (S-keepdrop DROPPED); status lists S-squeezecomplete. |
| 9 | DQ1 conjecture | GAP (important) | (a) Not first-order tight in general: outside logits fall by m on A_1 and 2m on A_2; v = (e1-e2)/sqrt2 + eta (e3-e4)/sqrt2, eta = m^(-1/2): loss <= C e^(-m)/m + e^(-2m+O(sqrt m)), eps_act(A_1) >= kappa e^(-m); ratio grows like m. Restrict DQ1 to comparable masses or refine TQ1 with per-node scales lambda_w (envelope (max lambda_w'/lambda_w) rho) minimised over lambda. (b) Route step (2) wrong cause: loss on A_2 is ~ mu_2 min(1, (a/eta)^2); balance gives ~ sqrt(mu_1 mu_2). (c) include TQ1.2 premises; N_m made by changing judgment weights. (d) chain rule over classes, not TI0. |
| 10 | TQ1.2 "can be > 0"; TQ1.3 | MINOR | prove via Grassmannian compactness; "needs" -> "uses"; example or delete "the sum". |
| 11 | TQ2.2 | GAP (easy) | generalise: k_S < rank Sigma_(W_S) => L_S > 0 (compact Grassmannian; F(Q) = 0 forces every Y_w(x) into Q). t in image of P_perp; f_w(0) = I_w from TI0 + Gibbs. |
| 12 | DQ1 rescaling remark | MINOR | eps_act reads logits only; F1 rescaling and copies leave logits unchanged: invariant. |
| 13 | numbers/claims outside Part C | NOT MATH | "60 nats", "far below", H-active stated as fact, Part C sentence inside a proof: move/qualify. |
| 14 | H-active | MINOR | no threshold, quantifiers, "well-trained" undefined; forward reference to Part B; measures the bound, not the loss. |
| 15 | TQ1.4 switch cost | NOT MATH | give a cost model or say "selects stored per-node arrays". |
| 16 | tags too wide | MINOR | TQ2.1 (under: none); TQ2.2 no active-class claims; TQ1.1 no R-active, S-squeezecomplete. |
| 17 | completeness premise on a graded bound | GAP | in the trunk completeness restricts only exactness; card it. |
| 18 | labels outside the four | MINOR | use the bar's labels. |
| 19 | literature | MINOR | packed unit = "superposition of mutually exclusive features" (Elhage et al. 2022), derived not assumed: add a compatibility-table row. |
| 20 | DQ1 (1)-(3), TQ1 cores, TQ2.2 core, Lean | OK | checked. |

## Content count
Structural: TQ1.2, TQ2.2, TQ2.3 (3). Bookkeeping: TQ1.1, TQ1.3, TQ1.4, TQ2.1, DQ1 (3) (5). The upper bound has no matching lower bound in the packing case (DQ1 unproved).

## Most important fixes
1. Withdraw "row 3 is never exact"; relabel TQ1.2/TQ1.4 ONE-BUILD; DQ1 a summary; "is any exact form possible for row 3?" to Shonu on a card.
2. Repair TQ2.1 (two cases), give TQ2.3's network, fix the (1 - mu_2) step, notation, generalise TQ2.2.
3. Tightness: prove the uneven-confidence NEG; restrict DQ1 or refine TQ1 with per-node scales.
