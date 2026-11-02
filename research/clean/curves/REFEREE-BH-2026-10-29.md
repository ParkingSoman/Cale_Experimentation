# Referee report: Part B.H (DH1 to DH3, TH0 to TH3) and D.16 (TH4), 2026-10-29

Referee: math-referee agent (reads only the paper and the Lean files). Lean: `lake build Coarse.Recur Coarse.Map`
passes, no `sorry`.

Checked and correct: TH0 parts 1 and 2 (kept J_w units, units of the same layer, idle units, units read by box tops);
TH1 lower and upper bounds, parts 2 to 4, the D2 version; TH2 part 1 restriction and (r) identity; the TH2 part 2
instance (gap 0, 1, 0, 1 = x_2 on P, every claim 0, monotone gaps at G_1 and G_2, exact at G_3); every number of TH3
(Sigma_1, Sigma_2, totals 7 and 3, eigenvalues, bounds 3/2 and 2, m_b = 0, bound 3); "every (r) sequence" justified.

| # | Kind | Finding | Fix applied |
|---|---|---|---|
| 1 | ERROR | TH4 ill-posed (no unit for summary size, no time bound); with poly-time summaries it conflicts with T57/T22 (one node, area P, k = 1) | TH4 restated in oracle form: merge map poly-time on rational summaries, single-node problems by an oracle; Route (1) case named |
| 2 | ERROR | "TM1 is the case of TH1 with parents in L - 1" overclaimed: TH1 needs a tree map, districts in every layer, shelter on J_w districts (unused), whole nodes | wording: TH1 at L - 1 is the case of TM1 for groups made of whole nodes; unused shelter premise dropped (shelter only on districts that meet T(S)) |
| 3 | ERROR/GAP | sibling merge / lift omit the step that creates a group from kept units; "every changed group is a lift" (TH2.2) and "all three are sibling merges" (TH3) false as written | creation step defined, counted as a sibling merge; TH2.2 and TH3 wording fixed |
| 4 | ERROR | D.16 Route (2) false (two units switching inside their areas, opposite gap signs); same in TH3's comment | restricted to the case of Route (1) |
| 5 | GAP | T61 claim rested on a script | proved in the text from T61's pre-activation table |
| 6 | GAP | TH2.2 "infimum attained" not shown; "(r) still reaches V_b" needs least builds | wording: V_b >= min over monotone vectors > 0; (r) remark made conditional on least builds existing |
| 7 | GAP | DH3 item 2 lets one node lie in two groups | sets W_S pairwise disjoint added |
| 8 | GAP | build wiring of new units unstated | sentence added to DH3 item 4 |
| 9 | BOOKKEEPING | v_S undefined when S's areas cover P | parenthetical added |
| 10 | BOOKKEEPING | DH2 copy invariance via DN1 (layer L only) | limited to DN1 copies |
| 11 | BOOKKEEPING | DM4's one-judgment-per-region rule vs repeated areas across layers | per node (l, R) |
| 12 | BOOKKEEPING | box layer wording; "area of every other node is empty"; fact (ii) misses J of other w in W_S | fixed |
| 13 | BOOKKEEPING | premises at every level follow from G_m; stronger statement: V_b monotone along sibling merges | TH2 part 1 restated as monotonicity, exactness as its corollary; inline identity cited |
| 14 | BOOKKEEPING | TH3 residual formula needs vhat in image of P_perp | added |
| 15 | GAP (hidden restriction) | "plain feed-forward" on no path | new SCOPE node S-plainff (PENDING), alias, card 3-24 question 4 |
| 16 | BOOKKEEPING | Lean coverage thin | Lean lines say what they cover |
| 17 | BOOKKEEPING | TH0 tag; TH3 is a one-hidden-layer result; card TH4 row | fixed |

Content count (referee): TH0, TH1 bookkeeping; TH2 part 1 structural (light), TH2 part 2 and TH3 structural
(counterexamples); TH4 ill-posed as first written.

Verdict (referee): Part B.H stands as mathematics after corrections. Every proof is correct for its instance or bound
and none needs a measurement, except the T61 comment (finding 5). Round-1 fixes not re-refereed.
