# Standing check: flexible method vs fixed reference

Shonu, 2026-10-20. This is a living rule. Edit the test as the methods change. Keep the idea.

## The question

The network is already trained. We refit it for different sizes with ONE flexible method. For any size, how does
that compare with spending resources to make that specific size good (a fixed reference with the same storage)?

## How to read it

| Result | Meaning | Action |
|---|---|---|
| Flexible matches fixed, or is very close | Very good. The flexible method can serve every tier. | Keep going. Test more circumstances. |
| Flexible beats fixed | Suspect the fixed reference first (too short, wrong objective, under-trained). | Strengthen the reference, then recheck. |
| Fixed beats flexible | Gains are left on the table. | Use the ladder (below) to find which part of the method misses them. |

## The ladder (template, first used in V2)

Rungs allow more freedom at EQUAL storage. Each gap between rungs names a cause.

| Rung | Freedom | A gap above the rung below means |
|---|---|---|
| A | The flexible method, one-step solve | (baseline) |
| B | The method's own weights, short fit | One-step solve is not enough |
| C | The method's own weights, long fit | The short fit stops too early (fitting bug) |
| D | All zero-storage weights near the layer, long fit | The method's class is too narrow |
| E | Whole model retrained at that size (GPU) | Gains need changes far from the layer |

## When to run it

- After every method change, as the smallest version first (1 layer, 2 widths).
- Across sizes, layers and text types before any claim that the method "works at every tier".
- Long fixed references (rung E, long trainings) go to a rented GPU.

## History

| Date | Test | Finding |
|---|---|---|
| 2026-10-20 | Q8 | A 2-min trained layer lost to RCF + window everywhere. Not a valid ceiling. |
| 2026-10-20 | V2 (L16, K48 / K768) | B ~ C: no fitting bug. D > C by 6-23 points: the fold class is too narrow. |
| 2026-10-20 | V3 (L16) | K48: layer L-1 gates carry the D gain. K768: kept gates and L-1 gates each about half. |
| 2026-10-20 | S1 / S1b (L16) | One shared set of L-1 gate changes for K48 + K768 keeps ~.6 of the per-width gain at K48, less at K768 (code). Sharing costs gains. |
| 2026-10-20 | G (L16, L22) | RCF-G (fit kept gates + L-1 gates too) beats RCF by 6.5-11.5 points. A cut start with the same free weights nearly catches up by 150 s. |
| 2026-10-20 | G75 (K1152) | RCF-G unstable at 75% width: L22 prose 34 -> -11 between 75 s and 150 s, below the trained layer. |
| 2026-10-20 | ES (K768, K1152) | RCF-G + early stopping: no collapse, beats the 2-min trained layer at K1152 on both layers. Pass 3 of 4 cells. |
