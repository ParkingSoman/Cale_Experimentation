# I5: does TQ31's hard network pass the essay? (agent's report, filed by the main session; the agent's tools refused the file write)

Verdict, as returned:

1. N itself is a lookup table with a small toggle. Not read as clean.
2. The numbers of D42 and D45 do not say this. On eps_V, N scores .08 to .41 and ResNet-18 scores .91 (a sample of 120 units, 40 points per unit).
3. The hard pattern does not need N. Build E has all computed errors of D42 and D45 at 0, and it carries a hard table of a new family, with family gates, shared poles and a coarse reader. These principles do not separate hard tables from clean ones. E was adjusted twice to reach 0.
4. The reason: TQ31's lower side reads only the table and the kept offsets, not how units carry the table. A principle about units separates only if it forces something on the table.
5. Most expected separators fail: hard tables score better than ResNet-18 on product error and spectral closure, and better or about equal on family closure once they have families.
6. One number separates in the right direction: closure of the competition graph, eps_cc = 1 - 3 triangles / wedges, on reciprocal competitive links. Weak: ResNet-18 is at .76. It makes no least computable, and it caps a factor at k = 1 only.
7. The sharpest difference: TQ31 lives at k = 1 with a private unit per area, so a free code is a real code. At k = 2 the fit falls to about 0 on sparse graphs (C5, sub(K4), Petersen: under 2e-5; K5: .019). On ResNet-18 the code is a read of 512 units for 20000 rows.

Advice: do not count I5 as met. If I5 means "error 0 on D42 and D45", build E meets it on these instances. That shows the principles are too weak. It does not show hard tables are clean. E reads as clean inside a family, and it has no shared cause across families. When that cause is added (build G), the errors rise.

Builds, all hand-built, table error 0 on P. Errors eps_V / V on pairs that matter / eps_G, eps_B / eps_pack / eps_unread / eps_agree.

| Build | What it is | Errors |
|---|---|---|
| N | DQ18 | .08 to .41 / .38 to .86 / empty / 0 / 0 / 0 to 1 |
| A | area unit plus two specialists never on together | 0 / 0 / empty / 0 / 0 / 0 |
| B | A with a gate and point units in front | all 0 |
| E | square table, family gate, shared poles, coarse reader, class judgments | all 0 |
| F | two poles shared by all areas of a camp, plus exception units | .10 to .26 / .21 to .40 / empty / .08 to .14 / .04 to .09 / .04 to .09 |
| G | E plus one unit per edge, on in both families, carrying the side link | .45 to .60 / .68 to .83 / 0, 0 / 0 / .15 to .17 / .15 to .40 |
| ResNet-18 | last hidden layer, sample | .91 / .89 / not in the cache / not in the cache / 0 / 0 |

Reuse scores worst, and lookups score best. So D42's V, at the last layer, rewards a network for not reusing anything. Comment (g) of D42 warns of this.

Comparison with ResNet-18: side links reciprocated: hard tables all, by build; ResNet-18 .03. Family links reciprocated: ResNet-18 .75. Share of a row's movement in one direction: hard 1, ResNet-18 .64. Effective off-own entries in a row: hard 2 to 6, ResNet-18 50. On ResNet-18, 92% of head links move against own, so the sign pattern is not alien. A family plus side links in a row's head is not a difference: it can be built.

Free code energy captured on ResNet-18 at k = 1, 2, 4, 8, 14, in the row's metric, on photos V did not see: .29, .44, .60, .75, .85. The read captures far less. Agrees with LADDER 52. One fit on one network, not a floor.

Scripts and logs in this folder: netlib.py, sqlib.py, a_read_N.py, b_rebuild.py, c_square_fit.py, d_compare.py, e_more.py, f_last.py, b_rebuild.log, c_square_fit.log, d_compare.log, e_more.log, d_compare.json. Not done: no build trained; G, B, pack not computed on ResNet-18; ResNet-50 not measured; D46, D21 not computed.
