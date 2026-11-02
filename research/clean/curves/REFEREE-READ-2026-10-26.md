# Referee report 23 (2026-10-26): the read index additions

Referee: the `math-referee` agent. Fixes: Appendix of COARSE-THEORY.md, "Referee report 23".

Correct: Lemma T46b for unit readers; Lemma T46c; Proposition T50 (every step recomputed; factor 1, not 1/2); T49 and
T48 for the weak form after report 22 (T49 factor c / T_u, not c / (2 T_u)). lake build passes, no sorry.

1. Two items named D35. 2. Item 5 cited a Part C result that was "not yet run". 3. Many statements did not say which
form of perfectly clean; eps had no agreement entry; T46 did not cover xi. 4. T46b did not cover class readers (fix:
dS/dt = Var_q(w)). 5. lambda_ind not defined for positions and pools. 6. T50 does not need the uniform law. 7. Comment
(f) out of date: a unit read only by readers that read it alone has xi = 0, so it is never agreement-clean (width-1
layers, copy chains). 8. The OR claim: for r = relu(W h_u + R), W > 0, no bias, R >= 0, xi = Cov(1[h_u = 0], 1[R = 0]);
with a threshold the sign can flip. 9. "T43 no longer refutes" overstates: it shows T43's networks are outside the
class, not that the class avoids hardness or is non-empty. 10. The agreement form excludes the family (apart members)
read by one coarse positive reader, against Remark B.K.1 (d). 11. Hidden nodes on paths (R-explaw; D35's path).
12. D35 not precise; route: nested helper detectors b = relu(<x~_i, x_(j)> - r') with sqrt(1 - eta/2) <= r' < r keep
T43's labels on the mix; only xi > 0 remains; if it works, the agreement form does not rescue Shonu's conjecture.
13. The H-route (3) verdict depends on the denominator (23/30 fails, 23/36 holds). 14. No Lean for the new results.
15. Small points.
