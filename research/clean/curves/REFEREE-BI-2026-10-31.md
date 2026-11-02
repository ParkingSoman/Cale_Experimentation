# Referee report: Part B.I, Part D.17 and the Remark (KL) of B.P (2026-10-31)

Agent: math-referee (reads only the paper and the Lean files). Lean `lake build` passed; no `sorry`.

**Bottom line (referee).** Every core proof is correct. No false step in TI0, TI1 parts 1 to 4, TI2, TI2.1 (except its
last sentence), TI3 parts 1 and 2, TI4 parts 1 to 3, or TI5 parts 1a, 1b and the numbers of part 2. The defects are
in the statements around the proofs: one false summary sentence ("only when" in item 5 and card 3-31); one overclaim
about the trunk (TI2.1 Comment); two premises missing from tags (TI2.1's last sentence, TI3 part 3); one claim stated
without proof (TI5 part 2, "directions tend to"); one uncertified computed result inside Part B; some labels missing
or misnamed.

Content count (referee): of the 7 proved results, 3 structural (TI3 part 2, TI4, TI5) and 4 bookkeeping (TI0, TI1,
TI2, TI2.1). The three that pass the bar (TI1, TI2 part 2, TI2.1) are bookkeeping; the structural ones do not reach V.

## Findings

1. TI2.1 Comment and card 3-31 (b): overclaim. T5's graded clause asks for a bound in every claim's eps; TI2.1 grades
   eps_lin only, every other claim exactly 0; DM6 completeness is not extended to DP1 or DL1 claims. Qualify labels.
2. B.I item 5 and card 3-31: ERROR. "D2's directions are KL's to this order only when the mean softmax is uniform" is
   false: sigma = (1/2, 1/4, 1/4), Sigma = l1 v v^T + l2 u u^T with v = (2,-1,-1)/sqrt 6, u = (0,1,-1)/sqrt 2, l1 > l2:
   top KL direction v is also D2's. Write "if", or "need not be otherwise".
3. TI2.1 last sentence uses TL4, which needs TL3's premises (not in the tag). Use TP3 part 1 instead.
4. TI3 part 3 uses on-inside and linear-read claims (through TI1, TI2); not in tag or path.
5. TI5 part 2 "KL-least directions tend to span(2,-1,-1)": not proved. Proof route: TI4's lower bound for a choice with
   span in a fixed line Q' has a constant uniform in Q'; psi has a unique minimiser; compactness.
6. TI3 Comment holds an uncertified computed result inside Part B: move to DI2. TI5's Lean line: drop "compared
   numerically to 4 digits".
7. Labels: TI2 part 1 = ONE-BUILD; label TI3 parts 1 and 3; TI4's label must not contain "OPT-KL"; TL2 is an
   inequality, not EQ.
8. DI1 Comment: I(R) is m_R times the mutual information; the D2 least at k = 0 is (1/2) E_w; I_w is the KL least at
   k = 0 only when J_w is flat.
9. B.I Status: list R-oninside / R-linread for TI2.1, TI5, DI1, DI2; "TY4 carries every item" is not proved (DL1, DP1
   are not defined for channel units with positions); TI0 and TI1 parts 2 to 4 do carry over.
10. TI1 part 2: say S' has k = 0, the nodes of W' carry the on-inside claim, S' and the S_i meet TH1's premises.
11. TI1 Comment: "only TH1's premises plus flat kept judgment units", not "only the district claims".
12. TI3 part 2: padding units when k_S > k; one line on why the minimum over Q exists.
13. TI4 H-Lipschitz step: use |E[(e-Ee)(v-Ev)(u-Eu)]| <= 2 |e|_inf sqrt(v^T H v u^T H u) <= |e||v||u|, c_0 = 1
    (DI1's route step (4) needs the dimension-free constant).
14. TI4 Comment: with several areas the judgment units' logit weights move too, not only the bias.
15. TI5 1b wording: one network on two mixes; "both mixes' B_w".
16. Lean `kl_chain` proves the algebraic core only; say the choices of lq, lb_c and the column sums are by hand.

## Items judged correct (referee)

Lemma TI0; TI1 parts 1, 3, 4 (part 3's y_N = c_N on R_0 from district claims alone); TI2 parts 1, 2; TI2.1's two
displayed relations and proof; TI3 part 1 and part 2's realisability, closedness and attainment; TI4 parts 1 to 3 (Taylor
constant 4/3, Pinsker bound on beta, lower bound by pointwise relaxation over any box, flat J_w, Ky Fan in the H
metric); TI5 (equal (E_w, Sigma_w, B_w), every claim at eps 0, the integer inequality, the concavity, the eigenvalues
1/4 and 3/8, the values 1/12, 1/16, 3/32); the Remark (KL); DI1 and DI2 are honest conjectures with routes.
