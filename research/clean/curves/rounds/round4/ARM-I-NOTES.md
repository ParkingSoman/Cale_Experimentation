SUPERSEDED in part (2026-11-02): the referee found a route that avoids the obstruction below. See Propositions TQ30 and TQ31 in COARSE-THEORY.md B.Q.12. The notes below are the record of the attempt.

# Round 4, Arm I: working notes (not the paper; nothing here is refereed)

Date: 2026-11-02. Plan: `~/.claude/plans/i-need-you-to-tingly-orbit.md`. Card 3-37.
Sources read at the text level: `sources/` (Chawin and Haviv 2025, Hardt et al. 2014, Huang 2013).

## What the sources give

| Source | Statement used | Read at |
|---|---|---|
| Chawin, Haviv 2025, Def 1.4, Thm 1.5, Cor 4.15, Lemma 4.13 | Completion gap: YES has B in [-theta, theta], rank <= d1. NO: every B in [-theta, theta] within eps on exposed entries has rank >= d2. Constant eps <= 1/7, theta up to 2^(2^(c d)). NP-hard. The partial matrix is A = 1 on the diagonal, 0 on edges of a graph H, a hole elsewhere | ch.txt lines 164-180, 1073-1100, 1190-1225 |
| same, Lemma 4.13 forward | YES instance is PSD with unit diagonal, so its completion has all entries in [-1, 1], even when theta in NO is large | ch.txt 1105-1115 |
| same, Thm 3.5, Thm 3.8, Lemma 4.7 | From X, Y with row norm <= theta, XY^t eps-fits G: coloring with (4 theta^2/(1-eps) + 1)^d colors. On line digraph: (2 theta/eta + 1)^(d m(d, 2 eta theta + eps)) colors of G | ch.txt 557, 638, 862 |
| Huang 2013, section 3 | Weighted FGLSS graph. YES: O(K^3) colors, perfect. NO: every independent set has weight fraction 2^(-Omega(K)). Deterministic reduction, NP-hard. Query-config vertices, conflict edges | hu.txt 160-230 |
| Chan 2013 via Huang | K-colorable on 1 - nu of vertices, NO: independent set at most 1/2^(K/2) + nu. Not read at source | hu.txt 67 |

## The reduction shape (fits Definition DQ18 with no change)

N(M, W; a, B) of Definition DQ18, with M = A (holes set to 0), W_wj = d on exposed entries and 1 on holes. All map claims hold at eps = 0 (already stated in DQ18).

Row w, class j, point x in A_w, e_xj = (l'[j] - l'[g]) - (y[j] - y[g]). From the pair {j, g}: KL(s_x || q_x) >= s_x[j] min(e^2, 1) / 38 (TQ23.a Step 1).
G_wj = (V_j - V_g)^T (c(x_w^+) - c(x_w^-)), rank <= k, G_wj - a M_wj = e_(x+),j - e_(x-),j.

YES: k >= d1 units write +- (a/2) B, exposed entries exact, holes move +- (a/2) B_wj with |B_wj| <= 1. Loss <= (a^2 tau / 8 m) * sum over holes of B_wj^2, up to the factor (1 + 3 alpha).

## OBSTRUCTION FOUND (2026-11-02): one bad entry cannot pay for the holes

Let delta = 1/d be the hole weight against an exposed entry. Weights below are in units of an exposed entry's tau.

```
NO, as stated, gives one bad entry:   D_NO  >= (1/2m) * c * (a eps)^2
YES pays on all holes:                D_YES <= (1/8m) * H * delta * a^2,   with H about m n holes
to read off |B| <= theta on a hole:   delta * (a theta)^2 >= (a eps)^2,    so delta >= eps^2 / theta^2

ratio R = D_NO / D_YES  <=  c' * theta^2 / (m n)
```

theta is a constant for fixed d. So R < 1 for large instances. This route gives nothing as stated.

This is not a detail of the proof: with hole entries of size poly(n), low rank fits become easy (Lemma 4.7 counts colors as theta^(d m), and the count exceeds n). The holes must be held at constant size, and one bad entry's budget is too small to pay for that in every row.

## THE REPAIR (a route, not proved)

Two changes.

1. Rms in place of sup. A budget on a whole row bounds the root mean square of the row's holes, not each hole. SVD factors X = U S^(1/2), Y = V S^(1/2) give |X|_F^2 = sum sigma_i <= d^(1/2) |B|_F. So most rows of X and Y have bounded norm (Markov), with no factor n. The hole mass mu = n delta (in units of tau) can then be any small constant.

2. Many bad rows in place of one. This needs a NO case in which every low-rank fit fails on a constant fraction of rows. Then m drops out of the ratio: R about rho * eps^2 / mu. Any fixed factor follows, by taking mu small, at constant a. So the loss is a constant share of the tail mass, not small: this would also reach the non-quadratic regime of I4.

Why many bad rows may be available.

```
fit on good rows, rank d           ->  (Thm 3.8 form on the good edges)  an f-coloring where every good edge is bichromatic
                                       with f = (2 theta / eta + 1)^(d m(d, eps')), a constant
Huang:  YES O(K^3) colors              rank d about log K on the line digraph, since b(d) >= K^3
        NO  alpha <= 2^(-Omega(K))     and log f is polylog(K) to K^small, far below K
monochromatic edges <= rho  ->  a color class of weight >= 1/f that spans edge mass <= rho
```

Missing step, the real gap: from "alpha small" to "any set of weight >= 1/f spans edge mass >= rho" for a chosen edge distribution on the FGLSS graph. Turan alone gives rho about 1 / (f alpha D), with D the degree, so it is useless on a dense graph. A route inside the PCP: take the distribution pick (function, table point p), then two queries reading p, conflict pairs only. At each point remove the minority answer. The removed weight is at most (edge mass at p) / (density at p), times the K^2 points a vertex reads. This turns a nearly independent set into an independent set of about half the weight. So rho is about 1 / poly(f, K), a constant. Not proved. It requires working with Huang's PCP, not only with its statement.

A randomized sparsification is not a shortcut: the union bound needs Turan's edge count, which is the same gap.

## Direct route, no line digraph: why it does not work today

Fit G directly, rank d >= colors. Unit-vector case: a cap of angle about 45 degrees holds an independent set, so alpha >= about 2^(-d/2) / poly(d). Chan's NO has alpha about 2^(-K/2) at K colors. These match up to poly(K), with the fit on the wrong side of the factor. So the direct route has no contradiction. Non-PSD X, Y costs more. Noted because the near match says the rank-K region is exactly at the edge.

## What this route cannot deliver

I5: the hard tables are PCP tables. I expect them to look nothing like a natural network's. Any "trained-like" statement is about stationarity, range and mass only.
