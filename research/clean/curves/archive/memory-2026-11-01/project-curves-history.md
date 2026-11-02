---
name: project-curves-history
description: "Curves track history before 2026-10-28 (later): activations vs edges, unit branch, zoom tree, clean definition, trunk resets, the InceptionV1 curves crush and the weights-only / exactness sessions"
metadata:
  type: project
---

Archive of [[project-curves-track]] (moved 2026-10-28 so the current file stays readable). Point-in-time notes.

## Before (2026-10-28): activations vs edges (UNIT-BRANCH.md A.5, A.6, B.6, B.7, D.3)
Shonu: the essay says activations mislead; understand why mathematically before choosing the next unit. Result: an
activation alone drops a unit's in-votes (TA3) and its readers (TA1, TA2, TA4); computation = weights x ON/OFF gates on
paths (TA5). Our co-firing unit (TU8) lets a box give an activation a NEW write in any direction (DU2 tethers only by graph),
so a softmax-unread unit (column alpha*1) still breaks U4 (TA6; a zero-column unit is already safe; fails under DU9); readers return only when the write is fixed (TA7: co-firing x reader overlap; two logits hides it). Fix
direction: write-tethered boxes (DA2, node R-edgetether PENDING, card 3-17), Proposition TA8 (tethered: help needs overlap in T's columns), graded Conjecture CU7. Next step: the edge unit
(sets of paths open together, CU2). Lean `Coarse/Edge.lean`; checks `research/clean/units/ta*.py`.

Started 2026-10-22 after the D-series (ConvNeXt block drop) stopped. Shonu wanted the essay's "curves & loops"
experiment (InceptionV1 3b crushed 480 -> 10) reproduced quickly, a training reference he can trust, then a
mathematical derivation of coarse from fine. Read `research/clean/curves/CURVES-LOG.md` first.

Key decisions and facts:
- Separate venv `research/clean/curves/.venv` (torch + lucent, MPS). Lucid InceptionV1 via `lucent.modelzoo`.
- Whole-layer crush into frozen later layers is hopeless (rank-10 PCA of 3b: KL 15.9). Training all later layers is
  unstable. So we crush ONE FAMILY: the 10 curve neurons -> k coarse neurons (`family.py`, exact smaller network).
- Removing the curve family costs KL .59; k=1 trained wins back most of it. The k=1 neuron is the essay's
  "curves & loops" neuron: a centred-loop detector (feature vis = one loop; crops = wheels, lenses, dials).
- Found law (function space): coarse(q) tracks sum over the family of each fine neuron moved to its own centre of
  curvature (group average over rotations about q), corr .52 vs .33 plain sum, reversed control ~.1.
- Readers (4a) read the family mostly by its lowest orientation harmonics (h0 average, h1 one wave).
- Pre-registered derivation (sum of moved kernels) FAILED (.366 vs removal .64). Fix path: project the moved ReLU sum
  onto one neuron (read), variance-keeping write (Bayes/|rho|). Best derived k=1 .242 dev; trained .140.
  The WRITE is the main gap (1x1 branch; training adds a loop->not-straight-line gate on 60 4a neurons).
- k=3: keep three members ~120 deg apart + Bayes estimate of the rest. SEALED: .139, LOSES to random-start trained
  k=3 (.122) by .017 (se .004); the dev claim that it won was noise (retracted). Random-start k=3 stays at the k=1
  level (.121). Trained from the derived start: .037, beats k=10 from random (.041). Report:
  research/clean/curves/CURVES-REPORT-2026-10-22.md (sealed cold test in section 6).
- Phase 5: circle family in 3b (11 members, tile radius 6-20 px) repeats it on sealed: derived start beats random
  start by .021 (se .0014); even-radius rule beats naive at k=2. Code: p5_drop.py, p5_train.py, p5_cold.py.
- Open: why the gate; why k=3 random fails but k=10 random works; k=1 vs k>=2 recipes not unified; middle/late layers.
- Shonu warned: be careful with standard interp tools (essay: they can hide cleanliness). Use raw activations,
  both poles, feature vis next to natural crops, no SAEs or simulation scores.

- 2026-10-22 night: new plan ~/.claude/plans/okay-i-want-you-optimized-perlis.md. Theory one-pager research/clean/curves/THEORY.md.
  RESULT (sealed, k=1, no training): write = minimize the intact net's second-order output damage at the 4a
  readers, ReLUs exact via one active-set step: .135 vs trained .122 (paired +.014, se .003); vmr .250, Bayes .311.
  Plain Fisher write .176 (needs all 35000 train images; 1500 over-fits). Read is not the gap (derived read = trained
  read under the Fisher write). Why training won: the plain Fisher write wakes readers that are off (zero gradient).
- BUG 2026-10-22: lucent inceptionv1 defaults to redirected ReLU (leaky 0.1 backward). load_model() now uses
  redirected=False; fv.py keeps True. True-gradient k=3 training matched the old run to step 1000 (rulers stand).

Related: [[feedback-derive-like-mathematician]], [[feedback-research-essay-rules]], [[project-map-track]].

- 2026-10-23 WEIGHTS-ONLY session (PAPER.md, log "WEIGHTS-ONLY"): Theorem 1 balance HOLDS on weights (3b median
  1.02, 3a .996; bottlenecks .76-.91 = open hole). Theorem 2: weights = error-weighted, gated TRAINING moments
  (direct check impossible on val: stable gradient orthogonal to theta). C1: max kernel overlap gives the curvature
  shifts from weights (direction right, scattered, r 1.5). C3: "1.5% in span" RETRACTED (raw-metric artifact; 61%
  in input metric). C4: moved sum = .42 corr with trained (twins .79). C5: weights-only read + same fitted write
  .357 vs fitted read .241. Theorem 5 (PROVED): adding self-labeled images leaves weights stationary but changes all
  plain moments -> no weights-only Phi in general. Fork for Shonu: plain moments from universal pixel statistics
  (dead leaves) pushed through the weights, vs strict weights-only limit.

- 2026-10-23 MIX session (plan ~/.claude/plans/i-d-like-you-to-squishy-yao.md, log section "MIX"): Shonu decided the
  image mix Q is a declared input (generalist = training mix; developer = list of capabilities). The generalist bar =
  the existing trained k=1 ruler (distilled on the train split). First job: data hunger (Fisher write unsettled at
  35000 images). Checks B0-B5 + cold test; PAPER.md sections 8+.

- 2026-10-23 afternoon BEAT session (plan ~/.claude/plans/okay-i-d-like-you-enumerated-wreath.md, log "BEAT"): audit
  verdicts: T4b theory weights NOT separable from simple rules (uniform cos .85-.93); W2 "gate" sign is already in the
  original weights (Bayes and plain column sum show it too) -> not a Fisher discovery. Work moves to k = 3 (random start
  stuck .122, member start .037 = headline bar B). Corollary 11.1 bumps never built.

- 2026-10-23 evening DATA-FREE session (plan ~/.claude/plans/i-want-you-to-majestic-milner.md, log "DATA-FREE",
  PAPER.md Part IV): generalist = no images at all. W4 logged: THE FORMULA (FITTED now) .0507 dev vs bar B .0361;
  M read + step1 .0443. Steps: strict baseline (Prop. 4 + balance gain), literature, conjectures A-E, assemble, cold.
  RESULTS (same evening): D1 strict Prop. 4 build FAILS (.754, worse than removal .664: column-sum cross-talk 15-20x).
  A (weight norm = activity) FAILS (Spearman .29); B (kernel overlap = co-firing) HOLDS (.76). E: class inversions
  (plain pixels, 256 steps, `e1_selfmade.py`) replace real images for the READ + lsq write at k = 3: .0696 vs real
  .0740 at equal count. But the Fisher write FAILS on them (.524 vs .174): the network is certain on its own inputs
  (1 - sum p^2 .0049 vs .125), Theorem 5 in action. Open fork for Shonu: boundary inversions (class a + nearest
  class b, 50/50) for the write; and whether a fixed generation rule counts as DERIVED. Paper sections 16-18.
- GENERATE (P1/P2, same night; PAPER.md Part V sections 19-22, log "GENERATE"): I1 InceptionV1 over-confident (ECE .26;
  irrelevant to P2). I2/I3: Fisher write ignores sure inputs (10498 unsure = all 35000). I4: logit level ranks natural
  but unbounded; conv1 has no pixel spectrum. Toy (`toy/`, 4 shapes, small CNN at balance): all generators far from
  truth; G2 (match weights) collapses: Theorem 21 plateau. Lemma 22.1 Euler scalar kappa exact (ratio 1.000).
  Theorem 22 first-order max-ent = G1 x e^{H(p)}, bounded by K (Cor. 22.2). Next: full max-ent multipliers from conv1.
- Theorem 23 night (sections 23-24): free-coefficient KKT reconstruction on the toy. Equation prefers the true train
  set 6x (best-c residual .047 vs rebuilt .26-.30). Support graded: least-sure 30% reproduce theta 3x better than random.
  Variable projection (exact c) lowered the residual but WORSENED the P2 writes (Fisher .189 -> .468; junk + concentration).
  THEOREM 24 (proved): weights fix the FIRST moment of gradients; the Fisher write needs a SECOND moment, which
  self-labeled inputs change freely -> the selection rule decides it; derived preference: MAXIMUM ENTROPY (spread), next
  plan = max-ent sampler. Published Buzaglo recipe (m = 2N, 1e4+ steps) never run. Best toy generator: G3 (lsq .080).
- Section 25 (Shonu: "how can frequency not show up in the network?"): weights' SIZE = how often x how wrong (no);
  class shares YES (last layer); neuron frequency via P(fires) = sum_y P(y) P(fires|y): class inversions give levels
  within x2 for ~80% of neurons, co-firing .78, but rank only ~half the reliability ceiling (.45 vs .97 in 3b): modes are
  not typical. Sections 19/24/25 = one gap (sample not maximize; spread not concentrate; typical not mode) = q(x).
- INTERROGATE (section 26): Neural Feature Ansatz. Raw N1 (Gram vs AGOP^.5, cos .90-.94) is NOT evidence: the no-data
  control gives .93-.95 (sandwich structure). Discriminating test: next-layer weights predict the REAL-image Fisher's
  off-diagonal structure .81-.87 vs the network's Fisher on noise .32-.45 (3b onward) -> weights hold natural-data metric
  info. Lemma 26.3: a constant metric cannot change an unconstrained write (GLS = OLS); the write's gain is per-image
  metric variation + gates. Weights' metric is usable for the WHAT (Phi) and constrained writes. Open: per-image metric.
- DISMANTLE (sections 27-32, same night): network-made build now .0771 (self read + lsq on class inversions); write
  coverage solved by the network's own reader pictures (Theorem 28.1, Prop. 28.2); constant from plain mean (28.4);
  M read + network-made Fisher write .0807 (beats lsq on same inputs by the real margin). Theorem 31 (SGD imprint: per-
  neuron imbalance stores per-sample gradient energies, N6 .62, N7 real .70 vs network-made .21-.31). Reweighting
  infeasible; imprint generator adversarial (N9 retracted, N10 failed). Section 32: three channels in weights
  (stationarity / symmetry imprints / alignment); the only remaining outside-info object = PAIRWISE activity-sensitivity
  covariation between different children (no channel carries it). Next: does the write need the off-diagonal part?
- IMPOSSIBILITY (sections 34-35): Theorem 34.3 (weights-only maps cannot be optimal for every consistent distribution;
  minimax bound). PROVED with a measured gap for exact-stationarity training (C1 only): adding network-labeled mass
  x100 moves the optimal write by .053 (X6). For SGD training, channel C2 (imprint DIRECTION, scale unknown) excludes
  every write-moving addition found (X4/X7 neutral = write-neutral; X8/X9 infeasible) -> evidence the imprint pins the
  write. Way around = generate inputs whose imbalance is parallel to D at ALL layers (one layer was adversarially gamed).
- 2026-10-24 THEOREM session (plan ~/.claude/plans/okay-so-the-last-harmonic-stallman.md): new paper NEURON-MATH.md.
  Target: weights-only C_k optimal in the network's own norm; trained nets = imperfect examples. Crux object = L6 (gate
  overlap of members with the new coarse unit). Log section "THEOREM (2026-10-24)".
  RESULTS (2026-10-24): Prop 6.1 (no exact orbit at a ReLU layer; Lean); angles from weights (median 22 deg); Prop 7.2
  write-in-span for reader matching (Lean); R2: trained write's out-of-span half is REAL (+.17 k=1, +.02 k=3) = readers'
  gating (Prop 7.5); k=1 twins differ only in flat directions (avg = twins, closest-twin gauge; balance gauge invalid:
  no weight decay in family.py); Shonu chose "readers adapt": reader biases free = no-op (dial = bias, Prop 9.1, B2
  .1217/.0350); k=3 in-span write direction = max-shift kernel overlap with the read (cos .91-.94). Lean proofs in
  research/clean/curves/lean/coarse (7.2 GB .lake, in DISK.md). Open: gate statistics; B3 (readers' other weights adapt)?
  B3 (Shonu chose "readers fully adapt": 4a input-conv weights on all 470 other 3b channels + biases free): k=0 ruler
  .0738, k=1 .0678, k=3 .0297 (2000 steps, capped). Prop 11.1 (= ThiNet/He 2017 reconstruction): Delta-W = W_F Lambda,
  Lambda = S_F,rest S_rest,rest^-1. Closed form (real moments, FITTED) .0647 beats the 2000-step k=0 ruler (-.0091, 2.9
  se); training started AT the formula .0446 (best known; A >= .020). Means ("how often") carry only .001-.004 of the
  network-made gap; the 3b COVARIANCE pattern carries ~.021 (R3h). Weights-only covariance from effective kernels in 3a
  space: .177 (81% of the real formula's gain). Network-made label stays "network-made" (generator-step test not run).
  Stronger from-zero B3 rulers (6000 steps): lr .01 -> .090 (train .033: over-fits 165k weights), lr .03 -> .342. Bigger
  step != stronger ruler. lr .003 6000-step run: .0564, BEATS the formula by .0082 (2.9 se) -> 11.3(1) REFUTED; the
  formula only beat a capped ruler. Formula-start .0446 stays best (formula = good starting point). k=3 unit worth vs best
  k=0 ~.015, not .044 (k=3 ruler also capped).
  G1 (12.4, one Gaussian from TRUE pre-act moments): means exact, family rows .714, Lambda cos ~.77 -> REFUTED.
  G2 (12.4', shared-contrast scale mixture): worse (.632); |p|^2 over 480 ch concentrates -> REFUTED. 12.3/12.5/12.6
  SUSPENDED: the mean-field recursion route is blocked at the ReLU step. Located obstacle: 12.4'' sparse mixture over a
  dictionary (directions = weights via effective kernels; the open part = how often dictionary items co-fire). Do not run
  12.4'' with fitted frequencies (proves nothing new); wait for a weights-only candidate or Shonu's call.

- 2026-10-24 EXACTNESS session (plan ~/.claude/plans/okay-for-this-session-twinkly-gosling.md, NEURON-MATH Part IV, log
  "EXACTNESS"): target Theorem E (training on D from F(m(D)) cannot lower KL on D) + Cor. E.1 (generalist). Real-moment
  formula .0647 is DISQUALIFIED as generalist; weights-only start = R3i .177. First: n-ladder, gated formula (13.1).
  RESULTS (same session, NEURON-MATH sections 13-15): Lean Exact.lean (quad_moment_suff, gated_write_in_span,
  exact_no_gain). R1: level-0 S needs only 64 images (within .003). R2: gated level 1 .0603, gates carry only .004 of the
  .020 exactness gap (13.6 REFUTED). R2b/R2c (exact 2nd-order KL term): training's gain over level 0 = COUPLING of errors
  across different readers AND positions (cross term .046 -> .0014); diagonal metric has the wrong sign; level 1 fails
  by gate flips. Prop 14.8 (Lean): reader-reader coupling can't move a write unless it varies per image. Prop 14.9:
  in-span coupled formula = one 4710 x 4710 solve. n=2000 over-fits (.249; <= 9 constraints/image); n=35000 .0552 TIES
  converged from-zero training (.0564), +.0106 above best known .0446. Next object: the coupling metric from the WEIGHTS.
  EXACTNESS test on coupled35k: training gains only -.001 (se .0015) BUT the curve leaves the start (dev .057 -> .12)
  and comes back; ends .0096 above best known .0446. Statement 15.1: "training gains nothing" != optimal (sharp point,
  large coefficients in flat directions). Exactness must be read WITH the best-known bar. Skipped this session: R4
  (generalist no-extra-info), R5 (k = 3 mix), R6 (developer subset), cold test.
- 2026-10-24 ROUTE A (Shonu chose A + "understand it on an image basis"; NEURON-MATH Part V section 16, log "ROUTE A"):
  Kbar = image-averaged coupling metric (3920 x 3920). Image meaning (16.6): CONTOUR INTEGRATION. Partners follow the
  co-circular rule (sharp test .696 vs .211); 5% of images carry 95.5% of the cross term (round things: melon rind, basin
  rim, lock dial); level-0 error runs along the contour, training removes it. Envelope = shared horizontal band (part
  photos, part network). Factorised build (Kbar from 2000 imgs) .0491: BEATS converged from-zero training by .0074
  (3.2 se), +.0045 vs best known. WEIGHTS-ONLY metric from 4a effective kernels (16.7: co-circular .780, 10/10) ->
  build .0570 (44% of the image metric's gain, -.0063 vs level 0p). Next: compose the metric through 4b+ (envelope +
  output weighting). Exactness run from factorised build: B3-k0-factorstart.
  Exactness run from factorised build -> .0430 = NEW best known k = 0 (-.0016, 2.7 se). The Adam kick-out (dev rises
  then re-descends) happens on EVERY start incl. level 0 -> 15.1 mechanism refuted; X-B3 recipe is not a local
  exactness test (15.2: use small steps / gradient norm next time).
  LOCAL exactness test (lr 3e-4, 500 steps, no kick): factorised build .0491 -> .0404 (local defect .009, new best known);
  same recipe from level 0 -> .0452 (formula start better by .0048, 8 se). Use the small-step recipe for exactness tests.
- 2026-10-24 CIRCUITS session (plan ~/.claude/plans/okay-last-session-landed-velvet-wren.md, NEURON-MATH Part VI, log
  "CIRCUITS"): Shonu: a new mathematical unit for a circuit when neurons are clean. Unit = EDGE between clean families
  (weight kernel + gate); circuit metric carried down by metric transport (exact). Two gaps: A (corrected: .0552 -> .0491 = .0061, plus .0018 pool fix; first stated as .0570 ->
  .0491 = .0079, Shonu's "56%"); B = Kbar build vs best known local-trained .0404 (.0087). Gate co-opening =
  same open object as 12.4''/L6. Checks I1-I3 (identity, eigenmodes, paths), A1-A3 (4a diag/off-diag/gate-avg), B1-B3.
  RESULTS (same session): transport identity CHECKED (2e-7). Gap A = coupling between 4a units (diagonal/importance
  metrics .055-.058; full metric no gates .0488); weights composed WITHOUT gates to 4d close it: .0489 dev, .0456 sealed
  = Kbar .0455 (depth CHOSEN; 5b/logits collapse: rank 24/8 vs true-gated 3223). Band + centred window come from the
  weights. Gap B = reader-specific, second order (L3 -.0116 vs KL -.0111); averaged metrics PENALISE training's move,
  per-image metric credits it -> per-image metric DIRECTION (gate pattern, L6) is the open object; 2-state split and
  per-image windows fail. Local exactness from 4d build -> .0411 (ties T). I3: coupling via round-shape 4d units, spread.
  Next: sampled-gate (Bernoulli) transport; network-made rates/activity. Activity moments still MIX-DERIVED.
  22.2 check: same-label metric images do NOT credit training's move (1.47) -> "which classes compete" refuted; only
  the image's own metric does. Gap B compact form OPEN.
- 2026-10-24 GAP B + DEPTH (Part VII): coarsening = OBS on the reader layer (Fisher form); level 0 = layer-wise OBS,
  factorised = KFC, training ~ full-Fisher OBS. Gap B = Kronecker error; NOT EKFAC diag, NOT top-80 modes; per-image
  gate-activity link; direction stable at ~8000 real images; exact one-step / Krylov reach ~40-50% (sealed -.0028);
  network-made inputs carry none of it (distribution information). DEPTH: true gates + UNIFORM output metric = Kbar;
  Bernoulli independent gates fail (.0520); 70% within-layer co-firing, 30% path coherence inside 4a-4e, nothing across
  4e->5a; gate patterns of 256 CLASS INVERSIONS carried to the logits = Kbar (dev .0491, sealed .0455): no depth cut.
  Open: e1 step-doubling check; Krylov with full output covariance (running); activity moments still MIX-DERIVED.
  Final this stretch: netmade-32 = netmade-256 metric; local exactness from netmade build .0403 (= best known T);
  Krylov with full output covariance no better (~40% of Gap B): truncation not the cause, Kronecker conditioning is.
  Next: coordinate-wise (Adam-like, diag F) preconditioner on Gap B; e1 step-doubling; network-made activity moments.
- 2026-10-24 evening FRAME (Shonu asked; plan ~/.claude/plans/last-session-made-quite-lively-flute.md, NEURON-MATH
  Part VIII): the project = COARSE-GRAINING whose target is the DISTILLATION OPTIMUM. Training is an approximate solver
  of that optimum; we seek its closed form. Distillation alone frames it as "query, then fit" (hides structure); fold-up
  alone does not say what "good" is. Statement S: answers (labels), metric (26.6, DERIVED) and questions (which inputs)
  all from the weights. "Gap B" renamed "the same-image term". Old 25.9 "outside the network" was ONE generator (class
  pictures, n=256) failing, not a theorem.
  RESULTS (same evening, NEURON-MATH Part VIII sections 28-29, log "NETWORK IS ENOUGH"): PROVED 28.1 moment equation,
  28.2 gain <= own loss (q's own F only), 28.3 transfer bound (coverage x leftover), 28.4 root condition (direction of ONE
  vector g_P(S); 256 photos give it to cos .94 under the uniform metric), 28.6 push = factorised + same-image covariance,
  28.9 term depends only on the law of 3b activity maps. CHECKED: class pictures are HARD (5.9x error), their factorised
  part is anti-aligned; photo term is 97% cross-position and class-INDEPENDENT (between-class part orthogonal).
  DERIVED question rule: one picture per family reader (368), reader pre-act at map centre, e1 settings
  (`c45_readerq.py`): cov cos .61; network-made end to end (gradient, KFC from the questions, length): dev -.0014, sealed
  -.0011 (58% of photo-256 step -.0019); class pictures -.0008. Start S still MIX-DERIVED. K4 step-doubling (512 steps)
  was running at session end. Next statement: CONTEXT/clutter around the object (reader questions = one patch in gray).
- 2026-10-24 NIGHT (NEURON-MATH 30-33, log after "K2-net"): GAP C: old stall = rank-10 sampled metric (31.1); exact
  uniform metric via JVP+VJP (31.2, `c50_exactcg.py`, max-pool index form) -> closed-form PCG on 2000 photos, derived
  stop (held-out L_ex) dev .0425 = 75% of S->T. Right objective = L_ex (exact reader gates, 31.7, C3). Gated frozen
  model is worse (flat directions). Centred KFC refuted. Plain-norm "drift" was parametrisation-dependent (withdrawn).
  GAP Q: KFC step 99.9% bias (A const 1 vs 7.6e4); bias-only closed form b = -K0^-1 c (32.1); c = 368 per-reader
  covariances. Reader questions .54-.58 match (= 32-64 photos' worth); held-out fitted weights .78 vs ceiling .91 ->
  remainder ~2/3 FREQUENCY of the network's own events, ~1/3 shape. Imprint D as frequency REFUTED (-.06). Open
  decision for Shonu: declare the event-frequency vector the mix input, or hunt a weights-only rule.
  Process: smoke run first, one GPU job, CPU math first (feedback-quick-iteration).
- 2026-10-24 LATE NIGHT: MATH-PAPER RESET (plan ~/.claude/plans/good-work-so-far-rustling-nautilus.md). New paper
  research/clean/curves/COARSE-THEORY.md (old papers = record; triage TRIAGE-2026-10-24.md: 243 items, 27 to prove,
  150 log only). Guards live: project .claude/settings.json hooks + tools/ + .math-mode switch (see
  [[feedback-math-paper-standalone]]). Proved: T0 (data enters the write only through one moment pair; facts only),
  T1 (H-clean+H-sym+H-read: circulant; k=1 read = sum of members, write = mean column, DATA-FREE under a constant
  metric; k=3 = bumps span; k=2 non-unique), T1' (sum of members is one ReLU only if hinges never cross: WHAT exact,
  HOW approximate), T3 (pairing slope: circle/curvature process 2 at short range rising to 5/2, direction process
  3/2; H-contour changed to the curvature process). Lean Model.lean builds; sympy + exact instance (model/) all ~1e-13.
  Conjectures with routes: T2 (loop at curvature centre, r = 1/kappa derived), T4 (symmetry -> few numbers; stability).
  Next: referee report, Part C eps (H-contour slope 2 vs 3/2 is a sharp test), Part D.
  UPDATE (same night, revision 3 of COARSE-THEORY.md): two math-referee rounds (REFEREE-A-B-2026-10-24.md, -r2.md),
  all findings fixed in the paper (Appendix tables). T1 split: T1a (under H-kron, H-pool; circulant premise) +
  Lemma T1b (plane model D8 + H-clean/H-sym/H-read). New T5: shared symmetry decouples several families by frequency
  (selection rule; data = one FxF matrix per frequency). T3 corrected: co-circular concentration holds for ANY
  curvature law (Var(phi-2y/s) = sigma^2 s^3/30); conditional peak only for Gaussian; T3.2 width-robust test = growth
  exponent of D(s)=Var(phi'-2y'/s): s^3 (curvature process) vs s (direction process). Exact instances (model/) all
  ~1e-14; symmetry-breaking control errors scale as delta^1. Part C proxies from R1 (weights only): H-clean R^2 .28,
  H-read R^2 .43, unequal member spacing -> strict symmetric model far from InceptionV1; Conjecture D1 (sampled field
  with the network's own measure mu_N) is the bridge. Part D written (chain with 4 missing links, D1-D5).
  Next: H-sym measured on photos directly (no network); the one InceptionV1 run goes to H-contour (estimator to be
  written in Part C first: covariance, frame map, quantization floor, decision rule, shuffle null).
  PART C (same night): H-sym on photos (pc_hsym.py, no network): eps(2) .106, eps(4) .073, H/V excess (gravity).
  H-contour smoke (pc_hcontour.py 64 imgs): frame rule FAILED (autocov axes vs R1 angles scatter 35 deg); reason
  "RF isotropy" REFUTED (eig ratios .34-.84). Verdict void; 2000 run NOT made. Open: define each member's tangent axis
  without R1 labels or autocov (math first). Two runs spent on that statement -> next step is a definition.
  REVISION 4 (same night): third referee (T5, Part C, Part D) fixed: T5 part 5 corrected (reads depend on Chat(n)),
  T5 is bookkeeping not structural; new Lemma T1c (gated writes shift along reader orbits); T3 part 4 (curvature-tuned:
  sigma^2 s^3/180); hypotheses H-joint, H-shot, H-detector, H-gram named; D1 restated as a limit; Part D chain has a
  Hypotheses column and 5 MISSING links. Linter now flags 0.xx decimals too. Structural results so far: T1a, T1b (+T1c);
  all behind H-kron or the plane model. Decision pending for Shonu: tangent-axis definition from results_g0_tuning.npz
  (no run) vs a new run.
- 2026-10-25 FOLD (plan ~/.claude/plans/last-session-did-really-shimmying-hartmanis.md; COARSE-THEORY.md rev 5, Part B.4):
  Verdict given to Shonu: rev 4 did NOT reach the end state (no tree defined; ReLU HOW fails by T1'; depth conjecture).
  Shonu chose restriction H-gate (one gate per family, gate path does not read values, kept at full size). PROVED: T8
  chain collapse (logits linear in build per input), T9 exact optimum over ALL builds = reduced-rank regression = PCA of
  centred logits, read of members (no H-kron needed), T7 HOW exact, T6 fold ladder (nested, merges compose), T10 with
  symmetry = same Fourier modes kept at every layer (pointwise edges only), T10.1 invariant outputs -> width 1 exact,
  T10.2/T10.3 KL bounds (factor n e^R). Tree (>=2 disjoint children) only 1->3; counting kills 3->5. Lean Fold.lean.
  Exact instance model/t9_instance.py all ~1e-15; trained toy toy2/ (Y1-Y8 all as predicted: students never beat the
  formula; relu1 variant shows the HOW gap). Referee 4 fixed. Open: spatial edges (D3b, multilinear), ReLU, stability.
  Referee round 5 (REFEREE-B4-2026-10-25-r2.md) fixed: D10 tree now needs NON-NEGATIVE units (non-negative reads of
  members) + >=2 disjoint children; levels 1->3 and 1->5 are trees, adjacent 3->5 never (count); Fejer statement only
  on the whole circle (fails for reads on N = 5 members). T10.3 = range-bounded builds only; Lean softmax_var_ge.
  Y9 (Shonu asked "does training converge to it?"): new T9 part 4 (near-optimal value => near-optimal centred outputs,
  bound 2 e (1 + sqrt(2 mu1/gap))^2; Lean gap_trace). toy2/c3_converge.py: 9 students, 12000 steps: excess -> 5e-6..5e-5
  rel, output distance -> 0.3-0.6%, never below the formula, bound never violated. Not proved: that training always
  reaches the minimum (deep linear nets: no bad local minima, Kawaguchi 2016, cited only).
- 2026-10-25 NOODLE (plan ~/.claude/plans/okay-you-can-look-fancy-goblet.md; COARSE-THEORY.md rev 6, Part B.5; plain
  summary research/clean/curves/LADDER.md). Shonu's word "noodlable" = fold to any size (down to 1 unit), optimal, instant
  size switch (= prefix of one stored build; he accepted that literal O(1) is impossible). PROVED: T12 floor for ANY
  architecture with a width-k linear head (under: none); D12/H-selfgate (gates read the value path through fixed reads,
  no gate path) -> T11 closed form among read-faithful builds (forced reads first, then PCA of residual), T11.1 r extra
  units below the top reach the T12 floor, T11.2 symmetry -> r = 1 (orbit sum); D13/H-sgate spatial edges -> T13 fold to
  "visible subspaces" (observability) is exact at the floor down to raw input. Lean Coarse/Noodle.lean. Instances
  model/t11_instance.py, model/t13_instance.py exact. Toy toy2/c4_selfgate.py (teacher_selfgate_sym.pt): Y10-Y12b as
  predicted; Y13: free-read students beat T11 at uniform width by 11-25% (never the floor) -> Conjecture D6. Honest
  costs: windowed random edges keep the bottom layer at full visible width; ReLU forces every read (no fold in-class).
  Referee 6 (REFEREE-B5-2026-10-25.md) fixed: T13 needs REACHABLE spaces (visible width = observable reads restricted
  to Rch_l; first version's 'k + r' claim false, now T13 part 4 proved for pointwise); T11.2 factor 1/2, needs
  single-column invariant reads; D6 restated existentially; T12 = D2 + affine head only. Round 2
  (REFEREE-B5-2026-10-25-r2.md): T13 parts 1-4 correct; fixed Rch_l = least a.s. subspace, visible_step on a submodule,
  T11.2 parts 3-4 need r = 1; new T13 part 5 (pointwise: d_l >= k; = k with gate path, minimal). Next: D6, D3b, ReLU.
- 2026-10-25 BOUNDARY (same session, B.6): Shonu's rule = never widen scope without the optimum inside it. D3b -> T14:
  below visible widths the optimal fold = best rank-1 approx of a 3-tensor (NP-hard, Hillar-Lim 2013 Thm 1.8/1.12);
  model/t14_instance.py. D6 -> T15 proved exactly (4 points: floor 0 < best 1/32 < T11 1/16; model/t15_instance.py).
  New Conjecture D7 (free-switch optimum NP-hard; obstacle: teacher must emit arbitrary labels). Lean rank1_fit.
  Exact-optimal scope = size counted at the top layer, lower layers at visible widths (+ r). Referee REFEREE-B6 run.
  Referee 7 (REFEREE-B6-2026-10-25.md) fixed: T14 retitled 'at width (1,1)', decision version only (Thm 1.8),
  builds via h = H^T eta / 2 (Thm 1.12); T15 now has a hand proof (closed union + 6-case no-exact-fit), 1/32 a Remark;
  D7 = decision problem with step gates (obstacles: teacher labels, free sign/bias, fixed-d is polynomial); D3b open
  for symmetric chains. Lean rank1_fit_eq, orth_head_split. Fixes not re-refereed.
- 2026-10-25 T16 (same session; Shonu: 'you don't understand why the formula is beaten; bar not met'): T16 = optimum
  over self-contained builds is min over EFFECTIVE GATE READS R of a closed form J_k(R) (gates depend only on R; then
  linear). T11 = J_k(teacher reads). Loss = gate error + width error. model/t16_instance.py: J search 1190 vs T11 12737
  vs students 1212 (k=2). Toy c5_why: free-read student ignores teacher gates, 94% kernel outside member span, mode 4.
  T14-CHECK-2026-10-25.md (sonnet): T14 correct; HL numbering arXiv v5 = 1.10/1.13; fixed n3 polynomial; odeco easy.
  Referee 8 (REFEREE-T16-2026-10-25.md) fixed: J_k as attained infimum for all R; part-5 premises; Cor T16.1 (identity
  edges suffice at uniform width); chamber convexity only n_y=2; 'not optimal' not 'not local min'. Open: is min_R J_k
  efficiently computable (D7); symmetric spatial chains (D3b).
- 2026-10-25 SWITCH FORMULA (plan ~/.claude/plans/okay-we-re-pretty-close-floating-quail.md; COARSE-THEORY.md rev 7,
  Part B.7; Lean Coarse/Switch.lean). Shonu: 'we copy the gates, I want a formula for the best switch'. Lemma T17a
  (under none): a build is a function of its kept features W, floor = E Var(y|W) (seen floor F(V)). H-gauss (Gaussian
  input, iid positions). T17 (affine gate, depth 1): best read for EVERY kept space V = teacher's gate argument
  regressed onto kept units (M2-projection of s onto V), head M Gamma, bias absorbs M C_e s; exact F(V); copy = s in V.
  T17.1 curved gates: same build, price <= L^2 sigma^2 (...), sigma^2 = unseen part of gate arg. T17.2 width 1: best
  kept unit on resolvent curve (mu I - Q)^{-1} stilde (copy at mu=inf, output eigvec at mu->lambda_j), mu = root of a
  degree 3n-2 polynomial. T17.3 width k: best Pi commutes with G(Pi) (self-consistent eigenproblem); D8 open.
  T18 subset fold (any law) <= T11. Checks: model/t17_sympy.py, t17_instance.py all pass; t18 on T16 instance weak
  (k=2: 9231 vs copy 12737 vs search 1190) because kept-space choice is the hard part. Referee 9
  (REFEREE-B7-2026-10-25.md, opus): all proofs correct; fixed overclaims (T9-end misnamed: Q = whitened Gram of
  P_perp M, not logit cov), linear-seen exact criterion, uniform T17.1 bound, step-gate psi, T18 V_T defined directly.
  New T17.3 part 4: width k = search over ONE vector p = Pi stilde on sphere |p|^2 = p^T stilde + Ky Fan (checked V7).
  Referee 10 (r2): D9 route was backwards (copositivity -> co-NP); now linear gate, one position, from
  NON-copositivity; D8 restated for rational kernel K' and value F. Next: D8 via part 4; resolvent curve for curved gates.
- 2026-10-25 LATE: TRUNK RESET (plan ~/.claude/plans/we-ve-been-doing-pretty-squishy-hanrahan.md). Shonu: theory drifted
  from the constant goal. Shonu fixed the trunk T*: WHOLE network, ANY size, ANY new units (not a subset), KL on the mix,
  nesting. New `research/clean/curves/DECISIONS.md`: trunk (hash-locked), 51 nodes (11 added after referee: hidden premises) (kinds ROOT/CONV/STRUCT/SCOPE/NEG,
  status DECIDED/PENDING/RETRO/DROPPED), tag-phrase aliases, path of all 47 tagged items, tree picture, 8 cards.
  `tools/ledger.py` (K1-K7, --report, --open) runs inside paper_lint + turn reminder; test_ledger.py in test_guards.
  CLAUDE.md rule 7; referee check 6 (paths). Paper Part 0 + D.1 path column; LADDER section 8. Distances: T1' 0, T12 1,
  T9/T16 3, T11 4, T13 5, B.7 T17-D8 8-9, family branch 3-14. Referee fixed: T14 does NOT fail to be ReLU (finite law:
  large bias) -- the build sets differ; 'exact fast rule impossible' is a BELIEF (conjecture), not a result. Root decision R-root (exact / fast / every network: which gives) PENDING; my
  recommendation A (structure = clean), B (bound) as the way to ReLU. DECIDED so far: R-gate, R-bench only.
  No new rung starts before R-root. TRUNK WORDING IS DRAFT: Shonu chose only goal + building blocks; asked
  him to confirm 'k in every layer', 'same kind as N', KL, 'does not train', nesting. R-gate/R-bench DECIDED rows are
  recorded from plans, not yet confirmed in chat. D.1 path column removed (pointed to ledger section 6 instead). See [[feedback-trunk-decisions]].
- 2026-10-25 LATE NIGHT: B.T AT THE TRUNK (plan ~/.claude/plans/okay-i-think-last-fancy-cosmos.md). Shonu: the R-root
  options A/B/C and the draft trunk wording were my guesses; push the general goal until a THEOREM forces a choice.
  COARSE-THEORY.md Part B.T (before B.0): Reading block (P finite list, plain ReLU, k per layer, fast = poly time),
  D15 (plain ReLU build, data vector), D16 (weak/strong nesting). T19 existence (2 logits, 1 hidden layer: finite union
  of polyhedral cones + coercive KL). T20 PROVED CONFLICT: teacher T = relu(x-1)+relu(-x-1) on {-3..3}: exact width-2
  fits are only {relu(x-1), relu(-x-1)} or {relu(x+1), relu(1-x)}; best width-1 uses none of them (D2 exact 16/5 < 85/26,
  KL certified by dual bound + interval arithmetic, margin ~6e-4). T21 exact rule (regions x signs x convex pieces,
  cost ~ n^{dk}; Arora et al. 2018 Thm 4.1). T22 PROVED CONFLICT: TRUNK-1 (best width-1 build within eps) NP-hard and no
  rho(d) S^{o(d)} under ETH; reduction from Multicolored Clique reusing Froese-Hertrich-Niedermeier 2022 point set,
  redone for KL + free sign/bias (midpoint step = monotone along segments) + weights + memorising teacher (Lemma T22a).
  Lean Coarse/Trunk.lean (relu_second_diff, t20_d2_gap_a/b, mono_mid_ge_min, broken_line_interp). Checks model/t20_nesting,
  t21_exact (training never beats exact; stuck 14/32), t22_reduction, trunk_exact.py. DECISIONS.md sections 2-3 rewritten:
  old A/B/C withdrawn; R-nest (N1/N2/N3) and R-root (A every-network / B every-mix / C exact / D fast) PENDING, options =
  theorem premises. New nodes S-twoclass, S-onehidden (partial results), N-onehidden, N-dim1 (NEG). Conjecture D17 (existence
  for n_y >= 3, depth). Referee report REFEREE-BT-2026-10-25.md.
  REFEREE 11 fixes + T23 (same night): referee pointed to Lim-Michalek-Qi (arXiv 1907.01507): their 2-output width-2
  example (6 points in R^2) carried to the trunk with 3 logits = Theorem T23: inf D_P = 0 NOT ATTAINED -> third proved
  fork R-least (L1 eps-best / L2 weight cap / L3 closure / L4 property). My own proof of non-attainment by second
  differences (model/t23_noattain.py). T21 restated (exact only for D2; KL = finitely many convex pieces). S-finite node
  (finite mix narrows positive results). T22 part 4: relative error 1+1/(16 N_V) also hard. N-dim1 removed (line
  embedding). D17 now: two logits, depth >= 2. Referee round 2: REFEREE-BT-2026-10-25-r2.md.
  Round 2 (REFEREE-BT-2026-10-25-r2.md): T23 re-derived and CORRECT. Fixes: node N-anynet (NEG, PENDING: "trained" =
  any network; the forks hold under the B.T readings only, and "every network" is an option in each fork); "T20/T22
  survive L1 (eps-best)" WITHDRAWN (unproved; 3a/3b conditional on 3-0); T22 part 4 factor 1+1/(16 n); option L5 (every
  mix); Lean t23_core. All gates clean. Next: Shonu answers R-least first (my rec L1), then R-nest (N2), R-root (A+B).

- 2026-10-25 LATE NIGHT (2): NEW TRUNK T** CONFIRMED (plan approval; hash stored): input N + goal G (weights over output
  classes) + sizes (k_1..k_L); rule reads only N and G (NO input data, no training); best for G; one-time computation per
  goal, then INSTANT switching (O(1) in |N|); score defined from N and G alone; exact closed form at eps(N) = 0, bound
  depending only on eps at eps > 0. Decisions DECIDED: R-least (clean, well-trained), R-nest (goal-indexed; switching kept),
  R-root (property of networks, no data). DROPPED: N-anynet, S-finite. New PENDING: R-score (packages i intrinsic / ii
  training law / iii worst case; working reading i), R-cleannet. Plan ~/.claude/plans/okay-i-think-last-fancy-cosmos.md.
  RESULT under T** (Part B.C): Lemma T24a (Gibbs: N optimal on its own labels for every input law), THEOREM T24 (T20's
  mirror network is perfectly clean; laws mu_1 uniform vs mu_2 {0..3}: disjoint best width-1 builds (KL, D2); D2 also for
  mirror-symmetric laws (1,1,3,3,3,1,1) best kink 14/11 vs (3,1,1,1,1,1,3) end-point unit) -> training-law score not
  fixed by N. Cor T24.1: worst case over consistent laws = worst single input. Fork R-score (i intrinsic / ii training
  law + stronger property e.g. SGD imprint / iii worst case); my rec (ii)-SGD. Proposed D20 (perfectly clean for group
  Gamma: equivariant plain ReLU, families = orbits, readout symmetric, every unit read) + D21 (eps from weights). Check
  model/t24_score.py. Referee REFEREE-BC-2026-10-25.md. Items 3-7 of the plan wait on R-score.
  Referee 13 fixed (REFEREE-BC-2026-10-25.md): T24a now iff (optimal on Q iff labels = N's); option (iii) ignores the goal
  (needs pointwise goal loss, node R-goal PENDING); (ii) must restrict INPUT law; D20 now reduced form + faithful action on
  layer 1 + explicit homomorphisms; open: can T22's hard instances be made perfectly clean. Ledger reminder now says
  trunk "confirmed". STOPPED at R-score card (3-1) for Shonu.
- 2026-10-25 LATE NIGHT (3): Shonu REVERSED no-data: TRUNK T*** (hash stored): N + mix P (generalist or developer) +
  sizes k_1..k_L; least D_P over any new units; one-time computation per mix ("the zoom"), then instant switching; exact
  at eps=0, bound in eps. R-score DECIDED (mix input), R-goal DROPPED, S-finite/N-finite off paths. Shonu's product
  picture = ZOOM: fold up neurons per sample. Part B.Z: D22 fold (unit reading only a group's pre-activations), D23 zoom
  (tree, node folds, cuts; switch cost = changed nodes). T20 IS a zoom (width-1 best = fold 5:1). THEOREM T25: least build
  over any units can read an input N ignores (x2 = answer on the mix) -> not a fold, even clean + symmetric mix =
  overfitting. Fork R-fold (a folds only = zoom [my rec] / b mix property / c any units). Insight: with 1 hidden layer,
  instant switching is easy by storage; the real test is DEPTH (layer-separability of best fold builds). model/t25_fold.py.
  Referee 14 fixed: D22 fold now = NON-NEGATIVE mix of members' pre-activations, NO new offset (Remark B.Z.0: silent
  unit relu(x2-100) + free offset or negative mix re-reads x2). T20 is a zoom for D2 exactly (best kink 8/5 = fold
  13 z1 + 3 z2); KL numerical. D23: cut = fold build with disjoint groups; switching cost ~ changed nodes (can be ~n),
  same as stored table; tree saves storage. Lean t25_mirror. STOPPED at R-fold card (3-2) for Shonu.
- 2026-10-25 LATE NIGHT (4) WHY FOLDS: Shonu asked for folds to follow from DEEP goals. Part B.Z subsection: D24 conditions
  G1 (use N's weights, not only answers = not distillation), G2 (unit reads only through N's pre-activations = no new
  distinction per mix), G3 (covers: fires only where a member fires). T26 any-unit = distillation (depends on N only via
  outputs on P); folds use weights (N1 bathtub vs N2 broken line, same answers, mix (3,3,3,1,1,1,1): 26/7 vs 84/11).
  T27 fold exact rule exponent ~ s k (fold mode in trunk_exact.py) vs T22 exponent d. T28 G2 <=> row-space fold; n_P <= d+1
  -> perfect any-unit copy of every N; pdim. T29 G3 <=> fold (D22 now with tau >= 0; Farkas, Schrijver 7.1h). T30 capstone.
  Lean fold_covers. Card 3-2 rewritten. Referee REFEREE-WHYFOLDS-2026-10-25.md.
  Referee 15 (REFEREE-WHYFOLDS): capstone OVERCLAIMED, fixed. Honest T30: only G3 (covering) forces folds (single ReLU,
  Z_S nonempty, domain R^d); G1 rules out any-unit (distillation) but not other N-dependent classes; G2 vacuous when first
  layer spans R^d. Prop T27.1: T22 hardness transfers to folds of large groups (s = 2d); fold exact cost exponent
  min(s,d)k. "Training" wording removed. Card 3-2: decision = "is G3 your meaning of fold up what exists?" (a/b/c).
- 2026-10-25 LATE NIGHT (5) REFRAME (Shonu): the project's question = WHAT DOES DISTILLATION CONVERGE TO (least-KL build
  over ANY units, the trunk as written), and the CONJECTURE: for clean networks it is a fold (essay: small model trained
  on same data has coarse versions of the big model's neurons). Folds are NOT to be argued from product constraints.
  R-fold DECIDED (any units kept). New: Lemma T31 (full-rank first layer => any unit is a loose fold), Conjecture D25
  (clean + full rank + invariant mix => least build has covering-fold units, one family each), node R-fullrank PENDING.
  Plan: exact test on cyclic families in R^2 (model/tf_clean_optimum.py) before any proof.
  RESULTS (same night): exact tests (model/tf_clean_optimum.py, tf_fold_gap.py, tf_broad_fold.py): CONJECTURE D25
  (covering folds optimal) REFUTED at solver accuracy: C3 family, 6-ring r=2, width 2, D2: any 1/36 vs covering 1/32 (gap
  1/8); C4 + centre width 1 gap ~2.4%. Optimal coarse units fire MORE broadly (non-negative mix, LOWERED threshold); at
  width 2 a pair along one member direction with opposite output signs. New CONJECTURE D26 (broad folds: m >= 0, offset
  free): holds at width 1 on 12-point R^3 instances where the cone is proper (any = broad, covering 5-9% worse); width 2
  untested. Certification pending; B.F/D.9 not yet refereed.
- 2026-10-25 LATE NIGHT (6) PROOFS (Shonu: "you haven't mathematically shown these principles ... until a fork
  theorem"): EXACT exhaustive computer-assisted proofs via Lemma T32a (dual certificates: sum pi v = 0, s L^T(pi v) in
  cone(rows) by Caratheodory; exact cell lists: rational interior points + Gordan; Lean cert_ineq). T32 (R^2, C4 family,
  8 pts, width 2, D2): any 3/80 vs covering >= 1/24. T33 (R^3 lean C4, 12 pts, width 1): 1 - relu(-x3/2) exact 0 vs
  broad folds >= 1/468 (negative side). T34 (R^3 two C4 families leaning opposite, 12 pts, width 1): sideways unit
  25/216 vs signed one-family folds >= 7/60. Corollary T35: family structure fails for D20-clean. FORK R-foldopt
  PENDING (a stronger clean: coarse unit present / b folds as restriction with losses / c characterise optimum).
  Scripts model/t32_cover_gap.py, t33_broad_gap.py, t34_signed_gap.py. Referee REFEREE-BF-2026-10-25.md.
  Referee 16 fixed: T35 now an EXISTENCE statement per instance; T34 also proved for KL (model/t34_kl.py: Fenchel
  bound log(1+e^f) >= s f + H(s), s = q - v, v rational meeting Lemma T32a conditions, interval arithmetic; 636/636).
  T34's unit is a covering fold of a CROSS-family group; group folds with free thresholds restrict nothing when layer
  positively spans R^d; covering folds of any group fail (T33, D2 and KL). Objective is NOT an escape. Fork R-foldopt
  options: (a) stronger clean (coarse unit present = "hanging-ears neuron"), (b) folds as restriction (losses >= 1/9 rel,
  0 vs >= 1/468, <= 25/216 vs >= 7/60), (c) characterise optimum directly. Rec (a).
- 2026-10-25 LATE NIGHT (7) THE OPTIMUM (Shonu: "understand what actually is the optimum within clean neural nets ...
  real interpretation"; plan okay-i-think-last-fancy-cosmos.md). Paper B.O: Lemma T36 (best builds' turned copies are
  best), D28 isotropy, Lemma T37 (finite symmetric criticality), D29 (axis) + D30 (fires where silent) REFUTED on rings
  (model/to_ring.py) -- only NUMERICALLY contradicted, not refuted. THEOREM T38 (one-sided derivatives, no Clarke): each
  unit's affine piece = LS fit of the PARTIAL residual on its half-space (backfitting); clean 'fit on region, average
  off it' only at width 1 (T38.1) or nested/disjoint regions; dial/detector = Definition D31 (firing mass >= 1/2); exact check
  model/t38_fixedpoint.py on T20/T32/T33/T34. T38.2: in a deep net the target = family output seen through the unit's
  OWN write, t = c^T W h / |c|^2 (orthogonal split, model/t38_write.py). T38.3: if readers' metric G couples positions,
  target mixes positions. H-dialfit (pc_dialfit.py, 3b k=1 ckpt): fires 83%; corr with position-local fit .37 (plain
  sum) / .41 (write target, slope 1.79), bar .8 -> FAILED; reading = T38.3 (readers reach neighbours; log's moved
  members). Card 3-3 has an "Update" table; LADDER section 14. Fork R-foldopt still PENDING. Open: which REGIONS units
  take (the NP-hard part, T22); a G that couples positions measured on 3b. Referee 17 (REFEREE-BO-2026-10-25.md) fixed:
  T38.2 least only among sign +1 builds; 4a reads 3b after max pool and crush has no bias shift, so T38.2 premises fail.
- 2026-10-26 GATED FOLD (plan ~/.claude/plans/first-thing-i-want-lexical-floyd.md; paper B.G + D.11; LADDER 15; card 3-4).
  Shonu: my D22 "fold" was narrower than his meaning; his = "the network's own gate decides the region, an easy formula
  the value" ("yes that is it"; toy networks only, all derived). T32-T34 toys had no gate unit, so they never tested it.
  New nodes PENDING: R-gatefold (H-front), R-flatgate (H-flatgate), S-keepfirst (2 hidden layers, build keeps layer 1).
  PROVED: T39 (region = gate's => least unit = one LS/logistic fit on G; part 0 reduces to 1-hidden on z = h(x));
  T39.1 (no member bends inside the gate => gate fold EXACT from N's weights: relu(sum of ON members' pre-acts - Delta_0
  + M(g - g1)) + Delta_0); T40 (members bend inside gate => gate region not least: 5/182 vs >= 7/182, exact, I1 uniform).
  D34 refuted; numerically false on all 8 cases (model/tg_gate_instance.py); least unit copies one opposite member pair,
  breaks symmetry. Dividing line = members straight inside the gate (not dial/detector). Candidate stronger clean for
  R-foldopt (a): "every family behind a gate and straight inside it" (no node yet). Next: several gates; n_y >= 3
  (rank-r inside a gate, ~T9); nested gates (zoom tree). Referee REFEREE-BG-2026-10-26.md. sympy lives in .venv only.
- 2026-10-26 SHAPE (same session; Shonu: "derive the general object for the small-network optimum, later results are
  special cases"; paper B.S, LADDER 16). D35 output set O_k(P) (centred outputs on the mix, any depth/n_y); D36 optimum =
  nearest point (inf); D37 activation pattern / piece / optimal pattern. T41: optimum = min over finitely many pieces
  (an optimal pattern always exists, even under T23); inside a piece output is multilinear in layers; rule error =
  price of the wrong piece. T41.1 (2 logits, 1 hidden): closed polyhedral cones (T19), convex (T21), projection when no
  constraint active (= T38 normal eqns). T42 TRANSFER BOUND (Lean Coarse/Shape.lean transfer_bound): optimum 1-Lipschitz
  in outputs; least build for N' is within 2 dist(N,N') of optimal (D2) -> trunk eps bound IF eps = output distance to
  nearest clean net (proposal under R-cleannet). Check model/ts_transfer.py: 80/80, excess <= .66 dist. I1: 56 closed
  pieces, 2 output vectors (B* + 90-degree turn). Referee-18 (B.G) fixes done: T39a split, T40 retitled (under: none,
  part 0 proves D20), Remark B.G.1 non-uniqueness, Lean Coarse/Gate.lean. Referee on B.S: REFEREE-BS-2026-10-26.md.
  Open: for clean N, a formula naming the optimal piece from the weights.
  REFEREE 19 (REFEREE-BS-2026-10-26.md) fixed: T20 optimal PATTERNS NEST ({2,3} in ({2,3},{-3,-2})), least points
  inside do not (kink 8/5 vs 1) -> switching fails inside a piece, not between pieces. One error notion = difference of
  losses; T42 part 3 in D2 units = 2 eps delta + 2 eps^2 (NOT eps-only); N' must have N's L and widths (any-width eps
  ~ 0 for generic mixes, referee sketch); T42 side branch (R-obj RETRO). T9/T16/T17 = analogues (gated class), T9 output
  set = determinantal cone. T41.1 closure lemma; T41.2 (layer 1 kept). Lean t41_min_over_pieces, t41_error. Shonu's
  distillation framing added to B.S intro (same object; distillation = local search; optimum depends on N only via
  outputs on the mix, T26).
- 2026-10-26 EVENING: CLEAN DEFINITION (plan ~/.claude/plans/seems-like-the-last-merry-frost.md; paper A.5 + B.K; LADDER 17;
  card 3-5). Shonu: optimum is the object, folds/gates = hypotheses (R-foldopt DECIDED (c)); gates PARKED (R-gatefold,
  R-flatgate, S-keepfirst DROPPED); goal = the RIGHT definition of clean, N-only (no mix), eps a list, multi-principle.
  New rule 8 in curves/CLAUDE.md; see [[feedback-general-form-first]]. A.5: criteria C1-C5; D38 (domain X, regions,
  readers, families); D39 principles Q1 moved copies (= D20), Q2 siblings apart, Q3 votes agree where units co-fire, Q4
  gate in front, Q5 coarse unit present. model/ta5_principles.py (exact witnesses): T20 clean; T22/T23 broken lines fail
  Q3; T32-T34 fail Q2. THEOREM T43 (Lean Coarse/Clean.lean; model/t43_clean_hard.py ALL OK): detector network D40
  (kappa blocks of narrow direction caps, domain sum of block norms <= 1, r = 1 - eta/8, one positive reader; depth L
  with coarse unit) is clean under Q1-Q3 (+Q5 at depth) and T22's hardness holds for it with S_kappa-invariant mixes
  (reduction from Clique via colored blow-up). Hardness lives in the MIX; best unit = union of N's own detectors
  (piece made of N's regions; which union is hard). New fork R-mixclean PENDING (card 3-5): (a) mix property [rec],
  (b) weaker fast, (c) error, (d) principle these nets fail. Open: families without symmetry (C4); eps for Q4/Q5.
  Referee 20 (REFEREE-BK-2026-10-26.md): proofs sound; fixed: families = orbits of N's own group Aut_X(N) (so real
  nets: trivial group, C4 open); Q2 strict vs broad (peak) = Shonu's call (strict also kills real broad tuning);
  Q4 needs a proper gate; T*** has no "fast": T43 = no poly-time zoom; option (e) keep trunk as is; (d) live (D40
  weights ~ m^4, box domain kills it). Rec: Shonu answers (e) first, then (d) bounded sharpness.
- 2026-10-26 NIGHT: WHERE THE PUZZLE HIDES (plan ~/.claude/plans/okay-so-last-session-deep-teapot.md; LADDER 18; cards
  3-8, 3-9). Shonu did not follow the paper words; the workers/listeners/exam picture worked then, REPLACED 2026-10-27 by real ML objects + diagrams (see
  [[feedback-user-working-style]]). His answers: rethink how mixes enter ("the mix exists because optimal is
  subjective; the big net teaches the small one on it"); if a listening test fails clean nets, find another; photos
  question ties to the mix. New nodes R-jobmix (ROOT) and R-listen (STRUCT), both PENDING. D48 job-weighted mix (user
  weights jobs + "nothing"; photos from base law). T51/T51.1: T43's hard mix IS a job-weighted mix of itself -> hardness
  moves to the base law's SILENT part (pattern the big net never had to learn). Remark B.K.2. D49 candidate read tests
  (deciding read delta, co-active read gamma); T52: delta >= 0 always (size test only), gamma = 0 for alternatives;
  shared reason: alternatives are silent when u is on. Next idea: score teams AND alternatives (co-firing above/below
  chance) vs random baseline. Check model/t51_jobmix_listen.py ALL OK. Referee 24 (REFEREE-JOBMIX-2026-10-26.md) fixed: T51.1 holds only when the
  base law is part of the input; SILENT (open, the hopeful case) when the base law is N's own data; new Lemma T53
  (point-target last-layer units: lambda = s_u (q1(x0) - E q1)) -> mash-up workers inside N make it unclean; D49
  renamed dec/coact; T52 part 3 premise added.
- 2026-10-27 TRUNK T4 + ZOOM TREE (plan ~/.claude/plans/id-like-you-to-tingly-harp.md). Shonu: the developer chooses the
  grouping (which neurons go coarse, to how many) AND the degrade order (a zoom tree). T4 confirmed by plan approval, hash
  stored. I chose option (b) under his delegation: new units read only their group's pre-activations (under (a) the
  grouping would not shape what a unit computes). Nodes: R-group DECIDED; R-groupopt (H-group: (a) = (b) for clean N)
  and R-recur (H-recur: some tree makes step-by-step (r) optimal) PENDING; cards 3-10, 3-11. Correction told to Shonu:
  T43 with one big group handed over is still hard (the rule must pick which detectors the unit covers). Paper work:
  new Part B.W (D50 classes a/b/r, D51 AND/OR co-firing, T54 order, T55 step cost, T57 big group hard on D40,
  T58 AND group collapses exactly, k <= rank+1, 2 for two logits). Bit-length growth across steps blocks a clean
  complexity form of "(r) fails for every fast tree" -> concrete tree search on a 4-detector D40 instance instead.
  RESULTS (same day): T54 order (r) in (b) in (a), existence; T55 one-step cost (D2 polynomial per step, no bound across
  steps: bit-length growth); T57 one big group = CLEAN-TRUNK (NP-hard); T58 AND collapse exact (rank + 1, 2 for two
  logits; Lean Coarse/Zoom.lean and_collapse; model/tr_and.py exact, ALL OK); T59 exact merge = LP, blend obstruction
  for any output weight/offset; T59.1/T59.2 on D40: a pair merges free iff its blend is not in the mix.
  model/tr_tree_search.py (kappa=2, m=2, D2 numeric): all 18 merge orders have an (r) path to the optimum; 10 on
  every tie branch (all good-pair-first ones). Referee 25 (REFEREE-BW-2026-10-27.md): H-group was vacuous (now over
  family grouping; content only for low-rank families), D52/H-recur trivial (now from N, two-child merges), D53
  refuted by T20 (now "one member's region"); all fixed; round 2 launched. LADDER section 19 written (new style).
  Round 2 (same day): proofs T57-T59.2 sound; D53 restated again (whole layer, D2, silent blend for every pair; the
  detector network refuted the first restatement), H-recur/D52 only for targets with groups of size 1, eps_recur over
  every level (toy = top level only). Round-2 fixes not re-refereed. Next: level-2 V_b comparison in tr_tree_search,
  clique lemma (one unit per block merges iff colorful clique), D53 on rings, deeper layers (Remark B.W.3).
- 2026-10-27 (later, "continue until a decision is needed"): T59.1-T59.3 general (zero-loss one-unit groups on D40 =
  colourful cliques; sigma linear-in-x proof); Lemma T60 (r copies a direct unit iff parts in children's row spaces;
  part 3: a dropped direction never returns); Cor T60.1 (H-recur holds on D40 with T43's mixes for no-bad-pair
  targets, every zoom). Level-by-level tree search: 16/18 orders optimal at every level; the 2 through {AC, BD} lose:
  Prop T61 (exact, model/tr_acbd.py): {AC, BD} has a zero-loss build with anti-detector units
  relu(-20 z_A - 40 z_C - 45/2) + relu(-40 z_B - 20 z_D - 45/2) - 18; the copy rule of (r) is the cause. Referee round 3
  fixed (T43's mixes only; T60 part 2 for pre-activations). T61 not refereed. STOPPED at card 3-12 (R-deep: meaning of
  "reads only its group" in layers >= 2; options A through N's wiring with decodes [rec], B free read, C one layer at a
  time) -- Shonu's decision.
- 2026-10-27 EVENING "what are the T61 neurons" (plan id-like-you-to-tingly-harp.md v3): B.W.2: D54 presence/split/dial,
  D55 conflicting segment/group; Lemma T62a dip identity (Lean dip_switch, dip_same_sign); T62b every silent blend is
  paid by a switching unit with a > 0; T62: family grouping {AC, BD} forbids presence units (cycle parity + case
  analysis, block swap + per-block reflection symmetries). Exploration model/tr_poles.py: free -> presence; one
  conflicting group -> clique presence (lossy); family groups together -> split dials (8/8 pieces); same-block {AB,CD}
  no zero loss. Essay reading: continuous family (free) -> detector; discrete alternatives (conflicting) -> contrast
  dial ("negative poles", I1). New: H-freefam (R-freefam PENDING), D57, card 3-13 (R-kinds; rec (b): rule checks each
  group by T59's LP and picks the kind). STOPPED at cards 3-12 and 3-13.
  Referee round 4 (B.W.2): no errors; added T62 part 4 (every zero-loss unit fires at exactly one of its members ->
  split); CORRECTION: T61's {A,C} unit is on at A, off at C, HIGHEST AT B (not A).
- 2026-10-27 NIGHT CIRCUITS (Shonu: unify AND/OR/detector/anti-detector, a circuit-level object, scalability; recalled
  NEURON-MATH Part V R2c "training cancels errors through layers" as a sign the frame is too small, NOT a method).
  Part B.X: D58 circuit, D59 kink profile (lines, kinks, dips, dip lines), D60 coarse circuit (new units in any
  layers). T63 kink formula any depth (Lean kink_green; tr_kinks.py exact on 300 random 2-3 layer nets). T64 one-layer
  unit = one monotone kink per line (gives T59 pt2, T62a, T62b). T65 covering bound tau(D) of strict dip lines;
  T65.1 = T20's impossibility. T66 tent map: depth k, 2 units/layer exact vs one layer >= 2^(k-1)-1 (Telgarsky 2016
  row added). Remark B.X.1 four kinds as kink patterns. D61 conjecture. Card 3-14 R-circuit (rec (b): trunk groups
  become circuits, coarse circuits in any layer). Sign note: dips come from UPWARD jumps (convex bends).
  Referee round 5 (B.X): T63-T66 correct; D61 was FALSE ("exactly when": T61 instance tau = 1, no exact width-1) ->
  restated for free families; content count 0 structural (elementary + known Telgarsky separation). Card 3-14
  recommendation changed to (a) keep T4, dip lines as a tool, until a clean network with a deep job is found.
  STOPPED at cards 3-12, 3-13, 3-14.
- 2026-10-27 LATE: UNIT BRANCH (plan ~/.claude/plans/oh-yeah-last-session-tingly-milner.md; paper
  research/clean/curves/UNIT-BRANCH.md, lab research/clean/units/, LADDER 22, card 3-15). Shonu: stop one-theorem-per-
  neuron-kind; derive the unit from demands (see [[feedback-derive-the-unit]]). Demands U1-U5 (U4 independence decides).
  PROVED: TU1 box well defined iff convex; TU2 silence -> exact additivity; TU3 |I| <= 1/2 E|dS||dT| (KL Hessian <= 1/2);
  TU4 linear: independent iff activity uncorrelated (blocks of Cov on the mix); TU5 covering boxes + unlinked + disjoint
  supports -> exact independence at ANY depth; TU6 T61 instance: free I <= -1/8, covering independent but >= 1/2 per
  group; TU7 co-firing breaks it (last layer); TU8 unit = co-firing component (covering); TU9 covering composes (nest);
  TU10 FREE boxes: no independence at all (an OFF member's negative pre-activation still varies; a free unit reads it
  and fixes the neighbour; hinge functions span functions on P); TU11 free -> unit = whole network. DU2 (free = covering
  under gates) WITHDRAWN. Lean Coarse/Unit.lean. New nodes R-indep (card 3-15, rec (a) covering), R-district,
  S-linunits. Node map: R-circuit answered, R-deep/R-kinds go away/absorbed, 5 change, 9 unchanged.
  REFEREE (REFEREE-UNIT-2026-10-27.md, 19 findings, 1 error) all fixed, not re-refereed: builds are feedforward (not
  N's layers); TU4 one-way (converse needs |S| <= n_y); TU5/TU8/TU9 use "covering on P" (silent at mix points outside
  the support; T59.2's presence unit is this, NOT a fold); TU8 for partitions of the whole last layer, "no cancellation"
  defined; TU11 = "no two COSTLY sets independent" (not "whole network"); H-district = co-firing HUB per district
  (essay's gate is upstream); R-circuit/R-deep only "partly informed". Toy (units/toy_district.py, toy_pcover.py): free
  boxes help across gated districts (DU2 withdrawn); strict covering costly; covering-on-P = free single optima.
- 2026-10-27 NIGHT: OUTPUT-FAITHFUL BOX (plan ~/.claude/plans/okay-nice-work-last-soft-peach.md; UNIT-BRANCH.md B.5; LADDER 23;
  card 3-15 rewritten; new node R-outfaith PENDING). Shonu: clean + well-trained should already be mostly independent;
  independence is graded like every clean principle; "what IS the unit in an image model?". Remark B.4: TU10 uses no
  principle of clean (true for any net) -> the free box reads below-zero values N's readers never get. DU6 output-faithful
  (box depends only on members' sent outputs, threshold theta for weak ones); TU12 (read units off on the weak set or
  global constants); TU13 (only channel = one constant where the set is silent; -(sigma_S+sigma_T) <= I); lab: constant
  channel is STRONG (|x| I=-13/12, toy (1,2) -0.107, T61 -0.37) -> my "harmless constant" prediction FAILED. DU7
  silence-preserving (essay: negative poles are gates, not opposites) + TU15: = covering on P, I = 0 exactly for sets
  never on together -> covering is DERIVED, not Shonu's choice. TU14 graded bound: strong co-firing + dropped weak writes
  (+ constants); essay figure 69.47% vs 70.08% (strongest quarter kept, Inception last layer) = Shonu's "72 vs 71".
  TU16 (descendants meet only at logits -> additive). Conjectures DU4 (district's coarse unit = its judgment, "general car
  neuron", under H-district-theta) and DU5 (meeting units; first version false, referee). Referee B5U: 2 errors 6 gaps,
  fixed, not re-refereed. Free logit bias cuts toy single losses ~12x (most single loss = a missing constant).
  Open: N-only formula for the graded term; DU4 check on the circle toy; InceptionV1 H-district-theta measurement.
- 2026-10-27 LATE NIGHT: WHAT "HELP" IS (Shonu: "what is that error really coming from?"; LADDER 24; card 3-16; node
  R-readbias PENDING; conjectures renamed CU1-CU3 to avoid clashes with Definitions DU4-DU6). Anatomy: original toy's
  free joint units = shared up/down units (help_anatomy.py). Sheltered and gate-in-front toys (toy_shelter.py,
  toy_q4.py): help stays huge (-0.24, -0.20) -> my predictions FAILED. Cause = TU19: one ReLU unit is one-sided, a
  contrast needs the READER'S THRESHOLD; builds froze readers' biases, so another box's constant smuggles it. With
  re-fit reader biases (DU8, toy_bias.py) help ~0 in all toys (+0.006, +0.0001, +0.0001, -0.003). Remaining sources
  (Remark B.6): split dial (TU20 exact, |x|: I = -11/16 -> I<0 means "one feature, regroup"), shared threshold (I>0,
  small). District's 1-unit coarse version = CONTRAST DIAL over its members (siblings on both poles, reader threshold
  sets the flip; cu1_dial.py) = essay's "taxi on the ambulance's negative pole". CU1 restated to dial form. My rec on
  3-16: readers re-fit bias (makes 3-15 principles optional). Second referee (TU15-TU20) launched.
  REFEREE 2 (REFEREE-B5U2): "re-fit readers remove help" FALSE in general: silence used as an opposite ("T off ->
  answer like S", via T's spare constant unit) survives DU8; DU7 forbids it; worth <= one more unit (TU18 pt 2). CU3
  withdrawn -> TU21 (two-sided bounds). Old conjectures DU1/DU3/DU2 renamed CU4/CU5/CU6. Rec on 3-16 now: DU8 + DU7.
- 2026-10-27: Shonu DECIDED R-readbias ("yea they can move their biases"). Asked "same error on the weights side?".
  DU9 (readers re-fit weights on kept units, node R-readweights PENDING) + TU22 (sheltered + kept judgment unit +
  re-weightable -> I = 0 exactly). toy_weights.py: singles drop (orig 0.0094 -> 0.0050), no new help; orig (1,1) I =
  +0.0086 (a cost: leaks collide on shared readers); pose -0.004 (help via N's pose units, read by boxes, not logits).
  LADDER 25. Not refereed: TU22.
