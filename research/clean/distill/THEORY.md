# THEORY: the tree of distinctions

Opened 2026-11-25. Plan: `~/.claude/plans/pasted-content-id-0373-while-federated-coral-of-ed9bfe10-43f8-417f-84ad-1f5ddb39ab93.md`.
Results of the checks: `CALC.md` section 3. Shonu on the idea: "i LOOOOVE your tree idea."

## 1. The concept

> 2026-11-26 night: this section is law v1. The live node card (law v2 tests, status per law) is `NODE.md`. This section is replaced only after the identity law (NODE test 1) is decided.

**A smaller network is the parent's tree of distinctions, cut higher, in every layer at once.**

```
parent block b         car ──┬── taxi        (fine readers in block b+1)
                             └── ambulance
coarse block b         car-branch  (read = root of taxi+ambulance, fires on both, writes one message)
coarse block b+1       readers of "car" only. The taxi-vs-ambulance reader is gone too.
```

| Part | What the concept says | Source |
|---|---|---|
| The tree | Each neuron draws one distinction. Its children in the next layer refine it (taxi vs ambulance under car) | Essay 223-241, 137, 815-821: every branch traces back to one local origin |
| Why the parent holds the small network | Derived from the essay (not the essay's own claim): training grows broad distinctions first. A small network runs out of room and stops higher. So its units sit in the parent, as the shared roots of branches | Essay "Why Networks Generalize", "More on Circuit Ecology", lines 23, 251, 771 |
| One coarse neuron | One branch of the parent: the siblings under one parent | Essay 239-251 |
| Its read and firing | It fires on the UNION of the branch's strongest firings (an OR). Not the siblings' shared read direction (an AND): that collapses onto one member (CALC 3.2-3.3) | Essay 251. CALC 3.5 |
| Its gate and firing | The threshold opens on the union of the branch. A coarse neuron fires more broadly | Essay 239-251 |
| Its write | The branch's message, as the COARSE readers downstream read it | New. Every past fold kept A's fine readers |
| Every layer at once | A distinction stays only if its writer AND its reader stay | Guiding light: the whole network |
| Order of cutting | Rank each distinction by how strongly it is written times how strongly it is read. Merge the weakest first. One ranking gives 95, 50 and 5% | Balanced truncation. Unified-calc rule |
| Stream and heads | Same rule. The stream has no fixed basis, so there the cut is a projection (the KD student's stream is a linear view of the teacher's, R^2 .97 -> .63). Heads merge as branches of heads | CALC 2.2, 2.4. Essay 461 (low confidence) |

Why it matters: past folds averaged look-alike neurons, and they kept fine readers that still wanted the fine message. The tree changes both in one operation.

## 2. Where others stop (the bar to pass)

| Work | Same as which part | Where it stops | Read |
|---|---|---|---|
| Model folding (2502.10216) | merge rule: mean read, summed write | k-means on look-alikes. LLM 20% removed | CALC section 1 |
| Neuron Merging (Kim et al. 2020, 2010.13160) | weight-only merge, compensation in the next layer | Each pruned neuron goes to its most cosine-similar kept neuron (threshold t = .1 conv, .45 LeNet), scaled by the norm ratio, added into the next layer's input weights. Exact only when the pruned neuron is a positive multiple of a kept one (ReLU). Per layer. ImageNet ResNet-34 at 30% removed: 37.4% vs 73.3% dense (pruning 17.1%). LeNet 80%: 80.0% vs 89.8% | Sections 1-3 word for word, ImageNet numbers as a summary |
| Hamadouche & Hussain 2026 (2607.05457) | the ranking (reach times read) | Balanced Gramians from activations and logit Jacobians, per layer. Used ONLY to pick each layer's width. The narrow network is then trained from scratch or distilled. No neuron identity is kept. Pruning to those widths with no training: MNIST 34.5%, CIFAR-10 11.4%. No nonlinear error bound | Method and results word for word (HTML) |
| SliceGPT (2401.15024) | stream projection | PCA with calibration text. 25-30% removed | CALC section 1 |
| Balanced truncation (Moore 1981); CompreSSM (2510.02823), Ezoe & Sato 2024 | the ranking | proven bound for linear systems. Used on SSM layers | search summary |
| DeepInversion, ZeroQ | data-free scale from BatchNorm statistics | ResNet only. CLIP's LayerNorm stores none | not yet read |

What the tree adds: the unit is a branch of siblings (different jobs under one root), not a near-copy. The ranking chooses WHICH units, not only how many. The cut is the same in every layer, so readers coarsen with writers.

## 3. Can the input be the network only?

| Piece | From weights only? | Reason |
|---|---|---|
| The tree (who is whose child) | yes | children = readers of a parent's write, in weights (Shonu's definition) |
| Coarse read directions | yes | roots are weight algebra |
| Coarse write directions | yes, once the cut is given | the message is read by the kept readers |
| Thresholds (when a gate opens) | not in general | a gate needs to know where inputs sit along its read. ResNet-18 stores a summary (BatchNorm mean and variance). CLIP stores none |
| How much each job counts (95 / 50 / 5) | not in general | the same weights with two input mixes give two different best small networks. This weighting is the goal: the future capability dial |

Structure comes from the weights. Scale and weighting come from a goal. Whether the weights alone hold enough about the training inputs to set the scale is the next test.

## 4. Facts to build with, and the next targets

| Item | Fact | How the tree handles it |
|---|---|---|
| Half-space | One GELU unit fires in one half-space | That is what makes a coarse neuron coarse: one root, one side. The fine splits are the part we choose to drop |
| Linear bound | The balanced-truncation bound is proven for linear maps | Inside one gate pattern the network is linear. A bound with gates is a target for the math track |
| Traces | The KD student is as close to the no-teacher baseline as to the teacher (.628 vs .616). Distillation's own mark is +.10 in deep blocks | Much of the small network is what any good network learns. That is the most stable part to calculate |
| Depth | The student runs one block behind in mid blocks | Next target: the same tree across depth |
| Write and coarse readers | The first check tests the read and the gate | Next check: build the coarse write and score KL to the teacher |

## 5. Our own record, read as leads

| Row | Result | What it tells the tree |
|---|---|---|
| N11-weights | summed children's read vs best single A neuron: .234/.241/.242 vs .206/.225/.239 | the SUM of children is the wrong root. The shared direction is the fix |
| OC-fold-w | best single A neuron beat the fold (.268 vs .247) | averaging is not the root |
| N11-ruleR-act | co-firing groups matched as well as children | the check includes a co-firing rival |
| VM12-absorb | from-random neurons are coarse in which photos they fire on, not cell by cell | read coarse neurons per photo |
| OC-morph | parent read + children's summed write kept .43 / .68, A's fine readers kept | the whole-network cut removes this handicap |
| OC-reads-w | sibling reads 93-98% private | the root is a small shared part. Its threshold must sit low enough to catch the whole branch |
| N11-rates | B fires .40, A .17 | broad coarse neurons (also logged as the cause of a recall artifact) |
| N4-agents | 16 of 30 Small neurons drop Base's object condition | the tree's prediction, by eye. A blind version comes later |

## 6. First check (CALC 3.2): the first root rule lost

| Group used for a root | Student neuron match vs one teacher neuron (blocks 4-12) | Photo use |
|---|---|---|
| readers of one parent's write (weight-children, k = 8) | .00 to .03 lower | none |
| parent's write as those readers read it (the one fix) | .01 to .03 lower | none |
| neurons that fire together (the registered rival) | .027-.064 higher (KD), .019-.060 higher (baseline) | per-photo (FIT half) |

C1, C2 and C3 failed. The rule's gaps are in the tool: adjacent MLP links only (no attention, no later blocks), class token only, k fixed at 8 with weak child scores. The co-firing gain shows up in the baseline too, so it does not show the student keeping the teacher's branches. Whether a branch should be defined by firing is Shonu's call. Next steps, in order: a high-variance-direction control, then children read through attention and later blocks.

Vote families (CALC 3.3, Shonu's idea: siblings = neurons that take the same votes from the same parents): a second photo sample of the same classes gives the same map (overlap 1.00). Weights alone give the same map in blocks 2-6 only (.86). Vote roots are .01-.03 below single teacher neurons in blocks 4-10, and +.007 / +.018 in blocks 11-12 (the baseline gains there too). Members co-fire only .11-.19. The root is a blurred copy of one member.

Control (CALC 3.4): the co-firing win was a lean toward high-variance directions.

Reconciled with the essay (CALC 3.5, Shonu: "if it disagrees, then our build is wrong, not the essay"): 3.2-3.4 used three wrong microscopes (shared direction instead of union, correlation over the quiet bulk instead of strongest firings, class token instead of strongest crop). With strongest crops, the numbers still show no union gain, but by eye (not blind) all 10 student neurons do the same kind of job as their best teacher neuron, and 6 of 10 cover what 2-3 teacher neurons split (camera + gun, clocks + screens, corn + timber + tractors). The numbers missed it for two named reasons: the loudest member owns the union's top photos, and exact-photo overlap of the top 1% is too strict for a broad job. The "narrow stream" target is withdrawn. Next: equal-share unions, a job-level score, and a blind reading.

## 7. CALC 3.6: co-firing units read as verbs. NOT a tree test (corrected 2026-11-25)

Plan: `~/.claude/plans/id-like-you-to-fizzy-cherny-of-28afb932-72ad-4399-8251-822044650347.md`. Checklist: `../NEURON-CHECKLIST.md`.

Correction (Shonu): 3.6 used the 3.5 "members", 3 teacher neurons per student picked by shared photos. They are co-firing units, not a parent and its children. The child rule (R1) was never used. A first version of this section read the T2-T6 fails as facts about the tree ("siblings do not gate", "convergent, not divergent", "the split does not hold"). Those conclusions are WITHDRAWN. They blamed the structure for my test's object.

What 3.6 does show, without tree words:

| Finding | Status |
|---|---|
| Each student neuron does one job; noun lists dissolve into one pattern | readers, J1 9 of 10 |
| Its co-firing teacher units do the same job or a facet of it | readers |
| Student words vs best-teacher words: no difference found (.756 vs .740) | J2 FAIL, but the sheets leak (shuffled words .625) |
| Block-12 patch firings are dead (class-token pooling) | code |
| Block 11: KD does 21% of jobs mainly on the class token (teacher .5%, baseline 5%) | description |

What the tree predicts and 3.6 did not test: children split what their parent lets through, and turn their siblings away. Shonu: a neuron is a distinction, so turning away is expected. A lead from the ResNet-18 record (onecoarse LOG 2479-2502): n23 votes down on s3:174 (n297's up condition), n385 votes down on s3:184 (n163's up condition). Whole-read cosines (-.035) average such votes out. The actual test is CALC 3.7.

## 8. The actual tree, first test (CALC 3.7, ResNet-18 parent s3:69, 2026-11-25)

Objects: parent s3:69; its 11 children by R1; B's coarse neuron n101.

| Tree part (section 1) | Status at this parent |
|---|---|
| Children refine the parent | holds: every child fires where the parent is on (C1, .69-.97 vs base .20) |
| One coarse neuron = the parent of a branch | C5 passed; how n101 was picked is not yet checked (possible circularity) |
| Children turn each other away | Seen in blind words at the negative extreme: inside the parent's group, 8 of 11 children's negative poles read like a sibling's positive pole (profile vs front-on, whole figure vs close layout, rays vs ring). Numbers agree for 4 of 8. Not seen on typical cells: siblings rise together there (C3 medians). Open: is it special to siblings? Null neurons' negative cells also fall on child cells (CALC 3.7) |

