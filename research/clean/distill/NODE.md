# NODE: the node card of the neuron tree

Opened 2026-11-26 night. Plan: `~/.claude/plans/pasted-content-id-352f-okay-i-humming-quiche-of-069b918e-cca3-4972-9957-c9421d316ff9.md`.
Rounds change ONE row of this card per test. A change that moves a number but changes no law is not a result.
History: `THEORY.md` section 1 is law v1 (2026-11-25). `FANOUT.md` holds rounds 1-3 (technical rounds).

## Concept line

A smaller network is the big network's tree, cut higher, in every layer at once. **Open (test 1, not decided in L1):** is a coarse node one big-network node (morph) or a merge of several? After L1: training does not widen kept gates; deleted jobs spread over many kept nodes; the stream needs every direction kept nodes use, so a narrow stream needs coarser nodes.

## Names (Shonu, 2026-11-25)

| Name | Object |
|---|---|
| morph | parent's read, children's summed write, one unit (a calculated build) |
| widened neuron | a kept neuron that keeps its job and takes a share of a cut job (what a retrain from A shows) |
| coarse neuron | a neuron of a network trained small FROM THE START (the essay's object) |

Answer keys: R students (from random) say what a coarse node IS. P students (cut from A, then trained; `*_scores.pt` field `kept` = A index of every neuron) say what training CHANGES in a kept node. P students are circular for the identity question.

## The card

| Part | Law v1 | Law v2 under test | Evidence | Status |
|---|---|---|---|---|
| Identity | coarse node = merged siblings (union) | coarse node = the parent (morph), with a wider gate | merges lost at 85-95% (CALC 3.2-3.5, FANOUT r2 #4, r3 #6). From-start: B wheel neuron covers ~8 A neurons; B fires .40 vs A .17. Essay 251 | open (test 1) |
| Write | the branch's message, as coarse readers read it | deleted nodes' messages are carried SPREAD over kept nodes; kept nodes keep their own write | kslice (a spread ridge fit) wins at 85-95% on both networks. L1 write read (weights, no photos): at 95% the trained write change lies in the deleted neurons' write span 1.5-5x more than an equal-size kept-span null (7 of 8 blocks); at 50% training rewrites the writes (change .9-3.8x the write) and the change is NOT specific to the deleted span | supported at 95% (build + weights); at 50% training does something else (open) |
| Parent | one parent; children = readers of the parent's write (R1) | one main parent, minor others; a deleted node's message goes to its main parent | L1 firing read (a side fact, not this law): a deleted job's firing pattern spreads over tens of kept nodes of the same layer, not to the most similar one. The R1 parent was not tested | **untested** |
| Gate and firing | the gate opens on the union of the branch | when children are deleted, the kept node's gate opens to cover where they fired | L1: networks trained small FROM RANDOM fire more broadly as they shrink, in every block from s2 on (s4b0 firing rate: A .19, 95% .19, 50% .24, 5% .50). P students do not widen their kept gates (ratio .74-1.08 in every A-rate tercile, so not a pull to the mean), except the last block (1.17-1.30); more training moved them toward 1 (s4b0 .88 -> 1.01) | **supported for coarse neurons** (from start); not shown in P students; BUILD it (L2) |
| Negative pole | (none) | a child turns its siblings away; the vote leaves with the deleted sibling | CALC 3.7 blind words 8/11; essay 353-357 | open (read only in L1) |
| Weak firing | (none) | a node's statistics come from its strong firings, both poles | essay 309, 353 | open (L2) |
| Which nodes stay | rank by write size x read size | keep nodes that many kept nodes build on; delete leaves first | weight order = random at 95% (r3 #5); essay 719 | open (L2) |
| Stream | cut by the same rule, as a projection (in r1-r3: its own ranking and error) | the stream keeps only directions that kept nodes write AND kept nodes read; its size follows from the node cut; no separate error (Shonu 2026-11-26) | L1 test 4 (CLIP, quick 200, pink only): the prediction 'no extra error' FAILED at every width below 767 (653: 30.5%, 700: 39.5%, 737: 45.5%, 767: 59.0% = node cut alone). Findings: pruned nodes still use almost every direction (99% of kept write weight energy in 605-738 dims per block at 85%); the 50% node cut alone is at chance with the full stream | **open**; no free width found yet; pink energy measure = a bug |
| Heads | branches of heads | a head is a node with the same laws | head merges break CLIP; essay 385, 461 | partly |
| Every layer at once | a distinction stays only if its writer and its reader stay | same | held in every round | supported |

## Generics (rules in node words, with the adapter per network)

| Rule | Node words | ResNet-18 adapter | CLIP ViT adapter | Status |
|---|---|---|---|---|
| kslice fold | each deleted node's write is carried by kept nodes of the same layer, in proportion to how well their firing predicts its firing (generic input); kept nodes keep their own write | inner channels of a block -> conv2 columns | MLP neurons -> `pr_w` columns | both networks |

## Round log

### Round L1 (2026-11-26 night)

Code: `fanout/anskey.py` (laws `gate`, `carrier`). Photo use: reading only, 1,000 test photos. Identity check (A vs A): ratio 1.00, job corr 1.000 in every block (PASS). Unit = inner neuron (conv1 -> bn1 of each block, before the ReLU). "Job" = per-photo max firing over cells.

**Test 3 answer key: does the gate open wider when children are deleted?** Prediction (written first): at 50% most kept gates open wider, more than at 95%, and each keeps its job (corr > .7).

| Block | 95% rate A -> trained (ratio) | 50% `W50P_s0_ph4` rate A -> trained (ratio, share wider) | 50% own-job corr | A-in-A |
|---|---|---|---|---|
| s1b0 | .604 -> .600 (1.00) | .782 -> .704 (.97, .38) | .757 | 1 |
| s2b1 | .257 -> .248 (.97) | .303 -> .272 (.90, .38) | .563 | 1 |
| s3b1 | .207 -> .198 (.96) | .258 -> .226 (.89, .32) | .537 | 1 |
| s4b0 | .194 -> .186 (.95) | .235 -> .233 (1.01, .52) | .528 | 1 |
| s4b1 (last) | .203 -> .207 (1.01) | .258 -> .319 (**1.21, .82**) | .490 | 1 |

Reading: CONTRADICTED in blocks 1-7. Training does not open the kept gates wider (median ratio .89-1.01). It holds only in the last block, the one the classifier reads (essay 287: memorization sits near the output). Second fact: the Minitron cut already keeps the broad neurons (s3b0 kept rate .43 vs cut .27), so "a coarse node fires broadly" is met by WHICH nodes stay, not by a gate change. Per the loop rule, the gate law is NOT built (training did not do it), except as a last-block row later.

**Test 1 partial (P student, not the identity test): does a kept node stay itself?** Best A match of each trained kept neuron = its own A neuron: 95%: 100% in every block (own corr .91-.99). 50%: 64-93% (own .49-.76; best other kept .26-.70; random pair .06-.40). 16-33% of kept neurons in blocks s1-s4b0 now match a CUT neuron better than themselves (7% in s4b1).

**Test 2 answer key: does a deleted node's job go to ONE kept node?** Gain = corr(trained kept j, A's cut d) minus corr(A's j, A's d). Null = the same gain toward other KEPT neurons.

| Block | cut | kept neurons gaining > .05 per cut node (null) | top1 / top2 gain | top1 null | top carrier = nearest kept in A |
|---|---|---|---|---|---|
| s2b0 | 64 | 23 (19) | .222 / .185 | .182 | 0% |
| s3b0 | 128 | 37 (32) | .222 / .183 | .187 | 2% |
| s3b1 | 128 | 33 (25) | .249 / .198 | .188 | 1% |
| s4b0 | 256 | 58 (54) | .243 / .198 | .215 | 1% |
| s4b1 | 256 | 41 (35) | .173 / .145 | .159 | 0% |

Lean toward cut jobs minus lean toward other kept jobs (best match): A-in-A -.020 to +.029; trained +.030 to +.075 in all 6 blocks s2-s4.

Reading: a deleted job is picked up a LITTLE (+.03 to +.06 over the null) and SPREAD over tens of kept neurons. No single carrier (top1 close to top2). The carrier is not the nearest kept neuron in A (0-6%), so "send it to the most similar kept neuron" (Neuron Merging, fold partner) is CONTRADICTED. This repeats VECMATH 12-13 (a repair stays spread). It supports the spread form (kslice). NOT tested yet: the morph form, where the deleted node's PARENT (one layer up) carries its message. This read only looked at carriers in the same layer.

**Test 4: does the stream size follow from the node cut, with no extra error?** CLIP, nodes cut at f (heads by floor activity, MLP by weight), MLP fold with kslice, pink input. Rule `fanout/rules/l1_stream.py`. Check: law mode at f = 1 gives KL 9e-7 (PASS). Prediction (written first): at 85% full ~45%, all ~30%, law about +5 over all and below full. Quick 200 photos, KL / agree:

| Stream mode | 95% (width 730) | 85% (width 653) | 70% | 50% |
|---|---|---|---|---|
| full: node cut only, stream 768 | .280 / 78.5% | 1.355 / 59.0% | 6.117 / 6.5% | 10.418 / .5% |
| all: rG basis from ALL readers and the full stream (law v1) | .538 / 69.5% | 3.307 / 33.5% | | |
| law: rG formula from KEPT readers and KEPT writers only | .584 / 70.5% | 3.581 / 30.5% | 9.766 / .5% (538) | 11.646 / 0% (384) |
| rand: random basis (null) | 1.209 / 57.0% | 9.553 / 1.0% | | |

Width sweep, nodes at 85%, law basis: 653: 30.5%. 700: 39.5%. 737: 45.5%. 767 (all centred directions): 59.0% = the full row, so the build is exact and every lost point comes from dropped directions.

Span read (`fanout/out/l1_span.json`, eigenvalues of G^1/2 C G^1/2 on pink): directions holding 99% of the weighted energy. All nodes: 609. Kept nodes at 95/85/50/25%: 604 / 577 / 454 / 314. Energy the uniform cut drops: 85%: .5% (kept) vs .6% (all). 50%: 1.5% vs 4.2%.

Readings:
1. The kept-only span IS smaller, and more so as the cut deepens (99% energy in 454 vs 609 directions at 50%). This is the law's prediction.
2. The law basis TIES the old basis at 85% (30.5% vs 33.5%, inside the quick noise). The prediction (+5) FAILED.
3. The energy measure is the bug. Dropping 30 directions with .1% of the measured energy costs 13.5 points of agree. Pink-noise second moments call directions empty that kept nodes still use (same lesson as step 0 F2 and round 3 #4).
4. By weights, kept nodes still write and read almost every direction. Directions holding 99% of the kept nodes' weight energy, per block (blocks 0-11): writes at 85% 605-738, at 50% 566-720, at 25% 486-640. Reads at 85% 503-743, at 25% 388-692 (`fanout/l1_span.py`, `fanout/l1_rank.py`, weights only). The uniform stream cut is 653 / 384 / 192. So the stream size that follows from a PRUNED node cut is close to full. The uniform cut of the stream is what breaks the law.
5. At 50% the node cut ALONE (stream at 768) is at chance (.5%). At 50% the stream is not the blocker. The nodes are (most likely the heads: round 1 heads-only 50% = 5.807, at the flat level).
6. Concept consequence: deleting nodes does not shrink the span that kept nodes need, because kept nodes keep their fine writes. The stream can shrink for free only if the nodes themselves get COARSER (fewer write directions). The KD student (ViT-T) has a 192-dim stream with coarse nodes. So test 4 points back at test 1: the identity of a coarse node decides the stream size.

**Test 1: what is a coarse node: one A node (morph) or the union of several (merge)?** R students (trained from random), each neuron matched to A neurons of the same block by per-photo max firing. U_k = corr with the elementwise MAX (union) of its top-k A matches. Null = top-1 plus random A neurons.

| Student | s3b0 U1 -> U5 (null5) | s4b0 U1 -> U5 (null5) | s4b1 U1 -> U5 (null5) | share of neurons U3 > U1 + .05, s4b1 |
|---|---|---|---|---|
| `W95R_s0` (95% width, from random) | .396 -> .423 (.227) | .333 -> .384 (.160) | .280 -> .330 (.112) | .34 |
| `W50R_s0` | .401 -> .431 (.229) | .338 -> .387 (.157) | .264 -> .320 (.112) | .42 |
| `W5R_s0_x4` (26 neurons in stage 4) | .415 -> .457 (.251) | .344 -> .372 (.131) | .245 -> .288 (.115) | .46 |
| `W50P_s0_ph4` (P start, circular) | .592 -> .542 | .551 -> .502 | .492 -> .442 | .13 |

Reading: a neuron trained from random matches the union of 2-5 A neurons better than its best single A neuron (+.03 to +.06, far above the null). But the gain is the SAME at 95%, 50% and 5% width. So this read shows how two separately trained networks relate. It does not see coarsening. Test 1 is NOT decided. The cause is the microscope (per-photo max at one block position), not the tree. Next microscope: compare reads and writes in weights (`vecread/`), or build both forms and score them.

**Corrections after review (same night).**
- Gate: re-read in the R students (`anskey.py rate`) and by A-rate tercile. The from-start students DO fire more broadly as they shrink. Status changed from "contradicted" to "supported for coarse neurons".
- Write: the firing read (test 2 above) cannot test a write law. A weights read (`anskey.py write`, no photos) replaces it (card row Write).
- Parent: the firing read did not use the R1 parent. Status back to untested.
- Stream: the prediction failed at every width. Status open, not supported. The 59.0% row (stream at 768) has 72,966,756 parameters: it is NOT an 85% network (the uniform 85% cut has 62,048,886).

| Block | write read 95% `W95P_s0`: change / write, share in cut span (null, chance) | 50% `W50P_s0_ph4` |
|---|---|---|
| s2b0 | .151, .035 (.016, .005) | 1.316, .294 (.310, .111) |
| s3b0 | .222, .064 (.018, .006) | 1.693, .335 (.331, .111) |
| s3b1 | .257, .034 (.021, .006) | 2.462, .340 (.319, .111) |
| s4b0 | .308, .036 (.012, .006) | 1.294, .245 (.373, .111) |
| s4b1 | .299, .094 (.035, .006) | 3.829, .677 (.638, .111) |

What round L1 skipped (said plainly): no critic agent; no held-800 read (every CLIP stream row is quick-200, pink input only); no ResNet arm for test 4; no prediction written before tests 1 and 2; test 2 not run on R1 parents; no named nodes (s3:69 and its R1 children were in the plan and were not used).

### Round L1 summary (between-round step 1-7)

Concept line before: "a coarse node = merged siblings" (v1). After: still OPEN between morph and merge, with three new facts:
1. Coarse neurons (trained small from random) fire more broadly the smaller the network. Kept neurons in a cut-then-trained network do not (yet).
2. At 95%, training moves kept writes into the deleted neurons' write directions, as the spread fold does. At 50%, training rewrites the writes, and the change is not "carry the deleted messages".
3. The stream needs every direction that kept nodes use. Pruned nodes keep fine writes, so they use almost the whole span. A narrow stream may need coarser nodes (not shown yet).

Withdrawn: none from rounds 1-3. Narrowed: round 2 #3 "the stream needs the right form AND input" is now read as a span-measure bug inside one law, not a separate stream problem.

Next laws (round L2), by predicted effect:
0. Gate build: open each kept node's gate as coarse neurons do (from the R-student rates), on named nodes first (s3:69's R1 children).
1. Heads are nodes: the 50% node cut is at chance even with the full stream. What law of a head node fails? (answer key: ResNet has no heads, so CLIP-KD by build only)
2. Identity, new microscope: morph build vs merge build at a readable part-only cut (CLIP MLP-only 70%), plus a weights-based match of R students.
3. Span-measure bug (bug round, small): measure "directions kept nodes use" from weights or the self-made input, not pink energy.
4. Parent carrier one layer up (the morph form of test 2).
5. Weak firing; which nodes stay (from the card).

Essay pass for L2: essay 435-461 (attention: "where it looks", "what it looks for", heads "broad and recursively generalized").

## Drift check (per round)

| Test | Object (picked by) | A node of the card? |
|---|---|---|
| L1-3 gate | every inner neuron of ResNet-18, kept set of the P student (Minitron, by the trainer) | yes (gate part) |
| L1-2 carrier | every cut inner neuron and every kept inner neuron of the same block | partly: tests same-layer carriers, not the R1 parent |
| L1-4 stream | every CLIP stream direction, kept heads and MLP neurons (r3_joint keep_units) | yes (stream part) |
| L1-1 identity | every inner neuron of the R students, matched to A by per-photo max | yes, but the microscope cannot see coarsening (flat across sizes) |
| all L1 reads | block statistics over every neuron | NO named nodes (s3:69 and its R1 children were skipped). L2 must start with them |
