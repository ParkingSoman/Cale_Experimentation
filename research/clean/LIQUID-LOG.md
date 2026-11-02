# Liquid log: shrinking attention, head size and stream width

Plan: `~/.claude/plans/okay-right-now-i-crystalline-platypus.md` (approved 2026-10-20). Code: `research/clean/liquid/`.
Results: `results/liquid-*.json`. Dev layers 12, 16, 22 only. Dev text: prose test blocks 0-39, code chars 0-400k.
Calibration: 64 prose train blocks + 64 code blocks (chars 400k+), the same as rcf_v1.
Currency: dCE (added next-word loss) per share of the layer's weights removed (layer = 3,538,944 weights).

## Methods check (2026-10-20)

| Step | Field method | What I used |
|---|---|---|
| S1 heads | Head importance by output (Michel et al. 2019 style) | Write energy of each head on calibration text |
| S2 QK | MHA2MLA S_2-norm (arXiv 2502.14837): per-head norm of 2D rotary chunks of q and k | Sum over the group's query heads x key norm (keys are shared). Known bias: favors low-frequency pairs (MHA2MLA-VLM, arXiv 2601.11464, prefers an MKL score). |
| S2 VO | Low-rank value projection (Palu style), whitened SVD | Whitened SVD of the group's output weights, one basis per key/value group |
| S3/S4 | SliceGPT (arXiv 2401.15024): fold norm scales, rotate by eigenvectors of the norm outputs' second moment, delete rows/columns | Same, with one GLOBAL basis (SliceGPT uses one per layer plus shortcut matrices) |

## S0 harness

Identity edit dCE = 0.0 exactly. Baseline CE: prose 3.004, code 1.800. One dev score = ~5 s.

## Log

### S1 query heads (2026-10-20, results/liquid-s1-heads.json, s1_heads.py)

Pre-registered gate: HEAD-RCF beats CUT-LS on all 3 layers, prose and code, 95% interval clear of 0. **FAIL by the letter.**
dCE prose / code. r = heads removed per key/value group (r1: 6 heads left, 6.25% of layer weights; r2: 3 left, 12.5%).

| L, r | cut | cut_ls (control) | merge_ls | head_rcf | MLP cut_ls_mix same bytes | MLP rcf_v1 same bytes |
|---|---|---|---|---|---|---|
| 12 r1 | .0284/.0124 | .0073/.0064 | .0050/.0046 | .0062/.0046 | .0017/.0013 | .0018/.0010 |
| 12 r2 | .0606/.0435 | .0174/.0124 | .0142/.0118 | .0151/.0106 | .0034/.0022 | .0076/.0043 |
| 16 r1 | .0151/.0157 | .0027/.0028 | .0052/.0030 | .0024/.0036 | .0022/.0009 | .0016/.0016 |
| 16 r2 | .0159/.0268 | .0070/.0079 | .0212/.0114 | .0052/.0070 | .0043/.0020 | .0033/.0019 |
| 22 r1 | .0178/.0136 | .0079/.0049 | .0107/.0073 | .0074/.0048 | .0026/.0004 | .0026/.0001 |
| 22 r2 | .0474/.0315 | .0141/.0100 | .0177/.0135 | .0126/.0096 | .0048/.0021 | .0045/.0011 |

- head_rcf - cut_ls: prose clear below 0 at r2 on all 3 layers (-.0015 to -.0023) and at r1 on L12. Code clear only on L12.
  Gain = 10-25% of the control's damage. The L-1 MLP route adds little for heads.
- merge_ls beats cut_ls only on L12. merge_sum (q averaged, o summed, no refit) is poor everywhere.
- **Main finding: whole query heads cost 2-4x more loss per byte than MLP neurons** (cut_ls - MLP cut_ls_mix clear above 0 in
  11 of 12 cells). In one ranked list, MLP neurons go long before whole heads.
- Plain reading: a head moves information between words. A route at the current word (L-1 neurons) cannot redo that move.
  The kept heads of the same group can (they read the same values), and the refit already uses them.

### S2 head-size dial (2026-10-20, results/liquid-s2-headsize.json, s2_headsize.py)

Identity check (keep 1.0): all arms dCE = 0.0000. One head dim per group = 2,304 weights on either side.
Pre-registered gate: at equal weights removed, the head-size dial beats whole-head CUT-LS (S1) on >= 2 of 3 layers.

At 6.25% of layer weights removed (prose / code):

| Layer | whole heads cut_ls (S1 r1) | both sides, keep 75% | VO only keep 50% (vo_svd_ls) | MLP cut_ls_mix same bytes |
|---|---|---|---|---|
| 12 | .0073/.0064 | .0032/.0037 | .0031/.0034 | .0017/.0013 |
| 16 | .0027/.0028 | .0019/.0022 | .0053/.0042 | .0022/.0009 |
| 22 | .0079/.0049 | .0036/.0036 | .0041/.0051 | .0026/.0004 |

At 12.5% (S1 r2 vs both sides keep 50%): L12 .0174/.0124 vs .0074/.0095 (better); L16 .0070/.0079 vs .0084/.0064
(mixed); L22 .0141/.0100 vs .0116/.0116 (mixed).

**Gate: PASS at 6.25% (3 of 3 layers, about half the cost of whole heads). Mixed at 12.5%.**

Other findings:
- The query-key side is nearly free at first: dropping 25% of rotary pairs (3.1% of layer weights) costs L16 -.0008/+.0014,
  L22 .0009/.0033, L12 .0034/.0065. It gets steep below 50%.
- qk_wnorm (pair score from weight norms only) is far worse than the activation score (qk_act) in all 9 cells
  (e.g. L16 keep 75%: +.134 vs -.001). Rotary pairs must be ranked on real activations.
- Flexible vs per-size refit: vo_svd (one nested order, no refit) is within .0005 of vo_svd_ls at keep 75%, and within
  .001-.008 at keep 25%. So the value side is nearly a free nested dial.
- vo_svd_ls beats the norm-ranked control mainly at keep 25% (all 6 cells clear) and on L22.
- Attention is still 1.5-3x more costly per byte than MLP neurons at the same share (except L16 prose). In one ranked list,
  MLP neurons go first, then the cheap first steps of head size, then the rest.

### S3 stream subspace at one layer (2026-10-20, results/liquid-s3-stream.json, s3_stream.py)

Identity check (d = 576, all bases): dCE = 0.0000. Saliency coordinates found from data (pooled E[x_hat_i^2] > 20x
median): 17, 230, 247, 260, 308, 371, 507 (includes 17, 247, 507 from the coarsening track). Kept first in every basis.
Pre-registered gate: best basis costs less per byte than whole-head CUT-LS (S1) AND than RCF neurons at the same layer.
**FAIL.** It is 3-6x more costly than RCF neurons in all 9 cells. Against whole heads at 12.5%: better on L12, tie on L16,
worse on L22.

dCE prose / code (removed share = 1 - d/576 of the layer's weights):

| L, d (removed) | pca_global | read_glob_ls | write_glob | coord (control) | pca_local (not free) | MLP rcf_v1 same bytes |
|---|---|---|---|---|---|---|
| 12, 504 (12.5%) | .0170/.0121 | .0109/.0085 | .0366/.0339 | .0211/.0221 | .0107/.0115 | .0076/.0043 |
| 12, 432 (25%) | .0422/.0267 | .0329/.0197 | .0750/.0641 | .0515/.0482 | .0270/.0222 | .0123/.0072 |
| 12, 288 (50%) | .0824/.0611 | .0776/.0611 | .1171/.0966 | .0992/.0801 | .0701/.0572 | .0280/.0174 |
| 16, 504 | .0212/.0152 | .0074/.0059 | .0439/.0397 | .0313/.0262 | .0167/.0110 | .0033/.0019 |
| 16, 432 | .0436/.0336 | .0264/.0196 | .0824/.0766 | .0696/.0580 | .0367/.0268 | .0088/.0051 |
| 16, 288 | .1150/.0957 | .0680/.0518 | .1711/.1447 | .1709/.1478 | .0784/.0637 | .0235/.0149 |
| 22, 504 | .0273/.0190 | .0391/.0256 | .0447/.0419 | .0411/.0307 | .0253/.0146 | .0045/.0011 |
| 22, 432 | .0600/.0377 | .0798/.0573 | .1047/.0857 | .0771/.0648 | .0608/.0379 | .0140/.0038 |
| 22, 288 | .1347/.0886 | .1478/.1101 | .1942/.1520 | .1522/.1197 | .1313/.0841 | .0385/.0139 |

- The essay's read basis (directions neurons read from, + LS read refit) is the best free basis at L12 and L16, beating
  SliceGPT-style PCA by up to 40%. PCA is best at L22. The write basis is the worst everywhere.
- A global basis costs little at L22 (pca_global ~ pca_local) but more at L16 (d288: .115 vs .078).
- Plain reading: one 576-wide slot narrowed at one layer removes a share of EVERY reader at once, untargeted. MLP removal
  can pick its weakest neurons. Per weight, untargeted width is costly. The stream's case rests on S4 (it also shrinks
  the word table and makes every layer smaller together).

### S2b flexible vs fixed, head size (2026-10-20, results/liquid-s2b-fixed.json, s2b_fixed_ref.py)

Standing check (memory "flexible vs fixed"). Both head sides at keep 50% (12.5% of layer weights). Fixed = start from the
per-size LS, then Adam on whole-model KL for 60 s (only 42-52 steps on the shared Mac: a WEAK reference).

| Layer | flexible (nested, no refit) | ls (per size) | fixed (60 s) | flexible - fixed (95%) |
|---|---|---|---|---|
| 12 | .0133/.0192 | .0074/.0095 | .0038/.0071 | prose +.0096 [.0054,.0144], code +.0121 [.0085,.0159] |
| 16 | .0081/.0117 | .0084/.0064 | .0068/.0046 | prose +.0013 (ns), code +.0071 [.0052,.0092] |
| 22 | .0121/.0155 | .0116/.0116 | .0098/.0098 | prose +.0023 (ns), code +.0058 [.0040,.0075] |

Reading: fixed beats flexible, so gains are left on the table, mostly on code and at L12. Even a 50-step fit finds them,
so the reference is far from a ceiling. Rule: find which part misses them. Lead: the query-key side has no compensation
at all (pairs are dropped, nothing refits), while the value side's free order is close to its refit (S2).

### S3b essay fix for the stream basis (2026-10-20, results/liquid-s3b-essay.json)

Essay reading of the S3 fail: "the directions neurons write to and read from are a good place to start", but a plain
weight Gram counts every reader the same, also readers that never fire. Fix tested: weight each reader by its real
activity on calibration text (read_act), and a read + write version (rw_act). Same harness, no MLP reference rerun.

| L, d | pca_global | read_glob_ls (S3 best) | read_act | read_act_ls | rw_act | rw_act_ls |
|---|---|---|---|---|---|---|
| 12, 504 | .0170/.0121 | .0109/.0085 | .0139/.0092 | **.0097/.0070** | .0185/.0195 | .0233/.0174 |
| 12, 432 | .0422/.0267 | .0329/.0197 | .0407/.0276 | .0306/.0228 | .0426/.0411 | .0479/.0392 |
| 12, 288 | .0824/.0611 | .0776/.0611 | .0904/.0699 | .0862/.0613 | .0916/.0754 | .1028/.0715 |
| 16, 504 | .0212/.0152 | .0074/.0059 | .0101/.0082 | **.0060/.0070** | .0202/.0165 | .0150/.0129 |
| 16, 432 | .0436/.0336 | .0264/.0196 | .0294/.0263 | **.0205/.0173** | .0401/.0338 | .0331/.0278 |
| 16, 288 | .1150/.0957 | .0680/.0518 | .1031/.0711 | **.0645/.0488** | .0820/.0641 | .0777/.0553 |
| 22, 504 | **.0273/.0190** | .0391/.0256 | .0325/.0238 | .0403/.0252 | .0407/.0303 | .0465/.0252 |
| 22, 432 | **.0600/.0377** | .0798/.0573 | .0726/.0621 | .0965/.0640 | .0846/.0632 | .0908/.0602 |
| 22, 288 | **.1347/.0886** | .1478/.1101 | .1519/.1255 | .1604/.1126 | .1504/.1197 | .1571/.1195 |

- read_act_ls beats read_glob_ls a little at L16 (all 3 sizes, prose clear) and at L12 d504/d432. Small gains (.001-.006).
- **Gate still FAIL.** The best stream arm is still 1.3-17x more costly per byte than RCF neurons (about 2-4x in most
  cells, worst at L22: d504 6x/17x).
- Adding writes hurts in 6 of 9 cells (tie or better only at d288). This agrees with S3: the write basis is worst.
- At L22 the read refit HURTS in every read arm, the reverse of L12/L16. PCA stays best at L22. Not explained.

### S2c essay fix for the value side (2026-10-20, results/liquid-s2c-vo-read.json, s2c_vo_read.py)

Essay reading: attention is understood "in terms of neurons", so rank a head's value directions by what later neurons
READ, not only by what the head writes. vo_read = whitened SVD in the metric of activity-weighted reads by all later
neurons. Control: vo_svd (plain whitened SVD).

| L, keep | vo_svd | vo_read | vo_svd_ls | vo_read_ls |
|---|---|---|---|---|
| 12, 50% | .0039/.0048 | .0040/.0050 | .0031/.0034 | .0032/.0032 |
| 12, 25% | .0113/.0079 | .0106/.0087 | .0088/.0065 | .0085/.0069 |
| 16, 50% | .0056/.0057 | .0058/.0067 | .0053/.0042 | .0054/.0047 |
| 16, 25% | .0254/.0276 | .0281/.0342 | .0233/.0194 | .0247/.0253 |
| 22, 50% | .0052/.0055 | .0045/.0050 | .0041/.0051 | .0031/.0045 |
| 22, 25% | .0148/.0132 | .0146/.0142 | .0124/.0130 | .0130/.0129 |

**Null.** All differences are within about .001 except L16 keep 25% (vo_read clearly WORSE, both texts) and L22 keep 50%
_ls prose (vo_read slightly better). Reading: the plain whitened SVD of what the group writes already holds what later
neurons need. The essay's lead is satisfied by the existing method. Our metric added noise, not information.

### S5 one ranked list at one layer (2026-10-20, results/liquid-s5-mix.json, s5_mix.py)

Per budget, every (qk keep, vo keep) on a small grid, the rest from MLP neurons (cut_ls). Allocation chosen on a separate
selection set (prose train 200-223 + code chars 1.6M+), then scored on dev. Check: rcf_v1.build adds its own code
calibration blocks (chars 400k+), so the S.cal[:64] call gives it the same mixed calibration. The reference is fair.
Pre-registered gate: the chosen mix beats MLP-only RCF at equal bytes on all 3 dev layers. **NOT ESTABLISHED** (advisor
review): the mix's MLP part is cut_ls, the reference is rcf_v1, so the comparison mixes two MLP methods. The fair
comparison is mix vs mlp_only_cut. The plan's composition check (RCF fitted last on top of the head edit) was not run:
RCF's fold targets self_attn.o_proj, which the head edit replaces with an AttnGuard. Open item.

| L, budget | chosen (qk, vo) | mix | MLP-only cut_ls | MLP-only rcf_v1 |
|---|---|---|---|---|
| 12, 10% | (1, 1) | .0030/.0020 | same | .0070/.0040 |
| 12, 25% | (1, 1) | .0103/.0055 | same | .0123/.0072 |
| 12, 40% | (1, 1) | .0185/.0107 | same | .0189/.0111 |
| 16, 10% | (.75, 1) | .0016/.0024 | .0036/.0010 | .0033/.0015 |
| 16, 25% | (.75, 1) | .0078/.0061 | .0103/.0045 | .0088/.0051 |
| 16, 40% | (.75, .75) | .0153/.0086 | .0196/.0096 | .0174/.0103 |
| 22, 10% | (1, 1) | .0040/.0013 | same | .0038/.0005 |
| 22, 25% | (1, 1) | .0147/.0052 | same | .0140/.0038 |
| 22, 40% | (.75, .75) | .0268/.0137 | .0333/.0116 | .0283/.0083 |

- The selection picked NO head shrink at L12, and none at L22 below 40%. MLP neurons go first there.
- Where it picked head size (L16 all budgets, L22 40%), prose improves clearly (-.002 to -.0065 vs cut_ls), but code gets
  worse in 3 of 4 cells. The selection score sums prose and code means, so prose dominates. This is a flaw of the
  selection objective, not a finding about code.
- Side note for the RCF session: at L12, plain cut_ls beats rcf_v1 at 10% and 25% (prose clear at 10%).
- Ranked order per layer: MLP neurons first. The first 25% of query-key pairs joins at L16 early and at L22 near 40%.
  Value-side cuts join only near 40%. Whole heads and stream width never enter.

### S2d which head side misses the gains (2026-10-20, results/liquid-s2d-f1.json, s2d_f1.py)

Follow-up to S2b (one fix of the step). Both sides keep 50% (12.5% of layer weights). Start from the per-size LS state.
15 s Adam on the attention block's own output error (32 cached calibration blocks, position 0 excluded), training only
one side's kept weights. Zero extra storage. First run had no trainable weights (`unfreeze(keys=...)` matches local
parameter names, not submodules). Fixed, then the fit error moved in every arm (e.g. L22 f1_all .714 -> .306).

| Layer | ls | f1_qk | f1_vo | f1_all | S2b fixed (60 s, whole-model KL) |
|---|---|---|---|---|---|
| 12 | .0074/.0095 | .0067/.0071 | .0096/.0090 | .0062/.0053 | .0038/.0071 |
| 16 | .0084/.0064 | .0065/.0051 | .0088/.0067 | .0061/.0047 | .0068/.0046 |
| 22 | .0116/.0116 | .0094/.0085 | .0125/.0119 | .0084/.0071 | .0098/.0098 |

- **The query-key side holds the missing gains.** f1_qk - ls is below 0 on all 3 layers for code (95% clear) and on L16,
  L22 for prose. f1_vo gives nothing (L12 prose is worse, +.0022 clear).
- f1_all (both sides, 15 s, local) matches or beats the 60 s whole-model fixed reference on L16 and L22. L12 prose is
  the exception (.0062 vs .0038).
- Essay reading: a dropped rotary pair removes part of every query-key score. Nothing compensates it, because the plain
  LS refit only touches the output weights. The routes that can capture the loss are the kept query and key pairs.
- Cost: the f1 weights are per size, not nested. A nested head-size dial needs the f1 delta per size stored, or a
  closed-form query refit that depends on size in a cheap way. Open item.

### S4 stream width, whole model (2026-10-20, results/liquid-s4-stream-whole.json + liquid-s4b-check.json)

One global basis, every layer's reads and writes restricted to it, word table rows projected. Removed share = 1 - d/576
of every matrix (word table too). Pre-registered gate: at 90% width, dCE per byte below whole-model 9-to-6 CUT-LS heads.
**FAIL. Stream knob paused for the 135M, as the plan says.**

S4b checks (fail protocol): identity at d = 576 gives dCE 0.0000 (with and without the fix below). Fix: S4 left the
final norm at full width, so its output left the subspace and the full table read it. That is really a second,
untied output table. S4b projects the final norm's output again, which gives the tied model (one extra d' x d' matrix).
The tied numbers are the ones to use.

| Arm | Removed | prose / code | per 10% removed |
|---|---|---|---|
| pca_global, tied head (S4b) | 10.1% | .641/.436 | .63/.43 |
| pca_global, tied head (S4b) | 25% | 2.19/2.11 | .88/.84 |
| pca_global, untied (S4) | 10.1% | .614/.394 | |
| coord (control) | 10.1% | .811/.695 | |
| read_glob / read_act | 10.1% | 1.55/1.29 / 1.28/1.13 | |
| pca_global_ls | 10.1% | .662/.496 | |
| blocks6_pca / blocks6_read_act | 10.1% | .793/.606 / 3.95/5.16 | |
| MLP cut_ls_mix every layer (K 1330) | 10.1% of layers (8.0% of all) | .305/.241 | .30/.24 |
| heads 9 to 6 cut_ls every layer | 6.25% of layers (4.9% of all) | .184/.164 | .29/.26 |

- Per byte of layer weights the stream costs about 2 times more than both controls at 90% width. Per byte of ALL weights
  (the stream also shrinks the word table) it is about 1.7 times. At 75% width it is much worse.
- The read bases, best at ONE layer (S3), collapse on the whole model (2-3 times worse than PCA). Reading: they describe
  what layer readers want, not what the stream carries. The output head is the biggest reader and is not in their Gram.
  Open fix (not run, the plan says pause): add the head's reads to the read Gram.
- The LS read refit hurts on the whole model. It rebuilds dropped parts from the intact model's statistics, but after
  many sliced layers those statistics no longer hold. SliceGPT fits layer by layer on the sliced model. Not tried.
- Per-block bases are worse than one global basis. Each block boundary throws away what earlier blocks wrote outside the
  new subspace, for later readers. This agrees with the essay: subspaces are followed across layers, not reset.
- Compounding: MLP cut at 10% in every layer costs about 3 times the sum of its one-layer costs (.0035 x 30 = .10 vs .30).
  Whole heads cost about 1 times (.006 x 30 = .18 vs .18). Per-layer ranks do not carry over equally to the whole model.
- Two control runs took 715 s and 614 s, over the 10-minute rule without a warning first. Noted.

Note (S2d storage, plan's "log extra numbers per size"): the query-key fit changes the kept q and k rows. At keep 50%
that is (9 + 3) heads x 32 x 576 = 221,184 numbers per layer per stored size (6.25% of a layer, 25% of attention).
Gap a nested method must close (flexible minus f1_all): L12 .0071/.0139, L16 .0020/.0070, L22 .0037/.0084.
