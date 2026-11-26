# CLIP-KD: a public teacher and a distilled student of the same shape

Opened 2026-11-25. Plan: `~/.claude/plans/right-now-we-have-concurrent-volcano-of-a9da21f6-3c9a-4a53-8bad-868c04c92a58.md`.
Calculation work that uses these files: `../CALC.md`.

## 1. Why

Shonu asked for a public image model with a residual stream, a full model, and smaller models
distilled from it, so the calculation has a known answer without our own training runs.
Sizes could differ from 95 / 50 / 5%. The architecture had to match. Shonu chose CLIP-KD.

Search result: no public family has one teacher with several same-depth students.
Runner-up: Proteus (DINOv2 teachers -> Ti / S / B / L from random). Each size has a different
teacher and depth. Ruled out: DINOv2, DINOv3, DeiT (CNN teacher), Scala, HydraViT, MatFormer
(nested slices, not trained small from the start).

## 2. The three models

Source: github.com/winycg/CLIP-KD, release CLIP-KDv0.1 (CVPR 2024). Files in `weights/` (git-ignored).

| Name | File | Vision blocks | Vision width | MLP width | Output | Vision params | md5 |
|---|---|---|---|---|---|---|---|
| teacher | `ViT_B_16-laion400m_e32.pt` | 12 | 768 | 3072 | 512 | 86.19M | `07283adc5c17899f2ed22d82b563c54b` |
| kd | `ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.pt` | 12 | 192 | 768 | 512 | 5.62M | `cb967b0426c746976d6ea1b4afeb5a1c` |
| baseline | `ViT_T_16_cc3m_12m_ep32.pt` | 12 | 192 | 768 | 512 | 5.62M | `a817041981a523c1b545e03c82fc9180` |

Training logs (`.log`) for kd and baseline are next to the weights.

Facts from the CLIP-KD code (`script/ViT_B_16_KD_Laion/ViT_T_16_kd.sh`, `src/training/main_kd.py`, `train.py`):

1. kd starts from random (`--pretrained` is empty). The teacher is frozen.
   kd's teacher is exactly our teacher file: the script names `vit_b_16-laion400m_e32-55e67d44.pt`,
   and the sha256 of our file starts with `55e67d44`.
2. kd's loss is NOT pure distillation (`src/open_clip/loss.py`). It is the sum of:
   the caption contrastive loss; ckd (weight 1) = KL between the student's and the teacher's
   image-text similarity logits; icl (weight 1) = contrastive loss across student and teacher
   embeddings; fd (weight 2000) = MSE between the normalized student and teacher 512 embeddings,
   image plus text.
3. Students train on CC3M + CC12M for 32 epochs. The teacher trained on Laion-400M.
4. baseline = the same student recipe with no teacher terms.
5. One seed per model. One student size.
6. The text towers also differ: teacher 512 wide, students 384 wide.

## 3. Zero-shot scores (step 5)

Code: `score_clip.py`. One torch job on the Mac (MPS). open_clip 4.2.2 + ftfy are installed in
`pylib/` (git-ignored), with no change to a shared environment. Run with
`curves/.venv/bin/python -I distill/clipkd/score_clip.py --per_class 5` from `research/clean`.

Each model scores 1,000 classes with its OWN text tower (80 OpenAI templates per class) and its own
logit scale. KL = KL(teacher ‖ student) on the 1,000 class probabilities.

| Model | Logit scale | Top-1 (5,000 photos) | README top-1 (full ImageNet val) | KL to teacher | Agrees with teacher's top-1 |
|---|---|---|---|---|---|
| teacher | 100.0 | 66.2% | 67.1% | 0 | 100% |
| kd | 56.57 | 41.1% | 42.6% | 1.637 | 47.5% |
| baseline | 74.56 | 27.8% | 30.6% | 3.800 | 30.9% |

Smoke run (1 photo per class, 1,000 photos): teacher 67.1%, kd 40.1%, baseline 28.8%.
The order teacher > kd > baseline holds in top-1 and in KL. The training logs give the same
final numbers as the README (kd 42.57%, baseline 30.55%). Our numbers are 0.9 to 2.8 points
lower. The cause is not tested. Sampling noise at 5,000 photos is about 0.7 points. One possible
cause: our photos are fixed 224 crops, not the CLIP resize-and-crop of the full val set.

**Target for a calculated ViT-T.** A calculated student has no text tower. So it is scored with the
TEACHER's classifier at the teacher's scale (100). Same 5,000 photos, numpy only, from `out/`:

| Model | Median cosine to teacher's 512 output | KL to teacher | Top-1 | Agrees with teacher's top-1 |
|---|---|---|---|---|
| kd (THE TARGET) | .778 | 1.719 | 40.9% | 49.0% |
| baseline | .011 | 11.347 | 0.1% | 0.1% |

The kd student writes into the teacher's output space. The baseline does not.

Run time: teacher 300 s, each student about 145 s. Building the text classifier takes most of it.

Outputs in `out/` (git-ignored): `clip_scores.json`, `score_clip_pc5.log`, and per model
`<name>_test.npz` with fields `emb` (image tower output before normalising), `logits`, `labels`,
`photo_ids`, `logit_scale`, `text_classifier`. `CALC.md` and `eval_fast.py` read these field names.
Do not rename them.

## 4. Photo use

| Item | Photo use |
|---|---|
| Scores in section 3 | reading photos, test split (odd positions in each class), first 5 per class = 5,000. Scoring only. Do not tune a calculation on these files. Tuning needs a held-split run of `score_clip.py` (not made yet). |
| Smoke run | first 1 per class of the same split = 1,000. Scoring only. |
| `out/teacher_held.npz`, `out/kd_held.npz` (made by the CALC session with `eval_fast.py`, 2026-11-25) | held photos = fit positions 8 and 9 per class = 2,000, stored class by class, scored with the TEACHER's classifier at scale 100. Teacher 64.6%, kd 39.2%. Use these for any tuning. |

## 5. Limits for the calculation

1. One student size (25% width, 6.5% of vision params). No 95% or 50% point.
2. One seed. The seed spread is unknown.
3. kd also learned from captions, so it is not a pure KL optimum of the teacher.
4. The student stream is 192 wide. A weight comparison with the teacher needs a stream map first.
   kd's 512 output lies in the teacher's output space (median cosine .778 on 5,000 photos,
   section 3). baseline's does not (.011).
5. Licence: the open_clip code is MIT. The CLIP-KD weights licence is not yet checked. Check it
   before any public claim.

## 6. Block hooks (open_clip 4.2.2)

| Read | Module |
|---|---|
| Stream after block i | `visual.transformer.resblocks[i]` (forward hook output) |
| MLP neuron firing in block i | `visual.transformer.resblocks[i].mlp.gelu` (exact GELU) |

Check the token axis on first use. Expected (N, 197, D), class token at index 0.
