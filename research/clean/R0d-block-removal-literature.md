# R0d: Block removal and healing, literature notes

Status key: [V] = checked against the arXiv abstract this session. [M] = from my memory of the paper, numbers approximate, NOT re-checked. [I] = my inference.

## 1. Residual nets: deleting blocks

- Veit, Wilber, Belongie 2016, "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", arXiv 1605.06431. [V] Paths of many lengths; lesion study shows paths are ensemble-like. [M] Deleting one block from ResNet-110 / ResNet-200 on CIFAR-10 / ImageNet costs little (ImageNet ResNet-200: roughly a few points top-1 error for a single deletion), while deleting a downsampling block hurts a lot. Error grows smoothly with number of deleted blocks. VGG breaks completely when one layer is deleted. No healing in this paper: it is a no-retrain measurement.
- Stochastic depth, Huang et al. 2016, arXiv 1603.09382. [M] Blocks randomly dropped in training (survival prob linear 1.0 to 0.5). Network trained this way tolerates block deletion at test time far better. [I] Our ConvNeXt-Tiny was trained with stochastic depth (drop path 0.1 in timm recipe), so some tolerance is built in.
- Layer pruning for CNNs: DBP (Wang et al. 2019, "Dynamically Block Pruning"?? title uncertain), "Layer Pruning via Fusible Residual Convolutional Block" (Xu et al. 2020), "Layer-wise pruning" works. [M] They remove whole blocks and then fine-tune with the full labelled set for tens of epochs with KD from the original. They report accuracy back within about 1 point of the original at 30-50 percent FLOPs cut. These use the full ImageNet train set (1.28M images), not 18k. [I] So their recovery numbers do not transfer to our data budget.
- ViT / DeiT block dropping: [M] Several papers ("Depth pruning / Block skipping in ViT", e.g. "Layer Dropping" in DeiT-B) report removing 1-2 of 12 blocks costs 0.5-2 points top-1 without training, and 1 epoch or short fine-tune recovers most of it. Not verified; I did not find a primary number I trust.

## 2. LLM layer removal and healing

- Gromov et al. 2024, "The Unreasonable Ineffectiveness of the Deeper Layers", arXiv 2403.17887. [V] Up to about half of layers removable with minimal QA degradation; healing is QLoRA; single 40GB A100. [M] Layers chosen by angular distance between block input and output; blocks at the deep end (but not the final layer) are the most removable. Healing = QLoRA on C4 (about 164M tokens in my memory, one-two epochs). Healing mainly rescues the next-token loss (smooth), and MMLU/BoolQ stay flat until a sharp collapse at 45-55 percent pruned. Without healing, QA accuracy falls off a cliff earlier. Healing helps loss much more than QA accuracy near the threshold. Final layer is always kept.
- ShortGPT, Men et al. 2024, arXiv 2403.03853. [M] Block Influence (BI) = 1 - cosine similarity of block input and output hidden states. Remove lowest-BI layers (mostly deep ones) with no training. Llama-2-13B: remove about 27 percent of layers and keep around 90 percent of benchmark score; MMLU stays reasonable. No healing.
- LaCo, Yang et al. 2024, arXiv 2402.11187. [M] Merges later layers into an earlier layer by adding parameter differences, with no training. Prunes about 25-30 percent of Llama-2.
- SLEB, Song et al. 2024, arXiv 2402.09025. [M] Iterative block removal using perplexity on a small calibration set (128 samples); no training; removes 20 percent of blocks of OPT / Llama-2 with small perplexity rise. Notes: the removal choice is greedy, re-evaluated after each removal.
- BlockPruner, Zhong et al. 2024, arXiv 2406.10594. [M] Removes MHA and MLP sub-blocks separately, using perplexity. Finer than whole-block removal; MLP sub-blocks are often more important than attention ones. [I] Suggests we could try removing only half of block 16 (the MLP part has no attention analogue in ConvNeXt, but depthwise-conv versus pointwise-MLP split is the analogue).
- LLM-Streamline, Chen et al. 2024, arXiv 2403.19135. [V] Replaces the pruned layers with a lightweight network trained on a small set. [M] Replacement is a single FFN or transformer layer; trained by MSE between input and output hidden states of the removed span, on about 30k-ish samples (small), while the rest stays frozen. Reports about 90 percent of original accuracy at 25 percent pruning of Llama-2-7B. A linear (least-squares) replacement is shown as a weaker baseline in the paper [M].
- Shortened LLaMA, Kim et al. 2024, arXiv 2402.02834. [M] Depth pruning plus LoRA healing; then continued pretraining (CPT) on large data recovers much more. LoRA on a small set recovers a lot at moderate pruning (20 percent), little at heavy pruning (35 percent); the paper's finding was that LoRA alone is insufficient at high ratios and CPT with billions of tokens is needed.

Take-away [I]: In LLMs, healing recovers mostly LOSS, using 10^7-10^9 tokens, with low-rank adapters. Our setting (one block out of 18, 18k images) is comparable to the LOW end of this: it removes about 5.5 percent of depth.

## 3. Training-free / closed-form compensation

- FLAP, An et al. 2023, arXiv 2312.11983. [M] Structured pruning with a bias term added to compensate for removed channels: bias = mean output of the removed channels pushed through the next layer's weights. Fixed, closed-form, needs only calibration means. Gain is largest when pruned channels have large mean and low variance. [I] Direct analogue for us: add the MEAN of block 16's residual branch output (per channel, per spatial position or global) as a constant. A one-line test.
- OBS / OBC / SparseGPT / ZipLM: Frantar and Alistarh. OBC arXiv 2208.11580, SparseGPT arXiv 2301.00774, ZipLM arXiv 2302.04089 (structured). [M] After removing weights, update the REMAINING weights in the same layer by solving a least-squares problem on calibration activations (Hessian X^T X). 128-ish calibration sequences are enough. These work at the level of weights inside a layer, not whole-block removal. [I] The analogue for a whole block: refit the next block's input weights (or add a linear map) by least squares so that output on calibration data matches the original. This is the same Hessian-based idea with 18k images giving a very well-posed system for a layer with fewer than a few thousand inputs.
- REPAIR, Jordan et al. 2022, arXiv 2211.08403. [V] Averaging / interpolation causes variance collapse; rescale preactivations to restore statistics: 60-100 percent barrier reduction, 74 percent for ResNet50 ImageNet. [I] After block removal, activation statistics (mean and std per channel) in the later layers shift. Re-estimating them is cheap. ConvNeXt uses LayerNorm (no running stats), so the fix is a per-channel affine or a scalar rescale of the residual stream, fitted on calibration data.
- Model folding, Wang et al. 2025 [M, arXiv id not confirmed]. Merges similar neurons by clustering and fixes data-free statistics via a REPAIR-like step; data-free.
- Linear replacement of a removed layer by least squares: appears as baseline in LLM-Streamline [M] and in "Linear layer replacement" ideas. [I] The residual stream difference f(x) = block(x) is approximated by W x + b fitted by ridge regression on the layer-16 input. A cheap, one-shot test with a closed-form answer. Expect partial recovery only; whether LN and GELU make the removed branch non-linear is the question.
- Data-free distillation (DFKD, DeepInversion, Yin et al. 2020, arXiv 1912.08795): [M] reaches near-teacher accuracy on ImageNet with synthesized images but costs many GPU hours. Not needed since we have 18k real images.

## 4. Avoiding overfitting in distillation healing

[M] general practice, various sources:
- Match intermediate features (MSE on block outputs) in addition to or instead of logit KL. Gives dense per-position targets (a feature map has thousands of numbers per image versus 1000 logits), which strongly regularises. FitNets (arXiv 1412.6550), LLM-Streamline do this.
- Strong augmentation (random-resized-crop, flips, mixup / cutmix) during healing. Teacher recomputes on augmented images, so the data set is effectively infinite. Key for our train 0.027 / val 0.104 gap.
- Low-rank (LoRA) or very few trainable weights (a bias, a per-channel scale, LN affine). Gromov, Shortened LLaMA use this. [I] Our 4.7M-weight block 17 on 18k images is easily memorised.
- Freeze most of the network; train a few scalars; early stop on val.
- Teacher-generated data: more images through the teacher is free (no labels needed). [I] Augmentation multiplies data at zero label cost.

## 5. Why residual nets tolerate removal

- Veit 2016 [V/M]: ensemble of paths; removing one block removes only paths through it, a small fraction.
- Iterative refinement view: Greff, Srivastava, Schmidhuber 2016, "Highway and Residual Networks learn Unrolled Iterative Estimation", arXiv 1612.07771 [M]. Blocks make small corrections to a shared representation, so each is partly redundant.
- Jastrzebski et al. 2017 arXiv 1710.04773 [M]: residual blocks in the same stage do iterative refinement; early blocks of a stage do most of the work.
- LLMs: Gromov [M]: deep layers have high cosine similarity between input and output, so the residual contribution is small relative to the stream norm. Residual stream norm grows with depth in pre-LN transformers, so later blocks add relatively less (Pre-LN "curse of depth", Sun et al. 2025, arXiv 2502.05795 [M]). [I] ConvNeXt applies LayerScale (gamma, init 1e-6) and the stream norm also grows; check ratio ||block output|| / ||stream|| at block 16. If it is small, removal is mild; our KL 0.104 says it is not negligible at the output level.
- Block 17 healing limit [I]: if block 16 writes into stream directions that the head (and block 17) read through a nonlinear LayerNorm, block 17 cannot remap them with a bounded perturbation. Final LN rescales by the stream norm; the head sees direction only.

## Expectations (all [I], from the above)

1. Papers that heal well use 10^2-10^3 times more data than 18k images (1.28M images or 10^8 tokens), or work at ratios far from "one block of 18".
2. In LLMs, healing recovers loss smoothly but QA accuracy only partly. Recovering 7-10 percent of KL with an overfit 4.7M-weight block is consistent with "too little data, too many free weights", not evidence that recovery is impossible.
3. Cheap closed-form candidates, ordered by effort: (a) mean-bias compensation (FLAP); (b) per-channel / scalar rescale of the stream (REPAIR); (c) ridge-regression linear replacement of block 16, fitted on stream features; (d) least-squares refit of block 17's first pointwise layer (OBS style); (e) feature-level distillation with augmentation on a LoRA / bias-only block 17.
4. Realistic target: full recovery is unlikely; I would expect closed-form fixes to land in the 10-50 percent KL recovery range and a well-regularised multi-block feature-distillation to reach perhaps 50-80 percent. These ranges are guesses, not paper numbers.

## Caveats

I could not re-verify most numbers (only abstracts of 2403.17887, 2403.19135, 2211.08403, 1605.06431 were fetched). Titles / ids marked [M] should be checked before citing. DBP citation is unsure. Model folding arXiv id not confirmed.

## Verified by the main session (2026-10-22): PRACTISE (Wang & Wu, arXiv 2303.00972; earlier 2202.07861)
Read from the full HTML (sections 1-5 of the paper text; first 100k characters).
- Setting: drop whole blocks from ImageNet CNNs, recover with tiny unlabeled sets (50-1000 images).
- Recovery: (a) recoverability = min over a 1x1-conv ADAPTOR inserted at the removed block, rest frozen, of the L2 feature
  error; blocks with the smallest recoverability / latency gain are dropped; (b) final recovery trains ALL remaining
  parameters with feature mimicking (L2 between original and pruned features before global pooling, following MiR), SGD
  lr .02, /10 at 40% and 80%, 2000 iterations, batch 64, no labels.
- ResNet-34 (73.31%), 16.3% latency cut: block drop + plain fine-tune 66.5 / 68.6 (50 / 500 images); PRACTISE 70.3 / 72.5.
  22.1% cut: 60.6 / 65.0 vs 68.0 / 71.8. Few-shot baselines at 50 images: KD 44.5, MiR 64.1.
- ResNet-50 (76.1%), 21% cut, 500 images: 74.58 (ImageNet), 72.68 with synthesized data-free images; all data 75.24.
- "Finetune convexity hypothesis": the loss along the line from the raw to the pruned model is smooth and roughly convex,
  more so for dropped blocks than for pruned filters; offered as why few-shot fine-tuning does not overfit.
- One dropped block before recovery can be large (ResNet-34 layer1.1: top-1 error 26.98% -> 54.3%).
Expectation for us: with feature mimicking and ALL remaining layers trainable, most of the loss is recoverable with
hundreds to thousands of images. Our block-17-only, logits-only reference (10%) is far below what the literature reaches.
Related, not read in full: CURL (1911.08114, residual-block pruning with limited data), DC-ViT (2403.18708), CORP
(closed-form compensation with a small calibration set, DeiT), UP-DeiT (2111.15127; block deletion + KD 50 epochs).
