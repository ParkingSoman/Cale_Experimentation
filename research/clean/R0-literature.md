# R0: Literature review for the "Clean" track

Date: 2026-10-19. Method: web search and page fetch. Each paper below was found on arXiv or a venue page.
Numbers come only from the abstracts or search summaries I read. I did not open every full paper.
Where I did not read the full paper, the row says "abstract only".

Our setting: SmolLM2-135M, SwiGLU MLP, 1536 neurons per layer. We want a narrower layer that keeps coarse concepts.

## Topic 1. Merging neurons

| Paper | What it did | What it found | What it means for us |
|---|---|---|---|
| Neuron Merging: Compensating for Pruned Neurons. Kim, Kim, Park, Jeon. NeurIPS 2020. arXiv 2010.13160 | It splits a layer's weights into a smaller weight set plus a scaling matrix. The scaling matrix folds the removed neurons into the kept ones. It needs no data. It uses cosine similarity between neurons. | VGG-16 on CIFAR-10 reached 93.16% accuracy with 64% of parameters removed and no fine-tuning (search summary). A reviewer noted the method needs ReLU. | This is the closest match to "keep a neuron and refit". It works because ReLU is positive-homogeneous. SwiGLU is not. This may explain why our merge rules fail. |
| Forget the Data and Fine-Tuning! Just Fold the Network to Compress (model folding). Wang, Sikic, Thiele, Saukh. ICLR 2025. arXiv 2502.10216 | It clusters similar neurons with k-means and merges each cluster. It needs no data and no fine-tuning. It adds a fix so activation variance does not collapse. | Tested on ResNet18 and LLaMA-7B. It beats other data-free methods, mostly at high sparsity. The abstract gives no numbers. | It is a direct data-free merge baseline on an LLM. It clusters by weight similarity. Our hand-picked groups are a similar test. Check how it handles the gate. NOT READ in full. |
| DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging. Verma, Murray, Duh. 2025. arXiv 2507.04517 | It moves signal from all neurons onto a smaller kept set. It uses optimal transport over activation signatures. A pruning rule picks which neurons stay. | It beats simple and strong width-pruning baselines (abstract only). It needs example tokens to compute activations, so it is NOT data-free. | It is the nearest work to "merge neurons in an LLM MLP". It is a linear re-mix of neurons, not a new neuron. That matches our "keep and refit" result. I could not confirm how it treats SwiGLU gate and up parts. |
| ZipIt! Merging Models from Different Tasks without Training. Stoica, Bolya, Bjorner, Ramesh, Hearn, Hoffman. ICLR 2024. arXiv 2305.03053 | It merges two models. It also allows merging features inside one model. | It reports 20 to 60% better than earlier merge work. | Its "merge within one model" step is the only part that matters to us. It supports the idea that a merged feature needs a correlation test first. Low relevance otherwise. |
| Model Fusion via Optimal Transport. Singh, Jaggi. NeurIPS 2020. arXiv 1910.05653 | It aligns neurons across two models with optimal transport. Then it averages. | Appeared in search only. Not read. | It is the root of OT-based merging. Background only. |

Gap I found: I did not find a paper that merges SwiGLU neurons without training and shows it works. One 2026 paper (arXiv 2607.22583, not read) says a gated MLP forces gate and up to be handled together. That fits our claim that one SwiGLU neuron cannot express a sum of two. I found no direct proof of that claim.

## Topic 2. Coarse to fine features

| Paper | What it did | What it found | What it means for us |
|---|---|---|---|
| Towards Monosemanticity. Bricken et al. Anthropic, Transformer Circuits Thread, 2023 (Oct 4) | It trained a sparse autoencoder on a 1-layer transformer. | It split a 512-neuron layer into more than 4000 features (search summary). The paper also shows "feature splitting": a larger dictionary splits one feature into finer ones. I did not open the splitting section. | Splitting is the same idea as our "right family, wrong member". A wide layer holds the fine children. A narrow layer should hold the parent. |
| A is for Absorption: Studying Feature Splitting and Absorption in SAEs. Chanin et al. NeurIPS 2025 (workshop 2024). arXiv 2409.14507 | It tested hundreds of SAEs on a first-letter task with known labels. | Parent features can stop firing when a child feature takes over ("absorption"). Changing SAE width or sparsity did not fix it. | A coarse unit does not equal the sum of its children. A parent can have holes where a child fires. Our trained narrow neurons may show the same holes. |
| Learning Multi-Level Features with Matryoshka SAEs. Bussmann, Nabeshima, Karvonen, Nanda. ICML 2025. arXiv 2503.17547 | It trains nested dictionaries at once. Each small prefix must rebuild the input alone. | On Gemma-2-2B and TinyStories: less absorption, better sparse probing and concept erasure. Reconstruction is a little worse than standard SAEs. Downstream loss is similar. I could not find a separate Nabeshima paper. NOT VERIFIED as a separate citation. | It is the closest recipe for "coarse neurons first, fine neurons after". We can train the narrow layer as a nested prefix. It also shows nesting costs a little reconstruction quality. |
| Matryoshka Representation Learning. Kusupati et al. NeurIPS 2022. arXiv 2205.13147 | It trains one vector whose first d dimensions each work alone. | Up to 14x smaller embeddings on ImageNet-1K with the same accuracy (search summary). | Same nesting idea. Source of the term "Matryoshka". |
| MatFormer: Nested Transformer for Elastic Inference. Devvrit, Kudugunta, Kusupati et al. NeurIPS 2024. arXiv 2310.07707 | It nests the first m neurons of each FFN. It trains all sizes together. | It uses 4 sizes (ratios 0.5, 1, 2, 4). A 850M model yields sub-models from 582M to 850M. Gemma 3n reportedly uses it. | It is the proof that nested neuron prefixes work in an FFN when trained. It differs from our no-training goal. It also shows trained prefixes are not merges of the big neurons. |

## Topic 3. Comparing units across two models

| Paper | What it did | What it found | What it means for us |
|---|---|---|---|
| Convergent Learning: Do different neural networks learn the same representations? Li, Yosinski, Clune, Lipson, Hopcroft. ICLR 2016. arXiv 1511.07543 | It compared neurons of separately trained nets. It used one-to-one matching, sparse prediction (one-to-many), and spectral clustering (many-to-many). | Some features match across nets and some do not. Early and late layers match more than middle layers (alphaXiv summary). | This is the exact tool for our experiment. Regress each new neuron on old neurons with a sparse penalty. Expect that some neurons will not be explained. |
| SVCCA. Raghu, Gilmer, Yosinski, Sohl-Dickstein. NeurIPS 2017. arXiv 1706.05806 | It compares layers after removing small directions. It then finds shared directions. | It works across layers and nets. Numbers not read. | It compares whole subspaces, not single neurons. Use it as a second view. A match in subspace with no sparse match means "projection". |
| Similarity of Neural Network Representations Revisited (CKA). Kornblith, Norouzi, Lee, Hinton. ICML 2019. arXiv 1905.00414 | It proposes CKA to compare layers. | CKA finds matching layers in nets with different starts. CCA-type scores fail when width exceeds the sample count. | Our layers have 1536 neurons. We need many tokens, or CCA-style scores will mislead us. CKA is a cheap whole-layer score. |
| Revisiting Model Stitching to Compare Neural Representations. Bansal, Nakkiran, Barak. NeurIPS 2021. arXiv 2106.07682 | It plugs one net's layer into another through a small learned map. | Good nets trained differently can be stitched with no accuracy drop. They argue stitching shows more than CKA. | The test for "projection": a learned linear map from wide units to narrow units. If a map works but is dense, there is no merge. |
| Universal Neurons in GPT2 Language Models. Gurnee et al. TMLR 2024. arXiv 2401.12181 | It correlated neurons across 5 GPT-2 seeds over 100M tokens. | Only 1 to 5% of neurons are universal. Those are usually clear and fall in a few families. | Most neurons do NOT have a twin in another run. So we should expect most trained narrow neurons to have no single original twin. |
| Sparse Crosscoders for cross-layer features and model diffing. Lindsey et al. Anthropic, Transformer Circuits Thread, 2024 (Oct 25) | A shared dictionary reads and writes several layers or models. | It finds shared features and model-specific features. It is a research note, not a paper. | A crosscoder across the wide and narrow model is a strong test. It gives shared and unique features directly. It costs more than regression. |

Key question answered? Not directly. No paper I found trains a narrow layer and tests whether its units are sparse mixes of a wide layer's units in an LLM. Convergent Learning did this for image nets and found mixed results.

## Topic 4. What small or compressed models lose

| Paper | What it did | What it found | What it means for us |
|---|---|---|---|
| What Do Compressed Deep Neural Networks Forget? Hooker, Moorosi, Clark, Bengio, Denton. 2019. arXiv 1911.05248 (authors from memory, search results disagreed on order) | It found "Pruning Identified Exemplars" (PIEs), the few examples a pruned net gets wrong. | Top-line accuracy stays about the same. The loss falls on rare, atypical, or noisy examples (image tasks). Compression hurts the long tail. | Same pattern as our guess: fine detail goes first. Top-line scores hide it. We should test on rare cases, not on the average. Images, not LMs. |
| Large Language Models Struggle to Learn Long-Tail Knowledge. Kandpal, Deng, Roberts, Wallace, Raffel. ICML 2023. arXiv 2211.08411 | It counted pre-training documents that mention each question's entities. | Accuracy tracks how often a topic appeared. Larger models help, but scale alone needs many orders of magnitude to cover rare facts. | Rare facts are what a narrow model drops. Build the test set by fact frequency. |
| Physics of Language Models 3.3: Knowledge Capacity Scaling Laws. Allen-Zhu, Li. ICLR 2025. arXiv 2404.05405 | It measured stored facts in bits across model sizes. | About 2 bits of knowledge per parameter, even with int8. | Fewer neurons means fewer bits. A narrow layer must drop something. Facts go first. This is a capacity limit, not a failure of method. |
| Does Knowledge Distillation Really Work? Stanton, Izmailov, Kirichenko, Alemi, Wilson. NeurIPS 2021. arXiv 2106.05945 | It compared teacher and student predictions. | A big gap often remains between student and teacher even when the student could match. They blame optimization. | If we train a narrow layer by distillation, a gap may be an optimizer problem, not a capacity problem. Check with a longer run. |
| Compact Language Models via Pruning and Knowledge Distillation (Minitron). Muralidharan et al., NVIDIA. NeurIPS 2024. arXiv 2407.14679 | It pruned a 15B model (depth, width, MLP, attention). It then retrained with distillation on under 3% of the data. | Up to 40x fewer training tokens. Up to 16% better MMLU than training from scratch. | Prune then distill is the standard baseline. Our tests should be compared to it. Width pruning of MLP neurons is included. |
| The Quantization Model of Neural Scaling. Michaud, Liu, Girit, Tegmark. NeurIPS 2023. arXiv 2303.13506 (authors from memory) | It measured per-token loss across Pythia 19M to 6.4B on about 10M tokens. | Most tokens have near-zero loss. Their share grows with scale. Mean loss comes from the rare high-loss tokens. Some tokens improve sharply and some smoothly. | Closest to "which tokens do small models miss". The loss gap sits on a few tokens. It suggests whole skills ("quanta"), not only fine details. Both readings are open. |
| Memorization across scales in Pythia. arXiv 2506.12321 (title and authors NOT VERIFIED) | It sorts samples into memorized only by small, only by large, both, or neither. | Seen in a search summary only. | It could answer "what does small lose versus large". Needs a full read. |

Gap I found: no paper I found shows, for LMs of one family, whether small models lose "family members" or "whole concepts". The Pythia studies point to the tail of rare tokens and facts.

## Topic 5. The word "coarse-graining"

| Paper | What it did | What it found | What it means for us |
|---|---|---|---|
| An exact mapping between the Variational Renormalization Group and Deep Learning. Mehta, Schwab. 2014. arXiv 1410.3831 | It maps a physics method (RG, which merges spins into blocks) to deep nets built from RBMs. | An exact mapping for RBM-based nets. | "Coarse-graining" here means merging units into blocks and keeping the large-scale behavior. This is our idea, in physics words. |
| Probabilistic models, compressible interactions, and neural coding. arXiv 2112.14334 | It treats coarse-graining as compression. | Abstract only. | Shows the two words are used as near-synonyms in neuroscience and physics. |
| Diffusion-Guided Renormalization of Neural Systems via Tensor Networks. arXiv 2510.06361 | Renormalization of neural systems. | About 10x compression in system size at one setting (search summary). | A use of the term for brain data, not LLMs. |

I did not find "coarse-graining" used as a standard term in LLM compression papers. It appears in physics and neuroscience papers.

## Top 5 lessons for our experiment

1. Expect that most trained narrow neurons have no single original twin. Universal Neurons found only 1 to 5% match across runs. A "merge exists" result for most neurons would be a surprise.
2. Use sparse regression with a count of how much variance is left over. Convergent Learning shows this method works, and it found that many units stay unexplained. Report the leftover share, not only the matched share.
3. A dense map is not a merge. Model stitching shows a learned linear map often works. So also report how many original neurons each narrow neuron needs. A merge means few. A projection means many.
4. Prefix training is the proven way to get coarse neurons. Matryoshka SAEs and MatFormer train nested prefixes. Both pay a small quality cost. Test a nested version as a control arm.
5. Judge the loss on the rare cases, not the average. Hooker, Kandpal and the Pythia studies all say the loss hides in the tail. Capacity (about 2 bits per parameter) says some loss is unavoidable. Also watch for absorption: a coarse neuron may go silent where a fine neuron would fire.

## Right word

| Field | Term |
|---|---|
| LLM and CNN compression | "Width pruning" or "structured pruning" (remove neurons). "Neuron merging" or "folding" (combine neurons). "Width reduction" (DOTResize). |
| Nested training | "Matryoshka" or "elastic" or "nested" sub-models (MatFormer, Matryoshka SAEs). |
| Interpretability | "Feature splitting" (coarse to fine) and "feature absorption". |
| Physics | "Coarse-graining" or "renormalization". |

Best choice for us: say "neuron merging" for methods that combine existing neurons. Say "width reduction" for the goal. Say "Matryoshka" for trained nested layers. Use "coarse-graining" only as a plain-language image. It is not standard in LLM compression.

## Citations not fully verified

| Item | Status |
|---|---|
| Separate Nabeshima Matryoshka SAE write-up | NOT VERIFIED (only the joint paper found) |
| arXiv 2506.12321 (Pythia memorization) | NOT VERIFIED (title and authors) |
| arXiv 2607.22583 (gated MLP remark) | NOT VERIFIED (snippet only) |
| Author lists for Hooker 2019 and Michaud 2023 | From memory. Titles and ids are verified |
| Model folding numbers | Abstract has none. NOT VERIFIED |
| Towards Monosemanticity splitting numbers | NOT VERIFIED. I did not open that section |
